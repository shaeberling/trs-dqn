"""Diagnostic-only score search by mutating a verified own-policy action suffix.

Candidate actions never become demonstrations, actor targets, evaluation
policies or promoted replays. The only search fitness is displayed score,
stage, and visible life survival as a tie-break; native states are opaque
same-build reset machinery.
"""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256, ENVIRONMENT_VERSION
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_macro_explore import command_ids
from .defense_snapshot import capture, restore
from .defense_world_data import disk_guard


HOLDS = (4, 8, 16, 32, 64)
COMMANDS = command_ids("effective-stage-one")


def mutate_plan(parent, rng, holds=HOLDS, commands=COMMANDS):
    parent = np.asarray(parent)
    if (parent.ndim != 1 or len(parent) < 1 or parent.dtype != np.uint8
            or not holds or min(holds) < 1 or not commands
            or min(commands) < 0 or max(commands) > 255):
        raise ValueError("invalid own-policy plan or symmetric mutation profile")
    plan = parent.copy()
    edits = []
    for _ in range(int(rng.integers(1, 5))):
        start = int(rng.integers(len(plan)))
        length = int(rng.choice(holds))
        command = int(rng.choice(commands))
        stop = min(len(plan), start+length)
        plan[start:stop] = command
        edits.append(dict(start=start, length=stop-start, command=command))
    return plan, edits


def outcome_rank(row):
    """Mission/stage first, real score next, visible survival only on ties."""
    return (int(row.get("mission_completed", False)), int(row["stage"]),
            int(row["score"]), int(row["actions"]))


def play(env, saved, plan, *, trace=False):
    obs = restore(env, saved)
    screens, rewards = ([obs[-1].copy()], []) if trace else (None, None)
    info = None
    for index, action in enumerate(plan):
        obs, reward, terminal, truncated, info = env.step(int(action))
        if trace:
            screens.append(obs[-1].copy())
            rewards.append(reward)
        if (info["life_lost"] or info["stage"] > saved.stage
                or info["mission_completed"] or terminal or truncated):
            break
    row = dict(actions=index+1, score=int(info["score"]), stage=int(info["stage"]),
               life_lost=bool(info["life_lost"]), mission_completed=bool(info["mission_completed"]),
               survived_horizon=not (info["life_lost"] or info["stage"] > saved.stage
                                     or info["mission_completed"] or terminal or truncated))
    if trace:
        return row, np.stack(screens), np.asarray(rewards, np.float32)
    return row


def verify_from_boot(env, seed, source_frames, source_actions, source_rewards,
                     anchor, plan, expected, screens, rewards):
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], source_frames[0])
    for index in range(anchor):
        obs, reward, terminal, truncated, info = env.step(int(source_actions[index]))
        if (reward != source_rewards[index] or terminal or truncated or info["life_lost"]):
            raise RuntimeError("own prefix failed original-boot reexecution")
        np.testing.assert_array_equal(obs[-1], source_frames[index+1])
    np.testing.assert_array_equal(obs[-1], screens[0])
    result = None
    for index, action in enumerate(plan[:expected["actions"]]):
        obs, reward, terminal, truncated, info = env.step(int(action))
        if reward != rewards[index]:
            raise RuntimeError("candidate reward failed original-boot reexecution")
        np.testing.assert_array_equal(obs[-1], screens[index+1])
        result = dict(actions=index+1, score=int(info["score"]), stage=int(info["stage"]),
                      life_lost=bool(info["life_lost"]),
                      mission_completed=bool(info["mission_completed"]))
        if info["life_lost"] or info["stage"] > 1 or terminal or truncated:
            if index+1 != expected["actions"]:
                raise RuntimeError("candidate boundary moved during boot verification")
            break
    for key in result:
        if result[key] != expected[key]:
            raise RuntimeError("candidate result failed original-boot reexecution")
    if result["stage"] < 2 and not result["mission_completed"]:
        raise RuntimeError("claimed later-stage discovery did not reproduce")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=144)
    parser.add_argument("--candidates", type=int, default=20_000)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--seed", type=int, default=547)
    args = parser.parse_args()
    if (args.output.exists() or min(args.anchor, args.horizon, args.candidates,
                                    args.elite) < 1 or args.seed < 0):
        parser.error("fresh output and positive bounded search settings required")
    report, frames, actions = analyze(args.bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if args.anchor >= first_loss or args.anchor+args.horizon > len(actions):
        parser.error("anchor must precede own first life loss and plan fit recorded game")
    manifest = json.loads((args.bundle/"manifest.json").read_text())["metadata"]
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as data:
        source_rewards = data["rewards"].copy()
    baseline_plan = np.asarray(actions[args.anchor:args.anchor+args.horizon], np.uint8)
    rng = np.random.default_rng(args.seed)
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    config = dict(bundle=str(args.bundle.resolve()), trace_sha256=sha256(args.bundle/"trace.npz"),
                  model_sha256=sha256(args.bundle/"model.safetensors"),
                  source_sha256=sha256(Path(__file__)), game_sha256=GAME_SHA256,
                  environment_version=ENVIRONMENT_VERSION, seed=manifest["result"]["seed"],
                  anchor=args.anchor, horizon=args.horizon, candidates=args.candidates,
                  elite=args.elite, search_seed=args.seed, symmetric_commands=list(COMMANDS),
                  replacement_holds=list(HOLDS), diagnostic_only=True, model_updates=0,
                  policy_inputs_changed=False, reward_changed=False,
                  searched_actions_never_training_data=True, promotion_eligible=False)
    write_json(args.output/"config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=manifest["max_steps"],
                     observation_stride=manifest.get("observation_stride", 1))
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        obs = env.reset(config["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        saved = None
        for index in range(first_loss):
            if index == args.anchor:
                saved = capture(env)
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            if (reward != source_rewards[index] or terminal or truncated
                    or (info["life_lost"] != (index+1 == first_loss))):
                raise RuntimeError("verified source life failed exact reexecution")
            np.testing.assert_array_equal(obs[-1], frames[index+1])
        baseline = play(env, saved, baseline_plan)
        if (baseline["actions"] != first_loss-args.anchor
                or baseline["score"] != report["lives"][0]["visible_score_at_loss"]
                or not baseline["life_lost"]):
            raise RuntimeError("source action plan failed baseline reexecution")
        baseline.update(candidate=0, plan_sha256=hashlib.sha256(baseline_plan).hexdigest())
        elites = [(outcome_rank(baseline), rng.random(), 0, baseline_plan)]
        best, best_plan = baseline.copy(), baseline_plan.copy()
        seen = {baseline_plan.tobytes()}
        discovered = False
        completed = 0
        with (args.output/"plans.jsonl").open("x") as stream:
            stream.write(json.dumps(dict(kind="own_policy_baseline",
                                         plan=baseline_plan.tolist(), **baseline))+"\n")
            for candidate in range(1, args.candidates+1):
                if stopped:
                    break
                parent = elites[int(rng.integers(len(elites)))] if rng.random() < .75 else elites[0]
                for _ in range(16):
                    plan, edits = mutate_plan(parent[3], rng)
                    if plan.tobytes() not in seen:
                        break
                if plan.tobytes() in seen:
                    continue
                seen.add(plan.tobytes())
                row = play(env, saved, plan)
                rank = outcome_rank(row)
                eligible = len(elites) < args.elite or rank >= elites[-1][0]
                if eligible:
                    elites.append((rank, rng.random(), candidate, plan.copy()))
                    elites.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    del elites[args.elite:]
                accepted = any(item[2] == candidate for item in elites)
                if rank > outcome_rank(best):
                    best, best_plan = dict(candidate=candidate, **row), plan.copy()
                    print(json.dumps(dict(event="new_best", **best)), flush=True)
                stream.write(json.dumps(dict(candidate=candidate, parent=parent[2], edits=edits,
                                             plan=plan.tolist(), accepted=accepted, **row))+"\n")
                completed = candidate
                if row["stage"] > 1 or row["mission_completed"]:
                    observed, screens, rewards = play(env, saved, plan, trace=True)
                    if observed != row:
                        raise RuntimeError("candidate outcome changed on snapshot reexecution")
                    verified = verify_from_boot(env, config["seed"], frames, actions, source_rewards,
                                                args.anchor, plan, observed, screens, rewards)
                    write_json(args.output/"discovery.json", dict(candidate=candidate, plan=plan.tolist(),
                        observed=observed, boot_verified=verified, learned_policy_success=False,
                        promotion_eligible=False))
                    discovered = True
                if candidate % 1000 == 0 or discovered:
                    stream.flush()
                    status = dict(completed=completed, requested=args.candidates, distinct=len(seen),
                                  best=best, baseline=baseline, elite_size=len(elites),
                                  discovery=discovered, rng_state=rng.bit_generator.state,
                                  stop_requested=stopped)
                    write_json(args.output/"status.json", status)
                    print(json.dumps({k:status[k] for k in ("completed", "distinct", "best",
                                                             "elite_size", "discovery")}), flush=True)
                    disk_guard(args.output)
                if discovered:
                    break
        write_json(args.output/"report.json", dict(config=config, completed=completed,
            requested=args.candidates, baseline=baseline, best=best,
            best_plan=best_plan.tolist(), distinct=len(seen), discovery=discovered,
            stop_requested=stopped, rng_state=rng.bit_generator.state,
            candidate_actions_not_training_examples=True, model_updates=0,
            protected_best_unchanged=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
