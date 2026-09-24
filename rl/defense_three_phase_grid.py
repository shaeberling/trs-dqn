"""Diagnostic-only exhaustive three-phase commands from verified own play.

No candidate action is supplied to a learner, used as a reward, or promoted
as a learned replay. Opaque emulator snapshots are only reset machinery.
"""

import argparse
import itertools
import json
from pathlib import Path
import shutil
import signal

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_phase_grid import COMMANDS
from .defense_snapshot import capture
from .defense_trajectory_search import play, outcome_rank, verify_from_boot
from .defense_world_data import disk_guard


DELAYS = (0, 8, 16)
HOLDS = (8, 16, 32)


def grid(delays=DELAYS, holds=HOLDS, commands=COMMANDS):
    if (not delays or not holds or not commands
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in delays)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in holds)
            or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 20
                   for x in commands)
            or len(set(delays)) != len(delays) or len(set(holds)) != len(holds)
            or len(set(commands)) != len(commands)):
        raise ValueError("distinct nonnegative delays, positive holds and legal commands required")
    return itertools.product(delays, holds, commands, holds, commands, holds, commands)


def phase_plan(baseline, delay, first_hold, first_command,
               second_hold, second_command, third_hold, third_command):
    baseline = np.asarray(baseline)
    values = (delay, first_hold, first_command, second_hold, second_command,
              third_hold, third_command)
    if (baseline.ndim != 1 or baseline.dtype != np.uint8
            or any(isinstance(x, bool) or not isinstance(x, int) for x in values)
            or delay < 0 or min(first_hold, second_hold, third_hold) < 1
            or any(not 0 <= command < 20 for command in
                   (first_command, second_command, third_command))
            or delay+first_hold+second_hold+third_hold > len(baseline)):
        raise ValueError("invalid three-phase plan")
    plan = baseline.copy()
    index = delay
    for hold, command in ((first_hold, first_command),
                          (second_hold, second_command),
                          (third_hold, third_command)):
        plan[index:index+hold] = command
        index += hold
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, choices=(339, 340), required=True)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 runs the entire fixed grid; positive prefix is an integrity smoke")
    args = parser.parse_args()
    total = len(DELAYS)*len(HOLDS)**3*len(COMMANDS)**3
    if (args.output.exists() or args.horizon < max(DELAYS)+3*max(HOLDS)
            or not 0 <= args.limit <= total):
        parser.error("fresh output, sufficient horizon and valid grid prefix required")
    report, frames, actions = analyze(args.bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if args.anchor >= first_loss or args.anchor+args.horizon > len(actions):
        parser.error("requires exact own first-life prefix and enough recorded action tail")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as trace:
        source_rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    baseline_plan = np.asarray(actions[args.anchor:args.anchor+args.horizon], np.uint8)
    config = dict(bundle=str(args.bundle.resolve()), trace_sha256=sha256(args.bundle/"trace.npz"),
                  model_sha256=sha256(args.bundle/"model.safetensors"),
                  source_sha256=sha256(Path(__file__)),
                  play_source_sha256=sha256(Path(__file__).with_name("defense_trajectory_search.py")),
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  seed=metadata["result"]["seed"], anchor=args.anchor, horizon=args.horizon,
                  total_candidates=total, requested_candidates=args.limit or total,
                  delays=list(DELAYS), holds=list(HOLDS), symmetric_commands=list(COMMANDS),
                  diagnostic_only=True, model_updates=0, policy_inputs_changed=False,
                  reward_changed=False, candidate_actions_never_training_data=True,
                  promotion_eligible=False)
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    np.savez_compressed(args.output/"baseline-actions.npz", actions=baseline_plan)
    config["baseline_actions_sha256"] = sha256(args.output/"baseline-actions.npz")
    write_json(args.output/"config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=metadata["max_steps"],
                     observation_stride=metadata.get("observation_stride", 1))
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
                    or info["life_lost"] != (index+1 == first_loss)):
                raise RuntimeError("verified first-life prefix failed native reexecution")
            np.testing.assert_array_equal(obs[-1], frames[index+1])
        baseline = play(env, saved, baseline_plan)
        if (baseline["actions"] != first_loss-args.anchor
                or baseline["score"] != report["lives"][0]["visible_score_at_loss"]
                or not baseline["life_lost"]):
            raise RuntimeError("own suffix failed same-state native reexecution")
        best = dict(candidate=0, **baseline)
        completed, discovered = 0, False
        counts = dict(above_baseline_score=0, later_stage=0, survived_horizon=0,
                      same_score_longer=0)
        with (args.output/"outcomes.jsonl").open("x", buffering=1) as stream:
            stream.write(json.dumps(dict(kind="own_policy_baseline", **best))+"\n")
            for candidate, plan_values in enumerate(grid(), 1):
                if stopped or candidate > config["requested_candidates"]:
                    break
                plan = phase_plan(baseline_plan, *plan_values)
                row = play(env, saved, plan)
                completed = candidate
                record = dict(candidate=candidate, delay=plan_values[0],
                              first_hold=plan_values[1], first_command=plan_values[2],
                              second_hold=plan_values[3], second_command=plan_values[4],
                              third_hold=plan_values[5], third_command=plan_values[6], **row)
                stream.write(json.dumps(record)+"\n")
                counts["above_baseline_score"] += int(row["score"] > baseline["score"])
                counts["later_stage"] += int(row["stage"] > 1)
                counts["survived_horizon"] += int(row["survived_horizon"])
                counts["same_score_longer"] += int(row["score"] == baseline["score"]
                                                   and row["actions"] > baseline["actions"])
                if outcome_rank(row) > outcome_rank(best):
                    best = record
                    print(json.dumps(dict(event="new_best", **best)), flush=True)
                if row["stage"] > 1 or row["mission_completed"]:
                    observed, screens, rewards = play(env, saved, plan, trace=True)
                    if observed != row:
                        raise RuntimeError("candidate outcome changed on snapshot reexecution")
                    verified = verify_from_boot(env, config["seed"], frames, actions,
                                                source_rewards, args.anchor, plan, observed,
                                                screens, rewards)
                    write_json(args.output/"discovery.json", dict(candidate=candidate,
                        phase=list(plan_values), observed=observed, boot_verified=verified,
                        learned_policy_success=False, promotion_eligible=False))
                    discovered = True
                if candidate % 1000 == 0 or discovered:
                    write_json(args.output/"status.json", dict(completed=completed,
                        requested=config["requested_candidates"], baseline=baseline,
                        best=best, counts=counts, stage_two_discovery=discovered,
                        stop_requested=stopped))
                    print(json.dumps(dict(event="progress", completed=completed,
                                          best=best, counts=counts,
                                          stage_two_discovery=discovered)), flush=True)
                    disk_guard(args.output)
                if discovered:
                    break
        write_json(args.output/"report.json", dict(**config, baseline=baseline,
            best=best, completed=completed, counts=counts,
            stage_two_discovery=discovered, stopped=stopped,
            outcomes_sha256=sha256(args.output/"outcomes.jsonl")))
        print(json.dumps(dict(event="finished", completed=completed, best=best,
                              counts=counts, stage_two_discovery=discovered,
                              stopped=stopped)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
