"""Diagnostic-only exhaustive two-phase command grid from verified own play.

No grid action enters a learner, reward, policy input or published replay.
Only displayed score and visible life/stage outcomes rank trials. Opaque
native snapshots avoid booting for every branch but are never decoded.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import signal

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_macro_explore import command_ids
from .defense_snapshot import capture
from .defense_trajectory_search import play, outcome_rank, verify_from_boot
from .defense_world_data import disk_guard


COMMANDS = command_ids("effective-stage-one")
DELAYS = (0, 8, 16)
HOLDS = (4, 8, 16, 24, 32)


def grid(delays=DELAYS, holds=HOLDS, commands=COMMANDS):
    """Fixed direction-symmetric enumeration; no outcome-steered sampling."""
    if (not delays or not holds or not commands
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in delays)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in holds)
            or any(isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < 20
                   for n in commands)
            or len(set(delays)) != len(delays) or len(set(holds)) != len(holds)
            or len(set(commands)) != len(commands)):
        raise ValueError("positive distinct delays, holds and physical commands required")
    return itertools.product(delays, holds, commands, holds, commands)


def phase_plan(baseline, delay, first_hold, first_command, second_hold, second_command):
    baseline = np.asarray(baseline)
    if (baseline.ndim != 1 or baseline.dtype != np.uint8
            or any(isinstance(n, bool) or not isinstance(n, int) for n in
                   (delay, first_hold, first_command, second_hold, second_command))
            or delay < 0 or min(first_hold, second_hold) < 1
            or not 0 <= first_command < 20 or not 0 <= second_command < 20
            or delay+first_hold+second_hold > len(baseline)):
        raise ValueError("invalid two-phase plan")
    plan = baseline.copy()
    plan[delay:delay+first_hold] = first_command
    plan[delay+first_hold:delay+first_hold+second_hold] = second_command
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=int, default=322)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = entire fixed 7500-plan grid; positive prefix for plumbing smoke")
    args = parser.parse_args()
    total = len(DELAYS)*len(HOLDS)**2*len(COMMANDS)**2
    if (args.output.exists() or args.anchor < 1 or args.horizon < 1
            or max(DELAYS)+2*max(HOLDS) > args.horizon or not 0 <= args.limit <= total):
        parser.error("fresh output, positive anchor/horizon and valid grid prefix required")
    report, frames, actions = analyze(args.bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if args.anchor >= first_loss or args.anchor+args.horizon > len(actions):
        parser.error("grid requires an exact own first-life prefix and enough recorded action tail")
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
    write_json(args.output/"config.json", config)
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
            raise RuntimeError("source suffix failed same-state reexecution")
        best = dict(candidate=0, **baseline)
        completed, stage_two = 0, False
        with (args.output/"outcomes.jsonl").open("x", buffering=1) as stream:
            stream.write(json.dumps(dict(kind="own_policy_baseline", **best))+"\n")
            for candidate, (delay, first_hold, first_command, second_hold, second_command) in enumerate(
                    grid(), 1):
                if stopped or candidate > config["requested_candidates"]:
                    break
                plan = phase_plan(baseline_plan, delay, first_hold, first_command,
                                  second_hold, second_command)
                row = play(env, saved, plan)
                completed = candidate
                stream.write(json.dumps(dict(candidate=candidate, delay=delay,
                    first_hold=first_hold, first_command=first_command,
                    second_hold=second_hold, second_command=second_command, **row))+"\n")
                if outcome_rank(row) > outcome_rank(best):
                    best = dict(candidate=candidate, delay=delay, first_hold=first_hold,
                                first_command=first_command, second_hold=second_hold,
                                second_command=second_command, **row)
                    print(json.dumps(dict(event="new_best", **best)), flush=True)
                if row["stage"] > 1 or row["mission_completed"]:
                    observed, screens, rewards = play(env, saved, plan, trace=True)
                    if observed != row:
                        raise RuntimeError("candidate outcome changed on snapshot reexecution")
                    verified = verify_from_boot(env, config["seed"], frames, actions, source_rewards,
                                                args.anchor, plan, observed, screens, rewards)
                    write_json(args.output/"discovery.json", dict(candidate=candidate,
                        phase=dict(delay=delay, first_hold=first_hold, first_command=first_command,
                                   second_hold=second_hold, second_command=second_command),
                        observed=observed, boot_verified=verified, learned_policy_success=False,
                        promotion_eligible=False))
                    stage_two = True
                if candidate % 500 == 0 or stage_two:
                    write_json(args.output/"status.json", dict(completed=completed,
                        requested=config["requested_candidates"], baseline=baseline, best=best,
                        stage_two_discovery=stage_two, stop_requested=stopped))
                    print(json.dumps(dict(event="progress", completed=completed,
                                          best=best, stage_two_discovery=stage_two)), flush=True)
                    disk_guard(args.output)
                if stage_two:
                    break
        write_json(args.output/"report.json", dict(**config, baseline=baseline, best=best,
            completed=completed, stage_two_discovery=stage_two, stopped=stopped,
            outcomes_sha256=sha256(args.output/"outcomes.jsonl")))
        print(json.dumps(dict(event="finished", completed=completed, best=best,
                              stage_two_discovery=stage_two, stopped=stopped)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
