"""Diagnostic-only early two-phase reachability from verified own play.

This grid deliberately follows its two unbiased early command phases with
RIGHT toward an opening already visible on the original screen. It is not a
learned controller, trainer, demonstration set or promotable model replay.
Only original displayed score, visible life/stage and raw screens are read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_gate_timing import visible_ship_column
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_macro_explore import command_ids
from .defense_phase_grid import grid, phase_plan
from .defense_snapshot import capture, restore
from .defense_world_data import disk_guard


ANCHOR, WAYPOINT, HORIZON = 280, 355, 512
DELAYS, HOLDS = (0, 8, 16, 24), (8, 16, 24)
COMMANDS = command_ids("effective-stage-one")
RIGHT = 4
MILESTONES = (339, WAYPOINT, 387, 407)


def candidate_plan(baseline, choice):
    baseline = np.asarray(baseline)
    if baseline.shape != (HORIZON-ANCHOR,) or baseline.dtype != np.uint8:
        raise ValueError("expected the exact own-action window")
    delay, first_hold, first_command, second_hold, second_command = choice
    if (delay not in DELAYS or first_hold not in HOLDS or second_hold not in HOLDS
            or first_command not in COMMANDS or second_command not in COMMANDS):
        raise ValueError("candidate is outside the fixed early-phase grid")
    plan = baseline.copy()
    plan[:WAYPOINT-ANCHOR] = phase_plan(
        baseline[:WAYPOINT-ANCHOR], delay, first_hold, first_command,
        second_hold, second_command)
    plan[WAYPOINT-ANCHOR:] = RIGHT
    return plan


def play_candidate(env, saved, plan, *, trace=False):
    obs = restore(env, saved)
    marks, first_above = {}, None
    frames, actions, rewards = ([obs[-1].copy()], [], []) if trace else (None, None, None)
    for offset, action in enumerate(plan, 1):
        obs, reward, terminal, truncated, info = env.step(int(action))
        absolute = ANCHOR+offset
        if trace:
            frames.append(obs[-1].copy())
            actions.append(int(action))
            rewards.append(float(reward))
        if absolute in MILESTONES:
            marks[str(absolute)] = dict(score=int(info["score"]),
                                         ship_column=visible_ship_column(obs[-1]),
                                         lives=int(info["lives"]),
                                         stage=int(info["stage"]))
        if first_above is None and info["score"] > 2620:
            first_above = absolute
        ended = bool(info["life_lost"] or info["stage"] > saved.stage
                     or info["mission_completed"] or terminal or truncated)
        if ended:
            break
    row = dict(actions=offset, absolute_frame=absolute, score=int(info["score"]),
               stage=int(info["stage"]), highest_stage=int(info["highest_stage"]),
               life_lost=bool(info["life_lost"]),
               mission_completed=bool(info["mission_completed"]),
               survived_horizon=not ended, first_above_2620=first_above,
               marks=marks)
    if trace:
        return row, (np.stack(frames), np.asarray(actions, np.uint8),
                     np.asarray(rewards, np.float32))
    return row


def verify_from_boot(env, seed, source_frames, source_actions, source_rewards,
                     saved, plan, expected):
    branch, (frames, executed, rewards) = play_candidate(env, saved, plan, trace=True)
    if branch != expected:
        raise RuntimeError("candidate changed on snapshot replay")
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], source_frames[0])
    for index in range(ANCHOR):
        obs, reward, terminal, truncated, info = env.step(int(source_actions[index]))
        if (reward != source_rewards[index] or terminal or truncated or info["life_lost"]):
            raise RuntimeError("own original-boot prefix changed")
        np.testing.assert_array_equal(obs[-1], source_frames[index+1])
    np.testing.assert_array_equal(obs[-1], frames[0])
    for index, action in enumerate(executed):
        obs, reward, terminal, truncated, info = env.step(int(action))
        if reward != rewards[index]:
            raise RuntimeError("candidate reward changed on original-boot replay")
        np.testing.assert_array_equal(obs[-1], frames[index+1])
    if (info["score"] != expected["score"] or info["stage"] != expected["stage"]
            or info["life_lost"] != expected["life_lost"]):
        raise RuntimeError("candidate outcome changed on original-boot replay")
    return (dict(verified=True, boot_prefix_actions=ANCHOR,
                 candidate_actions=len(executed),
                 method="original boot; every source/candidate action, reward and screen reproduced"),
            (frames, executed, rewards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 runs all 3600 fixed candidates; positive prefix is a plumbing smoke")
    args = parser.parse_args()
    total = len(DELAYS)*len(HOLDS)**2*len(COMMANDS)**2
    if args.output.exists() or not 0 <= args.limit <= total:
        parser.error("fresh output and a valid bounded grid prefix required")
    report, source_frames, source_actions = analyze(args.bundle)
    if (report["lives"][0]["visible_loss_frame"] != 407
            or report["lives"][0]["visible_score_at_loss"] != 2620
            or len(source_actions) < HORIZON):
        parser.error("requires the exact verified own first-life source")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as data:
        source_rewards = data["rewards"].copy()
        metadata = json.loads(str(data["metadata"]))
    baseline = np.asarray(source_actions[ANCHOR:HORIZON], np.uint8)
    score_at_waypoint = int(source_rewards[:WAYPOINT].sum())
    config = dict(bundle=str(args.bundle.resolve()), source_trace_sha256=sha256(args.bundle/"trace.npz"),
                  source_model_sha256=sha256(args.bundle/"model.safetensors"),
                  source_sha256=sha256(Path(__file__)),
                  phase_source_sha256=sha256(Path(__file__).with_name("defense_phase_grid.py")),
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  source_seed=metadata["result"]["seed"], anchor=ANCHOR,
                  waypoint=WAYPOINT, horizon=HORIZON, delays=list(DELAYS),
                  holds=list(HOLDS), symmetric_first_second_commands=list(COMMANDS),
                  third_command="RIGHT from waypoint to first boundary or horizon",
                  total_candidates=total, requested_candidates=args.limit or total,
                  source_score_at_waypoint=score_at_waypoint,
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  promotion_eligible=False, native_snapshots_opaque=True)
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    write_json(args.output/"config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=metadata["max_steps"],
                     observation_stride=metadata.get("observation_stride", 1))
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        obs = env.reset(config["source_seed"])
        np.testing.assert_array_equal(obs[-1], source_frames[0])
        saved = None
        for index in range(407):
            if index == ANCHOR:
                saved = capture(env)
            obs, reward, terminal, truncated, info = env.step(int(source_actions[index]))
            if (reward != source_rewards[index] or terminal or truncated
                    or info["life_lost"] != (index+1 == 407)):
                raise RuntimeError("verified source life failed exact reexecution")
            np.testing.assert_array_equal(obs[-1], source_frames[index+1])
        baseline_plan = np.asarray(source_actions[ANCHOR:407], np.uint8)
        baseline_row = play_candidate(env, saved,
            np.pad(baseline_plan, (0, HORIZON-407), constant_values=0))
        if (baseline_row["actions"] != 407-ANCHOR or baseline_row["score"] != 2620
                or not baseline_row["life_lost"]):
            raise RuntimeError("own suffix failed exact snapshot reexecution")
        config["baseline"] = baseline_row
        write_json(args.output/"config.json", config)
        counts = dict(survived_waypoint=0, full_score_waypoint=0,
                      full_score_readable_waypoint=0, score_above_2620=0,
                      horizon_survivors=0, later_stage=0)
        max_full_score_ship_column = None
        best_score, latest = 2620, 0
        discovery = None
        with (args.output/"outcomes.jsonl").open("x", buffering=1) as stream:
            for candidate, choice in enumerate(grid(DELAYS, HOLDS, COMMANDS), 1):
                if stop or candidate > config["requested_candidates"]:
                    break
                plan = candidate_plan(baseline, choice)
                row = play_candidate(env, saved, plan)
                latest = candidate
                mark = row["marks"].get(str(WAYPOINT))
                if mark is not None and mark["lives"] == saved.lives:
                    counts["survived_waypoint"] += 1
                    if mark["score"] >= score_at_waypoint:
                        counts["full_score_waypoint"] += 1
                        column = mark["ship_column"]
                        if column is not None:
                            counts["full_score_readable_waypoint"] += 1
                            max_full_score_ship_column = (column if max_full_score_ship_column is None
                                                          else max(column, max_full_score_ship_column))
                counts["score_above_2620"] += row["score"] > 2620
                counts["horizon_survivors"] += row["survived_horizon"]
                counts["later_stage"] += row["highest_stage"] >= 2
                best_score = max(best_score, row["score"])
                stream.write(json.dumps(dict(candidate=candidate, choice=list(choice),
                                             plan_sha256=hashlib.sha256(plan).hexdigest(),
                                             **row))+"\n")
                if (row["score"] > 2620 or row["survived_horizon"]
                        or row["highest_stage"] >= 2):
                    verification, trace = verify_from_boot(
                        env, config["source_seed"], source_frames, source_actions,
                        source_rewards, saved, plan, row)
                    frames, executed, rewards = trace
                    np.savez_compressed(args.output/"discovery-trace.npz",
                                        frames=frames, actions=executed, rewards=rewards)
                    discovery = dict(candidate=candidate, choice=list(choice), outcome=row,
                                     verification=verification,
                                     trace_sha256=sha256(args.output/"discovery-trace.npz"))
                    write_json(args.output/"discovery.json", discovery)
                    print(json.dumps(dict(event="verified_discovery", **discovery)), flush=True)
                    break
                if candidate % 250 == 0:
                    disk_guard(args.output)
                    status = dict(completed=candidate, counts=counts,
                                  best_score=best_score,
                                  max_full_score_ship_column=max_full_score_ship_column)
                    write_json(args.output/"status.json", status)
                    print(json.dumps(dict(event="progress", **status)), flush=True)
        result = dict(config=config, completed=latest, counts=counts,
                      max_full_score_ship_column=max_full_score_ship_column,
                      best_score=best_score, discovery=discovery,
                      stopped=stop, outcomes_sha256=sha256(args.output/"outcomes.jsonl"))
        write_json(args.output/"report.json", result)
        print(json.dumps(dict(event="finished", completed=latest, counts=counts,
                              best_score=best_score, discovery=discovery, stopped=stop)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
