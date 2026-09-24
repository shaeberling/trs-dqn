"""Diagnostic-only visible score and ship position after an earlier RIGHT hold.

No intervention action enters a learner, evaluation policy or replay.
Opaque native snapshots only accelerate exact reexecution.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_gate_timing import visible_ship_column
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore
from .defense_world_data import disk_guard


ANCHOR = 280
INTERMEDIATE = 339
WAYPOINT = 355
MAX_HOLD = 48
RIGHT = 4


def overlay_right(actions, anchor, start, hold):
    """Copy a recorded action segment and replace one bounded physical hold."""
    actions = np.asarray(actions)
    if (actions.ndim != 1 or actions.dtype != np.uint8
            or any(isinstance(x, bool) or not isinstance(x, int)
                   for x in (anchor, start, hold))
            or anchor < 0 or not anchor <= start < anchor+len(actions)
            or hold < 1 or hold > MAX_HOLD):
        raise ValueError("invalid verified action segment or RIGHT window")
    plan = actions.copy()
    begin = start-anchor
    plan[begin:min(len(plan), begin+hold)] = RIGHT
    return plan


def screen_mark(obs, info):
    return dict(displayed_score=int(info["score"]), stage=int(info["stage"]),
                ship_glyph_column=visible_ship_column(obs[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output directory required")
    report, frames, actions = analyze(args.bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if first_loss <= WAYPOINT or len(actions) < WAYPOINT:
        parser.error("verified first life must extend past the fixed waypoint")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    config = dict(bundle=str(args.bundle.resolve()),
                  trace_sha256=sha256(args.bundle/"trace.npz"),
                  model_sha256=sha256(args.bundle/"model.safetensors"),
                  source_sha256=sha256(Path(__file__)),
                  game_sha256=GAME_SHA256,
                  environment_version=ENVIRONMENT_VERSION,
                  seed=metadata["result"]["seed"],
                  anchor=ANCHOR, intermediate=INTERMEDIATE,
                  waypoint=WAYPOINT, max_hold=MAX_HOLD,
                  ordered_pairs=(WAYPOINT-ANCHOR)*MAX_HOLD,
                  held_command="RIGHT", diagnostic_only=True,
                  model_updates=0, candidate_actions_never_training_data=True,
                  promotion_eligible=False)
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    write_json(args.output/"config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=metadata["max_steps"],
                     observation_stride=metadata.get("observation_stride", 1))
    try:
        obs = env.reset(config["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        for index in range(ANCHOR):
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            if (reward != rewards[index] or terminal or truncated
                    or info["life_lost"]):
                raise RuntimeError("original first-life prefix failed native reexecution")
            np.testing.assert_array_equal(obs[-1], frames[index+1])
        saved = capture(env)
        base = np.asarray(actions[ANCHOR:WAYPOINT], np.uint8)
        baseline = {}
        for index, action in enumerate(base, ANCHOR):
            obs, reward, terminal, truncated, info = env.step(int(action))
            if (reward != rewards[index] or terminal or truncated
                    or info["life_lost"]):
                raise RuntimeError("recorded waypoint suffix failed native reexecution")
            np.testing.assert_array_equal(obs[-1], frames[index+1])
            if index+1 in (INTERMEDIATE, WAYPOINT):
                baseline[str(index+1)] = screen_mark(obs, info)
        for point in (INTERMEDIATE, WAYPOINT):
            if baseline[str(point)]["displayed_score"] != int(rewards[:point].sum()):
                raise RuntimeError("native visible score differs from source trace")
        counts = {"intermediate_full_score_readable": 0,
                  "waypoint_full_score_readable": 0,
                  "early_visible_boundary": 0}
        max_columns = {str(INTERMEDIATE): None, str(WAYPOINT): None}
        completed = 0
        with (args.output/"outcomes.jsonl").open("x", buffering=1) as output:
            for start in range(ANCHOR, WAYPOINT):
                for hold in range(1, MAX_HOLD+1):
                    plan = overlay_right(base, ANCHOR, start, hold)
                    obs = restore(env, saved)
                    row = dict(start=start, hold=hold, at_339=None, at_355=None,
                               boundary_frame=None, boundary_type=None)
                    for index, action in enumerate(plan, ANCHOR):
                        obs, _, terminal, truncated, info = env.step(int(action))
                        if index+1 == INTERMEDIATE:
                            row["at_339"] = screen_mark(obs, info)
                        if index+1 == WAYPOINT:
                            row["at_355"] = screen_mark(obs, info)
                        if info["life_lost"] or info["stage"] > 1 or terminal or truncated:
                            row["boundary_frame"] = index+1
                            row["boundary_type"] = ("later_stage" if info["stage"] > 1 else
                                                    "life_loss" if info["life_lost"] else
                                                    "episode_end")
                            counts["early_visible_boundary"] += 1
                            break
                    for point, key in ((INTERMEDIATE, "at_339"), (WAYPOINT, "at_355")):
                        mark = row[key]
                        if (mark is not None and (row["boundary_frame"] is None
                                                  or row["boundary_frame"] > point)
                                and mark["displayed_score"] >= baseline[str(point)]["displayed_score"]
                                and mark["ship_glyph_column"] is not None):
                            counts[("intermediate" if point == INTERMEDIATE else "waypoint")
                                   +"_full_score_readable"] += 1
                            old = max_columns[str(point)]
                            max_columns[str(point)] = (mark["ship_glyph_column"] if old is None else
                                                        max(old, mark["ship_glyph_column"]))
                    output.write(json.dumps(row)+"\n")
                    completed += 1
                    if completed % 500 == 0:
                        disk_guard(args.output)
        result = dict(**config, completed=completed, baseline=baseline,
                      counts=counts, max_full_score_readable_ship_columns=max_columns,
                      outcomes_sha256=sha256(args.output/"outcomes.jsonl"))
        write_json(args.output/"report.json", result)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
