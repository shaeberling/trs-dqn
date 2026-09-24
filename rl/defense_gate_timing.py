"""Read-only visible geometry at an own-policy two-gate loss.

This diagnostic restores opaque own replay prefixes only to inspect screens
after constant-key interventions. It never supplies actions or observations
to training and cannot promote an intervention as learned play.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


WALL_CODES = (0x83, 0x8c, 0xb0)


def visible_ship_column(screen):
    """Column of one distinctive, sometimes occluded ship sprite glyph."""
    if screen.shape != (16, 64) or screen.dtype != np.uint8:
        raise ValueError("expected one raw 16x64 visible frame")
    matches = np.flatnonzero(screen[14] == 0xa8)
    return int(matches[0]) if len(matches) == 1 else None


def visible_wall_runs(screen, row):
    """Inclusive column runs of the game's horizontal wall glyphs only."""
    if screen.shape != (16, 64) or screen.dtype != np.uint8 or not 1 <= row < 16:
        raise ValueError("expected one raw 16x64 visible frame and gameplay row")
    mask = np.isin(screen[row], WALL_CODES)
    mask[0] = mask[-1] = False
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(start), int(stop-1)] for start, stop in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


def parse_pair(value):
    try:
        first, second = (int(part) for part in value.split(":"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("expected FRAME:FRAME or FRAME:ROW") from None
    return first, second


def measure(bundle, branches, walls):
    report, frames, actions = analyze(bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if (not branches or not walls or any(not 0 <= anchor < sample < first_loss
                                          for anchor, sample in branches)
            or any(not 0 <= frame < first_loss or not 1 <= row < 16
                   for frame, row in walls)):
        raise ValueError("measurements must precede the first visible loss")
    with np.load(Path(bundle)/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]
        metadata = json.loads(str(trace["metadata"]))
    manifest = json.loads((Path(bundle)/"manifest.json").read_text())["metadata"]
    env = DefenseEnv(tstates=report["tstates"], max_steps=manifest["max_steps"],
                     observation_stride=metadata.get("observation_stride", 1))
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        wanted = {anchor for anchor, _ in branches}
        saved = {}
        for index in range(max(wanted)+1):
            if index in wanted:
                saved[index] = capture(env)
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            np.testing.assert_array_equal(obs[-1], frames[index+1])
            if reward != rewards[index] or terminal or truncated or info["life_lost"]:
                raise RuntimeError("own recorded prefix failed exact native replay")
        measured = []
        for anchor, sample in branches:
            obs = restore(env, saved[anchor])
            for _ in range(sample-anchor):
                obs, _, terminal, truncated, info = env.step(4)  # original RIGHT command
                if terminal or truncated or info["life_lost"]:
                    raise RuntimeError("RIGHT branch ended before requested visible sample")
            measured.append(dict(anchor=anchor, sample=sample,
                                 held_command="RIGHT", ship_column=visible_ship_column(obs[-1]),
                                 displayed_score=info["score"], stage=info["stage"]))
    finally:
        env.close()
    return dict(source_bundle=str(Path(bundle).resolve()),
                replay_trace_sha256=sha256(Path(bundle)/"trace.npz"),
                probe_source_sha256=sha256(Path(__file__)),
                first_visible_loss_frame=first_loss,
                recorded_ship_columns={str(i): visible_ship_column(frames[i])
                                       for i in sorted({first for first, _ in walls}
                                                       | {sample for _, sample in branches})},
                recorded_wall_runs=[dict(frame=i, row=row,
                                         runs=visible_wall_runs(frames[i], row))
                                    for i, row in walls],
                held_right_samples=measured, diagnostic_only=True,
                training_data_written=False, parameter_updates=0,
                promotion_eligible=False,
                limitations=["A visible sprite glyph can be occluded or animated away.",
                             "Wall glyph runs are visible geometry, not collision flags.",
                             "Visible life loss can lag the physical collision.",
                             "Held-key interventions are not learned-policy actions or training examples."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--branch", type=parse_pair, action="append", required=True,
                        help="own prefix anchor and held-RIGHT sample as ANCHOR:SAMPLE")
    parser.add_argument("--wall", type=parse_pair, action="append", required=True,
                        help="recorded screen wall line as FRAME:ROW")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic")
    result = measure(args.bundle, args.branch, args.wall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
