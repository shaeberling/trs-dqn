"""Forensic-only original-course pointer audit of verified own life losses.

This deliberately reads nonvideo RAM, so it MUST NOT be imported by a trainer,
policy, reward, curriculum, checkpoint selector, or replay promoter. It only
checks whether the repeated visible losses occur early or late in the original
immutable stage-one obstacle stream. Pointer position is not collision time.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from trs.cmd import CMD

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_ars_focus import load_source
from .defense_audit import Image, audit
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import restore


POINTER_LOW = 0x819B
COURSE_START = 0x76A9
COURSE_END = 0x7BD8
COURSE_ROWS = 126


def course_rows(path=Path("var/defense.cmd")):
    """Decode immutable source row boundaries after the existing binary audit."""
    audited = audit(path)
    if audited["stage_one_course"]["rows"] != COURSE_ROWS:
        raise ValueError("stage-one audit row count changed")
    image = Image()
    CMD(image).load(str(path))
    cursor, boundaries = COURSE_START, {COURSE_START: 0}
    for row in range(1, COURSE_ROWS+1):
        while True:
            op = image.data[cursor]
            if op not in range(5):
                raise ValueError("invalid original course opcode")
            cursor += 4 if op in (2, 3) else 1
            if cursor > COURSE_END:
                raise ValueError("course exceeded audited end marker")
            if op == 0:
                break
        boundaries[cursor] = row
    if cursor != COURSE_END or image.data[cursor] != 0:
        raise ValueError("course boundaries changed")
    return boundaries


def visible_loss_pointer(env, boundaries):
    pointer = env.trs.ram.peek(POINTER_LOW) | (env.trs.ram.peek(POINTER_LOW+1) << 8)
    if pointer not in boundaries:
        raise RuntimeError("live stage-one pointer is not an audited row boundary")
    return dict(next_stream_byte=f"0x{pointer:04X}", decoded_rows=boundaries[pointer])


def probe_bundle(bundle, boundaries):
    report, frames, actions = analyze(bundle)
    with np.load(Path(bundle)/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]
        metadata = json.loads(str(trace["metadata"]))
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        losses = []
        for index, action in enumerate(actions):
            obs, reward, terminal, truncated, info = env.step(int(action))
            if reward != rewards[index] or not np.array_equal(obs[-1], frames[index+1]):
                raise RuntimeError("frozen own replay differs from original verified trace")
            if info["life_lost"]:
                losses.append(dict(frame=index+1, displayed_score=info["score"],
                                   **visible_loss_pointer(env, boundaries)))
            if (terminal or truncated) and index+1 != len(actions):
                raise RuntimeError("own replay ended early")
    finally:
        env.close()
    expected = [(life["visible_loss_frame"], life["visible_score_at_loss"])
                for life in report["lives"]]
    if [(row["frame"], row["displayed_score"]) for row in losses] != expected:
        raise RuntimeError("native loss events disagree with verified replay metadata")
    return dict(bundle=str(Path(bundle).resolve()),
                trace_sha256=report["source_hashes"]["trace.npz"],
                model_sha256=report["source_hashes"]["model.safetensors"],
                verified_neural_actions=len(actions), losses=losses)


def probe_sources(archive, boundaries):
    archive = Path(archive).resolve(strict=True)
    index = json.loads((archive/"index.json").read_text())
    env = DefenseEnv(tstates=100_000, max_steps=0, observation_stride=1)
    rows, prior = [], {}
    try:
        for name in index["files"]:
            saved, actions, rewards, screens, metadata = load_source(archive/name)
            obs = restore(env, saved)
            for offset, (action, reward, screen) in enumerate(zip(
                    actions, rewards, screens, strict=True)):
                obs, actual, terminal, truncated, info = env.step(int(action))
                if actual != reward or not np.array_equal(obs[-1], screen):
                    raise RuntimeError("own source prefix failed exact reexecution")
                if info["life_lost"] and offset+1 != len(actions):
                    raise RuntimeError("source life ended before recorded boundary")
            if not info["life_lost"] or info["stage"] != 1:
                raise RuntimeError("source did not end at its stage-one visible loss")
            source = metadata["source"]
            seed = source["seed"]
            base = prior.get(seed, 0)
            if info["score"] != source["visible_loss_score"] or info["score"] < base:
                raise RuntimeError("visible source score mismatch")
            prior[seed] = info["score"]
            rows.append(dict(source=name, source_npz_sha256=metadata["npz_sha256"],
                             seed=seed, life=source["life"],
                             displayed_life_score=info["score"]-base,
                             **visible_loss_pointer(env, boundaries)))
    finally:
        env.close()
    return dict(archive=str(archive), index_sha256=sha256(archive/"index.json"),
                own_games=len(index["games"]), own_life_states=len(rows),
                decoded_row_histogram=dict(sorted(Counter(row["decoded_rows"] for row in rows).items())),
                losses=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path, nargs="+")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing forensic directory")
    boundaries = course_rows()
    bundles = [probe_bundle(bundle, boundaries) for bundle in args.bundles]
    sources = probe_sources(args.source_archive, boundaries) if args.source_archive else None
    output = dict(game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  original_binary_sha256=sha256("var/defense.cmd"),
                  static_audit_source_sha256=sha256(Path(__file__).with_name("defense_audit.py")),
                  probe_source_sha256=sha256(Path(__file__)),
                  course_start=f"0x{COURSE_START:04X}", course_end=f"0x{COURSE_END:04X}",
                  total_stage_one_stream_rows=COURSE_ROWS,
                  memory_address_read=f"0x{POINTER_LOW:04X}/0x{POINTER_LOW+1:04X}",
                  bundles=bundles, own_source_archive=sources,
                  diagnostic_only=True, hidden_ram_never_policy_input=True,
                  hidden_ram_never_reward_or_curriculum=True,
                  model_updates=0, replay_promotion_eligible=False,
                  limitations=["Pointer is sampled at visible ship-loss bookkeeping, not exact collision.",
                               "Decoded rows are stream progress, not rows safely navigated by a ship.",
                               "The original program may render obstacles ahead of or behind this pointer.",
                               "The forensic pointer and row index must never enter training or evaluation selection."])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    write_json(args.output/"report.json", output)
    print(json.dumps(dict(total_rows=COURSE_ROWS,
                          bundle_loss_rows=[[row["decoded_rows"] for row in item["losses"]]
                                            for item in bundles],
                          source_row_histogram=sources["decoded_row_histogram"] if sources else None)))


if __name__ == "__main__":
    main()
