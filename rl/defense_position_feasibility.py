"""Forensic-only first-gate position search; NEVER a policy or training source.

This probe deliberately reads private game RAM to test physical feasibility.
Its states, keys, private measurements and paths must not enter a learner,
reward, curriculum, checkpoint selector or published neural replay.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_screen_beam import SIDE_FIRE_COMMANDS
from .defense_snapshot import capture, restore
from .defense_world_data import disk_guard


ANCHOR = 280
SCHEDULE = (4,) * 10 + (1,) * 19  # First-gate approach through frame 339.
TARGET_FRAME = ANCHOR + sum(SCHEDULE)
PRIVATE_POSITION = 0x5E03
PRIVATE_SHIPS = 0x7CEF
POSITION_TARGET = 60  # Forensic byte threshold, not a policy feature.
SCORE_TARGET = 1080  # Visible first-obstacle score in this exact source life.
EXPECTED_MODEL_SHA256 = "125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171"


@dataclass
class Node:
    saved: object
    path: bytes
    score: int
    position: int
    screen: bytes
    source: bool = False


def private_values(env):
    ram = env.trs.ram
    return int(ram.peek(PRIVATE_POSITION)), int(ram.peek(PRIVATE_SHIPS))


def make_node(env, path, source=False):
    position, ships = private_values(env)
    if ships != 4 or env.lives != 4:
        raise RuntimeError("candidate has already lost its first ship")
    return Node(capture(env), path, int(env.score), position,
                env.observation()[-1].tobytes(), source)


def select(nodes, limit, rng, mandatory=None):
    """Reserve score, right-position and score/position-diversity quotas."""
    if not 1 <= limit <= 512:
        raise ValueError("beam must be 1..512")
    # Full native state distinguishes occluded but physically different paths.
    unique = {}
    for node in nodes:
        key = hashlib.blake2s(node.saved.native, digest_size=16).digest()
        prior = unique.get(key)
        if prior is None or (node.score, node.position) > (prior.score, prior.position):
            unique[key] = node
    pool = list(unique.values())
    rng.shuffle(pool)
    chosen = [mandatory] if mandatory is not None else []
    used = {id(mandatory)} if mandatory is not None else set()

    def take(ordered, target):
        for node in ordered:
            if len(chosen) >= target:
                break
            if id(node) not in used:
                chosen.append(node)
                used.add(id(node))

    take(sorted(pool, key=lambda n: (n.score, n.position), reverse=True), max(1, limit // 3))
    take(sorted(pool, key=lambda n: (n.position, n.score), reverse=True),
         min(limit, max(2, 2 * limit // 3)))
    buckets = {}
    for node in pool:
        buckets.setdefault((node.score // 100, node.position // 4), []).append(node)
    keys = list(buckets)
    rng.shuffle(keys)
    while len(chosen) < limit and any(buckets.values()):
        for key in keys:
            bucket = buckets[key]
            while bucket and id(bucket[-1]) in used:
                bucket.pop()
            if bucket:
                node = bucket.pop()
                chosen.append(node)
                used.add(id(node))
                if len(chosen) >= limit:
                    break
    return chosen


def run(bundle, output, beam_width=256, layers=0, seed=171):
    bundle, output = Path(bundle), Path(output)
    if (output.exists() or not 1 <= beam_width <= 512 or
            not 0 <= layers <= len(SCHEDULE) or seed < 0):
        raise ValueError("fresh output, beam 1..512, valid layer count and seed required")
    report, frames, actions = analyze(bundle)
    if (report["source_hashes"]["model.safetensors"] != EXPECTED_MODEL_SHA256
            or report["lives"][0]["visible_loss_frame"] <= TARGET_FRAME
            or report["result"]["score"] != 10480
            or report["tstates"] != 100000):
        raise ValueError("requires the protected verified 100,000-T-state first life")
    with np.load(bundle / "trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    frame_numbers = [ANCHOR]
    for duration in SCHEDULE:
        frame_numbers.append(frame_numbers[-1] + duration)
    config = dict(bundle=str(bundle.resolve()), model_sha256=sha256(bundle / "model.safetensors"),
                  trace_sha256=sha256(bundle / "trace.npz"), source_sha256=sha256(__file__),
                  game_sha256=GAME_SHA256, anchor=ANCHOR, target_frame=TARGET_FRAME,
                  schedule=list(SCHEDULE), beam=beam_width,
                  requested_layers=layers or len(SCHEDULE), seed=seed,
                  commands=list(SIDE_FIRE_COMMANDS), private_position_address="0x5E03",
                  private_ship_count_address="0x7CEF", position_target=POSITION_TARGET,
                  visible_score_target=SCORE_TARGET, diagnostic_only=True,
                  model_updates=0, training_data_written=False,
                  promotion_eligible=False, hidden_ram_used_by_learner=False)
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    write_json(output / "config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    rng = np.random.default_rng(seed)
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        wanted = set(frame_numbers)
        source = {}
        for index in range(TARGET_FRAME):
            if index in wanted:
                source[index] = make_node(env, bytes(actions[ANCHOR:index]), source=True)
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            if (reward != rewards[index] or terminal or truncated or info["life_lost"]):
                raise RuntimeError("original-boot source prefix diverged")
            np.testing.assert_array_equal(obs[-1], frames[index + 1])
        source[TARGET_FRAME] = make_node(env, bytes(actions[ANCHOR:TARGET_FRAME]), source=True)
        current = [source[ANCHOR]]
        expanded = dead = completed = 0
        with (output / "layers.jsonl").open("x", buffering=1) as stream:
            for layer, duration in enumerate(SCHEDULE[:config["requested_layers"]], 1):
                children = []
                for parent in current:
                    for action in SIDE_FIRE_COMMANDS:
                        restore(env, parent.saved)
                        live = True
                        for _ in range(duration):
                            obs, _, terminal, truncated, info = env.step(int(action))
                            _, ships = private_values(env)
                            if ships != 4 or terminal or truncated or info["life_lost"]:
                                live = False
                                dead += 1
                                break
                        expanded += 1
                        if live:
                            children.append(make_node(env, parent.path + bytes([action]) * duration))
                frame = frame_numbers[layer]
                current = select(children, beam_width, rng, mandatory=source[frame])
                complete = frame == TARGET_FRAME
                qualifying = [n for n in children if n.position >= POSITION_TARGET]
                both = [n for n in qualifying if n.score >= SCORE_TARGET]
                row = dict(layer=layer, frame=frame, duration=duration,
                           expanded=expanded, private_deaths=dead,
                           generated_alive=len(children), selected=len(current),
                           selected_max_score=max(n.score for n in current),
                           selected_max_position=max(n.position for n in current),
                           selected_max_position_at_score_target=max(
                               (n.position for n in current if n.score >= SCORE_TARGET), default=None),
                           generated_max_position_at_score_target=max(
                               (n.position for n in children if n.score >= SCORE_TARGET), default=None),
                           position_target_count=len(qualifying) if complete else None,
                           both_targets_count=len(both) if complete else None,
                           source_score=source[frame].score,
                           source_position=source[frame].position)
                stream.write(json.dumps(row) + "\n")
                write_json(output / "status.json", row)
                print(json.dumps(row), flush=True)
                completed = layer
                disk_guard(output)
        result = dict(config=config, completed_layers=completed,
                      final_frame=frame_numbers[completed], expanded=expanded,
                      private_deaths=dead, final=row,
                      layers_sha256=sha256(output / "layers.jsonl"),
                      original_boot_prefix_verified=TARGET_FRAME,
                      limitations=["A private RAM byte is used only to rank forensic branches.",
                                   "No search action or native byte may enter training or evaluation policy.",
                                   "A position threshold is not a stage clear or proof of mission feasibility.",
                                   "One fixed source life and finite beam can miss other routes."])
        write_json(output / "report.json", result)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=171)
    args = parser.parse_args()
    result = run(args.bundle, args.output, args.beam, args.layers, args.seed)
    print(json.dumps(dict(event="finished", **{k: result[k] for k in
                                              ("completed_layers", "final_frame", "expanded")})), flush=True)


if __name__ == "__main__":
    main()
