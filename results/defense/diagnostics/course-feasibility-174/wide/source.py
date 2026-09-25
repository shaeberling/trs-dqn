"""Forensic-only continuation of the verified first-ship survival route.

Private bytes select diagnostic branches. This module is deliberately separate
from every learner, evaluator and replay promoter; its actions are not examples.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_course_progress_probe import course_rows, visible_loss_pointer
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_position_feasibility import EXPECTED_MODEL_SHA256, make_node, private_values, select
from .defense_screen_beam import SIDE_FIRE_COMMANDS
from .defense_snapshot import restore
from .defense_world_data import disk_guard


SOURCE_FRAME = 428
SOURCE_ANCHOR = 280
SOURCE_SCORE = 390
SOURCE_POSITION = 92
SOURCE_ROWS = 37


def load_source(bundle, discovery):
    """Bind the starting point to both independently verified prior archives."""
    report, frames, actions = analyze(bundle)
    if (report["source_hashes"]["model.safetensors"] != EXPECTED_MODEL_SHA256
            or report["tstates"] != 100_000 or report["result"]["score"] != 10_480):
        raise ValueError("requires protected verified original-cadence replay")
    with np.load(Path(bundle) / "trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    directory = Path(discovery)
    prior = json.loads((directory / "discovery.json").read_text())
    record = directory / "diagnostic-discovery.npz"
    if sha256(record) != prior["diagnostic_trace_sha256"]:
        raise ValueError("forensic source archive hash mismatch")
    with np.load(record, allow_pickle=False) as data:
        route = np.asarray(data["actions"], np.uint8).copy()
        diagnostic_only = bool(data["diagnostic_only"])
    if (not diagnostic_only or route.ndim != 1 or len(route) != SOURCE_FRAME - SOURCE_ANCHOR
            or any(int(action) not in SIDE_FIRE_COMMANDS for action in route)
            or hashlib.sha256(route.tobytes()).hexdigest() != prior["outcome"]["action_sha256"]
            or prior["outcome"]["frame"] != SOURCE_FRAME
            or prior["outcome"]["score"] != SOURCE_SCORE
            or prior["outcome"]["private_position"] != SOURCE_POSITION
            or not prior["verification"]["all_screens_and_rewards_matched"]):
        raise ValueError("forensic source is not the verified row-37 route")
    return report, frames, actions[:SOURCE_ANCHOR].copy(), rewards[:SOURCE_ANCHOR], metadata, route


def replay_source(env, seed, frames, prefix, rewards, route, boundaries):
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], frames[0])
    for index, action in enumerate(prefix):
        obs, reward, terminal, truncated, info = env.step(int(action))
        if reward != rewards[index] or terminal or truncated or info["life_lost"]:
            raise RuntimeError("protected first-life prefix changed")
        np.testing.assert_array_equal(obs[-1], frames[index + 1])
    for action in route:
        _, _, terminal, truncated, info = env.step(int(action))
        if terminal or truncated or private_values(env)[1] != 4:
            raise RuntimeError("verified forensic source no longer survives")
    position, ships = private_values(env)
    rows = visible_loss_pointer(env, boundaries)["decoded_rows"]
    if (position, ships, env.score, info["stage"], rows) != (
            SOURCE_POSITION, 4, SOURCE_SCORE, 1, SOURCE_ROWS):
        raise RuntimeError("verified forensic source endpoint changed")


def verify_candidate(env, seed, frames, prefix, rewards, path, boundaries, expected):
    """Reboot twice and compare every screen and score reward of the route."""
    records = None
    for repeat in range(2):
        obs = env.reset(seed)
        np.testing.assert_array_equal(obs[-1], frames[0])
        observed = []
        for index, action in enumerate(path):
            obs, reward, terminal, truncated, info = env.step(int(action))
            if index < len(prefix):
                if reward != rewards[index] or info["life_lost"]:
                    raise RuntimeError("protected prefix changed during verification")
                np.testing.assert_array_equal(obs[-1], frames[index + 1])
            if terminal or truncated or private_values(env)[1] != 4:
                raise RuntimeError("forensic route did not retain its first ship")
            observed.append((obs[-1].tobytes(), reward))
        position, ships = private_values(env)
        rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                if info["stage"] == 1 else 126)
        if (position, ships, env.score, info["stage"], rows) != (
                expected["position"], 4, expected["score"], expected["stage"], expected["rows"]):
            raise RuntimeError("forensic endpoint changed during verification")
        if records is not None and observed != records:
            raise RuntimeError("forensic route screens or rewards changed across boots")
        records = observed
    return dict(verified=True, original_boot_runs=2, actions=len(path),
                every_screen_and_reward_matched=True, diagnostic_only=True,
                promotion_eligible=False)


def run(bundle, discovery, output, *, beam_width=256, target_rows=50, max_frame=600, seed=174):
    bundle, discovery, output = Path(bundle), Path(discovery), Path(output)
    if (output.exists() or not 1 <= beam_width <= 512 or not SOURCE_ROWS < target_rows <= 126
            or not SOURCE_FRAME < max_frame <= 2_000 or seed < 0):
        raise ValueError("fresh output, beam 1..512, later course target/frame and seed required")
    report, frames, prefix, rewards, metadata, route = load_source(bundle, discovery)
    boundaries = course_rows()
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    config = dict(bundle=str(bundle.resolve()), discovery=str(discovery.resolve()),
                  model_sha256=report["source_hashes"]["model.safetensors"],
                  trace_sha256=report["source_hashes"]["trace.npz"],
                  discovery_sha256=sha256(discovery / "diagnostic-discovery.npz"),
                  source_sha256=sha256(__file__), game_sha256=GAME_SHA256,
                  source_frame=SOURCE_FRAME, source_rows=SOURCE_ROWS,
                  target_rows=target_rows, max_frame=max_frame, beam=beam_width,
                  seed=seed, commands=list(SIDE_FIRE_COMMANDS),
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  promotion_eligible=False, hidden_ram_used_by_learner=False)
    write_json(output / "config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    rng = np.random.default_rng(seed)
    try:
        replay_source(env, metadata["result"]["seed"], frames, prefix, rewards, route, boundaries)
        current = [make_node(env, b"")]
        expanded = dead = 0
        max_live_frame = SOURCE_FRAME
        max_live_rows = SOURCE_ROWS
        discovery_result = None
        last = None
        with (output / "layers.jsonl").open("x", buffering=1) as stream:
            for frame in range(SOURCE_FRAME + 1, max_frame + 1):
                if not current:
                    break
                children = []
                layer_dead = 0
                layer_rows = SOURCE_ROWS
                for parent in current:
                    for action in SIDE_FIRE_COMMANDS:
                        restore(env, parent.saved)
                        _, _, terminal, truncated, info = env.step(int(action))
                        expanded += 1
                        if terminal or truncated or private_values(env)[1] != 4:
                            dead += 1
                            layer_dead += 1
                            continue
                        max_live_frame = frame
                        node = make_node(env, parent.path + bytes([action]))
                        rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                                if info["stage"] == 1 else 126)
                        layer_rows = max(layer_rows, rows)
                        max_live_rows = max(max_live_rows, rows)
                        if info["stage"] >= 2 or rows >= target_rows:
                            full_path = np.concatenate((prefix, route,
                                                        np.frombuffer(node.path, np.uint8)))
                            expected = dict(frame=frame, rows=rows, stage=info["stage"],
                                            score=node.score, position=node.position,
                                            first_ship_survived=True,
                                            action_sha256=hashlib.sha256(full_path.tobytes()).hexdigest())
                            proof = verify_candidate(env, metadata["result"]["seed"],
                                                     frames, prefix, rewards, full_path,
                                                     boundaries, expected)
                            np.savez_compressed(output / "diagnostic-discovery.npz",
                                                actions=full_path, diagnostic_only=np.array(True))
                            discovery_result = dict(outcome=expected, verification=proof,
                                                    diagnostic_trace_sha256=sha256(
                                                        output / "diagnostic-discovery.npz"))
                            write_json(output / "discovery.json", discovery_result)
                            break
                        children.append(node)
                    if discovery_result:
                        break
                if not discovery_result:
                    current = select(children, beam_width, rng) if children else []
                last = dict(frame=frame, expanded=expanded, private_deaths=dead,
                            layer_private_deaths=layer_dead, generated_alive=len(children),
                            selected=len(current), max_live_frame=max_live_frame,
                            max_live_rows=max_live_rows, layer_max_rows=layer_rows,
                            selected_max_score=max((n.score for n in current), default=None),
                            selected_min_position=min((n.position for n in current), default=None),
                            selected_max_position=max((n.position for n in current), default=None),
                            discovery=bool(discovery_result))
                stream.write(json.dumps(last) + "\n")
                write_json(output / "status.json", last)
                if frame % 8 == 0 or discovery_result:
                    print(json.dumps(last), flush=True)
                    disk_guard(output)
                if discovery_result:
                    break
        result = dict(config=config, final=last, expanded=expanded, private_deaths=dead,
                      discovery=discovery_result, layers_sha256=sha256(output / "layers.jsonl"),
                      original_boot_source_verified=True,
                      limitations=["Finite beam from one private-RAM-selected source can miss a route.",
                                   "Decoded stream row is not proof of safely traversing that obstacle.",
                                   "All searched actions are forensic only, never learned play or training data."])
        write_json(output / "report.json", result)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument("--target-rows", type=int, default=50)
    parser.add_argument("--max-frame", type=int, default=600)
    parser.add_argument("--seed", type=int, default=174)
    args = parser.parse_args()
    result = run(args.bundle, args.discovery, args.output, beam_width=args.beam,
                 target_rows=args.target_rows, max_frame=args.max_frame, seed=args.seed)
    print(json.dumps(dict(event="finished", **result["final"],
                          found=bool(result["discovery"]))), flush=True)


if __name__ == "__main__":
    main()
