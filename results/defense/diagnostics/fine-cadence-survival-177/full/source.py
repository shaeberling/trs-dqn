"""Quarantined physical feasibility test after the verified row-50 route.

The source and any result were selected with private game RAM. Neither actions,
snapshots nor private measurements from this module may enter a learner, its
reward, checkpoint selection, or a promoted neural replay.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_course_continuation import (
    SOURCE_FORK, SOURCE_FRAME, SOURCE_POSITION, SOURCE_ROWS, SOURCE_SCORE,
    load_source,
)
from .defense_course_feasibility import verify_candidate
from .defense_course_progress_probe import course_rows, visible_loss_pointer
from .defense_learning import sha256, write_json
from .defense_position_feasibility import make_node, private_values, select
from .defense_screen_beam import SIDE_FIRE_COMMANDS
from .defense_snapshot import restore
from .defense_world_data import disk_guard


SOURCE_TSTATES = 100_000
FINE_TSTATES = 50_000


def verify_mixed(env, seed, frames, prefix, rewards, source_actions, fine_actions,
                 boundaries, expected):
    """Replay the mixed-cadence path twice from original boot, every screen/reward."""
    records = None
    for _ in range(2):
        env.tstates = SOURCE_TSTATES
        obs = env.reset(seed)
        np.testing.assert_array_equal(obs[-1], frames[0])
        observed = []
        for index, action in enumerate(source_actions):
            obs, reward, terminal, truncated, info = env.step(int(action))
            if terminal or truncated or private_values(env)[1] != 4:
                raise RuntimeError("forensic source lost first ship during mixed replay")
            if index < SOURCE_FORK:
                if reward != rewards[index] or info["life_lost"]:
                    raise RuntimeError("protected neural prefix reward changed")
                np.testing.assert_array_equal(obs[-1], frames[index + 1])
            observed.append((obs[-1].tobytes(), reward))
        position, ships = private_values(env)
        if ((position, ships, env.score, env.stage,
             visible_loss_pointer(env, boundaries)["decoded_rows"]) !=
                (SOURCE_POSITION, 4, SOURCE_SCORE, 1, SOURCE_ROWS)):
            raise RuntimeError("row-50 source endpoint changed")
        env.tstates = FINE_TSTATES
        for action in fine_actions:
            obs, reward, terminal, truncated, info = env.step(int(action))
            if terminal or truncated or private_values(env)[1] != 4:
                raise RuntimeError("mixed-cadence result lost first ship")
            observed.append((obs[-1].tobytes(), reward))
        position, ships = private_values(env)
        rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                if info["stage"] == 1 else 126)
        if ((position, ships, env.score, info["stage"], rows) !=
                (expected["position"], 4, expected["score"],
                 expected["stage"], expected["rows"])):
            raise RuntimeError("mixed-cadence endpoint changed")
        if records is not None and observed != records:
            raise RuntimeError("mixed-cadence screens or rewards changed across boots")
        records = observed
    return dict(verified=True, original_boot_runs=2,
                original_cadence_actions=len(source_actions), fine_actions=len(fine_actions),
                every_screen_and_reward_matched=True, diagnostic_only=True,
                promotion_eligible=False)


def run(bundle, discovery, output, *, beam_width=256, target_rows=65,
        max_fine_actions=400, seed=177):
    bundle, discovery, output = Path(bundle), Path(discovery), Path(output)
    if (output.exists() or not 1 <= beam_width <= 512
            or not SOURCE_ROWS < target_rows <= 126
            or not 1 <= max_fine_actions <= 800 or seed < 0):
        raise ValueError("fresh output, beam 1..512, later row, fine-action cap and seed required")
    report, frames, prefix, rewards, metadata, source_actions = load_source(bundle, discovery)
    if report["tstates"] != SOURCE_TSTATES or len(source_actions) != SOURCE_FRAME:
        raise ValueError("source must be the verified original-cadence row-50 route")
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    config = dict(bundle=str(bundle.resolve()), discovery=str(discovery.resolve()),
                  model_sha256=report["source_hashes"]["model.safetensors"],
                  trace_sha256=report["source_hashes"]["trace.npz"],
                  source_action_record_sha256=sha256(discovery / "diagnostic-discovery.npz"),
                  source_sha256=sha256(__file__), game_sha256=GAME_SHA256,
                  source_frame=SOURCE_FRAME, source_rows=SOURCE_ROWS,
                  source_tstates=SOURCE_TSTATES, fine_tstates=FINE_TSTATES,
                  target_rows=target_rows, max_fine_actions=max_fine_actions,
                  beam=beam_width, seed=seed, commands=list(SIDE_FIRE_COMMANDS),
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  promotion_eligible=False, hidden_ram_used_by_learner=False)
    write_json(output / "config.json", config)
    env = DefenseEnv(tstates=SOURCE_TSTATES, max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    rng = np.random.default_rng(seed)
    boundaries = course_rows()
    try:
        expected_source = dict(position=SOURCE_POSITION, score=SOURCE_SCORE,
                               stage=1, rows=SOURCE_ROWS)
        source_proof = verify_candidate(env, metadata["result"]["seed"], frames,
                                        prefix, rewards, source_actions, boundaries,
                                        expected_source)
        env.tstates = FINE_TSTATES
        current = [make_node(env, b"")]
        expanded = dead = 0
        max_live_rows = SOURCE_ROWS
        discovery_result = last = None
        frontier_node = None
        with (output / "layers.jsonl").open("x", buffering=1) as stream:
            for fine_action in range(1, max_fine_actions + 1):
                if not current:
                    break
                children = []
                layer_dead = 0
                layer_rows = None
                for parent in current:
                    for action in SIDE_FIRE_COMMANDS:
                        restore(env, parent.saved)
                        _, _, terminal, truncated, info = env.step(int(action))
                        expanded += 1
                        if terminal or truncated or private_values(env)[1] != 4:
                            dead += 1
                            layer_dead += 1
                            continue
                        node = make_node(env, parent.path + bytes([action]))
                        rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                                if info["stage"] == 1 else 126)
                        layer_rows = rows if layer_rows is None else max(layer_rows, rows)
                        max_live_rows = max(max_live_rows, rows)
                        if info["stage"] >= 2 or rows >= target_rows:
                            fine_path = np.frombuffer(node.path, np.uint8)
                            full_path = np.concatenate((source_actions, fine_path))
                            expected = dict(fine_actions=fine_action, rows=rows,
                                            stage=info["stage"], score=node.score,
                                            position=node.position, first_ship_survived=True,
                                            action_sha256=hashlib.sha256(full_path.tobytes()).hexdigest())
                            verification = verify_mixed(
                                env, metadata["result"]["seed"], frames, prefix,
                                rewards, source_actions, fine_path, boundaries, expected)
                            np.savez_compressed(output / "diagnostic-discovery.npz",
                                                actions=full_path, diagnostic_only=np.array(True))
                            discovery_result = dict(outcome=expected, verification=verification,
                                                    diagnostic_trace_sha256=sha256(
                                                        output / "diagnostic-discovery.npz"))
                            write_json(output / "discovery.json", discovery_result)
                            break
                        children.append(node)
                    if discovery_result:
                        break
                if not discovery_result:
                    current = select(children, beam_width, rng) if children else []
                    if current:
                        frontier_node = max(current, key=lambda n: (n.score, n.position))
                last = dict(fine_action=fine_action, expanded=expanded,
                            private_deaths=dead, layer_private_deaths=layer_dead,
                            generated_alive=len(children),
                            selected=0 if discovery_result else len(current),
                            max_live_rows=max_live_rows, layer_max_rows=layer_rows,
                            selected_max_score=(max(n.score for n in current)
                                                if current and not discovery_result else None),
                            discovery=bool(discovery_result))
                stream.write(json.dumps(last) + "\n")
                write_json(output / "status.json", last)
                if fine_action % 8 == 0 or discovery_result:
                    print(json.dumps(last), flush=True)
                    disk_guard(output)
                if discovery_result:
                    break
        frontier_witness = None
        if discovery_result is None and frontier_node is not None:
            restore(env, frontier_node.saved)
            rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                    if env.stage == 1 else 126)
            fine_path = np.frombuffer(frontier_node.path, np.uint8)
            full_path = np.concatenate((source_actions, fine_path))
            expected = dict(fine_actions=len(fine_path), rows=rows, stage=env.stage,
                            score=frontier_node.score, position=frontier_node.position,
                            first_ship_survived=True,
                            action_sha256=hashlib.sha256(full_path.tobytes()).hexdigest())
            verification = verify_mixed(env, metadata["result"]["seed"], frames,
                                        prefix, rewards, source_actions, fine_path,
                                        boundaries, expected)
            np.savez_compressed(output / "frontier-witness.npz",
                                actions=full_path, diagnostic_only=np.array(True))
            frontier_witness = dict(outcome=expected, verification=verification,
                                    diagnostic_trace_sha256=sha256(output / "frontier-witness.npz"))
            write_json(output / "frontier-witness.json", frontier_witness)
        result = dict(config=config, source_verification=source_proof, final=last,
                      expanded=expanded, private_deaths=dead,
                      discovery=discovery_result, frontier_witness=frontier_witness,
                      layers_sha256=sha256(output / "layers.jsonl"),
                      limitations=["One searched source and a finite beam cannot prove stage reachability or impossibility.",
                                   "Decoded stream row is not visible stage passage.",
                                   "Private RAM and searched actions remain isolated from learning."])
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
    parser.add_argument("--target-rows", type=int, default=65)
    parser.add_argument("--max-fine-actions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=177)
    args = parser.parse_args()
    result = run(args.bundle, args.discovery, args.output, beam_width=args.beam,
                 target_rows=args.target_rows, max_fine_actions=args.max_fine_actions,
                 seed=args.seed)
    print(json.dumps(dict(event="finished", **result["final"],
                          found=bool(result["discovery"]))), flush=True)


if __name__ == "__main__":
    main()
