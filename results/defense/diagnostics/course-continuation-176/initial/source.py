"""Forensic-only search beyond the verified first-ship row-50 route.

Private game state rejects/ranks search branches. No path, hidden byte or
snapshot produced here may enter a learner, reward or neural replay promoter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_course_feasibility import verify_candidate
from .defense_course_progress_probe import course_rows, visible_loss_pointer
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_position_feasibility import EXPECTED_MODEL_SHA256, make_node, private_values, select
from .defense_screen_beam import SIDE_FIRE_COMMANDS
from .defense_snapshot import restore
from .defense_world_data import disk_guard


SOURCE_FRAME = 573
SOURCE_FORK = 280
SOURCE_ROWS = 50
SOURCE_SCORE = 450
SOURCE_POSITION = 108


def load_source(bundle, discovery):
    """Require the protected neural prefix and exact independently verified route."""
    report, frames, learned_actions = analyze(bundle)
    if (report["source_hashes"]["model.safetensors"] != EXPECTED_MODEL_SHA256
            or report["tstates"] != 100_000 or report["result"]["score"] != 10_480):
        raise ValueError("requires protected original-cadence neural replay")
    with np.load(Path(bundle) / "trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    directory = Path(discovery)
    prior_report = json.loads((directory / "report.json").read_text())
    proof = json.loads((directory / "discovery.json").read_text())
    record = directory / "diagnostic-discovery.npz"
    if (prior_report["config"]["game_sha256"] != GAME_SHA256
            or prior_report["config"]["model_sha256"] != EXPECTED_MODEL_SHA256
            or prior_report["config"]["trace_sha256"] != report["source_hashes"]["trace.npz"]
            or sha256(directory / "source.py") != prior_report["config"]["source_sha256"]
            or sha256(record) != proof["diagnostic_trace_sha256"]
            or prior_report["discovery"] != proof
            or not proof["verification"]["every_screen_and_reward_matched"]):
        raise ValueError("row-50 forensic source archive/proof mismatch")
    with np.load(record, allow_pickle=False) as data:
        actions = np.asarray(data["actions"], np.uint8).copy()
        diagnostic_only = bool(data["diagnostic_only"])
    outcome = proof["outcome"]
    if (not diagnostic_only or actions.ndim != 1 or len(actions) != SOURCE_FRAME
            or not np.array_equal(actions[:SOURCE_FORK], learned_actions[:SOURCE_FORK])
            or any(int(action) < 0 or int(action) >= 20 for action in actions)
            or hashlib.sha256(actions.tobytes()).hexdigest() != outcome["action_sha256"]
            or (outcome["frame"], outcome["rows"], outcome["stage"],
                outcome["score"], outcome["position"], outcome["first_ship_survived"])
            != (SOURCE_FRAME, SOURCE_ROWS, 1, SOURCE_SCORE, SOURCE_POSITION, True)):
        raise ValueError("row-50 forensic path differs from verified endpoint")
    return report, frames, learned_actions[:SOURCE_FORK].copy(), rewards[:SOURCE_FORK], metadata, actions


def run(bundle, discovery, output, *, beam_width=512, target_rows=75,
        max_frame=900, seed=176):
    bundle, discovery, output = Path(bundle), Path(discovery), Path(output)
    if (output.exists() or not 1 <= beam_width <= 512
            or not SOURCE_ROWS < target_rows <= 126
            or not SOURCE_FRAME < max_frame <= 2_000 or seed < 0):
        raise ValueError("fresh output, beam 1..512, later row/frame and seed required")
    report, frames, prefix, rewards, metadata, source_actions = load_source(bundle, discovery)
    boundaries = course_rows()
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    config = dict(bundle=str(bundle.resolve()), discovery=str(discovery.resolve()),
                  model_sha256=report["source_hashes"]["model.safetensors"],
                  trace_sha256=report["source_hashes"]["trace.npz"],
                  source_action_record_sha256=sha256(discovery / "diagnostic-discovery.npz"),
                  source_sha256=sha256(__file__), game_sha256=GAME_SHA256,
                  source_frame=SOURCE_FRAME, source_rows=SOURCE_ROWS,
                  target_rows=target_rows, max_frame=max_frame,
                  beam=beam_width, seed=seed, commands=list(SIDE_FIRE_COMMANDS),
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  promotion_eligible=False, hidden_ram_used_by_learner=False)
    write_json(output / "config.json", config)
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    rng = np.random.default_rng(seed)
    try:
        initial_expected = dict(position=SOURCE_POSITION, score=SOURCE_SCORE,
                                stage=1, rows=SOURCE_ROWS)
        source_proof = verify_candidate(env, metadata["result"]["seed"], frames,
                                        prefix, rewards, source_actions, boundaries,
                                        initial_expected)
        current = [make_node(env, b"")]
        expanded = dead = 0
        max_live_frame, max_live_rows = SOURCE_FRAME, SOURCE_ROWS
        discovery_result = last = None
        with (output / "layers.jsonl").open("x", buffering=1) as stream:
            for frame in range(SOURCE_FRAME + 1, max_frame + 1):
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
                        max_live_frame = frame
                        node = make_node(env, parent.path + bytes([action]))
                        rows = (visible_loss_pointer(env, boundaries)["decoded_rows"]
                                if info["stage"] == 1 else 126)
                        layer_rows = rows if layer_rows is None else max(layer_rows, rows)
                        max_live_rows = max(max_live_rows, rows)
                        if info["stage"] >= 2 or rows >= target_rows:
                            full_path = np.concatenate((source_actions,
                                                        np.frombuffer(node.path, np.uint8)))
                            expected = dict(frame=frame, rows=rows, stage=info["stage"],
                                            score=node.score, position=node.position,
                                            first_ship_survived=True,
                                            action_sha256=hashlib.sha256(full_path.tobytes()).hexdigest())
                            verification = verify_candidate(
                                env, metadata["result"]["seed"], frames, prefix, rewards,
                                full_path, boundaries, expected)
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
                last = dict(frame=frame, expanded=expanded, private_deaths=dead,
                            layer_private_deaths=layer_dead, generated_alive=len(children),
                            selected=0 if discovery_result else len(current),
                            max_live_frame=max_live_frame, max_live_rows=max_live_rows,
                            layer_max_rows=layer_rows,
                            selected_max_score=max((n.score for n in current), default=None)
                            if not discovery_result else None,
                            selected_min_position=min((n.position for n in current), default=None)
                            if not discovery_result else None,
                            selected_max_position=max((n.position for n in current), default=None)
                            if not discovery_result else None,
                            discovery=bool(discovery_result))
                stream.write(json.dumps(last) + "\n")
                write_json(output / "status.json", last)
                if frame % 8 == 0 or discovery_result:
                    print(json.dumps(last), flush=True)
                    disk_guard(output)
                if discovery_result:
                    break
        result = dict(config=config, source_verification=source_proof, final=last,
                      expanded=expanded, private_deaths=dead, discovery=discovery_result,
                      layers_sha256=sha256(output / "layers.jsonl"),
                      limitations=["A finite beam from one searched source can miss passable routes.",
                                   "Decoded stream row is not stage passage or proof of safe geometry.",
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
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--target-rows", type=int, default=75)
    parser.add_argument("--max-frame", type=int, default=900)
    parser.add_argument("--seed", type=int, default=176)
    args = parser.parse_args()
    result = run(args.bundle, args.discovery, args.output, beam_width=args.beam,
                 target_rows=args.target_rows, max_frame=args.max_frame, seed=args.seed)
    print(json.dumps(dict(event="finished", **result["final"],
                          found=bool(result["discovery"]))), flush=True)


if __name__ == "__main__":
    main()
