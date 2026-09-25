"""Forensic-only survival search after first-gate prepositioning.

Private game bytes select diagnostic branches and reject already-dead ships.
No branch, action, private value or snapshot is available to a learner.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_position_feasibility import (
    ANCHOR, EXPECTED_MODEL_SHA256, PRIVATE_POSITION, PRIVATE_SHIPS,
    SCORE_TARGET, TARGET_FRAME, make_node, private_values, select,
)
from .defense_screen_beam import SIDE_FIRE_COMMANDS
from .defense_snapshot import restore
from .defense_world_data import disk_guard


SURVIVAL_FRAME = 428
SCHEDULE = (4,) * 10 + (1,) * 19 + (2,) * 20 + (1,) * 49
assert ANCHOR + sum(SCHEDULE) == SURVIVAL_FRAME


def frame_numbers(schedule=SCHEDULE):
    frames = [ANCHOR]
    for duration in schedule:
        if duration not in (1, 2, 4):
            raise ValueError("invalid diagnostic action hold")
        frames.append(frames[-1] + duration)
    return frames


def verify_discovery(env, seed, frames, actions, rewards, anchor, path, expected):
    """Replay the candidate twice, including a fresh original boot.

    The first pass records its screens and rewards from the already-verified
    source anchor. The second proves that an original-boot execution agrees.
    """
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], frames[0])
    for index in range(anchor):
        obs, reward, terminal, truncated, info = env.step(int(actions[index]))
        if reward != rewards[index] or terminal or truncated or info["life_lost"]:
            raise RuntimeError("source prefix changed before diagnostic anchor")
        np.testing.assert_array_equal(obs[-1], frames[index + 1])
    anchor_obs = obs[-1].copy()
    candidate_frames, candidate_rewards = [anchor_obs], []
    for action in path:
        obs, reward, terminal, truncated, info = env.step(int(action))
        if terminal or truncated:
            raise RuntimeError("candidate terminated before survival target")
        candidate_frames.append(obs[-1].copy())
        candidate_rewards.append(reward)
    position, ships = private_values(env)
    if (ships != 4 or info["score"] != expected["score"]
            or info["stage"] != expected["stage"]
            or position != expected["private_position"]):
        raise RuntimeError("candidate changed on original-boot verification")
    # Reboot again and compare *every* diagnostic screen/reward, not only end state.
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], frames[0])
    for index in range(anchor):
        obs, reward, terminal, truncated, info = env.step(int(actions[index]))
        if reward != rewards[index] or terminal or truncated or info["life_lost"]:
            raise RuntimeError("second source prefix changed")
        np.testing.assert_array_equal(obs[-1], frames[index + 1])
    for index, action in enumerate(path):
        obs, reward, terminal, truncated, info = env.step(int(action))
        if reward != candidate_rewards[index] or terminal or truncated:
            raise RuntimeError("candidate reward/boundary changed")
        np.testing.assert_array_equal(obs[-1], candidate_frames[index + 1])
    if private_values(env) != (expected["private_position"], 4):
        raise RuntimeError("private survivor state changed")
    return dict(verified=True, original_boot_actions=anchor + len(path),
                repeated_original_boot_runs=2, all_screens_and_rewards_matched=True,
                diagnostic_only=True, promotion_eligible=False)


def run(bundle, output, *, beam_width=128, layers=0, seed=172):
    bundle, output = Path(bundle), Path(output)
    if (output.exists() or not 1 <= beam_width <= 512 or
            not 0 <= layers <= len(SCHEDULE) or seed < 0):
        raise ValueError("fresh output, beam 1..512, valid layer count and seed required")
    report, frames, actions = analyze(bundle)
    if (report["source_hashes"]["model.safetensors"] != EXPECTED_MODEL_SHA256
            or report["lives"][0]["visible_loss_frame"] <= TARGET_FRAME
            or report["result"]["score"] != 10480 or report["tstates"] != 100000):
        raise ValueError("requires the protected verified first life and action cadence")
    with np.load(bundle / "trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    numbered = frame_numbers()
    requested = layers or len(SCHEDULE)
    config = dict(bundle=str(bundle.resolve()), model_sha256=sha256(bundle / "model.safetensors"),
                  trace_sha256=sha256(bundle / "trace.npz"), source_sha256=sha256(__file__),
                  game_sha256=GAME_SHA256, anchor=ANCHOR, target_frame=SURVIVAL_FRAME,
                  schedule=list(SCHEDULE), beam=beam_width, requested_layers=requested,
                  seed=seed, commands=list(SIDE_FIRE_COMMANDS),
                  private_position_address=hex(PRIVATE_POSITION),
                  private_ship_count_address=hex(PRIVATE_SHIPS),
                  diagnostic_only=True, model_updates=0, training_data_written=False,
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
        wanted = set(numbered[:requested + 1]) & set(range(ANCHOR, TARGET_FRAME + 1))
        source = {}
        for index in range(TARGET_FRAME):
            if index in wanted:
                source[index] = make_node(env, bytes(actions[ANCHOR:index]), source=True)
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            if reward != rewards[index] or terminal or truncated or info["life_lost"]:
                raise RuntimeError("original-boot source prefix diverged")
            np.testing.assert_array_equal(obs[-1], frames[index + 1])
        if TARGET_FRAME in wanted:
            source[TARGET_FRAME] = make_node(
                env, bytes(actions[ANCHOR:TARGET_FRAME]), source=True)
        current = [source[ANCHOR]]
        expanded = dead = 0
        max_live_frame = ANCHOR
        discovery = None
        last_row = None
        with (output / "layers.jsonl").open("x", buffering=1) as stream:
            for layer, duration in enumerate(SCHEDULE[:requested], 1):
                if not current:
                    break
                children = []
                layer_dead = 0
                for parent in current:
                    for action in SIDE_FIRE_COMMANDS:
                        restore(env, parent.saved)
                        live = True
                        for offset in range(duration):
                            obs, _, terminal, truncated, info = env.step(int(action))
                            _, ships = private_values(env)
                            if ships != 4 or terminal or truncated or info["life_lost"]:
                                live = False
                                dead += 1
                                layer_dead += 1
                                break
                            max_live_frame = max(max_live_frame, numbered[layer - 1] + offset + 1)
                        expanded += 1
                        if not live:
                            continue
                        node = make_node(env, parent.path + bytes([action]) * duration)
                        if info["stage"] >= 2 or info["missions_completed"] or numbered[layer] >= SURVIVAL_FRAME:
                            expected = dict(score=node.score, stage=info["stage"],
                                            private_position=node.position,
                                            first_ship_survived=True, frame=numbered[layer],
                                            action_sha256=hashlib.sha256(node.path).hexdigest())
                            proof = verify_discovery(env, metadata["result"]["seed"],
                                                     frames, actions, rewards, ANCHOR,
                                                     node.path, expected)
                            discovery = dict(outcome=expected, verification=proof)
                            # A found path is quarantined as forensic evidence, never a learner replay.
                            np.savez_compressed(output / "diagnostic-discovery.npz",
                                                actions=np.frombuffer(node.path, np.uint8).copy(),
                                                diagnostic_only=np.array(True))
                            discovery["diagnostic_trace_sha256"] = sha256(
                                output / "diagnostic-discovery.npz")
                            break
                        children.append(node)
                    if discovery:
                        break
                last_row = dict(layer=layer, frame=numbered[layer], duration=duration,
                                expanded=expanded, private_deaths=dead,
                                layer_private_deaths=layer_dead,
                                generated_alive=len(children), max_live_frame=max_live_frame,
                                generated_max_score=max((n.score for n in children), default=None),
                                generated_max_position=max((n.position for n in children), default=None),
                                generated_max_position_at_score_target=max(
                                    (n.position for n in children if n.score >= SCORE_TARGET), default=None),
                                discovery=bool(discovery))
                if not discovery:
                    mandatory = source.get(numbered[layer])
                    current = (select(children, beam_width, rng, mandatory=mandatory)
                               if children or mandatory is not None else [])
                    last_row["selected"] = len(current)
                    last_row["selected_max_score"] = max((n.score for n in current), default=None)
                    last_row["selected_max_position"] = max((n.position for n in current), default=None)
                stream.write(json.dumps(last_row) + "\n")
                write_json(output / "status.json", last_row)
                print(json.dumps(last_row), flush=True)
                disk_guard(output)
                if discovery:
                    write_json(output / "discovery.json", discovery)
                    break
        result = dict(config=config, completed_layers=last_row["layer"] if last_row else 0,
                      final_frame=last_row["frame"] if last_row else ANCHOR,
                      expanded=expanded, private_deaths=dead,
                      max_live_frame=max_live_frame, discovery=discovery,
                      layers_sha256=sha256(output / "layers.jsonl"),
                      original_boot_source_prefix_verified=TARGET_FRAME,
                      limitations=["A finite beam from one verified first life can miss passable routes.",
                                   "Private RAM selects only forensic branches, never learner actions.",
                                   "Survival past frame 428 is not a stage clear or successful mission."])
        write_json(output / "report.json", result)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=128)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=172)
    args = parser.parse_args()
    result = run(args.bundle, args.output, beam_width=args.beam,
                 layers=args.layers, seed=args.seed)
    print(json.dumps(dict(event="finished", completed_layers=result["completed_layers"],
                          final_frame=result["final_frame"], expanded=result["expanded"],
                          discovery=bool(result["discovery"]))), flush=True)


if __name__ == "__main__":
    main()
