"""Diagnostic screen-conditioned beam search from an own verified life.

Every expansion acts in the original emulator. Visible score and raw-screen
diversity select branches; an occasionally visible ship glyph supplies only
a position *bin* for balanced coverage, never a preferred direction. Native
snapshots are opaque reset machinery. This is not a trained/published policy,
and searched actions cannot enter the learner as targets or demonstrations.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import signal

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_cells import BOTTOM_DETAIL_ENCODING, screen_cell_bottom_detail
from .defense_gate_timing import visible_ship_column
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_macro_explore import command_ids
from .defense_snapshot import capture, restore
from .defense_world_data import disk_guard


ANCHOR = 320
SCHEDULE = (4,)*4 + (1,)*12 + (4,)*4 + (2,)*32
HORIZON = ANCHOR+sum(SCHEDULE)
EARLY_ANCHOR = 280
EARLY_SCHEDULE = (4,)*10 + SCHEDULE
COMMANDS = command_ids("effective-stage-one")
SOURCE_LIFE_SCORE = 2620


@dataclass
class Node:
    saved: object
    path: bytes
    score: int
    cell: str
    ship_column: int | None
    source: bool = False

    @property
    def ship_bin(self):
        return -1 if self.ship_column is None else self.ship_column//4


def make_node(env, path, *, source=False):
    obs = env.observation()
    return Node(capture(env), path, int(env.score), screen_cell_bottom_detail(obs),
                visible_ship_column(obs[-1]), source)


def select_beam(nodes, limit, rng, *, mandatory=None, history_key=False):
    """Half balanced across visible position bins, then score; no right bias."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("positive beam width required")
    # Optionally retain visibly identical cells with distinct last actions:
    # screen aliasing can hide control timing. Native bytes never enter a key.
    def key(node):
        return (node.cell, node.path[-1] if node.path else -1) if history_key else node.cell

    randomized = [(node.score, float(rng.random()), node) for node in nodes]
    randomized.sort(key=lambda row: (row[0], row[1]), reverse=True)
    unique, seen = [], {key(mandatory)} if mandatory is not None else set()
    for _, _, node in randomized:
        if key(node) not in seen:
            seen.add(key(node))
            unique.append(node)
    selected = [mandatory] if mandatory is not None else []
    used = {mandatory.path} if mandatory is not None else set()
    groups = defaultdict(list)
    for node in unique:
        groups[node.ship_bin].append(node)
    bins = sorted(groups)
    if bins:
        bins = [bins[int(i)] for i in rng.permutation(len(bins))]
    diversity_target = max(1, limit//2)
    while len(selected) < diversity_target and any(groups.values()):
        for bucket in bins:
            while groups[bucket] and groups[bucket][0].path in used:
                groups[bucket].pop(0)
            if groups[bucket]:
                node = groups[bucket].pop(0)
                selected.append(node)
                used.add(node.path)
                if len(selected) >= diversity_target:
                    break
    for node in unique:
        if len(selected) >= limit:
            break
        if node.path not in used:
            selected.append(node)
            used.add(node.path)
    return selected[:limit]


def verify_discovery(env, seed, source_frames, source_actions, source_rewards,
                     anchor, anchor_saved, path, expected, output):
    obs = restore(env, anchor_saved)
    screens, rewards = [obs[-1].copy()], []
    for action in path:
        obs, reward, _, _, info = env.step(int(action))
        screens.append(obs[-1].copy())
        rewards.append(float(reward))
    if (info["score"] != expected["score"] or info["stage"] != expected["stage"]
            or bool(info["life_lost"]) != expected["life_lost"]):
        raise RuntimeError("discovery changed on snapshot replay")
    obs = env.reset(seed)
    np.testing.assert_array_equal(obs[-1], source_frames[0])
    for index in range(anchor):
        obs, reward, terminal, truncated, info = env.step(int(source_actions[index]))
        if reward != source_rewards[index] or terminal or truncated or info["life_lost"]:
            raise RuntimeError("original-boot source prefix changed")
        np.testing.assert_array_equal(obs[-1], source_frames[index+1])
    np.testing.assert_array_equal(obs[-1], screens[0])
    for index, action in enumerate(path):
        obs, reward, _, _, info = env.step(int(action))
        if reward != rewards[index]:
            raise RuntimeError("discovery reward changed from original boot")
        np.testing.assert_array_equal(obs[-1], screens[index+1])
    if (info["score"] != expected["score"] or info["stage"] != expected["stage"]
            or bool(info["life_lost"]) != expected["life_lost"]):
        raise RuntimeError("discovery outcome changed from original boot")
    np.savez_compressed(output/"discovery-trace.npz", frames=np.stack(screens),
                        actions=np.frombuffer(path, np.uint8).copy(),
                        rewards=np.asarray(rewards, np.float32))
    return dict(verified=True, original_boot_actions=anchor+len(path),
                trace_sha256=sha256(output/"discovery-trace.npz"),
                method="every original-boot action, visible screen and score increment reproduced")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=128)
    parser.add_argument("--layers", type=int, default=0,
                        help="0 completes the fixed schedule; positive prefix is a plumbing smoke")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--early-anchor", action="store_true",
                        help="start at frame 280 with ten extra four-action layers")
    parser.add_argument("--fine-cadence", action="store_true",
                        help="one action per layer from frame 320 onward")
    parser.add_argument("--history-key", action="store_true",
                        help="deduplicate by visible cell and last action")
    args = parser.parse_args()
    anchor = EARLY_ANCHOR if args.early_anchor else ANCHOR
    schedule = ((4,)*10 if args.early_anchor else ()) + (
        (1,)*(HORIZON-ANCHOR) if args.fine_cadence else SCHEDULE)
    if (args.output.exists() or not 1 <= args.beam <= 512
            or not 0 <= args.layers <= len(schedule) or args.seed < 0):
        parser.error("fresh output, beam 1..512, valid layer prefix and nonnegative seed required")
    report, source_frames, source_actions = analyze(args.bundle)
    first_loss = report["lives"][0]["visible_loss_frame"]
    if (first_loss != 407 or report["lives"][0]["visible_score_at_loss"] != SOURCE_LIFE_SCORE
            or len(source_actions) < HORIZON):
        parser.error("requires the exact verified own first-life source")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as data:
        source_rewards = data["rewards"].copy()
        metadata = json.loads(str(data["metadata"]))
    frames = [anchor]
    for duration in schedule:
        frames.append(frames[-1]+duration)
    config = dict(bundle=str(args.bundle.resolve()),
                  source_trace_sha256=sha256(args.bundle/"trace.npz"),
                  source_model_sha256=sha256(args.bundle/"model.safetensors"),
                  source_sha256=sha256(Path(__file__)),
                  cell_source_sha256=sha256(Path(__file__).with_name("defense_cells.py")),
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  source_seed=metadata["result"]["seed"], anchor=anchor,
                  horizon=HORIZON, source_visible_loss=first_loss,
                  source_life_score=SOURCE_LIFE_SCORE, schedule=list(schedule),
                  symmetric_commands=list(COMMANDS), beam=args.beam,
                  requested_layers=args.layers or len(schedule), search_seed=args.seed,
                  cell_encoding=BOTTOM_DETAIL_ENCODING,
                  history_key=args.history_key, fine_cadence=args.fine_cadence,
                  selection="half balanced across visible ship-column bins; half highest displayed score; source path protected",
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  snapshots_opaque=True, promotion_eligible=False)
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
    rng = np.random.default_rng(args.seed)
    discovery = None
    try:
        obs = env.reset(config["source_seed"])
        np.testing.assert_array_equal(obs[-1], source_frames[0])
        source_nodes = {}
        for index in range(first_loss):
            if index in frames:
                source_nodes[index] = make_node(
                    env, bytes(source_actions[anchor:index]), source=True)
            obs, reward, terminal, truncated, info = env.step(int(source_actions[index]))
            if (reward != source_rewards[index] or terminal or truncated
                    or info["life_lost"] != (index+1 == first_loss)):
                raise RuntimeError("verified source first life failed exact reexecution")
            np.testing.assert_array_equal(obs[-1], source_frames[index+1])
        if anchor not in source_nodes:
            raise RuntimeError("missing exact source anchor")
        anchor_saved = source_nodes[anchor].saved
        beam = [source_nodes[anchor]]
        latest_frame = anchor
        max_score, max_alive_frame, total_expansions, total_deaths = env.score, anchor, 0, 0
        completed_layers = 0
        with (args.output/"layers.jsonl").open("x", buffering=1) as stream:
            for layer, duration in enumerate(schedule[:config["requested_layers"]], 1):
                if stop or not beam:
                    break
                children = []
                alive_children, layer_deaths, layer_max_score = 0, 0, 0
                for parent in beam:
                    for command in COMMANDS:
                        obs = restore(env, parent.saved)
                        executed = 0
                        for _ in range(duration):
                            obs, _, terminal, truncated, info = env.step(command)
                            executed += 1
                            absolute = latest_frame+executed
                            max_score = max(max_score, int(info["score"]))
                            layer_max_score = max(layer_max_score, int(info["score"]))
                            ended = bool(info["life_lost"] or terminal or truncated)
                            if not ended:
                                max_alive_frame = max(max_alive_frame, absolute)
                            progress = (info["score"] > SOURCE_LIFE_SCORE
                                        or info["stage"] >= 2 or info["mission_completed"]
                                        or (absolute >= HORIZON and not ended))
                            if progress or ended:
                                break
                        total_expansions += 1
                        path = parent.path + bytes([command])*executed
                        if progress:
                            outcome = dict(layer=layer, frame=absolute, score=int(info["score"]),
                                           stage=int(info["stage"]), life_lost=bool(info["life_lost"]),
                                           mission_completed=bool(info["mission_completed"]),
                                           action_sha256=hashlib.sha256(path).hexdigest(),
                                           actions_from_anchor=len(path), ship_column=visible_ship_column(obs[-1]))
                            verification = verify_discovery(
                                env, config["source_seed"], source_frames, source_actions,
                                source_rewards, anchor, anchor_saved, path, outcome, args.output)
                            discovery = dict(outcome=outcome, verification=verification)
                            latest_frame = absolute
                            write_json(args.output/"discovery.json", discovery)
                            print(json.dumps(dict(event="verified_discovery", **discovery)), flush=True)
                            break
                        if ended:
                            layer_deaths += 1
                            total_deaths += 1
                            continue
                        alive_children += 1
                        children.append(make_node(env, path))
                    if discovery is not None:
                        break
                if discovery is not None:
                    break
                latest_frame = frames[layer]
                mandatory = source_nodes.get(latest_frame)
                if mandatory is not None:
                    children.append(mandatory)
                beam = select_beam(children, args.beam, rng, mandatory=mandatory,
                                   history_key=args.history_key)
                row = dict(layer=layer, absolute_frame=latest_frame, duration=duration,
                           generated=alive_children+layer_deaths,
                           alive_children=alive_children, life_losses=layer_deaths,
                           distinct_screen_cells=len({n.cell for n in children}),
                           distinct_keys=len({(n.cell, n.path[-1] if n.path else -1)
                                              if args.history_key else n.cell for n in children}),
                           selected=len(beam), selected_score_max=max((n.score for n in beam), default=None),
                           selected_ship_bins=sorted({n.ship_bin for n in beam}),
                           layer_max_score=layer_max_score,
                           selected_paths=[dict(path_hex=n.path.hex(), score=n.score,
                                                cell=n.cell, ship_column=n.ship_column,
                                                source=n.source) for n in beam])
                stream.write(json.dumps(row)+"\n")
                completed_layers = layer
                write_json(args.output/"status.json", {k:v for k,v in row.items() if k!="selected_paths"})
                print(json.dumps(dict(event="layer", **{k:v for k,v in row.items() if k!="selected_paths"})), flush=True)
                disk_guard(args.output)
        result = dict(config=config, completed_layers=completed_layers,
                      latest_frame=latest_frame, max_score=max_score,
                      max_alive_frame=max_alive_frame, total_expansions=total_expansions,
                      total_life_losses=total_deaths, discovery=discovery,
                      stopped=stop, layers_sha256=sha256(args.output/"layers.jsonl"))
        write_json(args.output/"report.json", result)
        print(json.dumps(dict(event="finished", completed_layers=result["completed_layers"],
                              latest_frame=latest_frame, max_score=max_score,
                              max_alive_frame=max_alive_frame, total_expansions=total_expansions,
                              discovery=discovery, stopped=stop)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
