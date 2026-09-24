"""Training-only chaining of random holds from own, visible-screen states.

This is exploration research, not a learned playing policy or a replay eligible
for promotion. Opaque native states are reset machinery only. Source prefixes
come from the agent's own verified games; they are never action targets.
Screen cells select diverse reset states, and displayed score is the only
game-return quantity. Every exploratory direction is sampled symmetrically.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from .defense_cells import CELL_ENCODING, screen_cell
from .defense_learning import sha256, write_json
from .defense_macro_explore import command_ids
from .defense_snapshot import capture, restore


HOLDS = (4, 8, 16, 32)


class FrontierArchive:
    """Bounded screen-cell reservoir; higher displayed score wins exact ties."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("positive archive capacity required")
        self.capacity = capacity
        self.cells = {}
        self.seen = set()

    def add(self, key, node_id, life_score, saved, rng):
        if not isinstance(key, str) or len(key) != 32 or life_score < 0:
            raise ValueError("invalid visible screen cell or score")
        self.seen.add(key)
        old = self.cells.get(key)
        if old is not None:
            if life_score <= old[1]:
                return False
            self.cells[key] = (node_id, life_score, saved)
            return True
        if len(self.cells) < self.capacity:
            self.cells[key] = (node_id, life_score, saved)
            return True
        if int(rng.integers(len(self.seen))) >= self.capacity:
            return False
        victim = tuple(self.cells)[int(rng.integers(self.capacity))]
        del self.cells[victim]
        self.cells[key] = (node_id, life_score, saved)
        return True

    def choose(self, rng):
        if not self.cells:
            raise ValueError("empty frontier archive")
        rows = tuple(self.cells.values())
        if rng.random() < .5:
            cutoff = float(np.quantile([row[1] for row in rows], .75))
            rows = tuple(row for row in rows if row[1] >= cutoff)
        return rows[int(rng.integers(len(rows)))]


def trace_path(records, leaf, action, length):
    macros = [(action, length)]
    current = leaf
    while records[current]["kind"] == "random_hold":
        row = records[current]
        macros.append((row["action"], row["length"]))
        current = row["parent"]
    root = records[current]
    return root["source"], root["prefix"], list(reversed(macros))


def verify_discovery(env, source_archive, source_name, prefix, macros, target_stage):
    from .defense_ars_focus import load_source
    saved, actions, rewards, screens, _ = load_source(source_archive/source_name)
    obs = restore(env, saved)
    frames, executed, earned = [obs[-1].copy()], [], []
    for index in range(prefix):
        action = int(actions[index])
        obs, reward, terminal, truncated, info = env.step(action)
        if (reward != rewards[index] or not np.array_equal(obs[-1], screens[index])
                or terminal or truncated or info["life_lost"]):
            raise RuntimeError("own source prefix failed exact reexecution")
        frames.append(obs[-1].copy())
        executed.append(action)
        earned.append(reward)
    for action, length in macros:
        for _ in range(length):
            obs, reward, terminal, truncated, info = env.step(action)
            frames.append(obs[-1].copy())
            executed.append(action)
            earned.append(reward)
            if info["life_lost"] or info["stage"] >= target_stage or terminal or truncated:
                break
        if info["life_lost"] or info["stage"] >= target_stage or terminal or truncated:
            break
    if info["stage"] < target_stage or info["life_lost"]:
        raise RuntimeError("frontier discovery did not reexecute")
    return (np.asarray(frames, np.uint8), np.asarray(executed, np.uint8),
            np.asarray(earned, np.float32), info)


def main():
    from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars_focus import load_source
    from .defense_world_data import disk_guard

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expansions", type=int, default=20_000)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--source-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=503)
    args = parser.parse_args()
    if (args.output.exists() or min(args.expansions, args.capacity, args.source_stride) < 1
            or args.seed < 0):
        parser.error("fresh output and positive bounded settings required")
    source_archive = args.source_archive.resolve(strict=True)
    index = json.loads((source_archive/"index.json").read_text())
    if len(index["files"]) < 1:
        parser.error("no own source states")
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/"source.py")
    config = dict(source_archive=str(source_archive), source_index_sha256=sha256(source_archive/"index.json"),
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  source_sha256=sha256(Path(__file__)), screen_cell_encoding=CELL_ENCODING,
                  effective_commands=list(command_ids("effective-stage-one")), holds=list(HOLDS),
                  args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  training_only=True, model_updates=0, policy_inputs_changed=False,
                  score_reward_unchanged=True, random_actions_not_learned_replay=True)
    write_json(args.output/"config.json", config)
    env = DefenseEnv(tstates=100_000, max_steps=0, observation_stride=1)
    archive, rng = FrontierArchive(args.capacity), np.random.default_rng(args.seed)
    records, counters, source_hashes = [], Counter(), {}
    max_seed_life_score, max_explore_life_score = 0, 0
    try:
        prior = {}
        for name in index["files"]:
            saved, actions, rewards, screens, metadata = load_source(source_archive/name)
            source_hashes[name] = metadata["npz_sha256"]
            source = metadata["source"]
            seed = source["seed"]
            base = prior.get(seed, 0)
            prior[seed] = source["visible_loss_score"]
            if (saved.stage != 1 or saved.tstates != env.tstates or saved.max_steps != 0
                    or saved.observation_stride != 1 or saved.score < base
                    or len(actions) < args.source_stride):
                raise ValueError("source not a compatible own stage-one life")
            obs = restore(env, saved)
            for prefix in range(len(actions)):
                if prefix % args.source_stride == 0:
                    life_score = env.score-base
                    key = screen_cell(obs)
                    node_id = len(records)
                    snapshot = capture(env)
                    if archive.add(key, node_id, life_score, snapshot, rng):
                        records.append(dict(kind="own_prefix", source=name, prefix=prefix,
                                            parent=None, base_score=base,
                                            life_score=life_score, cell=key,
                                            stage=env.stage))
                        counters["seeded"] += 1
                        max_seed_life_score = max(max_seed_life_score, life_score)
                obs, reward, terminal, truncated, info = env.step(int(actions[prefix]))
                if reward != rewards[prefix] or not np.array_equal(obs[-1], screens[prefix]):
                    raise RuntimeError("own source prefix failed exact native verification")
                if prefix < len(actions)-1 and (terminal or truncated or info["life_lost"]):
                    raise RuntimeError("own source ended before its recorded loss")
        write_json(args.output/"sources.json", dict(index_sha256=config["source_index_sha256"],
                                                    files=source_hashes, own_games=len(index["games"])))
        if not archive.cells:
            raise RuntimeError("no own pre-loss source cells seeded")
        with (args.output/"plans.jsonl").open("x") as stream:
            for attempt in range(args.expansions):
                parent_id, _, saved = archive.choose(rng)
                action = int(rng.choice(command_ids("effective-stage-one")))
                hold = int(rng.choice(HOLDS))
                obs = restore(env, saved)
                start_stage, start_score = saved.stage, saved.score
                boundary, earned = False, 0.
                for length in range(1, hold+1):
                    obs, reward, terminal, truncated, info = env.step(action)
                    earned += reward
                    counters["exploration_actions"] += 1
                    if (info["life_lost"] or info["stage"] != start_stage
                            or info["mission_completed"] or terminal or truncated):
                        boundary = True
                        break
                if env.score-start_score != earned:
                    raise RuntimeError("exploration reward differs from displayed score gain")
                if info["life_lost"]:
                    counters["life_losses"] += 1
                if info["stage"] > 1:
                    counters["stage_two"] += 1
                if info["mission_completed"]:
                    counters["mission"] += 1
                base = records[parent_id]["base_score"]
                life_score = env.score-base
                max_explore_life_score = max(max_explore_life_score, life_score)
                row = dict(attempt=attempt, parent=parent_id, action=action, hold=hold,
                           length=length, score_gain=env.score-start_score,
                           life_score=life_score, stage=info["stage"],
                           life_lost=bool(info["life_lost"]),
                           mission_completed=bool(info["mission_completed"]),
                           admitted=False)
                if not boundary:
                    key = screen_cell(obs)
                    node_id = len(records)
                    snapshot = capture(env)
                    if archive.add(key, node_id, life_score, snapshot, rng):
                        records.append(dict(kind="random_hold", parent=parent_id,
                                            action=action, length=length,
                                            source=None, prefix=None, base_score=base,
                                            life_score=life_score, cell=key, stage=env.stage))
                        counters["admitted"] += 1
                        row["admitted"] = True
                        row["node"] = node_id
                stream.write(json.dumps(row)+"\n")
                if (attempt+1) % 1000 == 0 or info["stage"] > 1:
                    stream.flush()
                    write_json(args.output/"status.json", dict(expansions=attempt+1,
                        archive_cells=len(archive.cells), distinct_cells=len(archive.seen),
                        seed_best_life_score=max_seed_life_score,
                        exploration_best_life_score=max_explore_life_score,
                        counts=dict(counters),
                        rng_state=rng.bit_generator.state))
                    print(json.dumps(dict(expansions=attempt+1,
                        cells=len(archive.cells), seed_best_life_score=max_seed_life_score,
                        exploration_best_life_score=max_explore_life_score,
                        stage_two=counters["stage_two"], counts=dict(counters))), flush=True)
                    disk_guard(args.output)
                if info["stage"] > 1:
                    source_name, prefix, macros = trace_path(records, parent_id, action, length)
                    frames, executed, rewards, verified = verify_discovery(
                        env, source_archive, source_name, prefix, macros, info["stage"])
                    np.savez_compressed(args.output/"discovery-trace.npz",
                                        frames=frames, actions=executed, rewards=rewards)
                    write_json(args.output/"discovery.json", dict(source=source_name,
                        source_prefix=prefix, macros=macros, exploratory_result=row,
                        verified_stage=verified["stage"], verified_score=verified["score"],
                        training_only=True, learned_policy_success=False,
                        promotion_eligible=False))
                    break
        write_json(args.output/"nodes.json", records)
        write_json(args.output/"report.json", dict(config=config, expansions=attempt+1,
            source_games=len(index["games"]), source_states=len(index["files"]),
            archive_cells=len(archive.cells), distinct_cells=len(archive.seen),
            seed_best_life_score=max_seed_life_score,
            exploration_best_life_score=max_explore_life_score,
            counts=dict(counters),
            rng_state=rng.bit_generator.state,
            discovery=(args.output/"discovery.json").exists(), model_updates=0,
            published_replay=False,
            limitations=["Random held-key branches are not neural-policy actions.",
                         "Screen-cell novelty chooses reset states, not rewards or actions.",
                         "Opaque native bytes are never decoded or fed to a model.",
                         "Visible life loss may lag physical collision."]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
