"""Collect fresh complete own games from an imagined-return actor.

No external exploration override: the recurrent policy records exactly the
action actually executed. Whole held-out seeds are assigned before collection.
This is new training experience, never an import of evaluation replays.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, GAME_SHA256, ENVIRONMENT_VERSION
from .defense_learning import IMAGINATION_ALGORITHM, load_policy, sha256, write_json
from .defense_world_data import SCHEMA, disk_guard


def collect(checkpoint, output, games=24, heldout=4, seed=67000):
    checkpoint, output = Path(checkpoint), Path(output)
    if not 0 < heldout < games or seed < 0 or output.exists():
        raise ValueError('new output and valid collection counts/seeds required')
    saved = json.loads((checkpoint.parent/'state.json').read_text())
    if saved['config'].get('algorithm') != IMAGINATION_ALGORITHM or saved['actor_updates'] < 1:
        raise ValueError('requires a trained own imagined-return actor')
    for name, digest in saved['hashes'].items():
        if sha256(checkpoint.parent/name) != digest:
            raise ValueError('actor checkpoint checksum mismatch')
    policy, config = load_policy(checkpoint)
    parent = {n: sha256(checkpoint.parent/n) for n in ('model.safetensors', 'state.json')}
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema=SCHEMA, purpose='new-own-experience-for-world-model',
        checkpoint=str(checkpoint.resolve()), parent_hashes=parent, game_sha256=GAME_SHA256,
        environment_version=ENVIRONMENT_VERSION, collection_source_sha256=sha256(Path(__file__)),
        tstates=config['tstates'], observation_stride=config.get('observation_stride', 1),
        games=games, heldout_games=heldout, first_seed=seed, epsilon=0.,
        exploration_max_repeat=1, exploration_exponent=1.5,
        behavior='learned categorical actor at temperature 1; no action overrides',
        algorithm=IMAGINATION_ALGORITHM, architecture=config['architecture'],
        assignment='final heldout_games sequential seeds fixed before collection',
        policy_imitation=False, hidden_state=False, existing_evaluation_data=False)
    write_json(output/'collection-config.json', metadata)
    env = DefenseEnv(tstates=config['tstates'], max_steps=0,
                     observation_stride=config.get('observation_stride', 1))
    records = []
    try:
        for game in range(games):
            disk_guard(output)
            obs = env.reset(seed+game)
            policy.reset_seed(seed+game+1_000_000)
            frames, actions, rewards, continues = [obs[-1].copy()], [], [], []
            while True:
                action = int(policy(obs[None])[0])
                obs, reward, terminal, truncated, info = env.step(action)
                frames.append(obs[-1].copy()); actions.append(action); rewards.append(reward)
                continues.append(not (terminal or info['life_lost']))
                if len(actions) % 256 == 0:
                    disk_guard(output)
                if terminal or truncated:
                    if truncated or not info['game_over'] or sum(rewards) != info['score']:
                        raise RuntimeError('collection did not finish an original complete game')
                    break
            filename = f'episode-{game:04d}.npz'
            np.savez_compressed(output/filename, frames=np.stack(frames),
                actions=np.array(actions, np.int32), rewards=np.array(rewards, np.float32),
                continuation=np.array(continues, np.float32))
            row = dict(file=filename, sha256=sha256(output/filename), seed=seed+game,
                split='heldout' if game >= games-heldout else 'train', steps=len(actions),
                result=info, exploration={'action_overrides': 0, 'temperature': 1.})
            records.append(row)
            write_json(output/'progress.json', dict(completed_games=len(records), records=records))
            print(json.dumps(dict(event='collected', **{k:row[k] for k in ('file', 'split', 'seed', 'steps')},
                                  score=info['score'], highest_stage=info['highest_stage'])), flush=True)
    finally:
        env.close()
    if parent != {n: sha256(checkpoint.parent/n) for n in parent}:
        raise RuntimeError('collection parent changed')
    write_json(output/'manifest.json', dict(**metadata, episodes=records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=24)
    parser.add_argument('--heldout', type=int, default=4)
    parser.add_argument('--seed', type=int, default=67000)
    args = parser.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(128*1024*1024)
    collect(args.checkpoint, args.output, args.games, args.heldout, args.seed)


if __name__ == '__main__':
    main()
