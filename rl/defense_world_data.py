"""Fresh own-experience collection and sequence sampling for dynamics learning.

Never accepts replay/evaluation bundles. Held-out whole games are assigned
before collection and never supplied to gradient updates.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256, ENVIRONMENT_VERSION
from .defense_learning import DQN_ALGORITHM, load_policy, sha256, write_json
from .persistent_exploration import PersistentExploration


SCHEMA = 'defense-own-dynamics-v1'


def disk_guard(path):
    if shutil.disk_usage(path).free < 5*1024**3:
        raise RuntimeError('less than 5 GiB free; preserving existing files and stopping')


def collect(checkpoint, output, games=24, heldout=4, seed=65000, epsilon=.25):
    checkpoint, output = Path(checkpoint), Path(output)
    if not 0 < heldout < games or seed < 0 or not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError('invalid collection settings')
    if output.exists():
        raise ValueError('refusing to overwrite data')
    policy, config = load_policy(checkpoint)
    if config['algorithm'] != DQN_ALGORITHM or config.get('allow_enter') or config.get('architecture'):
        raise ValueError('preflight collection requires ordinary own scalar DQN')
    parent = {n: sha256(checkpoint.parent/n) for n in ('model.safetensors', 'state.json')}
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema=SCHEMA, purpose='new-own-experience-for-world-model',
        checkpoint=str(checkpoint.resolve()), parent_hashes=parent, game_sha256=GAME_SHA256,
        environment_version=ENVIRONMENT_VERSION, collection_source_sha256=sha256(Path(__file__)),
        tstates=config['tstates'], observation_stride=config.get('observation_stride', 1),
        games=games, heldout_games=heldout, first_seed=seed, epsilon=epsilon,
        exploration_max_repeat=64, exploration_exponent=1.5,
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
            rng = np.random.default_rng(seed+game+700000)
            explorer = PersistentExploration(1, 20, 64, 1.5, rng)
            frames, actions, rewards, continues = [obs[-1].copy()], [], [], []
            while True:
                action = int(explorer.select(policy(obs[None]), epsilon)[0])
                obs, reward, terminal, truncated, info = env.step(action)
                frames.append(obs[-1].copy()); actions.append(action); rewards.append(reward)
                boundary = terminal or info['life_lost']
                continues.append(not boundary)
                explorer.reset(np.array([boundary]))
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
                result=info, exploration=explorer.stats())
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


class Sequences:
    def __init__(self, directory, split, length):
        directory = Path(directory)
        metadata = json.loads((directory/'manifest.json').read_text())
        if (metadata.get('schema') != SCHEMA or metadata.get('game_sha256') != GAME_SHA256
                or metadata.get('purpose') != 'new-own-experience-for-world-model'
                or metadata.get('existing_evaluation_data') is not False
                or split not in ('train', 'heldout') or length < 1):
            raise ValueError('not a compatible own-collection dataset')
        self.episodes, self.counts, self.files = [], [], []
        for row in metadata['episodes']:
            if row['split'] != split:
                continue
            path = directory/row['file']
            if path.parent.resolve() != directory.resolve() or sha256(path) != row['sha256']:
                raise ValueError('dataset path or checksum mismatch')
            with np.load(path, allow_pickle=False) as data:
                episode = tuple(data[k].copy() for k in ('frames', 'actions', 'rewards', 'continuation'))
            frames, actions, rewards, continuation = episode
            n = len(actions)
            if (n < 1 or frames.shape != (n+1, 16, 64) or frames.dtype != np.uint8
                    or actions.shape != (n,) or actions.dtype != np.int32
                    or np.any(actions < 0) or np.any(actions >= 20)
                    or rewards.shape != (n,) or continuation.shape != (n,)
                    or not np.isfinite(rewards).all() or np.any(rewards < 0)
                    or not np.isin(continuation, [0., 1.]).all() or continuation[-1] != 0
                    or n != row['steps'] or rewards.sum(dtype=np.float64) != row['result']['score']):
                raise ValueError('malformed own episode')
            if n >= length:
                self.episodes.append(episode); self.counts.append(n-length+1); self.files.append(row['file'])
        if not self.episodes:
            raise ValueError('no sufficiently long own episodes')
        self.length, self.ends = length, np.cumsum(self.counts)
        self.split, self._boundary_pools = split, {}

    def boundary_pool(self, burn):
        """Unique windows with a visible loss after burn-in; training only."""
        if self.split != 'train' or not 0 <= burn < self.length:
            raise ValueError('boundary sampling requires train split and valid burn-in')
        if burn not in self._boundary_pools:
            pools = []
            for i, episode in enumerate(self.episodes):
                eligible = np.zeros(self.counts[i], bool)
                for boundary in np.flatnonzero(episode[3] == 0):
                    low = max(0, int(boundary)-self.length+1)
                    high = min(int(boundary)-burn, self.counts[i]-1)
                    if low <= high:
                        eligible[low:high+1] = True
                pools.append(np.flatnonzero(eligible)+(self.ends[i-1] if i else 0))
            self._boundary_pools[burn] = np.concatenate(pools)
        return self._boundary_pools[burn]

    def sample(self, count, rng, boundary_fraction=0., burn=0):
        if (count < 1 or not np.isfinite(boundary_fraction) or not 0 <= boundary_fraction <= 1
                or not 0 <= burn < self.length):
            raise ValueError('positive batch size, valid burn-in and boundary fraction required')
        selected = rng.integers(int(self.ends[-1]), size=count)
        if boundary_fraction:
            pool = self.boundary_pool(burn)
            if not len(pool):
                raise ValueError('no eligible training loss windows')
            focused = rng.random(count) < boundary_fraction
            selected[focused] = pool[rng.integers(len(pool), size=int(focused.sum()))]
        rows = []
        for index in selected:
            episode = int(np.searchsorted(self.ends, index, side='right'))
            offset = int(index-(self.ends[episode-1] if episode else 0))
            frames, actions, rewards, continuation = self.episodes[episode]
            end = offset+self.length
            rows.append((frames[offset:end+1], actions[offset:end], rewards[offset:end]*.01,
                         continuation[offset:end]))
        return tuple(np.stack([r[k] for r in rows]) for k in range(4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=24)
    parser.add_argument('--heldout', type=int, default=4)
    parser.add_argument('--seed', type=int, default=65000)
    parser.add_argument('--epsilon', type=float, default=.25)
    args = parser.parse_args()
    import mlx.core as mx
    mx.set_cache_limit(128*1024*1024)
    collect(args.checkpoint, args.output, args.games, args.heldout, args.seed, args.epsilon)


if __name__ == '__main__':
    main()
