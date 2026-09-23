"""Fresh paired full-boot score check for two frozen-encoder ARS checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_ars_boot_search import score_means, shared_seed_jobs


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars import head_array, play_population, restore_checkpoint
    from .defense_learning import sha256, write_json
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--first-training-seed', type=int, required=True)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--minimum-gain', type=float, default=150.)
    args = parser.parse_args()
    if (args.output.exists() or min(args.games, args.envs) < 1
            or args.first_training_seed < 70000 or not np.isfinite(args.minimum_gain)
            or args.minimum_gain <= 0):
        parser.error('fresh output, training-only seeds and positive settings required')
    mx.set_cache_limit(128*1024*1024)
    parent, candidate = QNetwork(action_count=20), QNetwork(action_count=20)
    first = restore_checkpoint(parent, args.parent, np.random.default_rng(0))
    second = restore_checkpoint(candidate, args.candidate, np.random.default_rng(0))
    left, right = first['config'], second['config']
    for config in (left, right):
        if (config.get('game_sha256') != GAME_SHA256
                or config.get('environment_version') != ENVIRONMENT_VERSION
                or config.get('native_sha256') != sha256('libtrs.so')
                or config.get('action_names') != list(action_names())
                or config.get('allow_enter') or config.get('eval_max_steps') != 0):
            raise ValueError('incompatible complete-game action profile')
    if any(left[key] != right[key] for key in ('tstates', 'observation_stride')):
        raise ValueError('incompatible environment settings')
    base = dict(tree_flatten(parent.parameters()))
    trial = dict(tree_flatten(candidate.parameters()))
    if base.keys() != trial.keys() or any(not np.array_equal(np.array(base[key]), np.array(trial[key]))
            for key in base if not key.startswith('advantage.')):
        raise ValueError('checkpoints do not share the exact frozen visual encoder and value net')
    disk_guard(args.output.parent)
    heads = np.stack((head_array(parent), head_array(candidate)))
    jobs = shared_seed_jobs(2, args.games, args.first_training_seed)
    games, _ = play_population(parent, heads, jobs, envs=args.envs,
        tstates=left['tstates'], observation_stride=left['observation_stride'],
        keep_traces=False)
    means = score_means(games, 2, args.games)
    report = dict(parent=str(args.parent.resolve()), candidate=str(args.candidate.resolve()),
        parent_sha256=sha256(args.parent/'model.safetensors'),
        candidate_sha256=sha256(args.candidate/'model.safetensors'),
        source_sha256=sha256(Path(__file__)),
        exact_frozen_encoder_and_value=True,
        fitness='displayed score in complete paired games from boot',
        training_only=True, first_training_seed=args.first_training_seed,
        games_per_policy=args.games, parent_mean=float(means[0]),
        candidate_mean=float(means[1]), mean_gain=float(means[1]-means[0]),
        minimum_gain=args.minimum_gain,
        accepted=bool(means[1] >= means[0]+args.minimum_gain),
        parent_highest_stage=max(g['highest_stage'] for g in games if g['candidate'] == 0),
        candidate_highest_stage=max(g['highest_stage'] for g in games if g['candidate'] == 1),
        games=games)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in
        ('games_per_policy', 'parent_mean', 'candidate_mean', 'mean_gain',
         'accepted', 'parent_highest_stage', 'candidate_highest_stage')}))


if __name__ == '__main__':
    main()
