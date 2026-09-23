"""Diagnostic-only firing-key ablation on a frozen screen-only policy.

No ablated head is saved, selected as a learned update, or replay-promoted.
The test uses only complete-game visible score and stage observations.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_ars_boot_search import key_incidence, score_means, shared_seed_jobs


BIASES = (0., -.5, -1., -2., -4., -8., -100.)


def space_bias_heads(head, names, biases=BIASES):
    head = np.asarray(head, np.float32)
    biases = np.asarray(biases, np.float32)
    if (head.ndim != 2 or head.shape[0] != len(names) or head.shape[1] < 2
            or biases.ndim != 1 or not len(biases)
            or not np.isfinite(head).all() or not np.isfinite(biases).all()
            or float(biases[0]) != 0):
        raise ValueError('invalid diagnostic action heads')
    space = key_incidence(names)[:, 4]
    heads = np.broadcast_to(head, (len(biases), *head.shape)).copy()
    heads[:, :, -1] += biases[:, None]*space[None]
    return heads


def main():
    import mlx.core as mx
    from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars import head_array, play_population, restore_checkpoint
    from .defense_learning import sha256, write_json
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--first-training-seed', type=int, required=True)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--biases', nargs='+', type=float, default=BIASES)
    args = parser.parse_args()
    if (args.output.exists() or min(args.games, args.envs) < 1
            or args.first_training_seed < 70000):
        parser.error('new output and positive training-only seed settings required')
    mx.set_cache_limit(128*1024*1024)
    model = QNetwork(action_count=20)
    state = restore_checkpoint(model, args.checkpoint, np.random.default_rng(0))
    config = state['config']
    if (config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('native_sha256') != sha256('libtrs.so')
            or config.get('action_names') != list(action_names())
            or config.get('allow_enter') or config.get('eval_max_steps') != 0):
        raise ValueError('incompatible complete-game action profile')
    heads = space_bias_heads(head_array(model), action_names(), args.biases)
    disk_guard(args.output.parent)
    jobs = shared_seed_jobs(len(heads), args.games, args.first_training_seed)
    games, _ = play_population(model, heads, jobs, envs=args.envs,
        tstates=config['tstates'], observation_stride=config['observation_stride'],
        keep_traces=False)
    means = score_means(games, len(heads), args.games)
    report = dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint/'model.safetensors'),
        source_sha256=sha256(Path(__file__)),
        diagnostic_only=True, promotion_eligible=False,
        saved_policy_weights=False, parameter_updates=0,
        action_family='every command containing SPACE',
        biases=[float(value) for value in args.biases], training_only=True,
        first_training_seed=args.first_training_seed, games_per_variant=args.games,
        fitness='displayed score in complete paired games from boot',
        mean_scores=[float(value) for value in means],
        highest_stages=[max(g['highest_stage'] for g in games if g['candidate'] == index)
                        for index in range(len(heads))],
        complete_games=all(g['terminated'] and not g['truncated'] for g in games),
        games=games)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in
        ('games_per_variant', 'biases', 'mean_scores', 'highest_stages', 'complete_games')}))


if __name__ == '__main__':
    main()
