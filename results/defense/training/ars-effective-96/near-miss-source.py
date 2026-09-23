"""Recheck saved neural search near-misses on fresh complete training games.

This diagnostic uses only displayed final score to compare candidates. It
never decodes native state or supplies obstacle positions or target keys.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_ars_boot_search import score_means, shared_seed_jobs


def trial_head(plan, index, incumbent):
    plan = Path(plan)
    with np.load(plan, allow_pickle=False) as arrays:
        center, heads = arrays['center'].copy(), arrays['heads'].copy()
    incumbent = np.asarray(incumbent, np.float32)
    if (center.shape != incumbent.shape or not np.array_equal(center, incumbent)
            or heads.shape != (40, *incumbent.shape) or not 0 <= index < len(heads)
            or not np.isfinite(heads).all()):
        raise ValueError('plan is not a complete symmetric population around this incumbent')
    return heads[index]


def main():
    import mlx.core as mx
    from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars import head_array, play_population, restore_checkpoint
    from .defense_learning import sha256, write_json
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--trial', nargs=2, action='append', required=True,
                        metavar=('PLAN_NPZ', 'CANDIDATE_INDEX'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--first-training-seed', type=int, required=True)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--minimum-gain', type=float, default=150.)
    args = parser.parse_args()
    if (args.output.exists() or min(args.games, args.envs) < 1
            or args.first_training_seed < 70000 or not np.isfinite(args.minimum_gain)
            or args.minimum_gain <= 0):
        parser.error('fresh output, training-only seeds and positive settings required')
    mx.set_cache_limit(128*1024*1024)
    model = QNetwork(action_count=20)
    state = restore_checkpoint(model, args.parent, np.random.default_rng(0))
    config = state['config']
    if (config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('native_sha256') != sha256('libtrs.so')
            or config.get('action_names') != list(action_names())
            or config.get('allow_enter') or config.get('eval_max_steps') != 0):
        raise ValueError('incompatible complete-game action profile')
    center = head_array(model)
    trials = []
    for plan_text, index_text in args.trial:
        plan = Path(plan_text)
        index = int(index_text)
        trials.append(dict(plan=str(plan.resolve()), index=index, sha256=sha256(plan),
                           head=trial_head(plan, index, center)))
    if len({(row['plan'], row['index']) for row in trials}) != len(trials):
        raise ValueError('duplicate candidate trial')
    disk_guard(args.output.parent)
    heads = np.stack([center]+[row['head'] for row in trials])
    jobs = shared_seed_jobs(len(heads), args.games, args.first_training_seed)
    games, _ = play_population(model, heads, jobs, envs=args.envs,
        tstates=config['tstates'], observation_stride=config['observation_stride'],
        keep_traces=False)
    means = score_means(games, len(heads), args.games)
    selected = int(np.argmax(means[1:]))+1
    report = dict(parent=str(args.parent.resolve()),
        parent_sha256=sha256(args.parent/'model.safetensors'),
        source_sha256=sha256(Path(__file__)),
        trials=[{key: value for key, value in row.items() if key != 'head'} for row in trials],
        fitness='displayed score in complete paired games from boot',
        training_only=True, first_training_seed=args.first_training_seed,
        games_per_policy=args.games, means=[float(x) for x in means],
        gains=[float(x-means[0]) for x in means[1:]],
        highest_stages=[max(g['highest_stage'] for g in games if g['candidate'] == i)
                        for i in range(len(heads))],
        selected_candidate=selected, minimum_gain=args.minimum_gain,
        accepted=bool(means[selected] >= means[0]+args.minimum_gain),
        games=games)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in
        ('games_per_policy', 'means', 'gains', 'highest_stages',
         'selected_candidate', 'accepted')}))


if __name__ == '__main__':
    main()
