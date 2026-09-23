"""Combine symmetric full-game score trials into a neural-head proposal.

Only final displayed scores from the policy's own complete training games
weight saved, symmetric screen-feature perturbations. The combined proposal
must independently pass fresh complete games before replacing its parent.
"""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .defense_ars_boot_search import score_means, shared_seed_jobs, shortlist


RADII = (.25, .5, 1., 2.)


def score_gradient(directions, scales, means):
    """Paired finite differences, clipped against single-pair score outliers."""
    directions = np.asarray(directions, np.float32)
    scales = np.asarray(scales, np.float32)
    means = np.asarray(means, np.float64)
    if (directions.ndim != 3 or len(directions) != 20
            or directions.shape[1] != 20 or directions.shape[2] < 2
            or scales.shape != (20,) or means.shape != (40,)
            or not all(np.isfinite(x).all() for x in (directions, scales, means))
            or np.any(scales <= 0)):
        raise ValueError('invalid symmetric complete-game score population')
    differences = means[::2]-means[1::2]
    coefficients = np.clip(differences, -2000., 2000.)/(2*scales)
    return np.einsum('d,daf->af', coefficients, directions.astype(np.float64),
                     optimize=True), differences


def archived_proposal(run, center, generations):
    """Require one unchanged incumbent across all audited populations."""
    run = Path(run)
    center = np.asarray(center, np.float32)
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    completed = {row['generation']: row for row in rows if row.get('event') == 'generation'}
    if (generations < 1 or any(g not in completed or completed[g]['accepted']
                               for g in range(1, generations+1))):
        raise ValueError('aggregate requires complete unchanged-incumbent generations')
    gradient = np.zeros_like(center, np.float64)
    norms, provenance, differences = [], [], []
    from .defense_learning import sha256
    for generation in range(1, generations+1):
        population = run/f'population-{generation:06d}'
        plan_path, games_path = population/'plan.npz', population/'screen-games.json'
        with np.load(plan_path, allow_pickle=False) as arrays:
            saved_center = arrays['center'].copy()
            directions = arrays['directions'].copy()
            scales = arrays['scales'].copy()
            heads = arrays['heads'].copy()
        if (not np.array_equal(saved_center, center)
                or heads.shape != (40, *center.shape)
                or not np.allclose(heads[::2], center+scales[:, None, None]*directions, atol=1e-5)
                or not np.allclose(heads[1::2], center-scales[:, None, None]*directions, atol=1e-5)):
            raise ValueError('population was not symmetric around unchanged parent')
        games = json.loads(games_path.read_text())
        repetitions = len(games)//40
        means = score_means(games, 40, repetitions)
        contribution, paired = score_gradient(directions, scales, means)
        gradient += contribution
        differences.extend(float(value) for value in paired)
        norms.extend(np.linalg.norm((scales[:, None, None]*directions).reshape(20, -1), axis=1))
        provenance.append(dict(generation=generation, plan_sha256=sha256(plan_path),
            screen_games_sha256=sha256(games_path), screen_games=len(games),
            score_difference_mean=float(paired.mean()),
            score_difference_std=float(paired.std())))
    length = float(np.linalg.norm(gradient))
    reference = float(np.median(norms))
    if not np.isfinite(length) or length < 1e-8 or not np.isfinite(reference) or reference <= 0:
        raise ValueError('no usable aggregate score direction')
    unit = (gradient/length).astype(np.float32)
    heads = np.stack([center+sign*radius*reference*unit
                      for radius in RADII for sign in (1., -1.)])
    if not np.isfinite(heads).all():
        raise FloatingPointError('nonfinite aggregate candidate')
    return heads, dict(populations=provenance, source_generations=generations,
        paired_directions=len(differences), score_difference_mean=float(np.mean(differences)),
        score_difference_std=float(np.std(differences)),
        gradient_norm_before_normalization=length,
        reference_perturbation_norm=reference, radii=list(RADII),
        proposals='plus/minus of score-weighted aggregate; no favored key or route')


def main():
    import mlx.core as mx
    from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars import (acting_policy, head_array, play_population,
                              restore_checkpoint, save_checkpoint, set_head)
    from .defense_learning import (evaluate, game_rank, publish_best, sha256,
                                   summarize, write_json)
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--generations', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen-games', type=int, default=8)
    parser.add_argument('--compare-games', type=int, default=32)
    parser.add_argument('--confirm-games', type=int, default=64)
    parser.add_argument('--minimum-gain', type=float, default=150.)
    parser.add_argument('--first-training-seed', type=int, required=True)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()
    if (args.output.exists() or min(args.generations, args.screen_games,
                                     args.compare_games, args.confirm_games, args.envs) < 1
            or args.first_training_seed < 70000 or args.seed < 0
            or not np.isfinite(args.minimum_gain) or args.minimum_gain <= 0):
        parser.error('new output, positive settings and training-only seeds required')
    mx.set_cache_limit(128*1024*1024)
    model = QNetwork(action_count=20)
    rng = np.random.default_rng(args.seed)
    state = restore_checkpoint(model, args.parent, rng)
    config = dict(state['config'])
    if (config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('native_sha256') != sha256('libtrs.so')
            or config.get('action_names') != list(action_names())
            or config.get('allow_enter') or config.get('eval_max_steps') != 0):
        raise ValueError('incompatible complete-game action profile')
    source = Path(args.source_run)
    source_config = json.loads((source/'config.json').read_text())
    if source_config.get('initialization_model_sha256') != sha256(args.parent/'model.safetensors'):
        raise ValueError('aggregate source did not start from this exact model')
    center = head_array(model)
    candidates, provenance = archived_proposal(source, center, args.generations)
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'plan.npz', center=center, heads=candidates)
    config.pop('boot_search', None)
    config.update(parent_steps=state['steps'],
        initialization_checkpoint=str(args.parent.resolve()),
        initialization_model_sha256=sha256(args.parent/'model.safetensors'),
        initialization_state_sha256=sha256(args.parent/'state.json'),
        search_optimizer='score-weighted symmetric full-game aggregate; no gradients',
        training_method='fresh complete-boot selection of own-score aggregate head',
        fitness='displayed score in complete boot games only',
        aggregate_search=dict(source_run=str(source.resolve()),
            source_config_sha256=sha256(source/'config.json'),
            parent_model_sha256=sha256(args.parent/'model.safetensors'),
            minimum_gain=args.minimum_gain, **provenance),
        aggregate_source_sha256=sha256(Path(__file__)),
        args={key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()})
    write_json(args.output/'config.json', config)
    original_optimizer = Path(config['parent_checkpoint'])/'optimizer.npz'
    if sha256(original_optimizer) != config['parent_optimizer_sha256']:
        raise ValueError('original PPO optimizer changed')
    shutil.copy2(original_optimizer, args.output/'ppo-parent-optimizer.npz')
    save_checkpoint(model, args.output/'generation-000000', config,
        generation=0, training_steps=0, training_games=0, rng=rng,
        next_seed=args.first_training_seed)
    started = time.monotonic()
    metrics = (args.output/'metrics.jsonl').open('x')
    training_steps = training_games = 0
    def log(row):
        row = dict(row, elapsed_seconds=time.monotonic()-started)
        metrics.write(json.dumps(row)+'\n'); metrics.flush()
        print(json.dumps(row), flush=True)
    def progress(row):
        if row['event'] != 'training_game':
            log(row)
    def phase(name, heads, count, first_seed):
        nonlocal training_steps, training_games
        jobs = shared_seed_jobs(len(heads), count, first_seed)
        games, _ = play_population(model, heads, jobs, envs=args.envs,
            tstates=config['tstates'], observation_stride=config['observation_stride'],
            log=progress, keep_traces=False)
        write_json(args.output/f'{name}-games.json', games)
        training_steps += sum(game['steps'] for game in games)
        training_games += len(games)
        means = score_means(games, len(heads), count)
        log(dict(event='phase', phase=name, games=len(games),
            mean_min=float(means.min()), mean_max=float(means.max()),
            highest_stage=max(g['highest_stage'] for g in games)))
        disk_guard(args.output)
        return games, means
    try:
        screen, screen_means = phase('screen', candidates,
            args.screen_games, args.first_training_seed)
        selected = shortlist(screen_means, 2)
        compare_heads = np.concatenate((center[None], candidates[selected]))
        compare, compare_means = phase('compare', compare_heads,
            args.compare_games, args.first_training_seed+args.screen_games)
        best = int(np.argmax(compare_means[1:]))+1
        nominated = bool(compare_means[best] >= compare_means[0]+args.minimum_gain)
        confirm, confirm_means, confirm_heads = [], None, None
        if nominated:
            confirm_heads = np.stack((center, candidates[selected[best-1]]))
            confirm, confirm_means = phase('confirm', confirm_heads,
                args.confirm_games,
                args.first_training_seed+args.screen_games+args.compare_games)
        accepted = bool(nominated and
            confirm_means[1] >= confirm_means[0]+args.minimum_gain)
        ranked = ([('screen', game) for game in screen]
                  + [('compare', game) for game in compare]
                  + [('confirm', game) for game in confirm])
        winner_phase, winner = max(ranked, key=lambda pair: game_rank(pair[1]))
        winner_head = {'screen': candidates, 'compare': compare_heads,
                       'confirm': confirm_heads}[winner_phase][winner['candidate']]
        global_manifest = Path('results/defense/learned/best/manifest.json')
        global_rank = tuple(json.loads(global_manifest.read_text())['rank']) \
            if global_manifest.exists() else (-1, -1, -1)
        published = None
        if game_rank(winner) > global_rank:
            set_head(model, winner_head)
            save_checkpoint(model, args.output/'discovery', config,
                generation=0, training_steps=training_steps,
                training_games=training_games, rng=rng,
                next_seed=args.first_training_seed+args.screen_games+
                    args.compare_games+args.confirm_games, candidate=True)
            clean = {key: value for key, value in winner.items()
                     if key not in ('candidate', 'direction', 'sign', 'repetition')}
            write_json(args.output/'discovery/training-result.json',
                dict(phase=winner_phase, result=clean))
            published = publish_best(args.output/'discovery/model.safetensors',
                summarize([clean]), args.output/'artifacts', log=log)
        set_head(model, candidates[selected[best-1]] if accepted else center)
        next_seed = (args.first_training_seed+args.screen_games+
                     args.compare_games+args.confirm_games)
        save_checkpoint(model, args.output/'generation-000001', config,
            generation=1, training_steps=training_steps,
            training_games=training_games, rng=rng, next_seed=next_seed)
        report = dict(source='own archived complete boot-game displayed scores',
            selected_candidates=selected, candidate=selected[best-1],
            nominated=nominated, accepted=accepted,
            comparison_incumbent=float(compare_means[0]),
            comparison_candidate=float(compare_means[best]),
            confirmation_incumbent=float(confirm_means[0]) if nominated else None,
            confirmation_candidate=float(confirm_means[1]) if nominated else None,
            training_games=training_games, training_steps=training_steps,
            highest_stage=max(game['highest_stage'] for _, game in ranked),
            missions=max(game['missions_completed'] for _, game in ranked),
            independently_verified_discovery=published is not None)
        write_json(args.output/'report.json', report)
        log(dict(event='selection', **report))
        if accepted:
            evaluation = evaluate(acting_policy(model), range(10000, 10010),
                tstates=config['tstates'], max_steps=0, envs=10,
                observation_stride=config['observation_stride'], log=progress)
            write_json(args.output/'generation-000001/evaluation.json', evaluation)
            log(dict(event='validation', mean_score=evaluation['mean_score'],
                highest_stage=evaluation['highest_stage'],
                mission_games=evaluation['mission_games']))
            publish_best(args.output/'generation-000001/model.safetensors',
                evaluation, args.output/'artifacts', log=log)
        if sha256(args.output/'ppo-parent-optimizer.npz') != config['parent_optimizer_sha256']:
            raise RuntimeError('preserved PPO optimizer changed')
        log(dict(event='finished', **report))
    finally:
        metrics.close()


if __name__ == '__main__':
    main()
