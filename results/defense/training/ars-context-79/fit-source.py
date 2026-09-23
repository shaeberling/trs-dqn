"""Search a frozen screen encoder's action head using complete boot games.

Every candidate is first played from boot, including candidates that earn
fewer points before the familiar first-stage loss. Only displayed game score
selects updates. Saved native state is never used as a policy input or start.
"""

import argparse
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from .defense_ars_focus import perturbation_directions


KEY_FACTORS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')


def key_incidence(names):
    if len(names) != 20:
        raise ValueError('requires the fixed 20-command keyboard profile')
    incidence = np.zeros((len(names), len(KEY_FACTORS)), np.float32)
    for action, name in enumerate(names):
        tokens = () if name == 'NOOP' else name.split('+')
        if len(tokens) != len(set(tokens)) or any(token not in KEY_FACTORS for token in tokens):
            raise ValueError('unexpected keyboard command')
        for token in tokens:
            incidence[action, KEY_FACTORS.index(token)] = 1.
    return incidence


def key_factor_directions(rng, count, head_shape, names):
    """Coherently perturb every command containing a physical key.

    Symmetric candidates cover every key without a preferred direction.
    No stage, screen feature, obstacle location, or target action is supplied.
    """
    if (count < 1 or len(head_shape) != 2 or head_shape[0] != len(names)
            or min(head_shape) < 2):
        raise ValueError('invalid key-factor head shape')
    incidence = key_incidence(names)
    factors = rng.normal(size=(count, len(KEY_FACTORS), head_shape[1])).astype(np.float32)
    directions = np.einsum('ak,dkf->daf', incidence, factors, optimize=True)
    if not np.isfinite(directions).all():
        raise ValueError('nonfinite key-factor directions')
    return directions


def context_key_directions(rng, count, head_shape, names, basis):
    """Symmetric physical-key proposals along an own-screen feature contrast."""
    basis = np.asarray(basis, np.float32)
    if (count < 1 or len(head_shape) != 2 or head_shape[0] != len(names)
            or head_shape[1] != len(basis) or not np.isfinite(basis).all()):
        raise ValueError('invalid visible context basis/head')
    incidence = key_incidence(names)
    coefficients = rng.normal(size=(count, len(KEY_FACTORS))).astype(np.float32)
    directions = np.einsum('ak,dk->da', incidence, coefficients, optimize=True)[:, :, None]*basis
    if not np.isfinite(directions).all():
        raise ValueError('nonfinite visible-context directions')
    return directions.astype(np.float32)


def candidate_scales(rng, count, sigma, sigma_max=None):
    """Use paired, stratified log-scale radii without favoring any key."""
    if (count < 1 or not np.isfinite(sigma) or sigma <= 0
            or (sigma_max is not None and
                (not np.isfinite(sigma_max) or sigma_max <= sigma))):
        raise ValueError('invalid symmetric search radii')
    if sigma_max is None:
        return np.full(count, sigma, np.float32)
    return rng.permutation(np.geomspace(sigma, sigma_max, count).astype(np.float32))


def shared_seed_jobs(candidates, repetitions, first_seed):
    """All candidates see the same boot and action-sampling seeds."""
    if min(candidates, repetitions) < 1 or first_seed < 70000:
        raise ValueError('positive candidate count and training seeds required')
    return [dict(candidate=candidate, direction=candidate//2,
                 sign=candidate%2, repetition=repetition,
                 seed=first_seed+repetition)
            for candidate in range(candidates) for repetition in range(repetitions)]


def score_means(games, candidates, repetitions):
    scores = np.full((candidates, repetitions), np.nan)
    for game in games:
        candidate, repetition = game['candidate'], game['repetition']
        if (not 0 <= candidate < candidates or not 0 <= repetition < repetitions
                or np.isfinite(scores[candidate, repetition])):
            raise ValueError('duplicate or out-of-range full-game return')
        scores[candidate, repetition] = game['score']
    if len(games) != candidates*repetitions or not np.isfinite(scores).all():
        raise ValueError('missing or invalid full-game return')
    return scores.mean(axis=1)


def shortlist(means, count):
    means = np.asarray(means, np.float64)
    if (means.ndim != 1 or not len(means) or not np.isfinite(means).all()
            or not 1 <= count <= len(means)):
        raise ValueError('invalid complete-boot shortlist')
    return np.argsort(-means, kind='stable')[:count].tolist()


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .defense import action_names, GAME_SHA256, ENVIRONMENT_VERSION
    from .defense_ars import (acting_policy, head_array, play_population,
                              restore_checkpoint, save_checkpoint, set_head)
    from .defense_learning import (evaluate, game_rank, publish_best, sha256,
                                   summarize, write_json)
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--initialize', type=Path, help='complete own ARS centroid checkpoint')
    origin.add_argument('--resume', type=Path, help='complete boot-search generation checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=0, help='0 learns until stopped or mission verified')
    parser.add_argument('--directions', type=int, default=20)
    parser.add_argument('--direction-mode',
                        choices=('action-row', 'key-factor', 'failure-context-key'), default='action-row')
    parser.add_argument('--context-archive', type=Path,
                        help='own visible pre-loss training screens, required for failure-context-key')
    parser.add_argument('--sigma', type=float, default=.05)
    parser.add_argument('--sigma-max', type=float,
                        help='stratified upper perturbation radius; omitted uses one radius')
    parser.add_argument('--shortlist', type=int, default=4)
    parser.add_argument('--screen-games', type=int, default=4)
    parser.add_argument('--compare-games', type=int, default=16)
    parser.add_argument('--confirm-games', type=int, default=16)
    parser.add_argument('--minimum-boot-gain', type=float, default=100.)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=131)
    parser.add_argument('--first-training-seed', type=int, default=100000)
    parser.add_argument('--eval-every', type=int, default=5)
    args = parser.parse_args()
    if (args.output.exists() or args.generations < 0 or args.directions != 20
            or min(args.shortlist, args.screen_games, args.compare_games,
                   args.confirm_games, args.envs, args.eval_every) < 1
            or args.shortlist > 2*args.directions
            or args.first_training_seed < 70000 or args.seed < 0
            or not np.isfinite([args.sigma, args.minimum_boot_gain]).all()
            or (args.sigma_max is not None and
                (not np.isfinite(args.sigma_max) or args.sigma_max <= args.sigma))
            or (args.context_archive is None) != (args.direction_mode != 'failure-context-key')
            or min(args.sigma, args.minimum_boot_gain) <= 0):
        parser.error('new output, all 20 action rows and positive score-search settings required')
    mx.set_cache_limit(128*1024*1024)
    mx.random.seed(args.seed)
    model = QNetwork(action_count=20)
    rng = np.random.default_rng(args.seed)
    settings = {name: getattr(args, name) for name in
                ('directions', 'direction_mode', 'sigma', 'sigma_max', 'shortlist', 'screen_games',
                 'compare_games', 'confirm_games', 'minimum_boot_gain', 'envs')}
    settings['context_archive'] = str(args.context_archive.resolve()) if args.context_archive else None
    if args.initialize:
        state = restore_checkpoint(model, args.initialize, np.random.default_rng(0))
        config = dict(state['config'])
        config.update(parent_steps=state['steps'],
            initialization_checkpoint=str(args.initialize.resolve()),
            initialization_model_sha256=sha256(args.initialize/'model.safetensors'),
            initialization_state_sha256=sha256(args.initialize/'state.json'),
            initialization_generation=state['generations'],
            initialization_training_steps=state['training_steps'],
            boot_search=settings, search_optimizer='paired complete-boot neural score search; no gradients',
            training_method=('all-action coordinate-row population' if args.direction_mode == 'action-row'
                             else 'own visible-failure-context physical-key population'
                             if args.direction_mode == 'failure-context-key'
                             else 'all-physical-key factorized population')
                + ', score-gated on three fresh full-game training seed sets',
            fitness='displayed score in complete games from boot; no local score prerequisite')
        generation, training_steps, training_games = 0, 0, 0
        next_seed = args.first_training_seed
    else:
        state = restore_checkpoint(model, args.resume, rng)
        config = dict(state['config'])
        prior_settings = dict(config.get('boot_search', {}))
        prior_settings.setdefault('direction_mode', 'action-row')
        prior_settings.setdefault('sigma_max', None)
        prior_settings.setdefault('context_archive', None)
        if prior_settings != settings:
            parser.error('resume settings differ from saved boot search')
        config['boot_search'] = prior_settings
        generation, training_steps, training_games, next_seed = (state[name] for name in
            ('generations', 'training_steps', 'training_games', 'next_training_seed'))
    if (config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('native_sha256') != sha256('libtrs.so')
            or config.get('action_names') != list(action_names())
            or config.get('allow_enter') or config.get('eval_max_steps') != 0):
        raise ValueError('incompatible game or action profile')
    if args.generations and generation >= args.generations:
        parser.error('target must exceed restored generation')
    parent = Path(config['parent_checkpoint'])
    if sha256(parent/'optimizer.npz') != config['parent_optimizer_sha256']:
        raise ValueError('original PPO optimizer changed')
    immutable = {key: np.array(value).copy() for key, value in tree_flatten(model.parameters())
                 if not key.startswith('advantage.')}
    context_basis = None
    if args.context_archive is not None:
        from .defense_ars_context import visible_context
        context_basis, context = visible_context(model, args.context_archive,
            config['initialization_model_sha256'])
        if args.initialize:
            config['context'] = context
        elif config.get('context') != context:
            raise ValueError('resumed visible context differs from checkpoint')
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent/'optimizer.npz', args.output/'ppo-parent-optimizer.npz')
    if context_basis is not None:
        np.savez_compressed(args.output/'context-basis.npz', basis=context_basis)
    config.update(boot_search_source_sha256=sha256(Path(__file__)),
        search_source_sha256=sha256(Path(__file__).with_name('defense_ars.py')),
        args={key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()})
    write_json(args.output/'config.json', config)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as stream:
        def log(row):
            row = dict(row, elapsed_seconds=time.monotonic()-started)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(json.dumps(row), flush=True)
        def progress(row):
            if row['event'] != 'training_game':
                log(row)
        def preserve(location=None, candidate=False):
            disk_guard(args.output)
            location = location or args.output/f'generation-{generation:06d}'
            save_checkpoint(model, location, config, generation=generation,
                training_steps=training_steps, training_games=training_games,
                rng=rng, next_seed=next_seed, candidate=candidate)
            return location
        def validate(location):
            nonlocal stop
            result = evaluate(acting_policy(model), range(10000, 10010),
                tstates=config['tstates'], max_steps=0, envs=10,
                observation_stride=config['observation_stride'], log=progress)
            write_json(location/'evaluation.json', result)
            log(dict(event='validation', generation=generation,
                **{key: value for key, value in result.items() if key != 'games'}))
            publish_best(location/'model.safetensors', result,
                args.output/'artifacts', log=log)
            if result['mission_games']:
                stop = True
                log(dict(event='verified_mission_in_complete_boot_validation', generation=generation))
        def phase(location, name, heads, count, seed):
            nonlocal training_steps, training_games
            jobs = shared_seed_jobs(len(heads), count, seed)
            games, _ = play_population(model, heads, jobs, envs=args.envs,
                tstates=config['tstates'], observation_stride=config['observation_stride'],
                log=progress, keep_traces=False)
            write_json(location/f'{name}-games.json', games)
            training_steps += sum(game['steps'] for game in games)
            training_games += len(games)
            means = score_means(games, len(heads), count)
            log(dict(event='phase', generation=generation+1, phase=name,
                     games=len(games), mean_min=float(means.min()),
                     mean_max=float(means.max()), highest_stage=max(g['highest_stage'] for g in games)))
            disk_guard(args.output)
            return games, means
        checkpoint = preserve()
        validate(checkpoint)
        while not stop and (not args.generations or generation < args.generations):
            center = head_array(model)
            directions = (perturbation_directions(rng, args.directions,
                center.shape, coordinate_row=True) if args.direction_mode == 'action-row'
                else context_key_directions(rng, args.directions, center.shape,
                    action_names(), context_basis) if args.direction_mode == 'failure-context-key'
                else key_factor_directions(rng, args.directions, center.shape, action_names()))
            scales = candidate_scales(rng, args.directions, args.sigma, args.sigma_max)
            perturbations = scales[:, None, None]*directions
            heads = np.stack([center+perturbations,
                              center-perturbations], axis=1).reshape(-1, *center.shape)
            target = args.output/f'population-{generation+1:06d}'
            target.mkdir(exist_ok=False)
            np.savez_compressed(target/'plan.npz', center=center,
                                directions=directions, scales=scales, heads=heads)
            first_seed = next_seed
            screen, screen_mean = phase(target, 'screen', heads,
                args.screen_games, first_seed)
            selected = shortlist(screen_mean, args.shortlist)
            compare_heads = np.concatenate([center[None], heads[selected]], axis=0)
            compare, compare_mean = phase(target, 'compare', compare_heads,
                args.compare_games, first_seed+args.screen_games)
            best = int(np.argmax(compare_mean[1:]))+1
            nominated = compare_mean[best] >= compare_mean[0]+args.minimum_boot_gain
            confirm, confirm_mean = [], None
            accepted = False
            if nominated:
                confirm_heads = np.stack([center, heads[selected[best-1]]])
                confirm, confirm_mean = phase(target, 'confirm', confirm_heads,
                    args.confirm_games, first_seed+args.screen_games+args.compare_games)
                accepted = bool(confirm_mean[1] >= confirm_mean[0]+args.minimum_boot_gain)
            next_seed += args.screen_games+args.compare_games+args.confirm_games
            generation += 1
            ranked = ([('screen', game) for game in screen]
                      + [('compare', game) for game in compare]
                      + [('confirm', game) for game in confirm])
            winner_phase, winner = max(ranked, key=lambda row: game_rank(row[1]))
            winner_head = {'screen': heads, 'compare': compare_heads,
                           'confirm': confirm_heads if confirm else None}[winner_phase][winner['candidate']]
            all_games = screen+compare+confirm
            global_manifest = Path('results/defense/learned/best/manifest.json')
            global_rank = tuple(json.loads(global_manifest.read_text())['rank']) \
                if global_manifest.exists() else (-1, -1, -1)
            published = None
            if game_rank(winner) > global_rank:
                set_head(model, winner_head)
                discovery = preserve(target/'discovery', candidate=True)
                clean_winner = {key: value for key, value in winner.items()
                                if key not in ('candidate', 'direction', 'sign', 'repetition')}
                write_json(discovery/'training-result.json',
                    dict(phase=winner_phase, candidate=winner['candidate'], result=clean_winner))
                published = publish_best(discovery/'model.safetensors', summarize([clean_winner]),
                    args.output/'artifacts', log=log)
            following = heads[selected[best-1]] if accepted else center
            set_head(model, following)
            for key, value in tree_flatten(model.parameters()):
                if key in immutable:
                    np.testing.assert_array_equal(np.array(value), immutable[key])
            np.savez_compressed(target/'update.npz', screen_means=screen_mean,
                compare_means=compare_mean,
                confirm_means=np.asarray(confirm_mean if confirm_mean is not None else [], np.float64),
                following=following)
            checkpoint = preserve()
            log(dict(event='generation', generation=generation,
                training_steps=training_steps, training_games=training_games,
                selected_candidates=selected, candidate=selected[best-1],
                nominated=bool(nominated), accepted=accepted,
                comparison_incumbent=float(compare_mean[0]),
                comparison_candidate=float(compare_mean[best]),
                confirmation_incumbent=float(confirm_mean[0]) if confirm_mean is not None else None,
                confirmation_candidate=float(confirm_mean[1]) if confirm_mean is not None else None,
                highest_stage=max(g['highest_stage'] for g in all_games)))
            if winner['missions_completed'] and published is not None:
                stop = True
                log(dict(event='verified_mission_in_complete_training_boot_game',
                         generation=generation, replay=str(published)))
            if generation % args.eval_every == 0 or stop or generation == args.generations:
                validate(checkpoint)
        if sha256(args.output/'ppo-parent-optimizer.npz') != config['parent_optimizer_sha256']:
            raise RuntimeError('preserved PPO optimizer changed')
        log(dict(event='finished', generation=generation, training_steps=training_steps,
                 training_games=training_games, stopped=stop,
                 frozen_encoder_and_value_verified=True))


if __name__ == '__main__':
    main()
