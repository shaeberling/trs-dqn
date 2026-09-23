"""Real-game score search over the strong own policy's categorical action head."""

import argparse
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np


def main():
    # Keep native emulator spawn workers free of MLX/GPU initialization.
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .defense import action_names, GAME_SHA256, ENVIRONMENT_VERSION
    from .defense_ars import (ALGORITHM, acting_policy, ars_update, candidate_jobs, head_array,
                              play_population, restore_checkpoint, save_checkpoint, set_head)
    from .defense_learning import (evaluate, game_rank, policy_description, publish_best, sha256,
                                  summarize, verify_policy_trace, write_json)
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--initialize', type=Path, help='complete own PPO checkpoint; new search algorithm, not PPO resume')
    origin.add_argument('--resume', type=Path, help='complete ARS generation-boundary checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=6, help='absolute generation target; 0 is unlimited')
    parser.add_argument('--directions', type=int, default=16)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=.005)
    parser.add_argument('--step-size', type=float, default=.002)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--first-training-seed', type=int, default=70000)
    parser.add_argument('--eval-every', type=int, default=1)
    args = parser.parse_args()
    if (args.output.exists() or args.generations < 0 or min(args.directions, args.repetitions, args.envs, args.eval_every) < 1
            or args.first_training_seed < 70000 or args.seed < 0
            or not np.isfinite([args.sigma, args.step_size]).all() or min(args.sigma, args.step_size) <= 0):
        parser.error('new output, positive search settings and separate training seeds required')
    mx.set_cache_limit(128*1024*1024)
    mx.random.seed(args.seed)
    model, rng = QNetwork(action_count=20), np.random.default_rng(args.seed)
    generation, training_steps, training_games, next_seed = 0, 0, 0, args.first_training_seed
    settings = {k: getattr(args, k) for k in ('directions', 'repetitions', 'sigma', 'step_size')}
    if args.initialize:
        parent = args.initialize
        state = json.loads((parent/'state.json').read_text())
        old = state['config']
        if (old.get('algorithm', 'ppo') != 'ppo' or old.get('architecture') or old.get('allow_enter')
                or old.get('game') != 'defense' or old.get('game_sha256') != GAME_SHA256
                or old.get('environment_version') != ENVIRONMENT_VERSION
                or old.get('action_names') != list(action_names()) or not (parent/'optimizer.npz').is_file()):
            parser.error('requires a complete compatible own feedforward PPO checkpoint')
        model.load_weights(str(parent/'model.safetensors'))
        config = dict(game='defense', algorithm=ALGORITHM, game_sha256=GAME_SHA256,
            native_sha256=sha256('libtrs.so'), environment_version=ENVIRONMENT_VERSION,
            action_names=list(action_names()), allow_enter=False, tstates=old['tstates'],
            observation_stride=old.get('observation_stride', 1), eval_max_steps=0,
            parent_checkpoint=str(parent.resolve()), parent_steps=state['steps'],
            parent_model_sha256=sha256(parent/'model.safetensors'),
            parent_optimizer_sha256=sha256(parent/'optimizer.npz'),
            parent_state_sha256=sha256(parent/'state.json'),
            training_method='ARS V1 adapted to own pretrained screen encoder and sampled categorical head',
            search=settings, score_reward_only=True, frozen_encoder=True, frozen_value=True,
            source_paper='https://arxiv.org/html/1803.07055v1',
            initialization='all own PPO weights retained; old PPO Adam preserved but not used by new ARS optimizer')
    else:
        state = restore_checkpoint(model, args.resume, rng)
        config = dict(state['config'])
        if config['search'] != settings:
            parser.error('resume requires identical search settings')
        parent = Path(config['parent_checkpoint'])
        generation, training_steps, training_games, next_seed = (state[k] for k in
            ('generations', 'training_steps', 'training_games', 'next_training_seed'))
    if args.generations and generation >= args.generations:
        parser.error('target must exceed restored generation')
    if sha256(parent/'optimizer.npz') != config['parent_optimizer_sha256']:
        raise ValueError('preserved parent optimizer checksum mismatch')
    if (config.get('game') != 'defense' or config.get('game_sha256') != GAME_SHA256
            or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('action_names') != list(action_names()) or config.get('allow_enter')
            or config.get('native_sha256') != sha256('libtrs.so')):
        raise ValueError('incompatible search environment')
    policy_description(config)
    config.update(source_sha256=sha256(Path(__file__)),
        search_source_sha256=sha256(Path(__file__).with_name('defense_ars.py')),
        loader_source_sha256=sha256(Path(__file__).with_name('defense_learning.py')),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    mx.eval(model.state)
    immutable = {k: np.array(v).copy() for k, v in tree_flatten(model.parameters()) if not k.startswith('advantage.')}
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent/'optimizer.npz', args.output/'ppo-parent-optimizer.npz')
    shutil.copy2(parent/'state.json', args.output/'ppo-parent-state.json')
    write_json(args.output/'config.json', config)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True  # Finish and preserve the current real-game population.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(row):
            row = dict(row, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
        def preserve(candidate=False, destination=None):
            disk_guard(args.output)
            destination = destination or args.output/f'generation-{generation:06d}'
            save_checkpoint(model, destination, config, generation=generation, training_steps=training_steps,
                            training_games=training_games, rng=rng, next_seed=next_seed, candidate=candidate)
            return destination
        def validate(checkpoint, extra_game=None):
            result = evaluate(acting_policy(model), range(10000, 10010), tstates=config['tstates'],
                max_steps=0, envs=min(args.envs, 10), observation_stride=config['observation_stride'], log=record)
            write_json(checkpoint/'evaluation.json', result)
            record(dict(event='validation', generation=generation, candidate=extra_game is not None,
                        **{k: v for k, v in result.items() if k != 'games'}))
            promotion = result
            if extra_game is not None:
                promotion = summarize(result['games']+[extra_game])
                promotion.update(includes_verified_training_game=True,
                    validation_only={k: v for k, v in result.items() if k != 'games'},
                    selection_note='ten complete validation games plus separately verified training discovery')
                write_json(checkpoint/'promotion.json', promotion)
            publish_best(checkpoint/'model.safetensors', promotion, args.output/'artifacts', log=record)
        checkpoint = preserve()
        validate(checkpoint)
        while not stop and (not args.generations or generation < args.generations):
            disk_guard(args.output)
            center = head_array(model)
            directions = rng.normal(size=(args.directions, *center.shape)).astype(np.float32)
            heads = np.stack([center+args.sigma*directions, center-args.sigma*directions], axis=1).reshape(-1, *center.shape)
            jobs = candidate_jobs(args.directions, args.repetitions, next_seed)
            generation_dir = args.output/f'population-{generation+1:06d}'
            generation_dir.mkdir(exist_ok=False)
            np.savez_compressed(generation_dir/'plan.npz', center=center, directions=directions, heads=heads)
            write_json(generation_dir/'jobs.json', dict(jobs=jobs, config=config,
                generation=generation+1, rng_after_plan=rng.bit_generator.state,
                plan_sha256=sha256(generation_dir/'plan.npz')))
            games, traces = play_population(model, heads, jobs, envs=args.envs,
                tstates=config['tstates'], observation_stride=config['observation_stride'], log=record, keep_traces=True)
            write_json(generation_dir/'games.json', games)
            returns = np.empty((args.directions, 2, args.repetitions), np.float64)
            for game in games:
                returns[game['direction'], game['sign'], game['repetition']] = game['score']
            training_steps += sum(game['steps'] for game in games)
            training_games += len(games)
            next_seed += args.directions*args.repetitions
            following, stats = ars_update(center, directions, returns, args.step_size)
            np.savez_compressed(generation_dir/'update.npz', returns=returns, following=following)
            generation += 1
            # Preserve and verify a training discovery even if the averaged
            # parameter update itself does not retain that behavior.
            winner_index = max(range(len(games)), key=lambda i: game_rank(games[i]))
            winner = games[winner_index]
            global_best = Path('results/defense/learned/best/manifest.json')
            global_rank = tuple(json.loads(global_best.read_text())['rank']) if global_best.exists() else (-1, -1, -1)
            if game_rank(winner) > global_rank:
                set_head(model, heads[winner['candidate']])
                candidate = preserve(True, generation_dir/'candidate')
                game = {k: v for k, v in winner.items() if k not in ('candidate', 'direction', 'sign', 'repetition')}
                frames, actions, rewards = traces[winner_index]
                np.savez_compressed(candidate/'training-trace.npz', frames=frames, actions=actions, rewards=rewards)
                write_json(candidate/'training-result.json', game)
                verification = verify_policy_trace(candidate/'model.safetensors', frames, actions, rewards, game)
                write_json(candidate/'training-verification.json', verification)
                validate(candidate, game)
            set_head(model, following)
            for key, value in tree_flatten(model.parameters()):
                if key in immutable:
                    np.testing.assert_array_equal(np.array(value), immutable[key])
            checkpoint = preserve()
            record(dict(event='generation', generation=generation, training_steps=training_steps,
                training_games=training_games, best_training_game=winner, **stats))
            if generation % args.eval_every == 0 or stop or generation == args.generations:
                validate(checkpoint)
        if sha256(args.output/'ppo-parent-optimizer.npz') != config['parent_optimizer_sha256']:
            raise RuntimeError('preserved parent optimizer changed')
        record(dict(event='finished', generation=generation, training_steps=training_steps,
                    training_games=training_games, frozen_encoder_and_value_verified=True, stopped=stop))


if __name__ == '__main__':
    main()
