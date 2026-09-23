"""Train a screen-only Defense policy from its own pre-loss emulator states.

Opaque states are selected after visible life losses in complete *training*
games. A paired parameter search then measures only real displayed score
gained until the next visible life/stage boundary. Boot validation remains
ordinary complete games. No recorded source actions are used as targets.
"""

import argparse
from collections import deque
from dataclasses import fields
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np


def save_source(path, saved, actions, rewards, screens, metadata):
    from .defense_learning import write_json, sha256
    path = Path(path)
    if path.exists() or len(actions) != len(rewards) or len(actions) != len(screens):
        raise ValueError('invalid own-state source trace')
    values = {field.name: getattr(saved, field.name) for field in fields(saved)
              if field.name not in ('native', 'frames')}
    np.savez_compressed(path, native=np.frombuffer(saved.native, np.uint8),
        frames=saved.frames, actions=np.asarray(actions, np.uint8),
        rewards=np.asarray(rewards, np.float32), screens=np.asarray(screens, np.uint8))
    write_json(path.with_suffix('.json'), dict(snapshot_fields=values, source=metadata,
        npz_sha256=sha256(path),
        selection=f'{len(actions)} actions before own visible life loss'))


def load_source(path):
    from .defense_snapshot import DefenseSnapshot
    from .defense_learning import sha256
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != record['npz_sha256']:
        raise ValueError('own-state source checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        saved = DefenseSnapshot(native=data['native'].tobytes(), frames=data['frames'],
                                **record['snapshot_fields'])
        actions, rewards, screens = (data[name].copy() for name in ('actions', 'rewards', 'screens'))
    if (len(actions) != len(rewards) or len(actions) != len(screens)
            or screens.shape != (len(actions), 16, 64)):
        raise ValueError('invalid own-state source arrays')
    return saved, actions, rewards, screens, record


def verify_source(env, item):
    from .defense_snapshot import restore
    saved, actions, rewards, screens, _ = item
    obs = restore(env, saved)
    np.testing.assert_array_equal(obs, saved.frames[::saved.observation_stride])
    for action, reward, screen in zip(actions, rewards, screens, strict=True):
        obs, actual, _, _, _ = env.step(int(action))
        if actual != reward:
            raise RuntimeError('restored source reward mismatch')
        np.testing.assert_array_equal(obs[-1], screen)
    return len(actions)


def harvest(model, archive, *, count, first_seed, lookback, tstates, observation_stride, log):
    from .defense import DefenseEnv
    from .defense_ars import acting_policy
    from .defense_snapshot import capture
    from .defense_learning import game_rank, write_json
    if count < 1 or lookback < 1 or first_seed < 70000:
        raise ValueError('need own training seeds and positive lookback')
    archive.mkdir(parents=True, exist_ok=False)
    env = DefenseEnv(tstates=tstates, max_steps=0, observation_stride=observation_stride)
    policy = acting_policy(model)
    source_games, sources, total_steps = [], [], 0
    try:
        for seed in range(first_seed, first_seed+count):
            policy.reset_seed(seed+1_000_000)
            obs = env.reset(seed)
            history = deque(maxlen=lookback)
            life = 1
            while True:
                before = capture(env)
                action = int(policy(obs[None])[0])
                obs, reward, terminal, truncated, info = env.step(action)
                history.append((before, action, reward, obs[-1].copy()))
                if info['life_lost']:
                    if len(history) == lookback:
                        saved = history[0][0]
                        if saved.lives != before.lives or saved.stage != before.stage:
                            raise RuntimeError('selected state crosses a life or stage boundary')
                        filename = f'seed-{seed}-life-{life}.npz'
                        metadata = dict(seed=seed, life=life, visible_loss_frame=env.steps,
                            visible_loss_score=info['score'], snapshot_action=saved.steps,
                            snapshot_score=saved.score, snapshot_lives=saved.lives,
                            source_policy='own frozen strong neural policy',
                            training_only=True, promotion_eligible=False)
                        save_source(archive/filename, saved,
                            [row[1] for row in history], [row[2] for row in history],
                            [row[3] for row in history], metadata)
                        sources.append(filename)
                    history.clear()
                    life += 1
                if info['stage'] != before.stage:
                    history.clear()
                if terminal or truncated:
                    result = dict(seed=seed, **info)
                    if game_rank(result) is None:
                        raise ValueError('snapshot harvest requires complete own games')
                    total_steps += info['steps']
                    source_games.append(result)
                    log(dict(event='harvest_game', seed=seed, score=info['score'],
                             steps=info['steps'], states=len(sources)))
                    break
        if not sources:
            raise RuntimeError('own games yielded no full same-life rewind states')
        for filename in sources:
            verify_source(env, load_source(archive/filename))
        write_json(archive/'index.json', dict(files=sources, games=source_games,
            training_steps=total_steps, first_seed=first_seed, lookback=lookback,
            selection='own complete-game visible life losses, exactly lookback earlier',
            checked='restore exact native state and replay recorded source actions, screens and rewards'))
        log(dict(event='harvest_finished', games=len(source_games), snapshots=len(sources),
                 training_steps=total_steps))
    finally:
        env.close()
    return sources, source_games, total_steps


def play_segment(env, saved, infer, candidate, seed, *, keep_trace=False):
    from .defense_snapshot import restore
    obs = restore(env, saved)
    rng = np.random.default_rng(seed+1_000_000)
    start_score, start_stage = saved.score, saved.stage
    actions, rewards, frames = [], [], [obs[-1].copy()]
    earned = 0.
    for step in range(1, 30001):
        values = infer(obs[None], [candidate])[0]
        probabilities = np.exp(values-np.logaddexp.reduce(values))
        action = int((rng.random() > np.cumsum(probabilities)).sum().clip(0, len(values)-1))
        obs, reward, terminal, truncated, info = env.step(action)
        earned += reward
        if keep_trace:
            actions.append(action)
            rewards.append(reward)
            frames.append(obs[-1].copy())
        if info['life_lost'] or info['stage'] != start_stage or info['mission_completed'] or terminal or truncated:
            if truncated or earned != env.score-start_score:
                raise ValueError('focused segment reward mismatch or truncation')
            result = dict(score_gain=env.score-start_score, start_score=start_score,
                final_score=env.score, steps=step, stage=info['stage'],
                highest_stage=info['highest_stage'], life_lost=bool(info['life_lost']),
                mission_completed=bool(info['mission_completed']),
                terminated=bool(terminal), truncated=bool(truncated),
                source_action=saved.source_action, seed=seed)
            trace = (np.asarray(frames, np.uint8), np.asarray(actions, np.uint8),
                     np.asarray(rewards, np.float32)) if keep_trace else None
            return result, trace
    raise RuntimeError('focused segment exceeded 30000 neural actions without a visible boundary')


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .defense import DefenseEnv, GAME_SHA256, ENVIRONMENT_VERSION, action_names
    from .defense_ars import (acting_policy, ars_update, head_array, make_population_infer,
        restore_checkpoint, save_checkpoint, set_head)
    from .defense_learning import evaluate, publish_best, sha256, write_json
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--initialize', type=Path, help='full ARS checkpoint with strong own policy')
    origin.add_argument('--resume', type=Path, help='saved focused generation checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=8, help='absolute target; 0 keeps learning')
    parser.add_argument('--harvest-games', type=int, default=12)
    parser.add_argument('--harvest-seed', type=int, default=71000)
    parser.add_argument('--lookback', type=int, default=128)
    parser.add_argument('--directions', type=int, default=16)
    parser.add_argument('--snapshots-per-direction', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=.005)
    parser.add_argument('--step-size', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=81)
    parser.add_argument('--eval-every', type=int, default=1)
    args = parser.parse_args()
    if (args.output.exists() or args.generations < 0 or args.harvest_seed < 70000
            or min(args.harvest_games, args.lookback, args.directions,
                   args.snapshots_per_direction, args.eval_every) < 1
            or not np.isfinite([args.sigma, args.step_size]).all()
            or min(args.sigma, args.step_size) <= 0):
        parser.error('new output and positive focus settings required')
    mx.set_cache_limit(128*1024*1024)
    mx.random.seed(args.seed)
    model, rng = QNetwork(action_count=20), np.random.default_rng(args.seed)
    settings = {key: getattr(args, key) for key in ('directions', 'snapshots_per_direction', 'sigma', 'step_size')}
    if args.initialize:
        state = restore_checkpoint(model, args.initialize, rng)
        if state['generations'] != 0 or state['training_steps'] != 0:
            parser.error('focus starts from a full generation-zero ARS checkpoint')
        config = dict(state['config'])
        config.update(focus=dict(settings=settings, harvest_games=args.harvest_games,
            harvest_seed=args.harvest_seed, lookback=args.lookback,
            source_checkpoint=str(args.initialize.resolve()),
            source_model_sha256=sha256(args.initialize/'model.safetensors'),
            selection='own training-game visible life losses, opaque rewind',
            fitness='displayed score gain until next visible life/stage boundary',
            evaluation='independent complete games from boot; no reset states'),
            training_method='paired real-score ARS over own pre-loss state continuations')
        generation, training_steps, training_segments = 0, 0, 0
    else:
        state = restore_checkpoint(model, args.resume, rng)
        config = dict(state['config'])
        prior_focus = config.get('focus', {})
        if (prior_focus.get('settings') != settings
                or any(prior_focus.get(field) != getattr(args, field)
                       for field in ('harvest_games', 'harvest_seed', 'lookback'))):
            parser.error('focus resume settings differ')
        generation = state['generations']
        training_steps = state['training_steps']
        training_segments = state['training_games']
    if (config.get('game_sha256') != GAME_SHA256 or config.get('environment_version') != ENVIRONMENT_VERSION
            or config.get('native_sha256') != sha256('libtrs.so') or config.get('action_names') != list(action_names())
            or config.get('allow_enter') or config.get('eval_max_steps') != 0):
        raise ValueError('incompatible focus environment or action profile')
    if args.generations and generation >= args.generations:
        parser.error('target must exceed restored generation')
    parent = Path(config['parent_checkpoint'])
    if sha256(parent/'optimizer.npz') != config['parent_optimizer_sha256']:
        raise ValueError('original PPO optimizer changed')
    immutable = {key: np.array(value).copy() for key, value in tree_flatten(model.parameters())
                 if not key.startswith('advantage.')}
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent/'optimizer.npz', args.output/'ppo-parent-optimizer.npz')
    archive = args.output/'own-loss-states'
    started = time.monotonic()
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (args.output/'metrics.jsonl').open('x') as stream:
        def log(row):
            row = dict(row, elapsed_seconds=time.monotonic()-started)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(json.dumps(row), flush=True)
        if args.initialize:
            names, source_games, harvest_steps = harvest(model, archive, count=args.harvest_games,
                first_seed=args.harvest_seed, lookback=args.lookback, tstates=config['tstates'],
                observation_stride=config['observation_stride'], log=log)
            training_steps += harvest_steps
        else:
            source = Path(config['focus']['archive'])
            if not source.is_dir():
                source = args.resume.parent.parent/'own-loss-states'
            shutil.copytree(source, archive)
            index = json.loads((archive/'index.json').read_text())
            names = index['files']
            source_games = index['games']
            for name in names:
                if sha256(archive/name) != config['focus']['archive_hashes'][name]:
                    raise ValueError('resumed own-state archive differs from checkpoint')
                load_source(archive/name)
        if len(names) < args.snapshots_per_direction:
            raise ValueError('not enough own rewind states for paired training')
        config['focus']['archive'] = str(archive.resolve())
        config['focus']['archive_hashes'] = {name: sha256(archive/name) for name in names}
        config.update(focus_source_sha256=sha256(Path(__file__)),
            search_source_sha256=sha256(Path(__file__).with_name('defense_ars.py')),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        write_json(args.output/'config.json', config)
        sources = [load_source(archive/name)[0] for name in names]
        def preserve():
            disk_guard(args.output)
            location = args.output/f'generation-{generation:06d}'
            save_checkpoint(model, location, config, generation=generation,
                training_steps=training_steps, training_games=training_segments,
                rng=rng, next_seed=80000+generation*args.snapshots_per_direction)
            return location
        def validate(location):
            result = evaluate(acting_policy(model), range(10000, 10010), tstates=config['tstates'],
                max_steps=0, envs=10, observation_stride=config['observation_stride'], log=log)
            write_json(location/'evaluation.json', result)
            log(dict(event='validation', generation=generation,
                     **{k: v for k, v in result.items() if k != 'games'}))
            publish_best(location/'model.safetensors', result, args.output/'artifacts', log=log)
        location = preserve()
        validate(location)
        env = DefenseEnv(tstates=config['tstates'], max_steps=0,
            observation_stride=config['observation_stride'])
        try:
            while not stop and (not args.generations or generation < args.generations):
                disk_guard(args.output)
                chosen = rng.choice(len(sources), args.snapshots_per_direction, replace=False)
                center = head_array(model)
                directions = rng.normal(size=(args.directions, *center.shape)).astype(np.float32)
                heads = np.stack([center+args.sigma*directions,
                                  center-args.sigma*directions], axis=1).reshape(-1, *center.shape)
                target = args.output/f'population-{generation+1:06d}'
                target.mkdir(exist_ok=False)
                np.savez_compressed(target/'plan.npz', center=center, directions=directions,
                                    heads=heads, chosen=chosen)
                infer = make_population_infer(model, heads)
                returns = np.empty((args.directions, 2, len(chosen)), np.float64)
                rows, count_steps = [], 0
                best_row = None
                for direction in range(args.directions):
                    for sign in range(2):
                        candidate = 2*direction+sign
                        for repetition, index in enumerate(chosen):
                            sample_seed = 80000+generation*len(chosen)+repetition
                            result, _ = play_segment(env, sources[index], infer, candidate, sample_seed)
                            row = dict(direction=direction, sign=sign, candidate=candidate,
                                repetition=repetition, snapshot=names[index], **result)
                            rows.append(row)
                            count_steps += result['steps']
                            returns[direction, sign, repetition] = result['score_gain']
                            if best_row is None or (row['highest_stage'], row['mission_completed'], row['score_gain']) > (
                                    best_row['highest_stage'], best_row['mission_completed'], best_row['score_gain']):
                                best_row = row
                    log(dict(event='focus_progress', generation=generation+1,
                             directions_done=direction+1, segments=len(rows), actions=count_steps))
                write_json(target/'segments.json', rows)
                following, stats = ars_update(center, directions, returns, args.step_size)
                np.savez_compressed(target/'update.npz', returns=returns, following=following)
                generation += 1
                training_steps += count_steps
                training_segments += len(rows)
                if best_row['highest_stage'] > 1 or best_row['mission_completed']:
                    set_head(model, heads[best_row['candidate']])
                    discovery = target/'discovery'
                    save_checkpoint(model, discovery, config, generation=generation,
                        training_steps=training_steps, training_games=training_segments,
                        rng=rng, next_seed=80000+generation*len(chosen), candidate=True)
                    snapshot = sources[names.index(best_row['snapshot'])]
                    result, trace = play_segment(env, snapshot, make_population_infer(model,
                        head_array(model)[None]), 0, best_row['seed'], keep_trace=True)
                    for key in ('score_gain', 'stage', 'highest_stage', 'steps', 'mission_completed'):
                        if result[key] != best_row[key]:
                            raise RuntimeError('focused discovery failed neural reexecution')
                    np.savez_compressed(discovery/'training-trace.npz',
                        frames=trace[0], actions=trace[1], rewards=trace[2])
                    write_json(discovery/'training-result.json', best_row)
                    validate(discovery)
                set_head(model, following)
                for key, value in tree_flatten(model.parameters()):
                    if key in immutable:
                        np.testing.assert_array_equal(np.array(value), immutable[key])
                location = preserve()
                log(dict(event='generation', generation=generation,
                         training_steps=training_steps, training_segments=training_segments,
                         best_focused_segment=best_row, **stats))
                if generation % args.eval_every == 0 or stop or generation == args.generations:
                    validate(location)
        finally:
            env.close()
        if sha256(args.output/'ppo-parent-optimizer.npz') != config['parent_optimizer_sha256']:
            raise RuntimeError('preserved PPO optimizer changed')
        log(dict(event='finished', generation=generation,
                 training_steps=training_steps, training_segments=training_segments,
                 stopped=stop, frozen_encoder_and_value_verified=True))


if __name__ == '__main__':
    main()
