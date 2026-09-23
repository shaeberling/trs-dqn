"""Training-only random command exploration from own verified pre-loss states.

The saved native bytes are opaque reset machinery, never policy input. Random
command holds are an exploration intervention, not a learned-policy replay or
hand-picked route; no weights are updated or promoted by this probe.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


HOLDS = (8, 16, 32, 64)
BRANCHES = (0, 32, 64, 96)


def branches_for_lookback(lookback):
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 128:
        raise ValueError('requires an own-life lookback of at least 128 decisions')
    return tuple((lookback*i)//4 for i in range(4))


def command_ids(mode):
    if mode == 'all':
        return tuple(range(20))
    if mode == 'effective-stage-one':
        # Static key semantics: Space+arrows do not translate in stage one.
        # All eight directions, NOOP and fire retain equal sampling weight.
        return tuple(range(10))
    raise ValueError('unknown unbiased command sampler')


def draw_plan(rng, mode, lookback=128):
    commands = command_ids(mode)
    branch = int(rng.choice(branches_for_lookback(lookback)))
    segments = [(int(rng.choice(commands)), int(rng.choice(HOLDS)))
                for _ in range(2)]
    return branch, segments


def play_trial(env, saved, recorded_actions, recorded_rewards, recorded_screens,
               policy, seed, branch, segments, *, max_actions, keep_trace=False):
    from .defense_snapshot import restore

    if (not 0 <= branch < len(recorded_actions)
            or max_actions <= len(recorded_actions)
            or len(segments) != 2
            or any(not 0 <= action < 20 or hold not in HOLDS
                   for action, hold in segments)):
        raise ValueError('invalid own-state exploration plan')
    obs = restore(env, saved)
    policy.reset_seed(seed+1_000_000)
    start_stage, start_score = saved.stage, saved.score
    actions, rewards, frames = [], [], [obs[-1].copy()]

    def step(action):
        nonlocal obs
        obs, reward, terminal, truncated, info = env.step(action)
        if keep_trace:
            actions.append(action)
            rewards.append(reward)
            frames.append(obs[-1].copy())
        ended = bool(info['life_lost'] or info['stage'] != start_stage
                     or info['mission_completed'] or terminal or truncated)
        return reward, ended, info

    total_reward, steps = 0., 0
    for index in range(branch):
        reward, ended, info = step(int(recorded_actions[index]))
        if reward != recorded_rewards[index] or not np.array_equal(obs[-1], recorded_screens[index]):
            raise RuntimeError('own recorded prefix failed exact native replay')
        steps += 1
        total_reward += reward
        if ended:
            raise RuntimeError('own prefix ended before the saved visible loss')
    for action, hold in segments:
        for _ in range(hold):
            reward, ended, info = step(action)
            steps += 1
            total_reward += reward
            if ended:
                break
        if ended:
            break
    while not ended and steps < max_actions:
        action = int(policy(obs[None])[0])
        reward, ended, info = step(action)
        steps += 1
        total_reward += reward
    if not ended:
        result = dict(completed_boundary=False, steps=steps,
                      score_gain=None, highest_stage=env.highest_stage)
    else:
        if info['score']-start_score != total_reward:
            raise RuntimeError('exploration reward is not displayed score gain')
        result = dict(completed_boundary=True, steps=steps,
                      score_gain=info['score']-start_score,
                      final_score=info['score'], stage=info['stage'],
                      highest_stage=info['highest_stage'],
                      life_lost=bool(info['life_lost']),
                      mission_completed=bool(info['mission_completed']),
                      terminated=bool(info['terminated']), truncated=bool(info['truncated']))
    trace = (np.asarray(frames, np.uint8), np.asarray(actions, np.uint8),
             np.asarray(rewards, np.float32)) if keep_trace else None
    return result, trace


def main():
    import mlx.core as mx
    from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256
    from .defense_ars import acting_policy, restore_checkpoint
    from .defense_ars_focus import harvest, load_source
    from .defense_learning import sha256, write_json
    from .defense_world_data import disk_guard
    from .model import QNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harvest-games', type=int, default=12)
    parser.add_argument('--harvest-seed', type=int, required=True)
    parser.add_argument('--lookback', type=int, default=128)
    parser.add_argument('--attempts', type=int, default=512)
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--command-mode', choices=('all', 'effective-stage-one'), default='all')
    parser.add_argument('--max-actions', type=int, default=2000)
    args = parser.parse_args()
    if (args.output.exists() or args.harvest_seed < 70000
            or min(args.harvest_games, args.attempts, args.max_actions) < 1
            or args.seed < 0 or args.lookback < 128 or args.max_actions <= args.lookback):
        parser.error('fresh output and positive training-only settings required')
    mx.set_cache_limit(128*1024*1024)
    model = QNetwork(action_count=20)
    state = restore_checkpoint(model, args.checkpoint, np.random.default_rng(0))
    config = state['config']
    if (config['game_sha256'] != GAME_SHA256
            or config['environment_version'] != ENVIRONMENT_VERSION
            or config['eval_max_steps'] != 0 or config['allow_enter']
            or config['tstates'] != 100_000):
        raise ValueError('requires ordinary complete-game own-policy timing')
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output/'source.py')
    rng = np.random.default_rng(args.seed)
    policy = acting_policy(model)
    checkpoint_hash = sha256(args.checkpoint/'model.safetensors')
    settings = dict(checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_model_sha256=checkpoint_hash,
                    checkpoint_state_sha256=sha256(args.checkpoint/'state.json'),
                    source_sha256=sha256(Path(__file__)),
                    harvest_source_sha256=sha256(Path(__file__).with_name('defense_ars_focus.py')),
                    game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                    command_ids=list(command_ids(args.command_mode)),
                    branches=list(branches_for_lookback(args.lookback)), holds=list(HOLDS),
                    args={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
                    training_only=True, model_updates=0, native_snapshot_policy_input=False,
                    exploration_not_learned_replay=True)
    write_json(args.output/'config.json', settings)
    with (args.output/'metrics.jsonl').open('x') as stream:
        def log(row):
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            if row['event'] != 'trial' or row.get('highest_stage', 1) > 1:
                print(json.dumps(row), flush=True)

        archive = args.output/'own-loss-states'
        names, source_games, harvest_steps = harvest(model, archive,
            count=args.harvest_games, first_seed=args.harvest_seed,
            lookback=args.lookback, tstates=config['tstates'],
            observation_stride=config['observation_stride'], log=log)
        selected, prior = [], {}
        for name in names:
            item = load_source(archive/name)
            saved, _, _, _, metadata = item
            source = metadata['source']
            life_score = source['visible_loss_score']-prior.get(source['seed'], 0)
            prior[source['seed']] = source['visible_loss_score']
            if (life_score >= 2400 and saved.stage == 1
                    and source['snapshot_action']+args.lookback == source['visible_loss_frame']):
                selected.append((name, item))
        if not selected:
            raise RuntimeError('no own high-score stage-one failure windows')
        log(dict(event='selected_sources', count=len(selected),
                 total_own_states=len(names), harvest_games=len(source_games),
                 harvest_actions=harvest_steps,
                 selection='own complete-game displayed life score >= 2400; stage one'))
        env = DefenseEnv(tstates=config['tstates'], max_steps=0,
                         observation_stride=config['observation_stride'])
        results, discoveries = [], []
        try:
            for attempt in range(args.attempts):
                index = int(rng.integers(len(selected)))
                name, (saved, actions, rewards, screens, _) = selected[index]
                branch, segments = draw_plan(rng, args.command_mode, args.lookback)
                trial_seed = args.seed*1_000_000+attempt
                result, _ = play_trial(env, saved, actions, rewards, screens,
                    policy, trial_seed, branch, segments, max_actions=args.max_actions)
                row = dict(event='trial', attempt=attempt, source=name,
                           branch=branch, segments=segments, seed=trial_seed, **result)
                results.append(row)
                log(row)
                if result['completed_boundary'] and result['highest_stage'] > 1:
                    verified, trace = play_trial(env, saved, actions, rewards, screens,
                        policy, trial_seed, branch, segments,
                        max_actions=args.max_actions, keep_trace=True)
                    if verified != result:
                        raise RuntimeError('training-only stage discovery failed reexecution')
                    target = args.output/f'discovery-{attempt:06d}'
                    target.mkdir(exist_ok=False)
                    np.savez_compressed(target/'trace.npz',
                        frames=trace[0], actions=trace[1], rewards=trace[2])
                    write_json(target/'result.json', row)
                    discoveries.append(str(target))
                if (attempt+1) % 32 == 0:
                    log(dict(event='progress', attempts=attempt+1,
                             complete=sum(r['completed_boundary'] for r in results),
                             stage_two=sum(r['highest_stage'] > 1 for r in results),
                             best_score_gain=max((r['score_gain'] for r in results
                                                  if r['score_gain'] is not None), default=None)))
                disk_guard(args.output)
        finally:
            env.close()
        write_json(args.output/'report.json', dict(
            config=settings, selected_sources=len(selected),
            harvest_games=len(source_games), harvest_actions=harvest_steps,
            attempts=len(results), complete_boundaries=sum(r['completed_boundary'] for r in results),
            stage_two_segments=sum(r['highest_stage'] > 1 for r in results),
            mission_segments=sum(bool(r.get('mission_completed')) for r in results),
            best_score_gain=max((r['score_gain'] for r in results
                                 if r['score_gain'] is not None), default=None),
            discoveries=discoveries, rng_state=rng.bit_generator.state,
            model_updates=0, published_replay=False,
            limitations=['Training-only opaque-state resets, never complete games from boot.',
                         'Random command holds are interventions, not neural policy actions.',
                         'No stage result from this probe is a learned-policy success.']))
        log(dict(event='finished', attempts=len(results),
                 stage_two_segments=sum(r['highest_stage'] > 1 for r in results),
                 discoveries=len(discoveries)))


if __name__ == '__main__':
    main()
