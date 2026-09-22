"""Frozen learned-world actor calibration with real complete-game evaluation.

This is not yet an online Dreamer training loop. Forecast errors are measured
separately; only original-emulator outcomes may publish a local best replay.
"""

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from .defense import GAME_SHA256, ENVIRONMENT_VERSION, action_names
from .defense_imagination import ALGORITHM, ARCHITECTURE, ImaginationLearner, acting_policy
from .defense_learning import evaluate, publish_best, sha256, write_json
from .defense_world_data import Sequences, disk_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--every', type=int, default=500)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--eval-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=.999)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.output.exists() or min(args.updates, args.every, args.batch, args.games, args.eval_envs) < 1:
        parser.error('positive counts and a new output required')
    world_state = json.loads((args.world/'state.json').read_text())
    for name, digest in world_state['hashes'].items():
        if sha256(args.world/name) != digest:
            parser.error('world checkpoint checksum mismatch')
    metadata = world_state['metadata']
    if (metadata['dataset_sha256'] != sha256(args.data/'manifest.json')
            or metadata['architecture'] != 'small-gaussian-rssm-v1'):
        parser.error('frozen actor calibration requires matching world-model own dataset')
    collection = json.loads((args.data/'manifest.json').read_text())
    train = Sequences(args.data, 'train', 32)  # No held-out files enter this sampler.
    mx.set_cache_limit(128*1024*1024)
    learner = ImaginationLearner(seed=args.seed, horizon=args.horizon, gamma=args.gamma)
    rng = np.random.default_rng(args.seed)
    learner.model.world.load_weights(str(args.world/'world.safetensors'))
    learner.rebind()
    config = dict(game='defense', game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
        algorithm=ALGORITHM, architecture=ARCHITECTURE, world_hidden=128, world_stochastic=32,
        action_names=list(action_names()), allow_enter=False, tstates=collection['tstates'],
        observation_stride=collection['observation_stride'], eval_max_steps=0,
        world_checkpoint=str(args.world.resolve()), world_sha256=sha256(args.world/'world.safetensors'),
        data=str(args.data.resolve()), dataset_sha256=sha256(args.data/'manifest.json'),
        training_method='frozen own learned world; imagined REINFORCE actor and TD-lambda critic',
        actor_step_unit='imagined updates, not emulator actions', own_training_actions=sum(
            row['steps'] for row in collection['episodes'] if row['split']=='train'),
        world_updates=world_state['updates'], new_environment_training_actions=0,
        world_frozen=True, demonstrations=False, imitation_loss=False, reward_scale=.01,
        actor_source_sha256=sha256(Path(__file__).with_name('defense_imagination.py')),
        fit_source_sha256=sha256(Path(__file__)), world_source_sha256=metadata['model_source_sha256'],
        source_paper='https://arxiv.org/abs/2010.02193',
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    if args.resume:
        previous = learner.restore(args.resume, rng)
        for key in ('world_sha256', 'dataset_sha256', 'tstates', 'observation_stride'):
            if previous[key] != config[key]:
                parser.error('incompatible actor continuation: '+key)
    if learner.updates >= args.updates:
        parser.error('target must exceed restored actor updates')
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', config)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(row):
            row = dict(**row, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
        def checkpoint():
            disk_guard(args.output)
            destination = args.output/f'update-{learner.updates:06d}'
            learner.save(destination, config, rng)
            result = evaluate(acting_policy(learner.model), range(10000, 10000+args.games),
                tstates=config['tstates'], max_steps=0, envs=args.eval_envs,
                observation_stride=config['observation_stride'], log=record)
            write_json(destination/'evaluation.json', result)
            record(dict(event='validation', actor_updates=learner.updates,
                        **{k:v for k,v in result.items() if k!='games'}))
            # Actor initialization is only a baseline, never a trained replay.
            if learner.updates:
                publish_best(destination/'model.safetensors', result, args.output/'artifacts', log=record)
        checkpoint()
        while learner.updates < args.updates:
            frames, actions, rewards, continuation = train.sample(args.batch, rng)
            starts, alive = learner.starts(frames, actions, continuation)
            stats = learner.train(starts, alive)
            if learner.updates % 20 == 0:
                disk_guard(args.output)
                record(dict(event='update', actor_updates=learner.updates, **stats))
            if learner.updates % args.every == 0 or learner.updates == args.updates:
                checkpoint()
        # The frozen world must remain parameter-exact, not merely on disk unchanged.
        frozen = args.output/'frozen-world-check.safetensors'
        learner.model.world.save_weights(str(frozen))
        if sha256(frozen) != config['world_sha256']:
            raise RuntimeError('world parameters changed during actor calibration')
        record(dict(event='finished', actor_updates=learner.updates, frozen_world_verified=True))


if __name__ == '__main__':
    main()
