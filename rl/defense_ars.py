"""ARS-V1-style search of a learned categorical head over frozen screen features.

Whole real-game score returns only. Adapted from arXiv:1803.07055 section 3;
not its continuous-control linear policy or observation-whitening variant.
No MLX import at module scope: emulator workers do not initialize the GPU.
"""

import json
from pathlib import Path
import time

import numpy as np

from .defense_learning import game_rank, sha256, write_json
from .vector import VectorEnv


ALGORITHM = 'screen-feature-ars-v1'


def head_array(model):
    return np.concatenate([np.array(model.advantage.weight), np.array(model.advantage.bias)[:, None]], axis=1)


def set_head(model, head):
    import mlx.core as mx
    head = np.asarray(head)
    if head.shape != (*model.advantage.weight.shape[:1], model.advantage.weight.shape[1]+1) or not np.isfinite(head).all():
        raise ValueError('invalid action head')
    model.advantage.weight = mx.array(head[:, :-1].astype(np.float32))
    model.advantage.bias = mx.array(head[:, -1].astype(np.float32))
    mx.eval(model.state)


def logits(features, heads):
    """Same per-row arithmetic for population training and frozen replay."""
    import mlx.core as mx
    return mx.sum(features[:, None, :]*heads[:, :, :-1], axis=-1)+heads[:, :, -1]


def acting_policy(model, temperature=1.):
    import mlx.core as mx
    from .evaluate import categorical_policy
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    def infer(obs):
        head = mx.concatenate([model.advantage.weight, model.advantage.bias[:, None]], axis=1)
        heads = mx.broadcast_to(head, (len(obs), *head.shape))
        return logits(model.features(obs), heads)/temperature
    predict = mx.compile(infer, inputs=model.state)
    return categorical_policy(lambda obs: np.array(predict(mx.array(obs))))


def make_population_infer(model, heads):
    import mlx.core as mx
    heads = np.asarray(heads, np.float32)
    if heads.ndim != 3 or heads.shape[1:] != head_array(model).shape or not np.isfinite(heads).all():
        raise ValueError('invalid candidate head bank')
    bank = mx.array(heads)
    predict = mx.compile(lambda obs, indices: logits(model.features(obs), bank[indices]), inputs=model.state)
    return lambda obs, indices: np.array(predict(mx.array(obs), mx.array(indices, mx.int32)))


def ars_update(head, directions, returns, step_size):
    """All directions; normalize by SD of paired mean returns, not ranks."""
    head, directions, returns = (np.asarray(x) for x in (head, directions, returns))
    if (directions.ndim != head.ndim+1 or directions.shape[1:] != head.shape
            or returns.ndim != 3 or returns.shape[:2] != (len(directions), 2)
            or not min(returns.shape) or not np.isfinite(step_size) or step_size <= 0
            or any(not np.isfinite(x).all() for x in (head, directions, returns))):
        raise ValueError('invalid search update')
    means = returns.astype(np.float64).mean(axis=2)
    scale = float(means.std())
    if scale <= 1e-12:
        delta = np.zeros_like(head, np.float64)
    else:
        coefficients = (means[:, 0]-means[:, 1])/scale
        delta = step_size*np.tensordot(coefficients, directions.astype(np.float64), axes=1)/len(directions)
    following = (head.astype(np.float64)+delta).astype(np.float32)
    if not np.isfinite(following).all():
        raise FloatingPointError('nonfinite search weights')
    return following, dict(return_std=scale, update_norm=float(np.linalg.norm(delta)),
        skipped_zero_variance=scale <= 1e-12, mean_candidate_score=float(means.mean()),
        best_candidate_mean=float(means.max()))


def candidate_jobs(directions, repetitions, first_seed):
    if min(directions, repetitions) < 1 or first_seed < 70000:
        raise ValueError('positive population and separate training seeds required')
    # Signs share boot and action-sampling seeds. Directions and generations
    # use fresh seeds; validation seeds 10000..10009 are never optimization data.
    return [dict(candidate=2*d+sign, direction=d, sign=sign, repetition=r,
                 seed=first_seed+d*repetitions+r)
            for d in range(directions) for sign in range(2) for r in range(repetitions)]


def play_population(model, heads, jobs, *, envs=16, tstates=100000, observation_stride=1,
                    log=None, keep_traces=False):
    if not jobs or envs < 1 or any(not 0 <= j['candidate'] < len(heads) for j in jobs):
        raise ValueError('invalid population jobs')
    infer = make_population_infer(model, heads)
    count = min(envs, len(jobs))
    workers = VectorEnv(count, jobs[0]['seed'], game='defense', tstates=tstates,
                        max_steps=0, allow_enter=False, observation_stride=observation_stride)
    results, traces = [None]*len(jobs), [None]*len(jobs)
    rngs = [np.random.default_rng(j['seed']+1_000_000) for j in jobs]
    sums, actions_taken = np.zeros(len(jobs)), np.zeros(len(jobs), np.int64)
    pending = iter(range(count, len(jobs)))
    started, last_log = time.monotonic(), time.monotonic()
    try:
        if log:
            log(dict(event='population_workers', workers=workers.runtime()))
        observations = workers.reset([j['seed'] for j in jobs[:count]])
        active = {i: (i, obs) for i, obs in enumerate(observations)}
        for i, obs in enumerate(observations):
            if keep_traces:
                traces[i] = ([obs[-1].copy()], [], [])
        while active:
            indices = sorted(active)
            batch_jobs = [active[i][0] for i in indices]
            obs = np.stack([active[i][1] for i in indices])
            values = infer(obs, [jobs[j]['candidate'] for j in batch_jobs])
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite candidate logits')
            probabilities = np.exp(values-np.logaddexp.reduce(values, axis=-1, keepdims=True))
            uniforms = np.array([rngs[j].random() for j in batch_jobs])
            actions = (uniforms[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1).clip(0, values.shape[1]-1)
            stepped = workers.step(actions, indices=indices)
            for worker, j, action, row in zip(indices, batch_jobs, actions, stepped, strict=True):
                obs, reward, terminal, truncated, info, _ = row
                sums[j] += reward
                actions_taken[j] += 1
                if keep_traces:
                    traces[j][0].append(obs[-1].copy())
                    traces[j][1].append(int(action))
                    traces[j][2].append(float(reward))
                if terminal or truncated:
                    game = dict(**jobs[j], **info)
                    if game_rank(game) is None or sums[j] != game['score'] or actions_taken[j] != game['steps']:
                        raise ValueError('population fitness requires complete score-aligned real games')
                    results[j] = game
                    if log:
                        log(dict(event='training_game', **game))
                    following = next(pending, None)
                    if following is None:
                        del active[worker]
                    else:
                        obs = workers.reset([jobs[following]['seed']], indices=[worker])[0]
                        active[worker] = following, obs
                        if keep_traces:
                            traces[following] = ([obs[-1].copy()], [], [])
                else:
                    active[worker] = j, obs
            if log and time.monotonic()-last_log >= 10:
                log(dict(event='population_progress', finished=sum(g is not None for g in results),
                         games=len(jobs), actions=int(actions_taken.sum()), seconds=time.monotonic()-started))
                last_log = time.monotonic()
    finally:
        workers.close()
    if keep_traces:
        traces = [tuple(np.asarray(x, dtype=dtype) for x, dtype in zip(t, (np.uint8, np.uint8, np.float32))) for t in traces]
    return results, traces


def save_checkpoint(model, destination, config, *, generation, training_steps, training_games, rng, next_seed, candidate=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(destination/'model.safetensors'))
    write_json(destination/'state.json', dict(config=config, generations=generation,
        steps=config['parent_steps']+training_steps, training_steps=training_steps,
        training_games=training_games, next_training_seed=next_seed, search_rng=rng.bit_generator.state,
        candidate=candidate,
        search_optimizer='ARS V1 normalized finite differences; no momentum state',
        hashes={'model.safetensors': sha256(destination/'model.safetensors')}))


def restore_checkpoint(model, directory, rng):
    directory = Path(directory)
    state = json.loads((directory/'state.json').read_text())
    if (state['config'].get('algorithm') != ALGORITHM or state.get('candidate', False)
            or state['steps'] != state['config']['parent_steps']+state['training_steps']):
        raise ValueError('not a compatible full ARS checkpoint')
    if sha256(directory/'model.safetensors') != state['hashes']['model.safetensors']:
        raise ValueError('ARS checkpoint checksum mismatch')
    model.load_weights(str(directory/'model.safetensors'))
    rng.bit_generator.state = state['search_rng']
    return state
