"""Read-only stage-one command-alias diagnostic on an own verified PPO replay.

Never changes a policy, action space, reward or training input. Replay frames
are diagnostic inputs only. Reports command-group entropy, not physical-state
entropy or a new evaluation success rate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names, screen_info, validate_observation_stride
from .defense_learning import game_rank, load_policy, sha256


GROUPS = tuple((i,) for i in range(9)) + (tuple(range(9, 18)), (18,), (19,))


def statistics_probability_row(probabilities):
    """Normalize float32 softmax roundoff for reporting, never for sampling."""
    row = np.asarray(probabilities, dtype=np.float64)
    if row.shape != (20,) or not np.isfinite(row).all() or np.any(row < 0) or row.sum() <= 0:
        raise ValueError('invalid categorical probabilities')
    total = float(row.sum())
    return row/total, abs(total-1.)


def probability_summary(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 20 or not len(p) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=1e-6)):
        raise ValueError('requires normalized finite 20-action probability rows')
    p = p/p.sum(axis=1, keepdims=True)
    grouped = np.stack([p[:, group].sum(axis=1) for group in GROUPS], axis=1)
    raw_entropy = -(p*np.log(p.clip(1e-300))).sum(axis=1)
    group_entropy = -(grouped*np.log(grouped.clip(1e-300))).sum(axis=1)
    raw, coarse = float(raw_entropy.mean()), float(group_entropy.mean())
    return dict(observations=len(p), mean_raw_entropy_nats=raw,
                mean_group_entropy_nats=coarse, mean_alias_entropy_nats=max(0., raw-coarse),
                alias_fraction_of_entropy=max(0., raw-coarse)/raw if raw > 0 else 0.,
                mean_movement_probability=float(p[:, 1:9].sum(axis=1).mean()),
                mean_forward_fire_alias_probability=float(p[:, 9:18].sum(axis=1).mean()),
                mean_probabilities=dict(zip(action_names(), p.mean(axis=0).tolist(), strict=True)))


def replay_observation(frames, index, stride):
    """Reconstruct the environment's boot-padded visible-frame stack."""
    validate_observation_stride(stride)
    if (frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (16, 64)
            or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(frames)):
        raise ValueError('invalid replay frames or decision index')
    return frames[[max(0, index-offset*stride) for offset in (3, 2, 1, 0)]]


def probe(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle/'manifest.json').read_text())
    expected = manifest['hashes']
    required = ('model.safetensors', 'state.json', 'verification.json', 'trace.npz')
    if any(name not in expected for name in required):
        raise ValueError('requires a complete verified replay bundle')
    for name in required:
        if sha256(bundle/name) != expected[name]:
            raise ValueError('replay checksum mismatch: '+name)
    verification = json.loads((bundle/'verification.json').read_text())
    if (not verification.get('verified') or verification.get('temperature', 1) != 1
            or verification['checkpoint_sha256'] != expected['model.safetensors']):
        raise ValueError('requires verified original-policy sampling')
    config = json.loads((bundle/'state.json').read_text())['config']
    if (config.get('algorithm', 'ppo') != 'ppo' or config.get('allow_enter', False)
            or config.get('action_names') != list(action_names())):
        raise ValueError('requires the standard 20-action PPO profile')
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        frames, actions = trace['frames'], trace['actions']
        metadata = json.loads(str(trace['metadata']))
    result = metadata['result']
    if (game_rank(result) is None or result['highest_stage'] != 1
            or metadata['checkpoint_sha256'] != expected['model.safetensors']
            or metadata.get('temperature', 1) != 1 or not len(actions)
            or actions.ndim != 1 or len(frames) != len(actions)+1
            or verification['verified_actions'] != len(actions)):
        raise ValueError('requires a complete stage-one-only own-policy trace')
    stride = config.get('observation_stride', 1)
    if (metadata.get('observation_stride', 1) != stride
            or metadata['tstates'] != config['tstates']):
        raise ValueError('trace/checkpoint timing differs')
    if any(screen_info(frame)['stage'] not in (None, 1) for frame in frames):
        raise ValueError('stage-one alias grouping is invalid for later stages')

    import mlx.core as mx
    mx.set_cache_limit(256*1024*1024)
    policy, loaded_config = load_policy(bundle/'model.safetensors')
    if loaded_config != config:
        raise ValueError('configuration changed while loading')
    recurrent = config.get('architecture') is not None
    hidden = np.zeros((1, config['recurrent_hidden']), np.float32) if recurrent else None
    if not recurrent:
        from .model import QNetwork
        model = QNetwork(action_count=20)
        model.load_weights(str(bundle/'model.safetensors'))
        mx.eval(model.state)
        predict = mx.compile(model.policy_value, inputs=model.state)
    rng = np.random.default_rng(result['seed']+1_000_000)
    probabilities, normalization_errors = [], []
    for i, action in enumerate(actions):
        obs = replay_observation(frames, i, stride)[None]
        if recurrent:
            logits, hidden = policy.infer(obs, hidden)  # Preserve memory across visible ship losses.
        else:
            logits = np.array(predict(mx.array(obs))[0])
        probs = np.exp(logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True))
        selected = int((rng.random() > np.cumsum(probs[0])).sum().clip(0, 19))
        if selected != int(action):
            raise ValueError(f'reconstructed policy action differs at decision {i}')
        row, error = statistics_probability_row(probs[0])
        probabilities.append(row)
        normalization_errors.append(error)
    probabilities = np.asarray(probabilities)
    windows = []
    previous = 0
    for event in metadata['events']:
        if not event.get('life_lost'):
            continue
        stop = event['frame']
        if not previous < stop <= len(actions):
            raise ValueError('invalid visible loss-event order')
        start = max(previous, stop-128)
        windows.append(dict(first_decision=start, stop_decision_exclusive=stop,
                            visible_lives_after=event['lives'], **probability_summary(probabilities[start:stop])))
        previous = stop
    for name in required:
        if sha256(bundle/name) != expected[name]:
            raise RuntimeError('source changed during diagnostic: '+name)
    return dict(bundle=str(bundle), checkpoint_sha256=expected['model.safetensors'],
                trace_sha256=expected['trace.npz'], state_sha256=expected['state.json'],
                probe_source_sha256=sha256(Path(__file__)), result=result,
                diagnostic_only=True, promotion_eligible=False, parameter_updates=0,
                training_data_written=False, replay_actions_reproduced=len(actions),
                max_original_float32_probability_sum_error=max(normalization_errors),
                native_reexecution=False, original_native_verification=verification,
                command_groups=[list(group) for group in GROUPS],
                whole_replay=probability_summary(probabilities), pre_visible_loss_windows=windows,
                limitations=['One selected own-policy replay, not a fresh complete-game evaluation.',
                             'Grouping describes stage-one input commands, not unique physical outcomes.',
                             'Full trajectory includes animations; visible loss can lag physical collision.',
                             'Action reproduction uses original float32 probabilities; entropy uses renormalized float64 rows.',
                             'No policy, action-space or training behavior is changed by this diagnostic.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing result')
    result = probe(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(replay_actions_reproduced=result['replay_actions_reproduced'],
                          **result['whole_replay'])), flush=True)


if __name__ == '__main__':
    main()
