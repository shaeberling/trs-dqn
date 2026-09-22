"""Compare frozen DQN predictions with returns on its own verified replay.

Diagnostic only: no parameter updates, new actions, native-state reads or
training data. A selected trajectory cannot establish a causal learning bug.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_alias_probe import replay_observation
from .defense_learning import DQN_ALGORITHM, sha256
from .defense_loss_probe import loss_windows


def discounted_returns(rewards, boundaries, gamma, scale):
    rewards, boundaries = np.asarray(rewards), np.asarray(boundaries)
    if (rewards.ndim != 1 or not len(rewards) or boundaries.shape != rewards.shape
            or boundaries.dtype != np.bool_ or not boundaries[-1]
            or not np.isfinite(rewards).all() or np.any(rewards < 0)
            or not np.isfinite(gamma) or not 0 < gamma < 1
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('requires complete finite score rewards and terminal boundaries')
    result = np.empty(len(rewards), np.float64)
    following = 0.
    for i in range(len(rewards)-1, -1, -1):
        following = float(rewards[i])*scale + (0. if boundaries[i] else gamma*following)
        result[i] = following
    return result


def comparison(predictions, returns, scale):
    predictions, returns = np.asarray(predictions), np.asarray(returns)
    if (predictions.ndim != 1 or predictions.shape != returns.shape or not len(returns)
            or not np.isfinite(predictions).all() or not np.isfinite(returns).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('requires matching nonempty finite prediction/return vectors')
    # Display in discounted score units, not the optimizer's fixed scaled units.
    predicted, realized = predictions/scale, returns/scale
    error = predicted-realized
    return dict(observations=len(error), mean_predicted_discounted_score=float(predicted.mean()),
                mean_realized_discounted_score=float(realized.mean()),
                mean_prediction_minus_return=float(error.mean()),
                median_prediction_minus_return=float(np.median(error)),
                mean_absolute_error=float(np.abs(error).mean()),
                p90_absolute_error=float(np.quantile(np.abs(error), .9)))


def probe(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle/'manifest.json').read_text())
    hashes = manifest['hashes']
    required = ('model.safetensors', 'state.json', 'trace.npz', 'verification.json')
    for name in required:
        if name not in hashes or sha256(bundle/name) != hashes[name]:
            raise ValueError('replay checksum mismatch: '+name)
    verification = json.loads((bundle/'verification.json').read_text())
    config = json.loads((bundle/'state.json').read_text())['config']
    if (not verification.get('verified') or verification.get('temperature', 1) != 1
            or verification['checkpoint_sha256'] != hashes['model.safetensors']
            or config.get('algorithm') != DQN_ALGORITHM or config.get('allow_enter', False)
            or config.get('action_names') != list(action_names())
            or config.get('life_terminal') is not True):
        raise ValueError('requires verified ordinary 20-action DQN with life terminals')
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        frames, actions, rewards = trace['frames'], trace['actions'], trace['rewards']
        metadata = json.loads(str(trace['metadata']))
    stride = config.get('observation_stride', 1)
    if (metadata.get('checkpoint_sha256') != hashes['model.safetensors']
            or metadata.get('action_names') != list(action_names())
            or metadata.get('temperature', 1) != 1
            or metadata.get('observation_stride', 1) != stride
            or metadata.get('tstates') != config['tstates']
            or verification['verified_actions'] != len(actions)):
        raise ValueError('trace provenance or timing mismatch')
    lives = loss_windows(frames, actions, rewards, metadata)
    boundaries = np.zeros(len(actions), bool)
    for life in lives:
        boundaries[life['visible_loss_frame']-1] = True
    returns = discounted_returns(rewards, boundaries, config['gamma'], config['reward_scale'])

    import mlx.core as mx
    from .model import QNetwork
    mx.set_cache_limit(256*1024*1024)
    model = QNetwork(action_count=20)
    model.load_weights(str(bundle/'model.safetensors'))
    mx.eval(model.state)
    predict = mx.compile(model, inputs=model.state)
    selected_q = np.empty(len(actions), np.float64)
    for i, action in enumerate(actions):
        obs = replay_observation(frames, i, stride)[None]
        values = np.array(predict(mx.array(obs)))[0]
        if not np.isfinite(values).all() or int(values.argmax()) != int(action):
            raise ValueError(f'reconstructed greedy action differs at decision {i}')
        selected_q[i] = float(values[action])

    scale = config['reward_scale']
    reports = []
    for life in lives:
        start, stop = life['previous_visible_loss_frame'], life['visible_loss_frame']
        flash = life['first_major_white_flash_in_last_128_frames']
        marker = stop if flash is None else flash
        first = max(start, marker-64)
        anchors = []
        for distance in (128, 64, 32, 8, 1):
            i = max(start, marker-distance)
            anchors.append(dict(decision=i, requested_actions_before_marker=distance,
                                predicted_discounted_score=float(selected_q[i]/scale),
                                realized_discounted_score=float(returns[i]/scale)))
        reports.append(dict(life=life['life'], life_score=life['score_gained_this_life'],
                            visible_loss_frame=stop, flash_alignment_frame=flash,
                            pre_marker_window=[first, marker],
                            pre_marker=comparison(selected_q[first:marker], returns[first:marker], scale),
                            after_flash=(comparison(selected_q[flash:stop], returns[flash:stop], scale)
                                         if flash is not None else None), anchors=anchors))
    for name in required:
        if sha256(bundle/name) != hashes[name]:
            raise RuntimeError('source changed during diagnostic: '+name)
    return dict(bundle=str(bundle), source_hashes={k: hashes[k] for k in required},
                probe_source_sha256=sha256(Path(__file__)), result=metadata['result'],
                gamma=config['gamma'], reward_scale=scale, life_terminal=True,
                diagnostic_only=True, promotion_eligible=False, parameter_updates=0,
                training_data_written=False, native_reexecution=False,
                original_native_verification=verification,
                replay_actions_reproduced=len(actions),
                whole_replay=comparison(selected_q, returns, scale), lives=reports,
                limitations=[
                    'One selected verified replay, not a fresh evaluation or expected-return estimate.',
                    'Returns follow the frozen greedy policy on this trajectory; they are not optimal counterfactual returns.',
                    'Visible life boundaries match learning terminals but can lag physical collision.',
                    'White flashes align windows; they are not exact collision timestamps.',
                    'Prediction errors can reflect approximation, partial observations or policy/value mismatch; they do not prove a causal bug.',
                    'No policy action, reward, training input or parameter is changed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    result = probe(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(replay_actions_reproduced=result['replay_actions_reproduced'],
                          **result['whole_replay'])), flush=True)


if __name__ == '__main__':
    main()
