"""Compare frozen DQN predictions with returns on its own verified replay.

Diagnostic only: no parameter updates, new actions, native-state reads or
training data. A selected trajectory cannot establish a causal learning bug.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_alias_probe import GROUPS, replay_observation
from .defense_learning import DQN_ALGORITHM, QUANTILE_ALGORITHM, sha256
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


def distribution_summary(quantiles, actions, alternative_actions, weights, scale):
    """Selected-action return spread and counterfactual choice, never confidence."""
    z, actions, other, weights = map(np.asarray, (quantiles, actions, alternative_actions, weights))
    if (z.ndim != 2 or not len(z) or z.shape[1] < 2
            or actions.shape != (len(z),) or other.shape != actions.shape
            or actions.dtype.kind not in 'iu' or other.dtype.kind not in 'iu'
            or np.any(actions < 0) or np.any(actions >= 20) or np.any(other < 0) or np.any(other >= 20)
            or weights.shape != (z.shape[1],) or not np.isfinite(z).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('requires finite quantiles, choices, normalized weights and positive scale')
    low, high = int(.1*z.shape[1]), min(z.shape[1]-1, int(.9*z.shape[1]))
    group = np.empty(20, np.int32)
    for index, members in enumerate(GROUPS):
        group[list(members)] = index
    return dict(observations=len(z), quantiles=z.shape[1],
                low_fraction=(low+.5)/z.shape[1], high_fraction=(high+.5)/z.shape[1],
                mean_high_minus_low_discounted_score=float(np.mean(z[:, high]-z[:, low])/scale),
                mean_distorted_minus_mean_discounted_score=float(np.mean(z@weights-z.mean(axis=1))/scale),
                selected_adjacent_crossing_fraction=float(np.mean(z[:, :-1]>z[:, 1:]+1e-6)),
                counterfactual_different_actions=int(np.sum(actions != other)),
                counterfactual_disagreement_fraction=float(np.mean(actions != other)),
                counterfactual_different_stage_one_commands=int(np.sum(group[actions] != group[other])),
                counterfactual_stage_one_command_disagreement_fraction=float(np.mean(group[actions] != group[other])))


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
            or verification.get('quantile_power', 0) != 0
            or verification['checkpoint_sha256'] != hashes['model.safetensors']
            or config.get('algorithm') not in (DQN_ALGORITHM, QUANTILE_ALGORITHM) or config.get('allow_enter', False)
            or config.get('action_names') != list(action_names())
            or config.get('life_terminal') is not True):
        raise ValueError('requires verified mean-greedy scalar or quantile 20-action DQN with life terminals')
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        frames, actions, rewards = trace['frames'], trace['actions'], trace['rewards']
        metadata = json.loads(str(trace['metadata']))
    stride = config.get('observation_stride', 1)
    if (metadata.get('checkpoint_sha256') != hashes['model.safetensors']
            or metadata.get('action_names') != list(action_names())
            or metadata.get('temperature', 1) != 1
            or metadata.get('quantile_power', 0) != 0
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
    distributional = config['algorithm'] == QUANTILE_ALGORITHM
    if distributional:
        from .defense_quantile import QuantileQ, distortion_weights
        model = QuantileQ(action_count=20, quantiles=config['quantiles'])
        # Standardized read-only counterfactual for both neutral and risk models.
        # It does not replace their actual mean-greedy replay or training policy.
        weights = distortion_weights(config['quantiles'], 1.5)
        mlx_weights = mx.array(weights)
        selected_quantiles = np.empty((len(actions), config['quantiles']), np.float64)
        alternative_actions = np.empty(len(actions), np.int32)
        def infer(obs):
            values = model.quantile_values(obs)
            return mx.mean(values, axis=2), values, mx.argmax(mx.sum(values*mlx_weights, axis=2), axis=1)
    else:
        model = QNetwork(action_count=20)
    model.load_weights(str(bundle/'model.safetensors'))
    mx.eval(model.state)
    predict = mx.compile(infer if distributional else model, inputs=model.state)
    selected_q = np.empty(len(actions), np.float64)
    for i, action in enumerate(actions):
        obs = replay_observation(frames, i, stride)[None]
        if distributional:
            mean, quantiles, alternatives = predict(mx.array(obs))
            values = np.array(mean)[0]
            selected_quantiles[i] = np.array(quantiles)[0, action]
            alternative_actions[i] = int(alternatives[0].item())
        else:
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
        if distributional:
            reports[-1]['pre_marker_distribution'] = distribution_summary(
                selected_quantiles[first:marker], actions[first:marker], alternative_actions[first:marker],
                weights, scale)
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
                **(dict(quantile_distribution=dict(
                    diagnostic_power=1.5, actual_training_power=config['quantile_exploration_power'],
                    counterfactual_only=True,
                    whole_replay=distribution_summary(selected_quantiles, actions, alternative_actions, weights, scale),
                    limitation='Selected mean-greedy replay screens only, not the actual exploratory training states. '
                               'Higher-index quantiles are not sorted; crossing is reported, not repaired. '
                               'Stage-one alias grouping compares commands, not measured physical displacement. '
                               'Return spread is not epistemic uncertainty or a confidence interval.')) if distributional else {}),
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
