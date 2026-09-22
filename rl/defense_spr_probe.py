"""Read-only SPR representation checks on a frozen, verified own replay.

No training, emulator actions, reward changes or evaluation-to-training transfer.
Unweighted diagnostics are not the PER-weighted training objective.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_alias_probe import replay_observation
from .defense_learning import DQN_ALGORITHM, sha256
from .defense_loss_probe import analyze


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('requires at least two finite feature vectors')
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def summarize(targets, predictions, perturbed, persistence):
    target, prediction, alternative = map(unit, (targets, predictions, perturbed))
    unchanged = unit(persistence)
    if any(target.shape != x.shape for x in (prediction, alternative, unchanged)):
        raise ValueError('feature shapes differ')
    distance = lambda a, b: 1-np.clip(np.sum(a*b, axis=1), -1, 1)
    # Fit a constant to the first half, report all predictors on the other half.
    # Temporal halves of one selected game are NOT independent fresh samples.
    split = len(target)//2
    mean = target[:split].mean(axis=0)
    constant = mean/max(np.linalg.norm(mean), 1e-8)
    held = target[split:]
    return dict(vectors=len(target), comparison_vectors=len(held),
                mean_raw_feature_std=float(np.asarray(targets).std(axis=0).mean()),
                mean_unit_feature_std=float(target.std(axis=0).mean()),
                unit_variance_sum=float(target.var(axis=0).sum()),
                zero_target_vectors=int(np.sum(np.linalg.norm(targets, axis=1) < 1e-8)),
                recorded_action_distance=float(distance(held, prediction[split:]).mean()),
                rotated_action_distance=float(distance(held, alternative[split:]).mean()),
                constant_target_distance=float(distance(held, constant).mean()),
                unchanged_current_target_distance=float(distance(held, unchanged[split:]).mean()),
                prediction_action_sensitivity=float(distance(prediction, alternative).mean()))


def path_starts(length, loss_frames, horizon, spacing=8):
    if (not 1 <= horizon <= 5 or spacing < 1 or not loss_frames
            or loss_frames[-1] != length or loss_frames != sorted(set(loss_frames))
            or loss_frames[0] <= 0):
        raise ValueError('requires ordered complete life boundaries and a valid horizon')
    # Include the final observed terminal screen but never cross into a new life.
    result, start = [], 0
    for stop in loss_frames:
        result.extend(range(start, stop-horizon+1, spacing))
        start = stop
    return result


def trajectory_predictions(online, ema, auxiliary, observations, actions, following, key, dropout):
    """Match auxiliary_loss's root/target dropout keys and batch layout exactly."""
    import mlx.core as mx
    import mlx.nn as nn
    from .defense_spr import spatial_features
    batch, horizon = actions.shape
    root_key, target_key = mx.random.split(key)
    latent = spatial_features(online, observations, root_key, dropout)
    encoded = spatial_features(ema, following.reshape(batch*horizon, *observations.shape[1:]),
                               target_key, dropout)
    targets = nn.relu(ema.hidden(encoded.reshape(batch*horizon, -1))).reshape(batch, horizon, -1)
    # Independent current-screen noise: persistence must not reuse target masks.
    current_key = mx.random.split(key, 3)[2]
    current = spatial_features(ema, observations, current_key, dropout)
    current = nn.relu(ema.hidden(current.reshape(batch, -1)))
    other = latent
    predictions, alternatives = [], []
    for offset in range(horizon):
        recorded = actions[:, offset]
        latent = auxiliary.transition(latent, recorded)
        other = auxiliary.transition(other, (recorded+1) % 20)
        predictions.append(auxiliary.predictor(nn.relu(online.hidden(latent.reshape(batch, -1)))))
        alternatives.append(auxiliary.predictor(nn.relu(online.hidden(other.reshape(batch, -1)))))
    return (targets, mx.stack(predictions, axis=1), mx.stack(alternatives, axis=1),
            mx.broadcast_to(current[:, None], targets.shape))


def probe(checkpoint, bundle, training_dropout=False, seed=0):
    from .defense_spr import AUX_FILES, SprAuxiliary
    from .model import QNetwork
    import mlx.core as mx

    checkpoint = Path(checkpoint).resolve(strict=True)
    report, frames, actions = analyze(bundle)
    names = ('state.json', 'model.safetensors', *AUX_FILES)
    before = {name: sha256(checkpoint/name) for name in names}
    state = json.loads((checkpoint/'state.json').read_text())
    config = state['config']
    dropout = config['spr_dropout'] if training_dropout else 0.
    verification = report['original_native_verification']
    if (config.get('algorithm') != DQN_ALGORITHM or config.get('spr_weight', 0) <= 0
            or config.get('spr_architecture') != 'spatial-own-future-v1'
            or config.get('life_terminal') is not True or config.get('allow_enter', False)
            or verification.get('temperature', 1) != 1
            or verification.get('quantile_power', 0) != 0
            or before['model.safetensors'] != report['source_hashes']['model.safetensors']
            or before['state.json'] != report['source_hashes']['state.json']
            or state.get('spr_auxiliary_hashes') != {n: before[n] for n in AUX_FILES}):
        raise ValueError('requires matching full SPR checkpoint and verified ordinary replay')
    mx.set_cache_limit(128*1024*1024)
    online, ema, auxiliary = QNetwork(20), QNetwork(20), SprAuxiliary(20)
    for model, name in ((online, 'model.safetensors'), (ema, AUX_FILES[1]), (auxiliary, AUX_FILES[0])):
        model.load_weights(str(checkpoint/name))
    mx.eval(online.state, ema.state, auxiliary.state)
    horizon = config['n_step']
    starts = path_starts(len(actions), [l['visible_loss_frame'] for l in report['lives']], horizon)
    stride = config.get('observation_stride', 1)
    for first in range(0, len(actions), 32):
        observations = mx.array(np.stack([replay_observation(frames, i, stride)
            for i in range(first, min(first+32, len(actions)))]))
        values = np.array(online(observations))
        if (not np.isfinite(values).all()
                or not np.array_equal(values.argmax(axis=1), actions[first:first+32])):
            raise ValueError('recorded greedy actions do not match frozen checkpoint')
    targets, predictions, alternatives, persistence = [], [], [], []
    key = mx.random.key(seed)
    for first in range(0, len(starts), 16):
        indices = starts[first:first+16]
        observation = mx.array(np.stack([replay_observation(frames, i, stride) for i in indices]))
        recorded = mx.array(np.stack([actions[i:i+horizon] for i in indices]).astype(np.int32))
        following = mx.array(np.stack([[replay_observation(frames, i+j+1, stride)
                                        for j in range(horizon)] for i in indices]))
        batch_key, key = mx.random.split(key)
        outputs = trajectory_predictions(online, ema, auxiliary, observation, recorded, following,
                                         batch_key, dropout)
        for destination, value in zip((targets, predictions, alternatives, persistence), outputs):
            # Preserve the original clean probe's horizon-major within-batch order.
            destination.append(np.array(value).transpose(1, 0, 2).reshape(-1, 256))
    statistics = summarize(*(np.concatenate(x) for x in (targets, predictions, alternatives, persistence)))
    if before != {name: sha256(checkpoint/name) for name in names}:
        raise RuntimeError('checkpoint changed during diagnostic')
    # Recheck trace provenance after inference as well.
    final, _, _ = analyze(bundle)
    if final['source_hashes'] != report['source_hashes']:
        raise RuntimeError('replay changed during diagnostic')
    return dict(checkpoint=str(checkpoint), bundle=report['bundle'], source_hashes=before,
                trace_hashes=report['source_hashes'], probe_sha256=sha256(Path(__file__)),
                diagnostic_only=True, parameter_updates=0, training_data_written=False,
                native_reexecution=False, promotion_eligible=False, dropout=dropout, diagnostic_seed=seed,
                replay_actions_reproduced=len(actions),
                paths=len(starts), horizon=horizon, result=report['result'], statistics=statistics,
                limitations=[
                    'Selected verified replay only, not the exploratory training distribution.',
                    'Unweighted diagnostic, not the logged PER-weighted objective.',
                    'Fixed action-label rotation is a sensitivity probe, not simulated counterfactual evidence.',
                    'Constant baseline uses first-half targets; all distances compare on the second half.',
                    'Persistence baseline repeats the current EMA projection without any learned transition.',
                    'With dropout, current and future encodings use independent noise; distances include this noise.',
                    'Small variance or weak action sensitivity suggests a concern, not a causal proof.',
                    'Noncollapsed features or lower distance do not prove useful control or mission progress.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-dropout', action='store_true', help='use saved auxiliary dropout, never acting dropout')
    parser.add_argument('--seed', type=int, default=0, help='explicit diagnostic-only dropout key')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    result = probe(args.checkpoint, args.bundle, args.training_dropout, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['statistics'], indent=2))


if __name__ == '__main__':
    main()
