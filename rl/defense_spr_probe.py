"""Read-only SPR representation checks on a frozen, verified own replay.

No training, emulator actions, reward changes or evaluation-to-training transfer.
Clean (dropout-free) diagnostics are not the PER-weighted training objective.
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


def probe(checkpoint, bundle):
    from .defense_spr import AUX_FILES, SprAuxiliary, spatial_features
    from .model import QNetwork
    import mlx.core as mx
    import mlx.nn as nn

    checkpoint = Path(checkpoint).resolve(strict=True)
    report, frames, actions = analyze(bundle)
    names = ('state.json', 'model.safetensors', *AUX_FILES)
    before = {name: sha256(checkpoint/name) for name in names}
    state = json.loads((checkpoint/'state.json').read_text())
    config = state['config']
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
    for first in range(0, len(starts), 16):
        indices = starts[first:first+16]
        observation = mx.array(np.stack([replay_observation(frames, i, stride) for i in indices]))
        latent = spatial_features(online, observation, mx.random.key(0), 0.)
        current = spatial_features(ema, observation, mx.random.key(0), 0.)
        current = np.array(nn.relu(ema.hidden(current.reshape(len(indices), -1))))
        other = latent
        for offset in range(horizon):
            recorded = mx.array(actions[np.array(indices)+offset].astype(np.int32))
            latent = auxiliary.transition(latent, recorded)
            # A fixed label rotation, not a game-valid counterfactual trajectory.
            other = auxiliary.transition(other, (recorded+1) % 20)
            future = mx.array(np.stack([replay_observation(frames, i+offset+1, stride) for i in indices]))
            encoded = spatial_features(ema, future, mx.random.key(0), 0.)
            targets.append(np.array(nn.relu(ema.hidden(encoded.reshape(len(indices), -1)))))
            persistence.append(current)
            predictions.append(np.array(auxiliary.predictor(nn.relu(online.hidden(latent.reshape(len(indices), -1))))))
            alternatives.append(np.array(auxiliary.predictor(nn.relu(online.hidden(other.reshape(len(indices), -1))))))
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
                native_reexecution=False, promotion_eligible=False, dropout=0.,
                replay_actions_reproduced=len(actions),
                paths=len(starts), horizon=horizon, result=report['result'], statistics=statistics,
                limitations=[
                    'Selected verified replay only, not the exploratory training distribution.',
                    'Dropout-free unweighted diagnostic, not the logged PER-weighted objective.',
                    'Fixed action-label rotation is a sensitivity probe, not simulated counterfactual evidence.',
                    'Constant baseline uses first-half targets; all distances compare on the second half.',
                    'Persistence baseline repeats the current EMA projection without any learned transition.',
                    'Small variance or weak action sensitivity suggests a concern, not a causal proof.',
                    'Noncollapsed features or lower distance do not prove useful control or mission progress.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    result = probe(args.checkpoint, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['statistics'], indent=2))


if __name__ == '__main__':
    main()
