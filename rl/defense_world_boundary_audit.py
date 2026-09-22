"""Read-only distinction between recognizing and forecasting visible life loss.

Whole held-out own games only. Observed-posterior predictions see the arrival
screen and are explicitly NOT forecasts. One-step and longer priors never do.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .defense_learning import sha256, write_json
from .defense_world_data import Sequences
from .defense_world_model import WorldModel
from .defense_world_review import boundary_windows


def boundary_predictions(model, batch, context=8, lead=16, draws=16):
    if draws < 1 or not 0 < context < context+lead <= batch[1].shape[1]:
        raise ValueError('invalid context, lead or draw count')
    frames, actions, _, continuation = (mx.array(v) for v in batch)
    arrival = context+lead
    truth = np.asarray(continuation[:, arrival-1])
    predictions = {}
    for mode, sample, count in [('mean_latent', False, 1), ('sampled_latent', True, draws)]:
        values = {name: [] for name in ('observed_arrival', 'prior_1', 'prior_4', 'prior_8', 'prior_16')}
        for draw in range(count):
            keys = mx.random.split(mx.random.key(51000+draw), 20)
            states, _ = model.observe(frames[:, :arrival+1], actions[:, :arrival], keys[0], sample=sample)
            observed = tuple(s[:, arrival] for s in states)
            values['observed_arrival'].append(np.array(mx.sigmoid(model.continue_logit(model.features(observed))[:, 0])))
            for horizon in (1, 4, 8, 16):
                if horizon > lead:
                    continue
                start = arrival-horizon
                # This state was computed causally: no subsequent observations
                # affect its posterior, even though recognition above uses them.
                state = tuple(s[:, start] for s in states)
                for t in range(start, arrival):
                    state, _ = model.step(state, actions[:, t], keys[t-start+1], sample=sample)
                values[f'prior_{horizon}'].append(np.array(mx.sigmoid(model.continue_logit(model.features(state))[:, 0])))
        summary = {}
        for name, rows in values.items():
            if not rows:
                continue
            probabilities = np.stack(rows)
            average = probabilities.mean(axis=0)
            summary[name] = dict(continuation_probability_mean=float(average.mean()),
                continuation_probabilities=average.tolist(), brier_of_mean=float(np.mean((average-truth)**2)),
                mean_draw_brier=float(np.mean((probabilities-truth[None])**2)))
        predictions[mode] = summary
    return dict(windows=len(frames), context=context, lead=lead, sampled_draws=draws,
        true_continuation=truth.tolist(), predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 1:
        parser.error('new output and positive draw count required')
    state = json.loads((args.checkpoint/'state.json').read_text())
    if (state['metadata']['dataset_sha256'] != sha256(args.data/'manifest.json')
            or state['metadata']['length'] != 32 or state['metadata']['burn'] != 8):
        parser.error('requires matching 32-action/8-context dataset')
    for name, digest in state['hashes'].items():
        if sha256(args.checkpoint/name) != digest:
            raise ValueError('checkpoint checksum mismatch')
    mx.set_cache_limit(128*1024*1024)
    model = WorldModel()
    model.load_weights(str(args.checkpoint/'world.safetensors'))
    held = Sequences(args.data, 'heldout', 32)
    batch, origins = boundary_windows(held)
    report = boundary_predictions(model, batch, draws=args.draws)
    report.update(origins=origins, checkpoint_sha256=sha256(args.checkpoint/'world.safetensors'),
        dataset_sha256=sha256(args.data/'manifest.json'), source_sha256=sha256(Path(__file__)),
        training_updates=0, promotion_eligible=False,
        limitations=['Observed arrival is recognition, not anticipation or a policy input.',
            'Boundary-conditioned diagnostic, not population calibration or collision ground truth.',
            'Sampled-latent means and mean-latent predictions need not agree.',
            'No diagnostic trajectories are added to training.'])
    write_json(args.output, report)
    print(json.dumps({mode: {name: row['brier_of_mean'] for name, row in rows.items()}
                      for mode, rows in report['predictions'].items()}), flush=True)


if __name__ == '__main__':
    main()
