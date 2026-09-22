"""Frozen-world life-boundary readouts: recognition versus one-step prediction.

Diagnostic only. Heads learn from the existing own-training split; complete
held-out games are never sampled for updates. No acting policy is changed.
"""

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from .defense_learning import sha256, write_json
from .defense_world_data import Sequences, disk_guard
from .defense_world_model import WorldModel


def windows(count, length=32, burn=8):
    """Cover every action after initial burn once, including episode tails."""
    if count < length or not 0 <= burn < length:
        raise ValueError('invalid episode length or burn')
    starts = list(range(0, count-length+1, length-burn))
    if starts[-1] != count-length:
        starts.append(count-length)
    covered = burn
    for start in starts:
        first = max(burn, covered-start)
        yield start, first
        covered = start+length
    assert covered == count


def features(model, frames, actions, burn=8):
    """Observed arrival includes its screen; prior_1 never sees that screen."""
    states, _ = model.observe(frames, actions, mx.random.key(0), sample=False)
    observed = model.features(states)[:, burn+1:]
    previous = tuple(s[:, burn:-1].reshape(-1, s.shape[-1]) for s in states)
    prior, _ = model.step(previous, actions[:, burn:].reshape(-1), mx.random.key(0), sample=False)
    prior = model.features(prior).reshape(observed.shape)
    return observed, prior


def extract(model, dataset, batch=16):
    specs = [(episode, start, first) for episode, row in enumerate(dataset.episodes)
             for start, first in windows(len(row[1]), dataset.length)]
    rows = {k: [] for k in ('observed', 'prior_1', 'continuation', 'episode', 'action')}
    for offset in range(0, len(specs), batch):
        selected = specs[offset:offset+batch]
        f = np.stack([dataset.episodes[e][0][s:s+33] for e, s, _ in selected])
        a = np.stack([dataset.episodes[e][1][s:s+32] for e, s, _ in selected])
        values = [np.array(v) for v in features(model, mx.array(f), mx.array(a))]
        for i, (episode, start, first) in enumerate(selected):
            for mode, value in zip(('observed', 'prior_1'), values, strict=True):
                rows[mode].append(value[i, first-8:])
            rows['continuation'].append(dataset.episodes[episode][3][start+first:start+32])
            rows['episode'].append(np.full(32-first, episode, np.int32))
            rows['action'].append(np.arange(start+first, start+32, dtype=np.int32))
    result = {k: np.concatenate(v) for k, v in rows.items()}
    assert len(result['continuation']) == sum(len(row[1])-8 for row in dataset.episodes)
    for episode, row in enumerate(dataset.episodes):
        selected = result['episode'] == episode
        np.testing.assert_array_equal(result['action'][selected], np.arange(8, len(row[1])))
        np.testing.assert_array_equal(result['continuation'][selected], row[3][8:])
    return result


def metrics(logits, continuation):
    logits, continuation = np.asarray(logits, np.float64), np.asarray(continuation, np.float64)
    if logits.shape != continuation.shape or not np.isfinite(logits).all():
        raise ValueError('invalid predictions')
    truth = 1-continuation
    probability = np.exp(-np.logaddexp(0, logits))  # probability of loss, not continuation
    positives = truth.astype(bool)
    count = int(positives.sum())
    order = np.argsort(-probability, kind='stable')
    ranked, ranked_truth = probability[order], truth[order]
    # Group ties before calculating precision/recall; a constant predictor's
    # average precision must equal prevalence, not depend on episode order.
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked)-1]
    found = np.cumsum(ranked_truth)[ends]
    average_precision = float(np.sum(np.diff(np.r_[0., found])*found/(ends+1))/count) if count else None
    predicted = probability >= .5
    return dict(rows=len(truth), losses=count, prevalence=float(truth.mean()),
        brier=float(np.mean((probability-truth)**2)),
        always_survive_brier=float(truth.mean()),
        log_loss=float(np.mean(np.logaddexp(0, logits)-continuation*logits)),
        loss_brier=float(np.mean((probability[positives]-1)**2)) if count else None,
        mean_loss_probability_at_loss=float(probability[positives].mean()) if count else None,
        mean_loss_probability_elsewhere=float(probability[~positives].mean()) if (~positives).any() else None,
        loss_average_precision=average_precision,
        threshold_half=dict(true_positive=int((predicted & positives).sum()),
            false_positive=int((predicted & ~positives).sum()), false_negative=int((~predicted & positives).sum())))


def sample(labels, rng, batch, stratified):
    """Stratification with importance weights keeps ordinary empirical BCE."""
    if batch < 2 or batch % 2 or not np.isin(labels, [0., 1.]).all():
        raise ValueError('even positive batch and binary labels required')
    if not stratified:
        return rng.integers(len(labels), size=batch), np.ones(batch, np.float32)
    loss, safe = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if not len(loss) or not len(safe):
        raise ValueError('stratification requires both classes')
    indices = np.r_[rng.choice(loss, batch//2), rng.choice(safe, batch//2)]
    weights = np.r_[np.full(batch//2, 2*len(loss)/len(labels)),
                    np.full(batch//2, 2*len(safe)/len(labels))].astype(np.float32)
    return indices, weights


class Readout:
    def __init__(self, parent):
        self.head = nn.Sequential(nn.Linear(160, 128), nn.ELU(), nn.Linear(128, 1))
        self.head.update(parent.parameters())
        self.optimizer = optim.Adam(learning_rate=6e-4, eps=1e-5)
        self.optimizer.init(self.head.trainable_parameters())
        self.state = [self.head.state, self.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def _update(self, x, y, weights):
        def objective(head):
            values = nn.losses.binary_cross_entropy(head(x)[:, 0], y, with_logits=True, reduction='none')
            return mx.mean(values*weights)
        loss, grads = nn.value_and_grad(self.head, objective)(self.head)
        grads, norm = optim.clip_grad_norm(grads, max_norm=100.)
        self.optimizer.update(self.head, grads)
        return loss, norm

    def train(self, x, y, weights):
        result = self.update(mx.array(x), mx.array(y), mx.array(weights))
        mx.eval(result, self.state)
        values = [float(v) for v in result]
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite diagnostic update')
        return dict(loss=values[0], gradient_norm=values[1])


def evaluate(head, x, y):
    logits = np.concatenate([np.array(head(mx.array(x[start:start+2048])))[:, 0]
                             for start in range(0, len(x), 2048)])
    return metrics(logits, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--every', type=int, default=500)
    parser.add_argument('--batch', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() or min(args.updates, args.every, args.batch) < 1 or args.batch % 2:
        parser.error('new output, positive counts and even batch required')
    state = json.loads((args.world/'state.json').read_text())
    if (state['metadata']['dataset_sha256'] != sha256(args.data/'manifest.json')
            or state['metadata']['length'] != 32 or state['burn'] != 8):
        parser.error('requires matching 32-action/8-burn world and own dataset')
    for name, digest in state['hashes'].items():
        if sha256(args.world/name) != digest:
            raise ValueError('world checksum mismatch')
    mx.set_cache_limit(128*1024*1024)
    mx.random.seed(0)
    model = WorldModel()
    model.load_weights(str(args.world/'world.safetensors'))
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(world=str(args.world.resolve()), world_sha256=sha256(args.world/'world.safetensors'),
        dataset_sha256=sha256(args.data/'manifest.json'), source_sha256=sha256(Path(__file__)),
        model_source_sha256=sha256(Path(__file__).with_name('defense_world_model.py')),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        diagnostic_only=True, promotion_eligible=False, policy_training=False,
        feature_filter='mean latent; 32-action windows with 8-action burn; unique post-burn transitions',
        readout='parent continuation head weights, separate fresh Adam 6e-4 eps1e-5',
        stratification='half each class, weights 2*class_fraction; unbiased empirical BCE, no changed target')
    write_json(args.output/'config.json', config)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(row):
            row = dict(row, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
        cached, origins = {}, {}
        for split in ('train', 'heldout'):
            dataset = Sequences(args.data, split, 32)
            cached[split] = extract(model, dataset)
            origins[split] = dataset.files
            rows = cached[split]
            record(dict(event='features', split=split, rows=len(rows['continuation']),
                losses=int((rows['continuation'] == 0).sum()), games=len(dataset.files)))
            disk_guard(args.output)
            # Recomputable feature arrays stay local; hashes tie all diagnostics
            # to the exact arrays. Compact origins retain unique action indices.
            np.savez_compressed(args.output/f'{split}-origins.npz',
                **{k: rows[k] for k in ('continuation', 'episode', 'action')})
        if set(origins['train']) & set(origins['heldout']):
            raise ValueError('train/held-out overlap')
        import hashlib
        write_json(args.output/'features.json', dict(episode_files=origins,
            array_sha256={split: {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in rows.items()}
                          for split, rows in cached.items()}))
        for mode in ('observed', 'prior_1'):
            for stratified in (False, True):
                name = mode+('-stratified' if stratified else '-uniform')
                learner, rng = Readout(model.continue_logit), np.random.default_rng(0)
                for update in range(args.updates+1):
                    if update:
                        rows = cached['train']
                        selected, weights = sample(rows['continuation'], rng, args.batch, stratified)
                        stats = learner.train(rows[mode][selected], rows['continuation'][selected], weights)
                        if update % 100 == 0:
                            record(dict(event='update', arm=name, updates=update, **stats))
                    if update % args.every == 0 or update == args.updates:
                        disk_guard(args.output)
                        dest = args.output/name/f'probe-{update:06d}'
                        dest.mkdir(parents=True, exist_ok=False)
                        learner.head.save_weights(str(dest/'head.safetensors'))
                        mx.savez(str(dest/'optimizer.npz'), **dict(tree_flatten(learner.optimizer.state)))
                        write_json(dest/'state.json', dict(updates=update, arm=name, sampling_rng=rng.bit_generator.state,
                            config=config, hashes={n: sha256(dest/n) for n in ('head.safetensors', 'optimizer.npz')}))
                        report = {split: evaluate(learner.head, rows[mode], rows['continuation'])
                                  for split, rows in cached.items()}
                        write_json(dest/'audit.json', report)
                        record(dict(event='audit', arm=name, updates=update, **report))
        model.save_weights(str(args.output/'frozen-world-check.safetensors'))
        if sha256(args.output/'frozen-world-check.safetensors') != config['world_sha256']:
            raise RuntimeError('frozen world changed')
        if sha256(args.data/'manifest.json') != config['dataset_sha256']:
            raise RuntimeError('dataset changed')
        record(dict(event='finished', frozen_world_verified=True, updates_per_head=args.updates))


if __name__ == '__main__':
    main()
