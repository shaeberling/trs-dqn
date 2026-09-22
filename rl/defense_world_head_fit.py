"""Continue only the learned continuation head on own frozen world features.

Preserves the complete parent Adam and RNG state. Encoder/dynamics/reward
weights and their Adam moments do not change; the global Adam step advances.
This is an explicit head-only training phase, not an optimizer reset.
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
from .defense_world_life_probe import evaluate, extract, sample
from .defense_world_model import WorldLearner


class HeadLearner:
    def __init__(self, base):
        self.base = base
        self.state = [base.model.state, base.optimizer.state]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def _update(self, x, y, weights):
        base = self.base
        def objective(head):
            return mx.mean(nn.losses.binary_cross_entropy(
                head(x)[:, 0], y, with_logits=True, reduction='none')*weights)
        loss, grads = nn.value_and_grad(base.model.continue_logit, objective)(base.model.continue_logit)
        grads, norm = optim.clip_grad_norm(grads, max_norm=100.)
        # MLX accepts a gradient subtree; omitted parameters AND their moments
        # remain untouched. Sending zeros would incorrectly decay momentum.
        base.optimizer.update(base.model, {'continue_logit': grads})
        return loss, norm

    def train(self, x, y, weights):
        result = self.update(mx.array(x), mx.array(y), mx.array(weights))
        mx.eval(result, self.state)
        values = [float(v) for v in result]
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite head continuation update')
        self.base.updates += 1
        return dict(loss=values[0], gradient_norm=values[1])


def immutable_arrays(base):
    return {group+'.'+key: np.array(value).copy()
            for group, tree in [('model', base.model.parameters()), ('optimizer', base.optimizer.state), ('random', base.random)]
            for key, value in tree_flatten(tree)
            if not key.startswith('continue_logit.') and not (group == 'optimizer' and key == 'step')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2000, help='additional head-only updates')
    parser.add_argument('--every', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--features', choices=('observed', 'prior_1'), default='observed')
    parser.add_argument('--stratified', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or min(args.updates, args.every, args.batch) < 1 or args.batch % 2:
        parser.error('new output, positive counts and even batch required')
    saved = json.loads((args.world/'state.json').read_text())
    metadata = saved['metadata']
    if (metadata['dataset_sha256'] != sha256(args.data/'manifest.json') or metadata['length'] != 32
            or saved['burn'] != 8 or saved.get('byte_reconstruction')
            or saved.get('overshooting', {}).get('weight', 0.)):
        parser.error('requires matching plain 32-action/8-burn world and own dataset')
    mx.set_cache_limit(128*1024*1024)
    base, rng = WorldLearner(seed=0), np.random.default_rng(0)
    base.restore(args.world, rng)
    unchanged = immutable_arrays(base)
    initial_updates = base.updates
    metadata = dict(metadata, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        parent_world_sha256=sha256(args.world/'world.safetensors'), parent_updates=initial_updates,
        training_phase='continuation-head-only, cached mean-latent own features',
        head_features=args.features, head_batch=args.batch, head_stratified=args.stratified,
        head_additional_updates=args.updates, full_parent_optimizer_retained=True,
        fit_source_sha256=sha256(Path(__file__)),
        feature_source_sha256=sha256(Path(__file__).with_name('defense_world_life_probe.py')),
        model_source_sha256=sha256(Path(__file__).with_name('defense_world_model.py')))
    metadata['args']['resume'] = str(args.world.resolve())
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', metadata)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(row):
            row = dict(row, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
        cached, files = {}, {}
        for split in ('train', 'heldout'):
            dataset = Sequences(args.data, split, 32)
            cached[split], files[split] = extract(base.model, dataset), dataset.files
            record(dict(event='features', split=split, rows=len(cached[split]['continuation'])))
        if set(files['train']) & set(files['heldout']):
            raise ValueError('train/held-out overlap')
        learner = HeadLearner(base)
        def preserve():
            disk_guard(args.output)
            dest = args.output/f'update-{base.updates:06d}'
            base.save(dest, rng, metadata)
            audit = {split: {mode: evaluate(base.model.continue_logit, rows[mode], rows['continuation'])
                            for mode in ('observed', 'prior_1')} for split, rows in cached.items()}
            write_json(dest/'head-audit.json', audit)
            record(dict(event='audit', updates=base.updates, **audit))
        preserve()
        train = cached['train']
        while base.updates < initial_updates+args.updates:
            indices, weights = sample(train['continuation'], rng, args.batch, args.stratified)
            stats = learner.train(train[args.features][indices], train['continuation'][indices], weights)
            if base.updates % 100 == 0:
                record(dict(event='update', updates=base.updates, **stats))
            if (base.updates-initial_updates) % args.every == 0 or base.updates == initial_updates+args.updates:
                preserve()
        after = immutable_arrays(base)
        assert unchanged.keys() == after.keys()
        for key, value in unchanged.items():
            np.testing.assert_array_equal(value, after[key])
        assert int(base.optimizer.step) == base.updates
        if sha256(args.data/'manifest.json') != metadata['dataset_sha256']:
            raise RuntimeError('dataset changed')
        record(dict(event='finished', updates=base.updates, non_head_parameters_moments_and_rng_verified=True))


if __name__ == '__main__':
    main()
