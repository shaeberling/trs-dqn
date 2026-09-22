"""Train and audit the dynamics preflight; never acts or promotes a replay."""

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from .defense_learning import sha256, write_json
from .defense_world_data import Sequences, disk_guard
from .defense_world_model import WorldLearner


def audit(model, batch, context=8):
    """Mean-latent open-loop prediction. Future frames are targets ONLY.

    Rotating action labels measures sensitivity, not true counterfactuals.
    All methods are measured on exactly the same held-out windows.
    """
    frames, actions, rewards, continuation = (mx.array(v) for v in batch)
    if not 0 < context < actions.shape[1]:
        raise ValueError('context must precede at least one held-out transition')
    states, _ = model.observe(frames[:, :context+1], actions[:, :context], mx.random.key(0), sample=False)
    state = tuple(s[:, -1] for s in states)
    alternative = state
    previous = model.pixels(frames[:, context])
    rows = []
    for t in range(context, actions.shape[1]):
        state, _ = model.step(state, actions[:, t], mx.random.key(0), sample=False)
        alternative, _ = model.step(alternative, (actions[:, t]+1) % 20, mx.random.key(0), sample=False)
        feature, altered = model.features(state), model.features(alternative)
        prediction, rotated = model.decode(feature), model.decode(altered)
        truth = model.pixels(frames[:, t+1])
        # Changed graphics locations: persistence cannot win just by repeating black background.
        changed = mx.abs(truth[..., 0]-previous[..., 0]) > .01
        graphics = (prediction[..., 0]-truth[..., 0])**2
        reward_error = (model.reward(feature)[..., 0]-rewards[:, t])**2
        positive = rewards[:, t] > 0
        probability = mx.sigmoid(model.continue_logit(feature)[..., 0])
        values = dict(pixel_mse=mx.mean((prediction-truth)**2),
            graphics_mse=mx.mean(graphics), rotated_graphics_mse=mx.mean((rotated[..., 0]-truth[..., 0])**2),
            persistence_graphics_mse=mx.mean((previous[..., 0]-truth[..., 0])**2),
            changed_graphics_mse=mx.sum(graphics*changed)/mx.maximum(changed.sum(), 1),
            changed_graphics_count=changed.sum(),
            prediction_action_sensitivity=mx.mean((prediction-rotated)**2),
            reward_mse=mx.mean(reward_error), zero_reward_mse=mx.mean(rewards[:, t]**2),
            positive_reward_mse=mx.sum(reward_error*positive)/mx.maximum(positive.sum(), 1),
            positive_rewards=positive.sum(),
            continuation_brier=mx.mean((probability-continuation[:, t])**2),
            always_continue_brier=mx.mean((1-continuation[:, t])**2),
            visible_boundaries=mx.sum(1-continuation[:, t]))
        mx.eval(values)
        rows.append(dict(horizon=t-context+1, **{k: float(v) for k, v in values.items()}))
    return dict(windows=len(frames), context=context, horizons=rows,
        limitations=['Whole held-out own-collection games, never gradient-update data.',
                     'Deterministic mean-latent forecasts, not distribution-calibrated samples.',
                     'Pixel accuracy and action sensitivity do not establish useful control.',
                     'Rotated labels are a sensitivity check, not true counterfactual observations.',
                     'Continuation labels use visible life boundaries, not exact collision times.',
                     'No actor, imagination-based behavior learning or mission claim yet.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--length', type=int, default=32)
    parser.add_argument('--burn', type=int, default=8)
    parser.add_argument('--audit-windows', type=int, default=64)
    parser.add_argument('--every', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--extend-data', action='store_true',
                        help='explicitly continue on a checked union containing the entire previous dataset')
    parser.add_argument('--boundary-fraction', type=float, default=0.,
                        help='training-window mixture fraction containing a visible life loss after burn-in')
    parser.add_argument('--change-sampling', action='store_true',
                        help='explicitly change window sampling on a full optimizer/RNG continuation')
    args = parser.parse_args()
    if (args.output.exists() or min(args.updates, args.batch, args.every, args.audit_windows) < 1
            or not 0 < args.burn < args.length or not np.isfinite(args.boundary_fraction)
            or not 0 <= args.boundary_fraction <= 1):
        parser.error('use a new output, positive counts, and 0 < burn < length')
    if args.extend_data and not args.resume:
        parser.error('data extension requires a full optimizer/RNG resume')
    if args.change_sampling and not args.resume:
        parser.error('sampling change requires a full optimizer/RNG resume')
    mx.set_cache_limit(128*1024*1024)
    train, held = Sequences(args.data, 'train', args.length), Sequences(args.data, 'heldout', args.length)
    if set(train.files) & set(held.files):
        raise ValueError('train/held-out overlap')
    rng = np.random.default_rng(args.seed)
    frozen = held.sample(args.audit_windows, np.random.default_rng(82731))
    learner = WorldLearner(seed=args.seed, burn=args.burn)
    metadata = dict(data=str(args.data.resolve()), dataset_sha256=sha256(args.data/'manifest.json'),
        architecture='small-gaussian-rssm-v1', reward_scale=.01, batch=args.batch,
        length=args.length, burn=args.burn, train_games=len(train.episodes), heldout_games=len(held.episodes),
        boundary_fraction=args.boundary_fraction,
        train_windows=int(train.ends[-1]), heldout_windows=int(held.ends[-1]),
        fit_source_sha256=sha256(Path(__file__)),
        model_source_sha256=sha256(Path(__file__).with_name('defense_world_model.py')),
        dynamics_only=True, promotion_eligible=False,
        source_paper='https://arxiv.org/abs/1912.01603', args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    if args.resume:
        previous = learner.restore(args.resume, rng)
        if args.extend_data:
            union = json.loads((args.data/'manifest.json').read_text())
            if (union.get('dataset_operation') != 'immutable byte-identical union; no new trajectories or split changes'
                    or previous['dataset_sha256'] not in [s['sha256'] for s in union.get('source_manifests', [])]):
                parser.error('new dataset must be a checked union containing the previous dataset')
            # Verify that every previous episode is included byte-identically
            # with its original held-out label, not merely named in metadata.
            source = next(s for s in union['source_manifests'] if s['sha256']==previous['dataset_sha256'])
            old_path = Path(source['path'])/'manifest.json'
            if sha256(old_path) != previous['dataset_sha256']:
                parser.error('extension source provenance mismatch')
            old_rows = json.loads(old_path.read_text())['episodes']
            seeds = [r['seed'] for r in union['episodes']]
            if len(seeds) != len(set(seeds)):
                parser.error('extension contains overlapping episode seeds')
            members = {(r['seed'], r['sha256'], r['split']) for r in union['episodes']}
            if not all((r['seed'], r['sha256'], r['split']) in members for r in old_rows):
                parser.error('extension omitted or relabeled an original episode')
            metadata['previous_dataset_sha256'] = previous['dataset_sha256']
            metadata['dataset_extension'] = True
        elif previous['dataset_sha256'] != metadata['dataset_sha256']:
            parser.error('dataset differs; use an explicit checked union extension')
        for field in ('architecture', 'batch', 'length', 'burn', 'reward_scale'):
            if previous[field] != metadata[field]:
                raise ValueError('incompatible continuation: '+field)
        previous_fraction = previous.get('boundary_fraction', 0.)
        if previous_fraction != args.boundary_fraction:
            if not args.change_sampling:
                parser.error('sampling differs; declare --change-sampling')
            metadata['previous_boundary_fraction'] = previous_fraction
            metadata['sampling_change'] = True
    if learner.updates >= args.updates:
        parser.error('target updates must exceed restored updates')
    disk_guard(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', metadata)
    started = time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(row):
            row = dict(**row, elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps(row), flush=True)
        def preserve():
            disk_guard(args.output)
            destination = args.output/f'update-{learner.updates:06d}'
            learner.save(destination, rng, metadata)
            report = audit(learner.model, frozen, args.burn)
            write_json(destination/'audit.json', report)
            record(dict(event='audit', updates=learner.updates, first=report['horizons'][0],
                        last=report['horizons'][-1]))
        preserve()
        while learner.updates < args.updates:
            batch = train.sample(args.batch, rng, args.boundary_fraction, args.burn)
            stats = learner.train(batch)
            if learner.updates % 20 == 0:
                disk_guard(args.output)
                record(dict(event='update', updates=learner.updates,
                            sampled_loss_targets=int((batch[3][:, args.burn:] == 0).sum()), **stats))
            if learner.updates % args.every == 0 or learner.updates == args.updates:
                preserve()
        if sha256(args.data/'manifest.json') != metadata['dataset_sha256']:
            raise RuntimeError('dataset manifest changed during fit')
        # Recheck episode files too, not just the manifest.
        for row in json.loads((args.data/'manifest.json').read_text())['episodes']:
            if sha256(args.data/row['file']) != row['sha256']:
                raise RuntimeError('dataset episode changed during fit')
        record(dict(event='finished', updates=learner.updates))


if __name__ == '__main__':
    main()
