"""Inspect frozen forecasts around held-out visible losses, never training data."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw

from .defense_learning import sha256, write_json
from .defense_world_data import Sequences
from .defense_world_fit import audit
from .defense_world_model import WorldModel


def boundary_windows(dataset, context=8, lead=16):
    if not 1 <= lead <= dataset.length-context:
        raise ValueError('invalid visible-boundary lead')
    rows, provenance = [], []
    for filename, episode in zip(dataset.files, dataset.episodes, strict=True):
        frames, actions, rewards, continuation = episode
        for boundary in np.flatnonzero(continuation == 0):
            start = int(boundary-context-lead+1)
            end = start+dataset.length
            if start < 0 or end > len(actions):
                continue
            rows.append((frames[start:end+1], actions[start:end], rewards[start:end]*.01,
                         continuation[start:end]))
            provenance.append(dict(file=filename, start=start, visible_loss_action=int(boundary),
                                   visible_loss_horizon=lead))
    if not rows:
        raise ValueError('no complete boundary windows')
    return tuple(np.stack([r[k] for r in rows]) for k in range(4)), provenance


def render(model, batch, path, context):
    frames, actions = mx.array(batch[0]), mx.array(batch[1])
    # Display up to four cases; all cases enter the numerical audit.
    frames, actions = frames[:4], actions[:4]
    states, _ = model.observe(frames[:, :context+1], actions[:, :context], mx.random.key(0), sample=False)
    state = tuple(s[:, -1] for s in states)
    selected = (1, 4, 8, 16, 24)
    forecasts = {}
    for t in range(context, actions.shape[1]):
        state, _ = model.step(state, actions[:, t], mx.random.key(0), sample=False)
        if t-context+1 in selected:
            forecasts[t-context+1] = np.array(model.decode(model.features(state)))[..., 0]+.5
    targets = np.array(model.pixels(frames))[..., 0]+.5
    sheet = Image.new('RGB', (5*384, len(frames)*330+55), '#18202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), 'HELD-OUT visible-loss windows. Top: recorded graphics. Bottom: predicted graphics.', fill='white')
    draw.text((10, 26), 'Graphics channel only; predicted gray values show uncertainty/blur. No future screens supplied to predictor.', fill='white')
    for row in range(len(frames)):
        for col, horizon in enumerate(selected):
            x, y = col*384, row*330+55
            draw.text((x+5, y), f'Case {row+1}, horizon {horizon}', fill='white')
            for offset, picture in ((20, targets[row, context+horizon]), (178, forecasts[horizon][row])):
                pixels = np.round(np.clip(picture, 0, 1)*255).astype(np.uint8)
                sheet.paste(Image.fromarray(pixels).resize((384, 144), Image.Resampling.NEAREST), (x, y+offset))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    state = json.loads((args.checkpoint/'state.json').read_text())
    metadata = state['metadata']
    if (metadata['dataset_sha256'] != sha256(args.data/'manifest.json')
            or metadata['length'] != 32 or metadata['burn'] != 8):
        parser.error('requires matching 32-action/8-context dynamics preflight')
    for name, digest in state['hashes'].items():
        if sha256(args.checkpoint/name) != digest:
            raise ValueError('checkpoint checksum mismatch')
    before = sha256(args.checkpoint/'world.safetensors')
    mx.set_cache_limit(128*1024*1024)
    model = WorldModel()
    model.load_weights(str(args.checkpoint/'world.safetensors'))
    held = Sequences(args.data, 'heldout', 32)
    batch, origins = boundary_windows(held)
    report = audit(model, batch, context=8)
    report.update(checkpoint_sha256=before, dataset_sha256=metadata['dataset_sha256'],
                  review_source_sha256=sha256(Path(__file__)), origins=origins,
                  selection='held-out windows with visible loss at forecast action 16; excludes incomplete windows',
                  training_updates=0, promotion_eligible=False)
    args.output.mkdir(parents=True, exist_ok=False)
    render(model, batch, args.output/'forecasts.png', 8)
    if before != sha256(args.checkpoint/'world.safetensors'):
        raise RuntimeError('world model changed')
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(windows=len(origins), loss_horizon=report['horizons'][15])), flush=True)


if __name__ == '__main__':
    main()
