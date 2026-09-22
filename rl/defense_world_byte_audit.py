"""Read-only categorical screen reconstruction and causal forecast metrics."""

import hashlib
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def audit_bytes(model, decoder, batch, context=8):
    frames, actions = batch[:2]
    if not 0 < context < actions.shape[1]:
        raise ValueError('context must precede forecast transitions')
    horizons = [h for h in (1,4,8,16,24) if h <= actions.shape[1]-context]
    totals = {}
    digests = {}
    def record(name, logits, target, previous):
        ce = nn.losses.cross_entropy(logits, mx.array(target).astype(mx.int32), reduction='sum')
        predicted = mx.argmax(logits, axis=-1)
        mx.eval(ce, predicted)
        predicted = np.array(predicted).astype(np.uint8)
        if name not in totals:
            totals[name] = dict(cross_entropy_sum=0., cells=0, groups={})
            digests[name] = hashlib.sha256()
        row = totals[name]
        row['cross_entropy_sum'] += float(ce); row['cells'] += target.size
        digests[name].update(predicted.tobytes())
        masks = dict(all=np.ones_like(target, bool), changed=target!=previous,
            nonblank_ascii=(target>32)&(target<127), digits=(target>=48)&(target<=57),
            stars=target==42)
        for group, mask in masks.items():
            counts = row['groups'].setdefault(group, dict(cells=0, correct=0, persistence_correct=0))
            counts['cells'] += int(mask.sum())
            counts['correct'] += int(((predicted==target)&mask).sum())
            counts['persistence_correct'] += int(((previous==target)&mask).sum())
    for start in range(0,len(frames),8):
        f,a = mx.array(frames[start:start+8]),mx.array(actions[start:start+8])
        states,_ = model.observe(f,a,mx.random.key(0),sample=False)
        features = model.features(states)
        record('observed',decoder(features[:,context+1:]),np.array(f[:,context+1:]),np.array(f[:,context:-1]))
        state = tuple(s[:,context] for s in states)
        for h in range(1,max(horizons)+1):
            state,_ = model.step(state,a[:,context+h-1],mx.random.key(0),sample=False)
            if h in horizons:
                record(f'prior_{h}',decoder(model.features(state)),np.array(f[:,context+h]),np.array(f[:,context]))
    for name,row in totals.items():
        row['cross_entropy'] = row['cross_entropy_sum']/row['cells']
        row['predicted_bytes_sha256'] = digests[name].hexdigest()
        for group in row['groups'].values():
            group['accuracy'] = group['correct']/group['cells'] if group['cells'] else None
            group['persistence_accuracy'] = group['persistence_correct']/group['cells'] if group['cells'] else None
    return dict(windows=len(frames), context=context, measurements=totals,
        limitations=['Observed reconstruction is not a forecast.',
            'Prior rollouts use past filtered state and actual future actions, never future observations.',
            'Mean-latent predictions on held-out own games, not playing scores.',
            'All 256 visible byte values are targets; named groups are diagnostic only.'])
