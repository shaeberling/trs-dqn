"""Visible self-play context for proposing Defense search directions.

Only recorded rendered screens are opened from own life-loss training games.
Opaque native snapshots are checksummed as source files but never decoded,
restored, passed to the policy or used as a fitness signal here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .defense_learning import sha256


def contrast_basis(earlier, middle, later):
    """Affine feature contrast: mean earlier projection 0, later mean 1."""
    earlier, middle, later = (np.asarray(value, np.float32)
                              for value in (earlier, middle, later))
    if (earlier.ndim != 2 or earlier.shape != middle.shape or earlier.shape != later.shape
            or not len(earlier) or earlier.shape[1] < 2
            or not all(np.isfinite(value).all() for value in (earlier, middle, later))):
        raise ValueError('invalid visible self-play feature batches')
    early_mean = earlier.mean(axis=0)
    late_mean = (middle.mean(axis=0)+later.mean(axis=0))/2
    difference = late_mean-early_mean
    squared = float(np.dot(difference, difference))
    if not np.isfinite(squared) or squared < 1e-8:
        raise ValueError('own approach features have no usable contrast')
    weights = difference/squared
    bias = -float(np.dot(weights, early_mean))
    basis = np.concatenate((weights, [bias])).astype(np.float32)
    def describe(features):
        values = features@basis[:-1]+basis[-1]
        return dict(mean=float(values.mean()), median=float(np.median(values)),
                    minimum=float(values.min()), maximum=float(values.max()))
    return basis, dict(feature_count=earlier.shape[1], examples=len(earlier),
        difference_norm=float(np.sqrt(squared)), earlier=describe(earlier),
        middle=describe(middle), later=describe(later),
        construction='own visible screen features; centered early, unit mean middle/late contrast')


def visible_context(model, archive, expected_model_sha256):
    """Verify own training traces and compute context without reading native state."""
    import mlx.core as mx
    archive = Path(archive)
    index_path = archive/'index.json'
    source_config_path = archive.parent/'config.json'
    index = json.loads(index_path.read_text())
    source_config = json.loads(source_config_path.read_text())
    if (source_config.get('focus', {}).get('source_model_sha256') != expected_model_sha256
            or index.get('lookback') != 128 or len(index.get('files', ())) != 48
            or len(index.get('games', ())) != 12):
        raise ValueError('context requires original own 128-decision source games')
    names = index['files']
    if len(set(names)) != len(names) or any(not name.startswith('seed-') or not name.endswith('.npz') for name in names):
        raise ValueError('unexpected own source file names')
    early, middle, late = [], [], []
    hashes = {'index.json': sha256(index_path),
              'source-config.json': sha256(source_config_path)}
    for name in names:
        path = archive/name
        metadata_path = path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        digest = sha256(path)
        source = metadata['source']
        if (digest != metadata['npz_sha256'] or not source.get('training_only')
                or source['visible_loss_frame']-source['snapshot_action'] != 128):
            raise ValueError('unverified own visible source: '+name)
        hashes[name] = digest
        hashes[metadata_path.name] = sha256(metadata_path)
        with np.load(path, allow_pickle=False) as arrays:
            # Deliberately never access arrays['native'] or snapshot fields.
            first = arrays['frames'].copy()
            screens = arrays['screens']
            second = screens[60:64].copy()   # 64 decisions before visible loss.
            third = screens[92:96].copy()    # 32 decisions before visible loss.
        if (first.shape != (4, 16, 64) or second.shape != first.shape
                or third.shape != first.shape or any(
                    value.dtype != np.uint8 for value in (first, second, third))):
            raise ValueError('unexpected rendered-screen source shape')
        early.append(first); middle.append(second); late.append(third)
    features = [np.array(model.features(mx.array(np.stack(batch))))
                for batch in (early, middle, late)]
    basis, diagnostics = contrast_basis(*features)
    return basis, dict(source_archive=str(archive.resolve()),
        source_model_sha256=expected_model_sha256, source_hashes=hashes,
        basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        diagnostics=diagnostics,
        selection='own visible life loss, screens at 128 / 64 / 32 decisions before marker',
        native_snapshot_read=False, action_target_read=False,
        fitness='none; directions proposed only, whole boot displayed score selects updates')
