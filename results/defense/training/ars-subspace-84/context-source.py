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
        proposal_source_sha256=sha256(Path(__file__)),
        basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        diagnostics=diagnostics,
        selection='own visible life loss, screens at 128 / 64 / 32 decisions before marker',
        native_snapshot_read=False, action_target_read=False,
        fitness='none; directions proposed only, whole boot displayed score selects updates')


def approach_subspace(earlier, approach, components=4):
    """Early mean contrast plus visual-variation axes from own approach screens.

    The axes are proposals in a frozen neural feature space, not labels,
    routes, collision coordinates, actions, rewards or policy inputs.
    """
    earlier = np.asarray(earlier, np.float32)
    approach = np.asarray(approach, np.float32)
    if (earlier.ndim != 2 or approach.ndim != 3 or approach.shape[0] != 3
            or approach.shape[1:] != earlier.shape or earlier.shape[0] < components+1
            or earlier.shape[1] < components+2 or components < 1
            or not np.isfinite(earlier).all() or not np.isfinite(approach).all()):
        raise ValueError('invalid own-screen approach feature batches')
    mean_basis, mean_diagnostics = contrast_basis(earlier, approach[0], approach[1])
    mean_direction = mean_basis[:-1].astype(np.float64)
    mean_direction /= np.linalg.norm(mean_direction)
    deltas = (approach-earlier[None]).reshape(-1, earlier.shape[1]).astype(np.float64)
    residual = deltas-(deltas@mean_direction)[:, None]*mean_direction
    _, singular, axes = np.linalg.svd(residual, full_matrices=False)
    if singular[components-1] < 1e-6:
        raise ValueError('own approach features lack independent visual axes')
    early_mean = earlier.mean(axis=0).astype(np.float64)
    approach_features = approach.reshape(-1, earlier.shape[1]).astype(np.float64)
    basis = [mean_basis]
    rms_values = []
    for axis in axes[:components]:
        axis = axis.copy()
        pivot = np.argmax(np.abs(axis))
        if axis[pivot] < 0:
            axis *= -1
        centered = (approach_features-early_mean)@axis
        rms = float(np.sqrt(np.mean(centered**2)))
        if rms < 1e-6:
            raise ValueError('degenerate own-screen visual axis')
        weights = axis/rms
        basis.append(np.concatenate((weights, [-weights@early_mean])).astype(np.float32))
        rms_values.append(rms)
    basis = np.stack(basis)
    if not np.isfinite(basis).all():
        raise FloatingPointError('nonfinite visual proposal subspace')
    return basis, dict(examples=len(earlier), feature_count=earlier.shape[1],
        components=components, mean_contrast=mean_diagnostics,
        residual_singular_values=[float(x) for x in singular[:components]],
        component_rms_before_normalization=rms_values,
        construction='early mean contrast and principal variation of own visible approaches')


def visible_subspace(model, archive, expected_model_sha256, components=4):
    """Verified own rendered screens at -128, -96, -64 and -32 decisions."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .model import QNetwork
    archive = Path(archive)
    _, provenance = visible_context(model, archive, expected_model_sha256)
    source_config = json.loads((archive.parent/'config.json').read_text())
    source_weights = Path(source_config['focus']['source_checkpoint'])/'model.safetensors'
    if sha256(source_weights) != expected_model_sha256:
        raise ValueError('own visible source checkpoint changed')
    source_model = QNetwork(action_count=20)
    source_model.load_weights(str(source_weights))
    original = dict(tree_flatten(source_model.parameters()))
    current = dict(tree_flatten(model.parameters()))
    if original.keys() != current.keys() or any(
            not np.array_equal(np.array(original[key]), np.array(current[key]))
            for key in original if not key.startswith('advantage.')):
        raise ValueError('visual encoder differs from own-screen source policy')
    names = json.loads((archive/'index.json').read_text())['files']
    batches = [[], [], [], []]
    for name in names:
        path = archive/name
        if sha256(path) != provenance['source_hashes'][name]:
            raise ValueError('visible source changed during proposal construction')
        with np.load(path, allow_pickle=False) as arrays:
            # Never access arrays['native']; all four samples are own screen history.
            screens = arrays['screens']
            frames = (arrays['frames'].copy(), screens[28:32].copy(),
                      screens[60:64].copy(), screens[92:96].copy())
        if any(frame.shape != (4, 16, 64) or frame.dtype != np.uint8 for frame in frames):
            raise ValueError('unexpected rendered-screen source shape')
        for batch, frame in zip(batches, frames, strict=True):
            batch.append(frame)
    features = [np.array(model.features(mx.array(np.stack(batch)))) for batch in batches]
    basis, diagnostics = approach_subspace(features[0], np.stack(features[1:]), components)
    provenance = dict(provenance, basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        diagnostics=diagnostics,
        selection='own visible loss; screens 128 / 96 / 64 / 32 decisions before marker',
        subspace_components=components, frozen_encoder_match_source=True)
    return basis, provenance
