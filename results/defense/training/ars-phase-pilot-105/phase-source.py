"""Two late-approach visual axes from verified own pre-loss screens only.

These are proposal directions for a frozen screen-feature policy head, not
action labels, routes, rewards, hidden collision coordinates or policy inputs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .defense_ars_context import contrast_basis
from .defense_ars_early_context import visible_early_subspace
from .defense_learning import sha256


OFFSETS = (128, 96, 64, 32)


def phase_basis(earlier, middle, later):
    """Centered early→middle and independent middle→late feature changes."""
    earlier, middle, later = (np.asarray(batch, np.float32)
                              for batch in (earlier, middle, later))
    if (earlier.ndim != 2 or middle.shape != earlier.shape or later.shape != earlier.shape
            or len(earlier) < 3 or earlier.shape[1] < 3
            or not all(np.isfinite(batch).all() for batch in (earlier, middle, later))):
        raise ValueError('invalid own-screen phase features')
    first, first_diagnostics = contrast_basis(earlier, middle, middle)
    second, second_diagnostics = contrast_basis(middle, later, later)
    first_weights = first[:-1].astype(np.float64)
    late_weights = second[:-1].astype(np.float64)
    residual = late_weights - (late_weights@first_weights)/(first_weights@first_weights)*first_weights
    late_difference = later.mean(axis=0).astype(np.float64)-middle.mean(axis=0).astype(np.float64)
    late_response = float(residual@late_difference)
    if not np.isfinite(late_response) or abs(late_response) < 1e-6:
        raise ValueError('late visual change is not independent of earlier change')
    residual /= late_response
    bias = -float(residual@middle.mean(axis=0))
    basis = np.stack([first, np.concatenate((residual, [bias])).astype(np.float32)])
    if not np.isfinite(basis).all():
        raise FloatingPointError('nonfinite own-screen phase basis')
    def describe(features):
        values = features@basis[:, :-1].T+basis[:, -1]
        return [float(x) for x in values.mean(axis=0)]
    return basis, dict(examples=len(earlier), feature_count=earlier.shape[1],
        components=2, first_contrast=first_diagnostics,
        raw_late_contrast=second_diagnostics,
        late_independent_response_before_normalization=late_response,
        mean_projection_earlier=describe(earlier),
        mean_projection_middle=describe(middle),
        mean_projection_later=describe(later),
        construction='own rendered screen feature changes at -96/-64 and -64/-32; no target action')


def visible_phase_subspace(model, archive, minimum_life_score=2400):
    import mlx.core as mx

    archive = Path(archive)
    # Reuse the strict verified-source, frozen-encoder, score-filter and
    # no-hidden-state checks of the previously audited source loader.
    _, provenance = visible_early_subspace(model, archive, components=2,
        expected_offsets=OFFSETS, minimum_life_score=minimum_life_score)
    index = json.loads((archive/'index.json').read_text())
    batches = [[], [], []]
    last_score = {}
    for name in index['files']:
        path = archive/name
        if sha256(path) != provenance['source_hashes'][name]:
            raise ValueError('own screen changed after provenance verification')
        metadata = json.loads(path.with_suffix('.json').read_text())
        life_score = metadata['visible_score_at_loss']-last_score.get(metadata['seed'], 0)
        last_score[metadata['seed']] = metadata['visible_score_at_loss']
        if life_score < minimum_life_score:
            continue
        with np.load(path, allow_pickle=False) as arrays:
            if arrays.files != ['screens']:
                raise ValueError('unexpected own-screen source arrays')
            screens = arrays['screens'].copy()
        if screens.shape != (4, 4, 16, 64) or screens.dtype != np.uint8:
            raise ValueError('invalid verified own-screen sample')
        for batch, frame in zip(batches, screens[1:], strict=True):
            batch.append(frame)
    features = [np.array(model.features(mx.array(np.stack(batch)))) for batch in batches]
    basis, diagnostics = phase_basis(*features)
    return basis, dict(provenance,
        proposal_source_sha256=sha256(Path(__file__)),
        basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        diagnostics=diagnostics, subspace_components=2,
        selection='own verified rendered screens at 96 / 64 / 32 decisions before visible loss',
        fitness='none; fresh complete boot-game displayed score alone selects updates')
