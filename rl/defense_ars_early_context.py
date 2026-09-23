"""Verified early own-screen feature subspace for score-only proposals."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .defense_ars_context import approach_subspace
from .defense_ars_early_source import OFFSETS
from .defense_learning import sha256


def visible_early_subspace(model, archive, components=12, expected_offsets=OFFSETS,
                           minimum_life_score=0):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .model import QNetwork
    archive = Path(archive)
    index_path = archive/'index.json'
    index = json.loads(index_path.read_text())
    if (index.get('offsets') != list(expected_offsets) or len(index.get('files', ())) != 48
            or len(index.get('games', ())) != 12 or index.get('skipped')
            or not index.get('proposal_data_only')
            or index.get('native_snapshot_read') is not False
            or index.get('hidden_state_read') is not False
            or index.get('action_target_read') is not False):
        raise ValueError('requires exact verified own early-screen source')
    if minimum_life_score < 0 or isinstance(minimum_life_score, bool):
        raise ValueError('minimum own visible life score must be nonnegative')
    source_sha = index['source_model_sha256']
    checkpoint = Path(index['source_checkpoint'])
    if (sha256(checkpoint/'model.safetensors') != source_sha
            or sha256(checkpoint/'state.json') != index['source_state_sha256']
            or sha256(archive/'source.py') != index['source_sha256']):
        raise ValueError('own early-screen source provenance changed')
    original_model = QNetwork(action_count=20)
    original_model.load_weights(str(checkpoint/'model.safetensors'))
    original = dict(tree_flatten(original_model.parameters()))
    current = dict(tree_flatten(model.parameters()))
    if original.keys() != current.keys() or any(
            not np.array_equal(np.array(original[key]), np.array(current[key]))
            for key in original if not key.startswith('advantage.')):
        raise ValueError('visual encoder differs from early own-screen source')
    if any(not game['verification']['verified'] or
           game['verification']['checkpoint_sha256'] != source_sha
           for game in index['games']):
        raise ValueError('source complete-game verification missing')
    batches = [[], [], [], []]
    last_score, next_life, omitted = {}, {}, []
    hashes = {'index.json': sha256(index_path), 'source.py': sha256(archive/'source.py')}
    for name in index['files']:
        path = archive/name
        metadata_path = path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        digest = sha256(path)
        if (digest != metadata['npz_sha256']
                or metadata['source_checkpoint_sha256'] != source_sha
                or metadata['offsets'] != list(expected_offsets)
                or not metadata['training_only']):
            raise ValueError('own early rendered-screen source changed')
        hashes[name], hashes[metadata_path.name] = digest, sha256(metadata_path)
        with np.load(path, allow_pickle=False) as arrays:
            if arrays.files != ['screens']:
                raise ValueError('early source may contain only rendered screens')
            screens = arrays['screens'].copy()
        if screens.shape != (4, 4, 16, 64) or screens.dtype != np.uint8:
            raise ValueError('invalid early rendered-screen shape')
        seed, life = metadata['seed'], metadata['life']
        if (life != next_life.get(seed, 1) or not isinstance(seed, int)
                or not isinstance(life, int)
                or not isinstance(metadata['visible_score_at_loss'], int)):
            raise ValueError('invalid own per-life score provenance')
        life_score = metadata['visible_score_at_loss']-last_score.get(seed, 0)
        if life_score < 0:
            raise ValueError('decreasing own visible score')
        last_score[seed] = metadata['visible_score_at_loss']
        next_life[seed] = life+1
        if life_score < minimum_life_score:
            omitted.append(dict(seed=seed, life=life, visible_life_score=life_score))
            continue
        for batch, frames in zip(batches, screens, strict=True):
            batch.append(frames)
    if len(batches[0]) < components+1:
        raise ValueError('too few own score-qualified rendered screens')
    features = [np.array(model.features(mx.array(np.stack(batch)))) for batch in batches]
    basis, diagnostics = approach_subspace(features[0], np.stack(features[1:]),
                                            components=components)
    return basis, dict(source_archive=str(archive.resolve()),
        source_model_sha256=source_sha, source_hashes=hashes,
        proposal_source_sha256=sha256(Path(__file__)),
        basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),
        diagnostics=diagnostics, subspace_components=components,
        minimum_visible_life_score=minimum_life_score,
        omitted_own_lives=omitted,
        selection='own verified rendered screens '+ ' / '.join(map(str, expected_offsets))+
                  ' decisions before visible loss',
        native_snapshot_read=False, hidden_state_read=False,
        action_target_read=False, frozen_encoder_match_source=True,
        fitness='none; fresh whole-boot displayed score selects updates')
