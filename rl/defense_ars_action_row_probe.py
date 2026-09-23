"""Read-only matched-screen action probabilities for two frozen ARS heads.

This diagnoses a saved policy change on verified own rendered screens. It
neither executes an emulator nor changes a policy, reward or training input.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import action_names
from .defense_ars import head_array
from .defense_learning import sha256, write_json


def frozen_model(checkpoint):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .model import QNetwork

    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint/'state.json').read_text())
    model_hash = sha256(checkpoint/'model.safetensors')
    if (state['hashes']['model.safetensors'] != model_hash
            or state['config']['action_names'] != list(action_names())
            or state['config']['eval_max_steps'] != 0):
        raise ValueError('incompatible or changed complete-game checkpoint')
    model = QNetwork(action_count=len(action_names()))
    model.load_weights(str(checkpoint/'model.safetensors'))
    mx.eval(model.state)
    weights = {key: np.array(value) for key, value in tree_flatten(model.parameters())}
    return model, state, model_hash, weights


def own_screens(archive, parent_hash):
    archive = Path(archive)
    index = json.loads((archive/'index.json').read_text())
    source = archive.parent/'source.py'
    if (index['source_model_sha256'] != parent_hash
            or index['source_sha256'] != sha256(source)
            or index['offsets'] != [128, 96, 64, 32]
            or not index['proposal_data_only']
            or index['native_snapshot_read'] is not False
            or index['hidden_state_read'] is not False
            or index['action_target_read'] is not False
            or len(index['games']) != 8 or len(index['files']) != 32):
        raise ValueError('requires eight verified parent-screen training games')
    if any(not game['verification']['verified'] or
           game['verification']['checkpoint_sha256'] != parent_hash
           for game in index['games']):
        raise ValueError('source game verification missing')
    batches = []
    for name in index['files']:
        path = archive/name
        metadata = json.loads(path.with_suffix('.json').read_text())
        if (sha256(path) != metadata['npz_sha256']
                or metadata['source_checkpoint_sha256'] != parent_hash
                or metadata['offsets'] != index['offsets']
                or not metadata['training_only']):
            raise ValueError('own-screen source changed: '+name)
        with np.load(path, allow_pickle=False) as arrays:
            if arrays.files != ['screens']:
                raise ValueError('screen source contains extra arrays')
            screens = arrays['screens'].copy()
        if screens.shape != (4, 4, 16, 64) or screens.dtype != np.uint8:
            raise ValueError('unexpected rendered-screen stack')
        batches.append(screens)
    return index, np.stack(batches)


def summarize(probs, offsets):
    names = action_names()
    groups = {
        'pure_RIGHT': [names.index('RIGHT')],
        'pure_rightward_movement': [names.index(s) for s in
                                    ('RIGHT', 'UP+RIGHT', 'DOWN+RIGHT')],
        'pure_leftward_movement': [names.index(s) for s in
                                   ('LEFT', 'UP+LEFT', 'DOWN+LEFT')],
        'Space_containing': [i for i, name in enumerate(names) if 'SPACE' in name],
    }
    return [dict(decisions_before_visible_loss=offset,
                 mean_command_probability={name: float(probs[:, j, ids].sum(axis=-1).mean())
                                           for name, ids in groups.items()})
            for j, offset in enumerate(offsets)]


def analyze(parent, candidate, archive):
    import mlx.core as mx

    parent_model, parent_state, parent_hash, parent_weights = frozen_model(parent)
    candidate_model, candidate_state, candidate_hash, candidate_weights = frozen_model(candidate)
    if (parent_weights.keys() != candidate_weights.keys()
            or parent_state['config']['observation_stride'] !=
            candidate_state['config']['observation_stride']):
        raise ValueError('policies do not share architecture and input timing')
    changed = [key for key in parent_weights if
               not np.array_equal(parent_weights[key], candidate_weights[key])]
    if sorted(changed) != ['advantage.bias', 'advantage.weight']:
        raise ValueError('expected only the learned action-head change')
    first, second = head_array(parent_model), head_array(candidate_model)
    rows = np.flatnonzero(np.any(first != second, axis=1))
    if rows.tolist() != [action_names().index('RIGHT')]:
        raise ValueError('expected only the pure RIGHT command row to differ')
    index, screens = own_screens(archive, parent_hash)
    observations = screens.reshape(-1, 4, 16, 64)
    features = np.array(parent_model.features(mx.array(observations)))
    if not np.isfinite(features).all():
        raise ValueError('nonfinite frozen visual features')
    logits = np.stack([features@head[:, :-1].T+head[:, -1] for head in (first, second)])
    logits = logits.reshape(2, len(screens), 4, len(action_names())).astype(np.float64)
    probs = np.exp(logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True))
    if not np.allclose(probs.sum(axis=-1), 1., atol=1e-6):
        raise ValueError('invalid categorical action probabilities')
    delta = probs[1, :, :, action_names().index('RIGHT')] - probs[0, :, :, action_names().index('RIGHT')]
    return dict(diagnostic_only=True, training_data_written=False,
                parameter_updates=0, native_reexecution=False,
                parent_checkpoint=str(Path(parent).resolve()),
                candidate_checkpoint=str(Path(candidate).resolve()),
                parent_model_sha256=parent_hash, candidate_model_sha256=candidate_hash,
                source_index_sha256=sha256(Path(archive)/'index.json'),
                diagnostic_source_sha256=sha256(Path(__file__)),
                matched_parent_own_lives=len(screens), offsets=index['offsets'],
                only_changed_command='RIGHT',
                parent=summarize(probs[0], index['offsets']),
                candidate=summarize(probs[1], index['offsets']),
                right_probability_change=dict(mean=float(delta.mean()),
                                              min=float(delta.min()), max=float(delta.max())),
                limitations=['Counterfactual action probabilities on parent own screens, not candidate own trajectories.',
                             'Visible life-loss markers are not exact collision timestamps.',
                             'Probabilities do not prove actual movement or wall versus projectile contact.',
                             'No action target, route, native state or additional reward was supplied.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('parent_screen_archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite a diagnostic')
    report = analyze(args.parent, args.candidate, args.parent_screen_archive)
    write_json(args.output, report)
    print(json.dumps(dict(matched_lives=report['matched_parent_own_lives'],
                          right_probability_change=report['right_probability_change'])))


if __name__ == '__main__':
    main()
