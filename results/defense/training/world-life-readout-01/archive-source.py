"""Archive finished life readout/head/actor trials and verify actual state."""
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np

HELPER = Path('results/defense/training/world-model-bytes-01/archive-source.py')
helper = runpy.run_path(str(HELPER))
sha, read, tensors, same_arrays = (helper[name] for name in ('sha', 'read', 'tensors', 'same_arrays'))
ROOT = Path('results/defense/training/world-life-readout-01')


def copy_checked(run, name):
    source, dest = Path('runs')/run, ROOT/name
    events = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'finished'
    if not dest.exists():
        subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    for pattern in ('update-*/state.json', '*-uniform/probe-*/state.json', '*-stratified/probe-*/state.json'):
        for state in dest.glob(pattern):
            for name, digest in read(state)['hashes'].items():
                assert sha(state.parent/name) == digest
    return dest, dict(file_hashes=hashes, finished=events[-1])


def main():
    assert not (ROOT/'comparison.json').exists()
    report = {}
    runs = [('readouts', 'defense-world-life-readout-01'),
            ('head-short', 'defense-world-head-continuation-12'),
            ('head-long', 'defense-world-head-continuation-13'),
            ('actor-short', 'defense-imagination-life-head-10'),
            ('actor-long', 'defense-imagination-life-head-long-11')]
    for name, run in runs:
        dest, report[name] = copy_checked(run, name)
        if name == 'readouts':
            report[name]['final_audits'] = {p.parent.parent.name: read(p) for p in dest.glob('*/probe-002000/audit.json')}
            assert report[name]['finished']['frozen_world_verified']
        elif name.startswith('head-'):
            config = read(dest/'config.json')
            parent = Path(config['args']['world'])
            initial = dest/f"update-{config['parent_updates']:06d}"
            assert sha(initial/'world.safetensors') == sha(parent/'world.safetensors')
            for file in ('optimizer.npz', 'random.npz'):
                same_arrays(initial/file, parent/file)
            assert read(initial/'state.json')['sampling_rng'] == read(parent/'state.json')['sampling_rng']
            final = sorted(dest.glob('update-*'))[-1]
            a, b = tensors(parent/'world.safetensors'), tensors(final/'world.safetensors')
            assert a.keys() == b.keys()
            for key in a:
                if not key.startswith('continue_logit.'):
                    assert a[key] == b[key]
            assert any(a[k] != b[k] for k in a if k.startswith('continue_logit.'))
            with np.load(parent/'optimizer.npz') as a, np.load(final/'optimizer.npz') as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    if not key.startswith('continue_logit.') and key != 'step':
                        np.testing.assert_array_equal(a[key], b[key])
                assert int(b['step']) == read(final/'state.json')['updates']
            same_arrays(parent/'random.npz', final/'random.npz')
            report[name]['head_only_state_verified'] = True
            report[name]['final_audit'] = read(final/'head-audit.json')
        else:
            initial = dest/'update-002000'
            config = read(initial/'state.json')['config']
            parent = Path(config['args']['resume'])
            old, new = tensors(parent/'model.safetensors'), tensors(initial/'model.safetensors')
            assert {k: v for k, v in old.items() if k.startswith('actor.')} == {
                k: v for k, v in new.items() if k.startswith('actor.')}
            for file in ('critic.safetensors', 'critic-target.safetensors'):
                assert tensors(parent/file) == tensors(initial/file)
            for file in ('actor-optimizer.npz', 'value-optimizer.npz', 'random.npz'):
                same_arrays(parent/file, initial/file)
            assert read(parent/'state.json')['sampling_rng'] == read(initial/'state.json')['sampling_rng']
            world = Path(config['world_checkpoint'])/'world.safetensors'
            assert {k.removeprefix('world.'): v for k, v in new.items() if k.startswith('world.')} == tensors(world)
            assert sha(dest/'frozen-world-check.safetensors') == sha(world)
            evaluations = []
            for state in sorted(dest.glob('update-*/state.json')):
                result = read(state.parent/'evaluation.json')
                assert result['complete_games'] == 10 and not result['incomplete_games']
                evaluations.append(dict(actor_updates=read(state)['actor_updates'],
                    **{k: v for k, v in result.items() if k != 'games'}))
            best = dest/'artifacts/best'
            manifest, verification = read(best/'manifest.json'), read(best/'verification.json')
            for file, digest in manifest['hashes'].items():
                assert sha(best/file) == digest
            assert verification['verified']
            report[name].update(evaluations=evaluations, full_behavior_retained=True,
                replay_score=manifest['result']['score'], verified_actions=verification['verified_actions'])
    parent = Path('runs/defense-world-fit-10-byte-control/update-020000/world.safetensors')
    assert sha(ROOT/'readouts/frozen-world-check.safetensors') == sha(parent)
    world = tensors(parent)
    parent_head = {k.removeprefix('continue_logit.'): v for k, v in world.items() if k.startswith('continue_logit.')}
    for state in (ROOT/'readouts').glob('*/probe-000000/head.safetensors'):
        assert tensors(state) == parent_head
    control = Path('results/defense/training/world-model-bytes-01/actor-control')
    report['reused_control'] = dict(path=str(control),
        evaluations=[read(p) for p in sorted(control.glob('update-*/evaluation.json'))],
        initial_state_sha256=sha(control/'update-002000/state.json'))
    # All arms start with exactly the already-tested control's behavior state.
    for name in ('actor-short', 'actor-long'):
        initial = ROOT/name/'update-002000'
        for file in ('actor-optimizer.npz', 'value-optimizer.npz', 'random.npz'):
            same_arrays(initial/file, control/'update-002000'/file)
        a, b = tensors(initial/'model.safetensors'), tensors(control/'update-002000/model.safetensors')
        assert {k: v for k, v in a.items() if k.startswith('actor.')} == {
            k: v for k, v in b.items() if k.startswith('actor.')}
    best = Path('results/defense/learned/best')
    manifest = read(best/'manifest.json')
    for name, digest in manifest['hashes'].items():
        assert sha(best/name) == digest
    report.update(global_best_score=manifest['result']['score'], archive_helper_sha256=sha(HELPER),
        limitations=['Reused validation seeds, not a fresh gameplay success estimate.',
            'No change to actor input, score reward, game or train/held-out assignments.',
            'New-head diagnostic and full-Adam head-only training are separate experiments.',
            'Observed arrival sees its screen; recognition is not anticipation.',
            'Whole-game unique post-burn rows differ from the earlier selected 48 loss windows.'])
    (ROOT/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {n: v for n, v in report[k].items()
        if n in ('evaluations', 'replay_score', 'verified_actions')} for k in ('actor-short', 'actor-long')}, indent=2))


if __name__ == '__main__':
    main()
