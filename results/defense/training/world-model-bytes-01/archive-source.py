"""Preserve the completed categorical reconstruction comparison, with checks."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


ROOT = Path('results/defense/training/world-model-bytes-01')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def same_arrays(first, second):
    with np.load(first) as a, np.load(second) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def tensors(path):
    raw = path.read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    header, body = json.loads(raw[8:8+length]), raw[8+length:]
    return {k: (v['dtype'], v['shape'], body[v['data_offsets'][0]:v['data_offsets'][1]])
            for k, v in header.items() if k != '__metadata__'}


def archive(run, destination):
    source, dest = Path('runs')/run, ROOT/destination
    events = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'finished'
    if not dest.exists():
        subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    # Replay state.json retains source metadata but the replay bundle omits
    # training-only optimizer/critic files. Its own manifest is checked below.
    for state_path in dest.glob('*/state.json'):
        for name, digest in read(state_path).get('hashes', {}).items():
            assert sha(state_path.parent/name) == digest
    return dest, dict(file_hashes=hashes, finished=events[-1])


def main():
    assert not (ROOT/'comparison.json').exists()
    report = {}
    for arm, run in [('control', 'defense-world-fit-10-byte-control'),
                     ('bytes', 'defense-world-fit-11-byte-reconstruction'),
                     ('frozen-readout', 'defense-world-byte-frozen-readout-01'),
                     ('actor-control', 'defense-imagination-byte-control-08'),
                     ('actor-bytes', 'defense-imagination-byte-reconstruction-09')]:
        dest, report[arm] = archive(run, arm)
        if arm in ('control', 'bytes'):
            assert report[arm]['finished']['updates'] == 20000
            audit = read(dest/'update-020000/audit.json')
            report[arm]['uniform_forecast_brier'] = sum(
                h['continuation_brier'] for h in audit['horizons'])/24
            report[arm]['uniform_always_survive_brier'] = sum(
                h['always_continue_brier'] for h in audit['horizons'])/24
            report[arm]['boundary_predictions'] = read(
                dest/'update-020000/boundary-predictions.json')['predictions']
            report[arm]['loss_horizon_16'] = read(dest/'review-20000/report.json')['horizons'][15]
        if arm == 'bytes':
            report[arm]['byte_audit'] = read(dest/'update-020000/byte-audit.json')
        if arm == 'frozen-readout':
            report[arm]['byte_audit'] = read(dest/'probe-002000/byte-audit.json')
            assert report[arm]['finished']['frozen_world_and_optimizer_verified']
        if arm.startswith('actor-'):
            assert report[arm]['finished']['frozen_world_verified']
            assert report[arm]['finished']['actor_updates'] == 3000
            evaluations = []
            for checkpoint in sorted(dest.glob('update-*')):
                result = read(checkpoint/'evaluation.json')
                assert result['complete_games'] == 10 and not result['incomplete_games']
                evaluations.append(dict(actor_updates=read(checkpoint/'state.json')['actor_updates'],
                    **{k: v for k, v in result.items() if k != 'games'}))
            report[arm]['evaluations'] = evaluations
            best = dest/'artifacts/best'
            manifest, verified = read(best/'manifest.json'), read(best/'verification.json')
            for name, digest in manifest['hashes'].items():
                assert sha(best/name) == digest
            assert verified['verified']
            report[arm].update(replay_score=manifest['result']['score'],
                              verified_actions=verified['verified_actions'])

    parent = Path('runs/defense-world-fit-08-overshoot-control/update-018000')
    for arm in ('control', 'bytes'):
        initial = ROOT/arm/'update-018000'
        assert sha(initial/'world.safetensors') == sha(parent/'world.safetensors')
        for name in ('optimizer.npz', 'random.npz'):
            same_arrays(initial/name, parent/name)
        assert read(initial/'state.json')['sampling_rng'] == read(parent/'state.json')['sampling_rng']
    initial, frozen = ROOT/'bytes/update-018000', ROOT/'frozen-readout'
    assert sha(initial/'byte-decoder.safetensors') == sha(frozen/'probe-000000/byte-decoder.safetensors')
    for name in ('byte-optimizer.npz', 'random.npz'):
        same_arrays(initial/name, frozen/'probe-000000'/name)
    assert read(initial/'state.json')['sampling_rng'] == read(frozen/'probe-000000/state.json')['sampling_rng']
    assert sha(frozen/'frozen-world-check.safetensors') == sha(parent/'world.safetensors')
    same_arrays(frozen/'frozen-world-optimizer.npz', parent/'optimizer.npz')

    actor_parent = Path('runs/defense-imagination-overshoot-control-06/update-002000')
    old = tensors(actor_parent/'model.safetensors')
    for arm in ('control', 'bytes'):
        actor = ROOT/f'actor-{arm}'
        initial = actor/'update-002000'
        for name in ('actor-optimizer.npz', 'value-optimizer.npz', 'random.npz'):
            same_arrays(initial/name, actor_parent/name)
        for name in ('critic.safetensors', 'critic-target.safetensors'):
            assert tensors(initial/name) == tensors(actor_parent/name)
        new = tensors(initial/'model.safetensors')
        assert {k: v for k, v in old.items() if k.startswith('actor.')} == {
            k: v for k, v in new.items() if k.startswith('actor.')}
        world = tensors(ROOT/arm/'update-020000/world.safetensors')
        assert {k.removeprefix('world.'): v for k, v in new.items() if k.startswith('world.')} == world
        assert read(initial/'state.json')['sampling_rng'] == read(actor_parent/'state.json')['sampling_rng']
        assert sha(actor/'frozen-world-check.safetensors') == sha(ROOT/arm/'update-020000/world.safetensors')
    best = Path('results/defense/learned/best')
    manifest = read(best/'manifest.json')
    for name, digest in manifest['hashes'].items():
        assert sha(best/name) == digest
    report.update(initial_world_optimizer_and_rng_retained=True,
        identical_initial_byte_heads_and_sampling=True, frozen_world_and_optimizer_retained=True,
        initial_behavior_optimizer_and_rng_retained=True,
        global_best=dict(path=str(best.resolve()), score=manifest['result']['score']),
        limitations=['Reused validation seeds, not a fresh success-rate estimate.',
            'Observed-byte recognition is not future prediction or successful navigation.',
            'Frozen readout is diagnostic only; no policy training or promotion.',
            'Training uses only own visible trajectories, unchanged score reward and held-out assignments.'])
    (ROOT/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({arm: {k: v for k, v in report[arm].items()
        if k in ('evaluations', 'replay_score', 'verified_actions', 'uniform_forecast_brier')}
        for arm in ('control', 'bytes', 'actor-control', 'actor-bytes')}, indent=2))


if __name__ == '__main__':
    main()
