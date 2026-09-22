"""Preserve the completed direct-prior comparison and verify full-state lineage."""
import json
from pathlib import Path
import runpy
import subprocess

HELPER = Path('results/defense/training/world-model-bytes-01/archive-source.py')
helper = runpy.run_path(str(HELPER))
sha, read, tensors, same_arrays = (helper[name] for name in ('sha', 'read', 'tensors', 'same_arrays'))
ROOT = Path('results/defense/training/world-prior-output-01')


def copy_checked(run, name):
    source, dest = Path('runs')/run, ROOT/name
    events = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'finished'
    if not dest.exists():
        subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    for path in dest.glob('update-*/state.json'):
        for name, digest in read(path)['hashes'].items():
            assert sha(path.parent/name) == digest
    return dest, dict(file_hashes=hashes, finished=events[-1])


def main():
    assert not (ROOT/'comparison.json').exists()
    report = {}
    parent = Path('runs/defense-world-fit-10-byte-control/update-020000')
    actor_parent = Path('runs/defense-imagination-byte-control-08/update-002000')
    for arm, run in [('control', 'defense-world-fit-14-prior-control'),
                     ('prior', 'defense-world-fit-15-prior-output')]:
        dest, report[arm] = copy_checked(run, arm)
        assert report[arm]['finished']['updates'] == 22000
        initial, final = dest/'update-020000', dest/'update-022000'
        assert sha(initial/'world.safetensors') == sha(parent/'world.safetensors')
        for name in ('optimizer.npz', 'random.npz'):
            same_arrays(initial/name, parent/name)
        assert read(initial/'state.json')['sampling_rng'] == read(parent/'state.json')['sampling_rng']
        audit = read(final/'audit.json')
        report[arm].update(initial_full_world_state_retained=True,
            uniform_forecast_brier=sum(h['continuation_brier'] for h in audit['horizons'])/24,
            uniform_always_survive_brier=sum(h['always_continue_brier'] for h in audit['horizons'])/24,
            first_forecast=audit['horizons'][0], last_forecast=audit['horizons'][-1],
            boundary_predictions=read(final/'boundary-predictions.json')['predictions'],
            selected_loss_forecast=read(dest/'review-22000/report.json')['horizons'][15])
    same_arrays(ROOT/'control/update-022000/random.npz', ROOT/'prior/update-022000/random.npz')
    assert read(ROOT/'control/update-022000/state.json')['sampling_rng'] == read(
        ROOT/'prior/update-022000/state.json')['sampling_rng']
    def logged_targets(arm):
        events = [json.loads(line) for line in (ROOT/arm/'metrics.jsonl').read_text().splitlines()]
        return [(r['updates'], r['sampled_loss_targets']) for r in events if r['event'] == 'update']
    assert logged_targets('control') == logged_targets('prior')
    report['matching_final_rng_and_logged_training_targets'] = True
    for arm, run in [('actor-control', 'defense-imagination-prior-control-12'),
                     ('actor-prior', 'defense-imagination-prior-output-13')]:
        dest, report[arm] = copy_checked(run, arm)
        assert report[arm]['finished']['frozen_world_verified']
        assert report[arm]['finished']['actor_updates'] == 3000
        initial = dest/'update-002000'
        for name in ('actor-optimizer.npz', 'value-optimizer.npz', 'random.npz'):
            same_arrays(initial/name, actor_parent/name)
        for name in ('critic.safetensors', 'critic-target.safetensors'):
            assert tensors(initial/name) == tensors(actor_parent/name)
        old, new = tensors(actor_parent/'model.safetensors'), tensors(initial/'model.safetensors')
        assert {k: v for k, v in old.items() if k.startswith('actor.')} == {
            k: v for k, v in new.items() if k.startswith('actor.')}
        assert read(initial/'state.json')['sampling_rng'] == read(actor_parent/'state.json')['sampling_rng']
        world = ROOT/arm.removeprefix('actor-')/'update-022000/world.safetensors'
        assert {k.removeprefix('world.'): v for k, v in new.items() if k.startswith('world.')} == tensors(world)
        assert sha(dest/'frozen-world-check.safetensors') == sha(world)
        evaluations = []
        for path in sorted(dest.glob('update-*/state.json')):
            result = read(path.parent/'evaluation.json')
            assert result['complete_games'] == 10 and not result['incomplete_games']
            evaluations.append(dict(actor_updates=read(path)['actor_updates'],
                **{k: v for k, v in result.items() if k != 'games'}))
        best = dest/'artifacts/best'
        manifest, verification = read(best/'manifest.json'), read(best/'verification.json')
        for name, digest in manifest['hashes'].items():
            assert sha(best/name) == digest
        assert verification['verified']
        report[arm].update(initial_full_behavior_retained=True, evaluations=evaluations,
            replay_score=manifest['result']['score'], verified_actions=verification['verified_actions'])
    best = Path('results/defense/learned/best')
    manifest = read(best/'manifest.json')
    for name, digest in manifest['hashes'].items():
        assert sha(best/name) == digest
    report.update(global_best_score=manifest['result']['score'], helper_sha256=sha(HELPER),
        limitations=['Matched uniform own-data sampling and full parent Adam/RNG; only training objective differs.',
            'Prior outputs never receive the arrival screen; actual recorded actions condition predictions.',
            'Rotated-action sensitivity is not a true counterfactual outcome measurement.',
            'Reused validation seeds, not fresh success-rate estimates.',
            'No changed acting inputs, rewards, game binary or train/held-out assignments.'])
    (ROOT/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({arm: {k: v for k, v in report[arm].items()
        if k in ('uniform_forecast_brier', 'evaluations', 'replay_score', 'verified_actions')}
        for arm in ('control', 'prior', 'actor-control', 'actor-prior')}, indent=2))


if __name__ == '__main__':
    main()
