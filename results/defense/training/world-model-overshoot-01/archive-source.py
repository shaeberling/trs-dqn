import hashlib
import json
from pathlib import Path
import subprocess

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())

root = Path('results/defense/training/world-model-overshoot-01')
root.mkdir(exist_ok=True)
summary = {}
for arm, run_name in [('uniform', 'defense-world-fit-08-overshoot-control'),
                      ('focused', 'defense-world-fit-09-overshoot-four')]:
    source = Path('runs')/run_name
    events = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['updates'] == 18000
    dest = root/arm
    assert not dest.exists()
    subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    for checkpoint in dest.glob('update-*'):
        state = read(checkpoint/'state.json')
        for name, digest in state['hashes'].items():
            assert sha(checkpoint/name) == digest
    origin = read(source/'update-016000/state.json')
    final = read(source/'update-018000/state.json')
    audit = read(source/'update-018000/audit.json')
    boundaries = read(source/'update-018000/boundary-predictions.json')
    updates = [e for e in events if e['event'] == 'update']
    summary[arm] = dict(file_hashes=hashes, config=read(source/'config.json'),
        initial_hashes=origin['hashes'], final_hashes=final['hashes'],
        logged_sampled_loss_targets=sum(e['sampled_loss_targets'] for e in updates),
        logged_update_batches=len(updates), logged_target_count=len(updates)*8*24,
        uniform_forecast_brier=sum(r['continuation_brier'] for r in audit['horizons'])/24,
        uniform_always_survive_brier=sum(r['always_continue_brier'] for r in audit['horizons'])/24,
        boundary_predictions=boundaries['predictions'])
# Compare arrays by name, since NPZ key order is not optimizer state.
import numpy as np
for name in ('optimizer.npz', 'random.npz'):
    with np.load(root/'uniform/update-016000'/name) as a, np.load(root/'focused/update-016000'/name) as b:
        assert set(a.files) == set(b.files)
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
assert summary['uniform']['initial_hashes']['world.safetensors'] == summary['focused']['initial_hashes']['world.safetensors']
for arm in ('uniform', 'focused'):
    assert read(root/arm/'update-016000/state.json')['sampling_rng'] == read(root/'uniform/update-016000/state.json')['sampling_rng']
summary['arm_names'] = dict(uniform='original objective', focused='four-step latent overshooting, weight 1; NOT focused sampling')
summary['limitations'] = ['Both arms use identical uniform own-data sampling, not focused loss sampling.',
    'Only every twentieth training batch is logged; target counts are not totals over all updates.',
    'Frozen held-out forecast audit, not a playing-score result or a mission.',
    'No held-out/evaluation trajectories or oracle labels enter training.']
best = Path('results/defense/learned/best')
manifest = read(best/'manifest.json')
for name, digest in manifest['hashes'].items(): assert sha(best/name) == digest
summary['global_best_score'] = manifest['result']['score']
(root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({arm: {k:v for k,v in summary[arm].items() if k in ('uniform_forecast_brier', 'uniform_always_survive_brier', 'logged_sampled_loss_targets')} for arm in ('uniform', 'focused')}))
