import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

root = Path('results/defense/training/world-model-preflight-01')
assert not (root/'archive-audit.json').exists()
source = Path('rl/defense_world_data.py').read_bytes()
metadata = json.loads(Path('runs/defense-world-data-01/manifest.json').read_text())
# Only the empty-dataset validation guard changed after collection started.
# Recover and verify the exact source bytes identified by that collection.
old_source = source.replace(b'if (n < 1 or frames.shape', b'if (frames.shape')
assert hashlib.sha256(old_source).hexdigest() == metadata['collection_source_sha256']
assert json.loads(Path('runs/defense-world-fit-01/metrics.jsonl').read_text().splitlines()[-1])['event'] == 'finished'
root.mkdir(exist_ok=True)
for name, path in [('data', 'runs/defense-world-data-01'), ('fit', 'runs/defense-world-fit-01'),
                   ('review-600', 'runs/defense-world-review-600'), ('review-1000', 'runs/defense-world-review-1000')]:
    if not (root/name).exists():
        subprocess.run(['cp', '-cR', path, str(root/name)], check=True)
    files = [p for p in Path(path).rglob('*') if p.is_file()]
    for original in files:
        assert sha(original) == sha(root/name/original.relative_to(path))
(root/'collection-source-v1.py').write_bytes(old_source)
assert sha(Path('runs/defense-world-fit-01/update-001000/world.safetensors')) == sha(Path('runs/defense-world-fit-02/update-001000/world.safetensors'))
for name in ('optimizer.npz', 'random.npz'):
    with np.load(Path('runs/defense-world-fit-01/update-001000')/name) as first, np.load(Path('runs/defense-world-fit-02/update-001000')/name) as second:
        assert set(first.files) == set(second.files)
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
assert json.loads(Path('runs/defense-world-fit-01/update-001000/audit.json').read_text()) == json.loads(Path('runs/defense-world-fit-02/update-001000/audit.json').read_text())
summary = dict(dataset_sha256=sha(root/'data/manifest.json'),
    copied_files_hash_checked=True, initial_resume_model_optimizer_rng_identical=True,
    initial_resume_forecast_audit_identical=True,
    resume_comparison='Exact weight bytes, optimizer/RNG arrays by key, and forecast JSON; archive serialization bytes need not match.',
    current_collection_module_sha256=sha('rl/defense_world_data.py'),
    archived_collection_source_sha256=sha(root/'collection-source-v1.py'),
    collection_source_note='Only an empty-dataset reader guard was added after collection started; exact original source recovered and SHA checked.',
    training_actions=sum(e['steps'] for e in metadata['episodes'] if e['split']=='train'),
    heldout_actions=sum(e['steps'] for e in metadata['episodes'] if e['split']=='heldout'),
    no_policy_training_or_promotion=True,
    file_hashes={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
(root/'archive-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='file_hashes'},indent=2))
