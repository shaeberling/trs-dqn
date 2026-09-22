import hashlib
import json
from pathlib import Path
import subprocess

source = Path('runs/defense-world-actor-data-03')
dest = Path('results/defense/training/world-model-actor-collection-03')
manifest = json.loads((source/'manifest.json').read_text())
assert len(manifest['episodes']) == 24 and manifest['heldout_games'] == 4
assert manifest['existing_evaluation_data'] is False
assert not dest.exists()
subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
for row in manifest['episodes']:
    assert sha(dest/row['file']) == row['sha256']
    assert row['result']['game_over'] and row['result']['highest_stage'] == 1
    assert not row['result']['missions_completed']
summary = dict(file_hashes=hashes, splits={split: dict(
    games=sum(r['split']==split for r in manifest['episodes']),
    actions=sum(r['steps'] for r in manifest['episodes'] if r['split']==split),
    mean_score=sum(r['result']['score'] for r in manifest['episodes'] if r['split']==split)/sum(r['split']==split for r in manifest['episodes']))
    for split in ('train', 'heldout')}, training_updates=0,
    description='New own learned-actor experience; no action override or replay/evaluation imports.')
(dest/'archive-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary['splits']))
