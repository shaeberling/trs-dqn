import hashlib
import json
from pathlib import Path
import subprocess

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
root = Path('results/defense/training/world-model-feedback-01')
root.mkdir(exist_ok=False)
run = Path('runs/defense-world-fit-06-actor-feedback')
events = [json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
assert events[-1]['event'] == 'finished' and events[-1]['updates'] == 14000
hashes = {}
for source, dest in [(run, root/'fit'), (Path('runs/defense-world-data-feedback-01'), root/'data')]:
    subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    for path in source.rglob('*'):
        if path.is_file():
            digest = sha(path)
            assert sha(dest/path.relative_to(source)) == digest
            hashes[str((dest/path.relative_to(source)).relative_to(root))] = digest
for checkpoint in (root/'fit').glob('update-*'):
    state = json.loads((checkpoint/'state.json').read_text())
    for name, digest in state['hashes'].items(): assert sha(checkpoint/name) == digest
manifest = json.loads((root/'data/manifest.json').read_text())
assert len(manifest['episodes']) == 72 and manifest['heldout_games'] == 12
for row in manifest['episodes']: assert sha(root/'data'/row['file']) == row['sha256']
(root/'archive-audit.json').write_text(json.dumps(dict(file_hashes=hashes, dynamics_updates=14000,
    new_dynamics_updates=2000, no_evaluation_training_data=True), indent=2)+'\n')
print(json.dumps(dict(archived_files=len(hashes), dynamics_updates=14000)))
