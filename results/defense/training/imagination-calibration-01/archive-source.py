import hashlib
import json
from pathlib import Path
import subprocess

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def clone(source,destination):
    source,destination=Path(source),Path(destination)
    assert source.is_dir() and not destination.exists()
    subprocess.run(['cp','-cR',str(source),str(destination)],check=True)
    for path in source.rglob('*'):
        if path.is_file(): assert sha(path)==sha(destination/path.relative_to(source))

run=Path('runs/defense-imagination-calibration-01')
last=json.loads((run/'metrics.jsonl').read_text().splitlines()[-1])
assert last['event']=='finished' and last['frozen_world_verified'] and last['actor_updates']==1000
root=Path('results/defense/training/imagination-calibration-01')
clone(run,root)
versions=list((root/'artifacts/versions').iterdir());assert len(versions)==1
manifest=read(versions[0]/'manifest.json')
for name,digest in manifest['hashes'].items(): assert sha(versions[0]/name)==digest
verification=read(versions[0]/'verification.json');assert verification['verified']
assert verification['verified_actions']==1571 and manifest['result']['score']==320
for step in (0,500,1000):
    state=read(root/f'update-{step:06d}/state.json')
    for name,digest in state['hashes'].items(): assert sha(root/f'update-{step:06d}'/name)==digest
    evaluation=read(root/f'update-{step:06d}/evaluation.json')
    assert evaluation['complete_games']==10 and evaluation['incomplete_games']==0
    assert evaluation['highest_stage']==1 and evaluation['mission_games']==0
assert sha(root/'frozen-world-check.safetensors')==read(root/'config.json')['world_sha256']

mixed=Path('results/defense/training/world-model-mixed-01');assert not mixed.exists();mixed.mkdir()
clone('runs/defense-world-data-merged-01',mixed/'data')
for number in (5000,6000):
    clone(f'runs/defense-world-fit-03-mixed/update-{number:06d}',mixed/f'update-{number:06d}')
clone('runs/defense-world-mixed-review-6000',mixed/'review-6000')
subprocess.run(['cp','runs/defense-world-fit-03-mixed/config.json',str(mixed/'config.json')],check=True)
parts=[]
for line in Path('runs/defense-world-fit-03-mixed/metrics.jsonl').read_bytes().splitlines(keepends=True):
    if not line.endswith(b'\n'): break
    e=json.loads(line);parts.append(line)
    if e['event']=='audit' and e['updates']==6000: break
assert e['event']=='audit' and e['updates']==6000
(mixed/'metrics-through-006000.jsonl').write_bytes(b''.join(parts))
for name in ('world.safetensors','optimizer.npz','random.npz'):
    # Model bytes match; optimizer archives may reorder keys when restored.
    if name=='world.safetensors':
        assert sha(mixed/'update-005000'/name)==sha(Path('runs/defense-world-fit-02/update-005000')/name)
    else:
        import numpy as np
        with np.load(mixed/'update-005000'/name) as a,np.load(Path('runs/defense-world-fit-02/update-005000')/name) as b:
            assert set(a.files)==set(b.files)
            for k in a.files: np.testing.assert_array_equal(a[k],b[k])
complete=Path('runs/defense-world-fit-02/metrics.jsonl').read_bytes()
assert json.loads(complete.splitlines()[-1])['event']=='finished'
destination=Path('results/defense/training/world-model-preflight-01/continuation/metrics-complete.jsonl')
assert not destination.exists();destination.write_bytes(complete)
for destination in (root,mixed):
    hashes={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    (destination/'archive-audit.json').write_text(json.dumps(dict(all_files_hash_checked=True,
        frozen_world_or_initial_resume_verified=True,file_hashes=hashes),indent=2)+'\n')
print(json.dumps(dict(actor_complete=True,verified_actions=verification['verified_actions'],
                     actor_best=manifest['result']['score'],mixed_checkpoint=6000)))
