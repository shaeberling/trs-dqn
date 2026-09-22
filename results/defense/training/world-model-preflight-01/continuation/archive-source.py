import hashlib
import json
from pathlib import Path
import subprocess
import sys

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
step = int(sys.argv[1])
run = Path('runs/defense-world-fit-02')
root = Path('results/defense/training/world-model-preflight-01/continuation')
root.mkdir(exist_ok=True)
if not (root/'config.json').exists():
    subprocess.run(['cp', str(run/'config.json'), str(root/'config.json')], check=True)
assert sha(root/'config.json') == sha(run/'config.json')
checkpoint = f'update-{step:06d}'
assert not (root/checkpoint).exists()
parts=[]
for line in (run/'metrics.jsonl').read_bytes().splitlines(keepends=True):
    if not line.endswith(b'\n'): break
    event=json.loads(line);parts.append(line)
    if event['event']=='audit' and event['updates']==step: break
assert event['event']=='audit' and event['updates']==step
for source,destination in [(run/checkpoint,root/checkpoint),
        (Path(f'runs/defense-world-review-{step}'),root/f'review-{step}')]:
    assert source.is_dir() and not destination.exists()
    subprocess.run(['cp','-cR',str(source),str(destination)],check=True)
    for p in source.rglob('*'):
        if p.is_file(): assert sha(p)==sha(destination/p.relative_to(source))
log=root/f'metrics-through-{step:06d}.jsonl'
assert not log.exists();log.write_bytes(b''.join(parts))
state=json.loads((root/checkpoint/'state.json').read_text())
for name,digest in state['hashes'].items(): assert sha(root/checkpoint/name)==digest
best=Path('results/defense/learned/best')
manifest=json.loads((best/'manifest.json').read_text())
for name,digest in manifest['hashes'].items(): assert sha(best/name)==digest
print(json.dumps(dict(updates=step,checkpoint_hashes_checked=True,global_best_hashes_checked=True,
                     global_best_score=manifest['result']['score'])))
