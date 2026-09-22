import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

before=Path('runs/defense-option-learning-parity-before')
after=Path('runs/defense-option-learning-parity-after')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a,b=read(before/'latest/state.json'),read(after/'latest/state.json')
assert a['steps']==b['steps']==147456
assert {k:v for k,v in a.items() if k!='config'}=={k:v for k,v in b.items() if k!='config'}
assert sha(before/'latest/model.safetensors')==sha(after/'latest/model.safetensors')
assert sha(before/'latest/target.safetensors')==sha(after/'latest/target.safetensors')
parent=read(Path('results/defense/training/repeat-variable-calibration-01/checkpoint/state.json'))
assert b['updates']>parent['updates']
with np.load(before/'latest/optimizer.npz') as x,np.load(after/'latest/optimizer.npz') as y:
    assert x.files==y.files
    for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    count=len(x.files)
def events(run):
    result=[]
    for line in (run/'metrics.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['event'] in ('episode','curriculum_archive'):
            e.pop('wall_seconds');result.append(e)
    return result
assert events(before)==events(after)
old=subprocess.check_output(['/Library/Developer/CommandLineTools/usr/bin/git','show','aac051f:rl/defense_dqn.py'])
report=dict(before=str(before),after=str(after),new_base_actions=16384,
    new_updates=b['updates']-parent['updates'],
    model_sha256=sha(after/'latest/model.safetensors'),target_sha256=sha(after/'latest/target.safetensors'),
    optimizer_arrays_equal=count,all_nonconfiguration_state_equal=True,
    equal_game_and_archive_events=len(events(after)),native_sha256=sha(Path('libtrs.so')),
    original_trainer_commit='aac051f',original_executed_trainer_sha256=hashlib.sha256(old).hexdigest(),
    new_trainer_sha256=sha(Path('rl/defense_dqn.py')),
    limits=['Deterministic disabled-feature check with optimizer updates; not multi-option performance.',
            'Before trainer executes git-show source with original package/file globals. Its recorded trainer_source_sha256 reads the current file, so the execution source hash is recorded separately here.'])
output=Path('results/defense/diagnostics/multi-option-default-parity.json')
assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
