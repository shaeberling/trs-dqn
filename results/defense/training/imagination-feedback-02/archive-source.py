import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def tensors(path):
    raw = path.read_bytes(); length = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+length]); body = raw[8+length:]
    return {k: (v['dtype'], v['shape'], body[v['data_offsets'][0]:v['data_offsets'][1]])
            for k,v in header.items() if k != '__metadata__'}

source = Path('runs/defense-imagination-feedback-05')
dest = Path('results/defense/training/imagination-feedback-02')
events = [json.loads(l) for l in (source/'metrics.jsonl').read_text().splitlines()]
assert events[-1]['event'] == 'finished' and events[-1]['frozen_world_verified']
assert events[-1]['actor_updates'] == 2500 and not dest.exists()
subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
parent = Path('results/defense/training/imagination-feedback-01/update-001500')
start = dest/'update-001500'
for name in ('actor-optimizer.npz', 'value-optimizer.npz', 'random.npz'):
    with np.load(parent/name) as a, np.load(start/name) as b:
        assert set(a.files) == set(b.files)
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
for name in ('critic.safetensors', 'critic-target.safetensors'):
    assert tensors(parent/name) == tensors(start/name)
old, new = tensors(parent/'model.safetensors'), tensors(start/'model.safetensors')
assert {k:v for k,v in old.items() if k.startswith('actor.')} == {k:v for k,v in new.items() if k.startswith('actor.')}
assert any(old[k] != new[k] for k in old if k.startswith('world.'))
assert read(parent/'state.json')['sampling_rng'] == read(start/'state.json')['sampling_rng']
evaluations = []
for checkpoint in sorted(dest.glob('update-*')):
    state = read(checkpoint/'state.json')
    for name, digest in state['hashes'].items(): assert sha(checkpoint/name) == digest
    result = read(checkpoint/'evaluation.json')
    assert result['complete_games'] == 10 and result['highest_stage'] == 1 and not result['mission_games']
    evaluations.append(dict(actor_updates=state['actor_updates'], **{k:v for k,v in result.items() if k!='games'}))
manifest = read(dest/'artifacts/best/manifest.json')
for name, digest in manifest['hashes'].items(): assert sha(dest/'artifacts/best'/name) == digest
verified = read(dest/'artifacts/best/verification.json'); assert verified['verified']
result = dict(file_hashes=hashes, evaluations=evaluations, verified_actions=verified['verified_actions'],
    replay_score=manifest['result']['score'], all_behavior_state_retained_at_refresh=True,
    old_world_changed=True, parent_actor_updates=1500, additional_actor_updates=1000,
    final_world_frozen_sha256=sha(dest/'frozen-world-check.safetensors'))
(dest/'archive-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='file_hashes'}))
