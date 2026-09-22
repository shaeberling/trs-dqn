import hashlib
import json
from pathlib import Path
import subprocess

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
root = Path('results/defense/training/imagination-overshoot-comparison-01')
root.mkdir(exist_ok=False)
report = {}
for arm, name in [('uniform', 'defense-imagination-overshoot-control-06'), ('focused', 'defense-imagination-overshoot-four-07')]:
    source, dest = Path('runs')/name, root/arm
    events = [json.loads(l) for l in (source/'metrics.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['frozen_world_verified']
    subprocess.run(['cp', '-cR', str(source), str(dest)], check=True)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    assert hashes == {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    evaluations = []
    for checkpoint in sorted(dest.glob('update-*')):
        state = read(checkpoint/'state.json')
        for n, digest in state['hashes'].items(): assert sha(checkpoint/n) == digest
        evaluation = read(checkpoint/'evaluation.json')
        assert evaluation['complete_games'] == 10 and evaluation['highest_stage'] == 1
        assert not evaluation['incomplete_games'] and not evaluation['mission_games']
        evaluations.append(dict(actor_updates=state['actor_updates'], **{k:v for k,v in evaluation.items() if k!='games'}))
    manifest = read(dest/'artifacts/best/manifest.json')
    for n, digest in manifest['hashes'].items(): assert sha(dest/'artifacts/best'/n) == digest
    verified = read(dest/'artifacts/best/verification.json')
    assert verified['verified']
    report[arm] = dict(file_hashes=hashes, evaluations=evaluations,
        replay_score=manifest['result']['score'], verified_actions=verified['verified_actions'],
        frozen_world_sha256=sha(dest/'frozen-world-check.safetensors'))
report['arm_names'] = dict(uniform='original objective world', focused='four-step latent-overshoot world; NOT focused sampling')
report['limitations'] = 'Matched behavior continuations after two own-experience feedback cycles; reused complete boot games, not fresh success estimates; no shared-best replacement.'
(root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({arm: {k:v for k,v in report[arm].items() if k!='file_hashes'} for arm in ('uniform','focused')}, indent=2))
