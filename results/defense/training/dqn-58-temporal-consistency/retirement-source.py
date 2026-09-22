import gzip
import hashlib
import json
from pathlib import Path
import subprocess

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,parent in [('dqn-57-long-horizon-control','consistency-control-calibration-01'),
                    ('dqn-58-temporal-consistency','consistency-enabled-calibration-01')]:
    run=Path('runs')/('defense-'+name);root=Path('results/defense/training')/name
    raw=(run/'metrics.jsonl').read_bytes();events=[json.loads(l) for l in raw.splitlines()]
    assert events[-1]['event']=='stopped' and events[-1]['stop_requested']
    state=read(run/'latest/state.json');assert state['steps']==events[-1]['steps']
    dest=root/'final-checkpoint';assert not dest.exists()
    subprocess.run(['cp','-cR',str(run/'latest'),str(dest)],check=True)
    hashes={p.name:sha(p) for p in dest.iterdir()}
    assert hashes=={p.name:sha(p) for p in (run/'latest').iterdir()}
    packed=gzip.compress(raw,mtime=0);assert gzip.decompress(packed)==raw
    log=root/'metrics-complete.jsonl.gz';assert not log.exists();log.write_bytes(packed)
    rounds=[dict(steps=int(p.parent.name.split('-')[1]),evaluation=read(p)) for p in sorted(run.glob('step-*/evaluation.json'))]
    assert len(rounds)==6
    for r in rounds:
        assert r['evaluation']['complete_games']==10 and not r['evaluation']['incomplete_games']
        assert r['evaluation']['highest_stage']==1 and not r['evaluation']['mission_games']
    episodes=[e for e in events if e['event']=='episode']
    assert all(e['highest_stage']==1 and not e['missions_completed'] for e in episodes)
    assert all(e['full_game'] for e in episodes if e['worker']<2)
    start=read(Path('results/defense/training')/parent/'checkpoint/state.json')
    result=dict(final_steps=state['steps'],new_actions=state['steps']-start['steps'],
        new_boot_games=state['boot_episodes']-start['boot_episodes'],
        new_restored_segments=state['restored_segments']-start['restored_segments'],
        final_checkpoint_hashes=hashes,complete_log_sha256=hashlib.sha256(raw).hexdigest(),
        graceful_stop=events[-1],rounds=rounds,highest_logged_training_stage=1,
        reason='Six complete evaluation rounds without stage progression or improvement over the stronger preserved best; reallocating to learned dynamics and investigating failed life-loss forecasts. Not a wall-clock limit.',
        limitations='Reused complete boot seeds; original game completion remains unverified. Best replays retained.')
    p=root/'retirement.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rounds','final_checkpoint_hashes']},indent=2))

