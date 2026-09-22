import hashlib
import json
import os
from pathlib import Path
import sys

destination = Path('results/defense/training/world-model-bytes-01/collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/world-model-overshoot-01/collector-config.json').read_text())
    sources = old['sources'] + ['runs/'+name+'/artifacts' for name in (
        'defense-imagination-byte-control-08', 'defense-imagination-byte-reconstruction-09')]
    assert len(set(sources)) == len(sources) and not destination.exists()
    config = dict(sources=sources, output=old['output'], run=old['run'], interval=old['interval'],
        previous_pid=13565, previous_sources_preserved=True,
        new_sources='complete natively verified imagined-return actors, including future matched trials',
        policy_loader_sha256=hashlib.sha256(Path('rl/defense_learning.py').read_bytes()).hexdigest())
    destination.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable,'-u','-m','rl.defense_collect']
    for source in config['sources']: command += ['--source',source]
    command += ['--output',config['output'],'--run',config['run'],'--interval',str(config['interval'])]
    os.execv(sys.executable,command)
