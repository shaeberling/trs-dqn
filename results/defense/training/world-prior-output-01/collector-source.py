"""Preserve all existing sources and watch both direct-prior actor trials."""
import hashlib
import json
import os
from pathlib import Path
import sys

destination = Path(__file__).with_name('collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/world-life-readout-01/collector-long-config.json').read_text())
    sources = old['sources'] + ['runs/'+name+'/artifacts' for name in (
        'defense-imagination-prior-control-12', 'defense-imagination-prior-output-13')]
    assert len(set(sources)) == len(sources) and not destination.exists()
    config = dict(sources=sources, output=old['output'], run=old['run'], interval=old['interval'],
        previous_pid=46861, previous_sources_preserved=True,
        new_sources='matched learned direct-prior actors; complete-game native verification required',
        policy_loader_sha256=hashlib.sha256(Path('rl/defense_learning.py').read_bytes()).hexdigest())
    destination.write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources), config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable, '-u', '-m', 'rl.defense_collect']
    for source in config['sources']:
        command += ['--source', source]
    command += ['--output', config['output'], '--run', config['run'], '--interval', str(config['interval'])]
    os.execv(sys.executable, command)
