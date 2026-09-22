"""Keep every prior source and add the verified life-head actor trial."""
import hashlib
import json
import os
from pathlib import Path
import sys

destination = Path(__file__).with_name('collector-config.json')
long = '--long' in sys.argv
if long:
    destination = Path(__file__).with_name('collector-long-config.json')
if '--prepare' in sys.argv:
    previous = Path(__file__).with_name('collector-config.json') if long else Path(
        'results/defense/training/world-model-bytes-01/collector-config.json')
    old = json.loads(previous.read_text())
    sources = old['sources'] + ['runs/defense-imagination-life-head-long-11/artifacts' if long
                               else 'runs/defense-imagination-life-head-10/artifacts']
    assert len(set(sources)) == len(sources) and not destination.exists()
    config = dict(sources=sources, output=old['output'], run=old['run'], interval=old['interval'],
        previous_pid=44651 if long else 31529, previous_sources_preserved=True,
        new_sources='learned continuation-head actor, complete-game native verification required',
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
