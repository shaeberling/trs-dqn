"""Preserve all preceding replay sources and add focused ARS training."""

import json
import os
from pathlib import Path
import sys


destination = Path(__file__).with_name('collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/ars-59-head-search/collector-config.json').read_text())
    sources = old['sources']+['runs/defense-ars-focus-61-loss-snapshots/artifacts']
    assert len(set(sources)) == len(sources) and not destination.exists()
    config = dict(sources=sources, output=old['output'], run=old['run'],
                  interval=old['interval'], prior_sources_preserved=True,
                  new_source='focused neural head search on own pre-loss states')
    destination.write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources), config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable, '-u', '-m', 'rl.defense_collect']
    for source in config['sources']:
        command += ['--source', source]
    command += ['--output', config['output'], '--run', config['run'],
                '--interval', str(config['interval'])]
    os.execv(sys.executable, command)
