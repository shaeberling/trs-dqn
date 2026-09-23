"""Retain all prior sources and watch both multi-radius key-factor runs."""

import json
import os
from pathlib import Path
import sys


destination = Path(__file__).with_name('collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/ars-key-74/collector-config.json').read_text())
    sources = old['sources'] + [
        'runs/defense-ars-key-wide-pilot-75/artifacts',
        'runs/defense-ars-key-multiscale-pilot-76/artifacts',
        'runs/defense-ars-key-multiscale-77/artifacts',
    ]
    assert len(set(sources)) == len(sources) and not destination.exists()
    destination.write_text(json.dumps(dict(sources=sources, output=old['output'],
        run=old['run'], interval=old['interval'], prior_sources_preserved=True,
        new_sources='multi-radius physical-key complete-boot search'), indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources), config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable, '-u', '-m', 'rl.defense_collect']
    for source in config['sources']:
        command += ['--source', source]
    command += ['--output', config['output'], '--run', config['run'],
                '--interval', str(config['interval'])]
    os.execv(sys.executable, command)
