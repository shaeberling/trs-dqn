"""Add current visual action-row search to all prior best-replay sources."""

import json
import os
from pathlib import Path
import sys


destination = Path(__file__).with_name('collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/ars-focus-66/collector-config.json').read_text())
    sources = old['sources'] + ['runs/defense-ars-focus-67-feature-row/artifacts']
    assert len(set(sources)) == len(sources) and not destination.exists()
    destination.write_text(json.dumps(dict(sources=sources, output=old['output'],
        run=old['run'], interval=old['interval'], prior_sources_preserved=True,
        new_source='visual feature-row action search'), indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources), config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable, '-u', '-m', 'rl.defense_collect']
    for source in config['sources']:
        command += ['--source', source]
    command += ['--output', config['output'], '--run', config['run'],
                '--interval', str(config['interval'])]
    os.execv(sys.executable, command)
