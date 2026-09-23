"""Retain every earlier replay source and include long score-gated learning."""

import json
import os
from pathlib import Path
import sys


destination = Path(__file__).with_name('collector-config.json')
if '--prepare' in sys.argv:
    old = json.loads(Path('results/defense/training/ars-focus-67/collector-config.json').read_text())
    sources = old['sources'] + [
        'runs/defense-ars-focus-68-score-gate-pilot/artifacts',
        'runs/defense-ars-focus-69-score-gated/artifacts',
    ]
    assert len(set(sources)) == len(sources) and not destination.exists()
    destination.write_text(json.dumps(dict(sources=sources, output=old['output'],
        run=old['run'], interval=old['interval'], prior_sources_preserved=True,
        new_sources='score-gated visual action-row training'), indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources), config=str(destination))))
else:
    config = json.loads(destination.read_text())
    command = [sys.executable, '-u', '-m', 'rl.defense_collect']
    for source in config['sources']:
        command += ['--source', source]
    command += ['--output', config['output'], '--run', config['run'],
                '--interval', str(config['interval'])]
    os.execv(sys.executable, command)
