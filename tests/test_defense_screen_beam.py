import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_screen_beam import (ANCHOR, COMMANDS, SIDE_FIRE_COMMANDS, EARLY_ANCHOR,
                                    EARLY_SCHEDULE, HORIZON, SCHEDULE,
                                    Node, select_beam)


SOURCE = Path('results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay')


class DefenseScreenBeamTests(unittest.TestCase):
    def test_schedule_and_effective_commands(self):
        self.assertEqual(HORIZON, ANCHOR + sum(SCHEDULE))
        self.assertEqual(HORIZON, 428)
        self.assertEqual(EARLY_ANCHOR + sum(EARLY_SCHEDULE), HORIZON)
        self.assertEqual(set(COMMANDS), set(range(10)))
        self.assertEqual(set(SIDE_FIRE_COMMANDS), set(range(10)) | {18, 19})
        self.assertEqual(SCHEDULE[4:16], (1,) * 12)

    def test_selects_unique_visible_cells_and_preserves_source(self):
        nodes = [Node(None, bytes([i]), 10 + i, f'cell-{i}', i * 4)
                 for i in range(5)]
        nodes.append(Node(None, b'\x08', 100, 'cell-0', 0))
        mandatory = Node(None, b'\xff', 1, 'source-cell', None, True)
        selected = select_beam(nodes, 4, np.random.default_rng(1), mandatory=mandatory)
        self.assertEqual(len(selected), 4)
        self.assertIs(selected[0], mandatory)
        self.assertEqual(len({node.cell for node in selected}), len(selected))
        self.assertGreaterEqual(len({node.ship_bin for node in selected}), 3)
        with self.assertRaises(ValueError):
            select_beam(nodes, 0, np.random.default_rng(1))

    def test_history_key_retains_control_aliases_but_not_source_duplicate(self):
        mandatory = Node(None, b'\x01', 1, 'same', 20, True)
        nodes = [Node(None, b'\x02', 2, 'same', 20),
                 Node(None, b'\x01', 3, 'same', 20),
                 Node(None, b'\x03', 4, 'other', 24)]
        selected = select_beam(nodes, 3, np.random.default_rng(2),
                               mandatory=mandatory, history_key=True)
        self.assertEqual(len(selected), 3)
        self.assertIs(selected[0], mandatory)
        self.assertEqual({n.path for n in selected}, {b'\x01', b'\x02', b'\x03'})
        plain = select_beam(nodes, 3, np.random.default_rng(2), mandatory=mandatory)
        self.assertEqual(len(plain), 2)

    def test_native_exact_source_and_two_layer_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'beam'
            command = [sys.executable, '-m', 'rl.defense_screen_beam', str(SOURCE),
                       '--output', str(output), '--beam', '8', '--layers', '2']
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['completed_layers'], 2)
            self.assertEqual(report['latest_frame'], 328)
            self.assertEqual(report['config']['source_visible_loss'], 407)
            self.assertTrue(report['config']['diagnostic_only'])
            self.assertFalse(report['config']['promotion_eligible'])
            self.assertEqual(len((output / 'layers.jsonl').read_text().splitlines()), 2)

            side_output = Path(tmp) / 'side-fire'
            side_command = command[:]
            side_command[side_command.index(str(output))] = str(side_output)
            side_command.append('--side-fire')
            result = subprocess.run(side_command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            side_report = json.loads((side_output / 'report.json').read_text())
            self.assertEqual(side_report['config']['symmetric_commands'],
                             list(SIDE_FIRE_COMMANDS))
            self.assertTrue(side_report['config']['side_fire'])
            self.assertEqual(side_report['completed_layers'], 2)

            earlier_output = Path(tmp) / 'earlier'
            earlier_command = command[:]
            earlier_command[earlier_command.index(str(output))] = str(earlier_output)
            earlier_command.extend(('--anchor', '200', '--side-fire'))
            result = subprocess.run(earlier_command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            earlier_report = json.loads((earlier_output / 'report.json').read_text())
            self.assertEqual(earlier_report['config']['anchor'], 200)
            self.assertEqual(earlier_report['latest_frame'], 208)
            self.assertEqual(earlier_report['completed_layers'], 2)


if __name__ == '__main__':
    unittest.main()
