import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_early_phase_reach import (ANCHOR, WAYPOINT, HORIZON, DELAYS,
                                         HOLDS, COMMANDS, RIGHT, candidate_plan)
from rl.defense_phase_grid import grid


SOURCE = Path('results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay')


class DefenseEarlyPhaseReachTests(unittest.TestCase):
    def test_fixed_grid_and_early_two_phase_right_suffix(self):
        choices = list(grid(DELAYS, HOLDS, COMMANDS))
        self.assertEqual(len(choices), 3600)
        self.assertEqual(len(set(choices)), 3600)
        self.assertEqual(set(COMMANDS), set(range(10)))
        baseline = np.arange(HORIZON-ANCHOR, dtype=np.uint8)
        choice = (16, 24, 6, 24, 3)
        plan = candidate_plan(baseline, choice)
        np.testing.assert_array_equal(plan[:16], baseline[:16])
        np.testing.assert_array_equal(plan[16:40], np.full(24, 6, np.uint8))
        np.testing.assert_array_equal(plan[40:64], np.full(24, 3, np.uint8))
        np.testing.assert_array_equal(plan[64:WAYPOINT-ANCHOR], baseline[64:WAYPOINT-ANCHOR])
        np.testing.assert_array_equal(plan[WAYPOINT-ANCHOR:],
                                      np.full(HORIZON-WAYPOINT, RIGHT, np.uint8))
        np.testing.assert_array_equal(baseline, np.arange(HORIZON-ANCHOR, dtype=np.uint8))
        with self.assertRaises(ValueError):
            candidate_plan(baseline, (24, 32, 6, 24, 3))

    def test_native_verified_source_and_bounded_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'reach'
            command = [sys.executable, '-m', 'rl.defense_early_phase_reach',
                       str(SOURCE), '--output', str(output), '--limit', '10']
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['config']['baseline']['score'], 2620)
            self.assertEqual(report['config']['baseline']['absolute_frame'], 407)
            self.assertTrue(report['config']['baseline']['life_lost'])
            self.assertEqual(report['config']['total_candidates'], 3600)
            self.assertEqual(report['completed'], 10)
            self.assertEqual(len((output/'outcomes.jsonl').read_text().splitlines()), 10)
            self.assertEqual(report['config']['source_score_at_waypoint'], 2600)


if __name__ == '__main__':
    unittest.main()
