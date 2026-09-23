import unittest

import numpy as np

from rl.defense_ars_action_row_probe import summarize


class ActionRowProbeTests(unittest.TestCase):
    def test_uniform_command_probabilities_are_grouped_without_overlap(self):
        probabilities = np.full((3, 4, 20), 1 / 20)
        rows = summarize(probabilities, [128, 96, 64, 32])
        self.assertEqual([row['decisions_before_visible_loss'] for row in rows],
                         [128, 96, 64, 32])
        for row in rows:
            groups = row['mean_command_probability']
            self.assertAlmostEqual(groups['pure_RIGHT'], 1 / 20)
            self.assertAlmostEqual(groups['pure_rightward_movement'], 3 / 20)
            self.assertAlmostEqual(groups['pure_leftward_movement'], 3 / 20)
            self.assertAlmostEqual(groups['Space_containing'], 10 / 20)


if __name__ == '__main__':
    unittest.main()
