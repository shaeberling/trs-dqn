import unittest

import numpy as np

from rl.defense_macro_explore import (BRANCHES, HOLDS, LONG_HOLDS,
                                      branches_for_lookback, command_ids,
                                      draw_plan, hold_lengths)


class MacroExploreTests(unittest.TestCase):
    def test_symmetric_training_only_command_sets(self):
        self.assertEqual(command_ids('all'), tuple(range(20)))
        self.assertEqual(command_ids('effective-stage-one'), tuple(range(10)))
        with self.assertRaises(ValueError):
            command_ids('RIGHT')

    def test_random_plan_stays_within_declared_schedule_and_commands(self):
        for mode in ('all', 'effective-stage-one'):
            rng = np.random.default_rng(42)
            observed = set()
            for _ in range(500):
                branch, segments = draw_plan(rng, mode)
                self.assertIn(branch, BRANCHES)
                self.assertEqual(len(segments), 2)
                for action, hold in segments:
                    self.assertIn(action, command_ids(mode))
                    self.assertIn(hold, HOLDS)
                    observed.add(action)
            self.assertEqual(observed, set(command_ids(mode)))

    def test_earlier_own_life_branch_positions_remain_symmetric(self):
        self.assertEqual(branches_for_lookback(128), BRANCHES)
        self.assertEqual(branches_for_lookback(370), (0, 92, 185, 277))
        for invalid in (0, 127, True, 128.0):
            with self.assertRaises(ValueError):
                branches_for_lookback(invalid)
        rng = np.random.default_rng(7)
        for _ in range(30):
            self.assertIn(draw_plan(rng, 'all', 370)[0], (0, 92, 185, 277))

    def test_long_holds_do_not_prefer_a_direction(self):
        self.assertEqual(hold_lengths('short'), HOLDS)
        self.assertEqual(hold_lengths('long'), LONG_HOLDS)
        with self.assertRaises(ValueError):
            hold_lengths('RIGHT')
        rng = np.random.default_rng(8)
        actions, durations = set(), set()
        for _ in range(500):
            _, plan = draw_plan(rng, 'effective-stage-one', 370, 'long')
            actions.update(action for action, _ in plan)
            durations.update(hold for _, hold in plan)
        self.assertEqual(actions, set(range(10)))
        self.assertEqual(durations, set(LONG_HOLDS))


if __name__ == '__main__':
    unittest.main()
