import unittest

import numpy as np

from rl.defense_trajectory_search import COMMANDS, HOLDS, mutate_plan, outcome_rank


class DefenseTrajectorySearchTests(unittest.TestCase):
    def test_symmetric_contiguous_mutation_preserves_own_source(self):
        parent = np.arange(144, dtype=np.uint8) % 20
        original = parent.copy()
        rng = np.random.default_rng(547)
        sampled = set()
        for _ in range(100):
            plan, edits = mutate_plan(parent, rng)
            np.testing.assert_array_equal(parent, original)
            self.assertEqual(plan.shape, parent.shape)
            self.assertEqual(plan.dtype, np.uint8)
            self.assertTrue(1 <= len(edits) <= 4)
            for edit in edits:
                self.assertIn(edit["command"], COMMANDS)
                self.assertTrue(1 <= edit["length"] <= max(HOLDS))
                self.assertTrue(0 <= edit["start"] < len(parent))
                self.assertTrue(edit["start"]+edit["length"] <= len(parent))
                sampled.add(edit["command"])
        self.assertEqual(sampled, set(COMMANDS))
        with self.assertRaises(ValueError):
            mutate_plan(parent.astype(np.int32), rng)

    def test_real_score_precedes_visible_survival_tie_break(self):
        slow = dict(stage=1, score=2620, actions=144)
        early = dict(stage=1, score=2620, actions=119)
        higher_score = dict(stage=1, score=2640, actions=80)
        stage_two = dict(stage=2, score=100, actions=50)
        self.assertGreater(outcome_rank(slow), outcome_rank(early))
        self.assertGreater(outcome_rank(higher_score), outcome_rank(slow))
        self.assertGreater(outcome_rank(stage_two), outcome_rank(higher_score))


if __name__ == "__main__":
    unittest.main()
