import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from rl.defense_trajectory_search import COMMANDS, HOLDS, mutate_plan, outcome_rank, play


class DefenseTrajectorySearchTests(unittest.TestCase):
    def test_diversity_cell_is_recorded_only_at_a_surviving_visible_screen(self):
        class FakeEnv:
            count = 0

            def step(self, action):
                self.count += 1
                screen = np.zeros((4, 16, 64), np.uint8)
                screen[-1, 14, self.count] = 0xA8
                return (screen, 10, False, False,
                        dict(score=10*self.count, stage=1,
                             life_lost=self.count == 3, mission_completed=False))

        saved = SimpleNamespace(stage=1)
        initial = np.zeros((4, 16, 64), np.uint8)
        with patch("rl.defense_trajectory_search.restore", return_value=initial):
            early, frames, rewards = play(FakeEnv(), saved, np.zeros(4, np.uint8),
                                          trace=True, diversity_offset=2)
            loss = play(FakeEnv(), saved, np.zeros(4, np.uint8), diversity_offset=3)
        self.assertEqual(early["actions"], 3)
        self.assertEqual(early["score"], 30)
        self.assertIsInstance(early["checkpoint_cell"], str)
        self.assertEqual(len(early["checkpoint_cell"]), 32)
        self.assertIsNone(loss["checkpoint_cell"])
        self.assertEqual(frames.shape, (4, 16, 64))
        np.testing.assert_array_equal(rewards, [10., 10., 10.])

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
