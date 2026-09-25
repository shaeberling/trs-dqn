"""Pure reward-return arithmetic for the read-only Defense detour audit."""

import unittest

from rl.defense_score_detour_audit import discounted_return, required_next_reward


class ScoreDetourAuditTests(unittest.TestCase):
    def test_known_return_gap_and_next_reward_threshold(self):
        source = [0, 10, 0, 0]
        branch = [0, 0, 2]
        row = required_next_reward(source, 2, branch, 1, 0.5)
        self.assertEqual(row["source_known_return"], 10)
        self.assertEqual(row["branch_known_return"], 1)
        self.assertEqual(row["return_gap"], 9)
        self.assertEqual(row["hypothetical_next_action"], 4)
        self.assertEqual(row["next_reward_to_tie"], 36)
        self.assertEqual(required_next_reward(source, 2, branch, 1, 1)["next_reward_to_tie"], 8)

    def test_invalid_windows_and_discounts_rejected(self):
        for start, stop, discount in ((-1, 2, 1), (2, 1, 1), (0, 3, 1),
                                      (0, 2, 0), (0, 2, 1.1)):
            with self.subTest(start=start, stop=stop, discount=discount):
                with self.assertRaises(ValueError):
                    discounted_return([0, 1], start, stop, discount)
        with self.assertRaises(ValueError):
            required_next_reward([0, 1], 2, [0, 1], 2, 1)


if __name__ == "__main__":
    unittest.main()
