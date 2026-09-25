"""Pure selection checks for the opt-in forensic-only Defense probe."""

from types import SimpleNamespace
import unittest

import numpy as np

from rl.defense_position_feasibility import Node, select


def node(name, score, position):
    return Node(SimpleNamespace(native=name.encode()), name.encode(), score,
                position, name.encode())


class PositionFeasibilityTests(unittest.TestCase):
    def test_selection_respects_small_limit_and_mandatory_source(self):
        source = node("source", 200, 48)
        choices = [node("a", 300, 50), node("b", 100, 70)]
        self.assertEqual(select(choices, 1, np.random.default_rng(2)), [choices[0]])
        self.assertEqual(select(choices, 1, np.random.default_rng(2), source), [source])

    def test_selection_retains_score_and_position_frontiers(self):
        high_score = node("score", 1080, 44)
        high_position = node("position", 330, 70)
        other = node("other", 500, 52)
        chosen = select([high_score, high_position, other], 3, np.random.default_rng(5))
        self.assertEqual(len(chosen), 3)
        self.assertIn(high_score, chosen)
        self.assertIn(high_position, chosen)

    def test_identical_native_state_keeps_higher_score(self):
        lower = node("same", 300, 48)
        higher = node("same", 500, 50)
        chosen = select([lower, higher], 2, np.random.default_rng(7))
        self.assertEqual(chosen, [higher])


if __name__ == "__main__":
    unittest.main()
