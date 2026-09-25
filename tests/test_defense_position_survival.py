"""Pure guards for the forensic survival continuation's fixed schedule."""

import unittest

from rl.defense_position_survival import ANCHOR, SCHEDULE, SURVIVAL_FRAME, frame_numbers


class PositionSurvivalTests(unittest.TestCase):
    def test_schedule_hits_predeclared_survival_frame(self):
        numbered = frame_numbers()
        self.assertEqual(numbered[0], ANCHOR)
        self.assertEqual(numbered[-1], SURVIVAL_FRAME)
        self.assertIn(339, numbered)
        self.assertIn(391, numbered)
        self.assertTrue(all(b > a for a, b in zip(numbered, numbered[1:])))

    def test_bad_duration_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid diagnostic action hold"):
            frame_numbers((3,))

    def test_verification_frame_is_scheduled(self):
        self.assertIn(416, frame_numbers())


if __name__ == "__main__":
    unittest.main()
