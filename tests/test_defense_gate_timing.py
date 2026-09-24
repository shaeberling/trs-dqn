import argparse
import unittest

import numpy as np

from rl.defense_gate_timing import parse_pair, visible_ship_column, visible_wall_runs


class DefenseGateTimingTests(unittest.TestCase):
    def test_visible_geometry_extracts_only_recorded_glyphs(self):
        screen = np.full((16, 64), 0x80, np.uint8)
        screen[14, 24] = 0xa8
        screen[9, 1:21] = 0xb0
        screen[9, 31:63] = 0x8c
        screen[9, 26] = 0x97  # projectile is not treated as wall
        self.assertEqual(visible_ship_column(screen), 24)
        self.assertEqual(visible_wall_runs(screen, 9), [[1, 20], [31, 62]])
        screen[14, 25] = 0xa8
        self.assertIsNone(visible_ship_column(screen))
        screen[14, 24:26] = 0x80
        self.assertIsNone(visible_ship_column(screen))

    def test_rejects_non_screen_geometry_and_bad_pairs(self):
        frame = np.full((16, 64), 0x80, np.uint8)
        for invalid in (np.zeros((16, 63), np.uint8), frame.astype(np.float32)):
            with self.assertRaises(ValueError):
                visible_ship_column(invalid)
            with self.assertRaises(ValueError):
                visible_wall_runs(invalid, 9)
        for row in (0, 16, -1):
            with self.assertRaises(ValueError):
                visible_wall_runs(frame, row)
        self.assertEqual(parse_pair("341:388"), (341, 388))
        for invalid in ("341", "341:388:10", "a:b"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_pair(invalid)


if __name__ == "__main__":
    unittest.main()
