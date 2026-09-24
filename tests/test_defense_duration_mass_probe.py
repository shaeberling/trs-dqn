"""Forensic-only visible-history reconstruction for duration mass audits."""

import unittest

import numpy as np

from rl.defense_duration_mass_probe import stacked_frames


class DurationMassProbeTests(unittest.TestCase):
    def test_visible_stride_history_and_boot_padding(self):
        frames = np.empty((8, 16, 64), np.uint8)
        for index in range(len(frames)):
            frames[index].fill(index)
        stacked = stacked_frames(frames, [0, 1, 5, 7], 2)
        np.testing.assert_array_equal(stacked[:, :, 0, 0],
                                      [[0, 0, 0, 0], [0, 0, 0, 1],
                                       [0, 1, 3, 5], [1, 3, 5, 7]])
        with self.assertRaises(ValueError):
            stacked_frames(frames, [8], 2)


if __name__ == "__main__":
    unittest.main()
