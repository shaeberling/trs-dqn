import unittest

import numpy as np

from rl.defense_preposition_probe import MAX_HOLD, overlay_right


class PrepositionProbeTests(unittest.TestCase):
    def test_overlay_copies_and_clips_only_right_window(self):
        source = np.arange(20, dtype=np.uint8)
        changed = overlay_right(source, 280, 295, 12)
        np.testing.assert_array_equal(source, np.arange(20, dtype=np.uint8))
        np.testing.assert_array_equal(changed[:15], source[:15])
        np.testing.assert_array_equal(changed[15:], np.full(5, 4, np.uint8))

    def test_invalid_windows_rejected(self):
        source = np.zeros(20, np.uint8)
        for start, hold in ((279, 1), (300, 1), (280, 0), (280, MAX_HOLD+1)):
            with self.assertRaises(ValueError):
                overlay_right(source, 280, start, hold)
        with self.assertRaises(ValueError):
            overlay_right(source.astype(np.int32), 280, 280, 1)


if __name__ == "__main__":
    unittest.main()
