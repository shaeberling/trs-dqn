import unittest

import numpy as np

from rl.defense_ars_early_source import OFFSETS, visible_approaches


class EarlyOwnScreenSourceTests(unittest.TestCase):
    def test_four_life_windows_are_earlier_than_verified_visible_losses(self):
        frames = np.broadcast_to((np.arange(1521, dtype=np.uint16)%256)
                                 .astype(np.uint8)[:, None, None],
                                 (1521, 16, 64)).copy()
        events = [dict(frame=380*life, life_lost=True, score=2600*life)
                  for life in range(1, 5)]
        samples = visible_approaches(frames, events)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]['screens'].shape, (4, 4, 16, 64))
        self.assertEqual(int(samples[0]['screens'][0, -1, 0, 0]), (380-OFFSETS[0])%256)
        self.assertEqual(samples[-1]['visible_loss_frame'], 1520)
        with self.assertRaises(ValueError):
            visible_approaches(frames, events[:-1])
        with self.assertRaises(ValueError):
            visible_approaches(frames, [dict(frame=370, life_lost=True, score=2600)]+events[1:])


if __name__ == '__main__':
    unittest.main()
