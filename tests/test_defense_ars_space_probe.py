import unittest

import numpy as np

from rl.defense import action_names
from rl.defense_ars_space_probe import space_bias_heads


class SpaceActionAblationTests(unittest.TestCase):
    def test_only_space_command_biases_change(self):
        head = np.arange(20*257, dtype=np.float32).reshape(20, 257)
        heads = space_bias_heads(head, action_names(), (0., -2., -100.))
        self.assertEqual(heads.shape, (3, 20, 257))
        np.testing.assert_array_equal(heads[0], head)
        np.testing.assert_array_equal(heads[:, :9], np.broadcast_to(head[:9], (3, 9, 257)))
        np.testing.assert_array_equal(heads[:, 18], np.broadcast_to(head[18], (3, 257)))
        np.testing.assert_allclose(heads[:, 9, -1]-head[9, -1], [0, -2, -100])
        np.testing.assert_allclose(heads[:, 19, -1]-head[19, -1], [0, -2, -100])
        np.testing.assert_array_equal(heads[:, 9, :-1],
                                      np.broadcast_to(head[9, :-1], (3, 256)))
        with self.assertRaises(ValueError):
            space_bias_heads(head, action_names(), (-1.,))


if __name__ == '__main__':
    unittest.main()
