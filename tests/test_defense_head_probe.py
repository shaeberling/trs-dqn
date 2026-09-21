import unittest

import numpy as np

from rl.defense_head_probe import head_policy


class HeadProbeTests(unittest.TestCase):
    def test_fixed_head_not_mean_and_no_rng_draws(self):
        values = np.array([[[5., 0., 1.], [0., 7., 1.]],
                           [[0., 2., 9.], [8., 0., 1.]]])
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        rngs = [np.random.default_rng(3), np.random.default_rng(4)]
        before = [r.bit_generator.state for r in rngs]
        for head, expected in [(0, [0, 2]), (1, [1, 0])]:
            policy = head_policy(lambda x: values, head, 2)
            policy.reset_seed(10000)
            np.testing.assert_array_equal(policy(obs), expected)
            np.testing.assert_array_equal(policy.sample_with_rngs(obs, rngs), expected)
        self.assertEqual(before, [r.bit_generator.state for r in rngs])

    def test_rejects_wrong_head_shape_or_nonfinite_values(self):
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        for head in [-1, 2, True, .5]:
            with self.subTest(head=head), self.assertRaises(ValueError):
                head_policy(lambda x: None, head, 2)
        for values in [np.zeros((2, 3)), np.zeros((1, 2, 3)),
                       np.zeros((2, 3, 3)), np.full((2, 2, 3), np.nan)]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                head_policy(lambda x: values, 0, 2)(obs)


if __name__ == "__main__":
    unittest.main()
