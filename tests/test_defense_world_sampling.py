import unittest

import numpy as np

from rl.defense_world_data import Sequences


def sequences(split='train'):
    dataset = Sequences.__new__(Sequences)
    dataset.length, dataset.split, dataset._boundary_pools = 4, split, {}
    dataset.episodes = []
    for n, losses in [(10, [0, 4, 9]), (6, [2, 3, 5])]:
        frames = np.broadcast_to(np.arange(n+1, dtype=np.uint8)[:, None, None], (n+1, 16, 64)).copy()
        cont = np.ones(n, np.float32); cont[losses] = 0
        dataset.episodes.append((frames, np.arange(n, dtype=np.int32), np.arange(n, dtype=np.float32), cont))
    dataset.counts = [7, 3]
    dataset.ends = np.cumsum(dataset.counts)
    return dataset


class BoundarySamplingTests(unittest.TestCase):
    def test_pool_matches_exhaustive_unique_windows(self):
        dataset = sequences()
        for burn in (0, 1, 3):
            expected, base = [], 0
            for episode, count in zip(dataset.episodes, dataset.counts):
                expected.extend(base+i for i in range(count) if np.any(episode[3][i+burn:i+4] == 0))
                base += count
            np.testing.assert_array_equal(dataset.boundary_pool(burn), expected)
        # Final transition is not omitted, and a loss only in burn-in is excluded.
        self.assertIn(6, dataset.boundary_pool(3))
        self.assertNotIn(0, dataset.boundary_pool(3))

    def test_uniform_path_retains_rng_and_exact_original_sampling(self):
        dataset = sequences()
        rng, reference = np.random.default_rng(43), np.random.default_rng(43)
        frames, actions, rewards, cont = dataset.sample(50, rng)
        selected = reference.integers(10, size=50)
        for i, index in enumerate(selected):
            episode, offset = (0, index) if index < 7 else (1, index-7)
            source = dataset.episodes[episode]
            np.testing.assert_array_equal(frames[i], source[0][offset:offset+5])
            np.testing.assert_array_equal(actions[i], source[1][offset:offset+4])
            np.testing.assert_array_equal(rewards[i], source[2][offset:offset+4]*.01)
            np.testing.assert_array_equal(cont[i], source[3][offset:offset+4])
        self.assertEqual(rng.bit_generator.state, reference.bit_generator.state)

    def test_focused_windows_alignment_resume_and_rejection(self):
        dataset = sequences()
        rng = np.random.default_rng(9)
        state = rng.bit_generator.state
        batch = dataset.sample(128, rng, 1., 3)
        self.assertTrue(np.all(batch[3][:, 3] == 0))
        np.testing.assert_array_equal(batch[0][:, :-1, 0, 0], batch[1])
        np.testing.assert_array_equal(batch[2], batch[1].astype(np.float32)*.01)
        other = np.random.default_rng(); other.bit_generator.state = state
        for left, right in zip(batch, dataset.sample(128, other, 1., 3)):
            np.testing.assert_array_equal(left, right)
        for fraction in (-1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                dataset.sample(1, rng, fraction)
        with self.assertRaises(ValueError):
            sequences('heldout').sample(1, rng, .5, 1)
        for episode in dataset.episodes:
            episode[3][:] = 1
        dataset._boundary_pools.clear()
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            dataset.sample(1, rng, .5, 1)


if __name__ == '__main__':
    unittest.main()
