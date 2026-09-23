import unittest

import numpy as np

from rl.defense import action_names
from rl.defense_ars_boot_search import (key_factor_directions, shared_seed_jobs,
                                        score_means, shortlist)


class CompleteBootSearchTests(unittest.TestCase):
    def test_shared_seeds_and_complete_score_matrix(self):
        jobs = shared_seed_jobs(4, 3, 100000)
        self.assertEqual(len(jobs), 12)
        for candidate in range(4):
            self.assertEqual([job['seed'] for job in jobs if job['candidate'] == candidate],
                             [100000, 100001, 100002])
        games = [dict(**job, score=100*job['candidate']+job['repetition']) for job in jobs]
        np.testing.assert_array_equal(score_means(games, 4, 3), [1, 101, 201, 301])
        with self.assertRaises(ValueError):
            score_means(games[:-1], 4, 3)
        with self.assertRaises(ValueError):
            score_means(games+[games[0]], 4, 3)

    def test_shortlist_uses_only_score_and_stable_ties(self):
        self.assertEqual(shortlist([100., 200., 200., 50.], 3), [1, 2, 0])
        with self.assertRaises(ValueError):
            shortlist([100., np.nan], 1)

    def test_key_factors_cover_every_physical_key_symmetrically(self):
        directions = key_factor_directions(np.random.default_rng(4), 7,
                                           (20, 257), action_names())
        self.assertEqual(directions.shape, (7, 20, 257))
        np.testing.assert_array_equal(directions[:, 0], 0)  # NOOP has no key.
        np.testing.assert_allclose(directions[:, 10], directions[:, 1]+directions[:, 9], atol=1e-6)
        np.testing.assert_allclose(directions[:, 18], directions[:, 3]+directions[:, 4], atol=1e-6)
        np.testing.assert_allclose(directions[:, 19],
                                   directions[:, 18]+directions[:, 9], atol=1e-6)
        self.assertTrue(np.any(directions[:, 9] != 0))
        with self.assertRaises(ValueError):
            key_factor_directions(np.random.default_rng(4), 7, (20, 257), ('NOOP',))


if __name__ == '__main__':
    unittest.main()
