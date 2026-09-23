import unittest

import numpy as np

from rl.defense_ars_boot_search import shared_seed_jobs, score_means, shortlist


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


if __name__ == '__main__':
    unittest.main()
