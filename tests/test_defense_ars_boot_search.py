import unittest

import numpy as np

from rl.defense import action_names
from rl.defense_ars_boot_search import (candidate_scales, context_key_directions,
                                        key_factor_directions, shared_seed_jobs,
                                        score_means, shortlist, subspace_key_directions)


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

    def test_stratified_search_scales_preserve_both_radii(self):
        rng = np.random.default_rng(8)
        scales = candidate_scales(rng, 20, .012, .05)
        self.assertEqual(len(scales), 20)
        self.assertAlmostEqual(float(scales.min()), .012, places=6)
        self.assertAlmostEqual(float(scales.max()), .05, places=6)
        self.assertEqual(len(np.unique(scales)), 20)
        np.testing.assert_allclose(candidate_scales(rng, 4, .02), [.02]*4)
        with self.assertRaises(ValueError):
            candidate_scales(rng, 20, .02, .02)

    def test_contextual_key_proposals_retain_symmetric_command_structure(self):
        basis = np.linspace(-1, 1, 257, dtype=np.float32)
        directions = context_key_directions(np.random.default_rng(4), 7,
                                             (20, 257), action_names(), basis)
        self.assertEqual(directions.shape, (7, 20, 257))
        np.testing.assert_array_equal(directions[:, 0], 0)
        np.testing.assert_allclose(directions[:, 10], directions[:, 1]+directions[:, 9], atol=1e-6)
        np.testing.assert_allclose(directions[:, 19],
                                   directions[:, 3]+directions[:, 4]+directions[:, 9], atol=1e-6)
        with self.assertRaises(ValueError):
            context_key_directions(np.random.default_rng(4), 7,
                                   (20, 257), action_names(), np.ones(256))

    def test_visual_subspace_proposals_share_physical_keys_without_targets(self):
        basis = np.random.default_rng(8).normal(size=(5, 257)).astype(np.float32)
        directions = subspace_key_directions(np.random.default_rng(4), 7,
                                              (20, 257), action_names(), basis)
        self.assertEqual(directions.shape, (7, 20, 257))
        np.testing.assert_array_equal(directions[:, 0], 0)
        np.testing.assert_allclose(directions[:, 10], directions[:, 1]+directions[:, 9], atol=1e-6)
        np.testing.assert_allclose(directions[:, 19],
                                   directions[:, 3]+directions[:, 4]+directions[:, 9], atol=1e-6)
        self.assertTrue(np.isfinite(directions).all())
        with self.assertRaises(ValueError):
            subspace_key_directions(np.random.default_rng(4), 7,
                                     (20, 257), action_names(), np.ones((5, 256)))


if __name__ == '__main__':
    unittest.main()
