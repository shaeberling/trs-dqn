import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_learning import sha256
from rl.defense_q_probe import comparison, discounted_returns, distribution_summary, probe


class QProbeTests(unittest.TestCase):
    def test_quantile_summary_keeps_indices_and_reports_counterfactual_choices(self):
        z = np.array([[0., 1., 2., 3.], [3., 2., 1., 0.]])
        actions, other = np.array([0, 1]), np.array([1, 1])
        weights = np.array([.1, .2, .3, .4])
        before = z.copy()
        result = distribution_summary(z, actions, other, weights, .01)
        self.assertEqual(result['selected_adjacent_crossing_fraction'], .5)
        self.assertEqual(result['counterfactual_different_actions'], 1)
        self.assertEqual(result['counterfactual_disagreement_fraction'], .5)
        self.assertEqual(result['counterfactual_different_stage_one_commands'], 1)
        self.assertEqual(result['low_fraction'], .125)
        self.assertEqual(result['high_fraction'], .875)
        first = distribution_summary(z[:1], actions[:1], other[:1], weights, .01)
        self.assertAlmostEqual(first['mean_high_minus_low_discounted_score'], 300.)
        self.assertAlmostEqual(first['mean_distorted_minus_mean_discounted_score'], 50.)
        np.testing.assert_array_equal(z, before)
        aliased = distribution_summary(z, np.array([9, 18]), np.array([17, 19]), weights, .01)
        self.assertEqual(aliased['counterfactual_different_actions'], 2)
        self.assertEqual(aliased['counterfactual_different_stage_one_commands'], 1)
        with self.assertRaises(ValueError):
            distribution_summary(z, np.array([-1, 20]), other, weights, .01)
        for bad in [np.empty((0, 4)), np.full((2, 4), np.nan), np.zeros(2)]:
            with self.assertRaises(ValueError): distribution_summary(bad, actions, other, weights, .01)
        for bad in [weights*2, np.array([np.nan, .2, .3, .4]), np.array([-.1, .3, .4, .4])]:
            with self.assertRaises(ValueError): distribution_summary(z, actions, other, bad, .01)

    def test_frozen_quantile_actions_distribution_and_source_immutability(self):
        source = Path('results/defense/training/quantile-risk-calibration-01/replay')
        before = {p.name: sha256(p) for p in source.iterdir()}
        report = probe(source)
        self.assertEqual(report['replay_actions_reproduced'], 2538)
        self.assertEqual(sum(x['life_score'] for x in report['lives']), 10240)
        distribution = report['quantile_distribution']
        self.assertTrue(distribution['counterfactual_only'])
        self.assertEqual(distribution['actual_training_power'], 1.5)
        self.assertEqual(distribution['whole_replay']['quantiles'], 32)
        self.assertGreater(distribution['whole_replay']['mean_high_minus_low_discounted_score'], 0)
        self.assertTrue(all(life['pre_marker_distribution']['observations'] == 64 for life in report['lives']))
        self.assertFalse(report['native_reexecution'])
        self.assertFalse(report['training_data_written'])
        self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})

    def test_discounted_returns_stop_at_visible_life_boundary(self):
        rewards = np.array([10., 20., 100., 200.])
        boundaries = np.array([False, True, False, True])
        before = rewards.copy()
        result = discounted_returns(rewards, boundaries, .5, .01)
        np.testing.assert_allclose(result, [.2, .2, 2., 2.])
        np.testing.assert_array_equal(rewards, before)
        whole = discounted_returns(rewards, np.array([False, False, False, True]), .5, .01)
        np.testing.assert_allclose(whole, [.7, 1.2, 2., 2.])
        stats = comparison(np.array([.3, .1]), result[:2], .01)
        self.assertEqual(stats['observations'], 2)
        self.assertAlmostEqual(stats['mean_prediction_minus_return'], 0)
        self.assertAlmostEqual(stats['mean_absolute_error'], 10)

    def test_invalid_or_incomplete_return_inputs(self):
        for rewards, boundaries, gamma, scale in [([], [], .9, .01),
                ([1], [False], .9, .01), ([1], [1], .9, .01),
                ([np.nan], [True], .9, .01), ([-1], [True], .9, .01),
                ([1], [True], 1., .01), ([1], [True], .9, 0.)]:
            with self.assertRaises(ValueError):
                discounted_returns(rewards, boundaries, gamma, scale)
        for predictions, returns in [([], []), ([1], [1, 2]), ([np.nan], [1])]:
            with self.assertRaises(ValueError): comparison(predictions, returns, .01)

    def test_preserved_dqn_actions_and_sources_remain_exact(self):
        # Fixed committed bundle, never a moving best link or training source.
        source = Path('results/defense/training/dqn-33-persistent-resets/first-replay')
        before = {p.name: sha256(p) for p in source.iterdir()}
        report = probe(source)
        self.assertEqual(report['replay_actions_reproduced'], 2564)
        self.assertEqual(len(report['lives']), 4)
        self.assertEqual(sum(x['life_score'] for x in report['lives']), 10480)
        self.assertEqual(report['whole_replay']['observations'], 2564)
        self.assertFalse(report['training_data_written'])
        self.assertFalse(report['native_reexecution'])
        self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'bundle'
            shutil.copytree(source, dest)
            (dest/'verification.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                probe(dest)


if __name__ == '__main__':
    unittest.main()
