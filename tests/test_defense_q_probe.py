import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.defense_learning import sha256
from rl.defense_q_probe import comparison, discounted_returns, probe


class QProbeTests(unittest.TestCase):
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
