from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.defense import DefenseEnv
from rl.defense_ars import head_array, make_population_infer
from rl.defense_ars_focus import (harvest, load_source, perturbation_directions,
                                  play_segment, score_gated_choice, verify_source)
from rl.model import QNetwork


class FocusedARSTests(unittest.TestCase):
    def test_score_gate_uses_only_paired_training_score_and_rejects_ties(self):
        returns = np.array([[[130., 130.], [90., 90.]],
                            [[100., 100.], [110., 110.]]])
        self.assertEqual(score_gated_choice(returns, np.array([100., 100.]), 10.),
                         (0, 100., 130.))
        self.assertEqual(score_gated_choice(returns, np.array([120., 120.]), 20.)[0], None)
        self.assertEqual(score_gated_choice(returns+500, np.array([600., 600.]), 10.)[0], 0)
        with self.assertRaises(ValueError):
            score_gated_choice(returns, np.array([100.]), 10.)

    def test_bias_only_directions_preserve_feature_weights(self):
        rng = np.random.default_rng(10)
        directions = perturbation_directions(rng, 7, (20, 257), bias_only=True)
        self.assertEqual(directions.shape, (7, 20, 257))
        self.assertTrue(np.all(directions[:, :, :-1] == 0))
        self.assertTrue(np.any(directions[:, :, -1] != 0))
        basis = perturbation_directions(rng, 20, (20, 257), coordinate_bias=True)
        self.assertEqual(int(np.count_nonzero(basis)), 20)
        np.testing.assert_array_equal(basis[:, :, :-1], 0)
        np.testing.assert_array_equal(basis[:, :, -1].sum(axis=0), np.ones(20))
        np.testing.assert_array_equal(basis[:, :, -1].sum(axis=1), np.ones(20))
        with self.assertRaises(ValueError):
            perturbation_directions(rng, 16, (20, 257), coordinate_bias=True)
        rows = perturbation_directions(rng, 20, (20, 257), coordinate_row=True)
        nonzero_rows = np.any(rows != 0, axis=2)
        np.testing.assert_array_equal(nonzero_rows.sum(axis=0), np.ones(20))
        np.testing.assert_array_equal(nonzero_rows.sum(axis=1), np.ones(20))
        self.assertTrue(np.all(np.count_nonzero(rows, axis=(1, 2)) > 100))
        with self.assertRaises(ValueError):
            perturbation_directions(rng, 20, (20, 257), bias_only=True, coordinate_row=True)

    def test_native_own_loss_state_restore_and_paired_segment_identity(self):
        mx.random.seed(41)
        model = QNetwork(action_count=20)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)/'own-loss-states'
            events = []
            names, games, steps = harvest(model, archive, count=1, first_seed=73000,
                lookback=128, tstates=100000, observation_stride=1, log=events.append)
            self.assertEqual(len(games), 1)
            self.assertTrue(games[0]['terminated'])
            self.assertEqual(steps, games[0]['steps'])
            self.assertGreaterEqual(len(names), 1)
            item = load_source(archive/names[0])
            saved, actions, rewards, screens, record = item
            self.assertEqual(len(actions), 128)
            self.assertEqual(screens.shape, (128, 16, 64))
            self.assertEqual(record['source']['snapshot_action']+128,
                             record['source']['visible_loss_frame'])
            self.assertTrue(record['source']['training_only'])
            env = DefenseEnv(tstates=100000, max_steps=0)
            try:
                self.assertEqual(verify_source(env, item), 128)
                infer = make_population_infer(model, head_array(model)[None])
                first, trace_a = play_segment(env, saved, infer, 0, 84000, keep_trace=True)
                second, trace_b = play_segment(env, saved, infer, 0, 84000, keep_trace=True)
                self.assertEqual(first, second)
                for a, b in zip(trace_a, trace_b, strict=True):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(float(trace_a[2].sum()), first['score_gain'])
                self.assertTrue(first['life_lost'] or first['stage'] > saved.stage
                                or first['mission_completed'])
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
