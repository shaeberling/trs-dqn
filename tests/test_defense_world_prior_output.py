from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_bytes import ByteWorldLearner
from rl.defense_world_model import WorldLearner, WorldModel
from rl.defense_world_prior_output import predictions, prior_output_loss


class PriorOutputTests(unittest.TestCase):
    def test_actions_targets_and_burn_alignment(self):
        class Toy:
            def step(self, state, action, key, sample=True):
                self.actions = np.array(action)
                value = state[0]+action[:, None]
                return (value, value, value, mx.ones_like(value)), (value, mx.ones_like(value))
            def features(self, state): return state[0]
            def decode(self, x): return x[..., None, None]
            def pixels(self, x): return x[..., None]
            def reward(self, x): return x/10
            def continue_logit(self, x): return mx.zeros_like(x)
        actions = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.int32)
        cumulative = np.concatenate([np.zeros((2, 1)), np.cumsum(actions, axis=1)], axis=1)
        frames = mx.array(cumulative[..., None, None])
        states = tuple(mx.array(v) for v in (cumulative[..., None],)*3+(np.ones((2, 5, 1)),))
        rewards = mx.array(cumulative[:, 1:]/10)
        toy = Toy()
        terms = prior_output_loss(toy, states, frames, mx.array(actions), rewards, mx.ones((2, 4)), mx.random.key(0), 1)
        np.testing.assert_array_equal(toy.actions, actions[:, 1:].reshape(-1))
        np.testing.assert_allclose(np.array(terms), [0., 0., np.log(2)], atol=1e-6)
        changed = prior_output_loss(toy, states, frames, mx.array(actions), rewards+1, mx.ones((2, 4)), mx.random.key(0), 1)
        self.assertAlmostEqual(float(changed[1]), .5)
        with self.assertRaises(ValueError):
            predictions(toy, states, mx.array(actions), mx.random.key(0), 4)

    def test_prior_predictions_ignore_their_arrival_screen(self):
        mx.random.seed(3)
        model = WorldModel()
        rng = np.random.default_rng(3)
        f = rng.integers(128, 192, (2, 6, 16, 64), dtype=np.uint8)
        actions = mx.array(rng.integers(20, size=(2, 5), dtype=np.int32))
        states, _ = model.observe(mx.array(f), actions, mx.random.key(0), sample=False)
        before = [np.array(v) for v in predictions(model, states, actions, mx.random.key(0), 1, False)]
        f[:, 2:] = 32
        states, _ = model.observe(mx.array(f), actions, mx.random.key(0), sample=False)
        after = [np.array(v) for v in predictions(model, states, actions, mx.random.key(0), 1, False)]
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a[:, 0], b[:, 0])

    def test_extra_gradient_reaches_prior_and_outputs_not_filtered_targets(self):
        mx.random.seed(7)
        model = WorldModel()
        frames = mx.full((1, 5, 16, 64), 191, mx.uint8)
        actions = mx.array([[1, 2, 3, 4]])
        def objective(m):
            states, _ = m.observe(frames, actions, mx.random.key(0))
            return mx.sum(prior_output_loss(m, states, frames, actions, mx.ones((1, 4)),
                                           mx.zeros((1, 4)), mx.random.key(1), 1))
        loss, grads = nn.value_and_grad(model, objective)(model)
        self.assertTrue(np.isfinite(float(loss)))
        flat = dict(tree_flatten(grads))
        for prefix in ('prior.', 'memory.', 'action_input.', 'decoder.', 'reward.', 'continue_logit.'):
            self.assertTrue(any(np.any(np.array(v) != 0) for k, v in flat.items() if k.startswith(prefix)))
        for key, value in flat.items():
            if key.startswith(('encoder.', 'embedding.', 'posterior.')):
                np.testing.assert_array_equal(np.array(value), 0)

    def test_original_terms_and_exact_full_resume(self):
        rng = np.random.default_rng(6)
        batch = (rng.integers(128, 192, (1, 5, 16, 64), dtype=np.uint8),
                 np.array([[1, 2, 3, 4]], np.int32), np.zeros((1, 4), np.float32),
                 np.array([[1., 1., 1., 0.]], np.float32))
        learner = WorldLearner(seed=8, burn=1, prior_output_weight=.1)
        args = [mx.array(v) for v in batch]+[mx.random.key(4)]
        original, old_terms = learner.model.loss(*args, burn=1)
        extra, terms = learner.model.loss(*args, burn=1, prior_output_weight=.1)
        np.testing.assert_array_equal(np.array(old_terms), np.array(terms[:4]))
        self.assertAlmostEqual(float(extra), float(original+.1*mx.sum(terms[4:])), places=5)
        self.assertIn('prior_image', learner.train(batch))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint'
            learner.save(path, rng, dict(prior_outputs=learner.prior_outputs))
            other = WorldLearner(seed=11, burn=1, prior_output_weight=.1)
            other_rng = np.random.default_rng(11)
            other.restore(path, other_rng)
            self.assertEqual(rng.bit_generator.state, other_rng.bit_generator.state)
            self.assertEqual(learner.train(batch), other.train(batch))
            for a, b in ((learner.model.parameters(), other.model.parameters()),
                         (learner.optimizer.state, other.optimizer.state), (learner.random, other.random)):
                x, y = dict(tree_flatten(a)), dict(tree_flatten(b))
                self.assertEqual(x.keys(), y.keys())
                for key in x:
                    np.testing.assert_array_equal(np.array(x[key]), np.array(y[key]))
            plain = WorldLearner(seed=1, burn=1)
            with self.assertRaisesRegex(ValueError, 'objective differs'):
                plain.restore(path, np.random.default_rng(1))
            plain.restore(path, np.random.default_rng(1), allow_objective_change=True)
            self.assertNotIn('prior_image', plain.train(batch))
            with self.assertRaisesRegex(ValueError, 'byte/prior-output'):
                ByteWorldLearner(burn=1).restore(path, rng, allow_objective_change=True)
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                WorldLearner(prior_output_weight=weight)
        with self.assertRaises(ValueError):
            WorldLearner(prior_output_weight=.1, overshoot_distance=4, overshoot_weight=1.)


if __name__ == '__main__':
    unittest.main()
