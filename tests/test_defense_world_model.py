import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from rl.defense import GAME_SHA256
from rl.defense_learning import sha256
from rl.defense_world_data import SCHEMA, Sequences
from rl.defense_world_model import WorldLearner, WorldModel, normal_kl


class WorldModelTests(unittest.TestCase):
    def test_gaussian_kl_and_screen_encoding(self):
        a = mx.array([[0., 1.]])
        one = mx.ones_like(a)
        self.assertAlmostEqual(float(normal_kl(a, one, a, one)[0]), 0.)
        self.assertAlmostEqual(float(normal_kl(a, one, a+1, one)[0]), 1.)
        model = WorldModel()
        frames = np.full((1, 1, 16, 64), 128, np.uint8)
        frames[0, 0, 0, 0] = 191
        image = np.array(model.pixels(mx.array(frames)))
        self.assertEqual(image.shape, (1, 1, 48, 128, 2))
        np.testing.assert_array_equal(image[0, 0, :3, :2, 0], .5)
        np.testing.assert_array_equal(image[0, 0, 3:, :, 0], -.5)

    def test_posterior_is_causal_and_prior_uses_actions(self):
        mx.random.seed(13)
        model = WorldModel()
        rng = np.random.default_rng(12)
        frames = rng.integers(128, 192, (2, 5, 16, 64), dtype=np.uint8)
        other = frames.copy(); other[:, 3:] = 32
        actions = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        a, _ = model.observe(mx.array(frames), actions, mx.random.key(0), sample=False)
        b, _ = model.observe(mx.array(other), actions, mx.random.key(0), sample=False)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(np.array(left[:, :3]), np.array(right[:, :3]))
        state = tuple(s[:, 2] for s in a)
        p, _ = model.step(state, actions[:, 2], mx.random.key(1), sample=False)
        q, _ = model.step(state, (actions[:, 2]+1) % 20, mx.random.key(1), sample=False)
        self.assertGreater(float(mx.max(mx.abs(p[0]-q[0]))), 1e-6)
        self.assertEqual(model.decode(model.features(a)).shape, (2, 5, 48, 128, 2))

    def test_real_update_exact_resume_and_checksum(self):
        rng = np.random.default_rng(39)
        batch = (rng.integers(128, 192, (1, 5, 16, 64), dtype=np.uint8),
                 np.array([[1, 3, 4, 9]], np.int32), np.array([[0., .2, 0., 1.]], np.float32),
                 np.array([[1., 1., 1., 0.]], np.float32))
        learner = WorldLearner(seed=11, burn=1)
        before = np.array(learner.model.reward.layers[-1].weight)
        first = learner.train(batch)
        self.assertTrue(all(np.isfinite(v) for v in first.values()))
        self.assertFalse(np.array_equal(before, np.array(learner.model.reward.layers[-1].weight)))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'checkpoint'
            learner.save(checkpoint, rng, {'test': True})
            restored = WorldLearner(seed=87, burn=1)
            other_rng = np.random.default_rng(0)
            self.assertEqual(restored.restore(checkpoint, other_rng), {'test': True})
            np.testing.assert_array_equal(rng.integers(1000, size=10), other_rng.integers(1000, size=10))
            expected, actual = learner.train(batch), restored.train(batch)
            self.assertEqual(expected, actual)
            for (_, a), (_, b) in zip(tree_flatten(learner.model.parameters()),
                                     tree_flatten(restored.model.parameters()), strict=True):
                np.testing.assert_array_equal(np.array(a), np.array(b))
            expected_state = dict(tree_flatten(learner.optimizer.state))
            actual_state = dict(tree_flatten(restored.optimizer.state))
            self.assertEqual(expected_state.keys(), actual_state.keys())
            for name in expected_state:
                np.testing.assert_array_equal(np.array(expected_state[name]), np.array(actual_state[name]))
            (checkpoint/'random.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restored.restore(checkpoint, other_rng)

    def test_reward_alignment_and_burn_in_mask(self):
        model = WorldModel()
        frames = mx.full((1, 5, 16, 64), 128, mx.uint8)
        actions = mx.array([[1, 2, 3, 4]])
        rewards, continuation = mx.zeros((1, 4)), mx.ones((1, 4))
        key = mx.random.key(7)
        _, before = model.loss(frames, actions, rewards, continuation, key, burn=2)
        _, ignored = model.loss(frames, actions, mx.array([[100., 100., 0., 0.]]),
                                mx.array([[0., 0., 1., 1.]]), key, burn=2)
        np.testing.assert_array_equal(np.array(before), np.array(ignored))
        changed = mx.array([[0., 0., 2., 3.]])
        _, after = model.loss(frames, actions, changed, continuation, key, burn=2)
        states, _ = model.observe(frames, actions, key)
        predicted = model.reward(model.features(states)[:, 3:])[..., 0]
        expected = float(.5*mx.mean((predicted-changed[:, 2:])**2))
        self.assertAlmostEqual(float(after[1]), expected, places=5)
        np.testing.assert_array_equal(np.array(before)[[0, 2, 3]], np.array(after)[[0, 2, 3]])

    def test_open_loop_audit_does_not_observe_future_targets(self):
        from rl.defense_world_fit import audit
        model = WorldModel()
        rng = np.random.default_rng(93)
        frames = rng.integers(128, 192, (2, 6, 16, 64), dtype=np.uint8)
        batch = (frames, np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], np.int32),
                 np.zeros((2, 5), np.float32), np.ones((2, 5), np.float32))
        first = audit(model, batch, context=2)
        changed = frames.copy(); changed[:, 3:] = 191
        second = audit(model, (changed, *batch[1:]), context=2)
        for a, b in zip(first['horizons'], second['horizons'], strict=True):
            self.assertEqual(a['prediction_action_sensitivity'], b['prediction_action_sensitivity'])
            self.assertEqual(a['reward_mse'], b['reward_mse'])
            self.assertNotEqual(a['pixel_mse'], b['pixel_mse'])

    def test_sequence_splits_alignment_boundaries_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for i, split in enumerate(('train', 'heldout')):
                frames = np.broadcast_to(np.arange(7, dtype=np.uint8)[:, None, None]+i*50,
                                         (7, 16, 64)).copy()
                path = root/f'episode-{i}.npz'
                np.savez_compressed(path, frames=frames, actions=np.arange(6, dtype=np.int32),
                    rewards=np.arange(6, dtype=np.float32)*10,
                    continuation=np.array([1, 1, 0, 1, 1, 0], np.float32))
                rows.append(dict(file=path.name, sha256=sha256(path), split=split,
                                 steps=6, result={'score': 150}))
            manifest = dict(schema=SCHEMA, purpose='new-own-experience-for-world-model',
                game_sha256=GAME_SHA256, existing_evaluation_data=False, episodes=rows)
            (root/'manifest.json').write_text(json.dumps(manifest))
            train, held = Sequences(root, 'train', 3), Sequences(root, 'heldout', 3)
            self.assertFalse(set(train.files) & set(held.files))
            frames, actions, rewards, cont = train.sample(64, np.random.default_rng(0))
            np.testing.assert_array_equal(frames[:, :-1, 0, 0], actions)
            np.testing.assert_allclose(rewards, actions*.1, atol=1e-7)
            self.assertTrue(np.any(cont == 0))  # Life losses retained, not dropped/rewarded.
            self.assertTrue(np.all(frames < 50))
            self.assertTrue(np.all(held.sample(5, np.random.default_rng(0))[0] >= 50))
            (root/rows[0]['file']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Sequences(root, 'train', 3)


if __name__ == '__main__':
    unittest.main()
