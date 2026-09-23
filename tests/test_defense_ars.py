from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense import action_names, GAME_SHA256, ENVIRONMENT_VERSION
from rl.defense_ars import (ALGORITHM, acting_policy, ars_update, candidate_jobs, head_array,
    make_population_infer, play_population, restore_checkpoint, save_checkpoint, set_head)
from rl.defense_learning import load_policy, verify_policy_trace
from rl.model import QNetwork


def config():
    return dict(algorithm=ALGORITHM, game='defense', game_sha256=GAME_SHA256,
        environment_version=ENVIRONMENT_VERSION, action_names=list(action_names()),
        allow_enter=False, tstates=100000, observation_stride=1, eval_max_steps=0, parent_steps=123)


class ARSTests(unittest.TestCase):
    def test_update_orientation_scaling_and_zero_variance(self):
        head = np.zeros((1, 2), np.float32)
        directions = np.array([[[1., 0.]], [[0., 1.]]], np.float32)
        returns = np.array([[[4., 4.], [0., 0.]], [[0., 0.], [2., 2.]]])
        following, stats = ars_update(head, directions, returns, .1)
        expected = .1*np.array([[4., -2.]])/(2*returns.mean(axis=2).std())
        np.testing.assert_allclose(following, expected)
        other, _ = ars_update(head, directions, returns*10+80, .1)
        np.testing.assert_array_equal(following, other)
        unchanged, info = ars_update(head, directions, np.ones_like(returns), .1)
        np.testing.assert_array_equal(unchanged, head)
        self.assertTrue(info['skipped_zero_variance'])
        for invalid in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                ars_update(head, directions, returns, invalid)

    def test_common_pair_seeds_are_separate_from_validation(self):
        jobs = candidate_jobs(3, 2, 70000)
        self.assertEqual(len(jobs), 12)
        for d in range(3):
            plus = [j['seed'] for j in jobs if j['direction'] == d and j['sign'] == 0]
            minus = [j['seed'] for j in jobs if j['direction'] == d and j['sign'] == 1]
            self.assertEqual(plus, minus)
        self.assertEqual(len(set(j['seed'] for j in jobs)), 6)
        self.assertFalse(set(j['seed'] for j in jobs) & set(range(10000, 10010)))
        with self.assertRaises(ValueError):
            candidate_jobs(3, 2, 10000)

    def test_population_head_math_and_only_head_changes(self):
        mx.random.seed(1)
        model = QNetwork(action_count=20)
        rng = np.random.default_rng(3)
        obs = rng.integers(128, 192, (3, 4, 16, 64), dtype=np.uint8)
        before = {k: np.array(v).copy() for k, v in tree_flatten(model.parameters())}
        original = head_array(model)
        bank = np.stack([original, original+.001, original-.001])
        infer = make_population_infer(model, bank)
        actual = infer(obs, [0, 1, 2])
        for i in range(3):
            set_head(model, bank[i])
            logits = np.array(model.policy_value(mx.array(obs[i:i+1]))[0])[0]
            np.testing.assert_allclose(actual[i], logits, atol=1e-5, rtol=1e-5)
        for key, value in tree_flatten(model.parameters()):
            if not key.startswith('advantage.'):
                np.testing.assert_array_equal(before[key], np.array(value))

    def test_checkpoint_rng_next_update_loader_and_corruption(self):
        mx.random.seed(6)
        model = QNetwork(action_count=20)
        rng = np.random.default_rng(4)
        head = head_array(model)
        delta = rng.normal(size=(2, *head.shape)).astype(np.float32)
        updated, _ = ars_update(head, delta, np.array([[[10.], [5.]], [[3.], [7.]]]), .002)
        set_head(model, updated)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint'
            save_checkpoint(model, path, config(), generation=1, training_steps=100,
                            training_games=4, rng=rng, next_seed=70002)
            restored, other_rng = QNetwork(action_count=20), np.random.default_rng(0)
            state = restore_checkpoint(restored, path, other_rng)
            self.assertEqual(state['steps'], 223)
            self.assertEqual(state['next_training_seed'], 70002)
            np.testing.assert_array_equal(head_array(model), head_array(restored))
            first, second = rng.normal(size=(2, *head.shape)), other_rng.normal(size=(2, *head.shape))
            np.testing.assert_array_equal(first, second)
            a, _ = ars_update(head_array(model), first, np.array([[[4.], [1.]], [[2.], [6.]]]), .002)
            b, _ = ars_update(head_array(restored), second, np.array([[[4.], [1.]], [[2.], [6.]]]), .002)
            np.testing.assert_array_equal(a, b)
            loaded, _ = load_policy(path/'model.safetensors')
            original_policy = acting_policy(model)
            loaded.reset_seed(10); original_policy.reset_seed(10)
            obs = np.full((6, 4, 16, 64), 191, np.uint8)
            np.testing.assert_array_equal(loaded(obs), original_policy(obs))
            candidate = Path(temporary)/'candidate'
            save_checkpoint(model, candidate, config(), generation=1, training_steps=100,
                            training_games=4, rng=rng, next_seed=70002, candidate=True)
            with self.assertRaises(ValueError):
                restore_checkpoint(restored, candidate, other_rng)
            (path/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore_checkpoint(restored, path, other_rng)

    def test_native_population_matches_frozen_policy_reexecution(self):
        mx.random.seed(14)
        model = QNetwork(action_count=20)
        original = head_array(model)
        perturbation = np.random.default_rng(15).normal(0, .001, original.shape).astype(np.float32)
        bank = np.stack([original+perturbation, original-perturbation])
        jobs = candidate_jobs(1, 1, 99000)
        events = []
        games, traces = play_population(model, bank, jobs, envs=2, log=events.append, keep_traces=True)
        self.assertEqual(len(games), 2)
        workers = next(row['workers'] for row in events if row['event'] == 'population_workers')
        self.assertTrue(all(not row['mlx_loaded'] for row in workers))
        np.testing.assert_array_equal(head_array(model), original)
        with tempfile.TemporaryDirectory() as temporary:
            for i, (game, trace) in enumerate(zip(games, traces)):
                set_head(model, bank[game['candidate']])
                checkpoint = Path(temporary)/str(i)
                save_checkpoint(model, checkpoint, config(), generation=0, training_steps=0,
                    training_games=0, rng=np.random.default_rng(0), next_seed=70000)
                result = {k: v for k, v in game.items() if k not in ('candidate', 'direction', 'sign', 'repetition')}
                verified = verify_policy_trace(checkpoint/'model.safetensors', *trace, result)
                self.assertTrue(verified['verified'])
                self.assertEqual(verified['verified_actions'], game['steps'])


if __name__ == '__main__':
    unittest.main()
