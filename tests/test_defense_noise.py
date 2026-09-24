import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_noise import (PolicyBiasNoise, PolicyWeightNoise, PolicyKeyNoise,
                              PolicyDurationNoise, NoiseRollout)


class DefenseNoiseTests(unittest.TestCase):
    def test_duration_factor_noise_is_direction_neutral_and_replay_aligned(self):
        noise = PolicyDurationNoise(2, 20, 4, 4, np.random.default_rng(19))
        self.assertEqual(noise.values.shape, (2, 80))
        for worker in range(2):
            for duration in range(4):
                self.assertEqual(len(set(noise.values[worker, duration*20:(duration+1)*20])), 1)
        rollout = NoiseRollout(noise)
        before = rollout.record().copy()
        rollout.redraw(np.array([True, False]))
        after = rollout.record().copy()
        np.testing.assert_array_equal(after[1], before[1])
        self.assertFalse(np.array_equal(after[0], before[0]))
        bank, ids = rollout.arrays()
        np.testing.assert_array_equal(bank[ids], np.concatenate((before, after)))
        self.assertEqual(noise.draws, 3)
        zero_rng = np.random.default_rng(19)
        state = copy.deepcopy(zero_rng.bit_generator.state)
        zero = PolicyDurationNoise(2, 20, 4, 0, zero_rng)
        zero.redraw(np.array([True, True]))
        self.assertEqual(zero_rng.bit_generator.state, state)
        np.testing.assert_array_equal(zero.values, np.zeros((2, 80)))
        for spec in ((2, 20, 1, 4), (2, 20, 4, -1), (2, 20, 4, float('nan'))):
            with self.assertRaises(ValueError):
                PolicyDurationNoise(*spec, rng=np.random.default_rng(19))

    def test_duration_noise_cli_requires_joint_options_and_no_other_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'unused'
            for flags in (['--policy-duration-noise=-1'], ['--policy-duration-noise=nan'],
                          ['--policy-duration-noise=4'],
                          ['--policy-duration-noise=4', '--policy-bias-noise=1']):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_train',
                                         '--run', str(output), *flags],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('policy-duration-noise', result.stderr)
                self.assertFalse(output.exists())

    def test_key_factor_noise_is_direction_symmetric_and_correlated(self):
        noise = PolicyKeyNoise(2, 21, 1, np.random.default_rng(7))
        self.assertEqual(noise.values.shape, (2, 21))
        matrix = noise.matrix
        self.assertEqual(matrix.shape, (21, 7))
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1, 0, 0, 0])  # LEFT
        np.testing.assert_array_equal(matrix[4], [0, 0, 0, 0, 1, 0, 0])  # RIGHT
        np.testing.assert_array_equal(matrix[20], [0, 0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(float(matrix[6, 1]), 2**-.5)  # UP+RIGHT
        self.assertAlmostEqual(float(matrix[6, 4]), 2**-.5)
        self.assertAlmostEqual(float(matrix[5, 1]), 2**-.5)  # UP+LEFT mirror
        self.assertAlmostEqual(float(matrix[5, 3]), 2**-.5)
        np.testing.assert_allclose(np.sum(matrix*matrix, axis=1), 1, atol=1e-7)
        before = noise.values.copy()
        noise.redraw(np.array([False, True]))
        np.testing.assert_array_equal(noise.values[0], before[0])
        self.assertFalse(np.array_equal(noise.values[1], before[1]))
        self.assertEqual(noise.draws, 3)

    def test_key_factor_noise_rollout_and_zero_rng_contract(self):
        rng = np.random.default_rng(11)
        state = copy.deepcopy(rng.bit_generator.state)
        zero = PolicyKeyNoise(2, 20, 0, rng)
        zero.redraw(np.ones(2, dtype=bool))
        self.assertEqual(rng.bit_generator.state, state)
        np.testing.assert_array_equal(zero.values, np.zeros((2, 20)))
        noise = PolicyKeyNoise(2, 20, 1, np.random.default_rng(11))
        rollout = NoiseRollout(noise)
        first = rollout.record().copy()
        rollout.redraw(np.array([True, False]))
        second = rollout.record().copy()
        bank, ids = rollout.arrays()
        np.testing.assert_array_equal(bank[ids], np.concatenate((first, second)))
        for args in ((2, 19, 1), (2, 21, -1), (2, 20, float('nan'))):
            with self.assertRaises(ValueError):
                PolicyKeyNoise(*args, rng=np.random.default_rng(2))

    def test_key_factor_noise_fixed_interval_is_worker_local_and_recorded(self):
        noise = PolicyKeyNoise(2, 21, 1, np.random.default_rng(29), interval=3)
        rollout = NoiseRollout(noise)
        initial = rollout.record().copy()
        rollout.redraw(np.array([False, True]))
        after_boundary = rollout.record().copy()
        np.testing.assert_array_equal(after_boundary[0], initial[0])
        self.assertFalse(np.array_equal(after_boundary[1], initial[1]))
        rollout.redraw(np.array([False, False]))
        before_interval = rollout.record().copy()
        np.testing.assert_array_equal(before_interval, after_boundary)
        rollout.redraw(np.array([False, False]))
        after_interval = rollout.record().copy()
        self.assertFalse(np.array_equal(after_interval[0], before_interval[0]))
        np.testing.assert_array_equal(after_interval[1], before_interval[1])
        bank, ids = rollout.arrays()
        np.testing.assert_array_equal(bank[ids], np.concatenate(
            (initial, after_boundary, before_interval, after_interval)))
        self.assertEqual(noise.draws, 4)
        self.assertEqual(noise.elapsed.tolist(), [0, 2])
        for interval in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                PolicyKeyNoise(2, 21, 1, np.random.default_rng(3), interval=interval)

    def test_random_key_interval_renews_independently_and_preserves_rollout_bank(self):
        noise = PolicyKeyNoise(2, 21, 2, np.random.default_rng(31), interval_range=(3, 5))
        self.assertTrue(np.all((3 <= noise.periods) & (noise.periods <= 5)))
        initial = noise.values.copy()
        first_period = int(noise.periods[0])
        rollout = NoiseRollout(noise)
        recorded = []
        for step in range(first_period):
            recorded.append(rollout.record().copy())
            rollout.redraw(np.array([False, step == 0]))
            if step < first_period-1:
                np.testing.assert_array_equal(noise.values[0], initial[0])
        self.assertFalse(np.array_equal(noise.values[0], initial[0]))
        self.assertEqual(int(noise.elapsed[0]), 0)
        self.assertTrue(np.all((3 <= noise.periods) & (noise.periods <= 5)))
        bank, ids = rollout.arrays()
        np.testing.assert_array_equal(bank[ids], np.concatenate(recorded))
        periods = []
        for _ in range(64):
            noise.redraw(np.array([True, False]))
            periods.append(int(noise.periods[0]))
        self.assertTrue(all(3 <= period <= 5 for period in periods))
        self.assertGreater(len(set(periods)), 1)
        for interval_range in ((0, 4), (5, 4), (True, 5), (2.5, 5), [3, 5]):
            with self.assertRaises(ValueError):
                PolicyKeyNoise(2, 21, 1, np.random.default_rng(3), interval_range=interval_range)
        with self.assertRaises(ValueError):
            PolicyKeyNoise(2, 21, 1, np.random.default_rng(3), interval=3,
                           interval_range=(3, 5))
        rng = np.random.default_rng(10)
        before = copy.deepcopy(rng.bit_generator.state)
        zero = PolicyKeyNoise(2, 20, 0, rng, interval_range=(3, 5))
        zero.redraw(np.ones(2, dtype=bool))
        self.assertEqual(rng.bit_generator.state, before)
        np.testing.assert_array_equal(zero.values, np.zeros((2, 20)))

    def test_noise_persists_and_only_boundary_workers_are_redrawn(self):
        noise = PolicyBiasNoise(4, 20, 1, np.random.default_rng(42))
        original = noise.bias.copy()
        state = copy.deepcopy(noise.rng.bit_generator.state)
        for _ in range(100):
            noise.redraw(np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(noise.bias, original)
        self.assertEqual(noise.rng.bit_generator.state, state)
        self.assertEqual(noise.draws, 4)
        noise.redraw(np.array([False, True, False, True]))
        np.testing.assert_array_equal(noise.bias[[0, 2]], original[[0, 2]])
        self.assertFalse(np.array_equal(noise.bias[[1, 3]], original[[1, 3]]))
        self.assertEqual(noise.draws, 6)

    def test_zero_noise_does_not_consume_randomness(self):
        rng = np.random.default_rng(4)
        original = copy.deepcopy(rng.bit_generator.state)
        noise = PolicyBiasNoise(2, 20, 0, rng)
        noise.redraw(np.ones(2, dtype=bool))
        self.assertEqual(rng.bit_generator.state, original)
        np.testing.assert_array_equal(noise.bias, np.zeros((2, 20)))
        self.assertEqual(noise.draws, 0)

    def test_rng_resume_reproduces_new_episode_draws(self):
        first = PolicyBiasNoise(3, 20, .7, np.random.default_rng(5))
        restored_rng = np.random.default_rng(999)
        restored_rng.bit_generator.state = copy.deepcopy(first.rng.bit_generator.state)
        second = PolicyBiasNoise(3, 20, .7, restored_rng)
        first.redraw(np.ones(3, dtype=bool))
        np.testing.assert_array_equal(first.bias, second.bias)

    def test_bad_configuration_and_boundary_masks_rejected(self):
        for std in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PolicyBiasNoise(2, 20, std, np.random.default_rng(1))
        noise = PolicyBiasNoise(2, 20, 1, np.random.default_rng(1))
        for mask in (np.ones(2, dtype=int), np.ones(3, dtype=bool)):
            with self.assertRaises(ValueError):
                noise.redraw(mask)

    def test_cli_rejects_invalid_noise_and_sil_combination(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'unused'
            for flags in (['--policy-bias-noise=-1'], ['--policy-bias-noise=nan'],
                          ['--policy-bias-noise=1', '--sil-updates=1']):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_train',
                                         '--run', str(output), *flags],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('policy-bias-noise', result.stderr)
                self.assertFalse(output.exists())

    def test_weight_noise_cli_rejects_invalid_or_combined_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'unused'
            for flags in (['--policy-weight-noise=-1'], ['--policy-weight-noise=nan'],
                          ['--policy-weight-noise=.02', '--sil-updates=1'],
                          ['--policy-weight-noise=.02', '--policy-bias-noise=1']):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_train',
                                         '--run', str(output), *flags],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('policy-weight-noise', result.stderr)
                self.assertFalse(output.exists())

    def test_key_noise_cli_rejects_invalid_or_combined_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'unused'
            for flags in (['--policy-key-noise=-1'], ['--policy-key-noise=nan'],
                          ['--policy-key-noise=1', '--policy-bias-noise=1'],
                          ['--policy-key-noise=1', '--sil-updates=1'],
                          ['--policy-key-noise=1', '--allow-enter'],
                          ['--policy-key-noise-interval=-1'],
                          ['--policy-key-noise-interval=32']):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_train',
                                         '--run', str(output), *flags],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('policy-key-noise', result.stderr)
                self.assertFalse(output.exists())

    def test_random_key_interval_cli_rejects_bad_bounds_and_fixed_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'unused'
            for flags in (['--policy-key-noise-min-interval=3'],
                          ['--policy-key-noise=2', '--policy-key-noise-max-interval=5'],
                          ['--policy-key-noise=2', '--policy-key-noise-min-interval=5',
                           '--policy-key-noise-max-interval=3'],
                          ['--policy-key-noise=2', '--policy-key-noise-min-interval=3',
                           '--policy-key-noise-max-interval=5', '--policy-key-noise-interval=4']):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_train',
                                         '--run', str(output), *flags],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('policy-key-noise', result.stderr)
                self.assertFalse(output.exists())

    def test_rollout_bank_preserves_time_worker_and_shuffle_alignment(self):
        for noise in (PolicyBiasNoise(3, 20, 1, np.random.default_rng(4)),
                      PolicyWeightNoise(3, 20, 256, .02, np.random.default_rng(4))):
            rollout = NoiseRollout(noise)
            expected = []
            for step in range(20):
                expected.append(rollout.record().copy())
                rollout.redraw(np.array([step == 0, False, step == 0]))
            bank, indices = rollout.arrays()
            self.assertEqual(len(bank), 5)  # three initial draws, two redraws
            self.assertEqual(len(indices), 60)
            np.testing.assert_array_equal(bank[indices], np.concatenate(expected))
            order = np.random.default_rng(91).permutation(60)
            np.testing.assert_array_equal(bank[indices[order]], np.concatenate(expected)[order])
            # Neither subsequent noise draws nor a new rollout can mutate it.
            before = bank.copy()
            noise.redraw(np.ones(3, dtype=bool))
            np.testing.assert_array_equal(bank, before)
            with self.assertRaises(ValueError):
                NoiseRollout(noise).arrays()

    def test_weight_noise_matches_perturbed_parameters_including_encoder_gradient(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map
        from rl.model import QNetwork
        from rl.ppo import PPO
        agent = PPO(seed=9, action_count=20)
        obs = np.random.default_rng(17).integers(128, 192, (4, 4, 16, 64), dtype=np.uint8)
        draw = PolicyWeightNoise(1, 20, 256, .02, np.random.default_rng(91)).values
        noise = np.repeat(draw, 4, axis=0)
        original = np.array(agent.model.advantage.weight).copy()
        actions, logp, values = agent.act(obs, np.random.default_rng(3), head_weight_noise=noise)
        shifted = QNetwork(action_count=20)
        shifted.update(tree_map(lambda x: mx.array(x), agent.model.parameters()))
        shifted.advantage.weight = shifted.advantage.weight + mx.array(draw[0])
        logits, shifted_values = shifted.policy_value(mx.array(obs))
        lp = np.array(logits-mx.logsumexp(logits, axis=1, keepdims=True))
        np.testing.assert_allclose(logp, lp[np.arange(4), actions], atol=2e-6)
        np.testing.assert_array_equal(values, np.array(shifted_values))
        data = tuple(mx.array(x) for x in (obs, actions, logp, np.ones(4, np.float32), values+.1))
        (_, aux), grads = nn.value_and_grad(agent.model, agent._loss)(
            agent.model, *data, head_weight_noise=mx.array(noise))
        (_, other_aux), other_grads = nn.value_and_grad(shifted, agent._loss)(shifted, *data)
        self.assertAlmostEqual(float(aux[3].item()), 0, places=6)
        for (name, grad), (other_name, other_grad) in zip(tree_flatten(grads), tree_flatten(other_grads),
                                                       strict=True):
            self.assertEqual(name, other_name)
            np.testing.assert_allclose(np.array(grad), np.array(other_grad), atol=2e-6, rtol=2e-4,
                                       err_msg=name)
        np.testing.assert_array_equal(np.array(agent.model.advantage.weight), original)
        loss, aux = agent.update(*data, head_weight_noise=mx.array(noise))
        mx.eval(loss, aux, agent.state)
        self.assertTrue(np.isfinite(float(loss.item())))

    def test_sampler_and_loss_use_the_same_fixed_parameter_perturbation(self):
        import mlx.core as mx
        from rl.ppo import PPO
        agent = PPO(seed=9, action_count=20)
        obs = np.full((8, 4, 16, 64), 128, np.uint8)
        bias = PolicyBiasNoise(8, 20, .8, np.random.default_rng(91)).bias
        original = np.array(agent.model.advantage.bias).copy()
        actions, logp, values = agent.act(obs, np.random.default_rng(3), logit_bias=bias)
        logits = np.array(agent.predict(mx.array(obs))[0]) + bias
        expected = logits-np.logaddexp.reduce(logits, axis=1, keepdims=True)
        np.testing.assert_allclose(logp, expected[np.arange(8), actions], atol=1e-6)
        data = tuple(mx.array(x) for x in (obs, actions, logp, np.ones(8, np.float32), values))
        loss, metrics = agent._loss(agent.model, *data, logit_bias=mx.array(bias))
        self.assertAlmostEqual(float(metrics[0].item()), -1, places=5)
        self.assertAlmostEqual(float(metrics[3].item()), 0, places=6)
        _, wrong = agent._loss(agent.model, *data)
        self.assertGreater(float(wrong[3].item()), .01)
        np.testing.assert_array_equal(np.array(agent.model.advantage.bias), original)
        loss, metrics = agent.update(*data, logit_bias=mx.array(bias))
        mx.eval(loss, metrics, agent.state)
        self.assertTrue(np.isfinite(float(loss.item())))

    def test_disabled_sampling_and_frozen_evaluation_are_unchanged(self):
        import mlx.core as mx
        from rl.ppo import PPO
        agent = PPO(seed=21, action_count=20)
        obs = np.full((8, 4, 16, 64), 128, np.uint8)
        original = agent.act(obs, np.random.default_rng(42))
        zero = agent.act(obs, np.random.default_rng(42), logit_bias=np.zeros((8, 20), np.float32))
        for a, b in zip(original, zero):
            np.testing.assert_array_equal(a, b)
        zero_weights = agent.act(obs, np.random.default_rng(42),
                                 head_weight_noise=np.zeros((8, 20, 256), np.float32))
        for a, b in zip(original, zero_weights):
            np.testing.assert_array_equal(a, b)
        baseline = np.array(agent.predict(mx.array(obs))[0])
        agent.act(obs, np.random.default_rng(42), logit_bias=np.full((8, 20), 2, np.float32))
        np.testing.assert_array_equal(np.array(agent.predict(mx.array(obs))[0]), baseline)
        with self.assertRaises(ValueError):
            agent.act(obs, np.random.default_rng(42), logit_bias=np.zeros((1, 20)))
        with self.assertRaises(ValueError):
            agent.act(obs, np.random.default_rng(42), head_weight_noise=np.zeros((8, 20, 255)))


if __name__ == '__main__':
    unittest.main()
