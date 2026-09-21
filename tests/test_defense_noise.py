import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rl.defense_noise import PolicyBiasNoise


class DefenseNoiseTests(unittest.TestCase):
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
        baseline = np.array(agent.predict(mx.array(obs))[0])
        agent.act(obs, np.random.default_rng(42), logit_bias=np.full((8, 20), 2, np.float32))
        np.testing.assert_array_equal(np.array(agent.predict(mx.array(obs))[0]), baseline)
        with self.assertRaises(ValueError):
            agent.act(obs, np.random.default_rng(42), logit_bias=np.zeros((1, 20)))


if __name__ == '__main__':
    unittest.main()
