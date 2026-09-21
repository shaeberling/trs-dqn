import unittest
from unittest.mock import patch

import numpy as np

from rl.evaluate import categorical_policy, evaluate
from rl.temperature_probe import temperature_policy


def logits(obs):
    return np.sin(obs[:, -1, 0, 0:1]+np.arange(6)[None]).astype(np.float32)


class TemperatureProbeTests(unittest.TestCase):
    def test_unit_temperature_exact_actions_inputs_and_rng_consumption(self):
        seen = []
        def infer(obs):
            seen.append(obs.copy())
            return logits(obs)
        base, probe = categorical_policy(logits), temperature_policy(infer, 1)
        left = [np.random.default_rng(i) for i in (10, 11)]
        right = [np.random.default_rng(i) for i in (10, 11)]
        for step in range(20):
            obs = np.full((2, 4, 16, 64), step, np.uint8)
            np.testing.assert_array_equal(base.sample_with_rngs(obs, left),
                                          probe.sample_with_rngs(obs, right))
            np.testing.assert_array_equal(seen[-1], obs)
        self.assertEqual([g.random() for g in left], [g.random() for g in right])

    def test_scaled_logits_are_sampled_with_one_draw_per_action(self):
        obs = np.zeros((200, 4, 16, 64), np.uint8)
        raw = np.tile(np.arange(6, dtype=np.float32), (len(obs), 1))
        probe = temperature_policy(lambda batch: raw, .8)
        probe.reset_seed(32)
        uniform = np.random.default_rng(32).random(len(obs))
        scaled = raw/.8
        probabilities = np.exp(scaled-np.logaddexp.reduce(scaled, axis=-1, keepdims=True))
        expected = (uniform[:, None] > probabilities.cumsum(1)).sum(1).clip(0, 5)
        np.testing.assert_array_equal(probe(obs), expected)
        self.assertGreater(len(set(expected)), 1)

    def test_native_serial_parallel_equivalence(self):
        for temperature in (1, .8):
            serial = evaluate(temperature_policy(logits, temperature), [10000, 10001, 10000],
                              tstates=50000, observation_stride=2, max_steps=12)
            parallel = evaluate(temperature_policy(logits, temperature), [10000, 10001, 10000],
                                tstates=50000, observation_stride=2, max_steps=12, envs=2)
            self.assertEqual(serial['games'], parallel['games'])

    def test_invalid_temperature_before_inference(self):
        for value in (0, -1, np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                temperature_policy(lambda obs: self.fail('unexpected inference'), value)

    def test_final_seeds_rejected_before_model_load(self):
        from rl.temperature_probe import main
        with patch('sys.argv', ['probe', 'unused', '--temperature', '.8', '--seed', '40000',
                                '--output', 'unused.json']), patch('sys.stderr'), \
                patch('rl.temperature_probe.load_temperature_policy') as loader:
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
