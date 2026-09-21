import unittest
from unittest.mock import patch

import numpy as np

from rl.evaluate import categorical_policy, evaluate
from rl.temporal_probe import TemporalPolicy


def stack(value):
    return np.full((4, 16, 64), value, dtype=np.uint8)


def policy(seen=None):
    def logits(observations):
        if seen is not None:
            seen.append(observations.copy())
        return np.sin(observations[:, :, 0, 0].sum(axis=1, keepdims=True)
                      + np.arange(6)[None, :])
    return categorical_policy(logits)


class TemporalProbeTests(unittest.TestCase):
    def test_stride_one_exact_passthrough_and_rng_draws(self):
        seen = []
        base, probe = policy(), TemporalPolicy(policy(seen), stride=1)
        left = [np.random.default_rng(4), np.random.default_rng(9)]
        right = [np.random.default_rng(4), np.random.default_rng(9)]
        for step in range(12):
            obs = np.stack([stack(step), stack(step+3)])
            np.testing.assert_array_equal(base.sample_with_rngs(obs, left),
                                          probe.sample_with_rngs(obs, right))
            np.testing.assert_array_equal(seen[-1], obs)
        self.assertEqual([r.random() for r in left], [r.random() for r in right])
        self.assertEqual(probe.histories, {})

    def test_spaced_history_padding_and_input_copy(self):
        seen = []
        probe = TemporalPolicy(policy(seen))
        for step in range(9):
            obs = stack(step)[None]
            probe(obs)
            expected = [max(0, step-offset) for offset in (6, 4, 2, 0)]
            np.testing.assert_array_equal(seen[-1][0, :, 0, 0], expected)
            obs.fill(255)
        probe.reset_seed(123)
        probe(stack(77)[None])
        np.testing.assert_array_equal(seen[-1], stack(77)[None])
        self.assertEqual(len(probe.histories), 1)

    def test_interleaved_games_match_serial_even_with_duplicate_seeds(self):
        serial = [TemporalPolicy(policy()) for _ in range(3)]
        seeds = [41, 41, 9]
        for item, seed in zip(serial, seeds):
            item.reset_seed(seed)
        probe = TemporalPolicy(policy())
        rngs = [np.random.default_rng(seed) for seed in seeds]
        counts = [0, 0, 0]
        for jobs in ([0, 1], [1, 0], [0, 2], [2, 1], [1], [0, 2]):
            obs = np.stack([stack(30*job+counts[job]) for job in jobs])
            actual = probe.sample_with_rngs(obs, [rngs[job] for job in jobs])
            expected = [serial[job](frame[None])[0] for job, frame in zip(jobs, obs)]
            np.testing.assert_array_equal(actual, expected)
            for job in jobs:
                counts[job] += 1
        self.assertEqual(len(probe.histories), 3)
        self.assertEqual([r.random() for r in rngs],
                         [item.serial_rng.random() for item in serial])

    def test_native_serial_parallel_equivalence_and_stride_one_baseline(self):
        seeds = [10002, 10001, 10002]
        baseline = evaluate(policy(), seeds, max_steps=12, tstates=50_000)
        for stride in (1, 2):
            serial = evaluate(TemporalPolicy(policy(), stride), seeds,
                              max_steps=12, tstates=50_000)
            parallel = evaluate(TemporalPolicy(policy(), stride), seeds,
                                max_steps=12, tstates=50_000, envs=2)
            self.assertEqual(serial["games"], parallel["games"])
            if stride == 1:
                self.assertEqual(serial["games"], baseline["games"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            TemporalPolicy(policy(), 3)
        with self.assertRaises(ValueError):
            TemporalPolicy(lambda obs: np.zeros(len(obs)))
        probe = TemporalPolicy(policy())
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            probe.sample_with_rngs(np.stack([stack(0), stack(0)]), [rng, rng])
        with self.assertRaises(ValueError):
            probe.sample_with_rngs(stack(0)[None].astype(float), [rng])
        with self.assertRaises(ValueError):
            probe(np.stack([stack(0), stack(1)]))
        bad = stack(0)
        bad[0] = 1
        with self.assertRaises(ValueError):
            probe(bad[None])

    def test_final_seeds_rejected_before_model_load(self):
        with patch("sys.argv", ["probe", "unused", "--stride", "2", "--seed", "40000",
                                "--output", "unused.json"]), \
                patch("rl.temporal_probe.load_policy") as loader, \
                patch("sys.stderr"):
            from rl.temporal_probe import main
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
