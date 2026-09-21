import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rl.evaluate import EvaluationCancelled, EvaluationProgress, categorical_policy, evaluate
from rl.vector import VectorEnv


class FakeEnv:
    """Action-sensitive games with different lengths, including duplicates."""
    def __init__(self, tstates=100000, max_steps=0, observation_stride=1):
        self.max_steps = max_steps

    def reset(self, seed):
        self.seed, self.steps, self.score = seed, 0, 0
        self.done = False
        return np.array([seed, self.steps, self.score])

    def step(self, action):
        if self.done:
            raise AssertionError("completed game stepped again")
        self.steps += 1
        self.score += int(action)
        terminal = self.steps >= self.seed % 4 + 1
        truncated = bool(self.max_steps and self.steps >= self.max_steps and not terminal)
        self.done = terminal or truncated
        return (np.array([self.seed, self.steps, self.score]), action, terminal, truncated,
                dict(score=self.score, level=1+self.score//5, steps=self.steps,
                     terminated=terminal, truncated=truncated))

    def close(self):
        pass


class FakeVector:
    def __init__(self, count, seed, **config):
        self.envs = [FakeEnv(**config) for _ in range(count)]

    def reset(self, seeds, *, indices=None):
        indices = range(len(self.envs)) if indices is None else indices
        return np.stack([self.envs[i].reset(seed) for i, seed in zip(indices, seeds, strict=True)])

    def step(self, actions, *, indices):
        return [(*self.envs[i].step(action), None)
                for i, action in zip(indices, actions, strict=True)]

    def close(self):
        pass


def fake_policy():
    return categorical_policy(lambda obs: np.sin(obs[:, :1]+np.arange(6)[None, :]))


class ParallelEvaluationTests(unittest.TestCase):
    def test_progress_exposes_unfinished_games_without_counting_them_complete(self):
        rows = []
        with patch("rl.evaluate.time.monotonic", side_effect=[10, 19, 20]):
            reporter = EvaluationProgress(callback=rows.append)
            finished = [dict(score=12, level=1, terminated=True),
                        dict(score=50, level=1, terminated=False), None]
            active = [(10010, dict(steps=100000, score=292, level=5, waiting=True))]
            reporter(finished, active)
            self.assertEqual(rows, [])
            reporter(finished, active)
        self.assertEqual(rows[0], dict(event="evaluation_progress", evaluation_seconds=10,
                                      finished_games=2, complete_games=1, active_games=1,
                                      active=[dict(seed=10010, steps=100000, score=292,
                                                   level=5, waiting=True)]))

    def test_requested_cancellation_closes_workers_and_returns_no_result(self):
        for count in [1, 2]:
            calls = iter([False, True])
            with patch("rl.evaluate.BreakdownEnv", FakeEnv), patch("rl.vector.VectorEnv", FakeVector), \
                    patch.object(FakeEnv, "close") as serial_close, \
                    patch.object(FakeVector, "close") as parallel_close:
                with self.assertRaises(EvaluationCancelled):
                    evaluate(fake_policy(), [7, 6], envs=count, max_steps=0,
                             should_stop=lambda: next(calls))
                (serial_close if count == 1 else parallel_close).assert_called_once()

    def test_archived_complete_game_equivalence_and_checksums(self):
        directory = Path(__file__).resolve().parents[1]/"results"/"level10"
        manifest = json.loads((directory/"parallel-evaluation-check.json").read_text())
        baseline = json.loads((directory/"validation-lr5e5-18m5.json").read_text())
        self.assertEqual(manifest["seeds"], [10000, 10019])
        self.assertFalse(manifest["load_controlled"])
        for measurement in manifest["measurements"]:
            data = (directory/measurement["result"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), measurement["sha256"])
            result = json.loads(data)
            self.assertEqual(result["checkpoint_sha256"], manifest["checkpoint_sha256"])
            self.assertEqual(result["eval_envs"], measurement["eval_envs"])
            self.assertEqual(result["games"], baseline["games"])
            self.assertEqual([g["seed"] for g in result["games"]], list(range(10000, 10020)))
            self.assertEqual(result["complete_games"], 20)
            self.assertEqual(result["incomplete_games"], 0)
            self.assertTrue(all(g["terminated"] and not g["truncated"] for g in result["games"]))

    def test_scheduler_matches_serial_with_separate_rngs_and_duplicate_seeds(self):
        seeds = [7, 4, 5, 6, 7, 12, 13]
        with patch("rl.evaluate.BreakdownEnv", FakeEnv), patch("rl.vector.VectorEnv", FakeVector):
            for limit in [0, 2]:
                for epsilon in [0, 0.35, 1]:
                    expected = evaluate(fake_policy(), seeds, max_steps=limit, epsilon=epsilon)
                    for count in [2, 3, 20]:
                        actual = evaluate(fake_policy(), iter(seeds), envs=count,
                                          max_steps=limit, epsilon=epsilon)
                        self.assertEqual(actual.pop("eval_envs"), count)
                        self.assertEqual(actual, {k: v for k, v in expected.items() if k != "eval_envs"})
                        self.assertEqual([g["seed"] for g in actual["games"]], seeds)

    def test_stateless_deterministic_policy(self):
        policy = lambda obs: obs[:, 0] % 6
        with patch("rl.evaluate.BreakdownEnv", FakeEnv), patch("rl.vector.VectorEnv", FakeVector):
            serial = evaluate(policy, [7, 4, 5], max_steps=0)
            parallel = evaluate(policy, [7, 4, 5], envs=2, max_steps=0)
            self.assertEqual(serial["games"], parallel["games"])

    def test_native_worker_reset_and_partial_batch_match_serial(self):
        # Short real-emulator integration check; complete frozen-policy games
        # are additionally compared by the external validation benchmark.
        policy = categorical_policy(lambda obs: np.tile(np.arange(6), (len(obs), 1)))
        seeds = [10002, 10001, 10002]
        serial = evaluate(policy, seeds, max_steps=12, epsilon=0.2)
        parallel = evaluate(policy, seeds, envs=2, max_steps=12, epsilon=0.2)
        self.assertEqual(serial["games"], parallel["games"])
        self.assertEqual(parallel["complete_games"], 0)
        self.assertEqual(parallel["incomplete_games"], 3)

    def test_invalid_parallel_policy_fails_before_creating_workers(self):
        policy = lambda obs: np.zeros(len(obs), dtype=int)
        policy.reset_seed = lambda seed: None
        with patch("rl.vector.VectorEnv") as workers:
            with self.assertRaisesRegex(ValueError, "sample_with_rngs"):
                evaluate(policy, [1, 2], envs=2)
            workers.assert_not_called()
        with self.assertRaises(ValueError):
            evaluate(policy, [], envs=2)
        with self.assertRaises(ValueError):
            evaluate(policy, [1], envs=0)

    def test_categorical_policy_keeps_per_game_streams(self):
        seeds = [4, 8, 9]
        expected = []
        for seed in seeds:
            policy = fake_policy()
            policy.reset_seed(seed)
            expected.append([policy(np.array([[seed, step, 0]]))[0] for step in range(8)])
        batched = fake_policy()
        rngs = [np.random.default_rng(seed) for seed in seeds]
        actual = []
        for step in range(8):
            actual.append(batched.sample_with_rngs(np.array([[seed, step, 0] for seed in seeds]), rngs))
        np.testing.assert_array_equal(np.array(actual).T, expected)
        with self.assertRaises(ValueError):
            batched.sample_with_rngs(np.zeros((2, 3)), rngs)

    def test_invalid_worker_requests_send_nothing(self):
        workers = VectorEnv.__new__(VectorEnv)
        workers.pipes = [Mock(), Mock()]
        for call in [lambda: workers.step([1]),
                     lambda: workers.step([1, 2], indices=[0, 0]),
                     lambda: workers.step([1], indices=[2]),
                     lambda: workers.reset([1]),
                     lambda: workers.reset([1], indices=[-1])]:
            with self.assertRaises(ValueError):
                call()
        for pipe in workers.pipes:
            pipe.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
