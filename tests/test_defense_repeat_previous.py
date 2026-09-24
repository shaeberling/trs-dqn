import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from rl.defense_repeat_previous import (CONTINUE, POLICY_ACTION_NAMES,
                                       RepeatPreviousActions, RepeatPreviousPolicy,
                                       initialize_repeat_policy)
from rl.defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256


class SequencePolicy:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.seeds = []
        self.calls = []

    def reset_seed(self, seed):
        self.seeds.append(seed)

    def sample_with_rngs(self, obs, rngs):
        self.calls.append(tuple(rngs))
        return np.asarray(next(self.batches), np.int32)

    def __call__(self, obs):
        return np.asarray(next(self.batches), np.int32)


class RepeatPreviousTests(unittest.TestCase):
    def test_profile_and_training_executor_use_only_own_prior_key(self):
        self.assertEqual(POLICY_ACTION_NAMES[:-1], action_names(False))
        self.assertEqual(len(POLICY_ACTION_NAMES), 21)
        executor = RepeatPreviousActions(3)
        np.testing.assert_array_equal(executor.execute([CONTINUE, 4, 3]), [0, 4, 3])
        np.testing.assert_array_equal(executor.execute([2, CONTINUE, CONTINUE]), [2, 4, 3])
        executor.reset(np.array([False, True, False], dtype=bool))
        np.testing.assert_array_equal(executor.execute([CONTINUE]*3), [2, 0, 3])

    def test_invalid_choices_or_boundary_shapes_rejected(self):
        executor = RepeatPreviousActions(2)
        for choices in ([21, 0], [-1, 0], [0], [0.0, 1.0], [True, False]):
            with self.assertRaises(ValueError):
                executor.execute(choices)
        with self.assertRaises(ValueError):
            executor.reset([True])
        with self.assertRaises(ValueError):
            RepeatPreviousActions(0)

    def test_parallel_policy_keeps_separate_games_and_resets_on_life_loss(self):
        source = SequencePolicy([[4, 3], [CONTINUE, CONTINUE], [CONTINUE, CONTINUE]])
        policy = RepeatPreviousPolicy(source)
        left, right = object(), object()
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [left, right]), [4, 3])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [right, left]), [3, 4])
        policy.observe_boundaries(np.array([True, False]), [right, left])
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [left, right]), [4, 0])
        self.assertEqual(source.calls[1], (right, left))

    def test_serial_replay_seed_clears_previous_key(self):
        source = SequencePolicy([[4], [CONTINUE], [CONTINUE]])
        policy = RepeatPreviousPolicy(source)
        obs = np.zeros((1, 4, 16, 64), np.uint8)
        np.testing.assert_array_equal(policy(obs), [4])
        np.testing.assert_array_equal(policy(obs), [4])
        policy.reset_seed(123)
        np.testing.assert_array_equal(policy(obs), [0])
        self.assertEqual(source.seeds[-1], 123)

    def test_initialization_preserves_all_twenty_parent_logits_and_value(self):
        import mlx.core as mx
        from rl.model import QNetwork

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            original = QNetwork(action_count=20)
            original.save_weights(str(source/"model.safetensors"))
            config = dict(game="defense", game_sha256=GAME_SHA256,
                          environment_version=ENVIRONMENT_VERSION, algorithm="ppo",
                          action_names=list(action_names(False)), tstates=100000,
                          observation_stride=1)
            (source/"state.json").write_text(json.dumps(dict(config=config, steps=123)))
            extended = QNetwork(action_count=21)
            metadata = initialize_repeat_policy(extended, source, tstates=100000,
                                                observation_stride=1)
            observation = mx.array(np.zeros((2, 4, 16, 64), np.uint8))
            before, value_before = original.policy_value(observation)
            after, value_after = extended.policy_value(observation)
            np.testing.assert_allclose(np.array(after[:, :20]), np.array(before), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array(after[:, 20]), np.array(mx.mean(before, axis=1)),
                                       rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array(value_after), np.array(value_before), rtol=1e-5, atol=1e-5)
            self.assertEqual(metadata["source_training_steps"], 123)
            with self.assertRaises(ValueError):
                initialize_repeat_policy(QNetwork(21), source, tstates=50000,
                                         observation_stride=1)


if __name__ == "__main__":
    unittest.main()
