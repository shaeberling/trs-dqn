import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from rl.defense import GAME_SHA256, ENVIRONMENT_VERSION, action_names
from rl.defense_imagination import (ALGORITHM, ARCHITECTURE, ImaginationLearner,
    WorldActor, WorldPolicy, acting_policy, lambda_returns, survival_weights)
from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace


def config():
    return dict(game='defense', game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
        action_names=list(action_names()), allow_enter=False, tstates=100000,
        observation_stride=1, eval_max_steps=0, algorithm=ALGORITHM,
        architecture=ARCHITECTURE, world_hidden=128, world_stochastic=32)


def arrays(model):
    return {k: np.array(v).copy() for k, v in tree_flatten(model)}


class ImaginationTests(unittest.TestCase):
    def test_returns_alignment_terminal_and_weights(self):
        rewards = mx.array([[1., 2., 3.]])
        discounts = mx.array([[.5, 0., .5]])
        values = mx.array([[99., 10., 20., 30.]])
        np.testing.assert_allclose(np.array(lambda_returns(rewards, discounts, values, 1)), [[2., 2., 18.]])
        np.testing.assert_allclose(np.array(lambda_returns(rewards, discounts, values, 0)), [[6., 2., 18.]])
        np.testing.assert_array_equal(np.array(survival_weights(discounts)), [[1., .5, 0.]])
        for kwargs in ({'horizon': 0}, {'horizon': True}, {'gamma': 0}, {'entropy': float('nan')}):
            with self.assertRaises(ValueError):
                ImaginationLearner(**kwargs)

    def test_score_gradient_in_a_tiny_model(self):
        # Synthetic dynamics, solely a gradient unit test, never native-game data.
        class ToyWorld(nn.Module):
            @staticmethod
            def features(state):
                return mx.concatenate(state[:2], axis=-1)
            def step(self, state, action, key, sample=True):
                h = mx.concatenate([(action == 3)[:, None].astype(mx.float32),
                                     mx.zeros((len(action), 127))], axis=1)
                return (h, *state[1:]), None
            def reward(self, feature):
                return feature[:, :1]
            def continue_logit(self, feature):
                return mx.full((len(feature), 1), -100.)
        learner = ImaginationLearner(world=ToyWorld(), seed=13, horizon=1, entropy=0)
        start = (mx.zeros((64, 128)), mx.zeros((64, 32)), mx.zeros((64, 32)), mx.ones((64, 32)))
        before = float(mx.softmax(learner.model.actor(mx.zeros((1, 160))))[0, 3])
        for _ in range(80):
            learner.train(start, mx.ones(64))
        after = float(mx.softmax(learner.model.actor(mx.zeros((1, 160))))[0, 3])
        self.assertGreater(after, before*1.2)

    def test_frozen_world_exact_resume_and_target_sync(self):
        learner = ImaginationLearner(seed=7, horizon=3)
        rng = np.random.default_rng(13)
        frames = rng.integers(128, 192, (2, 5, 16, 64), dtype=np.uint8)
        actions = rng.integers(20, size=(2, 4), dtype=np.int32)
        continuation = np.array([[1, 0, 1, 0], [1, 1, 1, 0]], np.float32)
        start, alive = learner.starts(frames, actions, continuation, burn=2)
        np.testing.assert_array_equal(np.array(alive), [0, 1, 1, 1])
        world_before = arrays(learner.model.world.parameters())
        actor_before = arrays(learner.model.actor.parameters())
        learner.train(start, alive)
        self.assertTrue(any(not np.array_equal(v, arrays(learner.model.actor.parameters())[k]) for k,v in actor_before.items()))
        for k, value in world_before.items():
            np.testing.assert_array_equal(value, arrays(learner.model.world.parameters())[k])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint'
            learner.save(path, config(), rng)
            other = ImaginationLearner(seed=999, horizon=3)
            other_rng = np.random.default_rng(0)
            self.assertEqual(other.restore(path, other_rng), config())
            np.testing.assert_array_equal(rng.integers(100, size=8), other_rng.integers(100, size=8))
            self.assertEqual(learner.train(start, alive), other.train(start, alive))
            for a,b in ((learner.model.parameters(), other.model.parameters()),
                        (learner.actor_optimizer.state, other.actor_optimizer.state),
                        (learner.value_optimizer.state, other.value_optimizer.state)):
                left, right = arrays(a), arrays(b)
                self.assertEqual(left.keys(), right.keys())
                for k in left:
                    np.testing.assert_array_equal(left[k], right[k])
            learner.updates = 99
            learner.train(start, alive)
            for k,v in arrays(learner.critic.parameters()).items():
                np.testing.assert_array_equal(v, arrays(learner.target.parameters())[k])
            (path/'random.npz').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                other.restore(path, other_rng)

    def test_game_memory_and_own_action_feedback(self):
        seen = []
        def infer(obs, memory):
            seen.append(memory.copy())
            updated = memory.copy(); updated[:, :160] += 1
            logits = np.full((len(obs), 20), -100.); logits[:, 4] = 100.
            return logits, updated
        policy = WorldPolicy(infer)
        a, b = np.random.default_rng(1), np.random.default_rng(2)
        obs = np.zeros((2, 4, 16, 64), np.uint8)
        np.testing.assert_array_equal(policy.sample_with_rngs(obs, [a,b]), [4,4])
        policy.sample_with_rngs(obs[:1], [b])
        policy.sample_with_rngs(obs, [b,a])
        np.testing.assert_array_equal(seen[0], 0)
        np.testing.assert_array_equal(seen[1][:,160], [5])
        np.testing.assert_array_equal(seen[2][:,0], [2,1])
        policy.reset_seed(1); policy(obs[:1])
        np.testing.assert_array_equal(seen[-1], 0)
        with self.assertRaises(ValueError):
            policy.sample_with_rngs(obs, [a,a])

    def test_native_loader_parallel_serial_and_full_replay(self):
        learner = ImaginationLearner(seed=19, horizon=2)
        start = learner.model.world.initial(2)
        learner.train(start, mx.ones(2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint'
            learner.save(path, config(), np.random.default_rng(0))
            checkpoint = path/'model.safetensors'
            policy, saved = load_policy(checkpoint)
            parallel = evaluate(policy, [10000, 10001], tstates=100000, max_steps=0, envs=2)
            policy, _ = load_policy(checkpoint)
            serial = evaluate(policy, [10000, 10001], tstates=100000, max_steps=0, envs=1)
            self.assertEqual(parallel, serial)
            frames, actions, rewards, result, events = record_game(policy, 10000, tstates=100000, max_steps=0)
            self.assertEqual(result, parallel['games'][0])
            verified = verify_policy_trace(checkpoint, frames, actions, rewards, result)
            self.assertTrue(verified['verified'])
            self.assertEqual(verified['verified_actions'], len(actions))
            saved['architecture'] = 'unknown'
            state = json.loads((path/'state.json').read_text()); state['config'] = saved
            (path/'state.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                load_policy(checkpoint)


if __name__ == '__main__':
    unittest.main()
