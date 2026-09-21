import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_bootstrap import BootstrapLearner, BootstrapQ
from rl.defense_bootstrap_replay import BootstrapReplay
from rl.defense_learning import BOOTSTRAP_ALGORITHM, load_policy, policy_description
from rl.replay import NStep
from rl.train import checkpoint
from tests.test_defense_dqn import configuration


def config():
    return dict(configuration(), algorithm=BOOTSTRAP_ALGORITHM,
                bootstrap_heads=3, bootstrap_prior_scale=1.)


class DefenseBootstrapTests(unittest.TestCase):
    def test_mask_is_fixed_per_insert_and_follows_ring_slots_and_nstep(self):
        rng = np.random.default_rng(41)
        expected = np.random.default_rng(41)
        replay = BootstrapReplay(8, 3, .5, rng)
        frames = np.zeros((4, 16, 64), np.uint8)
        nstep = NStep(replay, n=2, gamma=.9)
        for i in range(12):
            nstep.append(frames+i, i, float(i), frames+i+1, True, False)
            np.testing.assert_array_equal(replay.masks[i % 8], expected.random(3) < .5)
        before = replay.masks.copy()
        rng_before = rng.bit_generator.state
        indices, batch = replay.sample(32, np.random.default_rng(9), .4)
        np.testing.assert_array_equal(batch[-1], before[indices])
        np.testing.assert_array_equal(replay.masks, before)
        self.assertEqual(rng_before, rng.bit_generator.state)
        self.assertTrue(np.any(before == 0))
        self.assertTrue(np.any(before == 1))
        self.assertEqual(batch[-1].shape, (32, 3))

    def test_masked_double_targets_use_corresponding_head(self):
        agent = BootstrapLearner(seed=3, heads=2)
        class Values:
            def __init__(self, table):
                self.table = mx.array(table)
            def head_values(self, states):
                return self.table[states]
        online = Values([[[1., 0.], [2., 0.]], [[2., 3.], [4., 1.]]])
        agent.target = Values([[[0., 0.], [0., 0.]], [[100., 4.], [8., 100.]]])
        # Head 0: y=2+.5*4=4, |TD|=3, Huber=2.5.
        # Head 1: y=2+.5*8=6, |TD|=4, Huber=3.5.
        args = (online, mx.array([0]), mx.array([0]), mx.array([2.]),
                mx.array([1]), mx.array([.5]), mx.array([1.]))
        loss, (errors, q) = agent._loss(*args, mx.array([[1., 0.]]))
        self.assertAlmostEqual(loss.item(), 2.5)
        np.testing.assert_allclose(np.array(errors), [3.5])
        self.assertAlmostEqual(q.item(), 1.5)
        self.assertAlmostEqual(agent._loss(*args, mx.array([[0., 1.]]))[0].item(), 3.5)
        self.assertEqual(agent._loss(*args, mx.array([[0., 0.]]))[0].item(), 0.)

    def test_prior_frozen_masked_head_gradients_target_sync_and_reload(self):
        agent = BootstrapLearner(seed=4, heads=3)
        rng = np.random.default_rng(5)
        obs = rng.integers(128, 192, (4, 4, 16, 64), dtype=np.uint8)
        original = {k: np.array(v) for k, v in tree_flatten(agent.online.parameters())}
        self.assertFalse(any(k.startswith('prior.') for k, _ in tree_flatten(agent.online.trainable_parameters())))
        masks = np.tile([1., 0., 1.], (4, 1)).astype(np.float32)
        batch = (obs, np.array([0, 1, 18, 19], np.int32), np.full(4, 3., np.float32),
                 obs, np.zeros(4, np.float32), np.ones(4, np.float32), masks)
        loss, errors, q = agent.train(batch)
        self.assertTrue(np.isfinite([loss, q]).all())
        self.assertEqual(errors.shape, (4,))
        after = {k: np.array(v) for k, v in tree_flatten(agent.online.parameters())}
        changed = {k for k in original if not np.array_equal(original[k], after[k])}
        self.assertTrue(any(k.startswith('learned.conv.') for k in changed))
        self.assertTrue(any(k.startswith('learned.hidden.0.') for k in changed))
        self.assertFalse(any(k.startswith('prior.') for k in changed))
        self.assertFalse(any(k.startswith('learned.'+layer+'.1.') for k in changed
                             for layer in ['hidden', 'value', 'advantage']))
        for k, v in tree_flatten(agent.target.parameters()):
            np.testing.assert_array_equal(original[k], np.array(v))
        head_values = np.array(agent.online.head_values(mx.array(obs)))
        np.testing.assert_array_equal(agent.actions(obs, np.array([0, 1, 2, 1])),
                                      head_values[np.arange(4), [0, 1, 2, 1]].argmax(axis=1))
        np.testing.assert_array_equal(agent.actions(obs), head_values.mean(axis=1).argmax(axis=1))
        with self.assertRaises(ValueError):
            agent.actions(obs, np.array([3, 0, 0, 0]))
        agent.sync_target()
        for k, v in tree_flatten(agent.target.parameters()):
            np.testing.assert_array_equal(after[k], np.array(v))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(agent, root, dict(config=config(), steps=4))
            policy, cfg = load_policy(root/'model.safetensors')
            np.testing.assert_array_equal(policy(obs), agent.actions(obs))
            self.assertIn('greedy mean', policy_description(cfg))
            loaded = BootstrapQ(20, 3, 1.)
            loaded.load_weights(str(root/'model.safetensors'))
            np.testing.assert_allclose(np.array(loaded.head_values(mx.array(obs))), head_values,
                                       rtol=1e-6, atol=1e-6)
            with self.assertRaisesRegex(ValueError, 'Temperature'):
                load_policy(root/'model.safetensors', temperature=.5)

    def test_invalid_bootstrap_configuration_rejected(self):
        from rl import defense_dqn
        with tempfile.TemporaryDirectory() as tmp:
            base = ['dqn', '--run', tmp+'/run', '--artifacts', tmp+'/artifacts']
            for extra in [['--bootstrap-heads', '1'], ['--bootstrap-heads', '-2'],
                          ['--bootstrap-prior-scale', 'nan'], ['--bootstrap-prior-scale', '-1'],
                          ['--bootstrap-probability', '0'], ['--bootstrap-probability', '1.1'],
                          ['--bootstrap-epsilon', '-.1']]:
                with self.subTest(extra=extra), patch.object(sys, 'argv', base+extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                    defense_dqn.main()
                self.assertEqual(e.exception.code, 2)

    def test_training_head_persists_across_ship_loss_until_complete_game(self):
        from rl import defense_dqn
        observed = []
        original_actions = BootstrapLearner.actions

        def capture(agent, observations, heads=None):
            observed.append(heads.copy())
            return original_actions(agent, observations, heads)

        class Workers:
            def __init__(self, *args, **kwargs):
                self.observations = np.full((2, 4, 16, 64), 128, np.uint8)
                self.tick = 0
            def runtime(self):
                return []
            def close(self):
                pass
            def step(self, actions):
                self.tick += 1
                terminal = self.tick % 3 == 0
                # A visible ship loss after every action must NOT change heads.
                return [(obs, 0., terminal, False, {'life_lost': True},
                         obs.copy() if terminal else None) for obs in self.observations]

        with tempfile.TemporaryDirectory() as tmp:
            args = ['dqn', '--run', tmp+'/run', '--artifacts', tmp+'/artifacts',
                    '--steps', '12', '--envs', '2', '--capacity', '32', '--warmup', '0',
                    '--train-every', '1000', '--bootstrap-heads', '3', '--bootstrap-epsilon', '0']
            with patch.object(sys, 'argv', args), patch.object(defense_dqn, 'VectorEnv', Workers), \
                    patch.object(BootstrapLearner, 'actions', capture), contextlib.redirect_stdout(io.StringIO()):
                defense_dqn.main()
        self.assertEqual(len(observed), 5)
        expected_rng = np.random.default_rng(97)
        expected_heads = expected_rng.integers(3, size=2)
        expected_rng.integers(20, size=2)  # first empty-replay action is random.
        for step, heads in enumerate(observed, 2):
            np.testing.assert_array_equal(heads, expected_heads)
            expected_rng.random(2)  # epsilon decision; epsilon is zero here.
            expected_rng.integers(20, size=0)
            if step % 3 == 0:
                expected_heads = np.array([expected_rng.integers(3) for _ in range(2)])

    def test_real_emulator_train_resume_and_exact_zero_update_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            args = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/'run'),
                    '--artifacts', str(p/'artifacts'), '--steps', '32', '--envs', '2',
                    '--capacity', '32', '--warmup', '4', '--train-every', '4',
                    '--batch-size', '4', '--n-step', '2', '--target-every', '4',
                    '--bootstrap-heads', '3', '--mlx-cache-mb', '128']
            result = subprocess.run(args, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            state = json.loads((p/'run/latest/state.json').read_text())
            self.assertEqual(state['config']['algorithm'], BOOTSTRAP_ALGORITHM)
            self.assertGreater(state['updates'], 0)
            rows = [json.loads(l) for l in (p/'run/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(not w['mlx_loaded'] for r in rows if r['event']=='workers_started'
                                for w in r['workers']))
            for target, limit in [('clone', '32'), ('resume', '64')]:
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(p/target),
                                         '--artifacts', str(p/(target+'-artifacts')),
                                         '--resume', str(p/'run/latest'), '--steps', limit],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            clone = json.loads((p/'clone/latest/state.json').read_text())
            for field in ['steps', 'updates', 'episodes', 'rng', 'bootstrap_rng']:
                self.assertEqual(clone[field], state[field])
            for file in ['model.safetensors', 'target.safetensors', 'optimizer.npz']:
                original = mx.load(str(p/'run/latest'/file))
                copied = mx.load(str(p/'clone/latest'/file))
                self.assertEqual(set(original), set(copied))
                for key in original:
                    np.testing.assert_array_equal(np.array(original[key]), np.array(copied[key]))
            later = json.loads((p/'resume/latest/state.json').read_text())
            self.assertEqual(later['steps'], 64)
            self.assertGreater(later['updates'], state['updates'])
            initial_weights = mx.load(str(p/'run/latest/model.safetensors'))
            resumed_weights = mx.load(str(p/'resume/latest/model.safetensors'))
            for key in initial_weights:
                if key.startswith('prior.'):
                    np.testing.assert_array_equal(np.array(initial_weights[key]), np.array(resumed_weights[key]))


if __name__ == '__main__':
    unittest.main()
