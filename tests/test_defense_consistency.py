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

from rl.defense_consistency import ConsistencyLearner, consistency_loss
from rl.model import Learner


class ConsistencyTests(unittest.TestCase):
    def test_online_choice_importance_weights_mask_and_stop_gradient(self):
        online = mx.array([[1., 3.], [4., 2.], [5., 1.]])
        target = mx.array([[100., 1.], [1., 100.], [9., 0.]])
        discounts, weights = mx.array([.5, 0., .5]), mx.array([1., .5, 2.])
        loss, gap, active = consistency_loss(online, target, discounts, weights)
        self.assertAlmostEqual(loss.item(), 8.5/3, places=6)
        self.assertEqual(gap.item(), 2.)
        self.assertAlmostEqual(active.item(), 2/3, places=6)
        gradient = mx.grad(lambda q: consistency_loss(q, target, discounts, weights)[0])(online)
        np.testing.assert_allclose(np.array(gradient), [[0., 1/3], [0., 0.], [-2/3, 0.]], atol=1e-7)
        target_gradient = mx.grad(lambda q: consistency_loss(online, q, discounts, weights)[0])(target)
        np.testing.assert_array_equal(np.array(target_gradient), 0.)

    def test_learning_terminal_mask_preserves_td_loss_and_gradients(self):
        agent = ConsistencyLearner(seed=619, weight=2.)
        rng = np.random.default_rng(20)
        obs = mx.array(rng.integers(256, size=(3, 4, 16, 64), dtype=np.uint8))
        following = mx.array(rng.integers(256, size=(3, 4, 16, 64), dtype=np.uint8))
        batch = (obs, mx.array([0, 1, 2]), mx.array([1., 2., 3.]), following, mx.zeros(3), mx.ones(3))
        agent.target.value.bias += 2.
        joint = agent._joint_loss(agent.online, *batch)
        base = Learner._loss(agent, agent.online, *batch)
        self.assertEqual(joint[0].item(), base[0].item())
        self.assertEqual(joint[1][3].item(), 0.)
        import mlx.nn as nn
        _, a = nn.value_and_grad(agent.online, agent._joint_loss)(agent.online, *batch)
        _, b = nn.value_and_grad(agent.online, lambda model, *args: Learner._loss(agent, model, *args))(agent.online, *batch)
        for (name, x), (other, y) in zip(tree_flatten(a), tree_flatten(b), strict=True):
            self.assertEqual(name, other)
            np.testing.assert_array_equal(np.array(x), np.array(y))

    def test_real_update_keeps_target_fixed_and_priorities_td_only(self):
        agent = ConsistencyLearner(seed=83, weight=1.)
        agent.target.value.bias += 2.
        rng = np.random.default_rng(82)
        obs = rng.integers(256, size=(4, 4, 16, 64), dtype=np.uint8)
        following = rng.integers(256, size=obs.shape, dtype=np.uint8)
        batch = (obs, np.arange(4, dtype=np.int32), np.arange(4, dtype=np.float32), following,
                 np.array([.99, 0., .99, .99], np.float32), np.ones(4, np.float32))
        expected = Learner._loss(agent, agent.online, *(mx.array(v) for v in batch))[1][0]
        expected = np.array(expected)
        before = {k: np.array(v) for k,v in tree_flatten(agent.target.parameters())}
        loss, errors, q = agent.train(batch)
        np.testing.assert_allclose(errors, expected, rtol=1e-5, atol=1e-5)
        stats = agent.consistency_stats()
        self.assertEqual(stats['local_updates'], 1)
        self.assertEqual(stats['last_bootstrapping_fraction'], .75)
        self.assertGreater(stats['last_weighted_consistency_loss'], 0.)
        self.assertAlmostEqual(loss, stats['last_weighted_td_loss']+stats['last_weighted_consistency_loss'], places=5)
        self.assertTrue(np.isfinite([loss, q]).all())
        for name, value in tree_flatten(agent.target.parameters()):
            np.testing.assert_array_equal(before[name], np.array(value))
        agent.sync_target()
        for (name,x),(other,y) in zip(tree_flatten(agent.online.parameters()),tree_flatten(agent.target.parameters()),strict=True):
            self.assertEqual(name,other)
            np.testing.assert_array_equal(np.array(x),np.array(y))

    def test_invalid_weights_and_incompatible_cli(self):
        for weight in [0., -1., 11., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                ConsistencyLearner(weight=weight)
        from rl import defense_dqn
        for extra in [['--tc-weight', '-1'], ['--tc-weight', 'nan'], ['--tc-weight', '11'],
                      ['--tc-weight', '1', '--quantiles', '32'], ['--tc-weight', '1', '--spr-weight', '1', '--compact-replay'],
                      ['--tc-weight', '1', '--learned-repeats', '1,4', '--n-step', '1']]:
            with patch.object(sys, 'argv', ['dqn', '--run', '/nonexistent/tc-test',
                    '--artifacts', '/nonexistent/tc-artifacts']+extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_learning_resume_and_standard_policy_replay(self):
        from rl.defense_learning import evaluate, load_policy, record_game, verify_policy_trace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/name),
                    '--artifacts', str(root/(name+'-artifacts'))]+extra, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            first = run('first', ['--tc-weight', '1', '--envs', '2', '--capacity', '256', '--compact-replay',
                '--batch-size', '4', '--warmup', '16', '--steps', '128', '--eval-every', '1000',
                '--max-episode-steps', '12', '--train-every', '4', '--target-every', '2', '--mlx-cache-mb', '64'])
            second = run('resumed', ['--resume', str(root/'first/latest'), '--steps', '192'])
            self.assertGreater(first['updates'], 0)
            self.assertGreater(second['updates'], first['updates'])
            self.assertEqual(second['config']['tc_weight'], 1.)
            self.assertEqual(second['consistency']['local_updates'], second['updates']-first['updates'])
            self.assertEqual(len(list((root/'resumed/latest').glob('*'))), 4)
            policy, config = load_policy(root/'resumed/latest/model.safetensors')
            evaluation = evaluate(policy, [10000, 10001], envs=2)
            self.assertEqual(evaluation['complete_games'], 2)
            for game in evaluation['games']:
                recorded = record_game(policy, game['seed'], tstates=config['tstates'], max_steps=0)
                self.assertEqual(recorded[3], game)
            self.assertTrue(verify_policy_trace(root/'resumed/latest/model.safetensors', *recorded[:3], recorded[3])['verified'])


if __name__ == '__main__':
    unittest.main()
