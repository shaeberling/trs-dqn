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
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from rl.ppo import PPO
from rl.reference import reference_kl


class ValueCoefficientTests(unittest.TestCase):
    def test_default_loss_and_gradients_exactly_preserve_old_formula(self):
        for anchor in [0., .2]:
            agent = PPO(seed=11, action_count=20, entropy=.002, reference_kl_weight=anchor)
            obs = np.random.default_rng(4).integers(128, 192, (4, 4, 16, 64), dtype=np.uint8)
            actions, logp, values = agent.act(obs, np.random.default_rng(3))
            args = tuple(mx.array(x) for x in (obs, actions, logp, [1., -.5, .1, -1.], values+.7))
            reference = mx.full((4, 20), -np.log(20)) if anchor else None

            def old_loss(model, obs, actions, old_logp, advantages, returns, reference_log_probs):
                logits, values = model.policy_value(obs)
                log_probs = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
                logp = mx.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
                ratio = mx.exp(logp-old_logp)
                actor = -mx.mean(mx.minimum(ratio*advantages, mx.clip(ratio, .8, 1.2)*advantages))
                critic = .5*mx.mean(mx.square(values-returns))
                entropy = -mx.mean(mx.sum(mx.exp(log_probs)*log_probs, axis=-1))
                loss = actor+.5*critic-agent.entropy*entropy
                return loss+anchor*reference_kl(log_probs, reference_log_probs) if anchor else loss

            (new, _), grads = nn.value_and_grad(agent.model, agent._loss)(agent.model, *args, reference)
            old, expected = nn.value_and_grad(agent.model, old_loss)(agent.model, *args, reference)
            np.testing.assert_array_equal(np.array(new), np.array(old))
            for (name, grad), (other, wanted) in zip(tree_flatten(grads), tree_flatten(expected), strict=True):
                self.assertEqual(name, other)
                np.testing.assert_array_equal(np.array(grad), np.array(wanted))

    def test_coefficient_scales_only_value_loss_gradient(self):
        obs = mx.array(np.random.default_rng(9).integers(128, 192, (4, 4, 16, 64), dtype=np.uint8))
        agents = [PPO(seed=11, action_count=20, value_coefficient=c) for c in [0., .1, .5]]
        logits, values = agents[0].model.policy_value(obs)
        logp = logits-mx.logsumexp(logits, axis=1, keepdims=True)
        args = (obs, mx.zeros(4, mx.int32), logp[:, 0], mx.array([1., -1., .5, -.5]), values+1.)
        results = [nn.value_and_grad(a.model, a._loss)(a.model, *args) for a in agents]
        for ((loss, metrics), _) in results[1:]:
            for actual, expected in zip(metrics, results[0][0][1], strict=True):
                np.testing.assert_array_equal(np.array(actual), np.array(expected))
        grads = [dict(tree_flatten(g)) for _, g in results]
        for name in grads[0]:
            base, small, normal = [np.array(g[name]) for g in grads]
            np.testing.assert_allclose(small-base, .2*(normal-base), atol=3e-6, rtol=3e-5)
            if name.startswith('advantage.'):
                np.testing.assert_array_equal(small, normal)

    def test_invalid_coefficient_rejected_before_run_creation(self):
        from rl import defense_train
        for value in [True, -1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PPO(value_coefficient=value)
        with tempfile.TemporaryDirectory() as tmp:
            for value in ['-1', 'nan', 'inf']:
                args = ['train', '--run', tmp+'/run', '--value-coefficient='+value]
                with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    defense_train.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((Path(tmp)/'run').exists())

    def test_real_training_resume_inheritance_override_and_default_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [sys.executable, '-m', 'rl.defense_train', '--envs', '2', '--rollout', '8',
                    '--batch-size', '16', '--epochs', '1', '--mlx-cache-mb', '128']

            def run(name, extra):
                command = base+['--run', str(root/name), '--artifacts', str(root/(name+'-artifacts'))]+extra
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())

            default = run('default', ['--steps', '32'])
            explicit = run('explicit', ['--steps', '32', '--value-coefficient', '.5'])
            for name in ['model.safetensors', 'optimizer.npz']:
                a, b = [mx.load(str(root/p/'latest'/name)) for p in ['default', 'explicit']]
                self.assertEqual(a.keys(), b.keys())
                for key in a:np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            self.assertEqual(default['rng'], explicit['rng'])
            small = run('small', ['--steps', '32', '--value-coefficient', '.1'])
            inherited = run('inherited', ['--resume', str(root/'small/latest'), '--steps', '48'])
            self.assertEqual(inherited['config']['value_coefficient'], .1)
            changed = run('changed', ['--resume', str(root/'small/latest'), '--steps', '48',
                                      '--value-coefficient', '.2'])
            self.assertEqual(changed['config']['value_coefficient'], .2)
            # Legacy checkpoints without this key still mean the old .5 weight.
            legacy = root/'default/latest/state.json'
            old = json.loads(legacy.read_text());del old['config']['value_coefficient']
            legacy.write_text(json.dumps(old))
            resumed = run('legacy', ['--resume', str(legacy.parent), '--steps', '48'])
            self.assertEqual(resumed['config']['value_coefficient'], .5)
            self.assertEqual(small['steps'], 32)


if __name__ == '__main__':
    unittest.main()
