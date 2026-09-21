import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from rl.env import ENVIRONMENT_VERSION, SHAPE
from rl.evaluate import load_policy
from rl.ppo import PPO
from rl.reference import ReferencePolicy, reference_kl, reference_metadata


class ReferenceTests(unittest.TestCase):
    def test_analytic_kl_and_gradients(self):
        logits = mx.array([[1., -2., 3., .1, -.4, .8], [1000., -1000., 0., 2., 3., 4.]])
        target = mx.array(np.log([[.1, .2, .3, .1, .1, .2], [.2, .1, .1, .2, .1, .3]]))
        def loss(student, teacher):
            return reference_kl(student-mx.logsumexp(student, axis=-1, keepdims=True), teacher)
        raw = np.array(logits)
        logp = raw-np.logaddexp.reduce(raw, axis=-1, keepdims=True)
        expected = np.mean(np.sum(np.exp(np.array(target))*(np.array(target)-logp), axis=-1))
        self.assertAlmostEqual(float(loss(logits, target).item()), float(expected), places=4)
        student_grad, teacher_grad = mx.grad(loss, argnums=(0, 1))(logits, target)
        np.testing.assert_allclose(np.array(student_grad),
                                   (np.exp(logp)-np.exp(np.array(target)))/2, atol=2e-7)
        np.testing.assert_array_equal(np.array(teacher_grad), np.zeros((2, 6)))
        self.assertAlmostEqual(float(reference_kl(target, target).item()), 0.)

    def test_metadata_validation_and_hash_identity(self):
        self.assertIsNone(reference_metadata(None, 0, 50000, 2))
        for weight in (-1, float('nan'), float('inf'), True, .1):
            with self.assertRaises(ValueError):
                reference_metadata(None, weight, 50000, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'model.safetensors'
            model.write_bytes(b'test identity only; never loaded as a model')
            config = dict(algorithm='ppo', environment_version=ENVIRONMENT_VERSION,
                          tstates=50000, observation_stride=2)
            (root/'state.json').write_text(json.dumps({'config': config}))
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            result = reference_metadata(model, .1, 50000, 2, digest)
            self.assertEqual(result['sha256'], digest)
            for timing, stride, expected in ((100000, 2, None), (50000, 1, None),
                                            (50000, 2, 'changed')):
                with self.assertRaises(ValueError):
                    reference_metadata(model, .1, timing, stride, expected)
            with self.assertRaises(ValueError):
                reference_metadata(model, 0, 50000, 2)
            config['algorithm'] = 'dqn'
            (root/'state.json').write_text(json.dumps({'config': config}))
            with self.assertRaises(ValueError):
                reference_metadata(model, .1, 50000, 2)

    def test_invalid_cli_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'must-not-exist'
            for value in ('nan', 'inf', '-1', '.1'):
                result = subprocess.run([sys.executable, '-m', 'rl.ppo', '--run', str(run),
                                         '--reference-kl-weight', value],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('reference', result.stderr)
                self.assertFalse(run.exists())

    def test_penalty_is_additive_and_requires_targets(self):
        agent = PPO(seed=11, reference_kl_weight=.1)
        rng = np.random.default_rng(7)
        obs = rng.integers(128, 192, size=(8, *SHAPE), dtype=np.uint8)
        actions, logp, values = agent.act(obs, rng)
        batch = tuple(mx.array(x) for x in (obs, actions, logp, np.ones(8, np.float32), values+1))
        targets = mx.full((8, 6), -np.log(6))
        anchored, aux = agent._loss(agent.model, *batch, targets)
        agent.reference_kl_weight = 0
        baseline, base_aux = agent._loss(agent.model, *batch)
        self.assertEqual(len(base_aux), 4)
        self.assertEqual(len(aux), 5)
        self.assertAlmostEqual(float(anchored.item()),
                               float(baseline.item())+.1*float(aux[-1].item()), places=6)
        agent.reference_kl_weight = .1
        with self.assertRaisesRegex(ValueError, 'requires rollout targets'):
            agent._loss(agent.model, *batch)

    def test_frozen_targets_optimizer_resume_and_standalone_inference(self):
        agent = PPO(seed=17, learning_rate=5e-6, reference_kl_weight=.1)
        rng = np.random.default_rng(23)
        obs = rng.integers(128, 192, size=(8, *SHAPE), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher, student = root/'teacher', root/'student'
            agent.save(teacher, {'config': {'algorithm': 'ppo'}})
            reference = ReferencePolicy(teacher/'model.safetensors')
            self.assertEqual(tree_flatten(reference.model.trainable_parameters()), [])
            before = {k: np.array(v).copy() for k, v in tree_flatten(reference.model.parameters())}
            targets = reference.targets(obs, 4)
            np.testing.assert_allclose(np.exp(targets).sum(axis=-1), 1, atol=2e-6)
            np.testing.assert_allclose(reference.targets(obs, 8), targets, atol=2e-6)
            with self.assertRaises(ValueError):
                reference.targets(obs, 0)
            actions, logp, values = agent.act(obs, rng)
            batch = tuple(mx.array(x) for x in (obs, actions, logp,
                np.linspace(-1, 1, 8, dtype=np.float32), values+1, targets))
            for _ in range(3):
                loss, aux = agent.update(*batch)
                mx.eval(loss, aux, agent.state)
            agent.save(student, {'config': {'algorithm': 'ppo',
                        'reference_policy': str(teacher/'model.safetensors'),
                        'reference_kl_weight': .1}})
            restored = PPO(seed=99, learning_rate=5e-6, reference_kl_weight=.1)
            restored.model.load_weights(str(student/'model.safetensors'))
            restored.optimizer.state = tree_unflatten(list(mx.load(str(student/'optimizer.npz')).items()))
            restored.compile()
            for _ in range(3):
                for learner in (agent, restored):
                    loss, aux = learner.update(*batch)
                    mx.eval(loss, aux, learner.state)
                    self.assertTrue(np.isfinite(float(loss.item())))
            for original, resumed in ((agent.model.parameters(), restored.model.parameters()),
                                      (agent.optimizer.state, restored.optimizer.state)):
                expected, actual = dict(tree_flatten(original)), dict(tree_flatten(resumed))
                self.assertEqual(expected.keys(), actual.keys())
                for name in expected:
                    np.testing.assert_allclose(np.array(actual[name]), np.array(expected[name]),
                                               rtol=1e-6, atol=1e-7, err_msg=name)
            for name, value in tree_flatten(reference.model.parameters()):
                np.testing.assert_array_equal(np.array(value), before[name])
            np.testing.assert_array_equal(reference.targets(obs, 4), targets)
            # The deployed learner needs only its own normal model checkpoint.
            agent.save(student, {'config': {'algorithm': 'ppo',
                        'reference_policy': str(teacher/'model.safetensors'),
                        'reference_kl_weight': .1}})
            (teacher/'model.safetensors').rename(teacher/'retained-source.safetensors')
            original, loaded = agent.policy(), load_policy(student/'model.safetensors')
            original.reset_seed(45)
            loaded.reset_seed(45)
            for _ in range(4):
                np.testing.assert_array_equal(original(obs), loaded(obs))


if __name__ == '__main__':
    unittest.main()
