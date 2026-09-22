from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_model import WorldLearner, WorldModel
from rl.defense_world_overshoot import latent_overshoot, validate


class OvershootingTests(unittest.TestCase):
    def test_action_target_alignment_and_complete_windows(self):
        class Toy:
            def __init__(self): self.actions = []
            def step(self, state, action, key, sample=True):
                self.actions.append(np.array(action))
                value = state[0]+action[:, None]
                return (value, value, value, mx.ones_like(value)), (value, mx.ones_like(value))
        actions = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]], np.int32)
        cumulative = np.concatenate([np.zeros((2,1)), np.cumsum(actions, axis=1)], axis=1)[...,None]
        states = tuple(mx.array(x) for x in (cumulative, cumulative, cumulative, np.ones_like(cumulative)))
        toy = Toy()
        self.assertEqual(float(latent_overshoot(toy, states, mx.array(actions), mx.random.key(3), 1, 3)), 3.)
        for d, selected in enumerate(toy.actions):
            np.testing.assert_array_equal(selected, actions[:,1+d:4+d].reshape(-1))
        altered = list(states); altered[2] = altered[2]+10
        self.assertAlmostEqual(float(latent_overshoot(Toy(), altered, mx.array(actions), mx.random.key(3), 1, 3)), 50.)
        for distance, weight in ((0,1),(1,1),(17,1),(True,1),(4,-1),(4,float('nan'))):
            with self.assertRaises(ValueError): validate(distance, weight)
        with self.assertRaises(ValueError):
            latent_overshoot(toy, states, mx.array(actions), mx.random.key(0), 5, 3)

    def test_only_prior_rollout_gets_extra_gradient(self):
        mx.random.seed(71)
        model = WorldModel()
        frames = mx.full((1,7,16,64), 191, mx.uint8)
        actions = mx.array([[1,2,3,4,5,6]])
        def objective(m):
            states, _ = m.observe(frames, actions, mx.random.key(1))
            states = (states[0], states[1], states[2]+10, states[3])
            return latent_overshoot(m, states, actions, mx.random.key(2), 1, 3)
        loss, grads = nn.value_and_grad(model, objective)(model)
        self.assertTrue(np.isfinite(float(loss)))
        flat = dict(tree_flatten(grads))
        self.assertTrue(any(np.any(np.array(v)!=0) for k,v in flat.items() if k.startswith('prior.')))
        for k, value in flat.items():
            if k.startswith(('encoder.', 'embedding.', 'posterior.', 'decoder.', 'reward.', 'continue_logit.')):
                np.testing.assert_array_equal(np.array(value), 0)

    def test_full_resume_and_explicit_objective_change(self):
        rng = np.random.default_rng(45)
        batch = (rng.integers(128,192,(1,7,16,64),dtype=np.uint8),
                 rng.integers(20,size=(1,6),dtype=np.int32), np.zeros((1,6),np.float32),
                 np.ones((1,6),np.float32))
        learner = WorldLearner(seed=8, burn=1, overshoot_distance=3, overshoot_weight=1.)
        self.assertIn('overshoot_kl', learner.train(batch))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint'
            learner.save(path,rng,{'overshooting':learner.overshooting})
            other = WorldLearner(seed=1,burn=1,overshoot_distance=3,overshoot_weight=1.)
            other_rng = np.random.default_rng(0)
            other.restore(path,other_rng)
            self.assertEqual(rng.bit_generator.state,other_rng.bit_generator.state)
            self.assertEqual(learner.train(batch),other.train(batch))
            for left,right in ((learner.model.parameters(),other.model.parameters()),
                               (learner.optimizer.state,other.optimizer.state), (learner.random,other.random)):
                a,b=dict(tree_flatten(left)),dict(tree_flatten(right))
                self.assertEqual(a.keys(),b.keys())
                for key in a: np.testing.assert_array_equal(np.array(a[key]),np.array(b[key]))
            changed = WorldLearner(seed=2,burn=1)
            with self.assertRaisesRegex(ValueError,'objective differs'):
                changed.restore(path,np.random.default_rng(1))
            changed.restore(path,np.random.default_rng(1),allow_objective_change=True)
            self.assertNotIn('overshoot_kl',changed.train(batch))


if __name__ == '__main__':
    unittest.main()
