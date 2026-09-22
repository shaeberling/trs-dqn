import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_bytes import ByteDecoder, ByteWorld, ByteWorldLearner
from rl.defense_world_model import WorldLearner, WorldModel


def exact(left, right):
    a,b = dict(tree_flatten(left)),dict(tree_flatten(right))
    assert a.keys() == b.keys()
    for key in a: np.testing.assert_array_equal(np.array(a[key]),np.array(b[key]))


class ByteWorldTests(unittest.TestCase):
    def batch(self):
        rng=np.random.default_rng(55)
        return (rng.integers(256,size=(1,5,16,64),dtype=np.uint8),
                rng.integers(20,size=(1,4),dtype=np.int32),
                np.array([[0.,.1,0.,.2]],np.float32),np.array([[1.,1.,1.,0.]],np.float32))

    def test_decoder_shape_alignment_and_original_terms(self):
        mx.random.seed(7)
        world,decoder=WorldModel(),ByteDecoder()
        system=ByteWorld(world,decoder)
        frames,actions,rewards,continuation=(mx.array(v) for v in self.batch())
        key=mx.random.key(9)
        core,terms=world.loss(frames,actions,rewards,continuation,key,burn=1)
        total,augmented=system.loss(frames,actions,rewards,continuation,key,1,2.)
        np.testing.assert_array_equal(np.array(terms),np.array(augmented[:4]))
        states,_=world.observe(frames,actions,key)
        logits=decoder(world.features(states)[:,2:])
        self.assertEqual(logits.shape,(1,3,16,64,256))
        expected=nn.losses.cross_entropy(logits,frames[:,2:].astype(mx.int32),reduction='mean')
        self.assertAlmostEqual(float(augmented[4]),float(expected),places=6)
        self.assertAlmostEqual(float(total),float(core+2*expected),places=5)

    def test_auxiliary_gradient_reaches_encoder_not_reward_head(self):
        learner=ByteWorldLearner(seed=4,burn=1)
        frames,actions,_,_=(mx.array(v) for v in self.batch())
        def objective(system):
            states,_=system.world.observe(frames,actions,mx.random.key(6))
            logits=system.decoder(system.world.features(states)[:,2:])
            return nn.losses.cross_entropy(logits,frames[:,2:].astype(mx.int32),reduction='mean')
        _,grads=nn.value_and_grad(learner.system,objective)(learner.system)
        flat=dict(tree_flatten(grads))
        for prefix in ('world.encoder.','world.posterior.','decoder.'):
            self.assertTrue(any(np.any(np.array(v)!=0) for k,v in flat.items() if k.startswith(prefix)))
        for k,v in flat.items():
            if k.startswith(('world.reward.','world.continue_logit.')):
                np.testing.assert_array_equal(np.array(v),0)

    def test_exact_extended_resume_and_explicit_full_parent_extension(self):
        batch=self.batch(); rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); parent=root/'parent'
            original=WorldLearner(seed=8,burn=1);original.train(batch)
            original.save(parent,rng,{'byte_reconstruction':None})
            learner=ByteWorldLearner(seed=21,burn=1)
            with self.assertRaisesRegex(ValueError,'objective differs'):
                learner.restore(parent,rng)
            learner.restore(parent,rng,allow_objective_change=True)
            self.assertEqual(learner.updates,1); self.assertEqual(learner.byte_updates,0)
            exact(original.model.parameters(),learner.model.parameters())
            exact(original.optimizer.state,learner.optimizer.state)
            exact(original.random,learner.random)
            stats=learner.train(batch);self.assertTrue(np.isfinite(list(stats.values())).all())
            saved=root/'extended';learner.save(saved,rng,{'byte_reconstruction':learner.byte_reconstruction})
            restored=ByteWorldLearner(seed=88,burn=1);other_rng=np.random.default_rng(0)
            restored.restore(saved,other_rng)
            self.assertEqual(rng.bit_generator.state,other_rng.bit_generator.state)
            self.assertEqual(learner.train(batch),restored.train(batch))
            for a,b in ((learner.model.parameters(),restored.model.parameters()),
                        (learner.decoder.parameters(),restored.decoder.parameters()),
                        (learner.optimizer.state,restored.optimizer.state),
                        (learner.byte_optimizer.state,restored.byte_optimizer.state),
                        (learner.random,restored.random)):
                exact(a,b)
            self.assertEqual(learner.byte_updates,restored.byte_updates)
            with self.assertRaisesRegex(ValueError,'byte reconstruction'):
                WorldLearner(burn=1).restore(saved,np.random.default_rng(0))
            # Acting remains loadable from the core-only file without auxiliary heads.
            acting=WorldModel();acting.load_weights(str(saved/'world.safetensors'))
            (saved/'byte-optimizer.npz').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                restored.restore(saved,np.random.default_rng(0))
        for weight in (0,-1,float('nan')):
            with self.assertRaises(ValueError):ByteWorldLearner(byte_weight=weight)

    def test_byte_forecasts_ignore_future_screens(self):
        from rl.defense_world_byte_audit import audit_bytes
        learner=ByteWorldLearner(seed=9,burn=1)
        batch=self.batch()
        first=audit_bytes(learner.model,learner.decoder,batch,context=1)
        changed=batch[0].copy();changed[:,2:]=32
        second=audit_bytes(learner.model,learner.decoder,(changed,*batch[1:]),context=1)
        for name,row in first['measurements'].items():
            if name.startswith('prior_'):
                self.assertEqual(row['predicted_bytes_sha256'],second['measurements'][name]['predicted_bytes_sha256'])
        observed=first['measurements']['observed']
        self.assertEqual(observed['cells'],3*16*64)
        self.assertEqual(observed['groups']['all']['cells'],observed['cells'])
        self.assertNotEqual(observed['cross_entropy'],second['measurements']['observed']['cross_entropy'])


if __name__=='__main__':
    unittest.main()
