import unittest
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_world_byte_probe import FrozenByteProbe
from rl.defense_world_bytes import ByteWorldLearner


class FrozenReadoutTests(unittest.TestCase):
    def test_readout_changes_but_world_and_its_optimizer_do_not(self):
        learner=ByteWorldLearner(seed=4,burn=1)
        def arrays(value):return {k:np.array(v).copy() for k,v in tree_flatten(value)}
        world,optimizer,head=arrays(learner.model.parameters()),arrays(learner.optimizer.state),arrays(learner.decoder.parameters())
        rng=np.random.default_rng(8)
        batch=(rng.integers(256,size=(1,5,16,64),dtype=np.uint8),rng.integers(20,size=(1,4),dtype=np.int32))
        probe=FrozenByteProbe(learner)
        for _ in range(3):probe.train(batch)
        for original,current in ((world,arrays(learner.model.parameters())),(optimizer,arrays(learner.optimizer.state))):
            self.assertEqual(original.keys(),current.keys())
            for key in original:np.testing.assert_array_equal(original[key],current[key])
        after=arrays(learner.decoder.parameters())
        self.assertTrue(any(not np.array_equal(value,after[key]) for key,value in head.items()))
        self.assertEqual(probe.updates,3)
        self.assertEqual(learner.updates,0)


if __name__=='__main__':unittest.main()
