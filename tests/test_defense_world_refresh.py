from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_imagination import ImaginationLearner
from rl.defense_learning import sha256
from rl.defense_world_model import WorldModel
from rl.defense_world_refresh import refresh_world


class WorldRefreshTests(unittest.TestCase):
    def test_retains_all_behavior_state_and_rejects_unrelated_world(self):
        learner = ImaginationLearner(seed=3, horizon=2)
        learner.train(learner.model.world.initial(2), mx.ones(2))
        def behavior():
            return {k: np.array(v) for k,v in tree_flatten([
                learner.model.actor.parameters(), learner.critic.parameters(), learner.target.parameters(),
                learner.actor_optimizer.state, learner.value_optimizer.state, learner.random])}
        before = behavior()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); old, new = root/'old', root/'new'
            old.mkdir(); new.mkdir()
            learner.model.world.save_weights(str(old/'world.safetensors'))
            mx.random.seed(999)
            updated = WorldModel(); updated.save_weights(str(new/'world.safetensors'))
            previous = dict(tstates=100000, observation_stride=1, dataset_sha256='old-data',
                            world_sha256=sha256(old/'world.safetensors'))
            current = dict(previous, dataset_sha256='union-data', world_sha256=sha256(new/'world.safetensors'))
            metadata = dict(args={'resume': str(old)}, previous_dataset_sha256='old-data')
            with self.assertRaisesRegex(ValueError, 'descend'):
                refresh_world(learner, previous, current, new, dict(metadata, args={'resume': str(new)}))
            with self.assertRaisesRegex(ValueError, 'dataset'):
                refresh_world(learner, previous, current, new, dict(metadata, previous_dataset_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                refresh_world(learner, previous, dict(current, world_sha256='wrong'), new, metadata)
            result = refresh_world(learner, previous, current, new, metadata)
            self.assertEqual(result['retained_actor_updates'], 1)
            self.assertEqual(learner.updates, 1)
            for key, value in before.items():
                np.testing.assert_array_equal(value, behavior()[key])
            actual = dict(tree_flatten(learner.model.world.parameters()))
            for key, value in tree_flatten(updated.parameters()):
                np.testing.assert_array_equal(np.array(value), np.array(actual[key]))


if __name__ == '__main__':
    unittest.main()
