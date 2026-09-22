import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.defense_imagination import ImaginationLearner
from rl.defense_learning import load_policy, record_game
from rl.defense_world_actor_data import collect
from rl.defense_world_data import Sequences
from tests.test_defense_imagination import config


class ActorCollectionTests(unittest.TestCase):
    def test_native_collection_memory_actions_splits_and_full_reproduction(self):
        learner = ImaginationLearner(seed=31, horizon=2)
        learner.train(learner.model.world.initial(2), mx.ones(2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, data = root/'checkpoint', root/'data'
            learner.save(checkpoint, config(), np.random.default_rng(0))
            rows = collect(checkpoint/'model.safetensors', data, games=2, heldout=1, seed=91000)
            self.assertEqual([r['split'] for r in rows], ['train', 'heldout'])
            self.assertEqual(len(Sequences(data, 'train', 32).episodes), 1)
            self.assertEqual(len(Sequences(data, 'heldout', 32).episodes), 1)
            policy, _ = load_policy(checkpoint/'model.safetensors')
            for row in rows:
                # This independent reproduction is a test only; no result from
                # it is used to create/modify the fresh training collection.
                frames, actions, rewards, result, _ = record_game(policy, row['seed'], tstates=100000, max_steps=0)
                with np.load(data/row['file']) as episode:
                    np.testing.assert_array_equal(episode['frames'], frames)
                    np.testing.assert_array_equal(episode['actions'], actions)
                    np.testing.assert_array_equal(episode['rewards'], rewards)
                    self.assertEqual(float(episode['continuation'][-1]), 0.)
                self.assertEqual(row['result']['score'], result['score'])
            with self.assertRaises(ValueError):
                collect(checkpoint/'model.safetensors', data, 2, 1)
            (checkpoint/'model.safetensors').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                collect(checkpoint/'model.safetensors', root/'bad', 2, 1)
            self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
