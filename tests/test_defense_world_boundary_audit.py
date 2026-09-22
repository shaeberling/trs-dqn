import unittest

import mlx.core as mx
import numpy as np

from rl.defense_world_boundary_audit import boundary_predictions
from rl.defense_world_model import WorldModel


class BoundaryAuditTests(unittest.TestCase):
    def test_forecasts_do_not_see_arrival_and_never_change_parameters(self):
        from mlx.utils import tree_flatten
        mx.random.seed(17)
        model = WorldModel()
        rng = np.random.default_rng(13)
        frames = rng.integers(128, 192, (2, 18, 16, 64), dtype=np.uint8)
        actions = rng.integers(20, size=(2, 17), dtype=np.int32)
        continuation = np.ones((2, 17), np.float32); continuation[:, -1] = 0
        batch = (frames, actions, np.zeros((2, 17), np.float32), continuation)
        before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        first = boundary_predictions(model, batch, context=1, draws=2)
        altered = frames.copy(); altered[:, -1] = 32
        second = boundary_predictions(model, (altered, *batch[1:]), context=1, draws=2)
        for mode in first['predictions']:
            for name in ('prior_1', 'prior_4', 'prior_8', 'prior_16'):
                self.assertEqual(first['predictions'][mode][name], second['predictions'][mode][name])
            self.assertNotEqual(first['predictions'][mode]['observed_arrival'],
                                second['predictions'][mode]['observed_arrival'])
        for k, v in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(before[k], np.array(v))
        with self.assertRaises(ValueError):
            boundary_predictions(model, batch, draws=0)


if __name__ == '__main__':
    unittest.main()
