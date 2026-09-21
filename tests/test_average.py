import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from rl.average import average_checkpoints, mean_parameters
from rl.evaluate import load_policy
from rl.model import QNetwork


class AveragingTests(unittest.TestCase):
    def test_uniform_parameter_mean(self):
        inputs = [{"x": mx.array([1., 3.])}, {"x": mx.array([3., 9.])},
                  {"x": mx.array([8., 0.])}]
        result = mean_parameters(inputs)
        np.testing.assert_array_equal(np.array(result["x"]), [4., 4.])
        np.testing.assert_array_equal(np.array(inputs[0]["x"]), [1., 3.])

    def test_single_parameter_set_is_identity(self):
        values = mx.array([-.125, 1., 1e-5])
        np.testing.assert_array_equal(np.array(mean_parameters([{"x": values}])["x"]),
                                      np.array(values))

    def test_reject_invalid_parameter_sets(self):
        valid = {"x": mx.array([1.])}
        for inputs in ([], [{}], [valid, {"y": mx.array([1.])}],
                       [valid, {"x": mx.array([1., 2.])}],
                       [{"x": mx.array([1], dtype=mx.int32)}],
                       [{"x": mx.array([float("nan")])}],
                       [{"x": mx.array([float("inf")])}]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                mean_parameters(inputs)

    def checkpoint(self, root, name, *, config=None, steps=100):
        directory = root/name
        directory.mkdir()
        model = QNetwork()
        model.save_weights(str(directory/"model.safetensors"))
        state = dict(steps=steps, config=config or dict(algorithm="ppo", run="same-run", tstates=100000))
        (directory/"state.json").write_text(json.dumps(state))
        return directory/"model.safetensors"

    def test_identity_round_trip_and_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.checkpoint(root, "source")
            original = source.read_bytes()
            out = root/"average"
            state = average_checkpoints([source], out)
            a, b = mx.load(str(source)), mx.load(str(out/"model.safetensors"))
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(state["evaluation_only"])
            self.assertFalse(state["resume_supported"])
            self.assertEqual(state["averaging"]["additional_training_actions"], 0)
            self.assertFalse((out/"optimizer.npz").exists())
            obs = np.zeros((3, 4, 16, 64), np.uint8)
            np.testing.assert_array_equal(load_policy(source)(obs),
                                          load_policy(out/"model.safetensors")(obs))

    def test_existing_output_and_duplicate_sources_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.checkpoint(root, "source")
            with self.assertRaises(FileExistsError):
                average_checkpoints([source], root)
            with self.assertRaises(ValueError):
                average_checkpoints([source, source], root/"new")
            self.assertFalse((root/"new").exists())

    def test_mismatched_configuration_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.checkpoint(root, "a")
            b = self.checkpoint(root, "b", config=dict(algorithm="ppo", run="other-run", tstates=100000))
            with self.assertRaises(ValueError):
                average_checkpoints([a, b], root/"new")
            self.assertFalse((root/"new").exists())


if __name__ == "__main__":
    unittest.main()
