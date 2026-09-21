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

from rl.defense import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256
from rl.defense_initialization import initialize_encoder
from rl.ppo import PPO


def config():
    return dict(game="defense", game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION, action_names=list(ACTION_NAMES),
                seed=99, gamma=.5)


def arrays(tree):
    return {k: np.array(v).copy() for k, v in tree_flatten(tree)}


class DefenseInitializationTests(unittest.TestCase):
    def test_only_encoder_is_copied_and_optimizer_heads_and_source_stay_unchanged(self):
        source, target = PPO(seed=9, action_count=20), PPO(seed=73, action_count=20)
        before = arrays(target.model.parameters())
        optimizer = arrays(target.optimizer.state)
        original = arrays(source.model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source.save(p, dict(config=config(), steps=123456, episodes=77))
            files = {f.name: f.read_bytes() for f in p.iterdir()}
            result = initialize_encoder(target.model, p)
            actual = arrays(target.model.parameters())
            for name in actual:
                expected = original[name] if name.startswith(("conv.", "hidden.")) else before[name]
                np.testing.assert_array_equal(actual[name], expected, err_msg=name)
            for name, value in arrays(target.optimizer.state).items():
                np.testing.assert_array_equal(value, optimizer[name])
            self.assertEqual(files, {f.name: f.read_bytes() for f in p.iterdir()})
            self.assertEqual(result["source_training_steps"], 123456)
            self.assertEqual(set(result["copied_parameters"]),
                             {k for k in actual if k.startswith(("conv.", "hidden."))})
            target.compile()
            obs = mx.full((2, 4, 16, 64), 128, dtype=mx.uint8)
            np.testing.assert_array_equal(np.array(target.model.features(obs)),
                                          np.array(source.model.features(obs)))

    def test_incompatible_and_malformed_sources_fail_before_any_model_change(self):
        source, target = PPO(seed=9, action_count=20), PPO(seed=73, action_count=20)
        before = arrays(target.model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source.save(p, dict(config=config(), steps=100))
            state = json.loads((p/"state.json").read_text())
            for change in ({"game": "breakdown"}, {"game_sha256": "wrong"},
                           {"environment_version": "wrong"}, {"allow_enter": True},
                           {"action_names": ["NOOP"]}):
                (p/"state.json").write_text(json.dumps(dict(state, config={**config(), **change})))
                with self.assertRaisesRegex(ValueError, "compatible"):
                    initialize_encoder(target.model, p)
            (p/"state.json").write_text(json.dumps(state))
            good = mx.load(str(p/"model.safetensors"))
            for broken in (dict(good, unknown=mx.zeros((1,))),
                           {**good, "hidden.weight": mx.zeros((2, 2))},
                           {**good, "hidden.bias": mx.full((256,), float("nan"))},
                           {**good, "hidden.bias": good["hidden.bias"].astype(mx.float16)}):
                with patch("mlx.core.load", return_value=broken), self.assertRaises(ValueError):
                    initialize_encoder(target.model, p)
            for name, value in arrays(target.model.parameters()).items():
                np.testing.assert_array_equal(value, before[name])

    def test_cli_disallows_initialization_together_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"unused"
            result = subprocess.run([sys.executable, "-m", "rl.defense_train", "--run", str(output),
                                     "--resume", "unused", "--initialize-encoder", "unused"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("not allowed with argument", result.stderr)
            self.assertFalse(output.exists())

    def test_changing_source_is_rejected_before_copying_parameters(self):
        source, target = PPO(seed=9, action_count=20), PPO(seed=73, action_count=20)
        before = arrays(target.model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source.save(p, dict(config=config(), steps=100))
            loaded = mx.load(str(p/"model.safetensors"))
            def concurrent_change(_):
                (p/"state.json").write_text(json.dumps(dict(config=config(), steps=200)))
                return loaded
            with patch("mlx.core.load", side_effect=concurrent_change), \
                    self.assertRaisesRegex(ValueError, "source changed"):
                initialize_encoder(target.model, p)
            for name, value in arrays(target.model.parameters()).items():
                np.testing.assert_array_equal(value, before[name])

    def test_fresh_cli_counters_and_resume_without_reapplying_initialization(self):
        source = PPO(seed=9, action_count=20)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source.save(p/"source", dict(config=config(), steps=123456, episodes=77))
            base = [sys.executable, "-m", "rl.defense_train"]
            first = subprocess.run(base + ["--run", str(p/"fresh"), "--artifacts", str(p/"artifacts"),
                "--initialize-encoder", str(p/"source"), "--seed", "73", "--envs", "1",
                "--rollout", "4", "--batch-size", "4", "--epochs", "1", "--steps", "4",
                "--eval-every", "1000", "--mlx-cache-mb", "64"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            events = [json.loads(line) for line in (p/"fresh/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["steps"], 0)
            saved = json.loads((p/"fresh/latest/state.json").read_text())
            self.assertEqual((saved["steps"], saved["episodes"]), (4, 0))
            self.assertEqual(saved["config"]["seed"], 73)
            self.assertEqual(saved["config"]["gamma"], .997)  # source config is not inherited
            self.assertIsNone(saved["config"]["resume"])
            self.assertEqual(saved["config"]["initialization"]["source_training_steps"], 123456)
            # A true resume must not load the old initialization source again.
            (p/"source").rename(p/"source-unavailable")
            second = subprocess.run(base + ["--run", str(p/"resumed"), "--resume", str(p/"fresh/latest"),
                "--artifacts", str(p/"resumed-artifacts"), "--steps", "8"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(second.returncode, 0, second.stderr)
            resumed = json.loads((p/"resumed/latest/state.json").read_text())
            self.assertEqual(resumed["steps"], 8)
            self.assertIsNone(resumed["config"]["initialize_encoder"])
            self.assertEqual(resumed["config"]["initialization"], saved["config"]["initialization"])


if __name__ == "__main__":
    unittest.main()
