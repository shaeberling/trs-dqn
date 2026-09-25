"""Stage-agnostic no-fire action profile for score-only neural training."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense import action_names
from rl.defense_learning import policy_description


class MovementOnlyTests(unittest.TestCase):
    def test_profile_is_nine_original_direction_ids_without_stage_logic(self):
        names = list(action_names(False)[:9])
        self.assertEqual(names[0], "NOOP")
        self.assertEqual(len(names), 9)
        self.assertTrue(all("SPACE" not in name for name in names))
        config = dict(algorithm="ppo", movement_only=True,
                      policy_action_names=names)
        self.assertIn("nine-movement", policy_description(config))
        for changed in (dict(policy_action_names=names + ["SPACE"]),
                        dict(canonical_fire=True), dict(allow_enter=True),
                        dict(learned_durations=[1, 4]),
                        dict(architecture="unsupported")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                policy_description({**config, **changed})

    def test_conflicting_profiles_fail_before_emulator_launch(self):
        from rl import defense_train

        for extra in (["--allow-enter"], ["--canonical-fire"],
                      ["--repeat-previous-action"], ["--learned-durations", "1", "4"],
                      ["--recurrent-hidden", "32"], ["--spatial-residual"],
                      ["--policy-bias-noise", ".5"], ["--sil-updates", "1"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", [
                    "defense_train", "--run", "/nonexistent/movement-only-test",
                    "--movement-only", *extra]), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_train.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_training_replay_and_exact_resume(self):
        from rl.defense_learning import load_policy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, "-m", "rl.defense_train", "--movement-only",
                       "--run", str(root/"run"), "--artifacts", str(root/"artifacts"),
                       "--steps", "64", "--seed", "47", "--envs", "2", "--rollout", "16",
                       "--batch-size", "32", "--epochs", "1", "--eval-every", "64",
                       "--eval-games", "2", "--eval-envs", "2", "--mlx-cache-mb", "64"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = root/"run/step-000000000064"
            state = json.loads((checkpoint/"state.json").read_text())
            self.assertEqual(state["steps"], 64)
            self.assertTrue(state["config"]["movement_only"])
            self.assertEqual(state["config"]["policy_action_names"],
                             list(action_names(False)[:9]))
            self.assertEqual(state["config"]["reward"],
                             "visible score difference only, constant scale for optimizer")
            evaluation = json.loads((checkpoint/"evaluation.json").read_text())
            self.assertEqual(evaluation["complete_games"], 2)
            self.assertTrue(json.loads((root/"artifacts/best/verification.json").read_text())["verified"])
            with np.load(root/"artifacts/best/trace.npz") as trace:
                self.assertTrue(np.all(trace["actions"] < 9))
            policy, config = load_policy(checkpoint/"model.safetensors")
            self.assertTrue(config["movement_only"])
            actions = policy.sample_with_rngs(np.zeros((2, 4, 16, 64), np.uint8),
                                              [np.random.default_rng(1), np.random.default_rng(2)])
            self.assertTrue(np.all(np.asarray(actions) < 9))
            resumed = subprocess.run([sys.executable, "-m", "rl.defense_train", "--run",
                                      str(root/"resume"), "--artifacts", str(root/"resume-artifacts"),
                                      "--resume", str(checkpoint), "--steps", "96"],
                                     capture_output=True, text=True, timeout=180)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(json.loads((root/"resume/latest/state.json").read_text())["steps"], 96)


if __name__ == "__main__":
    unittest.main()
