"""The unattended matched-control handoff is exact and fail-closed."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.defense_balanced_followup import (MATCHED_CONFIG, TARGET, control_command,
                                          treatment_condition, verify_terminal_evaluation)


class BalancedFollowupTests(unittest.TestCase):
    def make_treatment(self, root, *, steps=TARGET, event="stopped", changes=None):
        run = root/"treatment"
        latest = run/"latest"
        latest.mkdir(parents=True)
        config = {**MATCHED_CONFIG, **(changes or {})}
        (run/"config.json").write_text(json.dumps(config))
        (run/"status.json").write_text(json.dumps({"event": event, "pid": 1234}))
        (latest/"state.json").write_text(json.dumps({"steps": steps, "config": config}))
        (latest/"model.safetensors").touch()
        (latest/"optimizer.npz").touch()

    def test_waits_for_live_exact_treatment_then_requires_target_and_matching_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_treatment(root, event="progress")
            with patch("rl.defense_balanced_followup.process_command",
                       return_value=f"python -m rl.defense_train --run {root/'treatment'}"):
                self.assertEqual(treatment_condition(root), "live")
            with patch("rl.defense_balanced_followup.process_command", return_value=""):
                self.assertEqual(treatment_condition(root), "uncertain")
                self.make_treatment_update(root, event="stopped")
                self.assertEqual(treatment_condition(root), "complete")
                self.make_treatment_update(root, steps=TARGET-4096)
                self.assertEqual(treatment_condition(root), "stopped_early")
                self.make_treatment_update(root, steps=TARGET, config_change=("entropy", .02))
                self.assertEqual(treatment_condition(root), "incompatible")

    @staticmethod
    def make_treatment_update(root, *, event=None, steps=None, config_change=None):
        run = root/"treatment"
        status_path = run/"status.json"
        status = json.loads(status_path.read_text())
        if event is not None:
            status["event"] = event
        status_path.write_text(json.dumps(status))
        state_path = run/"latest/state.json"
        state = json.loads(state_path.read_text())
        if steps is not None:
            state["steps"] = steps
        if config_change is not None:
            key, value = config_change
            state["config"][key] = value
        state_path.write_text(json.dumps(state))

    def test_control_uses_same_experiment_settings_without_treatment_bias(self):
        command = control_command(Path("runs/defense-ppo-balanced-fire-160"), "python")
        self.assertEqual(command[:4], ["python", "-u", "-m", "rl.defense_train"])
        self.assertIn("--canonical-fire", command)
        self.assertIn("--life-terminal", command)
        self.assertNotIn("--balanced-canonical-init", command)
        self.assertIn("--no-balanced-canonical-init", command)
        for flag, value in (("--steps", str(TARGET)), ("--seed", "41"),
                            ("--rollout", "256"), ("--eval-every", "1048576"),
                            ("--gamma", ".997"), ("--reward-scale", ".01")):
            self.assertEqual(command[command.index(flag)+1], value)

    def test_terminal_evaluation_requires_complete_games_and_verified_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = root/"treatment-terminal-replay"
            replay.mkdir()
            evaluation = {"complete_games": 10, "incomplete_games": 0,
                          "checkpoint_sha256": "abc", "mean_score": 400}
            (root/"treatment-terminal-evaluation.json").write_text(json.dumps(evaluation))
            verification = {"verified": True, "checkpoint_sha256": "abc"}
            (replay/"verification.json").write_text(json.dumps(verification))
            self.assertEqual(verify_terminal_evaluation(root), evaluation)
            verification["checkpoint_sha256"] = "wrong"
            (replay/"verification.json").write_text(json.dumps(verification))
            with self.assertRaises(RuntimeError):
                verify_terminal_evaluation(root)


if __name__ == "__main__":
    unittest.main()
