"""PPO routing for own-life-loss practice; play/evaluation remain screen-only."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class DefensePPOLossCurriculumTests(unittest.TestCase):
    def test_invalid_options_fail_before_workers_start(self):
        from rl import defense_train

        invalid = (
            ["--curriculum-trigger", "life-loss"],
            ["--curriculum-trigger", "life-loss", "--curriculum-lookback", "64"],
            ["--curriculum-trigger", "life-loss", "--curriculum-probability", ".5"],
            ["--curriculum-restored-life-only"],
            ["--curriculum-restored-life-only", "--curriculum-probability", ".5"],
        )
        for extra in invalid:
            with self.subTest(extra=extra), patch.object(sys, "argv", [
                    "defense_train", "--run", "/nonexistent/ppo-own-loss", *extra]), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_train.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_training_archives_own_life_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, "-m", "rl.defense_train", "--run", str(root/"run"),
                       "--artifacts", str(root/"artifacts"), "--envs", "2", "--rollout", "16",
                       "--batch-size", "32", "--epochs", "1", "--steps", "4096",
                       "--eval-every", "10000", "--mlx-cache-mb", "64",
                       "--curriculum-probability", "1", "--curriculum-share",
                       "--curriculum-boot-envs", "1", "--curriculum-trigger", "life-loss",
                       "--curriculum-lookback", "16", "--curriculum-restored-life-only",
                       "--life-terminal"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = [json.loads(line) for line in (root/"run/metrics.jsonl").read_text().splitlines()]
            archives = [row for row in rows if row["event"] == "curriculum_archive"]
            self.assertTrue(archives)
            self.assertTrue(all(row["entry_kind"] == "life_loss_lookback" for row in archives))
            self.assertTrue(all(row["trigger_action"]-row["source_action"] == 16 for row in archives))
            episodes = [row for row in rows if row["event"] == "episode"]
            self.assertTrue(all(row["full_game"] for row in episodes if row["worker"] == 0))
            self.assertTrue(any(not row["full_game"] for row in episodes if row["worker"] == 1))
            state = json.loads((root/"run/latest/state.json").read_text())
            self.assertEqual(state["steps"], 4096)
            self.assertEqual(state["config"]["curriculum_trigger"], "life-loss")
            self.assertTrue(state["config"]["curriculum_restored_life_only"])
            self.assertFalse(state["config"]["curriculum_archive_saved"])
            resumed = subprocess.run([sys.executable, "-m", "rl.defense_train", "--run", str(root/"resume"),
                                      "--artifacts", str(root/"resume-artifacts"), "--resume",
                                      str(root/"run/latest"), "--steps", "4128"],
                                     capture_output=True, text=True, timeout=180)
            self.assertEqual(resumed.returncode, 0, resumed.stdout+resumed.stderr)
            after = json.loads((root/"resume/latest/state.json").read_text())
            self.assertEqual(after["steps"], 4128)
            self.assertEqual(after["config"]["curriculum_trigger"], "life-loss")
            self.assertTrue(after["config"]["curriculum_restored_life_only"])


if __name__ == "__main__":
    unittest.main()
