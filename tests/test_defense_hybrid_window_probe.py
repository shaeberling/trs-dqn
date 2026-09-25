import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class HybridWindowProbeTests(unittest.TestCase):
    def test_native_smoke_reexecutes_frozen_source_and_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"smoke.json"
            completed = subprocess.run([sys.executable, "-m", "rl.defense_hybrid_window_probe",
                "--smoke", "--output", str(output)], capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            report = json.loads(output.read_text())
        self.assertTrue(report["exact_parent_prefix_and_suffix_verified"])
        self.assertEqual(report["summary"]["source_first_visible_loss_action"], 412)
        self.assertEqual(report["summary"]["branches"], 2)
        self.assertEqual(len(report["branches"]), 2)
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["hidden_ram_reads"], 0)
        self.assertEqual(report["model_updates"], 0)
        self.assertEqual(report["reward_changes"], 0)
        self.assertFalse(report["branch_actions_as_demonstrations"])
        self.assertFalse(report["policy_promotion_eligible"])

    def test_existing_output_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exists.json"
            output.write_text("untouched")
            completed = subprocess.run([sys.executable, "-m", "rl.defense_hybrid_window_probe",
                "--smoke", "--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
