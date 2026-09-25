import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class InternalLossLagTests(unittest.TestCase):
    def test_native_forensic_probe_matches_frozen_original_boot_life(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"lag.json"
            completed = subprocess.run([sys.executable, "-m", "rl.defense_internal_loss_lag",
                "--output", str(output)], capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report["exact_original_boot_actions_rewards_screens_verified"], 412)
        self.assertEqual(report["first_visible_loss_action"], 412)
        self.assertEqual(report["internal_decrement"]["action_number"], 391)
        self.assertEqual(report["visible_lag_actions"], 21)
        self.assertTrue(report["internal_decrement"]["during_base_action"])
        self.assertGreater(report["internal_decrement"]["requested_tstates_threshold"], 0)
        self.assertLess(report["internal_decrement"]["requested_tstates_threshold"], 100000)
        self.assertEqual(report["audited_private_counter_address"], "0x7CEF")
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["model_updates"], 0)
        self.assertFalse(report["training_data_written"])
        self.assertFalse(report["hidden_counter_used_by_policy_or_reward"])
        self.assertFalse(report["promotion_eligible"])

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"exists.json"
            output.write_text("untouched")
            completed = subprocess.run([sys.executable, "-m", "rl.defense_internal_loss_lag",
                "--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
