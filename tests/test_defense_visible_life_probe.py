import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rl.defense_visible_life_probe import life_summary


class VisibleLifeProbeTests(unittest.TestCase):
    def test_summary_requires_four_consistent_visible_losses(self):
        game = dict(game_over=True, lives=0, steps=40, score=400,
                    visible_life_losses=[dict(action=10*i, displayed_score=100*i,
                                              visible_lives=4-i) for i in range(1, 5)])
        result = life_summary(dict(incomplete_games=0, games=[game]))
        self.assertEqual(result["first_visible_loss_actions"],
                         dict(mean=10., median=10., minimum=10, maximum=10))
        self.assertEqual(result["mean_actions_between_visible_losses"], [10.]*4)
        for changed in (dict(game, steps=41), dict(game, score=399),
                        dict(game, visible_life_losses=game["visible_life_losses"][:-1]),
                        dict(game, visible_life_losses=[game["visible_life_losses"][0]]*4)):
            with self.assertRaises(ValueError):
                life_summary(dict(incomplete_games=0, games=[changed]))
        with self.assertRaises(ValueError):
            life_summary(dict(incomplete_games=1, games=[game]))

    def test_native_cli_matches_independent_original_boot_recording(self):
        from rl.defense_learning import load_policy, record_game, sha256
        checkpoint = Path("results/defense/training/ppo-movement-only-164/pilot/checkpoint/model.safetensors")
        seed = 621111
        before = sha256(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"life-report.json"
            completed = subprocess.run([sys.executable, "-m", "rl.defense_visible_life_probe",
                str(checkpoint), "--output", str(output), "--seed", str(seed),
                "--games", "1", "--envs", "1"], capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            report = json.loads(output.read_text())
        self.assertEqual(report["checkpoint_sha256"], before)
        self.assertEqual(report["evaluation"]["complete_games"], 1)
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual(report["hidden_ram_reads"], 0)
        self.assertEqual(report["model_updates"], 0)
        self.assertFalse(report["checkpoint_selection_eligible"])
        policy, config = load_policy(checkpoint)
        recorded = record_game(policy, seed, tstates=config["tstates"], max_steps=0,
                               observation_stride=config.get("observation_stride", 1))
        game = report["evaluation"]["games"][0]
        self.assertEqual({key: value for key, value in game.items()
                          if key != "visible_life_losses"}, recorded[3])
        self.assertEqual([event["action"] for event in game["visible_life_losses"]],
                         [event["frame"] for event in recorded[4] if event["life_lost"]])
        self.assertEqual([event["displayed_score"] for event in game["visible_life_losses"]],
                         [event["score"] for event in recorded[4] if event["life_lost"]])
        self.assertEqual(sha256(checkpoint), before)


if __name__ == "__main__":
    unittest.main()
