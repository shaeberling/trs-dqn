"""The unattended memory comparison may advance only verified exact runs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl import defense_recurrent_balanced_followup as followup
from rl.defense_balanced_compare import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class RecurrentBalancedFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.treatment = self.root / "treatment"
        self.control = self.root / "control"
        self.monitor = self.root / "monitor"
        patches = (patch.object(followup, "TREATMENT", self.treatment),
                   patch.object(followup, "CONTROL", self.control),
                   patch.object(followup, "MONITOR", self.monitor),
                   patch.object(followup, "process_command", return_value=""))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def make_run(self, arm, steps=followup.TARGET):
        run = self.treatment if arm == "treatment" else self.control
        config = dict(followup.EXPECTED, run=str(run),
                      artifacts=str(Path(str(run) + "-artifacts")),
                      memory_scale=1. if arm == "treatment" else 0.,
                      resume=None, initialize_policy=None, initialize_encoder=None)
        write_json(run / "config.json", config)
        write_json(run / "status.json", dict(pid=123, event="stopped",
                                              stop_requested=False, steps=steps))
        write_json(run / "latest/state.json", dict(steps=steps, config=config))
        (run / "latest/model.safetensors").write_bytes(b"model")
        (run / "latest/optimizer.npz").write_bytes(b"optimizer")
        return config

    def test_exact_terminal_run_and_profile_check(self):
        self.make_run("treatment")
        self.make_run("control")
        self.assertEqual(followup.run_condition(self.treatment, "treatment"), "complete")
        self.assertEqual(followup.run_condition(self.control, "control"), "complete")
        self.assertIn("--memory-scale", followup.control_command())
        command = followup.control_command()
        self.assertEqual(command[command.index("--memory-scale") + 1], "0")
        self.assertEqual(command[command.index("--run") + 1], str(self.control))

        config = json.loads((self.control / "config.json").read_text())
        config["entropy"] = .02
        write_json(self.control / "config.json", config)
        self.assertEqual(followup.run_condition(self.control, "control"), "incompatible")

    def test_early_stop_fails_closed(self):
        self.make_run("treatment", steps=followup.TARGET - followup.INTERVAL)
        self.assertEqual(followup.run_condition(self.treatment, "treatment"), "stopped_early")

    def test_fixed_selection_prefers_mission_then_stage_then_mean(self):
        config = self.make_run("treatment")
        for index, step in enumerate(range(followup.INTERVAL, followup.TARGET + 1,
                                           followup.INTERVAL)):
            folder = self.treatment / f"step-{step:012d}"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(f"model-{index}".encode())
            (folder / "optimizer.npz").write_bytes(b"optimizer")
            write_json(folder / "state.json", dict(steps=step, config=config))
            write_json(folder / "evaluation.json", dict(
                complete_games=10, incomplete_games=0,
                games=[dict(seed=seed) for seed in range(10000, 10010)],
                checkpoint_sha256=digest(folder / "model.safetensors"),
                mean_score=10000 if index == 0 else 100,
                highest_stage=2 if index in (2, 3) else 1,
                mission_games=1 if index == 5 else 0))
        selection = followup.fixed_selection(self.treatment, "treatment")
        self.assertEqual(selection["selected"]["steps"], 6 * followup.INTERVAL)
        self.assertEqual(len(selection["candidates"]), 8)

    def test_fresh_later_stage_must_match_verified_replay(self):
        output = self.root / "fresh.json"
        replay = self.root / "replay"
        replay.mkdir()
        model_hash = "a" * 64
        seed = followup.FRESH_SEEDS[0]
        write_json(output, dict(complete_games=64, incomplete_games=0,
                                games=[dict(seed=value) for value in range(seed, seed + 64)],
                                checkpoint_sha256=model_hash, highest_stage=2,
                                mission_games=0))
        write_json(replay / "verification.json", dict(verified=True,
                                                       checkpoint_sha256=model_hash))
        write_json(replay / "manifest.json", dict(result=dict(highest_stage=1,
                                                               missions_completed=0)))
        (replay / "replay.html").write_text("replay")
        with self.assertRaises(RuntimeError):
            followup.checked_fresh(output, replay, model_hash, seed)
        write_json(replay / "manifest.json", dict(result=dict(highest_stage=2,
                                                               missions_completed=0)))
        self.assertEqual(followup.checked_fresh(output, replay, model_hash, seed)["highest_stage"], 2)


if __name__ == "__main__":
    unittest.main()
