"""Integrity/protocol checks for the frozen first-reach release, not new gameplay."""

import hashlib
import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/"models/breakdown-level5"
MODEL_SHA = "f84d6346798330742a1890e9ee6308fbc380c155bac9d24086dde10bf62d3b87"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Level5ArtifactTests(unittest.TestCase):
    def test_frozen_package_checksums_and_policy_configuration(self):
        names = set()
        for line in (RELEASE/"SHA256SUMS").read_text().splitlines():
            expected, name = line.split()
            self.assertEqual(Path(name).name, name)
            self.assertEqual(digest(RELEASE/name), expected)
            names.add(name)
        self.assertEqual(names, {"model.safetensors", "optimizer.npz", "state.json",
                                 "validation.json", "evaluation.json"})
        self.assertEqual(digest(RELEASE/"model.safetensors"), MODEL_SHA)
        state = json.loads((RELEASE/"state.json").read_text())
        self.assertEqual(state["steps"], 17502208)
        self.assertEqual(state["config"]["algorithm"], "ppo")
        self.assertEqual(state["config"]["environment_version"], "normalized-game-over-v2")
        self.assertEqual(state["best_level_rank"], [1, 1, 1, 10, 88.0])

    def test_original_model_and_replay_are_preserved(self):
        self.assertEqual(digest(ROOT/"models/breakdown/model.safetensors"),
                         "a54161326684ca4e63dec62e68053f9363a44640b75cab69f06bf59b85f7da69")
        self.assertEqual(digest(ROOT/"results/replay.html"),
                         "e71d228ee94949558b7006dbae6ee31543c895172b6aa73ba44d039ec6ac1a52")

    def test_complete_suites_and_independently_recomputed_statistics(self):
        suites = [(RELEASE/"validation.json", 10000, 20, 40000),
                  (RELEASE/"evaluation.json", 30000, 100, 0),
                  (ROOT/"results/level5/random.json", 30000, 100, 0)]
        for path, start, count, limit in suites:
            with self.subTest(suite=path.name, start=start):
                result = json.loads(path.read_text())
                games = result["games"]
                self.assertEqual([g["seed"] for g in games], list(range(start, start+count)))
                self.assertEqual(result["games_requested"], count)
                self.assertEqual(result["complete_games"], count)
                self.assertEqual(result["incomplete_games"], 0)
                self.assertEqual(result["max_steps"], limit)
                self.assertEqual(result["tstates"], 100000)
                self.assertEqual(result["epsilon"], 0)
                self.assertEqual(result["environment_version"], "normalized-game-over-v2")
                self.assertTrue(all(g["terminated"] and g["game_over"] and not g["truncated"]
                                    and g["steps"] > 0 and g["episode_reward"] == g["score"]
                                    for g in games))
                scores = [g["score"] for g in games]
                self.assertAlmostEqual(result["mean_score"], statistics.mean(scores))
                self.assertEqual(result["median_score"], statistics.median(scores))
                self.assertEqual(result["best_score"], max(scores))
                self.assertEqual(result["highest_complete_level"], max(g["level"] for g in games))
                for level in range(2, 6):
                    reached = sum(g["level"] >= level for g in games)
                    self.assertEqual(result["level_reach_counts"][str(level)], reached)
                    self.assertEqual(result["level_reach_rates"][str(level)], reached/count)
        validation = json.loads((RELEASE/"validation.json").read_text())
        success = next(g for g in validation["games"] if g["seed"] == 10016)
        self.assertEqual((success["score"], success["level"], success["steps"]), (278, 5, 15346))
        final = json.loads((RELEASE/"evaluation.json").read_text())
        self.assertEqual(final["checkpoint_sha256"], MODEL_SHA)
        self.assertFalse(final["deterministic_override"])
        self.assertEqual(final["policy"], "learned categorical, sampled")
        self.assertEqual(final["level_reach_counts"]["5"], 0)

    def test_archived_target_stop_and_mirrored_final_test(self):
        self.assertEqual((RELEASE/"evaluation.json").read_bytes(),
                         (ROOT/"results/level5/trained.json").read_bytes())
        path = ROOT/"results/level5/logs/level5-entropy003-extended.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "stopped")
        self.assertEqual(rows[-1]["steps"], 17502208)
        self.assertTrue(rows[-1]["target_met"])
        self.assertTrue(any(r["event"] == "validation" and r["steps"] == 17502208
                            and r["level_reach_counts"]["5"] == 1
                            and r["complete_games"] == 20 and r["incomplete_games"] == 0
                            for r in rows))


if __name__ == "__main__":
    unittest.main()
