"""No-training Cosmic model/evaluation/replay integration checks."""

import tempfile
from pathlib import Path
import unittest

from rl.cosmic import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256
from rl.cosmic_dqn import checkpoint
from rl.cosmic_learning import evaluate, load_policy, publish_best, sha256
from rl.model import Learner


class CosmicLearningTests(unittest.TestCase):
    def test_untrained_model_roundtrip_and_verified_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root/"frozen"
            config = dict(game="cosmic", game_sha256=GAME_SHA256,
                          environment_version=ENVIRONMENT_VERSION,
                          action_names=list(ACTION_NAMES),
                          algorithm="dueling-double-dqn-per-nstep",
                          tstates=100_000, eval_max_steps=0,
                          observation_stride=1)
            learner = Learner(seed=23, action_count=len(ACTION_NAMES))
            checkpoint(learner, frozen, dict(steps=0, updates=0, config=config))
            before = sha256(frozen/"model.safetensors")
            policy, loaded = load_policy(frozen/"model.safetensors")
            self.assertEqual(loaded, config)
            result = evaluate(policy, range(230, 232), tstates=100_000,
                              max_steps=0, envs=2)
            self.assertEqual(result["complete_games"], 2)
            self.assertEqual(result["incomplete_games"], 0)
            version = publish_best(frozen/"model.safetensors", result, root/"artifacts")
            best = root/"artifacts"/"best"
            self.assertTrue(best.is_symlink())
            self.assertEqual(best.resolve(), (root/"artifacts"/"versions"/version).resolve())
            self.assertIn("Cosmic Fighter", (best/"replay.html").read_text())
            self.assertEqual(before, sha256(best/"model.safetensors"))
            self.assertEqual(learner.optimizer.state["step"].item(), 0)


if __name__ == "__main__":
    unittest.main()
