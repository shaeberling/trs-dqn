import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from rl.defense import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_learning import game_rank, load_policy, publish_best, summarize
from rl.evaluate import categorical_policy
from rl.ppo import PPO
from rl.vector import VectorEnv


class DefenseLearningTests(unittest.TestCase):
    def test_twenty_action_ppo_updates_and_checkpoint_roundtrip(self):
        agent = PPO(seed=12, action_count=20)
        rng = np.random.default_rng(4)
        obs = rng.integers(128, 192, (8, 4, 16, 64), dtype=np.uint8)
        actions, logp, values = agent.act(obs, rng)
        self.assertEqual(agent.predict(mx.array(obs))[0].shape, (8, 20))
        self.assertTrue(np.all((actions >= 0) & (actions < 20)))
        before = np.array(agent.model.advantage.weight)
        loss, aux = agent.update(*(mx.array(x) for x in
                                  (obs, actions, logp, np.linspace(-1, 1, 8, dtype=np.float32),
                                   values+1)))
        mx.eval(loss, aux, agent.state)
        self.assertTrue(np.isfinite(loss.item()))
        self.assertFalse(np.array_equal(before, np.array(agent.model.advantage.weight)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = dict(game="defense", game_sha256=GAME_SHA256,
                          environment_version=ENVIRONMENT_VERSION, action_names=list(ACTION_NAMES))
            agent.save(path, dict(config=config, steps=32))
            loaded, _ = load_policy(path/"model.safetensors")
            original = agent.policy(31)
            loaded.reset_seed(31)
            np.testing.assert_array_equal(original(obs), loaded(obs))
            config["game"] = "breakdown"
            (path/"state.json").write_text(json.dumps(dict(config=config)))
            with self.assertRaises(ValueError):
                load_policy(path/"model.safetensors")

    def test_categorical_policy_does_not_clip_defense_to_six_actions(self):
        def logits(obs):
            values = np.full((len(obs), 20), -100., np.float32)
            values[:, 19] = 100
            return values
        policy = categorical_policy(logits)
        np.testing.assert_array_equal(policy(np.zeros((3, 4, 16, 64))), [19]*3)

    def test_optional_enter_profile_survives_checkpoint_roundtrip(self):
        agent = PPO(seed=42, action_count=21)
        obs = np.full((16, 4, 16, 64), 128, np.uint8)
        self.assertEqual(agent.predict(mx.array(obs))[0].shape, (16, 21))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = dict(game="defense", game_sha256=GAME_SHA256, allow_enter=True,
                          environment_version=ENVIRONMENT_VERSION, action_names=list(action_names(True)))
            agent.save(path, dict(config=config, steps=0))
            loaded, loaded_config = load_policy(path/"model.safetensors")
            self.assertTrue(loaded_config["allow_enter"])
            original = agent.policy(3)
            loaded.reset_seed(3)
            np.testing.assert_array_equal(original(obs), loaded(obs))
            config["allow_enter"] = False
            (path/"state.json").write_text(json.dumps(dict(config=config)))
            with self.assertRaises(ValueError):
                load_policy(path/"model.safetensors")

    def test_vector_defense_workers_have_no_gpu_and_match_serial(self):
        from rl.defense import DefenseEnv
        vector = VectorEnv(2, 7, game="defense", max_steps=10)
        try:
            self.assertTrue(all(not row["mlx_loaded"] for row in vector.runtime()))
            starts = vector.observations.copy()
            results = vector.step([19, 8])
        finally:
            vector.close()
        for worker, action in enumerate((19, 8)):
            env = DefenseEnv(max_steps=10)
            try:
                np.testing.assert_array_equal(env.reset(7+worker), starts[worker])
                actual = env.step(action)
                np.testing.assert_array_equal(actual[0], results[worker][0])
                self.assertEqual(actual[1:], results[worker][1:5])
            finally:
                env.close()

    def test_incomplete_games_cannot_win_and_missions_outrank_score(self):
        low = dict(terminated=True, truncated=False, missions_completed=0, highest_stage=1, score=9999)
        deep = dict(low, highest_stage=3, score=500)
        won = dict(deep, missions_completed=1, score=400)
        incomplete = dict(won, terminated=False, truncated=True)
        self.assertLess(game_rank(low), game_rank(deep))
        self.assertLess(game_rank(deep), game_rank(won))
        self.assertIsNone(game_rank(incomplete))
        result = summarize([low, deep, won, incomplete])
        self.assertEqual(result["complete_games"], 3)
        self.assertEqual(result["incomplete_games"], 1)
        self.assertEqual(result["mission_games"], 1)

    def test_failed_verification_never_publishes(self):
        result = dict(seed=10000, terminated=True, truncated=False, missions_completed=0,
                      highest_stage=1, score=100)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"artifacts"
            with patch("rl.defense_learning.load_policy", return_value=(None, dict(tstates=100000, eval_max_steps=0))), \
                 patch("rl.defense_learning.record_game", return_value=(None, None, None, result, [])), \
                 patch("rl.defense_learning.verify_policy_trace", side_effect=RuntimeError("mismatch")):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    publish_best(Path(tmp)/"model.safetensors", dict(games=[result]), output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
