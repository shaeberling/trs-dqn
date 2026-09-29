import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl.cosmic import (ACTIONS, ACTION_NAMES, CosmicEnv, GAME_OVER_TEXT,
                       GAME_SHA256, screen_info)
from rl.cosmic_learning import summarize
from rl.defense_smoke import write_replay
from rl.vector import VectorEnv
from trs import Key


def screen(hud=b"   50 [["):
    cells = np.full((16, 64), 128, np.uint8)
    cells[0, :len(hud)] = list(hud)
    cells[1, :5] = list(b"Empty")
    cells[1, 59:63] = list(b"Full")
    return cells


class CosmicTests(unittest.TestCase):
    def test_original_image_and_action_profile(self):
        self.assertEqual(hashlib.sha256(Path("var/cosmic.cmd").read_bytes()).hexdigest(),
                         GAME_SHA256)
        self.assertEqual(len(ACTIONS), 7)
        self.assertEqual(len(set(ACTIONS)), len(ACTION_NAMES))
        self.assertIn((Key.SPACE,), ACTIONS)
        self.assertIn((Key.RIGHT, Key.SPACE), ACTIONS)
        self.assertIn((Key.D,), ACTIONS)
        self.assertTrue(all(not set(keys) & {Key.CLEAR, Key.BREAK, Key.ENTER, Key._1}
                            for keys in ACTIONS))

    def test_visible_hud_blink_and_game_over(self):
        self.assertEqual(screen_info(screen())["score"], 50)
        self.assertEqual(screen_info(screen())["reserves"], 2)
        self.assertEqual(screen_info(screen(b"   10000 [["))["score"], 10_000)
        self.assertIsNone(screen_info(screen(b""))["score"])
        self.assertIsNone(screen_info(screen(b""))["reserves"])
        invalid = screen()
        invalid[1, 0] = 128
        self.assertIsNone(screen_info(invalid)["score"])
        terminal = screen(b"   50 ")
        terminal[9, 23:23+len(GAME_OVER_TEXT)] = list(GAME_OVER_TEXT)
        self.assertTrue(screen_info(terminal)["game_over"])
        self.assertEqual(screen_info(terminal)["reserves"], 0)

    def test_score_only_complete_games_and_exact_reexecution(self):
        for action, expected_score in ((0, 0), (3, 50)):
            env = CosmicEnv(max_steps=3_000)
            try:
                first = env.reset(7)
                self.assertEqual(first.shape, (4, 16, 64))
                self.assertEqual(first.dtype, np.uint8)
                np.testing.assert_array_equal(first, env.reset(7))
                frames, rewards = [first[-1]], []
                for _ in range(3_000):
                    obs, reward, done, truncated, info = env.step(action)
                    frames.append(obs[-1])
                    rewards.append(reward)
                    if done or truncated:
                        break
                self.assertTrue(done)
                self.assertFalse(truncated)
                self.assertEqual(info["score"], expected_score)
                self.assertEqual(sum(rewards), info["score"])
                self.assertEqual(info["reserves"], 0)
                self.assertTrue(screen_info(frames[-1])["game_over"])
                env.reset(7)
                for index, expected in enumerate(frames[1:]):
                    obs, reward, ended, truncated, _ = env.step(action)
                    np.testing.assert_array_equal(obs[-1], expected)
                    self.assertEqual(reward, rewards[index])
                    self.assertEqual(ended, index+2 == len(frames))
            finally:
                env.close()

    def test_blinking_startup_hud_settles_from_visible_screen(self):
        env = CosmicEnv()
        try:
            obs = env.reset(136)
            self.assertGreater(env.startup_settle_tstates, 0)
            self.assertEqual(screen_info(obs[-1])["score"], 0)
            self.assertEqual(screen_info(obs[-1])["reserves"], 3)
            np.testing.assert_array_equal(obs, env.reset(136))
        finally:
            env.close()

    def test_truncation_is_not_a_win_or_complete_game(self):
        env = CosmicEnv(max_steps=2)
        try:
            env.reset(3)
            with self.assertRaises(ValueError):
                env.step(True)
            env.step(0)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(summarize([dict(seed=3, **info)])["complete_games"], 0)
        finally:
            env.close()

    def test_vector_workers_use_cosmic(self):
        workers = VectorEnv(2, 11, game="cosmic", max_steps=2)
        try:
            self.assertEqual(workers.observations.shape, (2, 4, 16, 64))
            first = workers.step([0, 3])
            self.assertEqual(len(first), 2)
            second = workers.step([0, 3])
            self.assertTrue(all(result[3] for result in second))
            self.assertTrue(all(result[5] is not None for result in second))
        finally:
            workers.close()

    def test_self_contained_replay_uses_cosmic_title(self):
        frame = screen()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"replay.html"
            write_replay(target, np.stack([frame, frame]), [0],
                         dict(game="Cosmic Fighter", action_names=ACTION_NAMES,
                              tstates=100_000, trained_model=False))
            html = target.read_text()
            self.assertIn("<h1>Cosmic Fighter</h1>", html)
            self.assertNotIn("Obstacle Run / Missile Defense", html)


if __name__ == "__main__":
    unittest.main()
