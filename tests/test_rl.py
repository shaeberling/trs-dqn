import unittest
from unittest.mock import patch

import numpy as np

from rl.env import BreakdownEnv, SHAPE, screen_info
from rl.evaluate import level_rank, level_target_met, summary
from rl.replay import NStep, Replay


class EnvironmentTests(unittest.TestCase):
    def test_level_nine_to_ten_redraw_cannot_invent_level_nineteen(self):
        env = BreakdownEnv()
        try:
            env.reset(12)
            env.level = 9
            readings = iter([b"00019", b"00010", b"00010"])

            def partial_redraw(tstates):
                env.video[0, 59:64] = list(next(readings))

            with patch.object(env.trs, "run_for_tstates", side_effect=partial_redraw):
                obs, reward, done, truncated, info = env.step(3)
            self.assertEqual(info["level"], 10)
            self.assertEqual(env.level, 10)
            self.assertEqual(reward, 0)
            self.assertFalse(done or truncated)
            np.testing.assert_array_equal(obs[-1, 0, 59:64], list(b"00010"))
        finally:
            env.close()

    def test_digit_redraw_cannot_invent_score_reward(self):
        env = BreakdownEnv()
        try:
            env.reset(12)
            env.score, env.episode_reward = 9, 9
            readings = iter([b"00019", b"00010", b"00010"])

            def partial_redraw(tstates):
                env.video[0, 6:11] = list(next(readings))

            with patch.object(env.trs, "run_for_tstates", side_effect=partial_redraw):
                obs, reward, done, truncated, info = env.step(3)
            self.assertEqual(reward, 1)
            self.assertEqual(info["score"], 10)
            self.assertEqual(info["episode_reward"], 10)
            self.assertFalse(done or truncated)
            np.testing.assert_array_equal(obs[-1, 0, 6:11], list(b"00010"))
        finally:
            env.close()

    def test_graphics_space_in_terminal_message(self):
        screen = np.full((16, 64), 128, np.uint8)
        screen[0, 6:11] = list(b"00077")
        screen[0, 59:64] = list(b"00002")
        screen[10, 28:37] = list(b"GAME\x80OVER")
        self.assertEqual(screen_info(screen), dict(score=77, level=2, game_over=True, waiting=False))

    def test_native_terminal_stop_accepts_graphics_in_word_gap(self):
        env = BreakdownEnv()
        try:
            for gap in [32, *range(128, 192)]:
                with self.subTest(gap=gap):
                    env.reset(12)
                    env.video[10, 28:37] = list(b"GAME"+bytes([gap])+b"OVER")
                    self.assertLess(env.trs.run_for_tstates(100000), -90000)
                    _, _, terminal, truncated, info = env.step(3)
                    self.assertTrue(terminal)
                    self.assertFalse(truncated)
                    self.assertTrue(info["game_over"])
        finally:
            env.close()

    def test_native_terminal_stop_does_not_wildcard_ascii_letters(self):
        env = BreakdownEnv()
        try:
            env.reset(12)
            env.video[10, 28:37] = list(b"GAMEXOVER")
            self.assertGreaterEqual(env.trs.run_for_tstates(100000), 0)
        finally:
            env.close()

    def test_seeded_reset_and_space_cannot_skip_game_over(self):
        env = BreakdownEnv()
        try:
            start = env.reset(12)
            self.assertEqual(start.shape, SHAPE)
            np.testing.assert_array_equal(start, env.reset(12))
            reward_sum = 0
            lives_lost = 0
            for _ in range(2000):
                obs, reward, terminated, truncated, info = env.step(3)
                reward_sum += reward
                lives_lost += info["life_lost"]
                if terminated:
                    self.assertEqual(reward_sum, info["score"])
                    self.assertFalse(truncated)
                    self.assertEqual(lives_lost, 3)
                    self.assertTrue(screen_info(obs[-1])["game_over"])
                    break
            else:
                self.fail("Held SPACE skipped GAME OVER")
            with self.assertRaises(RuntimeError):
                env.step(0)
        finally:
            env.close()


class ReplayTests(unittest.TestCase):
    def test_n_step_terminal_and_truncation(self):
        replay = Replay(16)
        queue = NStep(replay, 3, 0.5)
        frames = [np.full(SHAPE, i, np.uint8) for i in range(5)]
        queue.append(frames[0], 1, 1, frames[1], False, False)
        queue.append(frames[1], 2, 2, frames[2], True, False)
        np.testing.assert_allclose(replay.returns[:2], [2, 2])
        np.testing.assert_array_equal(replay.discounts[:2], 0)
        np.testing.assert_array_equal(replay.next_obs[0], frames[2])
        queue.append(frames[3], 0, 4, frames[4], False, True)
        self.assertEqual(replay.discounts[2], 0.5)
        np.testing.assert_array_equal(replay.obs[2], frames[3])

    def test_prioritized_sampling_and_ring_wrap(self):
        replay = Replay(8)
        obs = np.zeros(SHAPE, np.uint8)
        for i in range(20):
            replay.add(obs, i % 6, float(i), obs, 0.9)
        self.assertEqual(replay.size, 8)
        replay.priorities(np.arange(8), np.array([1000]+[0]*7))
        indices, batch = replay.sample(1000, np.random.default_rng(1), 0.4)
        self.assertGreater((indices == 0).mean(), 0.95)
        self.assertTrue(np.all(batch[2] >= 12))
        self.assertAlmostEqual(replay.tree[1], replay.tree[replay.leaves:].sum())


class EvaluationTests(unittest.TestCase):
    def test_curriculum_segment_cannot_count_as_complete_game_or_target(self):
        result = summary([dict(score=15, level=1, terminated=True),
                          dict(score=700, level=10, terminated=True, full_game=False)])
        self.assertEqual(result["complete_games"], 1)
        self.assertEqual(result["incomplete_games"], 1)
        self.assertEqual(result["mean_score"], 15)
        self.assertEqual(result["highest_complete_level"], 1)
        self.assertIsNone(level_rank(result, 10))
        self.assertFalse(level_target_met(result, 10, 1))

    def test_incomplete_games_cannot_inflate_score_or_clear_count(self):
        result = summary([
            dict(score=10, level=1, terminated=True),
            dict(score=20, level=1, terminated=True),
            dict(score=100, level=2, terminated=False),
        ])
        self.assertEqual(result["complete_games"], 2)
        self.assertEqual(result["incomplete_games"], 1)
        self.assertEqual(result["mean_score"], 15)
        self.assertEqual(result["median_score"], 15)
        self.assertEqual(result["best_score"], 20)
        self.assertEqual(result["level_1_clears"], 0)
        self.assertEqual(result["highest_complete_level"], 1)
        self.assertEqual(result["level_reach_counts"]["2"], 0)
        self.assertIsNone(level_rank(result, 5))
        self.assertFalse(level_target_met(result, 2, 1))

    def test_reaching_level_five_is_distinct_from_clearing_it(self):
        result = summary([
            dict(score=10, level=1, terminated=True),
            dict(score=90, level=2, terminated=True),
            dict(score=250, level=5, terminated=True),
        ])
        self.assertEqual(result["level_reach_counts"], {"2": 2, "3": 1, "4": 1, "5": 1})
        self.assertAlmostEqual(result["level_reach_rates"]["5"], 1/3)
        self.assertTrue(level_target_met(result, 5, 1))
        self.assertFalse(level_target_met(result, 5, 2))
        self.assertFalse(level_target_met(result, 6, 1))

    def test_level_selection_prefers_deeper_games_over_mean_score(self):
        deeper = summary([dict(score=100, level=3, terminated=True),
                          dict(score=10, level=1, terminated=True)])
        shallower = summary([dict(score=90, level=2, terminated=True)]*2)
        self.assertGreater(level_rank(deeper, 5), level_rank(shallower, 5))
        empty = summary([])
        self.assertIsNone(empty["level_reach_rates"]["5"])
        self.assertIsNone(level_rank(empty, 5))
        self.assertFalse(level_target_met(empty, 5, 1))

    def test_truncated_high_level_cannot_meet_target(self):
        result = summary([dict(score=400, level=6, terminated=False)])
        self.assertEqual(result["highest_level"], 6)
        self.assertEqual(result["highest_complete_level"], 1)
        self.assertEqual(result["level_reach_counts"]["5"], 0)
        self.assertFalse(level_target_met(result, 5, 1))

    def test_level_ten_target_uses_displayed_level_not_score(self):
        below = summary([dict(score=999, level=9, terminated=True)])
        reached = summary([dict(score=600, level=10, terminated=True)])
        self.assertFalse(level_target_met(below, 10, 1))
        self.assertTrue(level_target_met(reached, 10, 1))
        self.assertFalse(level_target_met(reached, 11, 1))
        self.assertEqual(reached["level_reach_counts"]["10"], 1)
        self.assertGreater(level_rank(reached, 10), level_rank(below, 10))

    def test_unfinished_level_ten_cannot_meet_target(self):
        result = summary([dict(score=600, level=10, terminated=False),
                          dict(score=70, level=2, terminated=True)])
        self.assertEqual(result["highest_complete_level"], 2)
        self.assertEqual(result["level_reach_counts"]["10"], 0)
        self.assertIsNone(level_rank(result, 10))
        self.assertFalse(level_target_met(result, 10, 1))


if __name__ == "__main__":
    unittest.main()
