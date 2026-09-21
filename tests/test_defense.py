import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.defense import (ACTIONS, ACTION_NAMES, DefenseEnv, GAME_SHA256, action_set,
                        GAME_OVER_TEXT, screen_info)
from rl.defense_smoke import verify_trace, write_replay
from rl.defense_audit import audit
from trs import Key


def hud(left=b"280 **"):
    screen = np.full((16, 64), 128, np.uint8)
    row = left.ljust(29) + b"180020" + b" "*29
    screen[0] = list(row)
    return screen


class DefenseTests(unittest.TestCase):
    def test_archive_binary_identity(self):
        self.assertEqual(hashlib.sha256(Path("var/defense.cmd").read_bytes()).hexdigest(), GAME_SHA256)

    def test_original_game_wrap_is_static_evidence_not_a_claimed_win(self):
        report = audit()
        self.assertEqual(report["original_stage_cycle"], [1, 2, 3, 1])
        self.assertIn("YOU did it", report["success_text"])
        self.assertFalse(report["successful_gameplay_observed"])

    def test_hud_and_blink_are_not_zero(self):
        info = screen_info(hud())
        self.assertEqual((info["score"], info["lives"]), (280, 2))
        for left in (b"    **", b"", b"PLAYER 1", b"280 ***garbage"):
            self.assertIsNone(screen_info(hud(left))["score"])
        self.assertEqual(screen_info(hud(b"280"))["lives"], 0)
        wipe = hud()
        wipe[0, 63] = 128
        self.assertIsNone(screen_info(wipe)["score"])

    def test_screen_only_stage_and_mission_messages(self):
        for stage, message in enumerate((b"You are now entering the", b"Find your way through the",
                                         b"Now at last you enter the"), 1):
            screen = hud()
            screen[4, 10:10+len(message)] = list(message)
            self.assertEqual(screen_info(screen)["stage"], stage)
        screen = hud()
        screen[7, 22:43] = list(b"CONGRATULATIONS ! ! !")
        self.assertFalse(screen_info(screen)["mission_complete"])
        screen[8, 27:37] = list(b"YOU did it")
        self.assertTrue(screen_info(screen)["mission_complete"])
        screen[7, 23:41] = list(GAME_OVER_TEXT)
        self.assertTrue(screen_info(screen)["game_over"])

    def test_action_space_includes_all_directions_and_fires_no_menu_keys(self):
        self.assertEqual(len(ACTIONS), 20)
        self.assertEqual(len(set(ACTIONS)), 20)
        self.assertEqual(len(ACTION_NAMES), 20)
        self.assertIn((Key.LEFT, Key.RIGHT), ACTIONS)
        self.assertIn((Key.UP, Key.RIGHT, Key.SPACE), ACTIONS)
        for keys in ACTIONS:
            self.assertFalse(set(keys) & {Key.ENTER, Key.CLEAR, Key.BREAK, Key._1, Key._2})

    def test_optional_enter_is_a_selected_action_not_an_automatic_controller(self):
        self.assertEqual(action_set(True), ACTIONS + ((Key.ENTER,),))
        env = DefenseEnv(allow_enter=True)
        try:
            env.reset(12)
            losses = 0
            intro_entries = 0
            intro_visible = False
            for _ in range(3000):
                obs, _, done, truncated, info = env.step(20)
                visible = screen_info(obs[-1])["stage"] is not None
                intro_entries += visible and not intro_visible
                intro_visible = visible
                losses += info["life_lost"]
                if done or truncated:
                    break
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertEqual(losses, 4)
            self.assertEqual(intro_entries, 3)
            self.assertEqual(info["score"], 280)
            enter_steps = info["steps"]
            env.reset(12)
            for _ in range(3000):
                _, _, done, truncated, info = env.step(0)
                if done or truncated:
                    break
            self.assertTrue(done)
            self.assertEqual(info["score"], 280)
            self.assertLess(enter_steps, info["steps"])
        finally:
            env.close()

    def test_reset_and_complete_game_all_four_ships(self):
        env = DefenseEnv()
        try:
            start = env.reset(9)
            self.assertEqual(start.shape, (4, 16, 64))
            self.assertEqual(start.dtype, np.uint8)
            np.testing.assert_array_equal(start, env.reset(9))
            rng = np.random.default_rng(9)
            total, losses = 0, 0
            for _ in range(3000):
                obs, reward, done, truncated, info = env.step(int(rng.integers(len(ACTIONS))))
                total += reward
                losses += info["life_lost"]
                self.assertFalse(truncated)
                if done:
                    break
            else:
                self.fail("Expected complete random game")
            self.assertEqual(total, info["score"])
            self.assertEqual(losses, 4)
            self.assertEqual(info["lives"], 0)
            # Regression: first GAME OVER displayed 260, final visible score 280.
            self.assertEqual(info["score"], 280)
            self.assertEqual(screen_info(obs[-1])["score"], 280)
            self.assertTrue(screen_info(obs[-1])["game_over"])
            self.assertGreater(info["terminal_settle_tstates"], 0)
            with self.assertRaises(RuntimeError):
                env.step(0)
        finally:
            env.close()

    def test_held_fire_cannot_restart_after_game_over(self):
        env = DefenseEnv()
        try:
            env.reset(12)
            for _ in range(4000):
                obs, _, done, truncated, info = env.step(19)
                if done or truncated:
                    break
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertTrue(screen_info(obs[-1])["game_over"])
            self.assertEqual(info["lives"], 0)
        finally:
            env.close()

    def test_redraw_does_not_invent_score(self):
        env = DefenseEnv()
        try:
            env.reset(12)
            env.score = 80
            readings = iter((hud(b"180 ****"), hud(b"100 ****"), hud(b"100 ****")))
            with patch.object(env.trs, "run_for_tstates", side_effect=lambda _: env.video.__setitem__(slice(None), next(readings))):
                _, reward, _, _, info = env.step(0)
            self.assertEqual(reward, 20)
            self.assertEqual(info["score"], 100)
        finally:
            env.close()

    def test_scrolling_non_hud_animation_is_not_a_score_error(self):
        env = DefenseEnv()
        try:
            env.reset(12)
            count = 0

            def animate(_):
                nonlocal count
                count += 1
                env.video[0] = 128
                env.video[0, count] = ord("A")

            with patch.object(env.trs, "run_for_tstates", side_effect=animate):
                _, reward, done, truncated, info = env.step(0)
            self.assertEqual(count, 9)
            self.assertEqual(reward, 0)
            self.assertEqual(info["score"], 0)
            self.assertEqual(info["lives"], 4)
            self.assertFalse(done or truncated or info["life_lost"])
        finally:
            env.close()

    def test_unstable_numeric_hud_is_still_rejected(self):
        env = DefenseEnv()
        try:
            env.reset(12)
            count = 0

            def redraw(_):
                nonlocal count
                count += 1
                env.video[:] = hud(f"{count*20} ****".encode())

            with patch.object(env.trs, "run_for_tstates", side_effect=redraw):
                with self.assertRaisesRegex(RuntimeError, "numeric HUD did not settle"):
                    env.step(0)
            self.assertEqual(env.score, 0)
        finally:
            env.close()

    def test_truncation_and_action_validation(self):
        env = DefenseEnv(max_steps=1)
        try:
            env.reset(12)
            for bad in (-1, 20, True, 1.1):
                with self.assertRaises((TypeError, ValueError)):
                    env.step(bad)
            _, _, done, truncated, info = env.step(0)
            self.assertFalse(done)
            self.assertTrue(truncated)
            self.assertEqual(info["missions_completed"], 0)
        finally:
            env.close()

    def test_mission_counts_once_through_message_flicker(self):
        env = DefenseEnv()
        try:
            env.reset(12)
            win = hud(b"0 ****")
            win[8, 27:37] = list(b"YOU did it")
            intro = hud(b"0 ****")
            intro[4, 20:44] = list(b"You are now entering the")
            with patch.object(env.trs, "run_for_tstates"):
                for screen, expected in ((win, 1), (hud(b"0 ****"), 1), (win, 1),
                                         (intro, 1), (win, 2)):
                    env.video[:] = screen
                    _, reward, done, truncated, info = env.step(0)
                    self.assertEqual(info["missions_completed"], expected)
                    self.assertEqual(reward, 0)
                    self.assertFalse(done or truncated)
        finally:
            env.close()

    def test_trace_reexecution_and_portable_replay(self):
        env = DefenseEnv(max_steps=3)
        try:
            frames, rewards = [env.reset(7)[-1]], []
            actions = [0, 19, 8]
            for action in actions:
                obs, reward, _, _, _ = env.step(action)
                frames.append(obs[-1])
                rewards.append(reward)
        finally:
            env.close()
        verify_trace(frames, actions, rewards, 7, 100_000, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"replay.html"
            write_replay(path, frames, actions, {"test": True})
            html = path.read_text()
            self.assertNotIn("__FRAMES__", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("not a trained agent", html)


if __name__ == "__main__":
    unittest.main()
