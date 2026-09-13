"""RL wrapper: the policy gets ONLY the 1 KiB screen and its recent history.

No ball/paddle coordinates or other game memory are read. Score and episode
metadata are parsed from the same visible screen but never passed separately
to the model. Every gameplay key combination comes from the selected action.
"""

import ctypes
from collections import deque

import numpy as np

from main import CONFIGS
from trs import TRS, Key
from trs.native import wrapper

VIDEO = 0x3C00
SHAPE = (4, 16, 64)
ACTIONS = ((), (Key.LEFT,), (Key.RIGHT,), (Key.SPACE,),
           (Key.LEFT, Key.SPACE), (Key.RIGHT, Key.SPACE))
TEXT_TABLE = bytes(c if 32 <= c < 127 else 32 for c in range(256))


def screen_info(screen):
    text = screen.tobytes().translate(TEXT_TABLE)
    score = text[6:11]
    level = text[59:64]
    status = text[640:704]
    return {
        "score": int(score) if score.isdigit() else None,
        "level": int(level) if level.isdigit() else None,
        "game_over": b"GAME OVER" in status,
        "waiting": b"PASS BALL" in status or b"THAT HURTS" in status,
    }


class BreakdownEnv:
    def __init__(self, seed=0, tstates=100_000, max_steps=30_000):
        self.rng = np.random.default_rng(seed)
        self.tstates = tstates
        self.max_steps = max_steps
        self.trs = TRS(CONFIGS["breakdown"], original_speed=0, fps=30, no_ui=True)
        self.video = np.ctypeslib.as_array(self.trs.ram.ram)[VIDEO:VIDEO+1024].reshape(16, 64)
        wrapper.z80_set_video_stop.argtypes = (ctypes.c_ushort, ctypes.c_char_p, ctypes.c_int)
        self.frames = deque(maxlen=4)
        self.done = True

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        wrapper.z80_set_video_stop(0, b"", 0)
        self.trs.boot()
        # Seeded no-op start phase: <= 0.11 seconds, before the first collision.
        # Changes neither RAM nor the game's physics or rewards.
        self.start_tstates = int(self.rng.integers(0, 200_001))
        self.trs.run_for_tstates(self.start_tstates)
        wrapper.z80_set_video_stop(VIDEO+640+28, b"GAME\x80OVER", 9)
        frame = self.video.copy()
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(4))
        info = screen_info(frame)
        if info["score"] != 0 or info["level"] != 1 or info["game_over"]:
            raise RuntimeError(f"Unexpected boot screen: {info}")
        self.score, self.level, self.steps, self.episode_reward = 0, 1, 0, 0
        self.reserve_balls = int(np.count_nonzero(frame[0, 33:35] == 0x95))
        self.done = False
        return np.stack(self.frames)

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() is required after an episode ends")
        if not 0 <= int(action) < len(ACTIONS):
            raise ValueError(action)
        self.trs.keyboard.all_keys_up()
        for key in ACTIONS[int(action)]:
            self.trs.keyboard.key_down(key)
        self.trs.run_for_tstates(self.tstates)
        self.steps += 1
        frame = self.video.copy()
        info = screen_info(frame)
        # The Z80 redraws the five score digits sequentially. For example, 9
        # becoming 10 can briefly display 19. Let a changed HUD settle while
        # retaining exactly the policy-selected keys; no new action is chosen.
        # Unchanged HUDs incur no extra emulation work.
        hud_settle_tstates = 0
        if (info["score"] != self.score or info["level"] != self.level) and not info["game_over"]:
            for _ in range(8):
                previous_digits = frame[0, np.r_[6:11, 59:64]].copy()
                self.trs.run_for_tstates(2_000)
                hud_settle_tstates += 2_000
                frame = self.video.copy()
                info = screen_info(frame)
                if (info["game_over"] or (info["score"] is not None
                        and info["level"] is not None
                        and np.array_equal(previous_digits, frame[0, np.r_[6:11, 59:64]]))):
                    break
            else:
                raise RuntimeError("Score display did not settle")
        score = info["score"] if info["score"] is not None else self.score
        # There is no score reset within a game.
        if score < self.score:
            raise RuntimeError(f"Score went backwards: {self.score} -> {score}")
        reward = float(score-self.score)
        self.score = score
        self.level = max(self.level, info["level"] or self.level)
        reserve_balls = int(np.count_nonzero(frame[0, 33:35] == 0x95))
        life_lost = reserve_balls < self.reserve_balls or info["game_over"]
        self.reserve_balls = reserve_balls
        self.episode_reward += reward
        terminated = info["game_over"]
        truncated = bool(self.max_steps and self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        self.frames.append(frame)
        info.update(score=self.score, level=self.level, steps=self.steps,
                    terminated=terminated, truncated=truncated,
                    life_lost=life_lost,
                    hud_settle_tstates=hud_settle_tstates,
                    episode_reward=self.episode_reward, start_tstates=self.start_tstates)
        return np.stack(self.frames), reward, terminated, truncated, info

    def close(self):
        self.trs.keyboard.all_keys_up()
        wrapper.z80_set_video_stop(0, b"", 0)
