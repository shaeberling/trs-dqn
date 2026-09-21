"""Screen-only Obstacle Run environment; no hidden game RAM is read.

The archive calls this game Missile Defense; its existing launcher ID is defense.
One native emulator per process, as with Breakdown. This is a separate task:
Breakdown checkpoints, action IDs, curriculum and evaluation do not apply.
"""

from collections import deque
import ctypes
import hashlib
import operator
from pathlib import Path
import re

import numpy as np

from main import CONFIGS
from trs import TRS, Key
from trs.native import wrapper

VIDEO = 0x3C00
ENVIRONMENT_VERSION = "obstacle-run-screen-v1"
GAME_SHA256 = "9b887e46223eb2ce9d2ada2a7588435a90be51f9e5326a592193faeafa322644"
DIRECTIONS = ((), (Key.UP,), (Key.DOWN,), (Key.LEFT,), (Key.RIGHT,),
              (Key.UP, Key.LEFT), (Key.UP, Key.RIGHT),
              (Key.DOWN, Key.LEFT), (Key.DOWN, Key.RIGHT))
ACTIONS = DIRECTIONS + tuple(keys + (Key.SPACE,) for keys in DIRECTIONS) + (
    (Key.LEFT, Key.RIGHT), (Key.LEFT, Key.RIGHT, Key.SPACE))
ACTION_NAMES = tuple("+".join(key.name for key in keys) or "NOOP" for keys in ACTIONS)


def action_set(allow_enter=False):
    if not isinstance(allow_enter, (bool, np.bool_)):
        raise ValueError("allow_enter must be a boolean")
    return ACTIONS + ((Key.ENTER,),) if allow_enter else ACTIONS


def action_names(allow_enter=False):
    return tuple("+".join(key.name for key in keys) or "NOOP" for keys in action_set(allow_enter))


TEXT_TABLE = bytes(c if 32 <= c < 127 else 32 for c in range(256))
GAME_OVER_ADDRESS = 0x3DD7
GAME_OVER_TEXT = b"GAME OVER PLAYER 1"


def screen_info(screen):
    """Parse only visible text. Missing/blinking HUD fields are unknown, not zero."""
    raw = np.asarray(screen, dtype=np.uint8).reshape(16, 64).tobytes()
    text = raw.translate(TEXT_TABLE)
    row = raw[:64]
    # P1 field is 16 columns, high score centered, P2 field blank in one-player
    # mode. Require the unnormalized ASCII layout to reject screen wipes.
    hud = re.fullmatch(rb"([0-9]{0,7}) (\*{1,4}) *", row[:16])
    layout = (row[16:29] == b" " * 13 and
              re.fullmatch(rb"[0-9 ]{7}", row[29:36]) is not None and
              row[36:] == b" " * 28)
    score = int(hud[1]) if layout and hud and hud[1] else None
    lives = len(hud[2]) if layout and hud else None
    # Last ship's HUD has no stars. Do not infer death from any other blank HUD.
    zero = re.fullmatch(rb"([0-9]{1,7}) +", row[:16])
    if layout and zero:
        score, lives = int(zero[1]), 0
    stage = None
    for number, message in enumerate((b"You are now entering the",
                                      b"Find your way through the",
                                      b"Now at last you enter the"), 1):
        if message in text:
            stage = number
    return dict(score=score, lives=lives, stage=stage,
                game_over=GAME_OVER_TEXT in text[7*64:8*64],
                mission_complete=b"YOU did it" in text[8*64:9*64])


def positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name)
    value = operator.index(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(name)
    return value


class DefenseEnv:
    def __init__(self, seed=0, tstates=100_000, max_steps=30_000, allow_enter=False):
        # Optional learned menu action. No detection-triggered key injection:
        # only the selected policy action may press Enter after reset.
        self.actions = action_set(allow_enter)
        self.tstates = positive_integer(tstates, "tstates")
        if self.tstates > 1_000_000:
            raise ValueError("tstates must be <= 1,000,000 to sample transition messages")
        self.max_steps = positive_integer(max_steps, "max_steps", allow_zero=True)
        self.rng = np.random.default_rng(seed)
        if hashlib.sha256(Path(CONFIGS["defense"]["cmd"]).read_bytes()).hexdigest() != GAME_SHA256:
            raise RuntimeError("Defense executable differs from the audited game")
        self.trs = TRS(CONFIGS["defense"], original_speed=0, fps=30, no_ui=True)
        self.video = np.ctypeslib.as_array(self.trs.ram.ram)[VIDEO:VIDEO+1024].reshape(16, 64)
        wrapper.z80_set_video_stop.argtypes = (ctypes.c_ushort, ctypes.c_char_p, ctypes.c_int)
        wrapper.z80_set_video_text_stop.argtypes = (ctypes.c_ushort, ctypes.c_char_p, ctypes.c_int)
        self.frames = deque(maxlen=4)
        self.done = True

    def _wait(self, predicate, limit, quantum=100_000):
        for _ in range(limit // quantum):
            self.trs.run_for_tstates(quantum)
            if predicate(self.video):
                return
        raise RuntimeError("Defense startup did not reach the expected visible screen")

    def reset(self, seed=None):
        self.done = True
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        wrapper.z80_set_video_stop(0, b"", 0)
        self.trs.boot()
        # Jitter only the title/menu timing, not gameplay or hidden state.
        self.start_tstates = int(self.rng.integers(0, 200_001))
        self.trs.run_for_tstates(50_000_000 + self.start_tstates)
        self.trs.keyboard.key_down(Key.CLEAR)
        try:
            self._wait(lambda v: b"Enter number of players (1 or 2) ?" in v.tobytes(),
                       50_000_000)
            self.trs.keyboard.all_keys_up()
            self.trs.keyboard.key_down(Key._1)
            self._wait(lambda v: b"PLAYER 1" in v.tobytes(), 10_000_000)
        finally:
            self.trs.keyboard.all_keys_up()
        # Let the original introduction finish. No automatic gameplay actions.
        self._wait(lambda v: screen_info(v)["score"] == 0 and
                   screen_info(v)["lives"] == 4, 50_000_000, quantum=50_000)
        wrapper.z80_set_video_text_stop(GAME_OVER_ADDRESS, GAME_OVER_TEXT, len(GAME_OVER_TEXT))
        self.frames.clear()
        self.frames.extend(self.video.copy() for _ in range(4))
        self.score, self.lives, self.stage, self.highest_stage = 0, 4, 1, 1
        self.steps, self.missions = 0, 0
        self._mission_visible = False
        self.done = False
        return np.stack(self.frames)

    def _finish_terminal(self, frame, info):
        terminal_settle = 0
        if info["game_over"]:
            # The first GAME OVER draw precedes the last score/lives HUD update
            # (death animation at 7079 -> 7084). Finish only that animation,
            # keeping the selected keys, until the visible zero-ship HUD is
            # complete. No restart/abort combination exists in either action
            # profile; optional Enter skips intros, not this death animation.
            wrapper.z80_set_video_stop(0, b"", 0)
            for _ in range(100):
                self.trs.run_for_tstates(50_000)
                terminal_settle += 50_000
                frame = self.video.copy()
                info = screen_info(frame)
                if info["game_over"] and info["lives"] == 0 and info["score"] is not None:
                    self.trs.run_for_tstates(2_000)
                    terminal_settle += 2_000
                    frame = self.video.copy()
                    info = screen_info(frame)
                    if info["game_over"] and info["lives"] == 0 and info["score"] is not None:
                        break
            else:
                self.done = True
                raise RuntimeError("Final visible score not established after GAME OVER")
            wrapper.z80_set_video_text_stop(GAME_OVER_ADDRESS, GAME_OVER_TEXT, len(GAME_OVER_TEXT))
        return frame, info, terminal_settle

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() required before stepping")
        if isinstance(action, (bool, np.bool_)):
            raise ValueError(action)
        action = operator.index(action)
        if not 0 <= action < len(self.actions):
            raise ValueError(action)
        self.trs.keyboard.all_keys_up()
        for key in self.actions[action]:
            self.trs.keyboard.key_down(key)
        self.trs.run_for_tstates(self.tstates)
        frame = self.video.copy()
        info = screen_info(frame)
        # HUD copies use LDIR; settle a changed row with the SAME selected keys.
        # A blank score is intentional blinking and must not reset the reward.
        settle = 0
        if not np.array_equal(frame[0], self.frames[-1][0]) and not info["game_over"]:
            for _ in range(8):
                previous = frame[0].copy()
                self.trs.run_for_tstates(2_000)
                settle += 2_000
                frame = self.video.copy()
                info = screen_info(frame)
                if info["game_over"] or np.array_equal(previous, frame[0]):
                    break
            else:
                # A scrolling intro/wipe can keep changing row 0 without
                # displaying any HUD at all. Unknown visible fields carry no
                # reward and are retained from the last readable HUD below.
                # Do not abort learning because an animation is still moving.
                # Numeric HUDs must still stabilize; never accept a torn score.
                if info["score"] is not None or info["lives"] is not None:
                    raise RuntimeError(f"Defense numeric HUD did not settle: {frame[0].tobytes().hex()}")
        frame, info, terminal_settle = self._finish_terminal(frame, info)
        score = self.score if info["score"] is None else info["score"]
        if score < self.score:
            raise RuntimeError(f"Defense score went backwards: {self.score} -> {score}")
        reward = float(score - self.score)
        self.score = score
        lives = 0 if info["game_over"] else info["lives"]
        life_lost = lives is not None and lives < self.lives
        if lives is not None:
            if lives > self.lives:
                raise RuntimeError("Unexpected increase in visible ship count")
            self.lives = lives
        if info["stage"] is not None:
            self.stage = info["stage"]
            self.highest_stage = max(self.highest_stage, self.stage)
            self._mission_visible = False
        mission_completed = info["mission_complete"] and not self._mission_visible
        self.missions += int(mission_completed)
        if mission_completed:
            self._mission_visible = True
        self.steps += 1
        terminated = info["game_over"]
        truncated = bool(self.max_steps and self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        self.frames.append(frame)
        info.update(score=self.score, lives=self.lives, stage=self.stage,
                    highest_stage=self.highest_stage, missions_completed=self.missions,
                    mission_completed=mission_completed, life_lost=life_lost,
                    steps=self.steps, terminated=terminated, truncated=truncated,
                    episode_reward=float(self.score), hud_settle_tstates=settle,
                    terminal_settle_tstates=terminal_settle,
                    start_tstates=self.start_tstates)
        return np.stack(self.frames), reward, terminated, truncated, info

    def close(self):
        self.trs.keyboard.all_keys_up()
        wrapper.z80_set_video_stop(0, b"", 0)
