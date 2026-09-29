"""Screen-only Cosmic Fighter environment on the unmodified TRS-80 program."""

from collections import deque
import ctypes
import hashlib
import operator
from pathlib import Path
import re

import numpy as np

from main import CONFIGS
from trs import Key, TRS
from .env import validate_observation_stride


VIDEO = 0x3C00
GAME_SHA256 = "4999f2d853dddc0ee58198a03b597fd848d5011068037b1ed8f2926d218190e7"
ENVIRONMENT_VERSION = "cosmic-fighter-screen-v2-fire-tap"
GAME_OVER_TEXT = b"Game Over Player 1"
ACTIONS = ((), (Key.LEFT,), (Key.RIGHT,), (Key.SPACE,),
           (Key.LEFT, Key.SPACE), (Key.RIGHT, Key.SPACE), (Key.D,))
ACTION_NAMES = tuple(("+".join(key.name for key in keys) or "NOOP") +
                     (" (tap)" if Key.SPACE in keys else "") for keys in ACTIONS)
FIRE_RELEASE_NUMERATOR = 2
FIRE_RELEASE_DENOMINATOR = 5
TEXT_TABLE = bytes(c if 32 <= c < 127 else 32 for c in range(256))
HUD = re.compile(rb" *([0-9]{1,7}) (\[{0,9}) *")


def screen_info(screen):
    """Only visible video memory supplies score, reserve ships and terminal text.

    The score/ships line blinks. Unknown fields remain None; an all-blank
    frame is never treated as zero score or zero ships.
    """
    raw = np.asarray(screen, dtype=np.uint8).reshape(16, 64).tobytes()
    text = raw.translate(TEXT_TABLE)
    valid_layout = text[64:69] == b"Empty" and text[64+59:64+63] == b"Full"
    hud = HUD.fullmatch(text[:24]) if valid_layout else None
    score = int(hud[1]) if hud else None
    reserves = len(hud[2]) if hud else None
    return dict(score=score, reserves=reserves,
                game_over=GAME_OVER_TEXT in text[9*64:10*64],
                dock_prompt=b"Dock" in text)


class CosmicEnv:
    def __init__(self, seed=0, tstates=100_000, max_steps=5_000,
                 observation_stride=1):
        self.tstates = operator.index(tstates)
        self.max_steps = operator.index(max_steps)
        if not 1 <= self.tstates <= 1_000_000 or self.max_steps < 0:
            raise ValueError("invalid Cosmic action duration or episode cap")
        self.observation_stride = validate_observation_stride(observation_stride)
        if hashlib.sha256(Path(CONFIGS["cosmic"]["cmd"]).read_bytes()).hexdigest() != GAME_SHA256:
            raise RuntimeError("Cosmic Fighter executable differs from the audited image")
        self.rng = np.random.default_rng(seed)
        self.trs = TRS(CONFIGS["cosmic"], original_speed=0, fps=30, no_ui=True)
        self.video = np.ctypeslib.as_array(self.trs.ram.ram)[VIDEO:VIDEO+1024].reshape(16, 64)
        self.frames = deque(maxlen=3*self.observation_stride+1)
        self.done = True

    def observation(self):
        return np.stack(list(self.frames)[::self.observation_stride])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.trs.boot()
        # Seeded title-phase jitter changes only the original game's timing.
        self.start_tstates = int(self.rng.integers(0, 200_001))
        self.trs.run_for_tstates(self.start_tstates)
        frame = self.video.copy()
        info = screen_info(frame)
        self.startup_settle_tstates = 0
        for _ in range(100):
            if info["score"] is not None and info["reserves"] is not None:
                break
            self.trs.run_for_tstates(50_000)
            self.startup_settle_tstates += 50_000
            frame = self.video.copy()
            info = screen_info(frame)
        else:
            raise RuntimeError("Cosmic Fighter boot HUD never became readable")
        if info["score"] != 0 or info["reserves"] != 3 or info["game_over"]:
            raise RuntimeError(f"Unexpected Cosmic Fighter boot screen: {info}")
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(self.frames.maxlen))
        self.score, self.reserves, self.steps = 0, 3, 0
        self.done = False
        return self.observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() required after episode end")
        if isinstance(action, (bool, np.bool_)):
            raise ValueError(action)
        action = operator.index(action)
        if not 0 <= action < len(ACTIONS):
            raise ValueError(action)
        self.trs.keyboard.all_keys_up()
        # The original program only recognizes a new shot after it has polled
        # a released fire key. A zero-cycle release between two SPACE actions
        # is invisible to the emulated CPU and effectively holds fire forever.
        # Keep every decision at the configured duration: fire actions spend
        # 2/5 released and 3/5 pressed, other actions use the full duration.
        pressed_tstates = self.tstates
        if Key.SPACE in ACTIONS[action]:
            released_tstates = self.tstates*FIRE_RELEASE_NUMERATOR//FIRE_RELEASE_DENOMINATOR
            if released_tstates:
                self.trs.run_for_tstates(released_tstates)
                pressed_tstates -= released_tstates
        for key in ACTIONS[action]:
            self.trs.keyboard.key_down(key)
        self.trs.run_for_tstates(pressed_tstates)
        frame = self.video.copy()
        info = screen_info(frame)
        settle = 0
        # The HUD is drawn and blinked in pieces. Require two stable readable
        # copies before accepting a changed numeric field as score reward.
        if not info["game_over"] and (info["score"] is not None and info["score"] != self.score
                                        or info["reserves"] is not None and info["reserves"] != self.reserves):
            for _ in range(20):
                before = frame[0].copy()
                self.trs.run_for_tstates(2_000)
                settle += 2_000
                frame = self.video.copy()
                info = screen_info(frame)
                if info["game_over"] or (info["score"] is not None and
                        np.array_equal(before, frame[0])):
                    break
        if info["game_over"] and info["score"] is None:
            # Game-over text persists while the HUD blinks. Resolve the final
            # visible score without treating a blank frame as zero.
            for _ in range(30):
                self.trs.run_for_tstates(50_000)
                settle += 50_000
                frame = self.video.copy()
                info = screen_info(frame)
                if info["score"] is not None:
                    break
            else:
                raise RuntimeError("Final Cosmic Fighter score never became visible")
        score = self.score if info["score"] is None else info["score"]
        if score < self.score:
            raise RuntimeError(f"Visible Cosmic Fighter score went backwards: {self.score} -> {score}")
        reward = float(score-self.score)
        self.score = score
        reserves = 0 if info["game_over"] else info["reserves"]
        life_lost = reserves is not None and reserves < self.reserves
        extra_ship = reserves is not None and reserves > self.reserves
        if reserves is not None:
            self.reserves = reserves
        self.steps += 1
        terminated = info["game_over"]
        truncated = bool(self.max_steps and self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        self.frames.append(frame)
        info.update(score=self.score, reserves=self.reserves, life_lost=life_lost,
                    extra_ship=extra_ship, steps=self.steps, terminated=terminated,
                    truncated=truncated, episode_reward=float(self.score),
                    hud_settle_tstates=settle, start_tstates=self.start_tstates,
                    startup_settle_tstates=self.startup_settle_tstates)
        return self.observation(), reward, terminated, truncated, info

    def close(self):
        self.trs.keyboard.all_keys_up()
