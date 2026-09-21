"""Opaque same-build Defense saves for future own-experience training resets.

The native payload is never decoded or fed to the policy. This module neither
selects actions nor manufactures states; it restores exactly a captured state.
Boot and policy RNGs are not rewound. Ordinary evaluation never uses snapshots.
"""

import ctypes
from dataclasses import dataclass

import numpy as np

from .defense import ENVIRONMENT_VERSION, GAME_SHA256
from .snapshot import _configure
from trs.native import wrapper


@dataclass(frozen=True)
class DefenseSnapshot:
    native: bytes
    frames: np.ndarray
    score: int
    lives: int
    stage: int
    highest_stage: int
    steps: int
    missions: int
    mission_visible: bool
    start_tstates: int
    tstates: int
    max_steps: int
    action_count: int
    source_worker: int | None
    source_action: int
    source_full_game: bool
    game_sha256: str = GAME_SHA256
    environment_version: str = ENVIRONMENT_VERSION


def capture(env):
    if env.done or not env.trs.no_ui or env.trs.original_speed:
        raise ValueError("Defense snapshots require a live headless unthrottled environment")
    _configure()
    data = ctypes.create_string_buffer(wrapper.z80_snapshot_size())
    if not wrapper.z80_save_snapshot(data, len(data)):
        raise RuntimeError("Native snapshot failed; custom callbacks are unsupported")
    frames = np.stack(env.frames)
    frames.flags.writeable = False
    return DefenseSnapshot(
        data.raw, frames, env.score, env.lives, env.stage, env.highest_stage,
        env.steps, env.missions, env._mission_visible, env.start_tstates,
        env.tstates, env.max_steps, len(env.actions), getattr(env, "worker_id", None),
        getattr(env, "total_actions", env.steps), getattr(env, "full_game", True))


def restore(env, saved):
    if (not isinstance(saved, DefenseSnapshot) or saved.game_sha256 != GAME_SHA256
            or saved.environment_version != ENVIRONMENT_VERSION
            or not env.trs.no_ui or env.trs.original_speed
            or (env.tstates, env.max_steps, len(env.actions)) !=
               (saved.tstates, saved.max_steps, saved.action_count)
            or saved.frames.shape != (4, 16, 64) or saved.frames.dtype != np.uint8
            or not 1 <= saved.lives <= 4 or not 1 <= saved.stage <= saved.highest_stage <= 3
            or min(saved.score, saved.steps, saved.missions, saved.source_action) < 0):
        raise ValueError("Incompatible Defense snapshot configuration or screen history")
    _configure()
    data = ctypes.create_string_buffer(saved.native)
    # Native validation occurs before any native/Python state is changed.
    if not wrapper.z80_restore_snapshot(data, len(saved.native)):
        raise ValueError("Invalid or incompatible native snapshot")
    env.frames.clear()
    env.frames.extend(frame.copy() for frame in saved.frames)
    env.score, env.lives = saved.score, saved.lives
    env.stage, env.highest_stage = saved.stage, saved.highest_stage
    env.steps, env.missions = saved.steps, saved.missions
    env._mission_visible = saved.mission_visible
    env.start_tstates, env.done = saved.start_tstates, False
    return np.stack(env.frames)
