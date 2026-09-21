"""Opaque emulator snapshots, used only for self-generated training resets.

Same native build, action duration and observation stride only. Restoring does not rewind policy,
curriculum, or boot-phase RNGs. No internal emulator bytes are model inputs.
"""

import ctypes
from dataclasses import dataclass

import numpy as np

from trs.native import wrapper
from .env import validate_observation_stride


@dataclass(frozen=True)
class Snapshot:
    native: bytes
    frames: np.ndarray
    score: int
    level: int
    steps: int
    episode_reward: float
    reserve_balls: int
    start_tstates: int
    tstates: int
    source_worker: int | None = None
    source_action: int | None = None
    source_full_game: bool = True
    source_start_level: int = 1
    observation_stride: int = 1


def _configure():
    try:
        wrapper.z80_snapshot_size.argtypes = []
        wrapper.z80_snapshot_size.restype = ctypes.c_size_t
        for name in ("z80_save_snapshot", "z80_restore_snapshot"):
            fn = getattr(wrapper, name)
            fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            fn.restype = ctypes.c_int
    except AttributeError as error:
        raise RuntimeError("Rebuild the existing emulator with make for curriculum snapshots") from error


def capture(env):
    if env.done or not env.trs.no_ui or env.trs.original_speed:
        raise ValueError("Snapshots require a live, headless, unthrottled environment")
    _configure()
    data = ctypes.create_string_buffer(wrapper.z80_snapshot_size())
    if not wrapper.z80_save_snapshot(data, len(data)):
        raise RuntimeError("Native snapshot failed; custom callbacks are unsupported")
    frames = np.stack(env.frames)
    frames.flags.writeable = False
    return Snapshot(data.raw, frames, env.score, env.level, env.steps,
                    env.episode_reward, env.reserve_balls, env.start_tstates, env.tstates,
                    getattr(env, "worker_id", None), getattr(env, "total_actions", None),
                    getattr(env, "full_game", True), getattr(env, "segment_start_level", 1),
                    observation_stride=env.observation_stride)


def restore(env, saved):
    stride = validate_observation_stride(saved.observation_stride)
    if (env.tstates != saved.tstates or not env.trs.no_ui or env.trs.original_speed
            or env.observation_stride != stride
            or saved.frames.shape != (3*stride+1, 16, 64) or saved.frames.dtype != np.uint8):
        raise ValueError("Snapshot requires matching timing, observation stride and raw history")
    _configure()
    data = ctypes.create_string_buffer(saved.native)
    if not wrapper.z80_restore_snapshot(data, len(saved.native)):
        raise ValueError("Invalid or incompatible native snapshot")
    env.frames.clear()
    env.frames.extend(frame.copy() for frame in saved.frames)
    env.score, env.level, env.steps = saved.score, saved.level, saved.steps
    env.episode_reward, env.reserve_balls = saved.episode_reward, saved.reserve_balls
    env.start_tstates, env.done = saved.start_tstates, False
    return env.observation()
