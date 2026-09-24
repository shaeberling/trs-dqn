"""Coarse screen fingerprints for training-only, own-state archive diversity.

No model/backend imports or emulator access. The policy input is unchanged.
The HUD is omitted from this archive key so score/lives alone do not create
new cells. Gameplay graphics are decoded with the TRS-80's fixed six-bit raster.
"""

import hashlib

import numpy as np


CELL_ENCODING = "graphics-9x16x8-v1"
BOTTOM_DETAIL_ENCODING = "graphics-9x16x8+raw-bottom-three-rows-v1"


def screen_cell(frames):
    frames = np.asarray(frames)
    if frames.shape != (4, 16, 64) or frames.dtype != np.uint8:
        raise ValueError("screen cells require four raw uint8 screen frames")
    screen = frames[-1, 1:]
    graphics = np.where((screen >= 128) & (screen <= 191), screen-128, 0)
    bits = np.arange(6, dtype=np.uint8).reshape(3, 2)
    pixels = ((graphics[..., None, None] >> bits) & 1).transpose(0, 2, 1, 3).reshape(45, 128)
    counts = pixels.reshape(9, 5, 16, 8).sum(axis=(1, 3), dtype=np.uint16)
    coarse = np.minimum(7, counts*8//40).astype(np.uint8)
    return hashlib.blake2b(coarse.tobytes(), digest_size=16,
                           person=b"defense-cell-v1").hexdigest()


def screen_cell_bottom_detail(frames):
    """Coarse gameplay plus exact visible bytes in the bottom three rows.

    This is a training-reset fingerprint only. It neither detects an object
    nor supplies any new policy input, reward or action preference.
    """
    frames = np.asarray(frames)
    coarse = screen_cell(frames)
    return hashlib.blake2b(bytes.fromhex(coarse)+frames[-1, 13:16].tobytes(),
                           digest_size=16, person=b"defense-bottom1").hexdigest()
