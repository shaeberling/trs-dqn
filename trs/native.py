"""Load the existing emulator library from the repository on macOS/Linux."""

import ctypes
from pathlib import Path

wrapper = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "libtrs.so"))
