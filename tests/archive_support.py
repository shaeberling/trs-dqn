"""Skip historical integration suites when their local-only data is absent."""

from pathlib import Path
import unittest


_ARCHIVE_SENTINEL = (Path(__file__).resolve().parents[1] /
                     "results/defense/training/dqn-24-bootstrap/step-000000800000/model.safetensors")


def requires_defense_archive(cls):
    return unittest.skipUnless(
        _ARCHIVE_SENTINEL.is_file(),
        "historical Defense archive is local-only; see ARTIFACTS.md",
    )(cls)
