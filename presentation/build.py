"""Build the offline slide deck from audited results and real replay frames.

Run from the repository root: venv/bin/python presentation/build.py
No learning, inference, generated artwork, or emulator stepping occurs here.
"""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.env import screen_info
from rl.inspect_policy import decode_replay


def picture(frame):
    """Render the exact recorded character cells using the repository font."""
    image = Image.new("RGB", (512, 416), "#070c0e")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ROOT / "var/AnotherMansTreasureMIII64C.ttf"), 26)
    for row, cells in enumerate(frame):
        for col, code in enumerate(cells):
            draw.text((col * 8, row * 26 - 1), chr(0xe000 + int(code)),
                      font=font, fill="#d4f9e6")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    result_dir = ROOT / "results/level10"
    replay = (result_dir / "best/replay.html").read_bytes()
    audit = json.loads((result_dir / "all-eight-completion-audit.json").read_text())
    assert hashlib.sha256(replay).hexdigest() == audit["winning_replay"]["sha256"]
    metadata, frames, actions = decode_replay(replay.decode())
    assert metadata["game_won"] and metadata["steps"] == 49159
    assert len(frames) == len(actions) + 1
    selection = json.loads((result_dir / "all-eight-final-selection.json").read_text())
    fresh = json.loads((result_dir / "all-eight-fresh-test.json").read_text())
    assert fresh["complete_games"] == 100 and fresh["verified_wins"] == 0
    assert selection["combined"]["complete_games"] == 70
    assert selection["combined"]["verified_wins"] == 1
    config = json.loads((ROOT / "models/breakdown-all-eight/state.json").read_text())["config"]
    assert config["envs"] == 32 and config["curriculum_boot_envs"] == 16
    assert config["rollout"] == 256 and config["sil_updates"] == 4
    level8 = next(i for i, f in enumerate(frames) if screen_info(f)["level"] == 8)
    payload = {
        "metadata": metadata,
        "validation": selection["combined"],
        "fresh": {k: v for k, v in fresh.items() if k != "games"},
        "frames": {"hero": picture(frames[1800]), "level8": picture(frames[level8 + 60]),
                   "finish": picture(frames[-1])},
        "history": [picture(frames[i]) for i in (1794, 1796, 1798, 1800)],
        "frame_indices": {"hero": 1800, "level8": level8 + 60, "finish": len(frames) - 1},
        "model_sha256": metadata["checkpoint_sha256"],
        "replay_sha256": audit["winning_replay"]["sha256"],
    }
    # Keep only summary data, not the large validation trajectories.
    payload["validation"] = {k: v for k, v in payload["validation"].items() if k != "games"}
    template = (ROOT / "presentation/deck.html").read_text()
    assert template.count("__DECK_DATA__") == 1
    html = template.replace("__DECK_DATA__", json.dumps(payload).replace("</", "<\\/"))
    output = ROOT / "results/training-presentation.html"
    output.write_text(html)
    print(f"Built {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes), 10 slides.")


if __name__ == "__main__":
    main()
