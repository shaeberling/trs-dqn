"""Record a model-controlled complete game as a portable, interactive HTML replay."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import BreakdownEnv
from .evaluate import checkpoint_config, load_policy, policy_description


def record(checkpoint, output, seed, tstates, max_steps, deterministic=False):
    config = checkpoint_config(checkpoint)
    tstates = config.get("tstates", 100_000) if tstates is None else tstates
    policy = load_policy(checkpoint, deterministic=deterministic)
    env = BreakdownEnv(tstates=tstates, max_steps=max_steps)
    frames, actions = [], []
    try:
        obs = env.reset(seed)
        if hasattr(policy, "reset_seed"):
            policy.reset_seed(seed+1_000_000)
        frames.append(obs[-1].copy())
        while True:
            action = int(policy(obs[None])[0])
            obs, _, done, truncated, info = env.step(action)
            frames.append(obs[-1].copy())
            actions.append(action)
            if done or truncated:
                break
    finally:
        env.close()
    payload = bytearray(frames[0].tobytes())
    for previous, frame in zip(frames[:-1], frames[1:]):
        flat = frame.reshape(-1)
        changed = np.flatnonzero(flat != previous.reshape(-1))
        payload.extend(struct.pack("<H", len(changed)))
        for index in changed:
            payload.extend(struct.pack("<HB", index, int(flat[index])))
    atlas = Image.new("RGB", (16*8, 16*26), "black")
    draw = ImageDraw.Draw(atlas)
    font = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", 26)
    for c in range(256):
        draw.text((c % 16*8, c//16*26-1), chr(0xe000+c), font=font)
    stream = io.BytesIO()
    atlas.save(stream, format="PNG")
    metadata = dict(seed=seed, checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    policy=policy_description(config, deterministic),
                    deterministic_override=deterministic,
                    frames=len(frames), tstates=tstates, **info)
    template = Path(__file__).with_name("replay.html").read_text()
    html = (template.replace("__METADATA__", json.dumps(metadata).replace("</", "<\\/"))
            .replace("__FRAMES__", base64.b64encode(payload).decode())
            .replace("__ATLAS__", base64.b64encode(stream.getvalue()).decode())
            .replace("__ACTIONS__", json.dumps(actions)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(json.dumps(metadata, indent=2))
    if not info["terminated"]:
        raise SystemExit("Recording is truncated; it is not a complete evaluation game")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/replay.html"))
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--tstates", type=int, help="defaults to checkpoint action duration")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--deterministic", action="store_true", help="force PPO argmax as in evaluation")
    args = parser.parse_args()
    record(args.checkpoint, args.output, args.seed, args.tstates, args.max_steps, args.deterministic)
