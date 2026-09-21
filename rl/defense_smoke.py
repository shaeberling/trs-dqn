"""Run complete diagnostic games, not training or expert demonstrations.

Save a self-contained screen replay and an exactly re-executable action trace.
No model/learner imports; all scores and outcomes come from DefenseEnv's video.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .defense import ACTIONS, ACTION_NAMES, DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256


def verify_trace(frames, actions, rewards, seed, tstates, max_steps):
    env = DefenseEnv(tstates=tstates, max_steps=max_steps)
    try:
        np.testing.assert_array_equal(env.reset(seed)[-1], frames[0])
        for index, action in enumerate(actions):
            obs, reward, terminated, truncated, _ = env.step(int(action))
            np.testing.assert_array_equal(obs[-1], frames[index+1])
            if reward != rewards[index]:
                raise RuntimeError(f"Reward mismatch at action {index}")
            if (terminated or truncated) != (index == len(actions)-1):
                raise RuntimeError("Trace episode boundary mismatch")
    finally:
        env.close()


def write_replay(path, frames, actions, metadata):
    payload = bytearray(frames[0].tobytes())
    for previous, frame in zip(frames[:-1], frames[1:]):
        flat = frame.reshape(-1)
        changed = np.flatnonzero(flat != previous.reshape(-1))
        payload.extend(struct.pack("<H", len(changed)))
        for index in changed:
            payload.extend(struct.pack("<HB", index, int(flat[index])))
    atlas = Image.new("RGB", (128, 416), "black")
    draw = ImageDraw.Draw(atlas)
    font = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", 26)
    for code in range(256):
        draw.text((code % 16*8, code//16*26-1), chr(0xe000+code), font=font, fill="white")
    stream = io.BytesIO()
    atlas.save(stream, format="PNG")
    template = Path(__file__).with_name("defense_replay.html").read_text()
    if metadata.get("trained_model"):
        template = template.replace("diagnostic replay", "learned-policy replay").replace(
            "Emulator integration test — random/no-op actions, <strong>not a trained agent</strong>.",
            "<strong>Trained screen-only neural policy.</strong> No scripted gameplay controller.")
        template = template.replace("The recorded action sequence has been rerun and every resulting screen checked.",
            "The frozen model was reloaded and every neural action, screen and reward verified from boot.")
    path.write_text(template.replace("__META__", json.dumps(metadata).replace("</", "<\\/"))
                    .replace("__FRAMES__", base64.b64encode(payload).decode())
                    .replace("__ATLAS__", base64.b64encode(stream.getvalue()).decode())
                    .replace("__ACTIONS__", json.dumps([int(a) for a in actions])))


def run(output, games=10, seed=0, max_steps=30_000, tstates=100_000, policy="random"):
    if games < 1 or max_steps < 1:
        raise ValueError("games and max_steps must be positive")
    if policy not in ("random", "noop"):
        raise ValueError(policy)
    if hashlib.sha256(Path("var/defense.cmd").read_bytes()).hexdigest() != GAME_SHA256:
        raise RuntimeError("Game binary differs from the audited archive")
    # Never silently replace a previous evidence bundle.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    games_info, best = [], None
    for number in range(games):
        game_seed = seed + number
        rng = np.random.default_rng(game_seed)
        env = DefenseEnv(tstates=tstates, max_steps=max_steps)
        try:
            frames = [env.reset(game_seed)[-1]]
            actions, rewards, losses = [], [], []
            while True:
                action = int(rng.integers(len(ACTIONS))) if policy == "random" else 0
                obs, reward, terminated, truncated, info = env.step(action)
                frames.append(obs[-1])
                actions.append(action)
                rewards.append(reward)
                if info["life_lost"]:
                    losses.append(dict(step=info["steps"], remaining=info["lives"]))
                if terminated or truncated:
                    break
        finally:
            env.close()
        record = dict(seed=game_seed, policy=policy, **info, ship_losses=losses)
        games_info.append(record)
        rank = (terminated, info["missions_completed"], info["highest_stage"], info["score"])
        if best is None or rank > best[0]:
            best = (rank, np.asarray(frames), actions, rewards, record)
    _, frames, actions, rewards, record = best
    verify_trace(frames, actions, rewards, record["seed"], tstates, max_steps)
    metadata = dict(game="Obstacle Run / Missile Defense", game_sha256=GAME_SHA256,
                    environment_version=ENVIRONMENT_VERSION, action_names=ACTION_NAMES,
                    policy=policy, trained_model=False, verified_actions=len(actions),
                    tstates=tstates, max_steps=max_steps, **{"result": record})
    np.savez_compressed(output/"trace.npz", frames=frames, actions=np.asarray(actions, np.uint8),
                        rewards=np.asarray(rewards, np.float32), metadata=json.dumps(metadata))
    write_replay(output/"replay.html", frames, actions, metadata)
    sheet = Image.new("RGB", (1536, 446), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", 26)
    for panel, index in enumerate((0, min(100, len(frames)-1), len(frames)-1)):
        draw.text((panel*512+5, 5), f"Diagnostic frame {index} — {policy} actions", fill="white")
        for row, cells in enumerate(frames[index]):
            for col, code in enumerate(cells):
                draw.text((panel*512+col*8, 29+row*26), chr(0xe000+int(code)), font=font, fill="white")
    sheet.save(output/"screens.png")
    complete = [g for g in games_info if g["terminated"]]
    report = dict(**metadata, games=games_info, complete_games=len(complete),
                  mean_score=float(np.mean([g["score"] for g in complete])) if complete else None,
                  median_score=float(np.median([g["score"] for g in complete])) if complete else None,
                  best_score=max((g["score"] for g in complete), default=None),
                  highest_stage=max((g["highest_stage"] for g in complete), default=None))
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("games", "action_names")}, indent=2))
    if len(complete) != games:
        raise RuntimeError("Some games hit the diagnostic action limit; see truncated records")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--policy", choices=("random", "noop"), default="random")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
