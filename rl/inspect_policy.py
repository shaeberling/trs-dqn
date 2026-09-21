"""Inspect PPO probabilities on an existing replay; no training or new gameplay."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from .env import screen_info, validate_observation_stride
from .evaluate import checkpoint_config


def decode_replay(html):
    """Read the recorder's JSON/base64 literals without executing JavaScript."""
    def literal(name):
        match = re.search(rf"^const {name} = (.+);$", html, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing replay {name}")
        return json.loads(match.group(1))

    metadata, actions = literal("metadata"), literal("actions")
    match = re.search(r"^const data = Uint8Array.from\(atob\('([^']+)'\)", html, re.MULTILINE)
    if match is None:
        raise ValueError("Missing replay frame payload")
    data = base64.b64decode(match.group(1), validate=True)
    if len(data) < 1024:
        raise ValueError("Incomplete initial frame")
    frames, cursor = [np.frombuffer(data[:1024], np.uint8).copy()], 1024
    while cursor < len(data):
        if cursor + 2 > len(data):
            raise ValueError("Incomplete frame header")
        count, = struct.unpack_from("<H", data, cursor)
        cursor += 2
        if count > 1024 or cursor + count*3 > len(data):
            raise ValueError("Invalid frame delta length")
        frame = frames[-1].copy()
        for _ in range(count):
            index, value = struct.unpack_from("<HB", data, cursor)
            cursor += 3
            if index >= 1024:
                raise ValueError("Frame delta index is outside the screen")
            frame[index] = value
        frames.append(frame)
    if (len(frames) != metadata["frames"] or len(actions) != len(frames)-1
            or len(actions) != metadata["steps"]):
        raise ValueError("Replay frame/action counts do not match metadata")
    if any(type(action) is not int or not 0 <= action < 6 for action in actions):
        raise ValueError("Invalid recorded action")
    return metadata, np.stack(frames).reshape(-1, 16, 64), np.asarray(actions, np.int32)


def frame_stacks(frames, start, stop, observation_stride=1):
    # Replay frame i is the latest observation BEFORE recorded action i.
    stride = validate_observation_stride(observation_stride)
    indices = np.arange(start, stop)[:, None] + stride*np.arange(-3, 1)[None, :]
    return frames[np.maximum(indices, 0)]


def probability_metrics(probs, actions):
    if not len(actions):
        return {"actions": 0}
    probs = np.asarray(probs, np.float64)
    directions = probs[:, :3] + probs[:, 3:]
    serves = np.stack((probs[:, :3].sum(1), probs[:, 3:].sum(1)), axis=1)

    def entropy(p):
        return float(np.mean(-np.sum(p*np.log(np.maximum(p, 1e-300)), axis=1)))

    return {
        "actions": len(actions),
        "action_entropy_nats": entropy(probs),
        "direction_entropy_nats": entropy(directions),
        "serve_entropy_nats": entropy(serves),
        "mean_non_greedy_direction_probability": float(np.mean(1-directions.max(1))),
        "recorded_non_greedy_direction_fraction": float(np.mean(actions % 3 != directions.argmax(1))),
        "mean_serve_probability": float(serves[:, 1].mean()),
        "mean_recorded_action_probability": float(probs[np.arange(len(actions)), actions].mean()),
    }


def inspect(checkpoint, replay, batch_size):
    import mlx.core as mx
    from .model import QNetwork

    html = replay.read_text()
    metadata, frames, actions = decode_replay(html)
    observation_stride = validate_observation_stride(metadata.get("observation_stride", 1))
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if metadata.get("checkpoint_sha256") != model_hash:
        raise ValueError("Checkpoint hash differs from the recorded policy")
    if (checkpoint_config(checkpoint).get("algorithm") != "ppo"
            or metadata.get("deterministic_override")
            or metadata.get("policy") != "learned categorical, sampled"):
        raise ValueError("This diagnostic requires a sampled PPO recording")
    if not metadata.get("terminated") or metadata.get("truncated") or not len(actions):
        raise ValueError("Use a nonempty complete-game recording")
    model = QNetwork()
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    predict = mx.compile(model.policy_value, inputs=model.state)
    probs = np.empty((len(actions), 6), np.float32)
    for start in range(0, len(actions), batch_size):
        stop = min(len(actions), start+batch_size)
        logits, _ = predict(mx.array(frame_stacks(frames, start, stop, observation_stride)))
        logits = np.array(logits)
        probs[start:stop] = np.exp(logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True))
    rng = np.random.default_rng(metadata["seed"]+1_000_000)
    reproduced = (rng.random(len(actions))[:, None] > np.cumsum(probs, axis=1)).sum(1).clip(0, 5)
    mismatches = int(np.count_nonzero(reproduced != actions))
    if mismatches:
        raise ValueError(f"{mismatches} seeded actions differ; retry with --batch-size 1")
    info = [screen_info(frame) for frame in frames]
    if (info[-1]["score"] != metadata["score"] or not info[-1]["game_over"]
            or max(row["level"] or 1 for row in info) != metadata["level"]):
        raise ValueError("Decoded screen outcome differs from replay metadata")
    levels = np.asarray([row["level"] or 1 for row in info[:-1]])
    waiting = np.asarray([row["waiting"] for row in info[:-1]])
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": model_hash,
        "replay": str(replay), "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "seed": metadata["seed"], "score": metadata["score"], "level": metadata["level"],
        "observation_stride": observation_stride, "tstates": metadata.get("tstates"),
        "batch_size": batch_size, "seeded_action_mismatches": mismatches,
        "scope": "Post-hoc inference on one selected recorded game; not a performance estimate.",
        "all": probability_metrics(probs, actions),
        "playing": probability_metrics(probs[~waiting], actions[~waiting]),
        "waiting_for_serve": probability_metrics(probs[waiting], actions[waiting]),
        "by_visible_level": {str(level): probability_metrics(probs[levels == level], actions[levels == level])
                             for level in sorted(set(levels.tolist()))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    result = inspect(args.checkpoint, args.replay, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
