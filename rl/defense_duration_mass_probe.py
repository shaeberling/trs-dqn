"""Forensic-only frozen duration probabilities on a verified own replay.

These are hypothetical decisions at sampled visible frames, including frames
where a prior hold would force continuation. No weights, actions or rewards
are changed and this is never imported by a trainer.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .defense_duration_ppo import duration_action_names
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .model import QNetwork


def stacked_frames(frames, indices, stride):
    if (frames.ndim != 3 or frames.shape[1:] != (16, 64)
            or frames.dtype != np.uint8 or stride < 1
            or any(i < 0 or i >= len(frames) for i in indices)):
        raise ValueError("invalid visible replay frames or sample indices")
    return np.stack([frames[[max(0, i-3*stride), max(0, i-2*stride),
                             max(0, i-stride), i]] for i in indices])


def measure(checkpoint, bundle):
    checkpoint, bundle = Path(checkpoint), Path(bundle)
    report, frames, _ = analyze(bundle)
    if sha256(checkpoint) != report["source_hashes"]["model.safetensors"]:
        raise ValueError("checkpoint does not match verified own replay")
    config = json.loads((checkpoint.parent/"state.json").read_text())["config"]
    durations = config.get("learned_durations")
    if (config.get("algorithm") != "ppo" or config.get("architecture") is not None
            or not isinstance(durations, list)
            or config.get("policy_action_names") != list(duration_action_names(durations))):
        raise ValueError("requires an ordinary learned-duration Defense policy")
    with np.load(bundle/"trace.npz", allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata"]))
    stride = metadata.get("observation_stride", 1)
    if stride != config.get("observation_stride", 1):
        raise ValueError("replay/checkpoint observation stride mismatch")
    first = report["lives"][0]
    marker = first["first_major_white_flash_in_last_128_frames"]
    if marker is None or marker < 224:
        raise ValueError("first visible white alignment marker unavailable")
    model = QNetwork(action_count=20*len(durations))
    model.load_weights(str(checkpoint))
    windows = []
    for label, start, stop in (("first_life_before_marker", 0, marker),
                               ("marker_minus_224_to_128", marker-224, marker-128),
                               ("marker_minus_128_to_64", marker-128, marker-64),
                               ("marker_minus_64_to_0", marker-64, marker)):
        indices = list(range(start, stop, 4))
        observations = stacked_frames(frames, indices, stride)
        logits, _ = model.policy_value(mx.array(observations))
        mass = np.array(mx.softmax(logits, axis=-1)).reshape(len(indices), len(durations), 20).sum(axis=2)
        windows.append(dict(label=label, sample_frames=len(indices), range=[start, stop],
                            mean_duration_mass=mass.mean(axis=0).astype(float).tolist(),
                            max_duration_mass=mass.max(axis=0).astype(float).tolist()))
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                trace_sha256=report["source_hashes"]["trace.npz"],
                probe_source_sha256=sha256(Path(__file__)),
                durations=durations, observation_stride=stride,
                first_visible_loss_frame=first["visible_loss_frame"],
                first_major_white_marker_frame=marker, windows=windows,
                diagnostic_only=True, model_updates=0, training_data_written=False,
                promotion_eligible=False,
                limitations=["Sampled frames include forced continuation, not just actual option starts.",
                             "A visible white flash is an alignment marker, not a collision timestamp.",
                             "One selected replay is not a fresh-sample action-distribution estimate."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    result = measure(args.checkpoint, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
