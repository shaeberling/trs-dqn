"""Read-only frozen score-value probe on a verified own-policy loss replay.

This measures what the existing value head predicts before visible losses.
It does not train, choose actions, change reward or promote a replay.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_learning import sha256
from .defense_loss_probe import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic")

    report, frames, _ = analyze(args.bundle)
    state = json.loads((args.bundle/"state.json").read_text())
    config = state["config"]
    if config.get("algorithm") != "ppo" or config.get("recurrent_hidden"):
        parser.error("requires an ordinary feedforward PPO replay")
    with np.load(args.bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]

    import mlx.core as mx
    from .model import QNetwork
    names = config.get("policy_action_names", config["action_names"])
    model = QNetwork(action_count=len(names))
    model.load_weights(str(args.bundle/"model.safetensors"))
    mx.eval(model.state)

    offsets = (64, 48, 32, 16, 8)
    points = []
    for life in report["lives"]:
        stop, start = life["visible_loss_frame"], life["previous_visible_loss_frame"]
        for offset in offsets:
            index = stop-offset
            if index <= start:
                continue
            points.append((life["life"], offset, index, stop))
    stacks = np.stack([frames[np.maximum(0, np.arange(index-3, index+1))]
                       for _, _, index, _ in points])
    _, values = model.policy_value(mx.array(stacks))
    mx.eval(values)
    rows = [dict(life=life, decisions_before_visible_loss=offset,
                 predicted_remaining_score_points=float(value)/config["reward_scale"],
                 actual_remaining_visible_score_points=float(np.sum(rewards[index:stop])))
            for (life, offset, index, stop), value in zip(points, np.asarray(values), strict=True)]
    output = dict(bundle=str(args.bundle),
                  model_sha256=sha256(args.bundle/"model.safetensors"),
                  trace_sha256=sha256(args.bundle/"trace.npz"),
                  probe_source_sha256=sha256(Path(__file__)),
                  original_native_verification=report["original_native_verification"],
                  diagnostic_only=True, training_data_written=False,
                  model_updates=0, promotion_eligible=False,
                  rows=rows,
                  limitations=[
                      "One selected four-life replay, not a held-out calibration study.",
                      "Predictions are the score critic's values, not collision probabilities.",
                      "Visible life loss may lag the physical collision.",
                      "No value, frame, label or action from this diagnostic enters training."
                  ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
