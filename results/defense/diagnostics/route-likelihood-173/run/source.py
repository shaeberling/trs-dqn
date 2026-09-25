"""Frozen-policy likelihood of a quarantined forensic Defense route.

This is a read-only audit. Searched actions are never training examples,
policy inputs, rewards, checkpoint candidates or learned replays.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv
from .defense_canonical_fire import group_logits_numpy
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_world_data import disk_guard


ANCHOR = 280
SOURCE_LOSS_FRAME = 389
DIAGNOSTIC_FRAME = 428
MODEL_SHA256 = "125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171"
ROUTE_SHA256 = "b6bf6578d6d507d24e748c6159056b42ed1f498722839ee880e4588c2315bb15"


def physical_indices(actions):
    values = np.asarray(actions)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= 20)):
        raise ValueError("twenty-action physical command indices required")
    return np.where(values >= 18, values - 8, np.where(values >= 9, 9, values)).astype(np.int32)


def physical_log_probs(raw_logits, actions):
    grouped = group_logits_numpy(raw_logits)
    choices = physical_indices(actions)
    if len(grouped) != len(choices):
        raise ValueError("one action per screen")
    return grouped[np.arange(len(choices)), choices] - np.logaddexp.reduce(grouped, axis=1)


def likelihood_summary(log_probs, actions):
    log_probs = np.asarray(log_probs)
    commands = physical_indices(actions)
    if len(log_probs) != len(commands) or not len(commands) or not np.isfinite(log_probs).all():
        raise ValueError("finite nonempty action likelihoods required")
    return dict(actions=len(commands), mean_log_probability=float(log_probs.mean()),
                geometric_mean_physical_probability=float(np.exp(log_probs.mean())),
                below_one_percent=int((log_probs < np.log(.01)).sum()),
                physical_command_counts=np.bincount(commands, minlength=12).tolist())


def run(bundle, diagnostic, output):
    # Keep Metal loading out of pure tests and any emulator worker imports.
    import mlx.core as mx
    from .model import QNetwork

    bundle, diagnostic, output = Path(bundle), Path(diagnostic), Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite a diagnostic result")
    source_report, frames, actions = analyze(bundle)
    if (source_report["source_hashes"]["model.safetensors"] != MODEL_SHA256
            or source_report["tstates"] != 100000
            or source_report["lives"][0]["visible_loss_frame"] <= SOURCE_LOSS_FRAME):
        raise ValueError("requires the protected verified original-boot source")
    discovery = json.loads((diagnostic / "discovery.json").read_text())
    if (discovery.get("outcome", {}).get("action_sha256") != ROUTE_SHA256
            or discovery.get("outcome", {}).get("frame") != DIAGNOSTIC_FRAME
            or discovery.get("verification", {}).get("verified") is not True
            or discovery["verification"].get("original_boot_actions") != DIAGNOSTIC_FRAME):
        raise ValueError("requires the independently verified diagnostic route")
    with np.load(bundle / "trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    with np.load(diagnostic / "diagnostic-discovery.npz", allow_pickle=False) as trace:
        route = trace["actions"].copy()
        if trace["diagnostic_only"].item() is not True:
            raise ValueError("route is not quarantined as diagnostic")
    if (hashlib.sha256(route.tobytes()).hexdigest() != ROUTE_SHA256
            or len(route) != DIAGNOSTIC_FRAME - ANCHOR
            or metadata.get("observation_stride", 1) != 1):
        raise ValueError("diagnostic route or screen history differs")

    env = DefenseEnv(tstates=100000, max_steps=0)
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        for index in range(ANCHOR):
            obs, reward, terminal, truncated, info = env.step(int(actions[index]))
            if reward != rewards[index] or terminal or truncated or info["life_lost"]:
                raise RuntimeError("protected source prefix diverged")
            np.testing.assert_array_equal(obs[-1], frames[index + 1])
        diagnostic_obs = []
        for action in route:
            diagnostic_obs.append(obs.copy())
            obs, _, terminal, truncated, info = env.step(int(action))
            if terminal or truncated or info["life_lost"]:
                raise RuntimeError("quarantined route did not survive")
        if (env.steps != DIAGNOSTIC_FRAME or env.score != 390 or env.stage != 1
                or env.trs.ram.peek(0x7CEF) != 4):
            raise RuntimeError("diagnostic original-boot outcome changed")
    finally:
        env.close()
    source_obs = np.stack([frames[index-3:index+1]
                           for index in range(ANCHOR, SOURCE_LOSS_FRAME)])
    diagnostic_obs = np.stack(diagnostic_obs)
    source_actions = actions[ANCHOR:SOURCE_LOSS_FRAME].astype(np.int32)
    diagnostic_actions = route[:SOURCE_LOSS_FRAME-ANCHOR].astype(np.int32)
    model = QNetwork(action_count=20)
    model.load_weights(str(bundle / "model.safetensors"))
    mx.eval(model.state)
    source_logits = np.array(model.policy_value(mx.array(source_obs))[0])
    diagnostic_logits = np.array(model.policy_value(mx.array(diagnostic_obs))[0])
    source_on_source = physical_log_probs(source_logits, source_actions)
    source_on_diagnostic = physical_log_probs(diagnostic_logits[:len(source_actions)], source_actions)
    diagnostic_on_source = physical_log_probs(source_logits, diagnostic_actions)
    diagnostic_on_diagnostic = physical_log_probs(diagnostic_logits[:len(source_actions)],
                                                diagnostic_actions)
    later = physical_log_probs(diagnostic_logits[len(source_actions):], route[len(source_actions):])
    result = dict(model_sha256=MODEL_SHA256,
                  source_trace_sha256=source_report["source_hashes"]["trace.npz"],
                  diagnostic_route_sha256=ROUTE_SHA256,
                  diagnostic_record_sha256=sha256(diagnostic / "diagnostic-discovery.npz"),
                  source_sha256=sha256(__file__), anchor=ANCHOR,
                  divergence_window=[ANCHOR, SOURCE_LOSS_FRAME],
                  diagnostic_later_window=[SOURCE_LOSS_FRAME, DIAGNOSTIC_FRAME],
                  source_on_source=likelihood_summary(source_on_source, source_actions),
                  source_on_diagnostic=likelihood_summary(source_on_diagnostic, source_actions),
                  diagnostic_on_source=likelihood_summary(diagnostic_on_source, diagnostic_actions),
                  diagnostic_on_diagnostic=likelihood_summary(diagnostic_on_diagnostic, diagnostic_actions),
                  diagnostic_later=likelihood_summary(later, route[len(source_actions):]),
                  interpretation="Frozen model action likelihoods on recorded visible screens only",
                  diagnostic_only=True, model_updates=0, training_data_written=False,
                  promotion_eligible=False,
                  limitations=["Exact searched route probability is not probability of any surviving route.",
                               "The route was selected using private RAM for diagnostics only.",
                               "Cross-scoring uses different screen trajectories, not a controlled intervention.",
                               "The learned model, its replay and both live training arms are unchanged."])
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.diagnostic, args.output)
    print(json.dumps({key: result[key]["geometric_mean_physical_probability"] for key in
                      ("source_on_source", "source_on_diagnostic", "diagnostic_on_source",
                       "diagnostic_on_diagnostic", "diagnostic_later")}), flush=True)


if __name__ == "__main__":
    main()
