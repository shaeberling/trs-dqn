"""Read-only displayed-score return comparison for a quarantined route.

The route was private-state-selected elsewhere and is never a learner input,
demonstration, reward, checkpoint selector or promoted neural replay here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_position_feasibility import EXPECTED_MODEL_SHA256
from .defense_world_data import disk_guard


FORK = 280
DISCOUNTS = (1.0, 0.999, 0.997)


def discounted_return(rewards, start, stop, gamma):
    if not (0 <= start <= stop <= len(rewards) and 0 < gamma <= 1):
        raise ValueError("invalid return window or discount")
    return sum(float(rewards[index]) * gamma ** (index - start)
               for index in range(start, stop))


def required_next_reward(source_rewards, source_end, branch_rewards, fork, gamma):
    """Single hypothetical reward at the first decision after the branch trace."""
    if not 0 <= fork < source_end <= len(source_rewards) or len(branch_rewards) <= fork:
        raise ValueError("incompatible source/branch horizons")
    source = discounted_return(source_rewards, fork, source_end, gamma)
    branch = discounted_return(branch_rewards, fork, len(branch_rewards), gamma)
    gap = source - branch
    return dict(discount=gamma, source_known_return=source,
                branch_known_return=branch, return_gap=gap,
                hypothetical_next_action=len(branch_rewards) + 1,
                next_reward_to_tie=max(0.0, gap / gamma ** (len(branch_rewards) - fork)))


def run(bundle, discovery, output):
    bundle, discovery, output = Path(bundle), Path(discovery), Path(output)
    if output.exists():
        raise ValueError("fresh output required")
    report, frames, source_actions = analyze(bundle)
    if (report["source_hashes"]["model.safetensors"] != EXPECTED_MODEL_SHA256
            or report["tstates"] != 100_000 or report["result"]["score"] != 10_480):
        raise ValueError("requires the protected original-cadence learned replay")
    with np.load(bundle / "trace.npz", allow_pickle=False) as trace:
        source_rewards = trace["rewards"].copy()
        metadata = json.loads(str(trace["metadata"]))
    proof = json.loads((discovery / "discovery.json").read_text())
    if (proof["outcome"]["frame"] != 573 or proof["outcome"]["rows"] != 50
            or proof["outcome"]["score"] != 450
            or proof["outcome"]["stage"] != 1
            or not proof["verification"]["every_screen_and_reward_matched"]):
        raise ValueError("requires the verified original-boot row-50 diagnostic")
    record = discovery / "diagnostic-discovery.npz"
    if sha256(record) != proof["diagnostic_trace_sha256"]:
        raise ValueError("diagnostic action record hash mismatch")
    with np.load(record, allow_pickle=False) as data:
        actions = data["actions"].copy()
        diagnostic_only = bool(data["diagnostic_only"])
    if (not diagnostic_only or len(actions) != 573 or actions.ndim != 1
            or not np.array_equal(actions[:FORK], source_actions[:FORK])
            or hashlib.sha256(actions.tobytes()).hexdigest() != proof["outcome"]["action_sha256"]):
        raise ValueError("forensic path differs from the protected fork")
    source_end = report["lives"][0]["visible_loss_frame"]
    if source_end != 410 or source_rewards[:source_end].sum() != 2_620:
        raise ValueError("protected first-life reward/boundary changed")
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "source.py")
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    branch_rewards = []
    try:
        obs = env.reset(metadata["result"]["seed"])
        np.testing.assert_array_equal(obs[-1], frames[0])
        for index, action in enumerate(actions):
            obs, reward, terminal, truncated, info = env.step(int(action))
            if terminal or truncated or info["life_lost"]:
                raise RuntimeError("forensic branch ended before row-50 endpoint")
            if index < FORK:
                if reward != source_rewards[index]:
                    raise RuntimeError("protected prefix reward changed")
                np.testing.assert_array_equal(obs[-1], frames[index + 1])
            branch_rewards.append(reward)
        if (info["score"], info["lives"], info["stage"]) != (450, 4, 1):
            raise RuntimeError("forensic row-50 displayed endpoint changed")
    finally:
        env.close()
    branch_rewards = np.asarray(branch_rewards)
    if not np.array_equal(branch_rewards[:FORK], source_rewards[:FORK]):
        raise RuntimeError("shared source reward prefix changed")
    result = dict(game_sha256=GAME_SHA256,
                  source_sha256=sha256(__file__),
                  protected_model_sha256=report["source_hashes"]["model.safetensors"],
                  protected_trace_sha256=report["source_hashes"]["trace.npz"],
                  forensic_action_record_sha256=sha256(record),
                  original_boot_reexecuted=True, diagnostic_only=True,
                  policy_updates=0, training_data_written=False, promotion_eligible=False,
                  fork_action=FORK, fork_displayed_score=float(source_rewards[:FORK].sum()),
                  source_visible_first_loss_action=source_end,
                  source_post_fork_displayed_score=float(source_rewards[FORK:source_end].sum()),
                  branch_alive_through_action=len(actions),
                  branch_post_fork_displayed_score=float(branch_rewards[FORK:].sum()),
                  return_comparisons=[required_next_reward(
                      source_rewards, source_end, branch_rewards, FORK, gamma)
                      for gamma in DISCOUNTS],
                  interpretation=("A single hypothetical score increment on the next action "
                                  "would tie known discounted first-life returns from the fork. "
                                  "No such increment is observed or assumed. This is not PPO GAE, "
                                  "a full-game return, a stage-clear probability or a training target."))
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.discovery, args.output)
    print(json.dumps(dict(event="finished", fork_action=result["fork_action"],
                          returns=result["return_comparisons"])), flush=True)


if __name__ == "__main__":
    main()
