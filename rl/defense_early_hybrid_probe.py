"""Diagnostic-only early handoff between two frozen learned Defense policies.

The fixed switch schedule is not a learned controller or training example.
Only original screens, displayed score and visible boundaries rank outcomes.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_gate_timing import visible_ship_column
from .defense_hybrid_window_probe import SOURCE, NAVIGATION, checked_source, sample
from .defense_learning import load_policy, sha256, write_json
from .defense_snapshot import restore


STARTS = (0, 40, 80, 120, 160, 200, 240)
LENGTHS = (16, 32, 64, 96, 128)
REPEATS = 8
LIMIT = 520
MILESTONES = (160, 200, 240, 300, 340, 380, 390)


def navigation_seed(start, length, replicate):
    return 1_680_000 + 10_000*start + 100*length + replicate


def rollout(env, observation, parent, navigation, parent_rng, nav_rng,
            start, length, replicate, limit):
    trace_hash = hashlib.sha256()
    landmarks = {}
    nav_actions = 0
    for _ in range(start, limit):
        using_nav = nav_actions < length
        action = sample(navigation if using_nav else parent, observation,
                        nav_rng if using_nav else parent_rng)
        nav_actions += int(using_nav)
        observation, reward, terminal, truncated, info = env.step(action)
        trace_hash.update(bytes((action,)))
        trace_hash.update(np.float64(reward).tobytes())
        trace_hash.update(observation[-1].tobytes())
        if info["steps"] in MILESTONES:
            landmarks[str(info["steps"])] = dict(displayed_score=info["score"],
                                                  ship_column=visible_ship_column(observation[-1]),
                                                  visible_lives=info["lives"])
        if info["life_lost"] or info["highest_stage"] >= 2 or info["mission_completed"] or terminal or truncated:
            break
    return dict(start=start, window=length, replicate=replicate,
                navigation_rng_seed=navigation_seed(start, length, replicate),
                navigation_actions=nav_actions, last_observed_action=info["steps"],
                first_visible_loss_action=info["steps"] if info["life_lost"] else None,
                displayed_score=info["score"], highest_stage=info["highest_stage"],
                mission_completed=info["mission_completed"],
                visible_life_lost=info["life_lost"],
                survived_through_cap=(info["steps"] >= limit and not info["life_lost"] and
                                      info["highest_stage"] == 1),
                landmarks=landmarks, branch_action_reward_screen_sha256=trace_hash.hexdigest())


def branch_from_saved(env, parent, navigation, source, start, length, replicate, limit):
    observation = restore(env, source["saved"][start])
    parent_rng = np.random.default_rng()
    parent_rng.bit_generator.state = copy.deepcopy(source["rng_states"][start])
    nav_rng = np.random.default_rng(navigation_seed(start, length, replicate))
    return rollout(env, observation, parent, navigation, parent_rng, nav_rng,
                   start, length, replicate, limit)


def verify_from_boot(env, parent, navigation, source, actions, rewards, expected, limit):
    seed, start = source["report"]["result"]["seed"], expected["start"]
    parent_rng = np.random.default_rng(seed+1_000_000)
    observation = env.reset(seed)
    np.testing.assert_array_equal(observation[-1], source["frames"][0])
    for index in range(start):
        action = sample(parent, observation, parent_rng)
        if action != int(actions[index]):
            raise RuntimeError("independent boot prefix parent action changed")
        observation, reward, terminal, truncated, info = env.step(action)
        if (reward != rewards[index] or terminal or truncated or info["life_lost"] or
                not np.array_equal(observation[-1], source["frames"][index+1])):
            raise RuntimeError("independent boot prefix changed")
    if parent_rng.bit_generator.state != source["rng_states"][start]:
        raise RuntimeError("independent boot prefix policy RNG changed")
    nav_rng = np.random.default_rng(expected["navigation_rng_seed"])
    replayed = rollout(env, observation, parent, navigation, parent_rng, nav_rng,
                       start, expected["window"], expected["replicate"], limit)
    if replayed != expected:
        raise RuntimeError("independent original-boot branch did not reproduce")
    return dict(verified=True, original_boot_prefix_actions=start,
                branch_actions=replayed["last_observed_action"]-start,
                branch_action_reward_screen_sha256=replayed["branch_action_reward_screen_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="two-branch integration check only")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a diagnostic report")
    starts = (80,) if args.smoke else STARTS
    lengths = (64,) if args.smoke else LENGTHS
    repeats = 2 if args.smoke else REPEATS
    parent_hash, nav_hash = sha256(SOURCE/"model.safetensors"), sha256(NAVIGATION)
    parent, parent_config = load_policy(SOURCE/"model.safetensors")
    navigation, nav_config = load_policy(NAVIGATION)
    if (parent_config.get("algorithm") != "ppo" or parent_config.get("movement_only") or
            parent_config.get("learned_durations") or parent_config.get("recurrent_hidden") or
            nav_config.get("algorithm") != "ppo" or nav_config.get("movement_only") is not True or
            parent_config["tstates"] != nav_config["tstates"] or
            parent_config.get("observation_stride", 1) != nav_config.get("observation_stride", 1) or
            parent_config["game_sha256"] != nav_config["game_sha256"] or
            parent_config["game_sha256"] != GAME_SHA256):
        raise RuntimeError("requires compatible frozen full-control and movement-only PPO models")
    with np.load(SOURCE/"trace.npz", allow_pickle=False) as trace:
        source_actions, source_rewards = trace["actions"], trace["rewards"]
    env = DefenseEnv(tstates=parent_config["tstates"], max_steps=0,
                     observation_stride=parent_config.get("observation_stride", 1))
    try:
        source = checked_source(env, parent, SOURCE, starts)
        branches, confirmations, smoke_confirmations = [], [], []
        for start in starts:
            for length in lengths:
                for replicate in range(repeats):
                    row = branch_from_saved(env, parent, navigation, source,
                                            start, length, replicate, LIMIT)
                    branches.append(row)
                    if args.smoke:
                        smoke_confirmations.append(dict(start=start, window=length,
                            replicate=replicate, **verify_from_boot(
                                env, parent, navigation, source, source_actions,
                                source_rewards, row, LIMIT)))
                    elif (row["highest_stage"] >= 2 or row["displayed_score"] > 2620 or
                            row["last_observed_action"] >= source["first_loss"]+20):
                        confirmations.append(dict(start=start, window=length, replicate=replicate,
                            **verify_from_boot(env, parent, navigation, source,
                                               source_actions, source_rewards, row, LIMIT)))
    finally:
        env.close()
    if sha256(SOURCE/"model.safetensors") != parent_hash or sha256(NAVIGATION) != nav_hash:
        raise RuntimeError("frozen expert weights changed during diagnostic")
    summary = dict(branches=len(branches), source_first_visible_loss_action=source["first_loss"],
                   max_observed_action=max(row["last_observed_action"] for row in branches),
                   max_displayed_first_life_score=max(row["displayed_score"] for row in branches),
                   beyond_source_by_20=sum(row["last_observed_action"] >= source["first_loss"]+20
                                           for row in branches),
                   score_above_source=sum(row["displayed_score"] > 2620 for row in branches),
                   stage_two_or_later=sum(row["highest_stage"] >= 2 for row in branches),
                   missions=sum(row["mission_completed"] for row in branches),
                   survived_to_cap=sum(row["survived_through_cap"] for row in branches),
                   readable_ship_col_40_or_more_at_380=sum(
                       row["landmarks"].get("380", {}).get("ship_column") is not None and
                       row["landmarks"]["380"]["ship_column"] >= 40 for row in branches))
    result = dict(experiment="early frozen-neural handoff diagnostic, NOT a learned controller",
                  smoke=args.smoke, source_bundle=str(SOURCE), source_seed=source["report"]["result"]["seed"],
                  source_trace_sha256=source["report"]["source_hashes"]["trace.npz"],
                  source_parent_sha256=parent_hash, movement_model_sha256=nav_hash,
                  probe_source_sha256=sha256(__file__), game_sha256=GAME_SHA256,
                  starts=starts, lengths=lengths, repeats=repeats, cap=LIMIT,
                  milestones=MILESTONES, exact_parent_prefix_and_suffix_verified=True,
                  summary=summary, independently_verified_discoveries=confirmations,
                  smoke_branch_confirmations=smoke_confirmations,
                  branches=branches, diagnostic_only=True, hidden_ram_reads=0,
                  model_updates=0, reward_changes=0, training_data_written=False,
                  policy_promotion_eligible=False,
                  limitations=["Switch times are scripted and cannot be a final learned policy.",
                               "Only one own first-life seed is tested.",
                               "The frozen parent RNG does not consume skipped-window draws.",
                               "Ship glyph can be occluded; action count is not exact course depth."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
