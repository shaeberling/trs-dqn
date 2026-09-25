"""Diagnostic-only frozen-policy switching near one verified Defense loss.

The schedule is an intervention, NOT a learned policy or training example.
Only original displayed score, visible life/stage events and screens are read.
No branch can be promoted as a neural replay or used as a demonstration.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_learning import load_policy, sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


SOURCE = Path("results/defense/training/ppo-balanced-fire-continuation-161/fresh-comparison/continuation-612000-replay")
NAVIGATION = Path("results/defense/training/ppo-movement-only-164/extension/milestone-000002097152/checkpoint/model.safetensors")
STARTS = (300, 320, 340, 360)
LENGTHS = (16, 32, 48, 64)
REPEATS = 16
LIMIT = 520


def sample(policy, observation, rng):
    return int(policy.sample_with_rngs(observation[None], [rng])[0])


def checked_source(env, parent, bundle, starts):
    report, frames, actions = analyze(bundle)
    with np.load(bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]
        metadata = json.loads(str(trace["metadata"]))
    if (report["result"]["highest_stage"] != 1 or
            report["source_hashes"]["model.safetensors"] != sha256(bundle/"model.safetensors") or
            report["tstates"] != env.tstates or
            metadata.get("observation_stride", 1) != env.observation_stride or
            max(starts) >= report["lives"][0]["visible_loss_frame"]):
        raise RuntimeError("source replay is incompatible with the branch plan")
    seed = report["result"]["seed"]
    rng = np.random.default_rng(seed+1_000_000)
    observation = env.reset(seed)
    np.testing.assert_array_equal(observation[-1], frames[0])
    saved, rng_states = {}, {}
    first_loss = report["lives"][0]["visible_loss_frame"]
    for index in range(first_loss):
        if index in starts:
            saved[index] = capture(env)
            rng_states[index] = copy.deepcopy(rng.bit_generator.state)
        action = sample(parent, observation, rng)
        if action != int(actions[index]):
            raise RuntimeError(f"parent model RNG diverged at source action {index}")
        observation, reward, terminal, truncated, info = env.step(action)
        if (reward != rewards[index] or
                not np.array_equal(observation[-1], frames[index+1]) or
                terminal or truncated or info["life_lost"] != (index+1 == first_loss)):
            raise RuntimeError(f"original source did not reexecute at action {index}")
    if set(saved) != set(starts) or info["score"] != 2620:
        raise RuntimeError("source life or capture set changed")
    for start in starts:
        observation = restore(env, saved[start])
        np.testing.assert_array_equal(observation[-1], frames[start])
        rng.bit_generator.state = copy.deepcopy(rng_states[start])
        for index in range(start, first_loss):
            if sample(parent, observation, rng) != int(actions[index]):
                raise RuntimeError("restored parent RNG differs from source")
            observation, reward, terminal, truncated, info = env.step(int(actions[index]))
            if reward != rewards[index] or not np.array_equal(observation[-1], frames[index+1]):
                raise RuntimeError("restored parent suffix differs from source")
        if not info["life_lost"] or info["steps"] != first_loss or info["score"] != 2620:
            raise RuntimeError("restored parent did not reach its known visible loss")
    return dict(report=report, frames=frames, saved=saved, rng_states=rng_states,
                first_loss=first_loss)


def run_branch(env, parent, navigation, saved, parent_rng_state,
               start, length, replicate, limit):
    observation = restore(env, saved)
    parent_rng = np.random.default_rng()
    parent_rng.bit_generator.state = copy.deepcopy(parent_rng_state)
    navigation_seed = 1_650_000 + 10_000*start + 100*length + replicate
    navigation_rng = np.random.default_rng(navigation_seed)
    nav_actions = 0
    for _ in range(start, limit):
        use_navigation = nav_actions < length
        action = sample(navigation if use_navigation else parent, observation,
                        navigation_rng if use_navigation else parent_rng)
        nav_actions += int(use_navigation)
        observation, _, terminal, truncated, info = env.step(action)
        if info["life_lost"] or info["highest_stage"] >= 2 or info["mission_completed"] or terminal or truncated:
            break
    return dict(start=start, window=length, replicate=replicate,
                navigation_rng_seed=navigation_seed, navigation_actions=nav_actions,
                first_visible_loss_action=info["steps"] if info["life_lost"] else None,
                last_observed_action=info["steps"], displayed_score=info["score"],
                highest_stage=info["highest_stage"], mission_completed=info["mission_completed"],
                visible_life_lost=info["life_lost"], survived_through_cap=(
                    info["steps"] >= limit and not info["life_lost"] and info["highest_stage"] == 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="two-branch integration check, not outcome evidence")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a diagnostic")
    starts = (320,) if args.smoke else STARTS
    lengths = (32,) if args.smoke else LENGTHS
    repeats = 2 if args.smoke else REPEATS
    parent_hash, navigation_hash = sha256(SOURCE/"model.safetensors"), sha256(NAVIGATION)
    parent, parent_config = load_policy(SOURCE/"model.safetensors")
    navigation, nav_config = load_policy(NAVIGATION)
    if (parent_config.get("algorithm") != "ppo" or parent_config.get("movement_only") or
            parent_config.get("learned_durations") or parent_config.get("recurrent_hidden") or
            nav_config.get("algorithm") != "ppo" or nav_config.get("movement_only") is not True or
            parent_config["tstates"] != nav_config["tstates"] or
            parent_config.get("observation_stride", 1) != nav_config.get("observation_stride", 1) or
            parent_config["game_sha256"] != nav_config["game_sha256"] or
            parent_config["game_sha256"] != GAME_SHA256):
        raise RuntimeError("requires compatible ordinary parent and movement-only neural policies")
    env = DefenseEnv(tstates=parent_config["tstates"], max_steps=0,
                     observation_stride=parent_config.get("observation_stride", 1))
    try:
        source = checked_source(env, parent, SOURCE, starts)
        branches = []
        for start in starts:
            for length in lengths:
                for replicate in range(repeats):
                    branches.append(run_branch(env, parent, navigation, source["saved"][start],
                                               source["rng_states"][start], start, length,
                                               replicate, LIMIT))
    finally:
        env.close()
    if sha256(SOURCE/"model.safetensors") != parent_hash or sha256(NAVIGATION) != navigation_hash:
        raise RuntimeError("frozen expert weights changed during diagnostic")
    baseline = source["first_loss"]
    summary = dict(branches=len(branches), source_first_visible_loss_action=baseline,
                   max_observed_action=max(row["last_observed_action"] for row in branches),
                   beyond_source_by_20=sum(row["last_observed_action"] >= baseline+20
                                           for row in branches),
                   stage_two_or_later=sum(row["highest_stage"] >= 2 for row in branches),
                   missions=sum(row["mission_completed"] for row in branches),
                   survived_to_cap=sum(row["survived_through_cap"] for row in branches))
    result = dict(experiment="frozen-neural-expert random-window diagnostic, NOT a learned gate",
                  smoke=args.smoke, source_bundle=str(SOURCE), source_seed=source["report"]["result"]["seed"],
                  source_trace_sha256=source["report"]["source_hashes"]["trace.npz"],
                  source_parent_sha256=parent_hash,
                  movement_model_sha256=navigation_hash, source_sha256=sha256(__file__),
                  game_sha256=GAME_SHA256, starts=starts, lengths=lengths, repeats=repeats,
                  cap=LIMIT, exact_parent_prefix_and_suffix_verified=True,
                  summary=summary, branches=branches, diagnostic_only=True,
                  hidden_ram_reads=0, model_updates=0, reward_changes=0,
                  branch_actions_as_demonstrations=False,
                  policy_promotion_eligible=False,
                  limitations=["Switch schedule is scripted and cannot be a final policy.",
                               "Only one verified own first-life source state/seed is tested.",
                               "Post-window policy RNG has not consumed the skipped parent draws.",
                               "Action count includes animations and is not exact course depth."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
