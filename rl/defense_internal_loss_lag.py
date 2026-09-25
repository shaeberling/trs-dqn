"""Forensic-only timing of one original Defense ship loss.

The private counter is read only here, after a frozen own-policy replay exists.
No diagnostic value is a policy input, training reward, target or action source.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv, GAME_SHA256
from .defense_learning import sha256, write_json
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


SOURCE = Path("results/defense/training/ppo-balanced-fire-continuation-161/fresh-comparison/continuation-612000-replay")
INTERNAL_SHIPS = 0x7CEF  # original-game disassembly; forensic only
SCAN_FROM = 280


def private_ships(env):
    return int(env.trs.ram.peek(INTERNAL_SHIPS))


def run_recorded_key(env, action, requested_tstates):
    env.trs.keyboard.all_keys_up()
    for key in env.actions[int(action)]:
        env.trs.keyboard.key_down(key)
    env.trs.run_for_tstates(requested_tstates)
    return private_ships(env)


def first_decrement_threshold(env, saved, action, before):
    """Bisect a requested duration from the same exact pre-action state."""
    low, high = -1, env.tstates
    while high-low > 1:
        middle = (low+high)//2
        restore(env, saved)
        count = run_recorded_key(env, action, middle)
        if count < before:
            high = middle
        else:
            low = middle
    restore(env, saved)
    if run_recorded_key(env, action, high) != before-1:
        raise RuntimeError("private counter transition did not reproduce")
    restore(env, saved)
    return high


def measure(bundle):
    report, frames, actions = analyze(bundle)
    with np.load(bundle/"trace.npz", allow_pickle=False) as trace:
        rewards = trace["rewards"]
        metadata = json.loads(str(trace["metadata"]))
    visible_loss = report["lives"][0]["visible_loss_frame"]
    if (visible_loss != 412 or visible_loss <= SCAN_FROM or
            report["lives"][0]["visible_score_at_loss"] != 2620 or
            report["result"]["highest_stage"] != 1 or
            report["source_hashes"]["model.safetensors"] != sha256(bundle/"model.safetensors")):
        raise RuntimeError("requires the exact verified first-life source")
    env = DefenseEnv(tstates=report["tstates"], max_steps=0,
                     observation_stride=metadata.get("observation_stride", 1))
    try:
        observation = env.reset(report["result"]["seed"])
        np.testing.assert_array_equal(observation[-1], frames[0])
        if private_ships(env) != 4:
            raise RuntimeError("internal ship count did not start at four")
        event = None
        for index in range(visible_loss):
            before = private_ships(env)
            saved, base_after = None, None
            if index >= SCAN_FROM and event is None:
                if before != 4:
                    raise RuntimeError("private ship count changed before scan captured the event")
                saved = capture(env)
                base_after = run_recorded_key(env, actions[index], env.tstates)
                restore(env, saved)
                if base_after < before:
                    if base_after != before-1:
                        raise RuntimeError("unexpected private ship counter transition")
                    threshold = first_decrement_threshold(env, saved, actions[index], before)
                    event = dict(action_number=index+1, action_index=index,
                                 requested_tstates_threshold=threshold,
                                 interval_tstates=env.tstates, during_base_action=True,
                                 internal_ships_before=before, internal_ships_after=base_after,
                                 visible_hud_ships_before=env.lives,
                                 visible_score_before=env.score)
            observation, reward, terminated, truncated, info = env.step(int(actions[index]))
            if (reward != rewards[index] or
                    not np.array_equal(observation[-1], frames[index+1]) or
                    terminated or truncated or
                    info["life_lost"] != (index+1 == visible_loss)):
                raise RuntimeError(f"original source diverged at action {index+1}")
            after = private_ships(env)
            if saved is not None and base_after == before and after < before:
                if after != before-1:
                    raise RuntimeError("unexpected private ship counter change during HUD settle")
                event = dict(action_number=index+1, action_index=index,
                             requested_tstates_threshold=None,
                             interval_tstates=env.tstates, during_base_action=False,
                             internal_ships_before=before, internal_ships_after=after,
                             visible_hud_ships_before=before,
                             visible_score_before=info["score"]-reward)
            if event is None and after != 4:
                raise RuntimeError("private ship count changed before event was located")
        if event is None or private_ships(env) != 3 or info["score"] != 2620:
            raise RuntimeError("first private and visible ship-loss events did not reconcile")
        return dict(source_bundle=str(bundle), source_seed=report["result"]["seed"],
                    source_model_sha256=report["source_hashes"]["model.safetensors"],
                    source_trace_sha256=report["source_hashes"]["trace.npz"],
                    probe_source_sha256=sha256(__file__), game_sha256=GAME_SHA256,
                    audited_private_counter_address=f"0x{INTERNAL_SHIPS:04X}",
                    scan_from_action=SCAN_FROM, internal_decrement=event,
                    first_visible_loss_action=visible_loss,
                    visible_lag_actions=visible_loss-event["action_number"],
                    exact_original_boot_actions_rewards_screens_verified=visible_loss,
                    diagnostic_only=True, model_updates=0, training_data_written=False,
                    hidden_counter_used_by_policy_or_reward=False,
                    promotion_eligible=False,
                    limitations=["One frozen original-boot first life, not a distribution.",
                                 "The counter decrement is the original game's loss bookkeeping; "
                                 "the overlap test can precede it.",
                                 "Requested T-state bisection resolves an instruction interval, "
                                 "not a pixel-accurate collision moment.",
                                 "No internal counter value may enter training or acting."])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a forensic report")
    result = measure(SOURCE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(internal_decrement=result["internal_decrement"],
                          first_visible_loss_action=result["first_visible_loss_action"],
                          visible_lag_actions=result["visible_lag_actions"])))


if __name__ == "__main__":
    main()
