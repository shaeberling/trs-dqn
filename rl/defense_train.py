"""Train a fresh Defense PPO policy, preserving and verifying every new best.

Unlimited by default. SIGINT/SIGTERM saves a resumable latest checkpoint.
No hidden-state inputs, action overrides, demonstrations or game patches.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from .defense import action_names, screen_info, ENVIRONMENT_VERSION, GAME_SHA256, validate_observation_stride
from .defense_learning import evaluate, publish_best, sha256, summarize, write_json
from .ppo import PPO, gae
from .vector import VectorEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("results/defense/learned"))
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path)
    start.add_argument("--initialize-encoder", type=Path,
                       help="fresh learner using only an own Defense checkpoint's screen encoder")
    start.add_argument("--initialize-policy", type=Path,
                       help="own feedforward PPO checkpoint for a zero-output recurrent residual base")
    start.add_argument("--initialize-repeat-policy", type=Path,
                       help="own ordinary PPO checkpoint for a fresh continue-previous-action learner")
    start.add_argument("--initialize-duration-policy", type=Path,
                       help="own ordinary PPO checkpoint for a fresh learned key-duration policy")
    start.add_argument("--initialize-duration-checkpoint", type=Path,
                       help="copy own trained key-duration weights with fresh optimizer and policy RNG")
    parser.add_argument("--continue-initial-bias-offset", type=float, default=0.,
                        help="direction-neutral extra-action bias on own-policy initialization; 0 preserves mean row")
    parser.add_argument("--steps", type=int, default=0, help="absolute action limit; 0 = unlimited")
    parser.add_argument("--initialize-only", action="store_true",
                        help="save initialized weights/optimizer and exit without learning")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy", type=float, default=.02)
    parser.add_argument("--value-coefficient", type=float, default=.5,
                        help="weight on half mean-squared value error; default preserves prior PPO")
    parser.add_argument("--recurrent-hidden", type=int, default=0,
                        help="0 preserves feedforward PPO; positive adds screen-history GRU memory")
    parser.add_argument("--recurrent-own-action", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="also give the recurrent GRU its own previous physical key; never game state")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--memory-scale", type=float, default=1.,
                        help="recurrent residual scale; 0 is a matched memory-disabled control")
    parser.add_argument("--freeze-recurrent-base", action=argparse.BooleanOptionalAction, default=False,
                        help="train only recurrent residuals over an own initialized, frozen PPO base")
    parser.add_argument("--policy-bias-noise", type=float, default=0,
                        help="training-only actor bias noise std, fixed per life; 0 disables")
    parser.add_argument("--policy-weight-noise", type=float, default=0,
                        help="training-only actor output-weight noise std, fixed per life; 0 disables")
    parser.add_argument("--policy-key-noise", type=float, default=0,
                        help="training-only symmetric key-factor logit noise std, fixed per life; 0 disables")
    parser.add_argument("--policy-duration-noise", type=float, default=0,
                        help="training-only direction-neutral duration-factor logit noise std per life")
    parser.add_argument("--policy-key-noise-interval", type=int, default=0,
                        help="redraw key factors after N own actions per worker; 0 = only at visible boundaries")
    parser.add_argument("--policy-key-noise-min-interval", type=int, default=0,
                        help="random renewal: minimum own actions between key-factor redraws")
    parser.add_argument("--policy-key-noise-max-interval", type=int, default=0,
                        help="random renewal: maximum own actions between key-factor redraws")
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--reward-scale", type=float, default=.01,
                        help="constant units conversion; no shaping or clipping")
    parser.add_argument("--novelty-beta", type=float, default=0.,
                        help="training-only first visit to a HUD-free visible-screen cell per life; 0 disables")
    parser.add_argument("--alive-beta", type=float, default=0.,
                        help="training-only bonus for a visible gameplay HUD with a surviving ship")
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-enter", action=argparse.BooleanOptionalAction, default=False,
                        help="add Enter as a learned action; never automatically skip an intro")
    parser.add_argument("--repeat-previous-action", action=argparse.BooleanOptionalAction, default=False,
                        help="learn a twenty-first choice that continues the policy's own last key")
    parser.add_argument("--learned-durations", nargs="*", type=int, default=[],
                        help="learn joint physical keys and option holds, e.g. 1 4 16 64")
    parser.add_argument("--grouped-duration", action=argparse.BooleanOptionalAction, default=False,
                        help="fresh twelve-physical-command key-duration PPO with grouped fire aliases")
    parser.add_argument("--grouped-duration-weights", nargs="*", type=float, default=[],
                        help="initial direction-neutral hold-length masses for a fresh grouped-duration actor")
    parser.add_argument("--option-actor-gae", action=argparse.BooleanOptionalAction, default=False,
                        help="credit a completed learned hold's whole score return to its option start")
    parser.add_argument("--duration-explore-mix", type=float, default=0.,
                        help="training-only key-marginal-preserving uniform-duration mixture at option starts")
    parser.add_argument("--spatial-residual", action=argparse.BooleanOptionalAction, default=False,
                        help="learn a zero-initialized spatial mixing residual over an own duration policy")
    parser.add_argument("--extend-longest-duration", action="store_true",
                        help="append one longer hold to an own duration checkpoint with a direction-neutral actor row")
    parser.add_argument("--appended-longest-logit-offset", type=float, default=-2.,
                        help="uniform initial new-long-option logit relative to the copied former longest option")
    parser.add_argument("--duration-initial-logit-spacing", type=float, default=2.,
                        help="direction-neutral initial logit penalty per longer hold; only for fresh duration initialization")
    parser.add_argument("--canonical-fire", action=argparse.BooleanOptionalAction, default=False,
                        help="train a fixed twelve-choice categorical policy combining nine fire-key aliases")
    parser.add_argument("--balanced-canonical-init", action=argparse.BooleanOptionalAction, default=False,
                        help="fresh canonical-fire actor starts with equal mass on twelve physical choices")
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--observation-stride", type=int, default=1,
                        help="policy sees four visible frames spaced this many actions apart")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-max-steps", type=int, default=0)
    parser.add_argument("--mlx-cache-mb", type=int, default=1024)
    parser.add_argument("--sil-updates", type=int, default=0,
                        help="self-imitation updates from own training returns per rollout; 0 disables")
    parser.add_argument("--sil-capacity", type=int, default=32768)
    parser.add_argument("--sil-suffix-steps", type=int, default=2048)
    parser.add_argument("--sil-batch-size", type=int, default=512)
    parser.add_argument("--sil-loss-weight", type=float, default=.1)
    parser.add_argument("--sil-value-weight", type=float, default=.01)
    parser.add_argument("--curriculum-probability", type=float, default=0,
                        help="training-only own-reached-state reset probability; 0 disables")
    parser.add_argument("--curriculum-score-interval", type=int, default=20)
    parser.add_argument("--curriculum-per-bin", type=int, default=4)
    parser.add_argument("--curriculum-bins", type=int, default=16)
    parser.add_argument("--curriculum-lookback", type=int, default=0,
                        help="archive an own-play state this many actions before an archive event; 0 disables")
    parser.add_argument("--curriculum-trigger", choices=("progress", "life-loss"), default="progress",
                        help="archive on progress or rewind from this learner's own visible life losses")
    parser.add_argument("--curriculum-restored-life-only", action=argparse.BooleanOptionalAction, default=False,
                        help="end only restored training segments at their first visible life loss")
    parser.add_argument("--curriculum-cells", choices=("score", "screen", "age"), default="score",
                        help="archive higher score bins, diverse screens or later own-action life ages")
    parser.add_argument("--curriculum-screen-interval", type=int, default=32)
    parser.add_argument("--curriculum-age-interval", type=int, default=32)
    parser.add_argument("--curriculum-share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-boot-envs", type=int, default=0)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            if (hasattr(args, key) and key not in ("run", "resume", "artifacts", "steps", "initialize_encoder", "initialize_policy", "initialize_repeat_policy", "initialize_duration_policy", "initialize_duration_checkpoint", "initialize_only")
                    and "--"+key.replace("_", "-") not in explicit
                    and "--no-"+key.replace("_", "-") not in explicit):
                setattr(args, key, value)
        config = prior["config"]
        if (config.get("algorithm", "ppo") != "ppo"
                or config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
                or config.get("environment_version") != ENVIRONMENT_VERSION
                or config.get("action_names") != list(action_names(config.get("allow_enter", False)))):
            parser.error("Resume requires a compatible Defense checkpoint")
        if args.allow_enter != config.get("allow_enter", False):
            parser.error("Changing action profile requires a fresh run, not an incompatible optimizer resume")
        if args.balanced_canonical_init != config.get("balanced_canonical_init", False):
            parser.error("Changing canonical-fire initializer provenance on resume is invalid")
        if args.repeat_previous_action != config.get("repeat_previous_action", False):
            parser.error("Changing continue-action profile requires fresh initialization")
        if args.learned_durations != config.get("learned_durations", []):
            parser.error("Changing learned-duration profile requires fresh initialization")
        if (args.grouped_duration != config.get("grouped_duration", False)
                or args.grouped_duration_weights != config.get("grouped_duration_weights", [])):
            parser.error("Changing grouped-duration action profile or initializer requires a fresh run")
        if args.duration_explore_mix != config.get("duration_explore_mix", 0.):
            parser.error("Changing the duration behavior mixture on resume is invalid")
        if ("novelty_beta" in config and args.novelty_beta != config["novelty_beta"]):
            parser.error("Changing an established novelty reward on optimizer resume is invalid")
        if ("alive_beta" in config and args.alive_beta != config["alive_beta"]):
            parser.error("Changing an established alive reward on optimizer resume is invalid")
        if args.duration_initial_logit_spacing != config.get("duration_initial_logit_spacing", 2.):
            parser.error("Changing duration initialization spacing requires fresh initialization")
        if args.appended_longest_logit_offset != config.get("appended_longest_logit_offset", -2.):
            parser.error("Changing the appended duration's initialization prior on resume is invalid")
        if args.continue_initial_bias_offset != config.get("continue_initial_bias_offset", 0.):
            parser.error("Continue-action initialization offset cannot change on optimizer resume")
        from .recurrent_policy import RECURRENT_ARCHITECTURE, OWN_ACTION_RECURRENT_ARCHITECTURE
        from .defense_spatial_spec import SPATIAL_ARCHITECTURE
        if (args.recurrent_hidden != config.get('recurrent_hidden', 0)
                or args.recurrent_own_action != config.get('recurrent_own_action', False)
                or config.get('architecture') != ((OWN_ACTION_RECURRENT_ARCHITECTURE
                                                  if args.recurrent_own_action else RECURRENT_ARCHITECTURE)
                                                 if args.recurrent_hidden else
                                                 SPATIAL_ARCHITECTURE if args.spatial_residual else None)):
            parser.error("Changing learned architecture requires initialization, not an optimizer resume")
        if args.freeze_recurrent_base != config.get('freeze_recurrent_base', False):
            parser.error("Changing frozen parameters requires fresh initialization, not an optimizer resume")
        if not (args.resume/"optimizer.npz").exists():
            parser.error("Resume requires an original learner checkpoint with optimizer state")
    if min(args.envs, args.rollout, args.batch_size, args.epochs, args.eval_every,
           args.eval_games, args.eval_envs) < 1:
        parser.error("counts and intervals must be positive")
    if args.envs*args.rollout % args.batch_size:
        parser.error("envs * rollout must be divisible by batch-size")
    if (args.recurrent_hidden < 0 or (args.recurrent_own_action and not args.recurrent_hidden)
            or args.sequence_length < 1
            or not np.isfinite(args.memory_scale) or not 0 <= args.memory_scale <= 1
            or (args.recurrent_hidden and (args.rollout % args.sequence_length
                                          or args.batch_size % args.sequence_length))
            or (not args.recurrent_hidden and (args.initialize_policy or args.memory_scale != 1))
            or (args.recurrent_hidden and (args.initialize_encoder or args.sil_updates
                                          or args.policy_bias_noise or args.policy_weight_noise))):
        parser.error("invalid recurrent configuration, sequence sizes or unsupported SIL/noise/encoder-only mode")
    if args.freeze_recurrent_base and (not args.recurrent_hidden or args.memory_scale == 0
                                      or not (args.initialize_policy or args.resume)):
        parser.error("frozen recurrent base requires own full-policy initialization and enabled memory")
    if args.canonical_fire and (args.allow_enter or args.recurrent_hidden or args.sil_updates
                                or args.policy_bias_noise or args.policy_weight_noise):
        parser.error("canonical fire requires plain twenty-command feedforward PPO without SIL or policy noise")
    if args.balanced_canonical_init and (not args.canonical_fire or
            (not prior and (args.initialize_encoder or args.initialize_policy
                            or args.initialize_repeat_policy or args.initialize_duration_policy
                            or args.initialize_duration_checkpoint))):
        parser.error("balanced canonical initialization requires a fresh grouped-fire PPO actor")
    if args.repeat_previous_action and (args.allow_enter or args.recurrent_hidden or args.canonical_fire
                                        or args.sil_updates or args.initialize_encoder or args.initialize_policy):
        parser.error("continue-action learner requires ordinary feedforward PPO without Enter, SIL or other initialization")
    if args.initialize_repeat_policy and not args.repeat_previous_action:
        parser.error("--initialize-repeat-policy requires --repeat-previous-action")
    if args.learned_durations:
        from .defense_repeat import validate_spec
        try:
            validate_spec(20, args.learned_durations)
        except ValueError as error:
            parser.error(str(error))
        if (args.allow_enter or args.recurrent_hidden or args.canonical_fire
                or args.repeat_previous_action or args.sil_updates or args.initialize_encoder
                or args.initialize_policy or args.initialize_repeat_policy or not args.life_terminal
                or not (args.initialize_duration_policy or args.initialize_duration_checkpoint
                        or args.grouped_duration or args.resume)):
            parser.error("learned durations require own ordinary-policy initialization, life terminals, and plain feedforward PPO")
    elif args.initialize_duration_policy or args.initialize_duration_checkpoint:
        parser.error("duration initialization requires --learned-durations")
    if args.grouped_duration:
        if (not args.learned_durations or args.spatial_residual or args.extend_longest_duration
                or args.initialize_duration_policy or args.initialize_duration_checkpoint
                or args.policy_bias_noise or args.policy_weight_noise or args.policy_key_noise
                or args.policy_duration_noise):
            parser.error("grouped durations require fresh plain score-only option PPO without policy perturbations")
        from .defense_balanced_duration import balanced_duration_initial_bias
        try:
            balanced_duration_initial_bias(args.learned_durations, args.grouped_duration_weights)
        except ValueError as error:
            parser.error(str(error))
    elif args.grouped_duration_weights:
        parser.error("grouped-duration weights require --grouped-duration")
    if args.spatial_residual and (not args.learned_durations or args.recurrent_hidden
                                  or not (args.initialize_duration_checkpoint or args.resume)):
        parser.error("spatial residual requires own trained duration initialization and feedforward PPO")
    if args.extend_longest_duration and (args.spatial_residual
                                         or not (args.initialize_duration_checkpoint or args.resume)
                                         or (args.resume and not prior["config"].get("extend_longest_duration", False))):
        parser.error("extended longest hold requires own duration initialization or its exact resume")
    if (not np.isfinite(args.appended_longest_logit_offset)
            or not -4. <= args.appended_longest_logit_offset <= 4.
            or (not args.extend_longest_duration and args.appended_longest_logit_offset != -2.)):
        parser.error("appended-longest-logit-offset requires extension and must be in -4..4")
    if args.option_actor_gae and not args.learned_durations:
        parser.error("--option-actor-gae requires learned durations")
    if (not np.isfinite(args.duration_explore_mix) or not 0 <= args.duration_explore_mix < 1
            or (args.duration_explore_mix and (len(args.learned_durations) < 2
                                                  or args.spatial_residual))):
        parser.error("duration-explore-mix requires ordinary learned durations and a fraction in [0,1)")
    if (not np.isfinite(args.novelty_beta) or not 0 <= args.novelty_beta <= .5
            or (args.novelty_beta and args.sil_updates)):
        parser.error("novelty-beta must be in [0,0.5] and cannot be combined with SIL")
    if (not np.isfinite(args.alive_beta) or not 0 <= args.alive_beta <= .5
            or (args.alive_beta and (args.sil_updates or args.novelty_beta))):
        parser.error("alive-beta must be in [0,0.5] and cannot be combined with SIL or novelty")
    if (not np.isfinite(args.duration_initial_logit_spacing)
            or not 0 <= args.duration_initial_logit_spacing <= 20
            or (not args.learned_durations and args.duration_initial_logit_spacing != 2.)):
        parser.error("duration initial spacing must be in 0..20 and requires learned durations")
    if (not np.isfinite(args.continue_initial_bias_offset)
            or not 0 <= args.continue_initial_bias_offset <= 10
            or (args.continue_initial_bias_offset and not (args.initialize_repeat_policy or args.resume))):
        parser.error("Continue-action initial bias must be 0..10 and requires own-policy initialization")
    if min(args.steps, args.max_episode_steps, args.eval_max_steps, args.mlx_cache_mb) < 0:
        parser.error("limits must be nonnegative")
    if not 1 <= args.tstates <= 1_000_000:
        parser.error("tstates out of range")
    try:
        validate_observation_stride(args.observation_stride)
    except ValueError as error:
        parser.error(str(error))
    if (not np.isfinite(args.curriculum_probability) or not 0 <= args.curriculum_probability <= 1
            or min(args.curriculum_score_interval, args.curriculum_lookback) < 0
            or min(args.curriculum_per_bin, args.curriculum_bins, args.curriculum_screen_interval,
                   args.curriculum_age_interval) < 1
            or not 0 <= args.curriculum_boot_envs < args.envs
            or (args.curriculum_share and not args.curriculum_probability)
            or (args.curriculum_boot_envs and not args.curriculum_share)):
        parser.error("invalid own-experience curriculum settings")
    if args.curriculum_trigger == "life-loss" and (
            not args.curriculum_probability or not args.curriculum_lookback):
        parser.error("life-loss archive requires positive curriculum probability and lookback")
    if args.curriculum_restored_life_only and (not args.curriculum_probability or not args.life_terminal):
        parser.error("restored-life-only requires positive curriculum probability and life-terminal targets")
    if (args.sil_updates < 0 or min(args.sil_capacity, args.sil_suffix_steps, args.sil_batch_size) < 1
            or not np.isfinite(args.sil_loss_weight) or args.sil_loss_weight <= 0
            or not np.isfinite(args.sil_value_weight) or args.sil_value_weight < 0):
        parser.error("invalid own-experience self-imitation settings")
    if (not np.isfinite(args.policy_bias_noise) or args.policy_bias_noise < 0
            or (args.policy_bias_noise and args.sil_updates)):
        parser.error("policy-bias-noise must be finite/nonnegative and cannot be combined with SIL")
    if (not np.isfinite(args.policy_weight_noise) or args.policy_weight_noise < 0
            or (args.policy_weight_noise and (args.policy_bias_noise or args.sil_updates))):
        parser.error("policy-weight-noise must be finite/nonnegative and cannot be combined with bias noise or SIL")
    if (not np.isfinite(args.policy_key_noise) or args.policy_key_noise < 0
            or (args.policy_key_noise and (args.policy_bias_noise or args.policy_weight_noise
                                           or args.sil_updates or args.allow_enter or args.canonical_fire
                                           or args.recurrent_hidden
                                           or (args.learned_durations and not args.policy_duration_noise)))):
        parser.error("policy-key-noise requires ordinary feedforward Defense actions without other noise or SIL")
    if (not np.isfinite(args.policy_duration_noise) or args.policy_duration_noise < 0
            or (args.policy_duration_noise and (not args.learned_durations or args.policy_bias_noise
                                                or args.policy_weight_noise or args.sil_updates))):
        parser.error("policy-duration-noise requires learned durations without other noise or SIL")
    if args.policy_key_noise_interval < 0 or (args.policy_key_noise_interval and not args.policy_key_noise):
        parser.error("policy-key-noise-interval requires positive key noise and a nonnegative interval")
    if (args.policy_key_noise_min_interval or args.policy_key_noise_max_interval) and (
            not args.policy_key_noise or args.policy_key_noise_interval
            or not 1 <= args.policy_key_noise_min_interval <= args.policy_key_noise_max_interval):
        parser.error("random policy-key-noise intervals require positive noise, ordered positive bounds, "
                     "and no fixed interval")
    if (not all(np.isfinite(v) for v in (args.learning_rate, args.entropy, args.gamma,
                                        args.gae_lambda, args.reward_scale, args.value_coefficient))
            or args.learning_rate <= 0 or args.entropy < 0 or args.reward_scale <= 0
            or args.value_coefficient < 0
            or not 0 < args.gamma < 1 or not 0 < args.gae_lambda <= 1):
        parser.error("invalid optimizer/return parameters")
    if args.run.exists() and not args.resume:
        parser.error("run already exists; choose a new directory or --resume")
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    mx.set_cache_limit(args.mlx_cache_mb*1024*1024)
    agent_class, extra_agent = PPO, dict(canonical_fire=args.canonical_fire)
    if args.learned_durations:
        from .defense_duration_ppo import DurationPPO
        if args.grouped_duration:
            from .defense_grouped_duration_ppo import GroupedDurationPPO
            agent_class = GroupedDurationPPO
            extra_agent.update(duration_explore_mix=args.duration_explore_mix,
                               durations=tuple(args.learned_durations))
        elif args.spatial_residual:
            from .defense_spatial import SpatialDurationPPO
            agent_class = SpatialDurationPPO
        else:
            agent_class = DurationPPO
            extra_agent["duration_explore_mix"] = args.duration_explore_mix
    if args.recurrent_hidden:
        from .defense_recurrent import RecurrentPPO
        from .recurrent_policy import (RECURRENT_ARCHITECTURE,
                                       OWN_ACTION_RECURRENT_ARCHITECTURE, sequence_batches)
        agent_class = RecurrentPPO
        extra_agent = dict(hidden_size=args.recurrent_hidden, memory_scale=args.memory_scale,
                          freeze_base=args.freeze_recurrent_base,
                          own_action_input=args.recurrent_own_action)
    policy_action_count = (len(action_names(False))*len(args.learned_durations)
                           if args.learned_durations else
                           len(action_names(args.allow_enter)) + int(args.repeat_previous_action))
    agent = agent_class(seed=args.seed, learning_rate=args.learning_rate, entropy=args.entropy,
                        action_count=policy_action_count,
                        value_coefficient=args.value_coefficient, **extra_agent)
    rng = np.random.default_rng(args.seed)
    steps, episodes = 0, 0
    initialization = None
    if prior:
        agent.model.load_weights(str(args.resume/"model.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        agent.compile()
        rng.bit_generator.state = prior["rng"]
        steps, episodes = prior["steps"], prior["episodes"]
    else:
        if args.initialize_repeat_policy:
            from .defense_repeat_previous import initialize_repeat_policy
            try:
                initialization = initialize_repeat_policy(agent.model, args.initialize_repeat_policy,
                                                          tstates=args.tstates,
                                                          observation_stride=args.observation_stride,
                                                          bias_offset=args.continue_initial_bias_offset)
            except (OSError, ValueError, KeyError) as error:
                parser.error(str(error))
        elif args.initialize_duration_policy:
            from .defense_duration_ppo import initialize_duration_policy
            try:
                initialization = initialize_duration_policy(
                    agent.model, args.initialize_duration_policy, args.learned_durations,
                    tstates=args.tstates, observation_stride=args.observation_stride,
                    logit_spacing=args.duration_initial_logit_spacing)
            except (OSError, ValueError, KeyError) as error:
                parser.error(str(error))
        elif args.initialize_duration_checkpoint:
            from .defense_spatial import initialize_duration_checkpoint
            try:
                initialization = initialize_duration_checkpoint(
                    agent.model, args.initialize_duration_checkpoint, args.learned_durations,
                    tstates=args.tstates, observation_stride=args.observation_stride,
                    spatial=args.spatial_residual,
                    extend_longest=args.extend_longest_duration,
                    appended_logit_offset=args.appended_longest_logit_offset)
            except (OSError, ValueError, KeyError) as error:
                parser.error(str(error))
        elif args.initialize_policy:
            from .defense_initialization import initialize_policy
            try:
                initialization = initialize_policy(agent.model, args.initialize_policy,
                                                    allow_enter=args.allow_enter, tstates=args.tstates,
                                                    observation_stride=args.observation_stride)
            except (OSError, ValueError, KeyError) as error:
                parser.error(str(error))
        elif args.initialize_encoder:
            from .defense_initialization import initialize_encoder
            try:
                initialization = initialize_encoder(agent.model, args.initialize_encoder,
                                                    allow_enter=args.allow_enter)
            except (OSError, ValueError, KeyError) as error:
                parser.error(str(error))
        # Broad initial exploration; no preference for a hand-selected action.
        if not (args.initialize_policy or args.initialize_repeat_policy or args.initialize_duration_policy
                or args.initialize_duration_checkpoint):
            head = agent.model.base.advantage if args.recurrent_hidden else agent.model.advantage
            head.weight *= .1
            head.bias *= .1
            if args.balanced_canonical_init and not prior:
                from .defense_canonical_fire import balanced_fire_initial_bias
                head.bias = head.bias + mx.array(balanced_fire_initial_bias())
            if args.grouped_duration and not prior:
                from .defense_balanced_duration import balanced_duration_initial_bias
                head.bias = head.bias + mx.array(balanced_duration_initial_bias(
                    args.learned_durations, args.grouped_duration_weights))
        agent.compile()
    sil = None
    if args.sil_updates:
        from .sil import SILReplay, TrainingSuffixes, SelfImitation
        sil_replay = SILReplay(args.sil_capacity)
        sil_collector = TrainingSuffixes(sil_replay, args.envs, args.gamma, args.sil_suffix_steps,
                                         action_count=policy_action_count, score_reader=screen_info)
        sil = SelfImitation(agent, args.sil_loss_weight, args.sil_value_weight)
        sil_rng = np.random.default_rng(np.random.SeedSequence([args.seed, steps, 941]))
        if prior and "sil_rng" in prior:
            sil_rng.bit_generator.state = prior["sil_rng"]
    boot_episodes = prior.get("boot_episodes", episodes) if prior else 0
    restored_segments = prior.get("restored_segments", 0) if prior else 0
    noise = None
    if (args.policy_bias_noise or args.policy_weight_noise or args.policy_key_noise
            or args.policy_duration_noise):
        from .defense_noise import (PolicyBiasNoise, PolicyWeightNoise, PolicyKeyNoise,
                                    PolicyDurationNoise, PolicyKeyDurationNoise, NoiseRollout)
        noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, steps, 1873]))
        if prior and "policy_noise_rng" in prior:
            noise_rng.bit_generator.state = prior["policy_noise_rng"]
        # Emulator episodes restart on resume, so draw new episode perturbations.
        if args.policy_weight_noise:
            noise = PolicyWeightNoise(args.envs, policy_action_count,
                                      agent.model.advantage.weight.shape[1], args.policy_weight_noise, noise_rng)
            noise_argument, noise_kind = "head_weight_noise", "output-weight"
        elif args.policy_key_noise and args.policy_duration_noise:
            noise = PolicyKeyDurationNoise(args.envs, len(action_names(False)),
                                           len(args.learned_durations), args.policy_key_noise,
                                           args.policy_duration_noise, noise_rng,
                                           interval=args.policy_key_noise_interval,
                                           interval_range=(args.policy_key_noise_min_interval,
                                                           args.policy_key_noise_max_interval)
                                           if args.policy_key_noise_min_interval else None)
            noise_argument, noise_kind = "logit_bias", "factorized-key-and-duration-output-bias"
        elif args.policy_key_noise:
            noise = PolicyKeyNoise(args.envs, policy_action_count, args.policy_key_noise, noise_rng,
                                   interval=args.policy_key_noise_interval,
                                   interval_range=(args.policy_key_noise_min_interval,
                                                   args.policy_key_noise_max_interval)
                                   if args.policy_key_noise_min_interval else None)
            noise_argument, noise_kind = "logit_bias", "factorized-key-output-bias"
        elif args.policy_duration_noise:
            noise = PolicyDurationNoise(args.envs, len(action_names(False)),
                                        len(args.learned_durations), args.policy_duration_noise, noise_rng)
            noise_argument, noise_kind = "logit_bias", "factorized-duration-output-bias"
        else:
            noise = PolicyBiasNoise(args.envs, policy_action_count,
                                    args.policy_bias_noise, noise_rng)
            noise_argument, noise_kind = "logit_bias", "output-bias"
    args.run.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(game="defense", algorithm="ppo", game_sha256=GAME_SHA256,
                  native_sha256=sha256("libtrs.so"), environment_version=ENVIRONMENT_VERSION,
                  environment_source_sha256=sha256(Path(__file__).with_name("defense.py")),
                  trainer_source_sha256=sha256(Path(__file__)),
                  ppo_source_sha256=sha256(Path(__file__).with_name("ppo.py")),
                  model_source_sha256=sha256(Path(__file__).with_name("model.py")),
                  action_names=list(action_names(args.allow_enter)), observation="four raw 16x64 video-memory frames",
                  reward="visible score difference only, constant scale for optimizer",
                  policy="learned categorical, sampled", mlx=mx.__version__,
                  resume_semantics="optimizer and policy RNG restored; emulator episodes restart from boot")
    if args.novelty_beta:
        from .defense_cells import CELL_ENCODING
        config.update(reward=("visible score difference times reward scale plus training-only "
                              "first-visit HUD-free screen-cell bonus per visible life"),
                      novelty_encoding=CELL_ENCODING,
                      novelty_source_sha256=sha256(Path(__file__).with_name("defense_novelty.py")),
                      novelty_cells_source_sha256=sha256(Path(__file__).with_name("defense_cells.py")),
                      evaluation_reward="original displayed score only; no novelty bonus or shaping")
    if args.alive_beta:
        config.update(reward=("visible score difference times reward scale plus training-only "
                              "bonus for a visible gameplay HUD and surviving ship"),
                      alive_source_sha256=sha256(Path(__file__).with_name("defense_alive.py")),
                      evaluation_reward="original displayed score only; no alive bonus or shaping")
    if args.repeat_previous_action:
        from .defense_repeat_previous import POLICY_ACTION_NAMES, RepeatPreviousActions, RepeatPreviousPolicy
        config.update(policy_action_names=list(POLICY_ACTION_NAMES),
                      repeat_previous_source_sha256=sha256(Path(__file__).with_name("defense_repeat_previous.py")),
                      policy="learned categorical with own-previous-key continuation, sampled",
                      previous_action_reset="visible life loss or episode boundary; NOOP at boot/resume")
    if args.learned_durations:
        from .defense_duration_ppo import duration_action_names
        if args.grouped_duration:
            from .defense_balanced_duration import grouped_duration_action_names
            option_names = grouped_duration_action_names(args.learned_durations)
        else:
            option_names = duration_action_names(args.learned_durations)
        config.update(policy_action_names=list(option_names),
                      duration_source_sha256=sha256(Path(__file__).with_name("defense_duration_ppo.py")),
                      duration_executor_source_sha256=sha256(Path(__file__).with_name("defense_repeat.py")),
                      policy="learned categorical joint physical key-duration options, sampled",
                      actor_update=("semi-Markov score GAE at complete option starts; every base action trains the critic"
                                    if args.option_actor_gae else
                                    "only actual option starts; every base action trains the score-value critic"),
                      duration_reset="visible life loss or episode boundary; pending hold cancelled")
        if args.grouped_duration:
            config.update(policy="fresh grouped physical-key-duration categorical PPO, sampled",
                          grouped_duration_source_sha256=sha256(Path(__file__).with_name("defense_balanced_duration.py")),
                          grouped_duration_ppo_source_sha256=sha256(Path(__file__).with_name("defense_grouped_duration_ppo.py")),
                          grouped_duration_semantics="twenty raw logits per hold, nine fire aliases grouped to one Space; "
                                                     "twelve distinct original keyboard commands per hold",
                          grouped_duration_initialization="direction-neutral physical key mass for each supplied "
                                                          "hold length; random encoder and actor; no source policy")
        if args.duration_explore_mix:
            config.update(duration_exploration="training option starts: (1-mix)*joint actor + "
                          "mix*actor key marginal/uniform duration; PPO ratios use exact mixture",
                          evaluation_policy="unperturbed learned categorical, sampled")
    if args.recurrent_hidden:
        config.update(architecture=(OWN_ACTION_RECURRENT_ARCHITECTURE if args.recurrent_own_action
                                    else RECURRENT_ARCHITECTURE),
                      recurrent_source_sha256=sha256(Path(__file__).with_name('defense_recurrent.py')),
                      recurrent_policy_source_sha256=sha256(Path(__file__).with_name('recurrent_policy.py')),
                      policy=('learned recurrent categorical, sampled; screen and own-previous-key memory'
                              if args.recurrent_own_action else
                              'learned recurrent categorical, sampled; screen-history memory'),
                      memory_reset='zero at boot/actual environment reset, not visible life loss; cleared on resume',
                      recurrent_training='contiguous within-worker truncated BPTT; rollout initial states detached',
                      resume_semantics='optimizer and policy RNG restored; episodes and neural memory restart from boot')
        if args.freeze_recurrent_base:
            config['frozen_parameter_scope'] = 'base CNN, feature layer, actor and value heads; memory and residual heads trainable'
        if args.recurrent_own_action:
            config['own_action_input'] = ('one-hot previous chosen physical key; reset sentinel only at '
                                          'actual environment reset, never a hidden game-state read')
    if args.spatial_residual:
        from .defense_spatial_spec import SPATIAL_ARCHITECTURE
        config.update(architecture=SPATIAL_ARCHITECTURE,
                      spatial_source_sha256=sha256(Path(__file__).with_name("defense_spatial.py")),
                      spatial_spec_source_sha256=sha256(Path(__file__).with_name("defense_spatial_spec.py")),
                      policy="learned spatial-residual categorical key-duration options, sampled",
                      spatial_initialization="zero gate preserves transferred own-policy outputs")
    if initialization is not None:
        config.update(initialization=initialization,
                      initialization_source_sha256=sha256(Path(__file__).with_name(
                          "defense_repeat_previous.py" if args.initialize_repeat_policy else
                          "defense_duration_ppo.py" if args.initialize_duration_policy else
                          "defense_spatial.py" if args.initialize_duration_checkpoint else
                          "defense_initialization.py")))
    elif prior and "initialization" in prior["config"]:
        # Keep ancestry without reapplying initialization or requiring its source.
        config["initialization"] = prior["config"]["initialization"]
        config["initialization_source_sha256"] = prior["config"]["initialization_source_sha256"]
    if noise is not None:
        if args.policy_key_noise_min_interval:
            reset_rule = ("visible life loss, episode boundary, or independent uniformly sampled "
                          f"{args.policy_key_noise_min_interval}..{args.policy_key_noise_max_interval} "
                          "own-action periods per worker; fresh draw after resume")
        elif args.policy_key_noise_interval:
            reset_rule = (f"visible life loss, episode boundary, or each {args.policy_key_noise_interval} "
                          "own actions per worker; fresh draw after resume")
        else:
            reset_rule = "visible life loss or episode boundary; fresh draw after resume"
        period = "within-life" if (args.policy_key_noise_interval or
                                   args.policy_key_noise_min_interval) else "per-life"
        config.update(training_policy=f"learned categorical with {period} Gaussian {noise_kind} perturbations",
                      evaluation_policy="unperturbed learned categorical, sampled",
                      policy_noise_source_sha256=sha256(Path(__file__).with_name("defense_noise.py")),
                      ppo_source_sha256=sha256(Path(__file__).with_name("ppo.py")),
                      model_source_sha256=sha256(Path(__file__).with_name("model.py")),
                      policy_noise_reset=reset_rule)
    if args.canonical_fire:
        config.update(policy="learned categorical over fixed twelve command groups, sampled",
                      evaluation_policy="same fixed grouped categorical policy, sampled",
                      canonical_fire_source_sha256=sha256(Path(__file__).with_name("defense_canonical_fire.py")),
                      canonical_fire_semantics="commands 9..17 grouped by log-sum-exp and emitted as Space; "
                                               "commands 0..8 and 18..19 unchanged on every screen")
        if args.balanced_canonical_init:
            config["canonical_fire_initialization"] = (
                "fresh actor logits offset by -log(9) for each of nine forward-fire aliases; "
                "twelve distinct physical commands have approximately equal initial mass")
    if args.curriculum_probability:
        config.update(curriculum_archive_saved=False,
                      curriculum_source_sha256=sha256(Path(__file__).with_name("defense_curriculum.py")),
                      snapshot_source_sha256=sha256(Path(__file__).with_name("defense_snapshot.py")))
        if args.curriculum_cells == "screen":
            from .defense_cells import CELL_ENCODING
            config.update(curriculum_cell_encoding=CELL_ENCODING,
                          curriculum_cells_source_sha256=sha256(Path(__file__).with_name("defense_cells.py")))
        if args.curriculum_cells == "age":
            config["curriculum_cell_encoding"] = "own-actions-since-visible-life-or-stage-boundary-v1"
    write_json(args.run/("resume-config.json" if prior else "config.json"), config)
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)
    started, start_steps = time.monotonic(), steps
    stop = False
    workers = None
    recent = deque(maxlen=100)
    recent_restored = deque(maxlen=100)

    def log(event):
        row = dict(wall_seconds=round(time.monotonic()-started, 2), **event)
        log_file.write(json.dumps(row)+"\n")
        write_json(args.run/"status.json", {**row, "pid": os.getpid(), "training_steps": steps,
                                           "episodes": episodes, "updated_unix": time.time()})
        if row["event"] != "episode":
            print(json.dumps(row), flush=True)

    def state():
        saved = dict(steps=steps, episodes=episodes, boot_episodes=boot_episodes,
                     restored_segments=restored_segments, rng=rng.bit_generator.state, config=config)
        if sil is not None:
            saved.update(sil_rng=sil_rng.bit_generator.state, sil_replay_saved=False)
        if noise is not None:
            saved["policy_noise_rng"] = noise_rng.bit_generator.state
        return saved

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    next_eval = steps + args.eval_every
    last_log = time.monotonic()
    log(dict(event="start", steps=steps, config=config))
    try:
        curriculum = {}
        if args.curriculum_probability:
            curriculum = dict(curriculum=True, curriculum_probability=args.curriculum_probability,
                              curriculum_score_interval=args.curriculum_score_interval,
                              curriculum_per_bin=args.curriculum_per_bin, curriculum_bins=args.curriculum_bins,
                              curriculum_lookback=args.curriculum_lookback,
                              curriculum_trigger=args.curriculum_trigger,
                              curriculum_restored_life_only=args.curriculum_restored_life_only,
                              curriculum_cells=args.curriculum_cells,
                              curriculum_screen_interval=args.curriculum_screen_interval,
                              curriculum_age_interval=args.curriculum_age_interval,
                              curriculum_share=args.curriculum_share,
                              curriculum_boot_envs=args.curriculum_boot_envs)
        workers = VectorEnv(args.envs, args.seed+steps, game="defense", tstates=args.tstates,
                            max_steps=args.max_episode_steps, allow_enter=args.allow_enter,
                            observation_stride=args.observation_stride, **curriculum)
        obs = workers.observations
        if args.novelty_beta:
            from .defense_novelty import LifeScreenNovelty
            novelty = LifeScreenNovelty(obs, args.novelty_beta)
        else:
            novelty = None
        if args.alive_beta:
            from .defense_alive import visible_alive_bonus
        alive_bonus_sum = 0.
        alive_bonus_hits = 0
        repeat_actions = RepeatPreviousActions(args.envs) if args.repeat_previous_action else None
        if args.learned_durations:
            from .defense_repeat import RepeatedActions
            duration_actions = RepeatedActions(args.envs, action_count=12 if args.grouped_duration else 20,
                                               durations=args.learned_durations)
        else:
            duration_actions = None
        if args.recurrent_hidden:
            agent.reset_memory(args.envs)
            episode_starts = np.ones(args.envs, dtype=bool)
        log(dict(event="workers_started", workers=workers.runtime()))
        agent.save(args.run/"latest", state())
        while not stop and not args.initialize_only and (not args.steps or steps < args.steps):
            screens, actions_buffer, logps, values_buffer, rewards, boundaries = [], [], [], [], [], []
            actor_masks = []
            hidden_buffer, starts_buffer, previous_action_buffer = [], [], []
            noise_rollout = None if noise is None else NoiseRollout(noise)
            for _ in range(args.rollout):
                if args.recurrent_hidden:
                    hidden_buffer.append(agent.memory.copy())
                    starts_buffer.append(episode_starts.copy())
                    if args.recurrent_own_action:
                        previous_action_buffer.append(agent.previous_actions.copy())
                    episode_starts[:] = False
                exploration = {} if noise is None else {noise_argument: noise_rollout.record()}
                decision_mask = ((duration_actions.remaining == 0)
                                 if duration_actions is not None else None)
                actions, logp, values = agent.act(
                    obs, rng, **({"actor_mask": decision_mask} if decision_mask is not None else {}),
                    **exploration)
                screens.append(obs)
                actions_buffer.append(actions)
                logps.append(logp)
                values_buffer.append(values)
                if decision_mask is not None:
                    actor_masks.append(decision_mask.copy())
                next_obs, reward_row, boundary_row = [], [], []
                noise_boundaries = np.zeros(args.envs, dtype=bool) if noise is not None else None
                physical_actions = (duration_actions.select(actions) if duration_actions is not None else
                                    repeat_actions.execute(actions) if repeat_actions is not None else actions)
                if args.grouped_duration:
                    from .defense_balanced_duration import grouped_duration_physical_actions
                    physical_actions = grouped_duration_physical_actions(physical_actions)
                repeat_boundaries = np.zeros(args.envs, dtype=bool) if repeat_actions is not None else None
                duration_boundaries = np.zeros(args.envs, dtype=bool) if duration_actions is not None else None
                for worker, result in enumerate(workers.step(physical_actions)):
                    frame, reward, terminal, truncated, info, reset = result
                    reward *= args.reward_scale
                    if novelty is not None:
                        reward += novelty.step(worker, frame, life_lost=info["life_lost"], reset=reset)
                    if args.alive_beta:
                        extra_reward = visible_alive_bonus(
                            frame, args.alive_beta, life_lost=info["life_lost"],
                            terminal=terminal, truncated=truncated)
                        reward += extra_reward
                        alive_bonus_sum += extra_reward
                        alive_bonus_hits += bool(extra_reward)
                    learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                    if noise is not None:
                        noise_boundaries[worker] = terminal or truncated or info["life_lost"]
                    if repeat_actions is not None:
                        repeat_boundaries[worker] = terminal or truncated or info["life_lost"]
                    if duration_actions is not None:
                        duration_boundaries[worker] = terminal or truncated or info["life_lost"]
                    if sil is not None:
                        # Only this learner's own screens, selected actions and
                        # score returns; never evaluation/replay-file examples.
                        segment = sil_collector.append(worker, obs[worker], actions[worker], reward,
                                                       learning_terminal, truncated, info.get("full_game", True))
                        if segment is not None:
                            log(dict(event="sil_segment", action_counter=steps+worker+1, **segment))
                    if truncated and not learning_terminal:
                        value = (agent.bootstrap_value(frame[None], [worker]) if args.recurrent_hidden
                                 else agent.predict(mx.array(frame[None]))[1])
                        reward += args.gamma*float(value[0].item())
                    reward_row.append(reward)
                    boundary_row.append(learning_terminal or truncated)
                    next_obs.append(reset if reset is not None else frame)
                    if "curriculum_archive_add" in info:
                        log(dict(event="curriculum_archive", worker=worker, action_counter=steps+worker+1,
                                 **info["curriculum_archive_add"]))
                    if terminal or truncated:
                        if args.recurrent_hidden:
                            episode_starts[worker] = True
                        episodes += 1
                        if info.get("full_game", True):
                            boot_episodes += 1
                            recent.append(info)
                        else:
                            restored_segments += 1
                            recent_restored.append(info["episode_reward"])
                        log(dict(event="episode", worker=worker, action_counter=steps+worker+1,
                                 episode=episodes, **info))
                rewards.append(reward_row)
                boundaries.append(boundary_row)
                obs = np.stack(next_obs)
                if args.recurrent_hidden:
                    agent.reset_done(episode_starts)
                if noise is not None:
                    noise_rollout.redraw(noise_boundaries)
                if repeat_actions is not None:
                    repeat_actions.reset(repeat_boundaries)
                if duration_actions is not None:
                    duration_actions.reset(duration_boundaries)
                steps += args.envs
            last_value = (agent.bootstrap_value(obs) if args.recurrent_hidden
                          else agent.predict(mx.array(obs))[1])
            advantages, returns = gae(np.asarray(rewards, np.float32), np.asarray(values_buffer),
                                      np.asarray(boundaries, np.float32), np.array(last_value),
                                      args.gamma, args.gae_lambda)
            option_credit_stats = {}
            if duration_actions is not None:
                if args.option_actor_gae:
                    from .defense_duration_ppo import option_actor_gae
                    advantages, completed_mask = option_actor_gae(
                        rewards, values_buffer, boundaries, actor_masks, np.array(last_value),
                        duration_actions.remaining > 0, args.gamma, args.gae_lambda)
                    flat_mask = completed_mask.reshape(-1).astype(np.float32)
                    option_credit_stats = dict(
                        option_actor_completed=int(completed_mask.sum()),
                        option_actor_dropped_incomplete=int(np.asarray(actor_masks).sum()
                                                            -completed_mask.sum()))
                else:
                    flat_mask = np.concatenate(actor_masks).astype(np.float32)
                selected = advantages.reshape(-1)[flat_mask > 0]
                if not len(selected):
                    raise RuntimeError("duration rollout contained no policy decisions")
                advantages = ((advantages-selected.mean())/(selected.std()+1e-8)
                              *flat_mask.reshape(advantages.shape))
            else:
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            if args.recurrent_hidden:
                data = tuple(sequence_batches(x, args.sequence_length) for x in
                             (screens, actions_buffer, logps, advantages, returns))
                data += (sequence_batches(hidden_buffer, args.sequence_length)[:, 0],
                         sequence_batches(starts_buffer, args.sequence_length))
                if args.recurrent_own_action:
                    data += (sequence_batches(previous_action_buffer, args.sequence_length),)
            else:
                data = (np.concatenate(screens), np.concatenate(actions_buffer), np.concatenate(logps),
                        advantages.reshape(-1), returns.reshape(-1))
                if duration_actions is not None:
                    data += (flat_mask,)
            if noise is not None:
                noise_bank, noise_ids = noise_rollout.arrays()
            metrics = []
            for _ in range(args.epochs):
                order = rng.permutation(len(data[0]))
                epoch_metrics = []
                chunk_batch = args.batch_size // args.sequence_length if args.recurrent_hidden else args.batch_size
                for start in range(0, len(order), chunk_batch):
                    indices = order[start:start+chunk_batch]
                    extra = {} if noise is None else {noise_argument: mx.array(noise_bank[noise_ids[indices]])}
                    loss, aux = agent.update(*(mx.array(x[indices]) for x in data), **extra)
                    mx.eval(loss, aux, agent.state)
                    if not np.isfinite(float(loss.item())):
                        raise RuntimeError("Non-finite Defense PPO loss")
                    epoch_metrics.append([float(x.item()) for x in aux])
                metrics.extend(epoch_metrics)
                if np.mean(epoch_metrics, axis=0)[3] > .03:
                    break
            sil_metrics = {}
            if sil is not None:
                details = []
                if sil_replay.size:
                    for _ in range(args.sil_updates):
                        indices, batch = sil_replay.sample(args.sil_batch_size, sil_rng)
                        advantages, update = sil.train(batch)
                        sil_replay.priorities(indices, advantages)
                        details.append(update)
                sil_metrics = {"sil": {**sil_collector.metrics(), "updates": sil.updates,
                                       "rollout_updates": details}}
            if time.monotonic()-last_log >= 10:
                recent_summary = summarize(list(recent))
                noise_fields = {}
                if noise is not None:
                    noise_fields["policy_noise_draws"] = noise.draws
                    if isinstance(noise, PolicyKeyDurationNoise):
                        noise_fields["policy_noise_key_draws"] = noise.key.draws
                        noise_fields["policy_noise_duration_draws"] = noise.duration.draws
                    for kind, strength in (("policy_weight_noise_std", args.policy_weight_noise),
                                           ("policy_key_noise_std", args.policy_key_noise),
                                           ("policy_duration_noise_std", args.policy_duration_noise),
                                           ("policy_bias_noise_std", args.policy_bias_noise)):
                        if strength:
                            noise_fields[kind] = strength
                log(dict(event="progress", steps=steps, episodes=episodes,
                         boot_episodes=boot_episodes, restored_segments=restored_segments,
                         recent_restored=dict(segments=len(recent_restored),
                             mean_new_score=float(np.mean(recent_restored)) if recent_restored else None),
                         steps_per_second=(steps-start_steps)/(time.monotonic()-started),
                         recent={k: v for k, v in recent_summary.items() if k != "games"},
                         **({"novelty": novelty.metrics()} if novelty is not None else {}),
                         **({"visible_alive_bonus": dict(beta=args.alive_beta,
                                                          hits=alive_bonus_hits,
                                                          bonus_sum=round(alive_bonus_sum, 6))}
                            if args.alive_beta else {}),
                         actor_loss=float(np.mean(metrics, axis=0)[0]),
                         value_loss=float(np.mean(metrics, axis=0)[1]),
                         entropy=float(np.mean(metrics, axis=0)[2]),
                         approx_kl=float(np.mean(metrics, axis=0)[3]),
                         **({"duration_options": duration_actions.stats()}
                            if duration_actions is not None else {}),
                         **option_credit_stats,
                         mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
                         **noise_fields,
                         **sil_metrics))
                last_log = time.monotonic()
            if steps >= next_eval and not stop:
                directory = args.run/f"step-{steps:012d}"
                if directory.exists():
                    raise RuntimeError("Refusing to overwrite a historical checkpoint")
                agent.save(directory, state())
                agent.save(args.run/"latest", state())
                log(dict(event="validation_start", steps=steps, checkpoint=str(directory)))
                evaluation_policy = agent.policy()
                if args.repeat_previous_action:
                    evaluation_policy = RepeatPreviousPolicy(evaluation_policy)
                if args.learned_durations and not args.grouped_duration:
                    from .defense_duration_ppo import DurationCategoricalPolicy
                    evaluation_policy = DurationCategoricalPolicy(evaluation_policy, args.learned_durations)
                result = evaluate(evaluation_policy, range(args.eval_seed, args.eval_seed+args.eval_games),
                                  tstates=args.tstates, max_steps=args.eval_max_steps, envs=args.eval_envs,
                                  log=log, should_stop=lambda: stop, allow_enter=args.allow_enter,
                                  observation_stride=args.observation_stride)
                write_json(directory/"evaluation.json", result)
                log(dict(event="validation", steps=steps,
                         **{k: v for k, v in result.items() if k != "games"}))
                publish_best(directory/"model.safetensors", result, args.artifacts,
                             should_stop=lambda: stop, log=log)
                next_eval += args.eval_every
    except InterruptedError as error:
        log(dict(event="cancelled", reason=str(error), steps=steps))
    except BaseException as error:
        log(dict(event="error", error=repr(error), steps=steps))
        raise
    finally:
        agent.save(args.run/"latest", state())
        if workers is not None:
            workers.close()
        log(dict(event="stopped", steps=steps, stop_requested=stop,
                 **({"duration_options": duration_actions.stats()}
                    if duration_actions is not None else {})))
        log_file.close()


if __name__ == "__main__":
    main()
