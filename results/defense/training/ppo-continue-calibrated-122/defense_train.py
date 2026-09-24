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
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--memory-scale", type=float, default=1.,
                        help="recurrent residual scale; 0 is a matched memory-disabled control")
    parser.add_argument("--freeze-recurrent-base", action=argparse.BooleanOptionalAction, default=False,
                        help="train only recurrent residuals over an own initialized, frozen PPO base")
    parser.add_argument("--policy-bias-noise", type=float, default=0,
                        help="training-only actor bias noise std, fixed per life; 0 disables")
    parser.add_argument("--policy-weight-noise", type=float, default=0,
                        help="training-only actor output-weight noise std, fixed per life; 0 disables")
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--reward-scale", type=float, default=.01,
                        help="constant units conversion; no shaping or clipping")
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-enter", action=argparse.BooleanOptionalAction, default=False,
                        help="add Enter as a learned action; never automatically skip an intro")
    parser.add_argument("--repeat-previous-action", action=argparse.BooleanOptionalAction, default=False,
                        help="learn a twenty-first choice that continues the policy's own last key")
    parser.add_argument("--canonical-fire", action=argparse.BooleanOptionalAction, default=False,
                        help="train a fixed twelve-choice categorical policy combining nine fire-key aliases")
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
            if (hasattr(args, key) and key not in ("run", "resume", "artifacts", "steps", "initialize_encoder", "initialize_policy", "initialize_repeat_policy", "initialize_only")
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
        if args.repeat_previous_action != config.get("repeat_previous_action", False):
            parser.error("Changing continue-action profile requires fresh initialization")
        if args.continue_initial_bias_offset != config.get("continue_initial_bias_offset", 0.):
            parser.error("Continue-action initialization offset cannot change on optimizer resume")
        from .recurrent_policy import RECURRENT_ARCHITECTURE
        if (args.recurrent_hidden != config.get('recurrent_hidden', 0)
                or config.get('architecture') != (RECURRENT_ARCHITECTURE if args.recurrent_hidden else None)):
            parser.error("Changing recurrent architecture requires initialization, not an optimizer resume")
        if args.freeze_recurrent_base != config.get('freeze_recurrent_base', False):
            parser.error("Changing frozen parameters requires fresh initialization, not an optimizer resume")
        if not (args.resume/"optimizer.npz").exists():
            parser.error("Resume requires an original learner checkpoint with optimizer state")
    if min(args.envs, args.rollout, args.batch_size, args.epochs, args.eval_every,
           args.eval_games, args.eval_envs) < 1:
        parser.error("counts and intervals must be positive")
    if args.envs*args.rollout % args.batch_size:
        parser.error("envs * rollout must be divisible by batch-size")
    if (args.recurrent_hidden < 0 or args.sequence_length < 1
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
    if args.repeat_previous_action and (args.allow_enter or args.recurrent_hidden or args.canonical_fire
                                        or args.sil_updates or args.initialize_encoder or args.initialize_policy):
        parser.error("continue-action learner requires ordinary feedforward PPO without Enter, SIL or other initialization")
    if args.initialize_repeat_policy and not args.repeat_previous_action:
        parser.error("--initialize-repeat-policy requires --repeat-previous-action")
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
    if args.recurrent_hidden:
        from .defense_recurrent import RecurrentPPO
        from .recurrent_policy import RECURRENT_ARCHITECTURE, sequence_batches
        agent_class = RecurrentPPO
        extra_agent = dict(hidden_size=args.recurrent_hidden, memory_scale=args.memory_scale,
                          freeze_base=args.freeze_recurrent_base)
    policy_action_count = len(action_names(args.allow_enter)) + int(args.repeat_previous_action)
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
        if not (args.initialize_policy or args.initialize_repeat_policy):
            head = agent.model.base.advantage if args.recurrent_hidden else agent.model.advantage
            head.weight *= .1
            head.bias *= .1
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
    if args.policy_bias_noise or args.policy_weight_noise:
        from .defense_noise import PolicyBiasNoise, PolicyWeightNoise, NoiseRollout
        noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, steps, 1873]))
        if prior and "policy_noise_rng" in prior:
            noise_rng.bit_generator.state = prior["policy_noise_rng"]
        # Emulator episodes restart on resume, so draw new episode perturbations.
        if args.policy_weight_noise:
            noise = PolicyWeightNoise(args.envs, policy_action_count,
                                      agent.model.advantage.weight.shape[1], args.policy_weight_noise, noise_rng)
            noise_argument, noise_kind = "head_weight_noise", "output-weight"
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
    if args.repeat_previous_action:
        from .defense_repeat_previous import POLICY_ACTION_NAMES, RepeatPreviousActions, RepeatPreviousPolicy
        config.update(policy_action_names=list(POLICY_ACTION_NAMES),
                      repeat_previous_source_sha256=sha256(Path(__file__).with_name("defense_repeat_previous.py")),
                      policy="learned categorical with own-previous-key continuation, sampled",
                      previous_action_reset="visible life loss or episode boundary; NOOP at boot/resume")
    if args.recurrent_hidden:
        config.update(architecture=RECURRENT_ARCHITECTURE,
                      recurrent_source_sha256=sha256(Path(__file__).with_name('defense_recurrent.py')),
                      recurrent_policy_source_sha256=sha256(Path(__file__).with_name('recurrent_policy.py')),
                      policy='learned recurrent categorical, sampled; screen-history memory',
                      memory_reset='zero at boot/actual environment reset, not visible life loss; cleared on resume',
                      recurrent_training='contiguous within-worker truncated BPTT; rollout initial states detached',
                      resume_semantics='optimizer and policy RNG restored; episodes and neural memory restart from boot')
        if args.freeze_recurrent_base:
            config['frozen_parameter_scope'] = 'base CNN, feature layer, actor and value heads; memory and residual heads trainable'
    if initialization is not None:
        config.update(initialization=initialization,
                      initialization_source_sha256=sha256(Path(__file__).with_name(
                          "defense_repeat_previous.py" if args.initialize_repeat_policy else "defense_initialization.py")))
    elif prior and "initialization" in prior["config"]:
        # Keep ancestry without reapplying initialization or requiring its source.
        config["initialization"] = prior["config"]["initialization"]
        config["initialization_source_sha256"] = prior["config"]["initialization_source_sha256"]
    if noise is not None:
        config.update(training_policy=f"learned categorical with per-life Gaussian {noise_kind} perturbations",
                      evaluation_policy="unperturbed learned categorical, sampled",
                      policy_noise_source_sha256=sha256(Path(__file__).with_name("defense_noise.py")),
                      ppo_source_sha256=sha256(Path(__file__).with_name("ppo.py")),
                      model_source_sha256=sha256(Path(__file__).with_name("model.py")),
                      policy_noise_reset="visible life loss or episode boundary; fresh draw after resume")
    if args.canonical_fire:
        config.update(policy="learned categorical over fixed twelve command groups, sampled",
                      evaluation_policy="same fixed grouped categorical policy, sampled",
                      canonical_fire_source_sha256=sha256(Path(__file__).with_name("defense_canonical_fire.py")),
                      canonical_fire_semantics="commands 9..17 grouped by log-sum-exp and emitted as Space; "
                                               "commands 0..8 and 18..19 unchanged on every screen")
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
        repeat_actions = RepeatPreviousActions(args.envs) if args.repeat_previous_action else None
        if args.recurrent_hidden:
            agent.reset_memory(args.envs)
            episode_starts = np.ones(args.envs, dtype=bool)
        log(dict(event="workers_started", workers=workers.runtime()))
        agent.save(args.run/"latest", state())
        while not stop and not args.initialize_only and (not args.steps or steps < args.steps):
            screens, actions_buffer, logps, values_buffer, rewards, boundaries = [], [], [], [], [], []
            hidden_buffer, starts_buffer = [], []
            noise_rollout = None if noise is None else NoiseRollout(noise)
            for _ in range(args.rollout):
                if args.recurrent_hidden:
                    hidden_buffer.append(agent.memory.copy())
                    starts_buffer.append(episode_starts.copy())
                    episode_starts[:] = False
                exploration = {} if noise is None else {noise_argument: noise_rollout.record()}
                actions, logp, values = agent.act(obs, rng, **exploration)
                screens.append(obs)
                actions_buffer.append(actions)
                logps.append(logp)
                values_buffer.append(values)
                next_obs, reward_row, boundary_row = [], [], []
                noise_boundaries = np.zeros(args.envs, dtype=bool) if noise is not None else None
                physical_actions = repeat_actions.execute(actions) if repeat_actions is not None else actions
                repeat_boundaries = np.zeros(args.envs, dtype=bool) if repeat_actions is not None else None
                for worker, result in enumerate(workers.step(physical_actions)):
                    frame, reward, terminal, truncated, info, reset = result
                    reward *= args.reward_scale
                    learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                    if noise is not None:
                        noise_boundaries[worker] = terminal or truncated or info["life_lost"]
                    if repeat_actions is not None:
                        repeat_boundaries[worker] = terminal or truncated or info["life_lost"]
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
                steps += args.envs
            last_value = (agent.bootstrap_value(obs) if args.recurrent_hidden
                          else agent.predict(mx.array(obs))[1])
            advantages, returns = gae(np.asarray(rewards, np.float32), np.asarray(values_buffer),
                                      np.asarray(boundaries, np.float32), np.array(last_value),
                                      args.gamma, args.gae_lambda)
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            if args.recurrent_hidden:
                data = tuple(sequence_batches(x, args.sequence_length) for x in
                             (screens, actions_buffer, logps, advantages, returns))
                data += (sequence_batches(hidden_buffer, args.sequence_length)[:, 0],
                         sequence_batches(starts_buffer, args.sequence_length))
            else:
                data = (np.concatenate(screens), np.concatenate(actions_buffer), np.concatenate(logps),
                        advantages.reshape(-1), returns.reshape(-1))
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
                log(dict(event="progress", steps=steps, episodes=episodes,
                         boot_episodes=boot_episodes, restored_segments=restored_segments,
                         recent_restored=dict(segments=len(recent_restored),
                             mean_new_score=float(np.mean(recent_restored)) if recent_restored else None),
                         steps_per_second=(steps-start_steps)/(time.monotonic()-started),
                         recent={k: v for k, v in recent_summary.items() if k != "games"},
                         actor_loss=float(np.mean(metrics, axis=0)[0]),
                         value_loss=float(np.mean(metrics, axis=0)[1]),
                         entropy=float(np.mean(metrics, axis=0)[2]),
                         approx_kl=float(np.mean(metrics, axis=0)[3]),
                         mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
                         **({"policy_weight_noise_std" if args.policy_weight_noise else "policy_bias_noise_std":
                             noise.std, "policy_noise_draws": noise.draws}
                            if noise is not None else {}),
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
        log(dict(event="stopped", steps=steps, stop_requested=stop))
        log_file.close()


if __name__ == "__main__":
    main()
