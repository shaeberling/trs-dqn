"""Independent screen-only Defense Double DQN with prioritized own-experience replay.

No demonstrations, PPO trajectories, reward shaping or action controllers.
Optional training resets use opaque states reached by this learner itself.
Evaluation is greedy and always starts at the original game boot.
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

from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256, validate_observation_stride
from .defense_learning import (BOOTSTRAP_ALGORITHM, DQN_ALGORITHM, QUANTILE_ALGORITHM, REPEAT_ALGORITHM, evaluate, greedy_policy,
                               policy_description, publish_best, sha256, summarize, write_json)
from .replay import NStep, Replay
from .vector import VectorEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from-dqn", type=Path,
                        help="fresh quantile/action-duration learner initialized from own scalar DQN online weights")
    parser.add_argument("--learned-repeats", type=str,
                        help="optional joint learned action durations, e.g. 1,4,16,64; requires n-step 1")
    parser.add_argument("--steps", type=int, default=0, help="absolute action limit; 0 = unlimited")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--compact-replay", action=argparse.BooleanOptionalAction, default=False,
                        help="losslessly share identical visible frames in training replay")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--greedy-trace-cut", action=argparse.BooleanOptionalAction, default=False,
                        help="recompute current-policy cuts in own multi-step trajectories (compact scalar DQN)")
    parser.add_argument("--spr-weight", type=float, default=0.,
                        help="optional own-visual-future auxiliary loss; 0 preserves ordinary DQN")
    parser.add_argument("--inverse-weight", type=float, default=0.,
                        help="optional own-adjacent-screen action classification; no intrinsic reward")
    parser.add_argument("--reward-scale", type=float, default=.01)
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-enter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=16,
                        help="new aggregate actions per optimizer update")
    parser.add_argument("--epsilon-steps", type=int, default=1_000_000)
    parser.add_argument("--epsilon-final", type=float, default=.05)
    parser.add_argument("--exploration-max-repeat", type=int, default=1,
                        help="1 preserves ordinary epsilon-greedy; >1 enables training-only action persistence")
    parser.add_argument("--exploration-exponent", type=float, default=1.5,
                        help="bounded power-law duration exponent; must exceed one")
    parser.add_argument("--exploration-actions", choices=("uniform", "stage1-balanced"), default="uniform",
                        help="fixed training-only random-action distribution; greedy evaluation stays unchanged")
    parser.add_argument("--bootstrap-heads", type=int, default=0,
                        help="0 = ordinary DQN; at least 2 = per-game value-head exploration")
    parser.add_argument("--bootstrap-probability", type=float, default=.5)
    parser.add_argument("--bootstrap-prior-scale", type=float, default=1.)
    parser.add_argument("--bootstrap-epsilon", type=float, default=.01,
                        help="constant random-action probability after bootstrap warmup")
    parser.add_argument("--quantiles", type=int, default=0,
                        help="0 preserves scalar DQN; 2..256 enables fixed-quantile regression")
    parser.add_argument("--quantile-exploration-power", type=float, default=0.,
                        help="training-only upper-return power distortion, 0..4; 0 uses the mean")
    parser.add_argument("--target-every", type=int, default=2000,
                        help="optimizer updates per target-network copy")
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--observation-stride", type=int, default=1,
                        help="policy sees four visible frames spaced this many actions apart")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-max-steps", type=int, default=0)
    parser.add_argument("--mlx-cache-mb", type=int, default=512)
    parser.add_argument("--curriculum-probability", type=float, default=0,
                        help="training-only resets to this learner's own states; 0 disables")
    parser.add_argument("--curriculum-share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-boot-envs", type=int, default=0)
    parser.add_argument("--curriculum-boot-epsilon", type=float,
                        help="optional fixed post-warmup epsilon for reserved boot-only workers")
    parser.add_argument("--curriculum-lookback", type=int, default=0)
    parser.add_argument("--curriculum-trigger", choices=("progress", "life-loss"), default="progress",
                        help="archive on progress (default) or rewind from own visible life losses")
    parser.add_argument("--curriculum-restored-life-only", action=argparse.BooleanOptionalAction, default=False,
                        help="end only restored training segments at their first visible life loss")
    parser.add_argument("--curriculum-cells", choices=("score", "screen"), default="score",
                        help="training archive selection only; policy observations stay unchanged")
    parser.add_argument("--curriculum-score-interval", type=int, default=20)
    parser.add_argument("--curriculum-bins", type=int, default=16)
    parser.add_argument("--curriculum-per-bin", type=int, default=4)
    parser.add_argument("--curriculum-screen-interval", type=int, default=32)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            if (hasattr(args, key) and key not in ("run", "resume", "artifacts", "steps", "init_from_dqn")
                    and "--"+key.replace("_", "-") not in explicit
                    and "--no-"+key.replace("_", "-") not in explicit):
                setattr(args, key, value)
        config = prior["config"]
        expected_algorithm = (REPEAT_ALGORITHM if args.learned_repeats else BOOTSTRAP_ALGORITHM if args.bootstrap_heads else
                              QUANTILE_ALGORITHM if args.quantiles else DQN_ALGORITHM)
        if (config.get("algorithm") != expected_algorithm or config.get("game") != "defense"
                or config.get("game_sha256") != GAME_SHA256
                or config.get("environment_version") != ENVIRONMENT_VERSION
                or config.get("action_names") != list(action_names(args.allow_enter))
                or config.get("bootstrap_heads", 0) != args.bootstrap_heads
                or config.get("quantiles", 0) != args.quantiles):
            parser.error("Resume requires a compatible Defense DQN checkpoint")
        if not all((args.resume/name).is_file() for name in
                   ("model.safetensors", "target.safetensors", "optimizer.npz")):
            parser.error("DQN resume requires online, target and optimizer checkpoints")
    if args.learned_repeats is not None:
        from .defense_repeat import validate_spec
        try:
            durations = ([int(v) for v in args.learned_repeats.split(',')]
                         if isinstance(args.learned_repeats, str) else args.learned_repeats)
            _, args.learned_repeats = validate_spec(len(action_names(args.allow_enter)), durations)
        except (ValueError, TypeError) as error:
            parser.error(str(error))
        if (args.n_step != 1 or not args.life_terminal or args.exploration_max_repeat != 1
                or args.bootstrap_heads or args.quantiles or args.greedy_trace_cut
                or args.spr_weight or args.inverse_weight or args.exploration_actions != 'uniform'):
            parser.error('learned repeats require scalar one-option targets, life boundaries and no other exploration/auxiliary variant')
        if prior and tuple(prior['config'].get('learned_repeats', ())) != args.learned_repeats:
            parser.error('learned duration set must match the resumed model')
    if min(args.envs, args.batch_size, args.capacity, args.n_step, args.train_every,
           args.target_every, args.epsilon_steps, args.eval_every, args.eval_games, args.eval_envs) < 1:
        parser.error("counts and intervals must be positive")
    if min(args.steps, args.warmup, args.max_episode_steps, args.eval_max_steps, args.mlx_cache_mb) < 0:
        parser.error("limits must be nonnegative")
    if args.warmup > args.capacity:
        parser.error("warmup cannot exceed replay capacity")
    if (not all(np.isfinite(v) for v in
                (args.learning_rate, args.gamma, args.reward_scale, args.epsilon_final))
            or args.learning_rate <= 0 or not 0 < args.gamma < 1
            or args.reward_scale <= 0 or not 0 <= args.epsilon_final <= 1):
        parser.error("invalid optimizer, discount, reward scale or epsilon")
    if not 1 <= args.tstates <= 1_000_000:
        parser.error("tstates out of range")
    try:
        validate_observation_stride(args.observation_stride)
    except ValueError as error:
        parser.error(str(error))
    if (args.bootstrap_heads < 0 or args.bootstrap_heads == 1
            or not np.isfinite(args.bootstrap_probability) or not 0 < args.bootstrap_probability <= 1
            or not np.isfinite(args.bootstrap_prior_scale) or args.bootstrap_prior_scale < 0
            or not np.isfinite(args.bootstrap_epsilon) or not 0 <= args.bootstrap_epsilon <= 1):
        parser.error("invalid bootstrap heads, membership, prior scale or epsilon")
    if args.run.exists() and not args.resume:
        parser.error("run already exists; choose a new directory or --resume")
    if args.artifacts.resolve() == args.run.resolve():
        parser.error("checkpoint and replay-artifact roots must be separate")
    if (not np.isfinite(args.curriculum_probability) or not 0 <= args.curriculum_probability <= 1
            or min(args.curriculum_lookback, args.curriculum_score_interval) < 0
            or min(args.curriculum_bins, args.curriculum_per_bin, args.curriculum_screen_interval) < 1
            or not 0 <= args.curriculum_boot_envs < args.envs
            or (args.curriculum_share and not args.curriculum_probability)
            or (args.curriculum_boot_envs and not args.curriculum_share)):
        parser.error("invalid own-experience curriculum settings")
    if args.curriculum_trigger == "life-loss" and (
            not args.curriculum_probability or not args.curriculum_lookback):
        parser.error("life-loss archive requires positive curriculum probability and lookback")
    if args.curriculum_restored_life_only and (not args.curriculum_probability or not args.life_terminal):
        parser.error("restored-life-only requires positive curriculum probability and life-terminal targets")
    if args.curriculum_boot_epsilon is not None and (
            not args.curriculum_boot_envs or not np.isfinite(args.curriculum_boot_epsilon)
            or not 0 <= args.curriculum_boot_epsilon <= 1):
        parser.error("boot epsilon requires reserved shared-curriculum workers and a rate in 0..1")
    if args.bootstrap_heads and args.curriculum_probability:
        parser.error("own-state resets currently support ordinary DQN only")
    from .persistent_exploration import duration_distribution, worker_epsilons
    from .defense_exploration_actions import action_probabilities
    if args.exploration_actions != "uniform" and (args.exploration_max_repeat <= 1 or args.allow_enter):
        parser.error("balanced exploration requires persistent exploration and the standard 20 actions")
    exploration_action_probabilities = action_probabilities(args.exploration_actions, len(action_names(args.allow_enter)))
    try:
        duration_distribution(args.exploration_max_repeat, args.exploration_exponent)
    except ValueError as error:
        parser.error(str(error))
    if args.bootstrap_heads and args.exploration_max_repeat > 1:
        parser.error("persistent random exploration currently supports ordinary DQN only")
    if (args.quantiles != 0 and not 2 <= args.quantiles <= 256
            or not np.isfinite(args.quantile_exploration_power)
            or not 0 <= args.quantile_exploration_power <= 4
            or args.quantile_exploration_power and not args.quantiles):
        parser.error("invalid quantile count or exploration power")
    if args.quantiles and (args.bootstrap_heads or args.exploration_max_repeat > 1):
        parser.error("quantile DQN does not combine with bootstrap heads or persistent actions")
    if args.greedy_trace_cut and (args.bootstrap_heads or args.quantiles
                                 or not args.compact_replay or args.n_step > 32):
        parser.error("greedy trace cuts require compact scalar DQN and n-step in 1..32")
    if not np.isfinite(args.spr_weight) or not 0 <= args.spr_weight <= 10:
        parser.error("SPR weight must be finite and in 0..10")
    if args.spr_weight and (args.bootstrap_heads or args.quantiles or args.greedy_trace_cut
                           or not args.compact_replay or args.n_step > 5):
        parser.error("SPR requires compact scalar DQN, n-step in 1..5, and no trace cuts")
    if not np.isfinite(args.inverse_weight) or not 0 <= args.inverse_weight <= 10:
        parser.error("inverse weight must be finite and in 0..10")
    if args.inverse_weight and (args.bootstrap_heads or args.quantiles or args.greedy_trace_cut
                               or args.spr_weight or not args.compact_replay or args.n_step > 32):
        parser.error("inverse task requires compact scalar DQN, n-step in 1..32, and no other auxiliary task")
    initialization = None
    if args.init_from_dqn:
        if args.resume or not (args.quantiles or args.learned_repeats):
            parser.error("scalar initialization requires fresh quantile/action-duration DQN, not --resume")
        parent = json.loads((args.init_from_dqn/"state.json").read_text())
        pc = parent['config']
        if (pc.get('algorithm') != DQN_ALGORITHM or pc.get('game') != 'defense'
                or pc.get('game_sha256') != GAME_SHA256 or pc.get('environment_version') != ENVIRONMENT_VERSION
                or pc.get('action_names') != list(action_names(args.allow_enter))
                or pc.get('tstates') != args.tstates
                or pc.get('observation_stride', 1) != args.observation_stride):
            parser.error("scalar initialization requires compatible Defense DQN screen/action timing")
        initialization = dict(checkpoint=str(args.init_from_dqn), steps=parent['steps'],
                              model_sha256=sha256(args.init_from_dqn/'model.safetensors'),
                              state_sha256=sha256(args.init_from_dqn/'state.json'),
                              semantics="parent online encoder and dueling values tiled into coincident quantiles; "
                                        "target equals transferred online; fresh Adam, RNG, counters and replay")
        if args.learned_repeats:
            initialization['semantics'] = ('parent encoder/value copied, advantages tiled by duration; '
                'target equals transferred online; fresh Adam, RNG, counters and replay; no true duration-value equality assumed')

    # MLX is main-process-only: spawned emulator workers never import it.
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from .model import Learner
    from .train import checkpoint as base_checkpoint
    def checkpoint(learner, directory, saved):
        if args.spr_weight:
            # Auxiliary files precede the state marker; hashes reject a mixed
            # interrupted latest checkpoint. Ordinary/Breakdown format is unchanged.
            saved = dict(saved, spr_auxiliary_hashes=learner.save_auxiliary(directory))
        if args.inverse_weight:
            saved = dict(saved, inverse_auxiliary_hashes=learner.save_auxiliary(directory))
        base_checkpoint(learner, directory, saved)
    mx.set_cache_limit(args.mlx_cache_mb*1024*1024)
    if args.learned_repeats:
        from .defense_repeat_model import RepeatLearner
        agent = RepeatLearner(args.learning_rate, args.seed,
                             len(action_names(args.allow_enter)), args.learned_repeats)
    elif args.bootstrap_heads:
        from .defense_bootstrap import BootstrapLearner
        agent = BootstrapLearner(args.learning_rate, args.seed,
                                 action_count=len(action_names(args.allow_enter)),
                                 heads=args.bootstrap_heads, prior_scale=args.bootstrap_prior_scale)
    elif args.quantiles:
        from .defense_quantile import QuantileLearner
        agent = QuantileLearner(args.learning_rate, args.seed,
                                action_count=len(action_names(args.allow_enter)),
                                quantiles=args.quantiles, exploration_power=args.quantile_exploration_power)
    elif args.inverse_weight:
        from .defense_inverse import InverseLearner
        agent = InverseLearner(args.learning_rate, args.seed,
                               action_count=len(action_names(args.allow_enter)), weight=args.inverse_weight)
    elif args.spr_weight:
        from .defense_spr import SprLearner
        agent = SprLearner(args.learning_rate, args.seed,
                           action_count=len(action_names(args.allow_enter)),
                           horizon=args.n_step, weight=args.spr_weight)
    elif args.greedy_trace_cut:
        from .defense_trace import TraceLearner
        agent = TraceLearner(args.learning_rate, args.seed,
                             action_count=len(action_names(args.allow_enter)), horizon=args.n_step)
    else:
        agent = Learner(args.learning_rate, args.seed, action_count=len(action_names(args.allow_enter)))
    if initialization is not None:
        agent.initialize_scalar(args.init_from_dqn/'model.safetensors')
        if (sha256(args.init_from_dqn/'model.safetensors') != initialization['model_sha256']
                or sha256(args.init_from_dqn/'state.json') != initialization['state_sha256']):
            raise RuntimeError("scalar initialization source changed during load")
    rng = np.random.default_rng(args.seed)
    bootstrap_rng = np.random.default_rng(args.seed+2_000_000)
    steps, updates, episodes = 0, 0, 0
    if prior:
        agent.online.load_weights(str(args.resume/"model.safetensors"))
        agent.target.load_weights(str(args.resume/"target.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        if args.spr_weight:
            agent.restore_auxiliary(args.resume if prior['config'].get('spr_weight', 0) else None, prior)
        elif args.inverse_weight:
            agent.restore_auxiliary(args.resume if prior['config'].get('inverse_weight', 0) else None, prior)
        else:
            agent.state = [agent.online.state, agent.target.state, agent.optimizer.state]
            agent.update = mx.compile(agent._update, inputs=agent.state, outputs=agent.state)
        rng.bit_generator.state = prior["rng"]
        if args.bootstrap_heads:
            bootstrap_rng.bit_generator.state = prior["bootstrap_rng"]
        steps, updates, episodes = (prior[k] for k in ("steps", "updates", "episodes"))
    mx.eval(agent.state)
    boot_episodes = prior.get("boot_episodes", episodes) if prior else 0
    restored_segments = prior.get("restored_segments", 0) if prior else 0
    persistent = None
    if args.exploration_max_repeat > 1:
        from .persistent_exploration import PersistentExploration
        exploration_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 3089]))
        if prior and "persistent_exploration_rng" in prior:
            exploration_rng.bit_generator.state = prior["persistent_exploration_rng"]
        persistent = PersistentExploration(args.envs, len(action_names(args.allow_enter)),
                                          args.exploration_max_repeat, args.exploration_exponent,
                                          exploration_rng, exploration_action_probabilities)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(game="defense", algorithm=(REPEAT_ALGORITHM if args.learned_repeats else BOOTSTRAP_ALGORITHM if args.bootstrap_heads else
                                            QUANTILE_ALGORITHM if args.quantiles else DQN_ALGORITHM),
                  game_sha256=GAME_SHA256,
                  native_sha256=sha256(Path("libtrs.so")),
                  environment_version=ENVIRONMENT_VERSION,
                  environment_source_sha256=sha256(Path(__file__).with_name("defense.py")),
                  trainer_source_sha256=sha256(Path(__file__)),
                  model_source_sha256=sha256(Path(__file__).with_name("model.py")),
                  replay_source_sha256=sha256(Path(__file__).with_name("replay.py")),
                  action_names=list(action_names(args.allow_enter)), mlx=mx.__version__,
                  observation="four raw 16x64 video-memory frames",
                  reward="visible score difference only, constant scale for optimizer",
                  training_policy="epsilon-greedy learned Q-values; uniform random exploration",
                  policy="learned Q-values, greedy", evaluation_policy="learned Q-values, greedy",
                  replay_resume="refill from new own experience; no saved trajectories loaded",
                  resume_semantics="online, target, optimizer and RNG restored; episodes restart from boot",
                  priority_alpha=.6, priority_beta="0.4 to 1 over epsilon-steps aggregate actions")
    repeated = None
    if args.learned_repeats:
        from .defense_repeat import RepeatedActions, OptionReturns, LearnedRepeatPolicy
        repeated = RepeatedActions(args.envs, len(config['action_names']), args.learned_repeats)
        config.update(repeat_source_sha256=sha256(Path(__file__).with_name('defense_repeat.py')),
                      repeat_model_source_sha256=sha256(Path(__file__).with_name('defense_repeat_model.py')),
                      training_policy='epsilon-greedy joint action-duration options; epsilon per option start',
                      repeat_target='sum gamma**k times actual scaled score increments; gamma**actual_duration Double-Q bootstrap; zero on visible life loss',
                      repeat_warmup='independent one-base-action uniform choices until replay warmup',
                      repeat_boundary='cancel holds at visible life loss or episode end, in both training and evaluation',
                      repeat_observation='original base-action screen cadence; no skipped-history subsampling',
                      repeat_resume='restore Q/target/Adam/RNG; new boot games, empty option/replay state',
                      policy=policy_description(config), evaluation_policy=policy_description(config))
        if initialization is not None:
            config['repeat_initialization'] = initialization
        elif prior and 'repeat_initialization' in prior['config']:
            config['repeat_initialization'] = prior['config']['repeat_initialization']
    if args.bootstrap_heads:
        config.update(bootstrap_source_sha256=sha256(Path(__file__).with_name("defense_bootstrap.py")),
                      bootstrap_replay_source_sha256=sha256(Path(__file__).with_name("defense_bootstrap_replay.py")),
                      training_policy="uniform value head per complete game plus constant epsilon; random warmup",
                      bootstrap_membership="independent Bernoulli mask once per n-step insertion; empty masks allowed",
                      bootstrap_priority="mean absolute TD error over all heads",
                      bootstrap_resume="masks/replay refill; head redrawn for new boot game; priors restored from weights",
                      policy=policy_description(config), evaluation_policy=policy_description(config))
    if args.quantiles:
        config.update(quantile_source_sha256=sha256(Path(__file__).with_name("defense_quantile.py")),
                      quantile_locations="fixed midpoint fractions (i+0.5)/N; no prediction sorting",
                      quantile_loss="sum predicted quantiles, mean target quantiles; Huber kappa 1",
                      quantile_priority="unweighted per-transition quantile Huber loss divided by N",
                      quantile_target="online mean-greedy action, target quantiles; no risk distortion",
                      training_policy="epsilon-greedy learned quantiles; fixed training-only power distortion",
                      quantile_distortion="bin mass b**(1+power)-a**(1+power); return risk, not epistemic uncertainty",
                      policy=policy_description(config), evaluation_policy=policy_description(config))
        if initialization is not None:
            config['quantile_initialization'] = initialization
        elif prior and 'quantile_initialization' in prior['config']:
            config['quantile_initialization'] = prior['config']['quantile_initialization']
    curriculum = {}
    if args.curriculum_probability:
        curriculum = dict(curriculum=True, curriculum_probability=args.curriculum_probability,
                          curriculum_share=args.curriculum_share,
                          curriculum_boot_envs=args.curriculum_boot_envs,
                          curriculum_lookback=args.curriculum_lookback,
                          curriculum_trigger=args.curriculum_trigger,
                          curriculum_restored_life_only=args.curriculum_restored_life_only,
                          curriculum_cells=args.curriculum_cells,
                          curriculum_score_interval=args.curriculum_score_interval,
                          curriculum_bins=args.curriculum_bins, curriculum_per_bin=args.curriculum_per_bin,
                          curriculum_screen_interval=args.curriculum_screen_interval)
        config.update({k: v for k, v in curriculum.items() if k != "curriculum"})
        config.update(curriculum_archive_saved=False,
                      curriculum_source_sha256=sha256(Path(__file__).with_name("defense_curriculum.py")),
                      snapshot_source_sha256=sha256(Path(__file__).with_name("defense_snapshot.py")))
        if args.curriculum_trigger == "life-loss":
            config.update(curriculum_trigger_semantics=(
                "actual same-life state L decisions before visible loss; immediate stage entry; "
                "not a physical-collision timestamp; no snapshots restored during evaluation"))
        if args.curriculum_restored_life_only:
            config.update(curriculum_segment_semantics=(
                "restored segments truncate at first visible life loss unless native game over; "
                "boot games remain complete; life-terminal TD targets unchanged"))
        if args.curriculum_cells == "screen":
            from .defense_cells import CELL_ENCODING
            config.update(curriculum_cell_encoding=CELL_ENCODING,
                          curriculum_cell_source_sha256=sha256(Path(__file__).with_name("defense_cells.py")),
                          curriculum_selection="bounded bottom-k screen fingerprints; uniform cell reset")
    if args.compact_replay:
        config.update(replay_storage="exact-visible-frame-interning-v1",
                      frame_storage_source_sha256=sha256(Path(__file__).with_name("frame_storage.py")))
    if persistent is not None:
        config.update(training_policy="learned Q-values with training-only persistent uniform random exploration",
                      exploration_source_sha256=sha256(Path(__file__).with_name("persistent_exploration.py")),
                      exploration_duration="bounded power law proportional to length**(-exponent)",
                      exploration_epsilon="nominal uninterrupted exploratory-step fraction; boundary cuts can reduce it",
                      exploration_mean_duration=persistent.mean_duration,
                      exploration_warmup="unchanged independent uniform actions until replay warmup",
                      exploration_reset="cancel on visible ship loss, termination or truncation; no screen-triggered starts",
                      exploration_resume="restore separate RNG; active holds and local counters clear with new boot episodes")
    if exploration_action_probabilities is not None:
        config.update(training_policy="learned Q-values with training-only persistent fixed-distribution random exploration",
                      exploration_action_probabilities=exploration_action_probabilities.tolist(),
                      exploration_actions_source_sha256=sha256(Path(__file__).with_name("defense_exploration_actions.py")),
                      exploration_actions_semantics=(
                          "equal mass for twelve stage-1 command groups; forward-fire aliases share one group; "
                          "fixed across all screens/stages; all twenty actions remain possible; "
                          "warmup and greedy evaluation unchanged; not a route or state-conditioned rule"))
    if args.curriculum_boot_epsilon is not None:
        config.update(exploration_worker_roles="first curriculum_boot_envs use fixed boot epsilon; others use schedule",
                      exploration_worker_warmup="all workers remain independent uniform until replay warmup",
                      exploration_worker_source_sha256=sha256(Path(__file__).with_name("persistent_exploration.py")))
    if args.greedy_trace_cut:
        config.update(trace_source_sha256=sha256(Path(__file__).with_name("defense_trace.py")),
                      trace_replay_source_sha256=sha256(Path(__file__).with_name("defense_trace_replay.py")),
                      trace_target="up to n own rewards; stop before first later action not current online argmax; Double-Q bootstrap",
                      trace_ties="deterministic online argmax, matching normal greedy evaluation",
                      trace_resume="full network/target/Adam/RNG; replay refills; diagnostic counts clear")
    if args.spr_weight:
        config.update(spr_architecture="spatial-own-future-v1", spr_ema=agent.ema_rate,
                      spr_dropout=agent.dropout,
                      spr_source_sha256=sha256(Path(__file__).with_name('defense_spr.py')),
                      spr_replay_source_sha256=sha256(Path(__file__).with_name('defense_spr_replay.py')),
                      spr_sequence_source_sha256=sha256(Path(__file__).with_name('defense_trace_replay.py')),
                      spr_objective="PER-weighted mean valid-step cosine distance plus unchanged scalar TD loss",
                      spr_inference="ordinary QNetwork only; no dropout, predictor or future observations",
                      spr_resume="full Q/target/Adam plus auxiliary/EMA/Adam/key; own replay refills",
                      spr_conversion="own scalar Q/target/Adam/RNG preserved; auxiliary fresh; EMA copies online")
    if args.inverse_weight:
        from .defense_inverse import ARCHITECTURE
        config.update(inverse_architecture=ARCHITECTURE,
                      inverse_source_sha256=sha256(Path(__file__).with_name('defense_inverse.py')),
                      inverse_replay_source_sha256=sha256(Path(__file__).with_name('defense_inverse_replay.py')),
                      inverse_sequence_source_sha256=sha256(Path(__file__).with_name('defense_trace_replay.py')),
                      inverse_objective="PER-weighted actual-action cross entropy plus unchanged scalar TD loss",
                      inverse_inference="ordinary QNetwork only; no next screen or inverse classifier",
                      inverse_resume="full Q/target/Adam plus inverse head/Adam; own replay refills",
                      inverse_conversion="own scalar Q/target/Adam/RNG preserved; auxiliary head/Adam fresh")
    args.run.mkdir(parents=True, exist_ok=True)
    write_json(args.run/("resume-config.json" if prior else "config.json"), config)
    if args.bootstrap_heads:
        from .defense_bootstrap_replay import BootstrapReplay
        replay = BootstrapReplay(args.capacity, args.bootstrap_heads, args.bootstrap_probability,
                                 bootstrap_rng, compact=args.compact_replay)
    elif args.inverse_weight:
        from .defense_inverse_replay import InverseReplay
        from .defense_trace_replay import TraceNStep
        replay = InverseReplay(args.capacity, args.n_step, len(action_names(args.allow_enter)))
    elif args.spr_weight:
        from .defense_spr_replay import SprReplay
        from .defense_trace_replay import TraceNStep
        replay = SprReplay(args.capacity, args.n_step, len(action_names(args.allow_enter)))
    elif args.greedy_trace_cut:
        from .defense_trace_replay import TraceNStep, TraceReplay
        replay = TraceReplay(args.capacity, args.n_step, len(action_names(args.allow_enter)))
    else:
        replay = Replay(args.capacity, compact=args.compact_replay)
    buffer_type = TraceNStep if args.greedy_trace_cut or args.spr_weight or args.inverse_weight else NStep
    buffers = ([OptionReturns(replay, args.gamma, len(config['action_names']), args.learned_repeats)
                for _ in range(args.envs)] if repeated is not None else
               [buffer_type(replay, args.n_step, args.gamma) for _ in range(args.envs)])
    recent = deque(maxlen=100)
    recent_restored = deque(maxlen=100)
    started, start_steps = time.monotonic(), steps
    last_log = started
    next_eval, next_update = steps+args.eval_every, steps+args.train_every
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)
    stop, workers = False, None
    episode_heads = None

    def state():
        saved = dict(steps=steps, updates=updates, episodes=episodes,
                     boot_episodes=boot_episodes, restored_segments=restored_segments,
                     rng=rng.bit_generator.state, config=config)
        if args.bootstrap_heads:
            saved["bootstrap_rng"] = bootstrap_rng.bit_generator.state
        if persistent is not None:
            saved["persistent_exploration_rng"] = exploration_rng.bit_generator.state
            saved["persistent_exploration"] = persistent.stats()
        if repeated is not None:
            saved['learned_repeat_stats'] = repeated.stats()
            saved['learned_repeat_completed'] = sum(b.completed for b in buffers)
            saved['learned_repeat_interrupted'] = sum(b.interrupted for b in buffers)
        if args.greedy_trace_cut:
            saved["greedy_trace"] = agent.trace_stats()
        if args.spr_weight:
            saved['spr'] = agent.spr_stats()
        if args.inverse_weight:
            saved['inverse'] = agent.inverse_stats()
        return saved

    def log(row):
        row = dict(wall_seconds=round(time.monotonic()-started, 2), **row)
        log_file.write(json.dumps(row)+"\n")
        write_json(args.run/"status.json", dict(row, pid=os.getpid(), training_steps=steps,
                                              updates=updates, updated_unix=time.time()))
        print(json.dumps(row), flush=True)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(dict(event="start", steps=steps, config=config))
    try:
        workers = VectorEnv(args.envs, args.seed+steps, game="defense", tstates=args.tstates,
                            max_steps=args.max_episode_steps, allow_enter=args.allow_enter,
                            observation_stride=args.observation_stride, **curriculum)
        log(dict(event="workers_started", workers=workers.runtime()))
        observations = workers.observations
        last_metrics = {}
        while not stop and (not args.steps or steps < args.steps):
            epsilon = max(args.epsilon_final, 1-(1-args.epsilon_final)*
                          max(0, steps-args.warmup)/args.epsilon_steps)
            if args.bootstrap_heads:
                epsilon = args.bootstrap_epsilon
                if episode_heads is None:
                    episode_heads = rng.integers(args.bootstrap_heads, size=args.envs)
            epsilons = worker_epsilons(epsilon, args.envs, args.curriculum_boot_envs,
                                      args.curriculum_boot_epsilon)
            if replay.size < max(1, args.warmup):
                actions = rng.integers(len(config["action_names"]), size=args.envs)
            else:
                actions = (agent.actions(observations, episode_heads) if args.bootstrap_heads
                           else agent.actions(observations))
                if repeated is not None:
                    idle = np.flatnonzero(repeated.remaining == 0)
                    rates = np.asarray(epsilons)
                    exploring = idle[rng.random(len(idle)) < (rates if rates.ndim == 0 else rates[idle])]
                    actions[exploring] = rng.integers(len(config['action_names']) * len(args.learned_repeats), size=len(exploring))
                elif persistent is None:
                    explore = rng.random(args.envs) < epsilons
                    actions[explore] = rng.integers(len(config["action_names"]), size=int(explore.sum()))
                else:
                    actions = persistent.select(actions, epsilons)
            if repeated is not None:
                for worker in np.flatnonzero(repeated.remaining == 0):
                    buffers[worker].begin(observations[worker], int(actions[worker]))
                actions = repeated.select(actions)
            results = workers.step(actions)
            if persistent is not None:
                persistent.reset(np.asarray([terminal or truncated or info["life_lost"]
                                             for _, _, terminal, truncated, info, _ in results], dtype=bool))
            following = []
            for worker, (obs, reward, terminal, truncated, info, reset) in enumerate(results):
                learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                if repeated is not None:
                    buffers[worker].append(reward*args.reward_scale, obs, learning_terminal, truncated)
                else:
                    buffers[worker].append(observations[worker], int(actions[worker]),
                                           reward*args.reward_scale, obs, learning_terminal, truncated)
                following.append(reset if reset is not None else obs)
                if "curriculum_archive_add" in info:
                    log(dict(event="curriculum_archive", worker=worker, action_counter=steps+worker+1,
                             **info["curriculum_archive_add"]))
                if terminal or truncated:
                    episodes += 1
                    if info.get("full_game", True):
                        boot_episodes += 1
                        recent.append(info)
                    else:
                        restored_segments += 1
                        recent_restored.append(info["episode_reward"])
                    log(dict(event="episode", worker=worker, action_counter=steps+worker+1,
                             episode=episodes, **info,
                             **({"bootstrap_head": int(episode_heads[worker])} if args.bootstrap_heads else {})))
                    if args.bootstrap_heads:
                        # No change at ship loss: the selected head lasts the whole game.
                        episode_heads[worker] = rng.integers(args.bootstrap_heads)
            if repeated is not None:
                repeated.reset(np.asarray([t or u or info['life_lost']
                    for _, _, t, u, info, _ in results], dtype=bool))
            observations = np.stack(following)
            steps += args.envs
            while steps >= next_update:
                next_update += args.train_every
                if replay.size < max(1, args.warmup):
                    continue
                beta = min(1., .4+.6*steps/args.epsilon_steps)
                indices, batch = replay.sample(args.batch_size, rng, beta)
                loss, errors, q = agent.train(batch)
                if not np.isfinite([loss, q]).all() or not np.isfinite(errors).all():
                    raise RuntimeError("Non-finite Defense DQN update")
                replay.priorities(indices, errors)
                updates += 1
                if updates % args.target_every == 0:
                    agent.sync_target()
                last_metrics = dict(loss=loss, mean_q=q, priority_beta=beta)
                if args.greedy_trace_cut:
                    last_metrics['greedy_trace'] = agent.trace_stats()
                if args.spr_weight:
                    last_metrics['spr'] = agent.spr_stats()
                if args.inverse_weight:
                    last_metrics['inverse'] = agent.inverse_stats()
            if time.monotonic()-last_log >= 10:
                log(dict(event="progress", steps=steps, episodes=episodes, updates=updates,
                         boot_episodes=boot_episodes, restored_segments=restored_segments,
                         recent_restored=dict(segments=len(recent_restored),
                                              mean_new_score=float(np.mean(recent_restored))
                                              if recent_restored else None),
                         replay_size=replay.size, epsilon=epsilon,
                         **({"worker_epsilons": epsilons.tolist()}
                            if args.curriculum_boot_epsilon is not None else {}),
                         steps_per_second=(steps-start_steps)/(time.monotonic()-started),
                         recent={k: v for k, v in summarize(list(recent)).items() if k != "games"},
                         **({"replay_frame_storage": replay.frame_storage.stats()}
                            if args.compact_replay else {}),
                         **({"persistent_exploration": persistent.stats()} if persistent is not None else {}),
                         **({'learned_repeat_stats': repeated.stats()} if repeated is not None else {}),
                         mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
                         **last_metrics))
                last_log = time.monotonic()
            if steps >= next_eval and not stop:
                directory = args.run/f"step-{steps:012d}"
                if directory.exists():
                    raise RuntimeError("Refusing to overwrite a historical checkpoint")
                checkpoint(agent, directory, state())
                checkpoint(agent, args.run/"latest", state())
                log(dict(event="validation_start", steps=steps, checkpoint=str(directory)))
                policy = (LearnedRepeatPolicy(lambda obs: np.array(agent.online(mx.array(obs))),
                    len(config['action_names']), args.learned_repeats) if repeated is not None else
                    greedy_policy(lambda obs: np.array(agent.online(mx.array(obs)))))
                result = evaluate(policy, range(args.eval_seed, args.eval_seed+args.eval_games),
                                  tstates=args.tstates, max_steps=args.eval_max_steps,
                                  envs=args.eval_envs, log=log, should_stop=lambda: stop,
                                  allow_enter=args.allow_enter, observation_stride=args.observation_stride)
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
        checkpoint(agent, args.run/"latest", state())
        if workers is not None:
            workers.close()
        log(dict(event="stopped", steps=steps, stop_requested=stop))
        log_file.close()


if __name__ == "__main__":
    main()
