"""PPO with a screen-only categorical policy, optionally initialized by own RL."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

from .evaluate import (EvaluationCancelled, categorical_policy, evaluate,
                       level_rank, level_target_met, summary)
from .env import ENVIRONMENT_VERSION, validate_observation_stride
from .reference import ReferencePolicy, reference_kl, reference_metadata
from .vector import VectorEnv


def _load_backend():
    # spawn re-imports the entry module in every emulator worker. Only the
    # parent learner needs MLX; keep GPU initialization out of those workers.
    global mx, nn, optim, tree_flatten, tree_unflatten, QNetwork, write_json
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from .model import QNetwork
    from .train import write_json


def gae(rewards, values, boundaries, last_value, gamma, lam):
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards)-1 else values[t+1]
        continuation = 1-boundaries[t]
        delta = rewards[t]+gamma*continuation*next_value-values[t]
        carry = delta+gamma*lam*continuation*carry
        advantages[t] = carry
    return advantages, advantages+values


def validation_protocol(config):
    """Old checkpoints predate configurable validation seeds and action limits."""
    defaults = dict(eval_seed=10000, eval_games=5, eval_max_steps=20000, tstates=100000,
                    environment_version="literal-game-over-v1", eval_envs=1, observation_stride=1)
    return tuple(config.get(key, default) for key, default in defaults.items())


def update_selection(result, target_level, best_mean, best_levels, *, game_win=False):
    """Compute both records before saving any checkpoint's selection state."""
    rank = level_rank(result, target_level, game_win=game_win)
    improved_mean = rank is not None and result["mean_score"] > best_mean
    improved_levels = rank is not None and (best_levels is None or rank > best_levels)
    return (result["mean_score"] if improved_mean else best_mean,
            rank if improved_levels else best_levels, improved_mean, improved_levels)


class PPO:
    def __init__(self, seed=17, learning_rate=2.5e-4, entropy=0.01, reference_kl_weight=0,
                 action_count=6):
        if not np.isfinite(reference_kl_weight) or reference_kl_weight < 0:
            raise ValueError("reference KL weight must be finite and nonnegative")
        _load_backend()
        mx.random.seed(seed)
        self.model = QNetwork(action_count=action_count)
        self.optimizer = optim.Adam(learning_rate, eps=1e-5)
        self.optimizer.init(self.model.trainable_parameters())
        self.entropy = entropy
        self.reference_kl_weight = reference_kl_weight
        self.compile()

    def compile(self):
        self.state = [self.model.state, self.optimizer.state]
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(self.model.policy_value, inputs=self.model.state)
        mx.eval(self.state)

    def _loss(self, model, obs, actions, old_logp, advantages, returns, reference_log_probs=None,
              logit_bias=None):
        logits, values = model.policy_value(obs)
        if logit_bias is not None:
            logits = logits + mx.stop_gradient(logit_bias)
        log_probs = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        logp = mx.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
        logratio = logp-old_logp
        ratio = mx.exp(logratio)
        actor = -mx.mean(mx.minimum(ratio*advantages, mx.clip(ratio, .8, 1.2)*advantages))
        critic = .5*mx.mean(mx.square(values-returns))
        entropy = -mx.mean(mx.sum(mx.exp(log_probs)*log_probs, axis=-1))
        kl = mx.mean((ratio-1)-logratio)
        if self.reference_kl_weight:
            if reference_log_probs is None:
                raise ValueError("enabled reference penalty requires rollout targets")
            anchor = reference_kl(log_probs, reference_log_probs)
            return (actor+.5*critic-self.entropy*entropy+self.reference_kl_weight*anchor,
                    (actor, critic, entropy, kl, anchor))
        return actor+.5*critic-self.entropy*entropy, (actor, critic, entropy, kl)

    def _update(self, obs, actions, old_logp, advantages, returns, reference_log_probs=None,
                logit_bias=None):
        (loss, metrics), grads = nn.value_and_grad(self.model, self._loss)(
            self.model, obs, actions, old_logp, advantages, returns, reference_log_probs, logit_bias)
        grads, norm = optim.clip_grad_norm(grads, .5)
        self.optimizer.update(self.model, grads)
        return loss, metrics

    def act(self, obs, rng, logit_bias=None):
        logits, values = self.predict(mx.array(obs))
        logits, values = np.array(logits), np.array(values)
        if logit_bias is not None:
            if logit_bias.shape != logits.shape or not np.isfinite(logit_bias).all():
                raise ValueError("policy bias noise must match logits and be finite")
            logits = logits + logit_bias
        logp = logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        probs = np.exp(logp)
        actions = (rng.random(len(obs))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, logits.shape[1]-1)
        return actions.astype(np.int32), logp[np.arange(len(obs)), actions], values

    def policy(self, seed=0):
        return categorical_policy(lambda obs: np.array(self.predict(mx.array(obs))[0]), seed)

    def save(self, directory, state):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(directory/"model.tmp.safetensors"))
        (directory/"model.tmp.safetensors").replace(directory/"model.safetensors")
        mx.savez(str(directory/"optimizer.tmp.npz"), **dict(tree_flatten(self.optimizer.state)))
        (directory/"optimizer.tmp.npz").replace(directory/"optimizer.npz")
        write_json(directory/"state.json", state)


def memory_metrics():
    return dict(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                mlx_peak_bytes=mx.get_peak_memory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--initialize", type=Path, help="weights from this agent's own earlier RL")
    parser.add_argument("--resume", type=Path)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, default=0, help="absolute action counter; 0 = unlimited")
    budget.add_argument("--additional-steps", type=int, default=0,
                        help="actions to add after resuming; 0 = unlimited")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--reference-policy", type=Path,
                        help="training-only frozen checkpoint from this agent's own RL")
    parser.add_argument("--reference-kl-weight", type=float, default=0,
                        help="PPO-only KL(reference || learner) weight; 0 disables")
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--logit-scale", type=float, default=10)
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--eval-envs", type=int, default=1,
                        help="parallel validation workers; 1 preserves the serial protocol")
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-max-steps", type=int, default=20000,
                        help="per-validation-game action limit; 0 = complete games")
    parser.add_argument("--target-level", type=int, default=2,
                        help="displayed level to reach (5 means clearing levels 1-4)")
    parser.add_argument("--target-clears", type=int, default=3,
                        help="complete validation games that must reach target-level")
    parser.add_argument("--target-game-win", action=argparse.BooleanOptionalAction, default=False,
                        help="require verified original-game victories, not merely reaching a level")
    parser.add_argument("--stop-on-target", action=argparse.BooleanOptionalAction, default=True,
                        help="supervised learners continue until the supervisor verifies the full audit")
    parser.add_argument("--max-episode-steps", type=int, default=30000,
                        help="training episode action limit; 0 = no truncation")
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--observation-stride", type=int, default=1,
                        help="action spacing between the four input screen frames")
    parser.add_argument("--curriculum-probability", type=float, default=0,
                        help="training-only self-reached-state reset fraction; 0 disables")
    parser.add_argument("--curriculum-min-level", type=int, default=2)
    parser.add_argument("--curriculum-per-level", type=int, default=8)
    parser.add_argument("--curriculum-score-interval", type=int, default=0,
                        help="archive own within-level progress every N new points; 0 = level entries only")
    parser.add_argument("--curriculum-share", action=argparse.BooleanOptionalAction, default=False,
                        help="share same-run self-reached entries across training workers")
    parser.add_argument("--curriculum-boot-envs", type=int, default=0,
                        help="reserve this many workers for from-boot play; requires shared curriculum")
    parser.add_argument("--mlx-cache-mb", type=int, default=-1,
                        help="free-buffer cache in MiB; -1 keeps MLX default, 0 disables cache")
    parser.add_argument("--sil-updates", type=int, default=0,
                        help="own-experience self-imitation updates per rollout; 0 disables")
    parser.add_argument("--sil-capacity", type=int, default=32768)
    parser.add_argument("--sil-suffix-steps", type=int, default=2048,
                        help="bounded retained suffix per worker; commit only at a real learning terminal")
    parser.add_argument("--sil-batch-size", type=int, default=512)
    parser.add_argument("--sil-loss-weight", type=float, default=.1)
    parser.add_argument("--sil-value-weight", type=float, default=.01)
    parser.add_argument("--sil-priority-alpha", type=float, default=.6)
    parser.add_argument("--sil-priority-beta", type=float, default=.1)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        if prior.get("evaluation_only") or prior.get("resume_supported") is False:
            parser.error("evaluation-only checkpoint cannot resume training; use an original learner checkpoint")
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            flag = key.replace("_", "-")
            if (hasattr(args, key) and key not in ("run", "resume", "steps", "additional_steps", "initialize")
                    and "--"+flag not in explicit and "--no-"+flag not in explicit):
                setattr(args, key, value)
    try:
        args.observation_stride = validate_observation_stride(args.observation_stride)
    except ValueError as error:
        parser.error(str(error))
    if min(args.envs, args.rollout, args.batch_size, args.epochs,
           args.eval_every, args.eval_games, args.eval_envs) <= 0:
        parser.error("counts and intervals must be positive")
    if args.rollout*args.envs % args.batch_size:
        parser.error("rollout * envs must be divisible by batch-size")
    if args.target_level < 2 or not 1 <= args.target_clears <= args.eval_games:
        parser.error("target-level must be >= 2 and target-clears must be between 1 and eval-games")
    if min(args.steps, args.additional_steps, args.eval_max_steps, args.max_episode_steps) < 0:
        parser.error("action limits must be nonnegative")
    if (not 0 <= args.curriculum_probability <= 1 or args.curriculum_min_level < 2
                or args.curriculum_per_level < 1 or args.curriculum_score_interval < 0):
        parser.error("invalid curriculum probability, minimum level, or capacity")
    if args.curriculum_share and not args.curriculum_probability:
        parser.error("curriculum-share requires a positive curriculum-probability")
    if not 0 <= args.curriculum_boot_envs < args.envs:
        parser.error("curriculum-boot-envs must be between zero and envs minus one")
    if args.curriculum_boot_envs and not args.curriculum_share:
        parser.error("curriculum-boot-envs requires shared curriculum")
    if args.mlx_cache_mb < -1:
        parser.error("mlx-cache-mb must be -1 or nonnegative")
    if (args.sil_updates < 0 or min(args.sil_capacity,args.sil_suffix_steps,args.sil_batch_size) < 1
            or not np.isfinite(args.sil_loss_weight) or args.sil_loss_weight <= 0
            or not np.isfinite(args.sil_value_weight) or args.sil_value_weight < 0
            or not 0 <= args.sil_priority_alpha <= 1 or not 0 <= args.sil_priority_beta <= 1):
        parser.error("invalid SIL settings")
    expected_reference_hash = None
    if (prior and args.reference_policy is not None
            and str(args.reference_policy) == prior["config"].get("reference_policy")):
        expected_reference_hash = prior["config"].get("reference_checkpoint_sha256")
    try:
        reference_info = reference_metadata(args.reference_policy, args.reference_kl_weight,
                                            args.tstates, args.observation_stride,
                                            expected_reference_hash)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    _load_backend()
    if args.mlx_cache_mb >= 0:
        mx.set_cache_limit(args.mlx_cache_mb * 1024 * 1024)
    if args.additional_steps:
        args.steps = (prior["steps"] if prior else 0)+args.additional_steps
    args.run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.run/"config.json").exists():
        parser.error("run exists; use --resume or a new directory")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    initialization = args.initialize or (prior["config"].get("initialize") if prior else None)
    prior_actions = prior["config"].get("prior_training_actions", 0) if prior else 0
    if initialization and not prior_actions:
        initializer_state = Path(initialization).parent/"state.json"
        if initializer_state.exists():
            initializer = json.loads(initializer_state.read_text())
            prior_actions = initializer["steps"]+initializer["config"].get("prior_training_actions", 0)
    config.update(algorithm="ppo", mlx=mx.__version__, device=mx.device_info(),
                  environment_version=ENVIRONMENT_VERSION,
                  prior_training_actions=prior_actions,
                  observation="four raw 16x64 video-memory frames", reward="screen score difference only",
                  evaluation_policy="sample learned categorical distribution; seeded")
    if args.resume:
        config["resume_checkpoint_sha256"] = hashlib.sha256(
            (args.resume/"model.safetensors").read_bytes()).hexdigest()
    if reference_info:
        config["reference_checkpoint_sha256"] = reference_info["sha256"]
    write_json(args.run/("resume-config.json" if args.resume else "config.json"), config)
    agent = PPO(args.seed, args.learning_rate, args.entropy, args.reference_kl_weight)
    rng = np.random.default_rng(args.seed)
    steps, episodes, best, best_levels = 0, 0, -1.0, None
    if args.resume:
        agent.model.load_weights(str(args.resume/"model.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        agent.compile()
        steps, episodes = prior["steps"], prior["episodes"]
        if validation_protocol(prior["config"]) == validation_protocol(config):
            best = prior["best_mean"]
            if (prior["config"].get("target_level", 2) == args.target_level
                    and prior["config"].get("target_game_win", False) == args.target_game_win):
                saved_rank = prior.get("best_level_rank")
                best_levels = tuple(saved_rank) if saved_rank is not None else None
        rng.bit_generator.state = prior["rng"]
    elif args.initialize:
        agent.model.load_weights(str(args.initialize))
        agent.model.advantage.weight *= args.logit_scale
        agent.model.advantage.bias *= args.logit_scale
    mx.eval(agent.state)
    reference = ReferencePolicy(args.reference_policy) if reference_info else None
    sil = None
    if args.sil_updates:
        from .sil import SILReplay, TrainingSuffixes, SelfImitation
        sil_replay = SILReplay(args.sil_capacity,args.sil_priority_alpha)
        sil_collector = TrainingSuffixes(sil_replay,args.envs,args.gamma,args.sil_suffix_steps)
        sil = SelfImitation(agent,args.sil_loss_weight,args.sil_value_weight)
        sil_rng = np.random.default_rng(np.random.SeedSequence([args.seed,steps,941]))
        if prior and "sil_rng" in prior:
            sil_rng.bit_generator.state = prior["sil_rng"]
    environment = dict(tstates=args.tstates, max_steps=args.max_episode_steps,
                       observation_stride=args.observation_stride)
    if args.curriculum_probability:
        environment.update(curriculum=True, curriculum_probability=args.curriculum_probability,
                           curriculum_min_level=args.curriculum_min_level,
                           curriculum_per_level=args.curriculum_per_level,
                           curriculum_score_interval=args.curriculum_score_interval,
                           curriculum_share=args.curriculum_share,
                           curriculum_boot_envs=args.curriculum_boot_envs)
    envs = VectorEnv(args.envs, args.seed+steps, **environment)
    obs = envs.observations
    recent = deque(maxlen=100)
    beginning, last_log, start_steps = time.monotonic(), time.monotonic(), steps
    next_eval = (steps//args.eval_every+1)*args.eval_every
    stop, target_met = False, False
    action_origins = dict(from_boot=0, restored=0)
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)

    def log(row):
        row = {"wall_seconds": round(time.monotonic()-beginning, 2), **row}
        log_file.write(json.dumps(row)+"\n")
        if row["event"] not in ("episode", "curriculum_episode"):
            print(json.dumps(row), flush=True)

    def state():
        saved = dict(steps=steps, episodes=episodes, best_mean=best,
                     best_level_rank=best_levels,
                     rng=rng.bit_generator.state, config=config)
        if sil is not None:
            saved.update(sil_rng=sil_rng.bit_generator.state,sil_replay_saved=False)
        return saved

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log({"event": "start", "steps": steps, "config": config})
    try:
        log({"event": "worker_runtime", "steps": steps, "workers": envs.runtime()})
        while not stop and (not args.steps or steps < args.steps):
            screens, action_buffer, logps, value_buffer, rewards, boundaries = [], [], [], [], [], []
            rollout_origins = dict(from_boot=0, restored=0)
            for t in range(args.rollout):
                actions, logp, values = agent.act(obs, rng)
                screens.append(obs)
                action_buffer.append(actions)
                logps.append(logp)
                value_buffer.append(values)
                results = envs.step(actions)
                next_obs, reward_row, boundary_row = [], [], []
                for i, (frame, reward, terminal, truncated, info, reset) in enumerate(results):
                    full_game = info.get("full_game", True)
                    origin = "from_boot" if full_game else "restored"
                    rollout_origins[origin] += 1
                    action_origins[origin] += 1
                    next_obs.append(reset if reset is not None else frame)
                    learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                    if sil is not None:
                        # Raw score reward, before any PPO truncation bootstrap.
                        closed = sil_collector.append(i,obs[i],actions[i],reward,
                                                       learning_terminal,truncated,full_game)
                        if closed is not None:
                            log({"event":"sil_segment","steps":steps+i+1,
                                 "final_score":info["score"],**closed})
                    if truncated and not learning_terminal:
                        _, final_value = agent.predict(mx.array(frame[None]))
                        reward += args.gamma*float(final_value[0].item())
                    reward_row.append(reward)
                    boundary_row.append(learning_terminal or truncated)
                    if "curriculum_archive_add" in info:
                        log({"event": "curriculum_archive", "steps": steps+i+1,
                             "worker": i, **info["curriculum_archive_add"]})
                    if terminal or truncated:
                        episodes += 1
                        if full_game:
                            recent.append(info)
                        log({"event": "episode" if full_game else "curriculum_episode", **info,
                             "worker": i,
                             "episode_steps": info["steps"],
                             "steps": steps+i+1, "episode": episodes})
                rewards.append(reward_row)
                boundaries.append(boundary_row)
                obs = np.stack(next_obs)
                steps += args.envs
            _, last_value = agent.predict(mx.array(obs))
            advantages, returns = gae(np.asarray(rewards, np.float32), np.asarray(value_buffer),
                                      np.asarray(boundaries, np.float32), np.array(last_value),
                                      args.gamma, args.gae_lambda)
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            data = (np.concatenate(screens), np.concatenate(action_buffer), np.concatenate(logps),
                    advantages.reshape(-1), returns.reshape(-1))
            if reference is not None:
                # Query only learner-generated rollout screens, once per rollout.
                # Detached targets are reused across PPO epochs; SIL is unchanged.
                data += (reference.targets(data[0], args.batch_size),)
            metrics = []
            for epoch in range(args.epochs):
                order = rng.permutation(len(data[0]))
                for j in range(0, len(order), args.batch_size):
                    indices = order[j:j+args.batch_size]
                    loss, aux = agent.update(*(mx.array(x[indices]) for x in data))
                    mx.eval(loss, aux, agent.state)
                    metrics.append([float(x.item()) for x in aux])
                    if not np.isfinite(float(loss.item())):
                        raise RuntimeError("Non-finite PPO loss")
                if np.mean([m[3] for m in metrics[-len(order)//args.batch_size:]]) > .03:
                    break
            sil_metrics = {}
            if sil is not None:
                updates = []
                if sil_replay.size:
                    for _ in range(args.sil_updates):
                        indices, batch = sil_replay.sample(args.sil_batch_size,sil_rng,
                                                           args.sil_priority_beta)
                        positive, details = sil.train(batch)
                        sil_replay.priorities(indices,positive)
                        updates.append(details)
                sil_metrics = {"sil":{**sil_collector.metrics(),"updates":sil.updates,
                                      "rollout_updates":updates}}
            now = time.monotonic()
            if now-last_log >= 10:
                recent_result = summary(list(recent))
                log({"event": "progress", "steps": steps, "episodes": episodes,
                     "steps_per_second": round((steps-start_steps)/(now-beginning), 1),
                     "mean_score_100": recent_result["mean_score"],
                     "best_score_100": max((x["score"] for x in recent), default=0),
                     "highest_level_100": max((x["level"] for x in recent), default=1),
                     "complete_games_100": recent_result["complete_games"],
                     "incomplete_games_100": recent_result["incomplete_games"],
                     "level_reach_counts_100": recent_result["level_reach_counts"],
                     "level_reach_rates_100": recent_result["level_reach_rates"],
                     "actor_loss": float(np.mean(metrics, axis=0)[0]),
                     "value_loss": float(np.mean(metrics, axis=0)[1]),
                     "entropy": float(np.mean(metrics, axis=0)[2]),
                     "approx_kl": float(np.mean(metrics, axis=0)[3]),
                     **({"reference_kl": float(np.mean(metrics, axis=0)[4])}
                        if reference is not None else {}),
                     "rollout_action_origins": rollout_origins,
                     "training_action_origins": action_origins, **memory_metrics(), **sil_metrics})
                last_log = now
            if steps >= next_eval:
                directory = args.run/f"step-{steps:09d}"
                agent.save(directory, state())
                log({"event": "validation_start", "steps": steps,
                     "games": args.eval_games, "seed": args.eval_seed, "eval_envs": args.eval_envs})
                result = evaluate(agent.policy(), list(range(args.eval_seed, args.eval_seed+args.eval_games)),
                                  tstates=args.tstates, max_steps=args.eval_max_steps, verbose=True,
                                  envs=args.eval_envs, should_stop=lambda: stop,
                                  observation_stride=args.observation_stride,
                                  progress_callback=lambda row: log({"steps": steps, **row}))
                result["policy"] = "learned categorical, sampled"
                write_json(directory/"evaluation.json", result)
                log({"event": "validation", "steps": steps,
                     **{k: v for k, v in result.items() if k != "games"}})
                best, best_levels, improved_mean, improved_levels = update_selection(
                    result, args.target_level, best, best_levels, game_win=args.target_game_win)
                # Weights were saved before validation for crash recovery. Now
                # attach the records measured at this checkpoint, so resuming
                # it does not inherit the previous checkpoint's ranking.
                write_json(directory/"state.json", state())
                if improved_mean:
                    agent.save(args.run/"best", state())
                    write_json(args.run/"best"/"evaluation.json", result)
                if improved_levels:
                    agent.save(args.run/"best-level", state())
                    write_json(args.run/"best-level"/"evaluation.json", result)
                agent.save(args.run/"latest", state())
                next_eval += args.eval_every
                if level_target_met(result, args.target_level, args.target_clears,
                                    game_win=args.target_game_win):
                    target_met = True
                    if args.stop_on_target:
                        break
    except EvaluationCancelled as error:
        log({"event": "validation_cancelled", "steps": steps, "reason": str(error)})
    except BaseException as error:
        log({"event": "error", "steps": steps, "error": str(error)})
        raise
    finally:
        agent.save(args.run/"latest", state())
        log({"event": "stopped", "steps": steps, "target_met": target_met,
             "training_action_origins": action_origins, **memory_metrics(),
             **({"sil":{**sil_collector.metrics(),"updates":sil.updates}} if sil is not None else {})})
        envs.close()
        log_file.close()


if __name__ == "__main__":
    main()
