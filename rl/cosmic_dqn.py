"""Fresh, screen-only Cosmic Fighter DQN; score is the only reward.

Runs are bounded by default. SIGTERM saves a full optimizer checkpoint. Every
published replay is reexecuted with frozen weights from the original boot.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .cosmic import ACTION_NAMES, ENVIRONMENT_VERSION, GAME_SHA256
from .cosmic_learning import evaluate, load_policy, publish_best, summarize, write_json
from .model import Learner
from .replay import NStep, Replay
from .vector import VectorEnv


def checkpoint(learner, directory, saved):
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in (("model", learner.online), ("target", learner.target)):
        temporary = directory/f"{name}.tmp.safetensors"
        model.save_weights(str(temporary))
        temporary.replace(directory/f"{name}.safetensors")
    temporary = directory/"optimizer.tmp.npz"
    mx.savez(str(temporary), **dict(tree_flatten(learner.optimizer.state)))
    temporary.replace(directory/"optimizer.npz")
    write_json(directory/"state.json", saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=1_048_576,
                        help="absolute aggregate-action budget")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=.01,
                        help="constant optimizer units; game score remains the sole reward")
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=16)
    parser.add_argument("--target-every", type=int, default=2000)
    parser.add_argument("--epsilon-steps", type=int, default=500_000)
    parser.add_argument("--epsilon-final", type=float, default=.1)
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--observation-stride", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=262_144)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=70_000)
    parser.add_argument("--eval-max-steps", type=int, default=5_000)
    parser.add_argument("--mlx-cache-mb", type=int, default=512)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        original = prior["config"]
        if (original.get("game") != "cosmic" or original.get("game_sha256") != GAME_SHA256
                or original.get("environment_version") != ENVIRONMENT_VERSION
                or original.get("action_names") != list(ACTION_NAMES)):
            parser.error("resume requires a compatible Cosmic Fighter checkpoint")
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in original.items():
            if (hasattr(args, key) and key not in ("run", "artifacts", "resume", "steps")
                    and "--"+key.replace("_", "-") not in explicit):
                setattr(args, key, value)
        if not all((args.resume/name).is_file() for name in
                   ("model.safetensors", "target.safetensors", "optimizer.npz")):
            parser.error("resume requires online, target and optimizer files")
    if (min(args.steps, args.envs, args.batch_size, args.capacity, args.n_step,
            args.warmup, args.train_every, args.target_every, args.epsilon_steps,
            args.eval_every, args.eval_games, args.eval_envs, args.mlx_cache_mb) < 1
            or not 0 <= args.epsilon_final <= 1 or not 0 < args.gamma <= 1
            or not 0 < args.reward_scale <= 1 or not 1 <= args.tstates <= 1_000_000
            or args.observation_stride < 1 or args.max_episode_steps < 1
            or args.eval_max_steps < 1 or args.capacity < args.warmup
            or args.steps % args.envs):
        parser.error("invalid Cosmic DQN configuration")
    if args.run.exists():
        parser.error("run directory already exists; choose a new path")
    if args.artifacts.exists():
        parser.error("artifact directory already exists; choose a new path")
    if prior and args.steps <= prior["steps"]:
        parser.error("resume step limit must exceed checkpoint steps")
    args.run.mkdir(parents=True)
    args.artifacts.mkdir(parents=True)
    mx.set_cache_limit(args.mlx_cache_mb*1024*1024)
    learner = Learner(args.learning_rate, args.seed, action_count=len(ACTION_NAMES))
    rng = np.random.default_rng(args.seed)
    steps, updates, episodes, best_mean = 0, 0, 0, None
    if prior:
        learner.online.load_weights(str(args.resume/"model.safetensors"))
        learner.target.load_weights(str(args.resume/"target.safetensors"))
        learner.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        learner.optimizer.learning_rate = args.learning_rate
        learner.state = [learner.online.state, learner.target.state, learner.optimizer.state]
        learner.update = mx.compile(learner._update, inputs=learner.state, outputs=learner.state)
        rng.bit_generator.state = prior["rng"]
        steps, updates, episodes, best_mean = (prior[key] for key in
                                               ("steps", "updates", "episodes", "best_mean"))
    mx.eval(learner.state)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(game="cosmic", algorithm="dueling-double-dqn-per-nstep",
                  game_sha256=GAME_SHA256, environment_version=ENVIRONMENT_VERSION,
                  action_names=list(ACTION_NAMES), observation="four raw 16x64 video-memory frames",
                  reward="displayed-score difference only; constant optimizer scale",
                  policy="learned Q-values, greedy; epsilon-random training only",
                  replay_resume="refill from new own experience; no trajectory import")
    write_json(args.run/"config.json", config)
    replay = Replay(args.capacity, compact=True)
    buffers = [NStep(replay, args.n_step, args.gamma) for _ in range(args.envs)]
    recent = deque(maxlen=100)
    workers = None
    start_time = time.monotonic()
    start_steps = steps
    last_log = start_time
    next_eval = (steps//args.eval_every+1)*args.eval_every
    next_update = steps+args.train_every
    stop = False
    last_metrics = {}
    metrics = (args.run/"metrics.jsonl").open("a", buffering=1)

    def state():
        return dict(steps=steps, updates=updates, episodes=episodes, best_mean=best_mean,
                    rng=rng.bit_generator.state, config=config)

    def log(row):
        row = dict(wall_seconds=round(time.monotonic()-start_time, 2), **row)
        metrics.write(json.dumps(row)+"\n")
        write_json(args.run/"status.json", dict(row, pid=os.getpid(),
                   training_steps=steps, updates=updates, updated_unix=time.time()))
        if row["event"] != "episode":
            print(json.dumps(row), flush=True)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(dict(event="start", steps=steps, config=config))
    try:
        workers = VectorEnv(args.envs, args.seed+steps, game="cosmic", tstates=args.tstates,
                            max_steps=args.max_episode_steps,
                            observation_stride=args.observation_stride)
        observations = workers.observations
        log(dict(event="workers_started", workers=workers.runtime()))
        while not stop and steps < args.steps:
            epsilon = max(args.epsilon_final, 1-(1-args.epsilon_final)*
                          max(0, steps-args.warmup)/args.epsilon_steps)
            if replay.size < args.warmup:
                actions = rng.integers(len(ACTION_NAMES), size=args.envs)
            else:
                actions = learner.actions(observations)
                explore = rng.random(args.envs) < epsilon
                actions[explore] = rng.integers(len(ACTION_NAMES), size=int(explore.sum()))
            results = workers.step(actions)
            following = []
            for worker, (obs, reward, terminal, truncated, info, reset) in enumerate(results):
                buffers[worker].append(observations[worker], int(actions[worker]),
                                       reward*args.reward_scale, obs,
                                       terminal or info["life_lost"], truncated)
                following.append(reset if reset is not None else obs)
                if terminal or truncated:
                    episodes += 1
                    recent.append(info)
                    log(dict(event="episode", worker=worker, action_counter=steps+worker+1,
                             episode=episodes, **info))
            observations = np.stack(following)
            steps += args.envs
            while steps >= next_update:
                next_update += args.train_every
                if replay.size < args.warmup:
                    continue
                beta = min(1., .4+.6*steps/args.epsilon_steps)
                indices, batch = replay.sample(args.batch_size, rng, beta)
                loss, errors, mean_q = learner.train(batch)
                if not np.isfinite([loss, mean_q]).all() or not np.isfinite(errors).all():
                    raise RuntimeError("Non-finite Cosmic DQN update")
                replay.priorities(indices, errors)
                updates += 1
                if updates % args.target_every == 0:
                    learner.sync_target()
                last_metrics = dict(loss=loss, mean_q=mean_q, priority_beta=beta)
            if time.monotonic()-last_log >= 10:
                log(dict(event="progress", steps=steps, updates=updates, episodes=episodes,
                         epsilon=epsilon, replay_size=replay.size,
                         steps_per_second=(steps-start_steps)/(time.monotonic()-start_time),
                         recent={key: value for key, value in summarize(list(recent)).items()
                                 if key != "games"},
                         replay_frame_storage=replay.frame_storage.stats(),
                         mlx_active_bytes=mx.get_active_memory(),
                         mlx_peak_bytes=mx.get_peak_memory(), **last_metrics))
                last_log = time.monotonic()
            if steps >= next_eval and not stop:
                directory = args.run/f"step-{steps:012d}"
                if directory.exists():
                    raise RuntimeError("Refusing to overwrite a Cosmic checkpoint")
                checkpoint(learner, directory, state())
                log(dict(event="validation_start", steps=steps, checkpoint=str(directory)))
                policy, _ = load_policy(directory/"model.safetensors")
                result = evaluate(policy, range(args.eval_seed, args.eval_seed+args.eval_games),
                                  tstates=args.tstates, max_steps=args.eval_max_steps,
                                  observation_stride=args.observation_stride, envs=args.eval_envs)
                write_json(directory/"evaluation.json", result)
                log(dict(event="validation", steps=steps,
                         **{key: value for key, value in result.items() if key != "games"}))
                if (not result["incomplete_games"] and result["mean_score"] is not None
                        and (best_mean is None or result["mean_score"] > best_mean)):
                    version = publish_best(directory/"model.safetensors", result, args.artifacts)
                    best_mean = result["mean_score"]
                    log(dict(event="best_replay_published", steps=steps, version=version,
                             mean_score=best_mean))
                next_eval += args.eval_every
    except BaseException as error:
        log(dict(event="error", steps=steps, error=repr(error)))
        raise
    finally:
        checkpoint(learner, args.run/"latest", state())
        if workers is not None:
            workers.close()
        log(dict(event="stopped", steps=steps, stop_requested=stop))
        metrics.close()


if __name__ == "__main__":
    main()
