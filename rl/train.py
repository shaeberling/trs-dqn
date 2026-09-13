"""Train a screen-only agent on the Mac Mini until validation clears level 1."""

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .evaluate import evaluate
from .model import Learner
from .replay import NStep, Replay
from .vector import VectorEnv


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def checkpoint(learner, directory, state):
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in (("model", learner.online), ("target", learner.target)):
        temp = directory/f"{name}.tmp.safetensors"
        model.save_weights(str(temp))
        temp.replace(directory/f"{name}.safetensors")
    mx.savez(str(directory/"optimizer.tmp.npz"), **dict(tree_flatten(learner.optimizer.state)))
    (directory/"optimizer.tmp.npz").replace(directory/"optimizer.npz")
    write_json(directory/"state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/mlx-dqn"))
    parser.add_argument("--resume", type=Path, help="checkpoint directory; replay refills from new experience")
    parser.add_argument("--steps", type=int, default=0, help="total steps; 0 = until validation target")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--life-terminal", action="store_true",
                        help="end a training return at each on-screen lost ball; evaluation still uses full games")
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--epsilon-steps", type=int, default=250_000)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--exploration-repeat", type=int, default=1,
                        help="maximum Zipf(2) duration for random training actions; 1 = ordinary epsilon greedy")
    parser.add_argument("--target-every", type=int, default=2000, help="optimizer updates per target sync")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=20_000)
    parser.add_argument("--target-clears", type=int, default=3)
    parser.add_argument("--tstates", type=int, default=100_000)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            if (hasattr(args, key) and key not in ("run", "resume", "steps")
                    and "--"+key.replace("_", "-") not in explicit):
                setattr(args, key, value)
    if min(args.envs, args.batch_size, args.capacity, args.train_every, args.eval_every,
           args.epsilon_steps, args.target_every, args.eval_games) <= 0:
        parser.error("counts and intervals must be positive")
    args.run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.run/"config.json").exists():
        parser.error("run exists: choose another --run or use --resume")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(algorithm="dueling-double-dqn-per-nstep", mlx=mx.__version__,
                  device=mx.device_info(), observation="four raw 16x64 video-memory frames",
                  reward="screen score difference only", replay_resume="refill, not exact trajectory continuation")
    write_json(args.run/("resume-config.json" if args.resume else "config.json"), config)
    learner = Learner(args.learning_rate, args.seed)
    rng = np.random.default_rng(args.seed)
    steps, updates, episodes, best = 0, 0, 0, -1.0
    if args.resume:
        learner.online.load_weights(str(args.resume/"model.safetensors"))
        learner.target.load_weights(str(args.resume/"target.safetensors"))
        learner.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        learner.optimizer.learning_rate = args.learning_rate
        # Rebind compiled captures after replacing optimizer state.
        learner.state = [learner.online.state, learner.target.state, learner.optimizer.state]
        learner.update = mx.compile(learner._update, inputs=learner.state, outputs=learner.state)
        rng.bit_generator.state = prior["rng"]
        steps, updates, episodes, best = (prior[k] for k in ("steps", "updates", "episodes", "best_mean"))
    mx.eval(learner.state)
    replay = Replay(args.capacity)
    buffers = [NStep(replay, args.n_step, args.gamma) for _ in range(args.envs)]
    envs = VectorEnv(args.envs, args.seed+steps, tstates=args.tstates)
    observations = envs.observations
    recent = deque(maxlen=100)
    exploration_left = np.zeros(args.envs, np.int32)
    exploration_actions = np.zeros(args.envs, np.int32)
    losses, qs = deque(maxlen=100), deque(maxlen=100)
    start_time, start_steps, last_log = time.monotonic(), steps, time.monotonic()
    next_eval = (steps//args.eval_every+1)*args.eval_every
    next_update = steps+args.train_every
    stop = False
    target_met = False
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)

    def log(event):
        event = {"wall_seconds": round(time.monotonic()-start_time, 2), **event}
        log_file.write(json.dumps(event)+"\n")
        if event["event"] != "episode":
            print(json.dumps(event), flush=True)

    def state():
        return dict(steps=steps, updates=updates, episodes=episodes, best_mean=best,
                    rng=rng.bit_generator.state, config=config)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log({"event": "start", "steps": steps, "config": config})
    try:
        while not stop and (not args.steps or steps < args.steps):
            fraction = max(0, steps-args.warmup)/args.epsilon_steps
            epsilon = max(args.epsilon_final, 1-(1-args.epsilon_final)*fraction)
            warming_up = replay.size < args.warmup
            if args.exploration_repeat == 1:
                # Preserve the baseline's random stream for reproduction.
                if warming_up:
                    actions = rng.integers(6, size=args.envs)
                else:
                    actions = learner.actions(observations)
                    explore = rng.random(args.envs) < epsilon
                    actions[explore] = rng.integers(6, size=int(explore.sum()))
            else:
                actions = np.zeros(args.envs, np.int32) if warming_up else learner.actions(observations)
                explore = (rng.random(args.envs) < (1.0 if warming_up else epsilon)) & (exploration_left == 0)
                count = int(explore.sum())
                exploration_actions[explore] = rng.integers(6, size=count)
                exploration_left[explore] = np.minimum(rng.zipf(2, size=count), args.exploration_repeat)
                active = exploration_left > 0
                actions[active] = exploration_actions[active]
                exploration_left[active] -= 1
            results = envs.step(actions)
            next_observations = []
            for i, (obs, reward, terminal, truncated, info, reset) in enumerate(results):
                learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                buffers[i].append(observations[i], int(actions[i]), reward, obs, learning_terminal, truncated)
                next_observations.append(reset if reset is not None else obs)
                if terminal or truncated:
                    exploration_left[i] = 0
                    episodes += 1
                    recent.append(info)
                    log({"event": "episode", **info, "episode_steps": info["steps"],
                         "steps": steps+i+1, "episode": episodes})
            observations = np.stack(next_observations)
            steps += args.envs
            if replay.size >= args.warmup:
                while next_update <= steps:
                    indices, batch = replay.sample(args.batch_size, rng, min(1, 0.4+0.6*steps/2_000_000))
                    loss, errors, q = learner.train(batch)
                    if not np.isfinite(loss) or not np.all(np.isfinite(errors)):
                        raise RuntimeError("Non-finite learner update")
                    replay.priorities(indices, errors)
                    losses.append(loss)
                    qs.append(q)
                    updates += 1
                    next_update += args.train_every
                    if updates % args.target_every == 0:
                        learner.sync_target()
            else:
                next_update = steps+args.train_every
            now = time.monotonic()
            if now-last_log >= 10:
                log({"event": "progress", "steps": steps, "updates": updates, "episodes": episodes,
                     "steps_per_second": round((steps-start_steps)/(now-start_time), 1),
                     "epsilon": round(epsilon, 4), "replay_size": replay.size,
                     "mean_score_100": float(np.mean([x["score"] for x in recent])) if recent else None,
                     "best_score_100": max((x["score"] for x in recent), default=0),
                     "highest_level_100": max((x["level"] for x in recent), default=1),
                     "loss": float(np.mean(losses)) if losses else None,
                     "mean_q": float(np.mean(qs)) if qs else None,
                     "gpu_memory_mb": round(mx.get_active_memory()/1e6, 1)})
                last_log = now
            if steps >= next_eval:
                directory = args.run/f"step-{steps:09d}"
                checkpoint(learner, directory, state())
                result = evaluate(learner.actions, list(range(10_000, 10_000+args.eval_games)),
                                  tstates=args.tstates, max_steps=args.eval_max_steps)
                write_json(directory/"evaluation.json", result)
                log({"event": "validation", "steps": steps,
                     **{k: v for k, v in result.items() if k != "games"}})
                mean = result["mean_score"]
                if result["incomplete_games"] == 0 and mean is not None and mean > best:
                    best = mean
                    checkpoint(learner, args.run/"best", state())
                    write_json(args.run/"best"/"evaluation.json", result)
                checkpoint(learner, args.run/"latest", state())
                next_eval += args.eval_every
                if result["incomplete_games"] == 0 and result["level_1_clears"] >= args.target_clears:
                    target_met = True
                    log({"event": "validation_target_met", "steps": steps})
                    break
    except BaseException as error:
        log({"event": "error", "steps": steps, "error": str(error)})
        raise
    finally:
        checkpoint(learner, args.run/"latest", state())
        log({"event": "stopped", "steps": steps, "target_met": target_met,
             "checkpoint": str(args.run/"latest")})
        envs.close()
        log_file.close()


if __name__ == "__main__":
    main()
