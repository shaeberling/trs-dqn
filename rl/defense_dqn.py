"""Independent screen-only Defense Double DQN with prioritized own-experience replay.

No demonstrations, PPO trajectories, state resets, reward shaping or action
controllers. Evaluation is greedy and always starts at the original game boot.
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

from .defense import action_names, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import (DQN_ALGORITHM, evaluate, greedy_policy, publish_best,
                               sha256, summarize, write_json)
from .replay import NStep, Replay
from .vector import VectorEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=0, help="absolute action limit; 0 = unlimited")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=.01)
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-enter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=16,
                        help="new aggregate actions per optimizer update")
    parser.add_argument("--epsilon-steps", type=int, default=1_000_000)
    parser.add_argument("--epsilon-final", type=float, default=.05)
    parser.add_argument("--target-every", type=int, default=2000,
                        help="optimizer updates per target-network copy")
    parser.add_argument("--tstates", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-max-steps", type=int, default=0)
    parser.add_argument("--mlx-cache-mb", type=int, default=512)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            if (hasattr(args, key) and key not in ("run", "resume", "artifacts", "steps")
                    and "--"+key.replace("_", "-") not in explicit
                    and "--no-"+key.replace("_", "-") not in explicit):
                setattr(args, key, value)
        config = prior["config"]
        if (config.get("algorithm") != DQN_ALGORITHM or config.get("game") != "defense"
                or config.get("game_sha256") != GAME_SHA256
                or config.get("environment_version") != ENVIRONMENT_VERSION
                or config.get("action_names") != list(action_names(args.allow_enter))):
            parser.error("Resume requires a compatible Defense DQN checkpoint")
        if not all((args.resume/name).is_file() for name in
                   ("model.safetensors", "target.safetensors", "optimizer.npz")):
            parser.error("DQN resume requires online, target and optimizer checkpoints")
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
    if args.run.exists() and not args.resume:
        parser.error("run already exists; choose a new directory or --resume")
    if args.artifacts.resolve() == args.run.resolve():
        parser.error("checkpoint and replay-artifact roots must be separate")

    # MLX is main-process-only: spawned emulator workers never import it.
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from .model import Learner
    from .train import checkpoint
    mx.set_cache_limit(args.mlx_cache_mb*1024*1024)
    agent = Learner(args.learning_rate, args.seed, action_count=len(action_names(args.allow_enter)))
    rng = np.random.default_rng(args.seed)
    steps, updates, episodes = 0, 0, 0
    if prior:
        agent.online.load_weights(str(args.resume/"model.safetensors"))
        agent.target.load_weights(str(args.resume/"target.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        agent.state = [agent.online.state, agent.target.state, agent.optimizer.state]
        agent.update = mx.compile(agent._update, inputs=agent.state, outputs=agent.state)
        rng.bit_generator.state = prior["rng"]
        steps, updates, episodes = (prior[k] for k in ("steps", "updates", "episodes"))
    mx.eval(agent.state)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(game="defense", algorithm=DQN_ALGORITHM, game_sha256=GAME_SHA256,
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
    args.run.mkdir(parents=True, exist_ok=True)
    write_json(args.run/("resume-config.json" if prior else "config.json"), config)
    replay = Replay(args.capacity)
    buffers = [NStep(replay, args.n_step, args.gamma) for _ in range(args.envs)]
    recent = deque(maxlen=100)
    started, start_steps = time.monotonic(), steps
    last_log = started
    next_eval, next_update = steps+args.eval_every, steps+args.train_every
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)
    stop, workers = False, None

    def state():
        return dict(steps=steps, updates=updates, episodes=episodes,
                    rng=rng.bit_generator.state, config=config)

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
                            max_steps=args.max_episode_steps, allow_enter=args.allow_enter)
        log(dict(event="workers_started", workers=workers.runtime()))
        observations = workers.observations
        last_metrics = {}
        while not stop and (not args.steps or steps < args.steps):
            epsilon = max(args.epsilon_final, 1-(1-args.epsilon_final)*
                          max(0, steps-args.warmup)/args.epsilon_steps)
            if replay.size < max(1, args.warmup):
                actions = rng.integers(len(config["action_names"]), size=args.envs)
            else:
                actions = agent.actions(observations)
                explore = rng.random(args.envs) < epsilon
                actions[explore] = rng.integers(len(config["action_names"]), size=int(explore.sum()))
            results = workers.step(actions)
            following = []
            for worker, (obs, reward, terminal, truncated, info, reset) in enumerate(results):
                learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                buffers[worker].append(observations[worker], int(actions[worker]),
                                       reward*args.reward_scale, obs, learning_terminal, truncated)
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
            if time.monotonic()-last_log >= 10:
                log(dict(event="progress", steps=steps, episodes=episodes, updates=updates,
                         replay_size=replay.size, epsilon=epsilon,
                         steps_per_second=(steps-start_steps)/(time.monotonic()-started),
                         recent={k: v for k, v in summarize(list(recent)).items() if k != "games"},
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
                policy = greedy_policy(lambda obs: np.array(agent.online(mx.array(obs))))
                result = evaluate(policy, range(args.eval_seed, args.eval_seed+args.eval_games),
                                  tstates=args.tstates, max_steps=args.eval_max_steps,
                                  envs=args.eval_envs, log=log, should_stop=lambda: stop,
                                  allow_enter=args.allow_enter)
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
