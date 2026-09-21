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

from .defense import action_names, screen_info, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_learning import evaluate, publish_best, sha256, summarize, write_json
from .ppo import PPO, gae
from .vector import VectorEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("results/defense/learned"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=0, help="absolute action limit; 0 = unlimited")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy", type=float, default=.02)
    parser.add_argument("--gamma", type=float, default=.997)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--reward-scale", type=float, default=.01,
                        help="constant units conversion; no shaping or clipping")
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-enter", action=argparse.BooleanOptionalAction, default=False,
                        help="add Enter as a learned action; never automatically skip an intro")
    parser.add_argument("--tstates", type=int, default=100_000)
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
                        help="archive an own-play state this many actions before score progress; 0 disables")
    parser.add_argument("--curriculum-share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-boot-envs", type=int, default=0)
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
        if (config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
                or config.get("environment_version") != ENVIRONMENT_VERSION
                or config.get("action_names") != list(action_names(config.get("allow_enter", False)))):
            parser.error("Resume requires a compatible Defense checkpoint")
        if args.allow_enter != config.get("allow_enter", False):
            parser.error("Changing action profile requires a fresh run, not an incompatible optimizer resume")
        if not (args.resume/"optimizer.npz").exists():
            parser.error("Resume requires an original learner checkpoint with optimizer state")
    if min(args.envs, args.rollout, args.batch_size, args.epochs, args.eval_every,
           args.eval_games, args.eval_envs) < 1:
        parser.error("counts and intervals must be positive")
    if args.envs*args.rollout % args.batch_size:
        parser.error("envs * rollout must be divisible by batch-size")
    if min(args.steps, args.max_episode_steps, args.eval_max_steps, args.mlx_cache_mb) < 0:
        parser.error("limits must be nonnegative")
    if not 1 <= args.tstates <= 1_000_000:
        parser.error("tstates out of range")
    if (not np.isfinite(args.curriculum_probability) or not 0 <= args.curriculum_probability <= 1
            or min(args.curriculum_score_interval, args.curriculum_lookback) < 0
            or min(args.curriculum_per_bin, args.curriculum_bins) < 1
            or not 0 <= args.curriculum_boot_envs < args.envs
            or (args.curriculum_share and not args.curriculum_probability)
            or (args.curriculum_boot_envs and not args.curriculum_share)):
        parser.error("invalid own-experience curriculum settings")
    if (args.sil_updates < 0 or min(args.sil_capacity, args.sil_suffix_steps, args.sil_batch_size) < 1
            or not np.isfinite(args.sil_loss_weight) or args.sil_loss_weight <= 0
            or not np.isfinite(args.sil_value_weight) or args.sil_value_weight < 0):
        parser.error("invalid own-experience self-imitation settings")
    if (not all(np.isfinite(v) for v in (args.learning_rate, args.entropy, args.gamma,
                                        args.gae_lambda, args.reward_scale))
            or args.learning_rate <= 0 or args.entropy < 0 or args.reward_scale <= 0
            or not 0 < args.gamma < 1 or not 0 < args.gae_lambda <= 1):
        parser.error("invalid optimizer/return parameters")
    if args.run.exists() and not args.resume:
        parser.error("run already exists; choose a new directory or --resume")
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    mx.set_cache_limit(args.mlx_cache_mb*1024*1024)
    agent = PPO(seed=args.seed, learning_rate=args.learning_rate, entropy=args.entropy,
                action_count=len(action_names(args.allow_enter)))
    rng = np.random.default_rng(args.seed)
    steps, episodes = 0, 0
    if prior:
        agent.model.load_weights(str(args.resume/"model.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        agent.compile()
        rng.bit_generator.state = prior["rng"]
        steps, episodes = prior["steps"], prior["episodes"]
    else:
        # Broad initial exploration; no preference for a hand-selected action.
        agent.model.advantage.weight *= .1
        agent.model.advantage.bias *= .1
        agent.compile()
    sil = None
    if args.sil_updates:
        from .sil import SILReplay, TrainingSuffixes, SelfImitation
        sil_replay = SILReplay(args.sil_capacity)
        sil_collector = TrainingSuffixes(sil_replay, args.envs, args.gamma, args.sil_suffix_steps,
                                         action_count=len(action_names(args.allow_enter)), score_reader=screen_info)
        sil = SelfImitation(agent, args.sil_loss_weight, args.sil_value_weight)
        sil_rng = np.random.default_rng(np.random.SeedSequence([args.seed, steps, 941]))
        if prior and "sil_rng" in prior:
            sil_rng.bit_generator.state = prior["sil_rng"]
    boot_episodes = prior.get("boot_episodes", episodes) if prior else 0
    restored_segments = prior.get("restored_segments", 0) if prior else 0
    args.run.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(game="defense", algorithm="ppo", game_sha256=GAME_SHA256,
                  native_sha256=sha256("libtrs.so"), environment_version=ENVIRONMENT_VERSION,
                  environment_source_sha256=sha256(Path(__file__).with_name("defense.py")),
                  action_names=list(action_names(args.allow_enter)), observation="four raw 16x64 video-memory frames",
                  reward="visible score difference only, constant scale for optimizer",
                  policy="learned categorical, sampled", mlx=mx.__version__,
                  resume_semantics="optimizer and policy RNG restored; emulator episodes restart from boot")
    if args.curriculum_probability:
        config.update(curriculum_archive_saved=False,
                      curriculum_source_sha256=sha256(Path(__file__).with_name("defense_curriculum.py")),
                      snapshot_source_sha256=sha256(Path(__file__).with_name("defense_snapshot.py")))
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
                              curriculum_share=args.curriculum_share,
                              curriculum_boot_envs=args.curriculum_boot_envs)
        workers = VectorEnv(args.envs, args.seed+steps, game="defense", tstates=args.tstates,
                            max_steps=args.max_episode_steps, allow_enter=args.allow_enter, **curriculum)
        obs = workers.observations
        log(dict(event="workers_started", workers=workers.runtime()))
        agent.save(args.run/"latest", state())
        while not stop and (not args.steps or steps < args.steps):
            screens, actions_buffer, logps, values_buffer, rewards, boundaries = [], [], [], [], [], []
            for _ in range(args.rollout):
                actions, logp, values = agent.act(obs, rng)
                screens.append(obs)
                actions_buffer.append(actions)
                logps.append(logp)
                values_buffer.append(values)
                next_obs, reward_row, boundary_row = [], [], []
                for worker, result in enumerate(workers.step(actions)):
                    frame, reward, terminal, truncated, info, reset = result
                    reward *= args.reward_scale
                    learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                    if sil is not None:
                        # Only this learner's own screens, selected actions and
                        # score returns; never evaluation/replay-file examples.
                        segment = sil_collector.append(worker, obs[worker], actions[worker], reward,
                                                       learning_terminal, truncated, info.get("full_game", True))
                        if segment is not None:
                            log(dict(event="sil_segment", action_counter=steps+worker+1, **segment))
                    if truncated and not learning_terminal:
                        _, value = agent.predict(mx.array(frame[None]))
                        reward += args.gamma*float(value[0].item())
                    reward_row.append(reward)
                    boundary_row.append(learning_terminal or truncated)
                    next_obs.append(reset if reset is not None else frame)
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
                                 episode=episodes, **info))
                rewards.append(reward_row)
                boundaries.append(boundary_row)
                obs = np.stack(next_obs)
                steps += args.envs
            _, last_value = agent.predict(mx.array(obs))
            advantages, returns = gae(np.asarray(rewards, np.float32), np.asarray(values_buffer),
                                      np.asarray(boundaries, np.float32), np.array(last_value),
                                      args.gamma, args.gae_lambda)
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            data = (np.concatenate(screens), np.concatenate(actions_buffer), np.concatenate(logps),
                    advantages.reshape(-1), returns.reshape(-1))
            metrics = []
            for _ in range(args.epochs):
                order = rng.permutation(len(data[0]))
                epoch_metrics = []
                for start in range(0, len(order), args.batch_size):
                    indices = order[start:start+args.batch_size]
                    loss, aux = agent.update(*(mx.array(x[indices]) for x in data))
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
                         **sil_metrics))
                last_log = time.monotonic()
            if steps >= next_eval and not stop:
                directory = args.run/f"step-{steps:012d}"
                if directory.exists():
                    raise RuntimeError("Refusing to overwrite a historical checkpoint")
                agent.save(directory, state())
                agent.save(args.run/"latest", state())
                log(dict(event="validation_start", steps=steps, checkpoint=str(directory)))
                result = evaluate(agent.policy(), range(args.eval_seed, args.eval_seed+args.eval_games),
                                  tstates=args.tstates, max_steps=args.eval_max_steps, envs=args.eval_envs,
                                  log=log, should_stop=lambda: stop, allow_enter=args.allow_enter)
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
