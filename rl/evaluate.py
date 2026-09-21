"""Play fixed-seed complete games with a saved neural policy, or watch it."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .env import BreakdownEnv, ENVIRONMENT_VERSION, validate_observation_stride
from .outcome import OUTCOME_VERSION, verified_win


class EvaluationCancelled(RuntimeError):
    """A requested cancellation produces no selectable evaluation result."""


def complete_game(game):
    return bool(game["terminated"] and not game.get("truncated", False)
                and game.get("full_game", True))


class EvaluationProgress:
    def __init__(self, verbose=False, callback=None):
        self.verbose, self.callback = verbose, callback
        self.started = time.monotonic()
        self.next_report = self.started+10

    def __call__(self, games, active):
        if not self.verbose and self.callback is None:
            return
        now = time.monotonic()
        if now < self.next_report:
            return
        self.next_report = now+10
        finished = [game for game in games if game is not None]
        row = dict(event="evaluation_progress", evaluation_seconds=round(now-self.started, 2),
                   finished_games=len(finished),
                   complete_games=sum(complete_game(game) for game in finished),
                   active_games=len(active),
                   active=[dict(seed=seed, **{key: info.get(key) for key in
                                             ("steps", "score", "level", "waiting")})
                           for seed, info in active])
        if self.callback is not None:
            self.callback(row)
        else:
            print(json.dumps(row), flush=True)


def checkpoint_config(checkpoint):
    state_path = Path(checkpoint).parent/"state.json"
    return json.loads(state_path.read_text())["config"] if state_path.exists() else {}


def policy_description(config, deterministic=False):
    if config.get("algorithm") == "ppo" and not deterministic:
        return "learned categorical, sampled"
    return "learned action values/logits, argmax"


def summary(games):
    complete = [g for g in games if complete_game(g)]
    scores = [g["score"] for g in complete]
    highest = max((g["level"] for g in games), default=1)
    reach_counts = {str(level): sum(g["level"] >= level for g in complete)
                    for level in range(2, max(5, highest)+1)}
    result = {
        "games_requested": len(games), "complete_games": len(scores),
        "incomplete_games": len(games)-len(scores),
        "mean_score": float(np.mean(scores)) if scores else None,
        "median_score": float(np.median(scores)) if scores else None,
        "best_score": max(scores) if scores else None,
        "highest_level": highest,
        "highest_complete_level": max((g["level"] for g in complete), default=1),
        "level_1_clears": reach_counts["2"],
        "level_reach_counts": reach_counts,
        "level_reach_rates": {level: count/len(complete) if complete else None
                              for level, count in reach_counts.items()},
    }
    # Historical records remain byte/schema comparable; missing outcome data
    # is unknown, never a retroactively inferred loss or victory.
    if any('outcome_version' in g for g in games):
        assessed = [g for g in complete if g.get('outcome_version') == OUTCOME_VERSION]
        wins = sum(verified_win(g) for g in assessed)
        result.update(outcome_version=OUTCOME_VERSION, verified_wins=wins,
                      verified_win_rate=wins/len(complete) if complete else None,
                      outcome_assessed_games=len(assessed),
                      unverified_final_level_games=sum(g.get('win_status') == 'unverified_final_level'
                                                       for g in assessed),
                      missing_outcome_games=len(complete)-len(assessed))
    return result


def level_rank(result, target_level, *, game_win=False):
    """Rank complete validation suites by deeper-level consistency, then score."""
    if result["incomplete_games"] or not result["complete_games"]:
        return None
    return ((result.get('verified_wins', 0),) if game_win else ()) + tuple(result["level_reach_counts"].get(str(level), 0)
                 for level in range(target_level, 1, -1)) + (result["mean_score"],)


def level_target_met(result, target_level, target_clears, *, game_win=False):
    return (not result["incomplete_games"] and result["complete_games"] > 0
            and (result.get('verified_wins', 0) if game_win else
                 result["level_reach_counts"].get(str(target_level), 0)) >= target_clears)


def categorical_policy(infer_logits, seed=0):
    """Sample neural logits, optionally using one independent RNG per game."""
    rng = np.random.default_rng(seed)

    def choose(obs, uniforms):
        logits = np.asarray(infer_logits(obs))
        probs = np.exp(logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True))
        return (uniforms[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, logits.shape[1]-1)

    def sample(obs):
        return choose(obs, rng.random(len(obs)))

    def sample_with_rngs(obs, rngs):
        if len(obs) != len(rngs):
            raise ValueError("provide one policy RNG per observation")
        return choose(obs, np.array([generator.random() for generator in rngs]))

    def reset_seed(seed):
        nonlocal rng
        rng = np.random.default_rng(seed)

    sample.reset_seed = reset_seed
    sample.sample_with_rngs = sample_with_rngs
    return sample


def _parallel_games(policy, seeds, *, envs, tstates, max_steps, verbose, epsilon,
                    progress, should_stop, observation_stride):
    """Batch inference in the parent; isolate each native emulator in a worker.

    Policy must be stateless/deterministic or support sample_with_rngs. Games
    retain separate policy and epsilon RNG streams, independent of scheduling.
    """
    from .vector import VectorEnv

    if hasattr(policy, "reset_seed") and not hasattr(policy, "sample_with_rngs"):
        raise ValueError("parallel stochastic evaluation requires sample_with_rngs")
    count = min(envs, len(seeds))
    workers = VectorEnv(count, seed=seeds[0], tstates=tstates, max_steps=max_steps,
                        observation_stride=observation_stride)
    games = [None]*len(seeds)
    latest_info = [{"steps": 0} for _ in seeds]
    policy_rngs = [np.random.default_rng(seed+1_000_000) for seed in seeds]
    epsilon_rngs = [np.random.default_rng(seed+1_000_000) for seed in seeds]
    pending = iter(range(count, len(seeds)))
    try:
        observations = workers.reset(seeds[:count])
        active = {i: (i, obs) for i, obs in enumerate(observations)}
        while active:
            if should_stop is not None and should_stop():
                raise EvaluationCancelled("Evaluation cancelled before all requested games finished")
            indices = sorted(active)
            jobs = [active[i][0] for i in indices]
            observations = np.stack([active[i][1] for i in indices])
            if hasattr(policy, "sample_with_rngs"):
                actions = policy.sample_with_rngs(observations, [policy_rngs[j] for j in jobs])
            else:
                actions = policy(observations)
            actions = np.array(actions, copy=True)
            for slot, job in enumerate(jobs):
                rng = epsilon_rngs[job]
                if epsilon and rng.random() < epsilon:
                    actions[slot] = rng.integers(6)
            results = workers.step(actions, indices=indices)
            replacements = []
            for worker, job, result in zip(indices, jobs, results, strict=True):
                obs, _, terminal, truncated, info, _ = result
                latest_info[job] = info
                if terminal or truncated:
                    info["seed"] = seeds[job]
                    games[job] = info
                    if verbose:
                        print(json.dumps({"event": "evaluation_game", **info}), flush=True)
                    del active[worker]
                    next_job = next(pending, None)
                    if next_job is not None:
                        replacements.append((worker, next_job))
                else:
                    active[worker] = (job, obs)
            if replacements:
                observations = workers.reset([seeds[job] for _, job in replacements],
                                             indices=[worker for worker, _ in replacements])
                for (worker, job), obs in zip(replacements, observations, strict=True):
                    active[worker] = (job, obs)
            progress(games, [(seeds[job], latest_info[job]) for job, _ in active.values()])
    finally:
        workers.close()
    return games


def evaluate(policy, seeds, *, tstates=100_000, max_steps=100_000, verbose=False,
             epsilon=0.0, envs=1, progress_callback=None, should_stop=None,
             observation_stride=1):
    observation_stride = validate_observation_stride(observation_stride)
    seeds = list(seeds)
    if envs < 1 or not seeds:
        raise ValueError("provide positive envs and at least one seed")
    progress = EvaluationProgress(verbose, progress_callback)
    if envs > 1:
        games = _parallel_games(policy, seeds, envs=envs, tstates=tstates,
                                max_steps=max_steps, verbose=verbose, epsilon=epsilon,
                                progress=progress, should_stop=should_stop,
                                observation_stride=observation_stride)
        return {**summary(games), "epsilon": epsilon, "tstates": tstates,
                "max_steps": max_steps, "environment_version": ENVIRONMENT_VERSION,
                "observation_stride": observation_stride, "eval_envs": envs, "games": games}
    # Evaluation uses its own emulator process in training (the C core is global).
    env = BreakdownEnv(tstates=tstates, max_steps=max_steps, observation_stride=observation_stride)
    games = []
    try:
        for seed in seeds:
            obs = env.reset(seed)
            if hasattr(policy, "reset_seed"):
                policy.reset_seed(seed+1_000_000)
            rng = np.random.default_rng(seed+1_000_000)
            while True:
                if should_stop is not None and should_stop():
                    raise EvaluationCancelled("Evaluation cancelled before all requested games finished")
                action = int(policy(obs[None])[0])
                if epsilon and rng.random() < epsilon:
                    action = int(rng.integers(6))
                obs, reward, terminal, truncated, info = env.step(action)
                if terminal or truncated:
                    info["seed"] = seed
                    games.append(info)
                    if verbose:
                        print(json.dumps({"event": "evaluation_game", **info}), flush=True)
                    break
                progress(games, [(seed, info)])
    finally:
        env.close()
    return {**summary(games), "epsilon": epsilon, "tstates": tstates,
            "max_steps": max_steps, "environment_version": ENVIRONMENT_VERSION,
            "observation_stride": observation_stride, "eval_envs": envs, "games": games}


def load_policy(checkpoint, deterministic=False):
    import mlx.core as mx
    from .model import QNetwork

    model = QNetwork()
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    algorithm = checkpoint_config(checkpoint).get("algorithm", "dqn")
    if algorithm == "ppo" and not deterministic:
        infer_logits = mx.compile(model.policy_value, inputs=model.state)
        return categorical_policy(lambda obs: np.array(infer_logits(mx.array(obs))[0]))
    infer = mx.compile(lambda x: mx.argmax(model(x), axis=1), inputs=model.state)
    return lambda obs: np.array(infer(mx.array(obs)))


def watch(policy, seed, tstates, speed, observation_stride=1):
    # Drive the existing display on its UI thread, without its independent CPU
    # thread. Each emulation advance is controlled by the model's chosen action.
    import pyglet
    from trs.video import Video

    env = BreakdownEnv(seed=seed, tstates=tstates, max_steps=0,
                       observation_stride=observation_stride)
    obs = env.reset(seed)
    if hasattr(policy, "reset_seed"):
        policy.reset_seed(seed+1_000_000)
    window = Video(env.trs.ram, env.trs.keyboard, 30)
    done_until = 0.0

    def tick(dt):
        nonlocal obs, done_until
        if done_until:
            if time.monotonic() >= done_until:
                obs = env.reset()
                done_until = 0.0
            return
        action = int(policy(obs[None])[0])
        obs, _, terminal, _, info = env.step(action)
        window.set_caption(f"Breakdown — trained policy — score {info['score']} — level {info['level']}")
        if terminal:
            print(json.dumps(info), flush=True)
            done_until = time.monotonic()+2

    pyglet.clock.schedule_interval(tick, tstates/1_774_080/speed)
    try:
        window.mainloop()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--envs", type=int, default=1,
                        help="parallel evaluation workers; each game keeps its own policy RNG")
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--tstates", type=int, help="defaults to the checkpoint's recorded action duration")
    parser.add_argument("--observation-stride", type=int,
                        help="action spacing between input frames; defaults to checkpoint or 1")
    parser.add_argument("--max-steps", type=int, default=100_000,
                        help="0 = no truncation; truncated games are never reported as complete")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="use argmax even for a categorical PPO policy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    config = checkpoint_config(args.checkpoint) if args.checkpoint is not None else {}
    if args.tstates is None:
        args.tstates = config.get("tstates", 100_000)
    try:
        args.observation_stride = validate_observation_stride(
            config.get("observation_stride", 1) if args.observation_stride is None
            else args.observation_stride)
    except ValueError as error:
        parser.error(str(error))
    if min(args.games, args.envs) < 1:
        parser.error("--games and --envs must be positive")
    if args.envs != 1 and (args.random or args.watch):
        parser.error("--random and --watch require --envs 1 (preserves the original RNG protocol)")
    if args.random:
        rng = np.random.default_rng(args.seed)
        policy = lambda obs: rng.integers(6, size=len(obs))
    elif args.checkpoint is not None:
        policy = load_policy(args.checkpoint, deterministic=args.deterministic)
    else:
        parser.error("provide a checkpoint or --random")
    if args.watch:
        watch(policy, args.seed, args.tstates, args.speed, args.observation_stride)
        return
    result = evaluate(policy, list(range(args.seed, args.seed+args.games)),
                      tstates=args.tstates, max_steps=args.max_steps, verbose=True,
                      epsilon=args.epsilon, envs=args.envs,
                      observation_stride=args.observation_stride)
    result["checkpoint"] = str(args.checkpoint)
    result["deterministic_override"] = args.deterministic
    result["policy"] = "uniform random" if args.random else policy_description(config, args.deterministic)
    if args.checkpoint is not None:
        result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    if result["incomplete_games"]:
        raise SystemExit("Evaluation incomplete: increase --max-steps or improve the policy")


if __name__ == "__main__":
    main()
