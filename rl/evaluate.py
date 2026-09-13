"""Play fixed-seed complete games with a saved neural policy, or watch it."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .env import BreakdownEnv


def checkpoint_config(checkpoint):
    state_path = Path(checkpoint).parent/"state.json"
    return json.loads(state_path.read_text())["config"] if state_path.exists() else {}


def policy_description(config, deterministic=False):
    if config.get("algorithm") == "ppo" and not deterministic:
        return "learned categorical, sampled"
    return "learned action values/logits, argmax"


def summary(games):
    scores = [g["score"] for g in games if g["terminated"]]
    return {
        "games_requested": len(games), "complete_games": len(scores),
        "incomplete_games": len(games)-len(scores),
        "mean_score": float(np.mean(scores)) if scores else None,
        "median_score": float(np.median(scores)) if scores else None,
        "best_score": max(scores) if scores else None,
        "highest_level": max((g["level"] for g in games), default=1),
        "level_1_clears": sum(g["level"] >= 2 and g["terminated"] for g in games),
    }


def evaluate(policy, seeds, *, tstates=100_000, max_steps=100_000, verbose=False,
             epsilon=0.0):
    # Evaluation uses its own emulator process in training (the C core is global).
    env = BreakdownEnv(tstates=tstates, max_steps=max_steps)
    games = []
    try:
        for seed in seeds:
            obs = env.reset(seed)
            if hasattr(policy, "reset_seed"):
                policy.reset_seed(seed+1_000_000)
            rng = np.random.default_rng(seed+1_000_000)
            while True:
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
    finally:
        env.close()
    return {**summary(games), "epsilon": epsilon, "tstates": tstates,
            "max_steps": max_steps, "games": games}


def load_policy(checkpoint, deterministic=False):
    import mlx.core as mx
    from .model import QNetwork

    model = QNetwork()
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    algorithm = checkpoint_config(checkpoint).get("algorithm", "dqn")
    if algorithm == "ppo" and not deterministic:
        infer_logits = mx.compile(model.policy_value, inputs=model.state)
        rng = np.random.default_rng(0)

        def sample(obs):
            logits, _ = infer_logits(mx.array(obs))
            logits = np.array(logits)
            probs = np.exp(logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True))
            return (rng.random(len(obs))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 5)

        def reset_seed(seed):
            nonlocal rng
            rng = np.random.default_rng(seed)

        sample.reset_seed = reset_seed
        return sample
    infer = mx.compile(lambda x: mx.argmax(model(x), axis=1), inputs=model.state)
    return lambda obs: np.array(infer(mx.array(obs)))


def watch(policy, seed, tstates, speed):
    # Drive the existing display on its UI thread, without its independent CPU
    # thread. Each emulation advance is controlled by the model's chosen action.
    import pyglet
    from trs.video import Video

    env = BreakdownEnv(seed=seed, tstates=tstates, max_steps=0)
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
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--tstates", type=int, help="defaults to the checkpoint's recorded action duration")
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
    if args.games < 1:
        parser.error("--games must be positive")
    if args.random:
        rng = np.random.default_rng(args.seed)
        policy = lambda obs: rng.integers(6, size=len(obs))
    elif args.checkpoint is not None:
        policy = load_policy(args.checkpoint, deterministic=args.deterministic)
    else:
        parser.error("provide a checkpoint or --random")
    if args.watch:
        watch(policy, args.seed, args.tstates, args.speed)
        return
    result = evaluate(policy, list(range(args.seed, args.seed+args.games)),
                      tstates=args.tstates, max_steps=args.max_steps, verbose=True,
                      epsilon=args.epsilon)
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
