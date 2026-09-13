"""PPO with a screen-only categorical policy, optionally initialized by own RL."""

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .evaluate import evaluate
from .model import QNetwork
from .train import write_json
from .vector import VectorEnv


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


class PPO:
    def __init__(self, seed=17, learning_rate=2.5e-4, entropy=0.01):
        mx.random.seed(seed)
        self.model = QNetwork()
        self.optimizer = optim.Adam(learning_rate, eps=1e-5)
        self.optimizer.init(self.model.trainable_parameters())
        self.entropy = entropy
        self.compile()

    def compile(self):
        self.state = [self.model.state, self.optimizer.state]
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)
        self.predict = mx.compile(self.model.policy_value, inputs=self.model.state)
        mx.eval(self.state)

    def _loss(self, model, obs, actions, old_logp, advantages, returns):
        logits, values = model.policy_value(obs)
        log_probs = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        logp = mx.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
        logratio = logp-old_logp
        ratio = mx.exp(logratio)
        actor = -mx.mean(mx.minimum(ratio*advantages, mx.clip(ratio, .8, 1.2)*advantages))
        critic = .5*mx.mean(mx.square(values-returns))
        entropy = -mx.mean(mx.sum(mx.exp(log_probs)*log_probs, axis=-1))
        kl = mx.mean((ratio-1)-logratio)
        return actor+.5*critic-self.entropy*entropy, (actor, critic, entropy, kl)

    def _update(self, obs, actions, old_logp, advantages, returns):
        (loss, metrics), grads = nn.value_and_grad(self.model, self._loss)(
            self.model, obs, actions, old_logp, advantages, returns)
        grads, norm = optim.clip_grad_norm(grads, .5)
        self.optimizer.update(self.model, grads)
        return loss, metrics

    def act(self, obs, rng):
        logits, values = self.predict(mx.array(obs))
        logits, values = np.array(logits), np.array(values)
        logp = logits-np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        probs = np.exp(logp)
        actions = (rng.random(len(obs))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 5)
        return actions.astype(np.int32), logp[np.arange(len(obs)), actions], values

    def policy(self, seed=0):
        rng = np.random.default_rng(seed)

        def sample(obs):
            return self.act(obs, rng)[0]

        def reset_seed(seed):
            nonlocal rng
            rng = np.random.default_rng(seed)

        sample.reset_seed = reset_seed
        return sample

    def save(self, directory, state):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(directory/"model.tmp.safetensors"))
        (directory/"model.tmp.safetensors").replace(directory/"model.safetensors")
        mx.savez(str(directory/"optimizer.tmp.npz"), **dict(tree_flatten(self.optimizer.state)))
        (directory/"optimizer.tmp.npz").replace(directory/"optimizer.npz")
        write_json(directory/"state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--initialize", type=Path, help="weights from this agent's own earlier RL")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--logit-scale", type=float, default=10)
    parser.add_argument("--life-terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--target-clears", type=int, default=3)
    parser.add_argument("--tstates", type=int, default=100_000)
    args = parser.parse_args()
    prior = None
    if args.resume:
        prior = json.loads((args.resume/"state.json").read_text())
        explicit = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")}
        for key, value in prior["config"].items():
            flag = key.replace("_", "-")
            if (hasattr(args, key) and key not in ("run", "resume", "steps", "initialize")
                    and "--"+flag not in explicit and "--no-"+flag not in explicit):
                setattr(args, key, value)
    if min(args.envs, args.rollout, args.batch_size, args.epochs, args.eval_every, args.eval_games) <= 0:
        parser.error("counts and intervals must be positive")
    if args.rollout*args.envs % args.batch_size:
        parser.error("rollout * envs must be divisible by batch-size")
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
                  prior_training_actions=prior_actions,
                  observation="four raw 16x64 video-memory frames", reward="screen score difference only",
                  evaluation_policy="sample learned categorical distribution; seeded")
    write_json(args.run/("resume-config.json" if args.resume else "config.json"), config)
    agent = PPO(args.seed, args.learning_rate, args.entropy)
    rng = np.random.default_rng(args.seed)
    steps, episodes, best = 0, 0, -1.0
    if args.resume:
        agent.model.load_weights(str(args.resume/"model.safetensors"))
        agent.optimizer.state = tree_unflatten(list(mx.load(str(args.resume/"optimizer.npz")).items()))
        agent.optimizer.learning_rate = args.learning_rate
        agent.compile()
        steps, episodes, best = prior["steps"], prior["episodes"], prior["best_mean"]
        rng.bit_generator.state = prior["rng"]
    elif args.initialize:
        agent.model.load_weights(str(args.initialize))
        agent.model.advantage.weight *= args.logit_scale
        agent.model.advantage.bias *= args.logit_scale
    mx.eval(agent.state)
    envs = VectorEnv(args.envs, args.seed+steps, tstates=args.tstates)
    obs = envs.observations
    recent = deque(maxlen=100)
    beginning, last_log, start_steps = time.monotonic(), time.monotonic(), steps
    next_eval = (steps//args.eval_every+1)*args.eval_every
    stop, target_met = False, False
    log_file = (args.run/"metrics.jsonl").open("a", buffering=1)

    def log(row):
        row = {"wall_seconds": round(time.monotonic()-beginning, 2), **row}
        log_file.write(json.dumps(row)+"\n")
        if row["event"] != "episode":
            print(json.dumps(row), flush=True)

    def state():
        return dict(steps=steps, episodes=episodes, best_mean=best,
                    rng=rng.bit_generator.state, config=config)

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log({"event": "start", "steps": steps, "config": config})
    try:
        while not stop and (not args.steps or steps < args.steps):
            screens, action_buffer, logps, value_buffer, rewards, boundaries = [], [], [], [], [], []
            for t in range(args.rollout):
                actions, logp, values = agent.act(obs, rng)
                screens.append(obs)
                action_buffer.append(actions)
                logps.append(logp)
                value_buffer.append(values)
                results = envs.step(actions)
                next_obs, reward_row, boundary_row = [], [], []
                for i, (frame, reward, terminal, truncated, info, reset) in enumerate(results):
                    next_obs.append(reset if reset is not None else frame)
                    learning_terminal = terminal or (args.life_terminal and info["life_lost"])
                    if truncated and not learning_terminal:
                        _, final_value = agent.predict(mx.array(frame[None]))
                        reward += args.gamma*float(final_value[0].item())
                    reward_row.append(reward)
                    boundary_row.append(learning_terminal or truncated)
                    if terminal or truncated:
                        episodes += 1
                        recent.append(info)
                        log({"event": "episode", **info, "episode_steps": info["steps"],
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
            now = time.monotonic()
            if now-last_log >= 10:
                log({"event": "progress", "steps": steps, "episodes": episodes,
                     "steps_per_second": round((steps-start_steps)/(now-beginning), 1),
                     "mean_score_100": float(np.mean([x["score"] for x in recent])) if recent else None,
                     "best_score_100": max((x["score"] for x in recent), default=0),
                     "highest_level_100": max((x["level"] for x in recent), default=1),
                     "actor_loss": float(np.mean(metrics, axis=0)[0]),
                     "value_loss": float(np.mean(metrics, axis=0)[1]),
                     "entropy": float(np.mean(metrics, axis=0)[2]),
                     "approx_kl": float(np.mean(metrics, axis=0)[3])})
                last_log = now
            if steps >= next_eval:
                directory = args.run/f"step-{steps:09d}"
                agent.save(directory, state())
                result = evaluate(agent.policy(), list(range(10000, 10000+args.eval_games)),
                                  tstates=args.tstates, max_steps=20000)
                result["policy"] = "learned categorical, sampled"
                write_json(directory/"evaluation.json", result)
                log({"event": "validation", "steps": steps,
                     **{k: v for k, v in result.items() if k != "games"}})
                if not result["incomplete_games"] and result["mean_score"] > best:
                    best = result["mean_score"]
                    agent.save(args.run/"best", state())
                    write_json(args.run/"best"/"evaluation.json", result)
                agent.save(args.run/"latest", state())
                next_eval += args.eval_every
                if not result["incomplete_games"] and result["level_1_clears"] >= args.target_clears:
                    target_met = True
                    break
    except BaseException as error:
        log({"event": "error", "steps": steps, "error": str(error)})
        raise
    finally:
        agent.save(args.run/"latest", state())
        log({"event": "stopped", "steps": steps, "target_met": target_met})
        envs.close()
        log_file.close()


if __name__ == "__main__":
    main()
