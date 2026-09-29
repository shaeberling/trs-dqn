"""Frozen-policy evaluation and independently verified Cosmic Fighter replays."""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .cosmic import ACTION_NAMES, CosmicEnv, ENVIRONMENT_VERSION, GAME_SHA256
from .defense_smoke import write_replay
from .vector import VectorEnv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def summarize(games):
    complete = [game for game in games if game["terminated"] and not game["truncated"]]
    scores = [game["score"] for game in complete]
    return dict(games_requested=len(games), complete_games=len(complete),
                incomplete_games=len(games)-len(complete),
                mean_score=float(np.mean(scores)) if scores else None,
                median_score=float(np.median(scores)) if scores else None,
                best_score=max(scores, default=None), games=games)


def load_policy(checkpoint):
    """Load a frozen Cosmic DQN; reject another game's checkpoint."""
    import mlx.core as mx
    from .model import QNetwork

    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint.parent/"state.json").read_text())
    config = state["config"]
    if (config.get("game") != "cosmic" or config.get("game_sha256") != GAME_SHA256
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("action_names") != list(ACTION_NAMES)
            or config.get("algorithm") != "dueling-double-dqn-per-nstep"):
        raise ValueError("Incompatible Cosmic Fighter checkpoint")
    model = QNetwork(action_count=len(ACTION_NAMES))
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    predict = mx.compile(model, inputs=model.state)

    def policy(observations):
        values = np.array(predict(mx.array(observations)))
        if values.shape != (len(observations), len(ACTION_NAMES)) or not np.isfinite(values).all():
            raise RuntimeError("Invalid learned Cosmic Q-values")
        return values.argmax(axis=1)

    return policy, config


def evaluate(policy, seeds, *, tstates, max_steps, observation_stride=1, envs=8):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or envs < 1:
        raise ValueError("evaluation requires distinct seeds and positive workers")
    count = min(envs, len(seeds))
    workers = VectorEnv(count, seeds[0], game="cosmic", tstates=tstates,
                        max_steps=max_steps, observation_stride=observation_stride)
    games = [None]*len(seeds)
    pending = iter(range(count, len(seeds)))
    try:
        observations = workers.reset(seeds[:count])
        active = {index: (index, obs) for index, obs in enumerate(observations)}
        while active:
            indices = sorted(active)
            jobs = [active[index][0] for index in indices]
            batch = np.stack([active[index][1] for index in indices])
            actions = policy(batch)
            if len(actions) != len(indices):
                raise RuntimeError("Cosmic policy returned wrong action count")
            results = workers.step(actions, indices=indices)
            for worker, job, result in zip(indices, jobs, results, strict=True):
                obs, _, terminal, truncated, info, _ = result
                if terminal or truncated:
                    games[job] = dict(seed=seeds[job], **info)
                    replacement = next(pending, None)
                    if replacement is None:
                        del active[worker]
                    else:
                        active[worker] = (replacement,
                                          workers.reset([seeds[replacement]], indices=[worker])[0])
                else:
                    active[worker] = (job, obs)
    finally:
        workers.close()
    return summarize(games)


def record_game(policy, seed, *, tstates, max_steps, observation_stride=1):
    env = CosmicEnv(tstates=tstates, max_steps=max_steps,
                    observation_stride=observation_stride)
    try:
        obs = env.reset(seed)
        frames, actions, rewards = [obs[-1]], [], []
        while True:
            action = int(policy(obs[None])[0])
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(obs[-1])
            actions.append(action)
            rewards.append(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    return (np.asarray(frames), np.asarray(actions, np.uint8),
            np.asarray(rewards, np.float32), dict(seed=seed, **info))


def verify_policy_trace(checkpoint, frames, actions, rewards, result):
    policy, config = load_policy(checkpoint)
    found = record_game(policy, result["seed"], tstates=config["tstates"],
                        max_steps=config["eval_max_steps"],
                        observation_stride=config["observation_stride"])
    for expected, actual in zip((frames, actions, rewards), found[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    if found[3] != result:
        raise RuntimeError("Frozen Cosmic model replay outcome changed")
    return dict(verified=True, verified_actions=len(actions),
                checkpoint_sha256=sha256(checkpoint), game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION,
                method="reload weights; reproduce every neural action, reward and screen from boot")


def publish_best(checkpoint, evaluation, output):
    """Only complete scheduled evaluations can publish an immutable best."""
    checkpoint, output = Path(checkpoint), Path(output)
    if (evaluation["incomplete_games"] or not evaluation["complete_games"]
            or evaluation.get("evaluation_only")):
        raise ValueError("Incomplete or probe evaluations cannot publish a best")
    candidate = max(evaluation["games"], key=lambda game: game["score"])
    before = sha256(checkpoint)
    policy, config = load_policy(checkpoint)
    frames, actions, rewards, result = record_game(
        policy, candidate["seed"], tstates=config["tstates"],
        max_steps=config["eval_max_steps"], observation_stride=config["observation_stride"])
    if result != candidate:
        raise RuntimeError("Candidate game differs from scheduled evaluation")
    verification = verify_policy_trace(checkpoint, frames, actions, rewards, result)
    if sha256(checkpoint) != before:
        raise RuntimeError("Frozen Cosmic model changed during replay verification")
    version = f"step-{json.loads((checkpoint.parent/'state.json').read_text())['steps']:09d}-{before[:12]}-seed-{candidate['seed']}"
    directory = output/"versions"/version
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    metadata = dict(game="Cosmic Fighter", game_sha256=GAME_SHA256,
                    environment_version=ENVIRONMENT_VERSION, action_names=ACTION_NAMES,
                    policy="learned Q-values, greedy", trained_model=True,
                    checkpoint_sha256=before, verified_actions=len(actions),
                    tstates=config["tstates"], max_steps=config["eval_max_steps"],
                    observation_stride=config["observation_stride"], result=result)
    for name in ("model.safetensors", "state.json"):
        shutil.copy2(checkpoint.parent/name, directory/name)
    write_json(directory/"evaluation.json", evaluation)
    write_json(directory/"verification.json", verification)
    np.savez_compressed(directory/"trace.npz", frames=frames, actions=actions,
                        rewards=rewards, metadata=json.dumps(metadata))
    write_replay(directory/"replay.html", frames, actions, metadata)
    write_json(directory/"manifest.json", dict(metadata=metadata,
               hashes={path.name: sha256(path) for path in sorted(directory.iterdir())}))
    temporary = output/"best.tmp"
    temporary.symlink_to(Path("versions")/version)
    os.replace(temporary, output/"best")
    return version
