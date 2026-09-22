"""Defense-only evaluation, policy loading and verified best-effort publishing.

All play comes from a learned screen-only policy. No controller,
demonstration data, hidden-state reward, or Breakdown level semantics.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .defense import action_names, DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256, validate_observation_stride
from .defense_smoke import write_replay
from .vector import VectorEnv


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DQN_ALGORITHM = "dueling-double-dqn-per-nstep"
BOOTSTRAP_ALGORITHM = "bootstrapped-dueling-double-dqn-prior-per-nstep"
QUANTILE_ALGORITHM = "quantile-dueling-double-dqn-per-nstep"
REPEAT_ALGORITHM = "joint-action-duration-dueling-double-dqn-per"
IMAGINATION_ALGORITHM = "gaussian-rssm-imagined-reinforce"


def policy_description(config, temperature=1.0, *, quantile_power=None):
    algorithm = config.get("algorithm", "ppo")
    if quantile_power is not None:
        if (algorithm != QUANTILE_ALGORITHM or isinstance(quantile_power, bool)
                or not np.isfinite(quantile_power) or not 0 <= quantile_power <= 4
                or temperature != 1):
            raise ValueError("Quantile power overrides require quantile DQN, power 0..4 and temperature 1")
    from .recurrent_policy import RECURRENT_ARCHITECTURE
    architecture = config.get("architecture")
    if algorithm == IMAGINATION_ALGORITHM:
        if (architecture != 'screen-rssm-imagined-actor-v1' or config.get('allow_enter', False)
                or config.get('world_hidden') != 128 or config.get('world_stochastic') != 32):
            raise ValueError('unsupported learned-world policy configuration')
        return ('learned imagined-return actor, categorical sampling; screen/own-action memory'
                if temperature == 1 else 'learned imagined-return actor, temperature-scaled; screen/own-action memory')
    if architecture is not None:
        if architecture != RECURRENT_ARCHITECTURE or algorithm != 'ppo':
            raise ValueError("Unsupported Defense policy architecture")
        return ("learned recurrent categorical, sampled; screen-history memory" if temperature == 1 else
                "learned recurrent categorical, temperature-scaled; screen-history memory")
    if config.get('recurrent_hidden', 0):
        raise ValueError("Missing recurrent policy architecture")
    if algorithm == REPEAT_ALGORITHM:
        from .defense_repeat import validate_spec
        validate_spec(len(action_names(config.get('allow_enter', False))), config.get('learned_repeats', ()))
        if not config.get('life_terminal'):
            raise ValueError('learned repeats require the declared life-boundary protocol')
        return "learned joint action-duration Q-values, greedy; holds end at visible life boundaries"
    if algorithm == DQN_ALGORITHM:
        return "learned Q-values, greedy"
    if algorithm == BOOTSTRAP_ALGORITHM:
        return "learned bootstrap ensemble plus fixed priors, greedy mean Q-values"
    if algorithm == QUANTILE_ALGORITHM:
        if quantile_power:
            return f"learned score-return quantiles, greedy power-distorted expectation (power={quantile_power:g})"
        return "learned score-return quantiles, greedy mean Q-values"
    if algorithm != "ppo":
        raise ValueError("Unsupported Defense policy algorithm")
    return ("learned categorical, sampled" if temperature == 1 else
            "learned categorical logits, temperature-scaled sampling")


def greedy_policy(infer_values):
    """Same evaluation protocol as categorical policies; no exploration/RNG draws."""
    def sample(obs):
        values = np.asarray(infer_values(obs))
        if values.ndim != 2 or values.shape[0] != len(obs) or not np.isfinite(values).all():
            raise ValueError("Invalid learned Q-values")
        return values.argmax(axis=1)

    def with_rngs(obs, rngs):
        if len(obs) != len(rngs):
            raise ValueError("provide one policy RNG per observation")
        return sample(obs)

    sample.reset_seed = lambda seed: None
    sample.sample_with_rngs = with_rngs
    return sample


def game_rank(game):
    """Completed full games only; verified missions outrank score."""
    if not game.get("terminated") or game.get("truncated") or not game.get("full_game", True):
        return None
    return (game["missions_completed"], game["highest_stage"], game["score"])


def summarize(games):
    complete = [g for g in games if game_rank(g) is not None]
    scores = [g["score"] for g in complete]
    return dict(complete_games=len(complete), incomplete_games=len(games)-len(complete),
                mean_score=float(np.mean(scores)) if scores else None,
                median_score=float(np.median(scores)) if scores else None,
                best_score=max(scores, default=None),
                highest_stage=max((g["highest_stage"] for g in complete), default=1),
                mission_games=sum(g["missions_completed"] > 0 for g in complete),
                stage_2_games=sum(g["highest_stage"] >= 2 for g in complete),
                stage_3_games=sum(g["highest_stage"] >= 3 for g in complete),
                games=games)


def load_policy(checkpoint, *, temperature=1.0, quantile_power=None):
    from .model import QNetwork
    from .temperature_probe import temperature_policy
    import mlx.core as mx
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent/"state.json").read_text())["config"]
    if isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    policy_description(config, temperature, quantile_power=quantile_power)  # Reject unknown algorithms, never guess.
    validate_observation_stride(config.get("observation_stride", 1))
    names = action_names(config.get("allow_enter", False))
    if (config.get("game") != "defense" or config.get("game_sha256") != GAME_SHA256
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("action_names") != list(names)):
        raise ValueError("Checkpoint is not compatible with this Defense environment")
    if config.get('algorithm') == IMAGINATION_ALGORITHM:
        from .defense_imagination import WorldActor, acting_policy
        model = WorldActor()
        model.load_weights(str(checkpoint))
        mx.eval(model.state)
        return acting_policy(model, temperature), config
    if config.get("architecture") is not None:
        from .defense_recurrent import ResidualRecurrentNetwork
        from .recurrent_policy import RecurrentPolicy
        model = ResidualRecurrentNetwork(len(names), config['recurrent_hidden'], config['memory_scale'])
        model.load_weights(str(checkpoint))
        mx.eval(model.state)
        predict = mx.compile(model.step, inputs=model.state)
        def infer(obs, hidden):
            logits, _, updated = predict(mx.array(obs), mx.array(hidden))
            return np.array(logits), np.array(updated)
        return RecurrentPolicy(infer, config['recurrent_hidden'], temperature=temperature), config
    if config.get("algorithm") == REPEAT_ALGORITHM:
        model = QNetwork(action_count=len(names) * len(config['learned_repeats']))
    elif config.get("algorithm") == BOOTSTRAP_ALGORITHM:
        from .defense_bootstrap import BootstrapQ
        model = BootstrapQ(len(names), config["bootstrap_heads"], config["bootstrap_prior_scale"])
    elif config.get("algorithm") == QUANTILE_ALGORITHM:
        from .defense_quantile import QuantileQ
        model = QuantileQ(len(names), config["quantiles"])
    else:
        model = QNetwork(action_count=len(names))
    model.load_weights(str(checkpoint))
    mx.eval(model.state)
    if config.get("algorithm") == REPEAT_ALGORITHM:
        if temperature != 1:
            raise ValueError('temperature overrides do not apply to learned greedy durations')
        from .defense_repeat import LearnedRepeatPolicy
        predict = mx.compile(model, inputs=model.state)
        return LearnedRepeatPolicy(lambda obs: np.array(predict(mx.array(obs))),
                                   len(names), config['learned_repeats']), config
    if config.get("algorithm") in (DQN_ALGORITHM, BOOTSTRAP_ALGORITHM, QUANTILE_ALGORITHM):
        if temperature != 1:
            raise ValueError("Temperature overrides do not apply to a greedy DQN policy")
        if quantile_power:
            from .defense_quantile import distortion_weights
            weights = mx.array(distortion_weights(config['quantiles'], quantile_power))
            predict = mx.compile(lambda obs: mx.sum(model.quantile_values(obs)*weights, axis=2),
                                 inputs=model.state)
            return greedy_policy(lambda obs: np.array(predict(mx.array(obs)))), config
        predict = mx.compile(model, inputs=model.state)
        return greedy_policy(lambda obs: np.array(predict(mx.array(obs)))), config
    predict = mx.compile(model.policy_value, inputs=model.state)
    return temperature_policy(lambda obs: np.array(predict(mx.array(obs))[0]), temperature), config


def evaluate(policy, seeds, *, tstates=100_000, max_steps=0, envs=10, log=None,
             should_stop=lambda: False, allow_enter=False, observation_stride=1):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Validation requires distinct seeds")
    count = min(envs, len(seeds))
    workers = VectorEnv(count, seeds[0], game="defense", tstates=tstates, max_steps=max_steps,
                        allow_enter=allow_enter, observation_stride=observation_stride)
    games = [None] * len(seeds)
    rngs = [np.random.default_rng(s + 1_000_000) for s in seeds]
    pending = iter(range(count, len(seeds)))
    started, last_log = time.monotonic(), time.monotonic()
    try:
        observations = workers.reset(seeds[:count])
        active = {i: (i, obs) for i, obs in enumerate(observations)}
        while active:
            if should_stop():
                raise InterruptedError("Defense evaluation interrupted; no complete suite claimed")
            indices = sorted(active)
            jobs = [active[i][0] for i in indices]
            obs = np.stack([active[i][1] for i in indices])
            actions = policy.sample_with_rngs(obs, [rngs[j] for j in jobs])
            results = workers.step(actions, indices=indices)
            if hasattr(policy, 'observe_boundaries'):
                policy.observe_boundaries(np.asarray([t or u or info['life_lost']
                    for _, _, t, u, info, _ in results], dtype=bool), [rngs[j] for j in jobs])
            for worker, job, result in zip(indices, jobs, results, strict=True):
                obs, _, terminal, truncated, info, _ = result
                if terminal or truncated:
                    games[job] = dict(seed=seeds[job], **info)
                    replacement = next(pending, None)
                    if replacement is None:
                        del active[worker]
                    else:
                        reset = workers.reset([seeds[replacement]], indices=[worker])[0]
                        active[worker] = (replacement, reset)
                else:
                    active[worker] = (job, obs)
            if log and time.monotonic() - last_log > 10:
                log(dict(event="validation_progress", finished=sum(g is not None for g in games),
                         active=len(active), seconds=time.monotonic()-started))
                last_log = time.monotonic()
    finally:
        workers.close()
    return summarize(games)


def record_game(policy, seed, *, tstates, max_steps, should_stop=lambda: False, allow_enter=False,
                observation_stride=1):
    env = DefenseEnv(tstates=tstates, max_steps=max_steps, allow_enter=allow_enter,
                     observation_stride=observation_stride)
    policy.reset_seed(seed+1_000_000)
    try:
        obs = env.reset(seed)
        frames, actions, rewards, events = [obs[-1]], [], [], []
        while True:
            if should_stop():
                raise InterruptedError("Recording interrupted; previous best remains intact")
            action = int(policy(obs[None])[0])
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(policy, 'observe_boundaries'):
                policy.observe_boundaries(np.asarray([done or truncated or info['life_lost']], dtype=bool))
            frames.append(obs[-1])
            actions.append(action)
            rewards.append(reward)
            if info["mission_completed"] or info["life_lost"]:
                events.append(dict(frame=len(actions), **info))
            if done or truncated:
                break
    finally:
        env.close()
    return np.asarray(frames), np.asarray(actions, np.uint8), np.asarray(rewards, np.float32), dict(seed=seed, **info), events


def verify_policy_trace(checkpoint, frames, actions, rewards, result, *, should_stop=lambda: False,
                        temperature=1.0, quantile_power=None):
    """Reload frozen weights and re-run every neural action and screen from boot."""
    policy, config = load_policy(checkpoint, temperature=temperature,
                                 **(dict(quantile_power=quantile_power) if quantile_power is not None else {}))
    actual = record_game(policy, result["seed"], tstates=config["tstates"],
                         max_steps=config["eval_max_steps"], should_stop=should_stop,
                         allow_enter=config.get("allow_enter", False),
                         observation_stride=config.get("observation_stride", 1))
    for expected, found in zip((frames, actions, rewards), actual[:3], strict=True):
        np.testing.assert_array_equal(found, expected)
    if actual[3] != result:
        raise RuntimeError("Policy re-execution outcome mismatch")
    if game_rank(result) is None:
        raise ValueError("An incomplete replay cannot replace the best complete game")
    return dict(verified=True, verified_actions=len(actions), temperature=temperature,
                **(dict(quantile_power=quantile_power) if quantile_power is not None else {}),
                observation_stride=config.get("observation_stride", 1),
                checkpoint_sha256=sha256(checkpoint), game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION,
                method="reload weights; reproduce every policy action, reward and screen from boot")


def publish_best(checkpoint, evaluation, output, *, should_stop=lambda: False, log=None):
    """Append immutable bundle and atomically switch best symlink after verification."""
    if (evaluation.get('evaluation_only') or evaluation.get('promotion_eligible') is False
            or evaluation.get('temperature', 1) != 1 or evaluation.get('quantile_power') is not None):
        raise ValueError('evaluation-only policy probes cannot promote the standard training best')
    checkpoint, output = Path(checkpoint), Path(output)
    candidates = [g for g in evaluation["games"] if game_rank(g) is not None]
    if not candidates:
        return None
    candidate = max(candidates, key=game_rank)
    best = output/"best"
    if best.exists():
        old = json.loads((best/"manifest.json").read_text())
        if game_rank(candidate) <= tuple(old["rank"]):
            return None
    policy, config = load_policy(checkpoint)
    frames, actions, rewards, result, events = record_game(
        policy, candidate["seed"], tstates=config["tstates"], max_steps=config["eval_max_steps"],
        should_stop=should_stop, allow_enter=config.get("allow_enter", False),
        observation_stride=config.get("observation_stride", 1))
    if candidate != result:
        raise RuntimeError("Recorded best effort disagrees with parallel evaluation")
    verification = verify_policy_trace(checkpoint, frames, actions, rewards, result,
                                       should_stop=should_stop)
    state = json.loads((checkpoint.parent/"state.json").read_text())
    version = f"step-{state['steps']:012d}-{verification['checkpoint_sha256'][:12]}-seed-{result['seed']}"
    bundle = output/"versions"/version
    bundle.mkdir(parents=True, exist_ok=False)
    for name in ("model.safetensors", "state.json"):
        shutil.copy2(checkpoint.parent/name, bundle/name)
    write_json(bundle/"evaluation.json", evaluation)
    write_json(bundle/"verification.json", verification)
    metadata = dict(game="Obstacle Run / Missile Defense", game_sha256=GAME_SHA256,
                    environment_version=ENVIRONMENT_VERSION, action_names=config["action_names"],
                    policy=policy_description(config), trained_model=True,
                    verified_actions=len(actions), checkpoint_sha256=verification["checkpoint_sha256"],
                    tstates=config["tstates"], max_steps=config["eval_max_steps"], result=result,
                    events=events, training_steps=state["steps"],
                    observation_stride=config.get("observation_stride", 1))
    np.savez_compressed(bundle/"trace.npz", frames=frames, actions=actions, rewards=rewards,
                        metadata=json.dumps(metadata))
    write_replay(bundle/"replay.html", frames, actions, metadata)
    manifest = dict(version=version, rank=game_rank(result), result=result,
                    selected_for="best single complete effort, not mean evaluation score",
                    hashes={name: sha256(bundle/name) for name in
                            ("model.safetensors", "state.json", "evaluation.json", "verification.json",
                             "trace.npz", "replay.html")})
    write_json(bundle/"manifest.json", manifest)
    if best.exists() and not best.is_symlink():
        raise RuntimeError("Refusing to replace an unmanaged best directory")
    temporary = output/".best-next"
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(Path("versions")/version, target_is_directory=True)
    os.replace(temporary, best)
    if log:
        log(dict(event="best_replay_published", version=version, result=result,
                 verified_actions=len(actions)))
    return manifest
