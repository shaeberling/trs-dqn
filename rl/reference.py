"""Training-only regularization toward this agent's own frozen RL policy.

Only current learner observations are queried. No reference actions, rewards,
trajectories, optimizer state, or hidden emulator state enter the learner.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .env import ENVIRONMENT_VERSION


def reference_metadata(checkpoint, weight, tstates, observation_stride, expected_sha256=None):
    """Validate identity and observation semantics before starting GPU/workers."""
    if isinstance(weight, bool) or not np.isfinite(weight) or weight < 0:
        raise ValueError("reference KL weight must be finite and nonnegative")
    if checkpoint is None:
        if weight:
            raise ValueError("positive reference KL weight requires --reference-policy")
        return None
    if not weight:
        raise ValueError("--reference-policy requires a positive reference KL weight")
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint.parent/"state.json").read_text())
    config = state.get("config", {})
    if (config.get("algorithm") != "ppo"
            or config.get("environment_version") != ENVIRONMENT_VERSION
            or config.get("tstates", 100000) != tstates
            or config.get("observation_stride", 1) != observation_stride):
        raise ValueError("reference policy must be PPO with matching environment and screen timing")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("reference policy hash changed since the saved learner checkpoint")
    return {"checkpoint": str(checkpoint), "sha256": digest, "weight": weight}


def reference_kl(log_probs, reference_log_probs):
    """Mean forward KL(reference || learner); gradients only reach learner."""
    import mlx.core as mx

    target = mx.stop_gradient(reference_log_probs)
    return mx.mean(mx.sum(mx.exp(target)*(target-log_probs), axis=-1))


class ReferencePolicy:
    def __init__(self, checkpoint):
        # This class is constructed only in the parent learner, after resume.
        import mlx.core as mx
        from .model import QNetwork

        self.model = QNetwork()
        self.model.load_weights(str(checkpoint))
        self.model.freeze()
        mx.eval(self.model.state)
        self.predict = mx.compile(lambda obs: self.model.policy_value(obs)[0],
                                  inputs=self.model.state)

    def targets(self, observations, batch_size):
        """Cache six detached log-probabilities per current rollout screen."""
        import mlx.core as mx

        if batch_size < 1 or not len(observations):
            raise ValueError("reference target batches must be nonempty and positive")
        targets = []
        for offset in range(0, len(observations), batch_size):
            logits = self.predict(mx.array(observations[offset:offset+batch_size]))
            targets.append(np.array(logits-mx.logsumexp(logits, axis=-1, keepdims=True)))
        return np.concatenate(targets)
