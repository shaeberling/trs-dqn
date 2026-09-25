"""Fresh score-only PPO over twelve grouped keys and learned hold lengths.

The raw actor keeps twenty trainable rows per duration, while log-sum-exp
groups the nine stage-one fire aliases.  Its sampled option is a distinct
physical command and a hold length.  Neither grouping nor countdown reads
game RAM or steers from a hand-written route.
"""

import numpy as np

from .defense_balanced_duration import (group_duration_logits_mlx,
                                        group_duration_logits_numpy,
                                        grouped_duration_log_probs_mlx,
                                        grouped_duration_log_probs_numpy,
                                        grouped_duration_physical_actions)
from .defense_duration_ppo import DurationCategoricalPolicy, DurationPPO
from .defense_repeat import validate_spec
from .temperature_probe import temperature_policy


class GroupedDurationPolicy:
    """Run a grouped learned option policy with original keyboard commands."""

    def __init__(self, infer_logits, durations, temperature=1.0):
        self.duration_count = len(durations)
        grouped = lambda obs: group_duration_logits_numpy(infer_logits(obs), self.duration_count)
        self.inner = DurationCategoricalPolicy(temperature_policy(grouped, temperature),
                                               durations, action_count=12)

    def __call__(self, obs):
        return grouped_duration_physical_actions(self.inner(obs))

    def sample_with_rngs(self, obs, rngs):
        return grouped_duration_physical_actions(self.inner.sample_with_rngs(obs, rngs))

    def reset_seed(self, seed):
        self.inner.reset_seed(seed)

    def observe_boundaries(self, boundaries, rngs=None):
        self.inner.observe_boundaries(boundaries, rngs)


class GroupedDurationPPO(DurationPPO):
    """PPO objective uses exact grouped option likelihoods at real starts."""

    def __init__(self, seed=17, learning_rate=2.5e-4, entropy=.01,
                 reference_kl_weight=0, action_count=80, value_coefficient=.5,
                 canonical_fire=False, duration_explore_mix=0., durations=(1, 4, 16, 64)):
        _, durations = validate_spec(12, durations)
        if (action_count != 20 * len(durations)
                or canonical_fire or reference_kl_weight):
            raise ValueError("grouped duration PPO requires twenty raw rows per hold")
        self.durations = durations
        self.duration_count = len(durations)
        super().__init__(seed=seed, learning_rate=learning_rate, entropy=entropy,
                         reference_kl_weight=reference_kl_weight, action_count=action_count,
                         value_coefficient=value_coefficient, canonical_fire=False,
                         duration_explore_mix=duration_explore_mix)

    def _loss(self, model, obs, actions, old_logp, advantages, returns, actor_mask,
              reference_log_probs=None, logit_bias=None, head_weight_noise=None):
        import mlx.core as mx

        if reference_log_probs is not None or logit_bias is not None or head_weight_noise is not None:
            raise ValueError("grouped duration PPO does not use reference or output perturbations")
        logits, values = model.policy_value(obs)
        grouped = group_duration_logits_mlx(logits, self.duration_count)
        log_probs = grouped_duration_log_probs_mlx(grouped, self.duration_count,
                                                    self.duration_explore_mix)
        logp = mx.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
        mask = actor_mask.astype(mx.float32)
        decision_count = mx.maximum(mx.sum(mask), 1.)
        logratio = mx.where(mask > 0, logp - old_logp, 0.)
        ratio = mx.exp(logratio)
        actor = -mx.sum(mask * mx.minimum(ratio * advantages,
                                         mx.clip(ratio, .8, 1.2) * advantages)) / decision_count
        critic = .5 * mx.mean(mx.square(values - returns))
        sample_entropy = -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)
        entropy = mx.sum(mask * sample_entropy) / decision_count
        kl = mx.sum(mask * ((ratio - 1) - logratio)) / decision_count
        return actor + self.value_coefficient * critic - self.entropy * entropy, (actor, critic, entropy, kl)

    def act(self, obs, rng, *, actor_mask, logit_bias=None, head_weight_noise=None):
        import mlx.core as mx

        mask = np.asarray(actor_mask)
        if (mask.shape != (len(obs),) or mask.dtype != np.bool_
                or logit_bias is not None or head_weight_noise is not None):
            raise ValueError("one real-decision mask per worker; no output perturbations")
        logits, values = self.predict(mx.array(obs))
        grouped = group_duration_logits_numpy(np.array(logits), self.duration_count)
        log_probs = grouped_duration_log_probs_numpy(grouped, self.duration_count,
                                                      self.duration_explore_mix)
        probabilities = np.exp(log_probs)
        choices = (rng.random(len(obs))[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
        choices = choices.clip(0, grouped.shape[1] - 1).astype(np.int32)
        logp = log_probs[np.arange(len(obs)), choices]
        choices[~mask] = 0
        logp[~mask] = 0.
        return choices, logp, np.array(values)

    def policy(self, seed=0):
        import mlx.core as mx

        return GroupedDurationPolicy(lambda obs: np.array(self.predict(mx.array(obs))[0]),
                                     self.durations)
