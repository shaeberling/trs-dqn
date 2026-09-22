"""Optional multi-step latent regularizer, adapted from PlaNet section 4.

Uses only actual own action sequences and stopped posterior targets. No
future image enters the prior rollout, no extra reward or planner is added.
"""

import mlx.core as mx
import numpy as np


def validate(distance, weight):
    if (isinstance(distance, bool) or not isinstance(distance, int) or not 1 <= distance <= 16
            or not np.isfinite(weight) or weight < 0 or (weight > 0 and distance < 2)):
        raise ValueError('invalid latent overshooting distance/weight')


def latent_overshoot(model, states, actions, key, burn, distance):
    from .defense_world_model import normal_kl
    count = actions.shape[1]-distance-burn+1
    if distance < 2 or count < 1 or burn < 0:
        raise ValueError('overshooting requires complete post-burn action sequences')
    state = tuple(mx.stop_gradient(s[:, burn:burn+count].reshape(-1, s.shape[-1])) for s in states)
    keys = mx.random.split(key, distance+1)[1:]
    penalties = []
    for d in range(1, distance+1):
        selected = actions[:, burn+d-1:burn+d-1+count].reshape(-1)
        # No encoder or later observation is consulted during this rollout.
        state, (prior_mean, prior_std) = model.step(state, selected, keys[d-1], sample=True)
        if d == 1:
            continue  # The ordinary objective already trains one-step KL.
        mean, std = (mx.stop_gradient(s[:, burn+d:burn+d+count].reshape(-1, s.shape[-1]))
                     for s in states[2:])
        penalties.append(mx.mean(mx.maximum(normal_kl(mean, std, prior_mean, prior_std), 3.)))
    return mx.mean(mx.stack(penalties))
