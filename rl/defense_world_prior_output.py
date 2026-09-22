"""Optional supervised outputs from the causal one-step prior.

Past filtered states are stopped; actual arrival screens, score changes and
visible continuation are targets only. No new rewards or acting inputs.
"""

import mlx.core as mx
import mlx.nn as nn


def predictions(model, states, actions, key, burn, sample=True):
    if not 0 <= burn < actions.shape[1]:
        raise ValueError('prior output requires post-burn actions')
    previous = tuple(mx.stop_gradient(s[:, burn:-1].reshape(-1, s.shape[-1])) for s in states)
    prior, _ = model.step(previous, actions[:, burn:].reshape(-1), key, sample=sample)
    features = model.features(prior).reshape(len(actions), actions.shape[1]-burn, -1)
    return model.decode(features), model.reward(features)[..., 0], model.continue_logit(features)[..., 0]


def prior_output_loss(model, states, frames, actions, rewards, continuation, key, burn):
    image, score, logits = predictions(model, states, actions, mx.random.split(key, 2)[1], burn)
    target = model.pixels(frames[:, burn+1:])
    image_loss = .5*mx.mean(mx.sum((image-target)**2, axis=(-3, -2, -1)))
    reward_loss = .5*mx.mean((score-rewards[:, burn:])**2)
    continuation_loss = mx.mean(nn.losses.binary_cross_entropy(logits, continuation[:, burn:], with_logits=True))
    return mx.stack([image_loss, reward_loss, continuation_loss])
