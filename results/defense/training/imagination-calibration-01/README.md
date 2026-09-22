# First actor trained inside the learned world model

This is the first playable model-based policy in this branch, but it is
**not competitive yet**. All thirty complete calibration games lost in stage 1.
The separate verified best remains 10,480 points.

| Imagined actor updates | Mean | Median | Best |
| ---: | ---: | ---: | ---: |
| 0, untrained actor baseline | 276 | 280 | 280 |
| 500 | 294 | 300 | 320 |
| 1,000 | 282 | 280 | 300 |

Each row uses ten uncapped original-emulator games, reused seeds 10000–10009.
The [320-point replay](artifacts/best/replay.html) independently reproduced
**all 1,571 neural actions, rewards and screens from boot**. Initialization
is not eligible for replay promotion. This calibration remains outside the
shared collector and does not replace the stronger playing model.

## Learning and inputs

The actor and critic learn in the frozen own world model at 5,000 dynamics
updates. The method adapts the discrete-action REINFORCE behavior objective
from [DreamerV2](https://arxiv.org/abs/2010.02193), while retaining our smaller
Gaussian recurrent model rather than reproducing DreamerV2's categorical
latent architecture.

Rollouts start from filtered states of the original **20 training games**,
never their four held-out games. Each update samples four 32-action chunks,
uses an eight-action context, and imagines 15 transitions from their remaining
nonfinal states. Starts immediately following a visible life boundary are
masked. Future states and score increments are predictions from neural
networks, not queries of an emulator snapshot or game internals.

The actor has a 20-way categorical output and two 256-unit ELU hidden layers.
The critic has the same hidden sizes and a scalar output. Learning uses
discount .999, lambda .95, cumulative predicted-survival weights, Adam
4e-5 for the actor and 1e-4 for the critic, and norm cap 100. A separate
critic target is synchronized every 100 updates. Actor entropy coefficient
.001 is a policy regularizer; it is not added to score targets or returns.
The only reward model target remains actual visible score increments times
.01. No copied-action loss, demonstrations, route, or intrinsic reward is used.

REINFORCE gradients update the probability of the actor's **own sampled
imagined actions**, using a stopped-gradient return-minus-value advantage.
The critic regresses stopped lambda returns. Neither update changes the
world model. Its final saved weight bytes match the original world checkpoint
exactly; see [the preserved check](frozen-world-check.safetensors).

When playing, the policy observes the latest visible frame and retains only
its neural screen history and its own previous action. Filtering uses the
posterior mean, while imagined transitions sample Gaussian latents. The
acting path never calls reward/continuation prediction or an emulator branch.
No parser-derived life, score, stage or private RAM is fed into the policy.
Game memory is keyed by evaluation RNG identity, survives ship losses and
resets for each new game. All twenty original legal actions remain available.

## What is and is not demonstrated

The five new actor tests passed, including a synthetic score-gradient check,
terminal/return indexing, survival weights, frozen-world immutability, exact
model/actor/critic/Adam/RNG resume, target synchronization, per-game memory,
and complete native serial/parallel/reloaded replay agreement. The full
[387-test suite](regression-tests.txt) passed in 509.307 seconds.

This verifies implementation and replay plumbing, **not good play**. Predicted
returns rose during fitting, but real scores did not improve meaningfully.
The world model's loss forecasts remain poor, and its sparse data do not cover
every state/action combination the new actor can choose. These are concerns,
not a proven single cause of failure.

This calibration is offline behavior learning in a frozen own model, **not
yet an online Dreamer loop** that feeds the new actor's real experience back
into world-model learning. That feedback, stronger dynamics and further
real-game testing remain work toward the mission goal. A separate
[broader-data dynamics continuation](../world-model-mixed-01/README.md)
has finished; it does not silently alter this frozen calibration.

All three full actor checkpoints include their critic, target, both optimizers,
RNG and configuration. `steps` in these bundles means **actor updates**, not
emulator actions; the state and configuration label that unit explicitly.
The existing evaluator, standalone replay renderer and verifier support the
new architecture. Older DQN/PPO behavior is unchanged.

```bash
venv/bin/python -m rl.defense_imagine_fit \
  results/defense/training/world-model-preflight-01/continuation/update-005000 \
  results/defense/training/world-model-preflight-01/data \
  --output runs/defense-imagination-reproduction --updates 1000 --every 500 \
  --batch 4 --games 10 --eval-envs 4
```
