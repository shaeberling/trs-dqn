# Learned dynamics preflight: not yet a playing policy

This records the initial prediction-only milestone. A later
[actor calibration](../imagination-calibration-01/README.md) now plays complete
games, but performs poorly; no mission or later stage has been observed.

Repeated model-free trials still fail in stage 1. This experiment starts a
different path: learn an action-conditioned recurrent world model, then test
whether its forecasts are useful enough for subsequent behavior learning.
No actor or imagined-return learner has been added yet. Neither prediction
accuracy nor a completed calibration counts as game progress.

The model is inspired by [Dreamer](https://arxiv.org/abs/1912.01603), not a
reproduction. It uses a smaller Gaussian recurrent state-space model (128
deterministic, 32 stochastic dimensions), a convolutional screen encoder and
decoder, and score-change and continuation heads. A posterior observes the
current screen; a prior predicts from earlier latent state and the chosen
command only. The future screens used as training targets never enter an
open-loop forecast. Native MLX runs the model on this Mac.

## Experience and objective

The collector played **24 new complete boot games**, seeds 65000–65023,
using our own scalar DQN 33 at 6,962,144 actions, with nominal epsilon .25
and random holds up to 64 commands (power 1.5). All remained stage 1.
This is collection behavior, not a new policy evaluation or demonstration
dataset. No imitation loss trains the world model to copy those actions.

The split was assigned before collection: **20 training games / 39,162
actions**, and **four held-out games / 7,800 actions**. Training and held-out
episodes never overlap. Only raw visible video bytes, executed actions,
actual score increments, and visible life/episode boundaries are stored.
There are no emulator snapshots, private RAM, routes, expert actions or
existing evaluation trajectories. The [manifest](data/manifest.json)
hashes every compressed episode and records the parent weights and settings.

Windows are sampled uniformly over valid starts within whole episodes.
Each contains 32 actual transitions; the first eight warm up recurrent state.
Life losses stay in the sequence but have continuation target zero; episode
resets are never crossed. The screen uses the existing fixed semigraphics
and visible-ASCII encoding, without object or route extraction.

Training minimizes summed image reconstruction error, reward prediction
error, binary continuation loss and Gaussian posterior/prior KL with three
free nats. The reward target is the existing visible score increment times
.01. Reconstruction and continuation are model-learning objectives, not
additional game rewards. Adam is 6e-4, batch eight, gradient norm cap 100.
The initial fit completed **1,000 updates**, preserving full weights,
optimizer and RNG every 200 updates, including initialization.

## What the first check actually establishes

On 64 fixed windows from the four held-out games, deterministic mean-latent
graphics MSE at 1,000 updates is **.01711** one action ahead and **.04877**
24 actions ahead, versus **.07603 / .09194** when repeating the last screen.
That is a useful appearance-prediction result, not proof of correct dynamics.
Rotating future action labels gives **.01712 / .04995**: the benefit of the
correct actions is small. The [full audit](fit/update-001000/audit.json)
also reports changed-pixel errors, action sensitivity, score and boundary
baselines, and every horizon rather than only the favorable endpoints.

A second check deliberately selects held-out windows with a visible loss
at forecast action 16. **Twelve** complete windows qualify; final-life windows
without enough recorded aftermath are excluded. At that loss endpoint the
continuation Brier score is **.99946**, barely different from **1.0** for
always predicting survival. This model has not learned to anticipate those
losses. These are visible HUD boundaries, not exact collision timestamps.
The [forecast panels](review-1000/forecasts.png), visually reviewed, show
blurred barriers and poor event prediction. The numerical audit covers all
12 windows; the panel shows the first four. The earlier
[600-update review](review-600/report.json) is also retained.

There is therefore **no justification yet for trusting this model to train
navigation in imagination**. A continuation from the exact 1,000-update
state tests whether further learning improves the weak forecasts. This is
not a claim that more fitting will solve the game, and the held-out cases
remain excluded from updates.

Six focused tests cover Gaussian KL, screen rendering, causal filtering,
action dependence, real gradients, exact model/Adam/RNG resume, reward and
boundary alignment, burn-in masking, held-out isolation, and checksum
rejection. A first test failure was dictionary-order comparison of optimizer
arrays; comparing matching keys confirms exact values, not approximate
resume. The production continuation's initial weights, optimizer/RNG arrays
and forecast audit also match exactly. See [archive checks](archive-audit.json).

## Reproduction

```bash
# Fresh own collection; do not substitute saved evaluation replays.
venv/bin/python -m rl.defense_world_data \
  results/defense/training/dqn-33-persistent-resets/step-000006962144/model.safetensors \
  --output runs/defense-world-data-reproduction --games 24 --heldout 4 \
  --seed 65000 --epsilon .25

venv/bin/python -m rl.defense_world_fit runs/defense-world-data-reproduction \
  --output runs/defense-world-fit-reproduction --updates 1000 \
  --batch 8 --length 32 --burn 8 --audit-windows 64 --every 200

venv/bin/python -m rl.defense_world_review \
  runs/defense-world-fit-reproduction/update-001000 \
  runs/defense-world-data-reproduction --output runs/defense-world-review-reproduction
```

The collector enforces the unchanged 5 GiB free-space safeguard every 256
actions and before each episode; the fitter checks every 20 updates and
before checkpointing. Existing outputs are never overwritten. Completed
datasets and checkpoints are hash-checked. The collector's original source
is archived and hash-matched; a later empty-dataset validation guard changed
the module bytes, not the already-running collection method.

This experiment is deliberately excluded from the best-replay collector:
`world.safetensors` is not a playable policy. The original verified 10,480-point
replay remains available and unchanged. Successful mission completion is
still unproven.

## Follow-up: 2,000 updates and broader collection

The [continued state](continuation/update-002000/state.json) and
[held-out checks](continuation/update-002000/audit.json) are preserved.
One-action graphics MSE improves to **.01275**, but 24-action error worsens
to **.06567** on the same windows. At the twelve selected visible-loss
endpoints, continuation Brier remains **.99358** versus always-survive 1.0.
This is still poor boundary anticipation, not a readiness claim. The
continuation remains active toward 5,000 updates; no wall-clock limit was
introduced.

The full [382-test regression suite](regression-tests.txt) passed in 505.937
seconds. The stronger global best's entire manifest was hash-checked again
when preserving the continuation.

A separate new collection, `runs/defense-world-data-02`, uses the same own
parent with nominal epsilon **.05** and seeds **66000–66023**, keeping its
final four whole games held out again. It is intended to broaden coverage
of longer approaches: the first collection's exploratory training-game mean
was only **2,746**, despite starting from a stronger greedy policy. This
second collection is not yet part of the active fit, and no improvement is
assumed before checking its completed records.

That [second collection](../world-model-collection-02/README.md) subsequently
completed: 46,617 training actions and 9,412 held-out actions, all stage 1.
The training-game mean is 7,466, supporting broader later-approach coverage,
not deeper stage passage. No updates have used it yet. The original fit's
[3,000-update boundary review](continuation/review-3000/report.json) remains
poor (continuation Brier .99600 at the selected losses); its full model and
optimizer are preserved separately. The 5,000-update continuation is live.

It subsequently completed **5,000 updates**, with all checkpoint states and
the [complete log](continuation/metrics-complete.jsonl) preserved. Its final
one/24-action graphics errors are **.01005 / .07134** on the original fixed
held-out windows. At the twelve selected visible-loss endpoints, Brier is
still **.99518**. The [final panels](continuation/review-5000/forecasts.png),
visually inspected, retain blurred/mistimed forecasts despite sharper static
features. This did not establish reliable action consequences or loss
anticipation. The exact [original fitter source](fit-source-v1.py) is archived
and matches its recorded source SHA; the current fitter additionally supports
an explicit, checked dataset extension.
