# Candidate: auxiliary prediction of own future visual representations

Status: the optional `--spr-weight` implementation and native/gradient/resume
tests now exist. No performance benefit is established. This document retains
the initial design and acceptance criteria; see the current calibration status
in [DEFENSE.md](../../../DEFENSE.md#auxiliary-visual-prediction).
Existing live trials 43/44 and their acting policies are unchanged.

## Reason to test

Repeated score plateaus and aligned replay panels still show an unresolved
stage-1 obstacle sequence. Exploration, discount and return-target changes
have not yet produced a stage clear. This motivates testing representation
learning, but does **not** establish that visual representation is the cause.
The current encoder has a spatial 6×16×32 output and a 256-unit hidden layer;
an auxiliary temporal task could train those features even on transitions
with no immediate score change. Neither a loss reduction nor a better feature
diagnostic would count as mission progress.

## Primary method

[Schwarzer et al., Self-Predictive Representations, ICLR 2021](https://arxiv.org/abs/2007.05929)
predict future latent representations through action-conditioned transitions.
Targets use an exponential-moving-average encoder and projection, with stopped
gradients; an online predictor learns a cosine-similarity objective alongside
Q-learning. The paper uses five prediction steps and stops at episode boundaries.
Its non-augmentation variant uses encoder dropout and EMA coefficient .99.
This would be a Defense-specific adaptation, not a reproduction of the paper's
Rainbow architecture, training schedule or Atari results.

## Proposed local implementation

- Keep the existing acting QNetwork and 20 actions. No planner, latent rollout,
  dropout or auxiliary head runs during action selection or evaluation.
- Extract spatial features with the existing render table and three learned
  convolutions. Do not change screen encoding, game timing or ordinary Q-network
  inference. The auxiliary path alone normalizes spatial activations and uses
  explicit-key dropout; its random state must be checkpointed separately.
- Use a small learned convolutional transition model conditioned on a one-hot
  recorded action. Prefer per-location normalization over batch-dependent
  statistics so padded sequences cannot influence other examples. Reuse the
  hidden layer as projection, with a separate learned prediction head. Record
  these architectural differences from the paper explicitly.
- Retain up to five actual transitions using the checked compact trajectory
  storage. Supply the original stored aggregate n-step return/discount to the
  unchanged scalar Double-Q target, rather than reconstructing it at different
  precision. Trace cutting remains off for this experiment.
- Predict only actual same-life/episode observations, including the last actual
  terminal screen when available. Mask padding and never cross into reset
  observations. Future replay screens are training targets, never acting inputs.
- Preserve the score-delta reward, its constant scale and TD-error replay
  priorities. Auxiliary prediction loss is a representation-training loss, not
  an intrinsic reward, novelty-based priority, route label or new action rule.
- Keep the original online/target/Adam state on conversion from an own scalar
  parent. Auxiliary transition/predictor parameters, EMA state and their optimizer
  start explicitly fresh. A later SPR-enabled resume must restore all of them;
  missing auxiliary state must fail, not silently reinitialize.
- Add Defense-local checkpoint handling for auxiliary state rather than changing
  Breakdown's checkpoint format. Plain model weights and normal greedy loading
  remain sufficient for acting/replay; the full training checkpoint must retain
  every auxiliary parameter, optimizer slot, EMA tensor and random key.
- Start with a small explicit auxiliary coefficient (candidate .1), log its loss
  separately, and test zero-coefficient parity. This coefficient is a proposal,
  not a tuned optimum. Do not combine the first comparison with longer holds,
  high discount, trace cutting, quantiles or bootstrap heads.

## Required checks before any performance trial

1. Disabled/default training remains exactly unchanged: online/target weights,
   Adam arrays, counters, main RNG and persistent-exploration RNG.
2. Scalar-parent conversion reproduces all original Q-values and greedy actions
   before an auxiliary update; no evaluation trajectory becomes training data.
3. Auxiliary gradients reach the shared encoder and prediction model, but not
   the EMA targets. TD-only output heads do not receive an auxiliary-head gradient.
4. EMA arithmetic, explicit random-key progression, target synchronization and
   full checkpoint/resume are independently tested; zero-update resume is exact.
5. Padded or post-reset observations cannot affect loss or gradients. Every
   sampled action is aligned with its actual next observation, including life
   terminals and truncation. Ring reference accounting remains bounded.
6. Native own-state restores and reserved boot-only workers remain correct; worker
   processes do not import MLX. Tests use small own generated runs, not demos.
7. Ordinary frozen greedy evaluation and replay verification work without
   instantiating the auxiliary learner. No synthetic success fixture is gameplay
   evidence. Auxiliary feature-collapse checks cannot substitute for stage reach.

Only after these checks: an isolated same-parent calibration, 131,072 new own
actions and ten complete uncapped boot games, compared with the preserved
baseline and with all setting/lineage differences recorded. Keep it out of the
global collector until it becomes a checked full trial. Preserve negative
results, and require actual original mission completion for the thread goal.
