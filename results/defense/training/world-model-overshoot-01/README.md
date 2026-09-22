# Matched four-step latent prediction experiment

This tests an optional regularizer of the learned world, not a route, planner,
new observation or reward. The previous two real-experience feedback cycles
did not improve play, and longer forecasts remain weak.

The method adapts [PlaNet's latent overshooting](https://arxiv.org/html/1811.04551v5#S4):
roll out action-conditioned priors for several steps and compare them with
stopped posterior distributions. The paper reports mixed results and slightly
worse results for its final recurrent model, so this is a hypothesis test,
not a claimed proven remedy for this game.

Our variant keeps the entire original one-step objective and adds weight 1
times the mean Gaussian KL at distances 2, 3 and 4, with the same three-free-nat
floor. It uses every start after the eight-action context for which all four
actions fit inside the 32-action training chunk. Initial filtered states and
posterior targets are detached; gradients train the prior rollout only.
Future images are targets, never inputs to that rollout. No image decoding
or extra reward target is added to the regularizer. Actual visible life
transitions remain inside continuous original-game sequences; no boot episode
is crossed or fabricated.

The matched arms start from the exact 16,000-update world, Adam arrays and
sampling/model RNG. Both use identical own training data and sampling settings,
and target 18,000 total updates. Only one enables the new regularizer. Default
settings preserve the original learner; an objective change on full resume
requires explicit `--change-objective` and is recorded in checkpoint metadata.

Three new tests cover exact action/target indexing, stopped target/encoder
gradients, active prior gradients, full optimizer/RNG resume, invalid settings
and explicit objective-change gating. All six original world-model tests pass.
The separate [default-parity audit](default-parity.json) compares against the
exact saved pre-change source for four updates on 32 actual own training
windows: scalar metrics, model parameters, every Adam array and model RNG are
bit-identical. Its diagnostic updated states are never used for training.

The [global collector configuration](collector-config.json) now retains all
54 previous sources and adds seven complete-game verified imagined-actor
sources, including the upcoming matched actors. It was gracefully restarted
to load the new actor architecture. A stronger replay still requires native
verification and better game-outcome rank before publication.

```bash
venv/bin/python -m rl.defense_world_fit \
  results/defense/training/world-model-feedback-02/data \
  --output runs/defense-overshoot-reproduction \
  --resume results/defense/training/world-model-feedback-02/fit/update-016000 \
  --updates 18000 --batch 8 --length 32 --burn 8 --every 1000 \
  --overshoot-distance 4 --overshoot-weight 1 --change-objective
```

Omit the final three options for the original-objective control. Neither
prediction loss nor more gradient updates alone counts as progress toward
the original successful mission sequence.

## Completed result

Both arms finished 18,000 updates with all full states and logs preserved.
For legacy archive-helper naming only, `uniform/` is the original objective
and `focused/` is **latent overshooting, not focused data sampling**. Both
sample identical uniform batches: the 100 logged batches each contain 37
visible loss targets. Initial model weights, every Adam array and both RNG
states were checked equal before fitting.

On 48 selected held-out loss cases, mean-latent sixteen-step continuation
Brier is **.99188 control / .99335 overshooting**, versus always-survive 1.0.
Observed-arrival recognition is .90599 / .90977. Sixteen stochastic draws do
not resolve the failure. On uniform held-out windows, mean continuation Brier
is .0013223 / .0013167, versus always-survive .0013021; this tiny aggregate
difference does not indicate reliable loss anticipation.

The [matched actor comparison](../imagination-overshoot-comparison-01/README.md)
also fails: final means 120 / 112, all stage 1. There is no navigation benefit
demonstrated by this regularizer. All **399 regression tests** pass; the
[complete log](regression-tests.txt) records 339.521 seconds. Lower training
loss and passing tests are not counted as game progress.
