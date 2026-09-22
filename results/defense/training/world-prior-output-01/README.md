# Direct prior-output experiment

**Negative gameplay result.** Additional supervision of one-step prior outputs
reduces some screen-prediction errors but does not improve the learned actor.
All 60 complete validation games remain in stage 1. The stronger **10,480-point
verified global best remains unchanged**.

## Method and matched control

Both fits resume the complete original-objective world at 20,000 updates from
`runs/defense-world-fit-10-byte-control/update-020000`, continuing to 22,000.
This is the original world, not the recently overfit head-only variant.
The [unchanged own dataset](../world-model-feedback-02/data) has 80 training
games / 146,625 actions, and sixteen complete held-out games. Both fits use
batch 8, sequence 32, burn-in 8, identical sampling and the same Adam settings.

The opt-in `--prior-output-weight .1 --change-objective` adds one-step
prediction losses for rendered visible screens, actual score changes scaled
by .01, and visible continuation. Each forecast starts from the **stopped**
filtered state before the action and advances through the learned prior using
that recorded action. The arrival screen is a target only. The extra loss is
0.1 times the sum of the same image/reward/continuation losses used for the
observed-posterior outputs; the original objective and KL remain unchanged.
It is a supervised-output experiment on the existing Gaussian world, not a
claim to reproduce a new published algorithm.

The auxiliary gradient reaches the transition, prior and output heads, but
does not backpropagate through the starting encoder/posterior states. Those
still learn through the original objective. No new reward, hidden-memory
target, collision oracle, action override, demonstration or evaluation trace
is introduced. Acting inputs and architecture are unchanged. Combining this
loss with the earlier byte/overshooting auxiliaries is explicitly rejected.

Initial model weights, full Adam and both RNGs are checked equal. Final
sampling/filter RNG states and all logged sampled-loss counts also match.
Four actual-data updates under the **default** objective remain bit-identical
in losses, weights, Adam and RNG to the preserved pre-change source:
[default-parity.json](default-parity.json).

## Held-out forecasts

These use the same fixed 64 held-out windows, with eight context actions.
Lower errors are better. Mean-latent forecasts are not calibrated stochastic
outcome distributions, and rotated-action sensitivity is not a true
counterfactual outcome measurement.

| Final forecast metric | Control | Direct prior outputs |
| --- | ---: | ---: |
| One-step graphics MSE | 0.019716 | 0.018450 |
| 24-step graphics MSE | 0.025705 | 0.021089 |
| 24-step changed-graphics MSE | 0.386511 | 0.401624 |
| Mean continuation Brier across 24 horizons | 0.0013581 | 0.0013522 |

Always predicting survival gives Brier **0.0013021** on these same windows,
better than either learned forecast. Improvements in whole-screen MSE do not
imply better prediction of the small changing objects: the 24-step changed
graphics error actually worsens.

Across 48 separately selected held-out windows with a visible loss at horizon
16, continuation Brier is **0.984851 control / 0.984678 treatment**, versus 1
for always surviving. Observed-arrival recognition error is **0.862171 /
0.883413**. There is no useful loss-anticipation breakthrough. These selected
loss windows are not population calibration estimates; visible life decrement
lags the physical collision. Each final checkpoint retains its full boundary
audit, and each world directory includes the actual/predicted screen panels
in `review-22000`.

## Actual playing comparison

Both actors retain the full behavior state of
`defense-imagination-byte-control-08/update-002000`: actor, critic, target,
both Adam states and RNGs. Only the respective descendant world is refreshed.
Each then receives 1,000 additional imagined updates. Baseline evaluations
already use the refreshed world, not the unchanged parent policy.

| Actor updates | Control mean | Direct-prior mean |
| --- | ---: | ---: |
| 2,000, before additional actor learning | 306 | 314 |
| 2,500 | 120 | 122 |
| 3,000 | 120 | 110 |

Each cell is ten uncapped complete boot games on reused seeds 10000–10009,
not a fresh success-rate estimate. All 60 games lose in stage 1, with no stage
2/3 or mission completion. Both local bests score 340, with
[1,574 verified control commands](actor-control/artifacts/best/replay.html) and
[1,630 verified treatment commands](actor-prior/artifacts/best/replay.html).
These weak model-based actors fail earlier than the preserved strong policy;
their scores do not indicate reaching its recurring barrier.

## Preservation and tests

[comparison.json](comparison.json) records copy hashes, full-state warm-start
checks, matching RNG/target checks, audits and native replay verification.
`control`, `prior`, `actor-control` and `actor-prior` preserve every initial,
intermediate and final checkpoint and complete log. The global best's manifest
hashes were checked again. Collector configuration preserves all previous
sources and now watches **67**.

The full **414-test suite passes**, including four new prior-output tests:
action/target/burn alignment, arrival-screen causality, gradient routing,
exact full resume and explicit objective-change gates. Eighteen focused world
compatibility tests also passed separately. Logs:
[regression-tests.txt](regression-tests.txt), [focused-tests.txt](focused-tests.txt).

This short matched experiment does not justify extending the current actor
training on its own. Better broad image prediction has again failed to yield
better control; success still requires actual stage passage and verified
complete play. The code remains opt-in, and all stronger previous models are
preserved for further experiments.

```bash
venv/bin/python -m rl.defense_world_fit runs/defense-world-data-feedback-02 \
  --resume runs/defense-world-fit-10-byte-control/update-020000 \
  --output runs/reproduce-prior-control --updates 22000 --batch 8 --length 32 \
  --burn 8 --audit-windows 64 --every 1000
# Repeat into another new output with --prior-output-weight .1 --change-objective.

venv/bin/python -m rl.defense_imagine_fit \
  runs/reproduce-prior-control/update-022000 runs/defense-world-data-feedback-02 \
  --output runs/reproduce-prior-control-actor \
  --resume runs/defense-imagination-byte-control-08/update-002000 \
  --refresh-world --updates 3000 --every 500 --batch 4 --games 10 --eval-envs 4
# Repeat with the experimental world and another new actor output.
```
