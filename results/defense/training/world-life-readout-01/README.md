# Life-loss readouts and head-only continuation

**No gameplay improvement.** Separate readouts can recover more information
about visible life loss from the frozen world state. Training only its actual
continuation head improves recognition, but longer fitting worsens held-out
one-step prediction. Both subsequent learned actors still lose in stage 1.
The verified global best remains **10,480**, not replaced by these weak actors.

## Frozen-state diagnostic

`rl.defense_world_life_probe` uses the original-objective world at 20,000
updates from [the byte comparison's control](../world-model-bytes-01/control).
It extracts deterministic mean-latent features using 32-action windows and
eight actions of burn-in. Overlapping tail windows are masked so every action
after each game's first eight appears **exactly once**: 145,985 training rows
with 320 visible losses, and 29,239 held-out rows with 64 losses. All 80 training
and 16 held-out games keep their original assignments.

Observed-arrival features include the arrival screen: they measure recognition,
not prediction. One-step prior features use only earlier screens and the
recorded intervening action, never the arrival screen. Four separate readouts
start from the original continuation-head weights, each with a fresh diagnostic
Adam, and train for 2,000 updates at batch 1,024. Uniform sampling is compared
with half-loss/half-survival sampling weighted by twice each class's original
frequency. This preserves the same expected empirical BCE objective; it is
not a new reward or a balanced-class probability target.

| Held-out diagnostic, final | Brier error ↓ | Loss average precision ↑ | Losses detected / false alarms at 0.5 |
| --- | ---: | ---: | ---: |
| Original observed-arrival head | 0.002124 | 0.0688 | 0 / 1 |
| Observed, uniform | 0.001726 | 0.3344 | 11 / 2 |
| Observed, weighted stratification | 0.001786 | 0.3365 | 16 / 11 |
| Original one-step prior head | 0.002123 | 0.0572 | 0 / 0 |
| One-step prior, uniform | 0.001972 | 0.1734 | 2 / 1 |
| One-step prior, weighted stratification | 0.001993 | 0.2106 | 8 / 9 |

There are 64 actual losses. Constant always-survive Brier is 0.002189, and
constant-score average precision equals loss prevalence, also 0.002189. Tests
check tied-score handling and the direction of the loss probabilities. These
are correlated transitions from only sixteen held-out games, not independent
gameplay success estimates. Weighted stratification is not a decisive winner.

All readout weights/Adam/RNG states, intermediate audits, array hashes and
compact per-row episode/action provenance are in `readouts`. Frozen-world
weights are checked identical after all four fits. Features are recomputable
from the immutable own dataset and model; no diagnostic state is imported
into an acting policy.

## Actual model continuation, without resetting its optimizer

`rl.defense_world_head_fit` separately resumes the **complete** world and Adam
at 20,000 updates, then trains only its continuation head on observed-arrival
own-training features, with ordinary uniform sampling. It does not load the
diagnostic heads. Two thousand head-only updates produce `head-short` at
22,000 total updates. A full-state continuation adds 18,000 more, producing
`head-long` at 40,000: **20,000 original world updates plus 20,000 head-only
updates**, not 40,000 full-dynamics updates.

Only continuation-head weights and moments change. MLX receives a partial
gradient tree, not zero gradients for other parameters. Encoder, recurrence,
prior, decoder, reward head, their Adam moments and model RNG remain exact;
the global Adam step advances normally. Complete state remains compatible
with ordinary world continuation, and a real-update save/resume test checks
exact next-update equality. No optimizer history is silently discarded.

| Held-out full-game rows | Original | +2,000 head updates | +20,000 head updates |
| --- | ---: | ---: | ---: |
| Observed-arrival Brier ↓ | 0.002124 | 0.001745 | 0.001851 |
| Observed-arrival average precision ↑ | 0.0688 | 0.3357 | 0.3884 |
| Observed losses detected / false alarms | 0 / 1 | 13 / 5 | 19 / 20 |
| One-step prior Brier ↓ | 0.002123 | 0.002008 | 0.002526 |
| One-step prior average precision ↑ | 0.0572 | 0.1456 | 0.0880 |

Long fitting reduces training observed Brier to 0.001005, but held-out error
worsens relative to the short fit. Its one-step Brier exceeds even the
always-survive baseline. This is evidence of limited generalization and a
recognition/forecast mismatch, not proof that the latent contains no useful
information or that this is the sole cause of navigation failure.

The separate, earlier-style **48 selected nonfinal loss windows** are retained
in each final checkpoint's `boundary-predictions.json`. Sixteen-step Brier
remains poor: 0.98816 original, 0.97840 short, 0.96830 long. These selected
windows differ from the unique whole-game rows above and are not population
calibration estimates. Visible life decrement also lags the physical collision.

## Complete boot-game test

Both actors begin with the same retained behavior weights, critic, target,
two Adam states and RNGs as the prior control's 2,000-update actor. Only the
world's continuation head differs; the acting feature computation is
unchanged. Each actor then gets 1,000 additional imagined updates. The existing
[control continuation](../world-model-bytes-01/actor-control) is reused, not
rerun or counted as new evidence. Actual initial-state equality is checked.

| Actor updates | Existing control mean | Short-head mean | Long-head mean |
| --- | ---: | ---: | ---: |
| 2,000, before additional actor learning | 300 | 300 | 300 |
| 2,500 | 110 | 110 | 120 |
| 3,000 | 104 | 102 | 104 |

Each entry is ten uncapped complete boot games on reused seeds 10000–10009.
All **60 new** validation games lose in stage 1; no stage 2/3 or mission is
observed. Both local bests score 340 with **1,628 verified commands**:
[short-head replay](actor-short/artifacts/best/replay.html),
[long-head replay](actor-long/artifacts/best/replay.html). These actors fail
earlier than the strong 10,480-point policy; no success at its barrier is claimed.

## Evidence, tests and next direction

[comparison.json](comparison.json) records full copy hashes, actual unchanged
parameter/moment checks, full-state actor warm starts, evaluations and replay
verification. [archive-source.py](archive-source.py) preserves every initial,
intermediate and final checkpoint and complete log. Collector configuration
history is retained; the sole collector now watches all **65** sources. The
global best's manifest hashes were verified again.

The full **409-test suite** passed before the final head-only test was added.
All **six new tests** subsequently passed together, including head-only exact
resume and preservation of nonzero Adam moments. This is not a claim that the
full 410-test suite ran. Logs: [regression-tests.txt](regression-tests.txt),
[focused-tests.txt](focused-tests.txt).

This retires longer head-only fitting as the current approach. Recognizing a
loss after seeing its screen is not the same as learning useful action
consequences. The next model-based change should address that predictive
limitation and be judged by actual complete-game play, not another reduction
in reconstruction error. Score reward, screen-only inputs, original game and
no-demonstration rules remain unchanged.

```bash
venv/bin/python -m rl.defense_world_life_probe \
  runs/defense-world-fit-10-byte-control/update-020000 \
  runs/defense-world-data-feedback-02 --output runs/reproduce-life-readouts \
  --updates 2000 --every 500 --batch 1024

venv/bin/python -m rl.defense_world_head_fit \
  runs/defense-world-fit-10-byte-control/update-020000 \
  runs/defense-world-data-feedback-02 --output runs/reproduce-life-head \
  --updates 2000 --every 1000 --batch 1024 --features observed
# Continue its full update-022000 state with --updates 18000 --every 6000
# into a new output to reproduce the long phase. These are additional updates.

venv/bin/python -m rl.defense_imagine_fit \
  runs/reproduce-life-head/update-022000 runs/defense-world-data-feedback-02 \
  --output runs/reproduce-life-head-actor \
  --resume runs/defense-imagination-byte-control-08/update-002000 \
  --refresh-world --updates 3000 --every 500 --batch 4 --games 10 --eval-envs 4
```
