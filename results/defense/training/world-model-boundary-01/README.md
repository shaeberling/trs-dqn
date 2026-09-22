# Does loss-window sampling fix the missing death forecast?

It improves recognition on the selected loss cases, but **does not establish
reliable anticipation or better play**. Both full model/Adam/RNG states and
complete logs are preserved; see the [comparison](comparison.json).

The merged own training collection contains 85,779 transitions and only
160 visible life boundaries. Of its 84,539 complete 32-action windows,
2,920 include a boundary after the eight-action context. These are visible
parser labels already used by the existing life-terminal learner, not
hidden collision times or demonstrations.

Both arms start from the same 10,000-update model, exact Adam arrays and
training RNG. Each runs 2,000 additional dynamics updates, batch eight.
The control retains uniform-window sampling. The focused arm draws 50%
of windows from the unique eligible loss-window set and 50% uniformly.
Original frames, executed actions, score targets and continuation labels
are unchanged. Neither arm uses the eight held-out games for gradients.
No loss reward or hand-coded action is added.

This is an intentional sampling-distribution change, with no importance
correction and no unbiased probability-calibration claim. Default sampling
retains the original exact random draws. Resume requires an explicit
`--change-sampling` when changing the fraction.

## Held-out results at 12,000 updates

The following Brier errors are on 24 selected held-out loss cases; lower
is better, and always predicting survival gives 1.0. Observing the arrival
screen is recognition, **not a forecast**. Priors never see future screens.

| Mean-latent assessment | Uniform | Focused |
| --- | ---: | ---: |
| Observed loss arrival | .85693 | .73588 |
| One-step forecast | .86739 | .76856 |
| Four-step forecast | .87714 | .79839 |
| Eight-step forecast | .89136 | .79955 |
| Sixteen-step forecast | .98218 | .97001 |

Sixteen stochastic latent draws give sixteen-step mean-probability Brier
.98148 / .93788. This remains poor. More importantly, on the same **uniformly
sampled 64 held-out windows**, averaged across their 24 forecast steps,
continuation Brier worsens from **.002710 to .003974**; always surviving gives
.002604. Selected-loss improvement is therefore not overall calibration
improvement. The ordinary audit contains few losses and must not be read
as showing that a model can navigate safely.

Every twentieth training batch was logged: those 100 batches contain
28 / 414 loss targets out of 19,200 targets in each arm. These are logged
subsets, not counts across every gradient update. Both runs finished
normally at the predeclared update target, not at a wall-clock limit.

The [matched frozen-world actors](../imagination-boundary-comparison-01/README.md)
also failed to improve play: means 280 / 270 at 1,000 imagined updates,
all games stage 1. Only actual later-stage play would establish progress.

## Reproduce

```bash
venv/bin/python -m rl.defense_world_fit \
  results/defense/training/world-model-mixed-01/data \
  --output runs/defense-world-boundary-reproduction \
  --resume results/defense/training/world-model-mixed-01/update-010000 \
  --updates 12000 --batch 8 --length 32 --burn 8 --every 1000 \
  --boundary-fraction .5 --change-sampling
```

Omit the last two flags for the uniform control. New tests cover exact default
sampling/RNG parity, exhaustive eligible-window indexing and deduplication,
final boundaries, reward/action/frame alignment, exact sampling resume,
held-out rejection, and forecast independence from the arrival screen.
The full [393-test regression suite](regression-tests.txt) passed in 346.468
seconds before the separate actor-collection addition.
