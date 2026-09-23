# Score-gated candidate pilot

This completed two-generation native pilot checked a new training rule:
score from the same own pre-loss states nominates a neural candidate, then
actual complete games from fresh **training** boot seeds decide whether the
candidate replaces the current policy. Both comparisons use only displayed
score. The separate ten-game validation set is never used for an update.

Four complete games from own training seeds 74000–74003 supplied 16 native
states captured 128 decisions before visible life losses. All states were
restored and checked against the learner's recorded actions, screens and
score increments. The 20-action visual row search then played **328 paired
and incumbent focused segments** across two generations. Including harvest
and eight complete boot-gate games, this used **66,921 new training actions**.

In generation 1, a candidate improved focused mean score gain from
**2,055 to 2,440**, but its mean over four new complete boot games was only
**2,785**, versus **9,960** for the incumbent on the same seeds. The gate
rejected it. No candidate beat the incumbent focused score in generation 2.
The initial and final full-model SHA-256 both equal
`125346536cb1570a04dd65c34680d904c4d4e2b8924517cc5112bfc2a8bbb171`.
The ten reused validation games therefore stayed at mean 9,981, median
10,380, best 10,480, all stage 1. No new stage or mission was found.

`run/` preserves every state/trace, raw candidate return, boot score gate,
checkpoint, RNG and evaluation. The pilot demonstrates the earlier-route
regression the gate was designed to catch; it does not demonstrate improved
play yet.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-score-gate-pilot-reproduction \
  --harvest-games 4 --harvest-seed 74000 --lookback 128 \
  --directions 20 --snapshots-per-direction 4 --sigma .05 \
  --step-size .02 --coordinate-row --update-mode score-gated \
  --minimum-focus-gain 10 --minimum-boot-gain 0 \
  --boot-gate-games 4 --first-boot-training-seed 90000 \
  --generations 2 --seed 101 --eval-every 1
```
