# Paired-score continuation at the recurring barrier

This completed trial started from the fifth-generation neural checkpoint selected by
fresh complete boot **training** games in [run 69](../ars-score-gated-69/README.md).
Its source model and full RNG state remain in the archived parent checkpoint.
It collected 48 new own-policy loss states from 12 complete training games and
searched all 20 screen-dependent action rows symmetrically. Displayed score
alone nominated candidates; 16 paired complete boot training games had to show
at least 100 mean score points of improvement before accepting a change.
Evaluation games 10000–10009 are never used for selection. Native source
states are opaque restore data, never policy input. The policy still receives
only rendered screens, its own recurrent state and actions; no oracle or
scripted steering is used. Each completed generation preserves weights,
training data and RNG. The verified global best replay remains separate.

The run stopped cleanly at generation 11, with 1,804 focused continuations
and 903,375 new training actions. Two updates passed the stricter boot gate,
at generations 7 and 11, but the fixed ten-game validation means were 9,467
initially, 8,583 at generation 10 and 9,560 at the end. Every focused
continuation and complete boot game remained in stage 1; no mission was
completed. The global best replay remains unchanged. Full states and logs
are archived in [run](run/).

The initial checkpoint's apparent eight-game improvement from run 69 did
not generalize cleanly: on this run's 12 new harvest training seeds, its
mean was 8,823.3 versus 8,905.8 for the original strong parent evaluated
on those same seeds ([paired record](parent-paired-training-comparison.json)).
Of its 48 lives, 33 lost between 2,500 and 2,650
displayed points. The original parent's run-69 harvest had 45 of 48 lives
in that band. Neither displayed score nor the later visible loss marker
identifies the exact collision position. The repeated band does show that
the search is mostly recycling the same approach. The next experiment
screens candidates on complete boot games directly, avoiding short-window
score as a prerequisite for exploring a possible passage.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-score-gated-69/run/generation-000005 \
  --output runs/defense-ars-focus-70-robust-score \
  --harvest-games 12 --harvest-seed 76000 --lookback 128 \
  --directions 20 --snapshots-per-direction 4 --sigma .05 \
  --step-size .02 --coordinate-row --update-mode score-gated \
  --minimum-focus-gain 10 --minimum-boot-gain 100 \
  --boot-gate-games 16 --first-boot-training-seed 91000 \
  --generations 0 --seed 121 --eval-every 10
```
