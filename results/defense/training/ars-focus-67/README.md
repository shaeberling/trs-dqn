# Screen-dependent search over individual action rows

This finished, gracefully stopped trial preserved the strong neural screen encoder and changed
one of its 20 feature-to-action rows per candidate. Positive and negative
Gaussian changes cover all actions symmetrically each generation. Compared
with persistent action-bias changes, the effect depends on visible screen
features. No action or route is preferred by the trainer. Its sole training
fitness is displayed score gained in real emulator continuations from the
learner's own exact states, 128 decisions before visible ship losses.

Each snapshot is independently checked by reproducing its source screens,
actions and score increments, but those source actions are not training
targets. Complete boot games evaluated the frozen policy. A verified mission
in such a game would have stopped training. `run/` preserves all plans,
actual results, full model/RNG checkpoints, native-checked own states and
the original PPO optimizer lineage. `fit-source.py` exactly matches the
source hash recorded in the run configuration.

The larger row update caused severe regression. After **800 focused
continuations** and **112,382 new training actions**, none reached stage 2.
The original ten-game boot mean was 9,981; it fell to 5,179 at generation 1,
1,759 by generation 3, then **328** at the saved generation-5 stop. The final
median was 320, best 360, all stage 1. Despite this collapse in boot games,
median score gained from the old saved states remained 2,390. That gap shows
that optimizing these state continuations alone can harm the approach to
them. The global 10,480-point verified replay remains unchanged.

The run was stopped at its built-in generation boundary after the sustained
regression, preserving every completed update. This is a negative calibration
of `--step-size .5`, not evidence that smaller row updates or score-based
candidate acceptance would fail.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-67-feature-row \
  --harvest-games 12 --harvest-seed 71000 --lookback 128 \
  --directions 20 --snapshots-per-direction 4 --sigma .05 \
  --step-size .5 --coordinate-row --return-std-floor 20 \
  --generations 8 --seed 81 --eval-every 1
```
