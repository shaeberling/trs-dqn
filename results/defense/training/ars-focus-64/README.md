# Earlier own-state start for global action-preference search

This trial changed only the snapshot rewind from 128 to **256 neural
decisions** before each visible life loss, compared with
[trial 63](../ars-focus-63/README.md). It used the same strong neural parent,
12 own complete harvest games, 48 verified states, 16 unbiased Gaussian
action-bias directions, four shared states per direction, sigma 5, step 0.5,
and eight generations. Displayed score gained until the next life/stage
boundary remained the sole search fitness; complete games from boot were
used for validation and verified best replay selection.

Across **1,024 focused continuations** and **166,781 new training actions**
including state harvest, no candidate reached stage 2. Median score gained
from the earlier state was just **60**; the maximum was **2,560**, which
includes more approach distance than the 128-action trial and is not a direct
comparison of navigation. Broad global bias changes often caused very early
loss. Final ten-game boot mean was **9,875**, median **10,280**, best **10,380**;
all stage 1. No validation round passed stage 1. The global verified best
remains 10,480 points in stage 1.

`run/` preserves all original source-state records and native checks, the
full model and optimizer lineage, eight sets of candidate plans and actual
scores, RNG state, boot validations, and local replay. The run configuration
records the exact training source SHA-256 (the version of
`rl/defense_ars_focus.py` in commit `7eb85b3`).

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-64-reproduction \
  --harvest-games 12 --harvest-seed 71000 --lookback 256 \
  --directions 16 --snapshots-per-direction 4 --sigma 5 --step-size .5 \
  --bias-only --return-std-floor 20 --generations 8 --seed 81 --eval-every 1
```
