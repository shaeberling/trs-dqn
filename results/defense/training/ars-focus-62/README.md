# Wider perturbations at the same own-loss states

This matched comparison to [focused trial 61](../ars-focus-61/README.md)
started from the same frozen strong policy, harvested the same 48 own
pre-loss states from training seeds 71000–71011, and used the same search
RNG, 16 directions, four shared snapshots per direction and eight
generations. Only the categorical-head perturbation sigma changed from
**0.005 to 0.05**. The score from actual emulator continuations to the next
visible life or stage boundary remained the only training fitness.

All **1,024 focused continuations** remained in stage 1. Median score gain
from their respective saved states fell to **1,640** versus **2,410** in the
narrow trial; both had maximum gain **2,460**. The wider changes created more
different, often worse behavior but did not find a passage. The final ten
complete boot games averaged **10,356**, median **10,355**, best **10,480**;
all stage 1. The highest reused ten-game mean was 10,386 at generation 4,
below the global best's individual score and without stage passage.

`run/` preserves **129,054 new training actions**, full model/RNG
checkpoints, 12 complete source games, all opaque training states and their
exact source-trace checks, individual focused scores, plan/return arrays,
all boot evaluations, and the unchanged parent PPO optimizer. These are
reused validation seeds, not a fresh generalization test. The globally best
verified 10,480-point PPO replay remains available.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-62-reproduction \
  --harvest-games 12 --harvest-seed 71000 --lookback 128 \
  --directions 16 --snapshots-per-direction 4 --sigma .05 \
  --step-size .002 --generations 8 --seed 81 --eval-every 1
```
