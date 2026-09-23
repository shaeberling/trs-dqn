# Coordinate search close to the visible life loss

This completed trial matches [trial 65](../ars-focus-65/README.md) in strong
parent, source games, symmetric changes to all 20 action-head biases,
score-only fitness, and complete boot evaluation. It changes only the
own-state rewind from 256 to **64 decisions** before the visible life loss.
All captured states are verified by reexecuting their source actions and
screens; source actions are not training examples. The neural policy sees
only its usual screen history.

`run/` preserves the full model/RNG lineage, 48 verified own states, all
population plans and scores, and boot evaluations. It used **1,280 focused
continuations** and **110,160 new training actions** including state harvest.
No focused continuation or complete boot game reached stage 2. Median
score gain from these closer states was **1,520**, with maximum **2,290**;
these values cannot be compared directly with the earlier-start trial's
gain because the starting score differs. Candidate returns often differed
by only a few points, so the 20-point score-scale floor reduced updates.
The final ten-game boot mean was **9,827**, median **10,320**, best **10,480**;
all stage 1. The global 10,480-point verified replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-66-late-coordinate \
  --harvest-games 12 --harvest-seed 71000 --lookback 64 \
  --directions 20 --snapshots-per-direction 4 --sigma 5 --step-size 5 \
  --coordinate-bias --return-std-floor 20 --generations 8 \
  --seed 81 --eval-every 1
```
