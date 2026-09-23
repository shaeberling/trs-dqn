# Coordinate search close to the visible life loss

This running trial matches [trial 65](../ars-focus-65/README.md) in strong
parent, source games, symmetric changes to all 20 action-head biases,
score-only fitness, and complete boot evaluation. It changes only the
own-state rewind from 256 to **64 decisions** before the visible life loss.
All captured states are verified by reexecuting their source actions and
screens; source actions are not training examples. The neural policy sees
only its usual screen history.

The full run is under `runs/defense-ars-focus-66-late-coordinate/` during
training. Archive it here after completion. A complete boot game must show
and verify a new stage or successful mission before any such claim.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-66-late-coordinate \
  --harvest-games 12 --harvest-seed 71000 --lookback 64 \
  --directions 20 --snapshots-per-direction 4 --sigma 5 --step-size 5 \
  --coordinate-bias --return-std-floor 20 --generations 8 \
  --seed 81 --eval-every 1
```
