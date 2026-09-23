# Focused self-play on the recurring loss approach

This completed first focused calibration used the strong neural policy to play new
complete training games, captures opaque emulator states 128 actions before
its own visible life losses, and verifies each state by restoring it and
replaying recorded actions, screens and score increments. The recorded
actions are verification evidence only; training uses the saved state and
fresh actions sampled from each candidate neural policy. Candidate pairs
share the same state and action-sampling seed. Only visible score gained until
the next life or stage boundary determines the ARS head update.

Ordinary complete games from boot select and verify any globally better
replay. Captured native state is never a policy input, and evaluation never
resets to a training state. The 48 own states from 12 new full games, all
population plans, model checkpoints, RNG and results are preserved in `run/`.

Across eight generations, **1,024 focused segments** and **157,897 total new
actions** (including state harvest) yielded no stage-2 passage, either from
focused continuations or complete boot validation. Focused segment score gain
had median 2,410, maximum 2,460, and later rounds nearly collapsed to the
same outcome. The generation-1 10-game mean was 10,238; generation 8 finished
at **9,530**, median 10,345, best 10,480, all stage 1. Initial mean was 9,981.
The existing verified 10,480-point global best therefore remains unchanged.
These 10 boot seeds are reused selection seeds, not fresh test games.

On the 48 captured pre-loss training screens, the frozen parent gave a mean
41.7% probability to pure movement actions but a median of only 0.2%; 28
states were below 10%. Pure movement choices in the actual following 128
actions averaged about 59–62% by life, with right and left choices roughly
balanced. This read-only measurement does not prove displacement or a causal
collision, but it explains why many small head perturbations may keep the
same broad behavior. A wider unbiased perturbation is the next trial.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-61-loss-snapshots \
  --harvest-games 12 --harvest-seed 71000 --lookback 128 \
  --directions 16 --snapshots-per-direction 4 --sigma .005 \
  --step-size .002 --generations 8 --seed 81 --eval-every 1
```
