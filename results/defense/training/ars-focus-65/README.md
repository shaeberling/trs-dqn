# Symmetric search of individual learned action preferences

This in-progress trial searches all 20 action-head biases one at a time,
with both positive and negative changes of the same size. Every generation
includes every command exactly once in a random order. It leaves the
screen encoder, value head and learned feature-to-action weights fixed.
It begins from 48 verified states the frozen strong policy reached itself,
256 decisions before visible ship losses. Training uses only actual displayed
score gained until the next visible life/stage boundary. No direction or
action is preferred by the trainer.

Complete boot games test every generation, and a successful mission from
boot triggers independent replay verification and an automatic stop. All
state captures are opaque, used only to reset training episodes, and never
fed to the policy. Running states and metrics are under
`runs/defense-ars-focus-65-coordinate/`; archive them here once finished.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-65-coordinate \
  --harvest-games 12 --harvest-seed 71000 --lookback 256 \
  --directions 20 --snapshots-per-direction 4 --sigma 5 --step-size 5 \
  --coordinate-bias --return-std-floor 20 --generations 8 \
  --seed 81 --eval-every 1
```
