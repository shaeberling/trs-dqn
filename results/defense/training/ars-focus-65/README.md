# Symmetric search of individual learned action preferences

This completed trial searched all 20 action-head biases one at a time,
with both positive and negative changes of the same size. Every generation
includes every command exactly once in a random order. It leaves the
screen encoder, value head and learned feature-to-action weights fixed.
It begins from 48 verified states the frozen strong policy reached itself,
256 decisions before visible ship losses. Training uses only actual displayed
score gained until the next visible life/stage boundary. No direction or
action is preferred by the trainer.

Complete boot games tested every generation; any mission would have triggered
independent replay verification and an automatic stop. All state captures
were opaque training resets and never policy inputs. `run/` preserves the
48 verified own states, **1,280 focused continuations**, **346,725 new
training actions**, complete model/RNG checkpoints, per-action plans and
scores, boot validations, and the original PPO optimizer lineage.

No candidate continuation or complete boot game reached stage 2. Median
focused score gain was **2,540** from these 256-decision starts, close to
the maximum **2,560**; the training signal mostly saturated at the familiar
loss. The strongest 10-game mean was **10,388** at generation 2. The final
mean was **10,122**, median **10,370**, best **10,430**, all stage 1.
The globally best verified 10,480-point PPO replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-65-coordinate \
  --harvest-games 12 --harvest-seed 71000 --lookback 256 \
  --directions 20 --snapshots-per-direction 4 --sigma 5 --step-size 5 \
  --coordinate-bias --return-std-floor 20 --generations 8 \
  --seed 81 --eval-every 1
```
