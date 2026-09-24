# Stronger symmetric key-factor exploration

Run 123 tested direction-neutral per-life key-factor noise standard deviation
**1** from the confirmed run-121 score parent. It matched that parent's
stage-one score consistency but did not reach stage two in training or
evaluation. Its selected frozen policy still lost four 2,620-point lives
at the recurring right-opening barrier. This does not prove that stronger
coherent motor exploration is ineffective; the first trial used only 112
new complete boot training games.

Run 124 resumes **the same frozen run-121 full model/optimizer/policy RNG**
at counter 1,048,576, changing only key-factor noise standard deviation
from 1 to **2** relative to run 123. Each life draws zero-mean symmetric
factors for known physical keys; related commands share a factor, left and
right have identical distributions, and the network still selects every
action from raw visible screens. This is training-only exploration, with
unchanged displayed-score reward, no stage-based reward, no route rule,
no demonstration and no hidden game state. Unperturbed evaluation and
native replay use the same learned policy as before.

The bounded first gate is **1,048,576 new base actions**, ending at
counter **2,097,152**, with eight full optimizer/RNG checkpoints and
complete ten-game evaluations every 131,072 actions. Select by highest
fixed mean, earliest exact tie. Any observed stage-two or mission result
requires independent native replay verification. Otherwise, only a selected
fixed mean of at least **10,450** triggers 128 untouched matched complete
games against the run-121 score parent and run-123 factor-noise-1 selection.
If it misses that gate, archive the full run as negative. A score tie at
the 10,480 stage-one ceiling never replaces the global verified best.
This is a distinct exploration-strength test, not a claim that running
longer by itself will beat the visible barrier.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-key-noise-strong-124 \
  --artifacts runs/defense-ppo-key-noise-strong-124/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --policy-bias-noise 0 --policy-key-noise 2 \
  --steps 2097152 --eval-every 131072
```

The planned run completed **1,048,576** new base actions, 116 additional
complete boot training games and 4,172 restored segments in about 731
seconds. Its eight fixed ten-game means were **9,904 / 10,292 / 10,210 /
10,434 / 10,462 / 10,478 / 9,736 / 9,984**. The sixth complete
optimizer/RNG checkpoint at counter **1,835,008** was selected. No training
or validation game entered stage two. The full [run archive](run/) retains
every checkpoint, optimizer/RNG state, metrics and local artifacts; exact
trainer, environment, exploration, PPO, loader and snapshot sources are
preserved beside it. The full **471-test** suite passed before launch.

The predeclared [128 fresh matched games](comparison.json), seeds
603400–603527, averaged **10,402.34** for the selection, **10,456.88** for
the confirmed run-121 score parent and **10,473.59** for the weaker-noise
run-123 checkpoint. Paired wins/losses/ties were 18/17/93 versus run 121
and 19/19/90 versus run 123. The mean deficit is affected by a tail of
roughly 9,710–9,730-point run-124 games, despite 109/128 exact 10,480s;
none of the **384** fresh games reached stage two. This is not a confirmed
improvement. The score parent and shared verified best are unchanged.

The selected [native-verified local replay](fresh-selected-replay/replay.html)
has **2,500** learned decisions and four 2,620-point losses at the same
right-opening obstacle. [Loss-frame inspection](../../diagnostics/key-noise-strong-124-losses/README.md)
found only 10 selected `CONTINUE_PREVIOUS` choices, none in the four
pre-loss 64-action windows. Increasing symmetric key noise alone did not
create a learned escape behavior; an identical longer run is not supported
by these complete-game comparisons.
