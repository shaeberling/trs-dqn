# Diverse own-screen resets near visible losses

Three new PPO exploration tests (stronger per-life key noise, more full-boot
workers, and within-life key redraws) all ended at the same stage-one
obstacle. The [frozen score critic](../../diagnostics/score-value-barrier-01/README.md)
expects little additional displayed score after the late 1,500-point jump
on its failed trajectories. This motivates a different legal exploration
lever: preserve a wider variety of states that the learner itself reached
before visible loss. It is related to the archive/return/explore principle
in [Go-Explore](https://arxiv.org/abs/2004.12919), but it is not a
reimplementation of that algorithm or evidence that a passable path exists.

Run 127 resumes the **same full model/optimizer/policy RNG** from the
independently confirmed [run-121 score parent](../ppo-continue-121/README.md)
at counter **1,048,576**. It keeps the original four complete-boot workers,
twelve own-loss reset workers, 128-action lookback, 21-choice learned
categorical PPO, per-life bias noise, original 100,000-T-state action
cadence, four raw visible screen frames, displayed-score reward and
visible life boundaries. Only the **training reset archive** changes from
16 score bins × four states to at most **128 distinct coarse visible-screen
cells × one own state**. The cells exclude the HUD, use a fixed screen-only
fingerprint and bounded bottom-k hash admission. A restored state is an
opaque state this same learner actually visited 128 decisions before its
own visible ship loss; no game internals are decoded, and no route,
direction, collision bonus, demonstration or action override is provided.
Evaluation always starts from the original boot with no archive. Screen
diversity is not guaranteed to mean course progress.

The first bounded gate is **1,048,576 additional base actions**, ending at
counter **2,097,152**, with eight complete ten-game fixed-seed checks and
full optimizer/RNG checkpoints every 131,072 actions. Select highest fixed
mean, earliest exact tie. Independently replay-verify any observed stage-two
or mission result. Otherwise, only a selected fixed mean of at least
**10,450** triggers 128 untouched matched complete games on seeds
**604000–604127** against the confirmed run-121 parent; below that gate,
archive as negative. An equal 10,480 stage-one best never replaces the
protected global replay. Record actual archive occupancy and complete boot
games; do not assume equal occupancy from capacity.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-screen-frontier-127 \
  --artifacts runs/defense-ppo-screen-frontier-127/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --curriculum-cells screen --curriculum-bins 128 \
  --curriculum-per-bin 1 \
  --steps 2097152 --eval-every 131072
```

Before the full trial, the [native smoke](smoke/) completed 49,152 new
actions, 16 boot games and 42 restored segments. It logged 85 own-loss
archive events containing 31 distinct screen cells and independently
verified a 2,546-action local replay; its two-game check stayed in stage
one. The full **472-test** regression suite had passed before launch.

The production run then completed **1,048,576** new base actions, **114**
new complete boot games and **5,472** restored segments in about 746
seconds. Its eight fixed ten-game means were **10,480 / 10,462 / 10,222 /
10,194 / 10,440 / 10,460 / 10,224 / 10,446**. The first full
optimizer/RNG checkpoint at counter **1,179,648** was selected. No
training or validation game reached stage two. The [complete run archive](run/)
retains every checkpoint, optimizer/RNG state, metrics, local artifacts
and exact source snapshot.

The log encountered **170 distinct pre-loss screen fingerprints** over
time. The latest terminal inventory for every worker held **128 cells**,
the configured maximum; this is actual occupancy, not inferred from
capacity. Reserved boot-only workers still collected/shareable own states
but did not restore from the archive. These facts verify that the new reset
selection was exercised, not that cells corresponded to deeper progress.

On the predeclared [128 fresh matched complete games](comparison.json),
seeds 604000–604127, selected/confirmed-parent means were **10,402.42 /
10,475.86**. The selection had **24 paired wins, 14 losses and 90 ties**
and more exact 10,480s (**114/128** versus **103/128**), but three scores
below 9,000 versus none for the parent produced a **−73.44-point mean**.
All **256** fresh games stayed in stage one. Neither the paired-score
detail nor a fixed-seed ceiling is evidence of passage or a confirmed
new parent.

The selected [native-verified local replay](fresh-selected-replay/replay.html)
reproduces **2,563** learned decisions and four 2,620-point lives at the
same obstacle. [Loss-frame inspection](../../diagnostics/screen-frontier-127-losses/README.md)
found zero sampled continuation choices across the full frozen game.
The diverse own-screen archive did not produce a verified escape
behavior; the run-121 score parent and global verified best remain
protected. Repeating this exact archive setting longer is not supported
by its independent stage and score results.

A subsequent [exact-prefix timing diagnostic](../../diagnostics/gate-timing-127/README.md)
shows why the repeated loss is not simply a missing RIGHT command: an
early held RIGHT loses before the usual score plateau, while later held
RIGHT reaches toward but still misses the visible far-right opening.
Those intervention actions were not used to train or promote a model.
