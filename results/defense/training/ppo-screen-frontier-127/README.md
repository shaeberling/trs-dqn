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
