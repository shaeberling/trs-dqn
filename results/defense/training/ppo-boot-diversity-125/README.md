# More complete-game exploration alongside own-loss practice

The latest symmetric key-noise run completed 1,048,576 new actions but
generated only **116 complete boot games** versus **4,172 restored segments**.
Its 128 fresh-game result regressed and all lives still stopped at the
recurring right-opening barrier. Short reset segments are useful practice,
but a 12:4 restored/boot worker split might undersample complete approaches
to that obstacle. This is a training-allocation hypothesis, not a claim that
more time or a score tie establishes progress.

Run 125 resumes the exact full model, optimizer and policy RNG at counter
**1,048,576** from the independently confirmed [run-121 score parent](../ppo-continue-121/README.md).
It changes **only** the reserved complete-boot worker count **4 → 12** out
of 16; the four remaining workers still use the learner's own visible-loss
snapshots. Training still uses the same per-life 21-action actor-bias noise,
the same learned continuation choice, four visible screen frames, original
displayed-score reward, visible life boundaries, fixed action duration,
PPO settings and original game. No route rule, action override, hidden state,
demonstration or extra reward is introduced. Evaluation always starts from
boot with an unperturbed frozen policy.

The first bounded gate is **1,048,576 new base actions**, ending at counter
**2,097,152**, with eight complete ten-game fixed-seed checks and full
optimizer/RNG checkpoints every 131,072 actions. Select by highest fixed
mean, earliest exact tie. Any stage-two or mission observation requires
independent native replay verification. Otherwise, only a selected fixed
mean of at least **10,450** triggers 128 untouched matched complete games
on seeds **603600–603727** against run 121; below that gate, archive the
negative result without promoting a parent. A score tie at the stage-one
10,480 ceiling never replaces the protected global verified best. This
allocation test is distinct from the prior noise-strength, shorter-lookback
and longer-credit trials.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-boot-diversity-125 \
  --artifacts runs/defense-ppo-boot-diversity-125/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --curriculum-boot-envs 12 \
  --steps 2097152 --eval-every 131072
```
