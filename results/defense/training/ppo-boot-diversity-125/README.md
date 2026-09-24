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

The planned run completed **1,048,576** new base actions, **305** new
complete boot games and **1,331** restored segments in about 748 seconds.
The allocation change therefore produced roughly 2.6 times as many boot
games as run 124's 116, without eliminating own-loss practice. Its eight
fixed ten-game means were **10,405 / 10,432 / 10,245 / 10,472 / 10,444 /
10,466 / 10,480 / 10,456**. The seventh full optimizer/RNG checkpoint at
counter **1,966,080** was selected. No training or validation game entered
stage two. The [run archive](run/) retains every checkpoint, optimizer/RNG
state, metrics, local artifacts and exact source snapshot. The full
**471-test** suite passed before launch; no code changed for this trial.

The predeclared [128 fresh matched complete games](comparison.json), seeds
603600–603727, averaged **10,366.02** for the selection versus
**10,470.23** for the confirmed run-121 parent. Paired wins/losses/ties
were **18/37/73**, and exact 10,480 scores were **88/128** versus
**104/128**. The selection had four scores below 9,000 versus none for
the parent. All **256** fresh games stayed in stage one. Thus the perfect
ten-game fixed score did not generalize; this is not a new score parent.

The selected [native-verified local replay](fresh-selected-replay/replay.html)
has **2,562** learned decisions and four 2,620-point losses at the same
right-opening obstacle. The [read-only loss frames](../../diagnostics/boot-diversity-125-losses/README.md)
show 36 sampled continuation choices overall, but only 1/2/0/1 in the
four pre-loss windows. A much larger share of fresh boot trajectories did
not create a verified escape behavior; repeating this allocation alone is
not supported by its independent results. The confirmed run-121 score
parent and global verified best remain protected.
