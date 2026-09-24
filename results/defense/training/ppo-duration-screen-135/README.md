# Screen-diverse returns with joint key-duration exploration

The selected [joint key-duration learner](../ppo-duration-joint-134/README.md)
sampled 473 sixty-four-step options before its last progress report and
averaged near the stage-one score ceiling on fixed games. Yet its fresh
[native-verified replay](../../diagnostics/joint-134-course-progress/README.md)
still loses four lives at original course rows 33–34 of 126. Its second
fresh score comparison did **not** confirm a new score parent, so this
follow-up tests a different exploration lever rather than claiming a
stronger frozen policy.

Resume its **exact selected full optimizer/policy/noise RNG state** at
917,504 actions. Keep the learned joint physical-key/duration network,
per-life symmetric key-factor noise 2 and duration-factor noise 4, PPO
optimizer rate 2.5e-5, four boot-only workers, 128-action own-life-loss
lookback, four raw screen-frame input, and displayed-score-only reward.
Change only the training reset archive from 16 score bins × four own
states to at most 128 distinct coarse visible-screen cells × one own
state. The screen fingerprint excludes the HUD and only selects which
of this same learner's actually reached *opaque* states is revisited;
it supplies no route, direction, hidden pointer, collision oracle,
extra reward or demonstration. Evaluation always starts from game boot.
This combination was not tested by the earlier screen-diverse one-step
continuation or the joint-noise score-bin run individually.

Train **1,048,576 additional base actions**, to absolute counter
**1,966,080**, with eight fixed ten-game complete-game checks every
131,072 actions on seeds 10000–10009. Preserve every full optimizer/RNG
checkpoint. Select the earliest checkpoint with the highest stage rank,
breaking stage ties by fixed-game mean. Any stage-two candidate must be
independently native-replayed and freshly confirmed before promotion. If
all games stay in stage one, only a fixed mean at least **10,460** merits
a 64-game matched score check against the frozen input checkpoint and
the established ordinary-action parent on untouched seeds **607000–607063**;
it still cannot replace the protected global replay on score alone.
If two consecutive fixed checks average below 5,000, archive the run as
collapsed and stop it early. Report actual distinct screen-cell occupancy,
not just configured capacity.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-screen-135 \
  --artifacts runs/defense-duration-screen-135/artifacts \
  --resume results/defense/training/ppo-duration-joint-134/run/step-000000917504 \
  --curriculum-cells screen --curriculum-bins 128 --curriculum-per-bin 1 \
  --steps 1966080 --eval-every 131072 --eval-games 10 --eval-envs 10
```
