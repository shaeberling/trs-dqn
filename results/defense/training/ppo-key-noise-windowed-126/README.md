# Within-life, symmetric key-factor exploration

The repeated-loss [native counterfactual diagnostic](../../diagnostics/boot-diversity-125-losses/README.md)
found that holding a single physical command from a learned-policy state
does not pass the stage-one obstacle. A fixed RIGHT command changes visible
loss timing differently depending on whether the model is 192, 128 or 64
decisions before its observed loss. This does **not** identify a route or
provide a demonstration. It suggests a narrower exploration hypothesis:
the previous per-life factorized keyboard bias may be too temporally rigid
to sample different coherent movement phases within one life.

Run 126 resumes the same full [run-121 score parent](../ppo-continue-121/README.md)
used by the previous [stronger-noise run 124](../ppo-key-noise-strong-124/README.md).
It changes only the **training-only key-factor redraw period**, from
visible life/episode boundaries alone to those boundaries **or every 32
own actions per worker**. Key-factor standard deviation remains **2**,
with zero-mean left/right-symmetric draws; all other policy, optimizer,
screen, reward, action duration, boot/restored worker split and evaluation
settings match run 124. The fixed count is not a score, obstacle cue,
route rule or policy input. PPO records the actual noisy likelihood for
each action, including draws within a rollout. Frozen evaluation uses the
unchanged learned policy, with no perturbation or scripted action override.
No counterfactual diagnostic action enters training.

The bounded first gate is **1,048,576 new base actions**, ending at counter
**2,097,152**, with eight complete ten-game fixed-seed validations and full
optimizer/RNG checkpoints every 131,072 actions. Select by highest fixed
mean, earliest exact tie. Any apparent stage-two or mission observation
requires independent native replay verification. Otherwise, only a
selected fixed mean of at least **10,450** triggers 128 untouched matched
complete games on seeds **603800–603927** against run 121 and run 124.
Below that gate, archive the negative result. A 10,480-point stage-one tie
never replaces the protected global verified best.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-key-noise-windowed-126 \
  --artifacts runs/defense-ppo-key-noise-windowed-126/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --policy-bias-noise 0 --policy-key-noise 2 \
  --policy-key-noise-interval 32 \
  --steps 2097152 --eval-every 131072
```

Before production, the full **472-test** regression suite and all **14**
focused noise tests passed. A separate [single-update native smoke](smoke/)
resumed the same parent for 4,096 actions with interval 32, completed a
two-game frozen check at 10,480 mean (stage one), and independently verified
a 2,554-action local replay. Its artifacts are excluded from the shared
best collector and are not the predeclared production result.
