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

The production run completed **1,048,576** new actions, **114** complete
boot games and **4,099** restored segments in about 733 seconds. Its
eight fixed ten-game means were **10,469 / 10,480 / 10,475 / 10,468 /
10,404 / 10,480 / 10,228 / 10,232**. The second full optimizer/RNG
checkpoint at counter **1,310,720** was selected by the earliest-tie rule.
The logged redraw count exceeded **33,000** near the end, confirming that
the within-life schedule operated. No training or validation game reached
stage two. The [complete run archive](run/) preserves every checkpoint,
optimizer/RNG state, metrics and local artifacts; exact trainer,
environment, noise, PPO, loader and snapshot sources are beside it.

On the predeclared [128 fresh matched complete games](comparison.json),
seeds 603800–603927, selected / confirmed run-121 parent / per-life-noise
run-124 selection means were **10,428.05 / 10,445.78 / 10,396.48**.
Against the score parent, the selected model had **16 paired wins, 22
losses and 90 ties**; against the per-life-noise arm it had **15 wins, 21
losses and 92 ties**. All **384** fresh games stayed in stage one. The
positive mean difference over per-life noise is affected by score tails,
not accompanied by more paired wins or any new stage. It is not a new
confirmed parent.

The selected [native-verified local replay](fresh-selected-replay/replay.html)
reproduces **2,583** learned actions and four 2,620-point lives at the
same obstacle. [Loss-frame inspection](../../diagnostics/key-noise-windowed-126-losses/README.md)
found only one sampled continuation choice in the whole frozen game.
Within-life, symmetric exploration did not create a verified escape
behavior. The confirmed run-121 score parent and global verified best
remain unchanged; an identical longer run is not justified by these
fresh complete-game results.
