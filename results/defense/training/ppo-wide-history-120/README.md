# Adapt the score parent to wider visible-frame history

The strongest independently confirmed score parent (run 119, counter
9,718,528) still loses around the same right-opening barrier. In an
evaluation-only matched-seed probe of its frozen weights, doubling the
screen-frame spacing at unchanged 100,000-T-state actions reduced mean score
from 10,456.88 to 5,202.50 over 32 complete games; strides 4 and 8 fell
further. This is distribution shift, **not** proof that short history causes
the barrier. It motivates measuring whether ordinary screen/score PPO can
adapt to stride 2 without losing its established play.

Run 120 resumes the exact run-119 full model, optimizer and policy/noise RNG
state, changing **only observation stride from 1 to 2**. The input remains
four raw visible 16x64 frames, now spaced across six rather than three
ordinary actions. All own-visible-loss curriculum, per-life training-only
actor noise, fixed evaluation seeds, action duration and score-only reward
are inherited. Emulator episodes restart from boot on optimizer resume.

The planned first gate is four saved checkpoints over **524,288 new actions**,
at 131,072-action intervals. Complete ten-game unperturbed validations may
select a provisional checkpoint by highest mean score, earliest exact tie.
The gate succeeds if any game enters stage two or completes the mission, or
if fixed validation recovers at least 10,450 mean points. If none does,
archive the full negative run without presenting a new score parent. If it
does, compare the selected checkpoint with the frozen run-119 parent on at
least 64 new matched complete games before claiming an improvement. Any
native mission claim requires independent complete-game replay verification.
No frozen timing probe can promote the global best replay.

The full bounded run finished cleanly at counter **10,242,816**: 524,288
new base actions, four complete ten-game fixed-seed evaluations, all stage
one. Means were **10,349 / 10,323 / 10,394 / 10,452**. The final checkpoint
met the predeclared 10,450 recovery gate and was therefore selected for a
fresh comparison. Its full model/optimizer/RNG state, all four saved
checkpoints, metrics, source snapshots and local artifacts are under
[run/](run/). The short first gate did not establish barrier passage.

The independent [128 matched complete-game comparison](comparison.json) on
seeds 602100–602227 reverses the fixed-seed promise: selected stride-2
policy **10,344.77** mean versus the unchanged stride-1 parent **10,455.63**,
a **−110.86**-point difference. Selected improved 23 games, lost 80 and tied
25. It had five sub-9,000 games against the parent's zero, and 17 exact
10,480 games against the parent's 66. Neither policy entered stage two or
completed the mission. Thus the wider spacing is *not* a new score parent;
further training of the identical configuration is not justified by this
test. The selected checkpoint's [fresh best-effort replay](fresh-selected-replay/replay.html)
was independently verified for **2,539 learned actions**, scoring 10,480 in
stage one; it is kept locally and does not replace the global best.

The comparison is between each model at its own trained observation spacing,
which is the appropriate deployment comparison but is not a controlled
single-parameter attribution of why either learner scores as it does. The
frozen stride test remains a distribution-shift diagnostic only. No
diagnostic replay was used to update weights, reward or choose actions.

Frozen diagnostic records: [stride 1](../../validation/long-noise-119-frozen-stride-1-602000.json),
[stride 2](../../validation/long-noise-119-frozen-stride-2-602000.json),
[stride 4](../../validation/long-noise-119-frozen-stride-4-602000.json),
[stride 8](../../validation/long-noise-119-frozen-stride-8-602000.json).

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-wide-history-120 \
  --artifacts runs/defense-ppo-wide-history-120/artifacts \
  --resume results/defense/training/ppo-persistent-noise-long-119/milestone-000009718528 \
  --observation-stride 2 --steps 10242816 --eval-every 131072
```
