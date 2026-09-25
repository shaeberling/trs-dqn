# Movement-only pilot survives longer, but is still far from the old barrier

This post-freeze forensic audit reexecuted all **1,703** learned actions,
displayed rewards and visible screens of the independently
[verified 380-point fresh replay](../../training/ppo-movement-only-164/pilot/fresh-comparison/movement-620200-replay/replay.html)
from original boot. Only after each visible life loss did it read the
unchanged game's private stage-one stream pointer. The four losses occurred
at decoded rows **15 / 15 / 16 / 16 of 126**, with displayed cumulative
scores **80 / 180 / 280 / 380**.

Thus the fresh movement-only policy's matched score/action-count gain over
its no-learning control is real, but it has **not yet approached** the
strong firing policies' recurrent row-33/34 failure. The lower score is
not directly comparable because movement-only play cannot earn firing
bonuses. A longer extension is justified by the predeclared fresh survival
gate, not by a claim of course passage.

The [report](report.json) pins the game, trace, model and probe hashes.
The same audit on the frozen no-learning control's separate
[verified best replay](../../training/ppo-movement-only-164/pilot/fresh-comparison/baseline-620200-replay/replay.html)
found rows **14 / 15 / 12 / 16** at its four losses; its
[report](baseline-report.json) is preserved. These are the best-score
replays from **different seeds**, not a paired course-progress estimate.
They show at most a modest shift in sampled stream rows despite the
learner's clear paired score and game-length gain.
The hidden pointer is read only at visible loss bookkeeping, not the exact
collision instant, and is never a policy input, reward, curriculum target,
action source, checkpoint selector or model update. Reproduce with:

```sh
venv/bin/python -m rl.defense_course_progress_probe \
  results/defense/training/ppo-movement-only-164/pilot/fresh-comparison/movement-620200-replay \
  --output runs/defense-movement-only-164-pilot-course-progress-reproduction
```
