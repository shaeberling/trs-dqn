# Trial 163's stronger score still reaches the same early barrier

This is a forensic-only audit of the independently
[verified 9,930-point learned replay](../../training/ppo-grouped-duration-fresh-163/milestone-000006291456/verified-replay/replay.html).
The archived [report](report.json) reexecutes all **2,598** of that model's
original-boot actions, checking every score reward and visible frame against
the frozen trace. Only after each *visible* life loss does it read the
original immutable game's private stage-one stream pointer. That hidden
read is never a policy input, training reward, curriculum target, replay
selector, action source, or model update.

The four visible losses occurred at actions **410, 1,144, 1,877, 2,598**
with cumulative displayed scores **2,470, 4,990, 7,460, 9,930**. Every
loss found decoded stream row **34 of 126**. Thus the new grouped-duration
policy learned to score much better than its own early checkpoints, but the
verified game still repeatedly fails in the same early course region as
older score-trained models. The pointer is sampled at visible loss
bookkeeping, not the exact collision moment or a proof of impossibility.

The bundled [source](source.py) and report pin the game, trace and model
hashes. Reproduce with:

```sh
venv/bin/python -m rl.defense_course_progress_probe \
  results/defense/training/ppo-grouped-duration-fresh-163/milestone-000006291456/verified-replay \
  --output runs/defense-grouped-duration-163-course-progress-reproduction
```

This diagnostic does not alter the active 16,777,216-action training run,
its predeclared selection rule, or the protected global best replay.
