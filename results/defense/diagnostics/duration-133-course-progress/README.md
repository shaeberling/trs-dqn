# The new duration policies still fail at the same early obstacle

The [forensic report](report.json) re-executes two independently verified
learned-policy replay bundles byte-for-byte on the original emulator. It
samples the immutable game's stage-one stream pointer only when each visible
life loss has already occurred; this hidden value is **never** a training
observation, reward, curriculum target, policy decision or replay-selection
criterion. No model was updated from this report.

| Learned policy | Native-verified commands | Decoded stream row at four visible life losses |
| --- | ---: | --- |
| Conservative key-duration PPO | 2,540 | 33, 33, 34, 34 |
| Per-life direction-neutral duration exploration | 2,561 | 33, 33, 33, 33 |

The original first-stage stream has **126 rows**. These loss locations agree
with the earlier [independent course-progress audit](../course-progress-129/README.md),
despite the new policies sampling learned multi-action holds. The stream
pointer is not a collision-time sensor or proof of a passable route, so this
only localizes the repeated failure. It does show that high total score is
not near a stage clear: the four lives repeatedly collect points before the
same early segment.

Reproduce with `venv/bin/python -m rl.defense_course_progress_probe` and the
two archived `artifacts/best` bundles in
[the duration experiment](../../training/ppo-duration-133/README.md).
