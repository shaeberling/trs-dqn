# Selected grouped-duration replay still stops early

This forensic-only check reexecuted all **2,525** neural actions, score
rewards and visible screens of the already frozen
[14,680,064-action selected replay](../../training/ppo-grouped-duration-fresh-163/milestone-000014680064/verified-replay/replay.html)
from the original game boot. Only after each visible life loss did it read
the unchanged game's private stage-one stream pointer. The four losses
occurred at decoded rows **33, 34, 34, 34 of 126**. Displayed cumulative
scores were **2,470 / 4,940 / 7,460 / 9,930**.

The [report](report.json) pins the model, trace, game and probe source
hashes. Its hidden pointer is diagnostic only: never a policy input,
reward, curriculum target, checkpoint selector, action source or model
update. The read occurs at visible loss bookkeeping, so it does not
identify an exact collision instant or prove that passage is impossible.
The stronger fixed-game mean therefore still does not establish progress
past the recurring obstacle.

Reproduce into a new directory with:

```sh
venv/bin/python -m rl.defense_course_progress_probe \
  results/defense/training/ppo-grouped-duration-fresh-163/milestone-000014680064/verified-replay \
  --output runs/defense-grouped-duration-163-course-progress-14m-reproduction
```
