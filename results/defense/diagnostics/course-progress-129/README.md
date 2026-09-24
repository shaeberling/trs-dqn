# Repeated loss occurs early in the original stage-one stream

This [forensic report](report.json) is deliberately separate from learning.
It re-executes every physical action, visible screen and displayed-score
increment in two independently verified own-policy replay bundles, then
samples the original game's stage-one stream pointer **only at the observed
visible life-loss event**. The pointer is matched to row boundaries decoded
from the immutable original executable, whose existing static audit verifies
a **126-row** stage-one stream. The same procedure exactly replays all 48
previously archived own-life source continuations from twelve complete
neural games. Every source NPZ hash and both replay bundle hashes are checked.

| Verified own source | Lives | Decoded stream row at visible loss |
| --- | ---: | --- |
| [Confirmed run-121 replay](../../training/ppo-continue-121/fresh-confirm-selected-replay/replay.html) | 4 | 34, 33, 33, 34 |
| [Random-phase run-128 replay](../../training/ppo-random-phase-key-128/fresh-selected-replay/replay.html) | 4 | 34, 34, 34, 34 |
| [Older own-loss source archive](../../training/ars-score-gated-69/run/own-loss-states/) | 48 | row 28: 1; 31: 1; 32: 8; 33: 6; 34: 32 |

Thus the repeated 2,620–2,640-point loss cluster is **not close to
exhausting the original stage-one stream**: at the visible marker, the
decoder has consumed roughly row 33 or 34 of 126 in the strong selected
replays. The decoder may render ahead of the ship, and visible loss can lag
physical collision. This is **not** a claim that the ship safely passed 34
obstacle rows, an exact collision location, a passable route, or a measured
fraction of physical stage completion. Score alone did not reveal this
scale; it remains the only training reward.

The native pointer at `0x819B/0x819C` is a **forensic oracle** used here
only to audit game structure and interpret the repeated failures. It is
never supplied to the policy, reward, curriculum, training selection,
evaluation action choice, or replay promotion. No model weights changed.
The saved [source](source.py) and three focused regression tests reproduce
the exact row mapping and native-loss checks. The protected best replay
remains a stage-one loss, not a mission success.
The full **481-test** regression suite passed after this standalone probe
was added.

```sh
venv/bin/python -m rl.defense_course_progress_probe \
  results/defense/training/ppo-continue-121/fresh-confirm-selected-replay \
  results/defense/training/ppo-random-phase-key-128/fresh-selected-replay \
  --source-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-course-progress-reproduction
```
