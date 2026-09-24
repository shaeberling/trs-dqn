# Can two earlier command phases reach the next opening?

The repeated first-life loss occurs after **2,620** displayed points near
original stage-one stream rows 33–34. Existing fixed grids changed
commands from frame 322 or later, and a separate single-RIGHT study could
not move the ship far enough at frame 355 while retaining 2,600 points.
The two visible openings require a more coordinated approach. This bounded
diagnostic tests *earlier* two-command interventions from exact own-policy
frame **280**, then holds RIGHT from frame **355** toward the second
opening visible in the original screen. This is a **screen-informed
diagnostic intervention**, not a neutral neural training method. It uses
no hidden game RAM or course pointer, but its candidate commands are
hand-designed and must **never** be called a learned policy, supplied as
demonstrations, used as PPO targets/rewards, or promoted as best replay.

The sole source is the independently native-verified
[run-136 own complete-game replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html).
Reexecute every original screen and score increment from boot to its
first visible loss at frame **407**; take an opaque same-build emulator
snapshot at frame **280**. Enumerate all **3,600** combinations of:

- own-policy delay 0 / 8 / 16 / 24 actions;
- first physical command from NOOP, eight directions or Space, held
  8 / 16 / 24 actions;
- second independently selected command from the same symmetric ten,
  held 8 / 16 / 24 actions;
- resume the source's recorded commands to frame 355, then hold RIGHT
  until visible life loss, stage change, mission or frame 512.

The two command phases always finish by frame 355. Record real displayed
score and a sometimes-occluded ship glyph column at frames 339 / 355 /
387 / 407. Measure how many branches preserve the source's **2,600**
points at frame 355 and how far right they visibly reach. Continue **all**
surviving candidates, not merely those with high score or a detected
sprite. Stop the full grid early only if a branch exceeds the source's
**2,620 first-life points**, remains visibly alive through frame **512**,
or displays a later stage; reexecute every candidate action/reward/screen
from original boot before reporting such a discovery. A 512-frame survivor
would demonstrate passage of this local barrier, **not** stage-two or
mission completion. Preserve every candidate result and full source
hashes. Run a bounded plumbing smoke and the full repository regression
suite before production. Honor the 5 GiB disk guard.

```sh
venv/bin/python -u -m rl.defense_early_phase_reach \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/early-two-phase-reach-156/run
```

## Completed result

The 100-combination plumbing smoke reproduced the verified source life,
reached frame 355 in 35 branches and retained 2,600 points in ten. The
full repository suite passed **521 tests** before production.

The [full grid](run/report.json) completed all **3,600** predeclared
combinations. An independent read of its [complete outcome table](run/outcomes.jsonl)
confirmed **871** branches reached frame 355 and **69** of those retained
at least the source's 2,600 displayed points while still on the first
life. All 69 had a readable ship glyph, at columns **22–25**; the source
itself was at **25**. No branch exceeded **2,620** first-life points,
survived frame **512**, displayed stage two, or completed a mission. The
longest branch in this family lost by frame **409**; among the 69
full-score waypoint branches, the latest visible loss was frame **405**,
earlier than the source's **407**. No candidate triggered the original-
boot discovery verifier, because none met its predeclared progress gate.
The original-boot source prefix and its native snapshot suffix were
reexecuted byte-for-byte before the grid.

This is a negative result for this **specific** early two-command plus
RIGHT family from one exact own life. It does not prove that the openings
are physically impassable, nor does it justify repeating the same grid
longer. Single-RIGHT prepositioning and later three-phase grids were also
negative; together they suggest the needed approach changes more than
two early fixed phases or requires different timing before frame 280.
No searched action, screen measurement or native state entered training,
checkpoint selection, evaluation-policy action choice or the protected
learned replay.
