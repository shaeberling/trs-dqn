# Exhaustive two-phase test at the repeated visible opening

The original learned-policy replay repeatedly loses when the original
stage-one stream has decoded about 33–34 of 126 rows. Read-only visible
geometry shows two offset openings: from a specific own-play state,
holding a single key toward the second opening either collides earlier
or arrives short. Previous random mutation searches tried thousands of
sequences but did not **exhaust** a small, independently defined set of
two-phase commands. This diagnostic tests that finite set directly,
without selecting actions from a wall parser or hidden course pointer.

Use the independently [native-verified option-credit replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html)
as the sole source. Reexecute it from original boot through first-life
action **322**, capture an opaque same-build emulator state, and verify
the original suffix reaches its visible life loss at action **407** and
2,620 displayed points. From that exact state, test all **7,500**
combinations of:

- delays of 0, 8 or 16 original learned actions;
- a first symmetric physical command from NOOP, eight directions or Space,
  held 4, 8, 16, 24 or 32 actions;
- a second independently symmetric command and hold from the same sets;
- the remainder of the source's own recorded action sequence, to a
  512-action horizon or an earlier visible life/stage boundary.

The delay uses the original policy's already recorded actions, not a
scripted approach. The grid is fixed **before** observing outcomes and
exhausts every command pair equally. The commands are diagnostic
interventions, **not** training examples, actor targets or learned
replays. Only displayed score, visible life survival and visible stage
rank outcomes; snapshots are opaque reset machinery and no hidden game
RAM is read. A stage-two candidate must reproduce its entire screen,
reward and action trace from original boot. A horizon survivor without
a stage-two screen is reported only as a diagnostic, never a win.

First run a 20-candidate fixed-prefix plumbing smoke, then the complete
7,500-candidate grid with the same source. Preserve the full outcome
table, baseline action array, source hashes, exact grid definition,
intermediate status and any boot verification. This is not a new neural
policy and cannot replace the protected best replay. If no candidate
passes, the result bounds this **specific two-phase open-loop family**;
it does not prove the game is impossible or that a screen-conditioned
learner cannot coordinate a longer sequence.

```sh
venv/bin/python -u -m rl.defense_phase_grid \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output runs/defense-phase-grid-145 --anchor 322 --horizon 512
```

The [20-candidate plumbing smoke](smoke/report.json) exactly reproduced
the own-policy baseline: first-life loss after 85 actions from the
frame-322 anchor, 2,620 displayed points, stage one. No smoke candidate
improved it; that prefix is not an outcome estimate for the full grid.
Both focused grid tests and the full **503-test** repository suite passed
before production.
