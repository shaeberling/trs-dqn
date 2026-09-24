# Screen-conditioned timing search 157

This bounded **diagnostic** tests whether a verified learned policy's recurring
2,620-point first-life loss can be escaped by changing effective stage-one keys
at the visible timing window. The source is the original-boot, independently
verified fresh selected replay from `ppo-duration-credit-136`, seed 607200. Its
first visible life loss is at action 407. Search branches from an exact replayed
frame-320 state, tests all ten effective stage-one keys at each layer, and
continues to frame 428 or the first observed life loss. Durations shorten to one
action around frames 336–348. The beam retains up to 128 distinct raw visible
screen cells, half selected for balanced readable ship-column bins and half for
displayed score. The source policy path is retained while alive.

The native emulator snapshots are opaque reset machinery; no RAM contents,
route bytes, or hidden state inform branch ranking. A branch exceeding 2,620
points, reaching stage two, or remaining visibly alive at frame 428 must be
reexecuted from original boot with every action, reward, and screen matched.
Such a route is **not** a learned policy, not a training demonstration, and not
eligible to replace the published best replay. A negative result is evidence
only for this beam width, anchor, schedule, selection rule, and source seed.

Run with:

```sh
venv/bin/python -m rl.defense_screen_beam \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/screen-beam-157/run --beam 128
```

Two predeclared follow-ups test whether earlier positioning or tighter action
timing changes this result. The first starts at frame 280, using ten extra
four-action layers. The second also uses one-action layers from frame 320,
retains visibly identical cells with distinct last actions (control timing
can be hidden by a single screen), and widens the beam to 256. Both still
rank only visible screens and displayed score, never emulator state bytes.

```sh
venv/bin/python -m rl.defense_screen_beam \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/screen-beam-157/early-run \
  --beam 128 --early-anchor
venv/bin/python -m rl.defense_screen_beam \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/screen-beam-157/fine-history-run \
  --beam 256 --early-anchor --fine-cadence --history-key
```

## Completed results

All three searches reproduced the verified source prefix and finished their
predeclared scope. The [frame-320 run](run/report.json) expanded **30,920**
branches; the [frame-280 run](early-run/report.json) expanded **43,570**;
and the [fine, action-history run](fine-history-run/report.json) expanded
**237,700**. In each, the highest first-life visible score was **2,620**.
No branch triggered the stage-two, mission, greater-score, or frame-428
verification gate. Selected branches lost their first visible life by frame
408, 408, and 407 respectively. That visible event can lag a collision;
the report's `max_alive_frame` means *no visible decrement yet*, not
physical survival.

The supplied source's original screens show a centre opening at columns
**21–30** on row 14 at frame 348, with a readable ship glyph at column
**23**. At frame 388, a subsequent wall occupies columns **1–50** on row
13, leaving the opening to its right; the source glyph is at column **25**.
The fine run's farthest retained **full-score** readable ship glyph was at
column **39** at frame 388 (the earlier two runs reached 32 and 37).
These are visible-screen coordinates, not exact collision geometry. A
glyph can be absent or occluded, and screen-cell pruning means this is not
an exhaustive proof that the source life cannot be saved.

The result strengthens the training diagnosis: learning to collect the
repeated 2,620 points does not teach the time-sensitive transition between
these two openings. A useful successor should be measured first by actual
passage or stage change on original-boot learned-policy games, with score
and episode length secondary. A searched branch would still not be a
learned policy or admissible demonstration. The protected best weights
and verified replay are unchanged.

The full repository regression suite passed **525 tests** with native Mac
access. Each run archives its exact executed `source.py`, config, complete
selected-node ledger, and report; the later runs were made after the
earlier run's selection implementation was extended.
