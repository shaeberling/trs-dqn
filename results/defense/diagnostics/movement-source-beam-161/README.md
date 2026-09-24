# Screen-only continuation test from the new movement-heavy learned life

The fresh balanced-fire learner's independently native-verified
[10,220-point replay](../../training/ppo-balanced-fire-from-scratch-160/milestone-replay-000007344128/replay.html)
uses more ordinary movement near its losses than the protected best, but
still loses its first life at 2,550 points near the same two-opening region.
The prior [frame-200 screen beam](../screen-beam-early-159/README.md)
searched from a different, mature fire-heavy learned life. This diagnostic
holds that search method fixed and changes only the **own verified source
trajectory** to test whether the new approach admits a different local
continuation.

The beam starts at exact original-boot frame **200** of the new frozen
first life, holds one of the twelve distinct stage-one physical commands
for four actions per layer until frame 320, then branches every action to
frame **428** or visible first-life loss. Beam width is **256**. Selection
uses displayed score, visible ship-column bins and raw-screen-plus-last-
action diversity; the original learned path is retained while alive.
It does not use the game's hidden obstacle pointer, collision flags,
course data or a preferred movement direction. The seed, command set,
beam size, timing and selection rule match the earlier frame-200 search.
The only source-specific values are its verified first visible loss at
frame **414** and score **2,550**. The discovery score gate remains
**strictly above 2,620**, the established first-life ceiling, not merely
above the new source's lower score. A later stage, mission, or visibly
alive frame 428 also triggers a discovery. Each discovery must reproduce
every action, score increment and visible screen from original boot.

This is a bounded **diagnostic**, not a trained policy, expert example,
demonstration, additional reward or replay promotion. Searched actions
must never enter PPO training or become the neural replay. Failure would
bound only this source, beam and horizon, not prove the game impassable.
The protected global best and the live matched-control training are
unchanged. The output is under the ignored `runs/` directory until
verified and reviewed; the search has its own disk guard.

The five focused source/search tests pass, including an exact two-layer
native smoke from the new replay and unchanged old-source behavior.
The full Metal-enabled repository regression suite passes **537 tests**.

```sh
venv/bin/python -u -m rl.defense_screen_beam \
  results/defense/training/ppo-balanced-fire-from-scratch-160/milestone-replay-000007344128 \
  --output runs/defense-ppo-balanced-fire-160/movement-source-beam-161 \
  --beam 256 --anchor 200 --fine-cadence --history-key --side-fire
```

## Completed result

The [completed search](run/report.json) expanded **367,860** original-
emulator branches over **124** layers and ended at frame **414**, when no
retained first-life branch remained without visible loss. Maximum displayed
first-life score was **2,550**, equal to the new source; the last visibly
alive retained branch was at frame **413**. There was no score above the
predeclared 2,620 gate, no later stage, no mission, and no frame-428
survivor. At frame 388, the farthest readable ship glyph among selected
paths was column **30**. This does not identify an exact collision point
or show every route the pruned beam might have missed.

The full [layer history](run/layers.jsonl.gz) is compressed to avoid
retaining a duplicate 12 MiB text file. Decompression was compared
byte-for-byte with the finished run, and its SHA-256 is
`c32867115f5214dc0028b6aff9ee6d8fe8bfe2e97df78fb3eb996c65b25d2ea3`,
matching the report. The frozen source checkpoint hash in the archived
[configuration](run/config.json) matches the verified replay. The
[exact source script](run/source.py) and final [status](run/status.json)
are retained. The process exited normally; no discovered actions were
exported to a learner.

This negative result shows that swapping the earlier fire-heavy source
for the new movement-heavy one does **not** make the same score/diversity
beam find a passage. It does not establish that stage one is impossible;
beam pruning and the 428-frame horizon remain material limitations. It
does not justify simply repeating this search topology at larger width
without a distinct hypothesis. The live matched-control experiment and
protected best replay remain unchanged.
