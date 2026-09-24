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
