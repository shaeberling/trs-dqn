# Matched-seed check of the recurring stage-one loss

The independently confirmed generation-five parent and the generation-two
per-command candidate were each replayed from a fresh boot on the **same
eight training-only seeds, 490000–490007**. All sixteen complete four-life
game traces were reproduced from saved weights, actions, rewards and
screens before these 64 per-life samples were archived. The exact collector
script is [source.py](source.py); its hash and each source checkpoint hash
are recorded in [parent/index.json](parent/index.json) and
[candidate/index.json](candidate/index.json). The `parent/` and `candidate/`
folders retain the verified rendered-screen histories at 128 / 96 / 64 /
32 decisions before each visible life-loss marker, plus per-life displayed
score and frame. They contain no native state, action targets or hidden
collision labels.

| Policy | Complete games | Lives scoring 2,500–2,650 before visible loss | Stage two reached |
| --- | ---: | ---: | ---: |
| Confirmed parent | 8 | 31 / 32 | 0 |
| Generation-two candidate | 8 | 32 / 32 | 0 |

The parent had one early 1,060-point loss. On that seed the candidate's
complete-game score was 1,580 points higher. This one pair accounts for
94% of the candidate's 1,680-point aggregate advantage across these eight
games; across the other seven pairs the gain was just 100 points in total.
Thus the candidate can appear better on a small score sample by avoiding
an unusual early loss while **still failing at the same recurring late
stage-one score band in every life**. A separate, larger [64-seed paired
check](../../training/ars-action-row-99/paired-generation-2.json) found a
negative mean gain, so these eight games are a diagnostic, not a selection
or promotion result.

Displayed score and visible life-loss timing are not physical collision
coordinates. The screen histories and earlier
[barrier atlas](../ars-69-barrier-atlas/README.md) repeatedly show a
right-side opening, but they do not prove a wall hit rather than a
projectile hit. No weights, rewards, policy inputs, or global replay were
changed by this diagnostic.
