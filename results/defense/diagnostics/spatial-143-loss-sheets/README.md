# What the learned policies actually show before the repeated loss

The [verified-screen/action report](report.json) compares three frozen,
independently native-verified **selected single-game** replays: the new
[spatial residual](policy-1-losses.png), its [unchanged control](policy-2-losses.png),
and their [option-credit source](policy-3-losses.png). Every replay scores
2,620 on each of four lives and ends in stage one. Panels are unaltered
original video-memory glyphs at 64, 32, 8 and 1 actions before a visible
flash/loss alignment marker, **not** before a proven collision instant.

The existing read-only glyph locator finds the visible ship in sampled
pre-loss panels only around character columns **18–28**; many panels have
no unique detectable ship glyph because of animation/occlusion. The
later visible wall's right opening begins around column 51 in the
independent [timing audit](../gate-timing-127/README.md). In these
selected replays, none visibly approaches that opening. The spatial
model's last-64-action movement fractions were **45%, 45%, 53%, 56%**
across its four lives; the unchanged control's were **56%, 59%, 64%,
59%**. Those counts describe selected physical commands, not measured
displacement. Fire-with-arrow commands suppress movement in the original
stage-one game, and the screen can be blank during death animation.

This corroborates the repeated *approach* bottleneck despite different
network weights and action mixes. It does not establish the exact cause
of death, compare the policies over their full 64-game fresh samples,
or prove a route. No sheet, visible column estimate, hidden course
pointer or intervention is used as a policy input, reward, training
example or action override; no model was updated or replay promoted.
