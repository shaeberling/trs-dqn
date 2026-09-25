# Original-game loss timing across four frozen learned policies

The [single-life forensic audit](../internal-loss-lag-166/README.md) found a
21-decision delay between the original game's internal ship-count decrement
and the visible HUD decrement. This follow-up tests whether that finding is
an idiosyncrasy of one seed or a repeated game behavior. It remains
**diagnostic-only**: the private counter at `0x7CEF` is never a model input,
reward, boundary, curriculum feature, action source, or replay-ranking key.

Four independently verified, complete original-boot replays are fixed before
running the probe: the protected best, balanced-fire continuation seed
612059, fresh grouped-duration seed 615000, and movement-only extension seed
620600. The probe must reproduce **every recorded action, displayed reward
and screen** for all four lives of each game. At each original private
ship-count decrement, it restores the exact pre-action emulator state and
bisects that recorded key's requested duration; no probe action advances
the canonical replay. It then pairs that private event with the later
visible ship-loss event. The report also records only *visible* ship glyph
and wall-run measurements from frames immediately before the private event,
to keep the positioning question separate from the hidden timing label.

Decision use: a consistent lag across the score-heavy and no-fire policies
would rule out interpreting HUD loss as collision time in future
loss-centered reachability studies. Large variation would require a
policy-specific timing analysis. Neither outcome licenses hidden-state
supervision or proves a passable route; it only guides which *visible*
approach frames to investigate next.

## Completed result

The [full report](report.json) exactly reexecuted **9,361** original-boot
actions, visible rewards and raw screens across the four games. Both focused
native tests passed. All **16** private ship-count decrements paired with
their recorded visible life losses. The twelve non-final lives had reporting
lags of **11–21** decisions (mean **17.17**); the four final lives all had
lags of **three** decisions, consistent with the environment's special
GAME OVER score/HUD settlement path. These are *reported* action counts,
not exact collision timestamps; the original overlap check precedes the
private counter update.

The visible geometry is more important for the next experiment. Eight
decisions before each internal loss, readable ship glyphs for the three
high-scoring firing policies were at columns **20–26**. Their near-loss
screens repeatedly show a wall reaching column **50**, with the next broad
opening to its right. The movement-only policy instead had readable ship
glyphs at columns **47–52** in its first three lives, but died much earlier
in the course, as independently established by its post-freeze row audit.
Some sprite observations are missing or occluded, and rendered wall glyphs
are not collision flags. This supports a *two-skill, earlier-positioning*
diagnosis, not a scripted route or proof of physical reachability.

The private counter tells us that a visible-HUD-centered branch can include
many actions after the ship is already lost. It must remain forensic only.
Existing interventions at actions 300–360 on the firing source preceded its
action-391 internal decrement; their negative outcomes still stand. A
next bounded test should ask whether different earlier behavior can combine
rightward positioning with enough firing to survive the first obstacle.
It cannot be promoted as a learned replay or supplied as demonstrations.
The original game and protected best remain unchanged.
