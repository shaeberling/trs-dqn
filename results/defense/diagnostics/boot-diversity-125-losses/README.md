# More boot games did not change the visible loss pattern

The [read-only loss report](report.json) checks native replay verification
and renders recorded screen bytes for the selected
[boot-diversity policy](policy-1-losses.png) and the
[confirmed score parent](policy-2-losses.png). Both best-effort games earned
**2,620 displayed points on each of four lives** and lost near the familiar
broad right-opening barrier. The seeds differ, and a white-screen flash is
an alignment marker, not an exact collision label.

The selected frozen policy was re-executed at its verified replay seed. All
**2,562** recorded screens, physical actions and rewards matched exactly.
Instrumenting its sampled 21-way choices found **36**
`CONTINUE_PREVIOUS` selections across the game, but only **1 / 2 / 0 / 1**
in the four 64-action pre-loss windows. Physical pure-movement choices
occupied **51.6%, 54.7%, 50.0% and 57.8%** of those windows. These counts
do not measure net displacement or identify the exact fatal action.

No diagnostic frame or action entered training. The analysis changed no
model input, reward, policy, evaluation or replay; the source trace and
independent verification remain in the archived evaluation bundle.

An additional [native counterfactual probe](held-keys-128.json) first
replayed the selected model exactly to its own frame **279**, 128 decisions
before the first visible life loss, captured the opaque emulator state, and
reproduced the entire recorded suffix byte-for-byte. It then restored only
that state and tried each of the original 20 physical commands held until
the next visible boundary. The same procedure was repeated from
[192](held-keys-192.json) and [64](held-keys-64.json) decisions before the
loss. All 60 constant-key suffixes still lost a life in stage one.

The timing matters: holding RIGHT from the 192-action anchor delayed visible
loss to 49 actions versus 21 for held NOOP, and from the 128-action anchor
to 79 versus 56. But from the 64-action anchor it lost after **31** actions
versus **63** for NOOP. These are single-state counterfactual timings, not
evidence that RIGHT is a route: from the earlier anchors, fixed RIGHT also
earned far less score than the learned suffix. The physical outcome varies
within the same life, so an entire-life direction preference may be too
coarse. The probes are diagnostic interventions only; no generated action
sequence was supplied to training, used as a demonstration, or promoted as
learned play. They do not prove a passable path or identify collision time.
