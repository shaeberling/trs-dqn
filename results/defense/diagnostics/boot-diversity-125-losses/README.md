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
