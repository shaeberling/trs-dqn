# A full screen archive still yields the recurring four-life loss

The [read-only loss report](report.json) checks original native replay
verification and renders recorded screen bytes for the selected
[screen-frontier policy](policy-1-losses.png) and the
[confirmed score parent](policy-2-losses.png). Both selected best-effort
games earned **2,620 displayed points on each of four lives** and lost
near the familiar broad right-opening obstacle. Their seeds differ; a
white-screen flash is only an alignment marker, not an exact collision
label.

The selected frozen policy was re-executed at its replay seed. All
**2,563** recorded screens, physical actions and rewards matched exactly.
Instrumenting its sampled 21-way choices found **zero**
`CONTINUE_PREVIOUS` selections in the full game, including the four
64-action pre-loss windows. Physical pure-movement choices occupied
**60.9%, 67.2%, 59.4% and 70.3%** of those windows. These counts do not
measure net displacement or identify the fatal action.

The training archive had reached its 128-cell cap on every worker at the
last logged terminal inventory. Visual diversity in training therefore
did not translate into a distinct frozen-policy barrier behavior in this
replay. No diagnostic frame/action entered training, and the report does
not change reward, model input, evaluation or the protected global best.
