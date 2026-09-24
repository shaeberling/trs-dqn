# Within-life factor redraws still lead to the same visible loss

The [read-only loss report](report.json) checks native replay verification
and renders recorded screen bytes for the selected
[windowed-noise policy](policy-1-losses.png) and the
[confirmed score parent](policy-2-losses.png). Both best-effort games earned
**2,620 displayed points on each of four lives** and lost near the familiar
broad right-opening obstacle. Their seeds differ. A white-screen flash is
only an alignment marker, not an exact collision timestamp.

The selected frozen policy was re-executed at its verified replay seed.
All **2,583** screens, physical actions and rewards matched exactly.
Instrumenting its 21-way sampled choices found just **one**
`CONTINUE_PREVIOUS` selection overall, in the second of four 64-action
pre-loss windows. Physical pure-movement choices occupied **51.6%, 67.2%,
65.6% and 57.8%** of those windows. These counts do not measure net
displacement or prove which action caused the loss.

Training contained more than 33,000 scheduled key-factor draws; frozen
evaluation intentionally contains none. The diagnostic does not imply the
training noise was absent. It shows that this training change did not
produce a verified passing or persistent frozen-policy behavior here.
No diagnostic frame or action entered training, and the source trace and
independent verification remain in the archived evaluation bundle.
