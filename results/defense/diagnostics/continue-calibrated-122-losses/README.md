# Calibrated continuation is used, but still does not cross the barrier

The [read-only report](report.json) checks the hashes and existing native
verification of two best-effort replays, then renders only their recorded
screen bytes. The [calibrated policy](policy-1-losses.png) and
[uncalibrated score parent](policy-2-losses.png) each earned **2,620 points
on all four lives** before losing near the familiar broad right-opening
barrier. These are different selected seeds, not an exact matched trajectory
or collision-location proof. The white flash is an alignment heuristic.

The calibrated frozen policy was also re-executed at its selected replay
seed. Every saved screen, physical action, reward and final result matched
the native-verified trace, while its sampled 21-way choices were counted.
It chose `CONTINUE_PREVIOUS` **132 times in 2,576 decisions (5.12%)**:
across the four 64-action pre-loss windows, the counts were **2, 2, 2, 0**.
Thus the training did not suppress the new action completely, but continued
movement remained sparse at the decisive visible approaches. The physical
pure-movement fractions in those windows were **70.3%, 73.4%, 84.4%,
76.6%**—higher than the earlier score parent's selected best effort, yet
not proof of useful displacement toward the opening. The ship still appears
left of that opening as the barrier approaches.

These counts and panels are diagnostics only. They do not enter model input,
reward, curriculum selection or the action executor. The re-execution used
the saved `rl.defense_learning` policy and `record_game`, instrumenting only
its sampled-choice return and asserting exact equality with the archived
`trace.npz` arrays before reporting any count. No replay or model was edited.
