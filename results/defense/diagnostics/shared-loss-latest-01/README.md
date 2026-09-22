# Recurring failure review

The user's observation that the agents appear to fail in the same place is
supported by a fresh comparison of three preserved, independently verified
replays. These are selected trajectories, not a representative evaluation set.
The probe checks source hashes and reads only recorded screens/actions/rewards;
it does not run training, change policies, or feed evaluation data into replay.

| Selected policy | Points on each life | Visual interpretation |
| --- | --- | --- |
| Shared PPO best, 10,480 | 2,620 / 2,620 / 2,620 / 2,620 | Repeated right-opening barrier sequence; ship remains near centre/left as it approaches |
| DQN 43, 10,180 | 2,550 / 2,550 / 2,550 / 2,530 | Earlier centre-opening barrier near ship; right-opening barrier still farther up |
| DQN 44, 10,260 | 2,550 / 2,570 / 2,570 / 2,570 | Right-opening barrier sequence again, ship near centre/left |

See the original-screen panels for [the shared best](policy-1-losses.png),
[DQN 43](policy-2-losses.png), and [DQN 44](policy-3-losses.png), and the
[machine-readable provenance and action windows](report.json).

The useful diagnosis is a recurring navigation bottleneck, not literally one
identical collision across all policies. Equal score does not identify course
position. The visible life counter changes after the failure animation, and
the first major white flash is only an alignment heuristic, not an exact
collision timestamp. Some panels already contain animation. These screens do
not establish whether a wall or projectile causes an individual death.

Nor is this simply an absence of movement choices: pure directional actions
account for 51.6–71.9% of the best policy's four 64-action alignment windows,
48.4–56.3% for DQN 43, and 32.8–50.0% for DQN 44. Action counts are not measured
displacement, and firing with an arrow suppresses movement in stage 1.

## Consequence for training

Repeated near-ceiling scores are not evidence of passage. The pending SPR
calibration tests whether an auxiliary task predicting the learner's own future
visual features helps the shared encoder learn temporal information. It does
not establish that representation learning is the cause or the remedy.
Its score reward, screen-only acting inputs and ordinary greedy evaluation are
unchanged. Lower auxiliary loss is not success and could be misleading if
representations become nearly constant.

Judge it using complete boot games, independently verified replay, actual
stage reach, and comparable failure windows. Preserve the stronger existing
best regardless of its outcome. No hand-coded route, steering override,
collision label, extra reward, or demonstration follows from this review.
