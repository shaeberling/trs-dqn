# First-milestone loss-window check

These two panels re-render **native-verified own-policy replay traces** from
the first 256-action and matched 128-action PPO milestones. The read-only
`rl.defense_loss_probe` checks source hashes, visible scores and all four
loss events before rendering recorded screen bytes. It performs no gameplay,
learning or action selection, and its output is never training input.

- [256-action rewind panels](policy-1-losses.png): the four lives each gained
  **2,620** points in the selected 10,480-point replay.
- [128-action control panels](policy-2-losses.png): the four lives likewise
  gained **2,620** points in a separate selected 10,480-point replay.

The repeated broad-barrier sequence remains visible in both selected games.
Several panels show the ship near the centre/left as a right-side opening
approaches. This supports the user's observation of a common navigation
bottleneck, but the replays use **different seeds** and were selected for best
score. Equal life scores do not prove identical course positions or physical
collision sites. The white-screen marker is an alignment heuristic, not a
collision label; some panels already show loss animation. Only the planned
fresh complete-game comparison can judge the trained variants fairly.
Pure directional movement accounted for **54.7–64.1%** of the 64-action
windows before the alignment marker across these eight lives. Thus the
repeated failure is not simply an absence of movement commands. Counts do
not measure displacement, and in stage one adding Space to an arrow suppresses
movement; the trace alone cannot identify which choices caused a collision.

[Machine-readable report](report.json) records provenance, action windows and
the original independent neural-action verification for both traces.
