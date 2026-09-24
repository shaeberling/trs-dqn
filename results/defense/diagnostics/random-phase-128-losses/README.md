# Random-phase exploration still leaves the same visible bottleneck

The [read-only loss report](report.json) checks both original frozen replay
bundles and renders their recorded video bytes. The run-128 selected
[screen sheet](policy-1-losses.png) and unchanged run-121 parent
[screen sheet](policy-2-losses.png) each show four lives earning **2,620**
displayed points and losing near the familiar broad right-opening barrier.
The new policy's [local replay](../../training/ppo-random-phase-key-128/fresh-selected-replay/replay.html)
was independently verified for **2,569** learned actions; the parent replay
was independently verified too. Their seeds differ, and visual alignment
uses a major white flash when sampled, not an exact collision marker.

This confirms that the stronger random-phase training exploration did not
produce a distinct successful **frozen-policy** barrier behavior in these
selected games. It does not prove that each life collided with the same
wall, nor that a passable path does not exist. The diagnostic was never read
by the trainer, supplied as a demonstration or used to promote a replay.
