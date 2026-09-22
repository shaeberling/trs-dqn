# First real-experience feedback world update

The model completed **2,000 additional dynamics updates**, from 12,000 to
14,000, after adding [new actual actor experience](../world-model-actor-collection-03/README.md).
All full checkpoints, optimizer/RNG state, completed log, immutable union
data and source hashes are preserved here.

The checked union has 60 training games / 116,495 actions and twelve held-out
games / 23,304 actions. Earlier episodes and held-out assignments remain
unchanged. Differing policy parents are explicitly declared and recorded;
no evaluation replay or demonstration is imported. Sampling remains uniform,
since the focused-loss experiment did not improve overall forecasts or play.

At 14,000 updates the uniform held-out graphics MSE is .02023 at one step
and .03548 at 24 steps, versus persistence .05522 / .07234. This is a changed
held-out population, so compare with [its own starting audit](fit/update-012000/audit.json),
not directly with older four/eight-game aggregates.

The [36-case loss review](fit/review-14000/boundary-predictions.json) remains
poor: mean-latent Brier .90881 for recognizing observed arrival, .92265 one
step ahead and .98394 sixteen steps ahead. Always surviving gives 1.0 on
these selected cases. This world update is not evidence of barrier passage.

The [completed actor continuation](../imagination-feedback-01/README.md) uses
explicit `--refresh-world`. It restores the
full previous actor/critic/target/Adam/RNG state, verifies that the new world
declares the previous frozen world as its immediate parent and preserves or
explicitly extends its prior dataset, then replaces only world parameters.
Normal resume still rejects a changed world or dataset. The new unit test
verifies exact behavior-state retention and rejection of mismatched parent,
dataset and checksum; all five original actor tests also pass, including
complete native serial/parallel/reloaded replay agreement.

This is a manually orchestrated experience/model/actor feedback cycle, not
yet an unattended repeating Dreamer driver or a reproduction of the paper.
The world stays frozen within each actor-fitting phase. Actual complete-game
scores and native replay verification remain the outcome test. This first
cycle's final mean is 276, all stage-1 losses; it has not improved play.
