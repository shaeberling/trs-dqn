# Continuation policy still meets the recurring visible barrier

The two source replays in [report.json](report.json) were independently
re-executed from frozen learned weights before this read-only analysis. Both
best-effort traces scored **2,620 points on each of four lives** and ended in
stage one. The [new model's sheet](policy-1-losses.png) and the
[older parent's sheet](policy-2-losses.png) again show the broad right-opening
barrier approaching a ship left of that opening. This is visual evidence of
the same bottleneck pattern, not an exact collision-coordinate proof. The
selected replay seeds differ; white flashes are only alignment markers.

To test whether the new mechanism was actually used, the selected frozen
policy was loaded again at its confirmed best replay seed, and its 21-way
sampled choices were counted while `record_game` independently reproduced
**all 2,551 recorded physical actions, screens, rewards and final result**.
Only **one** sampled choice was `CONTINUE_PREVIOUS` across the complete game,
and **zero** occurred in any of the four 64-action pre-loss windows from
the report. These counts are diagnostic, not policy input, reward, action
override or training data. The score improvement across 256 fresh games
therefore cannot be attributed to sustained key holds in this verified trace.
It does not establish that no other game used the option.

The counter can be reproduced by loading the saved policy with
`rl.defense_learning.load_policy`, wrapping its `_execute` method to count
sampled choice `20`, and calling `record_game` at the saved replay seed;
compare the returned arrays byte-for-byte with the source `trace.npz` before
using the count. No source replay or global best was modified.
