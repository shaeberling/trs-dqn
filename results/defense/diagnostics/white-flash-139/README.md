# White-screen flashes are not safe early-terminal labels

The repeated life loss invites an obvious credit-assignment hypothesis:
the large white-screen flash can precede the HUD's visible life decrement
by several policy decisions. This is a **read-only, post-hoc audit** of
existing own-policy screens, not a new reward, trained model or collision
oracle. The tested marker is simply: more than half of the 15 gameplay
rows' raw video cells equal `0xBF` in one frame. An onset is a transition
from below to above that threshold. Visible life boundaries are taken
from already recorded full-game events or `continuation == 0` in own
training collections. A marker within 25 decisions before a boundary
counts as near-boundary; any earlier onset, excluding the first 32
decisions after a previous boundary, is a false early terminal candidate.
This window is an audit convention, **not** a collision timestamp.

In the [48 archived own pre-loss snippets](../../training/ars-score-gated-69/run/own-loss-states/),
36 non-final lives had a marker 6–19 actions before the visible decrement;
the 12 final lives had none in their last 128 actions. This tempting
alignment is a selection effect: every snippet contains only its own
last 128 decisions.

Across **20 separately archived, native-verified fresh selected replays**
under `results/defense/training/*/fresh-selected-replay/trace.npz` (also
including the four more recent `run/fresh-selected-replay` bundles), all
**60 non-final lives** had a marker in the final 25 decisions and all
**20 final lives** lacked one there. The near-loss onset preceded the
visible decrement by 8–17 decisions. But the same complete traces also
contained **556 early marker onsets** after a 32-decision post-boundary
grace period, far outside the final 25 decisions. Thus late alignment
alone gives poor specificity for an online terminal detector.

An independent [24-game own-play dynamics collection](../../training/world-model-mixed-01/data/)
contains **56,029 actions** and **96 visible life boundaries**. The
same marker has **1,402 early onsets** more than 25 decisions before the
next boundary without the post-boundary grace, or **688** after applying
the same 32-decision grace used above. Only 71 boundaries have any marker
within 25 decisions,
and 25 of those appear at the boundary itself rather than predicting it.
Every counted marker run in this collection is just one sampled frame,
whether near a boundary or not. The collection uses an older exploratory
DQN policy, so it is a broader-screen stress test rather than a matched
estimate of the current PPO policy's false-positive rate.

Conclusion: **do not train with a raw white-majority flash as an early
life terminal**. It would cut many valid trajectories. A more specific
visible event detector would need independent false-positive validation
on complete own games before touching PPO targets. No weights, reward,
reset logic, best replay or original game were changed by this audit.
