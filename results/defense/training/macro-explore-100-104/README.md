# Own-loss-state random-control exploration

This is a **training-only diagnostic**, not a learned policy or a published
replay. The independently confirmed stage-one policy first played twelve
complete training-seed games of its own. The harvester saved opaque native
states exactly 128 or 370 decisions before its own visible life losses,
then restored and replay-verified the recorded continuation. Only states
from lives gaining at least 2,400 displayed points were selected. Recorded
source actions reconstruct a chosen prefix exactly; they are never policy
inputs or training targets. The native bytes are not decoded.

Each trial then applies two randomly selected, fixed-duration command holds
and resumes the unchanged neural policy until the next visible life/stage
boundary. Branch points and hold lengths are sampled symmetrically; no
opening, route, preferred key, hidden collision marker or extra reward is
supplied. `all` samples all twenty keyboard commands; `effective` samples
NOOP, all eight movement commands and fire equally, avoiding stage-one
Space+arrow aliases without favoring any direction. **The random holds are
not claimed as learned-policy actions or used in evaluation from boot.**

| Archived trial | Own pre-loss lookback | Hold lengths | Continuations | Furthest past original visible-loss marker | Stage 2 |
| --- | ---: | --- | ---: | ---: | ---: |
| [All commands, short](all-short-128/report.json) | 128 | 8–64 | 256 | 23 decisions | 0 |
| [Distinct commands, short](effective-short-128/report.json) | 128 | 8–64 | 256 | 24 decisions | 0 |
| [All commands, larger sample](all-short-128-large/report.json) | 128 | 8–64 | 10,000 | 46 decisions | 0 |
| [All commands, earlier branch](all-short-370/report.json) | 370 | 8–64 | 256 | 25 decisions | 0 |
| [Distinct commands, long holds](effective-long-370/report.json) | 370 | 64–256 | 512 | 0 decisions | 0 |

The largest short-hold run did not find stage passage in **10,000** focused
continuations. Earlier short interventions also barely extended the life;
long random holds generally caused much earlier losses. Across all five
trials, **11,280 continuations reached no stage-two screen**. This is
evidence against these simple open-loop random interventions, not proof that
the barrier is unavoidable, that a specific obstacle caused every loss, or
that a screen-conditioned learner cannot improve. Visible loss may lag the
physical collision. The score gain in each report starts at its snapshot,
so it is not comparable across different lookback lengths.

Each trial directory retains its exact source, own native snapshots,
per-trial command plans and scores, RNG state, and report. No model weights
were updated or promoted; the verified global best replay is unchanged.
The reproducible probe is [rl/defense_macro_explore.py](../../../../rl/defense_macro_explore.py).
The full **449-test** repository suite passed after this new probe and its
four focused tests were added.
