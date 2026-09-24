# Visible prepositioning before the repeated two-gap loss

This is a **post-hoc diagnostic**, motivated by the native-verified
[three-phase failure](../three-phase-grid-150/README.md) and a small
exploratory held-RIGHT screen check. It is not an unbiased model-training
experiment or a claimed solution. On the same originally verified
[own learned replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html),
freeze the exact action-280 prefix. Enumerate every start from action
**280 through 354** and every RIGHT hold of **1 through 48** base
decisions, overlaying that one hold on the learner's original actions
through frame **355**. The grid has **3,600 start/hold pairs**, including
some action-equivalent pairs at the edge of the window. Record all real
visible score, life/stage and ship-glyph-column outcomes at frames
**339 and 355**; stop a branch at any earlier visible boundary.

The recorded learner has **1,080** displayed points and a readable ship
glyph at column **23** at frame 339; at frame 355 it has **2,600** and
the glyph at column **25**. Compare branch columns only among branches
that retain at least those respective source scores and have no visible
life/stage boundary. The sprite glyph can be occluded, so a missing
column is unknown rather than a failed position. An apparently farther-
right branch with much lower score has not preserved the earlier passage.
This is a visible-screen measurement, not a collision oracle or hidden
course-pointer read. The original game's score is the only return.

Preserve the complete table, exact source trace/model/game hashes, native
source-prefix verification, baseline score and screen equality, and code.
Candidate actions must not enter the policy, PPO/DQN data, a reward or
checkpoint selector, or the protected best replay. The test can show
whether a **single** early RIGHT hold safely prepositions this one learned
trajectory; it cannot rule out multiple switches, diagonals, earlier
divergence, or a screen-conditioned policy.

The [complete native run](run/report.json) finished all **3,600** ordered
start/hold pairs, preserved [every outcome](run/outcomes.jsonl), and
reexecuted the original prefix, scores and screen bytes exactly. Of the
2,249 branches still at the frame-355 waypoint, **934** had retained
the learner's 2,600 points, **855** of those had a readable ship glyph,
and their farthest visible column was **29**. At frame 339, **1,278**
full-score, boundary-free branches had a readable glyph; their maximum
was **25**. The original learner was at column 25 at frame 355 and
column 23 at frame 339. The other frame-355 survivors scored only 330,
1,080 or 1,100 points, so an apparently large rightward displacement can be
the result of missing an earlier score-bearing passage. Another
**1,351** branches reached a visible boundary before frame 355; none
reached a later stage. The copied archive matches the source run's file
hashes.

The full **511-test** regression suite passed after the probe was added.

This narrows one tempting explanation: a single early RIGHT hold does
not move this verified trajectory materially nearer the far-right
opening *while retaining its earlier displayed score*. It does not
identify a safe multi-action route or prove physical impossibility.
The next learned-policy test should change how the model anticipates
successive visible obstacles, while retaining from-boot complete-game
validation and the no-hidden-state boundary.
