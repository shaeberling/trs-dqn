# One more unbiased action phase at the repeated first-life barrier

The [verified later-turn two-phase grids](../late-phase-grid-149/README.md)
tested 15,000 fixed plans from adjacent first-life actions 339 and 340;
none exceeded the source's 2,620-point plateau or entered stage two.
The earlier action-322 grid was also negative. This bounded diagnostic
adds a third independently sampled short physical-command phase,
allowing a sequence to change direction twice (or choose the same key
again to extend a hold). It still does not condition actions on a new
screen and cannot itself satisfy the learned-policy goal.

Freeze the same originally verified
[run-136 own replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html),
its original first-life prefix and two anchors **339 and 340**. For
each anchor, test the Cartesian product of own-prefix delays
**0 / 8 / 16**, three holds independently **8 / 16 / 32** base
actions, and three commands independently drawn from the same ten
direction-symmetric effective stage-one commands used in run 149.
That is exactly **3 × 3³ × 10³ = 81,000** candidates per anchor.
After the three phases, play the source's recorded action suffix
until the next visible life/stage boundary or 512 actions. Plans that
reach a boundary mid-phase stop there; no future keys are injected.
The complete grid is fixed before seeing outcomes and has no wall
or route filter, preferred direction, hidden course read or score
bonus. Every candidate ranks only by actual displayed score and
visible life/stage/mission outcome.

First run a 20-candidate source-integrity smoke for each anchor,
then the full action-339 grid. If it independently reexecutes a
stage-two screen from original boot, stop that grid, preserve the
discovery and proceed to learned-policy research; otherwise run the
full action-340 grid. Both jobs check the 5 GiB disk safeguard and
handle SIGINT by saving current status. Preserve every candidate
choice and outcome, source hashes, fixed enumeration, original
baseline and full report. A candidate above 2,620 is diagnostic
progress only; it must not enter a demonstration set, trainer,
policy input, reward, checkpoint selector or promoted neural replay.
The protected best and all learned checkpoints stay untouched.
If both grids fail, the result bounds only this three-phase open-loop
family from two precise own states, not the game's solvability.

Both 20-candidate integrity smokes reproduced their source suffixes exactly.
The [action-339](anchor-339/report.json) and
[action-340](anchor-340/report.json) full grids then completed **81,000
candidates each**. Neither found a higher first-life score than **2,620**,
a later stage, a 512-action survivor, or even a same-score continuation
that lasted longer than the learned suffix. At action 339, **38,571**
candidates tied the score; at action 340, **42,162** did. Their complete
enumerated [outcomes](anchor-339/outcomes.jsonl) and
[adjacent outcomes](anchor-340/outcomes.jsonl), run configurations,
source copies and SHA-256 reports are preserved here. All four copied
run directories matched their original file hashes after archival.
The full **509-test** regression suite passed after this diagnostic code
was added.

The original visible frames give a concrete hypothesis, not a route:
at frame 339 the ship glyph is around column **23**, the lower wall
opens at columns **21–30**, and an approaching upper wall opens only
at **51–62**. By frame 387 the upper wall is near the ship's row, while
the recorded ship glyph is around column **25**. This directly supports
an earlier anticipation/positioning problem on this own trajectory.
It does not establish which action sequence can pass, whether the game
is solvable from these late states, or whether the policy lacks enough
screen information. In particular, more coarse late-phase bursts are
not an evidence-backed next training method; the next learned test
should focus on earlier screen-conditioned control and evaluate complete
fresh games at the same stage gate.
