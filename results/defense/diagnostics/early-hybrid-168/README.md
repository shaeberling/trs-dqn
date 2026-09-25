# Early frozen-policy handoff: can rightward positioning and firing coexist?

The all-life timing/geometry audit shows two incomplete learned behaviors:
high-scoring firing policies occupy the center near their late first-stage
loss, while the movement-only policy moves much farther right but loses at
an earlier obstacle. The [late handoff test](../hybrid-window-165/README.md)
changed behavior only at actions 300–360 and did not extend survival by more
than one action. This experiment asks whether an **earlier** temporary
movement-only window can leave the firing policy better positioned without
destroying its useful route.

The source is the same independently verified seed-612059 original-boot
first life of the frozen balanced-fire continuation. At one fixed action,
sample the frozen movement-only neural policy for one fixed window; otherwise
sample the frozen balanced-fire policy. The intervention times are scripted
and are **not** a learned controller, training experience, demonstration,
reward or promotable replay. Model inputs remain raw video-memory frames;
no private game state guides actions or branch selection.

Predeclared grid: starts **0, 40, 80, 120, 160, 200, 240**; window lengths
**16, 32, 64, 96, 128** base actions; **eight** independent movement-policy
draws per cell, for **280** branches. Stop at the first visible ship loss,
stage advance, or action 520. All parent source prefixes and restored
suffixes must reexecute exactly before branching. Record displayed score,
visible life/stage outcome and sometimes-occluded ship glyph columns at
actions 160, 200, 240, 300, 340, 380 and 390. No branch action trace is
saved as training data.

A branch earns an independent original-boot verification only if it reaches
stage two, exceeds 2,620 first-life displayed points, or remains visibly
alive at least **20 actions beyond** the source's loss at 412. If no branch
passes that gate, do **not** train a two-frozen-expert gate on this result.
Any positive diagnostic branch would show only reachability for a scripted
switch schedule, not satisfy the learned-policy goal. The protected best
model/replay and original game are never modified.

## Completed result

The [two-branch smoke](smoke-report.json) passed exact parent replay parity
and independently reproduced both switched branches from original boot;
both focused native tests and the full **579-test** native suite passed.
The [full 280-branch report](full-report.json)
completed the predeclared grid. No branch reached stage two, exceeded the
source's **2,620** first-life points, survived to action 520, or remained
visibly alive through action 432. The latest visible loss was action **415**,
only three actions later than the source. Fifteen branches reached action
412 or later, but none showed a meaningful passage. At action 380, just 37
branches had a readable ship glyph; the farthest was column **30**, well
short of the later right-side opening. Windows of 96 or 128 actions often
failed at an earlier obstacle and scored at most **140** points. Shorter
windows often recovered the firing policy's score but also its central
failure.

This fails the predeclared gate for a frozen-two-expert learned switch.
Combined control must be learned or explored in a materially different way;
merely changing *when* these two particular policies hand off is not enough
on this source life. The diagnostic is not a model update, successful
route, demonstration or replay candidate. The protected best is unchanged.
