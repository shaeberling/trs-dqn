# Obstacle Run / Missile Defense: integration and training feasibility

The user-confirmed new game is the executable in `Missile_Defense.zip`.
It identifies itself as **Obstacle Run**, by Arno Puder (1983/84), and is
already present as `var/defense.cmd`. No emulator rebuild, disk controller,
new ROM, binary patch or duplicate game asset is needed.

Latest outcome: the [movement-only PPO pilot and extension](results/defense/training/ppo-movement-only-164/README.md)
finished at **4,194,304** own actions without reaching stage two. Its selected
policy modestly improved displayed score over its shorter pilot on two
fresh 128-game matched sets, but all **512** games remained in stage one;
the selected replay lost near course row **16 of 126** on all four lives.
The stronger firing policy still loses near row **33–34**. A subsequent
[256-branch frozen-policy movement-window diagnostic](results/defense/diagnostics/hybrid-window-165/README.md)
from one independently verified firing-policy first life extended visible
survival by at most **one action** and never reached stage two. This is a
negative result for switching between those two experts, not a proof of
impossibility. More score-only continuation or a learned two-expert gate is
not justified by the current evidence. The protected 10,480-point model
and replay remain unchanged; no original mission has been observed.

Earlier outcome: the fresh [grouped key-duration PPO run](results/defense/training/ppo-grouped-duration-fresh-163/README.md)
completed all **16,777,216** planned own actions and selected its terminal
checkpoint on a **9,551** ten-game fixed mean. All **160** fixed and **512**
fresh matched comparison games (grouped and parent combined) ended in stage
one. Across two untouched 128-game matched sets, the grouped policy scored
**9,518.98 / 9,272.58** on average versus **9,601.48 / 9,647.34** for
the frozen score parent. Its best independently verified fresh replay scored
**10,080**; the protected global best remains **10,480** and no original
mission has been observed. Full selected state, all fixed results, complete
metrics, seven training best-effort versions and four verified fresh replays
are archived. The new policy is not promoted, and another mechanism is
needed before spending more compute on the repeated early barrier.
The subsequent movement-only trial (summarized above) retained the original
game's displayed-score reward and screen observation. It was a bounded test
with a frozen no-learning control, not a stage clear or replacement for the
protected best.

Current learned-policy experiment: the [fresh balanced-control PPO
comparison](results/defense/training/ppo-balanced-fire-from-scratch-160/README.md)
completed **8,388,608** own score-only, screen-only actions per arm. The
balanced initializer selected its terminal checkpoint (fixed ten-game mean
**8,502**), while the otherwise matched no-offset control selected its
4,194,304-action checkpoint (fixed mean **588**). On two untouched sets of
64 complete games each, balanced/control means were **9,382.97 / 579.69**
and **9,424.38 / 581.88**; balanced won all **128** paired games. Every
game still ended in stage one. Four local replays were independently
verified, and full selected/terminal state, fixed checks, logs and fresh
results are archived. A [post-training forensic replay comparison](results/defense/diagnostics/balanced-fire-160-loss-comparison/README.md)
shows that the treatment moves more often near loss yet still fails around
the original stage-one stream rows 31–34 of 126. Hidden course reads were
diagnostic only, never learner inputs, rewards or selection criteria. No
stage-two play or successful mission has been verified; the protected
[10,480-point best replay](results/defense/learned/best/replay.html) remains
unchanged.

The [predeclared balanced continuation](results/defense/training/ppo-balanced-fire-continuation-161/README.md)
then resumed the exact optimizer and policy RNG for **8,388,608** more
own actions and stopped normally at **16,777,216** total. Its seventh
checkpoint won the stage-then-fixed-mean selector (ten-game mean **10,258**).
On **128 fresh matched complete games**, it averaged **9,641.72** versus
**9,314.06** for its frozen parent, winning **102** seeds to **26**.
Both fresh replays independently verified, including a tied **10,480**
stage-one best. All **256** fresh games still ended in stage one, with no
successful mission. This confirms a score-training gain, not completion or
a replacement for the protected global-best replay. Full selected and
terminal states, all eight fixed checks, logs and fresh comparison are
archived; a different next mechanism is needed to pursue passage.

Storage retention is active, not just a warning threshold. After archiving
the balanced comparison, **14** unselected live checkpoint directories
(about **130 MiB**) were removed; selected/terminal optimizer states, every
fixed evaluation, logs and verified replays remain. A separate stopped
[DQN-24 archive](results/defense/training/dqn-24-bootstrap/) was checked
byte-for-byte against its local final and two intermediate checkpoints,
compressed full metrics and all **13** replay bundles. Only its duplicate
local checkpoint/replay directories (about **681 MiB**) were removed;
the tracked archive and lightweight local config/log/status remain. These
deleted local duplicates are recoverable from the tracked archive and
pushed branch. The continuation had a separate exact-PID 5.1 GiB disk
guard. After its completed comparison archive was pushed, seven
nonselected live checkpoint directories and duplicate local bundles were
pruned (about **95 MiB**), retaining selected/terminal full states and all
evaluation records.

A wider stopped-run retention pass then removed **171** additional local
DQN/PPO checkpoint directories (about **2.0 GiB** by `du`) only after each
directory matched a tracked archive copy byte-for-byte. The active seed-42
run and every unmatched local checkpoint were excluded. The archived copies
remain on the pushed branch, so these deleted duplicates are recoverable.
APFS free-space accounting did not immediately increase by the nominal
`du` total; the live free-space guard still uses actual filesystem space.

The next bounded [seed-42 balanced run](results/defense/training/ppo-balanced-seed42-162/README.md)
tests independent initialization/experience diversity rather than another
continuation of the seed-41 optimizer. It uses the same score-only,
screen-only rules, fixed stage-gated checks, a 5.1 GiB exact-run disk
guard and two untouched 128-game comparisons before any score-parent
claim. Earlier actor-head ARS searches were already negative, so that
method is not being redundantly relaunched.

That independent run has now completed its full **16,777,216**-action budget.
Its selected terminal checkpoint averaged **570** on ten fixed complete
games, still all stage one. On two untouched sets of 128 complete games,
seed 42 averaged **559.69 / 559.06** against the frozen seed-41 parent's
**9,491.48 / 9,593.59**; the parent won all **256** paired seeds, and all
**512** fresh games remained stage one. Four original-boot replays verified.
The full selected/terminal state, all 16 fixed evaluations, log and fresh
comparison are archived. No score-parent or global-best promotion occurred;
the next experiment must change the learning mechanism, not simply extend
seed 42. A direction-neutral grouped key-duration action prior has been
implemented and unit-tested separately, without changing this run's trainer.
After pushing the full archive, **16** stopped-run checkpoint directories
and local duplicate bundles were pruned (about **191 MiB** by `du`); the
selected full optimizer state, all fixed evaluations, complete metrics and
verified replays remain on the pushed branch. The 15 unselected optimizer
snapshots are not retained.

The next [fresh grouped key-duration PPO experiment](results/defense/training/ppo-grouped-duration-fresh-163/README.md)
changes temporal action exploration instead of restarting the same
single-action policy. Its fixed physical-command and duration prior is
direction-neutral, the reward remains visible score only, and its opt-in
training/evaluation/replay implementation passed the full **557-test** native
suite. A bounded verified smoke gates a fresh full run; no trial-163 result or
stage advance is claimed yet. The smoke passed at exactly 16,384 actions:
ten complete games averaged 254, a native-verified replay was archived,
and 134 actual 64-action options started. A fresh 16,777,216-action
production run used a 5.1-GiB exact-process disk guard and
fail-closed post-run comparison monitor. Only the archive-backed duplicate
local smoke directories (about 23 MiB) were deleted; the smoke checkpoint
and replay remain on the pushed branch.

At its fifth fixed production check, this fresh temporal policy averaged
**8,214** on ten complete original-boot games (best **9,880**), a large gain
over its first four checks. Its 5,242,880-action full optimizer checkpoint
and native-verified 2,569-action replay are archived in the trial folder.
All fifty fixed games so far remain stage one; this is score learning, not
mission passage or a global-best promotion. The planned full run continues.
The sixth fixed check then rose to a **9,355** ten-game mean (best **9,930**),
again all stage one. Its full checkpoint and independently verified replay
are archived as a new recovery milestone; the goal remains unfulfilled.
A [forensic audit of its verified replay](results/defense/diagnostics/grouped-duration-163-course-progress-6m/README.md)
found all four visible losses at the original stream's decoded row 34 of
126. That private pointer was read only after freezing the replay, never
as a policy observation, reward or checkpoint-selection feature. The
score gain still does not solve the early passage bottleneck.
At the eighth, halfway fixed check, the mean fell to **8,628** while the
best single game reached a new trial-best **9,980**. Its full checkpoint
and verified replay are archived alongside the stronger-by-mean sixth
checkpoint. All eighty fixed games so far remain stage one; the protected
10,480-point best and original game are unchanged.
At the twelfth fixed check, the mean edged up to **9,388**, making that
checkpoint the current trial selection by the frozen stage/mean rule.
Its full optimizer state and independently reproduced 9,910-point replay
are archived. All **120** fixed games to date remain stage one; the
halfway 9,980-point replay is still this trial's best individual effort.

Two unrelated stopped Defense bootstrap smokes also had a full local
`latest`, fixed-step checkpoint and replay bundle that matched their
already tracked archives byte-for-byte. Their six redundant local
directories (about **426 MiB** by `du`) were deleted after verification;
the archived model, optimizer, evaluation and replay files are still
available. APFS reported roughly 14 GiB free both before and after,
consistent with shared storage extents, so the live guard continues to
watch actual free space rather than nominal directory sizes.

Two follow-up diagnostics closed control-profile and earlier-approach gaps.
The [twelve-command side-fire beam](results/defense/diagnostics/screen-beam-side-fire-158/README.md)
added both distinct stage-one side-fire combinations; **290,508** branches
still capped at **2,620** points with no later stage. The
[frame-200 beam](results/defense/diagnostics/screen-beam-early-159/README.md)
tested **346,224** earlier screen-conditioned branches using all twelve
distinct controls and again found no passage. These are bounded searches,
not learned policies or proofs of impossibility; neither searched route
entered training. The best verified learned model and replay remain intact.

Latest diagnostic: the [screen-conditioned beam study](results/defense/diagnostics/screen-beam-157/README.md)
tested **312,190** original-emulator branches from a verified learned
first life, including an earlier frame-280 start, one-action timing and
visible-screen-plus-last-action diversity. None exceeded **2,620** points,
reached stage two, or remained visibly alive at frame 428. At frame 388,
the farthest retained full-score readable ship glyph was column **39**;
the next wall's visible opening begins beyond column **50**. This makes
the shared loss a concrete two-opening navigation/credit bottleneck, not
merely a coincident final score. The search is diagnostic only, not a
new learned policy or a proof that passage is impossible. The original-
boot verified best weights and replay remain protected. The full **525-test**
regression suite passes.

Current status: the [early two-phase reachability diagnostic](results/defense/diagnostics/early-two-phase-reach-156/README.md)
tested **3,600** exact original-emulator action combinations from a
native-verified own first life. All combinations lost by frame 409;
none exceeded the source's 2,620 points, survived to frame 512 or
entered stage two. Only 69 preserved 2,600 points at frame 355, with
the visible ship no farther right than the original. This is a bounded
diagnostic, not a learned model or proof of impossibility. The full
**521-test** suite passes, and the protected best remains unchanged.
The preceding [visible-gameplay survival experiment](results/defense/training/ppo-visible-survival-155/README.md)
tested 262,144 new PPO actions with an explicitly shaped, visible-HUD
survival reward. Its selected model averaged **10,454** on 64 untouched
complete games versus **10,471** for the matched score-only control, and
finished about 13 actions sooner per game. All 192 fresh games remained
in stage one, so its predeclared extension gate failed. Its full states
and verified replays are archived; nothing was promoted. The earlier
[visible-screen novelty experiment](results/defense/training/ppo-life-novelty-154/README.md)
tested a new, explicitly training-only auxiliary reward against a matched
score-only control, each for 262,144 new actions from the same PPO parent.
The novelty arm preserved strong fixed-seed score play but averaged
**10,348** on 64 untouched complete games versus **10,388** control and
**10,435** frozen parent; all 192 fresh games stayed in stage one. The
reward often marked ordinary moving screens as new, so this coarse signal
did not distinguish the required passage. Neither model was promoted; the
protected 10,480-point replay remains available. Full states/replays
are archived. Original
game completion remains unverified.

Historical status: **Defense training resumed after the user freed disk space**
(23 GiB available at restart). The full **414-test** suite passes, including
the supervisor checks previously blocked by the unchanged 5 GiB safeguard.
The latest [learned key-duration PPO investigation](results/defense/training/ppo-duration-133/README.md)
copied the strongest own screen-only PPO and trained its actor only at real
option starts. A conservative 131,072-action pilot and a direction-neutral
duration-noise 524,288-action trial both completed; the latter genuinely
sampled hundreds of 64-action holds, but none of its 80 fixed or 128 fresh
complete games reached stage two. Its selected checkpoint averaged 9,812
on 64 untouched games versus 9,560 for its frozen initializer. A
[separate forensic replay audit](results/defense/diagnostics/duration-133-course-progress/README.md)
places all eight new visible life losses at original stage-one stream row
33–34 of 126, matching the historical bottleneck. These hidden row reads
were diagnostic only, never training inputs or rewards. The protected
10,480-point global replay and original game's bits are unchanged. The
then-current full **489-test** regression suite passed.

The subsequent [joint key-and-duration exploration](results/defense/training/ppo-duration-joint-134/README.md)
sampled hundreds more long holds over 1,048,576 own training actions.
Its selected checkpoint averaged 10,474 on fixed validation, but **all 80**
validation games and both untouched comparison sets stayed in stage one.
Across 192 fresh matched games, its mean was **10,378.02** versus
**10,391.51** for the established ordinary-action score parent; rare
low-score failures prevent a score-parent promotion despite more paired
wins. A [fresh verified replay](results/defense/diagnostics/joint-134-course-progress/README.md)
still loses all four lives at decoded first-stage rows 33–34 of 126. The
original game, protected best replay and no-oracle training contract remain
unchanged. The full **491-test** regression suite passes with this joint mode.

The next [screen-diverse reset continuation](results/defense/training/ppo-duration-screen-135/README.md)
filled 128 own-visible-screen cells per worker and reached the exact
10,480 stage-one ceiling on four fixed ten-game checks. All 80 fixed and
192 fresh games nevertheless stayed stage one. Its selected checkpoint
averaged **10,319.69** on 64 fresh games versus **10,472.66** for its
frozen joint input; the apparent fixed-seed improvement did not transfer.
The [verified fresh replay audit](results/defense/diagnostics/screen-duration-135-course-progress/README.md)
still loses at original first-stage stream rows 33–34 of 126. The global
best replay is unchanged.

The [semi-Markov option-credit continuation](results/defense/training/ppo-duration-credit-136/README.md)
gave every learned hold's entire real displayed-score return to its
actor start while leaving the base-screen critic unchanged. It completed
1,048,576 new actions with all **80** fixed games in stage one. On two
fresh matched sets (**192 games each**), its selected checkpoint averaged
**10,406.41** against **10,398.28** for the established ordinary-action
parent, with four sub-9,000 scores each. This narrowly meets its
predeclared *score-training-parent* eligibility rule but not the actual
gameplay goal; its [verified replay](results/defense/diagnostics/option-credit-136-course-progress/README.md)
again loses at decoded rows 33–34 of 126. The protected replay remains
unchanged, and the full **494-test** suite passed before production.

An [expanded exact-prefix gate probe](results/defense/diagnostics/gate-window-137/README.md)
held all 20 physical keys from each of 18 adjacent verified first-life
frames before the common loss. All **360** diagnostic branches still lost
within 160 actions without exceeding the learned life’s 2,620-point
score. Holding RIGHT earlier died at 1,080–1,100; holding it later
reached 2,620 and died there. These intervention actions are **not**
training data or learned replays. The evidence favors testing
phase-changing exploration over merely lengthening a constant hold,
without proving any particular route.

The [phase-changing learned-option continuation](results/defense/training/ppo-duration-phase-137/README.md)
then renewed unbiased physical-key preferences every 24–64 own base actions
while retaining per-life duration factors. Its 1,048,576-action run
completed normally with all 80 fixed games in stage one. The earliest
10,480-mean checkpoint averaged **10,392.34** on 64 untouched complete
games, below its own frozen option-credit input's **10,421.25**; all 192
matched selected/input/ordinary-parent games remained stage one. The
selected replay was reexecuted for all 2,520 actions; its four visible
losses still occur at original course rows **33 / 33 / 34 / 34 of 126**.
It is archived as a negative result, not a new score parent or promoted
replay. The full **495-test** suite passed before production.

A separate [frozen cadence diagnostic](results/defense/diagnostics/cadence-138/README.md)
played the stronger option-credit input on 64 matched complete games per
100,000 / 50,000 / 25,000 T-state base-action timing. Native timing
averaged **10,475.16**; both finer evaluation-only overrides regressed to
**7,649.84 / 7,418.13**, losing all 64 paired games apiece and reaching
no later stage. This does not test retraining at finer cadence. Neither
trial changed the original game, reward, protected best replay or no-oracle
learning boundary.

The [white-flash credit diagnostic](results/defense/diagnostics/white-flash-139/README.md)
checked whether a screen-visible flash could mark the repeated life loss
earlier for PPO. It does align 8–17 actions before all 60 non-final
losses across 20 verified selected replays, but the same full traces have
556 false early onsets even after a 32-action post-life grace. An older
24-game own-play collection has 688 such early onsets after the same grace.
Therefore a raw white-majority screen cannot safely replace the actual
visible life boundary; no terminal-target change was made.

Two [own-trajectory mutation searches](results/defense/diagnostics/trajectory-search-140/README.md)
tested 20,000 distinct score-ranked candidate suffixes each from exact
verified first-life prefixes at frames 288 and
[160](results/defense/diagnostics/trajectory-early-141/README.md). Both
preserved the 2,620-point first-life plateau and extended visible survival
by at most two actions; neither produced extra score or stage two. The
entire candidate-plan/RNG histories are archived. These were diagnostics,
not neural training or learned replays; the result argues against simply
extending the same score-first local sequence search.

The next [screen-diverse trajectory selector](results/defense/diagnostics/trajectory-diverse-142/README.md)
kept 1,024 raw-visible-screen cells at the pre-gap approach and drew
10,060 of 20,000 mutation parents from that archive. It encountered
2,310 distinct cells, but still found no candidate above 2,620
first-life points or stage two; its best delayed visible loss by one
action. All plans, outcomes and RNG state are preserved, and the full
**498-test** suite passed before production. This was also diagnostic
only, not a trained policy or promoted replay. More of this exact
mutation topology is not supported by its observed outcome.

The next [learned spatial-residual comparison](results/defense/training/ppo-spatial-residual-143/README.md)
tests whether mixing neighboring screen features before the original dense
layer helps the policy generalize to the repeated opening. Its zero-gated
spatial and matched ordinary-duration arms start with exactly the same
own trained policy outputs and fresh optimizer/RNG, with no new reward or
game-state input. Both 16,384-action native smokes and a spatial resume
check finished, and their independently verified stage-one replays are
archived. The full **501-test** suite passes. The 524,288-action matched
stage gate has now finished: spatial/control best fixed means were
**10,456 / 10,462**, with all 80 fixed games still in stage one. On 64
untouched matched complete games each, spatial/control/frozen-source
means were **10,224.84 / 10,294.69 / 10,387.50**, all stage one. The
spatial arm also had six sub-9,000 games versus three and two,
respectively. Fresh verified replays still lose at decoded original
course rows 33–34. The spatial residual is a negative result, not a
score parent or promoted global replay; full states and records are
preserved.

The [bounded stronger phase-noise follow-up](results/defense/training/ppo-duration-strong-phase-144/README.md)
now tests whether more vigorous, still direction-neutral key/duration
exploration can escape the same failure point. It resumes the original
option-credit full state and changes training-only noise strength, not
screen input, displayed-score reward, or frozen evaluation. Its short
native smoke completed with a verified stage-one replay; the planned
524,288-action first gate stopped early on its predeclared collapse rule:
its first two noise-free ten-game means were **2,967 / 3,980**, all
stage one. It still sampled 540 long holds and nearly 8,000 key-factor
redraws before stopping, so the stronger exploration happened but did
not improve learned play. Full stopped model/optimizer/RNG state and a
native-verified local replay are archived. It is not extended, compared
on fresh seeds, or promoted; the protected best remains unchanged.

An [exhaustive two-phase gap diagnostic](results/defense/diagnostics/phase-grid-145/README.md)
then tested all **7,500** combinations of two unbiased short physical
commands from one exact native-verified learned first-life approach.
None exceeded the source's **2,620** first-life points, reached stage
two, or survived its 512-action horizon. Seven candidates at the same
score delayed visible loss by only one action. This is a bounded
black-box feasibility check, **not** a trained policy, extra reward,
demonstration or promoted replay; it rules out only that fixed
open-loop family from that particular state. The full **503-test**
suite passed before the production grid.

The next [fine-cadence learned-option probe](results/defense/training/ppo-duration-fine-cadence-146/README.md)
holds the strong option-credit weights fixed and compares native
100,000-T-state actions with 50,000-T-state actions on 32 matched
complete-game seeds. Stride-2 history at the fine cadence retains a
**10,477.19** mean versus **10,480** native, while stride-1 history
falls to **7,942.81**; all 96 games still lose in stage one. Matching
the original physical screen-history span therefore preserves nearly
all frozen score consistency while doubling decision opportunities.
The [selected-replay screen sheets](results/defense/diagnostics/spatial-143-loss-sheets/README.md)
show the ship still far from the right opening in recent policies.
The subsequent bounded **524,288-action** fine-cadence PPO adaptation
finished normally. Its fixed ten-game means were **10,476 / 10,480 /
10,470 / 10,458**, all stage one. Its earliest best checkpoint beat
its frozen same-timing parent by **21.88** then **25.47** mean points
on two independent matched 64-game sets, with no worse low-score tail;
it is eligible as a *score-training* parent, not a global-best replay
or a passage claim. The [verified replay audit](results/defense/diagnostics/fine-cadence-146-course-progress/README.md)
still finds losses at original stream rows **33 / 33 / 33 / 34 of 126**.
The same-place failure survives a doubled decision rate: further work
must address long-horizon exploration and credit at this bottleneck,
not simply replay the same fixed-seed ceiling.

The [long physical-hold integration trial](results/defense/training/ppo-duration-long-option-147/README.md)
tested whether restoring a 6.4-million-T-state learned hold at the
fine decision cadence could improve that approach. It copied the
confirmed fine-cadence model's 80 existing action logits exactly and
added a direction-neutral 128-step option. Three 16,384-action smokes
with uniform new-option logit offsets −2 / 0 / +2 started only
**1 / 2 / 4** long holds, below the predeclared exposure gate.
Their ten-game means were **10,470 / 10,214 / 9,220**, all stage one;
a +2 optimizer-resume integration check dropped further to **6,012**.
All four local best replays independently verified, and full states
are archived. The planned production comparison was not launched.
A [read-only frozen-logit audit](results/defense/training/ppo-duration-long-option-147/frozen-parent-duration-mass.json)
shows why: the source assigns roughly 98–99% probability to one-step
holds on selected pre-loss screens and around 0.001% to 64-step holds.
Appending a longer row to that collapsed duration distribution barely
explores it; a stronger flat prior also harms score. No stage passage
or best-replay promotion resulted.

The follow-up [duration-mixture test](results/defense/training/ppo-duration-mixture-148/README.md)
keeps the same cautious five-option actor but makes duration exploration
explicit during training. At each real option start an 8% mixture
draws a uniform duration while preserving the actor's physical-key
marginal. PPO stores and differentiates the *actual mixed likelihood*;
unperturbed complete-game evaluation is unchanged. Its treatment and
matched no-mixture control passed 16,384-action training and independent
replay smokes. The treatment genuinely started **52** 128-step holds
versus **one** in the control; the control's weights and optimizer bytes
exactly reproduce the earlier default-path run. Both full-state resume
checks also passed. This is an exposure/integration result, not stage
passage. The ensuing matched production test stopped the mixture arm
after two fixed means **3,616 / 2,305**, while its no-mixture control
completed all 524,288 actions with fixed means **10,472 / 9,956 /
10,416 / 10,446**; every game remained stage one. On 64 new matched
complete games, treatment/control/unchanged parent means were
**5,177.97 / 10,067.50 / 10,420.63**, with **61 / 10 / 1** games
below 9,000 and no later-stage games. The treatment lost to the parent
on all 64 paired seeds. Native-verified fresh replays still lose near
original stream rows **32–34**. This correctly exposes long actions
but hurts score, so neither new arm is promoted; full states and
replays are archived.

The [adjacent later-turn two-phase grids](results/defense/diagnostics/late-phase-grid-149/README.md)
then closed a specific timing gap in the earlier action-322 search:
from exact verified own first-life anchors **339 and 340**, they
tested **7,500** symmetric command/duration plans each. Neither
exceeded the learned 2,620-point first-life plateau, entered stage
two or survived its 512-action horizon. A few action-340 plans delayed
the visible loss by one action at equal score. These were diagnostic
counterfactuals only, never policy actions, rewards or demonstrations;
the frozen learned checkpoints and best replay are unchanged.

The [third-phase follow-up](results/defense/diagnostics/three-phase-grid-150/README.md)
expanded this bounded late correction family to **81,000** plans from
each of the same two verified anchors. All **162,000** full candidates
were executed in the original emulator, with complete outcomes and source
hashes preserved. None exceeded the source's **2,620** first-life points,
reached stage two, survived the 512-action horizon, or even extended
same-score survival. The recorded screen at action 339 visibly shows a
near lower opening at columns **21–30**, the ship near column **23**,
and the next upper opening at **51–62**; by action 387 the ship is
still near **25** as the latter wall approaches. This motivates an
*earlier screen-conditioned anticipation* experiment rather than more
late fixed-action substitutions. It is diagnostic evidence only, not a
learned escape route; no searched action entered training or promotion.

An [earlier visible prepositioning audit](results/defense/diagnostics/preposition-window-151/README.md)
then tested every **3,600** single-RIGHT start/hold pair from action
280 to the frame-355 waypoint of that same verified first life. At
frame 339, the farthest full-score readable ship glyph was column
**25** versus the learner's **23**. At frame 355, the farthest was
**29** versus **25**, among **855** readable branches that kept all
2,600 displayed points. Farther-right branches had typically missed
the earlier reward-bearing passage; no branch reached another stage.
The complete native outcome table and hashes are archived. This
rules out only one contiguous RIGHT correction, and reinforces the
need to investigate multi-step, earlier screen-conditioned control
without treating diagnostic actions as demonstrations.

A [matched small-duration-mixture PPO trial](results/defense/training/ppo-duration-short-mix-152/README.md)
then tested a 2% duration exploration floor over only the existing
1/4/16/64-action options, versus a no-mixture control from identical
verified fine-cadence weights. The treatment genuinely started
**1,593** 64-action options by its last progress event, versus **374**
for control, without the catastrophic fixed-score collapse seen with
the added 128-action option. Both arms completed **524,288** actions
and four complete-game checks, all stage one. On **64 new matched
complete games**, selected treatment/control/unchanged-parent means
were **10,105.31 / 10,032.50 / 10,412.34**, with **9 / 9 / 2**
sub-9,000 games and zero stage-two games. Both fresh-best replays
independently verify all learned actions and again lose four 2,620-
point lives. The treatment is negative versus its parent, so neither
new arm is promoted; complete optimizer/RNG states, evaluations and
replays are archived, and the protected best remains available.

An [own-previous-action recurrent PPO comparison](results/defense/training/ppo-own-action-memory-153/README.md)
then copied the strongest ordinary 20-action PPO into two zero-output
residual GRUs, freezing the same base. The treatment GRU received its
own prior chosen key plus screen history; control received only
screens. Both saved initializers exactly matched the parent's
original-boot actions/screens/rewards on two complete parity games,
and the full **514-test** suite passed. After 262,144 training actions
per arm, all 40 fixed games remained stage one. The first 64 fresh
matched games gave treatment/control/parent means **10,412.97 /
10,452.50 / 10,414.38**; a separately frozen 64-game confirmation
reversed the new-arm ordering at **10,457.19 / 10,424.06 / 10,412.81**.
All **384** fresh games stayed stage one. The control failed its
predeclared confirmation against treatment, and treatment had failed
its first-set score-parent gate, so neither is promoted. Both native-
verified fresh replays still lose four 2,620-point lives at the
recurring barrier. Full optimizer/RNG states, six fresh evaluation
tables and exact source metadata are archived; the protected global
best model and replay remain unchanged.

Matched five-option learned-duration / one-step full continuations (56/55)
retired after four stage-1-only rounds. Their calibrations averaged 6,266 / 6,365, all stage-1 losses;
this improves on their one-option calibrations but remains below the best.
Their first full means are 358 / 9,128: variable duration has regressed,
while the control recovers score without clearing stage 1.
The second means are 1,136 / 9,502, again without a later stage.
The third means are 1,701 / 9,484 and the fourth 1,271 / 4,280. A separate matched calibration
at discount .999 averaged 10,266 with temporal consistency versus 9,710 without,
all stage-1 losses. Full consistency/control continuations (58/57) have retired
after six complete evaluation rounds each, preserving final state and logs;
their first full means are **10,262 / 9,890**, with all 20 games still losing
in stage 1. The [current screen review](results/defense/diagnostics/consistency-full-losses-7293216/README.md)
confirms the recurring barrier approach, not new navigation progress.
Full states, log prefixes and verified replays are preserved. No success is
inferred from learning losses or near-ceiling scores.
The second full consistency/control means are **10,023 / 10,418** and the
third **10,328 / 10,214**, again all stage 1. The control's second replay
matches 10,480 (2,577 verified commands); consistency's third reaches 10,400
(2,523 verified commands). Neither promotes over the shared best.
The fourth through sixth consistency/control means are **10,337 / 10,174**,
**10,228 / 10,285**, and **10,337 / 10,207**. All 120 full-run validation games
lost in stage 1. The final consistency replay scores 10,420 with 2,499 verified
commands; the global best is unchanged.
A [world-model preflight](results/defense/training/world-model-preflight-01/README.md)
finished 5,000 dynamics updates on its first own collection. A separate
[imagined-return actor](results/defense/training/imagination-calibration-01/README.md)
is now playable and natively replay-verified, but performs poorly: mean
294, best 320 at 500 actor updates; mean 282 at 1,000. This is a frozen-world
calibration, not an online Dreamer loop or a navigation improvement.
The [broader-data continuation](results/defense/training/world-model-mixed-01/README.md)
finished at 10,000 world updates using 85,779 own training actions and eight
whole held-out games. It still misses visible life losses even when observing
the arrival screen, not just during long-range prediction. A completed
[matched sampling test](results/defense/training/world-model-boundary-01/README.md)
finds that 50% own-training loss-window sampling improves selected-loss
recognition but worsens overall held-out continuation error. The subsequent
[playing comparison](results/defense/training/imagination-boundary-comparison-01/README.md)
is negative: uniform/focused actors average 280 / 270 at 1,000 imagined
updates, with all 60 baseline and trained games losing in stage 1. Rewards,
inputs, game and held-out assignments remain unchanged. These offline results
motivate [new actor experience](results/defense/training/world-model-actor-collection-03/README.md)
for model feedback, not claiming that the recurring barrier has been solved.
That collection finished 24 fresh games, and the [first feedback world fit](results/defense/training/world-model-feedback-01/README.md)
completed 14,000 total updates using 60 training games / 116,495 actions with
twelve whole games held out. The [first actor feedback continuation](results/defense/training/imagination-feedback-01/README.md)
retained its complete learned behavior/optimizer/RNG state but finished with
mean 276 after 1,000 additional imagined updates, still all stage 1. Three
further focused tests passed after the full 393-test suite; the original five
actor tests were rerun successfully after adding explicit world refresh.
The [second feedback cycle](results/defense/training/imagination-feedback-02/README.md)
regressed to mean 110. A subsequent [matched four-step latent regularizer](results/defense/training/world-model-overshoot-01/README.md)
also failed to improve loss forecasts or play; final control/regularized actor
means are 120 / 112, all stage 1. Both complete states, all intermediate
checkpoints and verified local bests are retained. The global collector now
watches 67 sources and still preserves the stronger 10,480-point replay.
A [frozen text-reconstruction diagnostic](results/defense/diagnostics/world-text-reconstruction-01/README.md)
finds weak reconstruction of changed visible text even when observing the
arrival screen. It includes tolerance and star/space checks to avoid equating
small numeric errors with total information loss. This motivates inspecting
the reconstruction objective; it is not proof of the collision cause or a fix.
A completed [categorical visible-byte comparison](results/defense/training/world-model-bytes-01/README.md)
qualifies that diagnosis: a new decoder on the **frozen** parent recovers 90.91%
of nonblank ASCII, versus 91.22% with joint retraining. Poor rounded output did
not establish missing latent information. Life-loss forecasts remain poor,
and matched continued actor means fall to 104 / 102; all 60 complete validation
games lose in stage 1. These weak actors fail earlier than the strong shared
best. Full world/actor/decoder optimizer states, logs and verified local bests
are preserved. Five new byte/readout tests pass together after the full
403-test suite (the final frozen-readout test was added afterward).
A [frozen life-loss readout and head-only continuation](results/defense/training/world-life-readout-01/README.md)
then separated recognition from anticipation. Refitting the continuation head
with its full Adam preserved improves recognition but not play. Extending from
2,000 to 20,000 head-only updates worsens held-out one-step calibration; both
actor tests still lose in stage 1, finishing at means 102 / 104. Their 60 new
complete validation games use reused seeds, not fresh success estimates. Both
1,628-command local best replays and every intermediate state are preserved.
Six new readout/head tests pass after the full 409-test suite; the last
head-only test was added afterward. Further head-only fitting is retired.
A [matched direct-prior-output experiment](results/defense/training/world-prior-output-01/README.md)
adds optional supervision of the next screen, score change and continuation
without observing the arrival screen. It improves some broad graphics errors
but not 24-step changed-object error or useful loss anticipation. Continued
actor means finish at 120 control / 110 treatment, with all 60 games still
stage-1 losses. Full states, logs and both verified 340-point local replays
are preserved. The default objective remains parameter-exact in a real-data
parity check, and the full 414-test suite passes.
The earlier one-option continuations (54/53) retired after five rounds.
Their calibrations averaged 3,631 / 344, with all games still stage-1 losses:
a large relative difference against a regressed control, not a new best.
Their five full evaluation means are 1,800 / 377, 9,016 / 342,
8,379 / 1,328, 7,362 / 1,240 and 5,613 / 516. The variable-duration run's 9,980-point replay still shows the
recurring barrier sequence; the shared 10,480-point best remains unchanged.
Its run-best is 10,240, still stage 1. All final states and logs are preserved.
The lower-exploration
focused continuation (52) retired after six stage-1-only evaluations. Its calibration
averaged 10,297 versus high exploration's 10,214, but only one of ten paired
scores improved and all games remained in stage 1. The matched restored-life
continuations (49/50) retired cleanly after six stage-1-only evaluations each,
with all final state and logs preserved. Balanced-command run 51 also retired
after six stage-1-only evaluations; all six means were below its uniform-action
comparison. Final state and complete logs remain preserved.
The separate 128-action own-loss-lookback calibration finished below its
64-action counterpart, and an optional command-balanced exploration sampler
completed a near-tied calibration and a negative longer continuation. The 64-action
score-triggered / own-loss-triggered pair (47/48) retired after six stage-1-only
rounds with full state preserved. The focused calibration improved mean by 251
over its control, but the first full round scored 78 lower; all games still
lost in stage 1. Longer training is not a claimed breakthrough.
Visual prediction (45) and inverse-action classification (46) retired after
five and six stage-1-only evaluations, preserving final optimizers and logs.
The own-loss calibration completed below its
score-triggered control, with no stage passage; the longer pair tests adaptation,
not a demonstrated advantage.
Trace cutting (43) and longer persistence (44) retired
after six stage-1-only rounds each;
higher discount (42) retired after seven. Visual prediction's calibration improved mean score
but still failed at the recurring barrier; its full continuation tests longer
adaptation, not an established fix.
The inverse-action calibration scored below the baseline, and its frozen
classifier did not outperform an action-frequency prior. Its full continuation
tests longer adaptation, not a demonstrated capability.
A separate scalar-DQN calibration completed a 64-action own-state rewind against
the historical 128-action worker-split baseline, with both auxiliary losses off.
Its mean was **10,398 versus 9,466**, but all ten games still lost in stage 1;
the verified replay repeats the familiar barrier sequence. Its full state is
preserved, without replacing the shared best.
Worker-split exploration
(40) and its uniform control (41) retired after nine and seven evaluation
rounds, respectively, without a later stage.
Runs 33–39 have retired with full final state and logs preserved; the historical
updates below record their earlier trajectories. None has reached stage 2.
The trace-cut check changes training targets only, using the same own learned
parent and settings as the preserved five-step comparison. Its completed
calibration regressed to mean **8,324** versus **9,466**, all stage 1; run 43
tests longer adaptation, not a claimed solution to the recurring barrier.
The longer-persistence calibration improved mean to **10,097** versus **9,466**,
but all games remain stage-1 losses; run 44 tests whether it can advance.
Bootstrap DQN 24, PPO 29 and frozen-memory PPO 30 later retired after depth
plateaus or sustained regression, with all results preserved.
Training remains independent of Breakdown, with complete-game
validation and automatic verified best-effort replays. No history was deleted.
Full DQN trials 31/32 finished eight matched rounds without a new stage and
retired with full state preserved. Mean-score ranking varied; all games lost
in stage 1.
A bounded matched check of 5% versus 25% nominal persistent exploration
finished without a new stage; the higher rate scored worse. Neither full
run's settings changed.
Bounded follow-ups combine persistent exploration with their own newly
reached-state resets. Both have finished without a new stage; the 32-action
lookback improves the bounded comparison's mean and archive coverage, but
does not solve the shared failure pattern.
Full trials 33/34 subsequently retired after six complete paired rounds without
a stage clear, preserving final state and complete logs. Their mean differences
(resets minus control) were **+62 / −305 / +240 / +39 / +596 / +782**.
All 120 evaluation games were stage-1 losses.
A read-only prediction/return diagnostic does not support
gross near-loss Q-value inflation as the explanation.
A bounded archive-diversity pair favored screen fingerprints over score bins
by 384 mean points, with equal maximum archive capacity and identical parent
state. Both still fell below the parent and lost every evaluation in stage 1.
Full trials 35/36 tested those capacity-matched archive-selection arms.
Their first three full rounds average **10,198 / 10,133**, **9,836 / 10,172**
and **10,085 / 10,198**, respectively; every game still loses in stage 1.
Verified replay loss panels show the same broad
right-opening barrier sequence; archive diversity has not resolved it.
Run 35 later retired after six rounds without a stage clear, with its full
final state preserved. Run 36 continues; its sixth-round mean is **10,418**,
best **10,460**, still stage 1. Run 39 now gives the earlier 128-action-lookback
calibration a full continuation, testing more preparation time rather than
treating its lower short-check score as evidence against eventual progression.
A matched quantile-DQN calibration found risk distortion worse than neutral
exploration (**10,146 / 10,280** mean), without a later stage. Full trials
37/38 now continue both quantile arms from their own checked states. They add
no observations, demonstrations or rewards.
Their first full round reverses the short comparison's mean ranking
(neutral **9,939**, risk **10,283**), still without a later stage. Separately
playing the higher-return criterion scored worse on both frozen checkpoints;
those independently verified probes do not replace standard mean-greedy play.
The second standard round is **6,317 / 10,196**: neutral has regressed, while
risk retains near-ceiling scoring, still with no stage clear.
A successful mission has not yet been verified.
The [latest failure review](results/defense/diagnostics/shared-loss-latest-01/README.md)
compares the shared best with the preserved 10,180-point run-43 and 10,260-point
run-44 replays. The best repeats exactly 2,620 points on each life; best/44
panels show the recurring right-opening barrier sequence, whereas 43 fails
earlier near a centre-opening barrier. This supports a navigation bottleneck,
not identical collisions or a proven wall-versus-projectile cause. The review
is diagnostic only and adds no evaluation trajectories to training.
The current standard-policy best is **10,480 points**, with **2,580** neural
actions exactly reverified; its ten-game mean is **9,981**, median **10,380**, all stage 1.
Breakdown's frozen models, published site and results are unchanged. Shared
network/sampler code now supports configurable action counts while preserving
the original six-action defaults.

## Play and inspect

```sh
venv/bin/python main.py -m Play --game defense
```

Allow the animated title to appear. Enter shows instructions; **F1** maps to
the TRS-80 CLEAR key and starts player selection. Press **1** for one player.
Stage introductions finish automatically (Enter can skip them during manual
play). Arrow keys move, Space fires. F2 is BREAK.

| Input | Effect |
|---|---|
| Eight arrow directions, including diagonals | Movement |
| Space | Forward fire |
| Left + Right together | Side fire in stage 1 |
| Left + Right + Space | Both firing modes in stage 1 |
| CLEAR + BREAK (F1 + F2) | Abort to title; excluded from policy actions |

One important assembly-level distinction: stage 1 compares the **whole** key
byte for movement, so adding Space to an arrow stops movement while firing.
Stages 2 and 3 mask off Space before decoding movement and allow moving while
firing. The environment therefore offers **20 fixed actions**: nine movement/
no-op choices, those nine with Space, and the two side-fire combinations.
Some actions have equivalent effects in a particular stage. There is no
stage-aware controller choosing or overriding policy actions.

An optional **21st action, Enter**, is available with `--allow-enter`. The
neural policy alone chooses whether to press it; there is no automatic intro
skipper. It can dismiss stage introductions, but it cannot trigger CLEAR+BREAK
or select a new game. Existing 20-action checkpoints keep their original
action mapping and timing. A different action profile requires a fresh run,
not loading an incompatible optimizer/head into an existing checkpoint.

## Evidence and game identity

The supplied ZIP contains `command.CMD` and `disk_0.dmk`, not assembly source.
The analysis below is a read-only **disassembly of the original Z80 binary**.
`command.CMD` is byte-for-byte identical to `var/defense.cmd` (24,704 bytes).
The disk's active `DEFENSE.CMD` file was independently extracted in memory:
it is identical too. All 720 data-sector CRCs checked successfully.

Executable SHA-256:
`9b887e46223eb2ce9d2ada2a7588435a90be51f9e5326a592193faeafa322644`

DMK SHA-256:
`31bd3aca846f61b34ed60242d7404d4bad0f03cae67512fb74e0942932f8e801`

The CMD enters at `0xA870`. Static code addresses below are engineering audit
references, **not policy inputs or reward sources**.
The [reproducible ending audit](results/defense/game-ending-audit.json) checks
the instruction bytes directly; rerun with `venv/bin/python -m rl.defense_audit`.

| What the code establishes | Relevant addresses |
|---|---|
| CLEAR starts; one/two-player selection reads number-key matrix | `0x6F16`, `0x6F23`, `0x6F3A` |
| Four ships per player; one-player mode disables player 2 | `0x6F6F` |
| Three stage dispatches; increment stage, then wrap 4 to 1 | `0x6F86`–`0x700E` |
| Stage 1 firing and exact-byte movement comparisons | `0x758B`, `0x75AC`, `0x75E4` |
| Stage 2 movement masks Space; forward fire tests bit 7 | `0x8518`, `0x85B0` |
| Stage 3 movement masks Space; forward fire tests bit 7 | `0xA709`, `0xA7C4` |
| CLEAR + BREAK abort | `0x7683` |
| Decrement the current player's ships; draw game over at zero | `0x7010`, `0x7070` |
| Seven-digit score addition; periodic survival points | `0x7D86`, `0xABEB` |
| HUD formats score and `*` ship markers, then copies to video | `0x7DFD`–`0x7EFC` |
| Successful stage 3 calls congratulations animation | `0xA016`–`0xA01A`, `0xA2F9` |
| Writes `YOU did it` on row 8 | `0xA358`–`0xA361` |

## Observations, rewards and outcomes

`rl.defense.DefenseEnv` has the same basic reset/step shape as Breakdown but
independent game logic and action IDs. The observation is four raw `uint8`
16×64 screen frames: only video memory `0x3C00`–`0x3FFF`. The current game's
private score, ship coordinates, collision flags, random generator, stage
variables and off-screen graphics buffer are **not read by the environment**.

Booting uses CLEAR and 1 only before gameplay, waiting for the actual player
prompt. A seeded title-timing offset provides deterministic reset variation;
it is not a claim of independent, randomized game layouts. No automatic firing,
steering, life recovery, expert demonstrations or action oracle is supplied.

Visible signals:

- **Score:** leftmost 16 columns of row 0; variable-width digits followed by
  ships. The centered number is the high score, not the player's score.
- **Ships:** `*` characters following the score; initially four. A lost ship
  restarts that stage automatically. Blanking/scrolling screens mean unknown,
  not zero lives. Life-loss flags may lag the collision until the HUD updates.
- **Stage:** distinctive intro sentences: “You are now entering the”, “Find
  your way through the”, and “Now at last you enter the”. The last observed
  stage is retained during gameplay; it is not an always-visible level counter.
- **Mission completion:** `YOU did it` on row 8 (substring at video `0x3E1B`).
  Generic `CONGRATULATIONS` alone is insufficient: high-score entry also uses it.
- **Loss/episode end:** `GAME OVER PLAYER 1`, row 7 at `0x3DD7`.

The score deliberately blinks. Changed HUD copies are allowed to settle with
the **same chosen keys**. At game over, the original death animation must
finish updating the zero-ship score before reward is finalized: stopping at
the first GAME OVER can miss the last 20 points. The wrapper waits only for
that visible update, then stops. All reward is visible score difference;
there are no bonuses for lives, stages or mission detection.

The game loops its three stages. A defensible first achievement is **complete
one three-stage mission**, witnessed by `YOU did it`, while a full evaluation
game continues until all ships are gone. Higher score alone does not prove
mission completion. A diagnostic action cap is reported as **truncated**, never
as a win or complete loss.

The [stage-one course audit](results/defense/stage-one-course-audit.json) verifies
that the original first stage advances automatically when its scrolling
obstacle stream is exhausted, **not when a score threshold is reached**. Static
decoding finds 126 stream rows between `0x76A9` and the end marker at `0x7BD8`;
the original row-update countdown is six. These are engineering facts about
the immutable game asset, not measurements of any policy's progress. No stream
pointer, row counter, layout-derived route, or other hidden state is supplied
to the policy, reward, or curriculum. A learned full-game stage transition
must still be observed on screen before reporting a clear.

## Verified smoke tests and replays

| Controller (not learned) | Complete games | Mean | Median | Best | Highest stage |
|---|---:|---:|---:|---:|---:|
| Uniform random actions, seeds 0–9 | 10 | 286 | 280 | 320 | 1 |
| No input, seeds 100–109 | 10 | 280 | 280 | 280 | 1 |

All 20 diagnostic games detected four ship losses and ended at game over.
No mission was completed. These are integration baselines, not evidence of
learning. Survival itself earns points, so compare stage/mission progress as
well as score when evaluating a future model.

A separate read-only engineering diagnostic compared final visible totals
with the internal score for 20 random games, exposing the terminal-redraw
issue described above and confirming the fix. Those internal reads are not
in the environment, baseline recorder or any policy/reward path; no game
memory was patched and no training examples were produced from them.

- [Random-action replay](results/defense/random-baseline/replay.html): 1,603
  actions; every resulting screen and reward checked by re-execution.
- [No-input replay](results/defense/noop-baseline/replay.html): 1,535 actions,
  likewise checked.
- Full records: each directory's `report.json`; compressed screen/action/
  reward trace in `trace.npz`; no-input [screen samples](results/defense/noop-baseline/screens.png).

Reproduce into a **new directory** (existing evidence is never overwritten):

```sh
venv/bin/python -m rl.defense_smoke --output runs/defense-random-check --games 10 --seed 0
venv/bin/python -m rl.defense_smoke --output runs/defense-noop-check --games 10 --seed 100 --policy noop
venv/bin/python -m unittest tests.test_defense -v
node tests/test_defense_replay.js
```

## Is there enough information to train?

**Yes.** Inputs, keyboard actions, actual score reward, resets and loss boundaries
are established and exercised in the real emulator. The separate environment
is ready for a learner adapter:

```python
from rl.defense import DefenseEnv, ACTIONS

env = DefenseEnv(seed=7, max_steps=0)  # 0 means no artificial episode cap
try:
    observation = env.reset()
    # action = your_new_policy(observation), integer in [0, len(ACTIONS))
    observation, reward, terminated, truncated, info = env.step(0)
finally:
    env.close()
```

The new learner is a fresh 20-action MLX PPO policy on this Mac. It reuses the
screen encoder/optimizer implementation, **not Breakdown's weights, six-action
head, curriculum, level-ranking rules or game-specific evaluation**. The existing
`rl.ppo` / `rl.train` CLI remains Breakdown-only. Defense has its own command:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-ppo-01 \
  --artifacts results/defense/learned --envs 32 --rollout 128 \
  --batch-size 512 --epochs 4 --eval-every 100000 --eval-games 10 --eval-envs 10
```

The command defaults to unlimited training and complete, uncapped episodes.
Reward is the actual visible score delta multiplied by a fixed 0.01 for the
optimizer's units; there is no shaping or clipping. Ship loss is not a separate
learning terminal by default. SIGINT/SIGTERM saves `latest` and exits cleanly.
Resume with `--resume runs/defense-ppo-01/latest`; the optimizer and policy RNG
are restored, while emulator episodes restart from boot (not exact trajectory
continuation). Evaluation uses fixed validation seeds 10000–10009; do not use
fresh-test seeds to tune the model.

- Live unlimited progress: `runs/defense-dqn-33-persistent-resets/status.json`,
  `runs/defense-dqn-34-persistent-rate-control/status.json`,
  `runs/defense-dqn-35-screen-archive/status.json` and
  `runs/defense-dqn-36-score-archive-control/status.json`, each with an adjacent
  `metrics.jsonl`. Earlier trials have stopped cleanly; their outcomes and
  archived resumable checkpoints are recorded below. Confirm a status file's
  PID is still alive before treating it as evidence of a running learner.
- Completed bounded persistent-exploration comparison:
  `runs/defense-persistent-control-01/status.json` and
  `runs/defense-persistent-memory-01/status.json`. These are training checks,
  excluded from the global collector. The latter name refers to holding a
  random action, **not** a recurrent neural network.
- Completed bounded exploration-rate comparison:
  `runs/defense-persistent-rate-control-01/status.json` and
  `runs/defense-persistent-rate-high-01/status.json`, with adjacent logs.
  These short checks are excluded from the global collector.
- Completed bounded persistent-exploration/own-reset follow-up:
  `runs/defense-persistent-reset-calibration-01/status.json` (lookback 128,
  finished) and `runs/defense-persistent-reset-short-calibration-01/status.json`
  (lookback 32, finished), with adjacent logs. Both sources are excluded from
  the global collector.
- Completed bounded archive-diversity comparison:
  `runs/defense-dqn-screen-cells-calibration-01/status.json` and
  `runs/defense-dqn-score-cells-control-01/status.json`, with adjacent logs.
  These sources and the numerical-parity smoke runs are excluded from the collector.
- Historical checkpoints: `runs/defense-ppo-*/step-*/` and `runs/defense-dqn-*/step-*/`, including optimizer,
  configuration, policy weights and each completed validation suite.
- Stable best effort, once a validation candidate is verified:
  [replay](results/defense/learned/best/replay.html) and
  [weights](results/defense/learned/best/model.safetensors).
- Every promotion appends an immutable bundle under `results/defense/learned/versions/`
  and atomically switches the `best` symlink. Earlier versions are preserved.
  Each contains weights, configuration, evaluation, SHA-256 manifest, action/
  screen/reward trace, and a verification report.
- With concurrent learners, the single `rl.defense_collect` process owns that
  shared archive. Its status/log live in `runs/defense-collector/`. The learner
  processes write only to their separate experiment artifact roots.

After the storage pause, both original learner processes were confirmed gone
and every pause-checkpoint file was checked against its archived copy before
restart. DQN resumed at **7,383,056** actions and PPO at **15,674,112**.
Learning settings remain unchanged; newer optional features retain their
disabled/default behavior. The preserved
[DQN restart configuration](results/defense/training/dqn-24-bootstrap/resume-after-storage-config.json)
and [PPO restart configuration](results/defense/training/ppo-29-value-weight/resume-after-storage-config.json)
record the exact source hashes and ancestry. The
[259-test result](results/defense/diagnostics/post-storage-regression-tests.txt)
passed after space was freed. Both processes were observed advancing counters;
the original sole collector remained live, with short experiments excluded.

Weights, optimizer, DQN targets/priors and saved RNG states are restored, but
games restart from boot, DQN refills its replay buffer from new own experience,
and PPO refills its own-state archive. This is not exact trajectory or replay-
buffer continuation. Initial training-game scores during random DQN warmup are
not frozen-policy validation results. The stronger shared best is preserved.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-24-bootstrap \
  --artifacts runs/defense-dqn-24-bootstrap/artifacts \
  --resume results/defense/training/dqn-24-bootstrap/pause-checkpoint-000007383056 \
  --steps 0
venv/bin/python -u -m rl.defense_train --run runs/defense-ppo-29-value-weight \
  --artifacts runs/defense-ppo-29-value-weight/artifacts \
  --resume results/defense/training/ppo-29-value-weight/pause-checkpoint-000015674112 \
  --steps 0
```

These are the recorded restart commands, not commands to launch duplicate
learners while those runs are already active. Use the latest preserved
checkpoint and an unoccupied run directory for a later continuation.

Selection prioritizes completed missions, then highest stage, then score, among
**complete from-boot games only**. The stable best is a best single effort, not
a claim of reliable mean performance. Before promotion, the frozen weights are
reloaded and must exactly reproduce every neural action, reward and screen.
No unverified or truncated replay replaces the best.

The collector checks the isolated archives every 30 seconds, pins an immutable
source version, checks its checksums, and invokes the same full frozen-policy
verification before a stronger candidate can replace the global best. It does
not promote sampling diagnostics, change training, or push to GitHub. A
destination lock prevents duplicate collectors. Do not simultaneously point a
trainer directly at the shared archive while this collector owns it.

```sh
venv/bin/python -u -m rl.defense_collect \
  --source runs/defense-ppo-05-low-entropy/artifacts \
  --source runs/defense-ppo-06-life-boundary/artifacts \
  --source runs/defense-ppo-07-curriculum/artifacts \
  --source runs/defense-ppo-08-long-rollout/artifacts \
  --source runs/defense-ppo-09-curriculum-life/artifacts \
  --source runs/defense-ppo-10-long-credit/artifacts \
  --source runs/defense-ppo-11-exploration/artifacts \
  --source runs/defense-ppo-12-lookback/artifacts \
  --source runs/defense-ppo-13-screen-cells/artifacts \
  --source runs/defense-ppo-14-long-horizon/artifacts \
  --source runs/defense-ppo-15-bias-noise/artifacts \
  --source runs/defense-ppo-16-strong-bias-noise/artifacts \
  --source runs/defense-ppo-17-fresh-seed/artifacts \
  --source runs/defense-ppo-18-weight-noise/artifacts \
  --source runs/defense-ppo-19-moderate-weight-noise/artifacts \
  --source runs/defense-ppo-20-encoder-transfer/artifacts \
  --source runs/defense-ppo-21-long-lookback/artifacts \
  --source runs/defense-dqn-22-fresh/artifacts \
  --source runs/defense-ppo-23-life-age/artifacts \
  --source runs/defense-dqn-24-bootstrap/artifacts \
  --source runs/defense-ppo-25-matched-history/artifacts \
  --source runs/defense-dqn-26-own-resets/artifacts \
  --source runs/defense-ppo-27-matched-history-low-lr/artifacts \
  --source runs/defense-dqn-28-large-replay/artifacts \
  --source runs/defense-ppo-29-value-weight/artifacts \
  --source runs/defense-ppo-30-frozen-memory/artifacts \
  --source runs/defense-dqn-31-persistent/artifacts \
  --source runs/defense-dqn-32-persistent-control/artifacts \
  --output results/defense/learned --run runs/defense-collector --interval 30
```

The collector passed an isolated end-to-end check: it reloaded run 06's frozen
380-point policy and reproduced all 1,679 actions/screens/rewards before
publishing into the smoke-test directory. A second collector targeting the
live destination was rejected. The production collector is now running; it
leaves the existing global best untouched when source ranks are tied or lower.

Evaluate a frozen checkpoint independently (choose a new output path):

```sh
venv/bin/python -m rl.defense_evaluate \
  results/defense/learned/best/model.safetensors \
  --output runs/defense-fresh-evaluation.json --games 10 --seed 20000 --envs 10
```

The goal remains to observe and verify the original game's mission-ending
screen and subsequent behavior. The three-stage wrap in the binary is not
permission to relabel a ship-loss GAME OVER as a victory or manufacture more
levels. Keep improving until the successful completion sequence is observed.

### Experiment log: initial training and recovery

- `defense-ppo-01`: 531,200 sampled training actions before the strict HUD
  settling check stopped a worker. The trainer saved policy/optimizer state
  and closed its workers; this was not a completed training goal. Its full
  [log](results/defense/training/ppo-01/metrics.jsonl) and
  [configuration](results/defense/training/ppo-01/config.json) are preserved.
- Best frozen checkpoint from that run: step 303,104, score **340**, 1,615
  verified neural actions. Ten complete validation games: mean **310**, median
  **300**, highest stage **1**, no completed mission.
- The guard now permits continuously animated rows that contain no readable
  score/lives, while still rejecting unstable numeric HUDs. It makes no change
  to the timing or outputs of previously successful runs. Both archived learned
  replays were re-executed after the fix: all actions, screens and rewards were
  identical. The new run also records the environment source hash.
- `defense-ppo-02`: resumes run 01's saved policy, optimizer and RNG at action
  counter 531,200, with the same hyperparameters and a fresh set of from-boot
  emulator episodes. No demonstrations or hidden-state gameplay inputs added.
- Run 02 was subsequently checkpointed and paused cleanly at **1,542,912**
  cumulative actions. No stage 2 was observed. Its best frozen checkpoint,
  step **1,133,312**, reached **380** points (1,658 verified neural actions);
  ten-game mean **358**, median **360**, highest stage **1**. Later validation
  fell back to mean 294. All intermediate checkpoints remain in `runs/`, and
  its [complete log](results/defense/training/ppo-02/metrics.jsonl) is preserved.
- A [temporal probe](results/defense/temporal-probes/defense-temporal-932608-400k.json)
  held each frozen-model action for 400,000 rather than 100,000 T-states. On
  the same ten validation seeds, mean changed from 318 to 320 and best stayed
  340; neither protocol reached stage 2. This does not establish improvement.
  The [400k no-input check](results/defense/temporal-probes/noop-400k.json) still
  scored 280 and lost all four ships in ten complete games.
- `defense-ppo-03-enter` is a **fresh 21-action model**, not a resumed
  incompatible 20-action optimizer. Start command is the training command
  above with `--run runs/defense-ppo-03-enter --allow-enter`. All other
  hyperparameters remain the same. Its action counter starts at zero; the
  earlier 1,542,912 actions are separate prior experiment compute.
- The Enter-control test reduced a no-input game from about 1,535 to **1,043**
  decisions without changing its score (280), number of ship losses (four),
  or loss ending. All three between-life stage-1 intros were still observed
  in sampled screen frames, even with Enter continuously held. A 64-action
  GPU smoke test completed model reload and exact 21-action replay verification.
  This is an efficiency hypothesis to test through learning, not a mission win.
- Run 03 stopped cleanly at **1,073,152** actions. Its ten validation rounds
  never reached stage 2. The strongest round was step **200,704**: mean **326**,
  median **320**, best **340**; the last round fell to mean **286**. Its
  [full log](results/defense/training/ppo-03-enter/metrics.jsonl) and strongest
  [resumable checkpoint](results/defense/training/ppo-03-enter/step-000000200704/state.json)
  are archived, including weights and optimizer. The separate 380-point
  20-action best remains unchanged.
- A further [100-game random-action timing check](results/defense/temporal-probes/random-400k-100-games.json)
  at 400,000 T-states yielded mean **288.8**, median **280**, best **320**,
  all stage 1. No episodes hit its diagnostic cap. These diagnostic actions
  are not supplied to the learner.
- `defense-ppo-04-sil` resumes run 03's strongest compatible checkpoint with
  four optional self-imitation updates per rollout. All other training
  settings are unchanged. The auxiliary replay contains only this learner's
  own newly collected training screens, actions and discounted score rewards;
  no evaluation games, replay files, demonstrations or hidden-state targets.
  Positive return advantages select the useful updates. A bounded suffix
  ends at a real learning terminal; truncations are discarded. Replay memory
  is not checkpointed and refills after resuming (the sampling RNG is saved).
  The inherited action counter starts at 200,704, not at run 03's final count.
- Before activation, a [4,096-action smoke run](results/defense/training/sil-smoke-01/metrics.jsonl)
  exercised the auxiliary optimizer and complete-game evaluation. The optimizer
  made 64 PPO updates plus 16 self-imitation updates; its selected 300-point
  replay reproduced all **1,152** neural actions/screens/rewards after reloading
  frozen weights. This is an integration check, not improved performance.
- Run 04 then completed **1,032,192 additional actions**, stopping cleanly at
  inherited counter **1,232,896**, with 928 new complete training games.
  Across ten validation rounds, its best mean was **312**, below the starting
  checkpoint's **326**; the final round averaged **302**. Every round's best
  was at most 320, and no game reached stage 2. This configuration did not
  improve the policy. Its [complete log](results/defense/training/ppo-04-sil/metrics.jsonl)
  and [final resumable checkpoint](results/defense/training/ppo-04-sil/final-checkpoint/state.json)
  are retained; the newer lower-entropy continuation now has the machine.

Continue the self-imitation experiment from the archived starting checkpoint
into a new run directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-sil-reproduction \
  --resume results/defense/training/ppo-03-enter/step-000000200704 \
  --artifacts runs/defense-sil-reproduction/artifacts --sil-updates 4
```

`--sil-updates 0` (the default for a fresh run) leaves plain PPO unchanged.
Only one active production learner should write to `results/defense/learned`;
parallel experiments must use separate artifact directories. Breakdown keeps
its original action count and visible-score parser defaults.

### Sampling diagnostic: preserved 400-point replay

On the frozen 380-point checkpoint and the same ten validation seeds, reducing
sampling temperature from 1 to **0.5** increased mean score from **358 to 384**,
median from **360 to 380**, and best from **380 to 400**. Temperature **0.25**
gave mean/median/best **380**. Every game remained in stage 1. These are reused
validation results, not fresh testing or evidence of mission completion.

The [400-point replay](results/defense/sampling-probes/temperature-050/replay.html)
reproduced all **1,743** neural actions, screens and rewards after reloading the
original weights with temperature 0.5. Its separate bundle includes weights,
configuration, evaluation, trace and checksums. The HTML explicitly labels this
an evaluation-only sampling probe. It does not replace `learned/best`, whose
original temperature-1 policy and replay remain unchanged. Loading these same
weights without the temperature override reproduces the original policy, not
the diagnostic. This result motivates testing lower training entropy if the
current self-imitation run plateaus; it does not yet establish that doing so
will improve learning.

```sh
venv/bin/python -m rl.defense_evaluate \
  results/defense/sampling-probes/temperature-050/model.safetensors \
  --temperature .5 --seed 10000 --games 10 --envs 10 \
  --output runs/defense-temperature-reproduction.json \
  --replay-output runs/defense-temperature-reproduction
```

The optional recorder refuses incomplete suites, changed weights, existing
output directories and temporal overrides; it never promotes a diagnostic
into the live learner's best-artifact directory.

A later temperature-0.5 check on the frozen **500-point** model (counter
4,434,688) used the same ten validation seeds. Mean changed from **462 to 464**,
median stayed **460**, and best changed from **500 to 520**, all stage 1.
The [separate diagnostic replay](results/defense/sampling-probes/step-4434688-temperature-050/replay.html)
reproduced all **1,837** neural actions after reload. This small mean change
does not establish a useful sampling improvement, and the subsequently trained
standard-policy 560-point model is stronger. The diagnostic remains separate
from the shared best; its weights are the unchanged older 500-point model.

On the later frozen **10,280-point** model at counter **7,244,544**, two more
sampling diagnostics used the same ten validation seeds and unchanged action
timing. These are exploratory comparisons, not independent fresh-test results.

| Sampling temperature | Mean | Median | Best | Highest stage |
|---|---:|---:|---:|---:|
| 1 (original evaluation) | 8,834 | 9,115 | 10,280 | 1 |
| 0.5 | 8,951 | 9,125 | 10,260 | 1 |
| 1.5 | 9,107 | 9,815 | 10,280 | 1 |

Neither setting reached stage 2 or completed a mission. The
[temperature-0.5 bundle](results/defense/sampling-probes/step-7244544-temperature-050/replay.html)
and [temperature-1.5 bundle](results/defense/sampling-probes/step-7244544-temperature-150/replay.html)
preserve unchanged weights, complete evaluations, traces and reloaded-policy
verification. These diagnostic bundles are excluded from best-model promotion.
This small comparison does not establish that changing training entropy would
help; these probes did not change the active learners' settings.

An additional [action-timing diagnostic](results/defense/sampling-probes/step-8342272-tstates-50000.json)
used the frozen **10,480-point** model at counter **8,342,272**, ordinary
temperature-1 sampling, and the same ten validation seeds. Reducing each action
from **100,000 to 50,000 T-states** (twice as many policy decisions per emulated
second) gave mean **5,809**, median **5,785**, best **8,740**, all stage 1 and no
mission. The original timing gave mean **9,981**, median **10,380**, best
**10,480**. All diagnostic games completed without an action cap.
This is an explicit evaluation-only timing override, not a newly trained model
or a best-replay promotion. It provides no evidence that changing the frozen
policy's control rate alone helps; it does not rule out learning separately at
the shorter interval. Both active learners retain their original timing.

A read-only action-alias check used the frozen
[10,480-point policy and its own trace](results/defense/learned/versions/step-000008342272-125346536cb1-seed-10004/manifest.json),
querying 109 four-frame observations at action indices **280–388** before its
first ship loss. Mean categorical entropy was **0.21745 nats**; merging the
nine forward-fire aliases (action IDs 9–17, equivalent in stage 1) into one
probability gave **0.19499 nats**. Only **0.02246 nats**, about 10%, came from
variation within those aliases. Mean movement probability was **0.60089**;
forward-fire-only probability was **0.18815**. This narrow pre-collision window
does not support redundant fire choices as the dominant source of apparent
exploration. It is not an all-state or later-stage result. The diagnostic
changed no weights, controls or training inputs, and the 20-action set remains
unchanged.

The later `rl.defense_alias_probe` extends this to whole **verified own-policy
replays**, including recurrent memory carried across ship losses. It reconstructs
boot-padded screen stacks at the saved stride and must reproduce every original
sampled action using the original RNG before reporting statistics. Source hashes
are checked before and after. This is replay analysis, **not a new native game
evaluation**; the original native verification is retained in the report.
Float32 sampling is unchanged; entropy uses float64-renormalized probabilities
to remove small softmax rounding errors (maximum observed sum error below
0.000005 in these traces).

| Selected replay | Raw entropy (nats) | Grouped entropy (nats) | Alias share | Movement probability |
| --- | ---: | ---: | ---: | ---: |
| [Original best, 2,580 actions](results/defense/diagnostics/aliases-global-best-8342272.json) | 0.37958 | 0.25838 | 31.9% | 55.5% |
| [Frozen memory at 335,872, 2,494 actions](results/defense/diagnostics/aliases-frozen-memory-335872.json) | 0.61472 | 0.47975 | 22.0% | 57.5% |

Grouping merges forward-fire IDs 9–17 only; it describes the stage-one command
handler, not necessarily distinct physical outcomes. In the four 128-action
windows before **visible** ship-loss reports, alias entropy shares ranged from
15.3–25.9% for the original best and 22.0–25.4% for the recurrent replay.
The whole trajectories include animations, and visible loss can lag collision.
These two selected games have different seeds and trajectories: this is not a
matched-state causal comparison or a fresh success-rate estimate. Neither
trace shows redundant firing as most of its action entropy. This does not rule
out benefits from a different action set, but does not establish aliasing as
the dominant bottleneck here; controls and training remain unchanged. No
diagnostic frames/actions are used as training data or promotion candidates.

```bash
venv/bin/python -m rl.defense_alias_probe \
  results/defense/training/ppo-30-frozen-memory/first-replay \
  --output runs/defense-alias-reproduction.json
```

### Shared failure location: screen evidence

The user's observation that the policies fail at the same place prompted a
[four-policy, sixteen-life comparison](results/defense/diagnostics/shared-loss-01/report.json).
`rl.defense_loss_probe` reads only preserved, hash-checked, originally verified
own-policy traces. It reconciles each life total with recorded visible score
increments and renders the original screen bytes, without running new games,
reading private RAM, generating demonstrations, or changing training.

| Selected policy replay | Points earned on each of its four lives | Screen comparison |
| --- | --- | --- |
| Original global best | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-1-losses.png) |
| PPO 12 best effort | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-2-losses.png) |
| Frozen-memory PPO 30, 933,888 | 2,620 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-3-losses.png) |
| Independently trained bootstrap DQN 24 | 2,550 / 2,620 / 2,620 / 2,620 | [Four lives](results/defense/diagnostics/shared-loss-01/policy-4-losses.png) |

Visual inspection shows the same broad horizontal barrier approaching the
ship, with a large opening at the right, while these ships remain near the
center/left. Some frames show small breaks in the barrier; this is not proof
that one particular route or action is required. The selected actions differ
across policies and lives, but none of these traces negotiates this obstacle.
This supports a **shared behavioral bottleneck**, not just similar final
scores. It is still a selected-replay observation, not a population failure
rate, measured course index, or proof of the exact collision mechanism.

Rows of each sheet are lives; columns are 64, 32, 8 and 1 decisions before
an alignment marker. For the first three lives, the marker is the first sampled
frame in the last 128 with more than half the non-HUD cells solid white.
These flashes precede the visible life decrement by **11–18 decisions**.
They are **not exact physical collision timestamps**; even a preceding frame
can already be in the death animation. No such flash is sampled on the final
life (terminal settling advances the original animation), so that row explicitly
uses the visible loss endpoint instead. The sheets do not pretend these
different markers are precisely time-aligned collision events.

The records also show large **750–770** and **1,500–1,520** visible-score
increments shortly before this failure. A plausible learning explanation is
that the policies reliably collect the earlier reward but have not discovered
the longer sequence needed to continue past the barrier. This remains a
hypothesis, not a demonstrated cause. Movement choices still account for
34–72% of the selected 64-decision windows; action counts alone do not establish
effective displacement, and "the agent never moves" would be inaccurate.

The training implication is to judge subsequent experiments by escaping this
failure pattern and reaching new stages, not by another tiny increase in mean
score at the same 10,480 ceiling. Earlier lookback, longer-return, action-timing
and parameter-noise experiments are already documented below and did not clear
it; simply repeating them is not a new diagnosis. The live frozen-memory trial
continues as a controlled test, with no hand-coded right-turn rule, screen-derived
collision reward, or replay examples added to learning.

All **269 regression tests** pass. The new read-only checks cover synthetic
loss-window boundaries, absent-flash fallback, score reconciliation, invalid
traces, original-source immutability and checksum rejection, plus exact
feedforward/recurrent action reconstruction in the separate alias diagnostic.
The [full test log](results/defense/diagnostics/shared-loss-regression-tests.txt)
is preserved. No learner or environment behavior changed in this diagnostic.

```bash
venv/bin/python -m rl.defense_loss_probe \
  results/defense/learned/best \
  results/defense/training/ppo-12-lookback/best-effort \
  results/defense/training/ppo-30-frozen-memory/replay-peak-933888 \
  results/defense/training/dqn-24-bootstrap/replay-10410 \
  --output runs/defense-shared-loss-reproduction
```

### Frozen DQN prediction/return diagnostic

To test whether the shared losses accompany grossly inflated future-score
predictions, `rl.defense_q_probe` compares a frozen ordinary DQN's selected
Q-values with discounted returns on **that model's own verified replay**.
It reconstructs the exact boot-padded visible four-frame stacks, checks every
greedy action against the recorded action, and computes returns using the
checkpoint's gamma **0.997**, reward scale **0.01** and actual visible life
learning boundaries. It does not count the next life's rewards across a
learning terminal. Results are reported in discounted **score units** by
undoing the fixed optimizer scale, not as new game scores or optimal values.

Three selected traces reproduce **7,648** actions in total, with every source
file hash unchanged. Their native action/screen/reward verification was already
performed when the replays were published; this diagnostic does **not** run
the emulator again, update parameters, choose new actions or write training data.

| Frozen model / selected replay | Reconstructed actions | Whole-replay mean prediction minus realized discounted return |
| --- | ---: | ---: |
| [DQN 33 reset, 6,962,144](results/defense/diagnostics/q-return-reset-6962144.json) | 2,564 | −130.89 points |
| [DQN 34 no-reset, 6,962,144](results/defense/diagnostics/q-return-control-6962144.json) | 2,558 | −133.46 points |
| [DQN 31 low-rate persistence, 7,031,072](results/defense/diagnostics/q-return-persistent-7031072.json) | 2,526 | −137.93 points |

In their twelve 64-action windows before the visual alignment marker, mean
prediction-minus-return errors range from **−45.71 to +124.00** points. For
the nine lives with sampled white-flash windows, predicted remaining score
averages **7.15–81.56**, while the actual remaining return in those windows is
zero. Predictions generally fall sharply after the large score gains. Thus
these traces do **not** support the specific explanation that the selected
models consistently expect thousands more points while dying. Whole-trace
underprediction is not evidence of a general pessimistic bias: these are
selected high-scoring replays, not unbiased samples of expected return.

Neither small nor large error would establish the correct route or a causal
training defect. Approximation, partial observations, changes in policy and
selected-trajectory effects can all contribute. Flash alignment is not a
physical collision timestamp; final losses without a sampled flash use the
explicit visible-loss endpoint. No counterfactual action sequence is supplied,
and no policy/reward/environment setting changes on the basis of this check.
The matched exploration/reset trials continue unchanged.

Three focused tests pass: analytic discounted-return/life-boundary isolation,
invalid/incomplete input rejection, and exact frozen action reconstruction
with source immutability and checksum rejection. The full **279-test**
[regression suite passes](results/defense/diagnostics/q-probe-regression-tests.txt).
Reproduce into a new output path:

```bash
venv/bin/python -m rl.defense_q_probe \
  results/defense/training/dqn-33-persistent-resets/first-replay \
  --output runs/defense-q-return-reproduction.json
```

### Lower-entropy continuation

`defense-ppo-05-low-entropy` resumes the original 20-action checkpoint at
**1,133,312** actions (the 380-point model, mean 358). Its entropy coefficient
is **0.002**, down from 0.02; other PPO/gameplay settings are unchanged and
self-imitation is disabled. This directly tests learning with less pressure
to keep the action distribution diffuse. It does not scale inference logits:
validation remains at temperature 1. The starting model and optimizer are
[archived together](results/defense/training/ppo-05-low-entropy/start-checkpoint/state.json).

Its first ten-game validation, after 102,400 additional actions, had mean
**330**, median **330**, best **360**, all stage 1. Thus there is **no improvement
claim** yet. The next two rounds averaged 332 and 338, still below the starting
checkpoint. It initially ran alongside the self-imitation experiment, with
separate artifact roots to avoid competing writers; run 04 is now paused.
The experiment's local
verified replay is `runs/defense-ppo-05-low-entropy/artifacts/best/replay.html`;
the older 380-point policy and separate 400-point sampling probe remain archived.

Reproduce from the archived optimizer into a new run directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-low-entropy-reproduction \
  --resume results/defense/training/ppo-05-low-entropy/start-checkpoint \
  --artifacts runs/defense-low-entropy-reproduction/artifacts \
  --entropy .002 --sil-updates 0
```

Later, after **602,112 additional actions**, run 05 reached a ten-game mean
of **370**, median **380**, best **380**, still all stage 1. This is improved
validation consistency relative to the starting mean 358, not a new best
individual score or fresh-test success rate. The
[step-1,735,424 checkpoint](results/defense/training/ppo-05-low-entropy/step-000001735424/state.json)
preserves its weights, optimizer and complete evaluation. The next validation
means were 362 and 326, so this is not a claim of monotonic improvement.

Continued unchanged, run 05 subsequently set verified bests of **420**, **440**,
and **460**. At counter **3,234,560**, ten complete validation games averaged
**416**, median **420**, best **460**, still all stage 1 with no mission.
The collector reloaded the frozen weights and reproduced all **1,675** actions,
screens and rewards before publishing the current best. Its
[resumable checkpoint](results/defense/training/ppo-05-low-entropy/step-000003234560/state.json)
includes the optimizer and evaluation. The next two validation means were 412
and 410. This improvement came from ordinary from-boot training, not the
optional curriculum below; all intermediate replay bundles remain preserved.

Run 05 then regressed for six consecutive validation rounds (means 318, 296,
322, 324, 318, 316). It was stopped cleanly at **4,094,720** inherited actions:
**2,961,408 additional actions**, **1,824 new complete training games**, and
29 validation rounds. Its [complete log](results/defense/training/ppo-05-low-entropy/metrics.jsonl)
and [final resumable checkpoint](results/defense/training/ppo-05-low-entropy/final-checkpoint/state.json)
are preserved. Run 06 continued; the freed slot started the own-experience
curriculum from run 05's strongest archived checkpoint, not its regressed end.

### Ship-loss learning-boundary comparison

`defense-ppo-06-life-boundary` starts from exactly the same checkpoint and
settings as run 05, with only `--life-terminal` enabled. A configuration diff
confirms this is the only difference besides run/artifact directories. The
boundary prevents estimated returns from crossing a visible ship loss;
**reward remains the actual score delta**. The emulator is not reset after a
ship loss: the original game continues through its own introduction, and all
evaluation games still run from boot until all four ships are gone.

Its first ten-game validation, after **102,400 additional actions**, had mean
**368**, median **370**, best **380**, all stage 1, compared with run 05's mean
330 at the same training increment. This is an encouraging early comparison,
not stage completion or a statistically established success-rate improvement.
The second round fell to mean **314**, median **320**, best **340**; the early
gain is therefore not evidence of stable superiority.
Run 06 used its isolated archive throughout training;
run 06's local replay is
`runs/defense-ppo-06-life-boundary/artifacts/best/replay.html`.

Reproduce using the command above with a new run/artifact directory and
`--life-terminal`. Its exact
[configuration](results/defense/training/ppo-06-life-boundary/resume-config.json)
and [first validation](results/defense/training/ppo-06-life-boundary/first-validation.json)
are preserved. The original 380-point best and separate 400-point sampling
replay remain available; no unsuccessful candidate replaces them.

Run 06 subsequently reached **400 points with the ordinary temperature-1
policy**, at counter **2,034,432** (901,120 additional training actions).
Ten complete validation games: mean **360**, median **360**, best **400**,
all stage 1, no mission completion. The collector reloaded the frozen model
and verified all **1,682** actions, screens and rewards, then promoted it to
the [shared best replay](results/defense/learned/best/replay.html) and
[weights](results/defense/learned/best/model.safetensors).
This is a new trained-policy best, distinct from the older temperature-0.5
diagnostic. Neither demonstrates a stage clear. All previous versions remain.

Run 06 later matched the strongest ten-game validation mean, **416**, at
counter **3,435,264** (median 420, best 440), still all stage 1. It continued
unchanged while separate experiments ran alongside it.

At counter **4,434,688**, unchanged run 06 set a new standard-policy best of
**500 points**: ten complete validation games averaged **462**, median **460**,
still all stage 1, with no completed mission. The collector reloaded the frozen
weights and exactly verified all **1,901** actions, rewards and screens before
promoting the [shared replay](results/defense/learned/best/replay.html) and
[weights](results/defense/learned/best/model.safetensors). The
[resumable milestone](results/defense/training/ppo-06-life-boundary/step-000004434688/state.json)
also preserves its optimizer and full evaluation. This is **3,301,376 additional
actions** since its starting checkpoint, not training from scratch at that
counter. The earlier 460-point model and every prior replay remain archived.

Continuing unchanged, run 06 reached **560 points** at counter **4,733,696**
(**3,600,384 additional actions**). Ten complete validation games: mean **506**,
median **510**, best **560**, all stage 1, no mission. The collector verified
every one of the **1,975** neural actions/screens/rewards before promotion.
Its [resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000004733696/state.json)
preserves weights, optimizer and the complete evaluation. The 500-point bundle
and all earlier milestones remain available.

The next round, counter **4,836,096**, produced a **580-point** best effort,
with all **1,904** actions exactly verified by the collector. Its ten-game
mean was **480**, median **460**, all stage 1, with no completed mission.
This improves the best individual replay, not mean performance: the preceding
560-point checkpoint averaged 506 and remains archived separately. The
[580-point resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000004836096/state.json)
includes weights, optimizer and its complete evaluation.

After validation means of 522, 522, 542 and 560, unchanged run 06 reached
**600 points** at counter **5,434,112**. Ten complete validation games averaged
**564**, median **570**, best **600**, all stage 1 with no completed mission.
The collector reloaded the weights and verified all **2,003** actions, screens
and rewards before promotion. Its
[resumable checkpoint](results/defense/training/ppo-06-life-boundary/step-000005434112/state.json)
preserves the optimizer and complete evaluation alongside the shared replay.
This improves both the best effort and reused-seed validation mean, not a
fresh-test success rate or evidence of stage completion.

Run 06 eventually stopped cleanly at counter **7,199,488**, after **6,066,176
additional actions**, **3,423 new complete training games**, and **60 validation
rounds**. No game reached stage 2; its best-so-far effort stayed at 600 for
17 further rounds after first reaching that score. The strongest mean was
**586**; the latest matching
[checkpoint](results/defense/training/ppo-06-life-boundary/step-000007035648/state.json)
is preserved with optimizer and evaluation. Its
[complete log](results/defense/training/ppo-06-life-boundary/metrics.jsonl) and
[final resumable state](results/defense/training/ppo-06-life-boundary/final-checkpoint/state.json)
are archived. Run 08 subsequently reached 620 and later 10,280; its final
results are below. The freed slot tested curriculum resets with the successful
ship-loss learning boundaries.

### Own-experience curriculum: run 07

`rl.defense_snapshot` adapts the existing native snapshot API to Defense's
visible score, ships, stage, screen history and outcome bookkeeping. It saves
and restores an opaque emulator state reached through actual gameplay; it
does not decode internal bytes, edit game state, select actions, or expose
anything beyond the usual screen observation to a policy. It does not rewind
the boot RNG and rejects incompatible timing, action profiles and malformed
snapshots. No emulator rebuild was needed.

Native regression tests captured the original learned 380-point trajectory at
action 80, then reproduced every remaining **1,578** screen/action/reward
transition through game over in both the original process and a newly spawned
process. The native terminal-text stop was also restored correctly. These
recorded actions are test fixtures only, not training examples. The 21-action
profile and invalid-state rejection are tested separately.

`rl.defense_curriculum` optionally resets training workers to these unmodified,
own-reached states. Archive bins use visible stage and score earned since the
current life/stage began, not cumulative points from previous lost ships.
Each stage has a bounded number of bins and reservoir entries. Workers can
share snapshots reached within the same run; protected boot workers always
start normally. There is no archive-file loader or external demonstration input.

Restored segments are marked `full_game=False`, cannot enter best-game ranking,
and report only newly earned score. Training counters distinguish boot games
from restored segments, including in self-imitation provenance. Reward remains
the actual score delta, and evaluation always uses complete ordinary from-boot
games. Archives are not checkpointed: after resume they refill from new play.
The default curriculum probability is zero, leaving normal training unchanged.

An isolated [8,192-action smoke run](results/defense/training/curriculum-smoke-01/metrics.jsonl)
completed three boot games and four restored segments with protected-worker
sharing and self-imitation enabled. Its two-game validation mean was 290;
the selected replay reproduced all 1,548 actions. A further
[4,096-action resume check](results/defense/training/curriculum-smoke-resume-01/metrics.jsonl)
restored optimizer/RNG state, rebuilt archives from new play, and verified a
1,557-action replay. These are integration checks, not performance gains.

`defense-ppo-07-curriculum` resumed run 05's preserved 460-point checkpoint
at counter **3,234,560**. Only the curriculum is enabled: reset probability 0.5,
same-run snapshot sharing, and eight protected boot workers out of 32. All
PPO, reward, action, timing and evaluation settings are inherited unchanged;
self-imitation and ship-loss learning boundaries remain disabled. Its
[configuration](results/defense/training/ppo-07-curriculum/resume-config.json)
is archived. A new action counter segment is additional compute, not a
continuation from run 05's final counter. Reproduce in a new directory:

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-curriculum-trial \
  --resume results/defense/training/ppo-05-low-entropy/step-000003234560 \
  --artifacts runs/defense-curriculum-trial/artifacts \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8
```

The single collector has been restarted with run 07 as an additional isolated
source. Restored training segments cannot replace the best replay: promotion
still requires frozen-policy verification of a complete from-boot game.

After **102,400 additional actions**, run 07's
[first ten-game validation](results/defense/training/ppo-07-curriculum/first-validation.json)
had mean **410**, median **420**, best **440**, all stage 1 and no mission.
This is slightly below the starting checkpoint's mean 416, not improvement
yet. Its isolated replay reproduced all **1,733** neural actions after weight
reload; the then-current 460-point best remained unchanged. Boot-game and restored
segment counters remain separate in the live log.

The second validation mean rose to **420** and the third to **438** (median
**440**, best **460**) after **303,104 additional actions**, at counter
**3,537,664**. The [third checkpoint](results/defense/training/ppo-07-curriculum/step-000003537664/state.json)
preserves model, optimizer and the complete ten-game evaluation. This improves
consistency on reused validation seeds, not fresh-test performance: every game
remained in stage 1, with no mission completed. This did not replace the shared
best single-effort replay because its best score only tied the original
460-point model. Run 06 subsequently set the 600-point best described above.
Both historical runs were subsequently paused as described here.

After **2,027,520 additional actions**, run 07 stopped cleanly at inherited
counter **5,262,080**. It completed **974 new boot games** and **641 restored
practice segments**, with 20 complete validation rounds. Best validation mean
was 438, but the last three rounds each averaged 320; no game reached stage 2.
The early gain did not persist. Its [complete log](results/defense/training/ppo-07-curriculum/metrics.jsonl)
and [final resumable checkpoint](results/defense/training/ppo-07-curriculum/final-checkpoint/state.json)
are preserved, as is the earlier higher-mean checkpoint. The freed slot now
runs the following comparison; no archive states were transferred to it.

### Longer-rollout comparison: run 08

`defense-ppo-08-long-rollout` resumes the preserved 600-point checkpoint at
counter **5,434,112**, changing only **rollout length from 128 to 256**.
The [configuration](results/defense/training/ppo-08-long-rollout/resume-config.json)
inherits the same 32 workers, 512-example optimizer minibatches, four epochs,
learning rate, entropy, discount, ship-loss learning boundaries and score reward.
Curriculum and self-imitation are disabled. Game timing remains 100,000 T-states
per policy action, so this changes training batches, not the game or controls.

Motivation: in the [archived 460-point replay](results/defense/learned/versions/step-000003234560-5d80be3dfa03-seed-10000/replay.html),
three of thirteen non-overlapping full 128-action chunks had no reward; two
consisted entirely of explicitly visible introduction text. No full 256-action
chunk was entirely reward-free. These replay observations are not measurements of
every actual multi-worker training batch, nor proof of a cause. The hypothesis
is that longer batches mix introductions and active play more consistently.
They do not skip screens, inject actions, or add reward. Run 06 initially stayed
as an unchanged comparison; complete from-boot validation decides performance.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-rollout-reproduction \
  --resume results/defense/training/ppo-06-life-boundary/step-000005434112 \
  --artifacts runs/defense-long-rollout-reproduction/artifacts --rollout 256
```

The single collector includes run 08's isolated artifact directory. It still
requires exact frozen-policy replay verification before a stronger complete
effort can replace the shared best.

After **106,496 additional actions**, run 08's
[first validation](results/defense/training/ppo-08-long-rollout/first-validation.json)
averaged **552**, median **580**, best **600**, all stage 1 with no mission.
Its selected complete replay reproduced all **1,982** neural actions after
reloading the model. This is below the starting checkpoint's mean 564, not an
improvement claim. It tied the then-current shared best score and therefore did
not replace the existing 600-point bundle. Run 08 continued unchanged.

After **1,204,224 additional actions**, the longer-rollout trial reached
**620 points** at counter **6,638,336**. Ten complete validation games: mean
**574**, median **580**, best **620**, all stage 1 and no mission. The collector
reloaded the frozen model and exactly reproduced all **2,047** actions, screens
and rewards before promoting it. The
[resumable milestone](results/defense/training/ppo-08-long-rollout/step-000006638336/state.json)
preserves weights, optimizer and the complete evaluation. A preceding round
averaged 584, so the best single replay is not also the highest-mean checkpoint.
The 600-point model and all older versions remain available. This is progress
on reused validation seeds, not yet evidence of a stage clear or mission win.

Run 08 subsequently reached **10,280 points**, with all **2,495** neural
actions verified in its own [best replay](results/defense/training/ppo-08-long-rollout/best-effort/replay.html).
Its strongest ten-game mean was **9,358**, median **9,510**, best **10,280**, at
counter **8,940,288**; that [resumable checkpoint](results/defense/training/ppo-08-long-rollout/step-000008940288/state.json)
is preserved separately. It did not exceed 10,280 over five consecutive
validation rounds, and no training or evaluation game reached stage 2.
The run stopped cleanly at **9,325,312**, after **3,891,200 additional actions**,
**1,890 new complete training games** and **38 validation rounds**. Its
[complete log](results/defense/training/ppo-08-long-rollout/metrics.jsonl) and
[final model and optimizer](results/defense/training/ppo-08-long-rollout/final-checkpoint/state.json)
are archived. Its training slot now runs the longer-credit comparison below;
the independent run 09 was not interrupted.

### Curriculum with ship-loss boundaries: run 09

`defense-ppo-09-curriculum-life` resumes the preserved 620-point model at
counter **6,638,336**. Only own-experience curriculum resets are enabled relative
to that checkpoint: probability 0.5, same-run snapshot sharing, and eight
protected boot workers. All PPO settings, including **256-action rollouts and
ship-loss learning boundaries**, remain unchanged. The
[configuration](results/defense/training/ppo-09-curriculum-life/resume-config.json)
records this comparison. Run 08 initially remained the unchanged baseline;
it was later archived as described above. Run 09 was subsequently archived too,
after the results documented below.

This is not a restart of the earlier failed curriculum's final policy. Run 07
started from a weaker 460-point model, used 128-action rollouts and did not have
ship-loss learning boundaries. Run 09 starts with an empty archive and fills it
only from its own newly reached states; it loads no demonstrations or prior
snapshot archive. The same screen-only input, actual score-delta reward,
restored-segment bookkeeping and complete from-boot evaluation rules apply.
The collector includes its isolated output, but restored practice segments
remain ineligible to replace the best complete-game replay.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-curriculum-life-reproduction \
  --resume results/defense/training/ppo-08-long-rollout/step-000006638336 \
  --artifacts runs/defense-curriculum-life-reproduction/artifacts \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8
```

After **106,496 additional actions**, run 09's
[first validation](results/defense/training/ppo-09-curriculum-life/first-validation.json)
averaged **554**, median **560**, best **580**, all stage 1 and no mission.
Its replay reproduced all **1,927** neural actions after reload. This is below
the starting checkpoint's mean 574, not an improvement. Runtime checks after
34 boot games and 20 restored segments found no protected-worker restores or
segment reward-accounting violations. Both live runs continue unchanged; the
verified 620-point shared best remains preserved.

At counter **6,941,440**, after **303,104 additional actions**, run 09 reached
**2,950 points** in a complete ordinary from-boot validation game. Ten games:
mean **1,055**, median **650**, best **2,950**, all stage 1, no mission. The
collector reloaded the frozen weights and reproduced all **2,008** actions,
screens and rewards before promoting the shared best. Independent trace checks
confirmed an initial visible score of zero with four ships, total reward of
2,950, and only stage-1 introduction text; this is not a restored practice
segment or a claimed stage clear. The
[resumable milestone](results/defense/training/ppo-09-curriculum-life/step-000006941440/state.json)
preserves weights, optimizer and the complete evaluation. All older models
and verified replays remain available; both then-active runs continued unchanged.
The visible reward trace contains thirty 20-point increments and awards of
100, 750 and 1,500 points. The large score gain therefore does not by itself
demonstrate a longer survival time or a later stage.

The next checkpoint, counter **7,039,744** (**401,408 additional actions**),
reached **10,200 points**. Ten complete games averaged **7,019**, median **7,090**,
best **10,200**, all stage 1 with no completed mission. The collector verified
all **2,495** actions, screens and rewards from the reloaded frozen policy;
trace checks again confirmed zero initial score, four ships, and no stage-2
intro or mission message. Its
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007039744/state.json)
and immutable replay bundle are preserved separately from the 2,950-point
milestone. This remains a score improvement, not a verified stage clear.

At counter **7,146,240**, after **507,904 additional actions**, run 09 reached
**10,260 points**, with all **2,562** actions exactly verified from boot.
Ten complete validation games: mean **8,169**, median **8,030**, best **10,260**,
still all stage 1 and no mission. The
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007146240/state.json)
and complete replay bundle are preserved; both then-active experiments continued.

At counter **7,244,544**, after **606,208 additional actions**, run 09 reached
**10,280 points**, with all **2,519** actions exactly verified from boot.
Ten complete validation games: mean **8,834**, median **9,115**, best **10,280**,
still all stage 1 and no mission. The immutable replay bundle preserves its
weights, evaluation, action trace and verification alongside the earlier milestones.

Run 09 also improved consistency at counter **7,342,848**: ten complete games
averaged **9,527**, median **10,240**, best **10,280**. This ties the best
individual score, so it does not replace the best-effort replay. Its separate
[model and optimizer checkpoint](results/defense/training/ppo-09-curriculum-life/step-000007342848/state.json)
is preserved for resuming training. All ten games remained stage 1, without
a successful mission; these reused validation seeds are not a fresh test set.

At counter **8,342,272**, after **1,703,936 additional actions**, run 09 reached
**10,480 points**, with all **2,580** actions exactly verified from boot.
Ten complete validation games: mean **9,981**, median **10,380**, best **10,480**,
still all stage 1 and no mission. Its separate
[resumable checkpoint](results/defense/training/ppo-09-curriculum-life/step-000008342272/state.json)
and immutable replay bundle are preserved. No successful stage clear has
been observed in this run or the archived longer-rollout comparison.

Run 09 eventually stopped cleanly at counter **9,677,568**, after **3,039,232
additional actions**, **1,010 new complete boot games**, **740 restored practice
segments**, and **30 complete validation rounds**. Its best stayed at 10,480
for thirteen further rounds after first reaching that score, with no stage-2
or mission-success event in training or evaluation. The strongest mean was
**10,439**, median **10,460**, best **10,480**, at counter **9,243,392**; that
[model and optimizer](results/defense/training/ppo-09-curriculum-life/step-000009243392/state.json)
are preserved separately from the best-effort model. Its
[complete log](results/defense/training/ppo-09-curriculum-life/metrics.jsonl) and
[final resumable state](results/defense/training/ppo-09-curriculum-life/final-checkpoint/state.json)
are archived. The successful score milestones remain available. The freed
training slot then tested stronger exploration, while run 10 continued unchanged.

### Longer-credit comparison: run 10

`defense-ppo-10-long-credit` resumes the preserved **10,480-point** checkpoint
at counter **8,342,272**. The only changed training parameter is **GAE lambda
0.95 to 0.99**; the [configuration](results/defense/training/ppo-10-long-credit/resume-config.json)
was compared directly with its parent checkpoint. The existing return code
propagates an error backward within a learning segment by `gamma * lambda`
per action. With unchanged gamma 0.997, the new trace decays more slowly.
The hypothesis is that this helps assign delayed score rewards to earlier
choices; it is not a demonstrated improvement yet.

The screen input, 20 actions, timing, score-only reward, 256-action rollouts,
learning rate, entropy, curriculum settings, protected boot workers and
ship-loss boundaries are unchanged. Model, optimizer and policy RNG resume
from the saved checkpoint; the own-experience snapshot archive starts empty
and refills from newly played states. There are no game/episode action caps
or wall-clock limits. Three focused checks passed for analytic trace decay,
episode-boundary isolation and optimizer-resume equivalence before launch.
Run 09 initially continued unchanged as the reference experiment, and was later
archived as described above. Run 10 was later archived too. Experiments write isolated
artifacts, with the single verification collector watching their sources.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-credit-reproduction \
  --resume results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --artifacts runs/defense-long-credit-reproduction/artifacts --gae-lambda .99
```

After **106,496 additional actions**, run 10's
[first complete validation](results/defense/training/ppo-10-long-credit/first-validation.json)
averaged **9,275**, median **10,390**, best **10,460**, all stage 1 and no mission.
Its local best replay exactly reproduced all **2,568** neural actions after
reloading the weights; the [verification record](results/defense/training/ppo-10-long-credit/first-verification.json)
is preserved. This first result is below its starting checkpoint's mean 9,981
and best 10,480, so it does not replace the shared best or establish improvement.
The experiment continued unchanged beyond this initial validation.

Run 10 stopped cleanly at **9,988,864**, after **1,646,592 additional actions**,
**498 new complete boot games**, **375 completed restored segments** and
**16 validation rounds**. It tied 10,480 but never exceeded it or reached stage 2.
Its strongest ten-game mean was **10,473**, median **10,480**, best **10,480**,
at counter **9,046,784**. That
[resumable checkpoint](results/defense/training/ppo-10-long-credit/step-000009046784/state.json),
the [full log](results/defense/training/ppo-10-long-credit/metrics.jsonl),
[final model/optimizer](results/defense/training/ppo-10-long-credit/final-checkpoint/state.json)
and [verified best replay](results/defense/training/ppo-10-long-credit/best-effort/replay.html)
are preserved. The replay reproduces all **2,562** actions. This improved
score consistency, not stage completion. Its freed slot now tests the
earlier-state curriculum described below; run 11 continues unchanged.

### Stronger-exploration comparison: run 11

`defense-ppo-11-exploration` resumes the **same 10,480-point checkpoint** at
counter **8,342,272** as run 10. Its only changed parameter relative to that
parent is **entropy coefficient 0.002 to 0.01**, confirmed by comparing the
[saved configuration](results/defense/training/ppo-11-exploration/resume-config.json).
GAE lambda remains 0.95 here. Run 10 tested longer credit assignment and
run 11 tests stronger exploration, each changing one parameter from their
common preserved starting point. Neither intervention has yet established
that it can clear the first stage.

Motivation: the original curriculum run repeatedly learned high-scoring
stage-1 behavior without surviving to stage 2. Encouraging more varied learned
actions during training is a hypothesis for escaping that behavior, not a
guaranteed improvement. This is different from the earlier evaluation-only
sampling probes: ordinary temperature-1 sampling is retained at evaluation.
All screen-input, score-reward, timing, action, ship-loss, curriculum and
protected-boot settings remain unchanged. The snapshot archive starts empty,
using only this run's newly reached states. There are no wall-clock or action
limits. The collector includes this isolated source; the existing best model
and replay remain available throughout training.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-exploration-reproduction \
  --resume results/defense/training/ppo-09-curriculum-life/step-000008342272 \
  --artifacts runs/defense-exploration-reproduction/artifacts --entropy .01
```

After **106,496 additional actions**, run 11's
[first validation](results/defense/training/ppo-11-exploration/first-validation.json)
averaged **9,585**, median **10,140**, best **10,260**, all stage 1 and no mission.
Its local replay reproduced all **2,500** neural actions after reloading the
frozen model; the [verification record](results/defense/training/ppo-11-exploration/first-verification.json)
is preserved. This is below its starting model's mean 9,981 and best 10,480;
it does not replace the shared best. The experiment continues unchanged.

Run 11 was subsequently stopped gracefully at counter **10,865,408**, after
**2,523,136 additional actions**, 795 new completed boot games, 524 completed
restored segments and 25 ten-game validations. It tied 10,480 but never exceeded
it or reached stage 2. Its strongest validation mean was **10,450** (median
10,460) at counter **9,349,888**; the final validation mean was 9,353.
The [full log](results/defense/training/ppo-11-exploration/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-11-exploration/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-11-exploration/step-000009349888/state.json)
and [verified best replay](results/defense/training/ppo-11-exploration/best-effort/replay.html)
are retained. Increased entropy alone did not resolve this trial's bottleneck.

### Optional earlier-state curriculum

`--curriculum-lookback N` is disabled by default (`N=0`). With the option
enabled, a score-progress event can archive a state actually visited **N
actions earlier**, instead of the state at the reward event. The hypothesis
is that a practice reset with more lead-in may help when a rewarding state is
already close to a collision. It does not identify obstacles, select a route,
or provide actions, demonstrations, fabricated states, or extra reward.

The bounded opaque history contains at most `N+1` own-play snapshots per worker.
It clears at resets, detected ship losses, stage changes and terminal outcomes;
insufficient history is skipped. Newly reached stages are still archived
immediately. Archive buckets and reset score baselines use the **saved state's
own visible score**, not the later triggering score. Logs distinguish capture
and trigger actions/scores. Neither this history nor those labels reach the
policy. Snapshots and history are not loaded from demonstration files or
checkpointed; fresh runs refill them through their own gameplay. Ordinary
full-game evaluation still starts from boot without curriculum resets.

All **184 tests** passed, including exact earlier-state provenance, peer restore
and score accounting, life/stage boundary bookkeeping, protected-worker routing,
bounded history, and a full-game comparison proving that collection alone does
not change screens, rewards or outcomes. Stage-transition bookkeeping is unit
tested; this is not a claim of an actual learned stage-2 reach.

A separate [integration smoke run](results/defense/training/lookback-smoke-01/resume-config.json)
used four workers, a 32-action lookback and **16,384 additional training
actions**, then ten uncapped complete evaluation games. It collected 306 archive
entries, including 23 from an active restored segment; no restored segment had
finished when the bounded training smoke test ended. Full restored-episode
reward accounting is separately covered by the native regression tests.
Evaluation mean **9,854**, median **10,230**, best **10,260**, all stage 1 and
no mission. Its [frozen replay](results/defense/training/lookback-smoke-01/replay/replay.html)
reproduced all **2,433** neural actions. The log, model, optimizer, evaluation
and replay bundle are preserved. This is an integration check, not evidence
of improved performance; the smoke artifact source is excluded from the
production collector.

### Earlier-state curriculum: run 12

`defense-ppo-12-lookback` resumes run 10's strongest-mean checkpoint at
counter **9,046,784** (mean **10,473**, best **10,480**). Its only changed
training parameter is **curriculum lookback 0 to 32 actions**; the
[saved configuration](results/defense/training/ppo-12-lookback/resume-config.json)
was compared with the parent checkpoint, treating the previously absent
lookback option as its zero default. The curriculum source hash records the
new, tested implementation. All other training and game settings are inherited,
including GAE lambda 0.99, entropy 0.002, 32 workers, 256-action rollouts,
eight protected boot workers, score-only reward and ordinary evaluation.

The archive and bounded history start empty and fill only through new own play.
No prior snapshot archive or diagnostic trace is loaded. The run has no action
or wall-clock limit, and its isolated artifact source is included in the single
verification collector. Run 11 remains active as the exploration comparison.
This is an experiment aimed at the unresolved stage-1 bottleneck, not a claim
that the lookback hypothesis has improved performance.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-lookback-reproduction \
  --resume results/defense/training/ppo-10-long-credit/step-000009046784 \
  --artifacts runs/defense-lookback-reproduction/artifacts --curriculum-lookback 32
```

After **106,496 additional actions**, run 12's
[first validation](results/defense/training/ppo-12-lookback/first-validation.json)
averaged **10,431**, median **10,470**, best **10,480**, all stage 1 and no mission.
Its local replay reproduced all **2,544** neural actions; the
[verification record](results/defense/training/ppo-12-lookback/first-verification.json)
is preserved. This ties the shared best and is slightly below the starting
mean of 10,473, so it does not establish improvement or replace the existing
best-effort replay. The experiment continues unchanged.

Run 12 subsequently stopped gracefully at counter **11,094,784**, after
**2,048,000 additional actions**, 640 new completed boot games, 401 completed
restored segments and 20 ten-game validations. It never exceeded 10,480 or
reached stage 2. Its strongest mean was **10,474**, median 10,480, at counter
**10,447,616**; the final validation mean was 10,444. This tiny mean increase
over the parent (10,473) is not evidence of a meaningful gain on reused seeds.
The [full log](results/defense/training/ppo-12-lookback/metrics.jsonl),
[final checkpoint](results/defense/training/ppo-12-lookback/final-checkpoint/state.json),
[strongest-mean optimizer checkpoint](results/defense/training/ppo-12-lookback/step-000010447616/state.json)
and [best local replay](results/defense/training/ppo-12-lookback/best-effort/replay.html)
are preserved. Earlier resets alone did not resolve this trial's bottleneck.

### Optional screen-cell curriculum

`--curriculum-cells screen` changes the training archive's grouping from score
bins to coarse screen fingerprints. This is inspired by the archive-and-return
idea in [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9), not an
implementation of its full algorithm or demonstration-based robustification.
Only states reached by the current learner are retained; no external gameplay,
scripted actions, hidden-state labels or extra rewards are introduced.

The archive key decodes the latest frame's graphics characters, omits the HUD
row, averages the remaining 45×128 binary raster into 9×16 blocks, quantizes
each block to eight levels, then hashes the result. Text is absent from this
graphics-only key. The **policy input remains all four original screen frames**,
including text; this encoding is used only to group training reset states.
The default `--curriculum-cells score` retains the previous behavior.

Screen changes are sampled every `--curriculum-screen-interval` actions
(default 32). Each stage retains at most `--curriculum-bins` distinct cells,
using the smallest deterministic hash priorities as a score-independent,
bounded sample. Per-cell snapshots use reservoir sampling. This fixed
representation and bounded selection are implementation choices, not claims
to reproduce the paper's cell-selection scheme. Ordinary reset selection is
uniform across retained stages, cells and snapshots. A lookback, if enabled,
uses the actual earlier snapshot's screen key and score baseline.

All **190 regression tests** passed, including native snapshot restoration,
reward-free screen-cell collection, unchanged full-game screens/rewards,
peer routing, protected boot workers, bounded order-independent cell retention
and exact graphics encoding. No learned stage-2 or mission completion is
established by these tests.

A [bounded integration run](results/defense/training/screen-cells-smoke-01/resume-config.json)
trained for **16,384 new actions** from run 10's strongest-mean checkpoint.
It used four workers (one protected boot worker), 128 cells per stage, one
snapshot per cell, 32-action sampling and lookback, and reset probability 1
for the other workers. It logged 376 archive events covering 243 distinct
screen keys, including 105 events from restored play; four boot games and one
restored segment completed with exact visible-score reward accounting.
The [subsequent ten complete games](results/defense/training/screen-cells-smoke-01/evaluation.json)
averaged **8,009**, median **7,755**, best **10,280**, all stage 1. All **2,478**
actions of its [replay](results/defense/training/screen-cells-smoke-01/replay/replay.html)
were reproduced after reloading the model. This checks integration, not
improvement: performance is below the parent. Its model, optimizer and log are
preserved, and this smoke/probe source is excluded from the best collector.

### Diverse-screen curriculum: run 13

`defense-ppo-13-screen-cells` starts from the same run-10 checkpoint at counter
**9,046,784** as run 12. Relative to that continuing lookback comparison, it
changes archive grouping to screen cells, capacity from 16 score bins to 128
screen cells per stage, and snapshots per cell from four to one. Screen-key
changes are sampled every 32 actions. These are several archive changes, not
a single-parameter comparison. Both trials retain a 32-action lookback, GAE
lambda 0.99, entropy 0.002, 32 workers, 256-action rollouts, 50% reset probability
for eligible workers, and eight protected boot-only workers. The
[configuration](results/defense/training/ppo-13-screen-cells/resume-config.json)
records the exact settings and new implementation hashes.

The archive starts empty and uses only new own-play states. The smoke model
and archive are not reused. There are no wall-clock, episode-action or total
training-action limits. Ordinary ten-game evaluations start from boot, and
the single verification collector now includes this trial's isolated artifact
source. This run replaces the stopped exploration trial, not the preserved
best model or the continuing run 12.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-screen-cells-reproduction \
  --resume results/defense/training/ppo-10-long-credit/step-000009046784 \
  --artifacts runs/defense-screen-cells-reproduction/artifacts \
  --curriculum-lookback 32 --curriculum-cells screen --curriculum-bins 128 \
  --curriculum-per-bin 1 --curriculum-screen-interval 32
```

After **106,496 additional actions**, run 13's
[first ten complete validation games](results/defense/training/ppo-13-screen-cells/first-validation.json)
averaged **9,362**, median **8,960**, best **10,480**, all stage 1 and no mission.
Its local replay reproduced all **2,505** neural actions; the
[verification record](results/defense/training/ppo-13-screen-cells/first-verification.json)
is retained. This ties the shared best score but is below the parent mean;
it is not evidence of improvement. The full run continues unchanged.

Run 13 subsequently stopped gracefully at counter **10,570,496**, after
**1,523,712 additional actions**, 458 new completed boot games, 302 completed
restored segments and 15 ten-game validations. It tied 10,480 but never
exceeded it or reached stage 2. Its strongest mean was **10,422** (median
10,445) at **9,849,600**, below its parent's mean of 10,473; the final
validation mean was 10,271. The
[full log](results/defense/training/ppo-13-screen-cells/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-13-screen-cells/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-13-screen-cells/step-000009849600/state.json)
and [verified best replay](results/defense/training/ppo-13-screen-cells/best-effort/replay.html)
are preserved. This tested screen-cell configuration did not resolve the
observed bottleneck; the optional implementation remains available.

### Longer-return PPO: run 14

`defense-ppo-14-long-horizon` replaces the stopped run 12, while run 13 continues.
It resumes the preserved run-12 checkpoint at **10,447,616**, changing three
training settings together: rollout **256 → 1,024**, discount factor
**0.997 → 0.999**, and GAE lambda **0.99 → 0.999**. All other learning/game
settings are inherited, including score-bin curriculum, 32-action lookback,
eight boot-only workers, ship-loss learning boundaries, screen-only input and
visible-score-only reward. The newly introduced screen-cell mode stays off.
The [saved configuration](results/defense/training/ppo-14-long-horizon/resume-config.json)
records the settings and backward-compatible curriculum implementation hash.
There are no episode, total-action or wall-clock limits.

The motivation is longer credit assignment, not a proven diagnosis. In the
preserved 10,480-point trace, the four ship segments last 410, 723, 730 and
717 actions (later segments include inter-ship animations), each scoring
2,620. The old untruncated GAE residual kernel `(gamma*lambda)^k` has a
53.1-action half-life; the new one has a 346.4-action half-life. Actual direct
credit is also cut off by rollout ends and life boundaries; learned value
bootstrapping can carry information farther. Longer traces increase variance.
These calculations do not identify a route or show that longer credit will
escape the observed local behavior.

Testing different return horizons is also motivated by
[Agent57's discussion of long-term credit](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/).
This trial remains PPO: it does not implement Agent57, use its intrinsic rewards,
or introduce a meta-controller. All **191 regression tests** passed, including
analytic 1,024-step GAE weights, terminal masking and bootstrap propagation
at the new parameters. The same single collector includes this run and keeps
the previous verified best available until a genuine improvement is verified.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-long-horizon-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-long-horizon-reproduction/artifacts \
  --gamma .999 --gae-lambda .999 --rollout 1024
```

After four long rollouts (**131,072 additional actions**), run 14's
[first ten complete evaluation games](results/defense/training/ppo-14-long-horizon/first-validation.json)
averaged **10,208**, median **10,480**, best **10,480**, all stage 1 and no
mission. Its replay reproduced all **2,580** neural actions; the
[verification record](results/defense/training/ppo-14-long-horizon/first-verification.json)
is preserved. This is below its parent's mean and does not establish a gain.
The original shared best remains unchanged, and both full trials continue.

Run 14 subsequently stopped gracefully at **12,512,000**, after **2,064,384
additional actions**, 600 new completed boot games, 407 completed restored
segments and 20 ten-game validations. It never exceeded 10,480 or reached
stage 2. Its strongest mean was **10,458**, median 10,460, at **10,775,296**;
the final validation mean was 9,170. The
[full log](results/defense/training/ppo-14-long-horizon/metrics.jsonl),
[final optimizer checkpoint](results/defense/training/ppo-14-long-horizon/final-checkpoint/state.json),
[strongest-mean checkpoint](results/defense/training/ppo-14-long-horizon/step-000010775296/state.json)
and [verified best replay](results/defense/training/ppo-14-long-horizon/best-effort/replay.html)
are retained. The tested longer-return configuration did not resolve this
trial's bottleneck; this does not rule out other horizons or longer training.

### Optional persistent policy-bias exploration

`--policy-bias-noise STD` defaults to zero. When enabled, each training worker
draws independent zero-mean Gaussian offsets for the learned actor's output
bias parameters. The offsets persist across actions and rollout boundaries,
and are redrawn on a visible ship loss or episode boundary. They do not choose
an action, identify an obstacle, or encode a route. The learned screen-dependent
logits plus the sampled parameter offsets define the categorical policy.

The method is a restricted adaptation of
[parameter-space exploration](https://arxiv.org/abs/1706.01905): only output
biases are perturbed, with a fixed user-selected scale. It is not the paper's
whole-network, adaptive-scale algorithm. PPO retains each sample's offset and
uses that same offset in the updated log probability, ratio, entropy and KL
calculation. Noise is not optimized as a parameter. The value baseline remains
screen-only and unperturbed, which may increase estimation error/variance.
The separate noise RNG is saved; resume restarts emulator episodes and draws
fresh offsets from that RNG. No additional observation or reward is supplied.
Combining this option with the separate SIL path is rejected as untested.

Evaluation and published replays use the **unperturbed learned model**. Thus a
good noisy training episode cannot replace the standard-policy best on its
own. All **198 tests** passed, covering persistent and selective redraws, RNG
resume, zero-noise equivalence, matching sampling/learning likelihoods, finite
updates and unchanged ordinary sampling. Existing full runs 13 and 14 started
before this implementation and continue without policy-bias noise.

A separate [four-worker integration run](results/defense/training/bias-noise-smoke-01/resume-config.json)
used standard deviation 1 for **16,384 new actions**, starting from run 12's
strongest-mean model. Four boot games and one restored segment completed; the
noise RNG and optimizer are retained in the
[checkpoint](results/defense/training/bias-noise-smoke-01/checkpoint/state.json).
Its [ten unperturbed complete evaluation games](results/defense/training/bias-noise-smoke-01/checkpoint/evaluation.json)
averaged **10,201**, median **10,460**, best **10,480**, all stage 1. All **2,482**
actions of its [replay](results/defense/training/bias-noise-smoke-01/replay/replay.html)
were reproduced. This validates integration, not improved performance; the
smoke artifact source is excluded from the production collector.

### Persistent exploration: run 15

`defense-ppo-15-bias-noise` resumes the same run-12 checkpoint at counter
**10,447,616** used to start run 14, but retains its original 256-action
rollouts, discount 0.997 and GAE lambda 0.99. Its sole changed learning
parameter relative to that parent is **policy-bias noise 0 → 1**. The
[configuration](results/defense/training/ppo-15-bias-noise/resume-config.json)
records this and the tested implementation hashes. The previously absent
screen-cell setting retains its score-bin default; reward, observations,
controls, action timing and optimizer state are unchanged.

The smoke checkpoint is not reused. This is a third independent full learner,
with 32 workers, eight protected boot workers, 50% curriculum resets for
eligible workers, newly collected own-state archives and no action/time caps.
Runs 13 and 14 continue unchanged. Its complete-game evaluation uses the
unperturbed policy, and its isolated artifact source is included in the single
verification collector. Memory pressure and aggregate throughput are monitored
with the extra learner active; noisy training scores are not standard-policy
validation results and cannot directly promote the best replay.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-bias-noise-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-bias-noise-reproduction/artifacts --policy-bias-noise 1
```

After **106,496 additional actions**, run 15's
[first ten complete unperturbed games](results/defense/training/ppo-15-bias-noise/first-validation.json)
averaged **10,440**, median **10,460**, best **10,480**, all stage 1 and no mission.
Its replay reproduced all **2,501** neural actions; the
[verification record](results/defense/training/ppo-15-bias-noise/first-verification.json)
is preserved. This is below its starting mean of 10,474, not a gain. Initial
three-learner monitoring found roughly 1,670 aggregate training actions/second,
stable swap usage on the follow-up check and 37% reported free memory; these
are short observations, not a hardware benchmark. All three trials continue,
with the original verified best unchanged.

### Stronger persistent exploration: run 16

`defense-ppo-16-strong-bias-noise` replaces stopped run 13. It starts from
exactly the same counter-**10,447,616** checkpoint as the continuing run 15,
changing only **policy-bias noise standard deviation 1 → 2** relative to that
trial. The [saved configuration](results/defense/training/ppo-16-strong-bias-noise/resume-config.json)
was compared directly with run 15: apart from paths, that scalar is the sole
difference. It uses the same tested implementation, 32 workers, eight protected
boot workers, score-bin/lookback curriculum, ordinary score reward and uncapped
complete-game evaluation. There are no total-action or wall-clock limits.

The hypothesis is that larger, temporally consistent perturbations will sample
more different behaviors than scale 1. This is not evidence of better play and
may instead disrupt already learned behavior. Only a complete unperturbed
evaluation and verified replay can promote the shared best. Runs 14 and 15
continue unchanged, and the single collector includes run 16 while retaining
the earlier artifact sources.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-strong-bias-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-strong-bias-reproduction/artifacts --policy-bias-noise 2
```

After **106,496 additional actions**, run 16's
[first ten unperturbed complete games](results/defense/training/ppo-16-strong-bias-noise/first-validation.json)
averaged **10,314**, median **10,360**, best **10,480**, all stage 1 and no mission.
Its [verification record](results/defense/training/ppo-16-strong-bias-noise/first-verification.json)
confirms all **2,531** replay actions. This is below the common parent's mean
10,474 and run 15's first mean 10,440; no improvement is established. The
trial continues, and the existing shared best has not been replaced.

Run 16 subsequently stopped gracefully at **12,512,000** actions after
**2,064,384 new actions**, **653** additional complete boot games, **400**
restored training segments and **20** complete ten-game evaluations. No
training or evaluation record reached stage 2 or completed a mission. Best
remained **10,480**; peak mean was **10,455**, median **10,460**, at counter
**10,955,520**. The last evaluation at **12,454,656** averaged **10,434**,
median **10,440**, best **10,480**. The stronger fixed bias perturbation did
not escape the observed plateau in this trial. Its
[full log](results/defense/training/ppo-16-strong-bias-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-16-strong-bias-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-16-strong-bias-noise/step-000010955520/evaluation.json)
and [2,531-action verified replay](results/defense/training/ppo-16-strong-bias-noise/best-effort/replay.html)
are preserved. The fresh-seed and smaller output-weight-noise learners remain
active; the collector retains the stopped trial's immutable source.

### Independent initialization: run 17

`defense-ppo-17-fresh-seed` replaces stopped run 14. The checkpoint ancestry of
the continuing noise trials runs through 12 → 10 → 9 → 8 → 6 → 2 → 1, ending
at the original seed-41 initialization. Run 17 instead starts at **counter 0**
with **seed 73**, random network weights, a fresh optimizer and empty
own-experience archives. It loads no previous model or gameplay data. This
tests a new learning trajectory, not a continuation or a claim of immediate
improvement over the already trained models.

It uses the established 32-worker, 256-action-rollout settings: entropy 0.002,
discount 0.997, GAE lambda 0.99, ship-loss learning boundaries, score-bin
curriculum with 32-action lookback, 50% resets for eligible workers and eight
protected boot-only workers. Policy-bias noise and SIL are disabled. Input,
reward and the original game remain unchanged; complete evaluation uses the
same ten validation seeds. The [configuration](results/defense/training/ppo-17-fresh-seed/config.json)
records all settings. There are no action or wall-clock limits, and this
fresh model needs time to learn the early game again. The two established
noise trials continue alongside it; the collector includes all three and
preserves the existing best until a better verified result exists.

```sh
venv/bin/python -u -m rl.defense_train --run runs/defense-fresh-seed-reproduction \
  --artifacts runs/defense-fresh-seed-reproduction/artifacts --seed 73 \
  --rollout 256 --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8 \
  --curriculum-lookback 32
```

At **106,496 training actions**, run 17's
[first ten complete games](results/defense/training/ppo-17-fresh-seed/first-validation.json)
averaged **308**, median **300**, best **320**, all stage 1 and no mission.
Its [first replay](results/defense/training/ppo-17-fresh-seed/first-replay/replay.html)
reproduced all **1,604** neural actions; the
[optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/step-000000106496/state.json)
and full replay/model bundle are preserved as this independent run's baseline.
This is early learning from random initialization, far below the established
best, not a replacement for that model or evidence of stage progress.

At **1,900,544** actions, the fresh-seed learner's ten complete games averaged
**434**, median **440**, best **460**, still stage 1. That
[optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/step-000001900544/evaluation.json)
is preserved as an intermediate improvement over its 308-point initial
validation mean. Its first 460-point effort occurred at **1,400,832**; that
[checkpoint](results/defense/training/ppo-17-fresh-seed/step-000001400832/evaluation.json)
and its [verified replay](results/defense/training/ppo-17-fresh-seed/replay-460/replay.html)
are also preserved. These are two different models: the replay belongs to
the earlier first-best checkpoint, not the later higher-mean one. Neither
approaches the established 10,480-point global best. This independent learner
continues from its own experience, with no prior model or demonstration data.

### Optional state-dependent output-weight exploration

Run 15 (bias-noise SD 1) stopped gracefully at **12,708,608** actions after
**2,260,992 new actions**, **696** additional complete boot games, **497**
restored training segments and **22** complete ten-game evaluations. Neither
training nor evaluation reached stage 2. Best remained **10,480**; peak mean
was **10,470**, median **10,480**, at counter **11,455,232**. Its last evaluation
at **12,651,264** averaged **9,829**, median **10,470**, best **10,480**.
The [full log](results/defense/training/ppo-15-bias-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-15-bias-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-15-bias-noise/step-000011455232/evaluation.json)
and [2,501-action verified best replay](results/defense/training/ppo-15-bias-noise/best-effort/replay.html)
are preserved. This is a plateau in the tested configuration, not evidence
that every parameter-noise method fails. The stronger-bias and independent
fresh-seed learners continue unchanged.

`--policy-weight-noise STD` is a training-only alternative to bias noise; both
default to zero and cannot be combined. Each worker draws a Gaussian matrix
with the same shape as the learned actor's output weights, and retains it
until a visible life/episode boundary. For screen features `h`, the policy
logits are `(W + delta_W) h + b`. Unlike a constant bias offset, the effect
therefore depends on the screen. The value head is unperturbed. No action is
selected by a script, and no game data other than the screen is introduced.

This extends the restricted
[parameter-noise adaptation](https://arxiv.org/abs/1706.01905) to the actor's
output weight matrix, not the whole network. Its scale is fixed, not adaptive;
it is not a full reproduction of that paper or NoisyNet. Evaluation, model
exports and the shared-best collector still use the unperturbed learned
network. The stored weight format and default prediction path are unchanged.

PPO stores each actual life draw once in a rollout bank, with time-major
worker indices for sampling and shuffling; it does not duplicate the full
matrix for every observation. The original draw is reused when computing
the new policy likelihood. Gradients pass through the screen encoder exactly
as if the actor weights had been perturbed directly; noise itself is constant
for differentiation. The independent noise RNG is checkpointed, with fresh
life draws on resume. SIL combinations are rejected as untested.

All **201 regression tests** passed, including exact perturbed-weight
likelihood/gradient comparisons for every model parameter, finite updates,
rollout-bank ordering and snapshot isolation, zero-noise equivalence, and
invalid-mode checks. The stricter gradient-tree comparison was also rerun
separately. Existing production learners started before this change and
continue using their original settings.

Two isolated four-worker checks resumed run 12's 10,447,616-action checkpoint
and each trained for 16,384 additional actions. They used rollout 256, batch
256, one protected boot-only worker and otherwise inherited the parent
settings. Their unperturbed evaluations each contain ten complete games:

| Output-weight noise SD | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0 (matched control)](results/defense/training/weight-noise-control-01/checkpoint/evaluation.json) | 9,929 | 10,460 | 10,480 | 2,554 |
| [0.02](results/defense/training/weight-noise-smoke-01/checkpoint/evaluation.json) | 324 | 320 | 380 | 1,703 |
| [0.005](results/defense/training/weight-noise-smoke-02/checkpoint/evaluation.json) | 7,668 | 7,920 | 10,460 | 2,523 |

Both remained in stage 1 with no successful mission. These are regressions
from the parent's 10,474 validation mean, not successful exploration results.
The smaller noise did less harm in this small test, but it did not improve
the model. Full logs, configurations, optimizer checkpoints and verified
replays are preserved, including the failed larger-noise trial. Neither is
a collector source or a replacement for the global best. The matched no-noise
control also regressed, but retained much more performance than either noisy
trial. It used the same parent, seed, action count and training settings with
both noise options disabled. Its full checkpoint and verified replay are also
preserved. These single-seed checks do not establish general causality or
predict the outcome with the normal 32-worker training setup.

A subsequent [32-worker calibration](results/defense/training/weight-noise-calibration-01/resume-config.json)
used noise SD **0.005**, the same parent, and the parent's normal rollout 256,
batch 512, eight boot-only workers and all other learning settings. After
**32,768 new actions** (four rollouts), its ten complete unperturbed games
averaged **10,453**, median **10,460**, best **10,480**, still stage 1.
Its [replay](results/defense/training/weight-noise-calibration-01/replay/replay.html)
reproduced **2,553** actions. This retained the parent's performance much
better than the four-worker check; it is not an improvement or evidence that
noise helps exploration. The complete calibration log, optimizer and replay
bundle are preserved separately and excluded from automatic promotion.

`defense-ppo-18-weight-noise` continues from that calibration's optimizer at
counter **10,480,384**, with no training or episode action limit and complete
ten-game evaluations every 100,000 training actions (rounded to rollout
boundaries). Its [configuration](results/defense/training/ppo-18-weight-noise/resume-config.json)
retains noise SD 0.005 and the normal 32-worker settings. Resume restores the
policy, optimizer and RNGs but restarts emulator episodes and its own-state
archive; it draws fresh life perturbations. Evaluation remains unperturbed.
The sole collector includes this run alongside all previous sources, and
only verified strictly better efforts can replace the shared best.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-weight-noise-reproduction \
  --resume results/defense/training/weight-noise-calibration-01/checkpoint \
  --artifacts runs/defense-weight-noise-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Run 18's [first complete evaluation](results/defense/training/ppo-18-weight-noise/first-validation.json),
at **10,586,880** actions (**106,496** additional actions after calibration),
averaged **10,193**, median **10,460**, best **10,480**. All ten games ended
in stage 1 without a mission. Its [verification record](results/defense/training/ppo-18-weight-noise/first-verification.json)
confirms **2,502** reproduced actions; the source checkpoint and replay hashes
match. This is an early continuation result, not a new best. The full source
checkpoint and replay remain in the local run directory; the calibration's
complete bundle is committed above. The existing shared best remains unchanged.

Run 18 subsequently stopped gracefully at **12,233,472** actions after
**1,753,088** additional actions beyond calibration, **535** complete boot
games, **342** restored training segments and **17** ten-game evaluations.
No training or evaluation record reached stage 2 or completed a mission.
Best remained **10,480**; peak mean was **10,460**, median **10,460**, at
**12,086,016**. Its last evaluation at **12,184,320** averaged **8,520**,
median **9,145**, best **10,460**. The observed depth plateau, rather than a
wall-clock limit or a claim that this method can never work, prompted a new
experiment in its slot. The [full log](results/defense/training/ppo-18-weight-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-18-weight-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-18-weight-noise/step-000012086016/evaluation.json)
and [2,502-action verified best replay](results/defense/training/ppo-18-weight-noise/best-effort/replay.html)
are preserved. The larger-noise trial and fully fresh learner continue.

The larger SD **0.02** was also checked for four rollouts with the same normal
32-worker setup and the same run-12 parent, rather than inferring its behavior
solely from the failed four-worker check. Its
[ten complete unperturbed games](results/defense/training/weight-noise-calibration-02/checkpoint/evaluation.json)
averaged **7,783**, median **8,180**, best **10,480**, all stage 1. The
[replay](results/defense/training/weight-noise-calibration-02/replay/replay.html)
verified **2,520** actions. Full logs, configuration, optimizer and replay are
preserved, and this calibration is excluded from the collector. It was still
substantially more disruptive than SD 0.005 in this comparison, so no long
SD-0.02 run was launched. These are short, single-seed validation comparisons,
not independent success-rate estimates or proof about eventual learning.

The midpoint SD **0.01** passed the same four-rollout check: its
[ten complete games](results/defense/training/weight-noise-calibration-03/checkpoint/evaluation.json)
averaged **10,462**, median **10,480**, best **10,480**, all stage 1. Its
[replay](results/defense/training/weight-noise-calibration-03/replay/replay.html)
verified **2,524** actions. The full checkpoint, configuration, log and replay
are preserved. This retained performance in the short calibration, not a
stage advance or a demonstrated exploration benefit.

`defense-ppo-19-moderate-weight-noise` now continues from that calibration's
optimizer at **10,480,384**, with the same unlimited 32-worker settings and
100,000-action evaluation interval as run 18, but noise SD **0.01** instead of
0.005. Each trial has its own calibrated parent and RNG history. The
[configuration](results/defense/training/ppo-19-moderate-weight-noise/resume-config.json)
records this provenance. The collector includes both trials and the fresh
seed learner, with the same strict replay-verification gates. No larger-noise
regression was substituted for the shared best.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-moderate-weight-reproduction \
  --resume results/defense/training/weight-noise-calibration-03/checkpoint \
  --artifacts runs/defense-moderate-weight-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Run 19's [first ten complete games](results/defense/training/ppo-19-moderate-weight-noise/first-validation.json),
after **106,496** additional actions, averaged **9,728**, median **10,370**,
best **10,480**, all stage 1 without a mission. Its
[verification record](results/defense/training/ppo-19-moderate-weight-noise/first-verification.json)
confirms **2,580** reproduced actions. The local first-evaluation checkpoint
and immutable replay bundle have matching weights and validated manifests;
the full calibration bundle is committed above. This early result is below
its starting mean, not an improvement; training continues without replacing
the global best on a tied score.

### Read-only policy/value-gradient diagnostic

A [frozen-model diagnostic](results/defense/diagnostics/gradient-probe-8342272.json)
used the global-best model's own 2,580-action replay to compare its policy plus
entropy gradient with its weighted value-loss gradient. It reconstructed only
screen stacks, visible-score rewards and visible life boundaries; no optimizer
update occurred, and every model parameter was checked unchanged. In ten full
256-action trajectory blocks, the shared-encoder value/policy norm ratio was
**0.46–5.34**, median **1.45**. This does not show overwhelming value-gradient
dominance throughout that selected trajectory, and is not a diagnosis of the
plateau. It is not representative live training data: per-block normalization,
one selected game and the older checkpoint's GAE lambda 0.95 differ from the
continuing learners. The final 20-action partial block is recorded separately.
Reward scaling, optimizer and architecture remain unchanged on this evidence.

### Optional fresh decision heads with the learner's own screen encoder

`--initialize-encoder CHECKPOINT_DIRECTORY` starts a **new** Defense learner
using only the convolutional layers and shared 256-unit screen-feature layer
from an earlier own-trained Defense checkpoint. Both actor and value heads
remain freshly randomized from the new seed; the usual initial actor scale
of 0.1 still applies. The optimizer, action/episode counters, policy RNG,
emulator episodes and own-state archives are fresh. No recorded actions,
trajectories or opaque native snapshots are loaded. All encoder parameters
remain trainable, and observations/rewards/action selection are unchanged.

This is a proposed way to test a different decision policy without discarding
all learned visual features. It is inspired by partial-reset work such as
[Nikishin et al. (2022)](https://proceedings.mlr.press/v162/nikishin22a.html),
but **not a reproduction**: that paper retains replay data, while this PPO
experiment retains only learned encoder weights and starts new on-policy
experience. A fresh action counter reports additional training, not the
total cost including pretraining. Source checkpoint/state hashes and the
source's training-action count are explicitly recorded in configuration.
The checkpoint must match Defense's game, environment, action profile,
parameter names, shapes and dtypes; non-finite encoder weights and a source
that changes during loading are rejected before copying anything.

Initialization and optimizer resume are mutually exclusive. A later `--resume`
continues the new learner normally and preserves its ancestry without
reapplying initialization or requiring the old source path to remain present.
Without the new option, fresh-training and resume behavior are unchanged.
Initialization is applied only when requested for a new learner; it does not
alter already-running learners.

Before considering neuron recycling, a separate
[read-only activity probe](results/defense/diagnostics/activation-probe.json)
measured every policy-input screen from four frozen models' own replays.
Only **4–6 of 256** hidden units were never active in each sample; none of the
convolutional channels was entirely inactive. This does not establish a
widespread dead-neuron failure or rule out other representation problems.
The statistic is inspired by
[Sokar et al. (2023)](https://proceedings.mlr.press/v202/sokar23a.html); **ReDo
has not been implemented or used for training**. Selected replay coverage is
not a representative sample of every state the model could encounter.

All **206 regression tests** passed, including exact encoder-copy/head-
preservation checks, unchanged optimizer/source files, malformed or changing
source rejection, and a real-emulator fresh-start/resume test. An isolated
[four-worker integration run](results/defense/training/encoder-transfer-smoke-01/config.json)
used seed 73 and run 12's encoder at source counter 10,447,616, with fresh
heads/optimizer and its own counter starting at zero. After **16,384 new
actions**, its [ten complete games](results/defense/training/encoder-transfer-smoke-01/checkpoint/evaluation.json)
averaged **282**, median **280**, best **300**, all stage 1 with no mission.
Its [replay](results/defense/training/encoder-transfer-smoke-01/replay/replay.html)
reproduced **1,536** neural actions. Full logs, configuration, checkpoint and
replay are preserved. This verifies integration, not improved learning or
preservation of the old policy's performance. The run is excluded from the
collector. It preceded the full-size trial described below.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-encoder-transfer-check \
  --initialize-encoder results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-encoder-transfer-check/artifacts --seed 73 \
  --envs 4 --rollout 256 --batch-size 256 --entropy .002 --gae-lambda .99 \
  --life-terminal --curriculum-probability .5 --curriculum-share \
  --curriculum-boot-envs 1 --curriculum-lookback 32 --mlx-cache-mb 256 \
  --eval-envs 4 --eval-every 16384 --steps 16384
```

`defense-ppo-20-encoder-transfer` replaces stopped run 18. It initializes
directly from run 12's encoder at source counter **10,447,616**, not from the
four-worker smoke model. Its own counter starts at **zero**. The
[configuration](results/defense/training/ppo-20-encoder-transfer/config.json)
uses the same seed **73**, fresh-head initialization, 32 workers, rollout 256,
batch 512, entropy 0.002, discount 0.997, GAE lambda 0.99, life boundaries and
score-bin/lookback curriculum settings as run 17. Neither uses policy noise
or SIL. Apart from paths and initialization provenance, the only new config
field difference is an explicit zero for weight noise, which was absent
(and disabled) in run 17's earlier code. All copied features remain trainable.

This is a comparison of fresh versus own-pretrained visual features, not
training from scratch at the same total interaction cost: the source encoder
already embodies earlier learning. No output head, optimizer history,
trajectory or native state is transferred. Training and complete-game
evaluation are uncapped. The collector includes the new trial and retains
all previous artifact sources; the global best has not been replaced.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-encoder-transfer-reproduction \
  --initialize-encoder results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-encoder-transfer-reproduction/artifacts --seed 73 \
  --rollout 256 --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 8 \
  --curriculum-lookback 32
```

Run 20's [first ten complete games](results/defense/training/ppo-20-encoder-transfer/step-000000106496/evaluation.json)
at **106,496 additional actions** averaged **302**, median **300**, best **320**,
all stage 1. Its [replay](results/defense/training/ppo-20-encoder-transfer/first-replay/replay.html)
verified **1,603** neural actions, and the full optimizer/model bundle is
preserved. Run 17's corresponding first mean was 308 with the same 320 best;
this early single-seed comparison establishes **no transfer advantage**.
The decision heads are learning anew, and the source encoder's 10,447,616
earlier actions must not be omitted when discussing training cost. Both
fresh-policy trials continue without replacing the stronger global model.

Run 17 subsequently reached **600** at its own counter **3,604,480**:
[ten-game mean 570, median 580](results/defense/training/ppo-17-fresh-seed/step-000003604480/evaluation.json),
all stage 1. The full optimizer/model and a
[2,004-action verified replay](results/defense/training/ppo-17-fresh-seed/replay-600/replay.html)
are preserved alongside the earlier 460-point checkpoint. Run 20 reached
**460** at its own counter **704,512**, with
[mean 380, median 360](results/defense/training/ppo-20-encoder-transfer/step-000000704512/evaluation.json)
and a [1,673-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-460/replay.html).
These are different training ages, not a matched performance comparison.
Both are intermediate improvements within their own trials, below the shared
10,480-point policy; neither has reached stage 2 or a successful mission.

At its own counter **1,400,832**, run 20 improved to
[mean 444, median 440, best 480](results/defense/training/ppo-20-encoder-transfer/step-000001400832/evaluation.json).
Its full optimizer/model and [1,734-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-480/replay.html)
are preserved. Run 17 at the same fresh-action counter had mean **404**,
median **400**, best **460** on these validation seeds. This one-seed,
reused-validation comparison is a small local advantage for encoder transfer,
not evidence of better final performance or lower total training cost: run 20
also used an encoder pretrained for 10,447,616 actions. Both remain in stage 1.

Later saved milestones are run 17's **1,370**-point effort at **4,603,904**
actions ([mean 615, median 520](results/defense/training/ppo-17-fresh-seed/step-000004603904/evaluation.json),
[2,002 verified replay actions](results/defense/training/ppo-17-fresh-seed/replay-1370/replay.html))
and run 20's **580** at **1,802,240** new actions
([mean 540, median 550](results/defense/training/ppo-20-encoder-transfer/step-000001802240/evaluation.json),
[1,958 verified replay actions](results/defense/training/ppo-20-encoder-transfer/replay-580/replay.html)).
Full model/optimizer bundles accompany both. The isolated 1,370 effort did not
persist in subsequent run-17 batches, whose best scores returned to around 600;
it is a capability observation, not reliable performance. Neither trial has
advanced beyond stage 1, and both continue without changing the shared best.

Run 20 reached **600** at **2,203,648** new actions:
[ten-game mean 552, median 580](results/defense/training/ppo-20-encoder-transfer/step-000002203648/evaluation.json).
The full optimizer/model and [1,985-action verified replay](results/defense/training/ppo-20-encoder-transfer/replay-600/replay.html)
are preserved. All ten games remained in stage 1 without a mission.

Run 17's next rare breakthrough was **2,890** at **6,504,448** actions:
[ten complete games](results/defense/training/ppo-17-fresh-seed/step-000006504448/evaluation.json)
averaged **791**, median **560**, all stage 1. Its
[verified replay](results/defense/training/ppo-17-fresh-seed/replay-2890/replay.html)
reproduced **2,102** actions. The next batch at 6,602,752 returned to mean
**568**, median **570**, best **580**: this is not yet stable higher-scoring play.
Run 20 reached **640** at **3,301,376** new actions, with
[mean 582, median 580](results/defense/training/ppo-20-encoder-transfer/step-000003301376/evaluation.json)
and [2,081 verified actions](results/defense/training/ppo-20-encoder-transfer/replay-640/replay.html).
Both full model/optimizer checkpoints are preserved. Neither reached stage 2
or replaced the stronger shared best; the counters differ and run 20's encoder
also includes the previously documented pretraining cost.

Run 20 subsequently stopped normally at **3,817,472** new actions after
**1,656** complete boot games, **1,058** restored segments and **38** ten-game
validations. Its last mean/median/best were **580 / 580 / 600** at 3,801,088;
peak mean was **592**, median **600**, best **600** at 2,506,752. The
[final optimizer](results/defense/training/ppo-20-encoder-transfer/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-20-encoder-transfer/step-000002506752/evaluation.json)
and [full log](results/defense/training/ppo-20-encoder-transfer/metrics.jsonl)
are preserved alongside the earlier 640-point replay. No real training or
evaluation game reached stage 2 or a mission. The repeated score/depth plateau,
not a wall-clock deadline, motivated assigning its slot to bootstrap DQN.
The fresh-seed PPO lineage continued independently until the later plateau
described below.

Run 17 eventually stopped cleanly at **9,289,728** actions, with **3,969**
complete boot games, **2,649** restored segments and **92** complete ten-game
validations. There were **27** evaluations after its 6,504,448-action best;
none exceeded that checkpoint's mean or best score. The last ten means ranged
from **548 to 588**. Its final evaluation at **9,207,808** averaged **580**,
median **590**, best **600**. Neither training nor evaluation reached stage 2
or a mission. The [final optimizer checkpoint](results/defense/training/ppo-17-fresh-seed/final-checkpoint/state.json)
and [losslessly compressed full log](results/defense/training/ppo-17-fresh-seed/metrics.jsonl.gz)
are preserved; decompression was byte-checked against the unchanged local log.
The earlier full 2,890-point checkpoint and verified replay remain available.
This score/depth plateau, not elapsed time, prompted releasing its compute
for the continuing timing and value-learning experiments.

A [screen-encoding audit](results/defense/diagnostics/screen-encoding-audit.json)
also checked all **2,581** frames of the global-best replay. Its **112** distinct
character codes all fall within the encoder's graphics or ASCII ranges.
All **64** semigraphics glyphs matched the actual emulator font at their six
block interiors when compared with the model's rendering table. No dropped
character or graphics-bit-layout mismatch was found. This covers the recorded
stage-1 game, not unseen stages or the quality of learned visual features;
the input pipeline was left unchanged.

### Broader frozen-policy validation

The unchanged global-best model at counter **8,342,272** was evaluated on
[100 additional complete games](results/defense/validation/step-8342272-seeds-30000-30099.json),
seeds **30000–30099**, with eight workers, temperature 1, original 100,000-
T-state action duration and no episode action cap. Mean score was **9,482.3**,
median **10,280**, best **10,480**. **Zero** games reached stage 2, stage 3 or a
successful mission; all 100 ended normally at zero ships. The model's hash
was unchanged before/after evaluation. This broader batch did not reveal a
rare stage clear missed by the routine ten-game checks. It is additional
**validation**, not a held-out final test or proof that a clear is impossible.
The report remains evaluation-only and cannot automatically promote a replay.
No new best was found, and the existing verified best bundle is unchanged.

```bash
venv/bin/python -u -m rl.defense_evaluate \
  results/defense/learned/versions/step-000008342272-125346536cb1-seed-10004/model.safetensors \
  --output runs/defense-expanded-validation-reproduction.json \
  --games 100 --seed 30000 --envs 8 --max-steps 0
```

The longer-lookback run 21 later reached a ten-game mean **10,478**, median
**10,480**, best **10,480**, at counter **11,750,144**. Its
[full frozen checkpoint](results/defense/training/ppo-21-long-lookback/step-000011750144/evaluation.json)
was preserved and evaluated on the **same 100 broader validation seeds** above.
That [comparison](results/defense/validation/step-11750144-seeds-30000-30099.json)
gave mean **10,384.2**, median **10,480**, best **10,480**, with **63/100** games
matching that best and minimum score **7,940**. Relative to the older frozen
model on matching seeds, scores were higher in 96 games, equal in three and
lower in one; the mean difference was **901.9 points**. This is improved score
consistency on reused validation seeds, not a fresh final-test estimate or
proof of better stage-clearing ability.

All **100** games still ended at zero ships in **stage 1**, with no stage 2/3
or successful mission. Weights were hash-checked unchanged. A separate
[2,581-action verified replay](results/defense/validation/step-11750144-seeds-30000-30099-replay/replay.html)
preserves this exact model's best game; its bundle and report remain explicitly
evaluation-only and ineligible for automatic promotion. The shared best-effort
link remains unchanged on a tied score. The stronger average has not solved
the progression plateau.

### Longer own-state lead-in experiment

A [read-only replay timing check](results/defense/diagnostics/lookback-lead-in.json)
compared eligible lookback candidates within each life of the existing best
replay. The last eligible 32-action candidates were **51–63 actions** before
visible ship loss; 128-action candidates would be **147–159 actions** earlier.
These are candidate indices, not live archive selections or exact collision
times. The diagnostic created/restored no native states, and neither its
indices nor replay actions are supplied to training. The hypothesis is simply
that earlier own-reached resets leave more opportunity to change behavior.

The existing lookback option required no production-code change. A new
real-emulator test checks exact opaque snapshot/screen equality at lag 128,
bounded history, saved-state score metadata, and clearing at every ship loss
and reset. All **207 regression tests** passed. An isolated
[four-worker check](results/defense/training/lookback128-smoke-01/resume-config.json)
resumed run 12 at **10,447,616** and added **16,384** actions. Apart from paths,
its configuration differs from the earlier no-noise four-worker control only
in lookback **32 → 128**. Its 307 lagged archive events had exact 128-action
offsets; four boot games and one restored segment completed during training.

Its [ten complete validation games](results/defense/training/lookback128-smoke-01/checkpoint/evaluation.json)
averaged **9,304**, median **9,590**, best **10,480**, all stage 1, compared with
the control's mean **9,929**, median **10,460**, best **10,480**. This is not an
improvement. The [verified replay](results/defense/training/lookback128-smoke-01/replay/replay.html)
reproduced **2,578** neural actions. Full logs, optimizer and replay are
preserved; this short check is excluded from the global collector.

Run 19's moderate weight-noise trial stopped cleanly at **12,602,112** after
**2,121,728** additional actions, **636** boot games, **379** restored segments
and **21** ten-game evaluations. No observed stage advance or mission occurred.
Peak mean was **10,456**, median **10,480**, at **12,184,320**; the final
evaluation averaged **9,549**, median **10,035**, best **10,480**. Its
[full log](results/defense/training/ppo-19-moderate-weight-noise/metrics.jsonl),
[final optimizer](results/defense/training/ppo-19-moderate-weight-noise/final-checkpoint/state.json),
[peak-mean checkpoint](results/defense/training/ppo-19-moderate-weight-noise/step-000012184320/evaluation.json)
and [verified best replay](results/defense/training/ppo-19-moderate-weight-noise/best-effort/replay.html)
are preserved. Its observed depth plateau, not a wall-clock deadline, motivated
reassigning its slot.

`defense-ppo-21-long-lookback` now resumes directly from the same strong run-12
parent, **not** from the four-worker check. It changes lookback 32 to 128 with
32 workers, rollout 256, batch 512, discount 0.997, GAE lambda 0.99, entropy
0.002, life boundaries, shared score-bin curriculum and eight boot-only
workers. Noise and SIL remain off; training and evaluation are uncapped.
The [configuration](results/defense/training/ppo-21-long-lookback/resume-config.json)
records the current source hash and explicit defaults added since run 12.
This full-size trial tests the hypothesis despite the lower short-check mean;
no gain is claimed. Fresh-seed run 17 and own-encoder/fresh-head run 20 continue.
The sole collector retains all earlier sources and includes run 21, with the
same complete-game and frozen-policy verification gates.

Run 21's [first ten complete games](results/defense/training/ppo-21-long-lookback/step-000010554112/evaluation.json)
at **10,554,112** (**106,496** new actions) averaged **9,672**, median **10,385**,
best **10,480**, all stage 1 without a mission. Its
[first replay](results/defense/training/ppo-21-long-lookback/first-replay/replay.html)
verified **2,520** actions; the full model/optimizer checkpoint is preserved.
This is below its parent's mean and does not replace the shared best.

Run 21 eventually stopped cleanly at **12,380,928**, after **1,933,312** new
actions, **582** boot games, **394** completed restored segments and **19**
ten-game validations. Peak mean was **10,478** at **11,750,144** (the frozen
100-game comparison is above); the last evaluation averaged **10,210**,
median **10,460**, best **10,480**. No training or evaluation record reached
stage 2 or a mission. Its [full log](results/defense/training/ppo-21-long-lookback/metrics.jsonl)
and [final optimizer](results/defense/training/ppo-21-long-lookback/final-checkpoint/state.json)
are preserved in addition to the earlier checkpoints and verified replays.
The depth plateau, despite improved score consistency, motivated using its
slot for a different archive criterion; no wall-clock deadline forced the stop.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-long-lookback-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-long-lookback-reproduction/artifacts \
  --curriculum-lookback 128 --steps 0 --eval-every 100000
```

### Shorter-action learning check

An isolated [four-worker adaptation check](results/defense/training/short-action-smoke-01/resume-config.json)
resumed the same run-12 parent at **10,447,616**, changing action duration from
100,000 to **50,000 T-states**. It added **16,384 actions** with the same
settings as the earlier four-worker no-noise control; configuration differences
are only paths and action duration. This changes the policy's control cadence
without changing the original executable. It also halves the emulated-time
span of the four-frame input, rollout, lookback and unchanged per-action
discount horizon, so it is not a matched game-time or credit-horizon experiment.

Its [ten complete games](results/defense/training/short-action-smoke-01/checkpoint/evaluation.json)
averaged **5,261**, median **5,410**, best **7,370**, all stage 1 with no mission.
The [verified replay](results/defense/training/short-action-smoke-01/replay/replay.html)
reproduced **4,843** neural actions at the shorter cadence. Full configuration,
log, optimizer and replay are preserved. The check exited normally and is
excluded from the collector; the three full learners retain their timing.

To separate immediate timing mismatch from adaptation, the **exact same frozen
parent** was also evaluated at 50,000 T-states on the same ten seeds, without
parameter updates. This [parent comparison](results/defense/training/short-action-smoke-01/frozen-parent-at-50000.json)
averaged **7,939**, median **7,760**, best **10,310**, all stage 1. Its weights
were hash-checked unchanged, and the report is explicitly evaluation-only and
ineligible for promotion. Thus the brief adaptation batch reduced the mean
relative to this parent at the same cadence; it did not merely inherit the
entire observed drop from changing timing. The 100,000-T-state trained control
averaged **9,929**, median **10,460**, best **10,480**.

No boot game or restored segment finished during the short adaptation batch:
the four training episodes were still in progress at its checkpoint. Together
with its small size and changed physical-time horizons, this limits any claim
about eventual shorter-action learning. No full-size replacement was launched
on these results; the earlier-reset and fresh-policy trials continue.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-short-action-check \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-short-action-check/artifacts --tstates 50000 \
  --envs 4 --rollout 256 --batch-size 256 --curriculum-boot-envs 1 \
  --mlx-cache-mb 256 --eval-envs 4 --eval-every 16384 --steps 10464000
```

### Frozen action-rate and visual-history comparison

The earlier shorter-action result needs an important qualification. A
**frozen-parent history-spacing probe** now separates action cadence from
the screen-history span. `rl.defense_temporal_probe` reuses the generic raw-frame
history wrapper with Defense's evaluator. It changes no weights, rewards or
categorical sampling rule, supplies no actions, and is restricted to the ten
reused primary validation seeds. Results are explicitly evaluation-only and
ineligible for promotion. Complete games are uncapped; HUD/terminal settling
adds variable time beyond the nominal history span.

All rows below use run 12's **10,447,616** checkpoint and seeds 10000–10009:

| Action interval / frame stride | Nominal four-frame span | Mean | Median | Best |
| --- | ---: | ---: | ---: | ---: |
| [100,000 / 1, original validation](results/defense/training/ppo-12-lookback/step-000010447616/evaluation.json) | 300,000 | 10,474 | 10,480 | 10,480 |
| [50,000 / 1, frozen control](results/defense/validation/step-10447616-tstates-50000-stride-1.json) | 150,000 | 7,939 | 7,760 | 10,310 |
| [50,000 / 2, frozen probe](results/defense/validation/step-10447616-tstates-50000-stride-2.json) | 300,000 | 10,433 | 10,430 | 10,480 |

Intervals/spans are CPU T-states. All 30 games ended in stage 1 without a
mission. The repeated stride-1 probe reproduced **all ten complete game
records exactly** from the earlier 50,000-T-state evaluation, not merely its
summary. With stride 2, all ten paired scores increased; mean difference was
**+2,494**. The checkpoint hash remained unchanged throughout both probes.
This recovers most of the frozen model's original score consistency at the
faster action rate, but does not demonstrate deeper progression, reliable
winning, or learning at that rate. It also does not convert the earlier small
short-action training regression into a success.

This motivates a subsequent **training** experiment that preserves visual
history and accounts for the changed action duration in its credit/reset
horizons. Such training is not implemented by this probe. The production
learners and shared best remain unchanged. **227 regression tests** passed,
including real Defense frame-spacing/sampling tests, input-seed restrictions,
the no-cap/no-promotion contract and existing generic temporal-history tests.

```bash
venv/bin/python -m rl.defense_temporal_probe \
  results/defense/training/ppo-12-lookback/step-000010447616/model.safetensors \
  --tstates 50000 --stride 2 --envs 4 --output results/defense/validation/new-history-probe.json
# For the narrow-history control, use --stride 1 and a distinct output path.
```

### Trainable observation spacing

The frozen Defense timing probe now uses the environment's native spacing
instead of the wrapper, allowing **greedy DQN** as well as categorical PPO
without changing either policy's action rule. It records original and probe
spacing separately, retains the reused-seed restriction, and cannot promote
results. All **237 regression tests** passed. A
[PPO parity check](results/defense/validation/native-probe-ppo-parity.json)
reproduced all ten earlier wrapper game records exactly.

For ordinary DQN's frozen **1,700,000** checkpoint, the same ten seeds gave:

| Action interval / frame stride | Mean | Median | Best |
| --- | ---: | ---: | ---: |
| [100,000 / 1, baseline](results/defense/validation/dqn-1700000-timing-baseline.json) | 1,372 | 1,470 | 1,670 |
| [50,000 / 1](results/defense/validation/dqn-1700000-tstates-50000-stride-1.json) | 601 | 560 | 1,370 |
| [50,000 / 2](results/defense/validation/dqn-1700000-tstates-50000-stride-2.json) | 924 | 540 | 2,220 |

The baseline reproduced every saved game record exactly. All thirty games
ended in stage 1 without a mission, and weights remained unchanged. Wider
history partially recovers the mean at the faster cadence, but is still below
the original timing and has a lower median than either control. Its higher
single score is diagnostic only, not a new trained or promoted best. This
mixed result does not currently justify a separate faster-action DQN run;
ordinary DQN continues at its original timing while PPO tests learning at the
faster cadence.

Both Defense trainers now accept `--observation-stride` (positive integer,
default **1**). The policy still receives exactly four visible frames. Stride
2 retains seven consecutive frames and selects indices 0, 2, 4, 6; it adds
neither hidden-state input nor an action controller. Snapshot capture retains
all intermediate frames so resumed stacks are exact. Cross-stride or malformed
snapshots are rejected before changing emulator state, including shared score,
screen and age archives. Checkpoint evaluation, recording and replay verification
use the saved stride; legacy checkpoints default to 1.

The existing global best was reloaded under this implementation and reproduced
all **2,580** actions, screens and rewards at its original stride, with the
same **10,480**-point stage-1 result. Real-emulator tests compare stride 2
against the frozen probe wrapper, exercise snapshot continuation and peer
sharing, and check vector workers and both trainers' optimizer resumes.
All **236 regression tests** passed, including the storage checks below.
This is infrastructure for a training experiment, not evidence of a new stage.

Immutable Defense checkpoint/replay copies can be inventoried with
`python -m rl.defense_storage`; `--apply` replaces byte-identical duplicates
with macOS APFS copy-on-write clones. Every path and content hash is retained,
with independent future writes (not hard links). Live `latest`/`best` paths
and logs are excluded. The first pass replaced **393** duplicate files across
**278** groups, with about **1.6 GiB** more free space measured afterward;
concurrent training also changes free space. No historical checkpoint was
deleted, and the existing disk-space safety threshold was not lowered.

### Matched-history learning check and longer trial

The [isolated four-worker check](results/defense/training/matched-history-smoke-01/resume-config.json)
resumed the original run-12 optimizer at **10,447,616**, not a timing-probe
trajectory. It trained **32,768** new actions at 50,000 T-states with stride 2,
ending at **10,480,384**. Relative to the earlier 16,384-action, 100,000-T-state
four-worker control, rollout and batch doubled to **512**, lookback doubled
to **64**, gamma became **sqrt(0.997)** and GAE lambda **sqrt(0.99)**. These
preserve nominal visual, rollout, reset and discount horizons and the number
of minibatches per rollout. Variable HUD settling, new decision opportunities,
RNG use and changed optimization data prevent an exact game-time equivalence.

Four complete boot games and two restored segments finished during training.
All **326** archive events had exact 64-action source/trigger offsets; **60**
originated in restored segments. The check's
[ten complete evaluations](results/defense/training/matched-history-smoke-01/checkpoint/evaluation.json)
averaged **9,685**, median **10,415**, best **10,480**, all stage-1 losses.
This is below the frozen same-timing parent's mean **10,433** and the old
100,000-T-state short control's **9,929**, though above the earlier narrow-history
short-action check's **5,261**. It is not a controlled single-setting ablation
or a demonstrated improvement. Its
[5,089-action verified replay](results/defense/training/matched-history-smoke-01/replay/replay.html),
full optimizer and log are preserved; it exited normally and is excluded
from the collector.

`defense-ppo-25-matched-history` now tests this timing/history combination at
**32 workers**, resuming the original **10,447,616** parent directly, **not**
the short-check checkpoint. It uses rollout 512, batch 512, eight boot-only
workers, score-bin sharing at probability 0.5, lookback 64 and the adjusted
discounts above. Training and games have no action or wall-clock cap; complete
ten-game evaluations occur every approximately 200,000 actions. Batch size
stays 512 for memory headroom, so the larger rollout has more minibatches
than the original normal-size trial. The
[saved configuration](results/defense/training/ppo-25-matched-history/resume-config.json)
records all settings and source hashes. It replaces stopped run 23's compute
slot. The sole collector was restarted with stride-aware verification and this
full trial added; all old sources remain, and small probes remain excluded.
No stage-2 or mission success is claimed from launching it.

Its first [ten complete evaluations](results/defense/training/ppo-25-matched-history/step-000010660608/evaluation.json),
at **10,660,608** (**212,992** new actions), averaged **9,411**, median **9,590**,
best **10,440**. All ended in stage 1 without a mission. Its full optimizer
checkpoint and [5,061-action verified replay](results/defense/training/ppo-25-matched-history/first-replay/replay.html)
are preserved. This first batch is below the frozen timing-matched parent and
the short learning check; the longer trial continues, without replacing the
stronger shared best.

Run 25's second batch at **10,857,216** raised its own best to **10,460**,
but its mean fell to **8,750** (median **8,155**). The full
[checkpoint](results/defense/training/ppo-25-matched-history/step-000010857216/evaluation.json)
and [5,116-action verified replay](results/defense/training/ppo-25-matched-history/replay-10460/replay.html)
are preserved. The five successive validation means were **9,411 → 8,750 →
7,910 → 2,562 → 326**, with final median **300**, best **440** at **11,463,424**.
No complete game reached stage 2 or a mission.

The trial stopped cleanly at **11,512,576**, after **1,064,960** new actions,
**193** complete boot games and **91** restored segments. Its
[final optimizer checkpoint](results/defense/training/ppo-25-matched-history/final-checkpoint/state.json)
and [full log](results/defense/training/ppo-25-matched-history/metrics.jsonl)
are preserved alongside its first/strongest-mean and best-effort checkpoints.
This sustained loss of competence motivated stopping the configuration, not
a wall-clock limit. A separate half-learning-rate check starts from the
original strong parent, never the collapsed model; a learning-rate explanation
is a hypothesis, not a diagnosed cause of this regression.

The [half-learning-rate short check](results/defense/training/matched-history-low-lr-smoke-01/resume-config.json)
differs from the earlier four-worker matched-history check **only** in learning
rate (**0.00025 → 0.000125**) and output paths. Both start from the original
10,447,616 parent and train 32,768 new actions to **10,480,384**. The new
[ten-game result](results/defense/training/matched-history-low-lr-smoke-01/checkpoint/evaluation.json)
was mean **10,152**, median **10,450**, best **10,480**, versus the original
check's mean **9,685**, median **10,415**, best **10,480**. All games remained
stage-1 losses. This short comparison improves retention but remains below
the frozen timing-matched parent's mean **10,433**; it does not establish
deeper progress or explain the longer trial's failure.

The new check completed four boot games; no restored segment had finished
at its checkpoint. It recorded **329** archive events, all with exact
64-action lookback, **30** from ongoing restored segments. Its
[4,987-action verified replay](results/defense/training/matched-history-low-lr-smoke-01/replay/replay.html),
full optimizer and log are preserved, and it exited normally. It is excluded
from the collector and is not used as the parent of the longer trial.

`defense-ppo-27-matched-history-low-lr` now runs at the normal 32-worker size,
starting directly from the same original **10,447,616** parent. Its
[configuration](results/defense/training/ppo-27-matched-history-low-lr/resume-config.json)
differs from run 25's **only** in learning rate (0.000125) and output paths.
It retains unlimited learning/games, 50,000-T-state actions, stride 2,
rollout/batch 512, adjusted gamma/lambda, eight boot-only workers and lookback
64. It replaces stopped run 25's compute slot; DQN runs 22, 24 and 26 continue.
The larger rollout had increased minibatches per nominal game-time window;
halving the rate tests a smaller update size, not an exact optimizer-budget
equivalence because Adam and KL-based epoch stopping are nonlinear. The sole
collector includes run 27, with all older sources retained and small checks
excluded. No longer-run improvement is claimed from the short result alone.

Run 27's [first ten complete evaluations](results/defense/training/ppo-27-matched-history-low-lr/step-000010660608/evaluation.json),
at **10,660,608** (**212,992** new actions), averaged **9,933**, median
**10,385**, best **10,480**, all stage-1 losses. At the same first checkpoint,
run 25 averaged **9,411**, median **9,590**, best **10,440**. The lower-rate
trial therefore retains more score in this first batch, but neither shows
new depth, and long-run stability remains unproven. Its full optimizer and
[5,044-action verified replay](results/defense/training/ppo-27-matched-history-low-lr/first-replay/replay.html)
are preserved separately; the equal best score does not replace the shared
global replay.

At **11,250,432** (**802,816** new actions), its fourth batch averaged
[10,237, median 10,370, best 10,480](results/defense/training/ppo-27-matched-history-low-lr/step-000011250432/evaluation.json).
The full optimizer checkpoint is preserved. This compares with run 25's
same-counter mean **2,562**; the lower-rate variant has retained substantially
more score through these four batches. All ten games still lost in stage 1,
so this is retention, not new depth or proof of lasting stability.

The fifth batch, at **11,463,424**, fell to
[mean 6,929, median 7,900, best 10,460](results/defense/training/ppo-27-matched-history-low-lr/step-000011463424/evaluation.json).
That full checkpoint is preserved as well. Lowering the rate delayed the
earlier run's regression but has not eliminated it; none of these games
reached stage 2. One declining batch is not yet grounds to claim a permanent
plateau or to retire this trial.

The longer run did not recover its peak. It stopped cleanly at
**24,488,704** inherited actions: **14,041,088** new actions, **2,267** new
boot games, **1,461** restored segments and **70** completed ten-game
validations. Its strongest mean remained **10,237** at 11,250,432; none
reached stage 2 or a mission. The last complete batch at **24,455,936**
averaged **6,559**, median **6,335**, best **10,080**. The
[final checkpoint](results/defense/training/ppo-27-matched-history-low-lr/final-checkpoint/state.json),
[last validated checkpoint](results/defense/training/ppo-27-matched-history-low-lr/step-000024455936/evaluation.json)
and [losslessly compressed full log](results/defense/training/ppo-27-matched-history-low-lr/metrics.jsonl.gz)
are preserved alongside its earlier peak and verified best replay. This
extended depth plateau motivated replacing the run, not a wall-clock limit.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-matched-history-reproduction \
  --resume results/defense/training/ppo-12-lookback/step-000010447616 \
  --artifacts runs/defense-matched-history-reproduction/artifacts \
  --envs 32 --rollout 512 --batch-size 512 --tstates 50000 --observation-stride 2 \
  --gamma 0.9984988733093293 --gae-lambda 0.99498743710662 \
  --curriculum-lookback 64 --curriculum-boot-envs 8 --mlx-cache-mb 512 \
  --eval-envs 8 --eval-every 200000 --steps 0
# For run 27's lower-rate variant, use a distinct run/artifact directory and
# add --learning-rate 0.000125, keeping the original parent above.
```

### PPO value-loss weight comparison

`rl.defense_train --value-coefficient` controls the weight on **half the mean
squared value error** in PPO's existing joint loss. The default remains **0.5**,
so the complete value term is still 0.25 times mean squared error. A value of
0.1 makes that term 0.05 times mean squared error. The actor objective, entropy
coefficient, advantage/return calculation, reward, observations, action set
and evaluation policy are unchanged. This is not a reward bonus or a second
source of supervision. SIL's separate loss, when enabled, is unaffected.

The option is saved and inherited on resume, can be explicitly overridden,
and defaults to the old weight for older checkpoints. Current live learners
are not silently reconfigured. Source hashes now identify the Defense trainer,
PPO loss implementation and unchanged network in new configuration records.

This tests whether reducing the critic's contribution to the shared encoder
helps retain useful policy behavior. Rising value error around run 27's
regression motivates the comparison but does not establish causality; error
can also be a consequence of changed behavior. Adam's existing moments and
joint gradient clipping mean reducing this coefficient is not equivalent to
scaling a separate critic learning rate.

All **252 regression tests** passed. Targeted checks verify exact default loss
and every gradient against the previous formula (including reference-KL mode),
the isolated coefficient's effect on value/shared gradients without changing
actor-head gradients, invalid arguments, real-emulator default parity and
resume inheritance/override/legacy fallback. Paired short training checks use
the same original parent and settings, changing only the coefficient; their
sources are excluded from the shared best collector.

The completed matched pair used four workers, 512-step rollouts, batch 512,
50,000-T-state actions, stride 2, learning rate 0.000125, the adjusted gamma/
lambda and 64-action lookback of the lower-rate timing check above. Both
started directly from original parent **10,447,616** and collected **32,768**
new actions, ending at **10,480,384**. Configuration differences are only the
coefficient and output paths; no evaluation trajectory is training data.

| Value coefficient | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0.5 control](results/defense/training/value-control-01/checkpoint/evaluation.json) | 10,152 | 10,450 | 10,480 | [4,987](results/defense/training/value-control-01/replay/replay.html) |
| [0.1 reduced](results/defense/training/value-small-01/checkpoint/evaluation.json) | 8,679 | 7,940 | 10,460 | [5,084](results/defense/training/value-small-01/replay/replay.html) |

All twenty validation games lost in stage 1. The reduced weight scored lower
on nine of the ten paired seeds, higher on one, with a mean difference of
**−1,473**. It is not being promoted to a longer run. This negative short
comparison does not prove why PPO regressed, or rule out every other critic
configuration. The control additionally reproduced the earlier lower-rate
check's model bytes, all **26 optimizer arrays**, counters/RNG and all ten
game records **exactly**; see its [parity record](results/defense/training/value-control-01/parity.json).

Both checks exited normally; full models, optimizer states, logs, settings
and verified replays are preserved. Each completed four new boot games during
training; the control completed no restored segments and the reduced variant
completed two. Those restored segments are not counted as complete games.
The opposite-direction **1.0** coefficient check used the same parent and
settings, recorded [here](results/defense/training/value-large-01/resume-config.json).
Its [ten complete games](results/defense/training/value-large-01/checkpoint/evaluation.json)
averaged **10,449**, median **10,450**, best **10,480**. That is **297** more
than the matched control's mean, still entirely stage-1 losses. It completed
four new boot games and no restored segments during training, exited normally,
and preserved its full model/optimizer/log plus a
[5,104-action verified replay](results/defense/training/value-large-01/replay/replay.html).
This short result supports a longer comparison, not a claim of new depth,
durable stability or a diagnosed cause of the earlier regression.

`defense-ppo-29-value-weight` now runs at the normal **32-worker** size,
starting directly from original parent **10,447,616**, not from the short
check. Its [configuration](results/defense/training/ppo-29-value-weight/resume-config.json)
matches run 27's learning settings except for **value coefficient 1.0**
instead of the old implicit 0.5: learning rate 0.000125, 512-step rollouts,
batch 512, 50,000-T-state actions, stride 2, adjusted gamma/lambda, shared
score archive, eight boot-only workers and 64-action lookback. Training and
evaluation are uncapped; ten-game evaluation occurs every 200,000 actions.
It replaces stopped run 27's slot. The sole collector includes this source
and all historical sources; short checks remain excluded.

To reproduce, use run 27's command above with distinct output paths and
`--learning-rate 0.000125 --value-coefficient 1`.

Run 29's [first ten complete games at 10,660,608](results/defense/training/ppo-29-value-weight/step-000010660608/evaluation.json)
(**212,992** new actions) averaged **10,210**, median **10,460**, best
**10,480**. The full optimizer checkpoint and
[5,044-action verified replay](results/defense/training/ppo-29-value-weight/first-replay/replay.html)
are preserved. Its mean exceeds run 27's same-counter 9,933 by 277, but all
games still lost in stage 1. This is early retention, not proof of durable
stability or a depth gain, and the tied best does not replace the global replay.

Run 29 later reached [mean **10,461**, median **10,460**, best **10,480** at
11,250,432](results/defense/training/ppo-29-value-weight/step-000011250432/evaluation.json).
Its full optimizer checkpoint is preserved. Subsequent means fluctuated and
fell to **8,105** at **15,248,128**; all remained stage-1 losses. It subsequently
paused cleanly for disk pressure at **15,674,112** inherited actions
(**5,226,496** new actions). All **26** complete validation batches stayed in
stage 1; the last mean was **7,879** at **15,657,728**. The
[pause checkpoint](results/defense/training/ppo-29-value-weight/pause-checkpoint-000015674112/state.json),
[validation history](results/defense/training/ppo-29-value-weight/validation-at-000015674112.json)
and [losslessly compressed complete log](results/defense/training/ppo-29-value-weight/metrics-at-000015674112.jsonl.gz)
are preserved. This is not evidence that more compute has solved progression.

After storage recovery, run 29's
[first resumed validation at **15,887,104**](results/defense/training/ppo-29-value-weight/step-000015887104/evaluation.json)
averaged **8,966**, median **9,935**, best **10,440**, all ten stage-1 losses.
That is above its last pre-pause mean of 7,879, but below its preserved peak
10,461 and the shared best's score. The full optimizer checkpoint is preserved;
the continuation did not replace the global replay.

Run 29 subsequently retired cleanly at **18,279,168** inherited actions,
**7,831,552** new actions beyond its original parent. All **38** complete
ten-game validations stayed in stage 1; peak mean remained 10,461. The final
validation at **18,082,560** averaged **8,416**, median **8,595**, best **10,220**.
Its [last validation checkpoint](results/defense/training/ppo-29-value-weight/step-000018082560/evaluation.json),
[final optimizer checkpoint](results/defense/training/ppo-29-value-weight/final-checkpoint-000018279168/state.json),
[retirement record and curve](results/defense/training/ppo-29-value-weight/retirement-000018279168.json)
and [complete compressed log](results/defense/training/ppo-29-value-weight/metrics-at-000018279168.jsonl.gz)
are preserved. The stopped process was confirmed gone. Sustained regression
and no new stage—not a wall-clock limit—motivated releasing its compute for
the controlled persistent-exploration comparison below. Its earlier best replay
and every previous checkpoint remain available.

### Recurrent screen-history experiment

`rl.defense_train --recurrent-hidden 128 --sequence-length 32` adds a GRU to
the existing CNN's 256-feature output, with residual actor/value projections.
The motivating question is whether learned memory of earlier visible screens
helps beyond the four-frame input. Memory has been studied in
[Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527); this implementation
is **residual GRU PPO**, not a reproduction of that paper's LSTM DQN. We have
not established that insufficient history causes the current stage-1 plateau.

Only the policy's own screen history enters the recurrent state. There are
no game-private inputs, position labels, routes, oracle actions or extra
rewards. Memory starts at zero on boot or a real environment reset (including
a reset to an opaque own-reached state), not at a visible ship loss. Evaluation
always starts from boot; parallel games have independent memory keyed by game
identity, and evaluation never advances the learner's carried memory.

Training shuffles contiguous within-worker sequences, not individual frames.
The sequence length controls truncated backpropagation, while inference memory
can persist for the full game. Initial sequence states are detached from the
gradient; no burn-in is implemented. Carried states can therefore be stale
after an optimizer update. Resuming restores weights, optimizer and action RNG,
but emulator episodes and neural memory restart from boot. Own-state archives
do not contain saved neural memory. These are experiment limitations, not
claims of exact trajectory continuation.

`--initialize-policy` copies the entire compatible **own learned** feedforward
PPO into the residual base. The memory output projections start at zero,
preserving the parent's initial logits and values; the optimizer and action
counters start fresh. The base remains trainable. Saved provenance includes
the parent's model/configuration hashes and its **10,447,616** pretraining
actions: this is transfer from our own policy, not learning from scratch or
from demonstrations. `--memory-scale 0` supplies a matched memory-disabled
control with the same initialization and sequence batching. Default
`--recurrent-hidden 0` preserves the original feedforward path.

The [zero-update initializer](results/defense/training/recurrent-initial-01/checkpoint/state.json)
reproduced **all ten parent game records exactly** on reused validation seeds
10000–10009: mean **10,474**, median/best **10,480**, all stage-1 losses.
Its [replay](results/defense/training/recurrent-initial-01/replay/replay.html)
reverified **2,553** neural actions. This proves initial-policy preservation,
not an improvement. The initializer's checkpoint, fresh optimizer, provenance,
[parity record](results/defense/training/recurrent-initial-01/parity.json) and
complete-game evaluation are preserved separately from the shared best.

Regression tests cover sequence ordering, zero-residual identity, within-sequence
resets, per-game parallel memory, memory learning, disabled-memory gradients,
bootstrap/evaluation isolation, native full-game serial/parallel agreement,
saved-policy replay verification, invalid modes and real training/resume.
SIL and actor-noise modes are rejected with recurrent training until separately
implemented and tested.

The two **32,768-action** checks completed normally, using the same initializer,
four workers, 256-step rollouts, batch 512, 32-step sequences, learning rate
0.000125, entropy 0.002, value coefficient 0.5, 100,000-T-state actions and
stride 1. Shared own-score resets used probability 0.5, one boot-only worker
and lookback 32. Configuration differs only in memory scale and output paths.

| Memory | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [Enabled](results/defense/training/recurrent-memory-01/checkpoint/evaluation.json) | 9,499 | 10,300 | 10,440 | 2,462 |
| [Disabled control](results/defense/training/recurrent-control-01/checkpoint/evaluation.json) | 10,169 | 10,395 | 10,480 | 2,489 |

Both evaluated ten complete boot games on reused seeds 10000–10009; every
game lost in stage 1. Memory was **670 points worse in mean** than its matched
control and both regressed from the untouched parent's 10,474. This short
comparison does not establish a benefit from recurrence. The enabled check
completed **12** boot games and **one** restored segment during learning;
the control completed **10** boot games and **four** restored segments.
All checkpoints, optimizers, configurations, full logs and separately verified
replays are preserved. Neither short check enters the shared collector.

The initial full suite passed **258 tests**. After strengthening memory tests,
the **259-test** suite passed all seven recurrent tests and the other
non-supervisor checks, but **three supervisor checks failed** when actual free
disk space fell below **5 GiB**. The safety threshold and tests were not
weakened. No unlimited recurrent run was launched during that pause. After the
user freed space, all **259 tests passed**; the log is linked in the restart
section above.

A larger matched comparison then used the normal **32 workers** and
**131,072** new actions per arm, starting from the original zero-update
initializer, not either four-worker short-check result. Both used eight
boot-only workers, the same 256-step rollouts and 32-step sequences, batch 512,
and unchanged learning/timing settings. Each evaluated ten complete games on
the same reused validation seeds; all were stage-1 losses.

| Memory, 32-worker check | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [Enabled](results/defense/training/recurrent-memory-calibration-01/checkpoint/evaluation.json) | 10,032 | 10,430 | 10,480 | 2,521 |
| [Disabled control](results/defense/training/recurrent-control-calibration-01/checkpoint/evaluation.json) | 10,300 | 10,445 | 10,480 | 2,560 |

The enabled run completed **32** boot games and **12** restored training
segments; the control completed **32** boot games and **11** restored segments.
Both exited normally. Their configurations differ only in memory scale and
output paths, as recorded in the
[paired comparison](results/defense/training/recurrent-memory-calibration-01/comparison.json).
Memory was **268 points lower in mean**, and both remained below the
untouched parent's 10,474. This does not establish a benefit from this recurrent
setup at either tested size; it does not rule out all recurrent methods.
Neither check was promoted to an unlimited run or added to the collector.
Full model/optimizer checkpoints, configurations, logs and verified replays
are preserved, with the parent's 10,447,616 pretraining actions recorded in
both lineages. These are reused-seed tuning results, not fresh success rates.

To reproduce this larger pair, use the two bounded commands below with
`--envs 32 --curriculum-boot-envs 8 --steps 131072 --eval-every 131072
--eval-envs 8 --mlx-cache-mb 512` and new run/artifact paths. The two checks
still start from the same saved initializer; only the control sets memory
scale to zero.

```bash
# Save an exact-policy recurrent initializer without learning:
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-initial-reproduction \
  --artifacts runs/defense-recurrent-initial-reproduction/artifacts \
  --initialize-policy results/defense/training/ppo-12-lookback/step-000010447616 \
  --initialize-only --recurrent-hidden 128 --sequence-length 32 \
  --envs 4 --rollout 256 --batch-size 512 --learning-rate .000125 \
  --entropy .002 --gae-lambda .99 --life-terminal \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 1 \
  --curriculum-lookback 32 --eval-envs 4 --eval-every 32768 --mlx-cache-mb 256

# Matched bounded checks; initialize-only is deliberately not inherited:
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-memory-reproduction \
  --artifacts runs/defense-recurrent-memory-reproduction/artifacts \
  --resume results/defense/training/recurrent-initial-01/checkpoint --steps 32768
venv/bin/python -m rl.defense_train --run runs/defense-recurrent-control-reproduction \
  --artifacts runs/defense-recurrent-control-reproduction/artifacts \
  --resume results/defense/training/recurrent-initial-01/checkpoint \
  --memory-scale 0 --steps 32768
```

#### Frozen own-policy residual experiment

`--freeze-recurrent-base` freezes the own initialized CNN, feature layer and
original actor/value heads **before Adam is constructed**. Only the GRU and
its actor/value residual heads receive gradients and optimizer slots. Complete
checkpoints still save every frozen weight, so evaluation/replay needs no
external base model. The default remains fully trainable; fresh frozen mode
requires compatible `--initialize-policy` and enabled memory. Changing the
frozen set during optimizer resume is rejected rather than silently mixing
incompatible optimizer states.

This tests whether a fixed base can retain a useful starting representation
while a learned residual adapts it. A related decomposition appears in
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
That work combines conventional continuous control with a learned correction;
ours is **not a reproduction**: it adds categorical-logit/value residuals over
our own RL-trained neural policy. No conventional controller, action oracle,
demonstration or hidden game information enters this experiment. Freezing
guarantees unchanged base parameters, **not unchanged overall behavior or a
performance improvement**; the learned residual can still make play worse.

The [frozen initializer](results/defense/training/recurrent-frozen-initial-01/checkpoint/state.json)
has **byte-identical model weights and identical policy RNG** to the original
recurrent initializer. All **18** retained optimizer arrays exactly equal
their original zero-step values; the original **42-array** optimizer's base
slots are absent. The [parity record](results/defense/training/recurrent-frozen-initial-01/parity.json)
checks the saved inference settings too. This reuses the prior verified
initializer's identity; it is not a newly measured ten-game evaluation.
Provenance retains the own feedforward parent's **10,447,616** training actions.

All [**262 regression tests passed**](results/defense/diagnostics/frozen-recurrent-regression-tests.txt).
New checks establish exact base immutability while memory weights learn,
absence of base gradients/optimizer slots, complete checkpoint serialization,
native saved-policy replay reproduction, inherited freezing on real training
resume, incompatible-mode rejection and unchanged default initialization.

`defense-recurrent-frozen-calibration-01` completed **131,072** new actions with
32 workers, eight boot-only workers and the same learning settings as the
previous memory-enabled calibration. It starts from the exact-weight frozen
initializer, not the already-trained short result. Its
[ten complete evaluations](results/defense/training/recurrent-frozen-calibration-01/checkpoint/evaluation.json)
averaged **10,453**, median **10,460**, best **10,480**, all stage-1 losses.
That is **421** above the matched unfrozen-memory mean 10,032, and **153** above
the memory-disabled control's 10,300, but still **21 below** the untouched
parent's 10,474. This is better short-run retention, not new stage reach or
proof that recurrence improves on the original parent.

It completed **32** boot games and **10** restored segments during training,
then exited normally. The full checkpoint, optimizer, configuration, log and
[2,551-action verified replay](results/defense/training/recurrent-frozen-calibration-01/replay/replay.html)
are preserved. The [post-training comparison](results/defense/training/recurrent-frozen-calibration-01/comparison.json)
checks all **12** base arrays against the original feedforward parent:
**exactly unchanged**, while all **eight** memory/residual arrays changed.
The optimizer still has **18** arrays and no base slots. Both recurrent arms'
learning and environment settings match; freezing changes the trainable
parameter set, and all seeds are reused validation seeds, not fresh tests.
The bounded check remains excluded from the collector.

The improved retention supports an exploratory longer trial,
`defense-ppo-30-frozen-memory`, continuing this calibration's full optimizer
at **131,072** (plus the inherited base's **10,447,616** pretraining actions).
The [configuration](results/defense/training/ppo-30-frozen-memory/resume-config.json)
keeps 32 workers, 256-step rollouts, batch 512, 128 memory units, 32-step
sequences, learning rate 0.000125, entropy 0.002, value coefficient 0.5,
gamma 0.997, lambda 0.99, 100,000-T-state actions, stride 1, life boundaries,
shared own-score resets, eight boot-only workers and lookback 32. Training
and games are uncapped; ten-game validation runs every 200,000 actions.
Episodes, neural memory and own-state archives restart, not the optimizer.
Runs 24 and 29 continue independently. The sole collector was cleanly
restarted with the recurrent loader and this new full-run source, retaining
all historical sources and excluding all short checks. It still requires
frozen-policy replay verification before replacing the shared best.

Run 30's [first full-run validation at **335,872**](results/defense/training/ppo-30-frozen-memory/step-000000335872/evaluation.json)
(**204,800** actions beyond calibration) averaged **10,464**, median/best
**10,480**, all ten stage-1 losses. Its
[2,494-action verified replay](results/defense/training/ppo-30-frozen-memory/first-replay/replay.html)
and full optimizer checkpoint are preserved. An additional
[saved-weight check](results/defense/training/ppo-30-frozen-memory/step-000000335872/base-immutability.json)
confirms all **12** base arrays remain exactly equal to the original own
feedforward parent and the optimizer still excludes their slots. Since the
calibration, training completed **55** additional boot games and **17** restored
segments. This is continued retention, slightly above calibration's mean
10,453 but below the original parent's 10,474; it is not a stage clear. The
tied best score does not replace the global replay. Unlimited learning continues.

The [next four validation rounds](results/defense/training/ppo-30-frozen-memory/validation-through-000001138688.json)
had means **10,464**, **10,382**, **10,476** and **10,464**. The then-peak at
**933,888** is preserved as a
[full optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000000933888/evaluation.json).
Independent frozen re-evaluation exactly reproduced **all ten game records**,
and its [2,551-action replay](results/defense/training/ppo-30-frozen-memory/replay-peak-933888/replay.html)
reproduced every neural action, reward and screen from boot. All twelve base
arrays remain exactly equal to the original parent; no base optimizer slots
were introduced. Mean 10,476 is only **two points** above that parent's 10,474,
and below run 21's earlier 10,478. All games still end in stage 1, with no
mission. This separately preserved evaluation-only replay does not replace
the tied global best or supply training data.

Reviewing the [first eighteen complete validation rounds](results/defense/training/ppo-30-frozen-memory/validation-through-000003735552.json)
identified a later mean-score peak at **1,335,296**: **all ten games scored
10,480**, all ending in stage 1 without a mission. That previously unarchived
[full optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000001335296/evaluation.json)
is now preserved. Independent frozen re-evaluation reproduced **every one of
the ten complete game records exactly**, and the separate
[2,577-action replay](results/defense/training/ppo-30-frozen-memory/replay-peak-1335296/replay.html)
reproduced every neural action, screen and reward from boot. All twelve base
arrays still exactly match the original own feedforward parent, with no base
optimizer slots. The mean is six points above that parent's 10,474, not a
new single-game ceiling or stage clear. These are repeatedly used validation
seeds, not a fresh success-rate test. Later means varied again (10,328 at
3,735,552); all eighteen rounds stayed in stage 1. The stable shared replay
remains unchanged on a tied best; this selected peak is separately available
and excluded from training data and automatic evaluation-probe promotion.

PPO 30 subsequently stopped cleanly at **6,340,608** after **31** complete
ten-game validation rounds, all stage-1 losses, and no later-stage event in
training. It had not exceeded the score/depth ceiling first achieved at
1,335,296 after another **5,005,312** actions. Its last validation at
**6,332,416** again scored 10,480 in all ten games; this is consistent replay
of the same ceiling, not stage progression. Retirement releases compute for
the matched persistent-exploration/own-reset experiment, not because of a
wall-clock limit. All earlier peaks and verified replays remain intact.

The [full final checkpoint](results/defense/training/ppo-30-frozen-memory/final-checkpoint-000006340608/state.json),
[last complete evaluation and optimizer checkpoint](results/defense/training/ppo-30-frozen-memory/step-000006332416/evaluation.json),
[31-round retirement record](results/defense/training/ppo-30-frozen-memory/retirement-000006340608.json)
and [complete compressed log](results/defense/training/ppo-30-frozen-memory/metrics-at-000006340608.jsonl.gz)
are preserved. Final cumulative counts are **1,884** boot games and **1,291**
restored segments. The final post-update checkpoint has not itself been
evaluated and is not substituted for the independently verified selected peak.
A [read-only final parameter check](results/defense/training/ppo-30-frozen-memory/final-checkpoint-000006340608/base-immutability.json)
confirms all twelve base arrays remain byte-identical to the original own
feedforward parent, with only eighteen non-base optimizer arrays.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-recurrent-frozen-calibration-reproduction \
  --artifacts runs/defense-recurrent-frozen-calibration-reproduction/artifacts \
  --resume results/defense/training/recurrent-frozen-initial-01/checkpoint \
  --steps 131072

# Continue the verified calibration without a training/action limit:
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-frozen-memory-reproduction \
  --artifacts runs/defense-frozen-memory-reproduction/artifacts \
  --resume results/defense/training/recurrent-frozen-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

### Independent Double-DQN training path

`python -m rl.defense_dqn` provides a separate value-learning alternative to
the PPO trials. It reuses the repository's MLX dueling network, Double-Q target
calculation, prioritized replay and n-step return code, generalized to Defense's
20 actions while retaining Breakdown's six-action defaults. Method references:
[Double DQN](https://arxiv.org/abs/1509.06461),
[prioritized experience replay](https://arxiv.org/abs/1511.05952) and
[dueling networks](https://arxiv.org/abs/1511.06581). This is not Rainbow, a
bootstrapped ensemble, an exact paper reproduction or evidence of improvement.
It tests a different learning/update process after the repeated PPO depth
plateaus; no particular cause of those plateaus has been established.

Fresh DQN training starts with random weights and uses **only its own new
screen transitions**. It does not load PPO trajectories, evaluation traces,
demonstrations, stored emulator snapshots or pretrained features. The default
has no state resets; the optional own-state mode below collects its own new
states during training. Uniform random
epsilon-greedy exploration is training-only; complete-game evaluation and
exported replays use the **greedy learned Q-values**. The policy loader checks
the algorithm explicitly, rejects unsupported algorithms and refuses sampling-
temperature overrides for DQN. PPO sampling behavior remains unchanged, and
PPO/DQN optimizer resumes cannot be interchanged accidentally.

Defaults are eight emulator workers, batch 64, a bounded **50,000-transition**
buffer (409.6 MB for the two raw screen-stack arrays, plus small metadata),
five-step returns, discount 0.997, learning rate 0.0001, score scale 0.01 and
visible life boundaries. There is one optimizer update per 16 aggregate new
actions and a target-network copy every 2,000 updates. Replay priorities use
alpha 0.6; importance weights anneal from beta 0.4 to 1 over one million
aggregate actions. Warmup fills 10,000 transitions with random actions;
epsilon otherwise decays from 1 to 0.05 over one million actions after the
initial warmup allowance. There is no reward clipping, survival bonus,
intrinsic reward, scripted action sequence or training/evaluation action cap.

Checkpoints preserve the online network, independent target network, Adam
state, counters and RNG. Resume deliberately refills replay from new own
experience and restarts emulator episodes; it is not exact trajectory
continuation. The usual 10-game validation and reload-and-reproduce replay
gates apply. Target weights and optimizer remain in the resumable checkpoint;
watching the frozen policy requires only the online weights and its config.

All **213 regression tests** passed, including Double-Q action-selection/target-
evaluation arithmetic, 20-action updates, target isolation/synchronization,
greedy-policy round trips without RNG draws, episode-boundary/truncation
returns, malformed configuration rejection, real-emulator training/resume and
exact zero-update restoration of online, target, Adam and RNG state. Existing
PPO/Breakdown tests also passed; completed Breakdown artifacts are unchanged.

An isolated [four-worker integration run](results/defense/training/dqn-smoke-01/config.json)
started from random seed-97 weights and trained for **16,384 actions**. It used
an 8,192-transition buffer, 1,024-transition warmup and target copies every
256 updates; other learning defaults above were retained. Its
[ten complete greedy games](results/defense/training/dqn-smoke-01/checkpoint/evaluation.json)
averaged **328**, median **320**, best **360**, all stage 1 without a mission.
Its [verified greedy replay](results/defense/training/dqn-smoke-01/replay/replay.html)
reproduced **1,655** actions. Full online/target/optimizer state, logs and replay
are preserved. The run exited normally and is excluded from the collector.
This establishes end-to-end integration, not an advantage over PPO or progress
beyond the existing best. It is not initialized from the earlier PPO models.

`defense-dqn-22-fresh` now trains from random seed-97 weights with the full
defaults above, **not** from the small integration checkpoint. Its
[configuration](results/defense/training/dqn-22-fresh/config.json) preserves the
exact source hashes. This is an additional eight-worker learner, not another
32-worker PPO process; its fixed replay bound limits screen-storage growth.
Before launch, macOS reported 43% system-wide memory free. Resource use is
monitored alongside the three continuing PPO trials; no existing learner was
stopped while it was making fresh progress. The sole restarted collector now
includes this run and every previous source, but excludes the integration run.
The unchanged global-best PPO replay was also reloaded and all **2,580** actions
reproduced with the algorithm-aware loader before this launch. DQN has not yet
surpassed that model or established a stage clear.

Run 22's [first ten complete greedy games](results/defense/training/dqn-22-fresh/step-000000100000/evaluation.json)
at **100,000** fresh actions and **5,624** optimizer updates all scored **280**
(mean/median/best 280), with no stage advance or mission. Its
[first verified replay](results/defense/training/dqn-22-fresh/first-replay/replay.html)
reproduced **1,570** actions. Online/target weights and Adam state are preserved.
Training epsilon was still approximately **0.9145** at this checkpoint; greedy
evaluation contains none of that random exploration. This is an early baseline,
not an improvement over the integration run or PPO. Once the 50,000-transition
buffer filled, swap use remained near **5.3 GiB** and the other learners
continued progressing; MLX's reported DQN peak was about **296 MB** (excluding
the NumPy replay buffer and emulator processes). Resource use remains monitored.

At **400,000** actions, run 22 improved to
[mean 322, median 320, best 340](results/defense/training/dqn-22-fresh/step-000000400000/evaluation.json).
Its online/target/optimizer checkpoint and
[1,651-action verified greedy replay](results/defense/training/dqn-22-fresh/replay-340/replay.html)
are preserved. This is a small within-run improvement, still stage 1 and well
below the PPO models; it establishes no algorithm advantage.

At **900,000** actions, ordinary DQN reached
[mean 336, median 340, best 360](results/defense/training/dqn-22-fresh/step-000000900000/evaluation.json).
The full online/target/optimizer and
[1,672-action verified replay](results/defense/training/dqn-22-fresh/replay-360/replay.html)
are preserved. All ten games remained in stage 1; this remains a local
improvement, not a new global best or stage progression.

At **1,000,000**, ordinary DQN's
[mean 354, median 360, best 380](results/defense/training/dqn-22-fresh/step-000001000000/evaluation.json)
improved again. Its full checkpoint and
[1,695-action verified replay](results/defense/training/dqn-22-fresh/replay-380/replay.html)
are saved. These ten complete games still all ended in stage 1 without success.

At **1,200,000**, ordinary DQN reached
[mean 376, median 380, best 400](results/defense/training/dqn-22-fresh/step-000001200000/evaluation.json).
The full online/target/optimizer and
[1,703-action verified replay](results/defense/training/dqn-22-fresh/replay-400/replay.html)
are preserved. All ten games remained in stage 1, with no mission completion.

At **1,300,000**, ordinary DQN's best increased to **420**, although its
[mean 366 and median 360](results/defense/training/dqn-22-fresh/step-000001300000/evaluation.json)
fell from the previous batch. Its full checkpoint and
[1,785-action verified replay](results/defense/training/dqn-22-fresh/replay-420/replay.html)
are preserved. All ten games remained in stage 1 without a mission; a higher
single effort does not imply a better mean or deeper progression.

At **1,400,000**, ordinary DQN reached
[mean 458, median 470, best 500](results/defense/training/dqn-22-fresh/step-000001400000/evaluation.json).
Its full online/target/optimizer checkpoint and
[1,963-action verified replay](results/defense/training/dqn-22-fresh/replay-500/replay.html)
are preserved. All ten complete games still ended in stage 1 without a mission.

At **1,600,000**, ordinary DQN reached
[mean 991, median 995, best 1,390](results/defense/training/dqn-22-fresh/step-000001600000/evaluation.json),
with a [2,239-action verified replay](results/defense/training/dqn-22-fresh/replay-1390/replay.html).
At **1,700,000**, it improved to
[mean 1,372, median 1,470, best 1,670](results/defense/training/dqn-22-fresh/step-000001700000/evaluation.json),
with a [2,201-action verified replay](results/defense/training/dqn-22-fresh/replay-1670/replay.html).
Both full online/target/optimizer checkpoints are preserved. All twenty games
remained stage-1 losses. The next 1,800,000-action batch regressed to mean
392, median 400, best 440: improvement is not monotonic, and these milestones
do not replace the stronger PPO global best.

At **2,000,000**, DQN's
[ten-game mean was 1,187, median 1,260, best 2,060](results/defense/training/dqn-22-fresh/step-000002000000/evaluation.json).
The [2,089-action verified replay](results/defense/training/dqn-22-fresh/replay-2060/replay.html)
and full optimizer/target checkpoint are preserved. This is a new single-game
best for the independent DQN lineage, but its mean is below the 1,700,000-action
checkpoint's 1,372. All ten games remained stage-1 losses.

Its later **2,200,000** checkpoint improved the ten-game mean to
[1,467, median 1,270, best 2,020](results/defense/training/dqn-22-fresh/step-000002200000/evaluation.json).
That full online/target/optimizer checkpoint is also preserved. The lower
single-game best did not replace its existing 2,060-point replay, and no game
reached stage 2 or a mission.

Run 22 subsequently stopped after **3,030,192** actions, **188,761** updates
and **1,784** complete training games. Its [final resumable checkpoint](results/defense/training/dqn-22-fresh/final-checkpoint/state.json)
and [entire training log](results/defense/training/dqn-22-fresh/metrics.jsonl)
are preserved alongside the best-effort and peak-mean checkpoints above.
Across **30** complete validation batches, none reached stage 2 or a mission.
The [last batch at 3,000,000](results/defense/training/dqn-22-fresh/step-000003000000/evaluation.json)
averaged **356**, median **360**, best **380**. The six final validation means
were 324, 402, 362, 372, 376 and 356. This persistent regression, rather than
a wall-clock limit, motivated replacing its compute slot with run 28 below.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-reproduction \
  --artifacts runs/defense-dqn-reproduction/artifacts
# Later, resume with a fresh replay buffer from the saved online/target/Adam state:
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-continuation \
  --artifacts runs/defense-dqn-continuation/artifacts \
  --resume runs/defense-dqn-reproduction/latest
```

### Persistent random exploration

The shared-barrier replay diagnostic motivates testing **temporal persistence**
in exploration, without supplying a route or using the diagnostic frames as
training examples. [Temporally-Extended Epsilon-Greedy Exploration](https://arxiv.org/abs/2006.01782)
studies randomly sampled actions held for sampled durations. Our optional
ordinary-DQN adaptation is not a reproduction of its Atari agents: it uses
this repository's existing network, score-only reward and primitive-step
replay, with bounded durations and an occupancy-calibrated exploration rate.

`--exploration-max-repeat 1` (the default) executes the **unchanged** independent
epsilon-greedy branch, including its original RNG draws. A larger value enables
`rl.persistent_exploration`: sample uniformly among all existing actions, then
hold that action for a random number of original environment steps. For the
first comparison, durations are 1–64 with probability proportional to
`length**(-1.5)`. No observation, obstacle detector, location, score threshold,
or game-private state selects the start, action or duration. All directions,
firing aliases and no-op remain eligible; there is no preference for right.

Each held primitive step still produces its own screen, visible-score reward,
n-step replay transition and normal update opportunity. This is **not** a new
evaluation action repeat or a change in emulation speed. Visible life loss,
termination and truncation cancel outstanding holds. Replay warmup retains
the old independent uniform random actions. Saved online/target/optimizer and
replay semantics are unchanged. An independent exploration RNG is saved and
restored; active holds and local diagnostic counters clear when resume boots
new games. Bootstrap-head combinations are currently rejected explicitly.

To separate longer sequences from simply adding more random actions, epsilon
denotes a nominal exploratory-step fraction. If mean duration is `m`, an idle
worker starts a random hold with probability `epsilon / (m*(1-epsilon)+epsilon)`.
This is the renewal-process occupancy formula, not the paper's unadjusted
option-start epsilon. Here `m = 6.17855` and epsilon 0.05 gives start probability
**0.00844648**. Stationary uninterrupted occupancy is 5%; episode/life cuts and
changing epsilon can change realized occupancy. Logs and checkpoints retain
actual exploratory steps, duration/action counts and cancelled future steps.

Greedy complete-game validation remains **entirely learned, at the original
100,000-T-state decision cadence**. The evaluation loader never instantiates
the training exploration helper. No reward bonus, demonstration, hidden RAM,
scripted route, or evaluation-only action override is introduced.

The first paired calibration starts both arms from the same preserved ordinary
DQN-28 peak at **6,100,000**, not from either earlier short learning-rate check.
Each collects **131,072** new primitive actions and ends at **6,231,072**.
Both retain eight workers, compact capacity 200,000, batch 64, learning rate
0.0001, n-step 5, gamma 0.997, life learning boundaries, 10,000-entry replay
warmup, update interval 16, target interval 2,000, epsilon 0.05 and no own-state
resets. Each refills replay with its own newly collected experience. The
[control configuration](results/defense/training/persistent-control-01/resume-config.json)
and [persistent configuration](results/defense/training/persistent-repeat-01/resume-config.json)
differ only in maximum repeat, output paths and exploration provenance.
They share the same reused ten validation seeds; neither is a fresh test or
eligible source for the global collector. A complete stage clear remains the
objective, not a higher training score under random exploratory actions.

All **274 regression tests** pass. The five added tests cover the duration
distribution and occupancy calibration, exact holds and boundary cancellation,
RNG reproducibility, invalid CLI combinations, real native training and resume,
and byte-for-byte array equality of default versus explicitly disabled runs
(online, target and optimizer, with identical counters/RNG). A 384,000-decision
synthetic check exercises the nominal 5% occupancy and long-duration tail;
it is a sampler test, not game-performance evidence. The
[complete test log](results/defense/diagnostics/persistent-regression-tests.txt)
is preserved. Existing learners continue using their originally loaded code;
new checkpoints record the exact new trainer/helper source hashes.

Both bounded arms exited normally and independently verified their best replay:

| Calibration | Ten-game mean | Median | Best | Verified greedy replay |
| --- | ---: | ---: | ---: | --- |
| [Independent-action control](results/defense/training/persistent-control-01/checkpoint/evaluation.json) | 10,008 | 10,245 | 10,350 | [2,414 actions](results/defense/training/persistent-control-01/replay/replay.html) |
| [Persistent random exploration](results/defense/training/persistent-repeat-01/checkpoint/evaluation.json) | 10,240 | 10,180 | 10,440 | [2,485 actions](results/defense/training/persistent-repeat-01/replay/replay.html) |

All twenty games lost in stage 1. The [paired mean gain of **232**](results/defense/training/persistent-repeat-01/comparison.json)
comes largely from **one 2,370-point difference**: five seeds improved and five
worsened, and the persistent median is lower. Both means remain below the
untouched parent's 10,290. This is weak early evidence, not a demonstrated
exploration advantage or barrier passage. The control completed **57** new
boot games during training, versus **56** for persistence; neither used resets.
After warmup, persistence recorded **5,798 / 121,040** exploratory primitive
steps (**4.790%**), 929 sampled holds and 264 future steps cancelled at boundaries.
Full online/target/optimizer/RNG checkpoints, configurations, logs and verified
replays are preserved for both arms. Training games and exploratory actions
are not substituted for greedy validation results.

A [screen-only check of these two selected calibration replays](results/defense/diagnostics/shared-loss-persistent-01/report.json)
also does **not** establish passage through the diagnosed obstacle sequence.
The [control's life totals](results/defense/diagnostics/shared-loss-persistent-01/policy-1-losses.png)
are **2,600 / 2,580 / 2,600 / 2,570**; the
[persistent model's totals](results/defense/diagnostics/shared-loss-persistent-01/policy-2-losses.png)
are **2,600 / 2,600 / 2,600 / 2,640**. Some windows show explosion graphics near
the preceding center-gap barrier while the broad barrier is still well above
the ship; they are not all identical to the older best's later approach.
The same flash-alignment limitations apply, and no exact collision cause or
course index was measured. In particular, one 2,640-point life is not proof
of crossing the barrier. These are read-only views of already verified games,
not additional evaluations, learning inputs or a change to the running trial.

To test durability with a matched baseline, unlimited full trials now continue
each arm's own **6,231,072** checkpoint:
[DQN 31 persistent configuration](results/defense/training/dqn-31-persistent/resume-config.json)
and [DQN 32 control configuration](results/defense/training/dqn-32-persistent-control/resume-config.json).
They keep all learning settings, change the validation interval to 200,000,
and remove the training action cap; full games remain uncapped. Each restores
its online/target/optimizer and RNG states, boots new games and refills its
own replay buffer, so this is not exact continuation of prior trajectories.
They are not independently initialized replicates. Frozen-memory PPO 30 and
DQN 24 subsequently retired with all state preserved.
The sole collector was stopped cleanly,
confirmed gone, then restarted with both full-run sources and all 26 historical
sources. Short calibration sources remain excluded. Any new global best still
requires independent frozen-policy replay verification.

The [first full-run comparison at **6,431,072**](results/defense/training/dqn-31-persistent/comparison-at-000006431072.json)
adds **200,000** actions per arm after calibration (**331,072** after the common
original parent). It reverses the short-run mean advantage:

| Full trial | Ten-game mean | Median | Best | Verified greedy replay |
| --- | ---: | ---: | ---: | --- |
| [DQN 31 persistent](results/defense/training/dqn-31-persistent/step-000006431072/evaluation.json) | 9,373 | 10,280 | 10,310 | [2,483 actions](results/defense/training/dqn-31-persistent/first-replay/replay.html) |
| [DQN 32 control](results/defense/training/dqn-32-persistent-control/step-000006431072/evaluation.json) | 10,278 | 10,280 | 10,280 | [2,545 actions](results/defense/training/dqn-32-persistent-control/first-replay/replay.html) |

All twenty complete games lost in stage 1. Persistence is **905 points lower
on average**, with two higher, four lower and four tied seeds. Its exploratory
step fraction during this continuation was **5.212%** (9,902 / 189,968 after
warmup), not a sudden increase in random-action volume. The learners completed
**89** and **88** new boot games respectively, with no restored segments.
Both full online/target/optimizer/RNG checkpoints and independently verified
replays are preserved. The earlier, higher-scoring calibration replays are
unchanged. This does not establish that persistence helps; both trials continue
unchanged for further matched rounds, without treating a single batch as a
success or a definitive rejection. Neither replaced the shared 10,480 best.

The [second matched round at **6,631,072**](results/defense/training/dqn-31-persistent/comparison-at-000006631072.json)
adds **400,000** actions per arm after calibration (**531,072** after the common
parent). [Persistence](results/defense/training/dqn-31-persistent/step-000006631072/evaluation.json)
averaged **10,344**, median **10,350**, best **10,430**;
[control](results/defense/training/dqn-32-persistent-control/step-000006631072/evaluation.json)
averaged **10,042**, median **10,220**, best **10,360**. The mean difference is
now **+302**, with eight higher and two lower paired seeds, reversing the
first round's −905. All twenty games still lost in stage 1. Persistence is
54 points above the untouched parent's mean 10,290, but this fluctuating
comparison is not a durable advantage or a depth gain.

Both full checkpoints and their independently verified
[2,538-action persistent replay](results/defense/training/dqn-31-persistent/replay-10430/replay.html)
and [2,579-action control replay](results/defense/training/dqn-32-persistent-control/replay-10360/replay.html)
are preserved. Since calibration, they completed **174** and **170** new boot
games respectively, with no restored segments. The cumulative measured
persistent exploratory-step fraction was **4.930%**. Earlier checkpoints and
higher-scoring calibration replays remain untouched. Both learners continue
unchanged, and the shared best is still the original verified 10,480 replay.

The [third paired round at **6,831,072**](results/defense/training/dqn-31-persistent/comparison-at-000006831072.json)
adds **600,000** actions per arm after calibration. Persistence averaged
**10,303**, median **10,305**, best **10,420**; control averaged **10,188**,
median **10,240**, best **10,260**. The mean difference is **+115**. All twenty
games again lost in stage 1. Both full checkpoints and complete game records
are preserved; neither beat its earlier archived best replay, so those verified
replays remain the references. This is no stage-depth improvement.

At the [fourth round, **7,031,072**](results/defense/training/dqn-31-persistent/comparison-at-000007031072.json),
persistence reached a new personal peak mean **10,450**, median **10,455**,
best **10,480**. Its independently verified
[2,526-action replay](results/defense/training/dqn-31-persistent/replay-10480/replay.html)
is preserved with the full online/target/optimizer/RNG checkpoint. Control
averaged **10,199**, median/best **10,220**. All ten paired differences favor
persistence, averaging **+251**, but all twenty games still lost in stage 1.
The runs have completed **342** and **343** new boot games since calibration.
This matches the global score ceiling without passing it or establishing a
stage clear; the existing shared replay remains unchanged.

The [six-round comparison through **7,431,072**](results/defense/training/dqn-31-persistent/comparison-at-000007431072.json)
shows another reversal. At 7,231,072 the means were **10,195** persistent and
**9,866** control (+329). At 7,431,072 they were **10,239** persistent versus
**10,304** control (−65), with four higher and six lower paired persistent
scores. Persistent median/best were **10,250 / 10,320**; control median/best
were **10,320 / 10,440**, a new control-run personal best. Its independently
verified [2,588-action replay](results/defense/training/dqn-32-persistent-control/replay-10440/replay.html)
and both complete checkpoints are preserved. The earlier persistent 10,480
replay remains unchanged. Since calibration, the runs completed **508** and
**516** new boot games respectively. All **120** evaluation games across these
six reused-seed rounds per arm lost in stage 1. The evidence remains mixed on
mean score and entirely negative on a new stage; it is not a fresh success rate.

Both older trials subsequently retired cleanly after **eight** paired rounds,
all **160** reused-seed evaluation games still stage-1 losses. DQN 31 stopped
at **8,010,944** actions, **498,806** updates and **3,951** cumulative boot games;
DQN 32 stopped at **8,004,192**, **498,384** updates and **3,963** boot games.
Neither used restored segments. At their last evaluation, **7,831,072**, means
were **10,190** persistent and **8,859** control; bests **10,200** and **10,420**.
They had not exceeded their earlier score/depth ceilings. Retirement releases
compute for the archive-diversity comparison, not because of elapsed time.

The [DQN 31 final checkpoint and eight-round record](results/defense/training/dqn-31-persistent/retirement-000008010944.json)
and [DQN 32 equivalent](results/defense/training/dqn-32-persistent-control/retirement-000008004192.json)
are preserved, with full online/target/optimizer/RNG states, last complete
evaluation checkpoints and complete compressed logs alongside those records.
Final post-update checkpoints are not claimed as separately evaluated.
Their earlier verified best replays remain intact, and both historical sources
remain in the collector. DQN 31's trained lineage continues in trials 33/34;
the independent control's resumable state remains available too.

```bash
# Use distinct run/artifact paths for each arm. Set repeat to 1 for the control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reproduction \
  --artifacts runs/defense-persistent-reproduction/artifacts \
  --resume results/defense/training/dqn-28-large-replay/step-000006100000 \
  --steps 6231072 --eval-every 131072 \
  --exploration-max-repeat 64 --exploration-exponent 1.5

# Unlimited continuation of the verified persistence calibration:
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-full-reproduction \
  --artifacts runs/defense-persistent-full-reproduction/artifacts \
  --resume results/defense/training/persistent-repeat-01/checkpoint \
  --steps 0 --eval-every 200000
```

### Persistent exploration-rate comparison

The [saved duration counters at 6,631,072](results/defense/diagnostics/persistent-duration-6631072.json)
show **3,240** sampled holds over **174** new boot games, but only **131**
sampled durations of at least 32 steps and **53** of at least 48. These counts
include firing/no-op actions and animations; life/episode boundaries can
shorten a sampled hold. The saved marginals cannot identify long **movement**
holds or where they occurred. They do not prove insufficient exploration is
the cause, or that any particular duration would solve the barrier.

They motivate a controlled intensity check rather than assuming the 5% arm
already tried many long maneuvers at the relevant states. Both new bounded
arms start from the **same** preserved DQN-31 checkpoint at **6,631,072**
(mean 10,344), with identical online/target/optimizer and saved RNG ancestry.
Each adds **131,072** actions, ending at **6,762,144**. Their
[control configuration](results/defense/training/persistent-rate-control-01/resume-config.json)
and [higher-rate configuration](results/defense/training/persistent-rate-high-01/resume-config.json)
differ only in paths and **epsilon-final 0.05 versus 0.25**. Both use persistent
durations 1–64 with exponent 1.5, eight workers, compact capacity 200,000,
unchanged learning settings and no own-state resets. Replay refills from new
own experience after boot; no saved evaluation trajectory is loaded.

Under the existing occupancy formula, idle start probabilities are
**0.00844648** and **0.05118847**. Realized fractions can differ because of
boundary cancellation. All action IDs remain uniformly eligible; no direction,
position, obstacle, score threshold or route triggers exploration. This tests
**more random exploratory steps**, not a longer duration distribution or a
change to score-only rewards. Both frozen evaluations remain pure learned
greedy play at the ordinary cadence, on the same ten reused validation seeds.
The full low-rate run 31 and its independent-action control 32 continue
unchanged. Retired bootstrap run 24 released compute; neither bounded source
is included in the shared collector. Existing tested code implements this
comparison without a learner/environment change.

The [completed paired result](results/defense/training/persistent-rate-high-01/comparison.json)
does **not** support increasing the rate on this evidence:

| Nominal persistent fraction | Measured fraction after warmup | Ten-game mean | Median | Best | Verified replay |
| --- | ---: | ---: | ---: | ---: | --- |
| 5% control | 4.542% | 10,324 | 10,280 | 10,480 | [2,509 actions](results/defense/training/persistent-rate-control-01/replay/replay.html) |
| 25% higher rate | 23.869% | 10,115 | 10,110 | 10,260 | [2,458 actions](results/defense/training/persistent-rate-high-01/replay/replay.html) |

The higher rate is **209 points lower on average**, with nine lower paired
scores and one tie. All twenty complete validation games lost in stage 1.
The arms completed **56** and **65** new boot training games respectively,
also without a later-stage or mission event. Both stopped normally at their
predeclared action budget; their full online/target/optimizer/RNG checkpoints,
compressed complete logs and independently verified replays are preserved.
This is one bounded paired comparison on reused validation seeds, not a
definitive rejection of persistent exploration. Neither becomes a new unlimited
trial, and the global best remains unchanged.

A [read-only loss comparison](results/defense/diagnostics/shared-loss-persistent-rate-01/report.json)
reinforces the user's observation. The
[control's four lives](results/defense/diagnostics/shared-loss-persistent-rate-01/policy-1-losses.png)
each scored **2,620**, with the ship near the center/left as the broad barrier
with an opening on the right approaches. The
[higher-rate model](results/defense/diagnostics/shared-loss-persistent-rate-01/policy-2-losses.png)
scored **2,500 / 2,620 / 2,600 / 2,540** across its lives, including losses
around the preceding center-gap barrier. Neither shows passage through this
obstacle sequence. These images align to visible flashes or the final visible
loss, not exact collision timestamps; they do not establish a required route
or the underlying learning failure. No replay data was used for training.

```bash
# Use separate output paths and epsilon-final 0.05 for the matched control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-rate-reproduction \
  --artifacts runs/defense-persistent-rate-reproduction/artifacts \
  --resume results/defense/training/dqn-31-persistent/step-000006631072 \
  --steps 6762144 --eval-every 131072 --epsilon-final 0.25
```

### Persistent exploration with own-state resets

The higher-rate trial above mostly repeated full boot games without escaping
the common obstacle sequence. A bounded follow-up tests whether reusing states
reached during **its own new training** gives sustained random exploration more
opportunities from later positions. This is a hypothesis about training-state
coverage, not a claim that the archive necessarily concentrates on a specific
obstacle or contains a solution.

`defense-persistent-reset-calibration-01` starts from the **same 6,631,072**
online/target/optimizer/RNG parent as both rate checks, **not** from the
higher-rate trial's final weights. It adds **131,072** actions with nominal
persistent exploration 25%, lengths 1–64 and exponent 1.5. Its comparator is
the already completed `persistent-rate-high-01`, so the changed treatment is
the own-state reset setup: probability **0.5**, sharing, **two of eight**
workers permanently boot-only, and **128-action** archive lookback. All other
learning, observation, action timing, replay capacity and evaluation settings
match. This is one sequential paired lineage, not simultaneous independent
replicates. Wall-clock throughput is not its outcome measure.

The archive begins empty and uses the existing bounded visible-score cells.
Only newly reached opaque native snapshots are shared inside this run; none
are decoded, fabricated or taken from an evaluation replay or another learner.
Restored starting score is not credited as reward. Every primitive action
still comes from the learned policy or uniform, screen-independent random
exploration. The longer lookback is the existing generic archive setting,
not an obstacle detector or steering instruction. Boot games and restored
segments are reported separately, and all ten uncapped greedy evaluation
games start from boot. This short source is excluded from the global collector.

Two new regression tests cover this combination without changing production
code. A synthetic boundary fixture confirms held actions cancel both at a
visible life loss and at an own-state reset, replay ends on the actual terminal
screen, reset observations start new transitions, and only new score increments
are recorded. A real-emulator test exercises compact replay, new archive
creation, a reserved boot worker, restored segments and optimizer updates.
Its deliberately truncated episodes test plumbing, not successful gameplay.
All seven focused persistent-exploration tests pass. The full **276-test**
suite also [passes](results/defense/diagnostics/persistent-reset-regression-tests.txt).

An [interim training-only archive audit at **6,698,992**](results/defense/diagnostics/persistent-reset-interim-6698992.json) found that retained
save events had source progress at most **170** points within a life, despite
trigger progress reaching **2,620**. Looking back 128 actions rewinds across
the large score jumps. This does not by itself identify obstacle positions or
prove those states are unhelpful, but it cautions against claiming that this
configuration provides near-failure practice. A second bounded arm,
`defense-persistent-reset-short-calibration-01`, therefore starts from the same
6,631,072 parent with identical settings except **lookback 32** and output
paths. The intended comparison includes both actual archive coverage and
complete from-boot evaluation against lookback 128 and the existing no-reset
high-exploration comparator. Neither reset arm imports the other's experience.

The [lookback-128 trial has finished](results/defense/training/persistent-reset-calibration-01/comparison.json)
at **6,762,144**, with ten complete greedy games averaging **10,153**, median
**10,130**, best **10,310**, all stage-1 losses. That is **+38** over the
no-reset comparator's 10,115, with six higher and four lower paired scores,
and remains below the common parent's 10,344. Its best
[2,596-action replay](results/defense/training/persistent-reset-calibration-01/replay/replay.html)
was independently reproduced from frozen weights; full online/target/optimizer/
RNG state and complete compressed logs are preserved. It stopped normally at
the declared calibration budget and initially was not continued. The later
full run 39 revisits this decision after the shorter-lookback and archive
diversity trials failed to produce a stage clear; see below.

Training completed **50** new boot games and **39** restored segments, all
without a later-stage event. Both reserved workers stayed boot-only. All
**1,639** archive events had exactly 128-action lookback; the **127** retained
save events reached at most **190** points of source within-life progress,
while triggers reached 2,620. This confirms the earlier coverage limitation
through the end of this check, without claiming score measures course distance.
The shorter-lookback result follows below.

The [32-action-lookback arm](results/defense/training/persistent-reset-short-calibration-01/comparison.json)
also stopped normally at **6,762,144**. Its ten complete greedy games averaged
**10,339**, median **10,340**, best **10,390**, all stage-1 losses. Relative to
the no-reset high-exploration comparator this is **+224**, with all ten paired
scores higher. Relative to lookback 128 it is **+186**, with seven higher and
three tied. It is still **5 points below the common parent's mean**, with no
stage-depth gain. This one sequential paired lineage on reused seeds suggests
a useful setting to test further, not a demonstrated solution or fresh success
rate. Full weights/target/optimizer/RNG, compressed logs and its independently
verified [2,580-action replay](results/defense/training/persistent-reset-short-calibration-01/replay/replay.html)
are preserved. No global replay was replaced.

Training completed **54** new boot games and **30** restored segments, all
stage 1. Both boot-only workers remained boot-only. All **1,723** archive
events used exactly 32-action lookback; **191** retained save events reached
source within-life progress **2,570**, versus the long-lookback arm's **190**.
This establishes different saved-state coverage by visible score, not an
obstacle coordinate, exact collision timing, or a count of current archive
contents. The [read-only loss report](results/defense/diagnostics/shared-loss-persistent-reset-01/report.json)
records best-replay life scores of **2,500 / 2,570 / 2,620 / 2,620** for lookback
128 and **2,600 / 2,570 / 2,600 / 2,620** for lookback 32. These remain near the
old score ceiling, with no verified barrier passage or stage transition.

To test whether the bounded reset result lasts, full runs now continue each
arm's own **6,762,144** checkpoint:
[DQN 33 shorter-lookback reset configuration](results/defense/training/dqn-33-persistent-resets/resume-config.json)
and [DQN 34 no-reset configuration](results/defense/training/dqn-34-persistent-rate-control/resume-config.json).
Both retain nominal 25% persistent exploration, eight workers, compact replay
capacity 200,000, batch 64, learning rate 0.0001, n-step 5, discount 0.997,
life learning boundaries and the original 100,000-T-state/stride-1 timing.
Run 33 retains 0.5 shared own-state resets, two boot-only workers and lookback
32; run 34 retains no resets. Training is uncapped, with ten complete greedy
from-boot evaluations every 200,000 new actions, first at **6,962,144**.

Both restore their own online/target/optimizer and RNG states but refill
experience from newly booted games; run 33 also rebuilds its archive from
empty. Thus this is not exact continuation of trajectories, nor independent
random-seed replication. No evaluation replay or other run's snapshots are
training data. Trials 31/32 continue unchanged as the low-rate persistence/
independent-action comparison. Retiring PPO 30 releases capacity for these
four eight-worker DQN learners.

The sole replay collector was stopped cleanly and confirmed gone before
restarting with [all 30 full-run sources](results/defense/training/dqn-33-persistent-resets/collector-config.json),
including the new pair and all historical sources. Short calibrations remain
excluded. Any global replacement still requires an independently reproduced
frozen-policy replay and a higher stage/mission/score rank; the existing best
is never replaced just for a higher mean.

The [first full comparison at **6,962,144**](results/defense/training/dqn-33-persistent-resets/comparison-at-000006962144.json)
adds **200,000** actions per arm after calibration (**331,072** after the common
6,631,072 parent):

| Full trial | Ten-game mean | Median | Best | Independently verified replay |
| --- | ---: | ---: | ---: | --- |
| [DQN 33 own resets](results/defense/training/dqn-33-persistent-resets/step-000006962144/evaluation.json) | 10,259 | 10,430 | 10,480 | [2,564 actions](results/defense/training/dqn-33-persistent-resets/first-replay/replay.html) |
| [DQN 34 no resets](results/defense/training/dqn-34-persistent-rate-control/step-000006962144/evaluation.json) | 10,197 | 10,150 | 10,460 | [2,558 actions](results/defense/training/dqn-34-persistent-rate-control/first-replay/replay.html) |

All twenty complete games lost in stage 1. The reset advantage narrows to
**+62**, with eight higher and two lower paired scores, including one −1,350
outlier. This is not a durable advantage or a new stage; both means remain
below the common parent's 10,344. Matching the global single-game ceiling
does not replace the shared best. Both full online/target/optimizer/RNG
checkpoints, game records and verified replays are preserved.

The reset arm completed **83** new boot games and **45** restored segments;
control completed **99** new boot games, with none restored. Both had no
later-stage training event. Measured exploratory-action fractions were
**24.467%** and **24.388%**, respectively. All **2,753** reset-arm archive
events used 32-action lookback, the **241** retained save events reached
within-life source score **2,600**, and both reserved workers stayed boot-only.
These counts describe retained events over time, not current archive contents
or measured obstacle positions. Both learners continue unchanged for further
matched rounds; no new source data, reward or controller was introduced.

The [second full round at **7,162,144**](results/defense/training/dqn-33-persistent-resets/comparison-at-000007162144.json)
reverses that small advantage: resets averaged **9,942**, median **9,930**,
best **10,040**, versus control mean **10,247**, median **10,230**, best
**10,320**. The mean difference is now **−305**, with all ten paired reset
scores lower. All twenty complete games
again lost in stage 1, after **400,000** additional actions per arm since
calibration. Both complete online/target/optimizer/RNG checkpoints and game
records are preserved. Their first-round verified replays scored higher and
remain unchanged; no new replay is claimed for this lower-scoring round.
The inconsistent mean ranking does not establish a durable reset benefit or
stage-depth progress. Both full runs continue unchanged.

The [three-round record through **7,362,144**](results/defense/training/dqn-33-persistent-resets/comparison-through-000007362144.json)
includes the third round's reset mean **10,029**, median **10,080**, best
**10,180**, versus control mean **9,789**, median **9,820**, best **9,960**.
The +240 difference again reverses the previous round. All sixty evaluation
games across the three rounds lost in stage 1. Complete third-round game
records and model hashes are archived in that report; the earlier, stronger
full optimizer checkpoints and verified replays remain the preserved references.

The [five-round record through **7,762,144**](results/defense/training/dqn-33-persistent-resets/comparison-through-000007762144.json)
adds reset/control means **10,193 / 10,154** and **10,227 / 9,631**.
All **100** complete games across the five paired rounds lost in stage 1;
no mission or later stage was observed. These are repeatedly reused validation
seeds, not independent test games. The record includes complete per-game
results and immutable local model hashes; earlier stronger full optimizer
states and verified replays remain the archived references. The changing
score margin still does not demonstrate progression beyond the shared obstacle.

Runs 33/34 have now stopped cleanly with full state preserved at
[**8,133,536**](results/defense/training/dqn-33-persistent-resets/retirement.json)
and [**8,153,696**](results/defense/training/dqn-34-persistent-rate-control/retirement.json).
They added **1,371,392 / 1,391,552** actions, **529 / 710** complete boot games
and **362 / 0** restored segments since their calibration parents. Their sixth
round, at **7,962,144**, averaged **10,252 / 9,470**, median **10,260 / 9,280**,
best **10,280 / 9,940**, all stage 1. All **120** evaluation games across six
paired rounds lost in stage 1. The two final checkpoints, last evaluated full
checkpoints and complete compressed logs are archived. Final post-update
weights were not separately evaluated. This depth plateau prompted reassigning
compute to distributional learning, not a wall-clock limit; nothing was deleted.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reset-reproduction \
  --artifacts runs/defense-persistent-reset-reproduction/artifacts \
  --resume results/defense/training/dqn-31-persistent/step-000006631072 \
  --steps 6762144 --eval-every 131072 --epsilon-final .25 \
  --curriculum-probability .5 --curriculum-share \
  --curriculum-boot-envs 2 --curriculum-lookback 128

# Unlimited continuation of the verified 32-action-lookback arm:
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-reset-full-reproduction \
  --artifacts runs/defense-persistent-reset-full-reproduction/artifacts \
  --resume results/defense/training/persistent-reset-short-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
# For the matched comparator, use distinct output paths and resume
# results/defense/training/persistent-rate-high-01/checkpoint instead.
```

### Worker-role exploration allocation

The shared failure sequence persists across the archive and quantile trials.
Increasing exploration everywhere can also make the early course harder to
reach. An optional `--curriculum-boot-epsilon` now gives the existing reserved
boot-only workers a fixed post-warmup exploration rate; all other workers
retain the global epsilon schedule. Roles are fixed by worker index, not by
screens, scores, obstacle locations or a scripted route. All workers use the
same learned Q-network and share their own training replay and native-state
archive. Warmup remains independent uniform actions for every worker.

This borrows the actor-diversity idea from
[Ape-X / Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933),
which assigned different exploration rates to actors. It is **not a full
Ape-X implementation** or its published epsilon distribution: our workers
remain synchronous and this experiment combines two fixed worker roles with
the already-tested bounded persistent random exploration and own-state resets.
The paper does not establish that this particular combination will work here.

With persistent exploration, each worker's nominal exploratory-step fraction
is converted separately into a hold-start probability. Visible life/episode
boundaries still cancel holds; actual occupancy can differ from the nominal
rate and is now counted per worker. Setting a rate to zero does not interrupt
an already active hold; normal boundaries do. Evaluation never uses these
random holds or worker overrides. No new reward, policy input, demonstration,
hidden-state read or game-specific action selection is introduced.

The [split calibration configuration](results/defense/training/worker-epsilon-split-calibration-01/resume-config.json)
and [uniform control](results/defense/training/worker-epsilon-uniform-control-01/resume-config.json)
both resume own DQN 33 at **6,962,144**, with the same online/target/Adam/RNG
and **131,072** new actions each. The parent averaged **10,259**, best
**10,480**, all stage 1. Both use eight workers, compact replay 200,000,
batch 64, learning rate 1e-4, gamma .997, n-step 5, life terminals,
100,000 T-states/stride 1, duration maximum 64/exponent 1.5, shared score
archives 16×4, reset probability .5, two reserved boot workers and lookback
**128**. Replay and opaque native archives refill from new own experience;
saved evaluation traces are never training input.

- Split: two boot-only workers at **.05**, six other workers at **.9**.
- Control: all eight workers at **.6875**.

Both have the same nominal mean **.6875**. The comparison tests allocation,
not just adding more exploration; boundary cuts and resulting state visitation
may still change the realized mean. The [configuration comparison](results/defense/training/worker-epsilon-split-calibration-01/design.json)
checks that only this allocation and its metadata/output paths differ.
Both will use ten complete uncapped from-boot games on the reused validation
seeds 10000–10009. They are short controlled checks, excluded from the shared
best collector, not fresh success-rate estimates or a claimed stage clear.

The [before/after default-path check](results/defense/diagnostics/worker-epsilon-default-parity.json)
kept online/target files, all 26 Adam arrays, RNG states and existing exploration
counters exact over 64 native actions. It deliberately truncated tiny test
episodes and is not a performance result. Eleven focused tests cover scalar
versus vector equality, analytical/statistical occupancy, zero/one rates,
boundary cancellation, invalid CLI settings, both independent and persistent
native training, optimizer/RNG resume, warmup and unchanged greedy evaluation.
The full [297-test regression suite](results/defense/diagnostics/worker-epsilon-regression-tests.txt)
passed, as did the [11 focused tests](results/defense/diagnostics/worker-epsilon-focused-tests.txt).
Existing live learners retain their loaded implementation and settings.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-worker-epsilon-split-reproduction \
  --artifacts runs/defense-worker-epsilon-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128
# Matched control: distinct output paths, --epsilon-final .6875,
# omit --curriculum-boot-epsilon; all other arguments unchanged.
```

The slots come from two gracefully retired depth-plateaued learners, whose
full final online/target/Adam/RNG, complete compressed logs and validation
records are preserved:

- [DQN 36](results/defense/training/dqn-36-score-archive-control/retirement.json):
  stopped at **9,092,560**, after **1,999,344** new actions, **820** boot games,
  **482** restored segments and **nine** ten-game rounds, all stage 1. Peak
  mean **10,418**, best **10,460**; final evaluation mean **9,993**.
- [DQN 38](results/defense/training/dqn-38-quantile-risk/retirement.json):
  stopped at **1,346,528**, after **1,215,456** new actions, **530** boot games
  and **six** ten-game rounds, all stage 1. Peak mean **10,283**; final mean
  **9,800**. Its later best **10,350** has a preserved
  [2,475-action verified replay](results/defense/training/dqn-38-quantile-risk/replay-10350/replay.html).

Their final post-update checkpoints were not separately evaluated. Retirement
reflects depth plateaus, not a wall-clock limit. No artifacts were deleted.
The collector retains both historical sources; DQN 37 and DQN 39 continue.

DQN 37 also matched the global **10,480** at **931,072**, with mean **8,424**
and a [2,498-action verified replay](results/defense/training/dqn-37-quantile-neutral/replay-10480/replay.html).
Its [six-round curve](results/defense/training/dqn-37-quantile-neutral/curve-through-000001331072.json)
peaks at mean **10,408** at **1,131,072**, then **10,098** at **1,331,072**.
DQN 39's [first three rounds](results/defense/training/dqn-39-long-lookback/curve-through-000007362144.json)
average **9,882**, **9,994**, **10,301**; its new **10,340** best has a
[2,591-action verified replay](results/defense/training/dqn-39-long-lookback/replay-10340/replay.html).
These are score recoveries, not new depth. All remain stage 1; the shared
global best and completed Breakdown winning replay remain unchanged.

The [latest aligned loss panels](results/defense/diagnostics/shared-loss-worker-exploration-01/report.json)
compare those three newly preserved replays. Their per-life scores are
**[2620, 2620, 2620, 2620]**, **[2550, 2600, 2580, 2620]** and
**[2600, 2600, 2570, 2570]** respectively. Visual inspection of the neutral
and long-lookback sheets again shows the broad right-opening barrier while
the ship remains near the center/left. This supports the recurring-sequence
diagnosis, not an exact collision-coordinate or causal proof. White-flash
alignment is not a collision timestamp, and the final loss can omit the flash
between sampled screens. These are selected best replays, not representative
training rollouts; no diagnostic screen or action is supplied to training.

#### Worker-allocation calibration result and longer trial

Both checks exited normally at **7,093,216**. Their
[paired ten-game comparison](results/defense/training/worker-epsilon-split-calibration-01/comparison.json)
does **not** establish improvement: every game lost in stage 1, and both
means are below the parent's **10,259**.

| Allocation | Mean | Median | Best | Verified replay |
| --- | ---: | ---: | ---: | --- |
| Split .05/.9 | 9,466 | 10,310 | 10,410 | [2,440 actions](results/defense/training/worker-epsilon-split-calibration-01/replay/replay.html) |
| Uniform .6875 | 9,694 | 10,160 | 10,240 | [2,528 actions](results/defense/training/worker-epsilon-uniform-control-01/replay/replay.html) |

Split minus control averages **−228**; seven paired scores are higher and
three lower, including differences **−5,530** and **−2,310**. This is neither
a reliable benefit nor a fresh-seed success-rate estimate. Full online/target,
Adam/RNG, configs, logs and verified replay bundles are preserved for both.

The [split audit](results/defense/training/worker-epsilon-split-calibration-01/audit.json)
records **53** boot games and **42** restored segments, **1,251** archive
events and **122** retained save events. The
[control audit](results/defense/training/worker-epsilon-uniform-control-01/audit.json)
records **65**, **33**, **1,034** and **50** respectively. All source/trigger
offsets equal 128 and reserved workers remain boot-only. Realized exploration
is approximately **6.1%/3.7%** on the two split boot workers and **89–90%**
on the others; control workers are approximately **67–69%**. Boundary cuts
and finite sampling explain why these are not exactly the nominal rates.

Retained source within-life scores reach **190** versus **100**, with trigger
scores **2,620** versus **180**. This shows a difference in states supplying
the archive, not a verified route, collision location or new stage. Retained
events are not distinct current-state counts. All logged training episodes
also remain stage 1. The split design was exercised, but the passage goal
remains unmet.

Full [DQN 40 split](results/defense/training/dqn-40-worker-epsilon-split/resume-config.json)
and [DQN 41 control](results/defense/training/dqn-41-worker-epsilon-control/resume-config.json)
now continue their respective complete calibration states. Only output paths,
the limit to **unlimited**, and the full evaluation interval **200,000** change;
first full evaluation is due at **7,293,216**. Replay/native archives refill
from new own experience. This extends the matched trial's exposure, rather
than declaring the split arm superior based on its best score or reset count.
DQN 37 and 39 continue in the other two slots.

The old sole collector exited cleanly before its replacement started with
[all 37 full-run sources](results/defense/training/dqn-40-worker-epsilon-split/collector-config.json).
Historical sources remain included; short calibrations and evaluation-only
probes remain excluded. The shared 10,480-point best is unchanged, and future
promotion still requires complete frozen-policy replay verification.

At the matched **100,000-new-action** cutoff (**7,193,216**), the
[split startup audit](results/defense/training/dqn-40-worker-epsilon-split/startup-audit-000007193216.json)
records **42** completed boot games and **26** restored segments; **21/26**
completed restores originated from the two reserved low-exploration workers.
Those workers supplied **82/110** retained save events. The
[uniform control audit](results/defense/training/dqn-41-worker-epsilon-control/startup-audit-000007193216.json)
records **47** boot games and **34** restored segments; **8/34** came from
reserved workers, which supplied **11/59** retained events. Both preserve
boot-only reservations and exact 128-action archive offsets, and all logged
episodes remain stage 1. The role separation is producing a different supply
of own practice states, but this is mechanism evidence, not a performance or
stage-clear result. Each audit records the exact log-prefix hash and byte count;
its exploration counters come from the last progress row before the cutoff.

The [first full paired evaluation at **7,293,216**](results/defense/training/dqn-40-worker-epsilon-split/comparison-at-000007293216.json)
adds **200,000** actions per arm after calibration (**331,072** since their
common parent): split mean **9,755**, median **10,190**, best **10,340**;
control mean **9,832**, median **9,810**, best **9,940**. All twenty complete
games remain stage-1 losses. Split scores are higher on eight paired seeds,
lower on two (**−1,930**, **−2,020**), for mean difference **−77**. The higher
median/best does not establish a depth benefit or reliable superiority.

Both full online/target/Adam/RNG checkpoints and independently verified
[split replay, **2,581 actions**](results/defense/training/dqn-40-worker-epsilon-split/first-replay/replay.html)
and [control replay, **2,579 actions**](results/defense/training/dqn-41-worker-epsilon-control/first-replay/replay.html)
are preserved. The [split audit](results/defense/training/dqn-40-worker-epsilon-split/audit-at-000007293216.json)
records **82** new boot games and **61** restored segments, versus
[control's **99** and **61**](results/defense/training/dqn-41-worker-epsilon-control/audit-at-000007293216.json).
Retained save events number **137** versus **62**, maximum source within-life
score **190** versus **120**, and maximum trigger within-life score **2,620**
versus **2,450**. All offsets and reserved-worker boundaries remain correct;
all completed training episodes also remain stage 1. These are archive-event
statistics, not proof of course advancement. Both learners continue unchanged.

The [second-round comparison through **7,493,216**](results/defense/training/dqn-40-worker-epsilon-split/comparison-through-000007493216.json)
records split mean **8,851**, best **10,140**, versus control mean **9,165**,
best **9,860**. Both means regress from the first round, and all forty games
across the two paired rounds remain stage-1 losses. More exploratory practice
has not yet established a passage benefit. Frozen checkpoint hashes and all
per-game records are preserved with the comparison; the trials continue.

At **7,693,216**, the [third-round comparison](results/defense/training/dqn-40-worker-epsilon-split/comparison-through-000007693216.json)
shows a split-arm recovery: mean **10,299**, median **10,310**, best
**10,430**, versus control mean **9,236**, median **9,110**, best **9,840**.
Split scores are higher on all ten paired seeds, with mean difference
**+1,063**, after differences **−77** and **−314** in the first two rounds.
This single-round score advantage is not durable-superiority or stage-clear
evidence: all sixty games across the three rounds remain stage-1 losses.
The split arm's full online/target/Adam/RNG checkpoint and
[2,550-action verified replay](results/defense/training/dqn-40-worker-epsilon-split/replay-10430/replay.html)
are preserved separately; the global 10,480-point best is unchanged.

The worker-allocation [six-round comparison through **8,293,216**](results/defense/training/dqn-40-worker-epsilon-split/comparison-through-000008293216.json)
adds split/control means **7,310 / 4,013**, **9,784 / 6,022** and
**9,049 / 1,207**. The uniform control has substantially regressed; split
allocation better retains scoring in these later rounds, but all **120**
games across six paired rounds remain stage-1 losses. This is neither a new
stage nor an independent test success rate. Earlier stronger checkpoints and
verified replays remain preserved; live runs have not replaced the global best.

The uniform control (41) later stopped cleanly at **8,572,656**, after
**1,479,440** new actions, **738** complete boot games, **459** restored segments
and **seven** complete ten-game evaluation rounds. Its last mean recovered to
**7,228**, median **7,540**, best **8,300**, still below its calibration and
earlier peak. Every logged training episode and validation game stayed in
stage 1. The [retirement record](results/defense/training/dqn-41-worker-epsilon-control/retirement.json),
complete compressed log, last evaluated checkpoint and final full optimizer
checkpoint are preserved. The final post-update weights were not separately
evaluated. This retires the depth-plateaued control without deleting its stronger
first-round verified replay; the split arm continues. Compute is available to
the trace-cut calibration, not a claim that trace cutting has succeeded.

The split arm (40) subsequently stopped cleanly at **8,903,168**, after
**1,809,952** new actions, **765** complete boot games, **592** restored segments
and **nine** complete ten-game evaluation rounds. Its last mean was **10,256**,
median **10,240**, best **10,310**. Every logged training episode and all
90 validation games remained in stage 1. The
[retirement record](results/defense/training/dqn-40-worker-epsilon-split/retirement.json),
complete compressed log, last evaluated checkpoint and final full optimizer
checkpoint are preserved. Final post-update weights were not separately
evaluated; the earlier 10,430-point verified replay stays available. The depth
plateau, not a wall-clock limit, prompted reallocating compute to the duration
and return-target experiments.

### One-step target calibration under heavy exploration

The worker-allocation trial changes which experience is generated, but still
uses **uncorrected five-step returns**: `NStep` sums the next five actual
score rewards, then `Learner._loss` bootstraps with the target network's value
of the online network's greedy action. Intermediate exploratory actions remain
inside that return. This is the existing multi-step DQN design, not a newly
discovered implementation error. Under heavy exploration, however, a useful
first action can be followed by unrelated random actions, while the desired
evaluation policy follows learned greedy actions throughout.

The [off-policy return literature](https://arxiv.org/abs/1606.02647) motivates
checking that mismatch. It does **not** prove it caused this game's plateau.
The existing `--n-step 1` option instead bootstraps after one transition,
avoiding intermediate behavior actions in that particular target. It does not
remove function-approximation error, all off-policy risks, partial observability
or exploration difficulty. Multi-step targets can propagate newly observed
rewards faster, as discussed in [Rainbow](https://arxiv.org/abs/1710.02298), so
one-step is a tradeoff to test, not an assumed improvement. This experiment
is ordinary one-step Double DQN, **not Retrace** or an importance-correction
implementation.

The [one-step calibration](results/defense/training/one-step-split-calibration-01/resume-config.json)
resumes the same own DQN 33 checkpoint at **6,962,144** as the preserved
[five-step split calibration](results/defense/training/worker-epsilon-split-calibration-01/resume-config.json).
It collects **131,072** new actions and evaluates ten complete uncapped boot
games on reused seeds 10000–10009. The [configuration check](results/defense/training/one-step-split-calibration-01/design.json)
asserts that only **n-step 5 → 1** and output paths differ; production source
hashes match. This is a historical same-parent/configuration comparator, not
a new simultaneous or independent replication.

Both retain eight workers, compact replay 200,000, batch 64, learning rate
1e-4, gamma .997, reward scale .01, life terminals, 100,000 T-states/stride 1,
durations 1–64/exponent 1.5, two boot-only workers at epsilon .05 and six at
.9, own shared score archives 16×4, reset probability .5 and lookback 128.
Full online/target/Adam/RNG are restored; replay and native archives refill
from new own experience. No replay demonstration, oracle or extra reward is
introduced. The shorter queue fills replay sooner, so equal action counts
can include a slightly different number of optimizer updates; this will be
reported rather than silently described as an exactly update-matched trial.

Three additional regression tests cover immediate one-step insertion,
terminal/truncation discounts and final observations; an analytical synthetic
one-state MDP illustrating the return difference; and native persistent
worker-role training with exact online/target/Adam/RNG resume. The synthetic
fixture is test-only, not game data or a demonstration. The tiny native test
deliberately truncates episodes and is not a performance result. No production
code or existing live-run setting changes for this calibration.
All [300 regression tests](results/defense/diagnostics/one-step-regression-tests.txt)
and [nine focused Defense DQN tests](results/defense/diagnostics/one-step-focused-tests.txt)
passed.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-one-step-split-reproduction \
  --artifacts runs/defense-one-step-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 --n-step 1
```

The slot comes from [retired DQN 37](results/defense/training/dqn-37-quantile-neutral/retirement.json),
which stopped cleanly at **1,952,224** after **1,821,152** new actions,
**759** boot games and **nine** ten-game rounds, all stage-1 losses. Its
peak mean remains **10,408**, best **10,480**; the last evaluation mean is
**9,979**, median **10,240**, best **10,290**. Full final online/target/Adam/RNG,
last evaluated checkpoint, complete compressed log and all evaluation records
are preserved. The final post-update weights were not separately evaluated.
Its best replay remains available; no artifact was deleted. Retirement is
for the depth plateau, not a wall-clock budget. Runs 39/40/41 continue, and
the short one-step check is excluded from the shared collector.

#### One-step result: regression, not continued

The [one-step comparison](results/defense/training/one-step-split-calibration-01/comparison.json)
finished normally at **7,093,216** with mean **2,383**, median **540**, best
**7,500**, all ten stage-1 losses. Against the five-step control's **9,466**
mean, this is **−7,083**; nine paired scores are lower and one is higher.
The configuration is **not extended into a full run**. This rejects the tested
short conversion as an improvement, not every possible one-step learner or
the general concern about off-policy multi-step returns.

Its [2,302-action verified replay](results/defense/training/one-step-split-calibration-01/replay/replay.html),
full online/target/Adam/RNG and compressed training log are preserved. The
[audit](results/defense/training/one-step-split-calibration-01/audit.json)
records **59** boot games, **33** restored segments, exact 128-action offsets
and protected boot-only workers; no training episode reached a later stage.
It performed **7,568** new updates, versus control's **7,566**, because of
the shorter replay insertion lag. Both received exactly **131,072** new actions.

Read-only Q checks reconstruct every action on each model's own verified
selected replay: [one-step, **2,302 actions**](results/defense/diagnostics/one-step-split-q-calibration-7093216.json)
and [five-step, **2,440 actions**](results/defense/diagnostics/five-step-split-q-calibration-7093216.json).
Whole-replay mean predicted minus realized discounted score is **+68.08**
versus **−100.66**, with mean absolute errors **248.68** versus **186.90**.
These are different selected trajectories, not matched-state counterfactuals
or estimates of optimal returns; they do not establish the cause of the
performance regression. Sources remain unchanged, and neither diagnostic
provides data or parameter updates to training.

### Longer-discount calibration with five-step targets

The next [isolated calibration](results/defense/training/long-discount-split-calibration-01/resume-config.json)
returns to the five-step target and changes only **gamma .997 → .9995**
relative to the original worker-split calibration. Its
[configuration comparison](results/defense/training/long-discount-split-calibration-01/design.json)
asserts that only gamma and output paths differ, including identical production
source hashes. It starts from the same own DQN 33 checkpoint at **6,962,144**,
not from the regressed one-step weights, restores full online/target/Adam/RNG,
and collects **131,072** new actions before ten complete uncapped boot games.
Replay and native archives refill from own new experience.

This tests whether stronger weighting of later **actual score** helps the
current learned policy escape the early-reward routine. The reward itself,
screen input, action set, timing, persistent exploration split, own resets,
learning rate and target-copy interval are unchanged. The half-weight delay
increases from approximately **231 to 1,386 actions**; a reward 512 actions
later has weight approximately **.215 versus .774**. Visible ship-loss
learning terminals still stop credit across lives. Higher gamma cannot supply
an undiscovered reward, prove a new route, or guarantee better exploration.

The earlier long-horizon PPO trial also failed to advance. This is not a new
claim that longer horizons solve the game; it is a controlled parameter check
in the present value-learning, high-exploration setting. The same-parent
five-step calibration is a historical comparator, not a fresh simultaneous
replication, and seeds 10000–10009 remain reused validation seeds.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-long-discount-split-reproduction \
  --artifacts runs/defense-long-discount-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 --gamma .9995
```

The check uses the now-finished one-step slot; runs 39/40/41 and the sole
37-source collector continue unchanged. Both short calibrations are excluded
from global promotion. The 10,480-point stage-1 best remains verified and
available; the mission goal is not achieved.

#### Longer-discount result and full continuation

The [completed calibration comparison](results/defense/training/long-discount-split-calibration-01/comparison.json)
at **7,093,216** gives gamma .9995 mean **7,283**, median **7,240**, best
**10,320**, versus gamma .997 mean **9,466**, median **10,310**, best
**10,410**. Eight paired scores are lower and two higher; mean difference is
**−2,183**. All twenty compared games remain stage-1 losses. This is a short
score regression, not evidence that longer discounting improves passage.

Its full online/target/Adam/RNG, complete compressed log and
[2,398-action verified replay](results/defense/training/long-discount-split-calibration-01/replay/replay.html)
are preserved. The [audit](results/defense/training/long-discount-split-calibration-01/audit.json)
records **60** boot games, **43** restored segments, **1,271** archive events
and **124** retained save events, with exact 128-action offsets and protected
boot-only workers. Retained source within-life score reaches **190**, trigger
score **2,620**; these are not position measurements or stage-clear evidence.
All logged training episodes remain stage 1. Both discount calibrations
performed **7,566** updates over **131,072** new actions.

Full [DQN 42](results/defense/training/dqn-42-long-discount/resume-config.json)
now continues the higher-discount calibration unchanged, apart from output
paths, **unlimited** training and evaluation every **200,000** actions. Its
first full round is due at **7,293,216**. This longer trial asks whether the
changed objective recovers from retraining and yields actual stage progression;
the retained best-game competence warrants testing more exposure, not claiming
a benefit from the regressed mean. Unlike the one-step check's median **540**,
this check retained median **7,240**, but neither demonstrates later-stage play.
No evaluation trace becomes training input; replay/native archives refill.

Runs 39/40/41 continue. The old collector exited cleanly before the replacement
started with [all 38 full-run sources](results/defense/training/dqn-42-long-discount/collector-config.json).
Retired sources remain included and short calibrations remain excluded.
Promotion still requires frozen-policy native replay verification. No stage-2
reach or original mission completion has yet been observed.

The [new aligned screen comparison](results/defense/diagnostics/shared-loss-long-discount-01/report.json)
preserves the split run's 10,430-point replay and the longer-discount check's
10,320-point replay. Per-life scores are **[2620, 2620, 2620, 2570]** and
**[2600, 2570, 2550, 2600]**. They show the familiar broad obstacle sequence,
but not identical failure timing: the longer-discount panels include the
preceding center-opening barrier near the ship when the right-opening barrier
is still farther up the screen. The split replay reaches the latter barrier
near the ship. This is consistent with some earlier failures in the changed
policy, not a verified new passage. Neither alignment nor score establishes
the exact collision object or instant; projectile/wall causality is not
inferred from these panels. Selected replays are not representative training
rollouts, and no diagnostic frames/actions are supplied to learning.

Full higher-discount run 42's [first three rounds](results/defense/training/dqn-42-long-discount/comparison-through-000007693216.json)
average **9,927**, **10,230** and **10,262**, recovering the calibration's
score drop but still losing every game in stage 1. Differences from the
historical same-count normal-discount split arm are **+172**, **+1,379**
and **−37**. First-round median is **10,410**, best **10,460**; its
[2,584-action verified replay](results/defense/training/dqn-42-long-discount/first-replay/replay.html)
and full online/target/Adam/RNG checkpoint are preserved. Score recovery alone
does not show improved passage.

The [aligned full-run loss panels](results/defense/diagnostics/shared-loss-full-long-discount-01/report.json)
compare this replay with the split arm's 10,430-point effort. Higher-discount
life scores are **[2600, 2620, 2620, 2620]**. Unlike its earlier calibration,
these panels again show the broad right-opening barrier approaching a ship
remaining left of the gap. That supports recurrence of the familiar sequence,
not proof of the exact collision object or timestamp. The final loss has no
sampled white flash. No diagnostic trajectory becomes training data.

The higher-discount run (42) subsequently stopped cleanly at **8,517,856**,
after **1,424,640** new actions, **620** complete boot games, **468** restored
segments and **seven** full ten-game validation rounds. Its last mean was
**9,849**, median **9,845**, best **9,910**. Every logged training episode and
all 70 validation games remained in stage 1. The
[retirement record](results/defense/training/dqn-42-long-discount/retirement.json),
complete compressed log, last evaluated checkpoint, peak-mean checkpoint
(**10,262** at **7,693,216**) and final full optimizer state are preserved.
Final post-update weights were not separately evaluated. Its earlier 10,460-point
verified replay remains available. The stage-depth plateau, not a wall-clock
limit, prompted freeing compute for the next representation-learning candidate.

### Longer preparation-context continuation

Full [DQN 39](results/defense/training/dqn-39-long-lookback/resume-config.json)
continues the original **lookback-128** persistent-exploration calibration at
**6,762,144**. That check received only **131,072** new actions and **39**
completed restored segments, versus the shorter-lookback run 33's later
**1,371,392** additional actions and **362** restored segments. Run 33 never
cleared stage 1. The earlier 186-point short-check advantage for lookback 32
was not evidence that the longer preparation context cannot help stage depth.
Lower source score is not proof of less useful preparation: score is not
position, and the goal is passage rather than maximizing a short-run mean.

This is a continuation of an already checked experiment, not a new controller
or a claimed solution. It restores the long arm's online/target/Adam and both
RNG streams, with nominal 25% persistent exploration, durations 1–64/exponent
1.5, eight workers, compact replay 200,000, batch 64, learning rate 1e-4,
gamma .997, n-step 5, life terminals, 100,000 T-states and stride 1. Own resets
remain probability .5, shared score cells 16×4, two boot-only workers and
**128 actions before an archive event**, not a scripted collision timestamp.
All learning/exploration/archive settings match the preserved parent.
Replay and native-state archives refill from new own experience. No archived
evaluation trace or foreign native state is loaded.

The [startup log-prefix audit through **6,874,360**](results/defense/training/dqn-39-long-lookback/startup-audit.json)
observes **43** completed boot games, **32** restored segments and **1,314**
archive events, all with exact 128-action source/trigger offsets. Reserved
workers remain boot-only. All completed episodes are still stage 1. Retained
save events reach source within-life score **210**, while triggers reach
**2,640**; these are not position, collision or stage-clear measurements.

Only output paths, the limit to unlimited, and the full evaluation interval
of **200,000** change. Its first full evaluation is due at **6,962,144**.
The historical shorter-lookback/no-reset arms 33/34 provide same-lineage
comparators at matching additional action counts, not fresh independent or
contemporaneous replications. Later progression must still be observed in
complete from-boot play, with a verified replay; earlier static/visual evidence
does not establish success.

Its [first full round at **6,962,144**](results/defense/training/dqn-39-long-lookback/comparison-at-000006962144.json)
is now complete: mean **9,882**, median **9,870**, best **10,040**, all ten
stage-1 losses. Its [2,542-action replay](results/defense/training/dqn-39-long-lookback/first-replay/replay.html)
was independently verified, and the full online/target/Adam/RNG checkpoint is
preserved. Relative to the historical same-count first rounds, its mean is
**377 below** shorter-lookback run 33 and **315 below** no-reset run 34.
This first batch does not show a benefit. The longer trial continues unchanged
to address the previously much smaller exposure given to this lookback.

It completed **78** new boot games and **54** restored segments since its own
calibration, without a logged later-stage episode. All **2,448** archive events
had exact 128-action offsets; reserved workers remained boot-only. **168**
retained save events reached source within-life score **210**, while triggers
reached **2,640**. These are event counts, not distinct retained-state counts,
and the scores do not establish route or collision location.

Run 35's retired slot supplies compute. Runs 36/37/38 continue unchanged.
The prior collector exited cleanly before the new sole collector started
with [all 35 full-run sources](results/defense/training/dqn-39-long-lookback/collector-config.json).
Historical sources remain; calibration and evaluation-only probes remain
excluded. Shared best promotion still requires frozen-policy verification.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-persistent-long-lookback-full-reproduction \
  --artifacts runs/defense-persistent-long-lookback-full-reproduction/artifacts \
  --resume results/defense/training/persistent-reset-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

At **7,562,144**, the longer-lookback run's
[four-round curve](results/defense/training/dqn-39-long-lookback/curve-through-000007562144.json)
reaches mean **10,345**, median **10,330**, best **10,460**. Its
[2,562-action verified replay](results/defense/training/dqn-39-long-lookback/replay-10460/replay.html)
and complete online/target/Adam/RNG checkpoint are preserved. This improves
its own score record, but all forty validation games remain stage-1 losses;
no mission or later-stage progress has been observed. The run continues
unchanged, and the global best is not replaced by this lower score.

Run 39 subsequently stopped cleanly at **8,833,696**, after **2,071,552**
new actions, **813** complete boot games, **580** restored segments and
**ten** full ten-game validation rounds since its calibration. Every logged
training episode and validation game remained in stage 1. Its last evaluated
mean was **10,054**, median **10,050**, best **10,160**. The
[retirement record](results/defense/training/dqn-39-long-lookback/retirement.json),
complete compressed log, last evaluated checkpoint and final full optimizer
checkpoint are preserved. Final post-update weights were not separately
evaluated. Compute was reassigned because of the depth plateau, not a
wall-clock limit; its stronger earlier verified replay remains available.

### Current-policy trace-cutting calibration

The repeated obstacle sequence remains unresolved. One possible contributor
is learning multi-step targets from highly exploratory continuations: a random
hold after a useful action can change the return attributed to that action.
The failed one-step conversion does not establish the cause, and discarding
all longer returns also discards faster credit propagation. This experiment
tests retaining longer returns only through actions matching the current
learned greedy policy.

Optional `--greedy-trace-cut` uses at most `--n-step` own transitions. The root
action may be exploratory. At each later visible state, the **current online
network** recomputes its deterministic argmax; a different recorded next action
ends the return *before* that action's reward. The target network supplies
the bootstrap value of the online-selected action. Greedy continuations retain
their actual rewards, up to the horizon. Life/episode learning terminals stop
bootstrap; truncations and partial paths bootstrap from the actual final screen,
never the next reset. No collection-time greedy label is trusted after learning.

This is a finite-horizon, lagged Double-Q adaptation of Watkins-style cutting.
For a deterministic target policy, the clipped coefficient in
[Munos et al.'s Retrace formulation](https://arxiv.org/abs/1606.02647) becomes
one for its selected action and zero otherwise when lambda is one. That
motivates the cut without estimating probabilities for history-dependent
random holds. This implementation is **not** general stochastic-target Retrace
and does not inherit a tabular convergence guarantee for deep replay learning.

The optional trajectory replay shares exact visible frames with the compact
frame pool and retains per-step actions/rewards/terminal discounts. It preserves
the priority sampler and per-worker life/episode queues. Ring overwrites release
all frame references, including padding. Future recorded screens are used
**only for training targets**, never acting inputs. The acting model, ordinary
greedy boot evaluation, reward, native reset rules and action set are unchanged.
The option requires compact scalar DQN and a horizon of 1–32; it cannot combine
with quantile or bootstrap heads. With the flag off, the original path remains.

All [309 regression tests](results/defense/diagnostics/trace-cut-regression-tests.txt)
and [nine focused tests](results/defense/diagnostics/trace-cut-focused-tests.txt)
passed. Coverage includes target arithmetic, current-network changes, terminals,
truncation, deterministic ties, gradient isolation, replay priorities/RNG,
ring refcounts, independent workers, real MLX updates, exact resume, ordinary
evaluation loading, and actual native own-state resets. Tiny native checks are
not performance results or training parents. The
[64-action before/after default-path check](results/defense/diagnostics/trace-cut-default-parity.json)
preserves both networks, all 26 Adam arrays, RNGs and prior counters exactly.
The [zero-update real-parent conversion](results/defense/diagnostics/trace-cut-parent-conversion-parity.json)
also preserves both networks, Adam arrays, both RNGs and global counters.
Local exploration/trace counters clear on resume; replay and archives refill.

The [isolated calibration](results/defense/training/trace-cut-split-calibration-01/resume-config.json)
starts from original DQN 33 at **6,962,144**, not a smoke/conversion checkpoint.
It collects **131,072** new actions before ten uncapped boot games. Relative to
the historical worker-split five-step calibration, the
[configuration check](results/defense/training/trace-cut-split-calibration-01/design.json)
allows only the new trace option/metadata, trainer hash and output paths.
It retains gamma .997, n-step 5, two boot workers at epsilon .05, six others
at .9, persistent random holds 1–64, lookback 128 and all optimizer settings.
Reported backup-length histograms and cut fractions check whether the mechanism
is active; they are not passage evidence. Seeds 10000–10009 are reused validation
seeds, not fresh tests, and this historical comparison is not an independent
replication. No evaluation trace enters training. The calibration is excluded
from the shared best collector; runs 40/41/42 continue unchanged.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-trace-cut-split-reproduction \
  --artifacts runs/defense-trace-cut-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 --greedy-trace-cut
```

#### Trace-cut calibration result and full continuation

The [completed ten-game comparison](results/defense/training/trace-cut-split-calibration-01/comparison.json)
at **7,093,216** gives trace cutting mean **8,324**, median **8,320**, best
**10,160**, versus baseline mean **9,466**, median **10,310**, best **10,410**.
Seven paired scores are lower and three higher; mean difference is **−1,142**.
Every game remains a stage-1 loss. This is a short score regression, not a
verified benefit. The full online/target/Adam/RNG checkpoint, complete compressed
log and [2,449-action verified replay](results/defense/training/trace-cut-split-calibration-01/replay/replay.html)
are preserved. The check exited normally and remains excluded from the collector.

Its [audit](results/defense/training/trace-cut-split-calibration-01/audit.json)
records **59** new boot games, **37** completed restored segments, **1,203**
archive events and **103** retained save events, all with exact 128-action
offsets and protected boot-only workers. Retained source within-life score
reaches **170**, trigger score **2,620**; these do not measure obstacle position.
All logged training episodes are stage 1. Both calibration arms perform
**7,566** new updates. The trace mechanism processes **484,224** sampled
backups, with lengths 1–5 counted **[395728, 30358, 13975, 8612, 35551]**.
Mean length is **1.4674** and **91.787%** end at a nongreedy later action.
Thus most targets shorten, while some retain longer credit propagation; these
mechanism counts are not performance or passage evidence.

The [warmup parity check](results/defense/training/trace-cut-split-calibration-01/warmup-parity.json)
also finds all **56** logged archive events identical to the historical
baseline before the first learning update, excluding wall-time fields. It
does not compare every unlogged screen/action and supplies no training data.
The [loss panels](results/defense/diagnostics/shared-loss-trace-cut-calibration-01/report.json)
show life scores **[2500, 2580, 2480, 2600]**, with some center-opening failures
earlier than the familiar right-opening barrier. They do not establish exact
collision timing or an identical cause across lives.

The [frozen prediction/return check](results/defense/diagnostics/trace-cut-split-q-calibration-7093216.json)
reconstructs all **2,449** actions exactly, with mean prediction **1,005.73**
versus realized discounted score **1,103.88**, mean error **−98.15**, absolute
error **188.39**. The earlier baseline's selected replay had mean error
**−100.66**, absolute error **186.90**. These are different selected trajectories,
not matched-state estimates or causal evidence of improvement. The diagnostic
now explicitly reports zero post-flash actions when the first flash is on the
terminal screen; no fictitious sample is inserted, and learning is unchanged.
All [six focused diagnostic tests](results/defense/diagnostics/trace-cut-q-probe-tests.txt)
pass, including that exact replay edge case and preserved scalar/quantile
action reconstruction and source immutability.

Full [DQN 43](results/defense/training/dqn-43-greedy-trace-cut/resume-config.json)
continues the calibrated optimizer unchanged, with unlimited training and
evaluation every **200,000** actions, first at **7,293,216**. Its median retains
more play competence than the earlier one-step check's **540**, which warrants
testing longer adaptation but does not establish a trace-cut advantage. Replay
and native archives refill from new own experience; no calibration evaluation
trajectory is reused. Runs 40/42 continue unchanged. The old collector exited
before the sole replacement started with
[all 39 full-run sources](results/defense/training/dqn-43-greedy-trace-cut/collector-config.json),
including retired sources and excluding short calibrations. Independent
frozen-policy verification still gates shared best promotion. The verified
10,480-point stage-1 best remains unchanged; no mission has been observed.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-greedy-trace-cut-full-reproduction \
  --artifacts runs/defense-greedy-trace-cut-full-reproduction/artifacts \
  --resume results/defense/training/trace-cut-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

Run 43's [first full comparison at **7,293,216**](results/defense/training/dqn-43-greedy-trace-cut/comparison-at-000007293216.json)
gives mean **8,681**, median **8,300**, best **9,820**, versus the historical
five-step split arm's mean **9,755**. The mean difference is **−1,074**, with
nine paired scores lower and one higher. All games remain stage-1 losses.
The full checkpoint and [2,530-action verified replay](results/defense/training/dqn-43-greedy-trace-cut/first-replay/replay.html)
are preserved. The [audit](results/defense/training/dqn-43-greedy-trace-cut/audit-at-000007293216.json)
records **88** new boot games, **65** restored segments and **1,723** archive
events with correct lookback and protected boot workers. All logged training
episodes remain stage 1. Mean sampled backup length is **1.4205**, with
**92.724%** ending at a nongreedy later action. This first continuation batch
does not demonstrate a benefit; the original shared best is unchanged.

By **7,693,216**, run 43's [three-round comparison](results/defense/training/dqn-43-greedy-trace-cut/comparison-through-000007693216.json)
has means **8,681 / 8,227 / 9,415**, versus baseline **9,755 / 8,851 / 10,299**.
Every paired round remains lower in mean, and every game remains stage 1.
The third checkpoint's median is **10,020**, best **10,180**; its full optimizer
and [2,484-action verified replay](results/defense/training/dqn-43-greedy-trace-cut/replay-10180/replay.html)
are preserved. Recovery from the earlier regression is not a demonstrated
trace-cut advantage or stage passage.

Run 43 then [retired gracefully](results/defense/training/dqn-43-greedy-trace-cut/retirement.json)
at **8,308,744**, after **1,215,528** new actions, **541** boot games and **392**
restored segments since its calibration. All logged episodes and all **60**
complete evaluation games remain stage 1. Its
[six-round means](results/defense/training/dqn-43-greedy-trace-cut/comparison-through-000008293216.json)
are **8,681 / 8,227 / 9,415 / 8,089 / 4,828 / 9,074**. Differences from the
historical baseline are **−1,074 / −624 / −884 / +779 / −4,956 / +25**; later
rank reversals did not produce stage passage. Final full state and the latest
evaluated checkpoint at **8,293,216**, complete log and new
[2,575-action verified 10,280-point replay](results/defense/training/dqn-43-greedy-trace-cut/replay-10280/replay.html)
are preserved. Final weights were not separately evaluated. This was an
outcome-based retirement, not a wall-clock limit; historical collector sources
and all earlier replays remain intact.

### Longer random-persistence calibration

The [new isolated duration check](results/defense/training/long-persistence-split-calibration-01/resume-config.json)
changes the random-hold cap **64 → 256** relative to the historical worker-split
calibration. It uses the same original DQN 33 parent at **6,962,144**, not
trace-cut or high-discount weights. It collects **131,072** new actions, then
plays ten complete uncapped boot games on reused seeds 10000–10009.
This asks whether a broader duration tail discovers useful own experience
beyond the repeated barrier. It is not a scripted escape, preferred direction,
obstacle-triggered start, reward bonus or demonstration.

The method remains the existing bounded adaptation of
[temporally extended epsilon-greedy exploration](https://arxiv.org/abs/2006.01782):
uniformly sample one of all 20 actions and a power-law duration with exponent
1.5. Only **4.991%** of planned holds exceed 64 decisions, while expected
uninterrupted duration changes **6.17855 → 12.28982**. The start-probability
calibration retains the same nominal exploratory-step fractions: boot workers
.05, other workers .9. Their idle-start probabilities become **.00426427**
and **.42273732**, respectively. Life/episode cuts can still alter realized
occupancy, so actual duration/action/cancellation and worker counts are measured.

Every held primitive step still has its own screen, score reward and normal
learning opportunity. Life loss/episode boundaries cancel outstanding holds;
greedy boot evaluation has no random persistence. Gamma .997, n-step 5,
lookback 128, own-reset rules, optimizer, action cadence and observations all
remain unchanged. Trace cutting is **off**: this check does not combine two
new learning changes. The [configuration audit](results/defense/training/long-persistence-split-calibration-01/design.json)
records the cap/derived mean, paths and newer trainer hash/explicit false trace
flag; helper/model/replay/environment hashes remain identical to baseline.
The preserved default-path parity check covers the intervening trainer branch.

All [12 focused exploration tests](results/defense/diagnostics/long-persistence-focused-tests.txt)
pass, now including native parity at cap 256, occupancy for both worker rates,
observed long-tail samples and cancellation of maximum-length holds. These
sampler/native checks are not game-performance results or training parents.
The [complete 311-test suite](results/defense/diagnostics/long-persistence-regression-tests.txt)
also passes, including the terminal-screen Q-diagnostic regression.
Existing full trials 42/43 continue unchanged; the short check remains excluded
from shared promotion. This is a historical same-parent comparison, not an
independent replication or fresh success-rate estimate. Actual stage progression
and ultimately the original mission ending must still be observed and verified.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-long-persistence-split-reproduction \
  --artifacts runs/defense-long-persistence-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 \
  --exploration-max-repeat 256
```

#### Longer-persistence result and full continuation

The [completed comparison](results/defense/training/long-persistence-split-calibration-01/comparison.json)
gives cap 256 mean **10,097**, median **10,370**, best **10,430**, versus cap 64
mean **9,466**, median **10,310**, best **10,410**. Six paired scores are higher
and four lower; the **+631** mean difference includes a **+5,690** outlier and
a **−2,620** regression. It is a small, uneven validation improvement, not
evidence of reliable progress: all twenty compared games lose in stage 1.
Its [2,561-action verified replay](results/defense/training/long-persistence-split-calibration-01/replay/replay.html),
full online/target/Adam/RNG state and complete compressed log are preserved.
The bounded check exited normally; both arms performed **7,566** new updates.

The [mechanism audit](results/defense/training/long-persistence-split-calibration-01/audit.json)
records **57** new boot games, **46** completed restored segments, **1,246**
archive events and **111** retained save events. All offsets are exactly 128
actions and reserved workers remain boot-only; every logged training episode
is stage 1. Retained source within-life score reaches **190**, trigger **2,640**;
neither establishes a new passage. There are **376** planned holds above 64,
including **154** above 128, with maximum **256**. These are sampled durations,
not a claim that boundary-truncated holds all executed fully. Actual exploratory
fractions are **2.974% / 3.562%** for the two boot workers and **87.654–89.537%**
for the others. Nominal rates match baseline, but the realized rates do not
exactly match because trajectories, cancellations and finite samples differ.

Full [DQN 44](results/defense/training/dqn-44-long-persistence/resume-config.json)
continues this calibration unchanged, with unlimited actions and ten-game
evaluations every **200,000**, first at **7,293,216**. Replay and native archives
refill with new own experience; calibration replay actions are not training data.
The short comparison warrants testing the duration setting longer, not claiming
a solved barrier. Runs 42/43 remain unchanged. The prior collector exited
cleanly before its sole replacement started with
[all 40 full-run sources](results/defense/training/dqn-44-long-persistence/collector-config.json),
including retired sources but no short calibrations. Independent frozen-policy
native replay verification remains required before shared best promotion.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-long-persistence-full-reproduction \
  --artifacts runs/defense-long-persistence-full-reproduction/artifacts \
  --resume results/defense/training/long-persistence-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

Run 44's [first full comparison at **7,293,216**](results/defense/training/dqn-44-long-persistence/comparison-at-000007293216.json)
regressed to mean **6,404**, median **6,745**, best **7,810**. All ten paired
scores are below the historical cap-64 arm (mean **9,755**); mean difference is
**−3,351**. Every game remains a stage-1 loss. Its short-calibration advantage
did not persist into this first continuation checkpoint. The full optimizer
and [2,334-action verified replay](results/defense/training/dqn-44-long-persistence/first-replay/replay.html)
are preserved separately, without replacing the stronger shared best.
The [audit](results/defense/training/dqn-44-long-persistence/audit-at-000007293216.json)
records **88** new boot games, **60** restored segments and **1,912** archive
events with correct offsets and protected boot-only workers. All logged training
episodes remain stage 1. It sampled **586** planned holds above 64 decisions,
including **246** above 128; mechanism activity is not successful passage.
The trial continues unchanged for further complete-game measurements.

The duration arm's [three-round comparison through **7,693,216**](results/defense/training/dqn-44-long-persistence/comparison-through-000007693216.json)
has means **6,404 / 9,904 / 10,248**, recovering from its first full regression.
Differences from the cap-64 baseline are **−3,351 / +1,053 / −51**, so the mean
ranking varies and does not establish a sustained advantage. All games still
lose in stage 1. Its third median is **10,250**, best **10,260**; the full
checkpoint and [2,541-action verified replay](results/defense/training/dqn-44-long-persistence/replay-10260/replay.html)
are preserved. The 10,480-point shared best remains unchanged.

Run 44 subsequently [retired gracefully](results/defense/training/dqn-44-long-persistence/retirement.json)
at **8,335,312**: **1,242,096** new actions, **525** boot games and **448** restored
segments since its calibration. All logged training episodes and all **60**
complete evaluation games remained stage 1. The full final online/target/Adam
state, full latest evaluated state at **8,293,216**, and complete compressed log
are preserved; final weights were not separately evaluated. Its prior verified
10,260-point replay and third-round checkpoint remain intact.
The [six-round comparison](results/defense/training/dqn-44-long-persistence/comparison-through-000008293216.json)
has means **6,404 / 9,904 / 10,248 / 9,384 / 9,900 / 10,123**. Differences from
the cap-64 historical arm are **−3,351 / +1,053 / −51 / +2,074 / +116 / +1,074**;
later scoring gains never produced passage. Compute was reallocated to the
visual-prediction continuation based on this stage-depth plateau, not a wall-clock
limit. The collector retains 44's immutable best as a historical source.

### Auxiliary visual prediction

Optional `--spr-weight .1` tests an auxiliary visual-representation objective
based on [Self-Predictive Representations](https://arxiv.org/abs/2007.05929).
It predicts the learner's own future visual representations through a learned
action-conditioned model. This tests representation learning after repeated
stage-depth plateaus, not an established diagnosis of the obstacle failures.
The [initial design and acceptance criteria](results/defense/diagnostics/visual-prediction-design.md)
are preserved. No performance benefit or stage clear is yet established.

The acting model remains the existing QNetwork. Only the auxiliary training
path uses its 6×16×32 spatial features, per-example min/max normalization,
explicit-key dropout .5 after each convolution, two action-conditioned
32-channel convolutions, a location-wise LayerNorm, the shared 256-unit hidden
projection, and a separate 256-unit linear predictor. Targets use a separate
EMA encoder/projection with coefficient **.99** and stopped gradients.
This is a scalar-DQN adaptation: different encoder dimensions, LayerNorm instead
of batch normalization, mean valid-step rather than summed loss, and coefficient
**.1**. It is not a reproduction of the paper's Rainbow/Atari results.

The auxiliary loss averages cosine distance over actual valid transitions in
each sampled path, then applies existing PER sample weights. It is added to the
unchanged scalar Double-Q loss; **only TD errors set replay priorities**.
The same stored float32 n-step return and discount are used, not a recomputed
target with different arithmetic. Actual terminal screens are allowed; padding
is masked, and paths never cross a life/episode reset. Location-wise normalization
prevents padded examples from altering another example's normalization statistics.
Trace cutting is off. The option requires compact scalar replay and n-step 1–5;
quantile, bootstrap and trace-cut combinations are explicitly rejected.

This does **not** add an intrinsic reward, route label, scripted action,
demonstration or simulator lookahead. Rewards remain visible score delta times
the same constant. Future stored frames are auxiliary targets only; acting and
evaluation use no dropout, predictor, future screen or latent rollout. Replay
contains only new own experience. Existing full trials 43/44 are unchanged.

Online/target weights and original Adam remain in their standard files. Full
SPR training checkpoints additionally save `spr-aux.safetensors`,
`spr-ema.safetensors`, `spr-optimizer.npz` and `spr-rng.npz`, with checksums in
the final state marker. Missing or mismatched auxiliary state prevents resume.
An own scalar conversion preserves both Q networks, original Adam and existing
RNGs; auxiliary parameters/Adam/key start explicitly fresh and EMA copies the
loaded online network. Later SPR resumes restore all auxiliary state and key.
Local diagnostic counters clear while native archives and replay refill.
Breakdown's checkpoint format is unchanged; ordinary frozen Q weights still
suffice for acting and replay verification.

The [default-path before/after check](results/defense/diagnostics/spr-default-parity.json)
reproduces both networks, all **26** Adam arrays and prior counters/RNGs exactly.
The [zero-update actual-parent conversion](results/defense/diagnostics/spr-parent-conversion-parity.json)
preserves the original state and initializes EMA to the parent online weights
exactly. Tiny native tests are not performance results or training parents.
Focused tests cover stored TD-field/sampler identity, auxiliary gradients without
TD-head or target gradients, padding invariance, EMA arithmetic, independent
target synchronization, explicit-key resume, corrupted state rejection, native
own resets and MLX-free workers. A complete native game and independent frozen
replay publication also pass without using auxiliary inference.

All [six focused SPR tests](results/defense/diagnostics/spr-focused-tests.txt)
and the [317-test full regression suite](results/defense/diagnostics/spr-regression-tests.txt)
passed. A separate [production-size native integration check](results/defense/diagnostics/spr-production-smoke.json)
used eight workers, batch 64 and five-step paths for **16,384** new actions,
performing **398** joint updates with peak recorded MLX allocation **688,636,200
bytes** (about **657 MiB**). It exited normally. This was not a performance
evaluation; neither its weights nor trajectories initialize the calibration.

The [isolated SPR calibration](results/defense/training/spr-split-calibration-01/resume-config.json)
starts directly from original DQN 33 at **6,962,144**, collects **131,072** new
actions and then evaluates ten complete uncapped boot games on reused seeds
10000–10009. Its [configuration comparison](results/defense/training/spr-split-calibration-01/design.json)
allows only SPR settings/provenance, the newer trainer hash and explicit false
trace flag, and output paths. Acting-model, environment, baseline replay,
exploration and archive source hashes match the historical worker-split baseline.
The default-path parity check covers the intervening disabled branches.
It retains gamma .997, n-step 5, random-hold cap 64, two boot workers at epsilon
.05, six other workers at .9, lookback 128, and all original optimizer settings.
The original Q optimizer resumes; only the new auxiliary optimizer starts fresh.
This is a historical same-parent comparison, not an independent replication or
fresh success-rate estimate. The short calibration is excluded from the shared
collector. Its completed results and full continuation are below; the original
mission goal remains unachieved.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-spr-split-reproduction \
  --artifacts runs/defense-spr-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 --spr-weight .1
```

#### Visual-prediction calibration result and diagnostic

The [completed calibration](results/defense/training/spr-split-calibration-01/comparison.json)
at **7,093,216** has mean **10,362**, median **10,355**, best **10,460**, with all
ten games lost in stage 1. Its mean exceeds the historical same-parent baseline
by **896** points: six seed-paired scores increased and four decreased. Most of
the aggregate difference comes from two formerly poor games (**+5,650** and
**+2,460**); this is not evidence of a reliable new capability. The frozen
[2,567-action replay](results/defense/training/spr-split-calibration-01/replay/replay.html)
is independently verified. Its [loss panels](results/defense/diagnostics/shared-loss-spr-calibration-01/policy-1-losses.png)
again show the right-opening barrier sequence with the ship near centre/left;
life scores are **2,600 / 2,620 / 2,620 / 2,620**. Exact collision causes remain
unproven. The shared 10,480-point best is unchanged.

The [audit](results/defense/training/spr-split-calibration-01/audit.json) records
**131,072** new actions, **7,566** updates, **52** complete boot training games,
**50** restored segments and **1,286** archive events. All archive offsets are
128 and the reserved boot workers remained boot-only. All logged episodes stay
in stage 1. The auxiliary branch made **2,398,124** valid future predictions;
its final weighted cosine loss is **.01443**. Peak logged MLX allocation was
**688,619,040 bytes**. The process exited normally; full Q/target/Adam and all
four auxiliary state files, evaluation, hashes and complete compressed log are
preserved in the calibration directory.

`python -m rl.defense_spr_probe CHECKPOINT REPLAY --output NEW_JSON` is a
separate read-only diagnostic, never imported by training. It verifies full
auxiliary hashes and matching replay provenance, reconstructs every greedy
action, and checks source hashes again after inference. It samples five-step
paths without crossing visible life boundaries. Dropout is disabled and PER
weights are absent, so its values are **not** the logged training objective.

The [diagnostic including a persistence baseline](results/defense/diagnostics/spr-calibration-representation-with-persistence-7093216.json)
uses **320** paths / **1,600** target vectors. These representations are not
constant: total unit-vector variance is **.3187** and no target vector is zero.
On the second half of collected vectors, cosine distances are:

| Prediction | Mean distance (lower is better) |
| --- | ---: |
| Learned transitions, recorded actions | .104746 |
| Learned transitions, fixed rotated action labels | .104728 |
| Constant fitted to first-half targets | .174024 |
| Unchanged current EMA representation | .015516 |

Thus the learned predictor beats a constant but is substantially worse than
simply persisting the current features. Its mean prediction change under action
rotation is only **.000416** cosine distance. This gives no evidence yet of
useful action-conditioned prediction, despite its small training loss. It is
not a proof of representation collapse or a causal explanation of navigation:
nearby screens are correlated, this is one selected greedy game rather than
exploratory training data, and rotated actions are not simulated trajectories.
The initial report without the persistence baseline is retained separately.
All [12 focused diagnostic/SPR tests](results/defense/diagnostics/spr-probe-tests.txt)
pass, including deterministic repeated inference on this replay, source
immutability, invalid provenance, constant-feature fixtures, boundary masking,
and the existing native training/resume/replay checks. Training code is unchanged.

The follow-up `--training-dropout --seed N` diagnostic uses the saved auxiliary
dropout rate (**.5**) and explicit local noise keys. Its root/target key splitting
and future-screen batch layout exactly match `SprLearner.auxiliary_loss`, checked
against the actual unweighted auxiliary loss. It never uses or changes the
learner's saved random key. Four [focused tests](results/defense/diagnostics/spr-dropout-probe-tests.txt)
pass, including noise-key determinism, dropout-free numerical parity with the
earlier report, provenance checks and source immutability.

On the same frozen replay, the [seed-0](results/defense/diagnostics/spr-calibration-dropout-seed0-7093216.json)
and [seed-1](results/defense/diagnostics/spr-calibration-dropout-seed1-7093216.json)
checks produce:

| Dropout diagnostic | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| Recorded-action prediction distance | .062168 | .061343 |
| Rotated-action prediction distance | .062397 | .061425 |
| Constant-target distance | .101320 | .102411 |
| Unchanged-current-target distance | .060971 | .059287 |
| Unit-target variance sum | .198184 | .200934 |
| Prediction action sensitivity | .000705 | .000738 |

The representation remains nonconstant with dropout. Learned prediction gets
closer to the persistence baseline than in the clean diagnostic, but still does
not outperform it on either noise seed; the recorded-versus-rotated action
difference remains small. This narrows the earlier concern without establishing
useful controllable dynamics. Independent noise in current/future encodings is
part of these distances. None of these unweighted results equals the logged
PER-weighted loss, and neither same-replay noise repeat is an independent game
or training replication. Trial 45's settings remain unchanged while it adapts.

Full trial **45** resumes this exact checked state with unchanged settings and
ten complete uncapped boot games every **200,000** further actions. It tests
longer adaptation, not a claimed solution. Its
[configuration](results/defense/training/dqn-45-visual-prediction/resume-config.json)
changes only run/artifact/parent paths, the unlimited step budget and evaluation
interval. The sole [collector now includes 41 full trials](results/defense/training/dqn-45-visual-prediction/collector-config.json);
all short calibrations remain excluded. No diagnostic trace enters training.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-spr-full-reproduction \
  --artifacts runs/defense-spr-full-reproduction/artifacts \
  --resume results/defense/training/spr-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

Run 45's [first full comparison at **7,293,216**](results/defense/training/dqn-45-visual-prediction/comparison-at-000007293216.json)
has mean **9,766**, median **9,760**, best **9,780**. Every game remains a stage-1
loss. The historical baseline mean is **9,755**: two paired scores improved,
eight fell, for a net **+11** mean. The calibration's score advantage has not
become a convincing advantage in this first continuation round. Full state and
the [2,439-action verified replay](results/defense/training/dqn-45-visual-prediction/first-replay/replay.html)
are preserved, without replacing the global best.

The [audit](results/defense/training/dqn-45-visual-prediction/audit-at-000007293216.json)
records **200,000** new actions, **11,874** joint updates, **86** boot games,
**66** restored segments and **1,974** archive events, all with correct lookback
and protected boot workers. No logged episode reached stage 2. The
[loss panels](results/defense/diagnostics/shared-loss-spr-full-01/policy-1-losses.png)
show a mixture of earlier centre-opening and recurring right-opening barriers;
life scores are **2,450 / 2,430 / 2,450 / 2,450**. Neither equal score nor white
flashes prove exact collision locations or causes.

The [first full checkpoint's dropout diagnostic](results/defense/diagnostics/spr-full-dropout-seed0-7293216.json)
now gives recorded-action distance **.05225**, below persistence **.05830** and
the constant baseline **.10345** on its selected replay. However, rotating the
action labels still gives almost the same distance (**.05264**); prediction
sensitivity is only **.000636**. Thus it is too strong to say the predictor
never beats persistence after more training, but there is still no demonstrated
navigation benefit or substantial use of action labels. These are different
selected states from the calibration, not matched-state learning curves.
The [second noise seed](results/defense/diagnostics/spr-full-dropout-seed1-7293216.json)
gives the same ordering: recorded **.05180**, rotated **.05203**, persistence
**.06260**, constant **.10372**; this is a noise repeat, not another game.
Trial 45 continues unchanged.

At **7,493,216**, run 45's [second full comparison](results/defense/training/dqn-45-visual-prediction/comparison-through-000007493216.json)
has mean **8,151**, median **7,450**, best **9,800**; all ten games still lose in
stage 1. Its mean is **700** below the historical same-count baseline's **8,851**,
following the first-round **+11** difference. The full second checkpoint and
[2,541-action verified replay](results/defense/training/dqn-45-visual-prediction/replay-9800/replay.html)
are preserved. Auxiliary loss fell to **.00719**, but this is not gameplay
progress: no later stage has been observed and the shared best is unchanged.

The [third full comparison at **7,693,216**](results/defense/training/dqn-45-visual-prediction/comparison-at-000007693216.json)
recovers to mean **10,295**, median **10,275**, best **10,400**, still all stage 1.
The historical same-count baseline averages **10,299**: a **−4** difference,
not a demonstrated advantage. Its complete state and
[2,585-action verified replay](results/defense/training/dqn-45-visual-prediction/replay-10400/replay.html)
are preserved. Cumulative continuation totals are **600,000** actions,
**252** complete boot games and **189** completed restored segments; all
**5,884** logged archive offsets are 128, with reserved boot workers protected.

At **7,893,216**, the [fourth round](results/defense/training/dqn-45-visual-prediction/comparison-through-000007893216.json)
averages **10,134**, median **10,165**, best **10,280**, again all stage 1. This
is **+2,824** above the historical baseline's weak fourth round (7,310), following
differences **+11 / −700 / −4**. None of these score comparisons establishes a
stage clear. The preserved third-round 10,400 replay remains this full trial's
best; the stronger shared best is unchanged.

### Inverse-action representation experiment

The separate optional `--inverse-weight .01` tests whether learning to predict
recorded actions from adjacent visible screens helps the encoder. This follows
the inverse-classification idea in [Pathak et al., ICML 2017](https://arxiv.org/abs/1705.05363),
but **omits ICM's forward model and intrinsic reward**. It is not an ICM
reproduction. The [design, acceptance checks and limitations](results/defense/diagnostics/inverse-action-design.md)
are recorded separately. Weak action sensitivity in the SPR diagnostic motivates
the comparison; it does not prove the source of the navigation failure.

Training encodes each actual before/after four-frame screen stack with the same
online QNetwork, concatenates the two 256-dimensional features, and classifies
the recorded action through a 512→256 ReLU layer and a 20-class output. Both
encoder paths receive auxiliary gradients; the ordinary value/advantage heads
do not. The root action is paired with its **immediate** next screen, never the
n-step endpoint or a screen after a reset. Existing compact trajectory storage
retains exact original TD fields, priority sampling and sampler RNG.

The joint loss adds **.01** times PER-weighted action cross entropy to the
unchanged scalar Double-Q loss. Priorities remain TD errors only; joint gradients
are clipped to norm 10. Original Q Adam is preserved and the classifier uses a
separate Adam at the same learning rate. There is no dropout, EMA, intrinsic
reward, route label, action-semantic grouping or policy override. Acting and
evaluation remain ordinary greedy QNetwork inference from four past/current
visible frames. No classifier or next observation is used to select actions.

Full checkpoints additionally require `inverse-head.safetensors` and
`inverse-optimizer.npz`, with hashes and architecture identity checked on resume.
Scalar conversion keeps online/target/Adam and the original RNG states; only the
classifier and its Adam start fresh. Local exploration counters clear as on all
normal resumes, while persistent-exploration RNG restores exactly. Native
archives/replay refill from new own experience. Ordinary Q weights remain
sufficient for an independently verified replay, not a full training resume.

The [default before/after audit](results/defense/diagnostics/inverse-default-parity.json)
preserves both networks, all **26** original optimizer arrays and original state
fields bit-for-bit. The [actual-parent zero-update conversion](results/defense/diagnostics/inverse-parent-conversion-parity.json)
also preserves those arrays and original counters/RNG. All
[six focused tests](results/defense/diagnostics/inverse-focused-tests.txt) pass:
sampler/TD parity through ring wrap and boundaries, immediate-screen alignment,
gradient routing, TD-only priorities, target sync, default acting, full array-exact
resume, missing/corrupt auxiliary rejection, real own-state resets, MLX-free
workers and a complete native game with independent frozen replay verification.
The [full 327-test regression suite](results/defense/diagnostics/inverse-regression-tests.txt)
also passes. Existing Breakdown and Defense behavior remains covered.

A separate [production-sized integration check](results/defense/diagnostics/inverse-production-smoke.json)
used eight workers, capacity 200,000, batch 64 and five-step TD returns for
**16,384** new actions. It completed **398** joint updates / **25,472** own screen
pairs, with peak logged MLX allocation **509,643,200 bytes**, and exited normally.
This is not performance evidence; smoke/conversion models and trajectories are
not calibration parents or training data.

The [isolated calibration](results/defense/training/inverse-split-calibration-01/resume-config.json)
starts directly from the same original DQN-33 checkpoint
at **6,962,144**, using the historical worker-split baseline settings and
**131,072** new actions, followed by ten complete uncapped boot games on reused
seeds 10000–10009. It is not combined with SPR, trace cutting, quantiles or
bootstrap heads. Existing live trials are unchanged. A later frozen diagnostic
must compare action prediction with shuffled/repeated next screens and action
frequency baselines: aliases, persistent actions and starting-screen shortcuts
can all make classification accuracy misleading. Neither classification accuracy
nor score improvement alone proves passage through the recurring barrier.
The [configuration comparison](results/defense/training/inverse-split-calibration-01/design.json)
permits only inverse settings/provenance, the newer trainer hash and disabled
SPR/trace flags, and output paths. Acting-model, environment, ordinary replay,
exploration and archive source hashes match the historical baseline. Results are
below; the short calibration remains excluded from the shared collector.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-inverse-split-reproduction \
  --artifacts runs/defense-inverse-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 128 --inverse-weight .01
```

#### Inverse-action result and frozen controls

The [completed calibration](results/defense/training/inverse-split-calibration-01/comparison.json)
at **7,093,216** has mean **8,632**, median **9,420**, best **10,190**. Every game
remains a stage-1 loss. The mean is **834** below the historical baseline; one
paired score improved and nine fell. Full state including both auxiliary files,
the complete compressed log, and the
[2,550-action verified replay](results/defense/training/inverse-split-calibration-01/replay/replay.html)
are preserved. Its [loss panels](results/defense/diagnostics/shared-loss-inverse-calibration-01/policy-1-losses.png)
again show the right-opening barrier sequence with the ship near centre/left;
life scores are **2,550 / 2,520 / 2,570 / 2,550**. This did not solve the bottleneck.

The [audit](results/defense/training/inverse-split-calibration-01/audit.json)
records **131,072** new actions, **7,566** updates, **55** boot games, **37**
restored segments and **1,282** archive events. All recorded offsets are 128,
reserved workers stayed boot-only, and all logged episodes stayed in stage 1.
There were **484,224** sampled own pairs. Peak logged MLX allocation was
**509,635,520 bytes**. The process exited normally.

`python -m rl.defense_inverse_probe CHECKPOINT REPLAY --output NEW_JSON` is a
read-only control, not a trainer. It checks full classifier/checkpoint hashes,
matching replay provenance and all greedy actions, then compares paired-screen
classification with repeated-current and deterministically permuted-next-screen
inputs. A smoothed action-frequency prior fits only the first half of recorded
actions. All comparisons below use the same **1,275** second-half decisions:

| Classifier input/control | Cross entropy (lower is better) | Accuracy |
| --- | ---: | ---: |
| Actual adjacent screens | 2.7361 | 17.10% |
| Repeat current screen | 2.7334 | 17.25% |
| Permute next screens | 2.9504 | 13.18% |
| First-half action-frequency prior | 2.4668 | 31.22% |

The [frozen report](results/defense/diagnostics/inverse-calibration-classifier-7093216.json)
therefore does **not** demonstrate useful action-change recognition. Real next
screens perform almost the same as unchanged-current screens and worse than a
simple class-frequency control. Permuting screens hurts, but that alone does
not prove controllable dynamics: temporal/context relationships and policy
shortcuts are confounds. This is one selected greedy game, not the exploratory
training distribution. Temporal halves are not independent samples, and these
inference-only input substitutions are not playable counterfactual trajectories.
No diagnostic data enters training. All [three focused probe tests](results/defense/diagnostics/inverse-probe-tests.txt)
pass, including analytic metrics, invalid input, deterministic frozen inference,
source immutability and corrupted auxiliary-state rejection.

Full **DQN 46** now [continues the exact calibrated state](results/defense/training/dqn-46-inverse-action/resume-config.json)
with unchanged learner settings and ten complete uncapped boot games every
**200,000** new actions. This tests whether longer adaptation can establish the
intended classifier capability and improve gameplay; the negative short result
is not presented as an advantage. No SPR or other new change is combined with it.
The sole [collector includes 42 full-trial sources](results/defense/training/dqn-46-inverse-action/collector-config.json),
including retired histories and excluding all short calibrations. The original
10,480-point global best remains unchanged and verified.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-inverse-full-reproduction \
  --artifacts runs/defense-inverse-full-reproduction/artifacts \
  --resume results/defense/training/inverse-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

Run 46's [first full comparison at **7,293,216**](results/defense/training/dqn-46-inverse-action/comparison-at-000007293216.json)
has mean **9,870**, median **9,940**, best **10,000**, all stage-1 losses. The mean
is **115** above the historical baseline; two paired scores improved and eight
fell. This reversal of the short comparison is not a demonstrated navigation
advantage. The complete checkpoint and
[2,455-action verified replay](results/defense/training/dqn-46-inverse-action/first-replay/replay.html)
are preserved. Its [audit](results/defense/training/dqn-46-inverse-action/audit-at-000007293216.json)
records **200,000** new actions, **11,874** updates, **95** boot games, **70**
restored segments and **1,903** correctly offset archive events, with protected
boot workers and no logged later stage.

The [frozen classifier controls](results/defense/diagnostics/inverse-full-classifier-7293216.json)
on **1,228** second-half decisions give paired-screen cross entropy **2.5881** /
accuracy **20.28%**, versus repeated-current **2.5967 / 20.20%**, permuted-next
**2.7376 / 16.04%**, and frequency prior **2.6152 / 31.51%**. Paired cross entropy
now slightly beats the prior, but top-class accuracy remains lower and the
paired-versus-repeated distinction remains tiny. These are different selected
states from the calibration, not a matched-state causal learning curve. Run 46
continues unchanged; the global best remains intact.

At **7,493,216**, run 46's [second full comparison](results/defense/training/dqn-46-inverse-action/comparison-at-000007493216.json)
has mean **10,220**, median **10,215**, best **10,280**. This is **+1,369** over
the historical same-count baseline's 8,851, but every game remains a stage-1
loss. The full optimizer and
[2,494-action verified replay](results/defense/training/dqn-46-inverse-action/replay-10280/replay.html)
are preserved. Across **400,000** continuation actions, **188** complete boot
games and **132** completed restored segments, no logged episode reached a
later stage. All **3,856** archive offsets are 128 and boot workers remain
protected. The trial continues without replacing the shared best.

The [third and fourth full comparisons](results/defense/training/dqn-46-inverse-action/comparison-through-000007893216.json)
give means **9,455** and **10,265**, respectively, both all stage-1 losses. At
**7,893,216**, the fourth round has median **10,280** and a new run-best
**10,330**. Its [2,536-action verified replay](results/defense/training/dqn-46-inverse-action/replay-10330/replay.html),
complete model/target/Q-Adam/auxiliary-Adam state and checked log prefix are
preserved. The [audit](results/defense/training/dqn-46-inverse-action/audit-at-000007893216.json)
records **800,000** continuation actions, **49,374** updates, **366** complete
boot games, **263** completed restored segments and **7,773** correctly offset
archive events, with protected boot workers and no later stage in the log.
Mean differences versus the same-count historical baseline are
**+115 / +1,369 / −844 / +2,955**. The larger fourth difference reflects the
baseline's low 7,310 mean, not passage through the recurring failure sequence.

At **8,093,216**, the [fifth round](results/defense/training/dqn-46-inverse-action/comparison-through-000008093216.json)
regresses to mean **9,268**, median **9,525**, best **10,040**, all stage 1. This
is **−516** against the historical same-count baseline's 9,784. The preceding
10,330 verified replay remains the full trial's best. One million additional
training actions after calibration have not yet produced a stage clear.

### Intermediate own-state rewind calibration

The [selected-reset audit](results/defense/diagnostics/selected-reset-origins-45-46.json)
joins each completed restored segment in fixed log prefixes to its original
same-run archive event by source worker/action. It verifies chronological order,
matching starting score/stage, and the 128-action archive offset. Every one of
the **134** SPR and **50** inverse-action completed restored segments matched.
Their selected states have per-life scores **0–190** and **0–140**, respectively;
**110/134** and **39/50** originate from the two low-epsilon boot-only workers.

This confirms actual reuse of low-scoring own approach states, not just archive
creation. It does not establish exact obstacle positions or collision lead time:
score-based archive logs contain neither captured screens nor exact life-age.
The saved state is 128 actual decisions before a score-bin trigger, not before
a detected collision. The audit reads no opaque native payloads and creates
no snapshots, labels, demonstrations or training data.

The [new scalar-DQN calibration](results/defense/training/mid-lookback-split-calibration-01/resume-config.json)
tests an intermediate **64-action** rewind. This may reduce repeated lead-in
while retaining preparation time, but that is a hypothesis to test. PPO has
previously used 64; this is specifically a controlled scalar-DQN worker-split
comparison, not the first use of this duration anywhere in the project.
It starts from the original DQN-33 parent at **6,962,144** and collects **131,072**
new own actions, followed by ten complete uncapped boot games on reused seeds
10000–10009. The [configuration comparison](results/defense/training/mid-lookback-split-calibration-01/design.json)
changes only rewind length relative to the historical 128-action baseline,
apart from disabled optional flags, trainer provenance and output paths.

It retains gamma .997, five-step returns, random-hold cap 64, epsilon .05 on two
reserved boot workers and .9 on six others, reset probability .5 and the same
score-based archive capacity. SPR, inverse classification and trace cutting are
all **off**. The existing setting needs no trainer change; disabled-path parity
and the full regression checks already cover the unchanged code. Runs 45/46
remain untouched. All states are opaque saves of this trial's actual experience,
not evaluation replay states or hand-selected routes. The short calibration is
excluded from the shared collector.

The calibration completed normally at **7,093,216**. Its
[ten complete games](results/defense/training/mid-lookback-split-calibration-01/checkpoint/evaluation.json)
averaged **10,398**, median **10,410**, best **10,440**, with no later stage or
mission. The [paired comparison](results/defense/training/mid-lookback-split-calibration-01/comparison.json)
is **+932** over the historical 128-action baseline: seven higher, one lower,
two tied. Recoveries of **5,740** and **2,430** points on two weak baseline games
account for most of the aggregate gain. These reused seeds do not establish a
fresh success rate or passage through the navigation bottleneck.

The [audit](results/defense/training/mid-lookback-split-calibration-01/audit.json)
records **7,566** updates, **55** complete boot games, **34** completed restored
segments and **1,409** exact 64-action archive offsets, with reserved boot
workers protected. Archive source progress reached 330 versus trigger progress
2,620; neither value is an exact course position. Full online/target/optimizer
state and the complete compressed log are preserved with checked hashes.
The [2,502-action verified replay](results/defense/training/mid-lookback-split-calibration-01/replay/replay.html)
earns **2,620 / 2,620 / 2,580 / 2,620** across four lives. Its
[unaltered-screen loss panels](results/defense/diagnostics/shared-loss-mid-lookback-01/policy-1-losses.png)
again show the right-opening barrier sequence, with one earlier loss near the
centre-opening barrier. The rewind change has not resolved this failure.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-mid-lookback-split-reproduction \
  --artifacts runs/defense-mid-lookback-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 64 \
  --spr-weight 0 --inverse-weight 0
```

### Own-loss-triggered reset experiment

The selected-reset audit and repeated failure panels motivate testing the
**trigger**, not only the rewind duration. The default archive rewinds from
score-bin changes; it is not aligned to ship loss. Optional DQN
`--curriculum-trigger life-loss --curriculum-lookback 64` instead waits for an
actual visible life-counter loss and retains the exact opaque own snapshot
from 64 decisions earlier. It does so before clearing that life's history or
resetting its score baseline. The final lost life can also contribute a prior
live state. Short histories are skipped, never borrowed across a life/reset.
New visible stages still receive an immediate own snapshot.

This changes training starting states only. Archive keys, bounded retention,
sharing, boot-only workers and probabilistic resets remain unchanged. The
saved state's own score/life-age/screen key is used, not a fabricated label
from the later event. Policy input stays four raw visible frames; reward stays
visible score delta. There is no route, extra penalty, action override, hidden
RAM label or demonstration. A visible ship loss may lag physical collision;
64 decisions is a hypothesis, not a guarantee of a recoverable state. Evaluation
always plays ordinary complete games from boot without archive access.

The [disabled-path parity check](results/defense/diagnostics/loss-trigger-default-parity.json)
compares a 4,096-action diagnostic before and after the change: all **12** online,
**12** target and **26** optimizer arrays, every non-config state field and all
**41** archive/episode records are equal. Paths, provenance hashes and the
new default `progress` config entry differ. These capped diagnostic episodes
are not performance evidence. A separate
[zero-action parent conversion](results/defense/diagnostics/loss-trigger-parent-conversion-parity.json)
preserves the actual DQN-33 parent's weights, Adam, original counters and both
RNGs exactly. Local exploration statistics clear normally with boot resets.

The [production-setting smoke](results/defense/diagnostics/loss-trigger-production-smoke.json)
completed 16,384 new actions without an evaluation; it is not a performance
parent. Tests check all four real losses against exact earlier native bytes and
screens, unchanged complete-game observations/rewards for score/screen/age keys,
final-life capture, insufficient history, immediate stage-entry bookkeeping,
peer restoration, protected boot workers, absence of capture when disabled,
truncation without loss, and native DQN learning/resume with archive refill.
Stage-transition fixtures are not evidence of actual stage passage.

All [seven focused checks](results/defense/diagnostics/loss-trigger-focused-tests.txt)
passed, and the [full 337-test suite](results/defense/diagnostics/loss-trigger-regression-tests.txt)
passed in 326 seconds. The production smoke made **398** updates, completed
**eight** boot games and **one** restored segment, and emitted **38** correctly
offset own-loss archive events with boot workers protected. Its measured MLX
peak was **289,622,052 bytes**; it exited normally. This verifies execution,
not improved gameplay.

The [controlled calibration](results/defense/training/loss-trigger-split-calibration-01/resume-config.json)
started from original DQN 33 at **6,962,144**,
not the smoke or either auxiliary learner. It uses the same settings as the
completed intermediate-rewind check: 64-action lookback, gamma .997, five-step
returns, hold cap 64, split epsilon .05/.9, eight workers, reset probability .5,
score cells 16×4 and no auxiliary losses. The [configuration audit](results/defense/training/loss-trigger-split-calibration-01/design.json)
confirms that only the trigger changes, apart from paths and source provenance. Compare
131,072 new actions and ten complete uncapped boot games on the reused seeds;
judge actual stage passage separately from mean score. The short calibration
is excluded from the shared best collector.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-loss-trigger-split-reproduction \
  --artifacts runs/defense-loss-trigger-split-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 64 \
  --curriculum-trigger life-loss --spr-weight 0 --inverse-weight 0
```

The calibration exited normally at **7,093,216**. Its
[ten complete games](results/defense/training/loss-trigger-split-calibration-01/checkpoint/evaluation.json)
averaged **10,198**, median **10,180**, best **10,260**, all stage-1 losses.
The [paired comparison](results/defense/training/loss-trigger-split-calibration-01/comparison.json)
is **−200** relative to the 64-action progress-triggered control's 10,398;
all ten paired scores fell. This is a negative short result, not a navigation
improvement. The complete online/target/Adam state, full compressed log and
[2,493-action independently verified replay](results/defense/training/loss-trigger-split-calibration-01/replay/replay.html)
are preserved with checked hashes. The
[loss panels](results/defense/diagnostics/shared-loss-trigger-calibration-01/policy-1-losses.png)
show the recurring right-opening sequence and an earlier centre-opening loss;
life scores are **2,570 / 2,570 / 2,550 / 2,570**. Exact physical collision causes
remain unproven.

The [audit](results/defense/training/loss-trigger-split-calibration-01/audit.json)
records **7,566** updates, **61** boot games, **31** completed restored segments,
and **306** actual own-loss archive events. All offsets and life-age differences
are exactly 64, source lives precede the observed decrement, and reserved boot
workers remain protected. No logged training episode reached a later stage.
Every completed restored segment joins to a preceding same-run archive event
with matching starting score/stage; **24/31** sources came from boot-only workers.
Source progress reaches **1,100**, compared with the control archive's maximum
330. This confirms changed sampling, not a correct route or improved gameplay.
The [two-arm selected-reset report](results/defense/diagnostics/selected-reset-origins-mid64-vs-loss64.json)
preserves the actual selections and complete-log provenance, rather than
inferring reuse from archive occupancy.

Full **DQN 47** [continues the progress-triggered control](results/defense/training/dqn-47-mid-lookback-control/resume-config.json),
and **DQN 48** [continues the own-loss arm](results/defense/training/dqn-48-loss-trigger/resume-config.json),
each from its own exact calibrated optimizer at **7,093,216**, with unlimited
learning and evaluations every 200,000 actions. Their
[paired design](results/defense/training/dqn-48-loss-trigger/paired-design.json)
keeps all learning settings equal except the trigger and its semantics; paths
and their separately learned parent states differ. Own archives/replay refill
from new experience on resume. A negative short score comparison does not prove
that further exposure cannot help, but neither is continuation evidence that it
will. Both runs must be judged by actual boot-game stage/mission reach, with
matched action-count comparisons and the stronger global best preserved.

The previous collector exited cleanly before its replacement started. The sole
[collector now watches 44 full-trial sources](results/defense/training/dqn-48-loss-trigger/collector-config.json),
including 47/48 and every prior source, while excluding short calibrations.
Shared-best promotion still requires independent native replay verification.
The existing 10,480-point best remains unchanged.

The [first full paired evaluation at **7,293,216**](results/defense/training/dqn-48-loss-trigger/comparison-at-000007293216.json)
adds **200,000** actions per arm after calibration (**331,072** after their
original common parent). The progress-triggered control averages **10,368**,
median **10,450**, best **10,480**. The own-loss arm averages **9,889**, median
**10,030**, best **10,280**. Every one of the twenty complete boot games lost
in stage 1. The own-loss arm is **−479** in mean, with nine paired scores lower
and one higher, following its **−200** calibration difference. Neither actual
stage passage nor a score advantage has been demonstrated.

Both complete optimizer checkpoints, checked log prefixes and independently
verified replays are preserved:
[control, 2,569 actions](results/defense/training/dqn-47-mid-lookback-control/first-replay/replay.html)
and [own-loss, 2,575 actions](results/defense/training/dqn-48-loss-trigger/first-replay/replay.html).
The control merely ties the existing global single-game best and does not
replace it. The [screen-only loss panels](results/defense/diagnostics/shared-loss-trigger-full-01/report.json)
again show the recurring barrier sequence. The selected control replay earns
**2,620** on each life, versus **2,570** on each life in the own-loss replay;
score equality is not a claim of identical physical collisions.

The [control audit](results/defense/training/dqn-47-mid-lookback-control/audit-at-000007293216.json)
records **89** boot games, **64** completed restored segments and **2,164** archive
events. The [own-loss audit](results/defense/training/dqn-48-loss-trigger/audit-at-000007293216.json)
records **85**, **67** and **454**, respectively. Both made **11,874** updates,
kept correct 64-action offsets and protected reserved boot workers. All logged
training episodes remained in stage 1. Every completed restored segment joins
to its original same-run source: **17/67** own-loss selections exceed the
control's maximum selected per-life score of 330, reaching 1,100. These log
joins verify different sampling, not improved navigation or exact course
positions. The pair continues unchanged for another matched evaluation; this
negative first round is not presented as a successful intervention.

At **7,493,216**, the [second full comparison](results/defense/training/dqn-48-loss-trigger/comparison-at-000007493216.json)
adds **400,000** actions per arm after calibration. Control mean/median/best are
**10,051 / 10,060 / 10,310**; own-loss results are **9,757 / 9,980 / 10,080**.
All twenty games still lose in stage 1. Own-loss mean is **−294**, with seven
paired scores lower and three higher; its two full-round differences are
**−479 / −294**. Both complete online/target/Adam states and checked log prefixes
are preserved. Neither round set a new run-best, so no new replay was published:
the earlier independently verified 10,480 and 10,280 replays remain intact.

The cumulative [control audit](results/defense/training/dqn-47-mid-lookback-control/audit-at-000007493216.json)
has **176** boot games, **135** completed restored segments and **4,320** archive
events; the [own-loss audit](results/defense/training/dqn-48-loss-trigger/audit-at-000007493216.json)
has **170**, **130** and **906**. Each made **24,374** updates, with correct offsets,
protected boot workers, and no logged stage beyond 1. Of the own-loss selections,
**41/130** have source per-life score above the control's maximum of 330.

A [read-only restored-life coverage audit](results/defense/diagnostics/loss-trigger-restored-life-coverage-7493216.json)
checks an additional distinction: a reset selects the beginning of a training
segment, but the current trainer then continues through all remaining native
lives. The 130 completed restored segments contain **97,795** actions. Only
**1,978–7,522** are in their initial restored lives; **90,273–95,817** occur in
later native lives. For the 41 above-control-score starts, their initial lives
contribute **633–1,956** actions, or **0.158–0.489%** of the 400,000 total actions.

Those are bounds on the contribution of **completed restored segments**, not an
upper bound on all useful obstacle practice; still-active segments are excluded,
and ordinary boot/later lives can also reach the barrier. The audit reconciles
worker-local action counters with every completed segment length. When the
first visible loss has an archive event (or only one native life remained), its
timing is exact. Otherwise, enabled sharing and the same-life capture rule bound
the unarchived first loss to at most 64 new decisions; no timestamp is invented.
It reads no opaque snapshots and supplies no training examples or labels.
This evidence motivates examining practice frequency, but does not prove that
more frequent resets would work, or that these starts are recoverable. The
current matched pair remains unchanged.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-mid-lookback-full-reproduction \
  --artifacts runs/defense-mid-lookback-full-reproduction/artifacts \
  --resume results/defense/training/mid-lookback-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000

venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-loss-trigger-full-reproduction \
  --artifacts runs/defense-loss-trigger-full-reproduction/artifacts \
  --resume results/defense/training/loss-trigger-split-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
```

At **7,693,216**, the [third comparison](results/defense/training/dqn-48-loss-trigger/comparison-at-000007693216.json)
has control mean/median/best **10,412 / 10,410 / 10,460**, versus own-loss
**9,158 / 10,060 / 10,180**. All ten paired own-loss scores are lower
(mean difference **−1,254**); all twenty games remain stage 1. Both full
optimizer states and audited log prefixes are preserved. Neither sets a new
run-best, so the previously verified replays remain unchanged.

### Restored-life-only practice

The older probability-.5 reset pair completed two further rounds. At
**7,893,216**, [control versus own-loss](results/defense/training/dqn-48-loss-trigger/comparison-at-000007893216.json)
means were **10,426 / 10,280** (difference **−146**). At **8,093,216**,
[means were 8,328 / 9,934](results/defense/training/dqn-48-loss-trigger/comparison-at-000008093216.json)
(difference **+1,606**), reversing the ranking as the control regressed.
All forty games remained stage-1 losses. Full states and checked log prefixes
are preserved for both rounds, with no new run-best replay. Score-rank reversal
does not establish a consistent progression benefit from either trigger.

The coverage audit above motivates a separate optional
`--curriculum-restored-life-only` experiment. An already-restored training
segment now truncates at its first **visible** life loss, unless that loss
already ends the native game. The emulator's actual loss observation, score
and lives are untouched. Boot games remain complete, including the reserved
boot-only workers; a stage entry alone never cuts a segment. The vector worker
then uses the existing reset mechanism and its own opaque training archive.
No route, collision timestamp, evaluation trajectory, hidden-state input or
extra reward is introduced. This is practice allocation, not a claim that
saved states are recoverable.

DQN requires enabled curriculum and life-terminal learning for this option.
The existing n-step targets therefore terminate at exactly the same visible
loss as before. Restored cuts stay classified as training segments, never
complete boot games or ranked evaluation results. The default is **off**.
Seven focused tests cover real boot-game equality, exact restored trajectories
through loss (including opaque native bytes), identical n-step targets,
native last-life termination, vector reset observations, stage entry, CLI
validation, real learning and checkpoint resume. All **344 tests passed**.
[Default parity](results/defense/diagnostics/restored-life-default-parity.json)
is bit exact across Q/target/Adam, RNG/counters and 16 episode/archive events.
[Original-parent conversion](results/defense/diagnostics/restored-life-parent-conversion-parity.json)
also preserves all 50 parameter/optimizer arrays and both RNG states exactly.

The [production smoke check](results/defense/diagnostics/restored-life-production-smoke.json)
completed **16,384** new actions and **398** updates, with **eight** full boot
games and **46** restored segments. Of these, **29** cut with native lives
remaining; the rest ended at native game over. Boot workers remained protected,
and all archive offsets were 64. Neither diagnostic supplies the performance
parent or counts as complete-game performance evidence.

Two new calibrations start independently from the same original DQN-33
checkpoint at **6,962,144**, not a smoke checkpoint. Each receives **131,072**
new actions and ten complete uncapped boot evaluations on reused seeds
10000–10009. Both use life-loss lookback 64, split epsilon .05/.9, two boot-only
workers out of eight, five-step life-terminal returns, gamma .997, persistence
64 and no auxiliary loss. Both set reset probability **1**, unlike the older
.5 pair; this concentrates eligible-worker practice while retaining protected
boot games. Only restored-life-only differs between the two new arms.
They are excluded from the global collector. Comparisons against older runs
cannot isolate the probability change; the new pair can isolate the cutoff
configuration, but subsequent own experience and random streams may diverge.

```bash
# Repeat with --no-curriculum-restored-life-only and a separate run path for control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-restored-life-reproduction \
  --artifacts runs/defense-restored-life-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 64 \
  --curriculum-trigger life-loss --curriculum-probability 1 \
  --curriculum-restored-life-only --spr-weight 0 --inverse-weight 0
```

The [completed comparison](results/defense/training/restored-life-focused-calibration-01/comparison.json)
has control mean/median/best **9,963 / 9,945 / 10,080**, versus focused practice
**10,214 / 10,375 / 10,460**. Focused scores are higher on nine paired seeds
and lower on one (mean difference **+251**). All twenty complete games still
lose in stage 1. This is a modest matched score improvement, not stage progress;
the focused mean is also below the original parent's 10,259. Both full
Q/target/Adam checkpoints and complete compressed logs are preserved, alongside
the [2,519-action control replay](results/defense/training/restored-life-control-calibration-01/replay/replay.html)
and [2,502-action focused replay](results/defense/training/restored-life-focused-calibration-01/replay/replay.html).

Both arms made **7,566** updates. The control completed **18** boot games and
**146** restored segments; focused practice completed **19** and **2,200**, with
**1,515** first-loss cuts while native lives remained. Audits protect the two
boot workers, join all completed restored starts to their own archive sources,
and verify every archive's 64-action offset. Control completed restored
segments contain **86,323** actions, only **4,142–8,363** of them in their
initial restored lives. Focused segments contain **89,015** actions, all in
their initial lives. This verifies the intended practice-allocation change,
not that every such action is useful obstacle practice; active segments are
excluded and score is not exact course position.

The [read-only loss windows](results/defense/diagnostics/restored-life-calibration-losses-01/report.json)
show the familiar right-opening barrier sequence in both selected replays.
Focused life scores are **2,620 / 2,620 / 2,600 / 2,620**; control life scores
are **2,620 / 2,520 / 2,470 / 2,470**. Screens are aligned to a visible flash
or loss marker, not an exact physical collision. Neither demonstrates passage,
nor distinguishes wall collision from a projectile hit with certainty.

Full runs **49/50** now continue their respective preserved calibrations from
**7,093,216**, with unchanged arm settings, unlimited learning and evaluations
every **200,000** new actions. Their [control configuration](results/defense/training/dqn-49-restored-life-control/resume-config.json)
and [focused configuration](results/defense/training/dqn-50-restored-life-focused/resume-config.json)
retain lineage; episodes restart and archives/replay refill from new own play.
The sole collector now retains **46** full-trial sources, including all
historical sources and both new continuations, but no short calibrations.
The shared verified 10,480-point stage-1 replay remains unchanged. Runs 47/48
subsequently retired after six rounds, as recorded below.

Frozen read-only value probes reconstruct all **2,519 / 2,502** greedy actions
in the [control](results/defense/diagnostics/restored-life-control-calibration-values-01.json)
and [focused](results/defense/diagnostics/restored-life-focused-calibration-values-01.json)
calibration replays. Whole-replay mean absolute errors against realized
life-terminal discounted score are **166.29 / 182.50** points. Across the four
64-decision pre-marker windows, control signed errors are
**−14.28 / −1.93 / −20.91 / −51.97**, versus focused
**+70.23 / +122.50 / +129.11 / +76.16**. This does not demonstrate improved
calibration from focused practice. These are different selected trajectories,
not matched states or optimal counterfactual returns; flashes are not exact
collision timestamps. No actions, parameters, rewards or training data changed.

At **7,293,216**, the [first full comparison](results/defense/training/dqn-50-restored-life-focused/comparison-at-000007293216.json)
adds **200,000** actions per arm. Control mean/median/best are
**9,418 / 10,245 / 10,380**; focused results are **9,340 / 9,905 / 10,330**.
All twenty games lose in stage 1. Five paired scores improve and five regress;
focused mean difference is **−78**, so the short calibration's score advantage
has not held in this round. Both full optimizer states and checked log prefixes
are preserved with independently verified [2,560-action control](results/defense/training/dqn-49-restored-life-control/replay-10380/replay.html)
and [2,494-action focused](results/defense/training/dqn-50-restored-life-focused/replay-10330/replay.html)
replays. Neither displaces the shared best.

Each arm made **11,874** updates and completed **26** boot games. Control has
**192** completed restored segments; focused has **3,304**, including **1,915**
first-loss cuts. Every reset source joins to its own logged archive event;
offsets and boot-worker protection remain correct. A separate
[first-life coverage audit](results/defense/diagnostics/restored-life-full-first-coverage.json)
finds control initial-life contribution **3,953–10,757** actions versus focused
**140,721**, all among completed restored segments. Focused median duration is
**39** decisions; **360/3,304** exceed 64, versus **31/192** initial control
lives. Longer duration is not proof of obstacle passage or recoverability:
the actions differ, visible losses can lag collision, and active segments are
excluded. The intended allocation change persists, but progression does not.
The new full pair continues unchanged for another matched round.

The older probability-.5 trigger pair finished its
[sixth comparison](results/defense/training/dqn-48-loss-trigger/comparison-at-000008293216.json)
at **8,293,216**: control mean/median/best **10,318 / 10,325 / 10,430**,
own-loss **10,225 / 10,230 / 10,330**, all stage 1. Own-loss has one higher,
seven lower and two tied paired scores (mean difference **−93**). Its new
full-run best [10,330-point replay](results/defense/training/dqn-48-loss-trigger/replay-10330/replay.html)
verified **2,509** actions; the control's earlier verified 10,480 replay remains
its best. Both latest evaluation optimizers and full audited prefixes are saved.

The pair then stopped gracefully after the depth plateau:
[47 at 8,325,680](results/defense/training/dqn-47-mid-lookback-control/retirement.json)
(**1,232,464** new actions, **552** boot games, **407** restored segments), and
[48 at 8,318,096](results/defense/training/dqn-48-loss-trigger/retirement.json)
(**1,224,880**, **517**, **384**). Complete final Q/target/Adam checkpoints,
logs and all six evaluations are preserved. Neither training nor evaluation
logged a later stage or mission. The collector retains both historical sources.

A new [earlier-start calibration](results/defense/training/restored-life-long-lookback-calibration-01/resume-config.json)
tests **128** decisions of own-loss lookback against the preserved **64**-decision
focused calibration. Configuration comparison confirms only lookback and output
paths differ. Both start from the same original DQN-33 parent at **6,962,144**,
not their trial's final or smoke state; this new check gets the same **131,072**
actions and ten uncapped complete boot evaluations. It retains probability 1,
first-restored-life-only segments, split epsilon, protected boot workers,
unchanged score-only targets and no auxiliary loss. This tests more time to
change approach before a visible loss, not known collision lead time or a
scripted route. The short check is excluded from the collector; 49/50 continue.

The auxiliary trials freed their slots after depth plateaus, not time limits.
[Visual prediction](results/defense/training/dqn-45-visual-prediction/retirement.json)
stopped gracefully at **8,155,240**, with five full evaluation rounds; its final
round mean was **10,235**, best **10,290**. Its earlier verified 10,400 full-run
replay and 10,460 calibration replay remain preserved.
[Inverse action](results/defense/training/dqn-46-inverse-action/retirement.json)
stopped at **8,398,432**, with six rounds; its final mean was **10,214**, best
**10,240**, retaining the earlier verified 10,330 replay. Both full final
optimizers, auxiliary states, complete logs and last evaluation checkpoints
are preserved. No training episode or evaluation reached stage 2 or a mission.

The [earlier-start calibration finished](results/defense/training/restored-life-long-lookback-calibration-01/comparison.json)
at **7,093,216**, with mean/median/best **7,975 / 7,965 / 9,110**, versus the
64-decision focused calibration's **10,214 / 10,375 / 10,460**. Nine paired
scores regress and one improves (mean difference **−2,239**); all ten games
lose in stage 1. The [2,566-action verified replay](results/defense/training/restored-life-long-lookback-calibration-01/replay/replay.html),
full Q/target/Adam state and complete compressed log are preserved. It is not
being continued as a demonstrated improvement.

Its audit records **7,566** updates, **19** boot games, **994** completed
restored segments and **831** first-loss cuts. All **230** archive offsets
are exactly 128 decisions and boot workers stay protected. Completed restored
lives contain **88,831** actions; selected source per-life score ranges from
0 to 240. These are genuinely earlier states, not a known collision lead time.
The [saved loss windows](results/defense/diagnostics/focused-and-earlier-losses-01/report.json)
show the recurring barrier sequence; its life scores are
**1,720 / 2,470 / 2,450 / 2,470**, not new stage progression.

Meanwhile the new full pair reached **7,493,216**, adding **400,000** actions
each. The [second comparison](results/defense/training/dqn-50-restored-life-focused/comparison-at-000007493216.json)
is control mean/median/best **9,832 / 9,800 / 10,000**, versus focused
**9,641 / 9,670 / 10,460**. All twenty games remain stage-1 losses; four
focused scores improve and six regress (mean difference **−191**). Both full
optimizer states and audited log prefixes are preserved. Focused practice's
new full-run best [10,460-point replay](results/defense/training/dqn-50-restored-life-focused/replay-10460/replay.html)
verifies **2,568** actions. Its life scores **2,620 / 2,600 / 2,620 / 2,620**
and loss windows still show the familiar barrier. The shared best is unchanged.
Each arm made **24,374** updates and completed **48** boot games; restored
segment counts are **395 / 6,880**, with **4,527** focused first-loss cuts.
Both retain correct offsets, own-source provenance and protected boot workers.

### Balanced command-group exploration

The existing full restored-life pair completed two additional rounds without
a later stage. At **7,693,216**, the [third comparison](results/defense/training/dqn-50-restored-life-focused/comparison-at-000007693216.json)
has control mean/median/best **10,212 / 10,210 / 10,240**, versus focused
**9,860 / 10,280 / 10,400** (difference **−352**). Six focused paired scores
are higher and four lower; three lower games dominate the mean difference.
Both full states and hash-checked log prefixes are preserved, with no new
run-best replay. By then, control/focused completed **579 / 10,317** restored
segments and **68 / 69** boot games; focused cuts numbered **7,193**. Both
made **36,874** updates with correct archive offsets and protected boot workers.

At **7,893,216**, the [fourth comparison](results/defense/training/dqn-50-restored-life-focused/comparison-at-000007893216.json)
has mean **9,818 / 10,066**, reversing the ranking (focused **+248**).
All forty games across these two rounds remain stage-1 losses. Full optimizer
states and checked prefixes are retained for both rounds; no new run-best
replay is published. Score ranking varies, but neither setting has demonstrated
progression through the recurring obstacle.

The previous read-only alias diagnostics did not establish duplicated firing
choices as the dominant cause of the learned policy's entropy. They did not,
however, test the **training random-action distribution**. Uniform sampling
over the standard 20 keyboard combinations gives 45% of random starts to the
nine stage-1 forward-fire aliases (Space, or Space plus an arrow combination).
Stage 1 compares the whole movement byte, so these combinations fire without
moving. The eight movement directions together get only 40%.

Optional `--exploration-actions stage1-balanced` tests equal probability for
the twelve distinct stage-1 command groups: nine no-op/direction choices,
one forward-fire group and two side-fire combinations. Each group gets 1/12;
the nine forward-fire aliases share their group's probability equally (1/108
each). All twenty actions remain possible, and the learned Q head is unchanged.
Nominal movement probability becomes 2/3 and forward-fire-only 1/12; actual
exploratory-step proportions can differ because life boundaries cut holds.

This is a **fixed training-only sampling distribution**, not a screen-dependent
controller, route or obstacle detector. It never reads the current stage,
score, position or collision state and applies unchanged in later stages,
where the aliases need not be equivalent. Keeping every action possible avoids
removing later-stage move-and-fire choices, but the fixed bias may still be
unhelpful there. No particular movement direction is preferred. Warmup remains
independent uniform sampling, persistent duration/rate/boundary rules are
unchanged, and complete-game evaluation remains original learned greedy play.
The option requires persistent exploration and the standard 20-action profile.
The default `uniform` keeps the original integer RNG draws exactly.

Six focused tests pass, including empirical group probabilities, defensive
probability validation, disabled RNG parity, unchanged greedy choices, hold
reset/resume, CLI rejection, and native learning/checkpoint loading. A real
complete stage-1 game with random diagnostic actions is screen/reward/outcome
identical when its firing aliases are replaced by Space or Down+Right+Space;
these traces are not training demonstrations. The
[4,096-action default parity check](results/defense/diagnostics/balanced-actions-default-parity.json)
matches all 50 Q/target/Adam arrays, all nonconfiguration state fields and all
16 episode/archive records. The
[zero-action parent conversion](results/defense/diagnostics/balanced-actions-parent-conversion-parity.json)
also preserves weights, optimizer, counters and both RNG states exactly.

The [production smoke check](results/defense/diagnostics/balanced-actions-production-smoke.json)
completed **16,384** new actions and **398** updates, eight full boot games,
51 restored segments and 30 first-loss cuts. It exercised every action and
recorded **4,241** post-warmup exploratory steps: **6.08%** forward-fire aliases
and **61.47%** movement, without changing reserved worker roles or reset offsets.
This is plumbing evidence, not complete-game performance evidence. All **350
regression tests passed**. The same-parent performance calibration now starts
from original DQN-33 at **6,962,144**, not the smoke checkpoint, and gets
**131,072** new actions plus ten complete uncapped evaluations on reused seeds.
It matches the preserved 64-decision focused calibration's settings except
for the random-action distribution and recorded source metadata/output paths.
It retains split epsilon .05/.9, probability-1 shared own-loss resets, two
protected boot workers out of eight, first-restored-life-only segments,
five-step life-terminal returns, gamma .997 and no auxiliary loss. It is
excluded from the collector; the existing full runs remain unchanged.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-balanced-actions-reproduction \
  --artifacts runs/defense-balanced-actions-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --epsilon-final .9 \
  --curriculum-boot-epsilon .05 --curriculum-lookback 64 \
  --curriculum-trigger life-loss --curriculum-probability 1 \
  --curriculum-restored-life-only --spr-weight 0 --inverse-weight 0 \
  --exploration-actions stage1-balanced
```

The [balanced calibration completed](results/defense/training/balanced-actions-calibration-01/comparison.json)
at **7,093,216**, with mean/median/best **10,201 / 10,280 / 10,280**, versus
uniform's **10,214 / 10,375 / 10,460**. Two paired scores improve and eight
regress; the mean difference is only **−13**, with one +1,470 recovery offsetting
several smaller regressions. All ten games still lose in stage 1. This is
near-parity in mean, not an improvement or reliable stage progression.

The [2,540-action verified replay](results/defense/training/balanced-actions-calibration-01/replay/replay.html),
complete Q/target/Adam state and full compressed log are preserved. Its
[loss windows](results/defense/diagnostics/balanced-actions-calibration-losses-01/report.json)
repeat **2,570** points on each life around the familiar right-opening barrier.
The flash markers remain alignment aids, not exact collision timestamps or
proof of a wall-versus-projectile cause.

The checked run made **7,566** updates, completed **20** boot games and
**2,138** restored segments, with **1,561** first-loss cuts. All **321** archive
offsets were 64 and the two boot-only workers stayed protected. Completed
restored lives contributed **89,143** actions. Its **80,649** post-warmup
exploratory steps contained **65.80%** movement and **8.22%** forward-fire
aliases, confirming the intended sampling change without showing a stage clear.
Every action remained sampled. Reset sources still came only from own play.

Full run **51** now continues that preserved calibration at **7,093,216**,
with [unchanged settings](results/defense/training/dqn-51-balanced-actions/resume-config.json),
unlimited learning and 200,000-action evaluations. This tests longer adaptation,
not a claimed solution from the short check. Uniform focused run 50 remains
available as a matched action-count comparison with its own calibration parent;
the full trajectories and RNG states differ. The sole collector now includes
all **47** full-trial sources, retaining every historical source and excluding
short calibrations. The shared verified 10,480-point stage-1 best is unchanged.

The first two full balanced evaluations, at **7,293,216 / 7,493,216**, average
**7,324 / 8,836**, versus uniform focused run 50's **9,340 / 9,641** at the
same action counts. Both are regressions, and all twenty games lose in stage 1.
The [first comparison](results/defense/training/dqn-51-balanced-actions/comparison-at-000007293216.json)
and [second comparison](results/defense/training/dqn-51-balanced-actions/comparison-at-000007493216.json)
retain all reused-seed results. Both full Q/target/Adam checkpoints and log
prefixes are preserved, along with the first run-best
[10,180-point replay](results/defense/training/dqn-51-balanced-actions/replay-10180/replay.html).
The balanced continuation is still a test, not evidence of improved navigation.

Its [third complete evaluation](results/defense/training/dqn-51-balanced-actions/comparison-at-000007693216.json),
at **7,693,216**, averages **9,235**, median **9,820**, best **9,890**,
versus uniform focused's mean **9,860**. All ten games remain stage-1 losses;
the full checkpoint and log prefix are preserved without a new replay best.

The remaining balanced-action rounds also stay in stage 1. At
**7,893,216 / 8,093,216 / 8,293,216**, means are **9,918 / 9,345 / 9,568**,
versus uniform focused's **10,066 / 10,180 / 9,955**. Complete paired records
and full checkpoint states are preserved for rounds
[four](results/defense/training/dqn-51-balanced-actions/comparison-at-000007893216.json),
[five](results/defense/training/dqn-51-balanced-actions/comparison-at-000008093216.json)
and [six](results/defense/training/dqn-51-balanced-actions/comparison-at-000008293216.json).
All six full-round means are lower; none of these sixty complete games reached
a later stage. This is negative evidence for the tested distribution change.

Run 51 then [retired gracefully](results/defense/training/dqn-51-balanced-actions/retirement.json)
at **8,315,744**, after **1,222,528** new actions, **137** complete boot games
and **23,060** completed restored segments. No training stage 2 or mission was
logged. Its final Q/target/Adam state, complete compressed log and verified
10,180-point full-run replay remain preserved, as does its separate 10,280-point
calibration replay. The collector retains this historical source.

The uniform control/focused pair subsequently completed rounds five and six.
At **8,093,216**, mean scores were **10,236 / 10,180**; at **8,293,216**,
**9,875 / 9,955**. Every game still lost in stage 1. Full checkpoint states,
log prefixes and [fifth](results/defense/training/dqn-50-restored-life-focused/comparison-at-000008093216.json)
and [sixth](results/defense/training/dqn-50-restored-life-focused/comparison-at-000008293216.json)
comparisons are preserved. Across all six rounds, focused minus control mean
differences were **−78, −191, −352, +248, −56, +80**. Concentrating restored
practice on the initial life did not produce a stage clear in this experiment.

Both retired gracefully after that depth plateau, not a wall-clock limit:
control at **8,466,104** and focused at **8,434,768**. The
[control retirement](results/defense/training/dqn-49-restored-life-control/retirement.json)
records **1,372,888** additional actions, **150** boot games and **1,246**
completed restored segments; the
[focused retirement](results/defense/training/dqn-50-restored-life-focused/retirement.json)
records **1,341,552**, **147** and **22,612**, respectively. Neither logged a
later training stage or mission. Final Q/target/Adam states, complete compressed
logs and all previous best replays remain available. The collector retains
both historical sources while run 51 continues.

### Is the pre-loss practice state already unresponsive?

The repeated barrier failures also raise a timing question: the displayed
ship loss might lag the decisive mistake. An extended
[static binary audit](results/defense/diagnostics/loss-delay-static-audit.json)
confirms separate overlap, loss and HUD-refresh routines. The loss routine
decrements its internal counter, waits, and draws later; HUD formatting is
also gated by a countdown. This is read-only analysis of the original CMD,
not a runtime hidden-state input, a collision timestamp or a policy rule.

`rl.defense_response_probe` tests the narrower question of input responsiveness.
It first reproduces an already verified own-policy replay exactly, then makes
isolated opaque copies at fixed offsets before each visible loss. Each of the
20 legal actions is tested for four decisions from the identical state.
Only aggregate screen diversity is reported: no preferred action, branch
score, route, demonstration, training/reset data or replay-ranking candidate.
Original neural verification is reused; the original native trajectory is
reexecuted and every frame, reward and ending is checked. Source hashes are
checked again afterward, and the snapshot payload is never decoded.

The [balanced calibration probe](results/defense/diagnostics/balanced-actions-response-calibration-02.json)
exactly reproduced **2,540** actions; the
[focused full-run probe](results/defense/diagnostics/focused-actions-response-full-7493216.json)
reproduced **2,568**. In both replays, all four lives showed different gameplay
graphics for different inputs at **128, 64 and 32 decisions before visible loss**.
At eight decisions, the first three lives showed identical graphics; the final
life still showed differences. All branches lasted the full four decisions
without a visible boundary. HUD text is excluded from this graphics comparison.

This rules out input-inactive animations for these particular 64-decision
anchors, but **does not prove recoverability, movement or obstacle passage**.
It also does not cover every live training reset. Conversely, identical graphics
do not prove death: effects can be delayed or visually hidden. The final-life
ending uses different settling, so these offsets are not aligned physical
collision times. No training settings changed because of this diagnostic.

Three new tests cover graphics/HUD separation, common-prefix comparison and
repeatable exact native replay without source mutation or training output;
all **18** focused environment/audit/probe tests and the
[full 353-test suite](results/defense/diagnostics/response-probe-regression-tests.txt)
passed. The initial diagnostic
report is retained with its [exact source](results/defense/diagnostics/response-probe-source-v1.py);
the linked second report additionally asserts restored native video equality
before each branch. Neither version writes training examples.

### Random survival diagnostic and lower-exploration practice

A follow-up input audit found the expected keyboard mapping: emulator masks
08/10/20/40 correspond to up/down/left/right, matching the original stage-one
comparisons at 75E4. The original routine at AC0E reads keyboard row 3840
before dispatch. This revealed no mapping mismatch and required no native,
game or environment change; it does not establish that any particular route
is feasible.

The [isolated random-survival probe](results/defense/diagnostics/focused-random-survival-7493216.json)
first reproduces all **2,568** actions of the focused run's verified replay.
It then tests **256** random rollouts from each of **12** opaque states:
256, 128 and 64 decisions before each visible loss. Each rollout ends at
its first visible loss or at offset + 256 decisions. Actions use the existing
fixed balanced distribution and power-law persistence up to 64 decisions,
with epsilon 1. Common random sequences are used across anchors; neither
screens nor branch outcomes choose a preferred action or sequence.

All **3,072** rollouts lost the current life before the diagnostic horizon;
none reached stage 2, and none outlasted the original loss time by more than
32 decisions. At the 64-decision anchors, **59 / 16 / 56 / 7** rollouts,
respectively, outlasted the original visible-loss time at all; maximum observed
durations were **69 / 82 / 69 / 67** decisions. At 128 decisions, just one
rollout lasted longer than the reference, by three decisions; at 256, none did.
Random actions often lost earlier, especially from the earlier starts.

This does **not** prove unavoidability: finite random sampling is not exhaustive,
the anchors are selected replay states, and decision counts are neither exact
collision times nor course positions. Actions can also change the amount of
emulated work per decision. Only aggregate duration/censoring/stage counts are
exported, not action sequences, scores, snapshots or trajectories. Nothing from
these diagnostic rollouts enters the learner, its reset archive or ranking.

This motivates a bounded lower-exploration check, not a claimed diagnosis or
fix. [Its configuration](results/defense/training/restored-life-low-exploration-calibration-01/resume-config.json)
returns to the **same original 6,962,144-action Q/target/Adam/RNG checkpoint**
as the high-exploration focused calibration. Only nominal practice-worker
epsilon changes from **.9 to .25**, aside from paths/provenance and an explicit
`uniform` sampler default. Previously verified exact disabled-sampler parity
covers the newer sampler code. Boot workers retain .05; own-loss lookback is
64, probability is 1, and restored segments end at first visible loss.
Eight workers, two protected boot workers, uniform action starts, persistence
up to 64, gamma .997, five-step returns and all learner settings are unchanged.

The check permits **131,072 new actions**, followed by ten ordinary complete
uncapped boot games on the same reused validation seeds. It loads no diagnostic
trajectories or evaluation snapshots. This tests whether retaining more learned
behavior during focused practice is useful; earlier low-epsilon experiments
used different reset/segment configurations and did not clear the barrier.
The short check remains excluded from the best-replay collector.

The check finished normally at **7,093,216**. Its
[ten complete games](results/defense/training/restored-life-low-exploration-calibration-01/comparison.json)
average **10,297**, median **10,310**, best **10,380**, versus high exploration's
**10,214 / 10,375 / 10,460**. The +83 mean difference is driven by one +1,490
recovery; the other nine paired scores decline by 10–200 points. All games
remain stage-1 losses. This is not evidence of reliable improvement or a
resolved navigation bottleneck.

Full Q/target/Adam state and the complete log are preserved with the
[2,509-action verified replay](results/defense/training/restored-life-low-exploration-calibration-01/replay/replay.html).
The [loss windows](results/defense/diagnostics/low-exploration-calibration-losses-01/report.json)
show life totals **[2600, 2580, 2580, 2620]**, still around the familiar barrier
sequence, with some earlier failures. No physical collision timestamps or
specific steering instructions are inferred from these panels.

The [practice audit](results/defense/training/restored-life-low-exploration-calibration-01/audit.json)
records **7,566** updates, **20** complete boot games, **1,342** completed
restored segments, **1,037** first-loss cuts and **608** exact-64 archive events.
Protected workers stayed boot-only. Completed restored initial lives account
for **88,445** decisions, versus the high-exploration check's **89,015** across
2,200 segments: fewer, longer segments, not proven obstacle passage.
Exploration occupied **20,183 / 121,040** post-warmup decisions (**16.67%**
across all workers), consistent with lower practice epsilon, the protected
workers' .05 and cancellation at boundaries. All 20 actions were sampled.

Full run **52** continues this checked calibration's optimizer with
[unchanged settings](results/defense/training/dqn-52-low-exploration-focused/resume-config.json),
unlimited training and evaluations every 200,000 actions. Historical focused
run 50 supplies the same-action-count comparison; its own calibration and
later random trajectories differ. The sole collector now retains all **48**
full-trial sources, excluding short calibrations and every diagnostic branch.
The stronger shared 10,480-point replay and model remain unchanged.

Two diagnostic tests cover aggregate survival/censoring calculations, exact
native replay reproduction, repeatability, source checks and absence of
exported action sequences or training data. The
[full 355-test suite](results/defense/diagnostics/survival-probe-regression-tests.txt)
passed. No learner implementation, policy input, reward or original-game byte
changed in this experiment.

Full run 52 subsequently completed six ten-game rounds, all stage-1 losses.
At **7,293,216 / 7,493,216 / 7,693,216 / 7,893,216 / 8,093,216 / 8,293,216**,
its means were **9,780 / 10,220 / 9,986 / 9,944 / 10,170 / 10,134**. Relative
to high-exploration run 50 at the same action counts, the differences were
**+440 / +579 / +126 / −122 / −10 / +179**. These score changes did not produce
a later stage. All full states, log prefixes and paired records are preserved,
including the [first](results/defense/training/dqn-52-low-exploration-focused/comparison-at-000007293216.json)
and [sixth](results/defense/training/dqn-52-low-exploration-focused/comparison-at-000008293216.json)
comparisons. Its run-best [10,260-point replay](results/defense/training/dqn-52-low-exploration-focused/replay-10260/replay.html)
verified **2,530** base actions; the earlier 9,780 replay verified **2,497**.

The run [retired gracefully](results/defense/training/dqn-52-low-exploration-focused/retirement.json)
at **8,369,808**, after **1,276,592** additional actions, **141** complete boot
games and **17,343** completed restored segments. Training also logged no
stage 2 or mission. Final Q/target/Adam state and the complete compressed log
are preserved; the collector retains the historical source. This retirement
reflects a depth plateau, not a wall-clock limit.

### Learning how long to hold an action

The next experiment changes what the policy can learn, rather than merely
changing random exploration frequency. Inspired by
[Dynamic Frame-skip DQN](https://arxiv.org/abs/1605.05365), optional
`--learned-repeats 1,4,16,64 --n-step 1` gives the screen network one joint
Q-value per ordinary action and permitted duration. With 20 keys and four
durations this is 80 learned options; no direction, obstacle or route is
preferred by an execution rule. This is an adaptation, not a reproduction of
the paper's architecture, emulator or experimental protocol.

The original emulator and base action length stay unchanged. At a decision,
the network chooses a key combination and hold length; its executor counts
down that learned choice. Screen history continues updating at every base
action, so the next input is the same four recent raw video frames, not four
widely spaced macro frames. Greedy evaluation uses the same learned options.
Only visible life/episode boundaries cancel holds, consistently in training,
parallel evaluation and serial replay verification. No hidden game state,
oracle rollout, demonstration, extra observation or steering controller is used.

Each completed or boundary-interrupted option enters prioritized replay once,
with its actual observed discounted score sum and bootstrap multiplier
`gamma ** executed_base_actions`. The bootstrap is zero at a learning terminal;
time-limit truncation retains the actual-duration bootstrap. Double-Q chooses
the next joint option. No unexecuted outcomes or intermediate counterfactual
options are fabricated. Native games remain uncapped in performance evaluation.
The initial experiment uses one-option targets, life-terminal learning, and no
other auxiliary, distributional or persistent-random-action variant.

Exploration epsilon now applies **per option start**, not per base action.
The random choice is uniform over joint options. Actual time spent exploring
can differ substantially across duration sets; it must not be interpreted as
a matched exploratory-step fraction. Warmup still uses independent uniform
one-base-action choices. All logged training/evaluation budgets count base
actions, not the smaller number of option decisions.

`--init-from-dqn` transfers only the learner's own scalar encoder/value and
tiles its advantage head across durations. Target initially equals transferred
online; Adam, RNG, counters, replay and reset archives are **fresh**. Initial
duration-value equality is an initialization, not a claim that holding every
action longer really earns the same return. Exact ties initially choose the
shortest duration. Resuming a duration model restores its full Q/target/Adam/RNG
with the same duration set, while new boot games clear pending holds/replay.

Seven focused tests cover option execution, game-identity isolation, visible
boundaries, actual-duration discounts, one-step equivalence, parameter transfer,
learning/resume, CLI rejection and native parallel/serial/reloaded replay parity.
The [full 362-test suite](results/defense/diagnostics/learned-repeat-regression-tests.txt)
passes. A [4,096-action disabled-feature check](results/defense/diagnostics/learned-repeat-default-parity.json)
reproduced online/target bytes, all 26 optimizer arrays, nonconfiguration state
and **67** game/archive events exactly. The existing live learner was not
changed by this optional implementation.

The [initialization audit](results/defense/diagnostics/learned-repeat-initialization.json)
checks every copied/tiled parameter and fresh optimizer state. Both `(1,)` and
`(1,4,16,64)` reproduced the parent's complete **2,523-action, 10,430-point**
game exactly before any learning. That is initial compatibility on one seed,
not successful learned use of longer actions. Those diagnostic trajectories
never enter training. The separate
[16,384-action production smoke](results/defense/diagnostics/learned-repeat-production-smoke.json)
made **400** updates, completed **8** boot games and **33** restored segments,
and produced **58** exact-64 archive events with protected workers intact.
It executed **11,051** options across 16,384 base actions, completing 11,044
and interrupting 37; pending options are discarded at the normal stop. This
check had no performance evaluation and is not a calibration parent.

Two new calibrations start from the same original own scalar checkpoint at
6,962,144, with the fresh-state transfer described above. Their configurations
are identical except paths and the duration set:
[one-step control](results/defense/training/repeat-control-calibration-01/config.json)
versus [learned variable duration](results/defense/training/repeat-variable-calibration-01/config.json).
Both get **131,072 new base actions** and ten complete reused-seed boot games,
with eight workers, two protected boot workers, .25/.05 option-start epsilon,
gamma .997, batch 64, Adam 1e-4, compact capacity 200,000, warmup 10,000,
updates every 16 base actions, own-loss lookback 64 and first-restored-life-only
practice. Epsilon reaches its fixed value immediately after warmup. This matched
pair isolates learned duration availability; older five-step optimizer-resume
runs are not exact controls. Neither calibration is in the collector.

Both calibrations completed normally. The
[paired comparison](results/defense/training/repeat-variable-calibration-01/comparison.json)
gives mean/median/best **344 / 340 / 440** for the one-step control and
**3,631 / 3,680 / 7,550** for variable durations. All ten variable-duration
scores are higher, for a mean difference of **+3,287**, but both remain well
below the strong original parent and every game loses in stage 1. This supports
longer adaptation of the new option formulation, not a claim of improvement
over the preserved best or a demonstrated solution to the barrier.

Both full Q/target/Adam checkpoints and complete logs are preserved, with
independently reloaded/native-verified replays of
[1,770 control base actions](results/defense/training/repeat-control-calibration-01/replay/replay.html)
and [2,394 variable-duration base actions](results/defense/training/repeat-variable-calibration-01/replay/replay.html).
Verification includes the learned option choices, held outputs and visible
boundary resets; it does not pretend every held base action is a fresh neural
inference. The stronger shared model/replay is unchanged.

Each calibration made **7,568** updates and completed **24** boot games.
The [control audit](results/defense/training/repeat-control-calibration-01/audit.json)
records **1,748** restored segments and **88,985** initial-life base actions;
the [variable audit](results/defense/training/repeat-variable-calibration-01/audit.json)
records **1,351** and **88,993**, respectively. Their **546 / 582** archive
events have correct offsets and reserved workers stayed boot-only. Variable
duration executed **24,293 / 22,437 / 40,684 / 43,658** base actions under
1/4/16/64-step options, respectively, including interrupted holds. It started
33,738 options, completed 33,732 and interrupted 1,219 at boundaries. These
counts confirm the mechanism is used, not that a particular maneuver was learned.

Full trials [53 (control)](results/defense/training/dqn-53-repeat-control/resume-config.json)
and [54 (variable)](results/defense/training/dqn-54-learned-repeat/resume-config.json)
now continue their own calibrated optimizer states with unlimited learning and
200,000-base-action evaluations. Fresh boots refill their own replay and reset
archives, with no diagnostic/evaluation examples loaded. The sole collector was
restarted with tested duration-policy support and all **50** full-trial sources;
historical sources remain, and short calibrations stay excluded.

#### First full rounds and read-only duration review

The full continuations completed five paired rounds, each with ten
uncapped boot games on the same reused evaluation seeds. Every game still
loses in stage 1; these are not fresh success-rate estimates.

| Base actions since fresh initialization | One-step mean / best | Variable-duration mean / best |
| --- | --- | --- |
| [331,072](results/defense/training/dqn-54-learned-repeat/comparison-at-000000331072.json) | 377 / 610 | 1,800 / 5,180 |
| [531,072](results/defense/training/dqn-54-learned-repeat/comparison-at-000000531072.json) | 342 / 400 | 9,016 / 9,800 |
| [731,072](results/defense/training/dqn-54-learned-repeat/comparison-at-000000731072.json) | 1,328 / 2,180 | 8,379 / 9,980 |
| [931,072](results/defense/training/dqn-54-learned-repeat/comparison-at-000000931072.json) | 1,240 / 1,430 | 7,362 / 10,240 |
| [1,131,072](results/defense/training/dqn-54-learned-repeat/comparison-at-000001131072.json) | 516 / 610 | 5,613 / 8,450 |

All ten Q/target/Adam checkpoints, evaluation records, compressed log prefixes
and own-reset/option-accounting audits are preserved. The new variable-duration
replays verify **2,102**, **2,404** and **2,478** base commands, respectively;
the control's two published records verify **1,802** and **1,996**. The
[9,980-point replay](results/defense/training/dqn-54-learned-repeat/replay-9980/replay.html)
was superseded by the fourth round's 10,240-point run-best, not the stronger shared best. Neither evaluation nor
training logs through these checkpoints record stage 2 or a mission. No
diagnostic examples entered their replay buffers. The fourth-round
[10,240-point replay](results/defense/training/dqn-54-learned-repeat/replay-10240/replay.html)
verified **2,471** base commands.

Both trials retired gracefully after this depth plateau, not a wall-clock
limit. [Control 53](results/defense/training/dqn-53-repeat-control/retirement.json)
stopped at **1,229,144**, after **1,098,072** new actions, **165** complete boot
games and **17,312** restored segments. [Variable 54](results/defense/training/dqn-54-learned-repeat/retirement.json)
stopped at **1,250,896**, after **1,119,824** new actions, **166** boot games
and **17,434** restored segments. Full final Q/target/Adam states and complete
compressed logs are preserved. Neither training log records a later stage.
Their historical collector sources remain available.

The new read-only `rl.defense_repeat_probe` reconstructs every recorded base
command with frozen weights, the original visible history and visible life
boundaries. It checks source hashes, option countdowns and exact
planned/executed/cancelled accounting. It exports aggregate duration counts,
not actions to imitate or training examples. Two new tests cover frozen
reproduction/source integrity and window-edge accounting; the
[full 364-test suite](results/defense/diagnostics/learned-repeat-probe-regression-tests.txt)
passed in **354.014 seconds**. This probe reuses original native verification;
it is not another native evaluation.

For the [7,550-point calibration replay](results/defense/diagnostics/learned-repeat-use-calibration-01.json),
743 neural option decisions reproduce 2,394 base commands; **76.65%** of
commands belong to options longer than one step. For the first full
[5,180-point replay](results/defense/diagnostics/learned-repeat-use-full-331072.json),
676 decisions reproduce 2,102 commands, with **75.69%** belonging to longer
options. Near the visible losses, most commands instead belong to one- or
four-step options. Window counts do not label the physical collision time.

The [visible introduction-text breakdown](results/defense/diagnostics/learned-repeat-intro-use-01.json)
guards against mistaking waiting-screen holds for learned navigation. Of
1,232 base commands under 16-step options in each variable-duration replay,
**552** in the calibration and **577** in the first full replay occur with
recognized introduction text in the pre-action frame. Frames without that
text can still contain waits or animations: the complement is **not** an
active-gameplay label. This classification is diagnostic only and is not
fed to either learner. Long-hold usage alone therefore does not demonstrate
better obstacle handling.

Finally, [original-screen loss panels](results/defense/diagnostics/learned-repeat-full-losses-731072/README.md)
compare the shared best with the newest 9,980 replay. The latter earns
**2,500 / 2,450 / 2,600 / 2,430** per life, versus the best's four identical
2,620 totals. Both show the recurring right-opening barrier sequence, with
the ship still left of that opening in approach frames. This supports the
user's navigation-bottleneck observation, not identical collision positions
or a proven wall-versus-projectile cause. The experiment has recovered some
score but has not demonstrated passage through the bottleneck.

#### Optional multi-option returns

The one-option formulation removes the earlier scalar DQN's five-step return.
The short-duration control's large regression makes that an important
limitation to test; it does not prove credit assignment caused the barrier
failure. Optional `--repeat-n-step 5` now combines up to five **completed
options**, not five base actions. It still requires `--n-step 1` to keep the
ordinary DQN return setting distinct. Values 1..32 are supported; 1 preserves
the existing implementation, and values above 1 require learned durations.

`rl.defense_option_returns.MultiOptionReturns` queues only transitions produced
by the unchanged `OptionReturns`. It composes actual discounted score sums
using the product of each option's `gamma ** actual_duration`. It emits one
return per actual option start, including shorter suffixes at visible life
or episode boundaries. Learning terminals give zero bootstrap; truncation
retains its actual-duration bootstrap. No return crosses a boundary. No
shorter action-duration counterfactuals, extra rewards, hidden observations
or demonstration trajectories are generated. This uses uncorrected own
behavior multi-step returns, like the ordinary n-step baseline, not a claim
of unbiased off-policy targets.

Acting, four-frame history, hold cancellation and replay verification are
unchanged. Resume restores model/target/Adam/RNG but discards unfinished
options and queued return starts along with the existing replay refill.
Checkpoints separately count completed options, emitted returns and queued
starts; emitted plus queued must equal completed. The new return source hash
is recorded when enabled. The former 53/54 jobs retained their original code
and one-option settings until graceful retirement.

Five new tests cover actual-duration composition, interrupted options,
terminal/truncation flushing without cross-life leakage, observation ownership,
exact single-duration equivalence to ordinary five-step returns, single-option
equivalence, invalid CLI inputs, native learning/resume, parallel complete
games, serial recording and independent policy verification. All **14** focused
new/existing duration and diagnostic tests pass. The
[16,384-action default-path parity check](results/defense/diagnostics/multi-option-default-parity.json)
made **400 optimizer updates** and exactly reproduced Q/target bytes, all
26 optimizer arrays, all nonconfiguration state and 104 game/archive events.
The [full 369-test suite](results/defense/diagnostics/multi-option-regression-tests.txt)
also passed in **537.981 seconds**. The parity report records the executed old trainer's Git source hash separately from
its runtime file-hash metadata. The first 4,096-action check stayed inside
warmup, so the longer check is the evidence for learning parity.

Matched five-option calibrations use the same original own scalar checkpoint,
fresh transferred target/Adam/RNG initialization, 131,072 base actions and
ten complete reused-seed boot evaluations as the one-option calibrations.
Only the allowed durations differ between arms: `(1,)` and `(1,4,16,64)`.
Both retain the earlier eight-worker/two-protected-worker settings and own
loss-lookback practice. They are excluded from the global replay collector.
Their purpose is to test the credit-horizon hypothesis, not to claim a
navigation improvement from passing implementation tests.

Both calibrations finished normally. Their
[paired comparison](results/defense/training/repeat-five-variable-calibration-01/comparison.json)
gives control mean/median/best **6,365 / 5,390 / 8,660** and variable-duration
**6,266 / 6,060 / 7,040**. Variable duration is higher on six of ten reused
seeds but lower by **99** on mean. Every game remains a stage-1 loss. Relative
to the earlier one-option calibrations, means improve by **6,021** and
**2,635**, respectively; this supports testing the return horizon further,
not a claim of solving the barrier or beating the strong original parent.

Both full checkpoints, configurations and complete compressed logs are
preserved, with [control](results/defense/training/repeat-five-control-calibration-01/replay/replay.html)
and [variable](results/defense/training/repeat-five-variable-calibration-01/replay/replay.html)
replays verifying **2,481 / 2,409** base commands. Each made **7,566** updates.
The [control audit](results/defense/training/repeat-five-control-calibration-01/audit.json)
records **20** boot games, **1,606** restored segments, **88,456** initial-life
actions and **427** correct-offset archive events. The
[variable audit](results/defense/training/repeat-five-variable-calibration-01/audit.json)
records **24 / 1,438 / 89,106 / 582**, respectively. Protected workers stayed
boot-only. Emitted/queued return counts are **131,040 / 32** and
**36,466 / 24**; their sums equal completed options. Unfinished queues are
discarded on resume, not treated as fabricated terminals.

The [recorded variable replay panels](results/defense/diagnostics/multi-option-calibration-losses-01/policy-2-losses.png)
show failure earlier in the approach sequence on some lives, with per-life
totals **1,700 / 1,820 / 1,700 / 1,820**. Screen animations and delayed visible
life changes prevent identifying exact collisions from these panels. This
is not evidence of passing the recurring barrier. No diagnostic trajectories
are added to learning.

Full trials [55 (control)](results/defense/training/dqn-55-repeat-five-control/resume-config.json)
and [56 (variable)](results/defense/training/dqn-56-repeat-five-variable/resume-config.json)
continue their respective calibration optimizers with unlimited learning and
200,000-base-action evaluations. Fresh boots refill their own replay and reset
archives. The sole collector now includes **52** full-trial sources; the short
calibrations remain excluded and the stronger shared best is unchanged.

Their [first full comparison at 331,072](results/defense/training/dqn-56-repeat-five-variable/comparison-at-000000331072.json)
gives control mean **9,128**, best **10,200**, versus variable-duration mean
**358**, best **420**. All twenty complete games lose in stage 1. The
[control replay](results/defense/training/dqn-55-repeat-five-control/replay-10200/replay.html)
verifies **2,520** base commands and the
[variable replay](results/defense/training/dqn-56-repeat-five-variable/replay-420/replay.html)
verifies **1,774**. Both full optimizer states, log prefixes and return/reset
accounting audits are preserved. This first round reverses neither the
global plateau nor the variable arm's regression; both trials continue.

At [531,072](results/defense/training/dqn-56-repeat-five-variable/comparison-at-000000531072.json),
the second full round gives control mean/median/best **9,502 / 9,800 / 9,820**
and variable-duration **1,136 / 340 / 5,080**. All twenty games remain stage-1
losses. Both full states and log-prefix audits are preserved; the variable
arm's new [5,080 replay](results/defense/training/dqn-56-repeat-five-variable/replay-5080/replay.html)
verified **2,024** base commands. The control retains its earlier 10,200
run-best. Neither run changes the global best.

At [731,072](results/defense/training/dqn-56-repeat-five-variable/comparison-at-000000731072.json),
the third full means are **9,484 / 1,701** for control/variable, with best
scores **10,080 / 5,260**, again all stage 1. Full checkpoints, log prefixes
and accounting audits are preserved. The variable arm's
[5,260 replay](results/defense/training/dqn-56-repeat-five-variable/replay-5260/replay.html)
is verified separately; it does not replace the stronger shared best.

The [fourth round at 931,072](results/defense/training/dqn-56-repeat-five-variable/comparison-at-000000931072.json)
regresses to means **4,280 / 1,271**, best **7,650 / 4,470**, all stage 1.
Both full states and log prefixes are preserved. After four rounds without
stage progression or improvement over the shared best, compute was reassigned
to the matched stability experiment below. This is not a wall-clock stop or
a proof that further training could never help.

[Control 55 retired gracefully](results/defense/training/dqn-55-repeat-five-control/retirement.json)
at **966,848**, after **835,776** new actions, **96** boot games and **11,470**
restored segments. [Variable 56 retired](results/defense/training/dqn-56-repeat-five-variable/retirement.json)
at **998,448**, after **867,376** new actions, **141** boot games and **11,088**
restored segments. Final online/target/Adam states and complete compressed logs
are preserved, with no training stage 2 or mission. Run-best replays remain
10,200 and 5,260, respectively; historical collector sources are retained.

#### Frozen option-return error scale

Before attributing failure to the game's uneven award sizes, the read-only
`rl.defense_td_probe` reconstructs a selected verified replay's actual option
starts, executed durations and one/five-option returns. Every base command
must match the frozen policy. It loads the **matching saved target network**,
checks configuration/step/online-model identity and source hashes, then forms
the usual frozen Double-Q labels in scaled score units. Visible life boundaries
flush all return starts; every completed option must produce one diagnostic
row. Neither these rows nor the recorded trajectories enter training.

| Selected replay | Option-return rows | Mean absolute TD error | Fraction in linear Huber region | Unit-weight loss share from zero observed returns |
| --- | --- | --- | --- | --- |
| [Five-option control calibration](results/defense/diagnostics/multi-option-td-control-calibration-01.json) | 2,481 | 0.323 | 3.55% | 86.30% |
| [Five-option variable calibration](results/defense/diagnostics/multi-option-td-variable-calibration-01.json) | 1,284 | 1.481 | 36.92% | 82.92% |
| [Control full 331,072](results/defense/diagnostics/multi-option-td-control-full-331072.json) | 2,520 | 0.328 | 4.76% | 89.62% |
| [Variable full 331,072](results/defense/diagnostics/multi-option-td-variable-full-331072.json) | 614 | 0.813 | 22.64% | 87.35% |

The variable arm's selected full replay has lower mean TD error than its
calibration replay despite much worse score. Neither small residuals nor
lower diagnostic loss establish successful play. In all four traces, most
unit-weight Huber loss belongs to returns whose observed reward sum is zero,
not positive-award returns. This does **not** support the narrow explanation
that large observed awards dominate this particular diagnostic loss.
Zero observed reward does not mean zero bootstrap, and these groups are not
obstacle or collision labels.

The caveats matter: these are selected greedy boot replays, not samples of
the exploratory, restored-state, prioritized training distribution. Unit-weight
loss shares do not measure PER weighting or parameter-gradient contributions.
Saved-target labels are bootstrapped estimates, not ground-truth returns.
This analysis cannot rule out reward-scale problems or establish their cause.
The probe exports aggregate counts only, preserves all source files, reuses
the original native verification and makes no parameter updates.

Two focused tests pass, covering Huber arithmetic/group accounting, degenerate
losses, source integrity, exact one/five-option command reconstruction,
complete-boundary flushing and rejection of a mismatched target checkpoint.
The [full 371-test suite](results/defense/diagnostics/multi-option-td-regression-tests.txt)
passed in **382.737 seconds**. No learner or acting implementation changed
during this diagnostic review.

For possible future experiments, [Pohlen et al., section 3.2](https://arxiv.org/html/1805.11593#S3.SS2)
describe transforming value targets while keeping rewards unaltered. Section
3.3 separately proposes temporal-consistency regularization. Neither was
implemented at this diagnostic milestone; the following section records the
later isolated consistency experiment. Their full algorithm also uses expert
demonstrations, which remain excluded by this project's rules. Current live
training, reward scaling and acting policies are unchanged by this diagnostic.

#### Temporal-consistency stability experiment

The next optional scalar-DQN experiment adapts the temporal-consistency term
from [Pohlen et al., section 3.3](https://arxiv.org/html/1805.11593#S3.SS3).
It penalizes changes to the online next-state value used for bootstrapping,
relative to the saved target network. This is a stability hypothesis, not a
diagnosis proven by the selected-replay error audit. The paper's demonstration
system and imitation loss are not used. Nor is its nonlinear value transform
implemented here.

`--tc-weight 1` adds PER-importance-weighted Huber loss between online and
stopped target values for the **online-greedy** action at the n-step bootstrap
screen. The discrete argmax is explicitly stopped; gradients flow through
the selected online value only. Our adaptation masks rows with zero bootstrap
discount, so visible life/terminal transitions do not constrain unused future
values. Time-limit truncations retain their ordinary bootstrap behavior.
Loss is averaged over the whole batch, not renormalized by the active count.

Ordinary Double-Q labels, actual score rewards, return construction, replay
sampling and acting are unchanged. Priorities remain **absolute TD errors**,
not the auxiliary residual. Both losses share the existing clipped-gradient
Adam update; there are no extra networks, optimizer files or acting inputs.
Full Q/target/Adam/RNG resume stays compatible with scalar checkpoints; local
diagnostic counters restart. Weight zero retains the original `Learner` path.
The option is deliberately not combined with learned durations, quantile or
bootstrap heads, trace cutting, SPR or inverse-action learning.

The first candidate tests caught an MLX gradient-through-index error. It was
fixed by explicitly stopping the discrete argmax before any calibration was
launched; the [failed candidate test log](results/defense/diagnostics/consistency-initial-test-failure.txt)
is retained. All [five corrected focused tests](results/defense/diagnostics/consistency-focused-tests.txt)
pass, covering weighted loss arithmetic, terminal masks, stopped target/index
gradients, exact terminal-only TD-gradient equivalence, unchanged priorities,
fixed targets/synchronization, invalid settings, native learning/resume and
complete parallel/serial/reloaded greedy replay compatibility.

A [16,384-action disabled-feature check](results/defense/diagnostics/consistency-default-parity.json)
made **398** optimizer updates and exactly reproduced online/target bytes,
all **26** optimizer arrays, all nonconfiguration state and **201**
game/archive events. The two existing five-option full trials were not changed.
The [full 376-test suite](results/defense/diagnostics/consistency-regression-tests.txt)
passed in **492.895 seconds**.

The new matched calibrations resume the same original own scalar full state
at **6,962,144**, retain five-base-action returns, and receive **131,072** new
actions each. Both use discount **.999**, eight workers/two protected boot
workers, .25/.05 persistent exploration, own-loss lookback 64, probability-one
first-restored-life practice, compact replay 200,000, batch 64 and Adam 1e-4.
Only consistency weight **0 versus 1** and its provenance/output paths differ.
They preserve the parent's optimizer and target rather than using the fresh
duration-head initialization. Thus older .997 or differently initialized
experiments are not isolated controls for this comparison. Evaluation remains
ten complete uncapped boot games on reused seeds, and both calibrations stay
outside the global replay collector.

Both calibrations finished normally. The
[paired comparison](results/defense/training/consistency-enabled-calibration-01/comparison.json)
gives control mean/median/best **9,710 / 10,200 / 10,220** and consistency
**10,266 / 10,280 / 10,280**. All twenty complete boot games still lose in
stage 1. Nine consistency scores are higher and one ties; the **+556** mean
difference is mostly driven by two **+2,530** recoveries, while the other
differences are 0..80. This is stronger short-check scoring, not evidence of
passing the barrier, exceeding the shared best, or a fresh success-rate gain.

Both full Q/target/Adam checkpoints, configurations and complete compressed
logs are preserved. [Control](results/defense/training/consistency-control-calibration-01/replay/replay.html)
and [consistency](results/defense/training/consistency-enabled-calibration-01/replay/replay.html)
replays independently verify **2,549 / 2,517** base actions. The
[consistency loss panels](results/defense/diagnostics/consistency-calibration-losses-01/policy-2-losses.png)
show the recurring right-opening barrier sequence, with the ship left of
the opening in approach frames and **2,570** points on every life. Visible
flashes/life counters are not exact collision timestamps, and the panels
do not establish wall-versus-projectile deaths. No diagnostic examples enter
learning.

Each calibration made **7,566** optimizer updates and completed **20** boot
games. The [control audit](results/defense/training/consistency-control-calibration-01/audit.json)
records **1,384** restored segments, **88,592** initial-life actions and **602**
correct-offset archive events. The [consistency audit](results/defense/training/consistency-enabled-calibration-01/audit.json)
records **1,418 / 88,527 / 598**, respectively. Protected workers remained
boot-only. Final sampled consistency statistics are weighted TD loss **.02882**,
unscaled-by-coefficient consistency loss **.002643**, masked mean next-value
gap **.13417**, and **93.75%** bootstrapping rows. Those sampled losses are
implementation diagnostics, not measurements of navigation ability.

Full trials [57 (longer-horizon control)](results/defense/training/dqn-57-long-horizon-control/resume-config.json)
and [58 (temporal consistency)](results/defense/training/dqn-58-temporal-consistency/resume-config.json)
now continue their respective checked optimizers with **unlimited** training
and 200,000-action complete-game evaluations. The sole collector includes all
**54** full-trial sources, including the retired histories; neither short
calibration is eligible. The verified shared 10,480 best remains unchanged.

To reproduce this matched calibration, use the scalar lower-exploration
command below with `--gamma .999 --tc-weight 1`; use weight 0 and separate
paths for its control. Resume the corresponding preserved calibration
checkpoint with `--steps 0 --eval-every 200000` for the full continuation.

Reproduce the five-option calibration with the command below plus
`--repeat-n-step 5`; use `--learned-repeats 1` and separate paths for its control.

```sh
# Use --learned-repeats 1 and separate paths for the matched control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-learned-repeat-reproduction \
  --artifacts runs/defense-learned-repeat-reproduction/artifacts \
  --init-from-dqn results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --learned-repeats 1,4,16,64 --n-step 1 --steps 131072 --eval-every 131072 \
  --envs 8 --capacity 200000 --compact-replay --epsilon-final .25 --epsilon-steps 1 \
  --curriculum-boot-epsilon .05 --curriculum-probability 1 --curriculum-share \
  --curriculum-boot-envs 2 --curriculum-lookback 64 --curriculum-trigger life-loss \
  --curriculum-restored-life-only
```

To reproduce the earlier scalar lower-exploration calibration (not the
learned-duration experiment):

```sh
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-low-exploration-reproduction \
  --artifacts runs/defense-low-exploration-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --envs 8 --eval-envs 8 \
  --capacity 200000 --compact-replay --epsilon-final .25 \
  --curriculum-boot-epsilon .05 --curriculum-probability 1 \
  --curriculum-share --curriculum-boot-envs 2 --curriculum-lookback 64 \
  --curriculum-trigger life-loss --curriculum-restored-life-only \
  --spr-weight 0 --inverse-weight 0 --exploration-actions uniform
```

### Quantile score-return experiment

The scalar critic diagnostics did not establish gross Q inflation near loss,
and archive/exploration changes have not yet passed the recurring barrier.
An optional `--quantiles 32` now tests a different value representation:
fixed midpoint return quantiles using the pairwise quantile Huber loss from
[QR-DQN](https://arxiv.org/abs/1710.10044). The shared screen CNN feeds dueling
value/advantage heads per quantile. Loss sums over predicted quantiles and
averages over target quantiles, with Huber threshold 1. Double-Q selects the
next action by the online **mean** and obtains its quantiles from the target.
Existing score scaling, life-terminal n-step returns, Adam, gradient clipping
and own-experience replay remain. PER uses unweighted per-transition quantile
Huber loss divided by the quantile count, not a signed mean residual that can
cancel. This is an adaptation, not an exact paper reproduction.

Optional `--quantile-exploration-power 1.5` uses the positive power distortion
described in [IQN's risk-sensitive experiments](https://arxiv.org/abs/1806.06923)
for **training action selection only**. For a fixed quantile bin `[a,b]`, its
weight is `b**(1+power)-a**(1+power)`. Zero uses the ordinary mean. Values retain
their trained quantile indices; predictions are not sorted at action time.
Higher quantiles get more weight, but their spread is return variability, not
epistemic uncertainty or a confidence bound. Risk-seeking is not guaranteed to
help; the paper reports failures too. Unlike its IQN experiments, this model
has fixed quantiles and keeps both Bellman action selection and all evaluation
risk-neutral. There is no risk bonus, intrinsic reward, route or hidden input.

`--init-from-dqn` optionally transfers an own compatible scalar Defense model:
the encoder is copied, scalar dueling heads are tiled into initially coincident
quantiles, and target equals transferred online. Adam, RNG, counters and replay
start fresh; this is **not optimizer resume**. No fabricated return spread is
added. Compatibility includes game hash, screen history, action profile and
action timing; source hashes are checked before/after loading. Ordinary
quantile `--resume` restores its full optimizer/target/RNG and retains parent
provenance. Quantile count cannot change on resume. Bootstrap heads and
persistent action holds are rejected in this experimental mode.

The [default-path regression](results/defense/diagnostics/quantile-default-parity.json)
preserves byte-identical scalar online/target files, all **26** Adam arrays,
counters and RNG on a small native run. Its deliberately truncated games are
only plumbing evidence. A read-only
[own-model transfer check](results/defense/diagnostics/quantile-own-transfer-parity.json)
reproduces all **2,564** actions of the selected parent trace with both mean and
distorted selection before training. No trace is supplied to learning, and
this check is not a new emulator playthrough. All **nine** focused tests and
the full **289-test** suite pass, including analytic loss/gradient checks,
mean-versus-risk separation, exact optimizer resume and native initialization.

The matched neutral/risk calibrations initialize from run 33's
**6,962,144** checkpoint (mean **10,259**, best **10,480**) with 32 quantiles,
eight workers, compact capacity 200,000, batch 64, learning rate 1e-4, gamma
.997, n-step 5, life terminals, 100,000 T-states/action and history stride 1.
Both use 10,000-entry uniform warmup followed by 5% independent random actions
(`epsilon-steps=1`), no own-state resets and no persistent holds. Each permits
**131,072 new actions**, then ten complete mean-greedy boot games on reused
seeds 10000–10009. Only training power (0 versus 1.5) and output paths differ.
They have identical transferred online/target weights and fresh optimizer/RNG,
not the parent's scalar Adam. They are excluded from the global collector.
Neither finite calibration nor its initial action parity is a stage clear.

Both calibrations have finished normally at **131,072** new actions, with
**7,566** updates each. The [paired result](results/defense/training/quantile-risk-calibration-01/comparison.json)
does not favor risk distortion:

| Training selection | Mean | Median | Best | Verified mean-greedy replay |
| --- | ---: | ---: | ---: | --- |
| Neutral quantile mean | 10,280 | 10,280 | 10,280 | [2,592 actions](results/defense/training/quantile-neutral-calibration-01/replay/replay.html) |
| Power-distorted quantiles | 10,146 | 10,200 | 10,240 | [2,538 actions](results/defense/training/quantile-risk-calibration-01/replay/replay.html) |

Risk minus neutral is **−134**, with all ten paired scores lower. Both lose
every game in stage 1. They completed **55 / 56** boot training games, no
restored segments, and no logged later-stage episode. Full online/target/Adam/
RNG checkpoints, compressed logs and manifest-checked verified replays are
preserved. The [read-only loss windows](results/defense/diagnostics/shared-loss-quantile-calibration-01/report.json)
show life scores **2,570 × 4** for neutral and **2,570 / 2,570 / 2,550 / 2,550**
for risk; score alone does not identify a collision or establish passage.
Neutral's mean is 21 above the scalar parent's reused validation mean, but
both best scores are lower. This is not independent testing or a stage clear.

Full [DQN 37 neutral](results/defense/training/dqn-37-quantile-neutral/resume-config.json)
and [DQN 38 risk](results/defense/training/dqn-38-quantile-risk/resume-config.json)
continue each arm's own **131,072** full state, with unlimited learning and
200,000-action evaluation intervals. First full evaluations are due at
**331,072**. Unlike scalar initialization, this restores the quantile Adam,
target and RNG; replay refills from newly booted experience. Neither arm's
settings changed after seeing calibration results. This longer comparison
tests durability and the distributional learner, not a claimed advantage of
risk-seeking. Trials 35/36 continue separately.

The previous collector exited cleanly before the new sole collector started
with [all 34 full-run sources](results/defense/training/dqn-37-quantile-neutral/collector-config.json)
and the quantile-compatible loader. Calibrations remain excluded. A global
replacement still requires complete-game rank improvement and independent
frozen action/screen/reward verification. The shared 10,480-point replay is
unchanged.

```bash
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-quantile-risk-reproduction \
  --artifacts runs/defense-quantile-risk-reproduction/artifacts \
  --init-from-dqn results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --quantiles 32 --quantile-exploration-power 1.5 \
  --capacity 200000 --compact-replay --epsilon-steps 1 --epsilon-final .05 \
  --steps 131072 --eval-every 131072
# Matched control: distinct paths, --quantile-exploration-power 0.
```

### Quantile distribution and policy-criterion checks

The [first full quantile round at **331,072**](results/defense/training/dqn-37-quantile-neutral/comparison-at-000000331072.json)
adds **200,000** new actions to each calibrated model. Neutral completed
**82** new boot games and risk **83**, with no restored segments. Both full
online/target/Adam/RNG checkpoints are preserved. Standard mean-greedy results:

| Training criterion | Mean | Median | Best | Verified replay |
| --- | ---: | ---: | ---: | --- |
| Neutral | 9,939 | 10,120 | 10,360 | [2,535 actions](results/defense/training/dqn-37-quantile-neutral/first-replay/replay.html) |
| Power 1.5 | 10,283 | 10,280 | 10,330 | [2,534 actions](results/defense/training/dqn-38-quantile-risk/first-replay/replay.html) |

Risk minus neutral is **+344**, eight paired scores higher and two lower.
One pair contributes **2,480** of the **3,440** summed difference. All twenty
games lose in stage 1; this reversal of the short check is not a durable
advantage or a fresh success-rate estimate.

The [second-round record through **531,072**](results/defense/training/dqn-37-quantile-neutral/comparison-through-000000531072.json)
shows a marked neutral regression: mean **6,317**, median **5,370**, best
**7,880**, versus risk mean **10,196**, median **10,200**, best **10,220**.
The +3,879 risk margin is retention of stage-1 scoring, not new depth. All
forty standard games across both rounds lost in stage 1. Complete game records
and local immutable model hashes are preserved; first-round full states and
stronger verified replays remain the archived references. Neither run's
settings changed after these results.

By [**731,072**, the third round](results/defense/training/dqn-37-quantile-neutral/comparison-through-000000731072.json),
neutral recovered to mean **9,836**, median/best **10,400**, versus risk mean
**10,083**, median **10,240**, best **10,280**. Its new best
[2,538-action replay](results/defense/training/dqn-37-quantile-neutral/replay-10400/replay.html)
and complete third-round optimizer checkpoint are preserved. All sixty standard
games across three rounds lost in stage 1. The +247 risk mean margin comes
from only two higher paired games and eight lower ones; it is not consistent
per-seed dominance. The risk arm's stronger first-round replay remains its
archived reference. Neither this recovery nor fluctuating score margins
establish passage through the recurring obstacle or replace the shared best.

The read-only Q probe now also supports mean-greedy quantile replay bundles.
It reproduces every recorded action, checks checksums before/after, retains
trained quantile indices without sorting, and compares choices under a fixed
power-1.5 counterfactual. On the calibrated models' selected replays, the
[neutral report](results/defense/diagnostics/quantile-neutral-distribution-131072.json)
and [risk report](results/defense/diagnostics/quantile-risk-distribution-131072.json)
reproduce all **2,592 / 2,538** actions. Mean spans between midpoint fractions
0.109375 and 0.890625 are **110.72 / 133.34** discounted score units; these
are learned return spreads, not confidence intervals or proof of calibration.
Adjacent selected-action quantile crossings occur in **0.086% / 0.146%** of
comparisons. Whole-trace mean prediction minus realized greedy return is
**−92.74 / −68.44**; selection of good traces prevents interpreting that as
global underestimation.

Power weighting changes **776 / 609** raw action IDs. After collapsing the
nine equivalent stage-1 forward-fire commands, **644 / 474** choices still
differ (**24.85% / 18.68%**). In the four 64-decision pre-loss-alignment windows,
command differences are **23 / 17 / 31 / 19** for neutral and
**12 / 12 / 13 / 13** for risk. These are counterfactual choices on saved
screens, not new trajectories, actual training disagreement rates, measured
motion or evidence of a successful alternative route. Replay data never
enters learning.

To test those choices in actual play, `rl.defense_evaluate` now accepts an
explicit **evaluation-only** `--quantile-power 1.5`. Omission preserves all
standard mean-greedy behavior; explicit zero is mean-greedy but still a probe.
The option is rejected for other algorithms, nonfinite/out-of-range values
or temperature overrides. Its value is recorded in the evaluation, replay
metadata and independent native verification. Replay recording reloads the
frozen weights and reproduces actions, rewards, screens and outcome under
the same explicit criterion. Both collector and direct standard publisher
exclude evaluation-only reports. No trainer or default policy changed.

On the **331,072** frozen checkpoints, ten complete power-policy games on
the same reused seeds gave:

| Trained model | Probe mean | Median | Best | Verified probe replay |
| --- | ---: | ---: | ---: | --- |
| Neutral | 2,864 | 2,880 | 5,210 | [2,241 actions](results/defense/diagnostics/quantile-neutral-power-replay-331072/replay.html) |
| Risk | 8,863 | 9,350 | 10,280 | [2,503 actions](results/defense/diagnostics/quantile-risk-power-replay-331072/replay.html) |

All twenty probe games lost in stage 1. Mean changes versus the same weights'
standard play were **−7,075 / −1,420**. Thus this checked alternative criterion
did not hide a stage clear at these checkpoints/seeds. It does not establish
that every future distorted policy will fail. Mean-greedy remains standard;
the two ongoing learners are unchanged and no shared-best replacement occurs.
All [24 focused tests](results/defense/diagnostics/quantile-policy-probe-focused-tests.txt)
and [293 full regression tests](results/defense/diagnostics/quantile-policy-probe-regression-tests.txt)
pass, including native alternative-policy replay and promotion exclusion.

```bash
venv/bin/python -m rl.defense_evaluate \
  results/defense/training/dqn-38-quantile-risk/step-000000331072/model.safetensors \
  --quantile-power 1.5 --games 10 --seed 10000 --envs 8 \
  --output runs/defense-quantile-policy-check.json \
  --replay-output runs/defense-quantile-policy-check-replay
```

### DQN archive-diversity comparison

A [read-only inventory audit at action 7,242,407](results/defense/diagnostics/score-archive-inventory-7242407.json)
found all eight run-33 workers' latest logged terminal inventories had the same
**16 score cells / 53 snapshots**. Their lowest score-bin bound was **1,020**
within a life; ten of sixteen cells began at **2,420** or more. Since resets
choose a cell uniformly, those ten cells represent 62.5% of that inventory's
cell-selection probability. Nine of 117 completed restored segments lasted at
most 32 actions. These are logged inventories at different nearby episode ends,
not direct inspection of current native states. Score is not position, and
short segments do not prove a saved state was already doomed.

The existing score archive intentionally keeps the highest bins. It can
therefore omit lower-scoring approaches even if their screens differ. To test
that restriction, DQN now exposes the environment's already implemented
`--curriculum-cells score|screen`, `--curriculum-bins`, `--curriculum-per-bin`,
`--curriculum-score-interval` and `--curriculum-screen-interval`. Defaults remain
score cells, 16 bins, four entries, 20-point spacing and 32-action screen
sampling. The screen method uses the existing HUD-excluded 9×16 eight-level
graphics fingerprint, bounded bottom-k hash admission and uniform cell reset.
It is not object detection, a route, a novelty reward or a policy input.

An [exact pre-change/post-change regression](results/defense/diagnostics/dqn-screen-default-parity.json)
ran 4,096 actions with persistent exploration, own-state sharing and updates.
Online/target files are byte-identical, all **26** optimizer arrays are exactly
equal, and both RNG streams, exploration counters and episode counts match.
It completed five boot segments and three restored segments with deliberate
512-action truncation; this is plumbing evidence, not game success. Neither
smoke run supplies training data or weights to production. All **11** focused
[tests pass](results/defense/diagnostics/dqn-screen-focused-tests.txt), including real screen-archive capacity, reserved boot workers,
optimizer resume/option inheritance and unchanged from-boot evaluation.
The full **280-test** [regression suite passes](results/defense/diagnostics/dqn-screen-regression-tests.txt).

The bounded [screen arm](results/defense/training/dqn-screen-cells-calibration-01/resume-config.json)
and [score control](results/defense/training/dqn-score-cells-control-01/resume-config.json)
both start from run 33's **6,962,144** complete checkpoint (mean 10,259,
best 10,480), restoring the **same** online/target/optimizer and RNG states.
Each adds **131,072** actions, ending at **7,093,216**, with eight workers,
nominal 25% persistent exploration, durations 1–64/exponent 1.5, 0.5 reset
probability, two boot-only workers and lookback 32. Both have **128 cells ×
one snapshot per stage** as their maximum archive capacity. Actual occupied
counts may differ and must be reported, not assumed equal. Other learning,
compact-replay capacity, action timing and observation settings match.

Only paths, cell mode and its provenance metadata differ. Treatment includes
the mode's event selection: score changes at 20-point bins versus changed
screen cells sampled every 32 actions. This is not a key-only comparison or
an isolated comparison against run 33's smaller 16×4 archive. Both archives
start empty and receive only newly reached same-run snapshots; neither uses
the diagnostic replays or the other arm's experience. Ten complete greedy
evaluation games start from boot on the same reused seeds. Both short sources
are excluded from the global collector. The previous PPO screen-cell trial
did not clear stage 1; this tests the combination with persistent DQN, without
assuming screen diversity is sufficient. Trials 33/34 continue unchanged.

Both checks have now stopped normally at **7,093,216**. The
[complete paired result](results/defense/training/dqn-screen-cells-calibration-01/comparison.json)
favors the screen method within this bounded comparison:

| Archive selection | Ten-game mean | Median | Best | Independently verified replay |
| --- | ---: | ---: | ---: | --- |
| Screen fingerprints | 10,156 | 10,120 | 10,240 | [2,528 actions](results/defense/training/dqn-screen-cells-calibration-01/replay/replay.html) |
| Score bins, matched limit | 9,772 | 9,760 | 9,820 | [2,438 actions](results/defense/training/dqn-score-cells-control-01/replay/replay.html) |

All ten paired screen scores were higher, mean difference **+384**. All twenty
games nevertheless lost in stage 1, and both means remain below the common
parent's **10,259**. This one paired lineage on reused seeds is not a fresh
success rate, a durable advantage or a stage clear. Both full online/target/
optimizer/RNG checkpoints and complete compressed logs are preserved beside
their replays; the shared best remains unchanged.

Each arm completed **49** new boot games; screen selection completed **28**
restored segments and score selection **29**, without a later-stage event.
The latest logged terminal inventories had **128** entries on every screen
worker and **33** on every score worker. Equal capacity did not produce equal
occupancy, as anticipated. Both recorded source progress up to **2,600** and
trigger progress up to **2,620** within a life. These scores do not establish
obstacle passage, and having more screen cells does not prove they represent
useful new situations. Both reserved workers in each arm remained boot-only;
all archive events retained the exact 32-action lookback.

To test durability, full trials now continue each arm's own **7,093,216**
checkpoint: [DQN 35 screen archive](results/defense/training/dqn-35-screen-archive/resume-config.json)
and [DQN 36 score control](results/defense/training/dqn-36-score-archive-control/resume-config.json).
They retain all learning/archive settings and change only the output paths,
the training limit to unlimited, and the validation interval to **200,000**.
First full evaluations are due at **7,293,216**. Each restores its own
online/target/optimizer/RNG but rebuilds replay and its own archive from new
booted experience. This is one paired lineage, not independent seed replication
or exact trajectory continuation. No evaluation trace or saved foreign native
state is supplied to learning. Trials 33/34 continue unchanged.

The old collector stopped cleanly and was confirmed gone before the sole new
collector started with [all 32 full-run sources](results/defense/training/dqn-35-screen-archive/collector-config.json).
Historical sources remain included; short checks remain excluded. Any shared
best replacement still requires independent frozen-policy action/screen/reward
verification. The new capacity-matched pair is an experiment, not a claim that
screen cells have resolved the failure point.

The [first full round at **7,293,216**](results/defense/training/dqn-35-screen-archive/comparison-at-000007293216.json)
is now complete, after **200,000** new actions per arm:

| Archive selection | Ten-game mean | Median | Best | Verified replay |
| --- | ---: | ---: | ---: | --- |
| Screen fingerprints | 10,198 | 10,210 | 10,280 | [2,536 actions](results/defense/training/dqn-35-screen-archive/first-replay/replay.html) |
| Score bins, matched limit | 10,133 | 10,395 | 10,440 | [2,533 actions](results/defense/training/dqn-36-score-archive-control/first-replay/replay.html) |

The +65 mean difference comes from three higher and seven lower paired screen
scores. Its lower median and best score qualify the apparent mean advantage;
all twenty games lost in stage 1. Both full online/target/optimizer/RNG
checkpoints and manifest-checked, independently replay-verified bundles are
preserved. The stronger shared best is unchanged.

Screen selection completed **79** new boot games and **48** restored segments;
score selection completed **77 / 59**. All logged training episodes also
remained in stage 1. Latest terminal inventories contained **128** screen
entries versus **31** score entries per worker. Both reserved boot workers
remained boot-only and all archive events retained exact 32-action lookbacks.
More occupied cells have not yet yielded a later-stage experience.

The [read-only loss comparison](results/defense/diagnostics/shared-loss-screen-archive-01/report.json)
uses these frozen replays, with no training examples or parameter changes.
Screen selection gained **2,570 on all four lives**; score selection gained
**2,580 / 2,620 / 2,620 / 2,620**. Their
[screen-arm panels](results/defense/diagnostics/shared-loss-screen-archive-01/policy-1-losses.png)
and [control panels](results/defense/diagnostics/shared-loss-screen-archive-01/policy-2-losses.png)
show the same broad right-opening barrier sequence with the ship toward the
centre/left near losses. This is visual evidence of the recurring bottleneck,
not proof of identical collisions or a prescribed route. Flash alignment can
lag a collision; final-life settling can skip the flash entirely. Neither
score alone nor these selected high-scoring traces establish general success.
The full pair continues unchanged for its next matched evaluation.

The [three-round curve through **7,693,216**](results/defense/training/dqn-35-screen-archive/comparison-through-000007693216.json)
now records screen/control means **10,198 / 10,133**, **9,836 / 10,172** and
**10,085 / 10,198**: differences **+65 / −336 / −113**. All sixty complete
evaluation games lost in stage 1. The later rounds do not support a durable
advantage from screen diversity. Full per-game records and model hashes are
preserved in the curve; first-round full checkpoints and stronger verified
replays remain archived. The newer matched quantile experiment is separate;
neither evaluation data nor replay traces enter its training buffer.

The [six-round curve through **8,293,216**](results/defense/training/dqn-35-screen-archive/comparison-through-000008293216.json)
extends the mean differences to **+65 / −336 / −113 / +50 / −419 / −212**
(screen minus score). All **120** reused evaluation games lost in stage 1.
The sixth round is screen mean **10,206**, median **10,200**, best **10,220**,
versus score mean **10,418**, median **10,415**, best **10,460**. Both complete
evaluated checkpoints are archived. The score arm's new
[2,521-action verified replay](results/defense/training/dqn-36-score-archive-control/replay-10460/replay.html)
is preserved without replacing the stronger shared best.

Screen selection then [stopped cleanly at **8,451,456**](results/defense/training/dqn-35-screen-archive/retirement.json),
after **1,358,240** new actions, **534** new boot games and **336** restored
segments since its calibration. All logged completed training episodes also
remained in stage 1. Final online/target/Adam/RNG, the last evaluated full
checkpoint and complete compressed log are preserved; final post-update
weights were not separately evaluated. Its depth plateau and lack of a durable
mean advantage prompted reallocating the slot, not a wall-clock limit. Run 36
continues. No checkpoint, replay or log was deleted.

```bash
# Use distinct paths and curriculum-cells score for the capacity-matched control.
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-dqn-screen-cells-reproduction \
  --artifacts runs/defense-dqn-screen-cells-reproduction/artifacts \
  --resume results/defense/training/dqn-33-persistent-resets/step-000006962144 \
  --steps 7093216 --eval-every 131072 --curriculum-cells screen \
  --curriculum-bins 128 --curriculum-per-bin 1 --curriculum-screen-interval 32

# Unlimited continuation of the verified screen-cell arm:
venv/bin/python -u -m rl.defense_dqn \
  --run runs/defense-dqn-screen-archive-full-reproduction \
  --artifacts runs/defense-dqn-screen-archive-full-reproduction/artifacts \
  --resume results/defense/training/dqn-screen-cells-calibration-01/checkpoint \
  --steps 0 --eval-every 200000
# For the capacity-matched control, use separate paths and resume
# results/defense/training/dqn-score-cells-control-01/checkpoint instead.
```

### Lossless compact training replay

`rl.defense_dqn --compact-replay` optionally replaces dense duplicated screen
stacks with exact frame identities and reference-counted byte storage. Each
observation and next-observation still records its **four exact frames**;
there is no inference from temporal adjacency, worker order, n-step distance,
life boundaries or restored-state history. Byte-key equality resolves hash
collisions without approximation. When ring slots are overwritten, unreferenced
frames are released and their IDs reused; sampled arrays are independent,
writable copies. No native state is inspected and no RNG draw is added.

Actions, rewards, discounts, priorities, sampling weights, bootstrap memberships,
model inputs and update rules are unchanged. The option supports both ordinary
and bootstrap DQN, defaults off, and is inherited on optimizer resume. Replay
still refills from new experience after resume; it is not a loaded demonstration
archive. Configuration records the storage implementation hash. Logged storage
figures distinguish frame payload and index bytes from the equivalent dense
arrays; these figures **exclude Python metadata and the other replay/model
allocations**, and are not total process-memory measurements.

All **246 regression tests** passed. New checks exercise ring wraparound,
repeated boot frames, exact reference counts, sampling without aliases,
arbitrary history spacing, real own-state resets, n-step terminal/truncation
handling, identical priority trees, bootstrap masks and RNGs. Real-emulator
dense/compact training produced **exactly equal online, target and optimizer
arrays**, episode/update counters and RNG states for both ordinary and bootstrap
DQN, including ring reuse; compact resumes were also exercised.

This is intended to make larger own-experience buffers practical on the Mac
Mini, not to change the policy or claim a learning gain from storage alone.
The existing live learners have not been restarted or silently switched.

An [isolated compact-storage check](results/defense/training/compact-replay-smoke-01/config.json)
repeated the original fresh 16,384-action DQN check's learning settings. It
reproduced **every array** in the online model (12), target model (12) and
optimizer (26), the common counters/RNG state, and **all ten complete game
records exactly**. The [parity record](results/defense/training/compact-replay-smoke-01/parity.json)
documents this comparison. Mean/median/best remained **328 / 320 / 360**, all
stage 1; its [replay](results/defense/training/compact-replay-smoke-01/replay/replay.html)
verified **1,655** actions. Full weights, target, optimizer, configuration and
log are preserved. It exited normally and is excluded from the collector.

At its logged **15,416**-action point, the full 8,192-transition buffer held
**3,739** distinct frames: **3,828,736** payload bytes plus **262,144** index
bytes, about **3.90 MiB** versus **64 MiB** for dense screen arrays. Python
metadata and other allocations are excluded from this comparison. This
validates a substantial reduction in duplicated screen data in this check,
not a total-RAM or speedup claim, nor evidence that a larger buffer improves
learning. No recorded evaluation trace was used as training experience.

### Larger-buffer DQN comparison

`defense-dqn-28-large-replay` starts independently from fresh seed-97 weights,
with **200,000** replay transitions instead of run 22's **50,000**, using the
lossless compact storage above. It does not resume the storage check or any
previous model. Its [configuration](results/defense/training/dqn-28-large-replay/config.json)
retains ordinary Double DQN's eight workers, batch 64, learning rate 0.0001,
five-step returns, gamma 0.997, 100,000-T-state actions, stride 1, life
boundaries, 10,000-action warmup, update every 16 aggregate actions, target
copy every 2,000 updates and epsilon decay to 0.05. Own-state resets and
bootstrap heads are disabled. Training and complete-game evaluation remain
uncapped; evaluation runs every 100,000 actions from boot.

More retained own experience is a hypothesis to test, not a diagnosed fix
for run 22's regression. Source hashes differ because intervening optional
features were added; default compatibility and exact dense/compact training
parity were tested as documented above. No policy input, reward bonus or
evaluation demonstration has been added. The sole collector includes run 28
and retains all historical sources; other live learners continue unchanged.

Its [first ten complete games at 100,000](results/defense/training/dqn-28-large-replay/step-000000100000/evaluation.json)
averaged **268**, median **260**, best **280**, all stage-1 losses. This is below
run 22's same-counter mean 280, not an early performance improvement. The full
online/target/optimizer checkpoint and [1,477-action verified replay](results/defense/training/dqn-28-large-replay/first-replay/replay.html)
are preserved.

At **200,000**, it reached
[mean 316, median 320, best 320](results/defense/training/dqn-28-large-replay/step-000000200000/evaluation.json),
with a [1,631-action verified replay](results/defense/training/dqn-28-large-replay/replay-320/replay.html).
The full optimizer/target checkpoint is preserved. This is above run 22's
same-counter mean/best 280, but all ten games still lost in stage 1. One early
batch on reused seeds does not establish improved depth or durable stability.

At **400,000**, run 28 reached
[mean 328, median 320, best 340](results/defense/training/dqn-28-large-replay/step-000000400000/evaluation.json).
The intervening 300,000-action mean was 298, best 300, so the improvement was
not monotonic. The full checkpoint and [1,610-action verified replay](results/defense/training/dqn-28-large-replay/replay-340/replay.html)
are preserved. This is close to run 22's same-counter mean 322 and equal best
340; all games remained stage-1 losses, not evidence of a clear advantage.

Longer training produced substantial gains, but also repeated regressions.
Its strongest single game was **10,370** at **5,700,000** actions:
[mean 9,898, median 10,285](results/defense/training/dqn-28-large-replay/step-000005700000/evaluation.json),
with a [2,515-action verified replay](results/defense/training/dqn-28-large-replay/replay-10370/replay.html).
Its peak mean came at **6,100,000**:
[10,290, median 10,290, best 10,360](results/defense/training/dqn-28-large-replay/step-000006100000/evaluation.json).
Both full online/target/optimizer checkpoints and every intermediate
best-effort replay/checkpoint are preserved. All remained stage-1 losses.
This exceeds run 22's achieved best, but uses substantially more experience
and does not isolate a causal buffer-size advantage at matched compute.

Run 28 stopped cleanly at **7,400,000** actions, **461,874** updates and
**3,783** complete training games. There were **73** complete evaluation
batches; the 7,400,000 checkpoint was not evaluated before shutdown.
The last four completed means were **300, 364, 378, 362**. Its
[final checkpoint](results/defense/training/dqn-28-large-replay/final-checkpoint/state.json),
[last completed evaluation](results/defense/training/dqn-28-large-replay/step-000007300000/evaluation.json)
and [full log](results/defense/training/dqn-28-large-replay/metrics.jsonl)
are preserved. A larger buffer alone did not prevent regression or produce
stage 2. Matched short continuation checks now compare learning rates
0.0001 and 0.000025 from the **6,100,000** peak, with otherwise identical
fresh-buffer restart settings. Neither check is a collector source or a
claimed improvement before its complete-game results are available.

Those short checks have now finished. Both started from the same peak and
added **32,768** actions, ending at **6,132,768**, with eight workers and the
same 200,000-transition compact capacity. Each buffer refilled from new own
experience with the existing 10,000-transition random warmup; no saved
training or evaluation trajectories were loaded. Their configurations differ
only in learning rate and output paths.

| Learning rate | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [0.0001 control](results/defense/training/dqn-peak-control-01/checkpoint/evaluation.json) | 9,734 | 10,245 | 10,310 | [2,388](results/defense/training/dqn-peak-control-01/replay/replay.html) |
| [0.000025](results/defense/training/dqn-peak-low-lr-01/checkpoint/evaluation.json) | 5,367 | 5,190 | 7,580 | [2,286](results/defense/training/dqn-peak-low-lr-01/replay/replay.html) |

All twenty complete evaluation games lost in stage 1. Both means are below
the frozen parent's 10,290; the lower rate is substantially worse than the
matched restart control and is **not** being promoted to a full run. This
does not show that slowing updates fixes the long-run regression. Both
checks exited normally; full online/target/optimizer states, configuration,
logs and verified replays are preserved, separate from the collector.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-large-replay-reproduction \
  --artifacts runs/defense-dqn-large-replay-reproduction/artifacts \
  --seed 97 --capacity 200000 --compact-replay
```

### Ordinary DQN with its own reached-state resets

Ordinary DQN can now enable the same opaque, own-experience reset mechanism
used by PPO via `--curriculum-probability`, `--curriculum-share`,
`--curriculum-boot-envs` and `--curriculum-lookback`. This initial integration
defaults to score cells: visible score gains within the current life, 20-point
bins, 16 bins per stage and four retained states per bin. Boot-only workers
share newly reached states but never restore one. The archive is initially
empty, stays bounded, and is rebuilt on optimizer resume; native payloads are
neither decoded nor fabricated, serialized into model weights, or policy input.
No PPO trajectory, evaluation replay, demonstration or archived native state
is loaded into DQN training.

Only new visible-score differences enter n-step replay. At terminal/life
boundaries, pending returns end at the actual final screen, not the next
reset screen; restored starting score is not credited as reward. Boot-game
counts and recent scores are separated from restored-segment counts and new
segment reward. Full evaluation and replay verification still use the ordinary
from-boot environment without curriculum options. Bootstrap-head exploration
and these resets cannot yet be combined; the CLI rejects that combination.
The default probability is zero and existing learners continue unchanged.

All **240 regression tests** passed. New tests cover invalid combinations,
terminal-screen/return isolation, separate segment bookkeeping, unchanged
from-boot evaluation arguments, real own-state sharing, reserved boot workers,
and optimizer resume with fresh archives/replay. Truncated unit-test episodes
are bookkeeping checks, not evidence of actual game completion.

A paired short check resumed ordinary DQN's **1,700,000** checkpoint.
Each side trained **32,768** new actions, using four workers, an 8,192-transition
buffer refilled from new experience, 1,024 warmup transitions, and the parent's
batch 64, n-step 5, gamma 0.997, target-copy interval 2,000 and 100,000-T-state
action timing. One side enables 0.5 reset probability, shared score cells,
one boot-only worker and lookback 32; the other keeps resets disabled. Both
are excluded from the collector and evaluate ten complete games from boot.
Both exited normally at **1,732,768**, each with **107,608** cumulative optimizer
updates. Their configurations differ only in output paths and the reset setup
and its recorded provenance. Both refilled their replay buffers with new own
experience; neither loaded evaluation actions or another learner's states.

| Short continuation | Mean | Median | Best | Verified replay actions |
| --- | ---: | ---: | ---: | ---: |
| [No resets](results/defense/training/dqn-reset-control-01/checkpoint/evaluation.json) | 284 | 280 | 300 | [1,531](results/defense/training/dqn-reset-control-01/replay/replay.html) |
| [Own-state resets](results/defense/training/dqn-reset-smoke-01/checkpoint/evaluation.json) | 362 | 350 | 540 | [1,919](results/defense/training/dqn-reset-smoke-01/replay/replay.html) |

All twenty complete games were stage-1 losses. Reset scores were higher on nine
paired seeds and equal on one, mean difference **+78**, but **both** checks
regressed substantially from the parent's mean **1,372**. This one short pair
does not establish an overall learning advantage or explain that regression.
Worker count, replay capacity and warmup differ from the full production run.
The control completed 16 boot games; the reset side completed 14 boot games
and five restored segments. All **402** archive events had exact 32-action
lookback; **54** originated in already-restored segments. Worker 0 remained
boot-only, and every segment's logged reward equalled its new visible score.
Both complete checkpoints, configurations, logs and verified replays are
preserved separately; neither is a parent of the longer trial.

`defense-dqn-26-own-resets` instead starts **fresh at counter zero**, with seed
97 and the ordinary baseline's normal eight workers, 50,000-transition buffer,
10,000-transition random warmup, batch 64, n-step 5, gamma 0.997, target copies
every 2,000 updates and updates every 16 aggregate actions. Epsilon decays
from 1 to 0.05 over one million actions after warmup, as in run 22. It changes
the reset setup to probability 0.5, shared score cells, two permanently
boot-only workers and lookback 32. It uses neither short-check weights nor
pretrained features or native snapshots. The
[configuration](results/defense/training/dqn-26-own-resets/config.json)
records the fresh initialization and implementation hashes.

This longer fresh-start comparison tests whether resets help value learning;
it is not presented as a successful fix for the short-check regression.
Training and episodes are uncapped; ten from-boot evaluations occur every
100,000 actions. It uses part of stopped run 17's released compute. Runs 22,
24 and 25 continue unchanged. The sole collector includes run 26 and all
historical sources, excluding both short checks, with the same frozen-policy
verification before any global promotion.

Run 26's [first ten complete evaluations](results/defense/training/dqn-26-own-resets/step-000000100000/evaluation.json),
at **100,000** actions, averaged **204**, median **200**, best **260**, all
stage-1 losses without a mission. This is below ordinary run 22's same-counter
mean/median/best **280**, not an improvement. The full online/target/optimizer
checkpoint and [1,482-action verified replay](results/defense/training/dqn-26-own-resets/first-replay/replay.html)
are preserved. At that checkpoint, training had completed **50** boot games,
**27** restored segments and **5,624** optimizer updates. Segment counts are
not counted as full games. This fresh-start trial continues without promoting
over the stronger shared best.

At **200,000**, run 26 reached
[mean 282, median 280, best 300](results/defense/training/dqn-26-own-resets/step-000000200000/evaluation.json),
with a [1,598-action verified replay](results/defense/training/dqn-26-own-resets/replay-300/replay.html)
and full optimizer/target checkpoint preserved. All ten games remained stage-1
losses. This recovers from its first batch and is only slightly above ordinary
run 22's same-counter mean/best 280; it is not evidence of a robust advantage.

At **400,000**, run 26 reached
[mean 302, median 320, best 340](results/defense/training/dqn-26-own-resets/step-000000400000/evaluation.json),
with a [1,600-action verified replay](results/defense/training/dqn-26-own-resets/replay-340/replay.html).
At **600,000**, it reached
[mean 326, median 320, best 360](results/defense/training/dqn-26-own-resets/step-000000600000/evaluation.json),
with a [1,643-action verified replay](results/defense/training/dqn-26-own-resets/replay-360/replay.html).
Both full optimizer/target checkpoints are preserved; all games remained
stage-1 losses. These small lineage improvements do not replace the shared best.

After further training, run 26's best single game reached **8,360** at
**7,100,000** actions:
[mean 6,480, median 6,395](results/defense/training/dqn-26-own-resets/step-000007100000/evaluation.json),
with a [2,424-action verified replay](results/defense/training/dqn-26-own-resets/replay-8360/replay.html).
Its strongest mean was **6,901** at **7,200,000**, median **6,845**, best
**7,560**. Both complete checkpoints, plus all intermediate best-effort
checkpoints and replays, are preserved. Through 8,100,000, none reached stage
2 or a mission; the latest mean had fallen to 2,462. The learner continues
under review rather than treating its best score as current reliability.

Run 26 subsequently improved to **10,280** at **9,300,000**:
[mean 9,849, median 10,230](results/defense/training/dqn-26-own-resets/step-000009300000/evaluation.json),
with a [2,585-action verified replay](results/defense/training/dqn-26-own-resets/replay-10280/replay.html).
Its peak mean was [**10,130**, median/best **10,160**, at **9,100,000**](results/defense/training/dqn-26-own-resets/step-000009100000/evaluation.json).
All new best checkpoints and replays are preserved, including 9,780, 10,060,
10,180 and 10,220 points. After a sharp regression, its last seven validation
means were **334, 320, 316, 330, 400, 478, 472**. It stopped cleanly at
**10,487,648** actions, with **654,852** updates, **4,672** complete boot games
and **3,072** restored segments. None of its **104** complete validation
batches reached stage 2 or a mission. The
[final optimizer/target checkpoint](results/defense/training/dqn-26-own-resets/final-checkpoint/state.json),
[last validated checkpoint](results/defense/training/dqn-26-own-resets/step-000010400000/evaluation.json)
and [losslessly compressed full log](results/defense/training/dqn-26-own-resets/metrics.jsonl.gz)
are preserved. The original log and all local history remain intact. Its
sustained regression, not a wall-clock limit, prompted freeing its compute.

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-dqn-own-resets-reproduction \
  --artifacts runs/defense-dqn-own-resets-reproduction/artifacts --seed 97 \
  --curriculum-probability .5 --curriculum-share --curriculum-boot-envs 2 \
  --curriculum-lookback 32
```

### Bootstrapped value exploration

The optional `rl.defense_dqn --bootstrap-heads 5` path adapts
[Bootstrapped DQN](https://arxiv.org/abs/1602.04621) and
[randomized prior functions](https://arxiv.org/abs/1806.03335). A training
worker draws one value head uniformly at a new **complete game**, retaining
it across ship losses. This tests more temporally consistent exploration
than changing random actions independently at every step. The weights still
learn during that game; the head identity, not the network parameters, stays
fixed. A configurable constant epsilon adds occasional uniform actions, and
the initial empty-replay warmup remains uniformly random.

This implementation uses shared convolutional features followed by independent
256-unit hidden and dueling output heads. A separate randomly initialized
network of the same architecture supplies fixed additive priors. The prior
parameters are frozen, excluded from Adam, saved with both online and target
networks, and restored exactly. These are random screen-to-value functions,
not game knowledge, an oracle, demonstrations or additional reward.

Each own n-step transition receives independent Bernoulli head memberships
**once when inserted**. Memberships persist through sampling and ring-buffer
reuse replaces them with the new transition's memberships. Empty memberships
are allowed. Every head's Double-Q target uses its own online action choice
and corresponding target head, including the same prior. Masked Huber losses
are averaged over active memberships with replay importance weights; a shared
priority is the mean absolute TD error over all heads. Life-terminal returns
still stop at visible loss, even though the behavior head persists for the game.

Evaluation is a fixed, deterministic **greedy mean of ensemble Q-values plus
priors**, not a favorable-head search. Complete-game evaluation, frozen-weight
action/screen/reward verification and stage/mission-first ranking are unchanged.
PPO, ordinary DQN and bootstrap optimizer resumes cannot be interchanged, nor
can the number of heads change on resume. Replay/masks refill from new own
experience; online/target/prior/optimizer and both RNG states are restored,
but new boot episodes draw new heads. The policy still sees only four raw
screen frames and training reward remains scaled visible score difference.

This is **not an exact paper reproduction**: shared visual layers, dueling
heads, prioritized n-step replay, the greedy-mean evaluation rule, our optimizer
and TRS-80 observations are explicit adaptations. With five heads, checkpoints
are substantially larger than ordinary DQN; long-run evaluation/checkpoint
frequency must account for available disk space. Default `--bootstrap-heads 0`
retains ordinary DQN behavior. The four existing production learners do not
change their loaded algorithms.

Tests cover persistent replay masks, head-specific Double-Q arithmetic,
masked head gradients, frozen priors, target synchronization, greedy ensemble
reload, invalid configurations, exact zero-update optimizer/RNG cloning,
real-emulator training/resume, and head persistence across ship losses.
The full suite passed **224 tests** after integration.

The [isolated check](results/defense/training/bootstrap-smoke-01/config.json)
completed **16,384** own actions and **960** updates, then evaluated ten complete
games: [mean/median/best all **140**](results/defense/training/bootstrap-smoke-01/checkpoint/evaluation.json),
all stage 1 losses. Its
[verified replay](results/defense/training/bootstrap-smoke-01/replay/replay.html)
reproduced **1,208** actions. The full online/target/prior/optimizer, configuration
and log are preserved. It exited normally and is excluded from the collector.
The ordinary four-worker DQN check had mean **328**, median **320**, best **360**
at the same action count: the bootstrap check is substantially worse, not an
early performance advantage. This short check establishes working integration,
not efficacy. GPU peak was approximately 454 MB, excluding replay/emulator memory.

A [read-only head-agreement check](results/defense/diagnostics/bootstrap-smoke-head-agreement.json)
reconstructed all 1,208 input stacks from that own verified replay and reproduced
every mean-policy action. Pairwise **raw-action** agreement among heads was
**14.59%**, while the mean policy chose LEFT on **998** steps. Head preferences
differed; this was not identical-head behavior on the selected trajectory.
It does **not** measure individual-head game performance, establish the cause
of the low score, or represent the new production run. Action aliases further
limit behavioral interpretation. No weights, prior scale, evaluation rule or
training data were changed; the production trial continues to its original gate.

The subsequent [matched zero-prior check](results/defense/training/bootstrap-no-prior-smoke-01/config.json)
changed **only** `bootstrap_prior_scale` from 1 to 0, plus its output paths.
Seed, architecture, independent heads, initial random-network generation,
membership probability, epsilon, replay, optimizer, 16,384-action budget,
960 updates, four workers and ten validation seeds all remained the same.
All **36 frozen prior tensors** were identical across the saved checks and the
zero-prior online/target networks; the zero scale disables their contribution,
not their construction. Subsequent learned weights and trajectories naturally
diverge as a result of the changed policy.

Its [ten complete games](results/defense/training/bootstrap-no-prior-smoke-01/checkpoint/evaluation.json)
had mean/median/best **280 / 280 / 280**, all stage 1 losses, versus
**140 / 140 / 140** with scale 1 on the same seeds. The
[verified replay](results/defense/training/bootstrap-no-prior-smoke-01/replay/replay.html)
reproduced **1,482** actions. Its full checkpoint, config and log are preserved;
it exited normally and is excluded from the collector. This favors zero prior
in this short **single-training-seed** comparison, but remains below the
ordinary DQN check's mean 328. It does not establish a long-run advantage or
justify changing the ongoing production trial before its scheduled evaluation.

To reproduce this check, use the short-check command below with
`--bootstrap-prior-scale 0` and distinct run/artifact directories.

`defense-dqn-24-bootstrap` starts fresh at counter zero with seed 97,
**five heads**, prior scale **1**, membership probability **0.5**, epsilon **0.01**
after a **10,000-transition random warmup**, eight workers, batch 64, capacity
50,000, n-step 5, discount 0.997 and target synchronization every 2,000 updates.
It does not load the smoke weights, another learner's encoder, replay traces
or native snapshots. All episode/training caps are zero. Ten-game evaluations
occur every **200,000** actions; each full checkpoint is approximately **91 MiB**,
so storage remains monitored. Its
[configuration](results/defense/training/dqn-24-bootstrap/config.json) is preserved.
This deliberately longer test of coherent exploration replaces stopped run 20;
it is not justified as already stronger. Runs 17, 22 and 23 remain active.
The single collector retains all historical sources and includes run 24, with
the same independent full-replay verification gate and unchanged global best.

Run 24's [first ten complete games](results/defense/training/dqn-24-bootstrap/step-000000200000/evaluation.json),
at **200,000** actions and **11,874** updates, each scored **380**
(mean/median/best **380 / 380 / 380**), all stage 1 losses without a mission.
Its full online/target/prior/optimizer checkpoint and
[1,705-action verified replay](results/defense/training/dqn-24-bootstrap/first-replay/replay.html)
are preserved. Training had completed 128 boot games at that checkpoint.
Ordinary DQN's [same-counter ten games](results/defense/training/dqn-22-fresh/step-000000200000/evaluation.json)
each scored **280** on the same seeds. This is a local early advantage for the
ensemble setup, not evidence of improved depth, broad superiority or lower
compute cost. Architecture, priors and exploration schedule differ; it is not
a single-feature ablation or a comparison against a long zero-prior ensemble.
Both are one training seed with reused validation seeds. Run 24 continues
without replacing the stronger shared 10,480-point best.

At **800,000**, run 24's
[ten complete evaluations](results/defense/training/dqn-24-bootstrap/step-000000800000/evaluation.json)
all scored **520** (mean/median/best 520), still stage-1 losses. Its full
online/target/prior/optimizer checkpoint and
[2,006-action verified replay](results/defense/training/dqn-24-bootstrap/replay-520/replay.html)
are preserved. The intervening 400,000-action mean was 350 (best 380), and
the 600,000-action games all scored 320, so this recovery was not monotonic.
It is a new milestone for this lineage, not a global-best or depth improvement.

The long run eventually moved beyond that early plateau. At **5,200,000**,
it reached [mean 10,186, median 10,200, best 10,220](results/defense/training/dqn-24-bootstrap/step-000005200000/evaluation.json),
with a [2,524-action verified greedy-ensemble replay](results/defense/training/dqn-24-bootstrap/replay-10220/replay.html).
That full online/target/prior/optimizer checkpoint and all intervening
best-effort milestones are preserved. All ten games still lost in stage 1.
The next two means fell to 7,773 and 6,476, so this is a saved capability,
not a claim that the current policy is equally reliable. This run continues;
its earlier 800,000-action diagnostic below is not evidence about these
later learned heads.

At **6,000,000**, the ensemble improved to
[mean 10,258, median 10,260, best 10,280](results/defense/training/dqn-24-bootstrap/step-000006000000/evaluation.json),
with a [2,485-action verified replay](results/defense/training/dqn-24-bootstrap/replay-10280/replay.html).
At **7,000,000**, it reached
[mean **10,334**, median **10,330**, best **10,410**](results/defense/training/dqn-24-bootstrap/step-000007000000/evaluation.json),
with a [2,578-action verified replay](results/defense/training/dqn-24-bootstrap/replay-10410/replay.html).
Both full online/target/prior/optimizer checkpoints are preserved. This is
continued improvement within the bootstrap lineage, but still below the
shared 10,480-point best and entirely stage-1 losses. At 7,200,000 its mean
was 10,170, best 10,200. The learner subsequently paused cleanly for disk
pressure at **7,383,056** actions, **460,815** updates and **3,661** complete
training games. All **36** complete validation batches stayed in stage 1.
Its [pause checkpoint](results/defense/training/dqn-24-bootstrap/pause-checkpoint-000007383056/state.json),
[validation history](results/defense/training/dqn-24-bootstrap/validation-at-000007383056.json)
and [losslessly compressed complete log](results/defense/training/dqn-24-bootstrap/metrics-at-000007383056.jsonl.gz)
are preserved. Resuming restores weights, target, priors, optimizer and RNG,
but refills replay from new own experience; it is not an exact continuation
of the in-memory training buffer. No historical files were removed.

The first post-storage-resume DQN evaluation at
[**7,583,056**](results/defense/training/dqn-24-bootstrap/step-000007583056/evaluation.json)
averaged **10,090**, median **10,100**, best **10,200**, all ten stage-1 losses.
Its full online/target/prior/optimizer checkpoint is preserved. This is slightly
below the final pre-pause mean 10,170 and does not improve the saved 10,410-point
bootstrap best. The replay buffer refilled from new own experience after resume.

Run 24 later retired cleanly at **8,699,808** actions, **542,486** updates and
**4,216** complete boot games. All **42** ten-game validations stayed in stage 1;
its best remained 10,410 and peak mean 10,334 at seven million actions. The
last validation at **8,583,056** had mean **9,970**, median **9,980**, best **10,030**.
The [final online/target/prior/optimizer/RNG checkpoint](results/defense/training/dqn-24-bootstrap/final-checkpoint-000008699808/state.json),
[last validation checkpoint](results/defense/training/dqn-24-bootstrap/step-000008583056/evaluation.json),
[complete validation curve and retirement record](results/defense/training/dqn-24-bootstrap/retirement-000008699808.json)
and [full compressed log](results/defense/training/dqn-24-bootstrap/metrics-at-000008699808.jsonl.gz)
are preserved. The original process was confirmed exited. The prolonged depth
plateau motivated releasing compute for the controlled exploration-rate check;
this was not a wall-clock stop. No prior checkpoint or replay was removed.

A [frozen-head diagnostic at **800,000**](results/defense/diagnostics/bootstrap-800000-heads.json)
compared that checkpoint's greedy ensemble with each individual
learned head plus its saved random prior. Each fixed policy played ten complete,
uncapped games from boot on the reused seeds 10000–10009; a head was never
selected based on the current state or outcome. The ensemble reproduced all
ten saved game records exactly, and model/configuration hashes were unchanged.

| Fixed policy | Mean | Median | Best |
| --- | ---: | ---: | ---: |
| Ensemble | 520 | 520 | 520 |
| Head 0 | 482 | 500 | 520 |
| Head 1 | 508 | 500 | 520 |
| Head 2 | 520 | 520 | 520 |
| Head 3 | 440 | 420 | 520 |
| Head 4 | 368 | 380 | 400 |

All **60** games lost in stage 1. At this checkpoint, ensemble averaging is
not concealing a better single-head score or stage reach on these seeds.
This does not diagnose the training plateau or rule out different behavior at
future checkpoints. The probe makes no parameter updates, is excluded from
promotion and never supplies actions or traces to training. No production
evaluation rule was changed. All **248 regression tests** passed, including
fixed-head selection, shape validation and unchanged evaluation RNG checks.

Repeating the same frozen diagnostic at the much stronger
[5,200,000-action checkpoint](results/defense/diagnostics/bootstrap-5200000-heads.json)
again reproduced all ten saved ensemble game records exactly. Head 0 was
slightly stronger than the ensemble: mean **10,208** versus **10,186**, best
**10,300** versus **10,220**. Heads 1–4 averaged **7,663**, **9,574**, **6,722**
and **7,915**, respectively. All **60** games still lost in stage 1, and even
the best individual head remained below the shared 10,480-point effort.
Thus this later checkpoint contains a modest single-head score advantage,
but no hidden stage reach or mission completion on these seeds. Weights and
configuration were unchanged; no production policy or global replay was
replaced, and no diagnostic actions entered training.

```bash
venv/bin/python -m rl.defense_head_probe \
  results/defense/training/dqn-24-bootstrap/step-000000800000/model.safetensors \
  --envs 4 --output runs/defense-bootstrap-heads-reproduction.json
```

```bash
venv/bin/python -u -m rl.defense_dqn --run runs/defense-bootstrap-check \
  --artifacts runs/defense-bootstrap-check/artifacts --seed 97 \
  --envs 4 --batch-size 64 --capacity 8192 --warmup 1024 --target-every 256 \
  --steps 16384 --eval-every 16384 --eval-envs 4 --mlx-cache-mb 256 \
  --bootstrap-heads 5 --bootstrap-prior-scale 1 \
  --bootstrap-probability .5 --bootstrap-epsilon .01

# Independent unlimited production trial (fresh weights, standard DQN defaults):
venv/bin/python -u -m rl.defense_dqn --run runs/defense-bootstrap-reproduction \
  --artifacts runs/defense-bootstrap-reproduction/artifacts --seed 97 \
  --bootstrap-heads 5 --bootstrap-prior-scale 1 \
  --bootstrap-probability .5 --bootstrap-epsilon .01 --eval-every 200000
```

### Own-action-age archive experiment

`--curriculum-cells age --curriculum-age-interval 32` is an optional alternative
to score bins and coarse screen cells. It groups **actually reached** states
by visible stage and the learner's own action count since the last visible
ship-loss or stage boundary. A boot starts the counter at zero; each policy
action increments it. A real reset to an own saved state restores that state's
original counter, rather than inventing additional age or zeroing it at a new
training segment. No native payload is decoded or modified.

The motivation is to retain potentially useful states reached without an
immediate score increase. This changes **training reset selection only**:
the policy still receives the same four screen frames, the reward is still
only visible score change, and evaluation still starts from boot without an
archive. The counter is not a policy input, extra reward, hidden game timer,
position label, route or demonstration. It is also **not exact physical
survival or course progress**: visible loss can lag collision, and action
counts include intros/animations and may cover variable emulated time during
HUD settling. Later counter values need not correspond to better play.

Age-bin changes can trigger snapshots even at zero reward. The existing
lookback chooses the actual earlier same-life/stage snapshot and uses that
snapshot's age, score and baseline, not the later trigger's values. The
bounded archive keeps the largest age bins per visible stage, uses reservoir
sampling within each bin and samples stage/bin/state uniformly for a reset.
Defaults remain score bins; existing score/screen behavior and all live runs
remain unchanged unless a new learner explicitly selects this option.

All **218 regression tests** passed. New tests cover exact native/screen/age
equality at lag 128, bounded later-bin retention, counter clearing at visible
life/stage boundaries, saved-counter peer restore, malformed counters,
reward-free events and sharing without exposing snapshots to the model.
Full-game screen/reward comparisons also include age mode. Bookkeeping-only
stage-transition fixtures are unit tests, not evidence of an actual stage clear.

An isolated [four-worker check](results/defense/training/age-smoke-01/resume-config.json)
resumed run 12 at **10,447,616**, with lookback 128 and **16,384** new actions.
Relative to the earlier lookback-128 score-bin check, it changes the archive
criterion and its age interval/encoding, plus recorded source hashes and paths.
All **384** archive events had exact 128-action source/trigger offsets and
saved-age bin assignments. Four boot games completed; a restored segment
began generating own states but none had completed at the checkpoint.

Its [ten complete games](results/defense/training/age-smoke-01/checkpoint/evaluation.json)
averaged **8,456**, median **8,890**, best **10,460**, all stage 1, versus the
score-bin control's mean **9,304**, median **9,590**, best **10,480**. This is a
regression in that short comparison, not an improvement. Its
[2,552-action verified replay](results/defense/training/age-smoke-01/replay/replay.html),
full optimizer, config and log are preserved. It exited normally and is
excluded from the collector.

`defense-age-calibration-01` ([configuration](results/defense/training/age-calibration-01/resume-config.json))
tested the criterion at the normal **32-worker**
batch size, using run 21's stronger preserved parent at **11,750,144**, not the
four-worker smoke checkpoint. It retains rollout 256, batch 512, lookback 128,
eight boot-only workers and the parent's learning settings. The check permits
**131,072** new actions (4,096 per worker), so resets can actually be exercised
after complete boot games; merely filling archives would not test the proposed
mechanism. It used stopped run 21's slot and is excluded from the collector.

The check exited normally at **11,881,216**, with **32** complete boot games
and **seven** completed restored training segments. All **3,044** archive
events had correct source-age bins and 128-action offsets; **315** originated
in already-restored segments. Its [ten complete games](results/defense/training/age-calibration-01/checkpoint/evaluation.json)
averaged **10,387**, median **10,460**, best **10,480**, all stage 1 without a
mission. The [replay](results/defense/training/age-calibration-01/replay/replay.html)
verified **2,573** actions. Full logs, optimizer and replay are preserved. This
retains strong play but remains below the parent's ten-game mean **10,478**;
it is not evidence of improvement and has no matched 32-worker resumed control.

`defense-ppo-23-life-age` now continues from that checked calibration's
optimizer at **11,881,216**, with the same 32-worker settings, **unlimited**
training and 100,000-action evaluations. The
[configuration](results/defense/training/ppo-23-life-age/resume-config.json)
records its lineage; emulator episodes restart and archives refill from new
own experience. This longer trial tests whether reward-free later states help
progression, not a claimed success of the calibration. Runs 17, 20 and DQN 22
continue. The sole collector includes the new full trial and all old sources,
but neither small check. The shared best remains unchanged.

Run 23's [first ten complete games](results/defense/training/ppo-23-life-age/step-000011987712/evaluation.json),
at **11,987,712** (**106,496** actions after calibration), averaged **9,257**,
median **10,280**, best **10,360**, all stage 1 without a mission. Its
[verified replay](results/defense/training/ppo-23-life-age/first-replay/replay.html)
reproduced **2,507** actions. The full model/optimizer is preserved. This first
regular batch is below the calibrated parent, not an improvement; the longer
trial continues without replacing the stronger shared best.

By **12,184,320**, run 23 matched the shared best's **10,480** single-game
score, with [ten-game mean 9,710, median 10,460](results/defense/training/ppo-23-life-age/step-000012184320/evaluation.json).
All games remained in stage 1, with no mission completion. Its full optimizer
and [2,581-action verified replay](results/defense/training/ppo-23-life-age/replay-10480/replay.html)
are preserved separately. This recovers the score plateau but is still below
the starting calibration's mean; an equal best score does not promote the
global replay, and does not establish new progression.

Run 23 subsequently stopped cleanly at **14,174,976**, after **2,293,760**
new actions, **693** complete boot games, **449** completed restored segments
and **22** complete validation batches. None reached stage 2 or a mission.
Its strongest mean was
[10,464, median/best 10,480 at 13,486,848](results/defense/training/ppo-23-life-age/step-000013486848/evaluation.json);
the final validation mean was **9,984**, median **10,360**, best **10,460**.
The peak optimizer checkpoint, [full log](results/defense/training/ppo-23-life-age/metrics.jsonl)
and [final optimizer checkpoint](results/defense/training/ppo-23-life-age/final-checkpoint/state.json)
are preserved alongside its earlier verified best replay. Its depth plateau,
not a wall-clock budget, prompted reassigning the compute to a timing/history
experiment. The shared global replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_train --run runs/defense-age-calibration-reproduction \
  --resume results/defense/training/ppo-21-long-lookback/step-000011750144 \
  --artifacts runs/defense-age-calibration-reproduction/artifacts \
  --curriculum-cells age --curriculum-age-interval 32 \
  --eval-every 131072 --steps 11881216

# Continue the preserved calibration with unlimited learning:
venv/bin/python -u -m rl.defense_train --run runs/defense-life-age-reproduction \
  --resume results/defense/training/age-calibration-01/checkpoint \
  --artifacts runs/defense-life-age-reproduction/artifacts \
  --steps 0 --eval-every 100000
```

Remaining validation: stage 2/3 controls and mission-success detection are
supported by disassembly and parser tests, but **not yet exercised by an
unmodified complete playthrough reaching those stages**. Validate them when a
learned policy reaches them; do not patch memory or supply scripted expert play
to manufacture a success. The screen parser is single-player only. If a new
binary changes its HUD or awards extra ships, its assumptions must be audited
again. Keep models and diagnostics tied to the recorded executable hash.

## Real-game head search and the recurring barrier

The [six-generation ARS-style search](results/defense/training/ars-59-head-search/README.md)
held the strong PPO screen encoder fixed and searched its 20-action head from
complete, self-play game scores. It used 384 training games and 969,966 new
actions. The highest validation mean was 10,412; the final ten-game mean was
10,090, median 10,420, best 10,460. All remained in stage 1. The archived
full states and population returns make this negative trial reproducible.

The [verified final replay](results/defense/diagnostics/ars-59-final-replay/replay.html)
scored 2,620 / 2,620 / 2,620 / 2,600 on its four lives. Its
[unaltered screen panels](results/defense/diagnostics/shared-loss-ars-59/report.json)
show the ship left of the right-opening barrier again. The older global best
scored 2,620 on all four lives. The panel alignment uses visible white
flashes and later HUD changes, so it does not prove the exact collision time
or whether the wall or a projectile caused each loss.

The next [own-state focused calibration](results/defense/training/ars-focus-61/README.md)
restored 48 exact states captured 128 decisions before visible life losses in
12 new complete training games. It verified each state by replaying its
original subsequent actions and screens, then sampled new candidate actions
from those states. All 1,024 scored candidate continuations and all complete
boot validations still ended in stage 1. The small perturbations produced a
nearly flat score signal in later rounds. This motivates testing a wider
unbiased perturbation on the same sort of own-state continuations.

That [wider, matched trial](results/defense/training/ars-focus-62/README.md)
increased head sigma tenfold to 0.05. It changed many actions, but its 1,024
focused continuations still never reached stage 2, and median score gained
from the saved states fell from 2,410 to 1,640. All complete boot games also
remained in stage 1. The next test perturbs only the action-preference biases,
which can sustain a movement preference across screens while leaving the
learned visual feature weights fixed.

The [action-bias trial](results/defense/training/ars-focus-63/README.md)
also finished without passage in 1,024 focused continuations. Its median
score gain fell to 890 from the same 128-decision source positions and its
final ten-game boot mean was 9,510, all stage 1. The stronger global replay
is untouched. An earlier own-state start provides a separate test of whether
these perturbations need more approach time; it is not yet evidence of a
solution.

The [earlier 256-decision start](results/defense/training/ars-focus-64/README.md)
also failed: 1,024 focused continuations, none reached stage 2, and median
score gained from the saved state was only 60. These globally perturbed
action preferences often destroyed useful prior behavior. The next search
changes just one action preference at a time across all 20 commands, keeping
the remaining neural policy intact for each candidate.

The [256-decision coordinate trial](results/defense/training/ars-focus-65/README.md)
kept most candidate play strong: median score gained from those saved states
was 2,540, but **none of 1,280** focused continuations reached stage 2.
The best reused ten-game boot mean was 10,388; the final mean was 10,122,
all stage 1. The high, nearly equal per-candidate returns explain why
average score improvements alone are weak evidence at this obstacle.

Moving the coordinate-search start to [64 decisions before visible loss](results/defense/training/ars-focus-66/README.md)
did not help. All 1,280 continuations and all boot validation games remained
in stage 1; the final boot mean was 9,827. The closer state often left very
little score variation between actions. This does not identify the physical
collision point, but it limits the usefulness of a simple persistent action
preference at either 64 or 256 decisions before the visible loss.

The [screen-dependent single-row search](results/defense/training/ars-focus-67/README.md)
also failed its first calibration. It kept receiving high scores from the
old captured states while complete boot performance collapsed to a final
ten-game mean of 328, all stage 1. It stopped cleanly after five generations
and 800 focused continuations. This exposes a concrete training mismatch:
scores after the saved approach state alone do not protect the policy's
earlier route to that state. Future updates need a score-only acceptance
check that includes the earlier approach, while still avoiding training on
evaluation games.

The [two-generation score-gated pilot](results/defense/training/ars-score-gate-pilot-68/README.md)
confirmed that this mismatch can be caught using only fresh **training**
scores: a candidate's mean gain from own pre-loss states rose from 2,055 to
2,440, yet its four complete boot games averaged 2,785 against the parent's
9,960 on the same seeds. It was rejected, and the saved model hash remained
exactly the strong parent's. The full native logs and model/RNG states are
preserved. A longer, score-gated neural search followed.

That [16-generation score-gated run](results/defense/training/ars-score-gated-69/README.md)
accepted two candidates on fresh, paired complete boot **training** games.
The first improved eight-game mean score by 460; the second improved it by
only 48.75. Its later fixed ten-game validation mean was 9,204, all stage 1,
and none of its 2,624 focused continuations crossed to stage 2. This is
evidence that small boot-score margins can admit unstable updates at the
recurring barrier. The full run was stopped gracefully and archived. The
stronger training-selected generation-5 checkpoint is the parent for a
stricter continuation; the global verified best replay is unchanged.

That [stricter continuation](results/defense/training/ars-score-gated-70/README.md)
also stopped gracefully without stage passage after 11 generations, 1,804
focused continuations and 903,375 training actions. Its ten-game validation
means were 9,467 initially, 8,583 at generation 10 and 9,560 at the end.
The selected run-69 generation-5 parent averaged 8,823.3 on 12 new boot
training seeds, versus 8,905.8 for the original parent on those same seeds:
its prior eight-game gain was not a robust complete-game gain. The run-69
parent lost 45 of 48 lives between 2,500 and 2,650 displayed points; the
run-70 parent still lost 33 of 48 in that band. This quantifies the shared
failure region without claiming a physical collision coordinate. Since a
short pre-loss score gain can favor firing over a passage maneuver, a new
search will screen full boot games directly before any local score filter.

The [48-loss visible-screen atlas](results/defense/diagnostics/ars-69-barrier-atlas/README.md)
shows those recorded screens and actions directly. In its last-64-action
windows, the original policy chose 894 pure rightward versus 503 pure
leftward commands, but also 1,277 firing/side-fire commands, which can
interrupt stage-one movement. These windows can contain post-collision
animation, so this is a plausible training hypothesis, not proof of cause.

The [one-generation full-boot pilot](results/defense/training/ars-boot-pilot-71/README.md)
proved that all 40 symmetric action-row candidates can be played from boot,
shortlisted, compared on shared fresh training seeds, and independently
confirmed. It recorded 50 complete training games and 125,915 new actions.
The deliberately tiny two-game confirmation admitted a noisy update: the
fixed ten-game mean fell from 9,981 to 9,538, with no stage 2. This is a
functional test, not an improvement. The next run uses 4 / 16 / 16 fresh
boot games for screening / comparison / confirmation and keeps the original
strong policy as parent.

The ongoing [complete-boot search](results/defense/training/ars-boot-72/README.md)
reached generation 5 after 1,328 full training games and 3,335,069 new
actions. Its first candidate's 860.625-point comparison gain shrank to
33.125 points on a separate 16-game confirmation and was rejected. Three
later updates passed both gates, but the fixed ten-game boot mean at the
[restorable generation-5 milestone](results/defense/training/ars-boot-72/milestone-000005/state.json)
was 9,975 versus 9,981 for the original parent. All complete games still
ended in stage 1. This shows score stability under full-boot search, not
barrier passage or a new global best.

The complete-boot row search was then stopped gracefully at generation 11:
**2,864 complete training games** and **7,208,154 new actions**, all stage 1.
Five candidates passed both score gates, but fixed ten-game mean fell to
9,659 at generations 10 and 11, versus 9,981 at the original parent. Full
run files and the exact executing source are archived. Since a perturbation
of one command row cannot directly coordinate all commands sharing Space,
the next test factorizes candidate perturbations by physical keyboard key,
still symmetrically and without a prescribed movement direction.

The [one-generation key-factor pilot](results/defense/training/ars-key-pilot-73/README.md)
completed 50 full training games and 123,401 new actions. Its tiny two-game
confirmation admitted a candidate that lowered the fixed ten-game mean
from 9,981 to 9,327, all stage 1. This functional test confirms the native
factorized-policy path but not generalization. The ongoing
[key-factor search](results/defense/training/ars-key-74/README.md) starts
from the unchanged original policy and uses 4 / 16 / 32 distinct complete
training boot games for screening / comparison / confirmation, with a
150-point final margin. It changes learned physical-key preferences across
commands but gives no target key or route.

At the [generation-5 key-factor milestone](results/defense/training/ars-key-74/milestone-000005/state.json),
1,328 complete training games and 3,243,653 new actions had yielded one
update accepted on a separate 32-game confirmation set. Fixed ten-game boot
mean rose slightly from 9,981 to 10,043, but **all games still ended in
stage 1**. The model/RNG checkpoint is independently restorable; the
shared 10,480-point verified replay remains unchanged.

The factorized run then ended cleanly at generation 11 after **2,832 full
training games** and **6,833,703 new actions**, all stage 1. Its fixed
ten-game mean stayed 10,043 at generations 5, 10 and 11, with only one
accepted update. The complete run and exact executing source are archived;
the stronger global best replay remains untouched. A wider symmetric
physical-key perturbation is the next score-only test of the same bottleneck.

The [wider 0.05 pilot](results/defense/training/ars-key-wide-pilot-75/README.md)
made broad behavioral changes but degraded all one-game candidate scores
into the 280–7,750 range, with no stage 2 and no accepted update. A
[stratified-radius pilot](results/defense/training/ars-key-multiscale-pilot-76/README.md)
then covered 0.012–0.05 in one symmetric population: its strongest
one-game candidates came mostly from the conservative end, but it retained
broader candidates for exploration. Its apparent two-game comparison gain
reversed on independent confirmation, leaving the original model unchanged.
Both pilots are fully archived. The ongoing
[multi-radius complete-boot search](results/defense/training/ars-key-multiscale-77/README.md)
uses four screening, 16 comparison and 32 independent confirmation games
per candidate decision, with a 150-point final margin and no stage or
route-specific action rule.

At the [multi-radius generation-5 milestone](results/defense/training/ars-key-multiscale-77/milestone-000005/state.json),
1,264 complete training games and 2,997,896 new actions had produced one
score-gated update. Its fixed ten-game boot mean was 9,925 versus 9,981
at the original parent, and every game was still stage 1. The checkpoint
is preserved with full model/RNG state; no new best replay was promoted.

The multi-radius run ended cleanly at generation 11 after **2,832 complete
training games** and **6,726,107 new actions**. Only one update passed its
32-game confirmation; fixed ten-game mean stayed 9,925 at generations 5,
10 and 11. No candidate reached stage 2. The complete run and exact source
are archived, and the global verified best is unchanged. A read-only check
of the frozen encoder on the original policy's own saved visible screens
found that its approach features are separable from earlier same-life
features. The next test uses that self-play screen contrast solely to
propose context-sensitive key-weight changes; complete-game displayed score
remains the only update fitness, with no prescribed direction or route.

The [failure-context pilot](results/defense/training/ars-context-pilot-78/README.md)
verified the native path: a contrast computed solely from the original
policy's own rendered screens averaged approximately 0 at 128 decisions
before visible loss and 1 across the 64-/32-decision approach screens.
Symmetric key changes along this feature contrast produced candidate
scores from 300 to 10,480, but a two-game apparent gain reversed on
independent confirmation. All games were stage 1 and the original model
remained unchanged. The ongoing [context search](results/defense/training/ars-context-79/README.md)
uses 4 / 16 / 32 fresh complete boot-game sets and a 150-point final
margin. The contrast proposes where weights may matter; no action,
direction, hidden collision marker, demonstration or extra reward is
supplied to the policy.

The [context search](results/defense/training/ars-context-79/README.md)
then finished cleanly after **3,024 complete training games** and
**7,410,176 actions**. Two early score-gated updates raised the ten-game
mean from 9,981 to 10,155, but generations 5, 10 and 11 all evaluated at
10,155 and **no game reached stage 2**. Later short comparison gains
repeatedly shrank or reversed on 32 independent confirmation games. The
full native run, source and final model/RNG checkpoint are archived, and
the 10,480-point verified global replay is unchanged. A single learned
failure-context contrast may be too coarse to express the position- and
timing-dependent behavior needed at the shared opening; the next test
will use a broader subspace of the policy's own visible approach features,
while keeping complete-game displayed score as the only update fitness.

A [one-generation visual-subspace pilot](results/defense/training/ars-subspace-pilot-80/README.md)
used the same own rendered approach screens but added four principal
variation axes to an earlier-onset mean contrast. Its 54 complete boot
training games accepted one candidate, and ten-game mean rose from 9,981
to 10,266, still all stage 1. Because its pilot acceptance used only two
games per gate, an independent [32-seed paired training check](results/defense/training/ars-subspace-pilot-80/paired-confirmation.json)
compared that candidate with the original parent: 10,182.81 versus
9,802.19 mean displayed points on the same fresh seeds, a 380.63-point
gain, with the exact frozen visual encoder verified. Both policies still
ended in stage 1. This supports using the candidate as the next
training-selected starting point but does not establish passage.

The [long subspace search](results/defense/training/ars-subspace-81/README.md)
started from that independently confirmed pilot policy. At its
[generation-5 milestone](results/defense/training/ars-subspace-81/milestone-000005/state.json),
1,264 complete training games and 3,098,088 actions had produced no
further accepted update; fixed ten-game mean remained 10,266 and all
games remained in stage 1. A generation-4 candidate's independent
32-game gain was 149.375 points, just under the predeclared 150-point
gate, so it was rejected. The full model/RNG checkpoint is preserved,
and the verified global best replay is unchanged.

That run then stopped cleanly at generation 11 after **2,832 complete
training games** and **6,940,989 actions**, with zero accepted updates.
The fixed ten-game mean was still 10,266 at generations 10 and 11;
every training and validation game remained in stage 1. The complete
native archive and final model/RNG checkpoint are preserved. Two
predeclared-gate near-misses remain available in the saved population
plans for a larger fresh-training-seed score check; neither was quietly
promoted from this run.

That [64-seed paired check](results/defense/training/ars-subspace-near-miss-82/README.md)
has now finished. On 192 complete boot training games, the unchanged pilot
parent averaged **10,280.31**, while the generation-4 and generation-8
near-misses averaged **9,752.66** and **10,111.56**, respectively. Their
independent gains were negative (-527.66 and -168.75), and all games
stayed in stage 1. This directly refutes using either near-miss as a new
parent; it also confirms that short comparison margins near this barrier
can be misleading. The exact plans, paired results and probe source are
archived, with no weights or verified replay promoted.

A [wider 12-axis pilot](results/defense/training/ars-subspace-wide-pilot-83/README.md)
tested a larger own-screen proposal subspace and symmetric radii 1–8.
Its 40 one-game candidate scores ranged from 360 to 10,460; large radii
often destroyed the earlier route, while the top screen scores clustered
near radii 1–2.4. The best candidate lost 20 mean points in a separate
two-game comparison, so no update was accepted. Fixed ten-game mean
stayed 10,266, all stage 1. A more conservative radius with the same
broader visual subspace is the next complete-boot test.

The [conservative-radius twelve-axis search](results/defense/training/ars-subspace-84/README.md)
uses radii 0.5–2.5. At its [generation-5 milestone](results/defense/training/ars-subspace-84/milestone-000005/state.json),
1,328 complete training games and 3,280,299 actions had yielded no
accepted update or stage-2 game. A generation-5 candidate gained 507.5
points in the short 16-game comparison but only 134.69 on a separate
32-game confirmation, so the predeclared 150-point gate rejected it.
Fixed ten-game mean remains 10,266, all stage 1. The complete model/RNG
checkpoint is preserved and the global replay remains unchanged.

That run stopped cleanly at generation 11 after **2,960 complete training
games** and **7,325,563 actions**, with zero accepted updates or stage-2
games. Fixed ten-game mean remained 10,266 at generations 0, 5, 10 and
11. A generation-11 candidate's 741.88-point short-comparison gain fell
to 61.88 on independent confirmation. The complete native run and final
model/RNG checkpoint are archived. The repeated reversals motivate
testing a score-derived combination of many saved symmetric trials,
rather than selecting one apparent winner from a short comparison.

The [score-derived aggregate trial](results/defense/training/ars-aggregate-85/README.md)
combined all 220 saved symmetric direction pairs from those eleven
unchanged-incumbent generations, then played **288 new complete training
games** across screening, comparison and independent confirmation. Its
best candidate's +300-point 32-game comparison became **-21.09 points**
over 64 fresh confirmation games. All games were stage 1, so no update
or replay was promoted. Source hashes, candidate plans, full-game scores
and model/RNG states are archived. This eliminates one score-only way
of averaging the noisy directions; it does not prove the barrier is
unlearnable. A separate read-only action ablation can now test whether
the current policy's firing combinations interrupt movement at the
observed opening without altering the learned policy.

The diagnostic [Space-command ablation](results/defense/diagnostics/space-ablation-86/README.md)
then played 32 complete games per bias on shared fresh seeds. Almost
eliminating Space cut mean score from 10,023.75 to 1,808.75, so globally
disabling firing is not a viable shortcut. A modest -1 bias initially
looked +293.13 points better, but a separate [64-seed check](results/defense/diagnostics/space-ablation-87/README.md)
reduced that estimate to +154.53, and a predeclared final
[128-seed check](results/defense/diagnostics/space-ablation-88/README.md)
found only **+4.77 points**. Every game in all three checks stayed in
stage 1. No diagnostic head was saved or promoted. The exact per-game
records and source hashes are retained, supporting context-sensitive
control work rather than a hand-coded global firing rule.

The repeated losses may require an intervention well before the final
128 decisions. A new [early own-screen source](results/defense/training/ars-early-source-89/README.md)
replayed twelve complete training games from the independently confirmed
pilot-80 policy and retained only rendered screen histories 370 / 350 /
300 / 256 decisions before each of 48 visible life losses. Whole-trace
verification and file hashes are preserved. A frozen encoder found a
nontrivial early visual contrast, but this is proposal data only: the
acting policy receives no death clock, snapshot, route or extra signal.

The [early-screen pilot](results/defense/training/ars-early-pilot-90/README.md)
played 54 complete training games. Its best two-game comparison gained
770 points, but fresh two-game confirmation gained only 140, short of
the predeclared 150-point gate. No update was accepted, the fixed
ten-game mean remained 10,266, and all games were still stage 1.
The [longer early-screen search](results/defense/training/ars-early-91/README.md)
now tests the same proposal basis with independent 4 / 16 / 32-game
score gates. The sole verified-best collector watches this run as its
93rd source while preserving the existing 10,480-point global replay.

At the [generation-5 milestone](results/defense/training/ars-early-91/milestone-000005/state.json),
the longer early-screen run has played **1,392 complete training games**
and **3,451,610 actions**. One update passed both score gates at generation
4 (+298.13 / +225.63), raising the fixed ten-game mean from 10,266 to
10,388. A generation-5 short gain reversed below the confirmation margin.
Every training and validation game is still stage 1, so the repeated
barrier remains unresolved. The model/RNG milestone and exact source are
preserved; the in-run gates alone do not establish generalization.

That independent [64-seed-per-policy check](results/defense/training/ars-early-91/paired-confirmation.json)
averaged **10,218.91** for the previous pilot-80 parent and **10,373.28**
for the new early-window checkpoint: +154.38 points, just above the
predeclared 150-point gate. Both remained in stage 1. The exact frozen
encoder/value network was verified. This qualifies the checkpoint as a
score-selected parent for a new experiment but does not establish a
navigation breakthrough.

A fresh **training-only** pilot-80 replay localizes the repeated bottleneck:
the right-opening barrier is visible about 64 decisions before the first
visible loss, while the ship is still far left of the opening. A separate
[46-life screen-window source](results/defense/training/ars-bottleneck-source-92/README.md)
uses verified own screens at 128 / 96 / 64 / 32 decisions before loss;
two 140-point early-loss outliers are excluded from *proposal construction*
using only displayed per-life score. The first
[bottleneck-window pilot](results/defense/training/ars-bottleneck-pilot-93/README.md)
played 50 complete games but did not accept an update: its best candidate
was 50 points below the parent in the two-game comparison, and all games
remained stage 1. The longer early-window run continues independently.

A [symmetric effective-key pilot](results/defense/training/ars-effective-pilot-94/README.md)
then separated stage-one movement from Space-containing commands, which
do not translate the ship. This is an audited control-semantic prior, not
a hard-coded rightward route. Its best two-game candidate gained only
40 points, so no update was accepted; ten-game mean stayed 10,388 and
all games remained stage 1. The full source and results are preserved.

The original early-window run was interrupted during generation nine by
a code-edit/worker-spawn race, after generation eight had completed
2,176 training games and 5,405,520 actions. No state or replay was lost.
The [audited continuation](results/defense/training/ars-early-91-resume-95/README.md)
restores the exact generation-eight model/RNG checkpoint. The source
archive and feature-basis hashes match, and the re-run generation-nine
proposal plan and 160 screening games are byte-identical to the interrupted
prefix. The sole best-replay collector now watches all 96 artifact sources.

The [audited continuation](results/defense/training/ars-early-91-resume-95/README.md)
has finished the planned eleven generations: **3,024 logical complete
training games** and **7,527,929 actions**. A second in-run update passed
the 16-/32-game score gates at generation nine, but an independent
64-seed-per-policy comparison found only a +124.22-point gain over the
previous confirmed checkpoint, below the 150-point threshold for choosing
a new parent. Fixed ten-game mean ended at **10,397**, just 9 points above
the generation-five 10,388. All games remained stage 1, and the verified
global best replay stayed at 10,480. Full original/interrupted and resumed
records, checkpoints, exact source and replays are archived. The next
longer search will use the confirmed generation-five parent and the
more faithful movement-versus-Space proposal structure.

That [effective-key long run](results/defense/training/ars-effective-96/README.md)
is now active from the independently confirmed generation-five parent.
Its search treats firing combinations as non-translating in stage one,
samples all physical directions symmetrically, and retains the same
complete-boot score-only update gates. All **442 repository tests** passed
before launch. The sole collector now watches 97 sources and protects the
unchanged global best replay.

At the [generation-five effective-key milestone](results/defense/training/ars-effective-96/milestone-000005/state.json),
1,392 complete training games and 3,478,331 actions have yielded no
accepted update or stage-two game. A generation-four comparison gain
collapsed to +16.56 points on independent confirmation; a generation-five
candidate reached +148.44 on 32 confirmation games, narrowly below the
predeclared 150-point gate. It was not quietly promoted. Fixed ten-game
mean stays 10,388, and the 10,480-point global replay remains unchanged.
The full model/RNG checkpoint is preserved while the run continues.

The generation-five near-miss was also tested independently on
[64 fresh paired seeds](results/defense/training/ars-effective-96/near-miss-5.json):
candidate 10,238.28 versus unchanged parent 10,094.38, a **+143.91-point**
gain. This again misses the 150-point gate, so no weights are promoted.
Both policies stayed in stage one. The exact plan, score records and
probe source are preserved while the main run continues.

The [effective-key long run](results/defense/training/ars-effective-96/README.md)
then finished cleanly after **2,896 complete training games** and
**7,236,566 actions**, with zero accepted updates or stage-two games.
Fixed ten-game mean remained 10,388 at generations zero, five, ten and
eleven. The entire run, final model/RNG state, source and verified local
replays are preserved. The 10,480-point global replay is unchanged.
Repeating this coupled key-factor search longer is not supported by its
results; the next distinct test proposes score-gated changes to one
command row at a time using only the same verified own-screen features.

The first [per-command visual pilot](results/defense/training/ars-action-row-pilot-97/README.md)
played 50 complete boot training games from that unchanged parent. Its
best candidate gained only 70 points in the separate two-game comparison,
so no update was accepted. All games remained stage 1, fixed ten-game
mean stayed 10,388, and the global replay was unchanged. The exact
population plan and source are preserved. The next run broadens this
distinct proposal to two symmetric directions per command while keeping
complete-game score as its sole fitness.

A second [per-command pilot](results/defense/training/ars-action-row-wide-pilot-98/README.md)
covered each command twice with 40 symmetric directions. Its 94 full
training games found only a +10-point two-game comparison gain and no
stage-two game. Tied one-game screening scores clustered its shortlist
among early direction indices. The longer test will break *only exact
displayed-score ties* randomly while favoring distinct command rows;
higher scores always remain ahead of lower scores. This preserves
score-only selection while making exploration of the failure-window
action space less dependent on index order.

The [long per-command search](results/defense/training/ars-action-row-99/README.md)
is now active from the independently confirmed early-window parent with
two proposals for each of twenty commands, larger complete-boot score
gates and exact-score-only tie diversification. The full **444-test**
suite passed before launch. The best verified 10,480-point replay remains
the promotion floor.

At generation two, its score-only gates accepted a proposal changing
the pure `RIGHT` command row, but a separate 64-seed paired check
[reversed the apparent gain](results/defense/training/ars-action-row-99/paired-generation-2.json):
**-17.81** displayed points relative to the independently confirmed
parent, with both policies still in stage one. The model/RNG milestone,
proposal plan and exact check source are preserved. The live search
continues, but this short-sample update is not treated as a confirmed
improvement or promoted replay.

A fresh, [matched-seed own-screen diagnostic](results/defense/diagnostics/ars-action-row-99-matched-losses/README.md)
then reproduced eight complete games for each policy. The confirmed
parent lost 31 of 32 lives after gaining 2,500–2,650 points; the
candidate lost all 32 there. One unusual 1,060-point early parent loss
accounts for 94% of the candidate's apparent eight-game score advantage.
The recurring barrier is therefore a much more specific failure than
low average score alone suggests. These screens are diagnostic only;
visible loss timing is not an exact collision label or a route oracle.

A [same-screen action-probability probe](results/defense/diagnostics/ars-action-row-99-matched-losses/action-probabilities.json)
then confirmed the generation-two update changed only the pure `RIGHT`
row, but did not create a uniform steering bias: its mean probability on
parent-owned screens rose from 1.48% to 3.22% at 64 decisions before
visible loss, then fell from 7.60% to 5.66% at 32 decisions. This is a
read-only counterfactual, not evidence that a particular command caused
or prevented a physical collision. The focused 12-test suite passes.

The [generation-five search milestone](results/defense/training/ars-action-row-99/milestone-000005/state.json)
is preserved after **2,288 complete training games** and **5,769,059
actions**. A candidate's +256.25-point 16-game comparison reversed to
-82.50 in the fresh 32-game gate, so it was rejected. The fixed ten-game
mean is now **10,032** versus 10,388 at the start, all stage one. The
internally accepted generation-two head remains in the live run, but its
independent 64-seed check is negative; it is not a confirmed improvement.

At the [generation-nine milestone](results/defense/training/ars-action-row-99/milestone-000009/state.json),
**4,208 complete games and 10,604,410 actions** had produced a second
internally accepted update, still all stage one. A [new 64-seed check](results/defense/training/ars-action-row-99/paired-generation-9.json)
against the last independently confirmed parent reversed its short-run
gain: candidate **10,034.53**, parent **10,212.19**, a **-177.66-point**
difference. It is preserved but not treated as a confirmed parent or new
best replay. The planned run continues to its final checkpoint.

The [per-command run](results/defense/training/ars-action-row-99/README.md)
then finished cleanly at **11 generations, 5,136 complete training games
and 12,945,774 actions**. Every training/validation game was stage one.
The last candidate's +167.50-point comparison gain reversed to -318.44
on fresh confirmation games. Fixed ten-game mean ended at **10,286**,
below the initial 10,388. The entire model/RNG history and a separately
verified local 10,480-point replay are archived. The original global
best replay remains unchanged. Continuing the identical one-row search
is not supported by these results; subsequent experiments should test a
distinct way to explore the recurrent visible failure window while
keeping screen-only acting and score-only reward.
The full **445-test** repository suite passed after this archive.

The [own-loss-state random-control study](results/defense/training/macro-explore-100-104/README.md)
then tested **11,280 training-only continuations** from the confirmed
neural policy's own replay-verified stage-one loss approaches. Two
direction-neutral command holds per continuation were tried at 128- and
370-decision lookbacks, with both all-command and distinct-stage-one
sampling, plus longer holds. No continuation reached stage two. The
10,000-attempt short-hold run extended a visible life by at most 46
decisions past its original loss marker; long holds generally worsened
survival. These are randomized exploration actions, not learned-policy
play, and no weights or replay were promoted. This negative result
argues for screen-conditioned learning rather than more of the same
open-loop random holds; it does not prove the barrier is unavoidable.
The full **449-test** suite passes with this training-only probe.

The first [late own-screen phase-action pilot](results/defense/training/ars-phase-pilot-105/README.md)
then built two independent feature axes from the model's own verified
96-/64-/32-decision rendered-screen approaches. Symmetric proposals
covered all twenty command rows, and only whole-game displayed score
could accept an update. In **54 complete training games** its best
two-game comparison gain was 50 points, below the 150-point gate;
fixed ten-game mean remained 10,388 and every game stayed in stage one.
The full model/RNG state, proposal plan, source and verified local replay
are preserved. A longer score-gated test will check this focused visual
representation more thoroughly, with no prescribed steering direction.
The full **452-test** repository suite passes with this new proposal mode.

The [long late-phase visual search](results/defense/training/ars-phase-106/README.md)
is now active from the independently confirmed parent. It tests two
signed random directions per command, with fresh 4-/16-/64-game
complete-boot score gates and a 150-point confirmation margin. The
previous 100 collector sources plus the pilot and new run are monitored
by the sole independent replay collector. Exact code/configuration and
the source basis are preserved; the verified global best is unchanged.

At its [generation-five checkpoint](results/defense/training/ars-phase-106/milestone-000005/state.json),
this phase-conditioned run has completed **2,544 full training games / 6.41
million actions** with no stage-two reach or accepted proposal. Its fixed
ten-game mean remains **10,388**, and the parent model/RNG state is preserved.
The first three nominated comparisons improved on 16 games but failed to
clear the separate 64-game 150-point gain requirement (or reversed),
illustrating again why short-score gains cannot be taken as bottleneck
progress. The run continues to its planned eleven generations.

The next distinct training approach is now wired into PPO: optional
`--curriculum-trigger life-loss --curriculum-lookback L` archives only its
own actual same-life states L actions before visible loss, and optional
`--curriculum-restored-life-only` focuses restored segments on the first
visible loss. These existing native training-environment features are now
available to the PPO learner, with `--life-terminal` required for the
restored-life-only option. Reserved boot workers still play complete games;
evaluation never restores states. The policy still sees four rendered
screens and learns from displayed score only. A two-worker native training
and checkpoint-resume smoke test passes; a longer continuation will be
evaluated from the strong preserved PPO optimizer after the active search
finishes. This is a test of practice allocation, not a collision oracle or
proof that the saved states are recoverable.

The [first own-loss PPO continuation](results/defense/training/ppo-own-loss-107/README.md)
completed **131,072 new actions**, 508 restored practice segments and 24
new boot games. Its first and final fixed ten-game means were 10,269 and
10,137, both stage one. An independent matched **64-game fresh check**
found parent / first / final means **9,673.59 / 10,194.84 / 9,977.66**.
The first checkpoint reduced below-9,000-point games from 22 to seven,
improving 42 of 64 paired seeds, but none of the 192 fresh games reached
stage two. Its verified local 10,480 replay and both full optimizers are
preserved. This suggests the targeted practice can reduce early failures
without yet changing the recurring late barrier; the first checkpoint
will be the parent of a longer, more exploratory continuation.

That [longer PPO continuation](results/defense/training/ppo-own-loss-108/README.md)
used the independently stronger first optimizer with higher training
entropy. It completed **524,288 new actions**, 47 boot games and 1,859
restored own-loss segments. Its four fixed ten-game means were **9,464 /
8,903 / 10,178 / 10,457**, all stage one. The apparent final rebound
failed a new paired **64-game** check: final mean **10,037.19** versus
parent **10,183.28**, a **-146.09-point** change; below-9,000 games grew
from seven to eleven. No training or fresh game reached stage two.
The complete checkpoint/log/replay archive is preserved, but this is not
a confirmed successor to the first run-107 checkpoint. The sole replay
collector now watches **104** sources, including both new PPO runs, and
has not displaced the independently verified global best.

A [read-only fixed-ten-command probe](results/defense/diagnostics/action-profile-109/README.md)
then tested whether stage-one fire-key aliases were masking the repeated
failure. Simply removing their logits from the **frozen** strong PPO policy
collapsed 64-game mean score from **10,157.97 to 996.88**; every masked
game scored below 3,000, with no stage-two reach. This mask also discarded
most learned firing probability, so it is not a fair test of a newly trained
compact-action model. We will first check a probability-preserving alias
collapse. No trained model or verified best replay was changed.

In parallel, generation ten of the [late-phase ARS search](results/defense/training/ars-phase-106/README.md)
internally accepted a +203.91-point proposal on 64 confirmation games,
but an additional **64 fresh matched games** reversed it to **-95.63**
against the last independently confirmed parent. Both stayed in stage one.
Its full model/RNG milestone is preserved, but this short-sample head is
not a confirmed improvement or a replacement for the verified replay.

The [full phase-search run](results/defense/training/ars-phase-106/README.md)
then finished cleanly after **11 generations, 5,648 complete games and
14.24 million actions**. Its generation-eleven nominee also reversed in
the fresh gate (+248.13 comparison to -26.09 confirmation). Final fixed
ten-game mean/median/best were **10,450 / 10,450 / 10,480**, still all
stage one. Every checkpoint, candidate plan, full log and a 2,565-action
verified local replay are archived. The internally accepted generation-ten
head remains negative on the additional independent 64-seed check, so
the global best and last independently confirmed search parent stay put.

The [probability-preserving canonical-fire probe](results/defense/diagnostics/action-group-110/README.md)
then merged the nine stage-one fire-alias logits by log-sum-exp while
retaining their total probability and sending the draw through Space.
On 64 new matched full games, **all paired displayed scores matched**:
both means were **10,251.09**, all stage one. This read-only result
explains why the naive ten-command mask failed (it discarded learned
fire mass) and supports a trainable grouped-action PPO test. It does not
prove identical native trajectories or later-stage command equivalence;
the verified global best remains unchanged.

A [trainable canonical-fire PPO variant](results/defense/training/ppo-canonical-112/README.md)
now groups fire-key likelihoods by log-sum-exp in both categorical
sampling and PPO's actor/entropy loss, then emits canonical Space for
that fixed group on every screen. It keeps the twenty-logit neural head
and exact strong parent optimizer. A [same-parent ordinary-action control](results/defense/training/ppo-canonical-control-113/README.md)
will isolate this change over 524,288 new actions per arm; four workers
per arm remain full-boot and independent evaluation never restores a
snapshot. Native short training, replay verification, optimizer resume
and the full **462-test** suite pass. This is action abstraction, not a
scripted steering rule, hidden-state input, demonstration or new reward.

The [matched grouped-action and ordinary PPO continuations](results/defense/training/ppo-canonical-112/README.md)
both finished **524,288 new actions** from the same full parent optimizer.
The grouped arm's four fixed ten-game means were **9,457 / 9,789 /
8,446 / 8,668**; the [control](results/defense/training/ppo-canonical-control-113/README.md)
scored **10,438 / 10,428 / 10,139 / 9,314**. All remained stage one.
Choosing each arm's best checkpoint by those fixed games, then playing
**128 new matched complete games**, gave grouped / common parent /
control means **9,825 / 10,173.75 / 10,370.78**. The grouped arm doubled
sub-9,000 scores from 15 to 30 versus parent, despite improving 72 paired
seeds; the control reduced them to five, improving 91 paired seeds.
No fresh game reached stage two. The grouped abstraction is not a
confirmed successor. The control's +197.03 fresh mean gain makes its
first checkpoint a stronger **score** parent with full optimizer/RNG
preserved, but it still hits the repeated barrier. Both complete runs,
logs and verified local 10,480-point replays are archived; the global
best remains unchanged.

The next [earlier own-loss PPO trial](results/defense/training/ppo-early-loss-114/README.md)
tests a concrete response to the repeated visible barrier: rewind 256
decisions before the learner's own visible loss, versus a
[same-parent 128-decision control](results/defense/training/ppo-early-loss-control-115/README.md).
Both resume the independently confirmed ordinary-action score checkpoint
at 8,538,880 actions, retain four full-boot workers, and are scheduled for
524,288 new actions and four complete-game checks each. Their independent
collector sources are isolated; only a native-verified higher-ranked replay
can replace the global best. This is a training-reset comparison, not
collision localization, a hand-coded path, or evidence of stage passage.

The [longer own-loss rewind](results/defense/training/ppo-early-loss-114/README.md)
and [same-parent control](results/defense/training/ppo-early-loss-control-115/README.md)
have now both completed 524,288 new actions. Fixed ten-game means were
**10,385 / 10,193 / 10,292 / 10,429** for the 256-decision rewind and
**10,430 / 9,708 / 10,229 / 10,081** for the 128-decision control; every
game remained in stage one. Their validation-selected checkpoints were
then evaluated on **128 new matched complete games** alongside their
confirmed common parent. Fresh means were **10,399.69 / 10,291.80 /
10,416.48** for longer/control/parent, respectively. All 384 fresh games
lost in stage one. The longer rewind reduced sub-9,000 scores to one
versus two in the parent, but worsened 88 of 128 paired games and was
16.80 mean points lower. The control had nine sub-9,000 games. These are
preserved negative barrier results, not a new confirmed score parent.
The selected models' fresh-best replay traces each passed independent
neural-action verification; the globally ranked best replay is unchanged.

The next same-parent comparison tests two more specific explanations for
that repeated failure. [Run 116](results/defense/training/ppo-near-loss-116/README.md)
rewinds to the learner's own visible state 64, not 128, decisions before
loss; input-responsiveness diagnostics cover this region but do not prove
recoverability. [Run 117](results/defense/training/ppo-long-credit-117/README.md)
keeps the 128-decision window and changes only PPO GAE lambda from 0.95 to
0.995, so later displayed-score consequences can affect earlier action
advantages within a rollout. Both resume the confirmed control-113 model,
optimizer and RNG, with 524,288 new actions planned. The completed run-115
is their matched ordinary continuation. The independent collector now
watches **110** sources; the verified global best remains protected.

Both trials completed their predeclared 524,288 new actions and four fixed
ten-game checks. The 64-decision own-loss arm's fixed means were **9,949 /
9,729 / 10,346 / 9,680**; the longer-credit arm's were **7,888 / 9,445 /
10,152 / 10,402**. Every training and validation game stayed in stage one.
After validation-only checkpoint selection, [128 new matched complete games](results/defense/training/ppo-near-loss-116/comparison.json)
on seeds 600800–600927 produced parent / ordinary control / near-loss /
long-credit means **10,398.91 / 10,377.27 / 9,764.30 / 10,121.80**.
All **512 fresh games** remained in stage one. The closer reset yielded
4,203 short restored practice segments but 30 sub-9,000 fresh games versus
four for the parent; longer credit had 14. Neither is a confirmed successor.
Full model/optimizer/RNG histories and independently verified fresh-best
replays are archived in [run 116](results/defense/training/ppo-near-loss-116/README.md)
and [run 117](results/defense/training/ppo-long-credit-117/README.md).
The repeated visible navigation bottleneck and global verified best remain
unchanged.

The next [per-life parameter-noise PPO test](results/defense/training/ppo-persistent-noise-118/README.md)
returns to the strongest independently confirmed score checkpoint. Its
training-only actor-bias perturbation persists for each ship life while
the learned screen-conditioned policy still chooses every key. The
existing 128-decision own-loss continuation is the matched ordinary
control. Four complete-game checks are planned over 524,288 new actions,
then 128 fresh matched complete games. Only an unperturbed model's native
stage transition and verified replay can establish progress. The sole
collector now watches **111** sources; the global best remains protected.

The [persistent-noise PPO run](results/defense/training/ppo-persistent-noise-118/README.md)
completed 524,288 new actions with four fixed ten-game means **10,213 /
10,466 / 9,946 / 10,464**, all stage one. Validation selected the second
full optimizer/RNG checkpoint. On 128 new matched complete games, it
averaged **10,365.70** versus the confirmed parent's **10,344.77**, but
the paired median was zero and the trimmed mean nearly zero. The
predeclared second 128-game set reversed the mean margin (**10,386.02**
versus **10,418.28**). Across both sets, selected / parent means were
**10,375.86 / 10,381.52**; none of the 512 games reached stage two.
Per-life noise increased exact-ceiling frequency but not confirmed mean
score or barrier passage. All training/validation/fresh games stayed in
stage one; full histories and neural-verified fresh replays are preserved.
The strongest independently confirmed score parent and global verified
best replay remain unchanged.

A [longer continuation](results/defense/training/ppo-persistent-noise-long-119/README.md)
now resumes that noise trial's **full 8,801,024-action optimizer, policy RNG
and per-life noise RNG** from its validation score peak, rather than claiming
the borderline fresh score result as a new parent. It retains all screen-only,
displayed-score-only and own-loss training settings for **2,097,152 more
actions** and 16 planned unperturbed complete-game checks. A frozen
validation-selected checkpoint will then face 128 new matched games. The
single collector watches **112** sources and keeps the native-verified best
replay available independently of this exploratory run.

The [long persistent-exploration continuation](results/defense/training/ppo-persistent-noise-long-119/README.md)
finished **2,097,152 additional actions**, 220 boot games, 8,206 restored
segments and sixteen complete ten-game validations, with no stage-two
training or validation game. Its seventh checkpoint at **9,718,528** had
the earliest peak mean **10,470** and was frozen before fresh checking.
On two predeclared new 128-game seed sets, it averaged **10,436.99** versus
**10,370.47** for the last independently confirmed ordinary-action score
parent; sub-9,000 games fell from ten to one. The immediate noise parent
averaged **10,352.97** on the same 256 seeds. These are meaningful
independently checked **score** gains, but **all 768 fresh games** still
ended in stage one. The new full optimizer/RNG checkpoint can serve as a
score-training parent; it is not a mission solution, and the older native-
verified global best replay remains preserved. All model/optimizer/RNG
history, source snapshots and independently verified fresh-best replays
are archived with the run.

A [read-only loss-screen check](results/defense/diagnostics/long-noise-119-losses/README.md)
of the new score parent's and older parent's independently verified fresh
best-effort replays found **2,620 displayed points on every one of eight
lives**, again around the broad right-opening barrier sequence. Their
selected seeds differ, and the white flash is not a collision timestamp;
this does not prove identical physical failure sites. It does show that the
confirmed mean-score gain has not yet changed the recurring visible
late-life failure. The panels never enter training or action selection.

The recurring barrier prompted a [frozen screen-history check](results/defense/training/ppo-wide-history-120/README.md)
of the new score parent on the same 32 fresh complete-game seeds. At its
trained stride 1 it averaged **10,456.88** points; changing only the spacing
between its four input frames to strides 2, 4 or 8 yielded **5,202.50**,
**996.88** and **316.88**, respectively. All remained stage one. This
measures distribution shift, not whether longer history can be learned or
whether it would resolve the barrier. A bounded stride-2 PPO adaptation from
the exact full model/optimizer/RNG parent is therefore planned; the global
verified best remains protected.

Run 120 completed the planned **524,288** stride-2 adaptation actions. Its
fixed ten-game means recovered to **10,349 / 10,323 / 10,394 / 10,452**,
which triggered the predeclared independent check. On 128 new matched
complete games, the selected stride-2 checkpoint averaged **10,344.77**
versus **10,455.63** for the unchanged stride-1 score parent: **−110.86**.
Selected had five sub-9,000 collapses versus zero for the parent, and both
remained entirely in stage one. The selected policy's 10,480-point local
best-effort [replay](results/defense/training/ppo-wide-history-120/fresh-selected-replay/replay.html)
independently verifies 2,539 learned actions, but cannot displace the older
global best. Full optimizer/RNG, metrics, source and all four checkpoints are
preserved in the [run archive](results/defense/training/ppo-wide-history-120/README.md).
The recurring barrier remains unsolved; wider history alone was not a win.

A distinct [learned-continuation PPO pilot](results/defense/training/ppo-continue-121/README.md)
now tests whether the policy can sustain its *own* previous key across
ordinary action intervals. The twenty established actor rows and value/
screen encoder transfer unchanged from the run-119 score parent; a neutral
twenty-first `CONTINUE_PREVIOUS` row is learned with fresh PPO optimizer and
score-only experience. The one-key memory resets at visible life/episode
boundaries. The model never receives an obstacle detector, route script,
hidden RAM, demonstrated actions or stage-based reward. This is not the
earlier DQN multi-step hold experiment; every step remains a learned
decision. No-learning and one-update native smokes passed, including a
2,537-action independently verified local replay. The predeclared million-
action pilot and independent score/mission gates are in its README.

Run 121 finished all **1,048,576** planned actions and eight complete
ten-game checks. The fixed means were **10,148 / 10,426 / 10,388 / 10,450 /
10,114 / 10,314 / 10,328 / 10,472**; the final full state was selected.
Two predeclared, independent [128-game matched sets](results/defense/training/ppo-continue-121/comparison.json)
confirm a **+26.48-point score-consistency gain** over the run-119 parent
across 256 games per policy, with 114 paired wins, 29 losses and 113 ties.
The new model scored exactly 10,480 on 202/256 games versus 119/256 for
the parent. Nevertheless all **512** fresh games stayed in stage one.
The full run, source, optimizer/RNG states and native-verified local replays
are [preserved](results/defense/training/ppo-continue-121/README.md), while
the shared best remains the older equal-ranked 10,480-point replay.

The [new loss-screen diagnostic](results/defense/diagnostics/continue-121-losses/README.md)
again finds four 2,620-point lives near the same right-opening barrier. A
read-only re-execution matching every recorded screen/action/reward found
just **one** `CONTINUE_PREVIOUS` choice among 2,551 neural decisions and
none in the four pre-loss 64-action windows. Thus the score gain does not
demonstrate learned persistence through the barrier. A longer identical
run would not be a meaningful test of the proposed mechanism until the
continuation action is actually explored.

A read-only calibration on the run-121 model's own verified screens exposed
why the new action went unused: its initial mean selection probability was
only **0.0717%**, then **0.0116%** after training. A neutral **+7** bias to
that extra action row, leaving all twenty parent rows untouched, predicts
12.76% overall and 6.80% near the four observed loss windows. A separate
no-learning native replay then verified **320** learned continuation choices
among 2,531 actions (12.64%), across all lives. This is not a scripted
direction or an added reward; it makes the categorical action genuinely
available to PPO. A [predeclared calibrated pilot](results/defense/training/ppo-continue-calibrated-122/README.md)
will test whether that changes the repeated stage-one outcome while keeping
the earlier confirmed score parent and global best protected.

Run 122 completed all **1,048,576** calibrated-continuation actions. Its
fixed ten-game means were **10,126 / 10,406 / 10,406 / 10,416 / 10,166 /
10,407 / 10,243 / 10,456**, all stage one. The final checkpoint passed the
predeclared gate, but the [128 fresh matched games](results/defense/training/ppo-continue-calibrated-122/comparison.json)
averaged **10,450.94**, below both the run-121 score parent (**10,456.80**)
and run-119 ordinary parent (**10,453.13**) on the same seeds. All **384**
games remained stage one. The full optimizer/RNG history, exact source and
verified local replay are [archived](results/defense/training/ppo-continue-calibrated-122/README.md);
the older confirmed score parent and global best remain unchanged.

The [read-only loss check](results/defense/diagnostics/continue-calibrated-122-losses/README.md)
again shows four 2,620-point lives at the recurring right-opening barrier.
Unlike run 121, the calibrated model did use its extra action: a native-
matched replay sampled it **132 times in 2,576 decisions**, yet only
**2/2/2/0** times in the four 64-action pre-loss windows. Pure physical
movement choices occupied 70–84% of those windows, but that did not move
the policy beyond the obstacle. This rules out the simple explanation that
the option was *never* tried; it does not identify the exact fatal action
or prove that a learned action-duration mechanism could never work.

A [symmetric key-factor PPO exploration trial](results/defense/training/ppo-key-noise-123/README.md)
now tests a different failure hypothesis: independent per-command noise may
not produce coherent directional trials even when the screen policy knows
physical key combinations. Training-only zero-mean factors for each known
key make related commands fluctuate together, with left/right treated
identically; the frozen model and replay use no such noise. This is not a
hard-coded direction or route, and score-only reward and original episode
boundaries remain unchanged. A tiny optimizer-resume smoke passed, including
a native-verified 2,552-action stage-one replay. The bounded full pilot's
checkpoint and fresh-comparison gates are fixed in its README.

Run 123 completed all **1,048,576** planned additional actions with
training-only symmetric key-factor noise. Its fixed ten-game means were
**10,466 / 10,470 / 9,574 / 10,478 / 10,468 / 10,400 / 10,476 /
10,232**; the fourth full optimizer/RNG checkpoint was selected. No
training or validation game entered stage two. On [128 new matched games](results/defense/training/ppo-key-noise-123/comparison.json),
selected / confirmed run-121 parent / older ordinary parent averaged
**10,473.67 / 10,470.23 / 10,449.30**, all stage one. The selected
policy's +3.44-point margin over its actual score parent had 17 paired
wins and 18 losses, so it is not promoted as a new score parent. Its
[native-verified local replay](results/defense/training/ppo-key-noise-123/fresh-selected-replay/replay.html)
and complete [run archive](results/defense/training/ppo-key-noise-123/README.md)
are preserved; the shared best remains untouched.

The [read-only loss sheets](results/defense/diagnostics/key-noise-123-losses/README.md)
again show four 2,620-point lives near the broad right-opening barrier.
Byte-exact re-execution of the selected replay found **zero** frozen-policy
`CONTINUE_PREVIOUS` choices among 2,528 decisions, including all pre-loss
windows. Coherent training noise did not become a learned escape behavior.

The next [stronger key-factor pilot](results/defense/training/ppo-key-noise-strong-124/README.md)
changes only training-noise standard deviation **1 → 2**, restarting from
the same frozen run-121 full state rather than selecting run 123's near-tied
checkpoint. Its symmetric factors remain direction-neutral; complete-game
score is still the only reward and the original best replay is protected.
Eight fixed checks and any fresh comparison are predeclared in the run plan.

Run 124 completed its planned **1,048,576** stronger-noise actions and eight
ten-game checks (fixed means **9,904 / 10,292 / 10,210 / 10,434 / 10,462 /
10,478 / 9,736 / 9,984**). Its sixth checkpoint passed the fixed gate,
but [128 fresh matched games](results/defense/training/ppo-key-noise-strong-124/comparison.json)
averaged only **10,402.34**, against **10,456.88** for run 121 and
**10,473.59** for run 123. None of the 384 fresh games reached stage two.
Its [verified local replay](results/defense/training/ppo-key-noise-strong-124/fresh-selected-replay/replay.html)
again lost four 2,620-point lives at the recurring barrier, with just 10
sampled continuation choices in 2,500 actions and none in the pre-loss
windows. The [full negative run](results/defense/training/ppo-key-noise-strong-124/README.md)
and [read-only loss frames](results/defense/diagnostics/key-noise-strong-124-losses/README.md)
are preserved. The run-121 score parent and older global best remain
protected; stronger exploration noise by itself did not solve this obstacle.

The next [boot-diversity allocation test](results/defense/training/ppo-boot-diversity-125/README.md)
responds to the 116 complete boot games versus 4,172 short restored segments
in run 124. It resumes the confirmed run-121 full optimizer and changes only
four to twelve reserved boot workers, leaving four self-restoring workers.
This tests whether more complete approaches improve exploration of the
recurring barrier; it does not change model inputs, reward, actions or native
evaluation. The bounded gate and independent comparison are predeclared.

Run 125 completed all **1,048,576** planned actions, adding **305** full
boot games and **1,331** own-restored segments. Its fixed ten-game means
were **10,405 / 10,432 / 10,245 / 10,472 / 10,444 / 10,466 / 10,480 /
10,456**, all stage one. The seventh checkpoint's perfect fixed mean did
not generalize: [128 fresh matched games](results/defense/training/ppo-boot-diversity-125/comparison.json)
averaged **10,366.02** against **10,470.23** for the confirmed parent, with
18 paired wins, 37 losses and 73 ties. No fresh game reached stage two.
Its [verified local replay](results/defense/training/ppo-boot-diversity-125/fresh-selected-replay/replay.html)
again lost four 2,620-point lives at the recurring obstacle. The
[read-only loss sheets](results/defense/diagnostics/boot-diversity-125-losses/README.md)
show only 1/2/0/1 learned continuation choices in the four 64-action
pre-loss windows. The [full negative run](results/defense/training/ppo-boot-diversity-125/README.md)
is archived; the run-121 score parent and global best remain unchanged.

The [native held-key diagnostic](results/defense/diagnostics/boot-diversity-125-losses/README.md)
reproduced the run-125 selected replay exactly to its own pre-loss states,
then tried all 20 constant physical commands from 192, 128 and 64 actions
before the first visible loss. None of the 60 diagnostic suffixes passed
stage one. Holding RIGHT delayed visible loss versus NOOP from the earlier
two anchors but hastened it from the 64-action anchor, while earning much
less score than the learned suffix from the earlier anchors. This is a
single-state, diagnostic-only observation, not a route or training example.

A [bounded within-life key-noise test](results/defense/training/ppo-key-noise-windowed-126/README.md)
now tests whether zero-mean symmetric key factors changing every 32 own
actions, rather than staying fixed for the whole life, give the learned
policy more useful coherent exploration. It keeps the original visible-
score reward, screen-only policy and native evaluation. Full source, tests,
fixed selection and fresh-game gates are recorded in its run plan.

Run 126 completed all **1,048,576** planned within-life-noise actions and
eight fixed ten-game checks: **10,469 / 10,480 / 10,475 / 10,468 / 10,404 /
10,480 / 10,228 / 10,232**. Its 32-action redraw schedule logged over
33,000 factor draws, but no training or validation game reached stage two.
The selected second checkpoint's [128 fresh matched games](results/defense/training/ppo-key-noise-windowed-126/comparison.json)
averaged **10,428.05**, below the run-121 score parent (**10,445.78**)
and above the per-life-noise run-124 arm (**10,396.48**). Paired outcomes
did not favor the new policy over either comparator, and all **384** fresh
games remained in stage one. Its [verified local replay](results/defense/training/ppo-key-noise-windowed-126/fresh-selected-replay/replay.html)
again has four 2,620-point losses at the familiar obstacle, with just one
sampled continuation choice among 2,583 neural decisions. The
[full negative run](results/defense/training/ppo-key-noise-windowed-126/README.md)
and [loss sheets](results/defense/diagnostics/key-noise-windowed-126-losses/README.md)
are preserved. Neither this within-life schedule nor its per-life control
has produced a verified passing policy; the global best remains protected.

A [frozen score-value diagnostic](results/defense/diagnostics/score-value-barrier-01/README.md)
adds a training-signal clue without modifying the learner. On the confirmed
run-121 replay, its critic predicts about **1,433–1,481** additional
displayed points 64 decisions before each visible life loss (the suffixes
earn 1,520–1,540), then only **6–44** points 32 decisions before loss
(actual suffixes earn 0–20). The repeated late score jump is learned, but
the remaining return near the obstacle is small on these failed paths.
This is one selected four-life replay, not proof that all evasive actions
are unrewarded or that the game is impassable. It focuses the next search
on discovering genuinely higher-score own trajectories rather than another
stage-one score tie, while preserving the score-only/no-oracle rules.

The next [diverse own-screen reset trial](results/defense/training/ppo-screen-frontier-127/README.md)
keeps the confirmed run-121 model, optimizer, screen policy, reward and
own-loss 128-action rewind, but changes the training archive from 16×4
score bins to up to 128 distinct HUD-excluded visible-screen cells. It tests
whether broader approaches to the repeated obstacle are needed before
score-only learning can see new return. Cells select only previously
reached training reset states; they do not enter the policy or reward.
Fixed/fresh selection gates and actual-occupancy reporting are predeclared.

Run 127 completed its planned **1,048,576** actions and eight ten-game
checks (fixed means **10,480 / 10,462 / 10,222 / 10,194 / 10,440 /
10,460 / 10,224 / 10,446**), all stage one. Its own-loss archive
encountered **170 distinct screen cells** and every worker's latest
inventory filled all **128** slots, confirming a broader reset population.
Nevertheless, the selected first checkpoint's [128 fresh matched games](results/defense/training/ppo-screen-frontier-127/comparison.json)
averaged **10,402.42** versus **10,475.86** for the run-121 parent; all
**256** games still lost in stage one. The new policy won more individual
seed comparisons than it lost, but severe low-score tails lowered its
mean, so it is not a confirmed score parent. Its
[verified local replay](results/defense/training/ppo-screen-frontier-127/fresh-selected-replay/replay.html)
again lost four 2,620-point lives at the same obstacle, with zero sampled
continuation choices among 2,563 decisions. The
[full run](results/defense/training/ppo-screen-frontier-127/README.md)
and [loss sheets](results/defense/diagnostics/screen-frontier-127-losses/README.md)
are preserved, and the global best remains protected.

A standalone [native-verified course-progress audit](results/defense/diagnostics/course-progress-129/README.md)
found that the repeated learned loss occurs when the original stage-one
stream decoder has consumed about row **33–34 of 126**, not near the end of
the course. This hidden pointer is forensic-only and never enters training,
action choice, reward, curriculum or replay ranking. A subsequent
[100,000-expansion chained own-screen search](results/defense/training/frontier-search-129/README.md)
chained up to twelve unbiased held commands from the learner's own exact
states but found no score above its 2,640-point source best or stage-two
screen. No weights were updated by that search.

An [age-prioritized first-life frontier comparison](results/defense/training/frontier-age-130/README.md)
then used visible screen plus own action age **only to select training reset
states**. Its fixed 100,000 expansions explored 677,998 more emulator
actions and 7,553 screen/age cells. It prolonged a few low-scoring branches,
but never exceeded its 2,620-point source best, entered stage two or showed
a mission. At 2,600+ points, surviving branches extended at most twelve
actions beyond their own source's visible-loss age. Original source future
loss times were used only in post-hoc analysis. This negative result does
not justify simply extending the same random-hold search; the confirmed
learned score parent and verified global best replay remain unchanged.

A [fine-bottom-screen archive comparison](results/defense/training/frontier-bottom-detail-131/README.md)
tested a concrete aliasing concern: the coarse screen key can ignore one-
to-three-column visible ship shifts near the repeated gap. It kept run 129's
same source states, score-biased selection, random holds, capacity, seed and
100,000-expansion gate, changing only the reset-state fingerprint to include
exact bottom-three-row video bytes. It retained **15,640** distinct cells
and explored **847,199** new emulator actions, but still never exceeded its
2,640-point source best, entered stage two or showed a mission. At 2,600+
life points, surviving branches extended at most twelve actions beyond
their own source's visible-loss age. This bounds the simple archive-aliasing
explanation; it does not establish that the screen lacks enough information
for a learned policy or that a different exploration method would fail.

A [stronger learned-continuation PPO pilot](results/defense/training/ppo-continue-strong-cal-132/README.md)
tested the specific observation that earlier trained policies rarely chose
their `CONTINUE_PREVIOUS` option near the repeated loss. A neutral +9 extra-
action bias on the unchanged strong run-119 parent produced a no-learning
initializer with **10,454.38** mean on 32 fresh complete games and actual
continuation counts **18 / 10 / 9 / 7** in its four pre-flash approach
windows. The full 1,048,576-action score-only PPO continuation selected
its seventh checkpoint (fixed mean 10,456), but on 128 new matched complete
games averaged **10,319.69**, versus **10,476.56** for the confirmed run-121
score parent and **10,472.81** for the +7 predecessor. Every game remained
stage one. Its verified selected replay chose continuation 766 times in
2,586 actions overall, yet only **4 / 3 / 1 / 5** times in the four pre-
flash approach windows, again losing at 2,620 points per life. Full model,
optimizer, RNG, training logs, paired results and verified local replays
are preserved. This stronger bias is not a successor and cannot replace
the globally protected best replay.

A [same-screen frozen-logit comparison](results/defense/diagnostics/continue-suppression-132/README.md)
then isolated that action-use change from differing replay trajectories.
On the same four verified pre-flash 64-action windows, the untrained +9
initializer assigned an average **11.65** expected continuation choices
per window; the selected trained +9 model assigned only **0.89**. The
trained +7 model assigned **1.31**. This is direct evidence of learned
continuation suppression on those visible inputs, not a claim about its
cause or about whether any particular held direction would pass the gap.
Repeating neutral-bias calibration alone is not the next justified step;
a genuinely different learned temporal-control and credit-assignment
mechanism must be tested against the same complete-game stage gate.
