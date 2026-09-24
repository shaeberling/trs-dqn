# Small existing-duration exploration from the fine-cadence score parent

The earlier [8% duration-mixture trial](../ppo-duration-mixture-148/README.md)
successfully exposed a newly appended 128-action hold but severely
damaged score. The verified [two-gap diagnosis](../../diagnostics/preposition-window-151/README.md)
shows that one late fixed-key correction cannot preserve both passages.
This matched trial asks a narrower learned-policy question: can a much
smaller, direction-neutral duration mixture create useful sustained
movement using only the *already trained* 1 / 4 / 16 / 64-action
options, without the damaging 128-action option?

Both arms initialize from the same [run-146 fine-cadence score
parent](../ppo-duration-fine-cadence-146/full/step-000001441792/)
with exactly the same 80 actor rows, CNN, critic and fresh Adam/RNG.
The treatment uses `--duration-explore-mix 0.02`; the control uses 0.
At each real option start, this adds a 2% uniform-duration component
while preserving the actor's physical-key marginal. PPO logs and
differentiates its *actual* mixed likelihood. Evaluation uses each
frozen unperturbed neural policy, never the mixture or a search action.
Both arms otherwise share the original screen-only 50,000-T-state,
stride-2 observation protocol, displayed-score reward, semi-Markov
option credit, 16 workers including four from-boot, life-loss own-state
curriculum and symmetric training-only key/duration factors. No
course-pointer read, wall parser, hand-coded direction, demonstration
or diagnostic action enters training.

First perform separate **16,384-action integration smokes** from the
same source. Require a finite update, complete model/optimizer/RNG
state, ten complete fixed-seed games, and independently native-verified
learned replay for each. The treatment must actually start at least
**five** 64-action options; otherwise this test has not exercised its
mechanism and will not scale. If its ten-game smoke mean is below
**5,000**, do not launch production. Smoke checkpoints cannot be
selected as a new policy.

If those gates pass, start both production arms afresh from the frozen
source for **524,288 new base actions**. Evaluate ten complete games
at each **131,072-action** checkpoint. Gracefully stop an arm after two
consecutive fixed-check means below 5,000; otherwise run all four
checks. Select the earliest highest-stage checkpoint, breaking stage
ties by fixed mean. Any stage-two or mission result requires original-
boot replay verification and fresh complete-game confirmation before
global-best promotion.

If all remain in stage one, freeze selections before testing each
arm and the unchanged source on **64 new matched complete-game seeds
608800–608863**. A score-only training parent must beat both
comparators on mean without a worse sub-9,000 tail, then pass a
separately predeclared fresh confirmation. No score tie or
counterfactual action can replace the protected best replay. Preserve
full states, metrics, evaluation records, source hashes and any
verified local replays, even for a negative outcome. The 5 GiB
disk guard remains active throughout.

## Integration gate passed

Both [treatment](treatment-smoke/) and [control](control-smoke/)
completed their separate 16,384-action smokes with finite updates,
full saved optimizer/model/RNG checkpoints, ten complete fixed-seed
games and independently original-boot-verified learned replays
(**5,093 / 5,078** neural actions). The treatment started **62**
64-action options, versus **13** in the control, exceeding the
predeclared exposure gate. Their fixed means were **10,460 / 10,470**,
all stage one; the treatment remained far above the 5,000-point
collapse threshold. These are integration checks only. The production
arms now restart independently from the frozen run-146 source, not
from either smoke checkpoint.

## Completed production and frozen selection

The treatment and matched control each completed **524,288** new
training actions normally, preserving their full model, optimizer,
policy RNG, metrics and native-verified local replays. At the last
pre-final progress event (516,096 actions), treatment had started
**1,593** real 64-action options versus **374** for control; the
mechanism remained exposed without the severe score collapse of
run 148. Their ten-complete-game fixed means were:

| Checkpoint actions | Treatment, 2% | Control, 0% |
| ---: | ---: | ---: |
| 131,072 | 10,413 | 10,460 |
| 262,144 | 10,288 | 9,978 |
| 393,216 | 10,391 | 10,470 |
| 524,288 | 10,264 | 10,214 |

All **80** fixed games stayed in stage one. The predeclared rule freezes
the treatment's [first checkpoint](treatment-full/step-000000131072/)
and control's [third checkpoint](control-full/step-000000393216/).
No collapse stop applied and no smoke checkpoint was selected.

## Untouched complete-game comparison and disposition

The frozen selections and unchanged run-146 source each played **64
complete games on the same new seeds 608800–608863**, at their
original 50,000-T-state/stride-2 timing:

| Policy | Mean / median / best | Below 9,000 | Stage-two games |
| --- | --- | ---: | ---: |
| [Treatment](fresh-treatment-64.json) | **10,105.31 / 10,460 / 10,480** | 9 | 0 |
| [Control](fresh-control-64.json) | **10,032.50 / 10,480 / 10,480** | 9 | 0 |
| [Unchanged parent](fresh-parent-64.json) | **10,412.34 / 10,480 / 10,480** | 2 | 0 |

Treatment beat control on **18** paired seeds, lost on **28** and tied
on **18**, despite its slightly higher mean. Against the parent it
won **6**, lost **33** and tied **25**. It fell about **307** mean points
below its parent and had nine rather than two sub-9,000 games. Thus it
fails the score-parent eligibility rule and there is no basis for an
extra confirmation or promotion. The [treatment fresh replay](fresh-treatment-replay/replay.html)
and [control fresh replay](fresh-control-replay/replay.html) independently
reproduce all **5,044 / 5,051** neural decisions from original boot.
Each earns 2,620 points on all four lives and remains in stage one.

This is a useful negative training result: a small mixture of *existing*
holds explores sustained movement without catastrophic fixed-score
collapse, but it neither solves the visible two-gap barrier nor beats
the untouched fine-cadence score parent on fresh complete games. All
full states and evaluations are retained. The original game and
protected global best model/replay are unchanged; diagnostic searches
did not enter either learner.
