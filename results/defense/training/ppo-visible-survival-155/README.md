# Visible-gameplay survival reward at the repeated loss

The [coarse screen-novelty trial](../ppo-life-novelty-154/README.md) marked
124,560 of 262,144 actions as first visits yet did not improve fresh
score or reach stage two. Earlier [age-prioritized frontier search](../frontier-age-130/README.md)
selected longer-lived saved states but made no policy updates; its longer
survivors often had very low displayed scores. This test asks whether a
learned PPO policy can use a smaller *per-step survival* signal alongside
the original displayed-score reward. The signal is **0.05 optimizer-reward
units** only after a frame with a legible original gameplay HUD, a positive
visible ship count, and no visible life loss or episode end. It never pays
during a blank intro, on loss, or on game over. It reads no hidden state and
does not specify a direction, wall, opening, route, stage target or expert
action. This is explicitly **reward shaping**, not score-only training.
From-boot evaluation and the protected best replay remain original
score/stage-only.

Treatment resumes the exact full model, optimizer and policy/noise RNG of
the [ordinary score parent](../ppo-persistent-noise-long-119/run/step-000009718528/)
at action **9,718,528**. It inherits 16 native workers (four boot-only),
score scale 0.01, 256-step rollouts, 512-action minibatches, four epochs,
0.00025 Adam, 0.002 entropy, gamma 0.997, lambda 0.95, 100,000-T-state
actions, per-life symmetric logit-bias noise and own-visible-life-loss
training resets. The exact score-only matched control from
[run 154](../ppo-life-novelty-154/control-full/), itself a continuation of
the same parent and seeds with **no auxiliary reward**, is frozen for
comparison. The original parent is also frozen. Source code changes do not
alter the old control's action/reward path; a short native smoke checks the
new treatment integration.

Require focused tests and the whole repository suite. Run a **16,384-action
smoke** with native complete-game validation. If finite and its ten-game
fixed mean is at least 5,000, restart from the common parent for **262,144
new actions**, checking ten fixed complete games every 65,536 actions.
Stop after two consecutive means below 5,000. Select the earliest checkpoint
with the highest stage/mission, then fixed mean. Any stage-two/mission
training or validation event requires independent original-boot replay
verification. If all stay stage one, compare frozen treatment, prior
score-only control and original parent on 64 untouched matched complete
games, seeds **610500–610563**. A score-parent designation requires the
treatment to beat both on fresh mean and no worse sub-9,000 tail, followed
by a second untouched 64-game confirmation, seeds **610700–610763**.
There is no automatic global-best promotion for a tied 10,480 score.

If fresh stage progression is still absent, extend the treatment toward
1,048,576 new actions **only** if its first fresh-set mean score remains
at least 10,000, its sub-9,000 count is no more than control + 1, and its
mean complete-game action count exceeds the control by at least 20. This
is a predeclared exploratory survival gate, not evidence of passage; it
prevents spending longer compute on a learner that merely collapses score
or does not survive observably longer. Otherwise archive the negative
result and retain all full checkpoints, evaluation tables and verified
replays without promotion. The 5 GiB disk guard remains in force.

## Integration gate

The full **519-test** repository suite passed with the new optional reward.
The 16,384-action native smoke completed with finite PPO updates and a full
model/optimizer/RNG checkpoint. It paid the bonus on **8,373** visibly live
HUD frames (**418.65** optimizer-reward units), demonstrating that the new
signal is active but excludes many non-gameplay/loss frames. Its ten fixed
complete games averaged **10,232**, best **10,480**, all stage one—above
the 5,000-point integration floor. Its separate local best-effort replay
reexecuted **2,546** policy actions from original boot. The production arm
restarts from the unchanged frozen parent, not these smoke weights.

## Full treatment and disposition

The production treatment completed all **262,144** new actions at counter
**9,980,672** with 36 complete boot games and 887 own-reset segments.
No training episode entered stage two or completed the mission. Its final
progress record shows **149,393** bonus-paying visible-HUD actions and
**7,469.65** auxiliary optimizer-reward units, confirming that survival
shaping was used substantially. It did not automatically make the learned
policy live longer on complete games.

The four fixed ten-game means at +65,536 / +131,072 / +196,608 /
+262,144 actions were **10,461 / 9,867 / 10,400 / 10,356**. All 40
fixed complete games remained stage one. Under the frozen selection rule,
the earliest [9,784,064-action checkpoint](full/step-000009784064/) is
selected, not the final weights.

On **64 untouched matched complete original-boot games** (seeds
**610500–610563**), the selected treatment, old score-only control and
unchanged parent returned:

| Frozen policy | Mean / median / best score | Below 9,000 | Mean complete-game actions | Stage-two games |
| --- | --- | ---: | ---: | ---: |
| [Visible-survival treatment](fresh-treatment-64.json) | **10,453.75 / 10,480 / 10,480** | 0 | **2,527.75** | 0 |
| [Score-only control](fresh-control-64.json) | **10,470.94 / 10,480 / 10,480** | 0 | **2,540.41** | 0 |
| [Unchanged parent](fresh-parent-64.json) | **10,447.97 / 10,460 / 10,480** | 0 | **2,524.78** | 0 |

Treatment beat the control in only 7 paired scores, lost 23 and tied 34.
It beats the parent slightly in mean, but loses to the matched control and
has **12.66 fewer** complete-game actions than that control on these same
seeds. It therefore fails both the score-parent and predeclared survival
extension gates. All **192** fresh games are stage one, and the
[treatment](fresh-treatment-replay/replay.html),
[control](fresh-control-replay/replay.html) and
[parent](fresh-parent-replay/replay.html) best-effort replays were each
independently reexecuted from original boot with `verified: true`.
No checkpoint is promoted; the protected global best replay remains
unchanged.

The complete smoke/full training directories and their separately verified
local replay artifact directories were **moved**, not duplicated, from
`runs/defense-ppo-visible-survival-155/` into this permanent archive to
keep free space above 5 GiB. Full model, optimizer, RNG, validation and
metric records are retained. The change does not justify extending this
same per-step survival bonus without a new reason or method.
