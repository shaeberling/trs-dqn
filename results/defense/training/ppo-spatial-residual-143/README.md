# Learned spatial mixing at the repeated two-opening barrier

The best screen-only score policies and three bounded diagnostic searches
all stop near original stage-one course row 33–34. Open-loop mutations
from both early and late own-policy states never exceeded 2,620
first-life displayed points; retaining 2,310 distinct pre-gap screen
cells did not solve it. The current shared CNN flattens its 6×16 spatial
feature map directly into a dense layer. A screen-conditioned spatial
mixing residual may share useful wall/ship relationships across locations
that the flattened head learned only at earlier openings. This is a
learned architectural hypothesis, **not** a hand-coded gap detector,
route, action override, hidden-state input or new reward.

Add two trainable 5×5 convolutions over the existing final spatial
feature map, with a learned scalar residual gate initialized to zero.
Copy the full frozen [option-credit score parent's](../ppo-duration-credit-136/README.md)
CNN, value and joint key-duration actor weights into the unchanged base.
At initialization, spatial and original networks must produce the same
logits and values on real own screens, within numerical precision.
Both arms start from these same frozen weights with **fresh Adam and
policy RNG** (the source optimizer cannot cover new parameters): one
uses the spatial residual, the other is an otherwise matched standard
key-duration PPO control. No recorded actions or counterfactual search
plans are loaded as training examples. The policy continues to see only
four rendered 16×64 video-memory frames; learning uses only actual
displayed-score differences and visible life/stage boundaries.

Keep the source policy's 16 workers, four boot-only workers, 128-cell
own-screen loss archive, 128-action own lookback, joint duration choices
1/4/16/64, semi-Markov option-start actor credit, key/duration
training-noise standard deviations 2/4, learning rate 2.5e-5 and all
other PPO settings. Test exact initial output parity and a 16,384-action
optimizer/replay smoke for both arms. Then run a **matched 524,288-action
first gate** (four ten-game fixed checks on seeds 10000–10009, every
131,072 actions), preserving complete model/optimizer/RNG checkpoints
and independently verified local replays. Select each arm's earliest
highest stage/mean fixed checkpoint. Any stage-two or mission outcome
requires independent fresh-boot learned-policy verification. If both
remain stage one, only a spatial selection with fixed mean at least
10,450 proceeds to **64 untouched matched complete games**, seeds
607900–607963, against its fresh-optimizer control and the frozen
source policy. A score-only gain must also survive a separate fresh
confirmation before becoming a score-training parent. Neither a stage-one
score tie nor a search plan replaces the protected global best replay.
Stop an arm early if two consecutive fixed means fall below 5,000 and
archive that collapse rather than silently modifying the plan.

The two arms differ only by the zero-initialized, trainable spatial
residual; both deliberately reset Adam/RNG on identical transferred own
weights. The gate is designed to test whether this new learned inductive
bias enables later-stage play or merely perturbs the same score plateau.

The three focused tests pass: the two initial networks give **bit-identical**
logits and values on five real native screens, the scalar residual gate is a
trainable parameter and moves under a finite PPO update, and incompatible
own-checkpoint protocols are rejected. The planned 16,384-action native
smokes have also finished with finite losses and independently verified
replays. The spatial smoke's two fixed games scored 10,480/10,480 (mean
10,480); the otherwise matched standard-network control scored
10,460/10,480 (mean 10,470). All four games stayed in stage one. The
spatial gate moved from exactly zero to -0.0008952. These tiny reused-seed
smokes check plumbing only, **not** barrier passage or model superiority.
The full **501-test** regression suite passed. A separate 4,096-action
continuation from the spatial smoke's exact model, optimizer and RNG
finished normally at action 20,480; its checkpoint is also retained.

The smoke checkpoints, optimizer and RNG states, logs, and replays are under
`smoke-spatial/`, `smoke-control/` and `smoke-resume/` here (and under their
original `runs/` directories). The production comparison restarts from the
same frozen parent, not from either smoke checkpoint.

## Completed matched gate

Both planned 524,288-action runs finished normally. The four fixed
ten-game means on seeds 10000–10009 were:

| Actions | Spatial residual | Unchanged-network control |
| ---: | ---: | ---: |
| 131,072 | 10,445 | 10,445 |
| 262,144 | 10,198 | 10,428 |
| 393,216 | 10,440 | **10,462** |
| 524,288 | **10,456** | 10,221 |

All **80 fixed games** stayed in stage one. The earliest highest fixed
stage/mean checkpoints selected by the predeclared rule are the spatial
[524,288-action checkpoint](run-spatial/step-000000524288/) and control
[393,216-action checkpoint](run-control/step-000000393216/). Neither arm
hit the two-consecutive-check collapse criterion. All four full
model/optimizer/RNG checkpoints, training logs and native-verified local
training-best replays are retained in [run-spatial](run-spatial/) and
[run-control](run-control/). The latter training-best replays rank a
**single complete game**, not the selected fixed-game mean.

Because the spatial selection reached the predeclared 10,450 fixed-mean
threshold, all three **frozen** policies were evaluated on 64 untouched
matched complete games, seeds 607900–607963:

| Frozen policy | Mean score | Scores below 9,000 | Stage-two games |
| --- | ---: | ---: | ---: |
| [Spatial selection](fresh-spatial-64.json) | 10,224.84 | 6 | 0 |
| [Fresh-optimizer control](fresh-control-64.json) | 10,294.69 | 3 | 0 |
| [Unmodified source](fresh-source-64.json) | **10,387.50** | **2** | 0 |

Spatial versus control has 18 paired wins, 19 losses and 27 ties; versus
the unmodified source it has 13 wins, 23 losses and 28 ties. Its fresh
mean is 69.84 points below the control and 162.66 below the source,
with a worse low-score tail. All **192 fresh games** remained stage one.
This spatial-residual hypothesis therefore **fails** its stage gate and
does not qualify even as a score-training parent. No score-only tie or
early fixed-check peak replaces the protected global best replay.

Separate [fresh native-verified spatial](fresh-spatial-replay/replay.html)
and [control](fresh-control-replay/replay.html) replays reproduce every
screen, action and reward from boot. A post-hoc
[original-course forensic audit](../../diagnostics/spatial-143-course-progress/README.md)
finds visible losses at decoded original stage-one rows 34/33/34/34 and
34/34/34/33 respectively, of 126 rows. This hidden pointer was read only
after the frozen replays and never enters training, action choice, reward,
checkpoint selection or replay promotion. The negative result argues
against simply adding this particular spatial mixing residual to the same
score-only PPO protocol; it does not establish that spatial learning or
screen-only play in general cannot pass the obstacle.
