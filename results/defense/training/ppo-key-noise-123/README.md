# Symmetric key-factor exploration from the confirmed score parent

The recurring right-opening barrier still ends nearly every life at 2,620
displayed points. The calibrated learned-continuation pilot used its new
choice in native play but selected it only 2/2/2/0 times in its four
pre-loss 64-action windows and did not progress. A distinct hypothesis is
that independent per-command bias noise does not produce enough *coherent*
motor trials: related physical key combinations receive unrelated draws.

Run 123 changes **training-only exploration**, not the reward, emulator,
network observation or unperturbed evaluation policy. Once per visible life,
each training worker samples symmetric zero-mean Gaussian factors for the
known keys `UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, plus independent `NOOP`
and learned `CONTINUE_PREVIOUS` factors. A command receives the normalized
sum of its own key factors as a fixed actor-logit bias for that life. Thus
`RIGHT`, `UP+RIGHT` and `DOWN+RIGHT` fluctuate coherently; left receives
an identically distributed mirror. No rightward route, obstacle detector,
hidden RAM, demonstration, direction-conditioned reward or evaluation-time
controller is provided. Every executed choice is still sampled by the
screen-only learned categorical PPO policy; the actual noisy likelihood is
retained in each optimizer batch. Original displayed-score reward and own
visible life/episode boundaries are unchanged.

The run resumes the exact full optimizer/policy RNG checkpoint from the
independently confirmed [run-121 score parent](../ppo-continue-121/README.md)
at counter **1,048,576**. It changes per-life actor-bias exploration from
independent 21-way noise standard deviation 1 to key-factor noise standard
deviation 1, inheriting all other PPO, boot-worker, rewind and evaluation
settings. The independent noise RNG state resumes; emulator episodes and
new per-life factor draws start fresh. Unlike the older ARS command-factor
search, this remains end-to-end PPO on the trained screen network.

The first gate is **1,048,576 additional base actions**, ending at counter
**2,097,152**, with eight saved full optimizer/RNG checkpoints and complete
ten-game unperturbed evaluations every 131,072 actions. Select by fixed
mean score, earliest exact tie. Native stage-two/mission evidence requires
independent replay verification. Otherwise, if the selected fixed mean
reaches **10,450**, compare the frozen checkpoint on 128 fresh matched
complete games with the run-121 score parent and the run-119 ordinary-action
parent. Below that gate, archive the full negative result without presenting
a new score parent. A tie at the existing 10,480 stage-one best never
overwrites the protected verified global replay.

A 128-action optimizer-resume integration check with this new noise completed
ten native games at **10,476** mean, all stage one; a frozen 10,480-point
replay verified **2,552** actions. The smoke is not a production milestone,
score-parent selection or collector source. The full regression suite must
pass before the production run starts.

```bash
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-key-noise-123 \
  --artifacts runs/defense-ppo-key-noise-123/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --policy-bias-noise 0 --policy-key-noise 1 \
  --steps 2097152 --eval-every 131072
```

The planned run completed **1,048,576** new actions, **112** additional
complete boot training games, **4,038** own-restored segments and eight
unperturbed ten-game checks. Fixed means were **10,466 / 10,470 / 9,574 /
10,478 / 10,468 / 10,400 / 10,476 / 10,232**. The fourth checkpoint
at counter **1,572,864** was selected. No training or validation game
reached stage two. The full [run archive](run/) contains all model/optimizer/
RNG checkpoints, validation records, metrics and local artifacts; exact
trainer, environment, key-noise, PPO, loader and snapshot sources are
preserved alongside it. The full **471-test** suite passed before launch.

On the predeclared [128 fresh matched games](comparison.json), seeds
603200–603327, selected / confirmed run-121 score parent / ordinary
run-119 parent means were **10,473.67 / 10,470.23 / 10,449.30**. The
selected model's +3.44 points versus its stronger score parent came with
**17 paired wins, 18 losses and 93 ties**—an effective score tie, not a
confirmed new parent. It exceeded the older ordinary parent by 24.38 on
these seeds, but that parent was already superseded by run 121. All **384**
fresh games remained in stage one. Its [local best replay](fresh-selected-replay/replay.html)
independently verified **2,528** learned decisions and 10,480 points but
cannot outrank the protected global best.

A [read-only visible-loss diagnostic](../../diagnostics/key-noise-123-losses/README.md)
again found four 2,620-point lives at the recurring right-opening barrier.
The selected frozen policy sampled `CONTINUE_PREVIOUS` **zero** times across
its verified game; physical movement occupied roughly 48–58% of the four
pre-loss 64-action windows. The key-factor training noise was coherent by
construction, but that did not yield verified passage or a distinct learned
persisting action behavior. Extending this identical configuration is not
supported by its complete-game results. The run-121 confirmed score parent
and older global best remain unchanged.
