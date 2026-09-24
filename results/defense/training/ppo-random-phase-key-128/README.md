# Random-phase coherent key exploration at the repeated gap transition

The [verified screen/timing diagnostic](../../diagnostics/gate-timing-127/README.md)
shows two successive visible gaps in one selected life: holding RIGHT too
early loses before the ordinary score plateau, while holding it later
approaches but does not reach the second opening. This does not establish a
collision cause or an action recipe. It does motivate an exploration test
whose key preferences can switch at different times within a life. The
previous [fixed 32-action redraw](../ppo-key-noise-windowed-126/README.md)
is phase-locked to the start of each own life or restored segment.

Run 128 resumes the exact full model, optimizer and policy RNG from the
independently confirmed [run-121 score parent](../ppo-continue-121/README.md)
at counter **1,048,576**. It changes training-only exploration to symmetric
physical-key-factor logit noise with standard deviation **6** and an
independent **uniform 24–64 own-action renewal period** for each worker.
Every visible life/episode boundary also redraws both factors and period.
These periods are independent of score, obstacles and screen geometry; all
key factors have zero-mean, direction-neutral draws. There is **no chosen
direction, scripted movement, action override, collision reward or
demonstration**. PPO records the actual sampled noisy action likelihood at
every step; frozen evaluation uses the ordinary unperturbed 21-choice neural
policy and starts at boot. The four raw 16×64 screen frames, displayed-score
reward, original action duration, 4 boot + 12 own-loss-reset workers and
all other run-121 settings remain unchanged. Diagnostic intervention traces
are not read by the trainer.

The bounded first gate adds **1,048,576 base actions**, ending at counter
**2,097,152**, with eight complete ten-game fixed-seed validations and full
optimizer/RNG checkpoints every 131,072 actions. Select the checkpoint with
highest fixed mean, earliest exact tie. Any observed stage-two or mission
game must be independently replay-verified. Otherwise, only a selected
fixed mean of at least **10,450** triggers an independent 128 matched-game
comparison against run 121 on seeds **604200–604327**. Below that gate,
archive the negative trial without claiming score improvement. A 10,480
stage-one tie never replaces the protected global best replay. Record actual
noise redraws, boot games and own-state restorations; no planned count is
treated as evidence of coverage or passage.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-ppo-random-phase-key-128 \
  --artifacts runs/defense-ppo-random-phase-key-128/artifacts \
  --resume results/defense/training/ppo-continue-121/run/step-000001048576 \
  --policy-bias-noise 0 --policy-key-noise 6 \
  --policy-key-noise-min-interval 24 \
  --policy-key-noise-max-interval 64 \
  --steps 2097152 --eval-every 131072
```

Before production, the full **476-test** regression suite and all **16**
focused noise tests passed. A separate [one-update native smoke](smoke/)
resumed the same parent for 4,096 actions, completed two frozen games at
10,470 mean (both stage one), and independently re-executed a 2,572-action
10,480-point local replay. Its full model/optimizer/RNG and source-linked
records are retained but excluded from the shared best collector. It is
only an implementation check, not evidence of new game progress.

The production trial finished its planned **1,048,576 new actions** in about
730 seconds. It completed **130** new boot games and **4,779** restored
segments. The last periodic progress record logged **26,337** key-factor
draws, confirming the renewal schedule was exercised. Its eight fixed
ten-game means were **9,799 / 10,015 / 10,464 / 10,325 / 10,462 /
10,460 / 10,359 / 10,476**. The final full optimizer/RNG checkpoint at
counter **2,097,152** won the predeclared highest-mean selection. No
training or validation game reached stage two. The [complete run archive](run/)
retains every saved checkpoint, optimizer/RNG state, evaluation, metric and
local replay; the archived tree was content-checked against the live run.
Exact trainer, noise, PPO, model, environment, curriculum, snapshot,
continuation, evaluation and worker sources are saved beside it, and eight
recorded source hashes match.

The predeclared [128 fresh matched games](comparison.json) on seeds
604200–604327 gave selected / unchanged run-121 parent means **10,473.67 /
10,476.88**, a **−3.20**-point difference. The selected policy won **18**
paired seeds, lost **4**, and tied **106**; it earned the exact 10,480-point
stage-one ceiling **124/128** times versus the parent's **110/128**.
However, one selected **9,730**-point game against a 10,480-point parent
game offset those small wins. Neither arm had a score below 9,000, and all
**256** games stayed in stage one. The high ceiling count is genuine
score-consistency behavior, not passage or a confirmed improvement in mean
score. The original run-121 parent remains the confirmed parent.

The selected [local best replay](fresh-selected-replay/replay.html) and
[paired parent replay](fresh-parent-replay/replay.html) independently
reproduced **2,569 / 2,524** learned actions. Both earned four 2,620-point
lives near the same visible right-opening barrier; see the [read-only loss
sheets](../../diagnostics/random-phase-128-losses/README.md). Neither
replay is a win, and the protected global best remains unchanged. This
bounded trial does not support repeating the same random-phase setting
longer without a new source of learning signal or exploration coverage.
