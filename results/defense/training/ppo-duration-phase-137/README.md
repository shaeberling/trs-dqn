# Phase-changing exploration for learned key-duration options

Several screen-only PPO policies, including the provisional
[option-credit score parent](../ppo-duration-credit-136/README.md), still
lose four lives at original course stream row 33–34. A diagnostic-only
[exact-prefix grid](../../diagnostics/gate-window-137/README.md) held all
20 physical commands from each of 18 adjacent first-life frames; none
of the 360 constant-key branches passed that life's 2,620-point loss.
Holding RIGHT too early died earlier; starting later survived the first
hazard but still died at the repeated barrier. This suggests that a
change of movement preference within a life matters more than merely
lengthening one key hold. The counterfactual actions are **not** training
data or a supplied route.

Resume the **full selected 1,179,648-action model/optimizer/policy/noise
RNG state** from run 136. Keep its screen-only joint key-duration policy,
semi-Markov option-start actor credit, every-base-screen critic, visible
score reward, 128-cell own-screen life-loss reset archive, 16 workers
(four boot-only), learning rate 2.5e-5, key-factor noise standard
deviation 2 and per-life duration-factor noise standard deviation 4.
Change only **when the symmetric key factors redraw**: give each worker
an independent uniformly sampled **24–64 own-base-action** period,
renewed after each key draw and visible life/episode boundary. The
duration factors still redraw only at those visible boundaries. An
option already selected remains held until its learned countdown ends;
no timer or diagnostic chooses an action. Renewal periods are independent
of score, screen geometry and hidden game state. Evaluation removes all
noise and starts from original boot.

Train **1,048,576 additional base actions** to absolute counter
**2,228,224**, with eight complete ten-game fixed checks every 131,072
actions on seeds 10000–10009. Preserve every full optimizer/RNG
milestone and local native-verified replay. Select earliest highest
stage rank, then highest fixed-game mean. Any stage-two candidate
requires independent native replay and fresh confirmation. If all
games stay stage one, only a selected fixed mean at least **10,460**
triggers 64 matched complete games on untouched seeds **607600–607663**
against the frozen input option-credit checkpoint and the ordinary
score parent. No score-only result replaces the protected global
replay or counts as barrier passage. If two consecutive fixed means
fall below 5,000, archive and stop the collapsed trial early. Report
actual key and duration draw counts; no planned count is evidence by
itself.

The short 16,384-action integration smoke resumed the same full checkpoint
and stopped normally at action 1,196,032. It logged 299 joint noise draws
after 12,288 sampled base actions, a finite actor loss (-0.00612), value
loss (0.152), approximate KL (0.00281), 3,217 completed option-actor
samples and three incomplete options dropped. Its four-game fixed evaluation
averaged 10,445 points, reached only stage one, and its 10,480-point best
replay was independently reproduced for all 2,515 actions from boot. This
validates the machinery, **not** a stage passage or model improvement.
The full regression suite passed: **495 tests**.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-phase-137 \
  --artifacts runs/defense-duration-phase-137/artifacts \
  --resume results/defense/training/ppo-duration-credit-136/run/step-000001179648 \
  --policy-key-noise-min-interval 24 \
  --policy-key-noise-max-interval 64 \
  --steps 2228224 --eval-every 131072 --eval-games 10 --eval-envs 10
```
