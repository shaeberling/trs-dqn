# Continuing score-gated visual action search

This completed trial began from the unchanged strongest learned
Defense policy. Twelve new complete training games, seeds 75000–75011,
provided 48 native-verified states 128 decisions before the policy's own
visible ship losses. Each search round changes one of 20 action rows at a
time, with symmetric positive and negative screen-dependent perturbations.
All actions are covered without a hand-selected direction or route.

Displayed score gained from the same four sampled own states nominated a
candidate only if it beats the incumbent by at least 10 points. Eight
complete games from fresh training boot seeds then compare the two frozen
policies on the same seeds; the candidate is accepted only when its mean
displayed score is no worse. These games are training data, distinct from
the ten fixed complete-game validation seeds 10000–10009. The incumbent
model remains unchanged when either score check fails. A candidate that
reaches a new stage in any complete boot training game is preserved and
replayed from saved weights for verification before best-replay promotion.

The run was stopped cleanly at a checkpoint boundary after 16 generations,
2,624 focused continuations and 597,443 new training actions. Generations 5
and 6 passed both score gates. The fifth-generation candidate improved its
eight-game paired training mean from 9,656.25 to 10,116.25; the sixth gained
only 48.75 points on the same kind of comparison. The ten fixed complete-game
validation games averaged 9,204 at generations 10 and 16. No focused
continuation or complete boot game reached stage 2, and the shared verified
best replay remains unchanged. The small second gate margin and lower later
validation mean motivate a stricter gate from the preserved fifth-generation
checkpoint. Full checkpoints, population plans, native source states, scores,
RNG and logs are in [run](run/). There is no wall-clock limit on the goal.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-69-score-gated \
  --harvest-games 12 --harvest-seed 75000 --lookback 128 \
  --directions 20 --snapshots-per-direction 4 --sigma .05 \
  --step-size .02 --coordinate-row --update-mode score-gated \
  --minimum-focus-gain 10 --minimum-boot-gain 0 \
  --boot-gate-games 8 --first-boot-training-seed 90000 \
  --generations 0 --seed 111 --eval-every 10
```
