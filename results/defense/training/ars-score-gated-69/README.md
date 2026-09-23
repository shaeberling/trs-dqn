# Continuing score-gated visual action search

This currently running trial began from the unchanged strongest learned
Defense policy. Twelve new complete training games, seeds 75000–75011,
provide 48 native-verified states 128 decisions before the policy's own
visible ship losses. Each search round changes one of 20 action rows at a
time, with symmetric positive and negative screen-dependent perturbations.
All actions are covered without a hand-selected direction or route.

Displayed score gained from the same four sampled own states nominates a
candidate only if it beats the incumbent by at least 10 points. Eight
complete games from fresh training boot seeds then compare the two frozen
policies on the same seeds; the candidate is accepted only when its mean
displayed score is no worse. These games are training data, distinct from
the ten fixed complete-game validation seeds 10000–10009. The incumbent
model remains unchanged when either score check fails. A candidate that
reaches a new stage in any complete boot training game is preserved and
replayed from saved weights for verification before best-replay promotion.

Training has no generation or wall-clock limit. Checkpoints, population
plans, native source states, scores, RNG and replay artifacts accumulate in
`runs/defense-ars-focus-69-score-gated/` with a 5 GiB disk safety guard.
An observed successful mission in a verified complete boot game stops the
loop. The shared collector checks this source alongside all earlier runs.
No improvement or stage passage is claimed until recorded and verified.

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
