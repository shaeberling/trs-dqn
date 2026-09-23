# Action-preference search at own loss states

This completed comparison used the same strong parent and 48 verified
training states as focused trials 61 and 62. Its random search changes only
the categorical action-head biases. This gives each candidate a persistent
action preference while retaining all learned screen features and their
weights. The directions are unbiased Gaussian samples over all 20 actions;
no action is preferred by the trainer. Perturbation sigma is 5.0 and the
normalized update step is 0.5, with a 20-point minimum score scale to avoid
amplifying nearly equal returns. Displayed score to the next visible life or
stage boundary is the only fitness.

Complete games from boot select verified best efforts. Training states were
never evaluation starts or policy inputs. `run/` preserves all source states,
model checkpoints, optimizer lineage, training scores and boot evaluations.
`fit-source.py` is the exact source matching the run configuration hash;
the later verified-mission stop addition was not used by this calibration.

The **1,024 focused continuations** used **124,210 new training actions**
including the 12 complete harvest games. None reached stage 2. Median score
gain from the state was only **890**, maximum **2,460**; this broad global
preference change often harmed survival. The final 10 complete boot games
averaged **9,510**, median **10,260**, best **10,430**, all stage 1. The best
intermediate mean was 10,346 at generation 5; no generation cleared the
barrier. The global 10,480-point verified replay remains unchanged.

All three focused trials so far began 128 decisions before the visible loss.
This trial does not establish that a different starting distance would help.
It does show that global action-preference mutations of this magnitude did
not find a route on the sampled own states.

```bash
venv/bin/python -u -m rl.defense_ars_focus \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-focus-63-bias-search \
  --harvest-games 12 --harvest-seed 71000 --lookback 128 \
  --directions 16 --snapshots-per-direction 4 --sigma 5 --step-size .5 \
  --bias-only --return-std-floor 20 --generations 8 --seed 81 --eval-every 1
```
