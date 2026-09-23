# Complete-boot score search at the repeated stage-one bottleneck

This ongoing run starts from the original strongest screen-only policy,
whose frozen ten-game boot mean is 9,981 and whose globally verified best
replay scores 10,480. The policy's visual encoder and value branch are
frozen; symmetric changes to each of the 20 action rows are tested from
ordinary game boot. No candidate must first increase score in a short
pre-loss window. All candidate actions are neural samples from visible
screens, with no oracle, memory-state input, controller or demonstration.

Each generation plays all 40 candidates on the same four fresh training
seeds. The four highest complete-game displayed-score means are compared
with the unchanged incumbent on another 16 paired seeds. If one gains at
least 100 mean points, it is checked on a third, independent set of 16
paired training seeds before weights change. These are optimization games,
never the fixed validation seeds 10000–10009. Any game attaining a better
stage or score than the global best is separately replayed by reloading the
candidate weights and reproducing every action, reward and screen. The
collector promotes only a verified higher-ranked complete game.

There is no generation or wall-clock limit; training stops only for a
verified mission or a graceful checkpoint-boundary intervention. A 5 GiB
free-space guard prevents archive exhaustion. Full population outcomes,
weights, source hashes, RNG state and original PPO optimizer are kept in
`runs/defense-ars-boot-72/` while active.

At generation 5, the run had evaluated 1,328 complete training games and
3,335,069 new training actions. Three updates had passed the separate
16-game confirmation; the first promising comparison gained 860.625 mean
points but only 33.125 on confirmation and was correctly rejected. The
fixed ten-game boot validation at generation 5 averaged 9,975, effectively
the unchanged parent's 9,981; all were stage 1. Its independently restorable
[model, RNG and evaluation checkpoint](milestone-000005/state.json) is
preserved here (model SHA-256
`1b9da6f52aa6f7b4eb5e45ff418f6e6162280d48bc9a81a2e50de984ca87e9c0`).
This is a score-stability milestone, not passage or a new global best.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-boot-72 --generations 0 \
  --directions 20 --sigma .05 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 16 \
  --minimum-boot-gain 100 --envs 16 \
  --first-training-seed 110000 --seed 151 --eval-every 5
```
