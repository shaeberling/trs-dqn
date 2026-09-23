# Factorized physical-key search pilot

This completed one-generation native pilot tested candidate perturbations that share a
screen-dependent feature vector across every command containing the same
physical key (Up, Down, Left, Right, Space). Symmetric positive and negative
perturbations cover every key with no preferred direction. The resulting
20-command policy still acts only from rendered screens; game state and the
visible-loss atlas do not enter its inputs or reward. Complete boot-game
displayed scores are the only update fitness. This pilot uses deliberately
small samples to test the implementation, not to claim an improvement.

The 40 candidates completed boot games without a native or checkpoint error.
One passed the intentionally tiny two-game comparison and two-game
confirmation, but the fixed ten-game validation mean fell from 9,981 to
9,327; all games stayed in stage 1. The [full pilot archive](run/) retains
50 complete training games, 123,401 new actions, exact model/RNG state and
the preserved PPO optimizer. Its result is another warning against small
confirmation sets, not a generalization gain. The longer search starts from
the unchanged original strong model and uses fresh 4 / 16 / 32 boot-game
screening, comparison and confirmation sets, with a 150-point final margin.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-key-pilot-73 --generations 1 \
  --direction-mode key-factor --directions 20 --sigma .02 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --shortlist 2 --first-training-seed 120000 --seed 161 --eval-every 1
```
