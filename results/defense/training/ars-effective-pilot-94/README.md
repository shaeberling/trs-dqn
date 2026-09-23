# Effective-key screen-window pilot

The repeated stage-one loss happens near a right-opening barrier, and
the game's audited keyboard semantics show that pressing Space with an
arrow suppresses translation in stage one. Earlier symmetric key-factor
proposals treated an arrow-plus-Space command as if it also moved the
ship. This pilot used a more faithful **symmetric** action-effect map:
pure arrow commands receive movement factors; any Space command receives
a firing factor instead. All directions were random and signed, with no
chosen steering direction, route, collision coordinate, target action,
snapshot, oracle or reward shaping. The visual proposal basis came from
the 46 verified high-score own screen histories in
`../ars-bottleneck-source-92/`; acting still uses normal rendered screens.

Starting from the independently score-confirmed early-91 milestone,
the pilot played 50 complete training games and 124,815 neural actions.
Its best candidate gained just 40 points in a two-game comparison,
below the predeclared 150-point gate, so it was rejected without a
confirmation phase. Fixed ten-game mean stayed 10,388; every training
and validation game ended in stage one. This shows the structural
proposal path runs correctly, not that it solves the barrier.

The exact source, population plan, complete per-game results, model/RNG
checkpoints, config and verified local replay are preserved. The shared
10,480-point best replay is unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-effective-pilot-94 --generations 1 \
  --direction-mode bottleneck-effective-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 400000 --seed 271 --eval-every 1
```
