# Repeated-failure screen-window pilot

This native one-generation pilot uses the 46 score-cluster own lives from
`../ars-bottleneck-source-92/` to propose a frozen-encoder visual subspace
at 128 / 96 / 64 / 32 decisions before visible life loss. Two 140-point
early-loss outliers were excluded from proposal construction using only
displayed own per-life score; their verified screens remain in the source
archive for audit. Twelve principal variation axes plus a mean approach
contrast generated symmetric physical-key weight changes. No obstacle
coordinate, hand-coded route, target action, native snapshot or reward
shaping was used. The policy still acts from normal rendered screens.

The pilot played 50 complete training games and 123,935 neural actions.
Its best candidate was **50 points below** the parent on two fresh
comparison games, so no confirmation was triggered and no update was
accepted. Fixed ten-game mean stayed 10,266, with all training and
validation games ending in stage one. This small pilot does not rule out
the window, but it gives no evidence of passage or robust improvement.

The population plan, per-game results, complete checkpoints/RNG states,
config, metrics, exact source files and verified local replay are retained.
The global 10,480-point independently verified replay is unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-bottleneck-pilot-93 --generations 1 \
  --direction-mode bottleneck-subspace-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 370000 --seed 263 --eval-every 1
```
