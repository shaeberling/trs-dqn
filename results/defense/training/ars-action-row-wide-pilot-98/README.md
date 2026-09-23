# Two-proposals-per-command visual pilot

This native one-generation pilot extended the previous verified
screen-conditioned action-row search to **40 symmetric directions**,
two for each of the original twenty commands. The 46 high-score own
screen histories at 128 / 96 / 64 / 32 decisions before visible loss
supplied only a frozen feature subspace; no target direction, route,
collision coordinate, native state or extra reward was used. All
candidate choices were scored in complete games from boot.

The pilot played 94 complete training games and 236,549 neural actions.
Screening scores ranged from 5,440 to 10,480. The strongest separate
two-game comparison was only **+10 points** over the parent, below the
150-point gate, so no update was accepted. Fixed ten-game mean remained
10,388 and every game stayed in stage one. Many one-game screening
scores tied at the ceiling, and stable index-order tie breaking clustered
the shortlist among early direction indices. The next run changes only
that exact-score tie break: tied candidates are randomized and distinct
command rows are covered before repeats. Strictly higher displayed
scores always rank ahead of lower ones; fitness is unchanged.

The complete population plan, per-game scores, model/RNG checkpoints,
config, exact source and verified local replay are archived here. The
global 10,480-point best replay is unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-action-row-wide-pilot-98 --generations 1 \
  --direction-mode bottleneck-action-row --subspace-components 12 \
  --directions 40 --sigma .5 --sigma-max 4 --shortlist 6 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 460000 --seed 293 --eval-every 1
```
