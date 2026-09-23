# Screen-conditioned per-command pilot

The previous physical-key-factor search coupled pure movement commands
with many firing combinations and did not pass the recurring stage-one
opening. This pilot gives each of the original twenty commands one
symmetric random weight proposal in the frozen visual subspace built
from 46 replay-verified own high-score failure-window screens. Each
command is covered exactly once; no rightward action, obstacle coordinate,
route, hidden game memory, demonstration or target is selected by hand.
Only complete native-game displayed score from boot chooses an update.
The acting policy still receives only its normal rendered-screen history.

The pilot played 50 complete training games and 126,362 neural actions.
Candidate scores in one-game screening ranged from 5,520 to 10,480.
The strongest two-game comparison candidate was **+70 points** over the
unchanged parent, below the predeclared 150-point gate; no confirmation
or update followed. Fixed ten-game mean remained 10,388 and all games
stayed in stage one. Three of the four screened finalists happened to
reduce a Space-containing command, but that small, selected sample is
not evidence of a causal solution. The global 10,480-point verified
replay remains unchanged.

The exact source, proposal plan, all per-game scores, model/RNG states,
config and local verified replay are archived here. The subsequent
broader run will test two independent directions per command rather
than altering the policy from this inconclusive pilot.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-action-row-pilot-97 --generations 1 \
  --direction-mode bottleneck-action-row --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 4 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 450000 --seed 283 --eval-every 1
```
