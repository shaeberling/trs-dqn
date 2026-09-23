# Full-boot candidate screening pilot

One completed generation of the new complete-game action-row search checked native
replay, paired seed handling, checkpoint preservation and independent
score-gated confirmation. It starts from the original strong own screen
encoder and policy, before the short-window focused-search updates. This
small pilot is a functional test, not a claim of progression. The policy
never sees emulator state or any oracle; every scored candidate is played
from an ordinary boot with only displayed score as fitness.

All 40 symmetric action-row candidates were played in complete boot games.
Two shortlisted candidates were compared with the unchanged incumbent on
the same two fresh training seeds; one candidate then passed a two-seed
confirmation. In all, 50 complete training games and 125,915 new actions
were recorded. That acceptance was noisy: the independent fixed ten-game
validation mean fell from 9,981 to 9,538, and all games stayed in stage 1.
The pilot deliberately used tiny screening/confirmation samples to test the
native path. Its complete [run](run/) is preserved; it does not replace the
verified global best or serve as the parent for the longer search. The next
run uses four screening, sixteen comparison and sixteen separate final
confirmation games per candidate decision.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --output runs/defense-ars-boot-pilot-71 --generations 1 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --shortlist 2 --first-training-seed 105000 --seed 141 --eval-every 1
```
