# Longer early-screen score search

This run tests whether the recurring stage-one failure can be changed
earlier in the approach. Its proposal basis comes only from the agent's
own verified rendered-screen histories 370 / 350 / 300 / 256 decisions
before visible life loss. The frozen encoder/value network, original
environment, twenty native commands and screen-only acting input are
unchanged. Twenty symmetric key-weight directions are screened per
generation. New complete games from boot provide 4 / 16 / 32 fresh
training seeds for screening, comparison and confirmation; a candidate
must beat the incumbent by at least 150 displayed points in both latter
phases to be kept. No local death reward, hand-coded steering, hidden
state, demonstration or evaluation-seed selection enters the search.

The initialization is the independently confirmed pilot-80 checkpoint.
The pilot-90 short gain failed its separate confirmation gate, so this
long run does not inherit that candidate. Fixed seeds 10000–10009 are
used only for periodic whole-game validation and replay publication.
The sole collector now watches this run's local verified artifacts as
well as all 91 previous sources; it independently replays any purported
global improvement before promoting the shared best.

`collector-config.json` records the full collector source list. The live
run is `runs/defense-ars-early-91/`; model/RNG checkpoints, exact search
source, population plans, full-game scores and verified replays will be
copied here at milestones and completion. Stage passage and native
mission completion, not near-ceiling score alone, remain the goal.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-subspace-pilot-80/generation-000001 \
  --context-archive results/defense/training/ars-early-source-89 \
  --output runs/defense-ars-early-91 --generations 11 \
  --direction-mode early-subspace-key --subspace-components 12 \
  --directions 20 --sigma .5 --sigma-max 2.5 --shortlist 4 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 310000 --seed 251 --eval-every 5
```
