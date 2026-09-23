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

At [generation 5](milestone-000005/state.json), 1,392 complete training
games and 3,451,610 neural actions have been played. Generation 4
accepted one head change after gains of 298.13 points in the 16-game
comparison and 225.63 in a separate 32-game confirmation. Generation
5's apparent 260-point comparison gain shrank to 110.31 in its own
32-game confirmation and was rejected. The fixed ten-game mean is now
10,388 versus 10,266 at initialization, but **all games remain in stage
one**. The full checkpoint, RNG state, evaluation, exact search source
and log prefix are preserved; the global 10,480-point verified best is
unchanged. The small in-run score gates alone do not establish a robust
gain.

That [64-seed-per-policy paired check](paired-confirmation.json) has now
finished on seeds 390000–390063. The unchanged pilot-80 parent averaged
10,218.91 displayed points and the milestone candidate averaged
10,373.28, a **154.38-point gain**. Both remained in stage one. The gain
barely exceeds the predeclared 150-point threshold, so the candidate is
eligible as a score-selected parent for a different proposal experiment;
it is not evidence of stage passage. `pair-source.py` preserves the
exact score-only comparison and verifies byte-identical frozen encoder
and value weights.

The original process later stopped during generation-nine confirmation
because a worker imported a temporarily invalid code edit. The complete
generation-eight checkpoint and the partial generation-nine plan and
game records are preserved in `run/`, including the error log; none was
deleted or promoted. The [audited continuation](../ars-early-91-resume-95/README.md)
verified exact basis/source hashes, then reproduced the generation-nine
plan, 160 screening games and 80 comparison games byte-for-byte before
finishing the planned eleven generations. The original milestone remains
the independently score-confirmed parent.

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
