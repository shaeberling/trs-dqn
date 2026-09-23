# Long per-command visual search at the repeated barrier

This run tests a distinct way to change the policy at the repeated
stage-one opening. Its frozen feature basis is built from 46 of the
agent's own replay-verified rendered-screen histories, sampled 128 /
96 / 64 / 32 decisions before high-score visible life losses. The
neural policy still acts on only its normal four rendered screen frames.
No native memory, collision coordinate, route, prescribed key, example
action, intrinsic reward or life-loss clock is supplied.

Each generation covers all twenty original commands with **two signed
random visual-weight directions per command** (forty directions,
eighty candidates). Unlike the preceding key-factor experiments, a
candidate changes just one command's neural output row. Screening uses
four fresh complete games per candidate; a six-candidate shortlist is
compared with the incumbent on sixteen new games, and a nominated
winner needs at least a 150-point gain in a separate 32-game check.
Only displayed complete-game score determines ranking and acceptance.
Candidates with *exactly tied* screening scores are randomized and
distinct command rows are covered before repeats; no lower score can
outrank a higher score. The shortlist tie rule and RNG state are saved.

The starting point is the generation-five early-window checkpoint,
independently checked against its parent on 64 paired training seeds.
The later early-window update and the effective-key search did not
establish a better parent or reach stage two. Fixed seeds 10000–10009
are only for whole-game validation and replay publication, never weight
selection. The sole global collector watches this run and all 99 prior
artifact roots; it replays any proposed best before promotion.

The live output is `runs/defense-ars-action-row-99/`. Full model/RNG
checkpoints, population plans, game records, exact source and verified
replays will be archived here at milestones and completion. The goal
remains stage passage and native mission clear, not merely another
near-ceiling stage-one score.
The full repository suite passed **444 tests** immediately before this
run; `fit-source.py`, `context-source.py` and `search-source.py` match
their recorded hashes.

At generation two, after **928 complete training games and 2,339,213
actions**, one candidate passed the run's 16-game comparison and fresh
32-game confirmation (+194.38 and +207.50 displayed points). Its saved
proposal plan shows a change only to the pure `RIGHT` output row; the
search had covered every command twice and did not prescribe that row.
The exact model/RNG checkpoint and score records are saved as
`milestone-000002`. However, a separate [64-seed paired check](paired-generation-2.json)
on fresh training seeds 480000–480063 reversed the result: parent mean
**10,260.78**, candidate mean **10,242.97**, a **-17.81-point** change.
Both remained in stage one. This candidate is not a confirmed new parent
or a new global best, although the live run continues from its internal
score-gated state. The pair-check source is archived as `pair-source.py`.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-early-91/milestone-000005 \
  --context-archive results/defense/training/ars-bottleneck-source-92 \
  --output runs/defense-ars-action-row-99 --generations 11 \
  --direction-mode bottleneck-action-row --subspace-components 12 \
  --directions 40 --sigma .5 --sigma-max 4 --shortlist 6 \
  --screen-games 4 --compare-games 16 --confirm-games 32 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 470000 --seed 307 --eval-every 5
```
