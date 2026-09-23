# Own-screen failure-context proposal pilot

This one-generation native pilot uses only rendered screens from 12
complete games played by the original strongest policy. A frozen visual
feature contrast compares its own screens 128 decisions before visible
life loss with screens 64–32 decisions before that marker. This contrast
proposes where a learned key-preference change should act, not which key
to press. Twenty random physical-key coefficient directions are tested
symmetrically over radii 0.5–4.0. Every candidate acts from an ordinary
boot using only screen history and is scored by displayed game points.
No native snapshot content, oracle, demonstrated action, route, or extra
reward enters the policy or fitness. This small pilot is a functional and
signal test, not a performance claim.

The completed pilot played 50 complete training games and 122,646 new
actions, all stage 1. Screening scores ranged from 300 to 10,480. The top
candidate appeared 1,110 points better over two comparison games, but
was 615 points worse on two independent confirmation games, so the
unchanged original model was retained and fixed ten-game mean stayed
9,981. The [full pilot](run/), exact [trainer source](fit-source.py) and
[visible-context source](context-source.py) are archived. The saved context
record verifies all 48 source files; it accessed only their screen arrays.
Its early visual-feature projection averaged approximately zero, while
the 64- and 32-decision projections averaged 1.048 and 0.952. This
demonstrates a distinct context-sensitive proposal, not a learned solution.
The longer run uses 4 / 16 / 32 fresh complete boot-game sets and a
150-point final margin.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-context-pilot-78 --generations 1 \
  --direction-mode failure-context-key --directions 20 \
  --sigma .5 --sigma-max 4 --screen-games 1 \
  --compare-games 2 --confirm-games 2 --shortlist 2 \
  --first-training-seed 170000 --seed 211 --eval-every 1
```
