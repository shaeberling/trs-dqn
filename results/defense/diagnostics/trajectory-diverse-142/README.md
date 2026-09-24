# Retain different visible approaches before the repeated gap

Two complete score-first searches from verified own first-life frames
[288](../trajectory-search-140/README.md) and
[160](../trajectory-early-141/README.md) tested 20,000 action-sequence
mutations each. Neither exceeded 2,620 displayed first-life points;
their best visible life extensions were two actions. The score plateau
offers almost no ranking signal among alternative approaches.

This bounded diagnostic changes **selection**, not score reward or
keyboard actions. Keep the earlier frame-160 anchor and 272-action
horizon, source replay, symmetric ten-command / five-hold mutation
profile, 20,000 candidate bound, score elite size 64 and random seed
547. At exactly **160 actions after the anchor** (source frame 320,
before the visible first middle opening), hash only the newest raw
rendered screen's coarse gameplay fingerprint plus exact bottom
three gameplay rows, excluding the HUD. Retain up to **1,024 distinct
visible screen cells**, keeping the higher *real displayed-score* plan
within a cell. Draw 50% of mutation parents from those cells, 37.5%
from the existing score elite, and 12.5% from the current highest-ranked
elite (as in the score-only control). No ship detector, gap direction,
collision coordinate, hidden
course pointer, extra reward or hand-selected command enters selection.
Every candidate is still evaluated on actual displayed score, visible
life loss and stage; the screen cell only preserves diverse parents.

Predeclare a **200-candidate smoke** and then **20,000 distinct
production candidates**. Record actual checkpoint-reaching candidates,
distinct cells and archive occupancy, maximum first-life displayed
score, survival beyond the own frame-407 loss, and stage/mission events.
Stop early only for a later-stage observation independently reexecuted
from original boot. Candidate action sequences remain diagnostic-only:
they are **not** training examples, neural replays, demonstrations, or
eligible for the protected best. The original game and screen-only
learning contract remain unchanged. A positive diagnostic would require
a separate learned-policy training and full-game verification step.

```sh
venv/bin/python -u -m rl.defense_trajectory_search \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output runs/defense-trajectory-diverse-142 \
  --anchor 160 --horizon 272 --candidates 20000 \
  --elite 64 --seed 547 \
  --diversity-offset 160 --diversity-capacity 1024
```

The [200-candidate final-code smoke](smoke/report.json) exactly reproduced
the own baseline. It reached the visible checkpoint in 92 plans, admitted
44 distinct screen cells and drew 106 mutation parents from that archive;
this verifies that the diversity selector actually ran. It found no
extra score or stage passage and is excluded from production evidence.
The [default-mode parity smoke](default-parity/report.json) generated the
same 200 action plans, candidate by candidate, as the archived run-140
smoke (both projected plan streams have SHA-256
`1f91fa84a4ac6d0edb6a10816b51a3b6f7bf270770ecac548fc1980971a2f22b`).
The new code therefore leaves the old score-only selection path intact;
the diverse arm changes its parent pool as specified above.
The full **498-test** regression suite passed before production.
