# Higher-resolution visible-screen frontier (bounded comparison)

Run 129 chained unbiased held commands from the learner's own saved screens
but found no stage-two path in 100,000 expansions. Its archive key reduces
the 15 gameplay rows into coarse 5-row by 8-pixel bins. A read-only check
on the run-127 verified first-life replay found that moving the visible
ship sprite by one to three character columns near the recurring gap can
leave that key unchanged. This is *aliasing*, not total omission of the
sprite: a 19-column held-RIGHT difference did change the cell. The
[age-prioritized run 130](../frontier-age-130/README.md) found no score or
stage gain either; it selected longer-lived but often low-scoring states.

Run 131 isolates archive resolution. It retains run 129's same 48 own-life
source states, score-biased selection, 4,096-cell reservoir, ten symmetric
physical commands, 4/8/16/32-action holds, source stride eight, seed 503,
100,000 expansions and original emulator cadence. The **only intended
behavioral change** is that the archive key hashes its original coarse
visible-graphics fingerprint together with the exact raw video bytes of
the bottom three gameplay rows. This retains fine lower-screen differences
without detecting a ship, wall or gap, or preferring any action. The HUD
is still excluded. It is training-reset selection only: there is no new
policy input, scripted action, score shaping, hidden native input, model
update or demonstration. A random branch cannot be promoted as a learned
replay. The older verified best replay and confirmed score parent remain
protected.

The final-code 100-expansion smoke exactly re-executed all 48 sources,
explored 1,292 new emulator actions, reached 527 fine-bottom cells and
admitted 54 new states. It did not beat the 2,640-point source best or
enter stage two. All **483 regression tests** passed before the bounded
production search.

The predeclared gate is 100,000 expansions, or an exact native reexecution
of a newly observed stage-two screen. Report distinct cells, admitted
chains, new emulator actions, best exploratory *displayed* life score
versus the same 2,640 source best, stage/mission discoveries, and the
longest surviving branch beyond its own source visible-loss time (post-hoc
only). If neither score nor stage improves, do not extend the same random
held-key search unchanged.

```sh
venv/bin/python -u -m rl.defense_frontier_search \
  --source-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-frontier-bottom-detail-131 \
  --priority score --cell-encoding bottom-detail \
  --expansions 100000 --capacity 4096 --source-stride 8 --seed 503
```
