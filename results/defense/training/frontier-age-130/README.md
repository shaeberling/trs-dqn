# Visible-age frontier search (bounded comparison)

The 100,000-expansion screen/score frontier search in run 129 chained up to 12
held commands, but never exceeded the source's 2,640 displayed life points or
reached stage two. Accepted chains survived at most 24 actions beyond their
own source's visible-loss step. Earlier age-binned **PPO training** (run 23)
also failed to improve complete-game progression. This is a different,
training-only test of state *selection*: can a screen-and-visible-age frontier
retain longer surviving intermediate states than run 129's score-biased
archive, without altering the game reward or model?

An initial 100-expansion smoke with all 48 states exposed a confound:
later lives have about 300 extra visible between-life transition actions,
so an unadjusted age selector disproportionately favors them. The final
predeclared comparison therefore uses the **twelve first-life snapshots**,
one from each of the same verified own games. First-life age begins at the
same boot boundary. It uses the same original emulator,
same ten direction-symmetric physical commands, same 4/8/16/32-action holds,
same 4,096-cell capacity and 100,000-expansion bound. A cell is the existing
HUD-excluded coarse **visible screen** plus a 16-own-action age bin. Age starts
at zero at each observed visible life loss in the source game; later branches
continue that count. A same-cell replacement keeps the later surviving state.
Sampling is 50% from the top age decile, 25% from the 75th–90th percentile,
and 25% uniform. There is no preferred direction, route, collision oracle,
extra reward, model update, or demonstrated action. The source games' *future*
loss times are not used in selection; they are available only for post-hoc
comparison. Opaque snapshots only reset the emulator. This search cannot
produce a promoted neural replay.

Predeclared gate: stop at 100,000 expansions or a natively re-executed
stage-two screen. Report exploration actions, distinct screen/age cells,
source and exploratory maximum visible life age, the maximum *surviving* age,
score above 2,640 if any, and stage/mission discoveries. After completion,
compare each branch's age to its own source's visible loss age **only as a
diagnostic**, not a search signal. If still no score or stage gain, do not
extend this same random-hold selector unchanged. The protected best replay
and full learned checkpoints remain untouched.

The final-code 100-expansion first-life smoke exactly re-executed twelve
own sources and explored 1,078 new actions, with 157 distinct visible
screen/age cells and 30 admitted new states. Its greatest surviving age was
405 actions versus a seeded maximum of 401; its best displayed life score
was still 2,620. The full regression suite passed **482 tests** before a
subsequent config-metadata-only edit; the focused frontier tests and a fresh
final-code smoke passed after that edit.

```sh
venv/bin/python -u -m rl.defense_frontier_search \
  --source-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-frontier-age-130 \
  --priority age --age-cell-interval 16 --source-life 1 \
  --expansions 100000 --capacity 4096 --source-stride 8 --seed 509
```
