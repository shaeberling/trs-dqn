# Bounded finer-action feasibility check after row 50

The protected screen-only neural replay scores 10,480 but loses its first ship
around original stage-one stream row 33. A separate **private-RAM-guided,
forensic-only** route preserves that ship through row 50 at 100,000 T states
per action. Continuing one selected route at the same cadence reached row 60,
where all one-action successors of its retained beam died. This does not prove
the interface impassable; earlier branches were pruned and the starting route
was not unique.

This check asks a narrower engineering question: starting from the verified
row-50 original-boot route, does halving **only subsequent** key holds to
50,000 T states allow first-ship survival to decoded row 65 or an earlier
visible stage-two transition? It keeps the original game, the twelve
distinct stage-one physical commands and the same 256-state score/position-
diverse beam. Stop at the first target, beam extinction, or **400 fine
actions**; seed 177 is fixed. A four-state, one-fine-action smoke runs first.
Both positive and negative results require a saved path reexecuted twice
from original boot with every visible screen and displayed-score reward
matched. A negative finite-beam result is not an impossibility proof.

This diagnostic reads private ship position/count and stream pointer to
select branches. **No source or searched action, snapshot, private value or
result may enter a learner, replay buffer, reward, policy input, checkpoint
selector, or promoted learned replay.** A decoded row is not a visible
stage clear. The live trial-169 treatment/control comparison and protected
best are independent and remain unchanged.

```sh
venv/bin/python -m unittest tests.test_defense_fine_cadence_feasibility -v
venv/bin/python -u -m rl.defense_fine_cadence_feasibility \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-fine-cadence-survival-177-smoke \
  --beam 4 --target-rows 65 --max-fine-actions 1 --seed 177
venv/bin/python -u -m rl.defense_fine_cadence_feasibility \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-fine-cadence-survival-177-full \
  --beam 256 --target-rows 65 --max-fine-actions 400 --seed 177
```

Do not infer that this finer cadence is better for a **trained** policy from
this search. Frozen-policy overrides at finer cadence previously scored
worse; retraining and strict full-20, screen-only play would be a separate
controlled experiment if feasibility evidence warrants it.

## First bounded result and width-only follow-up

The 256-state search expanded **668,892** original-emulator branches. It
kept the first ship alive through **220 fine actions** after the row-50
source, reaching decoded row **60**, but all **3,072** next actions from its
retained states lost that ship at fine action 221. The archive includes
the first-ship-alive frontier's full mixed-cadence action record, reexecuted
twice from original boot; no learner receives it. This is a negative result
for that finite beam, not evidence that finer control is futile.

The earlier 100,000-T-state continuation also stalled at row 60, and a
different source-route study previously changed from failure to success
when its beam grew from 256 to 512. Do **one** width-only follow-up with
the same source, 50,000-T-state continuation, seed, commands, row-65 gate,
and 400-fine-action cap. If it fails, stop this row-50 source-route cadence
search rather than increasing width or tuning seeds indefinitely. Neither
width can prove full-stage impossibility.

```sh
venv/bin/python -u -m rl.defense_fine_cadence_feasibility \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-fine-cadence-survival-177-wide \
  --beam 512 --target-rows 65 --max-fine-actions 400 --seed 177
```

## Wider result and one cadence comparator

The wider 50,000-T-state search reached decoded row **65** after **333 fine
actions** from the same row-50 source. The first ship was still privately
alive, displayed score was **550**, and the original game still showed
**stage one**. All **906** actions of the saved mixed-cadence record were
reexecuted twice from original boot with every visible screen and score
reward matched. The search expanded **2,013,997** emulator branches. This
is physical-feasibility evidence only; no searched action is learner data.

Because the prior 100,000-T-state continuation used seed 176, run **one**
matched-width, matched-source, seed-177 control using the existing forensic
module. It uses the same twelve controls and beam 512, targets row 65 and
caps the continuation at 200 original-cadence actions (frame 773). This
does not perfectly equate exploration budgets or isolate a causal cadence
effect, but it checks whether the row-65 result was simply available at the
old cadence under this beam/seed. A negative finite beam remains uncertain;
do not keep tuning this source-route search afterward.

```sh
venv/bin/python -u -m rl.defense_course_continuation \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-fine-cadence-survival-177-original-control \
  --beam 512 --target-rows 65 --max-frame 773 --seed 177
```

## Matched-source outcome and interpretation

The 100,000-T-state control expanded **671,580** original-emulator
branches. Its retained beam reached at most decoded row **60** at frame
684; all **6,144** next commands lost the first ship at frame 685. It
never reached row 65 or stage two. The saved final selected witness is at
row **59**, score **510**; it is not falsely labeled as a row-60 witness.
That 684-action record was reexecuted twice from original boot with every
screen and score reward matched. Both this control and the finer-cadence
row-65 discovery were copied byte-for-byte from their stopped runs and
independently reexecuted again from the **archived copies** by the focused
native tests.

The positive result is narrow but useful: this original emulator, fixed
row-50 source and command set admit a first-ship-alive trajectory to row
65 when later actions last 50,000 T states. Under the matched seed and
beam, the 100,000-T-state search did not find one and died at row 60.
Different decision counts and beam pruning prevent a universal or clean
causal conclusion; a different old-cadence route may still pass. Neither
search produced a neural policy, a visible stage transition, or a mission.
The next learner decision should await the trial-169 untouched-game
comparison. If a finer-cadence learner is tried, it must start from boot,
use only raw screens, retain all 20 original controls, train on displayed
score only and be judged first by verified stage passage rather than this
private course pointer. Do not feed these forensic action records to it.

The new module passed the complete **607-test** repository suite before
archived-copy recheck tests were added; all **five** focused tests,
including both independent original-boot rechecks, passed afterward.
The protected learned best and live recurrent trial were not changed.
