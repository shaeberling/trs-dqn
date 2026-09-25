# Forensic first-ship continuation toward row 75

The independently original-boot-verified route in
[`course-feasibility-174`](../course-feasibility-174/README.md) keeps its first
ship alive through action **573**, at decoded row **50 of 126**, with only
450 displayed points. This is **private-RAM-selected forensic evidence**, not
a learned policy. Its actions and snapshots are quarantined from every
learner, reward, curriculum, demonstration set, checkpoint selector and
promoted replay.

This bounded follow-up asks whether the exact original emulator and the
unchanged **100,000-T-state** action cadence admit continuation of that
same first ship to decoded row **75**, or an original visible stage-two
transition if one occurs sooner. It is an engineering feasibility check,
not permission to train on the path or infer full-stage solvability.

Before running: hash-check the protected learned trace and archived 573-
action forensic record; require the first 280 actions to match and reexecute
the complete starting route twice from original boot. From that exact state,
expand the same twelve distinct stage-one physical commands for **one
decision at a time**. Use the same 512-state score/position-diverse beam,
seed **176**, and stop at decoded row 75, a visible stage-two transition,
beam extinction or action **900**. The original private ship count rejects
already-dead branches; private stream pointer and position rank/describe
forensic states only. No private byte reaches training. If a survivor is
found, reexecute its entire path twice from original boot with every screen
and displayed-score reward matched, and save the diagnostic action bytes
only in this quarantined archive. A finite-beam negative result cannot prove
the game or action cadence impossible.

```sh
venv/bin/python -m unittest tests.test_defense_course_continuation -v
venv/bin/python -u -m rl.defense_course_continuation \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-course-continuation-176 --beam 512 \
  --target-rows 75 --max-frame 900 --seed 176
```

An isolated native one-layer smoke uses a separate fresh output with
`--beam 4 --max-frame 574`; it is not a gameplay result. The ongoing matched
trial 169 and protected learned best remain unchanged.

## Bounded negative result

The one-layer native smoke passed after two exact original-boot source
reexecutions. The 512-state production search expanded **669,324** original-
emulator branches. It kept the same first ship internally alive through
action **684**, when the original pointer had decoded row **60 of 126**;
the best selected displayed score at that frontier was only **510**. At
frame 684 the search generated **324** live successors and retained **all
324** (below its 512-state capacity). All **3,888** possible one-action
successors of those states lost the first ship at frame 685. No selected
branch reached row 75, stage two or a mission.

The initial [smoke](smoke/report.json) and [negative search](initial/report.json)
retained only the source-route fresh-boot proof; they had not saved a
replayable action record for their farthest snapshot-selected frontier.
After noticing that gap, a **witness-only recording addition** was made.
The [witness smoke](witness-smoke/report.json) and [repeated full search](full/report.json)
used identical seed, starting route, beam, action cadence, ranking and stop
gate. Both new per-layer logs are **byte-identical** to their corresponding
initial logs, so the addition did not change the search path or result.
All four run bundles were copied byte-for-byte from stopped directories;
their archived source and layer-log hashes match their reports.

The repeated search saved a [frontier witness](full/frontier-witness.json)
at action **684** (score **510**, decoded row **60**, stage one). Its complete
[684-action record](full/frontier-witness.npz) was reexecuted twice from
original boot with every screen and displayed-score reward matched; a
separate process independently replayed the **archived copy** and confirmed
the same endpoint, ship count and action hash. This is still a searched
forensic path, not a model-generated game or a stage clear.

A finite beam had pruned earlier layers, and the row-50 source was
the first found path, not an optimized full-course route. Thus the zero-
successor final layer is **not** a physical impossibility proof; another
earlier trajectory, search selector, action cadence or control history may
pass. It does show that simply replaying this quarantined row-50 path and
continuing the same 512-state score/position beam does not reach row 75.
This source-route search is stopped here. No private-guided action or state
entered a learner, and no learned best replay was promoted.

The four focused tests and four original-emulator runs passed. The last
complete repository suite passed **601 tests before this isolated module**;
no trainer, environment, reward or evaluation code changed here.
