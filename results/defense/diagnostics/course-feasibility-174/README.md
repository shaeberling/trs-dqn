# Bounded forensic course-feasibility continuation

The original-boot-verified, private-RAM-selected route in
[`position-survival-172`](../position-survival-172/README.md) keeps its first
ship alive through action 428, at decoded stream row 37 of 126. It is **not a
learned policy** and remains quarantined from all training, rewards, model
selection and best-replay promotion. This separate check asks only whether a
finite search from that exact state can keep the same ship alive to decoded
row 50 (or observe an original stage-two transition) at the unchanged
100,000-T-state action cadence.

Before looking at the result: use the protected learned replay's first 280
verified actions and the archived 148-action forensic continuation. Reboot
the original game and verify that combined source. Explore twelve distinct
stage-one physical commands, one decision at a time, with a 256-state beam,
seed 174, through at most action 600. Retain score/position diversity as in
the earlier forensic search and use the original private ship count to reject
already-dead branches. The private stream pointer reports diagnostic course
depth only. If the target is reached, independently reexecute the entire
candidate twice from original boot, comparing every visible screen and score
reward, and quarantine the action bytes. If no survivor remains or frame 600
is reached, report that bounded negative result without claiming
impossibility.

No searched action, hidden byte, path or snapshot may enter a learner, its
replay archive, reward, curriculum, checkpoint selector or public learned
best. A decoded row is not itself proof of safe traversal or stage passage;
only a visible original stage transition would establish stage one cleared.
The ongoing recurrent treatment/control trial is independent and is not
changed by this diagnostic.

```sh
venv/bin/python -u -m rl.defense_course_feasibility \
  results/defense/learned/best \
  results/defense/diagnostics/position-survival-172/wide \
  --output runs/defense-course-feasibility-174 --beam 256 \
  --target-rows 50 --max-frame 600 --seed 174
```

An isolated one-layer integration smoke uses a separate fresh output and
`--beam 4 --max-frame 429`; it is never treated as the result. Preserve the
report, compact per-frame metrics, source copy and any verified discovery
before pruning stopped-run duplicates. The global learned best remains the
10,480-point screen-only policy and its verified replay.

## First bounded result and width-only check

The four-state one-layer native integration smoke completed normally at
frame 429. The predeclared 256-state run then expanded **240,156** exact
original-emulator branches. It retained a first-ship survivor through
frame **508**, with the original pointer at decoded row **44**, but all
**3,072** successors of its 256 retained states died internally at frame
509. It did not reach row 50 or stage two. Its compact layer log and report
remain at `runs/defense-course-feasibility-174` pending archival.

At frame 508, **444** generated branches were still internally alive but
the beam retained only 256. Thus the immediate zero-survivor result may be
a pruning artifact. One bounded follow-up keeps the same original source,
actions, one-decision cadence, row-50/frame-600 gate and selection rule,
changing only the width to **512** and using a fresh output directory. If
this also fails, stop this particular source-route width search and return
to the learner design rather than escalating beam size indefinitely.

```sh
venv/bin/python -u -m rl.defense_course_feasibility \
  results/defense/learned/best \
  results/defense/diagnostics/position-survival-172/wide \
  --output runs/defense-course-feasibility-174-wide --beam 512 \
  --target-rows 50 --max-frame 600 --seed 174
```

## Verified result and limit

The width-only follow-up expanded **859,069** exact original-emulator
branches and found a **450-point**, first-ship-alive path to action **573**,
at decoded original stream row **50 of 126**. The original game was still in
**stage one**. The probe rebooted the original game twice and matched every
visible screen and displayed-score reward across all 573 actions. A separate
fresh process independently replayed the saved full action record and
confirmed score 450, four private ships, stage one, row 50 and the same
private ship position 108. The complete action record is quarantined under
[`wide/diagnostic-discovery.npz`](wide/diagnostic-discovery.npz), with its
hash and verification flags in [`wide/discovery.json`](wide/discovery.json).

The [smoke](smoke/report.json), [256-state result](full/report.json) and
[512-state result](wide/report.json) were copied byte-for-byte from stopped
run directories; the archived source copies match their report hashes. The
archived action record itself passed another fresh-process original-boot
endpoint reexecution. The complete **599-test** repository suite passed with
host Metal access after the archive; the isolated focused tests and native
one-layer smoke had passed before either bounded search.
This establishes only that the chosen interface admits a first-ship survivor
well past the prior row-33/34 learned-loss region. It does **not** establish
full-stage reachability, a learned route, stage two or mission completion.
The wider beam's success also shows the 256-state frame-509 stop was not a
hard physical impossibility. No further width increase from this same
source is planned; any next search should answer a new feasibility question
or change the learning mechanism. The protected learned best is unchanged.
