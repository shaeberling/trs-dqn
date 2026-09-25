# Forensic first-ship survival follow-up

The protected learned replay's first ship loses internally near action 389,
while its HUD reports the loss later. The previous frame-280→339 position
search found many lower-score branches far to the right, but none that were
both rightward and retained the source's first 1,080 points. This follow-up
tests whether the lower-scoring routes can **actually keep the first ship
alive through action 428**, beyond the source's loss. A survivor is an
interface-feasibility discovery, **not** a stage clear, mission, learned
policy, or training demonstration.

Predeclared search: start at the exact original-boot action-280 snapshot of
the protected, neural-verified 10,480-point replay. Expand all twelve
distinct stage-one physical commands with a 128-state beam and seed 172.
Use ten four-action layers to 320, nineteen single-action layers to 339,
twenty two-action layers to 379, then 49 single-action layers to 428. The
beam retains score, rightward position and score/position diversity. The
original private ship count rejects internally dead branches, before the
visible HUD delay. Save compact layer metrics only unless a survivor is
found. A discovered path must be independently replayed twice from original
boot, matching every source and candidate screen and displayed-score reward;
only then quarantine its action bytes in this diagnostic archive. It must
never be imported into training, curriculum, reward, checkpoint selection,
or a learned replay. No stage claim may be inferred from mere survival.

The game binary, score parser, training runs, protected best weights/replay,
and current trial 169 are untouched. Run a short native smoke and the full
regression suite before the bounded production search. A negative result is
limited to this source life, beam, action cadence, ranking and horizon; it
does not prove the game impossible.

```sh
venv/bin/python -u -m rl.defense_position_survival \
  results/defense/learned/best \
  --output runs/defense-position-survival-172-full --beam 128
```

## Bounded result and predeclared original-boot follow-up

The 592-test native suite passed. The two-layer smoke passed and the full
search completed normally after **130,728** original-emulator expansions.
The private ship count stayed at four through action **416** on selected
branches, versus the protected replay's internal first loss during action
389, but all 1,536 generated branches from frame 416 had internally lost
the ship by frame 417. The longest surviving branches at 416 scored at most
**1,120**; the 2,620-point branches had disappeared by frame 393. No branch
reached 428 or stage two. The [full report](full/report.json) and all compact
[layers](full/layers.jsonl), plus the [smoke](smoke/report.json), were copied
byte-for-byte from the stopped runs. A private-snapshot search alone is not
yet a fresh-boot verified physical route.

One bounded follow-up now reuses the **same** seed, beam, schedule and source
through frame 416, changing only the verification gate from 428 to 416. It
must find a surviving branch, replay its exact action path twice from a new
original boot with every screen and score reward matched, then quarantine
the diagnostic action bytes. It must not enter training or promotion. If
verification fails, the private-snapshot observation is not accepted. If
it succeeds, this proves only that the interface allows this particular
first ship to live 27 decisions beyond the protected internal loss, not
that stage one is passable or that a learned policy can reach that branch.

```sh
venv/bin/python -u -m rl.defense_position_survival \
  results/defense/learned/best \
  --output runs/defense-position-survival-172-verify416 --beam 128 \
  --verification-goal-frame 416
```

The follow-up passed: a 1,120-point branch was verified with its first ship
still internally alive at action **416**, still in stage one. The probe
replayed all 416 actions from original boot twice, matching every source
screen and reward before the action-280 fork and every candidate screen and
reward after it. A separate fresh-boot process independently replayed the
archived path and confirmed action count 416, score 1,120, stage one, four
visible ships and four private ships. Its quarantined [action record](verify416/diagnostic-discovery.npz)
has SHA-256 `0a7ada99655615142627799d3d09a48a2b437f52ee524e8b5768e4ede308d56a`;
the [verification report](verify416/discovery.json) records path SHA-256
`167a4617133f655606f25deb8cf401cf73e8aae77f7c72535e537457802f96c2`.
An additional exact replay of the protected source life found its private
ship counter first decremented during action **389**, not 391; its original
screens and score rewards matched the archived neural trace to that point.
Thus the verified diagnostic route survived 27 further decisions at the
same 100,000-T-state action cadence. All [follow-up files](verify416/report.json)
were copied byte-for-byte from the stopped run, and each archived source
matches the SHA-256 recorded in its configuration.

This is evidence against the narrow idea that the existing action cadence
forces the *same* first-ship loss at action 389. It does not show survival
through the next obstacle, a stage transition, a mission, a screen-only
learned behavior, or a passable full course. Neither diagnostic path nor
private selector has entered a learner or changed the protected best.

## One wider-beam pruning check

At frame 416 the beam-128 run generated 168 live successors but retained
only 128. Its zero-survivor result at 417 could therefore be an artifact
of pruning exactly there. One final width check uses the identical original
source, all twelve commands, seed, schedule and frame-428 gate with beam
**512**. A survivor at 428 still requires the same two original-boot
replays before being recorded; otherwise report the farthest internal-live
frame and stop this width-only line of investigation. This is a bounded
physical-feasibility diagnostic, not policy training or permission to
promote a searched route.

```sh
venv/bin/python -u -m rl.defense_position_survival \
  results/defense/learned/best \
  --output runs/defense-position-survival-172-wide --beam 512
```

The width check succeeded at its narrow gate. It expanded **577,189**
original-emulator branches and found a 390-point path with the first ship
internally alive at action **428**—39 decisions beyond the protected
replay's first internal loss at 389. The [discovery](wide/discovery.json)
was reexecuted twice from original boot by the probe, matching every
source/candidate screen and score reward. A separate process replayed the
archived action record and confirmed 428 actions, score 390, stage one,
four visible ships, and four private ships. The [full report](wide/report.json),
compact [layers](wide/layers.jsonl), exact source/configuration and
[quarantined action record](wide/diagnostic-discovery.npz) were copied
byte-for-byte from the stopped run. The saved source and action hashes
match their records. The path is *not* a neural policy or a training
demonstration, and it may not enter any learner or best-replay promotion.

This rules out only the claim that the current cadence forces the learned
first-life collision by frame 428. It does **not** establish stage-one
passage: the original stream has 126 rows, the path remains in stage one,
and its low score shows the score/survival tradeoff rather than a mission.
No further beam-width scaling is planned from this result alone.
