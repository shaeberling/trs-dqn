# Trial 169 course-depth audit, not a matched gameplay comparison

The fresh recurrent treatment completed its predeclared 8,388,608 own
actions; the zero-memory control is still training. This read-only audit
asks whether the treatment's selected terminal 600-point best-effort
game advanced farther through the immutable first-stage obstacle stream
than its much weaker initial replay. It also records the control's first
420-point replay as an early reference, **not** as a matched final model.

Both [treatment terminal](treatment-terminal-replay/replay.html) and
[control first](control-first-replay/replay.html) are independently
neural-action-verified original-boot complete games. The existing
`rl.defense_course_progress_probe` reexecuted every recorded action,
screen and score reward before reading the original game's private
stream pointer **only for post-hoc forensics** at visible ship losses.
The pointer never enters a policy, reward, training reset, action choice,
or model selection.

| Frozen replay | Displayed score | Decoded stream rows at four visible losses |
|---|---:|---|
| Treatment, 8,388,608 actions | 600 | 21 / 21 / 21 / 21 |
| Control, 1,048,576 actions | 420 | 15 / 16 / 15 / 16 |

The treatment's separately archived first-check replay had reached rows
13 / 13 / 16 / 16. Its terminal selected replay therefore shows deeper
first-stage progress, but still far short of the protected 10,480-point
learner's recurring row-33/34 losses and the original stage's 126 rows.
No game here entered stage two or completed the mission. The treatment
and control snapshots differ in training age and replay seed; this table
**does not establish a causal memory advantage**. The planned two-arm
fresh comparison must finish before that claim.

The compact [report](report.json), exact [probe source](source.py), and
both complete replay bundles were copied byte-for-byte from their active
sources. Reexecuting the *archived copies* from original boot reproduced
the same four loss rows, model hashes and trace hashes for each replay.
The archive's probe source hash matches the report. These private rows
measure decoded stream progress at visible loss bookkeeping, not exact
collision times or rows safely navigated by a ship. No searched route or
private diagnostic value became learner data or a published best effort.
