# Trial 169 control: fifth-checkpoint course depth

This is an interim, **learned-policy** replay audit while the matched
zero-memory control continues training. At **5,242,880** own actions, its
fifth fixed evaluation played ten complete original-boot games on seeds
10000–10009: mean **3,384**, median **3,780**, best **3,830** displayed
points; every game ended in stage one, with no mission. The trainer's
run-specific best-effort replay is the 3,830-point game on seed 10003.
Its model SHA-256 is
`61ebae59682c6669ff965a3e187e4823d914b99f1408233ec42c5e01e0a203ef`.

The replay was already independently verified by reloading the saved neural
weights: all **2,339** actions, visible screens and score rewards matched
from original boot. The standard read-only course probe then reexecuted it
and decoded the original immutable stage-one stream pointer only at visible
ship-loss bookkeeping. That pointer is **never** a policy input, reward,
training reset, action choice or checkpoint-selection value.

| Frozen learned replay | Displayed score | Decoded stream rows at four visible losses |
|---|---:|---|
| Control, first check | 420 | 15 / 16 / 15 / 16 |
| Control, fifth check | **3,830** | **30 / 29 / 30 / 29** |
| Protected global best | 10,480 | 33–34 on all four lives |

The fifth selected game therefore shows more course progress than the
control's first replay, but still no stage-two passage and no global-best
promotion. These are different selected games at different training ages;
the table is **not** a fresh-sample estimate, a matched memory comparison
or proof that displayed score generally tracks depth. The completed
treatment's terminal selected replay lost at row 21 on each life; the
control's predeclared final checkpoints and untouched paired sets are still
pending. No causal memory conclusion follows yet.

The full [self-contained replay](replay/replay.html), weights, trace,
manifest, fixed evaluation and verification are archived together. The
[original source probe](original-probe/report.json) and independent
[archived-copy recheck](archived-probe/report.json) both reproduce the four
loss rows and model/trace hashes. Decoded rows report stream progress at
visible loss, not exact collision timing or geometry safely traversed.
This archive is read-only evidence; no saved game or private pointer enters
training.
