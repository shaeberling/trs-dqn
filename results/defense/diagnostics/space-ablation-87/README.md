# Independent 64-seed check of a modest firing-key bias

The first diagnostic ablation found that suppressing Space entirely
destroyed early play, while a small -1 logit bias to Space-containing
commands showed a possible score gain. This diagnostic-only follow-up
compared the unchanged screen-only policy with exactly that -1 variant
over 64 **new** paired complete boot games per policy (training seeds
270000–270063). It did not save altered weights or promote a replay.

The unchanged parent averaged **10,081.72** displayed points; the -1
variant averaged **10,236.25**, a +154.53-point mean difference. All
128 games ended in stage 1. The effect is much smaller than the first
32-game estimate (+293.13), and only barely exceeds the 150-point score
margin used in the model searches. A final larger independent check is
required before considering this diagnostic bias as a training-selected
starting point; it is not evidence of barrier passage.

`report.json` preserves every complete game and the same-seed pairing.
`probe-source.py` matches the source hash; the original model and global
best replay are untouched.
