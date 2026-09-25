# Fresh, visible-life timing does not support another no-fire extension

The opt-in [evaluation probe](../../../../rl/defense_visible_life_probe.py)
records only the original screen-derived life-loss events while the frozen
neural policy plays complete games from boot. It counts chosen neural
actions to the **first visible ship loss**, not hidden course rows or exact
collision time. It does not change actions, observations, reward, model
weights, checkpoint selection, or the protected replay. The focused native
test matched every visible loss frame and displayed score to an independent
original-boot replay; the full **571-test** repository suite passed.

We froze the no-learning control, the 1,048,576-action movement-only pilot,
and its predeclared 2,097,152-action extension selection. Each played the
same **64 previously unused timing-jittered original-boot games**,
seeds 621000–621063, with its saved original 100,000-T-state action cadence
and no action cap. All **192** games completed in stage one. Full game and
four visible-loss records per seed are retained in the three linked reports.

| Frozen policy | Mean displayed score | Mean / median actions to first visible loss |
| --- | ---: | ---: |
| [Untrained no-fire control](baseline-64.json) | 288.75 | 145.69 / 144 |
| [Movement pilot](pilot-64.json) | 365.94 | 182.84 / 188 |
| [Movement extension selection](extension-64.json) | 376.25 | 184.36 / 186 |

The extension outlasted the pilot on the first life in **17** matched games,
lost **42**, and tied **5**; its mean advantage was only **1.52 actions**.
Against the untrained control it outlasted **63** of 64 games, with a mean
advantage of **38.67 actions**. Thus movement-only training learned a real
early survival behavior, while the longer extension's score gain did not
reliably improve first-life survival. These matched timing measures do not
prove a course-row count: scores, ship-loss animation, scrolling and title
jitter can affect the timing proxy. The separate post-freeze
[private-pointer forensic audit](../movement-only-164-extension-course-progress/README.md)
found the selected extension replay still losing at row 16 of 126 on each
life, with one later training best-effort life at row 21. That pointer was
never used in this probe or any learner, reward, or selection rule.

Reproduce with `rl.defense_visible_life_probe` on the three linked archive
checkpoints, `--seed 621000 --games 64 --envs 8`, each with a new output path.
The JSON reports pin the exact model, source, game and environment hashes.
