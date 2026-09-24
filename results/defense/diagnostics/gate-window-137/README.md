# The apparent turn is not solved by holding one key

This is a **diagnostic-only** extension of the earlier
[visible gate-timing probe](../gate-timing-127/README.md), not training.
It replays one exact, native-verified first-life prefix of the learned
[run-127 game](../../training/ppo-screen-frontier-127/fresh-selected-replay/replay.html)
to each of actions **323–340**, just before that life's visible loss at
action **407**. At every anchor, it verifies the original suffix's score
and screen bytes, then separately holds each of the original **20 physical
commands** for up to 160 base actions or the next visible boundary.
The resulting **360** counterfactuals, source hashes and each anchor's
original-suffix check are preserved as `lead-*.json` here.

Every original learned suffix reaches **2,620** points before losing a
life. None of the 360 constant-key branches scores above 2,620, reaches
stage two, or remains alive for all 160 actions. In particular, holding
RIGHT from anchors **323–338** loses much earlier, at displayed score
**1,080–1,100**; starting at **339 or 340** avoids that early loss but
still loses at **2,620** after **51 or 50** held actions. The earlier
probe separately found that RIGHT from action 322 loses early, whereas
starting at 341 or later is too late to reach the visible far-right
opening. Across this one verified life, merely adding a longer constant
RIGHT hold in the previously untested interval does not solve the shared
failure. A phase-changing sequence may be necessary, but these branches
do **not** establish a viable route, exact collision time, or that the
game is passable from any sampled anchor.

These actions are interventions for diagnosis only. They are not fed to
any learner, experience buffer, reward function, checkpoint selector,
evaluation policy or promoted replay. The probe uses visible score and
life/stage outcomes, not the hidden course pointer. No model was
updated. Reproduce each file with `rl.defense_barrier_probe` on the linked
verified bundle, `--lead 67` through `--lead 84`, and `--max-actions 160`.
