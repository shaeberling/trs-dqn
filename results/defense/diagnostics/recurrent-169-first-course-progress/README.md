# Early recurrent replay: lower score also means earlier course loss

The first fresh own-action recurrent treatment checkpoint at 1,048,576
training actions produced a fully original-boot-verified
[380-point replay](replay/replay.html). The complete replay bundle,
including frozen model weights, trace, original evaluation and action
verification, is copied here byte-for-byte from the live run. The copied
bundle was separately reexecuted from original boot and again matched
all 1,611 actions, screens and rewards and the same four loss rows.
The [forensic report](report.json) independently reexecuted all **1,611**
recorded physical actions, visible score increments and screen frames in
the original emulator, then read the original stage-one stream pointer
only at the four visible ship-loss events. The pointer decoded to course
rows **13 / 13 / 16 / 16 of 126**, versus roughly **33–34** in the
protected 10,480-point learned replay. Thus this selected early recurrent
game did not trade shooting score for deeper stage-one navigation; it
failed substantially earlier in the original obstacle stream.

This is one selected replay, not a representative fresh-game comparison
or a causal assessment of recurrence. The pointer is sampled at *visible*
loss bookkeeping, not the physical collision instant, and decoded stream
rows are not rows safely cleared by the ship. It is private original-game
RAM used only after freezing and reexecuting the replay. It never enters
the trainer, policy, reward, curriculum, checkpoint selection, or replay
promotion. The treatment and planned matched control continue under their
predeclared score/stage protocol; this diagnostic is not an early-stop gate.
