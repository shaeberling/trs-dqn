# Second actor's fresh training collection

The continued [actor at 1,500 updates](../imagination-feedback-01/update-001500/state.json)
played 24 new boot games, seeds 68000–68023, without external action overrides.
Its first twenty games provide **30,130 training actions**, mean score 278;
the last four provide **6,063 held-out actions**, mean 270. All lost in stage 1.

The manifest fixes whole-game train/held-out assignments and records behavior
checkpoint hashes. No evaluation replay or demonstration is imported. The
archive audit checks byte-identical arrays and complete-game outcomes. This
collection feeds the [second world feedback update](../world-model-feedback-02/README.md),
not an imitation objective or reward bonus.
