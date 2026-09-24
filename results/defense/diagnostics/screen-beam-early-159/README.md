# Earlier screen-conditioned approach 159

The [twelve-physical-command search](../screen-beam-side-fire-158/README.md)
included stage-one side fire but still lost near the same second opening.
This diagnostic asks whether the exact learned state at frame 280 has
already committed to an unrecoverable approach. It branches from the same
independently verified first life at frame **200**, leaving 120 actions of
screen-conditioned search before the critical fine-timing interval.

The run uses all twelve distinct stage-one physical commands: NOOP, eight
directions, forward fire and both side-fire combinations. It holds a chosen
command for four actions per layer through frame 320, then branches every
action to a frame-428 horizon or first visible life loss. A 256-state beam
balances displayed score and readable ship-column diversity. Visible screen
fingerprints plus the last action retain control-timing alternatives; the
original learned path remains present while alive. No native snapshot byte
is decoded or used for ranking. A score above 2,620, later stage/mission,
or a visibly live frame 428 must be reexecuted from original boot, matching
all actions, rewards and screens.

This is a bounded diagnostic of one verified own life, not neural training,
an expert route, or a promotion candidate. It cannot prove the course is
impossible. The protected learned best weights and replay remain unchanged.

```sh
venv/bin/python -m rl.defense_screen_beam \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/screen-beam-early-159/run \
  --beam 256 --anchor 200 --fine-cadence --history-key --side-fire
```

The [completed run](run/report.json) expanded **346,224** original-emulator
branches and finished at frame **407** when no retained first-life path
remained without a visible loss. The maximum displayed first-life score was
**2,620**; no later stage, mission or frame-428 survivor appeared. The
farthest retained full-score readable ship glyph at frame 388 was column
**25**, still left of the right opening. This is a negative result for this
specific score/diversity beam, not for all paths from frame 200. Starting
earlier increases branch diversity but also makes the bounded beam prune
many possible long histories. It does not justify merely widening the
same search indefinitely. No searched action entered any learner.
