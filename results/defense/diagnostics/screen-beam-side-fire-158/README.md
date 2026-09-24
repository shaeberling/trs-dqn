# Distinct stage-one side-fire coverage 158

The preceding [screen-conditioned beam](../screen-beam-157/README.md)
tested directions and forward fire, but its ten-command profile omitted
the two distinct stage-one side-fire key combinations: LEFT+RIGHT (18)
and LEFT+RIGHT+SPACE (19). The trained policy can choose these commands.
The original assembly distinguishes them from movement and forward fire;
whether they affect the two-opening failure is not established. This
predeclared diagnostic reruns the strongest prior beam with all **twelve
distinct stage-one physical commands**. Fire+arrow aliases (10–17) are
excluded because the original stage-one movement decoder treats them as
forward fire, not additional movement choices.

Use the same independently original-boot-verified run-136 first life,
seed 607200, anchor frame 280 and frame-428 horizon. The first ten layers
hold a candidate key for four actions each; from frame 320 onward, branch
every action. Beam width 256 and the visible-screen-plus-last-action key
preserve control-timing alternatives. Selection balances displayed score
and readable ship-column bins without favoring a direction or firing key.
Every candidate is executed in the original emulator; native snapshots are
opaque. A score above 2,620, a later stage/mission, or no visible life loss
at frame 428 must be independently reexecuted from original boot, matching
all actions, screens and score increments. Searched actions never train or
replace the neural policy. This bounds only one beam/source, not the game.

```sh
venv/bin/python -m rl.defense_screen_beam \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output results/defense/diagnostics/screen-beam-side-fire-158/run \
  --beam 256 --early-anchor --fine-cadence --history-key --side-fire
```

The [completed run](run/report.json) expanded **290,508** native branches.
No candidate exceeded **2,620** displayed first-life points, reached a
later stage or mission, or reached frame 428 without a visible life loss.
All retained paths lost the first visible life by frame **408**. At frame
388 the farthest retained full-score readable ship glyph was at column
**38**; the visible right opening begins beyond column 50. There were
still readable glyphs at column 38 on some frame-390 paths, but none
passed the verification gate. The source learned replay itself used
commands 18 and 19 **21** and **12** times in its first life, including
**16** and **3** times after the frame-280 anchor. The omission from the
prior ten-command search was real, but including side fire did not solve
this source life within the specified beam. Visible loss can lag physical
collision, and beam pruning prevents an impossibility claim.
