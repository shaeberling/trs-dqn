# Forensic first-gate position feasibility, not a learner

The protected verified learned replay loses its first ship at original
stage-one stream row about 33. On this exact first life, a controlled
RIGHT hold from action 339 raises a private position byte from **48**
to **94** by action 387, but still loses its ship at row 33. The next
visible opening begins near column 51; independent calibration on
readable ship glyphs suggests position byte **60** corresponds roughly
to column 28. The question here is narrow: can the original unmodified
emulator reach that earlier preposition at action **339** while also
retaining the source's **1,080 displayed points** from the first
obstacle, or does this route family trade away one of those conditions?
This is neither a stage clear nor a proof that the position threshold
itself is sufficient to survive the next wall.

The [forensic program](../../../../rl/defense_position_feasibility.py)
reexecutes every action, score reward and screen of the exact protected
source from original boot to action 339. It takes an opaque snapshot at
source action **280**, then expands all twelve distinct stage-one
physical keyboard commands for ten four-action layers to action 320
and nineteen one-action layers to 339. A deterministic **256-state**
beam reserves capacity separately for high displayed score, rightward
position and score/position diversity. It uses the original private
position byte at `0x5E03` and private ship counter at `0x7CEF` only
within this **diagnostic**: the latter drops internally dead branches
before the visually delayed HUD loss. Native snapshots are used only
to fork exact original-game states. A finite beam can miss routes;
its output does not establish impossibility.

No private RAM value, search action, path, replay screen, geometry target
or selection rule enters PPO/DQN training, a reward, a curriculum,
evaluation-policy action choice, checkpoint selection or best-replay
promotion. Any path found would be a hand-directed search discovery,
**not** a learned-policy win or demonstration. Do not export its actions
to a learner. The original score remains the only learner reward, and
the protected model/replay and live trial 169 remain untouched.

The two-layer, beam-16 native smoke reproduced the exact source prefix,
expanded **168** branches without a private ship loss, and stopped at
action 288. Three pure beam-selection tests passed. Require the full
repository regression suite and a fresh-output, exact-source full run
before drawing a conclusion. Retain only compact layer metrics and the
forensic report; no emulator snapshots or searched trajectories are
written to disk.

```sh
venv/bin/python -u -m rl.defense_position_feasibility \
  results/defense/learned/best \
  --output runs/defense-position-feasibility-171-full \
  --beam 256
```

## Completed bounded result

The full **590-test** repository suite passed before production. The
[complete search](full/report.json) finished all 29 scheduled layers
at frame 339, expanding **81,780** original-emulator branches. It used
the private ship counter to reject **1,954** expansions after a private
ship decrement or terminal boundary, without waiting for the lagging
visible HUD; this is not an exact collision timestamp. The original-boot source
prefix matched every recorded action reward and screen. The compact
[layer history](full/layers.jsonl), configuration, report and exact
probe source were copied from the stopped run and compared byte-for-byte;
the two-layer [native smoke](smoke/report.json) is also retained.

At frame 339, **1,404** generated live branches had the position byte
at or above **60**, but **none** of those also retained the source's
1,080 displayed points. The farthest position among generated branches
with at least 1,080 points was **52**, versus **48** in the unchanged
source; the full beam also retained positions up to **116** at lower
scores. Thus the search found rightward prepositioning *or* the first
large score jump, not both within this bounded family. It did not reach
stage two, establish that the first score jump is physically required
for passage, or show that a different screen-conditioned route cannot
do both. The 60-byte threshold is an empirically calibrated position
proxy, not a collision rule, and no searched path was written to disk.
The protected learned model/replay and live training process are unchanged.
