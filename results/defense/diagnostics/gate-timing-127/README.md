# A repeated loss has two visible timing constraints

This is a **read-only diagnosis**, not training or a learned replay. The
source is the [native-verified selected replay](../../training/ppo-screen-frontier-127/fresh-selected-replay/replay.html)
from run 127 (2,563 reproduced neural decisions, four 2,620-point lives).
We used its exact recorded first-life prefix to restore three opaque emulator
states, re-executed each original suffix byte-for-byte, then held each of the
original 20 physical commands to the next visible boundary. The states are
never decoded or supplied to the policy, and the intervention sequences are
not training examples or replay promotions.

In the recorded screen at action 322, the ship's visible sprite starts near
character column **18**. An earlier horizontal wall has a narrow opening at
columns **21–30**; the next wall, already entering the screen, extends through
column **50**, leaving an opening from **51** rightward. At action 387, the
learned ship is still near column **26**, with that latter wall approaching
the bottom of the screen. These are rendered video-byte observations, not
hidden game coordinates or exact collision timestamps. The ship glyph can
temporarily disappear during animation.

| Exact own-prefix anchor | Original suffix | Held RIGHT | Held NOOP |
| --- | --- | --- | --- |
| [322](anchor-322.json) | Visible life loss after 85 actions, score 2,620 | Loss after 12, score 330 | Loss after 38, score 1,100 |
| [341](anchor-341.json) | Loss after 66, score 2,620 | Loss after 66, score 2,620 | Loss after 66, score 2,620 |
| [347](anchor-347.json) | Loss after 60, score 2,620 | Loss after 58, score 2,620 | Loss after 57, score 2,620 |

The controlled RIGHT branch from action 341 reaches visible ship column
**45** by action 388, still short of the later opening beginning at 51.
From action 347 it reaches column **43** by action 387. Starting RIGHT at
action 322 instead loses well before the usual 2,620-point event. Thus a
single direction held from either an early or late anchor does not solve
this selected life. The stronger claim that the walls caused these exact
losses would require a collision-time oracle, which this work does not use.

This suggests a **phase-sensitive, sustained-movement exploration problem**:
the policy must navigate the middle opening and then move substantially
toward the next opening. It does not prove a passable action sequence exists
from these anchors, that every life has identical geometry, or that this is
the only reason training remains in stage one. Training still receives only
four raw screen frames, displayed-score reward, and visible boundaries.
No model, reward, action profile, optimizer, or global best replay changed.

Reproduce the three exact-prefix probes with `rl.defense_barrier_probe` on
the linked replay bundle, using `--lead 85`, `--lead 66`, and `--lead 60`
respectively; the first visible life loss is at action 407. The archived
JSON records source trace and probe hashes, original-suffix verification,
all 20 physical-key outcomes, and the diagnostic-only limitations.

The independent [visible-geometry report](geometry.json) exactly replays
the own prefixes and samples the RIGHT branches at actions 388 and 387.
Its reproducible command is:

```sh
venv/bin/python -m rl.defense_gate_timing \
  results/defense/training/ppo-screen-frontier-127/fresh-selected-replay \
  --branch 341:388 --branch 347:387 --wall 322:9 --wall 322:2 \
  --output runs/defense-gate-timing-127-report.json
```
