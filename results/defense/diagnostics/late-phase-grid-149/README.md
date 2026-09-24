# Fixed two-phase grid from the narrow later turn window

The [first exhaustive two-phase grid](../phase-grid-145/README.md)
started at action 322 of a verified own first life. It varied when
to depart from the learned suffix by 0/8/16 decisions, so the latest
first intervention began at action 338. The [adjacent single-key
audit](../gate-window-137/README.md) found RIGHT at 338 lost at the
earlier wall, while RIGHT begun at 339 or 340 avoided that early loss
but still died at the next opening. The one-action boundary makes a
later two-phase grid an informative test of this narrow intervention
family; it does not prove the needed action sequence or route.

Before seeing new outcomes, fix two exact verified own first-life
anchors, **339 and 340**, from the same
[run-136 learned replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html).
For each anchor, run the existing unbiased 7,500-candidate grid:
recorded own actions for delay **0/8/16**, then each pair of the ten
direction-symmetric effective stage-one commands held **4/8/16/24/32**
base decisions, then the original recorded suffix to the next visible
life/stage boundary or a 512-action horizon. Original emulator state
snapshots only speed exact replay; they are never decoded. Rank only
actual displayed score, visible stage/mission and life outcome.

Run a 20-candidate plumbing smoke for each anchor, verifying its
unaltered suffix against the source trace, then both full grids.
Stop a grid early only on a stage-two screen reproduced from original
boot, or an ordinary handled interruption. Preserve every candidate
plan/outcome, RNG-independent enumeration, source hashes and full
report. Score above the source first-life 2,620 plateau is a useful
diagnostic, **not** a learned-policy achievement. No candidate action
may enter PPO, DQN, a demonstration set, score reward, live neural
action choice or the protected replay. If neither grid improves,
only this small fixed open-loop family is ruled out, not the game or
all screen-conditioned policies.

```sh
venv/bin/python -u -m rl.defense_phase_grid \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output runs/defense-late-phase-grid-149-anchor339 \
  --anchor 339 --horizon 512
```

The two [20-candidate plumbing smokes](smoke-339/report.json) and
[adjacent smoke](smoke-340/report.json) both reproduced the exact
original learned suffix at 2,620 points and stage one. The full
[action-339 grid](anchor-339/report.json) and
[action-340 grid](anchor-340/report.json) then completed all **7,500**
candidates apiece with their full outcome tables and source hashes.
Neither reached stage two, survived its 512-action horizon or scored
above **2,620**. At anchor 339, **3,832** candidates tied 2,620 and
none survived longer than the source's 68 remaining actions. At anchor
340, **4,133** tied 2,620; 27 delayed visible loss by one action
without extra score, while all other candidates died no later than
the 67-action source suffix. These counts are diagnostic outcomes,
not trained-policy play or evidence of a viable route.

This closes the adjacent timing gap left by the 322-anchor two-phase
grid for this fixed family. It suggests that reaching farther requires
more than two short open-loop replacements or a screen-conditioned
sequence, but proves neither. No plan was fed to training, and the
protected best replay did not change.
