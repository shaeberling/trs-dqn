# Forensic timing of one original first-life loss

The current reachability work is anchored on a *visible* ship loss, which may
occur well after the original program has already decided that the ship was
hit. This bounded audit measures that lag on the independently verified
balanced-fire continuation's seed-612059 first life (visible loss at action
412). It does not search for or provide a route.

The original binary audit identifies `0x7CEF` as the internal player-one
ship counter. This diagnostic alone may read it; the environment, learner,
reward, action selector, replay promoter and curriculum may not. The probe
reexecutes every original action, reward and rendered screen from boot,
without choosing an action. Starting at action 280, it checks when that
counter first changes from four to three. For a changing action it restores
the exact own pre-action state and bisects the requested T-state duration to
localize the counter change within that original action. Each counter check
uses a new restore and the same recorded key, so substepping does not alter
the canonical replay. It reports the internal event and later visible HUD
decrement separately. No hidden value or diagnostic action enters training.

If the internal decrement precedes the visible loss by many decisions, then
branches anchored just before the HUD event are too late to test avoidance.
That would guide *where to start future diagnostics*, not authorize hidden
state as a training signal, an early-death reward, or a claimed collision
oracle for the learned policy. If the two events are close, the repeated
failure needs another explanation. This is one source life, not a general
distribution across policies or seeds.

## Result

The [native report](report.json) reexecuted all **412** first-life actions,
displayed score increments and raw screens from original boot. The original
ship counter first changed from four to three during recorded action **391**,
at requested T-state threshold **34,783** of that 100,000-T-state action.
The visible HUD did not report the ship loss until action **412**: a
**21-decision** reporting lag on this life. Both focused native tests pass.

This moves the *latest plausible intervention point* earlier than a
visible-loss-centered interpretation suggests. Actions selected after the
internal decrement cannot save that ship, even though the HUD still shows
four. The result does **not** prove that the collision happened exactly at
that threshold, that every life has a 21-action lag, or that this lag alone
caused the failed learning runs. Prior interventions starting at actions
300–360 were early enough to precede this internal event; their negative
results remain valid for their tested action families. A useful next probe
would check several verified lives/policies and target visible geometry
*before* action 391, without making the private counter a training signal.

Reproduce with:

```sh
venv/bin/python -m rl.defense_internal_loss_lag \
  --output runs/defense-internal-loss-lag-166.json
```
