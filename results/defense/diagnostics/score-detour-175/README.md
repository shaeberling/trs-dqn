# Displayed-score cost of the verified lower-score detour

This is a read-only return audit, **not training**. Compare the protected
10,480-point learned replay's first life to the separately verified forensic
route that keeps the first ship alive to original stream row 50. Both share
the exact first **280** actions. The route was selected using private RAM in
an isolated emulator search and is quarantined: its actions, private bytes
and snapshots cannot become demonstrations, rewards, resets, policy inputs,
training data or a promoted neural replay.

Reexecute the archived route from original boot and read only the displayed
score increments returned by `DefenseEnv`. Hash-bind the protected replay and
forensic action archive, and require its shared pre-fork actions, screens and
score rewards to match. Stop the protected comparison at its first **visible**
ship-loss boundary, not at an inferred private collision time. For each
discount γ in **1.0 / 0.999 / 0.997**, compute the known first-life score
return from action 280 forward. Then solve for a *single hypothetical score
increment at the first action after the forensic trace* that would tie the
protected known return at that fork. No such future score is assumed to occur.

This arithmetic is not PPO's GAE estimator, a full-game return or a policy
gradient. It cannot establish that passage is impossible or predict stage-two
scores. It only quantifies how a delayed, lower-score detour competes under
the score reward and discount used by current training. Later hypothetical
rewards would require still larger nominal points when γ is below one.

```sh
venv/bin/python -m unittest tests.test_defense_score_detour_audit -v
venv/bin/python -u -m rl.defense_score_detour_audit \
  results/defense/learned/best \
  results/defense/diagnostics/course-feasibility-174/wide \
  --output runs/defense-score-detour-175
```

The original game, policy weights, live trial 169 and global best replay are
unchanged by this audit.

## Reexecuted result

The protected first life had **210** displayed points at the shared action-280
fork, then gained **2,410** before its visible loss at action 410. The
forensic branch shared the first 280 action/screen/reward records and gained
only **240** more displayed points through action 573, ending at 450 with
all four visible ships still present in stage one. The original-boot
reexecution and complete calculations are in the hash-bound
[`report.json`](run/report.json); the archived [source](run/source.py) is
byte-identical to the script that generated it.

| Discount γ | Protected known return | Detour known return | Extra points needed *at action 574* to tie |
|---:|---:|---:|---:|
| 1.0 | 2,410.00 | 240.00 | **2,170.00** |
| 0.999 | 2,265.27 | 221.69 | **2,739.70** |
| 0.997 | 2,003.03 | 193.52 | **4,363.96** |

The action-574 column is a counterfactual lower bound on the nominal score
needed if all missing return arrived at the *earliest next action*. It is
not an observed reward or prediction. At γ=0.997, the actual trial-169
discount, further delay increases the nominal amount required. GAE,
learned bootstrapping, action sampling and later course outcomes are not
represented by this table. The result supports keeping exploration design
separate from short-term score maximization while obeying the score-only
reward rule; it does not authorize importing the forensic route.

The two focused arithmetic tests and the complete **601-test** repository
suite passed with host Metal access after this module was added. The native
audit reexecuted all 573 original-game actions, and the archived report and
source were copied byte-for-byte from the stopped output.
