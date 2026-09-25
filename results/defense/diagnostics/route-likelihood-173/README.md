# Frozen-policy likelihood of the row-37 forensic route

This post-hoc read-only audit asks why the protected 10,480-point neural
policy has not sampled anything resembling the independently verified
low-score route that keeps the first ship alive to stream row 37. It is
**not** training or an alternate replay. The route was selected using
private game RAM in a separate forensic search; it cannot become a
demonstration, reward, policy input, reset archive or promoted learned
result.

Reload the exact protected 20-action feedforward weights. Reexecute the
quarantined route from original boot and compare the model's probabilities
for its own recorded physical actions and the searched physical actions
over the same action-index window 280–388. Sum the nine original stage-one
forward-fire aliases before computing physical-command probabilities, so
their different neural IDs do not make the searched route artificially
unlikely. Also cross-score both action lists on both sets of raw four-frame
screen observations, and report searched actions 389–427 separately.
Do not update weights, export observations, or use the result for training.

```sh
venv/bin/python -m rl.defense_route_likelihood \
  results/defense/learned/best \
  results/defense/diagnostics/position-survival-172/wide \
  --output runs/defense-route-likelihood-173
```

The result is an *exact-route, frozen-model* diagnostic, not an estimate of
the chance of finding **any** surviving route. The found path was selected
after a large search, its screens differ from the learner's screens, and
the model evaluated here is the older protected score winner, not trial
169's recurrent treatment or control. Later-stage solvability is not
inferred from likelihoods.

## Reproduced result

Three focused pure tests passed. The original-boot replay reached the
forensic route's unchanged 390-point, four-ship stage-one outcome before
the frozen model was scored; no policy parameters were updated. The
[compact report](run/report.json) and exact [probe source](run/source.py)
were copied byte-for-byte from the stopped output, and the source matches
its recorded SHA-256.

For the **109 physical commands at action indices 280–388**, geometric
mean chosen-command probabilities under the protected frozen model were:

| Actions scored | Screens from learned route | Screens from forensic route |
|---|---:|---:|
| Learned route's physical commands | 0.831 | 0.0367 |
| Forensic survivor's physical commands | 0.000000266 | 0.000000545 |

The model assigned probability below **1%** to **94/109** forensic
commands even on its *own* recorded screens, and to **96/109** on the
forensic screens. It assigned below 1% to none of its own commands on its
own screens. On the later 39 forensic decisions through action 427, the
survivor's choices had geometric mean probability **0.0000158**.

This supports a specific exploration diagnosis: the protected high-score
policy heavily suppresses the action family used by this low-score,
deeper-surviving route. Screen distribution shift matters too—the
learned-route actions fall from 0.831 to 0.0367 on forensic screens—but
it does not explain the forensic actions' low likelihood on the learner's
own screens. The exact searched path was deliberately selected from many
branches; these numbers **do not estimate the probability of any stage
passage or the chance that a different route is sampled**. They cannot
justify training on the path or reporting a learned win. No full-suite
claim is made for this isolated read-only module; the last complete
repository suite passed 594 tests before it was added.
