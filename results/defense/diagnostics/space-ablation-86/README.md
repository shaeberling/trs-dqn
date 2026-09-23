# Diagnostic firing-key ablation at the shared stage-one failure

The screen atlas suggested that commands containing Space might interrupt
movement while the right-side opening approaches. This probe tested that
hypothesis without training or saving any altered policy. It temporarily
shifted the existing neural action-head bias of every Space-containing
command by a fixed amount, then played 32 complete games from boot per
variant on the same fresh training seeds 260000–260031 and action-sampling
seeds. Only displayed score and visible stage were recorded. No hidden
state, route, collision coordinate, demonstration or reward change was
used; the ablations are diagnostic and ineligible for replay promotion.

| Space-command bias | Mean displayed score | Highest stage |
| ---: | ---: | ---: |
| 0 (unchanged parent) | 10,023.75 | 1 |
| -0.5 | 9,884.06 | 1 |
| -1 | 10,316.88 | 1 |
| -2 | 10,021.88 | 1 |
| -4 | 9,718.44 | 1 |
| -8 | 6,236.56 | 1 |
| -100 (effectively no Space) | 1,808.75 | 1 |

Suppressing Space entirely catastrophically hurts the earlier route, so
“just stop firing” is not a solution. The -1 diagnostic variant showed
a +293.13-point paired mean in this 32-game sample, but it also failed
in stage 1 and was not selected or saved. A separate 64-seed paired check
is needed to see whether that modest effect is real. `report.json`
retains all 224 complete-game outcomes, and `probe-source.py` matches its
recorded source hash. The original model and global best replay are intact.
