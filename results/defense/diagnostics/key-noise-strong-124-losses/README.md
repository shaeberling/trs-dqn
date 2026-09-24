# Stronger key noise leaves the recurring stage-one loss intact

The [read-only loss report](report.json) checks native replay verification
and renders recorded screen bytes for the selected
[run-124 policy](policy-1-losses.png) and the
[confirmed run-121 score parent](policy-2-losses.png). Both best-effort
games earned **2,620 displayed points on each of four lives** and lost near
the familiar broad right-opening barrier. The replay seeds differ; a white
screen flash is only an alignment marker, not an exact collision label.

The run-124 frozen policy was re-executed on its selected replay seed. All
**2,500** recorded screens, physical actions and rewards matched exactly.
Instrumenting only its own sampled 21-way choices found **10**
`CONTINUE_PREVIOUS` selections in the whole game, **zero** in each of the
four 64-action pre-loss windows. Physical pure-movement choices occupied
**60.9%, 60.9%, 57.8% and 56.3%** of those windows. These action counts do
not measure displacement or establish the exact fatal action.

The diagnostic did not change model input, reward, actions, selection or
replay. No diagnostic example enters training. Its source trace and exact
verification remain in the archived evaluation bundle.
