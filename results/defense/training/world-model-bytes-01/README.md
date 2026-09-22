# Visible-byte reconstruction: matched continuation and frozen readout

**Negative gameplay result.** An auxiliary categorical screen decoder improves
exact text reconstruction, but a decoder trained on the **unchanged** parent
world obtains almost the same accuracy. Neither continued actor gets beyond
stage 1. This does not solve the recurring navigation bottleneck or establish
that missing screen information caused it.

## Comparison

Both world fits resume the complete original-objective world at 18,000 updates
from `runs/defense-world-fit-08-overshoot-control/update-018000`, continuing to
20,000. Core weights, Adam, stochastic-filter RNG and dataset-sampling RNG are
checked equal at the start. The unchanged [feedback dataset](../world-model-feedback-02/data)
contains 80 own training games / 146,625 actions, with 16 complete games held
out. No evaluation trajectories, hidden RAM or demonstrations enter training.

The experimental `--byte-weight 1 --change-objective` adds mean cross-entropy
over all 1,024 visible cells and all 256 byte classes. Its new convolutional
decoder has a separate fresh Adam; the core retains its full optimizer. The
original pixel, score, continuation and latent objectives remain. No semantic
HUD masks, collision labels or extra rewards are used. Acting still uses the
same recurrent core and learned actor, not decoded text. This is an auxiliary
screen loss, not a categorical-latent Dreamer implementation.

A third run trains only the identical newly initialized decoder for 2,000
updates on the frozen 18,000-update parent. It starts with matching decoder,
decoder Adam and both sampling RNGs. Final core weights and core Adam are
verified unchanged. This is a diagnostic readout, not a playing policy or a
new world checkpoint.

The fixed 64 held-out windows supply 1,536 observed arrival screens. These
are **reconstruction**, not forecasts; correlated cells are not independent
observations or success probabilities.

| Observed exact-byte accuracy | Frozen-core readout | Joint reconstruction |
| --- | ---: | ---: |
| All screen cells | 98.06% | 98.10% |
| Changed cells | 89.34% | 89.51% |
| Nonblank ASCII | 90.91% | 91.22% |
| Digits | 76.13% | 76.28% |
| Stars | 80.19% | 80.27% |

This qualifies the [older rounded scalar-decoder diagnostic](../../diagnostics/world-text-reconstruction-01/README.md):
low exact-byte accuracy from that decoder did **not** demonstrate that the
latent lacked the information. The new readout has a different architecture
and target, so this is not a pure comparison of rounding methods. Keeping the
previous screen still outperforms both new readouts on digits and stars in
these mostly unchanged windows; high overall accuracy alone is not sufficient.

Life-loss forecasts remain poor. Across 48 held-out windows selected to have
a visible life decrement at forecast step 16, continuation Brier error is
**0.98816 control / 0.98940 joint** (lower is better; always predicting survival
scores 1). Even observed-arrival recognition scores **0.90479 / 0.89842**.
Uniform held-out forecast Brier is **0.0013149 / 0.0013187**, versus **0.0013021**
for always surviving. These selected losses are not exact collision timestamps,
and their error is not a population calibration estimate.

## Complete-game playing test

Both actors retain all behavior weights, critic, target critic, both Adam
states and RNGs from the validation-selected 2,000-update actor in
`defense-imagination-overshoot-control-06`. Only the frozen world is refreshed.
Each then gets 1,000 additional imagined updates. Baseline evaluations already
use the respective refreshed world; they are not the unchanged parent policy.

| Actor updates | Control mean | Joint reconstruction mean |
| --- | ---: | ---: |
| 2,000, before additional actor learning | 300 | 310 |
| 2,500 | 110 | 112 |
| 3,000 | 104 | 102 |

Each entry is ten uncapped complete boot games, reusing seeds 10000–10009.
All 60 games lose in stage 1; none reaches stage 2/3 or a successful mission.
These are validation comparisons, not fresh success-rate estimates. Both
local bests score 340, with [1,628 verified control commands](actor-control/artifacts/best/replay.html)
and [1,639 verified experimental commands](actor-bytes/artifacts/best/replay.html).
The much stronger global **10,480-point best remains unchanged**, with all
manifest hashes checked. These weak model-based actors fail earlier than that
best; they should not be described as reaching its barrier.

## Preserved evidence and next question

[comparison.json](comparison.json) records copy hashes, initial full-state
equality checks, frozen-core checks, metrics and replay verification. Every
initial/intermediate/final world and actor checkpoint is retained, including
auxiliary decoder/Adam and diagnostic readout state. `control`, `bytes`,
`frozen-readout`, `actor-control` and `actor-bytes` are complete copied run
trees. [archive-source.py](archive-source.py) verifies these copies without
overwriting existing evidence. `collector-config.json` preserves all previous
61 sources and adds these two actor sources; the collector watches 63.

The full suite passed **403 tests** before the frozen-readout test was added;
all **five** new byte/readout tests were then rerun together successfully.
This is not a claim that the complete 404-test suite was run. The logs are
[regression-tests.txt](regression-tests.txt) and [byte-tests.txt](byte-tests.txt).
[default-parity.json](default-parity.json) checks four actual-data updates:
default core losses, weights, Adam and RNG remain bit-identical to the original
source saved here. Tests cover causality, gradient routing and exact resume.

The next useful distinction is whether the frozen latent supports recognizing
visible life changes with a separately trained readout, versus whether the
recurrent prior can anticipate them. The current result does not justify
simply running these poor imagined actors longer, prescribing a route, or
changing rewards. Actual stage passage and a verified complete replay remain
the measure of success.

## Reproduction

Use new output paths. Parent states and the immutable dataset are archived in
the preceding experiment directories; original local commands were:

```bash
venv/bin/python -m rl.defense_world_fit runs/defense-world-data-feedback-02 \
  --resume runs/defense-world-fit-08-overshoot-control/update-018000 \
  --output runs/reproduce-byte-control --updates 20000 --batch 8 --length 32 \
  --burn 8 --audit-windows 64 --every 1000
# Repeat with another output and --byte-weight 1 --change-objective.

venv/bin/python -m rl.defense_world_byte_probe \
  runs/defense-world-fit-08-overshoot-control/update-018000 \
  runs/defense-world-data-feedback-02 --output runs/reproduce-frozen-readout \
  --updates 2000 --every 1000

venv/bin/python -m rl.defense_imagine_fit \
  runs/reproduce-byte-control/update-020000 runs/defense-world-data-feedback-02 \
  --output runs/reproduce-byte-control-actor \
  --resume runs/defense-imagination-overshoot-control-06/update-002000 \
  --refresh-world --updates 3000 --every 500 --batch 4 --games 10 --eval-envs 4
# Repeat for the joint world with a separate actor output.
```
