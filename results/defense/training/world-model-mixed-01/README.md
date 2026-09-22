# World-model continuation with both own collections

The [union manifest](data/manifest.json) preserves **40 training games /
85,779 actions** and **eight held-out games / 17,212 actions** from the two
fresh own collections. Original episode bytes and split labels are unchanged.
Every copy is hash-checked; duplicate sources, overlapping seeds and mismatched
game/timing/parent settings are rejected. No evaluation replay was imported.

The model continues from the exact 5,000-update world-model weights,
optimizer and RNG, with an explicit `--extend-data` declaration. This is
an intentional change of training distribution, not a claim of identical
subsequent gradients. The fitter checks that every old episode remains
present with its original held-out label. The normal resume path still
rejects an undeclared dataset change.

The continuation finished 10,000 total dynamics updates, saving state and
held-out audits every 1,000. At the first new checkpoint, **6,000**, the
[24 complete held-out loss windows](review-6000/report.json) still have poor
loss anticipation: Brier **.98399** at their visible-loss endpoint, versus
1.0 for always predicting survival. These windows include both collections,
so their aggregate error is not directly comparable to the earlier 12-case
review. The [initial mixed-data audit](update-005000/audit.json) supplies the
unchanged-parent baseline for this new distribution.

The final [10,000-update review](review-10000/report.json) remains poor:
visible-loss Brier .99550 versus 1.0 for always surviving. The 64-window
uniform audit has graphics MSE .00849 at one step and .05736 at 24 steps,
versus persistence .08582 / .09135. The main agent visually reviewed the
forecast panel: broad scene prediction improves, but ship/death dynamics
remain unreliable. All intermediate 7,000–10,000 full optimizer/RNG states,
loss reviews and the completed metrics log are preserved.

A [recognition-versus-forecast diagnostic](review-10000/boundary-predictions.json)
shows mean-latent Brier .89936 even when observing the actual loss arrival
screen, .89667 for a one-step prior and .99550 at sixteen steps. Sixteen
stochastic latent draws give .90418 / .89757 / .99078 respectively. Recognition
is not a forecast; these selected loss cases do not estimate population
calibration. The head's weakness is not solely a long-horizon imagination
error or a deterministic-mean inference artifact. This motivates a matched
training-window sampling test, not a claim that the game bottleneck is solved.

This is dynamics learning, not a new playing score. No actor has yet been
trained using this mixed-data model; the frozen initial actor calibration
remains a separate experiment. The existing global replay is unchanged.

Two additional dataset tests pass: exact episode/split preservation and
coverage of both sources, plus rejection of duplicate/overlapping/incompatible
collections. They ran separately from the already-running 387-test suite.

```bash
venv/bin/python -m rl.defense_world_merge \
  results/defense/training/world-model-preflight-01/data \
  results/defense/training/world-model-collection-02 \
  --output runs/defense-world-union-reproduction
venv/bin/python -m rl.defense_world_fit runs/defense-world-union-reproduction \
  --output runs/defense-world-mixed-reproduction \
  --resume results/defense/training/world-model-preflight-01/continuation/update-005000 \
  --extend-data --updates 10000 --batch 8 --length 32 --burn 8 --every 1000
```
