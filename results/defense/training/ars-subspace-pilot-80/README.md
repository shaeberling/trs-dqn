# Visible-approach subspace pilot

This one-generation native pilot tested a broader proposal distribution at
the recurring stage-one failure. A frozen encoder converted the original
policy's own 48 verified rendered-screen histories to features at 128, 96,
64 and 32 decisions before visible loss. A centered early-approach contrast
and four principal visual-variation axes formed a five-dimensional proposal
subspace. Twenty symmetric random physical-key changes were tested from
boot; none encoded a route, target action, score threshold, hidden state or
collision coordinate. The policy still observed only rendered screens,
and whole-game displayed score alone decided the pilot update.

The 54 complete training games used 134,736 neural actions. A candidate
improved the two-game comparison mean by 1,470 points and a separate
two-game confirmation mean by 1,560, so the pilot accepted it. Its fixed
ten-game validation mean was 10,266 versus 9,981 initially, but **all
games remained in stage 1**. Two-game gates are too small to establish
generalization; a separate larger fresh-training-seed confirmation is
required before treating this as a useful parent.

That independent [32-game-per-policy paired check](paired-confirmation.json)
used fresh training seeds 191000–191031. The original parent averaged
9,802.19 displayed points; the pilot candidate averaged 10,182.81, a
**380.63-point gain** over the same seeds. Both remained in stage 1. The
candidate passed the predeclared 150-point margin, so it is eligible as a
training-selected parent for the longer search. `pair-source.py` preserves
the exact score-only comparison code and verifies the frozen encoder/value
network is byte-identical between the two checkpoints. No validation seed
was used for this selection.

`generation-000001/` contains the full model/RNG checkpoint and ten-game
evaluation. `config.json`, `metrics.jsonl`, `population-000001/` and
`artifacts/` contain the exact search plan, complete-game outcomes and
verified replay artifacts. `fit-source.py` and `context-source.py` match
the recorded source hashes. The proposal never read native snapshots;
the best verified 10,480-point global replay remains unchanged.

```bash
venv/bin/python -u -m rl.defense_ars_boot_search \
  --initialize results/defense/training/ars-59-head-search/run/generation-000000 \
  --context-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-ars-subspace-pilot-80 --generations 1 \
  --direction-mode failure-subspace-key --directions 20 \
  --sigma .5 --sigma-max 4 --shortlist 4 \
  --screen-games 1 --compare-games 2 --confirm-games 2 \
  --minimum-boot-gain 150 --envs 16 \
  --first-training-seed 190000 --seed 223 --eval-every 1
```
