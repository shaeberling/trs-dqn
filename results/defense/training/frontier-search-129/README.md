# Chained own-screen frontier exploration

The [two-gap timing diagnostic](../../diagnostics/gate-timing-127/README.md)
suggests that passage may require several different movement phases. The
previous [random two-hold probe](../macro-explore-100-104/README.md) restored
each own state, applied two unbiased held commands, then resumed the frozen
policy; 11,280 such continuations did not reach stage two. Run 129 tests a
different exploration topology: it retains diverse *intermediate* screens
and chains further unbiased held commands from them. A short sequence can
therefore build on another short sequence, rather than returning to the
same failed parent after two holds.

The 48 input states come from twelve complete, replay-verified **own neural
games** in the already archived run-69 pre-loss source set. Each source is
restored and its recorded visible screens and score increments are exactly
re-executed before the search uses it. Prefix states every eight own actions
seed a bounded screen-cell archive; the HUD is excluded from the fingerprint.
An expansion selects an archived state uniformly half the time and from
the upper quartile of **displayed within-life score** half the time. It then
samples uniformly among the ten distinct stage-one physical commands (no-op,
eight directions, fire) and hold lengths **4, 8, 16 or 32**. Any surviving
new screen cell may enter a 4,096-cell reservoir. This is training-only
exploration with opaque same-build emulator resets; the native bytes are
never decoded or fed to a model. Cell novelty chooses reset states, not
actions or reward. No particular direction, route, opening or collision
condition is supplied. There are no model updates or demonstrations, and a
random-branch trace can never replace a learned-policy replay.

Predeclared bounded first gate: **100,000 macro expansions**, seed **503**,
at the original 100,000-T-state action cadence. Stop early only if an actual
stage-two screen or mission outcome is observed and the entire chain is
re-executed from its own source with matching visible rewards/screens. Store
all expansion plans, accepted-node ancestry, source hashes, RNG state and
periodic progress. If there is no stage-two discovery, report the explored
action count, distinct screen cells, losses and best *new* displayed
within-life score separately from the best score already present in source
prefixes. This is an exploration gate, not a claim that the model has learned
or a substitute for a complete-game mission.

```sh
venv/bin/python -u -m rl.defense_frontier_search \
  --source-archive results/defense/training/ars-score-gated-69/run/own-loss-states \
  --output runs/defense-frontier-search-129 \
  --expansions 100000 --capacity 4096 --source-stride 8 --seed 503
```

The final-code [100-expansion native smoke](smoke/) re-executed all 48 source
prefixes, explored 1,177 new actions, visited 345 coarse cells and admitted
40 new intermediate states. It did not reach stage two. Its source-best and
exploration-best displayed life scores were both 2,640; that equality is not
evidence of new progress. The smoke is excluded from any best-model
collector.

The full **478-test** regression suite passed before the production search.

The production [run archive](run/) completed all **100,000** expansions and
**871,071** new emulator actions. The complete plan log, accepted-node
ancestry, source hashes, RNG state, final status and exact source code are
preserved; the archived files were content-checked against the live run.
The 4,096-cell archive filled and encountered **10,898** distinct coarse
visible cells over time. It admitted **12,005** new nodes, including chains
of up to **12** random holds; **6,558** admitted nodes had at least four
chained holds. The ten physical commands were sampled roughly uniformly.
These counts show that the mechanism chained beyond the old two-hold probe,
not that it found a better course path.

The original own-source prefixes already contained a best displayed
within-life score of **2,640**. New random branches also reached 2,640
**10,841** times, but **never exceeded it**. The search recorded **53,057**
visible life losses, **zero** stage-two screens and **zero** mission outcomes.
The [checked analysis](analysis.json) separates seeded from exploratory
score, archive diversity and macro depth. This negative result argues
against simply extending the same unbiased short-hold frontier search;
it does not prove no screen-conditioned policy can pass the obstacle.
No weights were trained or promoted, and the original verified best replay
and confirmed score parent remain unchanged.
