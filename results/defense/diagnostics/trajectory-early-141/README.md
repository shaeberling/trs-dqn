# Earlier own-trajectory branch point

The bounded [frame-288 search](../trajectory-search-140/README.md) tested
20,000 score-ranked, symmetric mutations of one verified own-policy life.
It found no extra first-life points or stage passage and delayed the
visible loss by at most two actions. Frame 288 is just 34 decisions before
the first visible middle opening in the independent timing diagnosis;
it may be too late to change the ship's entry position for both openings.

This controlled sensitivity test uses the **same already-tested search
code**, verified source replay, 20,000-candidate bound, elite capacity 64,
seed 547, symmetric ten-command sampler, 4/8/16/32/64-action contiguous
replacement lengths, score-first ranking and boot-reexecution gate. Only
the branch point moves from **frame 288 to 160**, and the plan horizon
grows from **144 to 272** so both searches end at source frame 432 if
still alive. The original own-policy baseline now lasts 247 actions
after the anchor before its first visible loss at frame 407. All
candidate actions remain diagnostic-only and are never supplied to a
learner, promoted as a replay, or treated as an expert route. The source
game, reward and policy input are unchanged. Native snapshots are opaque
reset machinery; no course pointer or hand-selected direction enters
the search.

First run a 200-candidate integration smoke, then the predeclared
**20,000 distinct-candidate** production bound. Only a native-reexecuted
stage-two or mission observation stops it early. Report the maximum
displayed first-life score, survival beyond the same frame-407 source
loss, and all plans; do not claim that a higher score from one search
plan is a learned model.

```sh
venv/bin/python -u -m rl.defense_trajectory_search \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output runs/defense-trajectory-early-141 \
  --anchor 160 --horizon 272 --candidates 20000 \
  --elite 64 --seed 547
```

The [200-candidate smoke](smoke/report.json) exactly reproduced the
source baseline's 2,620 points and 247-action suffix, with no stage-two
observation. It checks the earlier native snapshot and bookkeeping;
it is excluded from production selection.

## Completed result

The full run completed all **20,000 distinct mutations** plus the
baseline. The baseline scored 2,620 and lost the first life after 247
actions from frame 160. The best mutant scored the same 2,620 and lost
after **249** actions. **2,789** plans matched 2,620 points and **1,846**
of them outlasted the baseline by at least one action, but none scored
above 2,620, survived the 272-action horizon, reached stage two or
completed a mission. The [full archive](run/) preserves each 272-action
plan, its real outcome, source checksums, code snapshot and RNG state.

The later [frame-288 comparison](../trajectory-search-140/README.md)
likewise gained at most two visible actions without extra score. Moving
the branch point 128 actions earlier did not overcome the flat
2,620-point outcome under this score-first elite selection. That is
negative evidence for this specific search topology, not proof that an
earlier screen-conditioned route is impossible. No search plan was used
to train or promote a policy; the verified global best remains intact.
