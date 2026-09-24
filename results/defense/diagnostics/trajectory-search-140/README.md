# Score-only mutation search around an own barrier approach

The verified option-credit policy repeatedly reaches 2,620 displayed
points on its first life and then loses at original stage-one stream row
33–34. Unbiased held-key frontier searches and within-life factor noise
have not reached stage two. This bounded diagnostic tests a different
exploration topology: preserve an entire successful *own-policy* approach
as the initial action sequence, then mutate several contiguous command
segments of selected candidate parents and retain high-scoring,
longer-surviving sequences.
It does **not** supply a route, preferred direction, obstacle coordinate,
collision oracle or extra reward.

Use the independently [native-verified run-136 fresh replay](../../training/ppo-duration-credit-136/run/fresh-selected-replay/replay.html)
as the sole source. Reexecute its own prefix from boot through first-life
frame **288**, then save that opaque same-build emulator state for efficient
branching. Verify the original suffix through its visible life loss at
frame **407**, displayed score **2,620**. Candidate plans cover **144 base
actions** after the anchor; the population starts with the source
policy's own recorded commands, and each descendant receives 1–4
independent uniform command replacements of
**4, 8, 16, 32 or 64** actions. Commands are sampled symmetrically from
the ten distinct stage-one physical keys (NOOP, eight movement keys and
SPACE). A small elite pool combines higher *displayed score* with later
visible life survival only as a tie-break. Every candidate plan and real
score/life/stage outcome is recorded. No hidden course pointer, screen
geometry parser, trained policy update, demonstration, or replay promotion
is involved. Parent sampling is 75% uniform from the current score elite
and 25% from its highest-ranked member; after improvement, that latter
member is not necessarily the original own-policy baseline. This detail
is pinned by the archived source code and plans.
Intermediate opaque states are reset machinery only, not
policy inputs. A discovery must be reexecuted exactly from the original
boot and independently checked before suggesting a learning follow-up.

Predeclare a **200-candidate implementation smoke**, then **20,000
production candidates**, seed 547, elite capacity 64, original 100,000
T-state cadence. Stop production early only on a native-reexecuted
stage-two or mission observation. The primary evidence is actual
stage/score outcome; report best score, maximum survival beyond the
original first-life loss, and candidate count even if all fail. A random
branch is not a learned model or a verified learned replay. Do not feed
any searched action sequence into PPO or promote it as learned play.

```sh
venv/bin/python -u -m rl.defense_trajectory_search \
  results/defense/training/ppo-duration-credit-136/run/fresh-selected-replay \
  --output runs/defense-trajectory-search-140 \
  --anchor 288 --horizon 144 --candidates 20000 \
  --elite 64 --seed 547
```

The [final-code 200-candidate smoke](smoke/report.json) exactly reproduced
the own-policy first-life baseline: 2,620 points and a visible loss 119
actions after the anchor. Its best mutation lasted **120** actions at the
same score; it did not enter stage two. The smoke only checks native
reexecution and search bookkeeping, not gameplay success.
The full **497-test** regression suite passed before production.

## Completed bounded search

The production search completed all **20,000 distinct mutated plans**
plus the own-policy baseline. **4,792** plans attained exactly the
baseline's **2,620** displayed first-life points, and **3,208** of those
lasted at least one action beyond the baseline's 119-action suffix before
visible loss. The best survived **121** actions after the frame-288
anchor, just **two** beyond the source, still at 2,620 points. **None**
scored above 2,620, survived the 144-action horizon, reached stage two,
or completed a mission. Every plan, mutation, real outcome, source hash,
RNG state and final report are in the [full run archive](run/).

This is evidence against further local score-ranked mutation from frame
288 with the same horizon and replacement profile. The anchor is only
34 actions before the first visible middle opening in the older
[two-gap timing report](../gate-timing-127/README.md); it may already be
too late to change the ship's entry position for the far-right opening.
That is a hypothesis, not a collision or feasibility proof. This
diagnostic has not trained or promoted any neural policy, and the global
verified best replay is unchanged.
