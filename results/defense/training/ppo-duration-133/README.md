# Learned key-duration PPO experiment

The prior screen-only learners repeatedly end their first stage near the same
visible obstacle sequence. A native-verified replay and screen-only geometry
probe showed two offset openings: moving right from too early loses before the
usual event, while starting late reaches only about column 43–45 before the
second opening at column 51. This suggests a phase-sensitive sustained-action
exploration problem, not proof of a solvable intervention. No emulator state,
obstacle index, geometry probe, or scripted action enters this learner.

This opt-in policy chooses one of the original 20 physical key combinations
and a hold of 1, 4, 16 or 64 base actions from four raw screen frames. It
keeps that *learned* choice until the hold expires, unless a visible life or
episode boundary cancels it. PPO's actor updates only on genuine option
starts; the score-value critic learns at every base action. Reward remains
the displayed score difference. The parent is the learner's own ordinary
[PPO checkpoint](../ppo-persistent-noise-long-119/milestone-000009718528/state.json),
not demonstrations. Its encoder, critic, and physical actor rows are copied;
the four duration choices receive direction-neutral prior logit offsets, and
the optimizer/RNG start fresh.

Frozen 16-game calibrations (seeds 605000–605015) found that duration prior
spacing 2 severely disrupts play (mean 1,868.13), spacing 4 retains much
of the parent (9,241.25), and spacing 5 retains more (10,305.63) but makes
long holds vanishingly rare. All stayed in stage one. Spacing 4 is selected
to explore durations, **not** on the basis of a validation win.

The first 16,384-action optimizer smoke from spacing 4 used inherited
learning rate 2.5e-4 and a longer GAE lambda 0.995. It regressed sharply:
the frozen baseline averaged 9,135.63 on the 16 fixed seeds 10000–10015,
whereas the trained checkpoint averaged 4,960; all stayed stage one. That
checkpoint is diagnostic only and never replaces the verified global best.

## Predeclared conservative pilot

Starting from the *frozen* spacing-4 initialized optimizer, train exactly
131,072 base actions with 16 workers, four boot-only workers, the inherited
own-state score curriculum and per-life actor-bias exploration. Use learning
rate **2.5e-5** and the parent's GAE lambda **0.95**, evaluate complete games
at 32,768-action intervals on ten fixed seeds 10000–10009. Retain every full
optimizer/RNG milestone. Select the earliest checkpoint with the highest
stage rank, then highest fixed-game mean if ranks tie. If none reaches stage
two, compare the selected checkpoint and frozen initializer on 32 fresh
matched complete games, seeds **606000–606031**. A score-only gain cannot
replace the global verified best or count as obstacle passage. A stage-two
candidate requires an independent native-verified replay and further fresh
confirmation before promotion. Preserve all outcomes, including failure.

The short-run command is:

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-ppo-133-pilot \
  --artifacts runs/defense-duration-ppo-133-pilot/artifacts \
  --resume runs/defense-duration-ppo-133-init-spacing4/latest \
  --steps 131072 --learning-rate 0.000025 --gae-lambda 0.95 \
  --eval-every 32768 --eval-games 10 --eval-envs 10
```

The conservative pilot finished all four checks: fixed ten-game means were
**9,802 / 9,512 / 9,590 / 10,300**, all stage one. The final checkpoint
was selected. Its native-verified local replay scored 10,480, but never
reached stage two. On the predeclared 32 fresh matched games it averaged
**9,411.56**, versus **9,130.94** for the frozen initializer; again all
64 games were stage one. This is a tentative score gain, not a pass of the
repeated obstacle. The learner used only about 80 sixteen-action options
through 118,784 training actions and no 64-action options. Its exploration
almost never exercised the sustained action this experiment was meant to
test. The global verified best replay is therefore unchanged.

## Direction-neutral duration exploration

The next bounded trial starts again from the frozen spacing-4 initializer to
isolate an exploration change. Each training life draws one Gaussian logit
offset *per duration*, shared across all 20 physical commands; it never
selects a key, a screen position or an obstacle phase. This is an on-policy
PPO perturbation recorded with the learner's own trajectories and removed
during evaluation. Use standard deviation **4**, learning rate **2.5e-5**,
GAE lambda **0.95**, 16 workers with four boot-only, and the same own-state
curriculum. Run **524,288** base actions, evaluating ten complete games every
65,536 actions on fixed seeds 10000–10009. Retain all full optimizer/RNG
milestones. If the full run finishes stage-one-only, select the earliest
highest fixed-game-mean checkpoint and compare it with the frozen initializer
on **64 fresh matched complete games, seeds 606200–606263**. A real stage-two
result instead requires independent native replay plus fresh confirmation.
Do not promote or claim success on score alone. If two consecutive fixed
checks have mean below 5,000, end the run as a documented failed trial
rather than spend more actions on a collapsed policy.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-ppo-133-factor-noise \
  --artifacts runs/defense-duration-ppo-133-factor-noise/artifacts \
  --resume runs/defense-duration-ppo-133-init-spacing4/latest \
  --steps 524288 --learning-rate 0.000025 --gae-lambda 0.95 \
  --policy-bias-noise 0 --policy-duration-noise 4 \
  --eval-every 65536 --eval-games 10 --eval-envs 10
```

The complete eight-check continuation finished cleanly. Fixed ten-game
means were **9,097 / 9,411 / 9,329 / 9,650 / 9,644 / 10,180 / 9,920 /
10,052**. All **80** games ended in stage one. The sixth, 393,216-action
[checkpoint](factor-noise/step-000000393216/state.json) was selected by the
predeclared rule. Through 512,000 training actions the learner had started
**2,582 sixteen-step** and **343 sixty-four-step** options; the previous
near-absence of sustained actions was genuinely changed.

On [64 fresh complete games](factor-noise/fresh-selected-64.json), selected
mean/median/best were **9,812.03 / 10,360 / 10,480**, versus **9,559.84 /
10,360 / 10,480** for the [frozen initializer](initializer-spacing4/fresh-matched-64.json)
on the identical seeds. It improved 31 paired games, worsened 30 and tied
three: the positive mean difference is not broad pairwise dominance. All
**128** fresh games remained stage one. The [native-verified local replay](factor-noise/artifacts/best/replay.html)
again scores 10,480 and ends in a loss; it does not replace the protected
global replay.

A separate [read-only forensic reexecution](../../diagnostics/duration-133-course-progress/README.md)
finds that both new learned replays lose all four lives at decoded stage-one
stream row **33 or 34 of 126**, as in earlier policies. This hidden pointer
was not used by the learner or evaluator. Together, these outcomes show that
direction-neutral long-hold exploration alone has not solved the repeated
early navigation failure. Simply extending this exact run is possible, but
there is no observed passage yet to justify claiming it will learn one.
The full **489-test** repository regression suite passed after adding this
policy and exploration mode.
