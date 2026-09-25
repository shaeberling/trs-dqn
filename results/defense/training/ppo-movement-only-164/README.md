# Fresh stage-agnostic movement-only PPO: remove the firing-score shortcut

The previous [grouped key-duration trial](../ppo-grouped-duration-fresh-163/README.md)
trained for 16,777,216 own actions, but all 160 fixed and 256 fresh new-policy
games lost in stage one. Its verified best fresh replay earns 2,520 displayed
points on each of four lives; the visible reward trace includes large
750/1,500-point firing gains shortly before the recurring loss. The
protected score parent earns 2,620 per life and also loses there. The
visible-screen diagnostic suggests the score objective may reward firing
more quickly than repositioning for the later opening. It does **not**
establish a safe route or prove that suppressing fire will work.

This trial changes the **fixed action profile**, not the reward: the fresh
policy has nine logits for the original keyboard IDs 0–8 (NOOP and eight
movement directions). Those same nine choices apply at **every stage**.
There is no stage-aware controller, action override, programmed direction,
demonstration, original-game patch, hidden memory input or auxiliary reward.
The actor still sees only four raw 16×64 video-memory frames; it learns from
the displayed score difference and visible episode/life boundaries. It can
never fire, including after a stage transition, so this may impair later
stages; any passage must be observed in the original game before expanding
the action profile. This is an exploration/action-competition test, not a
replacement for the globally protected best policy.

Predeclared sequence:

1. Require the full native regression suite and focused action-profile tests.
   Freeze a seed-44 **untrained initializer** with the exact production model
   and evaluate 64 complete original-boot games, seeds 620000–620063. Save
   the full state, evaluation and independently verified best replay. This is
   a no-learning control, not a candidate for promotion.
2. Run a separate **16,384-action** seed-44 smoke with 16 original-emulator
   workers, 256-action rollouts, batch 512, four epochs, learning rate
   0.00025, entropy 0.01, gamma 0.997, GAE lambda 0.95, visible-score scale
   0.01, 100,000 T-states per action and life-terminal targets. Require a
   finite optimizer update, exact stop/full optimizer and RNG state, all
   chosen keyboard IDs below 9, ten complete fixed games on seeds
   10000–10009 with mean at least 100, and a native-verified original-boot
   replay. Smoke weights are never production weights.
3. On passing smoke, restart **fresh** from seed 44 for a bounded
   **1,048,576 own-action** pilot with the same settings. Evaluate ten
   complete fixed original-boot games every 262,144 actions. Select mission
   first, then highest stage, then fixed-game mean, earliest on a tie. Any
   stage-two or mission claim requires independent original-boot neural
   replay verification.
4. Compare the frozen selected pilot checkpoint and the frozen untrained
   initializer on **128 untouched matched complete games**, seeds
   620200–620327. Report score, game length, stage and mission separately;
   verify one best-effort replay per arm. If still stage one, extend the
   selected learner toward 4,194,304 total actions only if its fresh mean
   displayed score exceeds the initializer by at least **20** points and its
   mean complete-game action count exceeds it by at least **20** actions.
   These are compute-allocation gates, not a new reward or checkpoint rank.
   If the gate fails, archive the negative result and change mechanism.

The exact-run disk guard stops new training gracefully below 5.1 GiB free.
Selected and terminal full states, all fixed evaluations, compressed metrics
and verified replays must be archived and pushed before deleting unselected
local checkpoints. The protected 10,480-point global replay is never
overwritten by a lower-score or unverified outcome.
