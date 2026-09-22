# Separate inverse-action representation experiment

The SPR calibration's clean and training-dropout probes found weak action
sensitivity; neither learned predictor beat the unchanged-current-feature
baseline on the selected replay. This motivates testing a complementary
representation objective, but does not establish the cause of navigation failure.

## Published idea and deliberate limits

[Pathak et al., ICML 2017](https://arxiv.org/abs/1705.05363) train an inverse
dynamics classifier to predict an agent's recorded action from features of its
current and next observations. Their full ICM method also uses a forward model
and intrinsic reward. **This experiment uses neither of those components.** It
is inverse-action auxiliary learning alongside scalar DQN, not an ICM reproduction
or a claim to inherit its exploration results.

## Implementation

- An optional `--inverse-weight .01` leaves the ordinary QNetwork acting path
  unchanged. Only training sees the next observation. No action override,
  simulation lookahead, extra reward, action-semantic grouping or hidden state.
- The same online network encodes actual adjacent four-frame observations into
  256-dimensional features. A separate classifier concatenates the pair, applies
  a 512→256 ReLU hidden layer, then predicts all 20 original action labels.
  Both encoder paths receive gradients. There is no dropout or EMA target.
- Train on the root action's **immediate** next observation, not the n-step
  endpoint. Existing checked compact trajectory storage provides this field;
  the original sampled TD fields, sampler RNG, returns and discounts stay exact.
  No synthetic or reset-crossing transitions are made.
- Add .01 times PER-weighted action cross entropy to the original Double-Q
  objective. Priorities remain TD errors only. Clip the joint gradient norm to
  10; retain the original Q Adam and use a separate Adam for the new classifier.
- Save `inverse-head.safetensors` and `inverse-optimizer.npz` before the state
  marker, with required hashes and architecture identity checked on resume.
  Scalar conversion preserves original online/target/Adam and main/exploration
  RNG; only auxiliary parameters/Adam start fresh. Ordinary frozen Q weights
  remain sufficient for evaluation and verified replay.
- This first experiment excludes SPR, trace cutting, quantiles and bootstrap
  heads. It adds no new task to existing live trials.

## Interpretation pitfalls

Actions with equivalent visible effects remain separate labels. Some actual
decisions have no visible effect, so perfect classification may be impossible.
Persistent exploration and a learned greedy policy also make actions partly
predictable from the starting observation alone. Class imbalance can inflate
accuracy. Accordingly, training accuracy is not evidence of controllable features
or successful navigation. A later frozen diagnostic should compare actual
screen pairs with shuffled/repeated next screens and action-frequency baselines;
diagnostic/evaluation traces must never become training examples.

## Acceptance and comparison

Required before a performance run: disabled/default array and RNG parity,
actual-parent zero-update conversion, gradient routing and TD-only priorities,
immediate-next-screen alignment through ring wraps and life/reset boundaries,
full auxiliary resume and corruption rejection, native own resets, frozen-game
verification, full regression tests and production-size memory/throughput check.
Tiny checks and their trajectories are not performance evidence or parents.

Then start directly from the same original DQN-33 parent at 6,962,144 actions,
with the historical worker-split baseline settings and 131,072 new own actions.
Evaluate ten complete uncapped boot games on reused seeds 10000–10009. Preserve
the full state and independently verified best replay, including negative results.
The short calibration stays outside the global collector. Passing the recurring
barrier and observing the original mission ending—not action accuracy or a lower
auxiliary loss—remains the goal.
