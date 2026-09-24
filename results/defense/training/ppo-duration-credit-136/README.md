# Credit the complete learned hold, not just its early steps

The joint key-duration policy repeatedly reaches a high stage-one score
but loses at decoded stream row 33–34. Its [screen-diverse continuation](../ppo-duration-screen-135/README.md)
filled all 128 screen cells per worker and still did not pass the obstacle;
perfect reused-seed checks regressed on 64 fresh games. A specific
learning-rule limitation remains: the current PPO actor advantage is
ordinary **base-step** GAE sampled only at a learned option's start.
With gamma .997 and lambda .95, a TD event 63 actions later receives
roughly `(.997 * .95) ** 63`, about **3.2%**, of its original weight at
the start of a 64-action hold. Thus a late collision or delayed score
may barely update the option that caused it.

Add an **opt-in semi-Markov actor GAE**. At each actual option start, sum
every displayed-score reward during the complete hold with ordinary
per-base-action gamma, then bootstrap from the next option start (or
zero at a visible learning boundary). Recursively carry advantage with
one lambda factor **per completed option**, not per forced base action.
If a hold extends beyond a rollout, omit that unfinished start from actor
updates rather than pretending it finished; all its real base transitions
still train the same critic. The actor still samples from four raw screen
frames and the only reward is the displayed score. The executor, action
choices, evaluation and native replay semantics do not change. This is
an attribution change, not an obstacle hint, extra reward or demonstration.

Use the **same frozen full 917,504-action joint checkpoint** as the
screen-diverse [control run 135](../ppo-duration-screen-135/README.md),
the same 128-cell visible-screen life-loss archive, 16 workers (four
boot-only), symmetric key noise 2 and duration noise 4, learning rate
2.5e-5, gamma .997 and GAE lambda .95. The sole planned training change
is `--option-actor-gae`. A short optimizer smoke may check finite losses
and exact replay but cannot select or tune the production model.

Train **1,048,576 additional base actions** to absolute counter
**1,966,080**, with eight complete ten-game validations every 131,072
actions on seeds 10000–10009. Retain all full optimizer/RNG checkpoints.
Select earliest highest stage rank, breaking ties by fixed-game mean.
Any stage-two result requires independent native replay and fresh
confirmation. If all remain stage one, compare the frozen selection,
the control's selected 1,310,720-action checkpoint, and the shared input
joint checkpoint on **64 new matched complete games, seeds 607200–607263**.
Score-only improvement cannot replace the protected best replay. If two
consecutive fixed checks average below 5,000, archive and stop early.

```sh
venv/bin/python -u -m rl.defense_train \
  --run runs/defense-duration-credit-136 \
  --artifacts runs/defense-duration-credit-136/artifacts \
  --resume results/defense/training/ppo-duration-joint-134/run/step-000000917504 \
  --curriculum-cells screen --curriculum-bins 128 --curriculum-per-bin 1 \
  --option-actor-gae --steps 1966080 --eval-every 131072 \
  --eval-games 10 --eval-envs 10
```

A separate 16,384-action optimizer smoke from the same frozen source
finished with finite updates, four complete fixed-seed games at 10,480
mean (all stage one), and a native-verified **2,553-action** local replay.
It only establishes that option-credit updates, exact replay and resume
semantics run end to end; the smoke checkpoint is excluded from production
selection and from any claim of obstacle passage.
The full **494-test** repository regression suite passed before production.
