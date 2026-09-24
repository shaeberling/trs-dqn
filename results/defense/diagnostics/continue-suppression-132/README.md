# PPO suppresses continuation on identical approach screens

This is a read-only comparison of three frozen **screen-only neural policies**
on the same four verified 64-action pre-flash windows from run 121. The
[report](report.json) records the exact replay and model hashes. The
untrained +9 initializer predicts an average **11.65** continuation choices
per window; the selected model after 917,504 PPO actions predicts only
**0.89** on those identical frames. The earlier trained +7 model predicts
**1.31**. These are sums of categorical probabilities, not sampled choice
counts or demonstrated actions.

The selected +9 model's own separately verified replay chose continuation
4 / 3 / 1 / 5 times in its four approach windows and lost all lives at
2,620 displayed points. Its 128-game fresh mean regressed sharply versus
both comparators. Thus simply increasing the initial neutral bias made
the option available but did not prevent PPO from suppressing it where
the repeated loss occurs. The identical-screen comparison rules out
different sampled trajectories as the sole explanation of the lower
propensity. It does **not** identify the exact gradient, prove the
navigation route, or show that sustained movement would pass the wall.

No model or replay was altered. No native hidden state, reward shaping,
demonstration or scripted action entered learning or the diagnostic.
