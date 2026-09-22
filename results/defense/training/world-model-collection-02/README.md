# Additional own-experience collection: lower exploration

Twenty new training games (seeds 66000–66019) contain **46,617 actions**;
four independently assigned held-out games (66020–66023) contain **9,412**.
All finish in stage 1. Training-game mean score is **7,466**, held-out mean
**7,920**. These are exploratory collection statistics, not evaluation of
a newly learned policy or evidence of stage passage.

The same own scalar DQN 33 plays with nominal epsilon .05, random holds up
to 64 commands, exponent 1.5, and unchanged original-game timing. The
[manifest](manifest.json) hashes all episodes and identifies the exact own
parent and collection source. Every copied file was hash-checked. Each file
contains only visible frames, executed actions, score changes and visible
life/episode boundaries. No saved evaluation replay, hidden state, route,
demonstration or imitation target was used.

This collection broadens the data beyond the earlier epsilon-.25 batch,
whose training-game mean was 2,746. Higher collection score suggests more
late-approach experience but is not a measured course index. These episodes
have **not yet been used for updates**: the current world-model continuation
still fits only the original collection's training split. Keep both sets of
held-out games excluded when constructing the next training comparison.
