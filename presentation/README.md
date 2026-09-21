# Training Breakdown: an illustrated HTML presentation

[Open the public presentation](https://breakdown-learned-replay.saschah.chatgpt.site/presentation.html).
Anyone with the link can view it; the original replay stays at the site's root.

The deliverable is [results/training-presentation.html](../results/training-presentation.html):
ten slides, self-contained and usable offline. External paper and public replay
links require internet access. No external fonts, JavaScript libraries or image
requests are needed to view the slides.

Controls: arrow keys / Space to advance, Home / End, **O** for slide overview,
**N** for speaker notes, **F** for fullscreen. Buttons also work on mobile.
Browser printing produces a ten-page landscape handout.

Rebuild from the repository root:

```sh
venv/bin/python presentation/build.py
```

`deck.html` is the editable template; do not open it as the finished deck.
The build reads frozen selection/test records and the verified winning replay,
checks their key invariants, and embeds exact recorded frames rendered with
the existing TRS-80 font. SVG diagrams are explanatory illustrations, not
measurements or generated gameplay. The outcome grids show actual counts,
grouped by category rather than game order.

The comparison distinguishes DeepMind's 2013 Atari paper and 2015 Nature DQN
recipe from this project's DQN-initialized PPO, self-generated curriculum,
self-imitation and own-policy reference. It explicitly separates a verified
validation victory from zero verified victories in 100 fresh complete games.
Primary papers are linked in the slides; project sources appear in footers
and speaker notes. The deck does not train or modify any policy or game.

Public deployment adds `presentation.html` alongside the existing replay's
`index.html`, without changing the replay URL or audience.
