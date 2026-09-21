# GOALS

## The assignment

Teach a neural network to play the TRS-80 game **Breakdown** — a
Breakout-style brick-breaking game — and play it well. This repository gives
you a complete, scriptable TRS-80 emulator and the game itself. The AI is
yours to build, from scratch, however you see fit within the rules below.

## The goal

Train an agent that **clears as many levels as possible**. Level 1 is a wall
of roughly 77 bricks (1 point each); clearing it advances to level 2 with a
fresh wall, and so on. A game ends after three lost balls.

Report performance as the mean, median, and best score over a fixed set of
complete games (at least 10), along with the highest level reached.

The current user-confirmed completion target is to **beat all eight original
levels**, without extending or modifying the game. The original game ends
after level 8; reaching that level alone does not count as a win. Preserve a
full, verified learned-policy winning replay, then evaluate the frozen
validation-selected model on 100 fresh complete games. Report verified wins
and unresolved final-level outcomes separately; never infer victory from a
score threshold or a nonexistent displayed level 9/10. See TRAINING.md for
the screen-only outcome proof and its conservative last-ball limitation.

## The rules

1. **The agent must learn.** Its play must come from a trained model, not
   from hand-written game logic. No scripted heuristics steering or
   overriding the policy at play time.
2. **Screen in, actions out.** The model's input is the screen contents only
   — pixels, or the video memory they are rendered from. No other game
   internals may be fed to the model.
3. **The score is the reward.** Train from the game's own score (plus
   episode boundaries). Your *tooling* may read video memory to parse the
   score and detect game state — the score is literally on screen — but the
   *policy* sees only the screen.
4. **No demonstrations.** No human play data, no scripted expert generating
   examples. The agent learns from its own experience (reinforcement
   learning).
5. **Any published deep-RL method is fair game.** The canonical starting
   point is DeepMind's DQN — Mnih et al., *"Human-level control through deep
   reinforcement learning"*, Nature 518, 529–533 (2015), which learned Atari
   games (including Breakout) from pixels alone — but you are free to use
   any successor or alternative from the literature.

## The game (facts you would learn in five minutes of play)

- Controls: **LEFT** / **RIGHT** move the paddle, **SPACE** serves.
- A game gives you **3 balls**. The first ball serves itself shortly after
  the game boots.
- When a ball is lost the game **pauses indefinitely** (showing "PASS BALL"
  or "THAT HURTS") until SPACE is pressed to serve the next one. After the
  third lost ball it shows "GAME OVER"; SPACE then starts a fresh game.
- On screen: the score is 5 ASCII digits at the top left (video memory
  `0x3C00 + 6`), the level is 5 digits at the top right (`0x3C00 + 59`), and
  status messages appear centered on character row 10.
- Each brick is worth 1 point.

## The emulator (how to drive it)

Set up per [README.md](README.md), then everything is scriptable from
Python. One emulator instance per process (the C core keeps global state).

```python
from trs import TRS, Key, Screenshot
from main import CONFIGS

# original_speed=0 → run as fast as the host allows (far faster than real
# time); no_ui=True → headless, no window.
trs = TRS(CONFIGS["breakdown"], original_speed=0, fps=2.0, no_ui=True)

trs.boot()                        # run the game's boot sequence (see main.py)

shot = Screenshot(trs.ram, (0, 0, 64, 16))   # viewport: x, y, w, h in cells

trs.keyboard.all_keys_up()        # release everything
trs.keyboard.key_down(Key.RIGHT)  # hold a key
trs.run_for_tstates(50000)        # advance the CPU; 50,000 T-states at
                                  # 1.77408 MHz ≈ 28 ms of game time

frame = shot.screenshot()         # numpy array, 48 rows × 128 cols, {0, 1}
byte = trs.ram.peek(0x3C00 + 6)   # read any byte of memory
```

Details worth knowing:

- The screen is 64×16 **character** cells. Graphics characters (codes
  0x80–0xBF) subdivide each cell into 2×3 pixels, giving the 128×48 pixel
  screenshot. **The screenshot renders only graphics characters** — text
  (score digits, messages) does not appear in the pixel output; read text
  directly from video memory with `ram.peek`.
- `trs.boot()` resets the machine and replays the boot sequence from the
  game's config in `main.py`; use it to restart cleanly between episodes.
- Keys stay held until released — call `all_keys_up()` (or `key_up`) between
  presses.
- To *watch* what your agent is doing, `TRS(..., no_ui=False)` +
  `trs.mainloop()` renders the screen in a window while your code drives the
  emulator from another thread.

## Deliverables

1. Training code, with instructions to reproduce a run from scratch.
2. Ongoing training logs/metrics so progress is visible during a run.
3. A saved trained model, plus an evaluation script that plays N complete
   games and reports mean / median / best score and highest level reached.
4. A way to watch the trained agent play.
5. Short notes on your approach, what you tried, and results.

## Ground rules

Treat this branch as the entire repository. If your clone happens to contain
other branches, do not read them — they contain a prior solution, and
looking would defeat the purpose of this clean-room exercise.
