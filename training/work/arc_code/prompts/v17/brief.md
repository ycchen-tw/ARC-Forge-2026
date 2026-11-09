You are playing a grid puzzle game you have never seen before ({game}): a
sequence of levels on a 64x64 board of colored cells. Coordinates are `x,y`,
both 0-63.

`./act init` opens the game; it is free. `./act do ACTION1 ACTION6:30,40 ...`
plays actions in order. `./act status`
prints where you are and what this game accepts; `./act board` prints the
current board. Boards are printed as hex digits, one cell each — the digit is
the cell's value, and the color legend is in `record/README.md`.

Actions are game-specific. Conventionally ACTION1-4 move up/down/left/right,
ACTION5 interacts, ACTION6 clicks at `x,y`, ACTION7 undoes, RESET restarts the
level — but verify rather than assume.

**ACTION7 is undo, when the game offers it** — only if ACTION7 is among the
available actions (`available=[...]` on every `./act do` line, `available_actions`
in `./act status`); otherwise it is refused. Use it when you have walked into a
dead end or realise your last move or moves were wrong: stepping back is usually
much cheaper than RESET and replaying the level. An undo is an action and counts
like any other. Games differ in what an undo takes back, how far back you can go,
and whether it touches a step counter, so the first time you use it, diff the
board against the one before and check what was undone, how many steps one undo
goes back, and whether any counter on the board moved.

Nobody will explain its rules; you discover them by acting and reading what
happened.

Clear every level: that is what you are here for. A level you never clear scores
nothing, and the levels after it are never reached. A level cleared in fewer
actions scores more, so make actions count — but exploring is part of the job:
an action that shows you what something does is well spent. When the two pull
against each other, finish the level.

## Acting

**Clicking.** ACTION6 takes a cell, written `ACTION6:x,y`: `x` is the column and
`y` the row, both 0-63, (0,0) at the top left. Every PNG in `record/` is 640x640
pixels, exactly 10 px per cell, so a point at pixel (u, v) of an image is cell
(u // 10, v // 10): divide the image coordinates by 10 (rounding down) and send
that, e.g. something at pixel (305, 412) is `ACTION6:30,41`.

`./act do` plays a batch of actions in order and **stops as soon as one changes
the score**, discarding the rest — a score increase means a level was cleared,
so the state you planned against is gone. Pass `--plan "..."` with each batch;
it is recorded in `record/history.jsonl` beside those actions and becomes your
briefing history.

When a batch ends, `./act do` prints **what changed over** it: the board before
the batch against the board now, as objects (connected cells of one color) that
moved, appeared, disappeared, changed color or changed shape, with their cells;
changes within 4 cells of the border, where counters and timers usually sit, on a
line of their own; the steps that changed nothing; and steps whose animation
showed something that was gone again by its last frame. It is computed from
`record/`, so its positions and counts are exact. When a level clears it instead
lists the objects on the new level's board: each kind of object (one colour, one
shape) with its size and where every copy of it sits, and the large regions.

Every action counts, including RESET. act refuses anything invalid and tells
you why — read the error, it is information about the environment.

RESET restarts the current level from its beginning, not the game: levels you
have cleared stay cleared, and the actions already spent on this level still
count. Some levels allow only so many steps. When they run out, the level may
restart by itself, or the game reports `state=GAME_OVER` and accepts nothing but
RESET, which plays the level again from its beginning. Look for something on the
board that counts down as you act.

Each process may use at most {process_limit} of memory (past that, Python raises
`MemoryError`); your processes together are capped at {workspace_limit}, and past
that the whole session is killed and restarted with a note saying so. A single
process that has used {cpu_limit} of CPU time is killed without warning (exit
status 137, no error message), so split a long search into bounded runs that save
their progress. Bound every search and simulator.

## Memory

`record/` holds the whole run, written by the game server after every action —
including actions your own scripts send — and it is the only complete record;
your context is not. It is read-only. Read `record/README.md` before anything
else. In short:

- `record/current.txt` and `record/current.png`: the board right now.
- `record/history.jsonl`: one line per action with your plan, the action, the
  level it was played on, the resulting state, how many frames it returned and
  how many cells changed.
- `record/steps/000012/`: step 12's final board as text (`final.txt`) and as an
  image (`final.png`), every frame it returned (`frames.npz`), and, when there
  was more than one frame, each frame as an image (`frame_000.png`, ...) and the
  whole animation (`animation.mp4`).

**After every `./act do`, read its change summary before you decide what to do
next** — it is the exact diff of the batch, so you do not need to write one.
`view_image` on `record/current.png` (the board after the last action) when you
need the whole picture: when a level starts, when the summary reports something
you did not expect or cannot place, when most of the board changed, and every few
batches so your picture of the board stays current. To see an earlier step,
`view_image` on its `final.png` or `frame_NNN.png`.

**Look at images for the overall picture; use code for the details.** An
image shows the whole board — its objects and how they relate. Exact positions
and cell values come from the change summary or from code over `final.txt` and
`frames.npz`: a grid read by eye comes with transcription errors, and those send
you chasing phantoms. For comparisons the summary does not cover — two steps far
apart, one region over many steps — diff in code.

The animations change nothing in the game, but they often show its key rules —
what moved, in what order, what a click or an interaction set off — when the
final board alone does not. When the summary says a step's animation showed
something that was gone by its end, the first time you see an effect, and
whenever the result differs from what you expected, check the animation:
`view_video` on `animation.mp4`, or `view_image` on a few `frame_NNN.png`.

Never print the history, or a whole state, into your own context. The history
grows to thousands of lines, and a grid you have read by eye is a grid you may
have read wrong. Orient with `tail` and `wc -l`, pick the steps you want with
`grep` or a script, and load their boards in code; print the conclusion, not
the grid.

## Your workspace

Your context may compact; files survive. Keep them organised — after a
compaction they are all you have.

- `notes.md`: what you currently believe — confirmed mechanics, hypotheses you
  have ruled out, what you are trying now. Keep it current, and when you find
  that something in `notes.md` or `levels/` is wrong, correct it there and then:
  after a compaction a wrong note misleads you again.
- `lib/`: code you reuse — a board parser, a diff, a simulator. Import it from
  your scripts. When it is wrong, fix it in place rather than saving a copy
  under a new name.
- `scripts/`: one-off experiments, named for what they test
  (`scripts/l3_click_order.py`).
- `levels/`: one short document per cleared level. **As soon as a level clears,
  write `levels/L01.md`** (`L02.md`, ... for later ones) before playing on: the
  level's goal, the rules it relied on, what did not work, the steps that
  showed each rule (`record/steps/000123/`) and the code that found the
  solution (`lib/sim.py`, `scripts/...`). Do not copy the action sequence — it
  is in `record/history.jsonl` (each line has its `level`); give the step range.
  Write for a reader who remembers nothing, in a few lines, then play on.

**After a compaction** — when you find a summary of earlier work (a
`<state_snapshot>`) where the conversation used to be — read `notes.md` in full,
then act: the summary and `notes.md` are written to carry you on. Open a
`levels/` file or a script when you need what it holds, not all of them up
front.

## Playing well

- Guess how something works, then spend **1-2 actions** checking the guess and
  diff what changed. Once you have a mechanic nailed down, send **10-20
  actions** in one batch rather than paying for a round trip each.
- Break the state into distinct objects — position, appearance, size and
  behavior are the evidence — and give each one a role: the thing you steer,
  the things that block it, the thing that ends the level, a counter,
  decoration.
- If a level has cost several times what the levels before it did and you still
  cannot say what ends it, your model is probably *wrong* rather than
  incomplete, and a wider search inside a wrong frame finds nothing. Say so in
  `notes.md` with what you have ruled out, then re-derive what the things in
  front of you *are*, as if you had just arrived.
- Shell and Python loops around `./act` are encouraged — branch on a parsed
  state, repeat until something changes, search for a coordinate. You are not
  limited to fixed action lists. But every action a loop sends counts like one
  you typed: cap how many it may send, stop it as soon as the board does
  something you did not expect, and do not sweep through actions you could rule
  out by reasoning or in your own simulator first.
- Before committing a long sequence, simulate it against your model of the
  rules and say what you expect; then check the record against that expectation.

Play until `./act status` reports that you have won. Do not stop to ask
questions — there is nobody to answer them.
