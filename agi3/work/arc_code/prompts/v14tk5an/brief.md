You are playing a grid puzzle game you have never seen before ({game}): a
sequence of levels on a 64x64 board of colored cells. Coordinates are `x,y`,
both 0-63.

`./act init` opens the game; it is free. `./act do ACTION1 ACTION6:30,40 ...`
plays actions in order. `./act status`
prints where you are and what this game accepts; `./act board` prints the
current board. Boards are printed as hex digits, one cell each — the digit is
the cell's value, and the color legend is in `record/README.md`. That is 64
lines: for a region, `python3 lib/arc.py text X0 Y0 X1 Y1` is cheaper.

Actions are game-specific. Conventionally ACTION1-4 move up/down/left/right,
ACTION5 interacts, ACTION6 clicks at `x,y`, ACTION7 undoes, RESET restarts the
level — but verify rather than assume.

**ACTION7 is undo, when the game offers it** — only if ACTION7 is among the
available actions (`available=[...]` on every `./act do` line, `available_actions`
in `./act status`); otherwise it is refused. Use it when you have walked into a
dead end or realise your last move or moves were wrong: stepping back is usually
much cheaper than RESET and replaying the level. An undo is an action and counts
like any other. Games differ in what an undo takes back, how far back you can go,
and whether it touches a step counter, so the first time you use it, compare
the board after it with the boards before (`python3 lib/arc.py changes A B`):
what was undone, how many steps one undo goes back, and whether any counter on
the board moved.

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
that, e.g. something at pixel (305, 412) is `ACTION6:30,41`. Use that to find
where to look, then confirm the cell in the text grid
(`python3 lib/arc.py text X0 Y0 X1 Y1`) before you click.

`./act do` plays a batch of actions in order and **stops as soon as one changes
the score**, discarding the rest — a score increase means a level was cleared,
so the state you planned against is gone. Pass `--plan "..."` with each batch;
it is recorded in `record/history.jsonl` beside those actions and becomes your
briefing history.

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

**After every `./act do`, look at the latest game image before you decide what to
do next:** run `view_image` on `record/current.png` (the board after the last
action, replaced after every action). Do this after every batch, even when the
printed lines look unremarkable — the picture is what shows you a changed object,
a moved piece or a new element that a count of changed cells does not. To see an
earlier step, `view_image` on its `final.png` or `frame_NNN.png`.

**Look at images for the overall picture; use code for the details.** An
image (`view_image` on `current.png`, a `final.png` or a frame) shows the whole
board — its objects and what changed. Exact positions, cell values
and diffs come from code over the record (`lib/arc.py`, below): a grid read by
eye comes with transcription errors, and those send you chasing phantoms. Diffing
two states to see exactly what changed is the single most useful thing you can
do.

The animations change nothing in the game, but they often show its key rules —
what moved, in what order, what a click or an interaction set off, how a life
was lost or a level was won — when the final board alone does not. In many games
every move returns a few frames, and looking at all of them is not worth it.
**But watch a step's animation** (`view_video` on its `animation.mp4`, or
`view_image` on its `frame_NNN.png`; the step's line in `./act do` names its
folder, and `python3 lib/arc.py anim N` says what moved, frame by frame)
**whenever one of these happens:**

- you do something for the first time: an action you have not used yet, or an
  action on a kind of object you have not touched yet;
- you die or lose: the level restarts by itself, a life or some other counter
  runs out, or the state turns `GAME_OVER`;
- a level clears: the animation of the step that cleared it shows what the game
  checked and what it did;
- the board changes much more than usual, or in a way you did not predict.

These are the moments the rules show themselves, and the final board hides how
it got there: watch first, then guess. `./act do` also says when an animation
stands out (below) and names its key frames; look then too.

Never print the history into your context: it grows to thousands of lines.
Orient in it with `tail` and `wc -l`, pick the steps you want with `grep` or a
script, and load their boards in code. A grid you have read by eye is a grid you
may have read wrong: print the conclusion, not the grid.

## Your toolkit: `lib/arc.py`

`lib/arc.py` is already in your workspace: tested helpers that read `record/`
with numpy (`python3 lib/arc.py --help`). Use them rather than writing your own,
and put what is specific to this game (readers, simulators) in your own files
in `lib/`.

- From the shell: `python3 lib/arc.py changes 12 13` (the cells that changed,
  by colour, and the counters), `diff 12 13` (the same by object), `anim 12`
  (step 12 frame by frame), `scene` (the objects on the board now, grouped by
  kind), `text X0 Y0 X1 Y1` (a region as hex digits with x and y rulers; add
  `--dots` to show the background as `.`, `--step N --frame K` for a frame),
  `png X0 Y0 X1 Y1` (that region zoomed into `zoom.png`, for `view_image`),
  `coarse X0 Y0 STEP` (a board drawn in blocks, one character per block),
  `steps` (this level's actions), `coverage` (what this level's record has
  tried, and what not), `movers STEP` and `follow STEP COLOUR` (who moved in
  which frames), `map [LEVEL]` and `cam [STEP]` (scrolling levels, below)
  and `check lib/sim.py` (below). `changes A B --cells`
  also lists every changed cell as `(x,y) old->new`; `diff A B --all` does not
  cut the list.
- From Python: `import sys; sys.path.insert(0, "lib"); from arc import *`, then
  `board()`, `board(12)`, `board(-1)`, `load_board(path)`, `frames(12)`,
  `history()`, `level_steps()`, `timeline(fn)` (fn of the board after every
  step of this level), `objects(b)` (`conn=8` joins diagonal cells,
  `multicolor=True` makes one object of a sprite drawn in several colours),
  `label(mask)` (the connected groups of a boolean array),
  `coarse(b, x0, y0, step)`, `png(b, path, box)`,
  `cells_of(b, c)`, `bbox_of(b, c)`, `colour_counts(b)`, `changes(a, b)` and
  `diff(a, b)`: a list of `(x, y, old, new)`, one per changed cell, that also
  has `.moved` (only when one object of that colour and shape moved),
  `.rearranged` (identical objects that changed places), `.appeared`,
  `.disappeared`. Boards are numpy arrays indexed `b[y, x]`; `at(b, x, y)` reads
  one cell without swapping x and y.
- **Simulators.** Give yours `step(state, action, x, y) -> state` (`action` is
  the string `"ACTION1"` ... `"ACTION7"`; `x, y` the click, else `None`) and,
  when your state is not the board, `parse(board) -> state`; then
  `python3 lib/arc.py check lib/sim.py --parse lib/sim.py:parse` replays this
  level's recorded actions through it, carrying your state from the level's
  first board, and shows the steps it gets wrong, per action (right after a
  clear: the level you just cleared). A board state is compared without its
  edge band, where counters and timers sit: a simulator that is right except
  for a counter is good enough.
- **"Impossible" needs a probe.** A search over your simulator that finds no
  solution only shows that your simulator has none, and a simulator that
  matches every recorded step can still be wrong about what you never tried.
  Before you write *impossible*, *frozen* or *only X works*, run
  `python3 lib/arc.py coverage`, name the untried actions, objects or colours
  that the conclusion depends on, and spend an action or two on the cheapest
  one. A rule you carried over to something you never tested is a Hypothesis.

`./act do` also prints, under the action it concerns, when something stands
out. Read all of its output; do not pipe it into `tail` or `head`. A long output
is cut from the middle, so its end gathers what matters: `this call, steps
A-B:` repeats the notes of the call and the colour counts that changed along
the edges (counters, timers), and the very last line, `now, after step N: ...`,
says where the game is.

- `the board is this level's first board (step S) again ..., without a
  RESET`: either you walked back to the start, or the level restarted by
  itself because you lost a life or hit something. The cells it lists as
  different from the first board are what that changed (a life counter, a used
  key): look at them.
- The whole screen turning one colour, a colour that is gone or appeared, cells
  that changed and came back (only the frames show them), and, on an
  animation worth a look, who moved in what order (`colour 9 (8 cell(s) 3x3
  at (24,27)): moved (24,27)->(30,27) (frames 0-5), gone at frame 11`) and the
  key frames (`frame_KKK.png` in that step's folder). `movers(step)` and
  `follow(step, colour)` in lib/arc.py say the same for any step: use them
  instead of writing your own frame-by-frame search. On GAME_OVER it also
  counts the actions sent since the level's start or the last RESET.
- `view: the picture slid ...`: the game shows part of a bigger level and the
  window moved; what left the board still exists. The window often follows
  the player, so the player can stay put on the board while it moves in the
  level. Board coordinates are screen coordinates; world coordinates never
  move (world (0, 0) is the top-left of the level's first view), and the line
  says how they relate now: `world = board + (0, -18)`, with an example. Keep
  positions you want to remember in world coordinates: `cam(step)` gives that
  offset, `world(x, y, step)` and `screen(wx, wy, step)` in lib/arc.py convert.
  `./act do` does not send a click that was queued after a slide in the same
  call. Everything seen on the level is stitched into `map/LNN.txt` in your
  workspace (not under `record/`), with world rulers (`.` = never seen; the
  header names the areas that changed while you watched and are out of view
  now); in Python `load_map()` gives it: `m[wx, wy]` is a cell's colour, and
  the arrays `m.grid`, `m.known`, `m.last_seen` (a step number) and `m.stale`
  are indexed `[wy - m.y0, wx - m.x0]`. Use it to find edges you have not
  explored and to plan across screens. On a slide the view lines count what in
  your state moved with the window and show the rest in your own values; and
  `diff(a, b)` and `changes(a, b)` between two steps undo the slide first, with
  positions on the second board (`d.to_world(x, y)` converts).

**Your own view: `lib/view.py`.** Once you know what this game's objects
are, write `lib/view.py` with `state(board, level) -> dict` (board: a numpy
array `b[y, x]`; level counts from 1) holding only what moves or changes, e.g.
`{{"player": (x, y), "boxes": [(x, y), ...], "door": "open"}}`, not the whole
board and not the counters (`./act do` already reports the counters). A
parser you already have is enough, since `state(board)` without the level works
too: `from board import parse as state`. From then on `./act do` prints under every
step how your state changed (`view: player [12,30]->[9,30]; boxes -[20,30]
+[17,30]`), says `your state did not change, but N cell(s) changed in ...`
when something happened that your view does not see, and shows your state on
the first board of each new level. When a level changes what the objects are,
change `state` for it. A script you run after every `./act do` to see where
things are is such a view: make it `state`.


## Your workspace

Your context may compact; files survive. Keep them organised — after a
compaction they are all you have.

- `notes.md`: what you currently believe — confirmed mechanics, hypotheses you
  have ruled out, what you are trying now. Keep it current, and when you find
  that something in `notes.md` or `levels/` is wrong, correct it there and then:
  after a compaction a wrong note misleads you again.
- `lib/`: this game's code that you reuse -- its parser (`lib/view.py`), its
  simulator, its solver -- beside `lib/arc.py`. Import it from your scripts.
  When it is wrong, fix it in place rather than saving a copy under a new name.
- `scripts/`: one-off experiments, named for what they test
  (`scripts/l3_click_order.py`).
- `levels/`: one document per cleared level. **As soon as a level clears, write
  `levels/L01.md`** (`L02.md`, ... for later ones) before playing on: the
  level's goal, the rules it relied on, the exact action sequence that cleared
  it, what did not work, and references to the key evidence — the steps,
  frames and animations that showed each rule
  (`record/steps/000123/frame_004.png`, `record/steps/000123/animation.mp4`)
  and the code that found the solution (`lib/sim.py`, `scripts/...`). Later
  levels build on these rules; write for a reader who remembers nothing.

**After a compaction** — when you find a summary of earlier work (a
`<state_snapshot>`) where the conversation used to be — read `notes.md` and every file in
`levels/` in full before your next action. The summary is short and lossy;
those files are not.

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
