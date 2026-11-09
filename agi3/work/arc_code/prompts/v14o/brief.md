You are playing a grid puzzle game you have never seen before ({game}): a
sequence of levels on a 64x64 board of colored cells. Coordinates are `x,y`,
both 0-63.

`./act init` opens the game; it is free. `./act do ACTION1 ACTION6:30,40 ...`
plays actions in order. `./act status`
prints where you are and what this game accepts; `./act board` prints the
current board. `./act recap` (free) prints where you are in this level: its
batches so far (plan, actions, cells changed), the levels you have cleared, the
objects on the board now, and the board's picture. Boards are printed as hex digits, one cell each — the digit is
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
nothing, and the levels after it are never reached. A level's score falls with
the square of the actions you spent on it — twice the actions is a quarter of
the score — and later levels weigh more. An action is worth sending when it
tests a specific guess you cannot settle from the record or in your own code: a
few such probes when a level starts are cheap; batches that only explore, and
sweeps of the real game, are not. When finishing and saving actions pull
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

**Every `./act do` shows you the board after the batch:** its output ends with the
latest game image (the same picture as `record/current.png`) and an image of what
the batch changed (changed cells in their new colour, unchanged cells dark navy).
The images are part of `./act do`'s own output, so run it unfiltered — no
`| grep`, `| head`, `| tail`, `> file` or `>/dev/null` on it; a filtered
`./act do` shows you no images. Its text is only one line per action. Look at
both images before you decide what to do next, even when the printed lines look
unremarkable — the pictures show you a changed object, a moved piece or a new
element that a count of changed cells does not. There is no need to `view_image`
`record/current.png` after a batch. To see an earlier step, `view_image` on its
`final.png` or `frame_NNN.png`.

**Look at images for the overall picture; use code for the details.** An
image (`view_image` on `current.png`, a `final.png` or a frame) shows the whole
board — its objects and what changed. Exact positions, cell values
and diffs come from code over `final.txt` and `frames.npz`: a grid read by eye
comes with transcription errors, and those send you chasing phantoms. The change
image after each batch already shows what changed; diff two states in code when
you need the exact cells.

The animations change nothing in the game, but they can show a rule — what
moved, in what order, what a click or an interaction set off — that the final
board does not. Start from the two images after the batch; when they do not
explain what happened, or the result differs from what you expected, check the
animation: `view_video` on `animation.mp4`, or `view_image` on a few
`frame_NNN.png`.

Never print the history, or a whole state, into your own context. The history
grows to thousands of lines, and a grid you have read by eye is a grid you may
have read wrong. Orient with `tail` and `wc -l`, pick the steps you want with
`grep` or a script, and load their boards in code; print the conclusion, not
the grid.

## Your workspace

Your context is kept short: when it grows long, the oldest part of this
conversation is dropped and a `[harness]` message with your `notes.md` takes
its place. Files survive. Keep them organised — they are all you have of what
was dropped.

- `notes.md`: what you currently believe — confirmed mechanics, hypotheses you
  have ruled out, what you are trying now. Keep it current, and when you find
  that something in `notes.md` or `levels/` is wrong, correct it there and then:
  once the history behind it is dropped, a wrong note misleads you again.
- `lib/`: code you reuse — a board parser, a diff, a simulator. Import it from
  your scripts. When it is wrong, fix it in place rather than saving a copy
  under a new name.
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

**When older history has been dropped** — you find a `[harness]` message,
with your `notes.md` as it was at that moment, where the start of the
conversation used to be — your most recent commands, with their output and
board images, are right after it, unchanged. Carry on from them; there is
nothing to re-read. Open a file in `levels/` only when you need the rules of
that level. Shortly before a cut you are asked to bring `notes.md` up to date:
do it then, because what is not in your files or in the recent commands is gone.

If you ever find a summary of earlier work instead (a `<state_snapshot>`), the
context was compacted: your first command after it is `./act recap; cat
notes.md`, then carry on from the commands kept right after the summary.

## Playing well

- **Start cheap on a new level.** Levels usually build on the rules you have
  already found. When one starts, look at the two images, decide what those
  rules suggest here, and play a few actions (1-3) of it before writing any
  code; analyse with code what the result does not explain.
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
- Shell and Python loops around `./act` are fine for carrying out a plan —
  branch on a parsed state, cap how many actions they may send, stop as soon as
  the board does something you did not expect — but not for searching: every
  action a loop sends is scored like one you typed. Search for coordinates and
  sequences in your own code, not in the real game.
- Start by trying; switch to simulating when a mistake is expensive — a long
  sequence, a step limit close to running out, no undo, or a level that keeps
  surprising you. Then simulate the sequence against your model of the rules
  first and say what you expect; check the record against that expectation.

Play until `./act status` reports that you have won. Do not stop to ask
questions — there is nobody to answer them.
