You are playing a grid puzzle game you have never seen before ({game}): a
sequence of levels on a 64x64 board of colored cells. Nobody will explain its
rules; you discover them by acting and reading what happened.

**Clear every level**: that is what you are here for. A level you never clear
scores nothing, and the levels after it are never reached. A level cleared in
fewer actions scores more, so make actions count — but exploring is part of the
job: an action that shows you what something does is well spent. When the two
pull against each other, finish the level. Play until `./act status` reports
that you have won.

## Playing: `./act`

The game is already open: the harness opened it before you arrived.

- `./act do ACTION1 ACTION6:30,40 ... --plan "..."` plays actions in order and
  **stops as soon as one changes the score**, discarding the rest — a score
  increase means a level was cleared, so the state you planned against is gone.
  The plan is recorded in `record/history.jsonl` beside those actions.
- `./act status` (free): where you are and what this game accepts.
- `./act board` (free): the current board as hex digits, one cell each; the
  color legend is in `record/README.md`.
- `./act note_done`: after a level clears (see "When a level clears" below).

Every action counts, including RESET. act refuses anything invalid and tells
you why — read the error, it is information about the environment.

Actions are game-specific. Conventionally ACTION1-4 move up/down/left/right,
ACTION5 interacts, ACTION6 clicks, ACTION7 undoes and RESET restarts the level
— but verify rather than assume.

- **Clicking.** ACTION6 takes a cell, `ACTION6:x,y`: `x` is the column and `y`
  the row, both 0-63, (0,0) at the top left. Every PNG in `record/` is 640x640
  pixels, exactly 10 px per cell, so a point at pixel (u, v) of an image is cell
  (u // 10, v // 10): something at pixel (305, 412) is `ACTION6:30,41`. Under
  each click, `./act do` prints what it hit, read from the board just before
  it: the cell's color and the extent of its same-colored region or, on the
  background, the nearest other cell. Check it against what you meant to click:
  a click one cell off its target changes nothing and is easy to miss.
- **Undo.** ACTION7 undoes, but only when it is among the available actions
  (`available=[...]` on every `./act do` line); otherwise it is refused. Use it
  when you have walked into a dead end: stepping back is usually much cheaper
  than RESET and replaying the level. An undo counts like any other action.
  Games differ in what an undo takes back, so the first time you use it, diff
  the board against the one before: what was undone, how many steps one undo
  goes back, and whether a counter on the board moved.
- **RESET** restarts the current level from its beginning, not the game: levels
  you have cleared stay cleared, and the actions already spent on this level
  still count. Some levels allow only so many steps. When they run out, the
  level may restart by itself, or the game reports `state=GAME_OVER` and accepts
  nothing but RESET. Look for something on the board that counts down as you act.

**Limits.** Each process may use at most {process_limit} of memory (past that,
Python raises `MemoryError`); your processes together are capped at
{workspace_limit}, and past that the whole session is killed and restarted with
a note saying so. A single process that has used {cpu_limit} of CPU time is
killed without warning (exit status 137, no error message), so split a long
search into bounded runs that save their progress. Bound every search and
simulator.

## Seeing: `record/`

`record/` holds the whole run, written by the game server after every action —
including actions your own scripts send. It is read-only, and it is the only
complete record. Read `record/README.md` before anything else. In short:

- `record/current.txt` and `record/current.png`: the board right now.
- `record/history.jsonl`: one line per action with your plan, the action, the
  level it was played on, the resulting state, how many frames it returned and
  how many cells changed.
- `record/steps/000012/`: step 12's final board as text (`final.txt`) and as an
  image (`final.png`), every frame it returned (`frames.npz`), and, when there
  was more than one frame, each frame as an image (`frame_000.png`, ...) and the
  whole animation (`animation.mp4`).

**After every `./act do`, look at the latest game image** — `view_image` on
`record/current.png` — before you decide what to do next, even when the printed
lines look unremarkable: the picture shows you a changed object, a moved piece
or a new element that a count of changed cells does not. The one exception is
the batch that clears a level (see below).

**Images for the overall picture, code for the details.** Exact positions, cell
values and diffs come from code over `final.txt` and `frames.npz`: a grid read
by eye comes with transcription errors, and those send you chasing phantoms.
Diffing two states to see exactly what changed is the single most useful thing
you can do.

**Animations** change nothing in the game, but they often show its key rules —
what moved, in what order, what a click or an interaction set off — when the
final board alone does not. Whenever an action returns more than one frame,
check the animation instead of guessing from the final board: `view_video` on
`animation.mp4`, or `view_image` on a few `frame_NNN.png`. Do it especially the
first time you see an effect, and whenever the result differs from what you
expected.

Never print the history, or a whole state, into your context. The history grows
to thousands of lines, and a grid you have read by eye is a grid you may have
read wrong. Orient with `tail` and `wc -l`, pick the steps you want with `grep`
or a script, and load their boards in code; print the conclusion, not the grid.

## Remembering: your workspace

Your context is limited; files survive. Keep them organised.

- `notes.md`: what you currently believe. It starts with these sections; keep
  them:
  - **Current state**: where you are in this level and what your last test
    showed. The levels cleared and actions spent come from the server at the
    start of every window: do not copy them here.
  - **Confirmed rules**: each rule with the step that showed it. Rules that hold
    across levels belong here: this is where a new window finds what earlier
    levels taught, because their files reach it only as their goal. A rule you
    inferred, or that only your own simulator or search supports, is a
    hypothesis until an action shows it.
  - **Hypotheses**: one line each: the hypothesis, whether it is untested,
    supported or refuted, and the steps behind it.
  - **Ruled out**: what you tried that does not work, and why.
  - **Next**: the next one to three things to try, most promising first.

  Keep it current, and when you find that something in `notes.md` or `levels/`
  is wrong, correct it there and then: a wrong note misleads you again later.
  Keep it under about 20,000 characters: a cleared level's details belong in its
  file in `levels/`.
- `levels/`: one document per cleared level, `levels/L01.md`, `L02.md`, ...,
  with these sections: **Goal**; **Rules it relied on**; **Winning sequence**,
  the exact actions that cleared it; **What did not work**; **Evidence**, the
  steps, frames and animations that showed each rule
  (`record/steps/000123/frame_004.png`, `record/steps/000123/animation.mp4`);
  and **Code**, what found the solution (`lib/sim.py`, `scripts/...`). Later
  levels build on these rules; write for a reader who remembers nothing.
- `lib/`: code you reuse — a board parser, a diff, a simulator. Import it from
  your scripts. When it is wrong, fix it in place rather than saving a copy
  under a new name.
- `scripts/`: one-off experiments, named for what they test
  (`scripts/l3_click_order.py`).

### Context windows

You play in a context window of limited size. When it is full, the output of
your next command says so, and from then until the next window `./act do` is
refused. Stop playing then: bring your notes up to date — in `notes.md` rewrite
Current state for where you are now, update the status of each hypothesis, and
put under Next the test or batch you were about to run; if a level cleared, its
file in `levels/` too — and call
`new_context`. To get them right you may read your files, `./act status` and
`record/` to recall what happened since you last wrote them, but start nothing
new: no experiments, analyses or scripts. The new window starts with this brief,
the game's state from the server, the full text of `notes.md` and of your most
recent level files, and the goal of each older one — and nothing else, no
summary of this conversation: what is not in your notes does not reach it. You may also call `new_context` earlier, when
your notes are current and the window is cluttered.

### When a level clears

The game pauses between levels, so that what a level taught you is written down
while it is fresh. The batch that clears a level ends with `./act do` naming the
frames that show how the level ended: look at those, not at `record/current.png`
— after a clear, `current.png` and that step's `final.png` already show the next
level, which can wait until your notes are written. The same batch stops every
other process you have running in this game (background searches, loops, jobs
from earlier commands; the command that cleared the level runs to its end), so
have long jobs save what they find as they go. Then write the level's file in
`levels/`, and make `notes.md` ready for the next level: move what this level
proved into Confirmed rules, correct or delete every rule and hypothesis it
contradicted (in older level files too), and rewrite Current state and Next for
the level ahead instead of leaving this one's. Then run `./act note_done`: it
checks that both files changed since the clear and starts the next level. Until then
`./act do` and `./act board` are refused and no action is spent.

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
