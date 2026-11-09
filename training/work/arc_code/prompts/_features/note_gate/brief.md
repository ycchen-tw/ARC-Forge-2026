## When a level clears

The game pauses between levels, so that what a level taught you is written down
while it is fresh. The batch that clears a level ends with `./act do` naming the
frames that show how the level ended: look at those, not at `record/current.png`
— after a clear, `current.png` and that step's `final.png` already show the next
level, which can wait until your notes are written. The same batch stops every
other process you have running in this game (background searches, loops, jobs
from earlier commands; the command that cleared the level runs to its end), so
have long jobs save what they find as they go. Then write the level's file in
`levels/`, bring `notes.md` up to date, and run `./act note_done`: it checks
that both files changed since the clear and starts the next level. Until then
`./act do` and `./act board` are refused and no action is spent.
