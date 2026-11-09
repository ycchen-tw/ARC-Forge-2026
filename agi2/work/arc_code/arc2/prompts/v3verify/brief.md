# Your task: verify a solution to one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid; the rule has to be applied to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 (a colour).

## The puzzle's data

`arc.py` in the workspace has every grid already parsed -- use it from Python instead of printing JSON:

```python
from arc import train, test, show, diff, render
# train: list of (input, output) numpy int arrays; test: list of input arrays
show(train[0][0])          # print a grid, one row per line, one digit per cell
diff(a, b)                 # where two grids differ
render(grid, "out.png")    # save a grid as a PNG, then look at it with view_image
```

The same grids are in `task/` (read-only): `task.json`, one `.txt` file per grid (one row per line, one digit per cell), and the `.png` images that were attached to your first message in this order: {image_order}. Printing `task.json` costs about three times as many tokens as `show()`; don't print it.

Grid sizes (rows x columns): {sizes}.

## What you inherit

Another solver has already solved this puzzle as well as it could. Its work is in `previous/` (read-only) and is yours to build on:

- `REPORT.md` -- its hand-off report: the rule, how the code implements it, the evidence per training pair, what happens on each test input, its doubts and the readings it rejected. Start here.
- `solution.py` (and any helper scripts), `notes.md` -- its code and working ledger.
- `check.txt` -- what check.py reported for its solution: <<PREVIOUS>>
- `test_J_output.txt` / `test_J_output.json` -- the output it submitted for each test input. That output is the first answer.

Two answers can be submitted per test input, and a test input counts as solved if either is exactly right. The first is the previous solver's. A second one is only worth giving if the first may be wrong.

## Your job: verify, and give a second answer only if you find a problem

Check the previous solution the way a careful reviewer would -- do not redo the puzzle from scratch, and do not just rerun its code:

1. Read REPORT.md and compare its rule with every training pair yourself: does the stated rule explain each output completely, including the cells the code gets right "by accident"? `python3 check.py previous/solution.py` re-runs its code on the training pairs.
2. Check that solution.py implements the stated rule, not something narrower that happens to fit the training pairs.
3. For each test input, look at the previous output next to the input (`show()` or `render()` both and view them): does every part of it follow the rule? Does the test input contain something the training pairs never show, and is it handled the way the rule implies?
4. Look at the doubts and rejected readings in REPORT.md: is any of them at least as plausible as the chosen one on some test input?

Then decide, and write `VERDICT.md` in the workspace:

- **`CONFIRMED`** (first line exactly that) when you found no problem: the rule explains every training pair, the code implements it, and every test output follows it. Below it, a few lines on what you checked. Then stop -- no second answer is needed, and do not spend budget looking for one.
- **`ALTERNATIVE`** (first line exactly that) when you found a problem -- a bug, a rule that does not fully explain the training pairs, a test input handled against the rule, or a competing reading that is at least as plausible on a test input. Below it, what the problem is. Then build the corrected or alternative answer: `solution.py` in the workspace with `transform(grid)` that passes every training pair (start from `cp previous/solution.py solution.py` if that helps) and gives a different output on at least one test input. If, for a test input, you have an output you produced and checked with code but that transform does not produce, save it to `answer/test_J.json` as a JSON list of rows; it replaces transform's output for that test input. Never type out a grid from memory. `python3 check.py` runs your solution.py and tells you for each test input whether your output is the SAME AS or differs from the previous one. When your session ends, your transform's output (or answer file) is the second answer.

If the previous solution does not pass every training pair, it cannot be confirmed: write ALTERNATIVE and solve the puzzle, using its report and code as a starting point but ready to drop its reading.

## Budget

You have about {budget_k}k output tokens in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning: write VERDICT.md as soon as you have a verdict, and keep solution.py holding your best version.

## Finishing

Stop -- reply with one short line and no tool call -- when VERDICT.md says CONFIRMED, or when it says ALTERNATIVE and your transform passes every training pair and differs from the previous output on at least one test input.
