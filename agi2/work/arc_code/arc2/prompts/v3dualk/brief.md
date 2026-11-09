# Your task: one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid. Find the rule, then apply it to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 (a colour).

## The puzzle's data and your toolkit

`arc.py` in the workspace has every grid already parsed, plus a toolkit of exact, tested functions for what puzzles keep needing. `from arc import *` gives you all of it -- in your scripts, and in solution.py too (check.py and the final run execute solution.py in this folder, next to arc.py).

```python
from arc import *          # train: list of (input, output) numpy int arrays; test: list of input arrays
overview()                 # every training pair and test input in one summary: sizes, colours, objects, changed cells
show(g)                    # a grid with column numbers on top and row numbers on the left
diff(a, b); changes(a, b)  # where two grids differ (printed / as a list)
objects(g)                 # connected objects: o.color o.cells o.size o.bbox (r0, c0, r1, c1) o.mask o.holes o.touches ...
print_match()              # each training pair's objects paired input -> output: same, moved, recolored, turned, copy, added, deleted
groups(g); find(g, pat)    # objects of the same shape (rotations / flips / scaling allowed); every place a pattern occurs
panels(g); period(g); symmetry(g); mirror_axes(g)    # separator lines, repeating tiles, symmetries
regions(g, wall); enclosed; flood; distances; ray; slide; move; gravity; paint; paste; orient; upscale; ...
sweep(rule, k=[1, 2, 3])   # a rule's variants on every training pair: which pass, and whether they agree on the test inputs
preview()                  # what solution.py outputs for every test input
render(grid, "out.png")    # save a grid as a PNG, then look at it with view_image
doc(); doc("find")         # the list of every tool; one tool's reference with examples (all in TOOLS.md, whose Caveats section lists the traps)
```

Use these instead of writing your own connected components, flood fill, rotations or bounding boxes: they handle 4/8-connectivity, the background colour, all 8 orientations and the bbox order the same way everywhere, and hand-written copies of exactly these are where most bugs in puzzle code come from (a missing flip, an ignored connectivity, a swapped bbox order). Get positions, counts, sizes and shapes from them, not by counting digits along a printed row: a position read by eye is often off by one, and every recount costs tokens. The background is each grid's most frequent colour unless you pass `bg=`; pass it when that colour is not the background, or when a test input may differ.

The same grids are in `task/` (read-only): `task.json`, one `.txt` file per grid (one row per line, one digit per cell), and the `.png` images that were attached to your first message in this order: {image_order}. Printing `task.json` costs about three times as many tokens as `show()`; don't print it.

Grid sizes (rows x columns): {sizes}.

## What you deliver

`solution.py` in the workspace, defining `transform(grid)`: it takes a grid (a list of lists of ints) and returns the output grid. **When your session ends -- because you finished, or because the budget ran out -- the harness runs `transform` on every test input, and that output is your answer.** So create solution.py early and keep it holding your best current version at all times.

`python3 check.py` runs solution.py on every training pair and prints PASS or which cells are wrong, plus the output size for each test input. Its result is the only evidence that counts -- not your memory of an earlier run.

If, for a test input, you have an output that you produced and checked with code but that `transform` does not produce, save it to `answer/test_J.json` as a JSON list of rows; it replaces transform's output for that test input. Never type out a grid from memory.

You may also deliver a second answer: `solution2.py`, a self-contained file with its own `transform(grid)` (copy what it needs; do not import solution.py). Each test input counts as solved if either answer is exactly right, and the second answer is used only where its output differs from the first. A solution2.py that does not pass every training pair (`python3 check.py solution2.py`) is ignored.

## Method

Keep your state in files, not in your head. As soon as you can state a hypothesis as code, put it in solution.py and run check.py: a mismatch shows exactly where the hypothesis is wrong. Keep `notes.md` as a short ledger: each time you verify a fact or a hypothesis fails, add one line saying what and where (which pair, which cells). If your context is compacted, notes.md, solution.py and your scripts are what you keep. Look at what transform does to every test input too: the same rule has to give a sensible output for each of them, and a test input your rule cannot handle usually means the rule is incomplete, not that the test is special.

## Budget

You have about {budget_k}k output tokens for this puzzle in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning, and solution.py is taken as it is. Spend the budget on testing hypotheses in code rather than on simulating grids in your head.

## Finishing

Passing every training pair is necessary, not sufficient: a rule that fits the training pairs can still be wrong on a test input, usually because the test input contains a case the training pairs never show (a new colour, a different size or count, objects touching or overlapping, an edge case of the rule). So when transform passes every training pair, look at its output for every test input before you stop -- `preview()` prints them, `render()` saves one to view -- and check, against the input, that every part of it follows the rule you found, the way the training outputs do. If something looks off, fix the rule and run check.py again.

You can deliver two answers (see What you deliver). If you weighed another reading that also passes every training pair and gives a different output on some test input, do not choose between the two: keep the one you find more likely in solution.py and put the other in solution2.py. Write solution2.py only for a reading you actually weighed against the training pairs, not just to fill the second answer.

When transform passes every training pair and its output for every test input holds up to that check, stop: reply with one short line and no tool call. That ends the session.
