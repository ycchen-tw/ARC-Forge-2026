# Your task: one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid. Find the rule, then apply it to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 (a colour).

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

## What you deliver

`solution.py` in the workspace, defining `transform(grid)`: it takes a grid (a list of lists of ints) and returns the output grid. **When your session ends -- because you finished, or because the budget ran out -- the harness runs `transform` on every test input, and that output is your answer.** So create solution.py early and keep it holding your best current version at all times.

`python3 check.py` runs solution.py on every training pair and prints PASS or which cells are wrong, plus the output size for each test input. Its result is the only evidence that counts -- not your memory of an earlier run.

If, for a test input, you have an output that you produced and checked with code but that `transform` does not produce, save it to `answer/test_J.json` as a JSON list of rows; it replaces transform's output for that test input. Never type out a grid from memory.

## Method

Keep your state in files, not in your head. As soon as you can state a hypothesis as code, put it in solution.py and run check.py: a mismatch shows exactly where the hypothesis is wrong. Keep `notes.md` as a short ledger: each time you verify a fact or a hypothesis fails, add one line saying what and where (which pair, which cells). If your context is compacted, notes.md, solution.py and your scripts are what you keep. Look at what transform does to every test input too: the same rule has to give a sensible output for each of them, and a test input your rule cannot handle usually means the rule is incomplete, not that the test is special.

## Budget

You have about {budget_k}k output tokens for this puzzle in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning, and solution.py is taken as it is. Spend the budget on testing hypotheses in code rather than on simulating grids in your head.

## Finishing

**Passing every training pair does not make the test output right.** Solutions that pass every training pair are still wrong on a test input in about one puzzle in six -- almost never because of a bug, but because the rule that fits the training pairs is not the intended one, or because the test input contains something the training pairs never show (a new colour, size, count or shape, objects touching, overlapping or at the border, an empty or ambiguous case) and the code handles it by an arbitrary choice. Looking at the test output and seeing that it follows your own rule does not catch either. So once transform passes every training pair, before you stop:

1. For each test input, list in notes.md what it has that no training input has, and for each item whether the training pairs actually force how your rule handles it or your code just picks one way.
2. Write down at least two other readings of the training pairs that explain every training output as well as yours does -- different ways to pick objects, colours, order, counts, anchors, directions, sizes or tie-breaks -- and compute, with code, what each gives on each test input.
3. Where another reading gives a different test output, decide which is more plausible given the training pairs and how ARC puzzles are designed: rules are simple, general and symmetric; outputs look complete and deliberate, not clipped, crowded or lopsided. `render()` the candidates and view them. Keep the most plausible one in solution.py (or `answer/test_J.json`), and note the runner-up in notes.md.

When transform passes every training pair and you have done these checks for every test input, stop: reply with one short line and no tool call. That ends the session.
