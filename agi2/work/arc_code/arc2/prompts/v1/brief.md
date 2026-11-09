# Your task: one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid. Find the rule, then apply it to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 (a colour).

## Files in task/ (read-only)

- `task/task.json`: `{{"train": [{{"input": grid, "output": grid}}, ...], "test": [{{"input": grid}}, ...]}}`; a grid is a list of rows, each a list of ints.
- `task/train_K_input.txt`, `task/train_K_output.txt`, `task/test_J_input.txt` (K and J count from 1): the same grids, one row per line, one digit per cell, no separators.
- `task/*.png`: the same grids as images. They were attached to your first message in this order: {image_order}.

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

When transform passes every training pair and gives a sensible output for every test input, stop: reply with one short line and no tool call. That ends the session.
