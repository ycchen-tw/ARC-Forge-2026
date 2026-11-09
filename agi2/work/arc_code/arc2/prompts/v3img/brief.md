# Your task: one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid. Find the rule, then apply it to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 shown as a colour: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, 9 maroon.

## The puzzle's data: images only

You see this puzzle only as images. They were attached to your first message; `cat task/IMAGES.md` shows all of them again (each image costs context, so do that only when you need to look again, for example after your conversation was reset). There is no file with the cell values, and no way to get them as text: read the grids off the images. Each image draws one cell as a solid square with thin dark grid lines, on a white margin.

Grid sizes (rows x columns): {sizes}.

## What you deliver

`solution.py` in the workspace, defining `transform(grid)`: it takes a grid (a list of rows, each a list of ints 0-9) and returns the output grid. solution.py must be self-contained: it runs in a separate sandbox with Python's standard library and numpy, without the workspace's other files. **When your session ends -- because you finished, or because the budget ran out -- the harness runs `transform` on every test input, and that output is your answer.** So create solution.py early and keep it holding your best current version at all times.

`python3 check.py` runs solution.py in the sandbox on every training input and every test input, and reports:
- for each training pair, PASS -- or what is wrong: the output has the wrong size, or how many cells are wrong; with an image of your output in which every wrong cell is outlined in white;
- for each test input, an image of transform's output.

Its result is the only evidence that counts -- not your memory of an earlier run. You can run check.py at most {check_limit} times for this puzzle; after that it no longer runs, so make each run test a real hypothesis about the rule.

Inside the sandbox, anything transform prints is discarded and digits in error messages are replaced by `#`; you never see cell values as text. Do not try to recover cell values some other way: no grids written out by hand inside solution.py or other files to probe check.py, and no using check.py's counts of wrong cells to search for or reconstruct grids. transform has to compute each output from its input by the rule you found.

## Method

Keep your state in files, not in your head. As soon as you can state a hypothesis as code, put it in solution.py and run check.py: the outlined cells show exactly where the hypothesis is wrong. Keep `notes.md` as a short ledger: each time you verify a fact or a hypothesis fails, add one line saying what and where (which pair, which part of the grid). If your context is compacted or reset, notes.md and solution.py are what you keep. Look at transform's output for every test input too: the same rule has to give a sensible output for each of them, and a test input your rule cannot handle usually means the rule is incomplete, not that the test is special.

## Budget

You have about {budget_k}k output tokens for this puzzle in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning, and solution.py is taken as it is. Spend the budget on testing hypotheses in code rather than on simulating grids in your head.

## Finishing

Passing every training pair is necessary, not sufficient: a rule that fits the training pairs can still be wrong on a test input, usually because the test input contains a case the training pairs never show (a new colour, a different size or count, objects touching or overlapping, an edge case of the rule). So when transform passes every training pair, look at check.py's image of its output for every test input before you stop, and check, against the test input's image, that every part of it follows the rule you found, the way the training outputs do. If something looks off, fix the rule and run check.py again.

When transform passes every training pair and its output for every test input holds up to that check, stop: reply with one short line and no tool call. That ends the session.
