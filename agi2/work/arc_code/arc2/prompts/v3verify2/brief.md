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

## Your job: find out whether the first answer is wrong, and give a second answer if it may be

**Passing every training pair does not make the test output right.** The previous solution already passes the training pairs in almost every puzzle, yet its test output is wrong in about one puzzle in six. When it is wrong, it is almost never a bug in the training pairs. It is one of these:

- a reading that fits every training pair but is not the intended rule, so it diverges on the test input;
- the test input has something the training pairs never show (a new colour, size, count, shape, contact, overlap, orientation, an object at the border, an empty or ambiguous case), and the code handles it by an arbitrary choice.

So the review that matters is on the test inputs. Re-implementing the stated rule and checking it on the training pairs again proves nothing new: it only shows the code matches its own reading, not that the reading is right on the test. Do not spend your budget there. Rerun `python3 check.py previous/solution.py` once to confirm it passes train, then go to the test inputs.

Work like a reviewer whose job is to find the error:

1. Read REPORT.md. Write down, in `notes.md`, at least two other readings of the training pairs that explain every training output as well as the chosen one does -- different ways to pick objects, colours, order, counts, anchors, directions, sizes or tie-breaks. Use the doubts and rejected readings in REPORT.md, and add your own.
2. For each test input, look at it next to its previous output (`render()` both and view them, and `show()` them). List what this test input has that no training input has. For each item, say what the training pairs actually show about it -- often nothing -- and whether the previous output's choice is forced by the training pairs or just one option.
3. Work out what each other reading from step 1 would output on each test input (with code, not by hand). Where it differs from the previous output, decide honestly which one is more plausible, given the training pairs and how ARC puzzles are usually designed: rules are simple, general and symmetric; outputs look complete and deliberate, not clipped, crowded or lopsided.
4. Look at the previous test output as a picture, the way you would look at a training output. Does it look like the training outputs do -- same kind of structure, nothing half-done, no stray, missing or overlapping cells, nothing that runs into the border or another object in a way the training outputs never do? Anything that looks odd is a sign the reading is wrong on this input.

Then decide, and write two files in the workspace.

`REVIEW.md` -- your review of the previous solution. Another solver may read it to build the final answers, so write it for someone who has not seen your work:

- **Verdict**: CONFIRMED or ALTERNATIVE, and the main reason in a line or two.
- **Test inputs**: for each one, what it has that no training input has; whether the previous output is forced by the training pairs or one choice among several; and how confident you are that the previous output is right (high / medium / low), with the reason.
- **Other readings**: each reading you found, which training pairs it explains, what it outputs on each test input compared with the previous output (the file with its code or output), and how plausible it is.
- **Second answer**: if you give one, what it changes and why it is the best alternative.
- **Open doubts**: anything you could not settle.

`VERDICT.md` -- the decision, first line exactly one word:

- **`CONFIRMED`** only when every test output is forced: every other reading you found either fails a training pair or gives the same test output, and nothing in the test inputs is left to an arbitrary choice. "The output follows the stated rule" is not a reason; the question is whether the stated rule is the right one for this input.
- **`ALTERNATIVE`** in every other case -- a bug, a rule that does not fully explain the training pairs, a test input handled by an arbitrary choice, or another reading that is about as plausible and gives a different test output. When you are unsure, write ALTERNATIVE: a second answer costs nothing if the first is right, and a wrong CONFIRMED wastes the second attempt. Then build the alternative answer -- the most plausible output that differs from the previous one: `solution.py` in the workspace with `transform(grid)` that passes every training pair (start from `cp previous/solution.py solution.py` if that helps) and gives a different output on at least one test input. If, for a test input, you have an output you produced and checked with code but that transform does not produce, save it to `answer/test_J.json` as a JSON list of rows; it replaces transform's output for that test input. Never type out a grid from memory. `python3 check.py` runs your solution.py and tells you for each test input whether your output is the SAME AS or differs from the previous one. When your session ends, your transform's output (or answer file) is the second answer.

If the previous solution does not pass every training pair, it cannot be confirmed: write ALTERNATIVE and solve the puzzle, using its report and code as a starting point but ready to drop its reading.

## Budget

You have about {budget_k}k output tokens in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning: write VERDICT.md and REVIEW.md as soon as you have a verdict (update REVIEW.md as you learn more), and keep solution.py holding your best version.

## Finishing

Stop -- reply with one short line and no tool call -- when REVIEW.md is complete and VERDICT.md says CONFIRMED, or when it says ALTERNATIVE and your transform passes every training pair and differs from the previous output on at least one test input.
