# Your task: one ARC puzzle

The puzzle has {n_train} training pairs. In each pair a single hidden rule turns the input grid into the output grid. Find the rule, then apply it to each of the {n_test} test input(s). A grid is a rectangle of cells, each cell an integer 0-9 (a colour).

## The puzzle's data and your toolkit

`arc.py` in the workspace has every grid already parsed and a toolkit of exact, tested functions for what puzzles keep needing. `from arc import *` gives you all of it, in your scripts and in solution.py. This card lists the tools puzzles use most; `doc()` lists every tool, and `doc("name")` prints one tool's full reference with examples.

```
grids      train = [(input, output), ...] and test = [input, ...], numpy int arrays
look       show(g): the grid with row / column numbers     diff(a, b): the differences, printed
           render(g, "x.png"), then view_image             overview(): every pair summarised in one print
cells      changes(a, b) -> [(r, c, old, new)] (printed as a summary when long)    where(g, colour) -> [(r, c)]
           (a list of more than 8 cells prints row by row: "r1 c2-6 | r2 c1" = row 1 columns 2 to 6, row 2 column 1)
           count(g) -> {{colour: n}}    background(g) -> the most frequent colour
objects    objects(g, bg=None, conn=8, multicolor=False, gap=0) -> [Obj];  print_objects(g) prints them, with masks
           bg: one colour, a list ([0, 5]: both are background) or -1 (every colour is an object); None: the most frequent
           multicolor=True: touching cells of different colours form one object
           gap=1: cells with one empty cell between them join too (dashed lines, dotted outlines)
           by_color(g) -> {{colour: Obj of all its cells}};  the cells of one colour as pieces: components(g == 3)
  Obj      .idx .color .colors {{colour: n}} .size .cells [(r, c)] .rel (the cells from the bbox corner) .h .w
           .bbox (r0, c0, r1, c1, inclusive) .mask (bool, bbox-sized) .grid() .holes .hole_cells .touches .is_rect
pieces     components(mask_or_cells, conn=8, gap=0) -> [cells]    adjacency(objs) -> {{idx: [idx it touches]}}
           gap(a, b) -> .rows .cols (empty lines between) .cheb .manh    cluster(objs, lambda a, b: gap(a, b).cheb <= 2)
           runs(row) -> [(colour, start, length)]
in -> out  print_match(): each pair's objects, input -> output: same / moved by (dr, dc) / recolored / turned /
           copy / added / deleted (pass multicolor=True when objects have several colours)
           explain_objects(): every input object's fate in every pair, and the one-property rules that give each fate
           ("size == 1", "touching colour 2") with what they pick in the test inputs -- and where those rules disagree
shapes     orient(g, "rot90"): the 8 orientations are ORIENTS (rot90 = counter-clockwise)
           shape_match(a, b) -> ["rot90", ...]    shape_key(o): equal for equal shapes    groups(g): objects by shape
           find(g, pattern, flips=False) -> [Hit .r .c .orient .cells] (-1 in the pattern = any colour)
           align(a, b) -> [Fit .orient .r .c]: how shape a lays exactly onto b; fit.apply(cells) = where a's cells go
strokes    segments(x): the straight runs a shape is drawn with    trace(x): a line's ends, branches, cells in order
           aligned(items): pairs of cells / objects on one row, column or diagonal
layout     panels(g) -> [Panel .grid .bbox], cut along full lines (panels(g, color=c) picks the line colour;
           panels(g, color=c, min_frac=0.8) also cuts on lines that marks interrupt)    lines(g): rows / columns mostly of one colour
           tiles(g, nr, nc)    squeeze(g): repeated rows / columns merged    period(g) -> .pr .pc .tile .wrong
           symmetry(g, ignore=c) -> {{"rot90": cells off, ...}}    mirror_axes(g, ignore=c)
           explain_panels(): which AND / OR / XOR / stacking of the panels gives the output
regions    enclosed(g, wall=c) -> [Obj] that the wall closes in    flood(g, (r, c), through=[..] or wall=[..]) -> cells
walking    ray(g, (r, c), "right") -> .cells .hit    line(p, q) -> cells (straight or 45 degrees)
           slide(g, obj, "down") -> free steps (.hit_colors: what stops it)
drawing    g = paint(g, cells, colour)   paste(g, patch, r, c)   move(g, obj, dr, dc)   gravity(g, "down")
           each returns a new grid (a bare paint(g, ...) changes nothing); off the grid raises unless clip=True
packing    pack(region, pieces, rotate=False, flip=False, cover=True): every way to place the pieces (Obj keep their
           colours) in the region (a mask, cells or a Box); print() of the result says how many -- more than one is a tie
testing    verify(rule): runs rule(grid) on every training pair -- PASS, or the error / size / wrong cells -- and on
           the test inputs; nothing is written, so any guess can be checked before it goes into solution.py
           cellwise(f): the transform that sets each cell (r, c) to f(g, r, c), for cell-by-cell rules
           check_fate(pred, "deleted"): is pred(o) True exactly for the objects that get that fate, in every pair?
           per_pair(f): f(input, output) for every pair on one line each -- counts, sizes, positions, formulas
           sweep(rule, k=[1, 2], mode=["a", "b"]): which settings of a rule pass, and where the passing ones differ
           on the test inputs; rival rules: sweep(lambda g, f: f(g), f=[rule_a, rule_b])
           unseen(): what the test inputs have that no training input has
before     preview(): what solution.py outputs for every test input
```

Before you write code for one of these jobs, call the tool: labelling cells or a flood fill -> objects / components / flood; rotating, flipping or comparing shapes -> orient / shape_match / shape_key; cutting along lines -> panels / lines; moving until blocked -> slide / gravity / ray; distances between objects, or grouping them -> gap / cluster; placing or tiling pieces -> pack; how one shape maps onto another -> align; checking a guess on every pair -> verify / check_fate / per_pair, not pair by pair in your head; two readings that both pass -> sweep. Hand-written copies of exactly these are where most bugs in puzzle code come from; read positions and counts from the tools, not by eye.

Conventions: (r, c) = (row, column), counted from 0; objects, components, groups and match join diagonal neighbours (conn=8), regions, enclosed, flood and adjacency do not (conn=4). The background is decided per grid (its most frequent colour) unless you pass `bg=`: pass it when a test input or an output may differ.

Workspace facts: run your scripts from this folder, and do not name a file like a Python module (`enum.py`, `types.py`); tool output above about 2,500 tokens is cut in the middle, so print one grid or one pair at a time; the sandbox refuses `rm`, and with it the whole command it is in.

The same grids are in `task/` (read-only): `task.json`, one `.txt` file per grid (one row per line, one digit per cell), and the `.png` images that were attached to your first message in this order: {image_order}. Printing `task.json` costs about three times as many tokens as `show()`; don't print it.

Grid sizes (rows x columns): {sizes}.

## What you deliver

`solution.py` in the workspace, defining `transform(grid)`: it takes a grid (a list of lists of ints) and returns the output grid. It may start with `from arc import *` and use every tool: check.py and the final run execute it in this folder, next to arc.py, exactly as your scripts run. **When your session ends -- because you finished, or because the budget ran out -- the harness runs `transform` on every test input, and that output is your answer.** So create solution.py early and keep it holding your best current version at all times.

`python3 check.py` runs solution.py on every training pair and prints PASS or which cells are wrong, plus the output size for each test input. Its result is the only evidence that counts -- not your memory of an earlier run.

If, for a test input, you have an output that you produced and checked with code but that `transform` does not produce, save it to `answer/test_J.json` as a JSON list of rows; it replaces transform's output for that test input. Never type out a grid from memory.

You may also deliver a second answer: `solution2.py`, a self-contained file with its own `transform(grid)` (copy what it needs; do not import solution.py). Each test input counts as solved if either answer is exactly right, and the second answer is used only where its output differs from the first. A solution2.py that does not pass every training pair (`python3 check.py solution2.py`) is ignored.

## Method

Keep your state in files, not in your head. As soon as you can state a hypothesis as code, check it with `verify(rule)` (or put it in solution.py and run check.py): a mismatch shows exactly where the hypothesis is wrong. Keep `notes.md` as a short ledger: each time you verify a fact or a hypothesis fails, add one line saying what and where (which pair, which cells). If your context is compacted, notes.md, solution.py and your scripts are what you keep. Look at what transform does to every test input too: the same rule has to give a sensible output for each of them, and a test input your rule cannot handle usually means the rule is incomplete, not that the test is special.

## Budget

You have about {budget_k}k output tokens for this puzzle in total, your reasoning included; `cat budget.txt` shows how much is used. When it runs out you are stopped mid-step without warning, and solution.py is taken as it is. Spend the budget on testing hypotheses in code rather than on simulating grids in your head.

## Finishing

Passing every training pair is necessary, not sufficient: a rule that fits the training pairs can still be wrong on a test input, usually because the test input contains a case the training pairs never show (a new colour, a different size or count, objects touching or overlapping, an edge case of the rule). So when transform passes every training pair, look at its output for every test input before you stop -- `preview()` prints them, `render()` saves one to view -- and check, against the input, that every part of it follows the rule you found, the way the training outputs do. If something looks off, fix the rule and run check.py again.

You can deliver two answers (see What you deliver). If you weighed another reading that also passes every training pair and gives a different output on some test input, do not choose between the two: keep the one you find more likely in solution.py and put the other in solution2.py. Write solution2.py only for a reading you actually weighed against the training pairs, not just to fill the second answer.

When transform passes every training pair and its output for every test input holds up to that check, stop: reply with one short line and no tool call. That ends the session.
