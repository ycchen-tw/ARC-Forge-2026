# arc.py toolkit reference

## Which tool

`from arc import *` (in solution.py too). Every grid argument also takes a list of lists or a bool mask.

    look       show(g)  diff(a, b)  render(g, "x.png")  overview()  changes(a, b)  where(g, c)  count(g)  bbox_of(g)
    objects    objects(g, bg=None, conn=8, multicolor=False, gap=0) -> [Obj]  print_objects(g)  by_color(g)
               Obj: .color .colors .size .cells .rel .bbox .mask .grid() .holes .hole_cells .touches .is_rect
    pieces     components(mask_or_cells, gap=0)  cluster(items, linked)  gap(a, b)  adjacency(objs)  aligned(items)
    in -> out  print_match()  explain_objects()  explain_select()  explain_panels()  color_map(a, b)
    shapes     orient(g, "rot90")  shape_match(a, b)  align(a, b)  shape_key(x)  groups(g)  find(g, pattern)
    strokes    segments(x)  trace(x)  line(p, q)  outline(box)  halo(obj)  ray(g, (r, c), "right")  slide(g, o, "down")
    layout     panels(g, color=None, min_frac=1.0)  lines(g)  tiles(g, nr, nc)  squeeze(g)  period(g)  symmetry(g)
               mirror_axes(g)  fill_by_symmetry(g, hole_colour, ["fliplr"])  upscale  downscale
    regions    enclosed(g, wall=c)  regions(g, wall)  flood(g, start, wall=[..])  distances(g, start)
    drawing    paint(g, cells, c)  paste(g, patch, r, c)  move(g, obj, dr, dc)  gravity(g, "down"): new grids
    packing    pack(region, pieces, rotate=False, flip=False, cover=True): every way the pieces fill the region
    testing    verify(rule)  cellwise(f)  check_fate(pred, "deleted")  per_pair(f)  sweep(rule, k=[1, 2])  unseen()
    before     preview()  what solution.py outputs for every test input (and returns them)

## About this file

`from arc import *` gives you the puzzle's grids and every tool below. **solution.py may import from arc too**
(`from arc import *` at its top): check.py and the final run both execute solution.py in this folder, next to arc.py.

`doc()` prints the table above and the list of tools; `doc("find")` prints one section of this file (from the
shell: `python3 arc.py doc find`); `doc("caveats")` the traps. `grep -n "### " TOOLS.md` lists them too.

A solution.py built on the toolkit:

    from arc import *

    def transform(grid):
        g = np.array(grid)                      # np is exported too
        bg = background(g)
        for o in objects(g, bg=bg):
            if o.holes:
                g = paint(g, o.hole_cells, o.color)
        return g                                # check.py accepts arrays and lists

The tools are tested (all 8 orientations, both connectivities, bbox order, edge cases); reusing them is safer than a
fresh copy, and a bug in a hand-written copy (a missing flip, an ignored conn, a swapped bbox order) is the most common
mistake in hand-written helpers.

## Conventions

- **grid**: a 2-D numpy int array. Every tool also accepts a list of lists. Tools never change their input; the ones
  that build a grid return a new array (`transform` may return it as is -- check.py converts arrays).
- **position** `(r, c)` = (row, column), counted from 0 at the top-left (Python or numpy ints; numbers print as plain
  ints, `5` not `np.int64(5)`). **bbox** = `Box(r0, c0, r1, c1)`, inclusive,
  rows first: a tuple that also has `.r0 .c0 .r1 .c1 .h .w`, `.rows` / `.cols` (ranges) and `.slices` (`g[b.slices]` is
  the crop). Write `b.h`, not `b[2] - b[0] + 1`; never unpack a bbox as `(r0, r1, c0, c1)`.
- **bg** (background): default = the grid's most frequent colour, decided per grid (overview and print_objects say
  when that guess is a close call). Pass `bg=` when that is wrong
  (a grid mostly covered by one big shape) and inside solution.py whenever a test input could have a different
  most-frequent colour.
- **conn**: 8 (default for objects: diagonal neighbours join one object) or 4. Region / flood tools default to 4
  (a diagonal gap does not leak).
- **multicolor**: False (default) = an object is one colour; True = touching non-bg cells of any colours form one object.
- **-1** means "no cell" / "any value": in `find` patterns, `paste` patches, `Obj.grid(-1)` and shape keys.
- **bool masks**: every tool that takes a grid also takes a bool array (True = 1, False = 0; its bg is False), e.g.
  `objects(g == 3)`, `symmetry(o.mask)`, `bbox_of(mask)`.
- `show`, `diff`, `overview`, `doc` and the `print_*` tools print and return nothing; `preview`, `explain_select`
  and `sweep` print and also return their results; all other tools return values and print nothing.
- `train` / `test` arrays are read-only: `g = np.array(test[0])` (or `.copy()`) before writing into one.
- Results are exact (computed from the grid). If a result does not fit what you see, change a setting (bg, conn,
  multicolor) instead of counting by hand.

## Grids

### train, test
The puzzle: `train` = list of (input, output) array pairs, `test` = list of test input arrays.

### Box, Cells
`Box(r0, c0, r1, c1)`: a bbox, inclusive, rows first; a tuple with `.r0 .c0 .r1 .c1 .h .w`, `.rows` `.cols` (ranges)
and `.slices` (so `g[b.slices]` is the crop). `Cells`: a list of `(r, c)`. Up to 8 cells print as a list; more
print row by row as column runs -- `r1 c2-6` is row 1, columns 2 to 6 -- which is shorter and easier to read than the
pairs; the list itself is complete (`list(cells)` prints every pair, `.tolist()` gives a plain list).

    >>> b = bbox_of([[0, 0, 0], [0, 3, 3]])
    >>> b, b.h, b.w, tuple(b), list(b.cols)
    ((r0=1, c0=1, r1=1, c1=2), 1, 2, (1, 1, 1, 2), [1, 2])
    >>> where([[0, 3, 3, 3, 0], [3, 0, 0, 0, 3], [3, 0, 0, 0, 3], [0, 3, 3, 3, 0]], 3)
    10 cells: r0 c1-3 | r1 c0,4 | r2 c0,4 | r3 c1-3

### show(g, coords=True, dot=None, box=None)
Print a grid with column numbers on top and row numbers on the left.

    >>> show([[0, 1, 0], [2, 2, 0]])
        012
      0 010
      1 220
    >>> show([[0, 1, 0], [2, 2, 0], [0, 0, 3]], dot=0, box=(1, 1, 2, 2))
        12
      1 2.
      2 .3

Grids wider than 10 get a second header row with the tens digit. `dot` = colour(s) printed as '.' (e.g. the bg);
`box` = print only that bbox, keeping the grid's row / column numbers. `coords=False` prints bare digits. A bool mask
prints as 1 / 0.

### diff(a, b, limit=40)
Print where two grids of the same size differ; above `limit` cells, grouped into clusters of touching cells.

    >>> diff([[1, 2], [3, 4]], [[1, 0], [3, 5]])
    2 of 4 cells differ
      (0,1): 2 vs 0
      (1,1): 4 vs 5

### changes(a, b)
`[(r, c, colour in a, colour in b)]` for every cell where two same-size grids differ (raises if the sizes differ).
Printing more than 20 changes gives a summary instead of the tuples: how many cells change from which colour to which,
and the clusters of touching changed cells with their rows and columns (the list itself is complete).

    >>> changes([[1, 2], [3, 4]], [[1, 0], [3, 5]])
    [(0, 1, 2, 0), (1, 1, 4, 5)]
    >>> print(changes(np.zeros((6, 6), int), np.pad(np.full((5, 5), 2), ((1, 0), (1, 0)))))
    25 changed cells: 0->2 x25
      rows 1-5 cols 1-5: 25 cells 0->2 x25

### render(g, path)
Save a grid as a PNG (same colours and cell size as the puzzle's own task/*.png); `path` is relative to the workspace,
e.g. `render(out, "out.png")`, then look at it with view_image.

### count(g)
`{colour: number of cells}`.

    >>> count([[0, 1, 1], [0, 0, 2]])
    {0: 3, 1: 2, 2: 1}

### background(g)
The most frequent colour (the smallest one on ties): what `bg=None` means in every tool.

    >>> background([[7, 7, 1], [7, 2, 2]])
    7

### where(g, color)
The `(r, c)` cells of one colour, in reading order (a Cells list: more than 8 print row by row, see Box, Cells).

    >>> where([[0, 1, 1], [0, 0, 1]], 1)
    [(0, 1), (0, 2), (1, 2)]

### crop(g, bbox)
The sub-grid inside a bbox (inclusive); `bbox` may also be an Obj, Panel or Hit (their bbox).

    >>> crop([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (1, 1, 2, 2)).tolist()
    [[5, 6], [8, 9]]

### bbox_of(g, bg=None)
The Box around all non-bg cells.

    >>> bbox_of([[0, 0, 0], [0, 3, 0], [0, 0, 4]])
    (r0=1, c0=1, r1=2, c1=2)

### color_map(a, b)
`{colour in a: colour in b}` if b is a cell-by-cell recolouring of a (same size, each colour always becomes the same
colour), else None.

    >>> color_map([[1, 2], [2, 0]], [[5, 2], [2, 0]])
    {0: 0, 1: 5, 2: 2}
    >>> color_map([[1, 1]], [[1, 2]]) is None
    True

### overview()
Print every training pair and test input in one summary. Per pair: sizes, colour counts, object
counts (with the input's bg for both grids; multicolour counts too when touching colours join objects), full lines
of one colour that cut the input (or a test input) into panels, a lattice of lines of the most frequent colour, whole-grid
relations ("the output is the input rot180 / recoloured 1->2 / scaled x2 / tiled 2x3"), for same-size pairs the
changed cells with their colour changes, print_match's counts (with multicolor=True when most objects are pieces of
multicolour ones) and a map of the changes (left out when most cells change), for other sizes the sub-grid place (and how many places), size ratios, "the output is input object #k (crop
...)", "the output is the input's sub-grid turned rot90 at (r,c)", "the input appears in the output at (r,c)" and
"the output has the size of input object #k's bbox". First, if the inputs' most frequent colours differ,
a line saying so (with a hint when every input's border is one colour), and the grids where two colours both look like
background; last, each test input with the colours no training input has, its multicolour count, and whether its
objects of different colours touch when no training input's do.

### preview(fn=None, path="solution.py", png=False)
Run transform (fn, or the one in solution.py) on every test input and print each output: size, colours, and the cells
that differ from the input ('.' = unchanged) when the sizes match, else the whole output. png=True also saves
out_test_<j>.png for view_image. Use it to look at the test outputs before you finish. An output check.py would
reject is reported as such. Returns the outputs as arrays (None for a rejected one): `outs = preview()`.

## Objects

### objects(g, bg=None, conn=8, multicolor=False, gap=0)
Connected components of the non-bg cells, as a list of Obj sorted by (top row, left column). `bg` = one colour, or a
list of colours that are all background (`bg=[0, 5]`: 0 and 5 are background, the rest are objects); `bg=-1`: every
colour is an object. The cells of one colour, wherever they are: `objects(g == 3)` (components) or `by_color(g)[3]`
(one Obj). `gap=k` also joins cells with up to k background cells between them: conn=8 in any direction
(Chebyshev distance up to k + 1), conn=4 only along a row or column (dashed lines, dotted outlines).

    >>> g = [[1, 1, 0, 2],
    ...      [1, 0, 0, 2],
    ...      [0, 0, 3, 0]]
    >>> [(o.idx, o.color, o.size, o.bbox) for o in objects(g)]
    [(0, 1, 3, (r0=0, c0=0, r1=1, c1=1)), (1, 2, 2, (r0=0, c0=3, r1=1, c1=3)), (2, 3, 1, (r0=2, c0=2, r1=2, c1=2))]
    >>> len(objects(g, conn=4))
    3
    >>> len(objects(g, multicolor=True))
    2

With conn=8 the 3 at (2,2) does not touch the 1s (no shared side or corner); with multicolor=True the 2s and the 3
touch diagonally and become one object. **The bg is decided per grid**: `objects(train[0][1])` may use another
bg than `objects(train[0][0])` -- pass `bg=` to compare like with like.

`objects(g, bg=-1)`: no colour is background, so the background's own regions are objects too.
`objects(mask)` on a bool array: the components of its True cells (color 1).

    >>> len(objects([[1, 0, 1, 0, 0, 1]])), len(objects([[1, 0, 1, 0, 0, 1]], gap=1)), len(objects([[1, 0, 1, 0, 0, 1]], gap=2))
    (3, 2, 1)

### by_color(g, bg=None)
`{colour: Obj}`: all the cells of each non-bg colour as one Obj, touching or not (bg as in objects).

    >>> by_color([[1, 0, 1], [0, 2, 0]])
    {1: #0 color 1 size 2 bbox r0-0 c0-2 hw 1x3, 2: #1 color 2 size 1 bbox r1-1 c1-1 hw 1x1}

### Obj
One object (also returned by `regions`). Attributes and methods:

- `idx` position in the list; `color` majority colour; `colors` `{colour: number of cells}`; `size` number of cells
- `cells` list of `(r, c)` in grid coordinates; `rel` the same cells relative to the bbox's top-left corner (the shape
  wherever it stands: `o.rel == p.rel` = same shape, same orientation); `bbox` Box; `h`, `w` bbox height / width;
  `rows`, `cols`, `slices` as for the Box; `center` bbox centre `(r, c)` (may be x.5)
- `mask` bool array of the bbox's size, not the grid's (True = the object's cell); `draw()` the mask as text rows
- `grid(bg=None)` the bbox with the object's colours and `bg` elsewhere (default: the bg objects() used; -1: "no cell")
- `holes` number of enclosed holes; `hole_cells` their cells, in grid coordinates (cells inside the bbox, not the
  object's, that cannot reach the bbox edge through 4-connected non-object cells -- they may hold other colours)
- `is_rect` fills its bbox; `is_frame` covers the whole border of its bbox (at least 3x3) but not all of the inside
  (a filled block is is_rect, not is_frame); `interior` the Box inside the border (bbox shrunk by 1)
- `touches` the grid edges it touches, from ('top', 'bottom', 'left', 'right')
- `encloses(other)` True if every cell of `other` lies in this object's holes

    >>> ring = objects([[0, 0, 0, 0, 0],
    ...                 [0, 4, 4, 4, 0],
    ...                 [0, 4, 7, 4, 0],
    ...                 [0, 4, 4, 4, 0]])
    >>> ring
    [#0 color 4 size 8 bbox r1-3 c1-3 hw 3x3 frame holes 1, #1 color 7 size 1 bbox r2-2 c2-2 hw 1x1]
    >>> ring[0].hole_cells, ring[0].interior, ring[0].encloses(ring[1]), ring[0].touches
    ([(2, 2)], (r0=2, c0=2, r1=2, c1=2), True, ('bottom',))
    >>> ring[0].grid().tolist()
    [[4, 4, 4], [4, 0, 4], [4, 4, 4]]

### print_objects(g, bg=None, conn=8, multicolor=False, gap=0)
Print one line per object (the Obj repr: colour, size, bbox, h x w, rect / frame, holes), plus its mask when the bbox
is at most 10x10. At most 30 objects are listed. With the default bg it says when two colours both look like
background (then pass both: `bg=[x, y]`).

    >>> print_objects([[0, 5, 5], [0, 5, 0], [6, 0, 0]])
    2 objects (bg 0, conn 8, same-colour)
    #0 color 5 size 3 bbox r0-1 c1-2 hw 2x2
        55
        5.
    #1 color 6 size 1 bbox r2-2 c0-0 hw 1x1

### labels(g, bg=None, conn=8, multicolor=False)
An int array of the grid's shape, numbered like scipy's `ndimage.label`: 0 on bg, `k + 1` on the cells of
`objects(...)[k]` (so the objects are 1..n).

    >>> labels([[1, 0, 2], [1, 0, 2]]).tolist()
    [[1, 0, 2], [1, 0, 2]]

### components(x, conn=8, gap=0)
Connected groups of a bool mask's True cells, or of a list of (r, c) cells, as cell lists sorted by top row, then
left column. `gap=k` also joins cells with up to k other cells between them (conn=8: any direction; conn=4: along a
row or column).

    >>> components([(0, 0), (0, 1), (2, 2), (3, 3)])
    [[(0, 0), (0, 1)], [(2, 2), (3, 3)]]
    >>> components([(0, 0), (0, 1), (2, 2), (3, 3)], conn=4)
    [[(0, 0), (0, 1)], [(2, 2)], [(3, 3)]]
    >>> components([(0, 0), (0, 1), (2, 2), (3, 3)], gap=1)
    [[(0, 0), (0, 1), (2, 2), (3, 3)]]

### adjacency(objs, conn=4)
`{idx: [idx of every other object that touches it]}` for a list of Obj; conn=4 = sharing a side, 8 = or a corner.

    >>> adjacency(objects([[1, 2, 0], [0, 0, 3]]))
    {0: [1], 1: [0], 2: []}
    >>> adjacency(objects([[1, 2, 0], [0, 0, 3]]), conn=8)
    {0: [1], 1: [0, 2], 2: [1]}

### cluster(items, linked)
Split any list (Obj, cells, numbers ...) into groups: two items share a group when `linked(a, b)` is True or a chain
of linked items joins them (single linkage). Groups keep the items' order. For objects within some distance:

    >>> objs = objects([[1, 0, 0, 2, 0, 0, 0, 3]])
    >>> cluster(objs, lambda a, b: gap(a, b).cols <= 2)
    [[#0 color 1 size 1 bbox r0-0 c0-0 hw 1x1, #1 color 2 size 1 bbox r0-0 c3-3 hw 1x1], [#2 color 3 size 1 bbox r0-0 c7-7 hw 1x1]]

### gap(a, b)
How far apart a and b are (each an Obj, a Hit, a list of cells, one cell `(r, c)` or a Box), as a Gap:
`rows` / `cols` = empty rows / columns strictly between their bboxes (0 when they overlap or touch on that axis),
`cheb` = the smallest max(|dr|, |dc|) between their cells (1 = touching, corners too; 0 = a shared cell), `manh` = the
smallest |dr| + |dc| (1 = sharing a side), `dr` / `dc` = where b is: -1 above / left of a, 1 below / right, 0 overlapping
on that axis.

    >>> gap([(0, 0), (1, 0)], [(4, 3)])
    Gap(rows=2, cols=2, cheb=3, manh=6; b is diagonal (+1, +1) of a)

## Shapes

### orient(g, name), ORIENTS, inverse(name)
Turn a grid (or mask) by one of the 8 rotations / flips in ORIENTS:
`id rot90 rot180 rot270 flipud fliplr transpose antitranspose`. **rot90 is counter-clockwise** (np.rot90);
rot270 is clockwise. `inverse(name)` undoes it (rot90 <-> rot270; the others undo themselves).

    >>> orient([[1, 2], [3, 4]], "rot90").tolist()
    [[2, 4], [1, 3]]
    >>> orient([[1, 2], [3, 4]], "fliplr").tolist(), inverse("rot90")
    ([[2, 1], [4, 3]], 'rot270')

### upscale(g, k), downscale(g, k, how="exact"), scale_of(g)
`upscale`: every cell becomes a k x k block (k = int or (rows, cols)). `downscale`: each block becomes one cell;
how="exact" raises unless every block is one colour, how="majority" takes the block's most frequent colour.
`scale_of`: the largest (kr, kc) with g == upscale(downscale(g, (kr, kc))); (1, 1) if g is not a block upscale.

    >>> upscale([[1, 2]], 2).tolist()
    [[1, 1, 2, 2], [1, 1, 2, 2]]
    >>> scale_of([[1, 1, 2, 2], [1, 1, 2, 2]])
    (2, 2)
    >>> downscale([[1, 1, 2, 2], [1, 3, 2, 2]], 2, how="majority").tolist()
    [[1, 2]]

### squeeze(g)
g with every run of identical consecutive rows kept once, then every run of identical consecutive columns: a picture
drawn in blocks of different sizes becomes one cell per block (downscale needs blocks of one size).

    >>> squeeze([[1, 1, 1, 2], [1, 1, 1, 2], [3, 3, 3, 3]]).tolist()
    [[1, 2], [3, 3]]

### shape_key(x, flips=True, scale=False, color=False, bg=0)
A hashable key that is equal for equal shapes: use it to group or look up shapes. x = Obj, bool mask, or grid (its
cells != bg; bg default 0 here -- a small pattern's most frequent colour is often the shape itself; bg=None = its
most frequent colour). `flips=True`: equal under the 8 rotations / flips. `scale=True`: equal under integer upscaling. `color`: False = cells only, True = colours must match too, "relabel" = the same colour pattern up to a
renaming of colours. The key is the canonical drawing as a tuple of rows (1 / colour = cell, 0 / -1 = no cell).

    >>> L1 = [[1, 0], [1, 1]]
    >>> L2 = [[2, 2], [0, 2]]
    >>> shape_key(L1) == shape_key(L2)
    True
    >>> shape_key(L1, flips=False) == shape_key(L2, flips=False)
    False
    >>> shape_key(L1) == shape_key(upscale(L1, 2)), shape_key(L1) == shape_key(upscale(L1, 2), scale=True)
    (False, True)

Careful with scale=True: every filled rectangle with the same side ratio gets the same key (a 2x2 block is a 1x1
cell scaled by 2).

### shape_match(a, b, bg=0, color=False)
The ways shape a turns into shape b: orientation names, `name*k` (a scaled up by k first) and `name/k` (a scaled
down by k first; k is read off the two sizes, any factor). a, b = Obj, bool mask or grid (cells != bg; default bg 0, bg=None = each grid's most frequent
colour), each cropped to its bbox. color=True also compares colours. Empty list = no match. Several names = several
ways give the same cells (a symmetric shape): any of them is right, and match / groups print them all as
`rot270|flipud`.

    >>> shape_match([[1, 0], [1, 1]], [[1, 1], [1, 0]])
    ['rot270', 'flipud']
    >>> shape_match([[1, 0], [1, 1]], upscale([[1, 1], [0, 1]], 2))
    ['rot180*2', 'transpose*2']

### align(a, b, flips=True, color=False)
Every way to lay shape a exactly onto shape b, as a list of Fit: turn a by `.orient` (about its own bbox), then put
its bbox's top-left at `(.r, .c)`; `.dr .dc` = how far that corner moved. a, b = Obj, Hit, a list of cells or a bool
mask, in grid coordinates; [] if no orientation of a has b's cells. flips=False tries only the 4 rotations;
color=True also needs the colours to match (both Obj). `fit.apply(cells)` moves any cells of a's grid the same way
-- where a marker or a part attached to a goes when a is turned and moved onto b. shape_match gives only the names;
use align when you need where the cells land.

    >>> a = [(0, 0), (1, 0), (1, 1)]; b = [(5, 7), (5, 8), (6, 7)]
    >>> align(a, b)
    [Fit(rot270, then by (+5,+7): a's bbox top-left (0,0) -> (5,7)), Fit(flipud, then by (+5,+7): a's bbox top-left (0,0) -> (5,7))]
    >>> align(a, b, flips=False)[0].apply([(2, 0)])
    [(5, 6)]

### groups(x, by="shape", flips=True, scale=False, bg=None, conn=8, multicolor=False), print_groups(...)
Objects (of grid x, or a list of Obj) grouped by equal shape, largest group first. `by`: "shape" (cells only),
"shape+color" (colours too), "pattern" (same colour pattern up to renaming colours). Each Group has `members`
(Obj list) and `relation[i]` = how members[0] turns into members[i] (shape_match names joined by "|"; "~" = only
after renaming colours). `print_groups` prints one line per group of 2+ (each member with its colour, at most 8
members, "filled rectangles" when they all are), then the objects whose shape is unique.

    >>> g = [[1, 0, 0, 2, 2],
    ...      [1, 1, 0, 0, 2],
    ...      [0, 0, 0, 0, 0],
    ...      [3, 3, 3, 0, 0]]
    >>> print_groups(g)
    3 objects in 2 groups by shape (rotations/flips allowed)
      group of 2: 2x2  #0 c1 id  #1 c2 rot180|transpose
          1.
          11
      unique: #2 c3 1x3

### find(g, pattern, flips=False, scales=(1,), recolor=False, bg=None, alone=False)
Every place a pattern occurs in g, as a list of Hit (`.r .c` top-left, `.h .w`, `.bbox`, `.cells` = the grid cells
the pattern's compared cells landed on, `.orient`, `.scale`, `.colors`). pattern = a grid in which -1 marks "any value", or an Obj (its cells must match, the rest of its bbox is
"any value"). `flips=True` tries all 8 orientations; `scales` the upscale factors to try; `recolor=True` allows a
one-to-one renaming of the pattern's colours (Hit.colors gives it), except that the grid's bg (bg=None: its most
frequent colour) is never renamed -- a recoloured hit is the shape in other colours, not a patch of background.
`alone=True` keeps only hits whose cells touch (4-neighbours) no other non-bg cell: the shape standing on its own,
not a piece of a bigger object. Overlapping hits are all listed. A Hit works wherever cells do (paint, halo, ...).

    >>> g = [[0, 5, 0, 0],
    ...      [5, 5, 0, 7],
    ...      [0, 0, 7, 7]]
    >>> find(g, [[-1, 5], [5, 5]])
    [at (0,0) 2x2 id]
    >>> find(g, [[-1, 5], [5, 5]], recolor=True)
    [at (0,0) 2x2 id, at (1,2) 2x2 id colours {5: 7}]
    >>> find(g, [[5, 5], [-1, 5]], flips=True)
    [at (0,0) 2x2 rot270]

A hit only checks the pattern's own cells (use alone=True when the shape must stand alone). `scales=3` is the same
as `scales=(3,)`. A bool pattern raises (it would be read as colours 0 / 1): pass `np.where(mask, colour, -1)`, or the Obj. When several orientations put the same cells at the same place (a symmetric pattern, or flipud
and rot270 above), the hit is listed once, with the first of them in ORIENTS order.

## Layout

### separators(g)
Rows and columns that are one colour all the way across: `{"rows": {r: colour}, "cols": {c: colour}}`.

### panels(g, color=None, min_frac=1.0)
The grid cut along full rows / columns of one colour, as a list of Panel (`.i .j` band indices, `.bbox`, `.grid`)
in reading order; thick lines count once; lines that stop before the edge do not cut. color=None chooses the
separator colour: a colour other than the grid's most frequent one first, then the most separate lines (a solid band
of several rows counts once), then the colour least used inside the panels. Lines of the grid's most frequent colour
count when they never meet that colour inside the panels, or when they cut the grid into a regular lattice (4 or more
panels of one size) -- blank background rows around pictures do not. No such colour: one panel, the whole grid. A full
line can also be part of the picture: check that the panels make sense, or pass `color=`.

    >>> g = [[1, 1, 5, 2, 2],
    ...      [1, 1, 5, 2, 2],
    ...      [5, 5, 5, 5, 5],
    ...      [3, 3, 5, 4, 0]]
    >>> panels(g)
    [panel (0,0) rows 0-1 cols 0-1 (2x2), panel (0,1) rows 0-1 cols 3-4 (2x2), panel (1,0) rows 3-3 cols 0-1 (1x2), panel (1,1) rows 3-3 cols 3-4 (1x2)]
    >>> panels(g)[3].grid.tolist()
    [[4, 0]]

Separator lines interrupted by marks or by crossing lines of another colour are not full lines: `min_frac=0.8` (with
`color=`) also cuts on rows / columns that are at least 80% that colour. `lines(g)` lists such lines.

    >>> h = [[1, 5, 2], [5, 6, 5], [3, 5, 4]]
    >>> len(panels(h, color=5)), len(panels(h, color=5, min_frac=0.6))
    (1, 4)

### lines(g, min_frac=0.6, bg=None)
Rows and columns (at least 3 cells long) whose most common colour fills at least `min_frac` of them, as Line
(`.axis` "row" / "col", `.i`, `.color`, `.n` of `.of` cells, `.marks` [(position along the line, colour)] for the
other cells, `.cells`), rows first. Lines of the background colour are left out (bg=None: the most frequent colour;
`bg=-1` keeps them).

    >>> lines([[0, 0, 0, 0], [3, 3, 5, 3], [0, 0, 0, 0]])
    [row 1: colour 3, 3/4; other c2=5]

### tiles(g, nr, nc)
The grid cut into nr x nc equal blocks with no separator lines (e.g. three stacked 3x3 blocks: `tiles(g, 3, 1)`), as
Panels in reading order; raises unless the sides divide evenly.

    >>> [p.grid.tolist() for p in tiles([[1, 2, 3, 4]], 1, 2)]
    [[[1, 2]], [[3, 4]]]

### symmetry(g, ignore=None)
For each rotation / flip that keeps the grid's shape: how many cells differ from the turned grid (0 = symmetric
that way). `ignore` = colour(s) treated as unknown on either side (e.g. a patch hiding part of a pattern).

    >>> symmetry([[1, 2, 1], [3, 4, 3]])
    {'rot180': 6, 'flipud': 6, 'fliplr': 0}
    >>> symmetry([[1, 2, 9], [3, 4, 3]], ignore=9)
    {'rot180': 4, 'flipud': 4, 'fliplr': 0}

### mirror_axes(g, ignore=None, min_overlap=0.25)
Mirror lines anywhere in the grid (not only through the centre): `[(kind, position, compared, mismatched)]`,
sorted by mismatched / compared, then most compared first. kind "col" = a vertical mirror line at column `position`
(x.5 = between two columns), "row" = horizontal; compared = cell pairs on both sides of the line with neither cell
ignored; a line with nothing compared is left out. Lines whose mirrored part covers less than min_overlap of the
width (height) are not listed (a line on a column counts that column). On a mostly empty grid many lines "match"
because background meets background: read `compared`, and pass the bg as ignore= to compare only the picture.

    >>> mirror_axes([[1, 2, 3, 2, 1, 9], [4, 5, 6, 5, 4, 9]])[0]
    ('col', 2.0, 4, 0)

Here columns 0-4 mirror around column 2 (4 cell pairs compared, none differ); column 5 is outside the mirror.

### fill_by_symmetry(g, ignore, using, partial=False)
A copy of g whose `ignore`-coloured cells (holes in a pattern) get the colour of their mirror image under the
symmetries in `using`: orientation names that keep the grid's shape ("fliplr", "rot180", ...) and / or mirror lines
as mirror_axes gives them (("col", 4.5, ...)). Repeated until nothing changes. Cells no symmetry can fill raise a
ValueError that lists them (a wrong symmetry is the usual cause); `partial=True` keeps their colour instead.

    >>> fill_by_symmetry([[1, 2, 9], [3, 4, 3]], 9, ["fliplr"]).tolist()
    [[1, 2, 1], [3, 4, 3]]
    >>> fill_by_symmetry([[9, 1, 2, 9, 1]], 9, [("col", 2.5)], partial=True).tolist()
    [[9, 1, 2, 2, 1]]

### period(g, ignore=None, max_wrong=0, axis=None)
The smallest tile (by area) that tiles g from the top-left corner, as a Period: `.pr .pc` (tile height / width),
`.tile` (array; each cell = the majority colour of the cells it covers), `.wrong` (cells that disagree), `.closest`
(when only the whole grid fits: the best smaller tile `(pr, pc, cells off)`, which the repr names), `.tiled()`
(the whole grid rebuilt from the tile: repairs the wrong and the ignored cells). `ignore` = colour(s) that do not
vote and are never wrong (holes to fill); `max_wrong` = how many disagreeing cells to accept (noise); axis=0 /
axis=1 = repeat only down the rows / only across the columns. A tile side shorter than the grid must repeat about
1.5 times or more (2 x side <= grid side + 1); the whole grid is always a period, so look at pr / pc (the repr shows the repeats). Since ignored cells never
disagree, a large `ignore` can make the tile too small: check `.tile` (-1 = a tile cell only ignored cells cover).

    >>> p = period([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 0]], ignore=0)
    >>> p, p.tile.tolist(), p.tiled().tolist()
    (period 2x2 (1.5 x 2 repeats), 0 cells off the tiling, [[1, 2], [3, 4]], [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2]])
    >>> period([[1, 2, 1, 2], [1, 2, 1, 7]], max_wrong=1)
    period 1x2 (2 x 2 repeats), 1 cells off the tiling

### runs(seq)
Run-length encoding of one row or column: `[(colour, start, length)]` (a 2-D grid raises: `runs(g[r])`, `runs(g[:, c])`).

    >>> runs([0, 0, 3, 3, 3, 0])
    [(0, 0, 2), (3, 2, 3), (0, 5, 1)]

### neighbors(g, color=None, conn=8, bg=None)
An int array: for every cell, how many of its 4 / 8 neighbours have `color` (a colour or colours; None = any colour
but bg).

    >>> neighbors([[1, 0, 1], [0, 0, 0]], color=1).tolist()
    [[0, 2, 0], [1, 2, 1]]

## Regions

### regions(g, wall, conn=4)
Connected groups of the cells that are not wall (wall = a colour, colours, or a bool mask of the grid's shape), as
multicolour Obj. A region whose `.touches` is empty does not reach the grid edge: the wall encloses it.

### enclosed(g, wall, conn=4)
The regions that do not touch the grid edge. With conn=4 a diagonal gap in the wall does not leak; conn=8 lets it
leak.

    >>> g = [[0, 0, 0, 0, 0],
    ...      [0, 2, 2, 2, 0],
    ...      [0, 2, 0, 2, 0],
    ...      [0, 2, 2, 0, 0]]
    >>> [r.cells for r in enclosed(g, wall=2)]
    [[(2, 2)]]
    >>> [r.cells for r in enclosed(g, wall=2, conn=8)]
    []

The inside cell (2,2) touches the gap at (3,3) only diagonally: closed with conn=4, open with conn=8.

### flood(g, start, conn=4, through=None, wall=None)
The cells reachable from `start` through cells of start's colour, or of the colour(s) in `through`, or of every colour
except the `wall` colour(s) (give one of through / wall).

    >>> flood([[1, 1, 0], [0, 1, 0], [1, 0, 1]], (0, 0))
    [(0, 0), (0, 1), (1, 1)]
    >>> flood([[1, 3, 0], [2, 2, 0], [0, 0, 0]], (0, 0), wall=0)
    [(0, 0), (0, 1), (1, 0), (1, 1)]

### distances(g, start, through=None, conn=4, bg=None)
BFS steps from `start` ((r, c), or a list of cells / an Obj: all at distance 0) through cells of the colour(s) in
`through` (default: the bg; bg=None = the most frequent colour), as an int array; -1 where unreachable. The start
cells themselves need not be passable; an empty start raises.

    >>> distances([[0, 0, 5], [0, 5, 0], [0, 0, 0]], (0, 0)).tolist()
    [[0, 1, -1], [1, -1, 5], [2, 3, 4]]

## Strokes and paths

### segments(x, min_len=2)
The straight runs a shape is drawn with: maximal horizontal, vertical and diagonal runs of at least `min_len` cells
of x (an Obj, cells or a bool mask), longest first, as Segment (`.direction` "horizontal" / "vertical" /
"diagonal \" (down-right) / "diagonal /" (down-left), `.cells` in order, `.start`, `.end`, `len()`). A cell can be in
several runs (a corner is in a horizontal and a vertical one).

    >>> segments([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
    [horizontal r0 c0-2 (3 cells), vertical c2 r0-2 (3 cells), diagonal \ (0,1)-(1,2) (2 cells)]

### trace(x, conn=8)
x (an Obj, cells or a bool mask) walked as a line, as a Trace: `.ends` (cells with one neighbour), `.branches`
(three or more), `.closed` (a loop), and `.cells` in order along it -- from the first end in reading order, or round
the loop -- or None when it branches or falls apart (`.note` says which). With conn=8 a diagonal step counts only
where no side neighbour already joins the two cells, so an L-corner or a staircase is still one path.

    >>> p = trace([(0, 0), (0, 1), (1, 2), (2, 2)])
    >>> p, p.cells, p.ends
    (path from (0, 0) to (2, 2), 4 cells, [(0, 0), (0, 1), (1, 2), (2, 2)], [(0, 0), (2, 2)])
    >>> trace([(0, 0), (0, 1), (0, 2), (1, 1)])
    no single path (1 branch cells); ends [(0, 0), (0, 2), (1, 1)], branches [(0, 1)]

### aligned(items)
Which pairs of items line up, as (i, j, how, where): single cells `(r, c)` on one row ("row", r), column ("col", c),
diagonal ("diagonal \", r - c) or anti-diagonal ("diagonal /", r + c); Obj / Hit when their bboxes share rows
("rows", (first, last)) or columns ("cols", (first, last)). i, j index the list.

    >>> aligned([(1, 1), (1, 6), (4, 4), (5, 2)])
    [(0, 1, 'row', 1), (0, 2, 'diagonal \\', 0), (1, 3, 'diagonal /', 7)]

## Drawing and moving

### DIRS, N4, N8
`DIRS`: direction names -> (dr, dc): up down left right up-left up-right down-left down-right.
`N4` = the 4 side offsets (up, down, left, right), `N8` = N4 + the 4 corner offsets.

### turn(direction, how)
A direction turned by 90 degrees: how = "cw" / "right" (clockwise on screen), "ccw" / "left", or "back". A DIRS name
gives a name back, a (dr, dc) gives a tuple.

    >>> turn("up", "cw"), turn("up", "left"), turn((0, 1), "cw"), turn("down-left", "back")
    ('right', 'left', (1, 0), 'up-right')

### adjacent(r, c, shape, conn=4)
The neighbours of (r, c) that lie inside a grid (shape = a grid or (H, W)), in N4 / N8 order.

    >>> adjacent(0, 0, (3, 3)), adjacent(1, 1, (3, 3), conn=8)[:4]
    ([(1, 0), (0, 1)], [(0, 1), (2, 1), (1, 0), (1, 2)])

### ray(g, start, direction, stop=None, bg=None)
Walk from `start` (not included) in a direction (a DIRS name or (dr, dc)) until a cell of a stop colour or the
edge. Returns a Ray: `.cells` crossed, `.hit` (the cell that stopped it, or None at the edge), `.hit_color`.
stop=None: any colour but bg; a colour / colours; "edge": run to the edge. A ray started on an object's own cell
stops at the object's next cell in that direction (start from the object's edge cell, or pass stop=).

    >>> r = ray([[3, 0, 0, 5, 0]], (0, 0), "right")
    >>> r, r.cells, r.hit
    (ray of 2 cells, stopped by colour 5 at (0, 3), [(0, 1), (0, 2)], (0, 3))

### line(p, q), outline(bbox)
`line`: the cells from p to q inclusive (same row, column or diagonal; raises otherwise). `outline`: the border
cells of a bbox (or an Obj's bbox).

    >>> line((0, 0), (2, 2))
    [(0, 0), (1, 1), (2, 2)]
    >>> outline((0, 0, 1, 2))
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

### halo(x, conn=8, shape=None), mask_of(cells, shape)
`halo`: the cells next to x (an Obj, Hit or cells) that are not x's own -- its outline ring; cells outside the grid
are left out (an Obj knows its grid; give shape = a grid or (H, W) for plain cells). `mask_of`: a bool array of a
grid's shape, True on the cells.

    >>> o = objects([[0, 0, 0], [0, 3, 0], [0, 0, 0]])[0]
    >>> halo(o, conn=4), int(mask_of(o, (3, 3)).sum())
    ([(0, 1), (1, 0), (1, 2), (2, 1)], 1)

### paint(g, cells, color, clip=False)
A copy of g with the cells (an Obj, a Hit, a bool mask, or (r, c) list) set to one colour. Cells off the grid raise, or are
skipped with clip=True.

    >>> paint([[0, 0, 0]], [(0, 0), (0, 2)], 4).tolist()
    [[4, 0, 4]]

### paste(g, patch, r, c, skip=-1, clip=False)
A copy of g with `patch` drawn with its top-left corner at (r, c). An Obj draws exactly its own cells in its own
colours (not the bg around them in its bbox); for a grid patch, cells equal to `skip` are not drawn (default -1;
pass the bg colour to paste a sub-grid "transparently" -- `paste(g, o.grid(), ...)` draws o's bg too). Cells off the grid
raise, or are dropped with clip=True.

    >>> paste([[0, 0, 0], [0, 0, 0]], [[5, 0], [5, 5]], 0, 1, skip=0).tolist()
    [[0, 5, 0], [0, 5, 5]]

### shift(cells, dr, dc)
The cells (Obj or (r, c) list) moved by (dr, dc); nothing is drawn.

### slide(g, obj, direction, bg=None)
How many steps obj (Obj or cells) can move in a direction before it would leave the grid or overlap a cell that is
neither bg nor its own. The result is that int (use it as a number), with what stops it: `.edge` (True: the grid edge),
`.hit` (the cells it would overlap next) and `.hit_colors`. No cells raises.

    >>> g = [[2, 0, 0, 8]]
    >>> s = slide(g, objects(g)[0], "right")
    >>> s, s + 1, s.edge, s.hit_colors
    (2 (then colour 8 at [(0, 3)]), 3, False, [8])
    >>> slide(g, objects(g)[1], "right").edge
    True

### move(g, obj, dr, dc, bg=None, clip=False, overwrite=False)
A copy of g with the Obj erased to bg and drawn again, in its own colours (it works on another or a blank grid too),
moved by (dr, dc). bg: the given one, else the bg the Obj was cut with, else g's most frequent colour. A destination
cell that holds another non-bg colour raises (the move would cover something) unless overwrite=True. Cells moved
off the grid raise, or are dropped with clip=True. For plain cells: `paint(g, shift(cells, dr, dc), colour)`.

    >>> move(g, objects(g)[0], 0, 2).tolist()
    [[0, 0, 2, 8]]

### gravity(g, direction="down", bg=None, conn=8, multicolor=False, fixed=None, rigid=True)
A copy of g in which every object slides in a direction as a rigid body until it rests on the edge or on another
cell; objects nearest that edge move first, repeated until nothing moves. `fixed` = colour(s) whose objects never
move (walls). Objects keep their shapes (use objects' conn / multicolor to say what moves together); rigid=False:
every cell falls on its own (sand), so shapes break up and pile up per column (per row for left / right).

    >>> gravity([[0, 2, 0], [0, 2, 2], [0, 0, 0], [5, 0, 0]]).tolist()
    [[0, 0, 0], [0, 0, 0], [0, 2, 0], [5, 2, 2]]
    >>> gravity([[0, 2, 0], [0, 2, 2], [0, 0, 0], [5, 0, 0]], rigid=False).tolist()
    [[0, 0, 0], [0, 0, 0], [0, 2, 0], [5, 2, 2]]
    >>> gravity([[1, 1], [0, 0], [0, 1]], rigid=False).tolist(), gravity([[1, 1], [0, 0], [0, 1]]).tolist()
    ([[0, 0], [0, 1], [1, 1]], [[0, 0], [1, 1], [0, 1]])

## Packing

### pack(region, pieces, rotate=False, flip=False, cover=True, limit=50, max_steps=500000)
Every way to place all the pieces inside the region, without overlap, as a list of Packing. Pieces are moved only
(translation) unless `rotate=True` (also 90 / 180 / 270 degrees) or `flip=True` (also mirror images).
`cover=True`: the pieces must fill every region cell (it raises when the areas do not add up); `cover=False`: region
cells may stay empty. `region`: a bool mask of the grid (`g == 0`), a list of (r, c), a Box, or an Obj (its cells);
use `outline(box)` for a frame. `pieces`: Obj (keeps its colours), lists of cells, bool masks, or small grids with -1
where the piece has no cell. Identical pieces are interchangeable (one packing is not listed once per swap).
Printing the result says how many packings there are and whether the search is complete -- more than one packing is
a reading the pieces alone do not decide. `Packing.placements` = [(piece index, orientation, r, c)] (r, c = top-left
of the turned piece's bbox), `.cells` = {(r, c): piece index}, `.grid(g)` draws the pieces on a copy of g (colourless
pieces need `colors=[...]`).

    >>> region = np.ones((2, 3), bool)
    >>> A = np.array([[1, 1], [1, -1]]); B = np.array([[2, 2], [-1, 2]])
    >>> pack(region, [A, B])
    no packing
    >>> ps = pack(region, [A, B], rotate=True)
    >>> ps
    4 packings (the search is complete: there are no others)
      packing: piece 0 id at (0,0), piece 1 rot270 at (0,1)
      packing: piece 0 rot90 at (0,0), piece 1 id at (0,1)
      packing: piece 0 rot180 at (0,1), piece 1 rot90 at (0,0)
      packing: piece 0 rot270 at (0,1), piece 1 rot180 at (0,0)
      they differ in where pieces [0, 1] go
    >>> ps[0].grid().tolist()
    [[1, 1, 2], [1, 2, 2]]

The search is a plain backtracking over placements: dozens of pieces on a 30x30 region can take long, and it stops
after `max_steps` steps (printing says so); the default 500,000 steps take about 3 s (check.py gives transform 30 s
per grid, so inside transform keep it at that or below).

## Input -> output

### match(a, b, bg=None, conn=8, multicolor=False)
Pair the objects of input a with the objects of output b, as a list of Link (`.kind`, `.a` input Obj, `.b` output
Obj, `.info`). Both grids are cut with the same settings; bg = a's most frequent colour unless given. Kinds, tried in
this order with each object used once:
- `same`; `moved` (info: by (dr,dc)); `recolored` (same cells; info: the colour changes); `turned` (rotated /
  flipped / scaled, colours kept; info: every orientation name that fits, joined by "|", and the new place);
- `copy`: an extra output object equal to any input object (maybe turned);
- `moved+recolored` (same shape, other colours and place); `changed` (overlapping cells, different shape; info:
  cells added / removed);
- `merged`: an input object (3+ cells) whose exact pattern is now part of one bigger output object that grew
  (`changed`) or has no input of its own, e.g. a shape that moved until it joined a wall of its colour (info: by
  (dr,dc) into out #k). Several merged inputs can share one output object, and that object is not also `added`;
- `added` (the remaining output objects), `deleted` (the remaining input objects).
A new colour that another input object of the same shape has is named ("colour of same-shape input #3"); a moved or
turned object that touches another output object says so ("now touches #1 c8"). Pairing is greedy (nearest first):
when several objects look alike (single cells), check that the pairing is the one you mean. When the two grids have
different sizes the pairing compares places in different grids and means little (print_match says so).

### print_match(a=None, b=None, bg=None, conn=8, multicolor=False)
match() as text; with no grids, for every training pair. More than 5 `same` objects are summed up in one line; at
most 12 lines of each other kind are listed per pair.

    >>> print_match([[2, 0, 0, 8]], [[0, 0, 2, 8]])
    2 input objects -> 2 output objects (bg 0, conn 8): 1 same, 1 moved
      same            in #1 c8 1x1 at (0,3)     out #1 c8 1x1 at (0,3)
      moved           in #0 c2 1x1 at (0,0)     out #0 c2 1x1 at (0,2)     by (+0,+2), now touches #1 c8

### explain_select(bg=None, conn=8, multicolor=False)
For puzzles whose output is one of the input's objects (its bbox crop, or its cells on bg; maybe turned or scaled):
prints which simple one-property rules pick that object in every training pair -- sizes, positions (marked "a
position"), "the only object with a colour / shape / size no other object has", "the only object (not) touching the
grid edge", "the only object (not) left-right symmetric" (also up-down, rot180, diagonal), ... -- grouped by the
objects they pick in the test inputs. Several groups = the training pairs do not decide between them; "no pick
(tie)" = the rule has no unique winner in that test input. With only one or two fitting rules it warns that a rule can
fit by chance. Returns `[(rule, [Obj picked in each test input, or None on a tie])]`; [] (with a message) when some
pair's output is not one of its input's objects with these settings (try other bg / conn / multicolor).

### explain_objects(bg=None, conn=8, multicolor=False, table=True)
For puzzles whose output is the input with some of its objects kept, deleted, recoloured, moved, turned or changed:
what happens to every input object in every training pair (print_match's pairing), one line per object with its
size, shape, flags and the colours it touches, and then, for each fate, the rules of one property that give exactly
the objects with that fate in every pair -- size, height, holes, "colour 5", "touching colour 2", "not a filled
rectangle", "the largest", ... (positions marked "a position"; two properties joined by "and" when no single one
fits, positions left out) -- with the objects each rule picks in every test input. Rules that agree on train but
pick different test objects are grouped apart, simplest first: the training pairs do not decide between them, and
that is where to look before you trust a rule. A fate seen in one pair only gets no rules (any rule fits it by
chance). Needs every output the size of its input. Returns `{fate: [(rule, [the Obj it picks in each test input])]}`;
table=False leaves out the per-object lines.

    train 1: 4 objects (bg 0, conn 8): 2 kept, 2 deleted
      #0  c1 size 4   2x2 at (0,0) holes 0 rect edge largest touches {} -> kept
      #1  c5 size 1   1x1 at (0,3) holes 0 rect edge touches {} -> deleted
      ...
    rules that give each fate in every training pair (one property; 'and' = two, positions left out):
      deleted (4 of 7 objects, in 2 of 2 pairs): 9 rule(s); they pick 3 different sets of test objects -- the training pairs do not decide:
        colour 5 (+2 more that pick the same)  ->  test 1: #0 #3
        size == 1 (+4 more that pick the same)  ->  test 1: #0 #2 #3

### explain_panels(color=None)
For puzzles whose input is cut by lines into equal panels and whose output has a panel's size: which cell-wise
combination gives the output in every training pair -- two panels joined by AND / OR / XOR / "first and not second"
/ "neither" (the second maybe flipped: fliplr, flipud, rot180), or all panels stacked in some order (a later panel's
non-background cells cover the earlier ones). A panel's cells count where they are not the background (the most
frequent colour inside the panels); the output's colour is reported ("drawn in colour 3"). Prints and returns the
fitting descriptions; [] (with the reason) when the input is not such panels. `color=` is passed to panels().

    output = panel 0 XOR panel 1 (non-background cells), drawn in colour 3

### sweep(fn, **space)
Try `fn(grid, **params)` for every combination of the values given as `name=[values, ...]` on every training pair
(grid = a list of lists, as transform gets it). Prints how many combinations pass all pairs (and how many raised,
with the first error -- a raising combination counts as failing, it is not hidden), the closest one if none passes,
and whether the passing combinations agree on the test inputs. An output check.py would reject (not a rectangle
of ints 0-9, a side outside 1..30, None, floats, bools) counts as raising. Passing combinations that give different test
outputs are readings the training pairs cannot tell apart: each group is listed with the test cells where it
differs from the first group. Returns the passing combinations as dicts.

    def rule(grid, conn, fill):
        g = np.array(grid)
        for o in objects(g, conn=conn):
            g = paint(g, o.hole_cells, fill if fill >= 0 else o.color)
        return g

    sweep(rule, conn=[4, 8], fill=[-1, 0, 1, 2])

Rival rules written as functions can be swept the same way; they are printed by name:

    sweep(lambda grid, f: f(grid), f=[rule_a, rule_b, rule_c])

### Counter, defaultdict, deque
Python's `collections` classes, exported by `from arc import *` so scripts need no import.

    >>> Counter([3, 3, 5]).most_common(1)
    [(3, 2)]

## Testing a hypothesis

The tools above find facts; these check a guess against every training pair at once -- the work that otherwise
happens cell by cell and object by object in your head, where it is slow and easy to get wrong. A guess that is
code is checked in one call, on every pair, and shows exactly where it fails.

### verify(rule, show=4)
Run `rule(grid) -> grid` (a candidate transform; it gets each grid as a list of lists, as transform does) on every
training pair and print, per pair, PASS or how it fails: an ERROR with the line it was raised on, a wrong size, or
the wrong cells -- how many, which colour changes (got -> want), the places they are in and the first `show` cells;
then what it gives on each test input. Nothing is written: test a hypothesis, or a variant of solution.py's rule,
without touching solution.py. Returns True when every training pair passes.

    >>> verify(lambda g: max(objects(g), key=lambda o: o.size).grid())
    train 1: PASS
    train 2: PASS
    2/2 training pairs pass
    test 1: output 2x2 (preview(rule) shows it)
    True
    >>> verify(lambda g: objects(g)[0].grid())
    train 1: wrong size 1x1, want 2x2
    train 2: PASS
    1/2 training pairs pass
    test 1: output 2x2 (preview(rule) shows it)
    False

### cellwise(f)
A transform that sets every output cell (r, c) to `f(g, r, c)` (g = the input as a read-only int array; f returns a
colour 0-9), for rules that decide each cell from the input around it -- check one with `verify(cellwise(f))`, and
`transform = cellwise(f)` in solution.py is a complete solution. The output has the input's size.

    >>> cellwise(lambda g, r, c: 9 - g[r, c])([[0, 9], [3, 4]]).tolist()
    [[9, 0], [6, 5]]

### check_fate(pred, fate, bg=None, conn=8, multicolor=False)
Test your own rule for which objects get a fate: `pred(o)` -- or `pred(o, g)` / `pred(o, g, objs)` to see the grid and
all its objects, for rules about neighbours, sizes relative to the others, alignment -- must be True exactly for the
input objects whose fate is `fate` ("deleted", "kept", "moved", "recolored", "recolored to 4", ... as
explain_objects names them; "recolored" covers every "recolored to c"), in every training pair (the pairing of
print_match). Prints whether it holds, each object where it does not (pair, object, pred's value, its fate), and
which objects pred picks in each test input. Returns those counter-examples ([] = it holds). Same-size pairs only.

    check_fate(lambda o: o.size == 1, "deleted")
    pred <=> deleted: holds for all 7 objects in 2 pairs (4 with that fate)
      test 1: pred picks #0 c5 1x1 at (0,0) #2 c6 1x1 at (2,0) #3 c5 1x1 at (2,3) of 4 objects
    check_fate(lambda o, g, objs: o.size == max(p.size for p in objs), "kept")
    pred <=> kept: fails on 1 of 7 objects:
      train 1 #3 c2 size 2 1x2 at (3,2): pred False, fate kept

### per_pair(f)
Print `f(input, output)` for every training pair and `f(input, None)` for every test input, one line each: check a
numeric or structural guess (counts, sizes, positions, a formula) on all pairs at once. Returns the values (the
Exception where f raised).

    >>> vals = per_pair(lambda a, b: (len(objects(a)), None if b is None else b.shape))
    train 1: (2, (2, 2))
    train 2: (2, (1, 3))
    test 1: (2, None)

### unseen(bg=None, conn=8, multicolor=False)
What each test input has that no training input has -- a grid size, a colour, a number of objects, an object size
or an object shape (any rotation) outside what the training inputs show: the cases a rule fitted to the training
pairs was never checked on. Look at what your rule does with exactly these before you stop. Prints and returns the
lines.

    >>> _ = unseen()
    test 1: colours [4, 6] (no training input has them)

## Caveats

The traps that the tools cannot decide for you:

- **bg is guessed per grid** (its most frequent colour). An output, a test input or a crop can have another most
  frequent colour than the training input: pass `bg=` explicitly in solution.py (overview warns when the inputs'
  guesses differ). Exception: `shape_key` / `shape_match` default to bg=0.
- **panels guesses the separator colour**; pass `color=` when you know it, and check the panels make sense.
- **conn differs**: objects / labels / components / groups / match / gravity / neighbors / halo default to 8;
  regions / enclosed / flood / distances / adjacency / adjacent default to 4. `find` defaults to flips=False, `shape_key` / `groups` to flips=True.
- `Obj.grid()` fills its bbox with the bg: `paste(g, o.grid(), r, c)` draws that bg too; `paste(g, o, r, c)` or
  `skip=bg` draws only the object.
- `symmetry` / `mirror_axes` on a sparse grid: bg meets bg and counts as matching; pass `ignore=bg` and read
  `compared`. `period(..., max_wrong=k)` with a large k (or a large ignore) accepts a trivial tile: check `.tile`.
- `match` is greedy; on many lookalike objects (single cells) the pairing may not be the one you mean, and on
  grids of different sizes it compares places in different grids. Hundreds of objects take a few seconds.
- Empty results are lists (`objects` of an all-bg grid = []); things that cannot be done raise (a crop off the grid,
  a cell off the grid without clip=True, a move onto another colour). An error message says what to change.
- `from arc import *` defines names such as `line`, `count`, `find`, `move`, `turn`, `test`: a variable of yours with
  one of those names hides the tool for the rest of that script.
- check.py is the judge: the final answer is what `transform` in solution.py returns on the test inputs, not what
  the tools print. It gives transform 30 s per grid.
- Tool output above about 2,500 tokens is cut in the middle (a 30x30 `show` is about 1,000): print one grid or one pair
  at a time, or use `diff` / `show(g, box=...)`.
- The sandbox refuses `rm`, and with it the whole command it is in: leave old files, or overwrite them.
- `paint`, `paste`, `move`, `gravity` and `fill_by_symmetry` return a new grid and leave g as it was:
  `g = paint(g, cells, 4)`; a bare `paint(g, cells, 4)` changes nothing.
- A list of cells (`o.cells`, `where(...)`) is not a grid: grid tools raise on it. `mask_of(cells, g)` makes a
  bool mask of g's shape, `o.mask` / `o.grid()` are the object's own bbox-sized views.
- What a tool refuses by default (paste / paint / move off the grid without clip=True, a move onto another colour) is a
  guard against bugs in your code, not a hint about the puzzle's rule.
- Run scripts from this folder (`python3 x.py` here). Do not name a file like a
  Python module (`enum.py`, `types.py`, `random.py`): it hides that module. scipy is not installed; `Counter`,
  `defaultdict` and `deque` come with `from arc import *`.

## Help

### doc(name=None)
`doc()` lists every tool with its first line; `doc("find")` prints that tool's section of this file (result types
work too: `doc("Hit")` is find's section, `doc("Obj.holes")` Obj's); `doc("conventions")` and `doc("caveats")` print
those two sections.
