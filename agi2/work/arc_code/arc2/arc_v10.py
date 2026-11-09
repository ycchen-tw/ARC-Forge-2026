"""This puzzle's grids and a toolkit of exact, tested functions. `from arc import *` (solution.py too).
train = [(input, output), ...], test = [input, ...]: numpy int arrays. Every grid argument also takes a list of
lists or a bool mask. doc("name") prints one tool's reference with examples; TOOLS.md has all of them.

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

The code below is long -- read TOOLS.md or doc(), not this file.
"""

from __future__ import annotations

import json
import operator
import re
from collections import Counter, defaultdict, deque
from functools import cached_property, reduce
from math import gcd
from pathlib import Path

import numpy as np

np.set_printoptions(legacy="1.25")     # numbers print as 5, not np.int64(5)

_HERE = Path(__file__).resolve().parent
_TASK = json.loads((_HERE / "task" / "task.json").read_text())
train = [(np.array(p["input"]), np.array(p["output"])) for p in _TASK["train"]]
test = [np.array(t["input"]) for t in _TASK["test"]]
for _g in [g for pair in train for g in pair] + test:
    _g.setflags(write=False)        # shared by every call: change a .copy(), never the puzzle itself

__all__ = ["np", "train", "test", "show", "diff", "render", "count", "where", "crop", "bbox_of", "color_map", "overview",
           "objects", "print_objects", "Obj", "Cells",
           "shape_key", "shape_match", "groups", "print_groups", "find", "orient", "ORIENTS", "inverse",
           "upscale", "downscale", "scale_of",
           "separators", "panels", "symmetry", "period",
           "regions", "enclosed", "flood",
           "DIRS", "ray", "line", "outline", "paint", "paste", "shift", "slide", "move", "runs", "neighbors",
           "mirror_axes", "Box",
           "match", "print_match", "explain_select", "doc",
           "background", "changes", "labels", "components", "adjacency", "distances", "N4", "N8", "turn",
           "adjacent", "gravity", "sweep", "preview", "tiles", "halo", "mask_of", "fill_by_symmetry",
           "Hit", "Panel", "Period", "Ray", "Slide", "Link", "Group", "Counter", "defaultdict", "deque",
           "by_color", "cluster", "gap", "Gap", "lines", "Line", "squeeze", "pack", "Packing", "Packings", "Changes",
           "align", "Fit", "segments", "Segment", "trace", "Trace", "aligned", "explain_objects", "explain_panels",
           "verify", "cellwise", "check_fate", "per_pair", "unseen"]

_PALETTE = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
            (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37))

_STEPS = {4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
          8: tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)}
_MAX_CELLS = 8        # a Cells list up to this long prints as a list; longer ones as rows of column runs
_MAX_CELL_ROWS = 30   # rows listed when printing a long Cells list
_MAX_CLUSTERS = 10    # diff clusters listed
_CLUSTER_CELLS = 6    # cells listed per diff cluster
_MAX_OBJECTS = 30     # print_objects lines
_MAX_MASK = 10        # print_objects draws masks up to this bbox height / width
_MAX_PER_KIND = 12    # print_match lines per kind of link and pair
_MAX_MEMBERS = 8      # print_groups members listed per group
_DIHEDRAL = (("id", lambda m: m), ("rot90", lambda m: np.rot90(m, 1)), ("rot180", lambda m: np.rot90(m, 2)),
             ("rot270", lambda m: np.rot90(m, 3)), ("flipud", np.flipud), ("fliplr", np.fliplr),
             ("transpose", lambda m: m.T), ("antitranspose", lambda m: np.rot90(m, 2).T))
ORIENTS = tuple(name for name, _ in _DIHEDRAL)
_ORIENT = dict(_DIHEDRAL)
_INVERSE = {"id": "id", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90", "flipud": "flipud",
            "fliplr": "fliplr", "transpose": "transpose", "antitranspose": "antitranspose"}
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
        "up-left": (-1, -1), "up-right": (-1, 1), "down-left": (1, -1), "down-right": (1, 1)}
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
_EMPTY = -1           # "no cell here" in shape keys, find patterns and Obj.grid(bg=-1)


# --------------------------------------------------------------------------- helpers

def _grid(g, what: str = "grid") -> np.ndarray:
    """g as a non-empty 2-D integer (or bool) array. Integral floats (np.zeros) are accepted, a Panel is its .grid;
    anything else raises."""
    if isinstance(g, Panel):
        g = g.grid
    elif isinstance(g, (Obj, Hit)):
        raise TypeError(f"{what}: got an {type(g).__name__}, not a grid -- use o.grid() (its bbox) or mask_of(o, g)")
    elif isinstance(g, Cells):
        raise TypeError(f"{what}: got a list of (r, c) cells, not a grid -- mask_of(cells, g) makes a bool mask of g's "
                        f"shape; paint(g, cells, c) draws them")
    a = np.asarray(g)
    if a.ndim == 3 and a.shape[0] == 2:
        raise ValueError(f"{what} must be one grid, got shape {a.shape}: that is a training (input, output) pair -- "
                         f"use `for a, b in train` or train[k][0] / train[k][1]")
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{what} must be a non-empty 2-D grid, got shape {a.shape}")
    if a.dtype.kind == "f":
        if not (np.isfinite(a).all() and np.array_equal(a, np.round(a))):
            raise ValueError(f"{what} has non-integer values")
        a = a.astype(np.int64)
    elif a.dtype.kind not in "iub":
        raise TypeError(f"{what} must hold integers, got dtype {a.dtype}")
    return a


def _bg(a: np.ndarray, bg):
    """The background colour: bg if given, else the most frequent colour (False for bool masks)."""
    if bg is not None:
        if isinstance(bg, (list, tuple, set, frozenset, np.ndarray)):
            raise TypeError(f"bg={bg!r}: this tool takes one bg colour (objects / print_objects / labels take a list)")
        return operator.index(bg)
    if a.dtype == bool:
        return False
    vals, n = np.unique(a, return_counts=True)
    return int(vals[n.argmax()])


def _bgs(a: np.ndarray, bg) -> tuple[int, set]:
    """(the bg colour objects record, the set of background colours): bg may be a list of colours."""
    if isinstance(bg, (list, tuple, set, frozenset, np.ndarray)):
        cs = [operator.index(v) for v in (sorted(bg) if isinstance(bg, (set, frozenset)) else bg)]
        if not cs:
            raise ValueError("bg: empty list of colours")
        return cs[0], set(cs)
    b = _bg(a, bg)
    return b, {b}


def _bg_arg(a: np.ndarray, bg):
    """bg as objects() takes it: the sorted list if a list of colours was given, else one colour (None: a's most
    frequent)."""
    if isinstance(bg, (list, tuple, set, frozenset, np.ndarray)):
        return sorted(_bgs(a, bg)[1])
    return _bg(a, bg)


def _bg_close_call(a: np.ndarray) -> str | None:
    """A note when two colours both look like background: together they cover at least 80% of the cells and the
    second is nearly as frequent as the first -- objects() with the default bg then lists the second one as objects."""
    if a.dtype == bool or a.size < 16:
        return None
    vals, n = np.unique(a, return_counts=True)
    if len(n) < 3:
        return None
    order = np.argsort(-n, kind="stable")
    top, second = int(n[order[0]]), int(n[order[1]])
    if top + second < 0.8 * a.size or second < 0.75 * top:
        return None
    x, y = int(vals[order[0]]), int(vals[order[1]])
    return (f"colours {x} ({top} cells) and {y} ({second}) both look like background: "
            f"objects(g, bg=[{x}, {y}]) lists the rest")


def _colors(c, what: str = "colour") -> set[int]:
    """An int or an iterable of ints as a set of colours."""
    if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
        return {int(c)}
    out = {operator.index(v) for v in c}
    if not out:
        raise ValueError(f"{what}: empty set of colours")
    return out


def _steps(conn: int, gap: int) -> tuple[tuple[int, int], ...]:
    """Neighbour offsets: conn=8 links cells up to gap + 1 apart in both directions (Chebyshev), conn=4 cells on one
    row or column up to gap + 1 apart (gap empty cells may lie between them)."""
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    gap = operator.index(gap)
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if gap == 0:
        return _STEPS[conn]
    d = gap + 1
    if conn == 8:
        return tuple((dr, dc) for dr in range(-d, d + 1) for dc in range(-d, d + 1) if dr or dc)
    return tuple(s for k in range(1, d + 1) for s in ((-k, 0), (k, 0), (0, -k), (0, k)))


def _components(a: np.ndarray, member: np.ndarray, same_color: bool, conn: int, gap: int = 0) -> list[list[tuple[int, int]]]:
    """Connected groups of the True cells of `member` (cells in reading order), sorted by (top row, left col)."""
    steps = _steps(conn, gap)
    h, w = a.shape
    seen = np.zeros(a.shape, bool)
    out = []
    for r0, c0 in np.argwhere(member).tolist():
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack, cells = [(r0, c0)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and member[y, x] and not seen[y, x] and (not same_color or a[y, x] == a[r0, c0]):
                    seen[y, x] = True
                    stack.append((y, x))
        out.append(sorted(cells))
    out.sort(key=lambda cl: (cl[0][0], min(c for _, c in cl), cl[0]))
    return out


def _cells_of(x, shape=None) -> list[tuple[int, int]]:
    """Cells given as an Obj or Hit, a bool mask of the grid's shape, or an iterable of (r, c)."""
    if isinstance(x, (Obj, Hit)):
        return list(x.cells)
    a = np.asarray(x)
    if a.dtype == bool and a.ndim == 2:
        if shape is not None and a.shape != tuple(shape):
            raise ValueError(f"mask shape {a.shape} differs from the grid shape {tuple(shape)}")
        return [(int(r), int(c)) for r, c in np.argwhere(a)]
    out = []
    for p in x:
        if isinstance(p, (int, np.integer)):
            raise TypeError("cells must be a list of (r, c) pairs (or an Obj / Hit / bool mask); one cell: [(r, c)]")
        r, c = p
        out.append((operator.index(r), operator.index(c)))
    return out


def _cells_mask(cells) -> tuple[np.ndarray, tuple[int, int]]:
    """(bool mask of the cells' bbox, its top-left (r0, c0))."""
    cs = _cells_of(cells)
    r0, c0 = min(r for r, _ in cs), min(c for _, c in cs)
    m = np.zeros((max(r for r, _ in cs) - r0 + 1, max(c for _, c in cs) - c0 + 1), bool)
    for r, c in cs:
        m[r - r0, c - c0] = True
    return m, (r0, c0)


def _pos(p, what: str) -> tuple[int, int]:
    """A position (r, c) as python ints; floats (e.g. Obj.center) raise."""
    try:
        r, c = p
        return operator.index(r), operator.index(c)
    except TypeError:
        raise TypeError(f"{what}: a position is (r, c) with integer r, c, got {p!r}") from None


def _dir(d) -> tuple[int, int]:
    if isinstance(d, str):
        if d not in DIRS:
            raise ValueError(f"direction must be one of {list(DIRS)} or (dr, dc), got {d!r}")
        return DIRS[d]
    dr, dc = d
    dr, dc = operator.index(dr), operator.index(dc)
    if (dr, dc) == (0, 0):
        raise ValueError("direction (0, 0) does not move")
    return dr, dc


def _view_lines(b: np.ndarray, keep=None, r0: int = 0, c0: int = 0) -> list[str]:
    """b with column numbers on top and row numbers on the left (as show), numbered from (r0, c0); cells where
    keep is False shown as '.'."""
    cols = range(c0, c0 + b.shape[1])
    lines = []
    if cols[-1] >= 10:
        lines.append("    " + "".join(str(c // 10 % 10) if c >= 10 else " " for c in cols))
    lines.append("    " + "".join(str(c % 10) for c in cols))
    lines += [f"{r0 + r:>3} " + "".join(str(int(b[r, c])) if keep is None or keep[r, c] else "." for c in range(b.shape[1]))
              for r in range(b.shape[0])]
    return lines


class Box(tuple):
    """A bbox (r0, c0, r1, c1), inclusive -- a plain tuple that also has .r0 .c0 .r1 .c1 .h .w."""
    __slots__ = ()

    def __new__(cls, r0, c0, r1, c1):
        return super().__new__(cls, (int(r0), int(c0), int(r1), int(c1)))

    r0 = property(lambda self: self[0])
    c0 = property(lambda self: self[1])
    r1 = property(lambda self: self[2])
    c1 = property(lambda self: self[3])
    h = property(lambda self: self[2] - self[0] + 1)
    w = property(lambda self: self[3] - self[1] + 1)
    rows = property(lambda self: range(self[0], self[2] + 1))
    cols = property(lambda self: range(self[1], self[3] + 1))
    slices = property(lambda self: (slice(self[0], self[2] + 1), slice(self[1], self[3] + 1)))

    def __repr__(self) -> str:
        return f"(r0={self[0]}, c0={self[1]}, r1={self[2]}, c1={self[3]})"


def _box(b) -> Box:
    """b as a Box: an Obj / Panel / Hit (its bbox) or a 4-tuple (r0, c0, r1, c1)."""
    if isinstance(b, (Obj, Panel, Hit)):
        return b.bbox
    r0, c0, r1, c1 = b
    return Box(r0, c0, r1, c1)


# --------------------------------------------------------------------------- show / diff / render

def show(g, coords: bool = True, dot=None, box=None) -> None:
    """Print a grid with column numbers on top and row numbers on the left; coords=False: bare digits.
    dot: a colour (or colours) printed as '.' (e.g. the bg). box: print only that bbox, keeping the grid's numbers."""
    a = _grid(g)
    r0 = c0 = 0
    if box is not None:
        r0, c0, r1, c1 = _box(box)
        if not (0 <= r0 <= r1 < a.shape[0] and 0 <= c0 <= c1 < a.shape[1]):
            raise ValueError(f"show: box {(r0, c0, r1, c1)} does not fit a grid of shape {a.shape}")
        a = a[r0:r1 + 1, c0:c1 + 1]
    keep = ~np.isin(a, list(_colors(dot, "dot"))) if dot is not None and dot is not False else None
    if not coords:
        print("\n".join("".join(str(int(v)) if keep is None or keep[r, c] else "." for c, v in enumerate(row))
                        for r, row in enumerate(a)))
        return
    if a.min() < 0 or a.max() > 9:
        raise ValueError("show(coords=True) needs cell values 0-9; use coords=False or show(g % 10)")
    print("\n".join(_view_lines(a, keep, r0, c0)))


def diff(a, b, limit: int = 40) -> None:
    """Where two grids differ; above `limit` cells, grouped into clusters of touching cells."""
    a, b = _grid(a, "a"), _grid(b, "b")
    if a.shape != b.shape:
        print(f"size differs: {a.shape} vs {b.shape}")
        return
    cells = np.argwhere(a != b)
    if len(cells) <= limit:
        print(f"{len(cells)} of {a.size} cells differ")
        for r, c in cells:
            print(f"  ({r},{c}): {a[r, c]} vs {b[r, c]}")
        return
    clusters = _components(a, a != b, False, 8)
    print(f"{len(cells)} of {a.size} cells differ in {len(clusters)} clusters")
    for cl in clusters[:_MAX_CLUSTERS]:
        rows, cols = [r for r, _ in cl], [c for _, c in cl]
        ca = ",".join(str(v) for v in sorted({int(a[r, c]) for r, c in cl}))
        cb = ",".join(str(v) for v in sorted({int(b[r, c]) for r, c in cl}))
        print(f"rows {min(rows)}-{max(rows)} cols {min(cols)}-{max(cols)}: {len(cl)} cells, a {{{ca}}} -> b {{{cb}}}")
        listed = "  ".join(f"({r},{c}) {int(a[r, c])}->{int(b[r, c])}" for r, c in cl[:_CLUSTER_CELLS])
        more = f"  ... +{len(cl) - _CLUSTER_CELLS}" if len(cl) > _CLUSTER_CELLS else ""
        print(f"  {listed}{more}")
    if len(clusters) > _MAX_CLUSTERS:
        rest = clusters[_MAX_CLUSTERS:]
        print(f"... {len(rest)} more clusters ({sum(len(cl) for cl in rest)} cells)")


def render(grid, path) -> None:
    """Save a grid as a PNG (same colours and cell size as task/*.png), to look at with view_image."""
    from PIL import Image
    a = np.asarray(grid).astype(int)
    c = max(16, -(-256 // max(a.shape)))
    rgb = np.array(_PALETTE, dtype=np.uint8)[a].repeat(c, axis=0).repeat(c, axis=1)
    rgb[::c, :] = 64
    rgb[:, ::c] = 64
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------- cells, colours

def _col_runs(cols: list[int]) -> str:
    """Sorted column numbers as runs: [2, 3, 4, 6] -> '2-4,6'."""
    out, start = [], cols[0]
    for prev, c in zip(cols, cols[1:] + [None]):
        if c is None or c != prev + 1:          # a run ends at prev
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = c
    return ",".join(out)


class Cells(list):
    """A list of (r, c). Up to 8 cells print as a list; more print row by row as column runs,
    '15 cells: r1 c2-6 | r2 c1 | r5 c0,6' (the list itself is complete; list(cells) prints every pair)."""

    def tolist(self) -> list:
        """A plain list of (r, c) tuples (Cells already is a list; this is for code written for numpy)."""
        return list(self)

    def __repr__(self) -> str:
        if len(self) <= _MAX_CELLS:
            return list.__repr__(self)
        rows: dict[int, set] = defaultdict(set)
        for r, c in self:
            rows[r].add(c)
        dup = len(self) - sum(len(v) for v in rows.values())
        parts = [f"r{r} c{_col_runs(sorted(cs))}" for r, cs in sorted(rows.items())]
        more = f" | ... +{len(parts) - _MAX_CELL_ROWS} rows" if len(parts) > _MAX_CELL_ROWS else ""
        return (f"{len(self)} cells" + (f" ({dup} repeated)" if dup else "") + ": "
                + " | ".join(parts[:_MAX_CELL_ROWS]) + more)


def where(g, color: int) -> Cells:
    """[(r, c)] of the cells with that colour, in reading order."""
    a = _grid(g)
    return Cells((int(r), int(c)) for r, c in np.argwhere(a == operator.index(color)))


def count(g) -> dict[int, int]:
    """{colour: number of cells}."""
    vals, n = np.unique(_grid(g), return_counts=True)
    return {int(v): int(k) for v, k in zip(vals, n)}


def bbox_of(g, bg=None) -> Box:
    """(r0, c0, r1, c1), inclusive, of the non-background cells."""
    a = _grid(g)
    cells = np.argwhere(a != _bg(a, bg))
    if not len(cells):
        raise ValueError(f"bbox_of: the grid has no cell other than its background {_bg(a, bg)} (pass bg= if that is wrong)")
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    return Box(r0, c0, r1, c1)


def crop(g, bbox) -> np.ndarray:
    """The sub-grid inside bbox (r0, c0, r1, c1), inclusive; bbox may be an Obj (its bbox)."""
    a = _grid(g)
    r0, c0, r1, c1 = _box(bbox)
    if not (0 <= r0 <= r1 < a.shape[0] and 0 <= c0 <= c1 < a.shape[1]):
        raise ValueError(f"crop: bbox {(r0, c0, r1, c1)} does not fit a grid of shape {a.shape}")
    return a[r0:r1 + 1, c0:c1 + 1].copy()


def background(g) -> int:
    """The most frequent colour (the smallest one on ties) -- what bg=None means everywhere."""
    return int(_bg(_grid(g), None))


class Changes(list):
    """A list of (r, c, old, new); print() lists up to 20 of them, more as a summary: how many cells change from which
    colour to which, and the clusters of touching changed cells (the list itself is complete)."""

    def __repr__(self) -> str:
        if len(self) <= 20:
            return list.__repr__(self)
        kinds = Counter((o, n) for _, _, o, n in self)
        out = [f"{len(self)} changed cells: " + ", ".join(f"{o}->{n} x{k}" for (o, n), k in kinds.most_common(8))
               + (f" ... {len(kinds) - 8} more kinds" if len(kinds) > 8 else "")]
        cells = [(r, c) for r, c, _, _ in self]
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        m = np.zeros((max(r for r, _ in cells) - r0 + 1, max(c for _, c in cells) - c0 + 1), bool)
        for r, c in cells:
            m[r - r0, c - c0] = True
        cl = _components(m, m, False, 8)
        val = {(r, c): (o, n) for r, c, o, n in self}
        for k in cl[:_MAX_CLUSTERS]:
            rs, cs = [r + r0 for r, _ in k], [c + c0 for _, c in k]
            kk = Counter(val[(r + r0, c + c0)] for r, c in k)
            out.append(f"  rows {min(rs)}-{max(rs)} cols {min(cs)}-{max(cs)}: {len(k)} cells "
                       + ", ".join(f"{o}->{n} x{n_}" for (o, n), n_ in kk.most_common(3)))
        if len(cl) > _MAX_CLUSTERS:
            out.append(f"  ... {len(cl) - _MAX_CLUSTERS} more clusters")
        return "\n".join(out)


def changes(a, b) -> Changes:
    """[(r, c, colour in a, colour in b)] for every cell where two same-size grids differ, in reading order.
    Printing it summarises more than 20 changes (by colour change and by cluster)."""
    a, b = _grid(a, "a"), _grid(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"changes: sizes differ, {a.shape} vs {b.shape}")
    return Changes((int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in np.argwhere(a != b))


def color_map(a, b) -> dict[int, int] | None:
    """{colour in a: colour in b} if b is a recolouring of a cell by cell (same shape, one target per colour), else None."""
    a, b = _grid(a, "a"), _grid(b, "b")
    if a.shape != b.shape:
        return None
    m = {}
    for x, y in zip(a.ravel().tolist(), b.ravel().tolist()):
        if m.setdefault(x, y) != y:
            return None
    return dict(sorted(m.items()))


# --------------------------------------------------------------------------- objects

class Obj:
    """One connected component (or region). idx = position in the list it came from."""

    def __init__(self, idx: int, a: np.ndarray, cells: list[tuple[int, int]], bg=None):
        if not cells:
            raise ValueError("Obj needs at least one cell")
        self.idx = idx
        self.cells = Cells(cells)
        self.size = len(cells)
        self.bg = bg
        rows, cols = [r for r, _ in cells], [c for _, c in cells]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        self.bbox = Box(r0, c0, r1, c1)
        self.h, self.w = r1 - r0 + 1, c1 - c0 + 1
        self.mask = np.zeros((self.h, self.w), bool)
        self.mask[[r - r0 for r in rows], [c - c0 for c in cols]] = True
        vals = [int(a[r, c]) for r, c in cells]
        self.colors = {v: vals.count(v) for v in sorted(set(vals))}
        self.color = max(self.colors, key=self.colors.get)      # majority colour, smallest on ties
        self._vals = a[r0:r1 + 1, c0:c1 + 1].copy()
        self._gshape = a.shape

    @property
    def rel(self) -> Cells:
        """The cells relative to the bbox's top-left corner: [(r - r0, c - c0)] (the shape wherever it stands)."""
        r0, c0 = self.bbox[0], self.bbox[1]
        return Cells((r - r0, c - c0) for r, c in self.cells)

    def grid(self, bg=None) -> np.ndarray:
        """The bbox with the object's colours, and bg (default: the bg objects() used; -1 if none) elsewhere."""
        fill = (self.bg if self.bg is not None else _EMPTY) if bg is None else operator.index(bg)
        return np.where(self.mask, self._vals, fill)

    @cached_property
    def hole_cells(self) -> Cells:
        """Cells inside the bbox that are not the object's and cannot reach the bbox edge (4-connected): its holes."""
        r0, c0, _, _ = self.bbox
        out = []
        for cl in _components(self.mask, ~self.mask, False, 4):
            if all(0 < r < self.h - 1 and 0 < c < self.w - 1 for r, c in cl):
                out += [(r + r0, c + c0) for r, c in cl]
        return Cells(sorted(out))

    @cached_property
    def holes(self) -> int:
        """Number of separate holes (4-connected groups of hole_cells)."""
        if not self.hole_cells:
            return 0
        m = np.zeros((self.h, self.w), bool)
        r0, c0, _, _ = self.bbox
        for r, c in self.hole_cells:
            m[r - r0, c - c0] = True
        return len(_components(m, m, False, 4))

    @property
    def touches(self) -> tuple[str, ...]:
        """The grid edges the object touches, from ('top', 'bottom', 'left', 'right')."""
        r0, c0, r1, c1 = self.bbox
        H, W = self._gshape
        return tuple(name for name, hit in (("top", r0 == 0), ("bottom", r1 == H - 1),
                                            ("left", c0 == 0), ("right", c1 == W - 1)) if hit)

    @property
    def is_rect(self) -> bool:
        """True if the object fills its bbox."""
        return bool(self.mask.all())

    @property
    def is_frame(self) -> bool:
        """True if the object covers the whole border of its bbox (at least 3x3) but not all of the inside."""
        if self.h < 3 or self.w < 3:
            return False
        m = self.mask
        return bool(m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all() and not m.all())

    @property
    def interior(self) -> Box:
        """The bbox shrunk by one cell on every side (the inside of a frame); raises if the bbox is under 3x3."""
        if self.h < 3 or self.w < 3:
            raise ValueError(f"interior: object #{self.idx} is {self.h}x{self.w}, it has no inside")
        r0, c0, r1, c1 = self.bbox
        return Box(r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    rows = property(lambda self: self.bbox.rows)
    cols = property(lambda self: self.bbox.cols)
    slices = property(lambda self: self.bbox.slices)

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the bbox (row, col); .5 when the bbox has an even side."""
        r0, c0, r1, c1 = self.bbox
        return (r0 + r1) / 2, (c0 + c1) / 2

    def encloses(self, other) -> bool:
        """True if every cell of `other` (an Obj or cells) lies in this object's holes."""
        hole = set(self.hole_cells)
        return bool(hole) and all(p in hole for p in _cells_of(other))

    def draw(self) -> list[str]:
        """The mask as text rows: the cell's colour digit, '.' where the object has no cell."""
        return ["".join(str(int(v)) if m else "." for v, m in zip(vrow, mrow)) for vrow, mrow in zip(self._vals, self.mask)]

    def __repr__(self) -> str:
        r0, c0, r1, c1 = self.bbox
        extra = f" colors {self.colors}" if len(self.colors) > 1 else ""
        flags = " rect" if self.is_rect and self.size > 1 else " frame" if self.is_frame else ""
        holes = f" holes {self.holes}" if self.holes else ""
        return (f"#{self.idx} color {self.color} size {self.size} bbox r{r0}-{r1} c{c0}-{c1} hw {self.h}x{self.w}"
                f"{extra}{flags}{holes}")


def objects(g, bg=None, conn: int = 8, multicolor: bool = False, gap: int = 0) -> list[Obj]:
    """Connected components of the non-bg cells (bg = most frequent colour, or the colour / list of colours given;
    bg=-1: no background), sorted by (top row, left col). gap=k also joins cells with up to k bg cells between them
    (conn=8: in any direction, conn=4: along a row or column)."""
    a = _grid(g)
    b, bgs = _bgs(a, bg)
    comps = _components(a, ~np.isin(a, list(bgs)), not multicolor, conn, gap)
    return [Obj(i, a, cl, bg=int(b)) for i, cl in enumerate(comps)]


def by_color(g, bg=None) -> dict[int, Obj]:
    """{colour: Obj of all the cells of that colour}, for every colour but the background (bg as in objects; the
    cells need not touch)."""
    a = _grid(g)
    b, bgs = _bgs(a, bg)
    out = {}
    for i, v in enumerate(sorted(set(np.unique(a).tolist()) - bgs)):
        out[int(v)] = Obj(i, a, [(int(r), int(c)) for r, c in np.argwhere(a == v)], bg=int(b))
    return out


def print_objects(g, bg=None, conn: int = 8, multicolor: bool = False, gap: int = 0) -> None:
    """One line per object, plus its mask when the bbox is at most 6x6."""
    a = _grid(g)
    b, bgs = _bgs(a, bg)
    objs = objects(a, sorted(bgs) if len(bgs) > 1 else b, conn, multicolor, gap)
    shown = sorted(bgs) if len(bgs) > 1 else int(b)
    print(f"{len(objs)} objects (bg {shown}, conn {conn}{f', gap {gap}' if gap else ''}, "
          f"{'multicolor' if multicolor else 'same-colour'})")
    note = _bg_close_call(a) if bg is None else None
    if note:
        print("  " + note)
    for o in objs[:_MAX_OBJECTS]:
        print(o)
        if 1 < o.h * o.w and o.h <= _MAX_MASK and o.w <= _MAX_MASK:
            print("\n".join("    " + row for row in o.draw()))
    if len(objs) > _MAX_OBJECTS:
        print(f"... {len(objs) - _MAX_OBJECTS} more")


def labels(g, bg=None, conn: int = 8, multicolor: bool = False) -> np.ndarray:
    """An int array of the grid's shape, numbered like scipy.ndimage.label: 0 on bg, k + 1 on the cells of
    objects(...)[k] (so 1..n)."""
    a = _grid(g)
    out = np.zeros(a.shape, np.int64)
    for o in objects(a, bg, conn, multicolor):
        for r, c in o.cells:
            out[r, c] = o.idx + 1
    return out


def components(x, conn: int = 8, gap: int = 0) -> list[Cells]:
    """Connected groups of a bool mask's True cells, or of a set of (r, c) cells, sorted by (top row, left col).
    gap=k also joins cells with up to k other cells between them (conn=8: any direction, conn=4: a row or column)."""
    a = np.asarray(x)
    if a.dtype == bool and a.ndim == 2:
        m = a
        off = (0, 0)
    else:
        cells = _cells_of(x)
        if not cells:
            return []
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        m = np.zeros((max(r for r, _ in cells) - r0 + 1, max(c for _, c in cells) - c0 + 1), bool)
        for r, c in cells:
            m[r - r0, c - c0] = True
        off = (r0, c0)
    return [Cells((r + off[0], c + off[1]) for r, c in cl) for cl in _components(m, m, False, conn, gap)]


def adjacency(objs, conn: int = 4) -> dict[int, list[int]]:
    """{idx: [idx of every other object that touches it]} for a list of Obj; conn=4: sharing a side, 8: or a corner."""
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    owner = {}
    for o in objs:
        for p in o.cells:
            owner[p] = o.idx
    out = {o.idx: set() for o in objs}
    for o in objs:
        for r, c in o.cells:
            for dr, dc in _STEPS[conn]:
                j = owner.get((r + dr, c + dc))
                if j is not None and j != o.idx:
                    out[o.idx].add(j)
    return {k: sorted(v) for k, v in out.items()}


def cluster(items, linked) -> list[list]:
    """items (any list: Obj, cells, numbers, ...) split into groups: two items are in one group if linked(a, b) is
    True, or a chain of linked items joins them (single linkage). Groups keep the items' order and are sorted by
    their first item. e.g. cluster(objs, lambda a, b: gap(a, b).cheb <= 2)."""
    items = list(items)
    n = len(items)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if root(i) != root(j) and linked(items[i], items[j]):
                parent[root(j)] = root(i)
    groups_: dict[int, list] = {}
    for i in range(n):
        groups_.setdefault(root(i), []).append(i)
    return [[items[i] for i in g] for g in sorted(groups_.values())]


class Gap:
    """How far apart two sets of cells are. rows / cols: empty rows / columns strictly between their bboxes (0 when
    the bboxes share or touch that axis); cheb: smallest max(|dr|, |dc|) over cell pairs (1 = touching, diagonals
    too; 0 = a shared cell); manh: smallest |dr| + |dc| (1 = sharing a side); dr, dc: the sign of b's position
    relative to a (-1 above / left, 1 below / right, 0 overlapping on that axis)."""

    def __init__(self, rows, cols, cheb, manh, dr, dc):
        self.rows, self.cols, self.cheb, self.manh, self.dr, self.dc = rows, cols, cheb, manh, dr, dc

    def __repr__(self) -> str:
        side = {(-1, 0): "above", (1, 0): "below", (0, -1): "left", (0, 1): "right", (0, 0): "overlapping"}.get(
            (self.dr, self.dc), f"diagonal ({self.dr:+d}, {self.dc:+d})")
        return (f"Gap(rows={self.rows}, cols={self.cols}, cheb={self.cheb}, manh={self.manh}; b is {side} of a)")


def gap(a, b) -> Gap:
    """The distance between a and b (Obj, Hit, cells or one cell (r, c), Box): see Gap."""
    def cells(x):
        if isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, (int, np.integer)) for v in x):
            return [(int(x[0]), int(x[1]))]
        if isinstance(x, Box):
            return [(r, c) for r in x.rows for c in x.cols]
        return _cells_of(x)
    A, B = cells(a), cells(b)
    if not A or not B:
        raise ValueError("gap: both sides need at least one cell")
    ar0, ar1 = min(r for r, _ in A), max(r for r, _ in A)
    ac0, ac1 = min(c for _, c in A), max(c for _, c in A)
    br0, br1 = min(r for r, _ in B), max(r for r, _ in B)
    bc0, bc1 = min(c for _, c in B), max(c for _, c in B)
    rows = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    cols = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    dr = 1 if br0 > ar1 else -1 if br1 < ar0 else 0
    dc = 1 if bc0 > ac1 else -1 if bc1 < ac0 else 0
    pa, pb = np.array(A), np.array(B)
    d = np.abs(pa[:, None, :] - pb[None, :, :])
    return Gap(rows, cols, int(d.max(axis=2).min()), int(d.sum(axis=2).min()), dr, dc)


# --------------------------------------------------------------------------- orientation and scale

def orient(g, name: str) -> np.ndarray:
    """g turned by one of ORIENTS: id rot90 rot180 rot270 (np.rot90 = counter-clockwise) flipud fliplr transpose antitranspose."""
    if name not in _ORIENT:
        raise ValueError(f"orient: name must be one of {ORIENTS}, got {name!r}")
    if isinstance(g, (Obj, Hit, Panel)):
        raise TypeError("orient takes a grid or mask: use o.grid(), o.grid(-1) or o.mask for an Obj")
    if isinstance(g, Cells):
        raise TypeError("orient takes a grid or mask, not a list of (r, c) cells: use o.mask or o.grid() of the Obj")
    a = np.asarray(g)
    if a.ndim != 2:
        raise ValueError(f"orient: needs a 2-D grid, got shape {a.shape}")
    return _ORIENT[name](a).copy()


def inverse(name: str) -> str:
    """The orientation that undoes `name` (rot90 <-> rot270; the others undo themselves)."""
    if name not in _INVERSE:
        raise ValueError(f"inverse: name must be one of {ORIENTS}, got {name!r}")
    return _INVERSE[name]


def _pair(k, what: str) -> tuple[int, int]:
    kr, kc = (k, k) if isinstance(k, (int, np.integer)) else k
    kr, kc = operator.index(kr), operator.index(kc)
    if kr < 1 or kc < 1:
        raise ValueError(f"{what}: factors must be >= 1, got {(kr, kc)}")
    return kr, kc


def upscale(g, k) -> np.ndarray:
    """Every cell becomes a k x k block (k = int, or (rows, cols))."""
    kr, kc = _pair(k, "upscale")
    return np.asarray(g).repeat(kr, axis=0).repeat(kc, axis=1)


def downscale(g, k, how: str = "exact") -> np.ndarray:
    """Each k x k block becomes one cell. how="exact": the inverse of upscale, raises if a block is not uniform;
    "majority": the block's most frequent colour (smallest on ties)."""
    a = np.asarray(g)
    kr, kc = _pair(k, "downscale")
    if a.shape[0] % kr or a.shape[1] % kc:
        raise ValueError(f"downscale: shape {a.shape} is not a multiple of {(kr, kc)}")
    if how == "exact":
        small = a[::kr, ::kc]
        if not np.array_equal(upscale(small, (kr, kc)), a):
            raise ValueError(f"downscale: the grid is not made of uniform {kr}x{kc} blocks (how='majority' votes)")
        return small.copy()
    if how != "majority":
        raise ValueError(f"downscale: how must be 'exact' or 'majority', got {how!r}")
    H, W = a.shape[0] // kr, a.shape[1] // kc
    out = np.zeros((H, W), a.dtype)
    for i in range(H):
        for j in range(W):
            vals, n = np.unique(a[i * kr:(i + 1) * kr, j * kc:(j + 1) * kc], return_counts=True)
            out[i, j] = vals[n.argmax()]
    return out


def _run_gcd(same_as_next) -> int:
    lengths, n = [], 1
    for same in same_as_next:
        if same:
            n += 1
        else:
            lengths.append(n)
            n = 1
    lengths.append(n)
    return reduce(gcd, lengths)


def scale_of(g) -> tuple[int, int]:
    """The largest (kr, kc) with g == upscale(downscale(g, (kr, kc)), (kr, kc)); (1, 1) if g is not a block upscale."""
    a = _grid(g)
    kr = _run_gcd([np.array_equal(a[i], a[i + 1]) for i in range(a.shape[0] - 1)])
    kc = _run_gcd([np.array_equal(a[:, j], a[:, j + 1]) for j in range(a.shape[1] - 1)])
    return kr, kc


def squeeze(g) -> np.ndarray:
    """g with every run of identical consecutive rows kept once, then every run of identical consecutive columns:
    a picture drawn in blocks of different sizes becomes one cell per block (downscale needs equal blocks)."""
    a = _grid(g)
    keep_r = [0] + [i for i in range(1, a.shape[0]) if not np.array_equal(a[i], a[i - 1])]
    a = a[keep_r]
    keep_c = [0] + [j for j in range(1, a.shape[1]) if not np.array_equal(a[:, j], a[:, j - 1])]
    return a[:, keep_c].copy()


# --------------------------------------------------------------------------- shapes

def _patch(x, bg, what: str) -> np.ndarray:
    """x cropped to its cells' bbox as an int array, _EMPTY where x has no cell.
    x = Obj (its cells), Cells (value 1), bool mask (True cells, value 1), or grid (cells != bg; bg None = its most
    frequent colour)."""
    if isinstance(x, Obj):
        return np.where(x.mask, x._vals, _EMPTY)
    if isinstance(x, Cells):
        if not x:
            raise ValueError(f"{what} has no cell (an empty list of cells)")
        return np.where(_cells_mask(x)[0], 1, _EMPTY)
    a = np.asarray(x)
    if a.dtype == bool:
        m = _grid(a, what)
        vals = m.astype(np.int64)
    else:
        vals = _grid(a, what).astype(np.int64)
        m = vals != _bg(vals, bg)
    cells = np.argwhere(m)
    if not len(cells):
        raise ValueError(f"{what} has no cell (all background)")
    (r0, c0), (r1, c1) = cells.min(axis=0), cells.max(axis=0)
    return np.where(m, vals, _EMPTY)[r0:r1 + 1, c0:c1 + 1]


def _relabel(p: np.ndarray) -> np.ndarray:
    """Colours renumbered 0, 1, ... in order of first appearance (reading order); _EMPTY stays."""
    order = {}
    for v in p.ravel().tolist():
        if v != _EMPTY and v not in order:
            order[v] = len(order)
    return np.vectorize(lambda v: order.get(v, _EMPTY), otypes=[np.int64])(p)


def shape_key(x, flips: bool = True, scale: bool = False, color=False, bg=0) -> tuple:
    """A hashable key that is equal for equal shapes.
    flips: the same under any of the 8 rotations / flips.  scale: the same under integer upscaling (one factor for
    rows and columns, so filled rectangles with the same side ratio share a key).
    color: False = cells only (colours ignored), True = colours must match too, "relabel" = same colour pattern up to
    a renaming of colours.  x = Obj, bool mask, or grid (its cells != bg; bg=None: its most frequent colour).
    The key is the canonical drawing as a tuple of rows (1 / colour = cell, 0 / -1 = no cell)."""
    if color not in (False, True, "relabel"):
        raise ValueError(f"shape_key: color must be False, True or 'relabel', got {color!r}")
    p = _patch(x, bg, "shape_key")
    p = p if color else (p != _EMPTY).astype(np.int64)
    if scale:
        k = gcd(*scale_of(p))                 # the same factor for rows and columns: a scaled copy, not a stretch
        p = p[::k, ::k]
    best = None
    for name in (ORIENTS if flips else ("id",)):
        q = orient(p, name)
        if color == "relabel":
            q = _relabel(q)
        k = (q.shape, tuple(q.ravel().tolist()))
        if best is None or k < best[0]:
            best = (k, q)
    return tuple(tuple(row) for row in best[1].tolist())


def _ratios(small, big) -> list[int]:
    """Integer factors k >= 2 by which a shape of size `small` could be scaled up to size `big` (either orientation)."""
    ks = {big[i] // small[j] for i in (0, 1) for j in (0, 1) if big[i] % small[j] == 0 and big[i] // small[j] >= 2}
    return sorted(ks)


def _names(names: list[str]) -> str:
    """shape_match names as one label: 'rot180|fliplr' (the ways are the same cells); all 8 orientations (a
    symmetric shape) are shown as just the scale ('*2')."""
    by_scale: dict[str, list[str]] = {}
    for n in names:
        cut = max(n.find("*"), n.find("/"))
        base, suf = (n[:cut], n[cut:]) if cut > 0 else (n, "")
        by_scale.setdefault(suf, []).append(base)
    return " or ".join((suf if len(bases) == 8 and suf else "|".join(bases) + suf) for suf, bases in by_scale.items())


def shape_match(a, b, bg=0, color: bool = False) -> list[str]:
    """Which of ORIENTS turn shape a into shape b: "name" (same size), "name*k" (a scaled up k times first),
    "name/k" (a scaled down k times first; k is read off the sizes). a, b = Obj, bool mask or grid (cells != bg; bg=None: each grid's most
    frequent colour), cropped to their bbox.
    color=False compares cells only; color=True the colours too."""
    pa, pb = _patch(a, bg, "shape_match a"), _patch(b, bg, "shape_match b")
    if not color:
        pa, pb = (pa != _EMPTY).astype(np.int64), (pb != _EMPTY).astype(np.int64)
    out = []
    cands = [("", pa)] + [(f"*{k}", upscale(pa, k)) for k in _ratios(pa.shape, pb.shape)]
    for k in _ratios(pb.shape, pa.shape):
        if pa.shape[0] % k == 0 and pa.shape[1] % k == 0 and np.array_equal(upscale(pa[::k, ::k], k), pa):
            cands.append((f"/{k}", pa[::k, ::k]))
    for suffix, q in cands:
        for name, f in _DIHEDRAL:
            t = f(q)
            if t.shape == pb.shape and np.array_equal(t, pb):
                out.append(name + suffix)
    return out


class Group:
    """Objects with the same shape key. relation[i] = how members[0] turns into members[i] (shape_match names)."""

    def __init__(self, key, members: list[Obj], relation: list[str]):
        self.key, self.members, self.relation = key, members, relation

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        o = self.members[0]
        rect = " filled rectangles" if all(m.is_rect for m in self.members) else ""
        listed = "  ".join(f"#{m.idx} c{m.color} {r}" for m, r in zip(self.members[:_MAX_MEMBERS], self.relation))
        more = f"  ... +{len(self) - _MAX_MEMBERS}" if len(self) > _MAX_MEMBERS else ""
        return f"group of {len(self)}: {o.h}x{o.w}{rect}  {listed}{more}"


def _as_objs(x, bg, conn, multicolor) -> list[Obj]:
    if isinstance(x, (list, tuple)) and all(isinstance(o, Obj) for o in x):
        return list(x)              # [] is an empty list of objects
    return objects(x, bg, conn, multicolor)


def _relation(x: Obj, y: Obj, color) -> str:
    """How x turns into y for groups(): shape_match names; with color="relabel", names that hold after renaming
    colours, marked "~" (renamed); "scaled" when only a scaling with different row / column factors relates them."""
    if color != "relabel":
        n = shape_match(x, y, color=color is True)
        return _names(n) if n else "scaled"
    px, py = _relabel(_patch(x, None, "x")), _patch(y, None, "y")
    names = []
    for k in [1] + _ratios(px.shape, py.shape):
        up = upscale(px, k)
        names += [n + ("" if k == 1 else f"*{k}") for n in ORIENTS
                  if orient(up, n).shape == py.shape and np.array_equal(_relabel(orient(up, n)), _relabel(py))]
    if not names:
        return "scaled"
    exact = shape_match(x, y, color=True)
    return _names(exact) if exact else _names(names) + "~"


def groups(x, by: str = "shape", flips: bool = True, scale: bool = False, bg=None, conn: int = 8,
           multicolor: bool = False) -> list[Group]:
    """Objects (of grid x, or a list of Obj) grouped by equal shape; largest group first.
    by: "shape" (cells only), "shape+color" (colours too), "pattern" (same colour pattern up to renaming colours)."""
    color = {"shape": False, "shape+color": True, "pattern": "relabel"}.get(by)
    if color is None:
        raise ValueError(f"groups: by must be 'shape', 'shape+color' or 'pattern', got {by!r}")
    objs = _as_objs(x, bg, conn, multicolor)
    buckets: dict[tuple, list[Obj]] = {}
    for o in objs:
        buckets.setdefault(shape_key(o, flips, scale, color), []).append(o)
    out = []
    for key, ms in buckets.items():
        rel = ["id"] + [_relation(ms[0], m, color) for m in ms[1:]]
        out.append(Group(key, ms, rel))
    out.sort(key=lambda g: (-len(g), g.members[0].idx))
    return out


def print_groups(x, by: str = "shape", flips: bool = True, scale: bool = False, bg=None, conn: int = 8,
                 multicolor: bool = False) -> None:
    """groups() as text: one line per group of 2 or more, then the objects whose shape is unique."""
    gs = groups(x, by, flips, scale, bg, conn, multicolor)
    n = sum(len(g) for g in gs)
    if not gs:
        print("0 objects")
        return
    multi = [g for g in gs if len(g) > 1]
    single = [g.members[0] for g in gs if len(g) == 1]
    print(f"{n} objects in {len(gs)} groups by {by} ({'rotations/flips allowed' if flips else 'no rotations'}"
          f"{', scaling allowed' if scale else ''})")
    for g in multi[:_MAX_OBJECTS]:
        print(f"  {g}")
        o = g.members[0]
        if o.h <= _MAX_MASK and o.w <= _MAX_MASK and o.h * o.w > 1:
            print("\n".join("      " + row for row in o.draw()))
    if single:
        print("  unique: " + "  ".join(f"#{o.idx} c{o.color} {o.h}x{o.w}" for o in single[:_MAX_OBJECTS])
              + (f" ... +{len(single) - _MAX_OBJECTS}" if len(single) > _MAX_OBJECTS else ""))


class Hit:
    """One place find() matched: top-left (r, c) of the placed pattern, its size, the orientation / scale used,
    and .cells: the grid cells the pattern's compared cells landed on."""

    def __init__(self, r, c, h, w, orient_name, scale, colors, mask=None):
        self.r, self.c, self.h, self.w, self.orient, self.scale, self.colors = r, c, h, w, orient_name, scale, colors
        self._mask = mask if mask is not None else np.ones((h, w), bool)

    @property
    def cells(self) -> Cells:
        return Cells((self.r + int(y), self.c + int(x)) for y, x in np.argwhere(self._mask))

    @property
    def bbox(self) -> Box:
        return Box(self.r, self.c, self.r + self.h - 1, self.c + self.w - 1)

    r0 = property(lambda self: self.r)
    c0 = property(lambda self: self.c)
    r1 = property(lambda self: self.r + self.h - 1)
    c1 = property(lambda self: self.c + self.w - 1)
    slices = property(lambda self: self.bbox.slices)

    def __repr__(self) -> str:
        how = self.orient + (f"*{self.scale}" if self.scale > 1 else "")
        cm = f" colours {self.colors}" if self.colors else ""
        return f"at ({self.r},{self.c}) {self.h}x{self.w} {how}{cm}"


def find(g, pattern, flips: bool = False, scales=(1,), recolor: bool = False, bg=None, alone: bool = False) -> list[Hit]:
    """Every place where `pattern` occurs in g. pattern = Obj (its cells, colours), or a grid in which -1 marks
    "any value" (cells that are not compared). flips: try all 8 orientations. scales: upscale factors to try.
    recolor: allow a one-to-one renaming of the pattern's colours (Hit.colors gives it); the grid's bg (bg=None: its
    most frequent colour) is never renamed, so a recoloured hit is the shape in other colours, not a patch of
    background. alone=True: only hits whose cells touch (4-neighbours) no other non-bg cell -- the shape standing
    on its own, not part of a bigger object. Overlapping hits are all listed; a symmetric pattern is listed once
    per place."""
    a = _grid(g)
    gbg = _bg(a, bg)
    if not isinstance(pattern, Obj) and np.asarray(pattern).dtype == bool:
        raise TypeError("find: a bool mask has no colours -- pass the Obj, or a grid with -1 where any value may be")
    if isinstance(scales, (int, np.integer)):
        scales = (scales,)
    p = _patch(pattern, bg, "find pattern") if isinstance(pattern, Obj) else _grid(pattern, "find pattern").astype(np.int64)
    if (p == _EMPTY).all():
        raise ValueError("find: the pattern has no compared cell")
    hits, seen = [], set()
    for k in scales:
        k = operator.index(k)
        for name in (ORIENTS if flips else ("id",)):
            q = orient(upscale(p, k), name)
            h, w = q.shape
            if h > a.shape[0] or w > a.shape[1]:
                continue
            m = q != _EMPTY
            qv = q[m]
            for r in range(a.shape[0] - h + 1):
                for c in range(a.shape[1] - w + 1):
                    win = a[r:r + h, c:c + w][m]
                    if recolor:
                        fwd, back = {}, {}
                        ok = True
                        for x, y in zip(qv.tolist(), win.tolist()):
                            if (x == gbg) != (y == gbg) or fwd.setdefault(x, y) != y or back.setdefault(y, x) != x:
                                ok = False
                                break
                        if not ok:
                            continue
                        cmap = {x: y for x, y in sorted(fwd.items()) if x != y}
                    else:
                        if not np.array_equal(win, qv):
                            continue
                        cmap = {}
                    sig = tuple(sorted(((r + int(y), c + int(x)), int(q[y, x])) for y, x in np.argwhere(m)))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    hit = Hit(r, c, h, w, name, k, cmap, m)
                    if alone:
                        own = set(hit.cells)
                        ring = {(y + dy, x + dx) for y, x in own for dy, dx in _STEPS[4]} - own
                        if any(0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x] != gbg for y, x in ring):
                            continue
                    hits.append(hit)
    hits.sort(key=lambda t: (t.r, t.c))
    return hits


# --------------------------------------------------------------------------- layout

def separators(g) -> dict[str, dict[int, int]]:
    """Rows and columns that are a single colour all the way across: {"rows": {r: colour}, "cols": {c: colour}}."""
    a = _grid(g)
    rows = {r: int(a[r, 0]) for r in range(a.shape[0]) if (a[r] == a[r, 0]).all()}
    cols = {c: int(a[0, c]) for c in range(a.shape[1]) if (a[:, c] == a[0, c]).all()}
    return {"rows": rows, "cols": cols}


class Panel:
    """One cell of a grid split by separator lines: band indices (i, j), bbox in the full grid, and the sub-grid."""

    def __init__(self, i, j, bbox, grid):
        self.i, self.j, self.bbox, self.grid = i, j, bbox, grid

    r0 = property(lambda self: self.bbox.r0)
    c0 = property(lambda self: self.bbox.c0)
    r1 = property(lambda self: self.bbox.r1)
    c1 = property(lambda self: self.bbox.c1)
    h = property(lambda self: self.bbox.h)
    w = property(lambda self: self.bbox.w)
    slices = property(lambda self: self.bbox.slices)

    def __repr__(self) -> str:
        r0, c0, r1, c1 = self.bbox
        return f"panel ({self.i},{self.j}) rows {r0}-{r1} cols {c0}-{c1} ({r1 - r0 + 1}x{c1 - c0 + 1})"


def _spans(n: int, lines: set[int]) -> list[tuple[int, int]]:
    out, start = [], None
    for i in range(n + 1):
        if i < n and i not in lines:
            start = i if start is None else start
        elif start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _line_cells(a: np.ndarray, sep, color: int) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[[r for r, v in sep["rows"].items() if v == color], :] = True
    m[:, [c for c, v in sep["cols"].items() if v == color]] = True
    return m


def _groups(idx) -> int:
    """How many runs of consecutive numbers idx holds (a thick line counts once)."""
    idx = sorted(idx)
    return sum(1 for i, v in enumerate(idx) if i == 0 or v != idx[i - 1] + 1)


def _lattice(ps: list) -> bool:
    """Panels that form a regular lattice: at least 4, all the same size, each at least 2 cells."""
    return len(ps) >= 4 and len({(p.h, p.w) for p in ps}) == 1 and ps[0].h * ps[0].w >= 2


class Line:
    """One row or column that is mostly one colour: axis "row" / "col", index i, colour, n cells of it out of `of`,
    marks [(position along the line, colour)] for its other cells."""

    def __init__(self, axis, i, color, n, of, marks):
        self.axis, self.i, self.color, self.n, self.of, self.marks = axis, i, color, n, of, marks

    @property
    def cells(self) -> Cells:
        return Cells((self.i, k) if self.axis == "row" else (k, self.i) for k in range(self.of))

    def __repr__(self) -> str:
        other = ("; other " + " ".join(f"{'c' if self.axis == 'row' else 'r'}{k}={v}" for k, v in self.marks[:12])
                 + (f" ... +{len(self.marks) - 12}" if len(self.marks) > 12 else "")) if self.marks else ""
        return f"{self.axis} {self.i}: colour {self.color}, {self.n}/{self.of}{other}"


def lines(g, min_frac: float = 0.6, bg=None) -> list[Line]:
    """Rows and columns (of at least 3 cells) whose most common colour fills at least min_frac of them -- separator
    lines even where marks or crossing lines interrupt them -- rows first, then columns. Lines of the background
    (bg=None: the most frequent colour; bg=-1: keep every colour) are left out."""
    a = _grid(g)
    if not 0 < min_frac <= 1:
        raise ValueError(f"lines: min_frac must be in (0, 1], got {min_frac}")
    skip = set() if bg is not None and not isinstance(bg, (list, tuple, set)) and operator.index(bg) == -1 \
        else _bgs(a, bg)[1]
    out = []
    for axis, b in (("row", a), ("col", a.T)):
        if b.shape[1] < 3:
            continue
        for i in range(b.shape[0]):
            vals, n = np.unique(b[i], return_counts=True)
            v, k = int(vals[n.argmax()]), int(n.max())
            if v in skip or k < min_frac * b.shape[1]:
                continue
            out.append(Line(axis, i, v, k, b.shape[1], [(int(j), int(b[i, j])) for j in np.flatnonzero(b[i] != v)]))
    return out


def panels(g, color=None, min_frac: float = 1.0) -> list[Panel]:
    """The grid cut along full rows / columns of the separator colour (thick lines count once), in reading order.
    color=None: among the colours whose full lines give at least 2 panels, one other than the grid's most frequent
    colour first, then the most separate lines (a thick line or a solid band counts once), then the one least used
    inside the panels; the most frequent colour only if it never occurs inside its panels or its lines cut the grid
    into a regular lattice (4 or more equal panels). No such colour -> one panel, the whole grid.
    min_frac < 1 (needs color=): also cut on rows / columns that are only mostly that colour (marks or crossing lines
    on them, as lines() lists them)."""
    a = _grid(g)
    if not 0 < min_frac <= 1:
        raise ValueError(f"panels: min_frac must be in (0, 1], got {min_frac}")
    if min_frac < 1:
        if color is None:
            raise ValueError("panels: min_frac < 1 needs color= (the colour of the lines)")
        color = operator.index(color)
        rows = {r for r in range(a.shape[0]) if a.shape[1] >= 3 and np.mean(a[r] == color) >= min_frac}
        cols = {c for c in range(a.shape[1]) if a.shape[0] >= 3 and np.mean(a[:, c] == color) >= min_frac}
        return [Panel(i, j, Box(r0, c0, r1, c1), a[r0:r1 + 1, c0:c1 + 1].copy())
                for i, (r0, r1) in enumerate(_spans(a.shape[0], rows))
                for j, (c0, c1) in enumerate(_spans(a.shape[1], cols))]
    sep = separators(a)
    if color is None:
        usable = {"rows": sep["rows"] if a.shape[1] >= 3 else {}, "cols": sep["cols"] if a.shape[0] >= 3 else {}}
        n = Counter(list(usable["rows"].values()) + list(usable["cols"].values()))
        gbg = _bg(a, None)
        cands = []
        for col, k in n.items():
            rest = a[~_line_cells(a, usable, col)]
            ps = panels(a, col)
            if not rest.size or len(ps) < 2:
                continue
            inside = float(np.mean(rest == col))
            if col == gbg and (inside > 0 and not _lattice(ps) or min(p.h * p.w for p in ps) < 2):
                continue                  # blank background rows / columns between pictures (or dots) are no separators
            groups = _groups(r for r, v in usable["rows"].items() if v == col) + \
                _groups(c for c, v in usable["cols"].items() if v == col)
            cands.append((col == gbg, -groups, inside, col))   # not the background, most lines, least used inside
        if not cands:
            return [Panel(0, 0, Box(0, 0, a.shape[0] - 1, a.shape[1] - 1), a.copy())]
        color = min(cands)[3]
    color = operator.index(color)
    rows = {r for r, v in sep["rows"].items() if v == color}
    cols = {c for c, v in sep["cols"].items() if v == color}
    out = []
    for i, (r0, r1) in enumerate(_spans(a.shape[0], rows)):
        for j, (c0, c1) in enumerate(_spans(a.shape[1], cols)):
            out.append(Panel(i, j, Box(r0, c0, r1, c1), a[r0:r1 + 1, c0:c1 + 1].copy()))
    return out


def tiles(g, nr: int, nc: int) -> list[Panel]:
    """The grid cut into nr x nc equal blocks (no separator lines), as Panels in reading order; raises unless the
    sides divide evenly."""
    a = _grid(g)
    H, W = a.shape
    if nr < 1 or nc < 1 or H % nr or W % nc:
        raise ValueError(f"tiles: a {H}x{W} grid does not split into {nr}x{nc} equal blocks")
    h, w = H // nr, W // nc
    return [Panel(i, j, Box(i * h, j * w, i * h + h - 1, j * w + w - 1), a[i * h:(i + 1) * h, j * w:(j + 1) * w].copy())
            for i in range(nr) for j in range(nc)]


def symmetry(g, ignore=None) -> dict[str, int]:
    """For each orientation that keeps the grid's shape: how many cells differ from the grid turned that way
    (0 = symmetric). ignore: a colour (or colours) treated as unknown on either side, e.g. a mask over a pattern."""
    a = _grid(g)
    skip = _colors(ignore, "ignore") if ignore is not None else set()
    out = {}
    for name in ORIENTS[1:]:
        t = orient(a, name)
        if t.shape != a.shape:
            continue
        bad = a != t
        if skip:
            bad &= ~np.isin(a, list(skip)) & ~np.isin(t, list(skip))
        out[name] = int(bad.sum())
    return out


class Period:
    """A tiling: g[r][c] == tile[r % pr][c % pc] except at `wrong`. tile cells that only ignored cells cover are -1."""

    def __init__(self, pr, pc, tile, wrong, shape, closest=None):
        self.pr, self.pc, self.tile, self.wrong, self._shape = pr, pc, tile, wrong, shape
        self.closest = closest          # (pr, pc, wrong) of the best smaller tile when only the whole grid fits

    def tiled(self) -> np.ndarray:
        """The full grid rebuilt from the tile (fixes the wrong / ignored cells); raises if a tile cell is unknown."""
        if (self.tile == _EMPTY).any():
            raise ValueError("Period.tiled: some tile cells (-1 in .tile) are only covered by ignored cells -- the tile "
                             "is probably too small; try max_wrong= instead of ignore=, or axis=")
        H, W = self._shape
        return np.tile(self.tile, (-(-H // self.pr), -(-W // self.pc)))[:H, :W]

    def __repr__(self) -> str:
        H, W = self._shape
        out = f"period {self.pr}x{self.pc} ({H / self.pr:.3g} x {W / self.pc:.3g} repeats), {len(self.wrong)} cells off the tiling"
        if self.closest is not None:
            pr, pc, k = self.closest
            out += (f" -- no smaller tile fits; closest: {pr}x{pc} with {k} cells off "
                    f"(period(g, max_wrong={k}) gives it)")
        return out


def period(g, ignore=None, max_wrong: int = 0, axis=None) -> Period:
    """The smallest tile (by area, then rows) that tiles g from the top-left corner, with each tile cell the majority
    colour of the cells it covers, and at most max_wrong cells disagreeing. A tile side shorter than the grid has to
    repeat about 1.5 times or more (2 * side <= grid side + 1), so a last column that happens to equal the first is
    no period. ignore: colour(s) that do not vote and are
    never wrong (cells to be filled in). axis=0: repeat down the rows only (the tile is as wide as the grid), axis=1:
    across the columns only. The whole grid is always a period."""
    a = _grid(g).astype(np.int64)
    H, W = a.shape
    if operator.index(max_wrong) < 0:
        raise ValueError(f"period: max_wrong must be >= 0, got {max_wrong}")
    skip = np.isin(a, list(_colors(ignore, "ignore"))) if ignore is not None else np.zeros(a.shape, bool)
    vals, codes = np.unique(a, return_inverse=True)        # vote on colour codes (any ints, -1 included)
    codes = codes.reshape(a.shape)
    rr, cc = np.indices(a.shape)
    if axis not in (None, 0, 1):
        raise ValueError(f"period: axis must be None, 0 or 1, got {axis!r}")
    sides_r = [p for p in range(1, H + 1) if 2 * p <= H + 1 or p == H]   # a shorter side repeats ~1.5 times or more
    sides_c = [p for p in range(1, W + 1) if 2 * p <= W + 1 or p == W]
    if axis == 0:
        sides_c = [W]
    elif axis == 1:
        sides_r = [H]
    closest = None
    for pr, pc in sorted(((pr, pc) for pr in sides_r for pc in sides_c), key=lambda t: (t[0] * t[1], t[0])):
        cls = (rr % pr) * pc + (cc % pc)
        votes = np.zeros((pr * pc, len(vals)), np.int64)
        np.add.at(votes, (cls[~skip], codes[~skip]), 1)
        tile = np.where(votes.sum(axis=1) > 0, vals[votes.argmax(axis=1)], _EMPTY).reshape(pr, pc)
        wrong = (a != tile[rr % pr, cc % pc]) & ~skip
        k = int(wrong.sum())
        if k <= max_wrong:
            trivial = (pr, pc) == (H, W) or (axis == 0 and pr == H) or (axis == 1 and pc == W)
            return Period(pr, pc, tile, Cells((int(r), int(c)) for r, c in np.argwhere(wrong)), a.shape,
                          closest if trivial else None)
        # a smaller tile that a few noisy cells spoil: worth naming when nothing else fits
        if pr * pc >= 2 and (closest is None or k < closest[2]) and k <= 0.25 * a.size:
            closest = (pr, pc, k)
    raise AssertionError("unreachable: the whole grid is a period")


# --------------------------------------------------------------------------- regions

def regions(g, wall, conn: int = 4) -> list[Obj]:
    """Connected groups of the cells that are not wall (wall = colour, colours, or bool mask), as multicolour Obj.
    A region with .touches == () does not reach the grid edge: it is enclosed by the wall."""
    a = _grid(g)
    w = np.asarray(wall)
    if w.dtype == bool and w.ndim == 2:
        if w.shape != a.shape:
            raise ValueError(f"regions: wall mask shape {w.shape} differs from the grid shape {a.shape}")
        is_wall = w
    else:
        is_wall = np.isin(a, list(_colors(wall, "wall")))
    return [Obj(i, a, cl) for i, cl in enumerate(_components(a, ~is_wall, False, conn))]


def enclosed(g, wall, conn: int = 4) -> list[Obj]:
    """The regions (see regions) that do not touch the grid edge. conn=4: a diagonal gap in the wall does not leak."""
    return [o for o in regions(g, wall, conn) if not o.touches]


def flood(g, start, conn: int = 4, through=None, wall=None) -> Cells:
    """Cells reachable from start=(r, c) through cells of start's colour, or of the colours in `through`, or of every
    colour except the `wall` colour(s) (give through or wall, not both)."""
    a = _grid(g)
    r, c = _pos(start, "flood")
    if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
        raise ValueError(f"flood: start {start} is outside the grid {a.shape}")
    if through is not None and wall is not None:
        raise ValueError("flood: give through= (colours it may cross) or wall= (colours it may not), not both")
    if wall is not None:
        through = set(range(10)) - _colors(wall, "wall")
        if not through:
            raise ValueError(f"flood: wall={wall!r} leaves no colour to flood through")
    ok = np.isin(a, list(_colors(through, "through"))) if through is not None else a == a[r, c]
    if not ok[r, c]:
        raise ValueError(f"flood: the start cell's colour {int(a[r, c])} is not in through={through}")
    seen = np.zeros(a.shape, bool)
    seen[r, c] = True
    stack, out = [(r, c)], []
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    steps = _STEPS[conn]
    while stack:
        y, x = stack.pop()
        out.append((y, x))
        for dy, dx in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1] and ok[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return Cells(sorted(out))


def distances(g, start, through=None, conn: int = 4, bg=None) -> np.ndarray:
    """BFS steps from start ((r, c), or a list of cells / an Obj: all at distance 0) through cells of the colour(s) in
    `through` (default: the bg -- bg=None: the most frequent colour); an int array, -1 where unreachable. Start
    cells count as passable."""
    a = _grid(g)
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    ok = np.isin(a, list(_colors(through, "through"))) if through is not None else a == _bg(a, bg)
    src = [_pos(start, "distances")] if not isinstance(start, (Obj, Hit)) and np.ndim(start) == 1 else _cells_of(start)
    if not src:
        raise ValueError("distances: no start cell")
    dist = np.full(a.shape, -1, np.int64)
    frontier = []
    for r, c in src:
        if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
            raise ValueError(f"distances: start cell {(r, c)} is outside the grid {a.shape}")
        dist[r, c] = 0
        frontier.append((r, c))
    d = 0
    while frontier:
        d += 1
        nxt = []
        for r, c in frontier:
            for dr, dc in _STEPS[conn]:
                y, x = r + dr, c + dc
                if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and ok[y, x] and dist[y, x] < 0:
                    dist[y, x] = d
                    nxt.append((y, x))
        frontier = nxt
    return dist


# --------------------------------------------------------------------------- drawing and moving

class Ray:
    """The cells a ray crossed (start excluded) and the cell that stopped it (None: it left the grid)."""

    def __init__(self, cells, hit, hit_color):
        self.cells, self.hit, self.hit_color = cells, hit, hit_color

    def __repr__(self) -> str:
        end = f"stopped by colour {self.hit_color} at {self.hit}" if self.hit else "left the grid"
        return f"ray of {len(self.cells)} cells, {end}"


def ray(g, start, direction, stop=None, bg=None) -> Ray:
    """Walk from start=(r, c) in direction (a DIRS name or (dr, dc)) until a cell of a stop colour or the edge.
    stop=None: any colour other than bg (bg = most frequent colour); stop="edge": never stop before the edge.
    A start on an object's own cell stops at the object's next cell; (dr, dc) = (2, 0) steps two cells at a time."""
    a = _grid(g)
    dr, dc = _dir(direction)
    if isinstance(stop, str):
        if stop != "edge":
            raise ValueError(f"ray: stop must be None, a colour, colours or 'edge', got {stop!r}")
        stops = set()
    elif stop is None:
        stops = None                      # any colour but the bg
        b = _bg(a, bg)
    else:
        stops = _colors(stop, "stop")
    r, c = _pos(start, "ray")
    cells = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
            return Ray(Cells(cells), None, None)
        v = int(a[r, c])
        if (v != b) if stops is None else (v in stops):
            return Ray(Cells(cells), (r, c), v)
        cells.append((r, c))


def turn(direction, how: str):
    """direction turned by 90 degrees: how = "cw" / "right" (clockwise on screen), "ccw" / "left", or "back".
    A DIRS name gives a name back, a (dr, dc) a tuple."""
    dr, dc = _dir(direction)
    if how in ("cw", "right"):
        t = (dc, -dr)
    elif how in ("ccw", "left"):
        t = (-dc, dr)
    elif how == "back":
        t = (-dr, -dc)
    else:
        raise ValueError(f"turn: how must be 'cw', 'right', 'ccw', 'left' or 'back', got {how!r}")
    if isinstance(direction, str):
        return next(k for k, v in DIRS.items() if v == t)
    return t


def adjacent(r: int, c: int, shape, conn: int = 4) -> list[tuple[int, int]]:
    """The neighbours of (r, c) that are inside a grid (shape = a grid or (H, W)), in N4 / N8 order."""
    H, W = np.shape(shape) if np.ndim(shape) == 2 else shape
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    r, c = _pos((r, c), "adjacent")
    return [(r + dr, c + dc) for dr, dc in (N4 if conn == 4 else N8) if 0 <= r + dr < H and 0 <= c + dc < W]


def line(p, q) -> Cells:
    """Cells from p to q inclusive; p, q must share a row, a column or a diagonal."""
    (r0, c0), (r1, c1) = _pos(p, "line"), _pos(q, "line")
    dr, dc = r1 - r0, c1 - c0
    if dr and dc and abs(dr) != abs(dc):
        raise ValueError(f"line: {p} and {q} are not on one row, column or diagonal")
    n = max(abs(dr), abs(dc))
    sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
    return Cells((r0 + i * sr, c0 + i * sc) for i in range(n + 1))


def outline(bbox) -> Cells:
    """The border cells of a rectangle bbox = (r0, c0, r1, c1) (or an Obj's bbox), in reading order."""
    r0, c0, r1, c1 = _box(bbox)
    return Cells(sorted({(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1)}))


def paint(g, cells, color: int, clip: bool = False) -> np.ndarray:
    """A copy of g with `cells` (Obj, bool mask, or (r, c) list) set to color. Cells outside the grid raise,
    or are skipped with clip=True."""
    a = _grid(g).astype(np.int64)
    color = operator.index(color)
    for r, c in _cells_of(cells, a.shape):
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            a[r, c] = color
        elif not clip:
            raise ValueError(f"paint: cell {(r, c)} is outside the grid {a.shape} (clip=True skips it)")
    return a


def paste(g, patch, r: int, c: int, skip=_EMPTY, clip: bool = False) -> np.ndarray:
    """A copy of g with patch (a grid, or an Obj: exactly its cells) drawn with its top-left corner at (r, c).
    Grid patch cells equal to `skip` are not drawn (default -1; pass the bg colour to paste a sub-grid "transparently").
    Cells outside g raise, or are dropped with clip=True."""
    a = _grid(g).astype(np.int64)
    r, c = _pos((r, c), "paste")
    if isinstance(patch, Obj):
        p, draw = patch._vals, patch.mask              # an Obj draws exactly its own cells
    else:
        p = _grid(patch, "paste patch")
        draw = p != skip
    for (y, x), v in np.ndenumerate(p):
        if not draw[y, x]:
            continue
        rr, cc = r + y, c + x
        if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
            a[rr, cc] = v
        elif not clip:
            raise ValueError(f"paste: patch cell ({y},{x}) lands at {(rr, cc)}, outside the grid {a.shape} (clip=True drops it)")
    return a


def runs(seq) -> list[tuple[int, int, int]]:
    """Run-length encoding of one row or column: [(colour, start, length)], in order."""
    arr = np.asarray(seq)
    if arr.ndim > 1 and sorted(arr.shape)[-2] > 1:
        raise ValueError(f"runs: takes one row or column, got shape {arr.shape} (use runs(g[r]) or runs(g[:, c]))")
    v = arr.ravel().tolist()
    if not v:
        raise ValueError("runs: empty sequence")
    out, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            out.append((int(v[start]), start, i - start))
            start = i
    return out


def neighbors(g, color=None, conn: int = 8, bg=None) -> np.ndarray:
    """For every cell, how many of its 4 / 8 neighbours have `color` (a colour or colours; None = any colour but bg)."""
    a = _grid(g)
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    hit = a != _bg(a, bg) if color is None else np.isin(a, list(_colors(color)))
    pad = np.pad(hit, 1).astype(np.int64)
    H, W = a.shape
    return sum(pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] for dr, dc in _STEPS[conn])


def mirror_axes(g, ignore=None, min_overlap: float = 0.25) -> list[tuple[str, float, int, int]]:
    """Mirror lines anywhere in the grid, not only through the centre: [(kind, position, compared, mismatched)],
    best first. kind "col" = a vertical mirror line at column `position` (x.5 = between two columns), "row" =
    horizontal. Cells of the `ignore` colour(s) are not compared; a line with nothing compared is left out. Only lines
    whose mirrored part covers at least min_overlap of the grid's width (col) / height (row) are listed; mismatched = 0
    is an exact mirror. Sorted by mismatched / compared, then most compared first."""
    a = _grid(g)
    skip = np.isin(a, list(_colors(ignore, "ignore"))) if ignore is not None else np.zeros(a.shape, bool)
    out = []
    for kind, b, sk in (("col", a, skip), ("row", a.T, skip.T)):
        n = b.shape[1]
        for twice in range(1, 2 * n - 2):          # mirror line at twice/2
            lo = max(0, twice - (n - 1))
            cols = [c for c in range(lo, (twice + 1) // 2) if 0 <= twice - c < n]
            covered = 2 * len(cols) + (twice % 2 == 0)      # an axis on a column covers that column too
            if not cols or covered < min_overlap * n:
                continue
            left = b[:, cols]
            right = b[:, [twice - c for c in cols]]
            ok = ~(sk[:, cols] | sk[:, [twice - c for c in cols]])
            if ok.any():                                    # nothing compared: no evidence either way
                out.append((kind, twice / 2, int(ok.sum()), int(((left != right) & ok).sum())))
    out.sort(key=lambda t: (t[3] / max(t[2], 1), -t[2]))
    return out


def halo(x, conn: int = 8, shape=None) -> Cells:
    """The cells next to x (an Obj, Hit or cells) that are not x's own: its outline ring. Cells outside the grid
    are left out (the grid of an Obj; shape = a grid or (H, W) for cells)."""
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    own = set(_cells_of(x))
    ring = {(r + dr, c + dc) for r, c in own for dr, dc in _STEPS[conn]} - own
    if isinstance(x, Obj):
        shape = x._gshape
    if shape is not None:
        H, W = np.shape(shape) if np.ndim(shape) == 2 else shape
        ring = {(r, c) for r, c in ring if 0 <= r < H and 0 <= c < W}
    return Cells(sorted(ring))


def mask_of(cells, shape) -> np.ndarray:
    """A bool array of the grid's shape (a grid or (H, W)) that is True on the cells (Obj, Hit or (r, c) list)."""
    H, W = np.shape(shape) if np.ndim(shape) == 2 else shape
    m = np.zeros((H, W), bool)
    for r, c in _cells_of(cells):
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f"mask_of: cell {(r, c)} is outside {H}x{W}")
        m[r, c] = True
    return m


def fill_by_symmetry(g, ignore, using, partial: bool = False) -> np.ndarray:
    """A copy of g whose cells of the `ignore` colour(s) get the colour of their mirror image under the symmetries
    in `using`: orientation names that keep the grid's shape ("fliplr", "rot180", ...) and / or mirror lines as
    mirror_axes gives them (("col", 4.5, ...), ("row", 3.0, ...)). Repeated until nothing changes. Cells no
    symmetry can fill raise a ValueError that lists them; partial=True keeps their colour instead."""
    a = _grid(g).astype(np.int64)
    skip = list(_colors(ignore, "ignore"))
    if not using:
        raise ValueError("fill_by_symmetry: using= needs at least one symmetry")
    H, W = a.shape
    rr, cc = np.indices(a.shape)
    maps = []
    for u in using:
        if isinstance(u, str):
            if orient(a, u).shape != a.shape:
                raise ValueError(f"fill_by_symmetry: {u} changes a {H}x{W} grid's shape")
            idx = orient(np.arange(H * W).reshape(H, W), u)
            maps.append((idx // W, idx % W))
        else:
            kind, pos = u[0], float(u[1])
            if kind == "col":
                mc = (2 * pos - cc).astype(int)
                maps.append((rr, mc))
            elif kind == "row":
                mr = (2 * pos - rr).astype(int)
                maps.append((mr, cc))
            else:
                raise ValueError(f"fill_by_symmetry: a mirror line is ('col' | 'row', position, ...), got {u!r}")
    changed = True
    while changed:
        changed = False
        for mr, mc in maps:
            ok = (mr >= 0) & (mr < H) & (mc >= 0) & (mc < W)
            src = np.full(a.shape, -1)
            src[ok] = a[mr[ok], mc[ok]]
            fill = np.isin(a, skip) & ok & ~np.isin(src, skip)
            if fill.any():
                a[fill] = src[fill]
                changed = True
    left = np.argwhere(np.isin(a, skip))
    if len(left) and not partial:
        listed = [(int(r), int(c)) for r, c in left[:12]]
        raise ValueError(f"fill_by_symmetry: {len(left)} cells of colour {sorted(skip)} have no mirror image outside "
                         f"the holes under {list(using)}: {listed}{' ...' if len(left) > 12 else ''} -- add a "
                         f"symmetry, or partial=True keeps them")
    return a


def shift(cells, dr: int, dc: int) -> Cells:
    """The cells (Obj or (r, c) list) moved by (dr, dc)."""
    dr, dc = _pos((dr, dc), "shift")
    return Cells((r + dr, c + dc) for r, c in _cells_of(cells))


class Slide(int):
    """slide()'s result: the number of free steps (an int), plus what stops it: .edge (True: the grid edge),
    .hit (the cells it would overlap next; empty at the edge) and .hit_colors (their colours)."""

    def __new__(cls, n, hit, hit_colors):
        self = super().__new__(cls, n)
        self.hit, self.hit_colors = hit, hit_colors
        return self

    edge = property(lambda self: not self.hit)

    def __repr__(self) -> str:
        end = "the grid edge" if self.edge else f"colour {'/'.join(map(str, self.hit_colors))} at {list(self.hit)[:4]}"
        return f"{int(self)} (then {end})"

    def __format__(self, spec: str) -> str:               # f"{steps}" stays a number
        return format(int(self), spec)


def slide(g, obj, direction, bg=None) -> Slide:
    """How many steps obj (Obj or cells) can move in direction before it would leave the grid or overlap a
    cell that is not bg and not its own (bg = most frequent colour). The result is that int, with .edge / .hit /
    .hit_colors saying what stops it."""
    a = _grid(g)
    b = _bg(a, bg)
    dr, dc = _dir(direction)
    own = set(_cells_of(obj))
    if not own:
        raise ValueError("slide: no cells to move")
    n = 0
    while True:
        nxt = [(r + (n + 1) * dr, c + (n + 1) * dc) for r, c in own]
        if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in nxt):
            return Slide(n, Cells(), [])
        hit = Cells(sorted((r, c) for r, c in nxt if a[r, c] != b and (r, c) not in own))
        if hit:
            return Slide(n, hit, sorted({int(a[r, c]) for r, c in hit}))
        n += 1


def move(g, obj, dr: int, dc: int, bg=None, clip: bool = False, overwrite: bool = False) -> np.ndarray:
    """A copy of g with obj (an Obj) erased to bg and drawn again, with its own colours, moved by (dr, dc).
    bg: the given one, else the bg the Obj was cut with, else g's most frequent colour. A destination cell that holds
    another non-bg colour raises (the move would cover something) unless overwrite=True. Cells moved off the grid
    raise, or are dropped with clip=True."""
    if not isinstance(obj, Obj):
        raise TypeError("move: obj must be an Obj (it carries the colours); use paint(shift(cells, dr, dc), colour)")
    dr, dc = _pos((dr, dc), "move")
    a = _grid(g).astype(np.int64)
    b = _bg(a, bg if bg is not None else obj.bg if obj.bg is not None and obj.bg >= 0 else None)
    vals = [int(v) for v in obj._vals[obj.mask]]           # the object's own colours, in its cells' order
    own = set(obj.cells)
    for r, c in obj.cells:
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            a[r, c] = b
    for (r, c), v in zip(sorted(obj.cells), vals):
        y, x = r + dr, c + dc
        if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
            if not overwrite and a[y, x] != b and (y, x) not in own:
                raise ValueError(f"move: ({y},{x}) holds colour {int(a[y, x])}; overwrite=True draws over it")
            a[y, x] = v
        elif not clip:
            raise ValueError(f"move: cell {(r, c)} would leave the grid {a.shape} (clip=True drops it)")
    return a


def gravity(g, direction="down", bg=None, conn: int = 8, multicolor: bool = False, fixed=None,
            rigid: bool = True) -> np.ndarray:
    """A copy of g in which every object (objects() with these settings) slides in direction as a rigid body until it
    rests on the edge or on another cell; objects nearest that edge move first, repeated until nothing moves.
    fixed: colour(s) whose objects never move (walls). rigid=False: every cell falls on its own (sand), so shapes
    break up and pile up per column (per row for left / right)."""
    a = _grid(g).astype(np.int64)
    b = _bg(a, bg)
    dr, dc = _dir(direction)
    keep = _colors(fixed, "fixed") if fixed is not None else set()
    movers = [o for o in objects(a, b, conn, multicolor) if not (set(o.colors) & keep)]
    units = [list(o.cells) for o in movers] if rigid else [[p] for o in movers for p in o.cells]
    cells = dict(enumerate(units))
    vals = {i: [int(a[r, c]) for r, c in u] for i, u in cells.items()}
    lead = lambda i: max(r * dr + c * dc for r, c in cells[i])   # how far along direction the object reaches
    moved = True
    while moved:
        moved = False
        for i in sorted(cells, key=lambda i: -lead(i)):
            n = slide(a, cells[i], (dr, dc), bg=b)
            if n:
                for r, c in cells[i]:
                    a[r, c] = b
                cells[i] = [(r + n * dr, c + n * dc) for r, c in cells[i]]
                for (r, c), v in zip(cells[i], vals[i]):
                    a[r, c] = v
                moved = True
    return a


# --------------------------------------------------------------------------- input -> output pairs

class Link:
    """How one input object relates to one output object. kind is one of:
    same, moved, recolored, turned (rotated / flipped / scaled, maybe moved), copy (an extra output object equal to
    some input object), moved+recolored, changed (overlaps in place with other cells), merged (its exact pattern is
    now part of a bigger output object, maybe moved), added, deleted."""

    def __init__(self, kind, a, b, info=""):
        self.kind, self.a, self.b, self.info = kind, a, b, info

    def __repr__(self) -> str:
        def d(o):
            r0, c0, r1, c1 = o.bbox
            return f"#{o.idx} c{o.color} {o.h}x{o.w} at ({r0},{c0})"
        src = d(self.a) if self.a is not None else "-"
        dst = d(self.b) if self.b is not None else "-"
        return f"{self.kind:<15} in {src:<22} out {dst:<22} {self.info}".rstrip()


_KIND_ORDER = {"same": 0, "moved": 1, "recolored": 2, "turned": 3, "copy": 4, "moved+recolored": 5, "changed": 6,
               "merged": 7, "added": 8, "deleted": 9}


def _offset(x: Obj, y: Obj) -> tuple[int, int]:
    return y.bbox[0] - x.bbox[0], y.bbox[1] - x.bbox[1]


def _recolor_info(x: Obj, y: Obj) -> str:
    m = {}
    for cx, cy in zip(np.asarray(x._vals)[x.mask].tolist(), np.asarray(y._vals)[y.mask].tolist()):
        m.setdefault(cx, set()).add(cy)
    return ", ".join(f"{k}->{'/'.join(map(str, sorted(v)))}" for k, v in sorted(m.items()) if v != {k})


def _pair_up(A: list[Obj], B: list[Obj], ok, used_a: set, used_b: set, pre=None, common: bool = False):
    """Greedy one-to-one pairing of unused objects with ok(x, y) (after the cheap test pre(x, y)), nearest bbox corner
    first; common=True: offsets that more candidate pairs share first (several objects moved the same way)."""
    cand = []
    for x in A:
        if x.idx in used_a:
            continue
        for y in B:
            if y.idx in used_b or (pre is not None and not pre(x, y)):
                continue
            info = ok(x, y)
            if info is not None:
                cand.append((_offset(x, y), x.idx, y.idx, x, y, info))
    freq = Counter(t[0] for t in cand) if common else Counter()
    cand.sort(key=lambda t: (-freq[t[0]], abs(t[0][0]) + abs(t[0][1]), t[1], t[2]))
    out = []
    for _, _, _, x, y, info in cand:
        if x.idx in used_a or y.idx in used_b:
            continue
        used_a.add(x.idx)
        used_b.add(y.idx)
        out.append((x, y, info))
    return out


def match(a, b, bg=None, conn: int = 8, multicolor: bool = False) -> list[Link]:
    """Pair the objects of input a with the objects of output b (same segmentation for both; bg = a's most frequent
    colour unless given). Tried in this order, each object used once: same, moved, recolored (same cells), turned
    (orientation / scale, colours kept); then an output object equal to any input object (maybe turned) is a copy;
    then moved+recolored (same shape), changed (overlapping cells); an unpaired input object whose exact pattern is
    inside a bigger output object is merged; the rest of the output objects are added, the rest of the input objects
    deleted."""
    a, b = _grid(a, "a"), _grid(b, "b")
    bgc = _bg_arg(a, bg)
    A, B = objects(a, bgc, conn, multicolor), objects(b, bgc, conn, multicolor)
    ua, ub = set(), set()
    links = []

    def exact(x, y):
        return x.mask.shape == y.mask.shape and np.array_equal(x.mask, y.mask) and np.array_equal(x.grid(-1), y.grid(-1))

    def same_mask(x, y):
        return x.mask.shape == y.mask.shape and np.array_equal(x.mask, y.mask)

    def pass_(kind, ok, pre=None, common=False):
        for x, y, info in _pair_up(A, B, ok, ua, ub, pre, common):
            links.append(Link(kind, x, y, info))

    def same_colours(x, y):
        return x.size == y.size and x.colors == y.colors and (x.h, x.w) == (y.h, y.w)

    def could_turn(x, y):          # colours in the same proportions, sizes related by k x k (either orientation)
        if set(x.colors) != set(y.colors):
            return False
        ratio = y.size / x.size
        if any(y.colors[c] * x.size != x.colors[c] * y.size for c in x.colors):
            return False
        k = ratio ** 0.5 if ratio >= 1 else (1 / ratio) ** 0.5
        if abs(k - round(k)) > 1e-9:
            return False
        k = round(k)
        small, big = ((x.h, x.w), (y.h, y.w)) if ratio >= 1 else ((y.h, y.w), (x.h, x.w))
        return sorted(big) == sorted((small[0] * k, small[1] * k))

    pass_("same", lambda x, y: "" if exact(x, y) else None, pre=lambda x, y: x.bbox == y.bbox)
    pass_("moved", lambda x, y: "by ({:+d},{:+d})".format(*_offset(x, y)) if exact(x, y) else None,
          pre=same_colours, common=True)
    pass_("recolored", lambda x, y: _recolor_info(x, y) if same_mask(x, y) else None, pre=lambda x, y: x.bbox == y.bbox)
    pass_("turned", lambda x, y: (lambda n: f"{_names(n)} at ({y.bbox[0]},{y.bbox[1]})" if n else None)(
        shape_match(x, y, color=True)), pre=could_turn)
    # an extra output object equal to some input object (paired or not, maybe turned) is a copy of it
    for y in B:
        if y.idx in ub:
            continue
        src = next((x for x in A if same_colours(x, y) and exact(x, y)), None)
        rel = None
        if src is None:
            for x in A:
                names = shape_match(x, y, color=True) if could_turn(x, y) else []
                if names:
                    src, rel = x, _names(names)
                    break
        if src is not None:
            ub.add(y.idx)
            links.append(Link("copy", src, y, (rel + " " if rel else "") + "at ({},{})".format(*y.bbox[:2])))
    pass_("moved+recolored", lambda x, y: "by ({:+d},{:+d}) ".format(*_offset(x, y)) + _recolor_info(x, y)
          if same_mask(x, y) else None, pre=lambda x, y: x.size == y.size and (x.h, x.w) == (y.h, y.w), common=True)
    # changed in place: largest overlap first

    def boxes_meet(x, y):
        return x.bbox[0] <= y.bbox[2] and y.bbox[0] <= x.bbox[2] and x.bbox[1] <= y.bbox[3] and y.bbox[1] <= x.bbox[3]

    cand = []
    for x in A:
        for y in B:
            if x.idx not in ua and y.idx not in ub and boxes_meet(x, y):
                n = len(set(x.cells) & set(y.cells))
                if n:
                    cand.append((-n, x.idx, y.idx, x, y))
    for _, _, _, x, y in sorted(cand, key=lambda t: t[:3]):
        if x.idx in ua or y.idx in ub:
            continue
        ua.add(x.idx)
        ub.add(y.idx)
        sx, sy = set(x.cells), set(y.cells)
        links.append(Link("changed", x, y, f"+{len(sy - sx)} -{len(sx - sy)} cells"))
    # an unpaired input object (3+ cells) whose exact pattern is now inside one bigger output object: merged into it.
    # Only into an output object that grew (changed) or has no input yet -- one that equals another input object
    # (same / moved / recolored / turned / copy) is already explained, a pattern found there is a coincidence
    explained = {l.b.idx for l in links if l.kind != "changed"}
    owner = {p: y.idx for y in B if y.idx not in explained for p in y.cells}
    merged_into = set()
    for x in A:
        if x.idx in ua or x.size < 3:
            continue
        best = None
        for h in find(b, x.grid(-1)):
            cells = [(h.r + r, h.c + c) for r, c in np.argwhere(x.mask).tolist()]
            ys = {owner.get(p) for p in cells}
            if len(ys) == 1 and None not in ys and B[next(iter(ys))].size > x.size:
                dr, dc = h.r - x.bbox[0], h.c - x.bbox[1]
                if best is None or abs(dr) + abs(dc) < best[0]:
                    best = (abs(dr) + abs(dc), dr, dc, B[next(iter(ys))])
        if best is not None:
            ua.add(x.idx)
            _, dr, dc, y = best
            links.append(Link("merged", x, y, f"by ({dr:+d},{dc:+d}) into out #{y.idx} c{y.color}"))
            merged_into.add(y.idx)
    # where a new colour comes from: another input object of the same shape that has it
    for l in links:
        if l.kind in ("recolored", "moved+recolored"):
            key = shape_key(l.a)
            src = [z for z in A if z.idx != l.a.idx and z.color == l.b.color and shape_key(z) == key]
            if src:
                l.info += f" (colour of same-shape input #{src[0].idx})"
    # a moved / turned object that now touches another output object: say which (rules like "move until it touches")
    for l in links:
        if l.kind in ("moved", "moved+recolored", "turned"):
            near = {(r + dr, c + dc) for r, c in l.b.cells for dr, dc in _STEPS[4]} - set(l.b.cells)
            hits = [o for o in B if o.idx != l.b.idx and near & set(o.cells)]
            if hits:
                l.info += ", now touches " + " ".join(f"#{o.idx} c{o.color}" for o in hits)
    for y in B:
        if y.idx not in ub and y.idx not in merged_into:      # an object built from merged inputs is not "added"
            links.append(Link("added", None, y))
    for x in A:
        if x.idx not in ua:
            links.append(Link("deleted", x, None))
    links.sort(key=lambda l: (_KIND_ORDER[l.kind], l.a.idx if l.a is not None else -1, l.b.idx if l.b is not None else -1))
    return links


def print_match(a=None, b=None, bg=None, conn: int = 8, multicolor: bool = False) -> None:
    """match() as text; with no grids, for every training pair. More than 5 `same` objects are summed up in one
    line; at most 12 lines of each other kind are listed."""
    if (a is None) != (b is None):
        raise ValueError("print_match: give both grids, or neither (all training pairs)")
    pairs = [(f"train {k}", x, y) for k, (x, y) in enumerate(train, 1)] if a is None else [("", a, b)]
    for name, x, y in pairs:
        x, y = _grid(x, "a"), _grid(y, "b")
        bgc = _bg_arg(x, bg)
        links = match(x, y, bgc, conn, multicolor)
        n_in, n_out = len(objects(x, bgc, conn, multicolor)), len(objects(y, bgc, conn, multicolor))
        kinds = Counter(l.kind for l in links)
        head = ", ".join(f"{n} {k}" for k, n in kinds.items())
        print(f"{name + ': ' if name else ''}{n_in} input objects -> {n_out} output objects (bg {bgc}, conn {conn}"
              f"{', multicolor' if multicolor else ''}): {head}")
        if x.shape != y.shape:
            print(f"  sizes differ ({x.shape[0]}x{x.shape[1]} -> {y.shape[0]}x{y.shape[1]}): objects are paired by their "
                  "places in two different grids, so moved / changed / added / deleted may mean nothing here "
                  "(overview() lists size relations)")
        same = [l for l in links if l.kind == "same"]
        rest = [l for l in links if l.kind != "same"]
        if len(same) > 5:
            ids = " ".join(f"#{l.a.idx}" for l in same[:20]) + (" ..." if len(same) > 20 else "")
            print(f"  same            {len(same)} objects unchanged: {ids}")
        else:
            rest = same + rest
        shown, hidden = Counter(), Counter()
        for l in rest:
            if shown[l.kind] < _MAX_PER_KIND:
                print(f"  {l}")
                shown[l.kind] += 1
            else:
                hidden[l.kind] += 1
        if hidden:
            print("  ... not listed: " + ", ".join(f"{n} more {k}" for k, n in hidden.items()) + " (match() returns all)")


# --------------------------------------------------------------------------- extraction rules

def _props(objs: list[Obj]) -> dict[str, list]:
    """Per-object properties used by explain_select, one list per property (same order as objs)."""
    shp = [shape_key(o) for o in objs]
    n_shape = Counter(shp)
    n_color = Counter(o.color for o in objs)
    sym = [symmetry(o.grid(-1)) for o in objs]
    return {
        "size": [o.size for o in objs], "height": [o.h for o in objs], "width": [o.w for o in objs],
        "bbox area": [o.h * o.w for o in objs], "number of colours": [len(o.colors) for o in objs],
        "holes": [o.holes for o in objs], "top row": [o.bbox[0] for o in objs], "bottom row": [o.bbox[2] for o in objs],
        "left column": [o.bbox[1] for o in objs], "right column": [o.bbox[3] for o in objs],
        "fill ratio": [o.size / (o.h * o.w) for o in objs],
        "objects with the same shape": [n_shape[k] for k in shp],
        "objects with the same colour": [n_color[o.color] for o in objs],
        "_colour": [o.color for o in objs], "_shape": shp,
        "_rect": [o.is_rect for o in objs], "_edge": [bool(o.touches) for o in objs],
        "_sym": [any(v == 0 for v in d.values()) for d in sym],
        "_sym_lr": [d["fliplr"] == 0 for d in sym], "_sym_ud": [d["flipud"] == 0 for d in sym],
        "_sym_rot": [d["rot180"] == 0 for d in sym],
        "_sym_diag": [d.get("transpose") == 0 or d.get("antitranspose") == 0 for d in sym],
    }


_POSITIONAL = ("top row", "bottom row", "left column", "right column")


def _selectors():
    """(name, f(props, n) -> index or None) for every selection rule explain_select tries."""
    out = []
    for p in ("size", "height", "width", "bbox area", "number of colours", "holes", "top row", "bottom row",
              "left column", "right column", "fill ratio", "objects with the same shape", "objects with the same colour"):
        for word, pick in (("largest", max), ("smallest", min)):
            def f(P, n, p=p, pick=pick):
                v = P[p]
                best = pick(v)
                idx = [i for i in range(n) if v[i] == best]
                return idx[0] if len(idx) == 1 else None
            out.append((f"{word} {p}" + (" (a position)" if p in _POSITIONAL else ""), f))
    for p, label in (("_colour", "colour"), ("_shape", "shape"), ("size", "size")):
        def f(P, n, p=p):
            c = Counter(P[p])
            idx = [i for i in range(n) if c[P[p][i]] == 1]
            return idx[0] if len(idx) == 1 else None
        out.append((f"the only object with a {label} no other object has", f))
    for p, yes, no in (("_rect", "a filled rectangle", "not a filled rectangle"),
                       ("_edge", "touching the grid edge", "not touching the grid edge"),
                       ("_sym", "symmetric (some rotation / flip)", "not symmetric in any way"),
                       ("_sym_lr", "left-right symmetric", "not left-right symmetric"),
                       ("_sym_ud", "up-down symmetric", "not up-down symmetric"),
                       ("_sym_rot", "symmetric under rot180", "not symmetric under rot180"),
                       ("_sym_diag", "symmetric about a diagonal", "not symmetric about a diagonal")):
        for want, label in ((True, yes), (False, no)):
            def f(P, n, p=p, want=want):
                idx = [i for i in range(n) if P[p][i] == want]
                return idx[0] if len(idx) == 1 else None
            out.append((f"the only object {label}", f))
    return out


def _as_output(o: Obj, b: np.ndarray, a: np.ndarray) -> str | None:
    """How output b is made from object o of input a: 'crop' (its bbox in the input), 'object' (its cells, bg
    elsewhere), or such with an orientation / scale; None if neither."""
    for base, g in (("crop", crop(a, o.bbox)), ("object", o.grid())):
        for k in [1] + _ratios(g.shape, b.shape):
            up = upscale(g, k)
            names = [n for n in ORIENTS if orient(up, n).shape == b.shape and np.array_equal(orient(up, n), b)]
            if names:
                turn_ = "" if "id" in names else " " + "|".join(names)
                return base + turn_ + ("" if k == 1 else f" *{k}")
    return None


def _pick_text(o) -> str:
    return f"#{o.idx} c{o.color} {o.h}x{o.w} at ({o.bbox[0]},{o.bbox[1]})" if o is not None else "no pick (tie)"


def explain_select(bg=None, conn: int = 8, multicolor: bool = False) -> list[tuple[str, list]]:
    """For puzzles whose output is one of the input's objects (its bbox crop, or its cells; maybe turned / scaled):
    print which simple selection rules ("largest size", "the only object not left-right symmetric", ...) pick that
    object in every training pair, and which object they pick in each test input -- rules that pick the same
    objects are listed together. Returns [(rule, [the Obj it picks in each test input, or None on a tie])].
    Only rules over one property are tried; a rule that fits only by chance looks the same as the real one."""
    targets = []
    for k, (a, b) in enumerate(train, 1):
        objs = objects(a, bg, conn, multicolor)
        hit = {o.idx: _as_output(o, b, a) for o in objs}
        hit = {i: h for i, h in hit.items() if h is not None}
        if not hit:
            print(f"train {k}: the output is not one of the input's objects (bg {_bg_arg(a, bg)}, conn {conn}"
                  f"{', multicolor' if multicolor else ''}), so there is nothing to select")
            return []
        targets.append((objs, hit))
    for k, (objs, hit) in enumerate(targets, 1):
        print(f"train {k}: output = " + "; ".join(f"#{i} as {h}" for i, h in hit.items()) + f"  (of {len(objs)} objects)")
    tobjs = [objects(t, bg, conn, multicolor) for t in test]
    if all(len(objs) == 1 for objs, _ in targets):
        print("every training input has one object (with these settings): the output is that object, nothing to select")
        return [("the only object", [o[0] if len(o) == 1 else None for o in tobjs])]
    good = []
    for name, f in _selectors():
        if all((i := f(_props(objs), len(objs))) is not None and i in hit for objs, hit in targets):
            picks = []
            for objs in tobjs:
                i = f(_props(objs), len(objs)) if objs else None
                picks.append(objs[i] if i is not None else None)
            good.append((name, picks))
    if not good:
        print(f"no single-property rule picks the output object in all {len(train)} pairs")
        return []
    by_pick: dict[tuple, list[str]] = {}
    for name, picks in good:
        by_pick.setdefault(tuple(o.idx if o is not None else None for o in picks), []).append(name)
    print(f"{len(good)} rules pick it in all {len(train)} pairs; grouped by what they pick in the test inputs:")
    for key, names in sorted(by_pick.items(), key=lambda kv: -len(kv[1])):
        picks = next(p for n, p in good if n == names[0])
        print("  " + "  ".join(f"test {j}: {_pick_text(o)}" for j, o in enumerate(picks, 1))
              + f"  <- {len(names)} rule(s): " + "; ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
    for j in range(len(test)):
        ids = {key[j] for key in by_pick}
        real = ids - {None}
        if len(real) > 1:
            print(f"  test {j + 1}: the rules pick {len(real)} different objects -- the training pairs do not decide")
    if len(good) <= 2:
        print(f"  only {len(good)} rule(s) fit: with {len(train)} training pairs a rule can fit by chance -- check it")
    return good


def _fate(l: Link) -> str:
    """What happened to an input object, as explain_objects names it."""
    if l.kind == "same":
        return "kept"
    if l.kind in ("recolored", "moved+recolored"):
        cols = set(np.asarray(l.b._vals)[l.b.mask].tolist())
        return l.kind + (f" to {cols.pop()}" if len(cols) == 1 else "")
    return l.kind


def _object_features(a: np.ndarray, objs: list[Obj], bgs: set) -> dict[str, list]:
    """explain_objects' per-object properties: explain_select's, plus what each object touches."""
    P = _props(objs)
    owner = {p_: o.idx for o in objs for p_ in o.cells}
    H, W = a.shape
    touch_any, touch_cols = [], []
    for o in objs:
        own = set(o.cells)
        ring = {(r + dr, c + dc) for r, c in own for dr, dc in _STEPS[8]} - own
        ring = {(r, c) for r, c in ring if 0 <= r < H and 0 <= c < W}
        touch_any.append(any(q in owner for q in ring))
        touch_cols.append(frozenset(int(a[q]) for q in ring if int(a[q]) not in bgs))
    sizes = P["size"]
    P["_touches"] = touch_any
    P["_touch_colours"] = touch_cols
    P["_largest"] = [v == max(sizes) and sizes.count(v) == 1 for v in sizes]
    P["_smallest"] = [v == min(sizes) and sizes.count(v) == 1 for v in sizes]
    return P


_FATE_NUMERIC = ("size", "height", "width", "bbox area", "number of colours", "holes", "top row", "bottom row",
                 "left column", "right column", "objects with the same shape", "objects with the same colour")
_FATE_BOOL = (("_rect", "a filled rectangle"), ("_edge", "touching the grid edge"),
              ("_touches", "touching another object"), ("_largest", "the largest"), ("_smallest", "the smallest"),
              ("_sym", "symmetric (some rotation / flip)"), ("_sym_lr", "left-right symmetric"),
              ("_sym_ud", "up-down symmetric"))


def _fate_rules(rows: list[dict]) -> list[tuple[str, int, callable]]:
    """(rule text, rank, f(features row) -> bool) for every one-property rule explain_objects tries; rows = every
    training object's features (for the values to compare with). rank orders rules from simple to contrived:
    0 yes / no and colour rules, 1 "== v", 2 thresholds, 3 positions."""
    out = []
    for p in _FATE_NUMERIC:
        vals = sorted({r[p] for r in rows})
        pos = p in _POSITIONAL
        tag = " (a position)" if pos else ""
        for v in vals:
            out.append((f"{p} == {v}{tag}", 3 if pos else 1, lambda r, p=p, v=v: r[p] == v))
        for v in vals[1:]:
            out.append((f"{p} >= {v}{tag}", 3 if pos else 2, lambda r, p=p, v=v: r[p] >= v))
        for v in vals[:-1]:
            out.append((f"{p} <= {v}{tag}", 3 if pos else 2, lambda r, p=p, v=v: r[p] <= v))
    for p, label in _FATE_BOOL:
        out.append((label, 0, lambda r, p=p: bool(r[p])))
        out.append((f"not {label}", 0, lambda r, p=p: not r[p]))
    for c in sorted({r["_colour"] for r in rows}):
        out.append((f"colour {c}", 0, lambda r, c=c: r["_colour"] == c))
        out.append((f"not colour {c}", 0, lambda r, c=c: r["_colour"] != c))
    for c in sorted({c for r in rows for c in r["_touch_colours"]}):
        out.append((f"touching colour {c}", 0, lambda r, c=c: c in r["_touch_colours"]))
        out.append((f"not touching colour {c}", 0, lambda r, c=c: c not in r["_touch_colours"]))
    return out


def _feature_rows(P: dict[str, list], n: int) -> list[dict]:
    return [{k: v[i] for k, v in P.items()} for i in range(n)]


def explain_objects(bg=None, conn: int = 8, multicolor: bool = False, table: bool = True) -> dict:
    """For puzzles whose output is the input with some objects kept, deleted, recoloured, moved, ...: what happens
    to every input object in every training pair (print_match's pairing), and which one-property rules ("size == 1",
    "touching colour 2", "not a filled rectangle", ...) give exactly the objects with each fate in every pair -- and
    which objects of each test input those rules pick. Rules that agree on train but pick different test objects
    are listed apart, simplest first: the training pairs do not decide between them. When no single rule fits a
    fate, two properties joined by "and" are tried (not positions). A fate seen in one pair only gets no rules (any
    rule would fit it by chance). table=False leaves out the per-object table.
    Returns {fate: [(rule, [the Obj it picks in each test input])]}."""
    rows, fates, pair_of = [], [], []
    for k, (a, b) in enumerate(train, 1):
        if a.shape != b.shape:
            print(f"train {k}: sizes differ ({a.shape[0]}x{a.shape[1]} -> {b.shape[0]}x{b.shape[1]}): explain_objects "
                  "needs every output the size of its input (print_match / overview for other puzzles)")
            return {}
        bgc = _bg_arg(a, bg)
        bgs = set(bgc) if isinstance(bgc, list) else {bgc}
        objs = objects(a, bgc, conn, multicolor)
        links = match(a, b, bgc, conn, multicolor)
        fate = {l.a.idx: (_fate(l), l.info) for l in links if l.a is not None and l.kind != "copy"}
        copied = {l.a.idx for l in links if l.kind == "copy"}
        P = _object_features(a, objs, bgs)
        R = _feature_rows(P, len(objs))
        rows += R
        fates += [fate[o.idx][0] for o in objs]
        pair_of += [k] * len(objs)
        kinds = Counter(fate[o.idx][0] for o in objs)
        print(f"train {k}: {len(objs)} objects (bg {bgc}, conn {conn}{', multicolor' if multicolor else ''}): "
              + ", ".join(f"{n} {f}" for f, n in kinds.most_common()))
        if table:
            for o, r in list(zip(objs, R))[:_MAX_OBJECTS]:
                flags = " ".join(w for w, on in (("rect", r["_rect"]), ("edge", r["_edge"]), ("largest", r["_largest"]),
                                                 ("smallest", r["_smallest"])) if on)
                tc = "{" + ",".join(map(str, sorted(r["_touch_colours"]))) + "}"
                f, info = fate[o.idx]
                print(f"  #{o.idx:<2} c{o.color} size {o.size:<3} {o.h}x{o.w} at ({o.bbox[0]},{o.bbox[1]}) holes {o.holes} "
                      f"{flags + ' ' if flags else ''}touches {tc} -> {f}{' ' + info if info else ''}"
                      + (" (also copied)" if o.idx in copied else ""))
            if len(objs) > _MAX_OBJECTS:
                print(f"  ... {len(objs) - _MAX_OBJECTS} more objects")
    tests = []
    for t in test:
        bgc = _bg_arg(t, bg)
        bgs = set(bgc) if isinstance(bgc, list) else {bgc}
        objs = objects(t, bgc, conn, multicolor)
        tests.append((objs, _feature_rows(_object_features(t, objs, bgs), len(objs))))
    rules = _fate_rules(rows)
    result = {}
    print("rules that give each fate in every training pair (one property; 'and' = two, positions left out):")
    for f in [f for f, _ in Counter(fates).most_common()]:
        want = [x == f for x in fates]
        n_pairs = len({k for k, x in zip(pair_of, fates) if x == f})
        head = f"  {f} ({sum(want)} of {len(want)} objects, in {n_pairs} of {len(train)} pairs)"
        if all(want):
            print(f"  {f}: every object -- no rule needed")
            continue
        if n_pairs < 2:
            print(head + ": seen in one pair only -- rules fit it by chance, none listed")
            continue
        fits = [(name, rank, g) for name, rank, g in rules if all(g(r) == w for r, w in zip(rows, want))]
        if not fits:
            need = [(name, rank, g) for name, rank, g in rules if rank < 3 and all(g(r) for r, w in zip(rows, want) if w)]
            for i in range(len(need)):
                for j in range(i + 1, len(need)):
                    (n1, r1, g1), (n2, r2, g2) = need[i], need[j]
                    if all((g1(r) and g2(r)) == w for r, w in zip(rows, want)):
                        fits.append((f"{n1} and {n2}", 4 + r1 + r2, lambda r, g1=g1, g2=g2: g1(r) and g2(r)))
        if not fits:
            print(head + ": no rule of one or two of these properties")
            continue
        fits.sort(key=lambda t: t[1])
        groups_: dict[tuple, list] = {}
        picks_by_rule = []
        for name, rank, g in fits:
            picks = [[o for o, r in zip(objs, R) if g(r)] for objs, R in tests]
            picks_by_rule.append((name, picks))
            groups_.setdefault(tuple(tuple(o.idx for o in pk) for pk in picks), []).append(name)
        result[f] = picks_by_rule
        print(head + f": {len(fits)} rule(s)" + (f"; they pick {len(groups_)} different sets of test objects -- the "
                                                  "training pairs do not decide:" if len(groups_) > 1 else ":"))
        for key, names in list(groups_.items())[:4]:            # the simplest rule of each group first
            picked = "  ".join(f"test {j}: " + (" ".join(f"#{i}" for i in ids) if ids else "none")
                               for j, ids in enumerate(key, 1))
            print(f"    {names[0]}" + (f" (+{len(names) - 1} more that pick the same)" if len(names) > 1 else "")
                  + f"  ->  {picked}")
        if len(groups_) > 4:
            print(f"    ... {len(groups_) - 4} more sets")
    return result


def _combine(how: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return {"AND": x & y, "OR": x | y, "XOR": x ^ y, "first and not second": x & ~y, "neither": ~x & ~y}[how]


def explain_panels(color=None) -> list[str]:
    """For puzzles whose input is cut by lines into equal panels and whose output has a panel's size: which
    cell-wise combination of panels gives the output in every training pair -- two panels joined by AND / OR /
    XOR / "first and not second" / "neither" (the second maybe flipped), or all panels stacked in some order
    (later ones drawn over earlier ones). Masks are each panel's non-background cells (the background = the most
    frequent colour inside the panels). Prints and returns the descriptions that fit; color= is panels()' color."""
    data = []
    for k, (a, b) in enumerate(train, 1):
        ps = panels(a, color)
        if len(ps) < 2 or len({p.grid.shape for p in ps}) != 1 or ps[0].grid.shape != b.shape:
            print(f"train {k}: the input is not {len(ps)} equal panels of the output's size {b.shape[0]}x{b.shape[1]} "
                  f"(panels(train[{k - 1}][0]{'' if color is None else f', {color}'}) gives {len(ps)} panels"
                  + (f" of {sorted({p.grid.shape for p in ps})}" if ps else "") + ")")
            return []
        cells = np.concatenate([p.grid.ravel() for p in ps])
        vals, n = np.unique(cells, return_counts=True)
        pbg = int(vals[n.argmax()])
        obg = pbg if pbg in b else _bg(b, None)
        data.append(([p.grid for p in ps], pbg, b, obg))
    npan = {len(d[0]) for d in data}
    if len(npan) != 1:
        print(f"the inputs have different numbers of panels {sorted(npan)}: no fixed combination")
        return []
    n = npan.pop()
    found = []

    def colour_note(masks_out):
        cs = [sorted(set(b[m].tolist())) for (_, _, b, _), m in zip(data, masks_out)]
        if all(len(c) == 1 for c in cs):
            same = {c[0] for c in cs}
            return f"drawn in colour {same.pop()}" if len(same) == 1 else "drawn in one colour per pair: " + ", ".join(
                str(c[0]) for c in cs)
        return "in several colours"

    turns = ("id", "fliplr", "flipud", "rot180")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in turns:
                for how in ("AND", "OR", "XOR", "first and not second", "neither"):
                    if how in ("AND", "OR", "XOR", "neither") and (j < i and t == "id"):
                        continue              # symmetric: (i, j) already tried as (j, i)
                    outs, ok = [], True
                    for ps, pbg, b, obg in data:
                        y = ps[j] != pbg
                        y = orient(y, t)
                        if y.shape != b.shape:
                            ok = False
                            break
                        m = _combine(how, ps[i] != pbg, y)
                        if not np.array_equal(m, b != obg):
                            ok = False
                            break
                        outs.append(m)
                    if ok:
                        second = f"panel {j}" + ("" if t == "id" else f" {t}")
                        found.append(f"output = panel {i} {how} {second} (non-background cells), {colour_note(outs)}")
    if n <= 5:
        import itertools
        for order in itertools.permutations(range(n)):
            ok = True
            for ps, pbg, b, obg in data:
                out = np.full(b.shape, pbg)
                for k in order:
                    out = np.where(ps[k] != pbg, ps[k], out)
                if not np.array_equal(out, b):
                    ok = False
                    break
            if ok:
                found.append("output = the panels stacked, drawn in the order " + ", ".join(map(str, order))
                             + " (a later panel's non-background cells cover the earlier ones)")
    if not found:
        print(f"no combination of two panels (AND / OR / XOR / and-not / neither, flips) or stacking of the {n} "
              "panels gives the output in every pair")
    for line in found[:12]:
        print(line)
    if len(found) > 12:
        print(f"... {len(found) - 12} more")
    return found


def _invalid(out) -> str | None:
    """Why check.py would reject a transform output (None if it would accept it): a rectangle of ints 0-9, 1..30
    per side (numpy arrays of ints are fine; floats, bools and None are not)."""
    if out is None:
        return "returned None"
    v = out.tolist() if hasattr(out, "tolist") else out
    if not isinstance(v, (list, tuple)) or not v or not all(isinstance(r, (list, tuple)) and r for r in v):
        return f"not a non-empty list of rows ({type(out).__name__})"
    if len({len(r) for r in v}) != 1:
        return "rows have different lengths"
    for r in v:
        for x in r:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not 0 <= x <= 9:
                return f"cell {x!r} is not an int 0-9"
    if not (1 <= len(v) <= 30 and 1 <= len(v[0]) <= 30):
        return f"size {len(v)}x{len(v[0])} is outside 1..30"
    return None


def _params(p: dict) -> str:
    """A sweep combination as text: functions by name."""
    return "{" + ", ".join(f"{k}={v.__name__}" if callable(v) and hasattr(v, "__name__") else f"{k}={v!r}"
                           for k, v in p.items()) + "}"


def sweep(fn, **space) -> list[dict]:
    """Try fn(grid, **params) for every combination of the values in `space` (name=[values, ...]) on every training
    pair (grid = a list of lists, as transform gets it). Prints how many combinations pass all pairs (and how many raised, with the first error), and whether the
    passing ones agree on the test inputs -- groups of passing combinations that give different test outputs are the
    readings the training pairs cannot tell apart. Returns the passing combinations as dicts.
    Rival rules can be swept too: sweep(lambda g, f: f(g), f=[rule_a, rule_b])."""
    import itertools
    names = list(space)
    for k, v in space.items():
        if not isinstance(v, (list, tuple, range)) or not len(v):
            raise ValueError(f"sweep: {k}= must be a non-empty list of values, got {v!r}")
    combos = [dict(zip(names, vs)) for vs in itertools.product(*(space[k] for k in names))]
    passing, errors, first_error, best = [], 0, None, None
    for p in combos:
        try:
            wrong = 0
            for x, y in train:
                got = fn(x.tolist(), **p)
                bad = _invalid(got)
                if bad:
                    raise ValueError(f"check.py would reject the output: {bad}")
                out = np.asarray(got)
                wrong += y.size if out.shape != y.shape else int((out != y).sum())
        except Exception as e:  # counted and reported below: a sweep over rule variants expects some to fail
            errors += 1
            first_error = first_error or f"{_params(p)}: {type(e).__name__}: {e}"
            continue
        if wrong == 0:
            passing.append(p)
        elif best is None or wrong < best[0]:
            best = (wrong, p)
    print(f"{len(combos)} combinations: {len(passing)} pass every training pair"
          + (f", {errors} raised (first: {first_error})" if errors else ""))
    if not passing:
        if best:
            print(f"  closest: {_params(best[1])} with {best[0]} wrong cells over all pairs")
        return passing
    outs = {}
    for p in passing:
        try:
            outs_t = [fn(t.tolist(), **p) for t in test]
            bad = next((b for b in map(_invalid, outs_t) if b), None)
            key = (f"a test output check.py would reject: {bad}",) if bad else \
                tuple(json.dumps(np.asarray(o).tolist()) for o in outs_t)
        except Exception as e:  # noqa: BLE001 -- a combination that passes train but breaks on a test input is shown
            key = (f"raises {type(e).__name__}: {e}",)
        outs.setdefault(key, []).append(p)
    if len(outs) == 1:
        print("  all of them give the same test outputs")
    else:
        print(f"  they give {len(outs)} different sets of test outputs (the training pairs cannot tell them apart):")
        ref = next(iter(outs))
        for k, ps in outs.items():
            where_ = ""
            if k is not ref and not k[0].startswith(("a test output", "raises")) and not ref[0].startswith(("a test output", "raises")):
                parts = []
                for j, (x, y) in enumerate(zip(ref, k), 1):
                    x, y = np.array(json.loads(x)), np.array(json.loads(y))
                    if x.shape != y.shape:
                        parts.append(f"test {j}: other size {y.shape[0]}x{y.shape[1]}")
                    elif (x != y).any():
                        cells = [(int(r), int(c)) for r, c in np.argwhere(x != y)]
                        parts.append(f"test {j}: {len(cells)} cells differ, e.g. {cells[:4]}")
                where_ = "  vs the first: " + "; ".join(parts)
            print(f"    {len(ps)} combination(s), e.g. {_params(ps[0])}" + (f"  ({k[0]})" if k[0].startswith(("a test output", "raises")) else "")
                  + where_)
    print("  (fn gets each grid as a list of lists, as transform does)")
    return passing


def preview(fn=None, path: str = "solution.py", png: bool = False) -> list:
    """Run transform (fn, or the one in solution.py) on every test input and print each output: its size, colours,
    and the cells that differ from the input ('.' = unchanged) when the sizes match, else the whole output.
    png=True also saves out_test_<j>.png for view_image. Returns the outputs (None where check.py would reject one)."""
    if fn is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_preview_solution", _HERE / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.transform
    outs = []
    for j, t in enumerate(test, 1):
        got = fn(t.tolist())
        bad = _invalid(got)
        if bad:
            print(f"test {j}: check.py would reject this output: {bad}")
            outs.append(None)
            continue
        out = _grid(np.asarray(got, dtype=np.int64), f"transform(test {j})")
        outs.append(out)
        print(f"test {j}: input {t.shape[0]}x{t.shape[1]} -> output {out.shape[0]}x{out.shape[1]}  colours {count(out)}")
        if out.shape == t.shape:
            m = out != t
            print(f"  {int(m.sum())} cells differ from the input ('.' = same):")
            print("\n".join("  " + l for l in _view_lines(out, m)))
        else:
            print("\n".join("  " + l for l in _view_lines(out)))
        if png:
            render(out, _HERE / f"out_test_{j}.png")
            print(f"  saved out_test_{j}.png")
    return outs


# --------------------------------------------------------------------------- testing a hypothesis

def _where_raised(e: BaseException) -> str:
    """' (file.py line N)' for the innermost frame outside arc.py, or ''."""
    import traceback
    frames = [f for f in traceback.extract_tb(e.__traceback__) if Path(f.filename).resolve() != Path(__file__).resolve()]
    if not frames:
        return ""
    f = frames[-1]
    return f" ({Path(f.filename).name} line {f.lineno})"


def _wrong_cells_text(got: np.ndarray, want: np.ndarray, show: int) -> str:
    """'6 of 100 cells wrong (got->want 0->3 x4, 3->0 x2) in 2 places: rows 1-2 cols 3-4 (4) ...; first: (1,3) got 0 want 3 ...'"""
    m = got != want
    cells = [(int(r), int(c)) for r, c in np.argwhere(m)]
    kinds = Counter((int(got[r, c]), int(want[r, c])) for r, c in cells)
    clusters = _components(m, m, False, 8)
    places = "; ".join(f"rows {min(r for r, _ in k)}-{max(r for r, _ in k)} cols {min(c for _, c in k)}-"
                       f"{max(c for _, c in k)} ({len(k)})" for k in clusters[:4]) + (" ..." if len(clusters) > 4 else "")
    first = ", ".join(f"({r},{c}) got {int(got[r, c])} want {int(want[r, c])}" for r, c in cells[:show])
    return (f"{len(cells)} of {want.size} cells wrong (got->want " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in kinds.most_common(4))
            + f") in {len(clusters)} place{'s' if len(clusters) > 1 else ''}: {places}; first: {first}")


def verify(rule, show: int = 4) -> bool:
    """Run rule(grid) -> grid -- a candidate transform; it gets each grid as a list of lists, as transform does -- on
    every training pair and print, per pair, PASS or how it fails: an error (with the line), a wrong size, or the wrong
    cells (how many, which colour changes, where, the first few); then what it gives for each test input. Nothing is
    written: test a hypothesis without touching solution.py. Returns True when every training pair passes."""
    ok = 0
    for k, (a, b) in enumerate(train, 1):
        try:
            got = rule(a.tolist())
        except Exception as e:  # reported: a rule that raises on a pair fails it
            print(f"train {k}: ERROR {type(e).__name__}: {e}{_where_raised(e)}")
            continue
        bad = _invalid(got)
        if bad:
            print(f"train {k}: check.py would reject the output: {bad}")
            continue
        out = np.asarray(got, dtype=np.int64)
        if out.shape != b.shape:
            print(f"train {k}: wrong size {out.shape[0]}x{out.shape[1]}, want {b.shape[0]}x{b.shape[1]}")
        elif np.array_equal(out, b):
            print(f"train {k}: PASS")
            ok += 1
        else:
            print(f"train {k}: " + _wrong_cells_text(out, b, show))
    print(f"{ok}/{len(train)} training pairs pass")
    for j, t in enumerate(test, 1):
        try:
            got = rule(t.tolist())
        except Exception as e:  # reported: what the rule does on the test inputs is part of the answer
            print(f"test {j}: ERROR {type(e).__name__}: {e}{_where_raised(e)}")
            continue
        bad = _invalid(got)
        if bad:
            print(f"test {j}: check.py would reject the output: {bad}")
            continue
        out = np.asarray(got, dtype=np.int64)
        same = f", {int((out != t).sum())} cells differ from the input" if out.shape == t.shape else ""
        print(f"test {j}: output {out.shape[0]}x{out.shape[1]}{same} (preview(rule) shows it)")
    return ok == len(train)


def cellwise(f):
    """A transform that sets every output cell (r, c) to f(g, r, c) -- g the input as an int array -- for rules that
    decide each cell from the input around it: verify(cellwise(f)), or transform = cellwise(f) in solution.py.
    The output has the input's size; f must return a colour 0-9."""
    def rule(grid):
        g = _grid(grid).copy()
        g.setflags(write=False)              # f reads the input; writing into it would change later cells' input
        out = np.zeros(g.shape, np.int64)
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                v = f(g, r, c)
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 9:
                    raise ValueError(f"cellwise: f(g, {r}, {c}) returned {v!r}, not a colour 0-9")
                out[r, c] = v
        return out
    rule.__name__ = getattr(f, "__name__", "cellwise")
    return rule


def _call_pred(pred, o: Obj, g: np.ndarray, objs: list[Obj]):
    import inspect
    n = len([p for p in inspect.signature(pred).parameters.values()
             if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    if n not in (1, 2, 3):
        raise TypeError(f"check_fate: pred takes (o), (o, g) or (o, g, objs), not {n} arguments")
    return pred(*(o, g, objs)[:n])


def check_fate(pred, fate: str, bg=None, conn: int = 8, multicolor: bool = False) -> list:
    """Test your own rule for which objects get a fate: pred(o) -- or pred(o, g) / pred(o, g, objs) with the grid and
    all its objects, for rules about neighbours -- must be True exactly for the input objects whose fate is `fate`
    ("deleted", "kept", "moved", "recolored", "recolored to 4", ... as explain_objects names them; "recolored"
    covers every "recolored to c"), in every training pair (same pairing as print_match). Prints whether it holds,
    every object where it does not, and which objects pred picks in each test input. Returns the counter-examples
    [(pair number, Obj, pred value, its fate)] -- [] when the rule holds."""
    rows, seen = [], Counter()
    for k, (a, b) in enumerate(train, 1):
        if a.shape != b.shape:
            raise ValueError(f"check_fate: train {k} changes size ({a.shape} -> {b.shape}); fates need same-size pairs")
        bgc = _bg_arg(a, bg)
        objs = objects(a, bgc, conn, multicolor)
        fates = {l.a.idx: _fate(l) for l in match(a, b, bgc, conn, multicolor) if l.a is not None and l.kind != "copy"}
        for o in objs:
            f = fates[o.idx]
            seen[f] += 1
            rows.append((k, o, bool(_call_pred(pred, o, a, objs)), f))

    def has(f):
        return f == fate or f.startswith(fate + " ")
    if not any(has(f) for _, _, _, f in rows):
        raise ValueError(f"check_fate: no object's fate is {fate!r} (bg {bg!r}, conn {conn}); the fates here: "
                         + ", ".join(f"{f} x{n}" for f, n in seen.most_common()))
    bad = [(k, o, v, f) for k, o, v, f in rows if v != has(f)]
    name = getattr(pred, "__name__", "pred")
    name = "pred" if name == "<lambda>" else name
    if not bad:
        print(f"{name} <=> {fate}: holds for all {len(rows)} objects in {len(train)} pairs "
              f"({sum(has(f) for *_, f in rows)} with that fate)")
    else:
        print(f"{name} <=> {fate}: fails on {len(bad)} of {len(rows)} objects:")
        for k, o, v, f in bad[:12]:
            print(f"  train {k} #{o.idx} c{o.color} size {o.size} {o.h}x{o.w} at ({o.bbox[0]},{o.bbox[1]}): "
                  f"pred {v}, fate {f}")
        if len(bad) > 12:
            print(f"  ... {len(bad) - 12} more")
    for j, t in enumerate(test, 1):
        bgc = _bg_arg(t, bg)
        objs = objects(t, bgc, conn, multicolor)
        picked = [o for o in objs if _call_pred(pred, o, t, objs)]
        print(f"  test {j}: pred picks " + (" ".join(f"#{o.idx} c{o.color} {o.h}x{o.w} at ({o.bbox[0]},{o.bbox[1]})"
                                                   for o in picked[:10]) if picked else "none")
              + (f" ... +{len(picked) - 10}" if len(picked) > 10 else "") + f" of {len(objs)} objects")
    return bad


def per_pair(f) -> list:
    """Print f(input, output) for every training pair and f(input, None) for every test input, one line each -- the
    quick way to check a numeric or structural guess (counts, sizes, positions) on all pairs at once instead of
    working it out pair by pair. Returns the values (an Exception where f raised)."""
    out = []
    for name, a, b in [(f"train {k}", a, b) for k, (a, b) in enumerate(train, 1)] + \
            [(f"test {j}", t, None) for j, t in enumerate(test, 1)]:
        try:
            v = f(a, b)
        except Exception as e:  # reported per pair: a guess that breaks on one pair is shown, not hidden
            print(f"{name}: ERROR {type(e).__name__}: {e}{_where_raised(e)}")
            out.append(e)
            continue
        print(f"{name}: {v!r}")
        out.append(v)
    return out


def unseen(bg=None, conn: int = 8, multicolor: bool = False) -> list[str]:
    """What each test input has that no training input has: a grid size, colour, number of objects, object size or
    object shape (any rotation) outside what the training inputs show -- the cases a rule fitted to the training
    pairs was never checked on. Prints and returns the lines (empty when nothing is new)."""
    sizes, colours, counts, osizes, shapes = set(), set(), [], set(), set()
    for a, _ in train:
        objs = objects(a, _bg_arg(a, bg), conn, multicolor)
        sizes.add(a.shape)
        colours |= set(np.unique(a).tolist())
        counts.append(len(objs))
        osizes |= {o.size for o in objs}
        shapes |= {shape_key(o) for o in objs}
    lines_ = []
    for j, t in enumerate(test, 1):
        objs = objects(t, _bg_arg(t, bg), conn, multicolor)
        new = []
        hs, ws = [h for h, _ in sizes], [w for _, w in sizes]
        if not (min(hs) <= t.shape[0] <= max(hs) and min(ws) <= t.shape[1] <= max(ws)):
            new.append(f"size {t.shape[0]}x{t.shape[1]} (train inputs {min(hs)}-{max(hs)} x {min(ws)}-{max(ws)})")
        cs = sorted(set(np.unique(t).tolist()) - colours)
        if cs:
            new.append(f"colours {cs} (no training input has them)")
        if not min(counts) <= len(objs) <= max(counts):
            new.append(f"{len(objs)} objects (train inputs {min(counts)}-{max(counts)})")
        big = [o for o in objs if o.size not in osizes]
        if big:
            new.append("object sizes not in train: " + " ".join(f"#{o.idx} size {o.size}" for o in big[:8])
                       + (" ..." if len(big) > 8 else ""))
        odd = [o for o in objs if shape_key(o) not in shapes]
        if odd:
            new.append("shapes no training object has: " + " ".join(f"#{o.idx} {o.h}x{o.w} c{o.color}" for o in odd[:8])
                       + (" ..." if len(odd) > 8 else ""))
        for x in new:
            lines_.append(f"test {j}: {x}")
        if not new:
            lines_.append(f"test {j}: nothing the training inputs do not have (sizes, colours, object counts, sizes, shapes)")
    for x in lines_:
        print(x)
    return lines_


# --------------------------------------------------------------------------- geometry of cell sets

def _turn_cell(y: int, x: int, h: int, w: int, name: str) -> tuple[int, int]:
    """Where cell (y, x) of an h x w box lands after orient(box, name) (np.rot90 = counter-clockwise)."""
    return {"id": (y, x), "rot90": (w - 1 - x, y), "rot180": (h - 1 - y, w - 1 - x), "rot270": (x, h - 1 - y),
            "flipud": (h - 1 - y, x), "fliplr": (y, w - 1 - x), "transpose": (x, y),
            "antitranspose": (w - 1 - x, h - 1 - y)}[name]


class Fit:
    """One way align() lays a onto b: turn a by `orient` (about its bbox), then put its bbox's top-left at (r, c).
    .apply(cells) moves any cells given in a's grid coordinates the same way (e.g. a marker beside a)."""

    def __init__(self, name, a_box, r, c):
        self.orient, self._a, self.r, self.c = name, a_box, r, c

    @property
    def dr(self) -> int:
        """Row shift of the bbox's top-left corner (a pure move when orient is 'id')."""
        return self.r - self._a.r0

    @property
    def dc(self) -> int:
        return self.c - self._a.c0

    def apply(self, cells) -> Cells:
        r0, c0 = self._a.r0, self._a.c0
        h, w = self._a.h, self._a.w
        out = []
        for r, c in _cells_of(cells):
            y, x = _turn_cell(r - r0, c - c0, h, w, self.orient)
            out.append((self.r + y, self.c + x))
        return Cells(out)

    def __repr__(self) -> str:
        how = "moved" if self.orient == "id" else self.orient + ", then"
        return f"Fit({how} by ({self.dr:+d},{self.dc:+d}): a's bbox top-left ({self._a.r0},{self._a.c0}) -> ({self.r},{self.c}))"


def align(a, b, flips: bool = True, color: bool = False) -> list[Fit]:
    """Every way to lay shape a exactly onto shape b: an orientation (all 8 with flips=True, else the 4 rotations)
    and the place it goes, as Fit; [] if no turn of a has b's cells. a, b = Obj, Hit, cells or a bool mask of the
    grid, in grid coordinates. color=True also needs the colours to match (both Obj). Fit.apply(cells) maps other
    cells of a's grid the same way, e.g. where a's attached marker goes."""
    def norm(x, what):
        if color:
            if not isinstance(x, Obj):
                raise TypeError(f"align(color=True): {what} must be an Obj (cells and masks have no colours)")
            return x.bbox, np.where(x.mask, x._vals, _EMPTY)
        cells = _cells_of(x)
        if not cells:
            raise ValueError(f"align: {what} has no cell")
        m, (r0, c0) = _cells_mask(cells)
        return Box(r0, c0, r0 + m.shape[0] - 1, c0 + m.shape[1] - 1), np.where(m, 1, _EMPTY)
    abox, pa = norm(a, "a")
    bbox_, pb = norm(b, "b")
    out = []
    for name in (ORIENTS if flips else ("id", "rot90", "rot180", "rot270")):
        q = orient(pa, name)
        if q.shape == pb.shape and np.array_equal(q, pb):
            out.append(Fit(name, abox, bbox_.r0, bbox_.c0))
    return out


class Segment:
    """A straight run of cells: direction 'horizontal', 'vertical', 'diagonal \\' (down-right) or
    'diagonal /' (down-left); .cells in order from .start to .end; len() = number of cells."""

    def __init__(self, direction, cells):
        self.direction, self.cells = direction, Cells(cells)
        self.start, self.end = cells[0], cells[-1]

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        (r0, c0), (r1, c1) = self.start, self.end
        where_ = (f"r{r0} c{c0}-{c1}" if self.direction == "horizontal" else f"c{c0} r{r0}-{r1}"
                  if self.direction == "vertical" else f"({r0},{c0})-({r1},{c1})")
        return f"{self.direction} {where_} ({len(self)} cells)"


_SEG_DIRS = (("horizontal", (0, 1)), ("vertical", (1, 0)), ("diagonal \\", (1, 1)), ("diagonal /", (1, -1)))


def segments(x, min_len: int = 2) -> list[Segment]:
    """The maximal straight runs (horizontal, vertical, both diagonals) of at least min_len cells in x (an Obj,
    cells, or a bool mask), longest first, then in reading order. A cell can be in several runs (a corner is in
    a horizontal and a vertical one): this lists the strokes a shape is drawn with, not a partition."""
    if operator.index(min_len) < 2:
        raise ValueError(f"segments: min_len must be >= 2 (single cells are runs in every direction), got {min_len}")
    own = set(_cells_of(x))
    out = []
    for name, (dr, dc) in _SEG_DIRS:
        for r, c in sorted(own):
            if (r - dr, c - dc) in own:
                continue                      # not the start of a run
            run = [(r, c)]
            while (run[-1][0] + dr, run[-1][1] + dc) in own:
                run.append((run[-1][0] + dr, run[-1][1] + dc))
            if len(run) >= min_len:
                out.append(Segment(name, run))
    out.sort(key=lambda s: (-len(s), s.start, s.direction))
    return out


class Trace:
    """trace()'s result. .ends: cells with one neighbour; .branches: cells with three or more; .closed: a loop with
    no ends or branches; .cells: the cells in order along the path (from the first end in reading order, or round
    the loop from its first cell) -- None when the cells branch or fall apart, as .note says."""

    def __init__(self, cells, ends, branches, closed, note):
        self.cells, self.ends, self.branches, self.closed, self.note = cells, ends, branches, closed, note

    def __repr__(self) -> str:
        if self.cells is not None:
            kind = "closed loop" if self.closed else f"path from {self.ends[0]} to {self.ends[-1]}"
            return f"{kind}, {len(self.cells)} cells"
        return f"no single path ({self.note}); ends {self.ends}, branches {self.branches}"


def trace(x, conn: int = 8) -> Trace:
    """Walk the cells of x (an Obj, cells or a bool mask) as a line: its ends, its branch points, and the cells in
    order when it is one simple path or loop. With conn=8 a diagonal step counts only where no side-neighbour
    cell already joins the two cells (so a staircase or an L-corner is still a path, not a triangle)."""
    own = set(_cells_of(x))
    if not own:
        raise ValueError("trace: no cells")
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")

    def nbrs(p):
        r, c = p
        out = [(r + dr, c + dc) for dr, dc in _STEPS[4] if (r + dr, c + dc) in own]
        if conn == 8:
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                q = (r + dr, c + dc)
                if q in own and (r + dr, c) not in own and (r, c + dc) not in own:
                    out.append(q)
        return sorted(out)

    deg = {p: nbrs(p) for p in own}
    ends = sorted(p for p, n in deg.items() if len(n) == 1)
    branches = sorted(p for p, n in deg.items() if len(n) >= 3)
    lone = sorted(p for p, n in deg.items() if not n)
    parts = components(sorted(own), conn=conn)
    if len(own) == 1:
        return Trace(Cells(sorted(own)), [], [], False, "")
    if len(parts) > 1 or lone:
        return Trace(None, ends, branches, False, f"{len(parts)} separate pieces")
    if branches:
        return Trace(None, ends, branches, False, f"{len(branches)} branch cells")
    start = ends[0] if ends else min(own)
    order, prev = [start], None
    while True:
        nxt = [q for q in deg[order[-1]] if q != prev and q not in order[-2:]]
        if not nxt or nxt[0] == start:
            break
        prev = order[-1]
        order.append(nxt[0])
    closed = not ends
    if len(order) != len(own):
        raise RuntimeError(f"trace: walked {len(order)} of {len(own)} cells of a path without branches (bug)")
    return Trace(Cells(order), ends, branches, closed, "")


def aligned(items) -> list[tuple]:
    """Pairs of items that line up, as (i, j, how, shared): single cells (r, c) on one row ("row", r), column
    ("col", c), diagonal ("diagonal \\", r - c) or anti-diagonal ("diagonal /", r + c); Obj / Hit by the rows or
    columns their bboxes share ("rows", (r_first, r_last)) / ("cols", (c_first, c_last)). i, j index items."""
    items = list(items)
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if isinstance(a, (Obj, Hit)) or isinstance(b, (Obj, Hit)):
                if not (isinstance(a, (Obj, Hit)) and isinstance(b, (Obj, Hit))):
                    raise TypeError("aligned: give all cells (r, c) or all Obj / Hit, not a mix")
                A, B = a.bbox, b.bbox
                lo, hi = max(A.r0, B.r0), min(A.r1, B.r1)
                if lo <= hi:
                    out.append((i, j, "rows", (lo, hi)))
                lo, hi = max(A.c0, B.c0), min(A.c1, B.c1)
                if lo <= hi:
                    out.append((i, j, "cols", (lo, hi)))
                continue
            (r0, c0), (r1, c1) = _pos(a, "aligned"), _pos(b, "aligned")
            if r0 == r1:
                out.append((i, j, "row", r0))
            if c0 == c1:
                out.append((i, j, "col", c0))
            if r0 - c0 == r1 - c1:
                out.append((i, j, "diagonal \\", r0 - c0))
            if r0 + c0 == r1 + c1:
                out.append((i, j, "diagonal /", r0 + c0))
    return out


# --------------------------------------------------------------------------- packing

def _piece(x, k: int) -> tuple[np.ndarray, bool]:
    """(patch: bbox-sized int array, -1 where the piece has no cell; whether it has colours)."""
    if isinstance(x, Obj):
        return np.where(x.mask, x._vals, _EMPTY), True
    if isinstance(x, (Cells, list, tuple)) and len(x) and all(isinstance(q, tuple) and len(q) == 2 for q in x):
        m, _ = _cells_mask(x)
        return np.where(m, 0, _EMPTY), False
    a = np.asarray(x)
    if a.ndim == 2 and a.dtype == bool:
        if not a.any():
            raise ValueError(f"pack: piece {k} has no cell")
        (r0, c0), (r1, c1) = np.argwhere(a).min(axis=0), np.argwhere(a).max(axis=0)
        return np.where(a, 0, _EMPTY)[r0:r1 + 1, c0:c1 + 1], False
    if a.ndim == 2 and a.dtype.kind in "iu":
        if (a == _EMPTY).all():
            raise ValueError(f"pack: piece {k} has no cell")
        return a.astype(np.int64), True
    raise TypeError(f"pack: piece {k} must be an Obj, a list of (r, c) cells, a bool mask, or a small grid with -1 "
                    f"where it has no cell; got {type(x).__name__}")


def _region(x, what: str = "pack: region") -> set[tuple[int, int]]:
    if isinstance(x, Box):
        return {(r, c) for r in x.rows for c in x.cols}
    if isinstance(x, (Obj, Hit, Panel)):
        return set(x.cells) if not isinstance(x, Panel) else _region(x.bbox, what)
    if isinstance(x, (list, tuple)) and x and all(isinstance(q, tuple) and len(q) == 2 for q in x):
        return set(_cells_of(x))
    a = np.asarray(x)
    if a.ndim == 2 and a.dtype == bool:
        return {(int(r), int(c)) for r, c in np.argwhere(a)}
    raise TypeError(f"{what} must be a bool mask of the grid (e.g. g == 0), a list of (r, c) cells, a Box or an Obj")


class Packing:
    """One way to place every piece: .placements [(piece index, orientation, r, c)] (r, c = top-left of the turned
    piece's bbox) and .cells {(r, c): piece index}."""

    def __init__(self, placements, patches, colored):
        self.placements = placements
        self._patches, self._colored = patches, colored
        self.cells = {}
        for k, name, r, c in placements:
            q = orient(patches[k], name)
            for y, x in np.argwhere(q != _EMPTY):
                self.cells[(r + int(y), c + int(x))] = k

    def grid(self, g=None, colors=None, bg: int = 0) -> np.ndarray:
        """The pieces drawn in place on a copy of g (None: a bg-filled grid just big enough). Obj / grid pieces keep
        their colours (turned with the piece); colourless pieces (cells, masks) take colors[k]."""
        if g is None:
            H = max(r for r, _ in self.cells) + 1
            W = max(c for _, c in self.cells) + 1
            out = np.full((H, W), operator.index(bg), np.int64)
        else:
            out = _grid(g).astype(np.int64)
        for k, name, r, c in self.placements:
            q = orient(self._patches[k], name)
            if not self._colored[k]:
                if colors is None:
                    raise ValueError(f"Packing.grid: piece {k} has no colours (cells or a mask): pass colors=[...]")
                q = np.where(q != _EMPTY, operator.index(colors[k]), _EMPTY)
            for y, x in np.argwhere(q != _EMPTY):
                rr, cc = r + int(y), c + int(x)
                if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                    raise ValueError(f"Packing.grid: piece {k} lands at {(rr, cc)}, outside the grid {out.shape}")
                out[rr, cc] = q[y, x]
        return out

    def __repr__(self) -> str:
        return "packing: " + ", ".join(f"piece {k} {name} at ({r},{c})" for k, name, r, c in self.placements)


class Packings(list):
    """pack()'s result: a list of Packing; .complete is False when the search stopped at `limit` or `max_steps`."""

    complete = True
    note = ""

    def __repr__(self) -> str:
        if not self:
            head = "no packing" + ("" if self.complete else f" found before the search stopped ({self.note})")
            return head
        head = (f"{len(self)} packing{'s' if len(self) > 1 else ''} "
                + ("(the search is complete: there are no others)" if self.complete else f"found, the search stopped ({self.note}): there may be more"))
        lines_ = [head] + [f"  {p}" for p in self[:6]] + ([f"  ... +{len(self) - 6} more"] if len(self) > 6 else [])
        if len(self) > 1:
            differ = sorted({k for p in self[1:] for k, *rest in p.placements
                             if tuple(rest) != next(tuple(x[1:]) for x in self[0].placements if x[0] == k)})
            lines_.append(f"  they differ in where pieces {differ} go")
        return "\n".join(lines_)


def pack(region, pieces, rotate: bool = False, flip: bool = False, cover: bool = True, limit: int = 50,
         max_steps: int = 500_000) -> Packings:
    """Every way to place all the pieces inside the region without overlap (pieces may be translated; rotate=True
    also turns them by 90 / 180 / 270 degrees, flip=True also mirrors them). cover=True: they must fill every
    region cell (the areas must add up, or it raises); cover=False: region cells may stay empty.
    region: a bool mask of the grid (g == 0), a list of (r, c) cells, a Box or an Obj. pieces: Obj (keeps its
    colours), cells, bool masks, or small grids with -1 where there is no cell. Identical pieces are interchangeable
    (a packing is listed once, not once per swap). Returns at most `limit` packings; printing says whether the
    search found them all -- several packings are readings the pieces alone do not decide."""
    cells = _region(region)
    if not cells:
        raise ValueError("pack: the region has no cell")
    pieces = list(pieces)
    if not pieces:
        raise ValueError("pack: no pieces")
    pat, colored = zip(*(_piece(x, k) for k, x in enumerate(pieces)))
    names = ("id",) + (("rot90", "rot180", "rot270") if rotate else ()) + \
        (("flipud", "fliplr", "transpose", "antitranspose") if flip and rotate else ("flipud", "fliplr") if flip else ())
    options = []                     # per piece: [(name, [(dy, dx)] in reading order)]
    for q0 in pat:
        seen, opts = set(), []
        for name in names:
            q = orient(q0, name)
            key = (q.shape, tuple(q.ravel().tolist()))
            if key in seen:
                continue
            seen.add(key)
            opts.append((name, [(int(y), int(x)) for y, x in np.argwhere(q != _EMPTY)]))
        options.append(opts)
    area = sum(len(o[0][1]) for o in options)
    if cover and area != len(cells):
        raise ValueError(f"pack: the pieces have {area} cells and the region {len(cells)}: they cannot fill it exactly "
                         f"(cover=False lets region cells stay empty)")
    if area > len(cells):
        raise ValueError(f"pack: the pieces have {area} cells, more than the region's {len(cells)}")
    same = [next(j for j in range(k + 1) if pat[j].shape == pat[k].shape and np.array_equal(pat[j], pat[k]))
            for k in range(len(pat))]                 # the first piece identical to piece k
    order = sorted(cells)
    out = Packings()
    used, free, placed = [False] * len(pat), set(cells), []
    steps = [0]

    def fits(k, name, rel, r, c):
        return all((r + dy, c + dx) in free for dy, dx in rel)

    def put(k, name, rel, r, c):
        for dy, dx in rel:
            free.discard((r + dy, c + dx))
        used[k] = True
        placed.append((k, name, r, c))

    def take(k, rel, r, c):
        for dy, dx in rel:
            free.add((r + dy, c + dx))
        used[k] = False
        placed.pop()

    def done() -> bool:
        if len(out) >= limit:
            out.complete, out.note = False, f"limit={limit}"
            return True
        if steps[0] >= max_steps:
            out.complete, out.note = False, f"max_steps={max_steps}"
            return True
        return False

    def record():
        out.append(Packing(sorted(placed), pat, colored))

    def solve_cover() -> bool:
        steps[0] += 1
        if done():
            return True
        if all(used):
            record()
            return False
        r0, c0 = next(p for p in order if p in free)
        for k in range(len(pat)):
            if used[k] or any(not used[j] for j in range(same[k], k) if same[j] == same[k]):
                continue                     # identical pieces go in index order
            for name, rel in options[k]:
                fy, fx = rel[0]
                r, c = r0 - fy, c0 - fx
                if fits(k, name, rel, r, c):
                    put(k, name, rel, r, c)
                    stop = solve_cover()
                    take(k, rel, r, c)
                    if stop:
                        return True
        return False

    def solve_free(k: int, last: dict) -> bool:
        steps[0] += 1
        if done():
            return True
        if k == len(pat):
            record()
            return False
        for name, rel in options[k]:
            for (r0, c0) in order:
                fy, fx = rel[0]
                r, c = r0 - fy, c0 - fx
                key = (r, c, name)
                if same[k] != k and key <= last.get(same[k], (-10**9,)):
                    continue                 # identical pieces in increasing places
                if fits(k, name, rel, r, c):
                    put(k, name, rel, r, c)
                    prev = last.get(same[k])
                    last[same[k]] = key
                    stop = solve_free(k + 1, last)
                    if prev is None:
                        del last[same[k]]
                    else:
                        last[same[k]] = prev
                    take(k, rel, r, c)
                    if stop:
                        return True
        return False

    if cover:
        solve_cover()
    else:
        solve_free(0, {})
    return out


# --------------------------------------------------------------------------- overview

def _subgrid_places(o: np.ndarray, i: np.ndarray) -> list[tuple[int, int]]:
    H, W = i.shape
    h, w = o.shape
    return [(r, c) for r in range(H - h + 1) for c in range(W - w + 1) if np.array_equal(i[r:r + h, c:c + w], o)]


def _whole_grid_relations(a: np.ndarray, b: np.ndarray) -> list[str]:
    """Ways the whole output is made from the whole input: turned, recoloured, scaled, tiled."""
    out = []
    names = [n for n in ORIENTS[1:] if orient(a, n).shape == b.shape and np.array_equal(orient(a, n), b)]
    if names:
        out.append("the input " + "|".join(names))
    if a.shape == b.shape and not np.array_equal(a, b):
        cm = color_map(a, b)
        if cm is not None:
            out.append("the input recoloured " + ", ".join(f"{x}->{y}" for x, y in cm.items() if x != y))
    for k in _ratios(a.shape, b.shape):
        if b.shape == (a.shape[0] * k, a.shape[1] * k) and np.array_equal(upscale(a, k), b):
            out.append(f"the input scaled x{k}")
    for k in _ratios(b.shape, a.shape):
        if a.shape == (b.shape[0] * k, b.shape[1] * k) and np.array_equal(upscale(b, k), a):
            out.append(f"the input scaled down /{k}")
    if b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0 and b.shape != a.shape:
        n = (b.shape[0] // a.shape[0], b.shape[1] // a.shape[1])
        if np.array_equal(np.tile(a, n), b):
            out.append(f"the input tiled {n[0]}x{n[1]}")
    return out


def _border_colour(g: np.ndarray):
    e = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    return int(e[0]) if (e == e[0]).all() else None


def _panel_lines(a: np.ndarray, bga: int) -> list[str]:
    """overview's lines about full rows / columns of one colour that cut the grid into panels."""
    out = []
    sep = separators(a)
    for col in sorted({v for v in list(sep["rows"].values()) + list(sep["cols"].values())}):
        rs = [r for r, v in sep["rows"].items() if v == col] if a.shape[1] >= 5 else []
        cs = [c for c, v in sep["cols"].items() if v == col] if a.shape[0] >= 5 else []
        ps = panels(a, col) if rs or cs else []
        if len(ps) < 2:
            continue
        if col != bga:
            out.append(f"full lines of colour {col}: rows {rs} cols {cs} -> {len(ps)} panels "
                       f"(separators, or part of the picture; panels(g, {col}))")
        elif _lattice(ps):
            # lines of the most frequent colour that form a regular lattice (blank rows elsewhere do not)
            out.append(f"full lines of colour {col} (the most frequent colour) at regular steps: rows {rs} cols {cs} "
                       f"-> {len(ps)} panels of {ps[0].h}x{ps[0].w} (panels(g, {col}))")
    return out


def overview() -> None:
    """Every training pair and test input in one summary. Exact numbers, computed from the grids."""
    seen = set()
    for a, _ in train:
        seen |= set(np.unique(a).tolist())
    inputs = [(f"train {k}", a) for k, (a, _) in enumerate(train, 1)] + [(f"test {j}", t) for j, t in enumerate(test, 1)]
    bgs = {name: _bg(g, None) for name, g in inputs}
    if len(set(bgs.values())) > 1:
        print("the background (most frequent colour) differs between the inputs: "
              + ", ".join(f"{n} {v}" for n, v in bgs.items()) + " -- tools use each grid's own unless you pass bg=")
        borders = {_border_colour(g) for _, g in inputs}
        if len(borders) == 1 and None not in borders:
            print(f"  every input's border is all colour {borders.pop()}: if that is the background, pass it as bg=")
    close = {}
    for name, g in inputs:
        note = _bg_close_call(g)
        if note:
            key = re.sub(r" \(\d+ cells\)| \(\d+\)", "", note)        # the same two colours, whatever the counts
            close.setdefault(key, []).append(name)
    for note, names in close.items():
        print(f"{', '.join(names)}: {note}")
    train_multi = train_touch = False        # some training input: many touching objects / any touching colours
    for k, (a, b) in enumerate(train, 1):
        bga = _bg(a, None)
        print(f"train {k}: input {a.shape[0]}x{a.shape[1]} -> output {b.shape[0]}x{b.shape[1]}")
        print(f"  colours  input {count(a)}  output {count(b)}")
        oa, ob = objects(a, bga), objects(b, bga)
        print(f"  objects  input {len(oa)}  output {len(ob)}  (bg {bga} for both)")
        oam, obm = objects(a, bga, multicolor=True), objects(b, bga, multicolor=True)
        train_touch = train_touch or len(oam) < len(oa)
        multi = len(oa) >= 4 and 2 * len(oam) <= len(oa)       # many objects are pieces of multicolour ones
        if multi:
            train_multi = True
            print(f"  multicolour objects (touching colours joined, multicolor=True)  input {len(oam)}  output {len(obm)}")
        for line in _panel_lines(a, bga):
            print("  input has " + line)
        rel = _whole_grid_relations(a, b)
        if rel:
            print("  the output is " + " / ".join(rel))
        if a.shape == b.shape:
            m = a != b
            moves = Counter((int(a[r, c]), int(b[r, c])) for r, c in np.argwhere(m))
            listed = ", ".join(f"{x}->{y} x{n}" for (x, y), n in moves.most_common(8))
            print(f"  {int(m.sum())} of {a.size} cells change" + (f": {listed}" if moves else ""))
            if m.any():
                if multi and len(oam) <= _MAX_OBJECTS and len(obm) <= _MAX_OBJECTS:
                    kinds = Counter(l.kind for l in match(a, b, multicolor=True))
                    print("  multicolour objects (print_match(multicolor=True) for details): "
                          + ", ".join(f"{n} {k}" for k, n in kinds.items()))
                elif len(oa) <= _MAX_OBJECTS and len(ob) <= _MAX_OBJECTS:
                    kinds = Counter(l.kind for l in match(a, b))
                    print("  objects (print_match for details): " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
                if m.sum() > 0.4 * a.size:
                    print(f"  most cells change: no map here -- show(train[{k - 1}][1]) prints the output")
                else:
                    print("  the output where it differs from the input ('.' = same as the input):")
                    print("\n".join("  " + line for line in _view_lines(b, m)))
        else:
            places = _subgrid_places(b, a) if b.size >= 4 else []
            if places:
                more = f" (and {len(places) - 1} other places)" if len(places) > 1 else ""
                print(f"  the output is the input's sub-grid with top-left corner at row {places[0][0]}, "
                      f"column {places[0][1]}{more}")
            elif b.size >= 4 and b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1] or \
                    b.size >= 4 and b.shape[1] <= a.shape[0] and b.shape[0] <= a.shape[1]:
                hits = [h for h in find(a, b, flips=True) if h.orient != "id"]
                if hits:
                    print(f"  the output is the input's sub-grid turned {hits[0].orient} at ({hits[0].r},{hits[0].c})"
                          + (f" (and {len(hits) - 1} other places)" if len(hits) > 1 else ""))
            if a.size >= 4 and a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1] and a.shape != b.shape:
                hits = find(b, a, flips=True)
                if hits:
                    print(f"  the input appears in the output at ({hits[0].r},{hits[0].c}) {hits[0].orient}"
                          + (f" and {len(hits) - 1} more times" if len(hits) > 1 else ""))
            elif not rel and b.size >= 4:
                if b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0:
                    print(f"  output size = input size x ({b.shape[0] // a.shape[0]}, {b.shape[1] // a.shape[1]})")
                elif a.shape[0] % b.shape[0] == 0 and a.shape[1] % b.shape[1] == 0:
                    print(f"  output size = input size / ({a.shape[0] // b.shape[0]}, {a.shape[1] // b.shape[1]})")
            src = [o for o in oa if _as_output(o, b, a)] if len(oa) <= _MAX_OBJECTS and b.size >= 4 else []
            if src:
                print(f"  the output is input object " + ", ".join(f"#{o.idx} ({_as_output(o, b, a)})" for o in src)
                      + " (explain_select for rules that pick it)")
            same_box = [o for o in oa if (o.h, o.w) == b.shape and o not in src] if len(oa) <= _MAX_OBJECTS else []
            if same_box and not src and not places:
                o = same_box[0]
                print(f"  the output has the size of input object #{o.idx}'s bbox ({o.h}x{o.w} at ({o.bbox[0]},{o.bbox[1]}))"
                      + (f" and {len(same_box) - 1} more" if len(same_box) > 1 else ""))
    for j, t in enumerate(test, 1):
        new = sorted(set(np.unique(t).tolist()) - seen)
        bgt = int(_bg(t, None))
        ot, otm = objects(t, bgt), objects(t, bgt, multicolor=True)
        print(f"test {j}: input {t.shape[0]}x{t.shape[1]}  colours {count(t)}  objects {len(ot)} (bg {bgt})"
              + (f"  colours no training input has: {new}" if new else ""))
        if train_multi:
            print(f"  multicolour objects (touching colours joined, multicolor=True)  input {len(otm)}")
        elif len(otm) < len(ot) and not train_touch:
            print(f"  objects of different colours touch here ({len(ot)} objects, {len(otm)} with multicolor=True); "
                  f"in no training input do they")
        for line in _panel_lines(t, bgt):
            print("  test input has " + line)


# --------------------------------------------------------------------------- doc

def doc(name: str | None = None) -> None:
    """doc(): every tool with its one-line summary. doc("find"): that tool's full reference from TOOLS.md.
    doc("conventions") / doc("caveats"): the rules every tool follows / the traps."""
    import re
    text = (_HERE / "TOOLS.md").read_text()
    sections, owner, cur = [], {}, None
    for line_ in text.splitlines():
        if line_.startswith("## ") and line_[3:].strip().lower() in ("conventions", "caveats", "which tool"):
            cur = [line_]
            owner[line_[3:].strip().lower()] = cur
        elif line_.startswith("### "):
            head = line_[4:]
            while re.search(r"\([^()]*\)", head):
                head = re.sub(r"\([^()]*\)", "", head)
            names = [n.strip() for n in head.split(",") if n.strip()]
            cur = [line_]
            sections.append((names, cur))
            for n in names:
                owner[n] = cur
        elif line_.startswith("## "):
            cur = None
        elif cur is not None:
            cur.append(line_)
    if name is None:
        if "which tool" in owner:
            print("\n".join(owner["which tool"]).rstrip() + "\n")
        print("tools (doc('name') prints one; TOOLS.md has all of them):")
        for names, body in sections:
            first = next((l for l in body[1:] if l.strip()), "").strip()
            if len(first) > 90:
                first = first[:first.rfind(" ", 0, 87)] + " ..."
            print(f"  {', '.join(names):<34} {first}")
        print("doc('conventions'): positions, bbox, bg, conn, -1; doc('caveats'): the traps (bg per grid, ...)")
        return
    aliases = {"Hit": "find", "Panel": "panels", "Period": "period", "Ray": "ray", "Slide": "slide", "Link": "match",
               "Group": "groups", "Line": "lines", "Gap": "gap", "Packing": "pack", "Packings": "pack",
               "Changes": "changes", "Fit": "align", "Segment": "segments", "Trace": "trace"}
    key = aliases.get(name.split(".")[0], name.split(".")[0])
    key = key.lower() if key.lower() in ("conventions", "caveats", "which tool") else key
    if key not in owner:
        raise KeyError(f"doc: no section for {name!r}; doc() lists every tool")
    print("\n".join(owner[key]).rstrip())


if __name__ == "__main__":
    import sys
    if len(sys.argv) in (2, 3) and sys.argv[1] == "doc":
        doc(sys.argv[2] if len(sys.argv) == 3 else None)
    elif sys.argv[1:] == ["overview"]:
        overview()
    else:
        raise SystemExit("usage: python3 arc.py doc [name]  |  python3 arc.py overview")
