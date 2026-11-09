"""This puzzle's grids and a toolkit of exact, tested query functions.
TOOLS.md documents every tool with examples; doc() lists them, doc("find") prints one. The code below is long --
read TOOLS.md, not this file.

    from arc import *        # everything below, numpy as np included; solution.py may import from arc as well

    grids     train test show diff changes render count where background crop bbox_of Box color_map overview preview
    objects   objects print_objects Obj (.color .colors .cells .size .bbox .h .w .mask .grid() .holes .is_frame ...)
              labels components adjacency
    shapes    shape_key shape_match groups print_groups find orient ORIENTS inverse upscale downscale scale_of
    layout    separators panels tiles symmetry mirror_axes fill_by_symmetry period runs neighbors
    regions   regions enclosed flood distances
    drawing   DIRS N4 N8 turn adjacent ray line outline halo mask_of paint paste shift slide move gravity
    pairs     match print_match explain_select sweep
    help      doc() / doc("name") / doc("caveats")

Conventions: (r, c) = (row, column) from the top-left; bbox = Box(r0, c0, r1, c1) inclusive (.h .w too);
bg (background) = the grid's most frequent colour unless you pass bg=; conn = 8 for objects, 4 for regions / flood.
"""

from __future__ import annotations

import json
import operator
from collections import Counter
from functools import cached_property, reduce
from math import gcd
from pathlib import Path

import numpy as np

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
           "Hit", "Panel", "Period", "Ray", "Link", "Group"]

_PALETTE = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
            (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37))

_STEPS = {4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
          8: tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)}
_MAX_CELLS = 40       # cells shown when printing a Cells list
_MAX_CLUSTERS = 10    # diff clusters listed
_CLUSTER_CELLS = 6    # cells listed per diff cluster
_MAX_OBJECTS = 30     # print_objects lines
_MAX_MASK = 6         # print_objects draws masks up to this bbox height / width
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
    """g as a non-empty 2-D integer (or bool) array. Integral floats (np.zeros) are accepted; anything else raises."""
    a = np.asarray(g)
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
        return operator.index(bg)
    if a.dtype == bool:
        return False
    vals, n = np.unique(a, return_counts=True)
    return int(vals[n.argmax()])


def _colors(c, what: str = "colour") -> set[int]:
    """An int or an iterable of ints as a set of colours."""
    if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
        return {int(c)}
    out = {operator.index(v) for v in c}
    if not out:
        raise ValueError(f"{what}: empty set of colours")
    return out


def _components(a: np.ndarray, member: np.ndarray, same_color: bool, conn: int) -> list[list[tuple[int, int]]]:
    """Connected groups of the True cells of `member` (cells in reading order), sorted by (top row, left col)."""
    if conn not in _STEPS:
        raise ValueError(f"conn must be 4 or 8, got {conn!r}")
    steps = _STEPS[conn]
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
    keep = ~np.isin(a, list(_colors(dot, "dot"))) if dot is not None else None
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

class Cells(list):
    """A list of (r, c); print() stops after 40 cells with '... +N more' (the list itself is complete)."""

    def __repr__(self) -> str:
        if len(self) <= _MAX_CELLS:
            return list.__repr__(self)
        return list.__repr__(self[:_MAX_CELLS])[:-1] + f", ... +{len(self) - _MAX_CELLS} more]"


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


def changes(a, b) -> list[tuple[int, int, int, int]]:
    """[(r, c, colour in a, colour in b)] for every cell where two same-size grids differ, in reading order."""
    a, b = _grid(a, "a"), _grid(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"changes: sizes differ, {a.shape} vs {b.shape}")
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in np.argwhere(a != b)]


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


def objects(g, bg=None, conn: int = 8, multicolor: bool = False) -> list[Obj]:
    """Connected components of the non-bg cells (bg = most frequent colour), sorted by (top row, left col)."""
    a = _grid(g)
    b = _bg(a, bg)
    comps = _components(a, a != b, not multicolor, conn)
    return [Obj(i, a, cl, bg=int(b)) for i, cl in enumerate(comps)]


def print_objects(g, bg=None, conn: int = 8, multicolor: bool = False) -> None:
    """One line per object, plus its mask when the bbox is at most 6x6."""
    a = _grid(g)
    b = _bg(a, bg)
    objs = objects(a, b, conn, multicolor)
    print(f"{len(objs)} objects (bg {int(b)}, conn {conn}, {'multicolor' if multicolor else 'same-colour'})")
    for o in objs[:_MAX_OBJECTS]:
        print(o)
        if 1 < o.h * o.w and o.h <= _MAX_MASK and o.w <= _MAX_MASK:
            print("\n".join("    " + row for row in o.draw()))
    if len(objs) > _MAX_OBJECTS:
        print(f"... {len(objs) - _MAX_OBJECTS} more")


def labels(g, bg=None, conn: int = 8, multicolor: bool = False) -> np.ndarray:
    """An int array of the grid's shape: the idx of the object (as objects() numbers them) each cell belongs to, -1 on bg."""
    a = _grid(g)
    out = np.full(a.shape, -1, np.int64)
    for o in objects(a, bg, conn, multicolor):
        for r, c in o.cells:
            out[r, c] = o.idx
    return out


def components(x, conn: int = 8) -> list[Cells]:
    """Connected groups of a bool mask's True cells, or of a set of (r, c) cells, sorted by (top row, left col)."""
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
    return [Cells((r + off[0], c + off[1]) for r, c in cl) for cl in _components(m, m, False, conn)]


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


# --------------------------------------------------------------------------- orientation and scale

def orient(g, name: str) -> np.ndarray:
    """g turned by one of ORIENTS: id rot90 rot180 rot270 (np.rot90 = counter-clockwise) flipud fliplr transpose antitranspose."""
    if name not in _ORIENT:
        raise ValueError(f"orient: name must be one of {ORIENTS}, got {name!r}")
    if isinstance(g, (Obj, Hit, Panel)):
        raise TypeError("orient takes a grid or mask: use o.grid(), o.grid(-1) or o.mask for an Obj")
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


# --------------------------------------------------------------------------- shapes

def _patch(x, bg, what: str) -> np.ndarray:
    """x cropped to its cells' bbox as an int array, _EMPTY where x has no cell.
    x = Obj (its cells), bool mask (True cells, value 1), or grid (cells != bg; bg None = its most frequent colour)."""
    if isinstance(x, Obj):
        return np.where(x.mask, x._vals, _EMPTY)
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


def panels(g, color=None) -> list[Panel]:
    """The grid cut along full rows / columns of the separator colour (thick lines count once), in reading order.
    color=None: among the colours whose full lines give at least 2 panels, one other than the grid's most frequent
    colour first, then the most lines, then the one least used inside the panels; the most frequent colour only if it
    never occurs inside its panels. No such colour -> one panel, the whole grid."""
    a = _grid(g)
    sep = separators(a)
    if color is None:
        usable = {"rows": sep["rows"] if a.shape[1] >= 3 else {}, "cols": sep["cols"] if a.shape[0] >= 3 else {}}
        n = Counter(list(usable["rows"].values()) + list(usable["cols"].values()))
        gbg = _bg(a, None)
        cands = []
        for col, k in n.items():
            rest = a[~_line_cells(a, usable, col)]
            if not rest.size or len(panels(a, col)) < 2:
                continue
            inside = float(np.mean(rest == col))
            if col == gbg and inside > 0:
                continue                  # blank background rows / columns between pictures are no separators
            cands.append((col == gbg, -k, inside, col))   # not the background, most lines, least used inside
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

    def __init__(self, pr, pc, tile, wrong, shape):
        self.pr, self.pc, self.tile, self.wrong, self._shape = pr, pc, tile, wrong, shape

    def tiled(self) -> np.ndarray:
        """The full grid rebuilt from the tile (fixes the wrong / ignored cells); raises if a tile cell is unknown."""
        if (self.tile == _EMPTY).any():
            raise ValueError("Period.tiled: some tile cells (-1 in .tile) are only covered by ignored cells -- the tile "
                             "is probably too small; try max_wrong= instead of ignore=, or axis=")
        H, W = self._shape
        return np.tile(self.tile, (-(-H // self.pr), -(-W // self.pc)))[:H, :W]

    def __repr__(self) -> str:
        H, W = self._shape
        return f"period {self.pr}x{self.pc} ({H / self.pr:.3g} x {W / self.pc:.3g} repeats), {len(self.wrong)} cells off the tiling"


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
    for pr, pc in sorted(((pr, pc) for pr in sides_r for pc in sides_c), key=lambda t: (t[0] * t[1], t[0])):
        cls = (rr % pr) * pc + (cc % pc)
        votes = np.zeros((pr * pc, len(vals)), np.int64)
        np.add.at(votes, (cls[~skip], codes[~skip]), 1)
        tile = np.where(votes.sum(axis=1) > 0, vals[votes.argmax(axis=1)], _EMPTY).reshape(pr, pc)
        wrong = (a != tile[rr % pr, cc % pc]) & ~skip
        if int(wrong.sum()) <= max_wrong:
            return Period(pr, pc, tile, Cells((int(r), int(c)) for r, c in np.argwhere(wrong)), a.shape)
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


def flood(g, start, conn: int = 4, through=None) -> Cells:
    """Cells reachable from start=(r, c) through cells of start's colour (or of the colours in `through`)."""
    a = _grid(g)
    r, c = _pos(start, "flood")
    if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
        raise ValueError(f"flood: start {start} is outside the grid {a.shape}")
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


def fill_by_symmetry(g, ignore, using) -> np.ndarray:
    """A copy of g whose cells of the `ignore` colour(s) get the colour of their mirror image under the symmetries
    in `using`: orientation names that keep the grid's shape ("fliplr", "rot180", ...) and / or mirror lines as
    mirror_axes gives them (("col", 4.5, ...), ("row", 3.0, ...)). Repeated until nothing changes; cells no
    symmetry can fill keep their colour."""
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
    return a


def shift(cells, dr: int, dc: int) -> Cells:
    """The cells (Obj or (r, c) list) moved by (dr, dc)."""
    dr, dc = _pos((dr, dc), "shift")
    return Cells((r + dr, c + dc) for r, c in _cells_of(cells))


def slide(g, obj, direction, bg=None) -> int:
    """How many steps obj (Obj or cells) can move in direction before it would leave the grid or overlap a
    cell that is not bg and not its own (bg = most frequent colour)."""
    a = _grid(g)
    b = _bg(a, bg)
    dr, dc = _dir(direction)
    own = set(_cells_of(obj))
    if not own:
        raise ValueError("slide: no cells to move")
    n = 0
    while True:
        nxt = [(r + (n + 1) * dr, c + (n + 1) * dc) for r, c in own]
        if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or (a[r, c] != b and (r, c) not in own) for r, c in nxt):
            return n
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
    bgc = _bg(a, bg)
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
        bgc = _bg(x, bg)
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
            print(f"train {k}: the output is not one of the input's objects (bg {_bg(a, bg)}, conn {conn}"
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
            if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 9:
                return f"cell {x!r} is not an int 0-9"
    if not (1 <= len(v) <= 30 and 1 <= len(v[0]) <= 30):
        return f"size {len(v)}x{len(v[0])} is outside 1..30"
    return None


def sweep(fn, **space) -> list[dict]:
    """Try fn(grid, **params) for every combination of the values in `space` (name=[values, ...]) on every training
    pair (grid = a list of lists, as transform gets it). Prints how many combinations pass all pairs (and how many raised, with the first error), and whether the
    passing ones agree on the test inputs -- groups of passing combinations that give different test outputs are the
    readings the training pairs cannot tell apart. Returns the passing combinations as dicts."""
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
            first_error = first_error or f"{p}: {type(e).__name__}: {e}"
            continue
        if wrong == 0:
            passing.append(p)
        elif best is None or wrong < best[0]:
            best = (wrong, p)
    print(f"{len(combos)} combinations: {len(passing)} pass every training pair"
          + (f", {errors} raised (first: {first_error})" if errors else ""))
    if not passing:
        if best:
            print(f"  closest: {best[1]} with {best[0]} wrong cells over all pairs")
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
        print(f"  they give {len(outs)} different sets of test outputs:")
        for k, ps in outs.items():
            print(f"    {len(ps)} combination(s), e.g. {ps[0]}" + (f"  ({k[0]})" if isinstance(k[0], str) else ""))
    print("  (fn gets each grid as a list of lists, as transform does)")
    return passing


def preview(fn=None, path: str = "solution.py", png: bool = False) -> None:
    """Run transform (fn, or the one in solution.py) on every test input and print each output: its size, colours,
    and the cells that differ from the input ('.' = unchanged) when the sizes match, else the whole output.
    png=True also saves out_test_<j>.png for view_image."""
    if fn is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_preview_solution", _HERE / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.transform
    for j, t in enumerate(test, 1):
        got = fn(t.tolist())
        bad = _invalid(got)
        if bad:
            print(f"test {j}: check.py would reject this output: {bad}")
            continue
        out = _grid(got, f"transform(test {j})")
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
    for k, (a, b) in enumerate(train, 1):
        bga = _bg(a, None)
        print(f"train {k}: input {a.shape[0]}x{a.shape[1]} -> output {b.shape[0]}x{b.shape[1]}")
        print(f"  colours  input {count(a)}  output {count(b)}")
        oa, ob = objects(a, bga), objects(b, bga)
        print(f"  objects  input {len(oa)}  output {len(ob)}  (bg {bga} for both)")
        sep = separators(a)
        for col in sorted({v for v in list(sep["rows"].values()) + list(sep["cols"].values())}):
            rs = [r for r, v in sep["rows"].items() if v == col] if a.shape[1] >= 5 else []
            cs = [c for c, v in sep["cols"].items() if v == col] if a.shape[0] >= 5 else []
            n = len(panels(a, col)) if rs or cs else 0
            if n >= 2 and col != bga:
                print(f"  input has full lines of colour {col}: rows {rs} cols {cs} -> {n} panels (separators, or part of the picture)")
        rel = _whole_grid_relations(a, b)
        if rel:
            print("  the output is " + " / ".join(rel))
        if a.shape == b.shape:
            m = a != b
            moves = Counter((int(a[r, c]), int(b[r, c])) for r, c in np.argwhere(m))
            listed = ", ".join(f"{x}->{y} x{n}" for (x, y), n in moves.most_common(8))
            print(f"  {int(m.sum())} of {a.size} cells change" + (f": {listed}" if moves else ""))
            if m.any():
                if len(oa) <= _MAX_OBJECTS and len(ob) <= _MAX_OBJECTS:
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
        print(f"test {j}: input {t.shape[0]}x{t.shape[1]}  colours {count(t)}  objects {len(objects(t))} (bg {int(_bg(t, None))})"
              + (f"  colours no training input has: {new}" if new else ""))


# --------------------------------------------------------------------------- doc

def doc(name: str | None = None) -> None:
    """doc(): every tool with its one-line summary. doc("find"): that tool's full reference from TOOLS.md.
    doc("conventions") / doc("caveats"): the rules every tool follows / the traps."""
    import re
    text = (_HERE / "TOOLS.md").read_text()
    sections, owner, cur = [], {}, None
    for line_ in text.splitlines():
        if line_.startswith("## ") and line_[3:].strip().lower() in ("conventions", "caveats"):
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
        print("tools (doc('name') prints one; TOOLS.md has all of them):")
        for names, body in sections:
            first = next((l for l in body[1:] if l.strip()), "").strip()
            if len(first) > 90:
                first = first[:first.rfind(" ", 0, 87)] + " ..."
            print(f"  {', '.join(names):<34} {first}")
        print("doc('conventions'): positions, bbox, bg, conn, -1; doc('caveats'): the traps (bg per grid, ...)")
        return
    aliases = {"Hit": "find", "Panel": "panels", "Period": "period", "Ray": "ray", "Link": "match", "Group": "groups"}
    key = aliases.get(name.split(".")[0], name.split(".")[0])
    key = key.lower() if key.lower() in ("conventions", "caveats") else key
    if key not in owner:
        raise KeyError(f"doc: no section for {name!r}; doc() lists every tool")
    print("\n".join(owner[key]).rstrip())
