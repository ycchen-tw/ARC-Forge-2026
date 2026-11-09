"""arc: read this game's record/ from Python. Standard library + numpy. It sends no actions.

    import sys; sys.path.insert(0, "lib"); from arc import *     (from the workspace)
    python3 lib/arc.py --help                                    (the same from the shell)

    board()            the board now (record/current.txt): numpy uint8 (64, 64), indexed board()[y, x]
    board(12)          the board after step 12 (step 0 = the initial board); board(-1) after the last step
    load_board(path)   a board you saved: hex text (like record/current.txt), .npy, or .npz (its last frame)
    at(b, x, y)        the cell at column x, row y -- b[y, x], so you cannot swap them by accident
    cells_of(b, c)     every (x, y) of colour c;  bbox_of(b, c) -> (x0, y0, x1, y1) or None
    colour_counts(b)   {colour: cells}, most common first
    frames(12)         every frame step 12 returned: uint8 (frames, 64, 64); [-1] is its final board
    history()          record/history.jsonl as a list of dicts, one per step
    objects(b)         same-colour 4-connected components as Obj (color, size, x0, y0, x1, y1, cells,
                       shape); objects(b, colors=[8, 0xe], min_size=2) filters; conn=8 joins diagonal cells;
                       multicolor=True makes one Obj of each connected group of non-background cells, whatever
                       their colours (o.colors counts them)
    track(a, b)        pairs the objects of board a with those of board b (same colour and shape, nearest
                       first; then same colour and overlapping): a list of (obj_in_a or None, obj_in_b or None)
    scene(b)           the objects grouped by kind (same colour and shape) and where they are
    diff(a, b)         what changed from a to b (boards or step numbers): a list of (x, y, old, new), one per
                       changed cell, that also knows the objects -- d.moved, d.appeared, d.disappeared,
                       d.recolored, d.reshaped; print(diff(a, b)) says it by object
    text(b, box)       hex digits of a region, box = (x0, y0, x1, y1) inclusive; rulers=True numbers the
                       columns and rows, dots=True shows the background as '.'
    changes(a, b)      the changed cells by colour ("9->0: 24 cells in (10,20)-(15,25)") and the counters: colour
                       counts that changed in the two outer rows / columns on each side
    timeline(fn)       fn(board) after every step of the level you are on: [(step, action, value), ...]
    level_steps()      the recorded steps of the level you are on (level_steps(2): of level 2), in order
    anim(12)           what happened frame by frame in step 12 (an action that returned several frames)
    shift(a, b)        (dx, dy, share) when board b is board a moved as a whole, else None (./act do already
                       follows a scrolling view and keeps its map in map/LNN.txt)
    check(step, parse) replays the recorded actions of the level you are on through your simulator, in your
                       own state: step(state, action, x, y) -> state, parse(board) -> state; says which steps
                       it gets wrong, by action, and what the record never tried (coverage); the rest by
                       name: level=, key= (what to compare), threaded=, show=, edge=
    coverage()         what this level's record has tried: each action, each colour clicked, and what not
    label(mask)        the connected groups of True cells of a boolean array, as Obj (scipy is not installed)
    coarse(b, x0, y0, s)  the board read on a lattice of s x s blocks starting at (x0, y0): each block's most
                       common colour, as a smaller array (text() prints it)
    png(b, path, box)  save the board, or a region of it, as a PNG with the game's colours, zoomed, for
                       view_image (PIL is not installed)
    replace_once(path, old, new)
                       edit a file: fails loudly unless old occurs exactly once

x is the column and y the row, both 0-63, (0,0) at the top left.
"""

import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

__all__ = ["board", "load_board", "at", "cells_of", "bbox_of", "colour_counts", "frames", "history", "objects",
           "track", "scene", "diff", "changes", "text", "timeline", "replace_once", "Obj", "Diff", "ActError",
           "level_steps", "anim", "shift", "check", "coverage", "label", "coarse", "png", "np"]

#: the workspace is the directory that holds lib/ (this file is lib/arc.py)
WORKSPACE = Path(__file__).resolve().parent.parent
RECORD = WORKSPACE / "record"
EDGE = 4
LARGE = 512
SCREEN = 1024
MAX_LINES = 16
#: the game's colours 0-f (the ARC-AGI toolkit's)
PALETTE = ("ffffff", "cccccc", "999999", "666666", "333333", "000000", "e53aa3", "ff7bcc", "f93c31", "1e93ff",
           "88d8f1", "ffdc00", "ff851b", "921231", "4fcc30", "a356d6")


class ActError(Exception):
    """A request lib/arc.py cannot answer (no such step, a board off the record, a bad argument)."""


def _grid(layer):
    return np.asarray(layer, dtype=np.uint8)

def _parse_board(path):
    rows = path.read_text().splitlines()
    if not rows or len({len(r) for r in rows}) != 1:
        raise ActError("%s is not a rectangular board" % path)
    return np.array([[int(c, 16) for c in row] for row in rows], dtype=np.uint8)

def frames(step):
    """Every frame step ``step`` returned, in order: uint8 (frames, 64, 64); [-1] is its final board."""
    path = RECORD / "steps" / ("%06d" % step) / "frames.npz"
    if not path.is_file():
        raise ActError("step %d returned no frames (%s does not exist)" % (step, path))
    with np.load(path) as data:
        return data["frames"]

def history():
    """record/history.jsonl: one dict per step (step, level, action, x, y, plan, state, levels_completed,
    available_actions, frames, changed_cells)."""
    path = RECORD / "history.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def text(grid, box=None, rulers=False, dots=False):
    """Hex digits of ``grid`` (one line per row); ``box`` = (x0, y0, x1, y1) inclusive. ``rulers``: two header lines
    with the tens and units of each column's x, and each row's y in front. ``dots``: the most common colour of the
    whole grid (the background) as '.'."""
    grid = _as_grid(grid)
    bg = next(iter(colour_counts(grid))) if dots else None
    x0, y0 = 0, 0
    if box is not None:
        x0, y0, x1, y1 = box
        grid = grid[y0:y1 + 1, x0:x1 + 1]
    rows = ["".join("." if v == bg else "%x" % v for v in row) for row in grid]
    if not rulers:
        return "\n".join(rows)
    xs = range(x0, x0 + grid.shape[1])
    return "\n".join(["    " + "".join(str(x // 10 % 10) for x in xs), "    " + "".join(str(x % 10) for x in xs)]
                     + ["%3d %s" % (y0 + i, row) for i, row in enumerate(rows)])


def load_board(path):
    """A board you saved: hex text (one line per row, like record/current.txt), .npy, or .npz (its "frames", last
    frame; else its only array)."""
    path = Path(path)
    if path.suffix == ".npy":
        return _as_grid(np.load(path))
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "frames" in data.files:
                return _as_grid(data["frames"][-1])
            if len(data.files) != 1:
                raise ActError("%s holds %s: which one is the board?" % (path, data.files))
            return _as_grid(data[data.files[0]])
    return _parse_board(path)


def cells_of(grid, color):
    """Every (x, y) of colour ``color``, top to bottom."""
    ys, xs = np.nonzero(_as_grid(grid) == color)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]


def bbox_of(grid, color):
    """(x0, y0, x1, y1) inclusive around every cell of ``color``; None when there is none."""
    ys, xs = np.nonzero(_as_grid(grid) == color)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None


def colour_counts(grid):
    """{colour: number of cells}, the most common colour (usually the background) first."""
    values, counts = np.unique(_as_grid(grid), return_counts=True)
    return {int(v): int(n) for n, v in sorted(zip(counts, values), key=lambda p: (-p[0], p[1]))}

def _as_grid(value):
    if value is None:
        return board()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return board(int(value))
    grid = np.asarray(value)
    if grid.ndim != 2:
        raise ActError("a board is a 2-D array, got shape %r" % (grid.shape,))
    return grid

class Obj:
    """A connected component. x is the column, y the row; the box is inclusive. ``color`` is its colour (the most
    common one when it has several, see ``colors``)."""

    def __init__(self, color, cells, colors=None):
        self.color = int(color)
        self.colors = colors or {self.color: len(cells)}
        self.cells = frozenset(cells)
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        self.x0, self.y0, self.x1, self.y1 = min(xs), min(ys), max(xs), max(ys)
        self.size = len(self.cells)
        self.w, self.h = self.x1 - self.x0 + 1, self.y1 - self.y0 + 1
        #: the cells relative to the top-left of the box: equal shapes = same form, any position
        self.shape = frozenset((x - self.x0, y - self.y0) for x, y in self.cells)

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def __getitem__(self, name):
        """o["color"] works like o.color (scripts often treat objects as dicts)."""
        if not isinstance(name, str) or name.startswith("_") or not hasattr(self, name):
            raise KeyError("Obj has no field %r; fields: color, colors, cells, size, x0, y0, x1, y1, w, h, shape, box, "
                           "edge" % (name,))
        return getattr(self, name)

    @property
    def edge(self):
        """Entirely inside the edge band (often a counter or timer)."""
        return self.x0 >= 64 - EDGE or self.y0 >= 64 - EDGE or self.x1 < EDGE or self.y1 < EDGE

    def __repr__(self):
        return "Obj(color %x%s, %dx%d, %d cells, at (%d,%d)-(%d,%d))" % (
            self.color, "" if len(self.colors) == 1 else " +" + "".join("%x" % c for c in sorted(self.colors)
                                                                          if c != self.color),
            self.w, self.h, self.size, self.x0, self.y0, self.x1, self.y1)

_STEPS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_STEPS8 = _STEPS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _component(grid, start, seen, steps=_STEPS4, bg=None):
    # bg None: cells of the start's colour; else every cell that is not bg (a multi-colour object)
    h, w = grid.shape
    color = grid[start[1], start[0]]
    stack, cells = [start], []
    seen.add(start)
    while stack:
        x, y = stack.pop()
        cells.append((x, y))
        for dx, dy in steps:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and \
                    (grid[ny, nx] == color if bg is None else grid[ny, nx] != bg):
                seen.add((nx, ny))
                stack.append((nx, ny))
    if bg is None:
        return Obj(color, cells)
    counts = {}
    for x, y in cells:
        counts[int(grid[y, x])] = counts.get(int(grid[y, x]), 0) + 1
    return Obj(max(counts, key=lambda c: (counts[c], -c)), cells, counts)

def _components(grid, seeds, steps=_STEPS4, bg=None):
    seen, out = set(), []
    for start in sorted(seeds, key=lambda c: (c[1], c[0])):
        if start not in seen:
            out.append(_component(grid, start, seen, steps, bg))
    return out

def objects(grid=None, colors=None, min_size=1, max_size=None, conn=4, multicolor=False, bg=None):
    """The connected components of ``grid`` (default: the board now), top to bottom. By default one colour each,
    4-connected; ``conn=8`` also joins diagonal neighbours; ``multicolor=True`` makes an object of every connected
    group of cells that are not the background ``bg`` (default: the most common colour), whatever their colours --
    a sprite drawn in several colours is then one Obj (``o.colors`` counts them)."""
    grid = _as_grid(grid)
    h, w = grid.shape
    if conn not in (4, 8):
        raise ActError("conn is 4 or 8, not %r" % (conn,))
    steps = _STEPS8 if conn == 8 else _STEPS4
    if multicolor:
        if bg is None:
            bg = next(iter(colour_counts(grid)))
        mask = grid != bg
        if colors is not None:
            mask &= np.isin(grid, list(colors))
        seeds = [(int(x), int(y)) for y, x in zip(*np.nonzero(mask))]
        out = _components(grid, seeds, steps, bg)
    else:
        if colors is None:
            seeds = [(x, y) for y in range(h) for x in range(w)]
        else:
            mask = np.isin(grid, list(colors))
            seeds = [(int(x), int(y)) for y, x in zip(*np.nonzero(mask))]
        out = _components(grid, seeds, steps)
    return [o for o in out if o.size >= min_size and (max_size is None or o.size <= max_size)]

def scene(grid=None, limit=24):
    """The objects of ``grid`` (default: the board now) grouped by kind -- same colour and same shape --
    with where each one is (top-left x,y), largest kinds first; large regions summarised at the end."""
    grid = _as_grid(grid)
    comps = objects(grid)
    kinds = {}
    for o in comps:
        if o.size <= LARGE:
            kinds.setdefault((o.color, o.shape), []).append(o)
    order = sorted(kinds.values(), key=lambda m: (-m[0].size, -len(m), m[0].color))
    out = ["%d objects of %d kinds (same colour, 4-connected; x,y = column,row of the top-left corner):"
           % (sum(len(m) for m in order), len(order))]
    for members in order[:limit]:
        o = members[0]
        spots = sorted((m.x0, m.y0) for m in members)
        out.append("  %scolor %x %dx%d (%d cells) at %s%s%s" % (
            "%dx " % len(members) if len(members) > 1 else "", o.color, o.w, o.h, o.size,
            " ".join("(%d,%d)" % xy for xy in spots[:24]), " ..." if len(spots) > 24 else "",
            " [edge band]" if all(m.edge for m in members) else ""))
    rest = order[limit:]
    if rest:
        parts = []
        for members in rest[:REST_KINDS]:
            o = members[0]
            spots = sorted((m.x0, m.y0) for m in members)
            parts.append("%scolor %x %dx%d at %s%s" % ("%dx " % len(members) if len(members) > 1 else "", o.color,
                                                       o.w, o.h, " ".join("(%d,%d)" % xy for xy in spots[:6]),
                                                       " ..." if len(spots) > 6 else ""))
        out.append("  smaller kinds: " + "; ".join(parts) + (
            "; ... and %d more kinds" % (len(rest) - REST_KINDS) if len(rest) > REST_KINDS else ""))
    large = sorted(((o.size, o.color) for o in comps if o.size > LARGE), reverse=True)
    if large:
        out.append("  large regions: " + ", ".join("color %x (%d cells)" % (c, n) for n, c in large))
    return "\n".join(out)

#: kinds past scene's limit listed on its last line
REST_KINDS = 60


def _edge_cell(x, y, w, h):
    return x < EDGE or y < EDGE or x >= w - EDGE or y >= h - EDGE

def label(mask, conn=4):
    """The connected groups of True cells of a boolean array ``mask`` (what scipy.ndimage.label finds; scipy is not
    installed), top to bottom, as Obj (``o.cells``, ``o.box``, ``o.size``; ``o.color`` is 1)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ActError("a mask is a 2-D array, got shape %r" % (mask.shape,))
    if conn not in (4, 8):
        raise ActError("conn is 4 or 8, not %r" % (conn,))
    seeds = [(int(x), int(y)) for y, x in zip(*np.nonzero(mask))]
    return _components(mask.astype(np.uint8), seeds, _STEPS8 if conn == 8 else _STEPS4)


def coarse(grid, x0, y0, step, w=None, h=None):
    """The board read on a lattice: block (i, j) is the ``step`` x ``step`` square whose top-left is
    (x0 + i*step, y0 + j*step), and its value is the block's most common colour. Returns a uint8 array [j, i] of
    ``h`` rows and ``w`` columns (default: as many whole blocks as fit), so text(coarse(...)) prints it."""
    grid = _as_grid(grid)
    if step < 1 or x0 < 0 or y0 < 0:
        raise ActError("coarse needs x0, y0 >= 0 and step >= 1, got %r, %r, %r" % (x0, y0, step))
    w = (grid.shape[1] - x0) // step if w is None else w
    h = (grid.shape[0] - y0) // step if h is None else h
    if w < 1 or h < 1 or x0 + w * step > grid.shape[1] or y0 + h * step > grid.shape[0]:
        raise ActError("%d x %d blocks of %d from (%d,%d) do not fit on the board" % (w, h, step, x0, y0))
    blocks = grid[y0:y0 + h * step, x0:x0 + w * step].reshape(h, step, w, step).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(h, w, step * step)
    counts = np.stack([(blocks == c).sum(axis=2) for c in range(16)], axis=2)
    return counts.argmax(axis=2).astype(np.uint8)


def png_scale(w, h):
    """Pixels per cell png() uses for a region of w x h cells: about 640 pixels on its longer side, 4 to 40."""
    return max(4, min(40, 640 // max(w, h)))


def png(grid, path, box=None, scale=None):
    """Save ``grid`` (or its region ``box`` = (x0, y0, x1, y1) inclusive) as a PNG in the game's colours, each cell
    ``scale`` x ``scale`` pixels (default png_scale: about 640 pixels on the longer side) with its top and left pixel
    lines a little darker, so view_image can look at a region up close: pixel (u, v) is cell
    (x0 + u // scale, y0 + v // scale). Returns the path."""
    grid = _as_grid(grid)
    if box is not None:
        x0, y0, x1, y1 = box
        grid = grid[y0:y1 + 1, x0:x1 + 1]
    if not grid.size:
        raise ActError("empty region %r" % (box,))
    scale = png_scale(grid.shape[1], grid.shape[0]) if scale is None else scale
    rgb = np.frombuffer(bytes.fromhex("".join(PALETTE)), dtype=np.uint8).reshape(16, 3)
    img = np.repeat(np.repeat(rgb[grid], scale, axis=0), scale, axis=1)
    if scale >= 4:
        dark = (rgb.astype(np.uint16) * 3 // 4).astype(np.uint8)[grid]
        img[::scale, :] = np.repeat(dark, scale, axis=1)
        img[:, ::scale] = np.repeat(dark, scale, axis=0)
    height, width, _ = img.shape
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), img.reshape(height, width * 3)], axis=1).tobytes()

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return path


class Diff(list):
    """What changed from ``a`` to ``b``: a list of (x, y, old, new), one per changed cell, top to bottom -- so
    len(d), for x, y, old, new in d, d[:5] and sorted(d) work. By object: d.moved (only when one object of that
    colour and shape went from one place to one other), d.rearranged (several identical objects, or single cells,
    changed places: which went where is not known), d.appeared, d.disappeared, d.recolored, d.reshaped (lists of
    dicts; positions (x, y), boxes (x0, y0, x1, y1) inclusive); print(d) says it. Same-colour 4-connected objects:
    a sprite in several colours shows up as several -- changes(a, b) lists the cells by colour."""

    def __init__(self, a, b):
        if a.shape != b.shape:
            raise ActError("boards of different shapes: %r vs %r" % (a.shape, b.shape))
        h, w = b.shape
        ys, xs = np.nonzero(a != b)
        super().__init__((int(x), int(y), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs))
        self.a, self.b = a, b
        self.cells = list(self)
        self.count = len(self.cells)
        changed = {(x, y) for x, y, _, _ in self.cells}
        # a changed cell near the border is the edge band's (a counter, a timer) unless an object it belongs to,
        # before or after, reaches out of the band: then it is that object's
        band = {c for c in changed if _edge_cell(c[0], c[1], w, h)}
        reach = set()
        for grid in (a, b):
            for o in _components(grid, band):
                if o.size <= LARGE and any(not _edge_cell(x, y, w, h) for x, y in o.cells):
                    reach |= o.cells & band
        self.edge = sorted(band - reach)
        inner = changed - set(self.edge)
        self.inner = len(inner)
        self.moved, self.recolored, self.reshaped, self.appeared, self.disappeared = [], [], [], [], []
        self.rearranged = []
        self.large = {}
        self.new_screen = len(inner) > SCREEN
        if inner and not self.new_screen:
            self._match(a, b, inner)

    def _match(self, a, b, cells):
        # Seeding from changed cells alone misses the unchanged half of an object (the ring left
        # around a hole, a neighbour that merged in), so each side also seeds the other with the
        # cells of its small components that kept their colour.
        def kept(grid, comps):
            return {p for o in comps if o.size <= LARGE for p in o.cells if grid[p[1], p[0]] == o.color}
        old = _components(a, cells)
        new = _components(b, cells | kept(b, old))
        old = _components(a, cells | kept(a, new))
        for color in {o.color for o in old + new if o.size > LARGE}:
            net = sum(int(b[y, x] == color) - int(a[y, x] == color) for x, y in cells)
            if net:
                self.large[color] = net
        old = [o for o in old if o.size <= LARGE]
        new = [o for o in new if o.size <= LARGE]
        used = set()
        # a move is claimed only when it is unambiguous: one object of that colour and shape left its place and one
        # took a new place, and it is bigger than a cell; several identical ones (or single cells) are "rearranged"
        kinds = {}
        for o in old:
            kinds.setdefault((o.color, o.shape), ([], []))[0].append(o)
        for i, n in enumerate(new):
            if (n.color, n.shape) in kinds:
                kinds[(n.color, n.shape)][1].append(i)
        for (color, shape), (gone, came) in kinds.items():
            stayed = {i for i in came if any(new[i].cells == o.cells for o in gone)}
            gone = [o for o in gone if not any(new[i].cells == o.cells for i in came)]
            came = [i for i in came if i not in stayed]
            if not gone or len(gone) != len(came):
                continue
            if len(gone) == 1 and len(shape) > 1:
                o, n = gone[0], new[came[0]]
                self.moved.append({"color": o.color, "size": o.size, "from": (o.x0, o.y0), "to": (n.x0, n.y0),
                                   "dx": n.x0 - o.x0, "dy": n.y0 - o.y0, "box_before": o.box, "box_after": n.box})
            else:
                self.rearranged.append({"color": color, "size": len(shape), "count": len(gone),
                                        "w": gone[0].w, "h": gone[0].h,
                                        "from": sorted((o.x0, o.y0) for o in gone),
                                        "to": sorted((new[i].x0, new[i].y0) for i in came)})
            used.update(came)
            for o in gone:
                old.remove(o)
        for o in old[:]:
            for i, n in enumerate(new):
                if i not in used and n.cells == o.cells and n.color != o.color:
                    used.add(i)
                    old.remove(o)
                    self.recolored.append({"color": o.color, "new_color": n.color, "size": o.size, "box": o.box})
                    break
        settled = set(used)
        for o in old[:]:
            for i, n in enumerate(new):
                # two objects merging into one both overlap it, so only moves and recolours are excluded
                if i in settled or n.color != o.color:
                    continue
                if n.x0 <= o.x1 and o.x0 <= n.x1 and n.y0 <= o.y1 and o.y0 <= n.y1:
                    used.add(i)
                    old.remove(o)
                    self.reshaped.append({"color": o.color, "size_before": o.size, "size_after": n.size,
                                          "box_before": o.box, "box_after": n.box,
                                          "cells_differ": len(o.cells ^ n.cells)})
                    break
        self.disappeared = [{"color": o.color, "size": o.size, "box": o.box} for o in old]
        self.appeared = [{"color": n.color, "size": n.size, "box": n.box}
                         for i, n in enumerate(new) if i not in used]

    @property
    def added(self):
        return self.appeared

    @property
    def removed(self):
        return self.disappeared

    @property
    def box(self):
        """Box of every changed cell, or None."""
        if not self.cells:
            return None
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    def lines(self, limit=MAX_LINES):
        """The lines print(d) shows: at most ``limit`` object lines (None: all of them)."""
        def box(b):
            return "%dx%d at (%d,%d)-(%d,%d)" % (b[2] - b[0] + 1, b[3] - b[1] + 1, b[0], b[1], b[2], b[3])
        if not self.cells:
            return ["nothing changed"]
        if self.new_screen:
            ys = [c[1] for c in self.cells]
            xs = [c[0] for c in self.cells]
            return ["most of the board changed (%d cells in (%d,%d)-(%d,%d)): a new screen or a reset"
                    % (self.count, min(xs), min(ys), max(xs), max(ys))]
        out = ["moved: color %x %s (%d cells) by dx=%+d dy=%+d" % (m["color"], box(m["box_before"]), m["size"],
                                                                   m["dx"], m["dy"]) for m in self.moved]
        out += ["rearranged: %d color %x %dx%d objects (%d cells each) were at %s, now at %s (which went where is not "
                "known)" % (r["count"], r["color"], r["w"], r["h"], r["size"],
                            " ".join("(%d,%d)" % p for p in r["from"][:8]) + (" ..." if r["count"] > 8 else ""),
                            " ".join("(%d,%d)" % p for p in r["to"][:8]) + (" ..." if r["count"] > 8 else ""))
                for r in self.rearranged]
        out += ["recolored: color %x %s (%d cells) -> color %x" % (r["color"], box(r["box"]), r["size"],
                                                                   r["new_color"]) for r in self.recolored]
        out += ["reshaped: color %x %s (%d cells) -> %s (%d cells)" % (
            r["color"], box(r["box_before"]), r["size_before"], box(r["box_after"]), r["size_after"])
            for r in self.reshaped]
        out += ["disappeared: color %x %s (%d cells)" % (d["color"], box(d["box"]), d["size"])
                for d in self.disappeared]
        out += ["appeared: color %x %s (%d cells)" % (d["color"], box(d["box"]), d["size"]) for d in self.appeared]
        if limit is not None and len(out) > limit:
            cut = {}
            for line in out[limit:]:
                kind = line.split(":")[0]
                cut[kind] = cut.get(kind, 0) + 1
            out = out[:limit] + ["... and %d more: %s (d.lines(None), or diff A B --all, prints them all)" % (
                len(out) - limit, ", ".join("%d %s" % (n, k) for k, n in cut.items()))]
        out += ["large color %x region (over %d cells, e.g. background): %+d cells" % (c, LARGE, n)
                for c, n in sorted(self.large.items())]
        if self.edge:
            xs = [c[0] for c in self.edge]
            ys = [c[1] for c in self.edge]
            out.append("edge band (within %d cells of the border, often a counter or timer): %d cells in "
                       "(%d,%d)-(%d,%d)" % (EDGE, len(self.edge), min(xs), min(ys), max(xs), max(ys)))
            if not self.inner:
                out.append("nothing changed away from the edge band")
        return out

    def __repr__(self):
        return "\n".join(self.lines())

    __str__ = __repr__


def board(step=None):
    """The board now (record/current.txt), or after step ``step``. A step that returned no frame left the
    board as it was, so its board is the one of the last step before it that had one."""
    if step is None:
        path = RECORD / "current.txt"
        if not path.is_file():
            raise ActError("no board yet (%s does not exist): open the game with ./act init" % path)
        return _parse_board(path)
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise ActError("step must be an int, got %r" % (step,))
    if step < 0 or not (RECORD / "steps" / ("%06d" % step)).is_dir():
        last = max((int(p.name) for p in (RECORD / "steps").iterdir() if p.name.isdigit()), default=-1)
        if step < 0 and last + 1 + step >= 0:
            step = last + 1 + step                  # board(-1): after the last step, board(-2): the one before
        else:
            raise ActError("no step %d in the record: steps are 0..%d (board(-1) is the last)" % (step, last))
    for n in range(int(step), -1, -1):
        path = RECORD / "steps" / ("%06d" % n) / "final.txt"
        if path.is_file():
            return _parse_board(path)
    raise ActError("no board recorded at or before step %d" % step)


def at(grid, x, y):
    """The cell at column ``x``, row ``y``."""
    grid = _as_grid(grid)
    h, w = grid.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ActError("(%d,%d) is off the %dx%d board" % (x, y, w, h))
    return int(grid[y, x])


def track(a, b, min_size=1):
    """Pair the objects of ``a`` with those of ``b``: same colour and shape, nearest first; what is left,
    same colour and overlapping boxes. Unpaired objects come with None on the other side."""
    old = objects(a, min_size=min_size)
    new = objects(b, min_size=min_size)
    pairs, used_old, used_new = [], set(), set()
    candidates = sorted((abs(n.x0 - o.x0) + abs(n.y0 - o.y0), i, j)
                        for i, o in enumerate(old) for j, n in enumerate(new)
                        if o.color == n.color and o.shape == n.shape)
    for _, i, j in candidates:
        if i not in used_old and j not in used_new:
            used_old.add(i)
            used_new.add(j)
            pairs.append((old[i], new[j]))
    for i, o in enumerate(old):
        if i in used_old:
            continue
        for j, n in enumerate(new):
            if j not in used_new and n.color == o.color and n.x0 <= o.x1 and o.x0 <= n.x1 \
                    and n.y0 <= o.y1 and o.y0 <= n.y1:
                used_old.add(i)
                used_new.add(j)
                pairs.append((o, n))
                break
    pairs += [(o, None) for i, o in enumerate(old) if i not in used_old]
    pairs += [(None, n) for j, n in enumerate(new) if j not in used_new]
    return pairs


def diff(a, b=None):
    """What changed from ``a`` to ``b`` (boards or step numbers; ``b`` defaults to the board now)."""
    return Diff(_as_grid(a), board() if b is None else _as_grid(b))


def _same(p, r):
    # states may nest dicts, lists, tuples and numpy arrays; a list and a tuple with equal items are the same
    if isinstance(p, np.ndarray) or isinstance(r, np.ndarray):
        p, r = np.asarray(p), np.asarray(r)
        return p.shape == r.shape and bool((p == r).all())
    if isinstance(p, dict) and isinstance(r, dict):
        return p.keys() == r.keys() and all(_same(p[k], r[k]) for k in p)
    if isinstance(p, (list, tuple)) and isinstance(r, (list, tuple)):
        return len(p) == len(r) and all(_same(a, b) for a, b in zip(p, r))
    same = p == r
    return bool(same.all()) if isinstance(same, np.ndarray) else bool(same)


def _show(p, r):
    # how a prediction differs from what happened, in one line
    if isinstance(p, np.ndarray) and isinstance(r, np.ndarray) and p.shape == r.shape and p.ndim == 2:
        ys, xs = np.nonzero(p != r)
        return "%d cell(s) differ, e.g. %s" % (len(xs), ", ".join(
            "(%d,%d) predicted %x, real %x" % (x, y, p[y, x], r[y, x]) for y, x in list(zip(ys, xs))[:4]))
    if isinstance(p, dict) and isinstance(r, dict):
        keys = [k for k in sorted(set(p) | set(r), key=str)
                if k not in p or k not in r or not _same(p[k], r[k])]
        return "; ".join("%r: %s" % (k, "no such key in the prediction" if k not in p else
                                      "no such key in the real state" if k not in r else _show(p[k], r[k]))
                         for k in keys[:3]) + ("; and %d more key(s)" % (len(keys) - 3) if len(keys) > 3 else "")
    cut = lambda v: (repr(v)[:200] + " ...") if len(repr(v)) > 200 else repr(v)  # noqa: E731
    return "predicted %s, real %s" % (cut(p), cut(r))


def check(step, parse=None, *, level=None, key=None, threaded=True, show=3, edge=False):
    """Replay the recorded actions of ``level`` (default: the level you are on) through your simulator, in your own
    state space, print where it disagrees with what happened and what the record never tried, and return
    {"compared", "wrong": [(step, action, x, y, why), ...], "by_action": {action: [right, compared]}}.

        step(state, action, x, y) -> state   action "ACTION1".."ACTION7"; x, y the click (None otherwise)
        parse(board) -> state                default: the board itself (numpy uint8, b[y, x])
        key(state) -> what to compare        default: the whole state; e.g. key=lambda s: s["player"] to check
                                             only what your simulator models
        threaded=True                        the state is carried from the level's first board through your step,
                                             so hidden state (counters, held items) survives; after a wrong step it
                                             goes on from parse(the real board). threaded=False: every step starts
                                             from parse(the recorded board before it).

    When the state is the board, the edge band (EDGE cells along the border, where counters and timers sit) is not
    compared unless edge=True. A RESET re-parses the board. The step that cleared the level is compared with its last
    frame before the next level (not compared when it returned one frame). Right after a clear (nothing played on
    the new level yet) it checks the level just cleared. "All match" covers only what the record tried: read the
    coverage lines before you trust a plan, or an "impossible", from your simulator."""
    parse = parse or (lambda b: b.copy())
    key = key or (lambda s: s)
    rows = [r for r in level_steps(level) if r["step"] >= 1]
    if not rows and level is None and history():
        rows = [r for r in level_steps(history()[-1]["level"]) if r["step"] >= 1]
        print("check: nothing played on the new level yet; checking level %d" % history()[-1]["level"])
    if not rows:
        raise ActError("no recorded action on that level yet; --level N (check(..., level=N)) checks level N")
    if not edge:
        inner = lambda v: v[EDGE:-EDGE, EDGE:-EDGE] if isinstance(v, np.ndarray) and v.ndim == 2 else v  # noqa: E731
        key = (lambda k: lambda s: inner(k(s)))(key)
    state = parse(board(rows[0]["step"] - 1))
    done = rows[0].get("levels_completed", 0)
    wrong, skipped, overs, by_action = [], [], [], {}
    for r in rows:
        s, action, x, y = r["step"], r["action"], r.get("x"), r.get("y")
        if action == "RESET":
            state = parse(board(s))
            skipped.append("RESET %d" % s)
            continue
        real_board = board(s)
        if r.get("levels_completed", 0) > done:
            fs = frames(s)
            if len(fs) < 2:
                skipped.append("clear %d (one frame)" % s)
                continue
            real_board = fs[-2]
        before = state if threaded else parse(board(s - 1))
        try:
            predicted = step(before, action, x, y)
        except Exception as exc:
            raise ActError("your step raised on step %d (%s %s,%s): %s: %s" % (s, action, x, y, type(exc).__name__,
                                                                              exc)) from exc
        real = parse(real_board)
        tally = by_action.setdefault(action, [0, 0])
        tally[1] += 1
        if _same(key(predicted), key(real)):
            tally[0] += 1
            state = predicted
        else:
            wrong.append((s, action, x, y, _show(key(predicted), key(real))))
            state = real
        if r.get("state") == "GAME_OVER":
            overs.append(s)
    compared = sum(n for _, n in by_action.values())
    print("check: %d step(s) of level %d compared (%s): %d match, %d wrong" % (
        compared, rows[0]["level"], "threaded" if threaded else "each from the record", compared - len(wrong),
        len(wrong)))
    print("  by action: " + ", ".join("%s %d/%d" % (a, ok, n) for a, (ok, n) in sorted(by_action.items())))
    for s, action, x, y, why in wrong[:show]:
        print("  step %d %s%s: %s" % (s, action, "" if x is None else " %d,%d" % (x, y), why))
    if len(wrong) > show:
        print("  ... and %d more wrong step(s): %s" % (len(wrong) - show, ", ".join(str(w[0]) for w in wrong[show:])))
    if skipped or overs:
        print("  not compared: %s%s" % (", ".join(skipped) or "-",
                                         "; GAME_OVER at step(s) %s" % ", ".join(map(str, overs)) if overs else ""))
    print(coverage(rows[0]["level"]))
    return {"compared": compared, "wrong": wrong, "by_action": by_action}


def changes(a, b=None, limit=10, cells=0):
    """The cells that changed from ``a`` to ``b`` (boards or step numbers; ``b`` defaults to the board now), by colour:
    one line per old->new pair with its count and box, most cells first; then the counters: on each side, the colour
    counts that changed in the two outer rows or columns; then, with ``cells`` = N, the first N changed cells as
    (x,y) old->new."""
    a, b = _as_grid(a), board() if b is None else _as_grid(b)
    if a.shape != b.shape:
        raise ActError("boards of different shapes: %r vs %r" % (a.shape, b.shape))
    ys, xs = np.nonzero(a != b)
    if not len(xs):
        return "nothing changed"
    pairs = {}
    for x, y in zip(xs, ys):
        pairs.setdefault((int(a[y, x]), int(b[y, x])), []).append((int(x), int(y)))
    out = ["%d cell(s) changed in (%d,%d)-(%d,%d):" % (len(xs), xs.min(), ys.min(), xs.max(), ys.max())]
    for (o, n), where in sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]:
        cx = [c[0] for c in where]
        cy = [c[1] for c in where]
        out.append("  %x->%x: %d cell(s) in (%d,%d)-(%d,%d)" % (o, n, len(where), min(cx), min(cy), max(cx), max(cy)))
    if len(pairs) > limit:
        out.append("  ... and %d more colour pairs" % (len(pairs) - limit))
    h, w = b.shape
    sides = (("top rows 0-1", (slice(0, 2), slice(None))),
             ("bottom rows %d-%d" % (h - 2, h - 1), (slice(h - 2, h), slice(None))),
             ("left columns 0-1", (slice(None), slice(0, 2))),
             ("right columns %d-%d" % (w - 2, w - 1), (slice(None), slice(w - 2, w))))
    for name, part in sides:
        before, after = colour_counts(a[part]), colour_counts(b[part])
        moved = ["colour %x %d->%d" % (c, before.get(c, 0), after.get(c, 0)) for c in sorted(set(before) | set(after))
                 if before.get(c, 0) != after.get(c, 0)]
        if moved:
            out.append("  %s (counters?): %s" % (name, ", ".join(moved)))
    if cells:
        listed = ["(%d,%d) %x->%x" % (x, y, a[y, x], b[y, x]) for y, x in zip(ys[:cells], xs[:cells])]
        out.append("  cells (x,y) old->new, top to bottom%s:" % ("" if len(xs) <= cells else
                                                                 ", the first %d of %d" % (cells, len(xs))))
        out += ["    " + " ".join(listed[i:i + 8]) for i in range(0, len(listed), 8)]
    return "\n".join(out)


def timeline(fn, level=None):
    """[(step, action, fn(board after that step)), ...] over the level you are on (or ``level``), starting with the
    board the level started from."""
    rows = level_steps(level)
    if not rows:
        raise ActError("no recorded step on that level yet")
    first = rows[0]["step"] - 1 if rows[0]["step"] >= 1 else 0
    out = [(first, "start", fn(board(first)))] if rows[0]["step"] >= 1 else []
    return out + [(r["step"], r["action"], fn(board(r["step"]))) for r in rows]


def coverage(level=None):
    """What the record of ``level`` (default: the level you are on) has tried: each action and how often it changed
    the board, the colours clicked, the available actions never tried and the colours on the board never clicked."""
    rows = [r for r in level_steps(level) if r["step"] >= 1]
    if not rows:
        return "coverage: no recorded action on this level yet"
    tried = {}
    clicks = {}
    for r in rows:
        if r["action"] == "RESET":
            continue
        changed = (r.get("changed_cells") or 0) > 0
        n = tried.setdefault(r["action"], [0, 0])
        n[0] += 1
        n[1] += changed
        if r.get("x") is not None:
            c = clicks.setdefault(int(board(r["step"] - 1)[r["y"], r["x"]]), [0, 0])
            c[0] += 1
            c[1] += changed
    out = ["coverage of level %d (steps %d-%d): %s" % (rows[0]["level"], rows[0]["step"], rows[-1]["step"], ", ".join(
        "%s x%d (%d changed the board)" % (a, n, ch) for a, (n, ch) in sorted(tried.items())))]
    available = ["ACTION%d" % a for a in (rows[-1].get("available_actions") or [])]
    never = [a for a in available if a not in tried]
    if never:
        out.append("  never tried on this level: " + ", ".join(never))
    if clicks:
        out.append("  clicks by the colour they hit: " + ", ".join(
            "%x x%d (%d changed the board)" % (c, n, ch) for c, (n, ch) in sorted(clicks.items())))
    if "ACTION6" in available:
        now = colour_counts(board(rows[-1]["step"]))
        unclicked = [(c, n) for c, n in list(now.items())[1:] if c not in clicks]
        if unclicked:
            out.append("  colours on the board never clicked on this level: " + ", ".join(
                "%x (%d cells)" % cn for cn in unclicked))
    return "\n".join(out)


def replace_once(path, old, new):
    """Replace ``old`` with ``new`` in ``path``; raise unless ``old`` occurs exactly once."""
    path = Path(path)
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise ActError("%s: the text to replace occurs %d times, not once" % (path, n))
    path.write_text(text.replace(old, new, 1))


def level_steps(level=None):
    """The rows of record/history.jsonl played on ``level`` (default: the level you are on now), in order."""
    rows = history()
    if not rows:
        return []
    if level is None:
        level = rows[-1]["level"]
        if rows[-1].get("levels_completed", 0) >= level and rows[-1].get("state") != "WIN":
            level += 1               # the last action cleared a level: you are on the next one, nothing played yet
    return [r for r in rows if r["level"] == level]


def anim(step, limit=15):
    """What happened frame by frame in ``step``, each frame against the one before it (frame 0 against the board
    before the step); runs of unchanged frames are one line, and on a step that cleared a level the last frame is
    the next level's first board."""
    fs = frames(step)
    if len(fs) < 2:
        return "step %d returned %d frame" % (step, len(fs))
    rows = {r["step"]: r for r in history()}
    cleared = step in rows and step - 1 in rows and \
        rows[step].get("levels_completed", 0) > rows[step - 1].get("levels_completed", 0)
    out = ["step %d: %d frames%s" % (step, len(fs), "; it cleared the level: frame %d is the next level's first board"
                                     % (len(fs) - 1) if cleared else "")]
    prev = board(step - 1) if step >= 1 else None
    still = None
    shown = 0
    for i, f in enumerate(fs[:-1] if cleared else fs):
        if prev is not None and prev.shape == f.shape and not (prev != f).any():
            still = (still[0], i) if still else (i, i)
            prev = f
            continue
        if still:
            out.append("  %s: no change" % ("frame %d" % still[0] if still[0] == still[1] else "frames %d-%d" % still))
            still = None
        if shown == limit:
            out.append("  ... and more changing frames: diff(frames(%d)[k - 1], frames(%d)[k]) shows frame k"
                       % (step, step))
            prev = None
            break
        if prev is None or prev.shape != f.shape:
            out.append("  frame %d: (no board before it to compare)" % i)
        else:
            lines = Diff(prev, f).lines()
            out.append("  frame %d: %s" % (i, "; ".join(lines[:4]) + (" ..." if len(lines) > 4 else "")))
        shown += 1
        prev = f
    if still:
        out.append("  %s: no change" % ("frame %d" % still[0] if still[0] == still[1] else "frames %d-%d" % still))
    return "\n".join(out)


def shift(a, b, max_shift=16, min_share=0.85):
    """(dx, dy, share) when board ``b`` is board ``a`` moved by (dx, dy) as a whole -- the view scrolled -- judged on
    b's non-background cells (``share`` of them match); None when no shift explains b better than no shift."""
    a, b = _as_grid(a), _as_grid(b)
    if a.shape != b.shape:
        raise ActError("boards of different shapes: %r vs %r" % (a.shape, b.shape))
    h, w = b.shape
    values, counts = np.unique(b, return_counts=True)
    ys, xs = np.nonzero(b != values[np.argmax(counts)])
    if len(xs) < 30:
        return None

    def agree(dx, dy):
        sx, sy = xs - dx, ys - dy
        ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if ok.sum() * 3 < len(xs):
            return 0.0
        return float((a[sy[ok], sx[ok]] == b[ys[ok], xs[ok]]).mean())
    still = agree(0, 0)
    best = max(((agree(dx, dy), dx, dy) for dy in range(-max_shift, max_shift + 1)
                for dx in range(-max_shift, max_shift + 1) if dx or dy), key=lambda r: r[0])
    if best[0] < min_share or best[0] < still + 0.3:
        return None
    return best[1], best[2], round(best[0], 3)


def _load_fn(spec, default):
    """'lib/sim.py:step' -> the function (default name ``default``)."""
    import importlib.util
    path, _, name = spec.partition(":")
    spec_obj = importlib.util.spec_from_file_location("_agent_" + Path(path).stem, path)
    if spec_obj is None:
        raise ActError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec_obj)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec_obj.loader.exec_module(module)
    fn = getattr(module, name or default, None)
    if not callable(fn):
        raise ActError("%s has no function %r" % (path, name or default))
    return fn


def _cli(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="python3 lib/arc.py", description="read record/ from the shell")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diff", help="what changed between two steps (default: the last step -> now)")
    p.add_argument("a", type=int, nargs="?")
    p.add_argument("b", type=int, nargs="?")
    p.add_argument("--all", action="store_true", help="every object line, not the first %d" % MAX_LINES)
    p = sub.add_parser("anim", help="frame by frame view of one step")
    p.add_argument("step", type=int)
    p = sub.add_parser("scene", help="the objects of a board, grouped by kind (default: now)")
    p.add_argument("step", type=int, nargs="?")
    p = sub.add_parser("text", help="hex digits of a region with x/y rulers: text X0 Y0 X1 Y1 [--step N] "
                                    "[--frame K] [--dots]")
    p.add_argument("box", type=int, nargs=4)
    p.add_argument("--step", type=int)
    p.add_argument("--frame", type=int, help="frame K of that step's animation instead of its final board")
    p.add_argument("--dots", action="store_true", help="show the background colour as '.'")
    p = sub.add_parser("changes", help="the changed cells by colour, and the counters (default: the last step)")
    p.add_argument("a", type=int, nargs="?")
    p.add_argument("b", type=int, nargs="?")
    p.add_argument("--cells", type=int, nargs="?", const=200, default=0,
                   help="also list the changed cells as (x,y) old->new (the first N, default 200)")
    p = sub.add_parser("png", help="a region as a zoomed PNG for view_image: png X0 Y0 X1 Y1 [--step N] [--frame K] "
                                   "[--scale S] [--out zoom.png]")
    p.add_argument("box", type=int, nargs=4)
    p.add_argument("--step", type=int)
    p.add_argument("--frame", type=int, help="frame K of that step's animation instead of its final board")
    p.add_argument("--scale", type=int, help="pixels per cell (default: about 640 pixels on the longer side)")
    p.add_argument("--out", default="zoom.png")
    p = sub.add_parser("coarse", help="the board read on a lattice of STEP x STEP blocks from (X0, Y0), each block "
                                      "as its most common colour: coarse X0 Y0 STEP [--step N]")
    p.add_argument("lattice", type=int, nargs=3)
    p.add_argument("--step", type=int)
    p = sub.add_parser("shift", help="is the board after step B the board after step A moved as a whole?")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p = sub.add_parser("steps", help="the recorded actions of a level (default: the one you are on)")
    p.add_argument("level", type=int, nargs="?")
    p = sub.add_parser("check", help="replay this level's actions through your simulator: check lib/sim.py[:step] "
                                     "[--parse lib/sim.py:parse] [--each]")
    p.add_argument("sim")
    p.add_argument("--parse", help="FILE[:NAME] of parse(board) -> state (default: the board is the state)")
    p.add_argument("--level", type=int)
    p.add_argument("--each", action="store_true", help="start every step from the recorded board before it")
    p = sub.add_parser("coverage", help="what this level's record has tried, and what not")
    p.add_argument("level", type=int, nargs="?")
    args = parser.parse_args(argv)
    if args.cmd == "diff":
        rows = history()
        b = args.b if args.b is not None else (rows[-1]["step"] if rows else 0)
        a = args.a if args.a is not None else max(b - 1, 0)
        print("step %d -> step %d:" % (a, b))
        print("\n".join(diff(a, b).lines(None if args.all else MAX_LINES)))
    elif args.cmd == "anim":
        print(anim(args.step))
    elif args.cmd == "scene":
        print(scene(args.step))
    elif args.cmd == "text":
        if args.frame is not None:
            if args.step is None:
                raise ActError("--frame needs --step")
            grid = frames(args.step)[args.frame]
        else:
            grid = args.step
        print(text(grid, tuple(args.box), rulers=True, dots=args.dots))
    elif args.cmd == "changes":
        rows = history()
        b = args.b if args.b is not None else (rows[-1]["step"] if rows else 0)
        a = args.a if args.a is not None else max(b - 1, 0)
        print("step %d -> step %d:" % (a, b))
        print(changes(a, b, cells=args.cells))
    elif args.cmd == "png":
        if args.frame is not None and args.step is None:
            raise ActError("--frame needs --step")
        grid = frames(args.step)[args.frame] if args.frame is not None else args.step
        x0, y0, x1, y1 = args.box
        scale = args.scale or png_scale(x1 - x0 + 1, y1 - y0 + 1)
        print("%s: (%d,%d)-(%d,%d)%s, %d pixels per cell: pixel (u, v) is cell (%d + u // %d, %d + v // %d)" % (
            png(grid, args.out, tuple(args.box), scale), x0, y0, x1, y1,
            "" if args.step is None else " of step %d" % args.step + (
                "" if args.frame is None else " frame %d" % args.frame), scale, x0, scale, y0, scale))
    elif args.cmd == "coarse":
        x0, y0, step = args.lattice
        print("blocks of %d from (%d,%d); block (i, j) covers x %d+%d*i.., y %d+%d*j..:"
              % (step, x0, y0, x0, step, y0, step))
        print(text(coarse(args.step, x0, y0, step), rulers=True))
    elif args.cmd == "shift":
        print(shift(args.a, args.b))
    elif args.cmd == "steps":
        for r in level_steps(args.level):
            print("%6d %-8s%s %s" % (r["step"], r["action"], "" if r.get("x") is None else " %d,%d" % (r["x"], r["y"]),
                                     "changed %s" % r.get("changed_cells")))
    elif args.cmd == "check":
        check(_load_fn(args.sim, "step"), _load_fn(args.parse, "parse") if args.parse else None, level=args.level,
              threaded=not args.each)
    elif args.cmd == "coverage":
        print(coverage(args.level))


if __name__ == "__main__":
    _cli(sys.argv[1:])
