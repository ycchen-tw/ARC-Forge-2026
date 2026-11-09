"""arc: read this game's record/ from Python. Standard library + numpy. It sends no actions.

    import sys; sys.path.insert(0, "lib"); from arc import *     (from the workspace)
    python3 lib/arc.py --help                                    (the same from the shell)

    board()            the board now (record/current.txt): numpy uint8 (64, 64), indexed board()[y, x]
    board(12)          the board after step 12 (step 0 = the initial board)
    at(b, x, y)        the cell at column x, row y -- b[y, x], so you cannot swap them by accident
    frames(12)         every frame step 12 returned: uint8 (frames, 64, 64); [-1] is its final board
    history()          record/history.jsonl as a list of dicts, one per step
    objects(b)         same-colour 4-connected components as Obj (color, size, x0, y0, x1, y1, cells,
                       shape); objects(b, colors=[8, 0xe], min_size=2) filters. Two objects of different
                       colours are always separate; touching cells of one colour are one object.
    track(a, b)        pairs the objects of board a with those of board b (same colour and shape, nearest
                       first; then same colour and overlapping): a list of (obj_in_a or None, obj_in_b or None)
    scene(b)           the objects grouped by kind (same colour and shape) and where they are
    diff(a, b)         what changed from a to b (boards or step numbers) by object: moved, appeared,
                       disappeared, recolored, reshaped, plus the edge band; print(diff(a, b))
    text(b, box)       hex digits of a region, box = (x0, y0, x1, y1) inclusive
    level_steps()      the recorded steps of the level you are on (level_steps(2): of level 2), in order
    anim(12)           what happened frame by frame in step 12 (an action that returned several frames)
    shift(a, b)        (dx, dy, share) when board b is board a moved as a whole (the view scrolled), else None
    check_sim(step)    replays every recorded action of the level you are on through your simulator
                       step(board, action, x, y) -> board and reports the first one it gets wrong
    replay(sim, first, last)
                       the same over recorded steps first..last of one level
    replace_once(path, old, new)
                       edit a file: fails loudly unless old occurs exactly once

x is the column and y the row, both 0-63, (0,0) at the top left.
"""

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["board", "at", "frames", "history", "objects", "track", "scene", "diff", "text", "replay",
           "replace_once", "Obj", "Diff", "ActError", "level_steps", "anim", "shift", "check_sim"]

#: the workspace is the directory that holds lib/ (this file is lib/arc.py)
WORKSPACE = Path(__file__).resolve().parent.parent
RECORD = WORKSPACE / "record"
EDGE = 4
LARGE = 512
SCREEN = 1024
MAX_LINES = 12


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

def text(grid, box=None):
    """Hex digits of ``grid`` (one line per row); ``box`` = (x0, y0, x1, y1) inclusive."""
    grid = _as_grid(grid)
    if box is not None:
        x0, y0, x1, y1 = box
        grid = grid[y0:y1 + 1, x0:x1 + 1]
    return "\n".join("".join("%x" % v for v in row) for row in grid)

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
    """A same-colour 4-connected component. x is the column, y the row; the box is inclusive."""

    def __init__(self, color, cells):
        self.color = int(color)
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
            raise KeyError("Obj has no field %r; fields: color, cells, size, x0, y0, x1, y1, w, h, shape, box, edge"
                           % (name,))
        return getattr(self, name)

    @property
    def edge(self):
        """Entirely inside the edge band (often a counter or timer)."""
        return self.x0 >= 64 - EDGE or self.y0 >= 64 - EDGE or self.x1 < EDGE or self.y1 < EDGE

    def __repr__(self):
        return "Obj(color %x, %dx%d, %d cells, at (%d,%d)-(%d,%d))" % (
            self.color, self.w, self.h, self.size, self.x0, self.y0, self.x1, self.y1)

def _component(grid, start, seen):
    h, w = grid.shape
    color = grid[start[1], start[0]]
    stack, cells = [start], []
    seen.add(start)
    while stack:
        x, y = stack.pop()
        cells.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and grid[ny, nx] == color:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return Obj(color, cells)

def _components(grid, seeds):
    seen, out = set(), []
    for start in sorted(seeds, key=lambda c: (c[1], c[0])):
        if start not in seen:
            out.append(_component(grid, start, seen))
    return out

def objects(grid=None, colors=None, min_size=1, max_size=None):
    """Every same-colour 4-connected component of ``grid`` (default: the board now), top to bottom."""
    grid = _as_grid(grid)
    h, w = grid.shape
    if colors is None:
        seeds = [(x, y) for y in range(h) for x in range(w)]
    else:
        mask = np.isin(grid, list(colors))
        seeds = [(int(x), int(y)) for y, x in zip(*np.nonzero(mask))]
    out = _components(grid, seeds)
    return [o for o in out if o.size >= min_size and (max_size is None or o.size <= max_size)]

def scene(grid=None, limit=16):
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
            " ".join("(%d,%d)" % xy for xy in spots[:12]), " ..." if len(spots) > 12 else "",
            " [edge band]" if all(m.edge for m in members) else ""))
    if len(order) > limit:
        out.append("  ... and %d more kinds" % (len(order) - limit))
    large = sorted(((o.size, o.color) for o in comps if o.size > LARGE), reverse=True)
    if large:
        out.append("  large regions: " + ", ".join("color %x (%d cells)" % (c, n) for n, c in large))
    return "\n".join(out)

def _edge_cell(x, y, w, h):
    return x < EDGE or y < EDGE or x >= w - EDGE or y >= h - EDGE

class Diff:
    """What changed from ``a`` to ``b``. Every list holds dicts; positions are (x, y), boxes are
    (x0, y0, x1, y1) inclusive. ``cells`` is every changed cell as (x, y, old, new)."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        if a.shape != b.shape:
            raise ActError("boards of different shapes: %r vs %r" % (a.shape, b.shape))
        h, w = b.shape
        ys, xs = np.nonzero(a != b)
        self.cells = [(int(x), int(y), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs)]
        self.count = len(self.cells)
        changed = {(x, y) for x, y, _, _ in self.cells}
        self.edge = sorted(c for c in changed if _edge_cell(c[0], c[1], w, h))
        inner = changed - set(self.edge)
        self.inner = len(inner)
        self.moved, self.recolored, self.reshaped, self.appeared, self.disappeared = [], [], [], [], []
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
        for o in old[:]:
            best = None
            for i, n in enumerate(new):
                if i in used or n.color != o.color or n.size != o.size or n.shape != o.shape or n.cells == o.cells:
                    continue
                distance = abs(n.x0 - o.x0) + abs(n.y0 - o.y0)
                if best is None or distance < best[0]:
                    best = (distance, i)
            if best is not None:
                n = new[best[1]]
                used.add(best[1])
                old.remove(o)
                self.moved.append({"color": o.color, "size": o.size, "from": (o.x0, o.y0), "to": (n.x0, n.y0),
                                   "dx": n.x0 - o.x0, "dy": n.y0 - o.y0, "box_before": o.box, "box_after": n.box})
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

    def lines(self):
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
        out += ["recolored: color %x %s (%d cells) -> color %x" % (r["color"], box(r["box"]), r["size"],
                                                                   r["new_color"]) for r in self.recolored]
        out += ["reshaped: color %x %s (%d cells) -> %s (%d cells)" % (
            r["color"], box(r["box_before"]), r["size_before"], box(r["box_after"]), r["size_after"])
            for r in self.reshaped]
        out += ["disappeared: color %x %s (%d cells)" % (d["color"], box(d["box"]), d["size"])
                for d in self.disappeared]
        out += ["appeared: color %x %s (%d cells)" % (d["color"], box(d["box"]), d["size"]) for d in self.appeared]
        if len(out) > MAX_LINES:
            out = out[:MAX_LINES] + ["... and %d more object changes" % (len(out) - MAX_LINES)]
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

    def __bool__(self):
        return bool(self.cells)


def board(step=None):
    """The board now (record/current.txt), or after step ``step``. A step that returned no frame left the
    board as it was, so its board is the one of the last step before it that had one."""
    if step is None:
        path = RECORD / "current.txt"
        if not path.is_file():
            raise ActError("no board yet (%s does not exist): open the game with ./act init" % path)
        return _parse_board(path)
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ActError("step must be a non-negative int, got %r" % (step,))
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


def replay(sim, first, last):
    """Run ``sim(board, action, x, y) -> board`` over the recorded steps ``first``..``last`` (inclusive),
    each from the recorded board before it, and return the first step whose prediction differs from the
    record as {"step", "action", "x", "y", "cells": [(x, y, predicted, actual), ...]}; None when all match.
    Stops at the end of the level ``first`` was played on (the next board is another level)."""
    rows = {r["step"]: r for r in history()}
    if first < 1 or last < first:
        raise ActError("replay needs 1 <= first <= last, got %r..%r" % (first, last))
    if first not in rows:
        raise ActError("step %d is not in record/history.jsonl" % first)
    level = rows[first]["level"]
    resets = []
    for step in range(first, last + 1):
        row = rows.get(step)
        if row is None:
            raise ActError("step %d is not in record/history.jsonl" % step)
        if row["level"] != level:
            print("replay: steps %d-%d match (step %d is on another level)%s" % (first, step - 1, step, _skipped(resets)))
            return None
        if row["action"] == "RESET":
            resets.append(step)          # a RESET restarts the level: nothing for a simulator to predict
            continue
        predicted = np.asarray(sim(board(step - 1).copy(), row["action"], row.get("x"), row.get("y")))
        actual = board(step)
        if predicted.shape != actual.shape:
            raise ActError("sim returned shape %r for step %d, the board is %r" % (predicted.shape, step,
                                                                                    actual.shape))
        ys, xs = np.nonzero(predicted != actual)
        if len(xs):
            cells = [(int(x), int(y), int(predicted[y, x]), int(actual[y, x])) for y, x in zip(ys, xs)]
            print("replay: step %d (%s) is the first wrong one: %d cell(s), e.g. %s" % (
                step, row["action"], len(cells),
                ", ".join("(%d,%d) predicted %x got %x" % c for c in cells[:6])))
            return {"step": step, "action": row["action"], "x": row.get("x"), "y": row.get("y"), "cells": cells}
        if row.get("levels_completed", 0) > rows.get(step - 1, row).get("levels_completed", 0):
            print("replay: steps %d-%d match (step %d cleared the level)%s" % (first, step, step, _skipped(resets)))
            return None
    print("replay: steps %d-%d all match%s" % (first, last, _skipped(resets)))
    return None


def _skipped(resets):
    return "; RESET step(s) %s not compared" % ", ".join(map(str, resets)) if resets else ""


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


def anim(step):
    """What happened frame by frame in ``step``: one line per frame that differs from the one before it."""
    fs = frames(step)
    if len(fs) < 2:
        return "step %d returned %d frame" % (step, len(fs))
    out = ["step %d: %d frames" % (step, len(fs))]
    start = board(step - 1) if step >= 1 else None
    prev = start
    for i, f in enumerate(fs):
        if prev is None or prev.shape != f.shape:
            out.append("  frame %d: (no board before it to compare)" % i)
        elif not (prev != f).any():
            out.append("  frame %d: same as before" % i)
        else:
            lines = Diff(prev, f).lines()
            out.append("  frame %d: %s" % (i, "; ".join(lines[:4]) + (" ..." if len(lines) > 4 else "")))
        prev = f
    if start is not None and start.shape == fs[-1].shape:
        moving = np.zeros(start.shape, dtype=bool)
        for f in fs:
            moving |= f != start
        back = int((moving & (fs[-1] == start)).sum())
        if back:
            out.append("  %d cell(s) changed during the animation and are back to how they were in the end" % back)
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


def check_sim(step_fn, level=None):
    """Replay every recorded action of ``level`` (default: the level you are on) through
    ``step_fn(board, action, x, y) -> board``, each from the recorded board before it. Prints and returns the first
    step it gets wrong (see replay); None when all match."""
    rows = [r for r in level_steps(level) if r["step"] >= 1]
    if not rows:
        raise ActError("no recorded action on that level yet")
    return replay(step_fn, rows[0]["step"], rows[-1]["step"])


def _load_step_fn(spec):
    """'lib/sim.py:step' (default name: step) -> the function."""
    import importlib.util
    path, _, name = spec.partition(":")
    spec_obj = importlib.util.spec_from_file_location("_agent_sim", path)
    if spec_obj is None:
        raise ActError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    fn = getattr(module, name or "step", None)
    if not callable(fn):
        raise ActError("%s has no function %r" % (path, name or "step"))
    return fn


def _cli(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="python3 lib/arc.py", description="read record/ from the shell")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diff", help="what changed between two steps (default: the last step -> now)")
    p.add_argument("a", type=int, nargs="?")
    p.add_argument("b", type=int, nargs="?")
    p = sub.add_parser("anim", help="frame by frame view of one step")
    p.add_argument("step", type=int)
    p = sub.add_parser("scene", help="the objects of a board, grouped by kind (default: now)")
    p.add_argument("step", type=int, nargs="?")
    p = sub.add_parser("text", help="hex digits of a region: text X0 Y0 X1 Y1 [--step N]")
    p.add_argument("box", type=int, nargs=4)
    p.add_argument("--step", type=int)
    p = sub.add_parser("shift", help="is the board after step B the board after step A moved as a whole?")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p = sub.add_parser("steps", help="the recorded actions of a level (default: the one you are on)")
    p.add_argument("level", type=int, nargs="?")
    p = sub.add_parser("check", help="replay this level's actions through your simulator: check lib/sim.py[:step]")
    p.add_argument("sim")
    p.add_argument("--level", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "diff":
        rows = history()
        b = args.b if args.b is not None else (rows[-1]["step"] if rows else 0)
        a = args.a if args.a is not None else max(b - 1, 0)
        print("step %d -> step %d:" % (a, b))
        print(diff(a, b))
    elif args.cmd == "anim":
        print(anim(args.step))
    elif args.cmd == "scene":
        print(scene(args.step))
    elif args.cmd == "text":
        print(text(args.step, tuple(args.box)))
    elif args.cmd == "shift":
        print(shift(args.a, args.b))
    elif args.cmd == "steps":
        for r in level_steps(args.level):
            print("%6d %-8s%s %s" % (r["step"], r["action"], "" if r.get("x") is None else " %d,%d" % (r["x"], r["y"]),
                                     "changed %s" % r.get("changed_cells")))
    elif args.cmd == "check":
        check_sim(_load_step_fn(args.sim), args.level)


if __name__ == "__main__":
    import sys
    _cli(sys.argv[1:])
