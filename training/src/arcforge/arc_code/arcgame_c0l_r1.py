#!/usr/bin/env python3
"""arcgame: play this game from Python. Standard library + numpy.

    from arcgame import *

    r = act("ACTION1", "ACTION6:30,40", plan="what this batch tests")
                      plays the actions in order and stops as soon as a level clears
                      or the state becomes WIN / GAME_OVER; returns a Batch
                      (print(r) for one line per action). Call it as often as you like,
                      also inside loops -- board() and status() follow every action.
    board()           the board now: numpy uint8 array (64, 64), indexed board()[y, x]
    board(12)         the board after step 12 (step 0 = the initial board)
    frames(12)        every frame step 12 returned: uint8 array (frames, 64, 64)
    status()          free: state, levels_completed, win_levels, actions_used, available
    history()         record/history.jsonl as a list of dicts, one per step
    objects(grid)     same-colour 4-connected components, as Obj (color, size, x0, y0, x1, y1,
                      cells, shape); objects(grid, colors=[8, 0xe], min_size=2) filters
    scene(grid)       the objects grouped by kind (same colour and shape) and where they are
    diff(a, b)        what changed from a to b (boards or step numbers), by object: moved,
                      appeared, disappeared, recolored, reshaped, plus the edge band;
                      print(diff(a, b)) for a summary. diff() = before the last act() of this
                      process -> now (or, without one, the step before now -> now).
    text(grid, box)   hex digits of a region, box = (x0, y0, x1, y1) inclusive

Actions: "ACTION1".."ACTION7", "RESET", and clicks as "ACTION6:x,y" or ("ACTION6", x, y);
x is the column and y the row, both 0-63. Every action counts, including RESET. An action
the game does not accept raises ActError (nothing is spent); the ones before it in the same
act() call were played.

When a command that played actions finishes, its output ends with the board now, an image
of what changed during the command and the change by object (on a new level: its scene()),
whatever you did with its printed output.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["act", "board", "frames", "status", "history", "objects", "scene", "diff", "text",
           "Batch", "Step", "Obj", "Diff", "ActError"]

BASE_URL = os.environ.get("ARC_BASE_URL", "http://127.0.0.1:8731").rstrip("/")
API_KEY = os.environ.get("ARC_API_KEY", "")
USER_AGENT = "arcgame/1"
COORD_ACTION = "ACTION6"
TERMINAL = ("WIN", "GAME_OVER")
RECORD = Path("record")
EDGE = 4            # cells this close to the border are reported apart (counters and timers sit there)
LARGE = 512         # components bigger than this are background / large regions, not objects
SCREEN = 1024       # more inner cells than this changed: a new screen, object matching is meaningless
MAX_LINES = 12
SUMMARY_LINES = 10   # object lines at the end of a command
#: the process that runs the agent's commands; its child is the command's top process
DRIVER_COMM = "codex"
OWNER_ENV = "ARCGAME_OWNER"
EMIT_LOG = ".arcgame/emit.jsonl"


class ActError(Exception):
    pass


# ---------------------------------------------------------------- server

def _request(method, path, payload=None):
    if not API_KEY:
        raise ActError("ARC_API_KEY is not set")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("X-API-Key", API_KEY)
    req.add_header("User-Agent", USER_AGENT)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code == 429 and attempt < 5:
                time.sleep(min(float(exc.headers.get("Retry-After") or 1), 10.0))
                continue
            raise ActError("%s %s -> %s: %s" % (method, path, exc.code, detail)) from None
        except urllib.error.URLError as exc:
            raise ActError("%s %s -> cannot reach %s: %s" % (method, path, BASE_URL, exc.reason)) from None
    raise ActError("%s %s -> still rate-limited after retries" % (method, path))


def _status():
    return _request("GET", "/api/status")


def _names(available):
    return ["ACTION%d" % a for a in (available or [])]


def status():
    """Free: where you are. ``step`` is the number of actions used so far (the last step in record/)."""
    s = _status()
    return {"started": bool(s.get("started")), "state": s.get("state"),
            "levels_completed": s.get("levels_completed") or 0, "win_levels": s.get("win_levels"),
            "actions_used": s.get("actions_used", 0), "step": s.get("actions_used", 0),
            "available": _names(s.get("available_actions"))}


def _grid(layer):
    return np.asarray(layer, dtype=np.uint8)


def _parse_board(path):
    rows = path.read_text().splitlines()
    if not rows or len({len(r) for r in rows}) != 1:
        raise ActError("%s is not a rectangular board" % path)
    return np.array([[int(c, 16) for c in row] for row in rows], dtype=np.uint8)


def board(step=None):
    """The board now (free), or after step ``step``. A step that returned no frame left the
    board as it was, so its board is the one of the last step before it that had one."""
    if step is None:
        s = _status()
        if not s.get("started"):
            raise ActError("the game is not open yet: act() opens it, or run ./act init")
        frame = s.get("frame") or []
        if not frame:
            raise ActError("the server returned no board")
        return _grid(frame[-1])
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ActError("step must be a non-negative int, got %r" % (step,))
    for n in range(int(step), -1, -1):
        path = RECORD / "steps" / ("%06d" % n) / "final.txt"
        if path.is_file():
            return _parse_board(path)
    raise ActError("no board recorded at or before step %d" % step)


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


# ---------------------------------------------------------------- playing

class Step:
    """One action that was played: ``step`` is its number in record/steps/."""

    def __init__(self, token, obs):
        rec = obs.get("record") or {}
        self.token = token
        self.step = rec.get("step")
        self.frames = len(obs.get("frame") or [])
        self.changed_cells = rec.get("changed_cells")
        self.state = obs.get("state")
        self.levels_completed = obs.get("levels_completed") or 0
        self.win_levels = obs.get("win_levels")
        self.available = _names(obs.get("available_actions"))
        self.record_error = rec.get("error")

    def __repr__(self):
        changed = "?" if self.changed_cells is None else self.changed_cells
        out = "%s -> step %s: state=%s level=%s/%s, %d frame(s), %s cell(s) changed" % (
            self.token, self.step, self.state, self.levels_completed, self.win_levels, self.frames, changed)
        if self.record_error:
            out += " (RECORD FAILED: %s)" % self.record_error
        return out


class Batch:
    """What one act() call played. ``before``/``after`` are the boards around it."""

    def __init__(self, before, levels_before):
        self.before = before
        self.after = before
        self.levels_before = levels_before
        self.steps = []
        self.dropped = 0

    @property
    def last(self):
        return self.steps[-1] if self.steps else None

    @property
    def state(self):
        return self.last.state if self.steps else None

    @property
    def cleared(self):
        return bool(self.steps) and self.last.levels_completed > self.levels_before

    @property
    def available(self):
        return self.last.available if self.steps else []

    def diff(self):
        return diff(self.before, self.after)

    def __repr__(self):
        lines = [repr(s) for s in self.steps]
        if self.dropped:
            why = "level cleared" if self.cleared else "state=%s" % self.state
            lines.append("%s -- the remaining %d action(s) were not sent" % (why, self.dropped))
        if self.steps:
            lines.append("available now: %s" % ", ".join(self.last.available))
        return "\n".join(lines) or "(no action played)"


def _tokens(actions):
    out = []
    for a in actions:
        if isinstance(a, list):
            out += _tokens(a)
        elif isinstance(a, tuple):
            if len(a) != 3 or a[0] != COORD_ACTION:
                raise ActError("a tuple action must be ('ACTION6', x, y), got %r" % (a,))
            out.append("%s:%d,%d" % (COORD_ACTION, a[1], a[2]))
        elif isinstance(a, str):
            out.append(a.strip())
        else:
            raise ActError("an action is a string like 'ACTION1' or 'ACTION6:30,40', got %r" % (a,))
    return out


def _payload(token):
    name, _, coords = token.partition(":")
    if name == COORD_ACTION:
        x, sep, y = coords.partition(",")
        if not sep:
            raise ActError("%s needs a cell, e.g. %s:24,20 or ('%s', 24, 20)" % (name, name, name))
        try:
            return name, {"x": int(x), "y": int(y)}
        except ValueError:
            raise ActError("%s: x and y must be integers, got %r" % (name, coords)) from None
    if coords:
        raise ActError("%s takes no coordinates" % name)
    return name, {}


def act(*actions, plan=None):
    """Play the actions in order; stop as soon as a level clears or the state is WIN / GAME_OVER.
    Opens the game first if it is not open yet (free). ``plan`` is recorded with the first action."""
    tokens = _tokens(actions)
    if not tokens:
        raise ActError("act() needs at least one action")
    payloads = [_payload(t) for t in tokens]          # every token is checked before anything is sent
    s = _status()
    if not s.get("started"):
        s = _request("POST", "/api/cmd/RESET", {})
        _WINDOW.seen(s)
    current = s.get("frame") or []
    batch = Batch(_grid(current[-1]) if current else None, s.get("levels_completed") or 0)
    for index, (token, (name, payload)) in enumerate(zip(tokens, payloads)):
        if index == 0 and plan:
            payload = dict(payload, plan=str(plan))
        obs = _request("POST", "/api/cmd/%s" % name, payload)
        _WINDOW.seen(obs)
        batch.steps.append(Step(token, obs))
        if obs.get("frame"):
            batch.after = _grid(obs["frame"][-1])
        if obs.get("state") in TERMINAL or batch.cleared:
            batch.dropped = len(tokens) - index - 1
            break
    _LAST["batch"] = batch
    return batch


_LAST = {"batch": None}


# ---------------------------------------------------------------- objects and diffs

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


def diff(a=None, b=None):
    """What changed from ``a`` to ``b`` (boards or step numbers; ``b`` defaults to the board now).
    ``a`` defaults to the board before the last act() of this process, or else the step before now."""
    if a is None and b is None and _LAST["batch"] is not None:
        a, b = _LAST["batch"].before, _LAST["batch"].after
    if a is None:
        step = status()["step"]
        a = max(step - 1, 0)
        b = step if b is None else b
    return Diff(_as_grid(a), _as_grid(b))


# ---------------------------------------------------------------- board images at the end of a command

def _stat(pid):
    with open("/proc/%d/stat" % pid) as handle:
        raw = handle.read()
    return raw[raw.index("(") + 1:raw.rindex(")")], int(raw[raw.rindex(")") + 2:].split()[1])


def _ancestors():
    """[(pid, comm)] from this process up to (not including) pid 0."""
    out, pid = [], os.getpid()
    while pid > 0 and len(out) < 64:
        try:
            comm, ppid = _stat(pid)
        except (OSError, ValueError):
            break
        out.append((pid, comm))
        pid = ppid
    return out


def _stream():
    """Where the driver reads this command's output: stdout (else stderr) of the process the
    driver started for it -- a pipe or a terminal, wherever this process's own stdout goes."""
    chain = _ancestors()
    for (pid, _), (_, parent) in zip(chain, chain[1:]):
        if parent != DRIVER_COMM:
            continue
        for fd in (1, 2):
            path = "/proc/%d/fd/%d" % (pid, fd)
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target.startswith("pipe:") or target.startswith("/dev/pts/"):
                try:
                    return os.open(path, os.O_WRONLY | os.O_APPEND), "driver-fd%d" % fd
                except OSError:
                    continue
        return None, "redirected"
    return None, "no-driver"


def _alive_ancestor(owner):
    pid, _, start = owner.partition(":")
    if not pid.isdigit():
        return False
    for ancestor, _ in _ancestors()[1:]:
        if ancestor == int(pid):
            return _start_time(ancestor) == start
    return False


def _start_time(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            raw = handle.read()
        return raw[raw.rindex(")") + 2:].split()[19]
    except (OSError, IndexError):
        return ""


class _Window:
    """The actions played while this process ran (its own and its children's), and the image refs
    of the boards around them. Only the outermost arcgame process of a command shows images: it
    reads /api/status when it is imported and again when it exits."""

    def __init__(self):
        owner = os.environ.get(OWNER_ENV, "")
        self.nested = bool(owner) and _alive_ancestor(owner)
        self.start = None        # /api/status at import (outermost process only)
        self.error = None
        self.refs = {}           # step -> image ref, for steps of this process that returned a board
        if self.nested:
            return
        os.environ[OWNER_ENV] = "%d:%s" % (os.getpid(), _start_time(os.getpid()))
        try:
            self.start = _status()
        except ActError as exc:
            self.error = str(exc)

    def seen(self, obs):
        rec = obs.get("record") or {}
        if rec.get("image_ref") and obs.get("frame"):
            self.refs[rec["step"]] = rec["image_ref"]


_WINDOW = _Window()


def _ref_with_board(ref):
    step = int(ref.split("/")[1])
    return ref if (RECORD / "steps" / ("%06d" % step) / "final.txt").is_file() else None


def _images():
    """Text for the end of the command: what was played, the board now, what changed."""
    after = _status()
    if _WINDOW.start is not None:
        before = _WINDOW.start
        used0 = before.get("actions_used", 0) if before.get("started") else 0
        first = before.get("image_ref") if before.get("started") else _WINDOW.refs.get(0)
    else:                        # /api/status failed at import: only this process's own steps are known
        before, used0, first = {}, min(_WINDOW.refs) - 1, None
    used1 = after.get("actions_used", 0)
    if not after.get("started") or used1 <= used0:
        return None
    levels0, levels1 = before.get("levels_completed") or 0, after.get("levels_completed") or 0
    last = after.get("image_ref") and _ref_with_board(after["image_ref"])
    if not last and _WINDOW.refs:
        last = _WINDOW.refs[max(_WINDOW.refs)]
    first = first and _ref_with_board(first)
    lines = ["[arcgame] this command played %d action(s), steps %d-%d: levels completed %d/%s%s, state=%s, "
             "available=%s" % (used1 - used0, used0 + 1, used1, levels1, after.get("win_levels"),
                               " (+%d)" % (levels1 - levels0) if levels1 > levels0 else "", after.get("state"),
                               ", ".join(_names(after.get("available_actions"))))]
    if not last:
        lines.append("(board image unavailable: the server sent no image reference)")
        return "\n".join(lines) + "\n"
    step = last.split("/")[1]
    lines += ["Board now (step %s, the same picture as record/current.png):" % step,
              "[[arc-image:board:%s]]" % last]
    if first and levels1 == levels0 and first != last:
        lines += ["Changes during this command (step %s -> %s): changed cells in their new colour, "
                  "unchanged cells dark navy (not a game colour):" % (first.split("/")[1], step),
                  "[[arc-image:diff:%s:%s]]" % (first, last)]
        change = Diff(board(int(first.split("/")[1])), board(int(step))).lines()
        lines += ["By object:"] + ["  " + line for line in change[:SUMMARY_LINES]]
        if len(change) > SUMMARY_LINES:
            lines.append("  ... %d more lines: print(diff(%s, %s))" % (len(change) - SUMMARY_LINES,
                                                                      first.split("/")[1], step))
    elif levels1 > levels0:
        lines += ["New level: " + scene(board(int(step)), limit=SUMMARY_LINES)]
    return "\n".join(lines) + "\n"


def _log(route, summary):
    try:
        path = Path(os.environ.get("HOME", ".")) / EMIT_LOG
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"t": round(time.time(), 3), "pid": os.getpid(), "route": route,
                                     "summary": summary}) + "\n")
    except OSError:
        pass


def _emit():
    if _WINDOW.nested or _WINDOW.start is None and not _WINDOW.refs:
        return
    try:
        out = _images()
    except ActError as exc:
        if not _WINDOW.refs:     # the server is unreachable and this process played nothing itself
            return
        out = "(board images unavailable: %s)\n" % exc
    if out is None:
        return
    for handle in (sys.stdout, sys.stderr):
        try:
            handle.flush()
        except (OSError, ValueError):
            pass
    fd, route = _stream()
    _log(route, out.splitlines()[0])
    if fd is None:
        try:
            sys.stdout.write(out)
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
        return
    try:
        os.write(fd, out.encode())
    finally:
        os.close(fd)


def _terminated(signum, frame):
    # `timeout` and session cleanup send SIGTERM; the default action skips atexit and the images
    sys.exit(128 + signum)


atexit.register(_emit)
if not _WINDOW.nested and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL \
        and threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, _terminated)
