"""act's view tracker: follows the view of a level that scrolls, stitches what was seen into map/LNN.txt, and says
after each step whether the picture slid. act runs this as one child process per `./act do` (it needs numpy; act is
standard library only): one JSON request per step on stdin, one JSON answer per line on stdout.

Shift convention: a content shift (dx, dy) means new[y][x] == old[y - dy][x - dx] -- the picture moved by (dx, dy),
the window moved by (-dx, -dy). The window position is relative to the level's first view.

Two modes (argv[2]): "screen" (tk2/tk3, the default) writes map/LNN.txt indexed from its own top-left, which moves as
the map grows; "fixed" (tk4) writes it in world coordinates that never move -- world (0, 0) is the top-left of the
level's first view, board (x, y) = world (x + cam_x, y + cam_y) -- lists the cells out of view whose colour changed
while they were watched (a thing that moved off, or a change), also writes map/LNN.npz for lib/arc.py's load_map(), logs every step's window position to
.act_view/camera.jsonl for cam(step), and answers each step with the slide (dx, dy) so act can undo it in state
lines."""
import json
import os
import sys
from pathlib import Path

import numpy as np

N = 64                 # ARC-AGI-3 boards are 64 x 64
PAD = 128              # FFT size: a shift search over a 64-cell board without wrap-around
ERR_MAX = 0.12         # a shift must explain at least 88% of the new board's informative cells
GAIN_RATIO = 3.0       # ... leave at most a third of the mismatch that standing still leaves
ERR0_MIN = 0.08        # ... and standing still must leave at least 8% unexplained
CUT_ERR = 0.25         # neither a shift nor standing still explains the new board: the picture was replaced
PER_FRAME_MAX = 12     # largest shift searched between two frames of one step's animation
DIRECT_MAX = 48        # largest shift searched between the boards before and after a step (one-frame steps)
MIN_CELLS = 40         # fewer informative cells than this: too little to tell a shift
RELOCATE_SHARE = 0.9   # a redrawn picture is placed on the map where at least this share of its cells agree
HEX = "0123456789abcdef"
STALE_BOXES = 6        # most stale areas named in a fixed-mode map's header


def informative(board):
    """cells that differ from a 4-neighbour: a uniform area matches any shift"""
    b = board.astype(np.int16)
    m = np.zeros(b.shape, bool)
    m[1:, :] |= b[1:, :] != b[:-1, :]
    m[:-1, :] |= b[:-1, :] != b[1:, :]
    m[:, 1:] |= b[:, 1:] != b[:, :-1]
    m[:, :-1] |= b[:, :-1] != b[:, 1:]
    return m


def _onehot_fft(board, mask):
    planes = np.zeros((16, PAD, PAD))
    for c in np.unique(board[mask]):
        planes[c, :N, :N] = (board == c) & mask
    return np.fft.rfft2(planes)


def match_table(a, b, mb):
    """hits[d]: cells p of mb with a[p - d] == b[p]; cnt[d]: cells p of mb with p - d on the board"""
    fa = _onehot_fft(a, np.ones((N, N), bool))
    fb = _onehot_fft(b, mb)
    hits = np.fft.irfft2((fb * np.conj(fa)).sum(0), s=(PAD, PAD))
    ma = np.zeros((PAD, PAD))
    ma[:N, :N] = 1
    mbb = np.zeros((PAD, PAD))
    mbb[:N, :N] = mb
    cnt = np.fft.irfft2(np.fft.rfft2(mbb) * np.conj(np.fft.rfft2(ma)), s=(PAD, PAD))
    return np.rint(hits).astype(int), np.rint(cnt).astype(int)


def best_shift(a, b, max_shift, min_overlap_share=0.25):
    """(shift, err, err0): err is the mismatch share at the best shift, err0 at no shift; shift None when b has too
    few informative cells. Ties go to the smaller shift."""
    mb = informative(b)
    n = int(mb.sum())
    if n < MIN_CELLS:
        return None, 1.0, 1.0
    hits, cnt = match_table(a, b, mb)
    err0 = 1 - hits[0, 0] / max(cnt[0, 0], 1)
    r = np.arange(-max_shift, max_shift + 1)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    h, c = hits[dy % PAD, dx % PAD], cnt[dy % PAD, dx % PAD]
    ok = (c >= max(MIN_CELLS, min_overlap_share * n)) & ~((dx == 0) & (dy == 0))
    if not ok.any():
        return None, 1.0, err0
    err = np.where(ok, 1 - h / np.maximum(c, 1), 9.0)
    i = np.unravel_index(np.argmin(np.round(err, 4) * 1000 + np.abs(dx) + np.abs(dy)), err.shape)
    return (int(dx[i]), int(dy[i])), float(err[i]), float(err0)


def classify(a, b, max_shift):
    """'same' | 'shift' | 'still' | 'cut' | 'unclear' | 'sparse', and the shift"""
    if (a == b).all():
        return "same", (0, 0)
    d, e, e0 = best_shift(a, b, max_shift)
    if d is None:
        return "sparse", (0, 0)
    if e <= ERR_MAX and e0 >= max(ERR0_MIN, GAIN_RATIO * e):
        return "shift", d
    if e0 <= ERR_MAX * 2:
        return "still", (0, 0)
    if min(e, e0) >= CUT_ERR:
        return "cut", (0, 0)
    return "unclear", (0, 0)


def estimate(before, frames):
    """{'kind': 'none'|'shift'|'cut', 'shift': (dx, dy), 'pairs': [(a, b, d), ...]} for one step. An animation is
    followed frame by frame (scrolls move 1-3 cells a frame: small, unambiguous on repeated textures) and the shifts
    summed; a one-frame step is compared directly, and only when at least 1/5 of the cells changed."""
    seq = [before] + list(frames)
    kinds, pairs, total = [], [], [0, 0]
    if len(frames) >= 2:
        for a, b in zip(seq, seq[1:]):
            k, d = classify(a, b, PER_FRAME_MAX)
            kinds.append(k)
            if k == "cut":
                break
            if k == "shift":
                total[0] += d[0]
                total[1] += d[1]
                pairs.append((a, b, d))
        if "cut" not in kinds and total != [0, 0]:
            return {"kind": "shift", "shift": tuple(total), "pairs": pairs}
    k, d = classify(seq[0], seq[-1], DIRECT_MAX)
    if k == "shift" and int((seq[0] != seq[-1]).sum()) * 5 < seq[0].size:
        k = "still"
    if k == "shift":
        return {"kind": "shift", "shift": d, "pairs": [(seq[0], seq[-1], d)]}
    if k == "cut" or "cut" in kinds:
        return {"kind": "cut", "shift": None, "pairs": []}
    return {"kind": "none", "shift": (0, 0), "pairs": []}


class LevelMap:
    """The window's top-left in level coordinates (the level's first view is at (0, 0); it moves opposite to the
    picture), every cell seen (last colour), and the HUD rows / columns, learned as lines that stay put while the
    picture slides; they are not painted."""

    fixed = False              # the fixed mode also drops cells painted from a line before it was known to be HUD

    def __init__(self, board, step, fixed=False):
        self.fixed = fixed
        self.pos = (0, 0)
        self.cells = {}
        self.first = {}        # the colour a cell had when it was first seen
        self.seen = {}         # the last step a cell was in view
        self.src = {}          # the board (x, y) a cell was last painted from: dropped once that row / column is HUD
        self.votes = {}
        self.lost = False
        self.scrolled = False
        self.paint(board, step)

    def hud(self, axis):
        return {i for (ax, i), (pinned, world) in self.votes.items() if ax == axis and pinned >= 2 and world == 0}

    def learn_hud(self, a, b, d):
        inf = informative(b)
        dx, dy = d
        for axis in (0, 1):
            for i in range(N):
                line = inf[i] if axis == 0 else inf[:, i]
                if line.sum() < 4:
                    continue
                ys, xs = (np.full(N, i), np.arange(N)) if axis == 0 else (np.arange(N), np.full(N, i))
                sy, sx = ys - dy, xs - dx
                ok = (sy >= 0) & (sy < N) & (sx >= 0) & (sx < N) & line
                pinned = bool((a[ys, xs] == b[ys, xs])[line].all())
                agree = float((a[sy[ok], sx[ok]] == b[ys[ok], xs[ok]]).mean()) if ok.sum() >= 4 else None
                votes = self.votes.setdefault((axis, i), [0, 0])
                if pinned and agree is not None and agree < 0.8:
                    votes[0] += 1
                elif agree is not None and agree >= 0.95 and not pinned:
                    votes[1] += 1

    def paint(self, board, step):
        rows, cols = self.hud(0), self.hud(1)
        x0, y0 = self.pos
        for y in range(N):
            if y in rows:
                continue
            for x in range(N):
                if x not in cols:
                    key = (x0 + x, y0 + y)
                    self.cells[key] = int(board[y, x])
                    self.first.setdefault(key, int(board[y, x]))
                    self.seen[key] = step
                    self.src[key] = (x, y)

    def restart(self, board, step):
        # RESET: the same level from its first view; the map is the level's layout, so it is kept. A RESET restores
        # the level's first state exactly (196 of 196 in v34), so the fixed mode also puts every cell back to the
        # colour it was first seen in: a box or an enemy last seen elsewhere is back where it started
        self.pos = (0, 0)
        self.lost = False
        if self.fixed:
            self.cells = dict(self.first)
        self.paint(board, step)

    def update(self, est, board, step):
        if est["kind"] == "shift":
            for a, b, d in est["pairs"]:
                self.learn_hud(a, b, d)
            rows, cols = self.hud(0), self.hud(1)
            for key in [k for k, (x, y) in self.src.items() if self.fixed and (y in rows or x in cols)]:
                # painted before that line was known to be HUD
                for d in (self.cells, self.first, self.seen, self.src):
                    del d[key]
            dx, dy = est["shift"]
            self.pos = (self.pos[0] - dx, self.pos[1] - dy)
            self.scrolled = True
        elif self.scrolled and (est["kind"] == "cut" or self.lost):
            # a level that never slid keeps its window at (0, 0): a redraw there is a flash or a new screen
            self.lost = not self.relocate(board)
        if not self.lost:
            self.paint(board, step)

    def relocate(self, board):
        """put a redrawn picture where it agrees best with the cells seen so far; False if nowhere reaches
        RELOCATE_SHARE"""
        xs = [k[0] for k in self.cells]
        ys = [k[1] for k in self.cells]
        x0, y0 = min(xs), min(ys)
        pw, ph = max(xs) - x0 + 1 + 2 * N, max(ys) - y0 + 1 + 2 * N
        known = np.zeros((ph, pw), bool)
        colour = np.zeros((ph, pw), np.int16)
        for (x, y), v in self.cells.items():
            known[y - y0 + N, x - x0 + N] = True
            colour[y - y0 + N, x - x0 + N] = v
        inf = informative(board)
        for y in self.hud(0):
            inf[y, :] = False
        for x in self.hud(1):
            inf[:, x] = False
        fft = np.fft.rfft2
        hits = np.zeros((ph, pw))
        for c in np.unique(board[inf]):
            m = np.zeros((ph, pw))
            m[:N, :N] = (board == c) & inf
            hits += np.fft.irfft2(fft((colour == c) & known) * np.conj(fft(m)), s=(ph, pw))
        m = np.zeros((ph, pw))
        m[:N, :N] = inf
        total = np.rint(np.fft.irfft2(fft(known) * np.conj(fft(m)), s=(ph, pw)))
        share = np.where(total >= 100, np.rint(hits) / np.maximum(total, 1), 0)
        oy, ox = np.unravel_index(np.argmax(share), share.shape)
        if share[oy, ox] < RELOCATE_SHARE:
            return False
        self.pos = (int(ox) - N + x0, int(oy) - N + y0)
        return True

    def bounds(self):
        xs = [k[0] for k in self.cells]
        ys = [k[1] for k in self.cells]
        return min(xs), min(ys), max(xs), max(ys)

    def text(self, level):
        x0, y0, x1, y1 = self.bounds()
        head = ("# level %d as seen so far: %d x %d cells, '.' = never seen, HUD rows/columns left out. Map column = "
                "board x %+d, map line = board y %+d for the board you see now (the lines below are map lines 0, 1, "
                "...)." % (level, x1 - x0 + 1, y1 - y0 + 1, self.pos[0] - x0, self.pos[1] - y0))
        lines = [head]
        for y in range(y0, y1 + 1):
            lines.append("".join(HEX[self.cells[(x, y)]] if (x, y) in self.cells else "." for x in range(x0, x1 + 1)))
        return "\n".join(lines) + "\n"

    def stale(self, key):
        """out of view now, and last seen in a colour it did not have at first: a thing that moved off there, or
        something that changed -- it may be different by now"""
        x, y = key
        inside = not self.lost and 0 <= x - self.pos[0] < N and 0 <= y - self.pos[1] < N
        return not inside and self.cells[key] != self.first[key]

    def world_text(self, level):
        x0, y0, x1, y1 = self.bounds()
        now = ("the board's place in it is unknown since the picture was redrawn" if self.lost else
               "the board you see now is world x %d..%d, y %d..%d: world = board + (%d, %d)"
               % (self.pos[0], self.pos[0] + N - 1, self.pos[1], self.pos[1] + N - 1, self.pos[0], self.pos[1]))
        head = ("# level %d in world coordinates, which never move: world (0, 0) is the top-left of the level's first "
                "view, x grows to the right, y down. This map covers world x %d..%d, y %d..%d; %s. Each cell is the "
                "colour last seen there, '.' = never seen, HUD rows/columns left out. Python: sys.path.insert(0, "
                "'lib'); from arc import load_map; m = load_map(%d); m.at(x, y)" % (level, x0, x1, y0, y1, now, level))
        xs = range(x0, x1 + 1)
        lines = [head]
        boxes = self.stale_boxes()
        if boxes:
            lines.append("# out of view now and changed while you watched (a thing that moved off, or a change: it may "
                         "be different now): " + ", ".join("(%d,%d)-(%d,%d) %d cells" % b for b in boxes[:STALE_BOXES])
                         + (" and %d more area(s)" % (len(boxes) - STALE_BOXES) if len(boxes) > STALE_BOXES else ""))
        if x0 < 0:
            lines.append("      " + "".join("-" if x < 0 else " " for x in xs))
        lines += ["      " + "".join(str(abs(x) // 10 % 10) for x in xs), "      " + "".join(str(abs(x) % 10) for x in xs)]
        for y in range(y0, y1 + 1):
            row = "".join("." if (x, y) not in self.cells else HEX[self.cells[(x, y)]] for x in xs)
            lines.append("%5d %s" % (y, row))
        return "\n".join(lines) + "\n"

    def stale_boxes(self):
        """the 4-connected areas of stale cells, largest first: (x0, y0, x1, y1, cells) in world coordinates"""
        todo = {k for k in self.cells if self.stale(k)}
        boxes = []
        while todo:
            seed = todo.pop()
            group, stack = [seed], [seed]
            while stack:
                x, y = stack.pop()
                for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if n in todo:
                        todo.discard(n)
                        group.append(n)
                        stack.append(n)
            gx, gy = [g[0] for g in group], [g[1] for g in group]
            boxes.append((min(gx), min(gy), max(gx), max(gy), len(group)))
        return sorted(boxes, key=lambda b: (-b[4], b[1], b[0]))

    def arrays(self):
        """grid (-1 = never seen), first colour, last step seen (-1 = never) and stale, as world-anchored arrays
        whose [0, 0] is world (x0, y0)"""
        x0, y0, x1, y1 = self.bounds()
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        grid, first, seen = np.full(shape, -1, np.int16), np.full(shape, -1, np.int16), np.full(shape, -1, np.int32)
        stale = np.zeros(shape, bool)
        for (x, y), v in self.cells.items():
            grid[y - y0, x - x0], first[y - y0, x - x0], seen[y - y0, x - x0] = v, self.first[(x, y)], self.seen[(x, y)]
            stale[y - y0, x - x0] = self.stale((x, y))
        return {"grid": grid, "first": first, "seen": seen, "stale": stale, "x0": x0, "y0": y0,
                "pos": np.array(self.pos), "lost": self.lost,
                "hud_rows": np.array(sorted(self.hud(0)), np.int32), "hud_cols": np.array(sorted(self.hud(1)), np.int32)}

    def state(self):
        keys = np.array(sorted(self.cells), dtype=np.int32).reshape(-1, 2)
        return {"pos": np.array(self.pos), "keys": keys,
                "values": np.array([self.cells[tuple(k)] for k in keys.tolist()], dtype=np.uint8),
                "first": np.array([self.first[tuple(k)] for k in keys.tolist()], dtype=np.uint8),
                "seen": np.array([self.seen[tuple(k)] for k in keys.tolist()], dtype=np.int64),
                "src": np.array([self.src[tuple(k)] for k in keys.tolist()], dtype=np.int32).reshape(-1, 2),
                "votes": np.array([[ax, i, p, w] for (ax, i), (p, w) in sorted(self.votes.items())],
                                  dtype=np.int32).reshape(-1, 4),
                "flags": np.array([self.lost, self.scrolled])}

    @classmethod
    def load(cls, data):
        self = cls.__new__(cls)
        self.pos = tuple(int(v) for v in data["pos"])
        self.cells = {(int(x), int(y)): int(v) for (x, y), v in zip(data["keys"], data["values"])}
        self.first = {(int(x), int(y)): int(v) for (x, y), v in zip(data["keys"], data["first"])}
        self.seen = {(int(x), int(y)): int(v) for (x, y), v in zip(data["keys"], data["seen"])}
        self.src = {(int(x), int(y)): (int(sx), int(sy)) for (x, y), (sx, sy) in zip(data["keys"], data["src"])}
        self.votes = {(int(ax), int(i)): [int(p), int(w)] for ax, i, p, w in data["votes"]}
        self.lost, self.scrolled = (bool(v) for v in data["flags"])
        return self


def _ways(dx, dy):
    picture, window = [], []
    if dy:
        picture.append("%s %d" % ("down" if dy > 0 else "up", abs(dy)))
        window.append("%s %d" % ("up" if dy > 0 else "down", abs(dy)))
    if dx:
        picture.append("%s %d" % ("right" if dx > 0 else "left", abs(dx)))
        window.append("%s %d" % ("left" if dx > 0 else "right", abs(dx)))
    return " and ".join(picture), " and ".join(window)


def _span(values):
    return "%d" % values[0] if len(values) == 1 else "%d-%d" % (values[0], values[-1])


class Tracker:
    def __init__(self, ws, mode="screen"):
        if mode not in ("screen", "fixed"):
            raise ValueError("view tracker mode %r: screen or fixed" % mode)
        self.fixed = mode == "fixed"
        self.record = ws / "record"
        self.maps = ws / "map"
        self.state_path = ws / ".act_view" / "state.npz"
        self.camera_path = ws / ".act_view" / "camera.jsonl"
        self.camera = []       # (step, level, x, y) rows not yet written; x, y None while the place is unknown

    def frames(self, step):
        with np.load(self.record / "steps" / ("%06d" % step) / "frames.npz") as data:
            frames = data["frames"]
        if frames.ndim != 3 or frames.shape[1:] != (N, N):
            raise ValueError("step %d: frames of shape %s, not (n, %d, %d)" % (step, frames.shape, N, N))
        return frames

    def open(self):
        rows = [json.loads(line) for line in (self.record / "history.jsonl").read_text().splitlines() if line.strip()]
        self.level, start, done = 1, 0, 0
        for row in rows[1:]:
            if row["levels_completed"] > done:
                done, start, self.level = row["levels_completed"], row["step"], row["levels_completed"] + 1
        later = [(row["step"], row["action"]) for row in rows if row["step"] > start]
        resumed = None
        if self.state_path.is_file():
            with np.load(self.state_path) as data:
                saved = {k: data[k] for k in data.files}
            if int(saved["start"]) == start and int(saved["last"]) <= (later[-1][0] if later else start) \
                    and (not self.fixed or self._camera_last() >= int(saved["last"])):
                resumed = saved
        if resumed is None:
            self.start, self.last = start, start
            self.prev = self.frames(start)[-1]
            self.map = LevelMap(self.prev, start, self.fixed)
            self._log(start)
        else:
            self.start, self.last = int(resumed["start"]), int(resumed["last"])
            self.prev = resumed["prev"]
            self.map = LevelMap.load(resumed)
            self.map.fixed = self.fixed
        for step, action in later:
            if step > self.last:
                self.step(step, action, False)

    def _camera_last(self):
        # the last step .act_view/camera.jsonl has (-1: none); save() writes it before the state, so a state ahead
        # of it means a save that was cut off: the level is replayed from the record
        if not self.camera_path.is_file():
            return -1
        lines = [line for line in self.camera_path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1])["step"] if lines else -1

    def _log(self, step):
        m = self.map
        self.camera.append((step, self.level, None if m.lost else m.pos[0], None if m.lost else m.pos[1]))

    def _world(self):
        # an addition with a worked example: "board (x, y) = world (x+0, y-18)" was read as an equation and its sign
        # flipped in about 30% of the uses in round 4
        m = self.map
        if m.lost:
            return "the window's place in the level is unknown since the picture was redrawn"
        cx, cy = m.pos
        return "world = board + (%d, %d) now (board (10, 5) is world (%d, %d))" % (cx, cy, 10 + cx, 5 + cy)

    def step(self, step, action, cleared):
        """(lines to print for this step, whether the window moved or became unknown, the slide (dx, dy) or None);
        the screen mode answers the first two"""
        lines, moved, slide = self._step(step, action, cleared)
        if self.fixed:
            self._log(step)
            return lines, moved, slide
        return lines, moved

    def _step(self, step, action, cleared):
        if step != self.last + 1:
            raise ValueError("view tracker: step %d follows step %d" % (step, self.last))
        frames = self.frames(step)
        self.last = step
        m = self.map
        if cleared:
            lines = []
            if m.scrolled:
                self.write()
                lines.append("map: level %d's map stays in map/L%02d.txt; level %d gets its own when its view first "
                             "slides" % (self.level, self.level, self.level + 1))
            self.level += 1
            self.start, self.prev = step, frames[-1]
            self.map = LevelMap(self.prev, step, self.fixed)
            return lines, False, None
        if action == "RESET":
            m.restart(frames[-1], step)
            self.prev = frames[-1]
            if not m.scrolled:
                return [], False, None
            if self.fixed:
                return (["view: RESET put the window back at the level's first view: world = board again; "
                         "map/L%02d.txt keeps what you saw" % self.level], False, None)
            return (["view: RESET put the window back at the level's first view; map/L%02d.txt keeps what you saw"
                     % self.level], False, None)
        was_lost, was_pos = m.lost, m.pos
        est = estimate(self.prev, frames)
        m.update(est, frames[-1], step)
        self.prev = frames[-1]
        if est["kind"] == "shift":
            dx, dy = est["shift"]
            picture, window = _ways(dx, dy)
            rows = [y for y in range(N) if not 0 <= y - dy < N and y not in m.hud(0)]
            cols = [x for x in range(N) if not 0 <= x - dx < N and x not in m.hud(1)]
            new = " and ".join(part for part in (("rows %s" % _span(rows)) if rows else "",
                                                 ("columns %s" % _span(cols)) if cols else "") if part)
            x0, y0, x1, y1 = m.bounds()
            if self.fixed:
                # no "the things did not move": in bp35 and lf52 the camera follows the player, which did move
                return ["view: the picture slid %s (the window moved %s): what stayed put in the level is now %+d, %+d "
                        "cells away on the board%s. World coordinates never move: %s; map/L%02d.txt covers world x "
                        "%d..%d, y %d..%d"
                        % (picture, window, dx, dy, "; %s came into view" % new if new else "", self._world(),
                           self.level, x0, x1, y0, y1)], True, (dx, dy)
            where = ("the board's place on it is unknown since the picture was redrawn" if m.lost else
                     "the board's top-left is at its column %d, line %d" % (m.pos[0] - x0, m.pos[1] - y0))
            return ["view: the picture slid %s (the window moved %s) -- what was at (x, y) is now at (x%+d, y%+d)%s. "
                    "The things did not move by themselves; what left the board still exists. Map so far: "
                    "map/L%02d.txt (%d x %d cells), %s"
                    % (picture, window, dx, dy, "; %s came into view" % new if new else "", self.level,
                       x1 - x0 + 1, y1 - y0 + 1, where)], True, (dx, dy)
        if m.scrolled and est["kind"] == "cut":
            if m.lost:
                return ["view: the picture was redrawn, not slid, and matches nothing seen so far: the window "
                        "position is unknown until it does"], True, None
            if self.fixed:
                return ["view: the picture was redrawn, not slid; it matches map/L%02d.txt: %s"
                        % (self.level, self._world())], m.pos != was_pos or was_lost, None
            x0, y0, _, _ = m.bounds()
            return ["view: the picture was redrawn, not slid; it matches map/L%02d.txt with the board's top-left at "
                    "column %d, line %d" % (self.level, m.pos[0] - x0, m.pos[1] - y0)], m.pos != was_pos or was_lost, \
                None
        return [], False, None

    def write(self):
        if self.map.scrolled:
            self.maps.mkdir(exist_ok=True)
            path = self.maps / ("L%02d.txt" % self.level)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(self.map.world_text(self.level) if self.fixed else self.map.text(self.level))
            os.replace(tmp, path)
            if self.fixed:
                tmp = self.maps / ("L%02d.tmp.npz" % self.level)
                np.savez(tmp, level=self.level, **self.map.arrays())
                os.replace(tmp, self.maps / ("L%02d.npz" % self.level))

    def save(self):
        self.write()
        self.state_path.parent.mkdir(exist_ok=True)
        if self.fixed and self.camera:
            # before the state: a replay after a cut-off save only repeats rows (the last one of a step wins)
            with self.camera_path.open("a") as f:
                for step, level, x, y in self.camera:
                    f.write(json.dumps({"step": step, "level": level, "x": x, "y": y}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.camera = []
        tmp = self.state_path.with_name("state.tmp.npz")
        np.savez(tmp, start=self.start, last=self.last, prev=self.prev, **self.map.state())
        os.replace(tmp, self.state_path)


def main(ws, mode="screen"):
    tracker = Tracker(Path(ws), mode)
    tracker.open()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        ask = json.loads(line)
        if ask.get("end"):
            break
        answer = tracker.step(ask["step"], ask["action"], ask["cleared"])
        if tracker.fixed:
            lines, moved, slide = answer
            m = tracker.map
            print(json.dumps({"lines": lines, "moved": moved, "shift": slide,
                              "pos": None if m.lost else list(m.pos)}), flush=True)
        else:
            lines, moved = answer
            print(json.dumps({"lines": lines, "moved": moved}), flush=True)
    tracker.save()
    print(json.dumps({"saved": True}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
