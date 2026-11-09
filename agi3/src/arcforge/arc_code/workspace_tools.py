"""``workspace_tools`` (RSI batch 1, item C, prompts v14rc): ``lib/arc.py`` and ``./bg``, placed in the workspace before the game opens.

Blind-spot report (docs/research/codex_arc3_blindspots_20261018.md): 46–60% of Qwen's commands are ad-hoc python,
5.4–5.8% of them end in a Traceback; x/y mix-ups, merging of same-colour objects, ``str.replace`` silently doing nothing
when it does not match; ``pkill -f`` 193 times, exit 143 67 times. So provide a tested tool for reading boards / objects /
diffs / replays, and a ``./bg`` that manages background jobs with pidfiles.

The board reading, objects and diff in ``lib/arc.py`` are read-only functions **extracted verbatim** from ``arcgame.py``
(duckx c0L's library, tested in tests/test_arc_code_arcgame.py), so both sides run the same object and diff code;
arcgame's acting, images and HTTP are not brought in -- this arm still acts only through ``./act``.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Do not import arcgame: it installs atexit / signal handlers at import time (needed by the acting process in the jail); only read its source.
ARCGAME = Path(__file__).with_name("arcgame.py")
#: Definitions taken verbatim from arcgame.py (read-only, they only look at record/).
FROM_ARCGAME = ("_grid", "_parse_board", "frames", "history", "text", "_as_grid", "Obj", "_component",
                "_components", "objects", "scene", "_edge_cell", "Diff")

ARC_HEADER = '''"""arc: read this game's record/ from Python. Standard library + numpy. It sends no actions.

    import sys; sys.path.insert(0, "lib"); from arc import *     (from the workspace)

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
    replay(sim, first, last)
                       runs sim(board, action, x, y) -> board over recorded steps first..last of one level
                       and returns the first step it gets wrong (None when all match), printing it
    replace_once(path, old, new)
                       edit a file: fails loudly unless old occurs exactly once

x is the column and y the row, both 0-63, (0,0) at the top left.
"""

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["board", "at", "frames", "history", "objects", "track", "scene", "diff", "text", "replay",
           "replace_once", "Obj", "Diff", "ActError"]

#: the workspace is the directory that holds lib/ (this file is lib/arc.py)
WORKSPACE = Path(__file__).resolve().parent.parent
RECORD = WORKSPACE / "record"
'''

ARC_TAIL = '''

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
            if j not in used_new and n.color == o.color and n.x0 <= o.x1 and o.x0 <= n.x1 \\
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
    for step in range(first, last + 1):
        row = rows.get(step)
        if row is None:
            raise ActError("step %d is not in record/history.jsonl" % step)
        if row["level"] != level:
            print("replay: steps %d-%d match (step %d is on another level)" % (first, step - 1, step))
            return None
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
            print("replay: steps %d-%d match (step %d cleared the level)" % (first, step, step))
            return None
    print("replay: steps %d-%d all match" % (first, last))
    return None


def replace_once(path, old, new):
    """Replace ``old`` with ``new`` in ``path``; raise unless ``old`` occurs exactly once."""
    path = Path(path)
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise ActError("%s: the text to replace occurs %d times, not once" % (path, n))
    path.write_text(text.replace(old, new, 1))
'''

BG_SOURCE = '''#!/usr/bin/env python3
"""bg: background jobs in this workspace, tracked by a pid file -- no pkill, no nohup.

    ./bg run NAME CMD [ARG...]   start CMD in the background; its output goes to logs/NAME.log
    ./bg status                  every job: running (for how long) or exited (with its exit code)
    ./bg log NAME [N]            the last N (default 40) lines of logs/NAME.log
    ./bg kill NAME               stop the job and everything it started

A job is one process group of its own; kill stops only that group, after checking that the pid in
.bg/NAME.json is still the process that was started.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent      # this file is lib/bg.py
STATE = WS / ".bg"
LOGS = WS / "logs"


def _start_time(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            raw = handle.read()
    except OSError:
        return None
    return int(raw[raw.rindex(")") + 2:].split()[19])


def _job(name):
    path = STATE / (name + ".json")
    if not path.is_file():
        sys.exit("bg: no job named %r (./bg status lists them)" % name)
    return json.loads(path.read_text())


def _alive(job):
    return job.get("exit") is None and job["start"] is not None and _start_time(job["pid"]) == job["start"]


def _group_alive(pgid):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue                          # the process ended while we looked
        if int(raw[raw.rindex(")") + 2:].split()[2]) == pgid:
            return True
    return False


def _supervise(name, cmd):
    # the job's own session: this process and everything the command starts share one process group
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / (name + ".log"), "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=WS)
        code = proc.wait()
    path = STATE / (name + ".json")
    job = json.loads(path.read_text())
    job["exit"], job["ended"] = code, time.time()
    path.write_text(json.dumps(job))


def cmd_run(name, cmd):
    if not name.replace("_", "").replace("-", "").isalnum():
        sys.exit("bg: a job name is letters, digits, - and _ (got %r)" % name)
    if not cmd:
        sys.exit("bg: run NAME CMD [ARG...]")
    STATE.mkdir(exist_ok=True)
    path = STATE / (name + ".json")
    if path.is_file() and _alive(json.loads(path.read_text())):
        sys.exit("bg: %s is still running; ./bg kill %s first or use another name" % (name, name))
    proc = subprocess.Popen([sys.executable, __file__, "_supervise", name, *cmd], cwd=WS,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    for _ in range(50):
        start = _start_time(proc.pid)
        if start is not None:
            break
        time.sleep(0.01)
    path.write_text(json.dumps({"name": name, "cmd": cmd, "pid": proc.pid, "start": start,
                                "started": time.time(), "exit": None}))
    print("started %s (pid %d); output in logs/%s.log" % (name, proc.pid, name))


def cmd_status():
    jobs = sorted(STATE.glob("*.json")) if STATE.is_dir() else []
    if not jobs:
        print("no background jobs")
    for path in jobs:
        job = json.loads(path.read_text())
        now = time.time()
        if job.get("exit") is not None:
            state = "exited %d after %d s" % (job["exit"], job["ended"] - job["started"])
        elif _alive(job):
            state = "running for %d s" % (now - job["started"])
        else:
            state = "gone (killed, or ended without reporting)"
        print("%-16s %s -- %s" % (job["name"], state, " ".join(job["cmd"])[:80]))


def cmd_log(name, n):
    _job(name)
    log = LOGS / (name + ".log")
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    print("\\n".join(lines[-n:]) if lines else "(logs/%s.log is empty)" % name)


def cmd_kill(name):
    job = _job(name)
    if not _alive(job):
        print("%s is not running" % name)
        return
    os.killpg(job["pid"], signal.SIGTERM)
    for _ in range(50):
        if not _group_alive(job["pid"]):
            break
        time.sleep(0.1)
    else:
        os.killpg(job["pid"], signal.SIGKILL)
    job["exit"], job["ended"] = -signal.SIGTERM, time.time()
    (STATE / (name + ".json")).write_text(json.dumps(job))
    print("stopped %s" % name)


def main(argv):
    if argv[:1] == ["_supervise"]:
        return _supervise(argv[1], argv[2:])
    if argv[:1] == ["run"] and len(argv) >= 2:
        return cmd_run(argv[1], argv[2:])
    if argv == ["status"]:
        return cmd_status()
    if argv[:1] == ["log"] and len(argv) in (2, 3):
        return cmd_log(argv[1], int(argv[2]) if len(argv) == 3 else 40)
    if argv[:1] == ["kill"] and len(argv) == 2:
        return cmd_kill(argv[1])
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
'''

BG_SHIM = '#!/bin/sh\nexec python3 "$(dirname "$0")/lib/bg.py" "$@"\n'


def arc_source() -> str:
    """lib/arc.py: header + definitions extracted verbatim from arcgame.py + record-reading board / at / track / diff / replay / replace_once."""
    src = ARCGAME.read_text(encoding="utf-8")
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    missing = [name for name in FROM_ARCGAME if name not in defs]
    if missing:
        raise ValueError(f"arcgame.py no longer defines {missing}; update workspace_tools.FROM_ARCGAME")
    values = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("EDGE", "LARGE", "SCREEN", "MAX_LINES")}
    if len(values) != 4:
        raise ValueError(f"arcgame.py constants changed: found {sorted(values)}")
    consts = "".join(f"{k} = {values[k]!r}\n" for k in ("EDGE", "LARGE", "SCREEN", "MAX_LINES"))
    body = "\n\n".join(ast.get_source_segment(src, defs[name]) for name in FROM_ARCGAME)
    error = '''

class ActError(Exception):
    """A request lib/arc.py cannot answer (no such step, a board off the record, a bad argument)."""
'''
    return ARC_HEADER + consts + error + "\n\n" + body + "\n" + ARC_TAIL


def sources() -> dict[str, str]:
    """Workspace-relative path -> content (also part of ARM_STATE's prompt fingerprint: the briefing tells the agent to use them)."""
    return {"lib/arc.py": arc_source(), "lib/bg.py": BG_SOURCE, "bg": BG_SHIM}


def install(ws: Path) -> None:
    for rel, text in sources().items():
        path = ws / rel
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755 if rel == "bg" else 0o644)
