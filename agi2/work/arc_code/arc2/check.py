"""Run solution.py's transform on this puzzle's grids.

    python3 check.py            training pairs: PASS, or which cells are wrong; test inputs: output size
    python3 check.py FILE.py    the same for another file's transform (no snapshot)
    (harness) check.py --harvest [solution file]  task JSON on stdin -> one line "ARC2_HARVEST <json>" on stdout

Every time check.py passes at least as many training pairs as the best run so far, it copies solution.py to
.best/solution.py (with .best/meta.json). If solution.py later gets broken, the harness falls back to that copy.

The harness keeps its own copy of this file and runs it with --harvest when the session ends;
the copy in the workspace is for the agent.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import time
import signal
import sys
import traceback
from pathlib import Path

CALL_SECONDS = 30
MARK = "ARC2_HARVEST "


class CallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CallTimeout(f"transform took longer than {CALL_SECONDS} s")


def valid_grid(value) -> str | None:
    """None if value is a rectangular list of rows of ints 0-9 with sides 1..30, else why not."""
    if not isinstance(value, list) or not value:
        return "not a non-empty list of rows"
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            return "a row is not a non-empty list"
        if width is None:
            width = len(row)
        elif len(row) != width:
            return "rows have different lengths"
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 9:
                return f"cell {v!r} is not an int 0-9"
    if not 1 <= len(value) <= 30 or not 1 <= width <= 30:
        return f"size {len(value)}x{width} is outside 1..30"
    return None


def as_grid(value):
    """Lists, tuples and numpy arrays of ints become a list of lists of int."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        value = [list(r.tolist() if hasattr(r, "tolist") else r) for r in value]
        value = [[int(v) if hasattr(v, "__index__") and not isinstance(v, bool) else v for v in r]
                 for r in value]
    return value


def load_transform(path: Path):
    # solution.py may import arc.py and the agent's own helper modules: from its own folder first, then from the
    # workspace (the cwd) -- the .best/solution.py snapshot lives one level down, and harvest runs python -I
    for d in dict.fromkeys([Path.cwd().resolve(), path.resolve().parent]):
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "transform", None)
    if not callable(fn):
        raise AttributeError("solution.py does not define transform(grid)")
    return fn


def call(fn, grid):
    """(grid, None) or (None, error text). The agent's prints go to stderr."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CALL_SECONDS)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            out = as_grid(fn([list(r) for r in grid]))
    except MemoryError as error:
        # The agent's frames still hold whatever filled the process, so formatting the traceback can raise
        # MemoryError again and kill the harvest (2026-10-11, RL group g001298: rc 1 stopped the whole run).
        # Drop the frames first; the pair scores as an error like any other exception.
        error.__traceback__ = None
        return None, "MemoryError: the transform ran out of memory"
    except BaseException as error:  # noqa: BLE001 -- the agent's code may raise anything; it is reported
        tb = traceback.extract_tb(error.__traceback__)
        where = next((f"solution.py line {f.lineno}" for f in reversed(tb) if f.filename.endswith("solution.py")), "")
        return None, f"{type(error).__name__}: {error}" + (f" ({where})" if where else "")
    finally:
        signal.alarm(0)
    problem = valid_grid(out)
    return (out, None) if problem is None else (None, f"returned an invalid grid: {problem}")


def compare(got, want) -> str:
    if len(got) != len(want) or len(got[0]) != len(want[0]):
        return f"FAIL: size {len(got)}x{len(got[0])}, expected {len(want)}x{len(want[0])}"
    wrong = [(r, c) for r in range(len(want)) for c in range(len(want[0])) if got[r][c] != want[r][c]]
    if not wrong:
        return "PASS"
    shown = ", ".join(f"({r},{c}) got {got[r][c]} want {want[r][c]}" for r, c in wrong[:8])
    more = f" (+{len(wrong) - 8} more)" if len(wrong) > 8 else ""
    return f"FAIL: {len(wrong)} of {len(want) * len(want[0])} cells wrong [row,col from 0]: {shown}{more}"


def report(workspace: Path, solution: Path | None = None) -> None:
    task = json.loads((workspace / "task" / "task.json").read_text())
    own = solution is None
    path = workspace / "solution.py" if own else solution
    try:
        fn = load_transform(path)
    except BaseException as error:  # noqa: BLE001 -- reported to the agent
        print(f"cannot load {path.name}: {type(error).__name__}: {error}")
        sys.exit(1)
    passed = 0
    for k, pair in enumerate(task["train"], 1):
        got, error = call(fn, pair["input"])
        verdict = f"ERROR: {error}" if error else compare(got, pair["output"])
        passed += verdict == "PASS"
        print(f"train {k}: {verdict}")
    print(f"{passed}/{len(task['train'])} training pairs pass")
    if own:
        snapshot(workspace, passed, len(task["train"]))
    for j, test in enumerate(task["test"], 1):
        override = workspace / "answer" / f"test_{j}.json"
        got, error = call(fn, test["input"])
        line = f"test {j}: ERROR: {error}" if error else f"test {j}: output {len(got)}x{len(got[0])}"
        # run_arc2_seq：第二個 session 的工作區有前一個 session 的輸出，標出這個輸出跟它一不一樣
        previous = workspace / "previous" / f"test_{j}_output.json"
        if got is not None and previous.exists():
            same = json.loads(previous.read_text()) == got
            line += f"  ({'SAME AS' if same else 'differs from'} previous/test_{j}_output.json)"
        if own and override.exists():
            line += f"  (answer/test_{j}.json present: it replaces this output)"
        print(line)


def snapshot(workspace: Path, passed: int, total: int) -> None:
    if passed == 0 or not (workspace / "task" / ".snapshot").exists():   # 標記檔由 harness 依 config 放
        return
    best = workspace / ".best"
    meta = best / "meta.json"
    before = json.loads(meta.read_text())["passed"] if meta.exists() else 0
    if passed < before:
        return
    best.mkdir(exist_ok=True)
    shutil.copyfile(workspace / "solution.py", best / "solution.py")
    meta.write_text(json.dumps({"passed": passed, "total": total, "t": time.time()}))
    print(f"(saved a copy as .best/solution.py: {passed}/{total})")


def harvest(workspace: Path, solution: Path) -> None:
    # The agent's code runs in this process and can leave sys.stdout pointing elsewhere (2026-10-07: a MemoryError
    # inside redirect_stdout's exit left it on stderr and the harness lost the result line): the result line goes to
    # a copy of the real stdout taken before any of the agent's code runs.
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    task = json.loads(sys.stdin.read())
    result = {"train": [], "test": [], "load_error": None}
    try:
        fn = load_transform(solution)
    except BaseException as error:  # noqa: BLE001 -- recorded; the harness decides
        fn, result["load_error"] = None, f"{type(error).__name__}: {error}"
    for pair in task["train"]:
        got, error = call(fn, pair["input"]) if fn else (None, "no transform")
        result["train"].append({"verdict": f"ERROR: {error}" if error else compare(got, pair["output"])})
    for j, grid in enumerate(task["test"], 1):
        entry = {"transform": None, "transform_error": None, "answer_file": None, "answer_error": None}
        if fn:
            entry["transform"], entry["transform_error"] = call(fn, grid)
        override = workspace / "answer" / f"test_{j}.json"
        if override.exists():
            try:
                value = as_grid(json.loads(override.read_text()))
                problem = valid_grid(value)
                entry["answer_file" if problem is None else "answer_error"] = value if problem is None else problem
            except (ValueError, OSError) as error:
                entry["answer_error"] = f"{type(error).__name__}: {error}"
        result["test"].append(entry)
    out.write("\n" + MARK + json.dumps(result) + "\n")
    out.flush()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--harvest"] and len(sys.argv) <= 3:
        # Older, already-running harnesses launch this copied checker directly on the host.
        # Enter the same sandbox before importing any generated code; the trusted parent
        # sets this marker after entering bubblewrap or the task's POSIX identity,
        # and clears the host environment.
        if os.environ.get("ARC2_HARVEST_SANDBOX") != "1":
            helper = Path(__file__).resolve().with_name("harvest_sandbox.py")
            spec = importlib.util.spec_from_file_location("harvest_sandbox", helper)
            sandbox = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sandbox)
            command = sandbox.argv(Path.cwd(), helper.parent, Path(sys.prefix),
                                   [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]])
            os.execv(command[0], command)
        harvest(Path.cwd(), Path(sys.argv[2]) if len(sys.argv) == 3 else Path.cwd() / "solution.py")
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".py") and Path(sys.argv[1]).is_file():
        report(Path(__file__).resolve().parent, Path(sys.argv[1]).resolve())
    elif sys.argv[1:]:
        sys.exit(f"usage: python3 {sys.argv[0]} [FILE.py]")
    else:
        report(Path(__file__).resolve().parent)
