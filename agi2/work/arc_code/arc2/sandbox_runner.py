"""Image-only mode: run the agent's transform in a sandbox; the results leave only through this one harness channel.

    (harness) sandbox_runner.py < {"source": <solution.py text>, "inputs": [grid, ...]}

stdout is a single result JSON line: {"load_error": str|None, "outputs": [[grid|None, error|None], ...]}.
The transform's own prints and stderr are all discarded (fd 1/2 point to /dev/null; the result is written to an
fd dup'd before the run). Error messages keep only the exception type, the solution.py line number, and the
message with digits replaced by # -- the agent cannot get at any cell value.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import CALL_SECONDS, as_grid, valid_grid  # noqa: E402  (the harness's copy of check.py)


class CallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CallTimeout(f"transform took longer than {CALL_SECONDS} s")


def describe(error: BaseException) -> str:
    tb = traceback.extract_tb(error.__traceback__)
    line = next((f.lineno for f in reversed(tb) if f.filename == "solution.py"), None)
    if isinstance(error, SyntaxError) and error.filename == "solution.py":
        line = error.lineno
    message = re.sub(r"\d+", "#", str(error.msg if isinstance(error, SyntaxError) else error))[:160]
    return f"{type(error).__name__}: {message}" + (f" (solution.py line {line})" if line else "")


def main() -> None:
    request = json.loads(sys.stdin.read())
    result_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    result = {"load_error": None, "outputs": []}
    fn = None
    namespace = {"__name__": "solution", "__file__": "solution.py"}
    try:
        exec(compile(request["source"], "solution.py", "exec"), namespace)
        fn = namespace.get("transform")
        if not callable(fn):
            raise AttributeError("solution.py does not define transform(grid)")
    except BaseException as error:  # noqa: BLE001 -- the agent's code may raise anything; it is reported
        fn, result["load_error"] = None, describe(error)
    signal.signal(signal.SIGALRM, _alarm)
    for grid in request["inputs"]:
        if fn is None:
            result["outputs"].append([None, "no transform"])
            continue
        signal.alarm(CALL_SECONDS)
        try:
            value = as_grid(fn([list(row) for row in grid]))
            problem = valid_grid(value)
            result["outputs"].append([value, None] if problem is None else [None, f"returned an invalid grid: {problem}"])
        except BaseException as error:  # noqa: BLE001 -- reported
            result["outputs"].append([None, describe(error)])
        finally:
            signal.alarm(0)
    data = (json.dumps(result) + "\n").encode()
    while data:
        data = data[os.write(result_fd, data):]


if __name__ == "__main__":
    main()
