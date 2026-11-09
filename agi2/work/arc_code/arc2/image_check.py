"""Image-only mode (``"view": "image"``): the agent sees only images and cannot get at any cell value.

  task     The workspace has no task.json / .txt / .png. In each request sent to the model the model proxy replaces
           the text marker ``[[image:NAME]]`` with that image (ImageStore); the images live only in the harness's
           memory, out of reach of the agent's shell.
  check    The workspace's check.py is only a client: it POSTs to its own task's proxy (/arc2/check). The harness
           reads the workspace's solution.py as root, has the sandbox run every train and test input, and returns
           text: which train pairs pass; for a wrong pair, whether the size is wrong or how many cells are wrong,
           with an image of the agent's output (wrong cells framed in white); an output image for each test input.
  sandbox  A dedicated uid per task (not the agent's uid) runs sandbox_runner.py in fresh user+mount+ipc+net
           namespaces: this run's workspace directory and every world-writable directory are covered with tmpfs,
           so the transform has nowhere to leave anything the agent could read; prints are discarded, error
           messages keep only type and line number, digits become #.

Side channels left open (they take deliberate effort): non-digit characters in error messages, the dimensions
when the size is wrong, the /proc cmdline while the transform runs.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import threading
from pathlib import Path

import arc2_render as render

MARKER = re.compile(r"\[\[image:([A-Za-z0-9_.-]+)\]\]")
#: Directories covered with tmpfs in the sandbox: `find / -type d -perm -0002` on the VM (2026-10-04 G4: these
#: five). This run's tasks/ is added on top.
WORLD_WRITABLE = ("/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue", "/run/lock")
SOURCE_MAX_BYTES = 1 << 20
#: The sandbox's tmpfs did not mount: a harness fault, not to be counted as the agent's transform failing.
MOUNT_FAILED = 97


class ImageStore:
    """Name → PNG. The proxy substitutes images by name on every request (the same history yields the same bytes
    every time, so the prefix cache is unaffected)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.images: dict[str, str] = {}      # name → data URL

    def put(self, name: str, png: bytes) -> str:
        import base64
        if not MARKER.fullmatch(f"[[image:{name}]]"):
            raise ValueError(f"bad image name {name!r}")
        with self.lock:
            if name in self.images:
                raise RuntimeError(f"image {name} stored twice")
            self.images[name] = "data:image/png;base64," + base64.b64encode(png).decode()
        return f"[[image:{name}]]"

    def get(self, name: str) -> str | None:
        with self.lock:
            return self.images.get(name)


def _split(text: str, store: ImageStore, text_type: str) -> list[dict] | None:
    """Text → content parts (markers replaced by images); None if there is no recognized marker (left as is)."""
    parts, pos, found = [], 0, False
    for m in MARKER.finditer(text):
        url = store.get(m.group(1))
        if url is None:            # a nonexistent name the model typed itself: keep the original text
            continue
        found = True
        if m.end() > pos:
            parts.append({"type": text_type, "text": text[pos:m.end()]})
        parts.append({"type": "input_image", "image_url": url})
        pos = m.end()
    if not found:
        return None
    if pos < len(text):
        parts.append({"type": text_type, "text": text[pos:]})
    return parts


def inject_images(request: dict, store: ImageStore) -> int:
    """Replace markers in user messages and tool outputs of a Responses request with images; returns how many were
    replaced. assistant / reasoning items are left alone."""
    n = 0
    for item in request.get("input") or []:
        kind = item.get("type")
        if kind in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            if isinstance(output, str):
                parts = _split(output, store, "input_text")
                if parts is not None:
                    item["output"] = parts
                    n += sum(p["type"] == "input_image" for p in parts)
            elif isinstance(output, list):
                item["output"], k = _expand(output, store)
                n += k
        elif kind in (None, "message") and item.get("role") == "user" and isinstance(item.get("content"), list):
            item["content"], k = _expand(item["content"], store)
            n += k
    return n


def _expand(content: list, store: ImageStore) -> tuple[list, int]:
    out, n = [], 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
            parts = _split(part["text"], store, "input_text")
            if parts is not None:
                out += parts
                n += sum(p["type"] == "input_image" for p in parts)
                continue
        out.append(part)
    return out, n


class Sandbox:
    """Runs sandbox_runner.py as a dedicated uid in isolated namespaces. With fence none (local testing) it runs
    directly."""

    def __init__(self, user: str | None, python: Path, runner: Path, hide: list[str], data_kb: int) -> None:
        self.user, self.python, self.runner, self.hide, self.data_kb = user, python, runner, hide, data_kb

    def argv(self) -> list[str]:
        py = [str(self.python), "-I", str(self.runner)]
        if self.user is None:
            return ["/bin/bash", "-c", 'ulimit -d "$1"; shift; exec "$@"', "sandbox", str(self.data_kb), *py]
        mounts = " && ".join(f"mount -t tmpfs -o size=64m,mode=1777 none {d}" for d in self.hide)
        inner = f"{mounts} || exit {MOUNT_FAILED}; cd / && exec \"$@\""
        return ["/bin/bash", "-c", 'ulimit -d "$1"; shift; exec runuser -u "$@"', "sandbox", str(self.data_kb),
                self.user, "--", "env", "-i", "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "HOME=/tmp", "MPLBACKEND=Agg",
                "unshare", "--user", "--map-root-user", "--mount", "--ipc", "--net", "--",
                "/bin/sh", "-c", inner, "sandbox-inner", *py]

    def run(self, source: str, inputs: list) -> dict:
        seconds = 30 + 35 * len(inputs)
        try:
            out = subprocess.run(self.argv(), input=json.dumps({"source": source, "inputs": inputs}),
                                 capture_output=True, text=True, timeout=seconds, cwd="/")
        except subprocess.TimeoutExpired:
            return {"load_error": f"the sandbox did not finish within {seconds} s", "outputs": [[None, "timeout"]] * len(inputs)}
        lines = out.stdout.strip().splitlines()
        if out.returncode == 0 and len(lines) == 1:
            try:
                result = json.loads(lines[0])
            except ValueError:
                result = None
            if (isinstance(result, dict) and isinstance(result.get("outputs"), list)
                    and len(result["outputs"]) == len(inputs)):
                return result
        if self.user is not None and out.returncode == MOUNT_FAILED:
            raise RuntimeError(f"sandbox setup failed: {out.stderr[-400:]}")
        # The transform corrupted the result channel, or the process was killed (RLIMIT / memory): this sentence
        # is all the agent sees.
        return {"load_error": f"the sandbox returned no readable result (exit {out.returncode}; transform "
                              "may have crashed the process or written to its output channel)",
                "outputs": [[None, "no result"]] * len(inputs)}


class ImageChecker:
    """check / answer collection for one task. Images in the result text go out as markers, which the proxy
    replaces with the real images."""

    def __init__(self, tid: str, task: dict, ws: Path, uid: int | None, sandbox: Sandbox, store: ImageStore,
                 limit: int) -> None:
        self.tid, self.task, self.ws, self.uid, self.sandbox, self.store = tid, task, ws, uid, sandbox, store
        #: Max check runs per task (failed ones included): the wrong-cell count can serve as an oracle for inferring
        #: cells; with enough runs the whole grid could be binary-searched out.
        self.limit = limit
        self.refused = 0
        self.lock = threading.Lock()
        self.calls = 0
        self.best: tuple[int, str] | None = None      # (train pairs passed, source): updated when passed ≥ previous best
        self.inputs = [p["input"] for p in task["train"]] + [t["input"] for t in task["test"]]

    def read_source(self) -> str | tuple[None, str]:
        """Read the workspace's solution.py as root: symlinks not followed; must be a regular file owned by this
        task's uid."""
        path = self.ws / "solution.py"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None, "there is no solution.py in the workspace"
        except OSError as error:
            return None, f"solution.py cannot be opened ({type(error).__name__}; it must be a regular file, not a link)"
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or (self.uid is not None and st.st_uid != self.uid):
                return None, "solution.py must be a regular file you created"
            if st.st_size > SOURCE_MAX_BYTES:
                return None, f"solution.py is larger than {SOURCE_MAX_BYTES} bytes"
            data = os.read(fd, SOURCE_MAX_BYTES + 1)
        finally:
            os.close(fd)
        try:
            return data.decode()
        except UnicodeDecodeError:
            return None, "solution.py is not UTF-8 text"

    def evaluate(self, source: str) -> dict:
        """A dict shaped like check.py --harvest output (answer_file is always absent: image-only mode collects only
        the transform)."""
        r = self.sandbox.run(source, self.inputs)
        n = len(self.task["train"])
        h = {"load_error": r["load_error"], "train": [], "test": [], "outputs": r["outputs"]}
        for pair, (got, error) in zip(self.task["train"], r["outputs"][:n], strict=True):
            h["train"].append({"verdict": f"ERROR: {error}" if error else _compare(got, pair["output"])})
        for got, error in r["outputs"][n:]:
            h["test"].append({"transform": got, "transform_error": error, "answer_file": None, "answer_error": None})
        return h

    def harvest(self, source: str | None = None) -> dict:
        if source is None:
            source = self.read_source()
        if isinstance(source, tuple):
            n_test = len(self.task["test"])
            return {"load_error": source[1], "train": [{"verdict": "ERROR: no transform"}] * len(self.task["train"]),
                    "test": [{"transform": None, "transform_error": "no transform", "answer_file": None,
                              "answer_error": None}] * n_test}
        return self.evaluate(source)

    def check(self) -> str:
        with self.lock:
            if self.calls >= self.limit:
                self.refused += 1
                return (f"check.py has been run {self.limit} times, its limit for this puzzle, and will not run again. "
                        "When your session ends, the harness still runs transform from solution.py on the test inputs.\n")
            self.calls += 1
            k = self.calls
            source = self.read_source()
            used = f"(check {k} of {self.limit} for this puzzle)"
            if isinstance(source, tuple):
                return f"cannot run solution.py: {source[1]}\n{used}\n"
            h = self.evaluate(source)
            if h["load_error"]:
                return f"cannot load solution.py: {h['load_error']}\n{used}\n"
            lines, passed = [], 0
            n = len(self.task["train"])
            for j, (pair, entry) in enumerate(zip(self.task["train"], h["train"], strict=True), 1):
                got = h["outputs"][j - 1][0]
                verdict = entry["verdict"]
                if verdict == "PASS":
                    passed += 1
                    lines.append(f"train {j}: PASS")
                elif got is None:
                    lines.append(f"train {j}: {verdict}")
                else:
                    want = pair["output"]
                    same = len(got) == len(want) and len(got[0]) == len(want[0])
                    wrong = [[a != b for a, b in zip(ra, rb)] for ra, rb in zip(got, want)] if same else None
                    mark = self.store.put(f"check{k}_train{j}", render.png(got, wrong))
                    if same:
                        cells = sum(map(sum, wrong))
                        lines.append(f"train {j}: FAIL -- {cells} of {len(want) * len(want[0])} cells wrong "
                                     f"(outlined in white); your output: {mark}")
                    else:
                        lines.append(f"train {j}: FAIL -- your output is {len(got)}x{len(got[0])}, expected "
                                     f"{len(want)}x{len(want[0])} (rows x columns); your output: {mark}")
            lines.append(f"{passed}/{n} training pairs pass")
            for j, (got, error) in enumerate(h["outputs"][n:], 1):
                if error:
                    lines.append(f"test {j}: ERROR: {error}")
                else:
                    mark = self.store.put(f"check{k}_test{j}", render.png(got))
                    lines.append(f"test {j}: output {len(got)}x{len(got[0])}: {mark}")
            if self.best is None or passed >= self.best[0]:
                self.best = (passed, source)
            lines.append(used)
            return "\n".join(lines) + "\n"


def _compare(got, want) -> str:
    """Same verdict as check.py.compare, but lists no cell values."""
    if len(got) != len(want) or len(got[0]) != len(want[0]):
        return f"FAIL: size {len(got)}x{len(got[0])}, expected {len(want)}x{len(want[0])}"
    wrong = sum(a != b for ra, rb in zip(got, want) for a, b in zip(ra, rb))
    return "PASS" if not wrong else f"FAIL: {wrong} of {len(want) * len(want[0])} cells wrong"


CLIENT = '''"""Run solution.py on this puzzle (in a sandbox) and report the result.

    python3 check.py

Training pairs: PASS, or how the output is wrong (size, or how many cells) with an image of your output
(wrong cells outlined in white). Test inputs: an image of transform's output for each.
"""
import sys
import urllib.request

req = urllib.request.Request("http://127.0.0.1:{port}/arc2/check", data=b"", method="POST")
with urllib.request.urlopen(req, timeout=900) as r:
    sys.stdout.write(r.read().decode())
'''


def client_source(port: int) -> str:
    return CLIENT.replace("{port}", str(port))
