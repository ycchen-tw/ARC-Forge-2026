"""Reporting, staging bookkeeping, and the finders for every mounted input.

Forked on 2026-09-25 from work/arc_agi3/arc_code_qwen/notebook/01_prelude.py
(the old arm, which retires once this one has scored). Emitted as one cell by
work/arc_code/kaggle/build_notebook.py, which concatenates the files here in
name order.
"""


# Imported here for every cell: the notebook is one module split across cells,
# so a name this cell never uses may be the one the serve cell needs.
import base64
import datetime
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORK = pathlib.Path("/kaggle/working")
# Everything the *agent* touches -- its venv, the coding CLI, arc-code's
# checkout and the run directory. None of it is evidence and none of it belongs
# in the commit: /kaggle/working is both the output and a 20 GB budget, and the
# venv, the CLI and the checkout used to be committed with it. The run directory
# is the one thing worth keeping and MAIN copies that back; MAIN's finally then
# removes this tree.
#
# It also has to be reachable when /kaggle is not -- the fence variant that hides
# the corpus does it by closing /kaggle to every unprivileged uid -- which is why
# this is not simply WORK.
#
# jail.sh gives the session a private tmpfs /tmp, so nothing here is visible
# inside the jail except what it binds through by absolute path: the venv,
# act.py, the codex binary and the game's own workspace. That was already true
# under /kaggle, which the jail does not bind at all, so the move does not widen
# or narrow what a game can see.
AGENT = pathlib.Path("/tmp/arcrun")
RUN = AGENT / "arm-run"
# The new arm is carried with its repo layout -- work/arc_code/ beside
# src/arcforge/ -- because run_arm.py finds the arcforge package at ../../src
# from itself. It lands under AGENT, not WORK: under the relocate fence /kaggle
# is closed to every game's uid, and codex, running as that uid, has to read the
# driver files (its base instructions and compaction prompt) out of this tree.
CODE = AGENT / "code"
UNIT = CODE / "work" / "arc_code"
T0 = time.monotonic()
# Background work that must not outlive a failure. ThreadPoolExecutor registers
# an atexit hook that joins its workers *before* daemon threads are killed, so a
# reader left running would hold the kernel open until it finished the file.
PENDING = []
state = {"stages": {}, "complete": False}


def save():
    (WORK / "kaggle-arm-state.json").write_text(json.dumps(state, indent=2))


def disk():
    usage = shutil.disk_usage(WORK)
    return f"{usage.free / 2**30:.1f}/{usage.total / 2**30:.1f} GiB free"


def say(*parts):
    """The only channel out of a running kernel is stdout.

    kaggle-arm-state.json records everything, but it cannot be read until the
    session ends, so a nine-hour run that printed only at the end would be
    unobservable for nine hours and would report a failure once, at the end.
    """
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def stage(name, fn):
    begin = time.monotonic()
    say("STAGE", name, "begin")
    try:
        value = fn()
    except BaseException as error:
        state["stages"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        save()
        while PENDING:
            PENDING.pop()()
        say("STAGE", name, "FAILED", f"{type(error).__name__}: {error}")
        raise
    seconds = round(time.monotonic() - begin, 1)
    state["stages"][name] = {"ok": True, "seconds": seconds,
                             "value": value if isinstance(value, (str, int, float, list, dict)) else None}
    save()
    say("STAGE", name, f"ok in {seconds}s")
    return value


def run(argv, timeout, env=None, cwd=None, log=None):
    """Redirected output still has to reach the exception.

    With `log=` the child's stdout goes to a file and `done.stdout` is None, so
    the message used to read `... exited 1:` and stop. That file is only
    retrievable if the session ends cleanly -- which is not the case you need it
    in -- so the tail is read back and carried in the error.
    """
    path = WORK / (log or "stage")
    handle = path.open("a") if log else None
    try:
        done = subprocess.run(argv, timeout=timeout, env=env or os.environ.copy(), cwd=cwd,
                              stdout=handle or subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    finally:
        if handle is not None:
            handle.close()
    output = done.stdout
    if output is None:
        output = path.read_text(errors="replace") if path.exists() else ""
    if done.returncode:
        raise RuntimeError(f"{argv[0]} exited {done.returncode}: {output[-4000:]}")
    return output[-2000:]


def mounted(name):
    """Find a mounted input by name rather than by an assumed path.

    A dataset can land under its own slug or under competitions/<slug>/, and
    Kaggle mounts some inputs as symlinks -- a glob that declines to follow one
    is indistinguishable from a dataset that never attached. Raises rather than
    returning None: every caller here needs what it asked for.
    """
    root = pathlib.Path("/kaggle/input")
    for depth in range(4):
        for entry in root.glob("/".join(["*"] * depth + [name])):
            return entry
    for parent, dirs, _ in os.walk(root, followlinks=True):
        if name in dirs:
            return pathlib.Path(parent) / name
        if parent.count("/") > 6:
            dirs[:] = []
    raise RuntimeError(f"{name} is not mounted under /kaggle/input; mounted: "
                       + ", ".join(sorted(p.name for p in root.iterdir())))


def mounted_glob(pattern):
    """The same lesson as mounted(), one level down: find a mounted *file*.

    A dataset's files sit at /kaggle/input/datasets/<owner>/<slug>/<file>, so the
    `*/<pattern>` this used to be matched nothing -- and paired with
    `next(..., None)` that turned a dataset which had attached perfectly into a
    silently different code path. Raises, and names what is actually mounted.
    Ambiguity is an error too: two matches mean the caller's pattern is wrong,
    and picking one of them would be a guess.
    """
    root = pathlib.Path("/kaggle/input")
    # Walked rather than globbed at a fixed depth. The official AOT dataset keeps
    # its library at datasets/<owner>/<slug>/official-sm120-aot/fused_moe_120/,
    # which is one level deeper than the five this used to search, and the glob
    # reported it as "not mounted" -- indistinguishable from a dataset that never
    # attached. os.walk also follows the symlinks Kaggle mounts some inputs as.
    hits = []
    for parent, dirs, files in os.walk(root, followlinks=True):
        if parent.count(os.sep) - str(root).count(os.sep) > 8:
            dirs[:] = []
            continue
        hits += [pathlib.Path(parent) / f for f in files
                 if pathlib.PurePath(f).match(pattern)]
    hits = sorted(hits)
    if len(hits) > 1:
        raise RuntimeError(f"{len(hits)} inputs match {pattern!r}: {hits}")
    if hits:
        return hits[0]
    raise RuntimeError(f"nothing matches {pattern!r} under /kaggle/input; mounted: "
                       + ", ".join(sorted(p.name for p in root.iterdir())))


def only(root, pattern):
    """One match or an error -- never a silently-chosen first of several."""
    hits = sorted(root.rglob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {pattern!r} under {root}, found {hits}")
    return hits[0]


def unpack_arm():
    """Lay down the arm's own code, carried in the notebook rather than mounted.

    One deterministic gzipped tar; `filter="data"` refuses an absolute path or a
    `..` on the way out. Member names are repo-relative, so they -- not this
    function -- decide where things land, and that is checked rather than assumed.
    Everything is world-readable and nothing is world-writable: the games' uids
    read the driver files and fence scripts from here, and root executes them.
    """
    CODE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(PAYLOAD)))) as bundle:
        bundle.extractall(CODE, filter="data")
    for name in ("run_arm.py", "fence_posix.sh", "posix_fence.py"):
        if not (UNIT / name).is_file():
            raise RuntimeError(f"the carried bundle laid no {name} at {UNIT}")
    for path in [CODE, *CODE.rglob("*")]:
        path.chmod((0o755 if path.is_dir() or path.suffix == ".sh" else 0o644))
    return sorted(str(p.relative_to(CODE)) for p in CODE.rglob("*") if p.is_file())
