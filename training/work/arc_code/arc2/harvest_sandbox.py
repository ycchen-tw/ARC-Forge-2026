"""Isolate generated text solutions while the harness collects their answers."""
from __future__ import annotations

import shutil
from pathlib import Path


def argv(workspace: Path, harness: Path, runtime: Path, command: list[str]) -> list[str]:
    """Only expose system libraries, the public workspace, checker and Python runtime."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("fence none harvest requires bubblewrap; refusing to execute generated code on the host")
    args = [bwrap, "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    # Build a fresh root rather than mounting host data and trying to hide answer directories.
    args += ["--ro-bind", "/usr", "/usr"]
    for name in ("bin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            args += ["--symlink", str(path.readlink()), str(path)]
        elif path.exists():
            args += ["--ro-bind", str(path), str(path)]
    cache = Path("/etc/ld.so.cache")
    if cache.exists():
        args += ["--ro-bind", str(cache), str(cache)]
    args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in (runtime, harness):
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(workspace), str(workspace), "--chdir", str(workspace),
             "--setenv", "HOME", str(workspace), "--setenv", "TMPDIR", "/tmp",
             "--setenv", "LANG", "C.UTF-8", "--setenv", "MPLBACKEND", "Agg",
             "--setenv", "ARC2_HARVEST_SANDBOX", "1",
             "--setenv", "PATH", "/usr/bin:/bin", "--", *command]
    return args
