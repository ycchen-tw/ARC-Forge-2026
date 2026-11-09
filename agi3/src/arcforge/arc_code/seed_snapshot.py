"""Workspace snapshots taken live inside gen-2 games, later used as prior-context seeds for ARC3 RL (the contract of
``level_seed.py``).

First-generation seeds were replayed after the fact by ``work/arc_code/trace_replay.py`` (about 2% of files differed
from the original game, lf52's frames were nondeterministic, and two levels cleared within one call could not be split).
Here the middleware (``--seed-snapshots``) takes them live at two points instead:

- **Level change** (``LKK``): before level K's (K≥2) first counted action is sent to the official server, taken inside
  that game's lock. The record's last step is exactly level K−1's clearing action and ``current.txt`` is its
  ``final.txt``: precisely the shape ``level_seed.record_tail`` wants.
- **Window change** (``Wnnn``): the gen-2 SessionStart hook calls ``POST /api/snapshot`` in a new window (``compact``, or
  ``startup`` taking over an old workspace) before assembling the status pack; the middleware again takes it inside the
  lock. The record then stops at an arbitrary step (mid-level); starting RL from there would need the in-level actions
  replayed separately -- kept for later.

Copying: ``record/steps/`` is written by the middleware (``O_EXCL``, never changed once written, read-only for the
agent), so it is hard-linked; everything else is deep-copied (an agent overwriting with ``>`` modifies the inode in
place, and a hard link would make the snapshot change with it). :data:`EXCLUDED` (driver home, logs, gen-2 flags) and
``__pycache__`` are not copied. Symlinks and special files are not copied and files that vanish mid-copy (temp files of
background processes) are skipped; both are recorded in ``meta.json``'s ``skipped`` -- such a snapshot is not exactly
the workspace, and a later gate decides whether to use it.

Output: ``<root>/<game dir name>/<name>/ws`` and ``meta.json``; written as ``<name>.tmp`` and then renamed, so a visible
``<name>/meta.json`` means it is complete.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
from pathlib import Path

from .level_seed import tree_sha256

#: Top-level workspace entries kept out of snapshots: codex's driver home (rollout, sqlite, hooks log), the driver's log, gen-2 flags,
#: and the pace file the RL proxy writes (act_client.PACE_FILE: that episode's output tokens; an episode loading the seed has its own proxy rewrite it)
EXCLUDED = frozenset({".driver", "driver.log", "driver_digest.jsonl", ".note_gate", ".context_full", ".pace.json"})
#: Hard-linked subtree (written by the middleware, unchanged once written)
LINKED = ("record", "steps")
NAME = re.compile(r"^(L\d{2}|W\d{3})$")


class SnapshotError(RuntimeError):
    pass


def level_name(level: int) -> str:
    return f"L{level:02d}"


def window_name(window: int) -> str:
    return f"W{window:03d}"


def exists(root: Path, game_dir: str, name: str) -> bool:
    return (Path(root) / game_dir / name / "meta.json").is_file()


def record_last(workspace: Path) -> dict:
    """The last row of record/history.jsonl (the server-side progress at snapshot time)."""
    path = Path(workspace) / "record" / "history.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SnapshotError(f"{path} is empty")
    return json.loads(lines[-1])


def _copy_tree(src: Path, dst: Path, skipped: list[dict]) -> int:
    """src -> dst (dst does not exist). Returns the number of files copied."""
    files = 0

    def unreadable(error: OSError) -> None:            # os.walk silently skips unlistable directories by default: record them
        skipped.append({"path": Path(error.filename).relative_to(src).as_posix(), "why": type(error).__name__})

    for dirpath, dirnames, filenames in os.walk(src, onerror=unreadable):
        here = Path(dirpath)
        rel = here.relative_to(src)
        if not rel.parts:
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED]
            filenames = [f for f in filenames if f not in EXCLUDED]
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        linked = rel.parts[:2] == LINKED
        (dst / rel).mkdir()
        for name in list(dirnames):
            mode = os.lstat(here / name).st_mode
            if stat.S_ISLNK(mode):
                skipped.append({"path": (rel / name).as_posix(), "why": "symlink"})
                dirnames.remove(name)
        for name in sorted(filenames):
            path = here / name
            rel_path = (rel / name).as_posix()
            try:
                mode = os.lstat(path).st_mode
                if not stat.S_ISREG(mode):
                    skipped.append({"path": rel_path, "why": "symlink" if stat.S_ISLNK(mode) else "special"})
                    continue
                if linked:
                    os.link(path, dst / rel / name)
                else:
                    shutil.copy2(path, dst / rel / name, follow_symlinks=False)
            except FileNotFoundError:
                skipped.append({"path": rel_path, "why": "vanished"})
                continue
            files += 1
    return files


def take(workspace: Path, root: Path, name: str, meta: dict) -> dict:
    """Take one snapshot. Raise if ``name`` already exists (the caller checks :func:`exists` first). Returns what was written to ``meta.json``."""
    workspace, root = Path(workspace), Path(root)
    if not NAME.match(name):
        raise SnapshotError(f"snapshot name {name!r} is not LKK or Wnnn")
    out = root / workspace.name / name
    tmp = root / workspace.name / f"{name}.tmp"
    if out.exists():
        raise SnapshotError(f"{out} already exists")
    if tmp.exists():
        shutil.rmtree(tmp)                     # a previous snapshot stopped halfway (middleware killed mid-copy): incomplete, retake
    tmp.mkdir(parents=True)
    t0 = time.monotonic()
    skipped: list[dict] = []
    files = _copy_tree(workspace, tmp / "ws", skipped)
    last = record_last(tmp / "ws")
    full = {**meta, "name": name, "prefix": str(workspace), "files": files, "skipped": skipped,
            "record_last": {k: last.get(k) for k in ("step", "level", "state", "levels_completed", "win_levels")},
            "tree_sha256": tree_sha256(tmp / "ws"), "copy_seconds": round(time.monotonic() - t0, 3),
            "unix": time.time()}
    (tmp / "meta.json").write_text(json.dumps(full, indent=1, ensure_ascii=False) + "\n")
    os.rename(tmp, out)
    return full
