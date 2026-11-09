"""Shared parts of the idle-slot read-only reviewer: stall signals, snapshots, report delivery
(docs/plans/arc3_readonly_reviewer_20261014.md).

The reviewer is read-only with respect to the main game: what it gets is a **fixed snapshot** -- the record cut at
some step, copies of the player's notes and code at that moment -- not the main game's workspace. The snapshot uses
the same layout as the player's workspace (``record/``, ``notes.md``, ``levels/``, ``lib/``, ``scripts/``), so the
player's code runs as is inside it; the player's briefing is saved separately as ``player_AGENTS.md``, and only the
reviewer's own briefing is ``AGENTS.md``.

During a review the harness moves the snapshot forward (``refresh_snapshot``): ``record/`` gains the player's later
steps and ``record/live/`` holds the player's current notes and code; the copies at the snapshot's top level stay at
the moment the review started (the reviewer can change them itself; the harness does not touch them).

``record/steps/<n>/`` is written once and never changed: hardlink whenever possible (Kaggle's machine vanishes once
80–90 GiB are written to disk). **Hardlinked inodes are shared with the main game**, so the fence must hand the
snapshot's ``record/`` to the reviewer read-only (jail.sh's ``ARCSEC_RECORD`` ro-bind; under posix root-owned, go-w).
``history.jsonl`` keeps being appended to, so it is always copied and cut at the snapshot's step; ``notes.md`` and
``lib/`` are edited in place by the player, so they are always copied too.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from . import record as record_mod

#: The report the reviewer writes (in its own workspace)
REPORT = "REVIEW.md"
#: Delivery directory in the player's workspace: review/R01.md, R02.md…
DELIVERY_DIR = "review"
#: Cap on a delivered report: it stays in the player's context (reading it costs), so a longer one is cut with a note
REPORT_MAX_BYTES = 12_000
#: Caps on copying the player's code into the snapshot: per file, total (players often write search caches as big files)
CODE_FILE_MAX = 2_000_000
CODE_TOTAL_MAX = 48_000_000
CODE_DIRS = ("lib", "scripts")
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
#: Live snapshot (``refresh_snapshot``): the player's current notes.md, levels/, lib/, scripts/ are mirrored into the snapshot's
#: record/live/, with the step and time of the latest update in record/LIVE.json. All under record/: the posix fence keeps record/
#: for root, read-only for the reviewer
LIVE_DIR = "live"
LIVE_STATUS = "LIVE.json"


# ---------------------------------------------------------------- stall signals

def read_history(path: Path) -> list[dict]:
    """The committed part of record/history.jsonl (the last line may be half-written, so skip it; any other bad line blows up)."""
    raw = path.read_bytes()
    out = []
    for line in raw[:raw.rfind(b"\n") + 1].splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def level_timeline(rows: list[dict]) -> dict:
    """Compute each level's start and cost from the history.

    A level starts at "the step that cleared the previous level" (level 1 starts at step 0); the ``level`` field is the
    level the action was played on, so on the clearing step ``levels_completed`` has already gone up while ``level`` is
    still the old level. Returns the current level's start (step, time), the seconds and actions spent on it, and the
    seconds spent on each earlier level.
    """
    if not rows:
        raise ValueError("empty history")
    starts = [(0, rows[0]["step"], float(rows[0]["time"]))]    # (levels_completed, step, time)
    for row in rows:
        done = int(row["levels_completed"])
        if done > starts[-1][0]:
            starts.append((done, row["step"], float(row["time"])))
    last = rows[-1]
    current_done, start_step, start_time = starts[-1]
    durations = [b[2] - a[2] for a, b in zip(starts, starts[1:])]
    return {"levels_completed": int(last["levels_completed"]), "win_levels": int(last["win_levels"]),
            "level": current_done + 1, "state": last["state"], "step": int(last["step"]),
            "level_start_step": start_step, "level_start_time": start_time,
            "level_actions": int(last["step"]) - start_step, "last_action_time": float(last["time"]),
            "previous_level_seconds": durations}


def stall(history: Path, now: float) -> dict | None:
    """A game's current stall signal; None when history has no line yet (game not opened)."""
    if not history.exists():
        return None
    rows = read_history(history)
    if not rows:
        return None
    t = level_timeline(rows)
    prev = t["previous_level_seconds"]
    t["level_seconds"] = max(0.0, now - t["level_start_time"])
    t["since_action_seconds"] = max(0.0, now - t["last_action_time"])
    t["median_previous_seconds"] = statistics.median(prev) if prev else None
    return t


# ---------------------------------------------------------------- snapshot

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _link_or_copy(src: Path, dst: Path, link: bool) -> str:
    if src.is_symlink() or not src.is_file():
        # The record holds only regular files written by the middleware. Following a symlink would mean root hardlinking / copying a file from elsewhere to where the reviewer can read it
        raise RuntimeError(f"{src} is not a regular file: refusing to snapshot it")
    if link:
        try:
            os.link(src, dst, follow_symlinks=False)
            return "link"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "copy"


def _copy_code(src_root: Path, dst_root: Path, budget: list[int], skipped: list[str]) -> int:
    n = 0
    for path in sorted(src_root.rglob("*")):
        rel = path.relative_to(src_root)
        if any(part in SKIP_DIRS for part in rel.parts) or path.is_symlink():
            if path.is_symlink():
                skipped.append(f"{src_root.name}/{rel} (symlink)")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            skipped.append(f"{src_root.name}/{rel} (not a regular file)")     # FIFO / socket: copyfile would hang or blow up
            continue
        size = path.stat().st_size
        if size > CODE_FILE_MAX or budget[0] + size > CODE_TOTAL_MAX:
            skipped.append(f"{src_root.name}/{rel} ({size} bytes)")
            continue
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        budget[0] += size
        n += 1
    return n


def _render_step(step_dir: Path, png_size: int) -> int:
    """When the record lacks PNGs (rare6's packets dropped the media), redraw final.png and frame_NNN.png from frames.npz.
    Drawn exactly like the middleware's record.png; the mp4 needs ffmpeg and is not redone."""
    made = 0
    npz = step_dir / "frames.npz"
    if not npz.exists():
        return 0
    frames = np.load(npz)["frames"]
    if not (step_dir / "final.png").exists():
        (step_dir / "final.png").write_bytes(record_mod.png(frames[-1], png_size))
        made += 1
    if len(frames) > 1 and not (step_dir / "frame_000.png").exists():
        for i, grid in enumerate(frames):
            (step_dir / f"frame_{i:03d}.png").write_bytes(record_mod.png(grid, png_size))
            made += 1
    return made


def build_snapshot(ws: Path, dest: Path, *, link: bool = True, render_missing: bool = False,
                   png_size: int = 640) -> dict:
    """Freeze the current state of the player's workspace ``ws`` into ``dest`` (must not exist). Returns the snapshot's
    identity (also written as SNAPSHOT.json).

    The record is cut at the last committed step of history.jsonl; steps/ goes only up to that step. ``render_missing``:
    redraw missing PNGs (for offline tests; live records already have images, and redrawing would write into hardlinked
    directories, so it is not allowed live).
    """
    if dest.exists():
        raise FileExistsError(dest)
    rec = ws / "record"
    # The reader is the harness (root under posix): never follow symlinks in the player's workspace (following one would read for the player what it cannot read)
    for path in (rec, rec / "steps", rec / "history.jsonl", rec / "README.md"):
        if path.is_symlink():
            raise RuntimeError(f"{path} is a symlink: refusing to snapshot it")
    history = rec / "history.jsonl"
    raw = history.read_bytes()
    committed = raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in committed.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{ws}: no committed history to snapshot")
    last_step = int(rows[-1]["step"])
    if render_missing and link:
        raise ValueError("render_missing writes into step directories: it needs link=False")
    dest.mkdir(parents=True)
    out_rec = dest / "record"
    (out_rec / "steps").mkdir(parents=True)
    (out_rec / "history.jsonl").write_bytes(committed)
    shutil.copyfile(rec / "README.md", out_rec / "README.md")
    modes = {"link": 0, "copy": 0}
    rendered = 0
    for step in range(0, last_step + 1):
        src = rec / "steps" / f"{step:06d}"
        if src.is_symlink() or not src.is_dir():
            raise RuntimeError(f"{ws}: history has step {step} but record/steps/{step:06d} is missing")
        dst = out_rec / "steps" / src.name
        dst.mkdir()
        for f in sorted(src.iterdir()):
            modes[_link_or_copy(f, dst / f.name, link)] += 1
        if render_missing:
            rendered += _render_step(dst, png_size)
    # A step without frames writes no final.txt (record.py): its board is that of the nearest earlier step
    final = next(d for d in (out_rec / "steps" / f"{s:06d}" for s in range(last_step, -1, -1))
                 if (d / "final.txt").exists())
    shutil.copyfile(final / "final.txt", out_rec / "current.txt")
    if (final / "final.png").exists():
        shutil.copyfile(final / "final.png", out_rec / "current.png")
    skipped: list[str] = []
    budget = [0]
    copied = {}
    def plain(path: Path, kind) -> bool:
        if path.is_symlink():
            skipped.append(f"{path.name} (symlink)")
            return False
        return kind(path)

    for name in ("notes.md",):
        if plain(ws / name, Path.is_file):
            shutil.copyfile(ws / name, dest / name)
            copied[name] = 1
    if plain(ws / "levels", Path.is_dir):
        copied["levels"] = _copy_code(ws / "levels", dest / "levels", budget, skipped)
    for name in CODE_DIRS:
        if plain(ws / name, Path.is_dir):
            copied[name] = _copy_code(ws / name, dest / name, budget, skipped)
    if plain(ws / "AGENTS.md", Path.is_file):
        shutil.copyfile(ws / "AGENTS.md", dest / "player_AGENTS.md")
    t = level_timeline(rows)
    meta = {"source": str(ws), "made": time.time(), "step": last_step,
            "levels_completed": t["levels_completed"], "win_levels": t["win_levels"],
            "level": t["level"], "level_start_step": t["level_start_step"],
            "level_actions": t["level_actions"], "state": t["state"],
            "history_bytes": len(committed), "history_sha256": _sha256(committed),
            "record_files": modes, "rendered_png": rendered, "copied": copied,
            "code_bytes": budget[0], "skipped": skipped}
    (dest / "SNAPSHOT.json").write_text(json.dumps(meta, indent=1))
    return meta


def _write_atomic(target: Path, data: bytes) -> None:
    """Write a temp file in the same directory, then os.replace: readers see the old file or the new one, never a half-written one."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, target)


def _read_nofollow(path: Path) -> bytes:
    """A file in the player's workspace: the reader is root, so a file swapped for a symlink after the check is still not followed (O_NOFOLLOW)."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _mirror_live(ws: Path, live: Path) -> dict:
    """The player's current notes.md, levels/, lib/, scripts/ -> ``live``: changed files (size or mtime) are replaced whole,
    files the player deleted are deleted too; caps, symlinks and skipped directories as in build_snapshot (``_copy_code``)."""
    live.mkdir(exist_ok=True)
    wanted: dict[Path, Path] = {}
    skipped: list[str] = []
    budget = 0
    sources = []
    if (ws / "notes.md").is_symlink():
        skipped.append("notes.md (symlink)")
    elif (ws / "notes.md").is_file():
        sources.append((Path("notes.md"), ws / "notes.md", False))
    for name in ("levels", *CODE_DIRS):
        root = ws / name
        if root.is_symlink():
            skipped.append(f"{name} (symlink)")
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(ws)
            if any(part in SKIP_DIRS for part in rel.parts) or path.is_dir() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                skipped.append(f"{rel} ({'symlink' if path.is_symlink() else 'not a regular file'})")
                continue
            sources.append((rel, path, True))
    for rel, path, counted in sources:
        st = path.stat()
        if counted and (st.st_size > CODE_FILE_MAX or budget + st.st_size > CODE_TOTAL_MAX):
            skipped.append(f"{rel} ({st.st_size} bytes)")
            continue
        budget += st.st_size if counted else 0
        target = live / rel
        wanted[target] = path
        try:
            have = os.stat(target, follow_symlinks=False)
        except FileNotFoundError:
            have = None
        if have is not None and (have.st_size, have.st_mtime_ns) == (st.st_size, st.st_mtime_ns):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, _read_nofollow(path))
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=False)
    removed = 0
    for path in sorted(live.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
        elif path not in wanted:
            path.unlink()
            removed += 1
    return {"files": len(wanted), "bytes": budget, "removed": removed, "skipped": skipped}


def refresh_snapshot(ws: Path, dest: Path, step: int) -> dict:
    """During a review, bring the snapshot ``dest`` up to the current state of the player's workspace ``ws``; ``step`` is
    the step the snapshot currently reaches.

    ``record/`` gains the steps committed since (as in build_snapshot: hardlinks), ``history.jsonl`` and ``current.txt`` /
    ``current.png`` are replaced with the latest; the player's current notes.md, levels/, lib/, scripts/ are mirrored into
    ``record/live/``; ``record/LIVE.json`` records which step and when this update reached. All written under record/: the
    posix fence hands it back to root, go-w, every session, so the reviewer can read but not change it, and root writing
    into it is not led astray by symlinks the reviewer planted. The copies at the snapshot's top level (from the moment the
    review started) are not touched. Returns the content of LIVE.json.
    """
    rec = ws / "record"
    for path in (rec, rec / "steps", rec / "history.jsonl"):
        if path.is_symlink():
            raise RuntimeError(f"{path} is a symlink: refusing to refresh the snapshot from it")
    raw = (rec / "history.jsonl").read_bytes()
    committed = raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in committed.splitlines() if line.strip()]
    last = int(rows[-1]["step"])
    if last < step:
        raise RuntimeError(f"{ws}: history ends at step {last}, before the snapshot's step {step}")
    out_rec = dest / "record"
    for n in range(step + 1, last + 1):
        src = rec / "steps" / f"{n:06d}"
        if src.is_symlink() or not src.is_dir():
            raise RuntimeError(f"{ws}: history has step {n} but record/steps/{n:06d} is missing")
        dst = out_rec / "steps" / src.name
        dst.mkdir()
        for f in sorted(src.iterdir()):
            _link_or_copy(f, dst / f.name, True)
    if last > step:
        _write_atomic(out_rec / "history.jsonl", committed)
        final = next(d for d in (out_rec / "steps" / f"{s:06d}" for s in range(last, -1, -1))
                     if (d / "final.txt").exists())
        _write_atomic(out_rec / "current.txt", (final / "final.txt").read_bytes())
        if (final / "final.png").exists():
            _write_atomic(out_rec / "current.png", (final / "final.png").read_bytes())
    live = _mirror_live(ws, out_rec / LIVE_DIR)
    t = level_timeline(rows)
    status = {"refreshed_unix": time.time(), "step": last, "level": t["level"],
              "levels_completed": t["levels_completed"], "win_levels": t["win_levels"], "state": t["state"],
              "live": live}
    _write_atomic(out_rec / LIVE_STATUS, json.dumps(status, indent=1).encode())
    return status


# ---------------------------------------------------------------- delivery

def report_text(path: Path) -> str:
    """The reviewer's report, checked before delivery: exists, non-empty, UTF-8; an overlong one is cut with a note."""
    data = path.read_bytes()
    text = data.decode("utf-8")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    if len(data) > REPORT_MAX_BYTES:
        text = data[:REPORT_MAX_BYTES].decode("utf-8", errors="ignore")
        text += f"\n\n[harness] report cut at {REPORT_MAX_BYTES} bytes.\n"
    return text


#: The reviewer workspace delivered along with the report (the scripts it cites are in review_work/ and run on the snapshot): per-file and total caps
WORK_FILE_MAX = 2_000_000
WORK_TOTAL_MAX = 16_000_000
#: Not delivered: codex's home, the driver log, both briefings
WORK_SKIP = {".driver", "AGENTS.md", "player_AGENTS.md", "REVIEW.md", ".tmp", ".arcgame"}


def delivery_header(meta: dict, current: dict | None, frozen: str | None = None, library: bool = False,
                    skipped: list[str] | None = None, tail: str | None = None) -> str:
    """The passage the harness adds on delivery: which step the report is based on, which step the player is at now, and where
    the reviewer's scripts run. ``tail``: the reviewer prompt set's ``delivery_tail.md`` (from r8), inserted before the last
    sentence; absent when there is none (r7 and earlier)."""
    head = (f"[harness] A read-only reviewer studied a copy of your record up to step {meta['step']} "
            f"(level {meta['level']}, {meta['levels_completed']} level(s) cleared), your notes, levels/ and code "
            f"from that moment. It could not play and did not see anything you have done since.")
    if current is not None and current["level"] != meta["level"]:
        # e2e2: the review took 13 minutes, and by delivery the player had gone from level 1 to level 3 -- advice about that level is mostly void, rule findings may still help
        head += (f" You have left level {meta['level']} since: you are now at step {current['step']} (level "
                 f"{current['level']}). Advice about level {meta['level']}'s layout no longer applies; findings "
                 f"about how the game works may still hold, so check them before you rely on them.")
    elif current is not None and current["step"] != meta["step"]:
        head += (f" You are now at step {current['step']} (level {current['level']}); check whether the findings "
                 f"still apply before you act on them.")
    if frozen is not None:
        head += (f" The frozen workspace it worked in -- that copy of your record, notes and code, plus its own "
                 f"scripts under review_work/ -- is {frozen}/; the commands it cites run from there "
                 f"(`cd {frozen} && python3 review_work/...`). Your own notes.md, lib/ and record/ are untouched.")
        if library:
            head += (" The `arcgame` module its scripts import (read-only helpers over record/) is copied into "
                     "that review_work/.")
        if skipped:
            head += (f" Not copied there (size limits or not regular files): {', '.join(skipped[:5])}"
                     f"{f' and {len(skipped) - 5} more' if len(skipped) > 5 else ''}.")
    return head + (" " + " ".join(tail.split()) if tail else "") + " You decide what to do with it.\n\n"


def _copy_work(src: Path, dst: Path) -> dict:
    """Reviewer workspace -> delivery directory: record/ is hardlinked (the snapshot is already hardlinks or copies, written
    once and never changed), the rest is copied, and anything over the cap is skipped and noted."""
    dst.mkdir()
    linked = copied = 0
    total = 0
    skipped = []
    # Copy review_work/ first: when the cap is hit, what gets dropped is the player's big files (search caches), not the scripts the report cites
    for path in sorted(src.rglob("*"), key=lambda p: (p.relative_to(src).parts[0] != "review_work", p)):
        rel = path.relative_to(src)
        if rel.parts[0] in WORK_SKIP or rel.name.startswith("driver.") or any(p in SKIP_DIRS for p in rel.parts):
            continue
        target = dst / rel
        if path.is_symlink():
            skipped.append(f"{rel} (symlink)")
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif rel.parts[0] == "record":
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target, follow_symlinks=False)
            linked += 1
        elif not path.is_file():
            skipped.append(f"{rel} (not a regular file)")
        else:
            size = path.stat().st_size
            if size > WORK_FILE_MAX or total + size > WORK_TOTAL_MAX:
                skipped.append(f"{rel} ({size} bytes)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            total += size
            copied += 1
    return {"linked": linked, "copied": copied, "bytes": total, "skipped": skipped}


def _copy_library(src: Path, dst: Path) -> list[str]:
    """Put the reviewer's arcgame (given only in arms where the player has no library) into the delivered review_work/: when
    the player runs ``python3 review_work/x.py`` as cited, sys.path[0] is right there. If the reviewer wrote a file of the
    same name, keep it and do not overwrite."""
    modules = sorted(src.glob("*.py"))
    dst.mkdir(exist_ok=True)
    copied = []
    for path in modules:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"review library {path} is not a regular file")
        if (dst / path.name).exists():
            continue
        shutil.copyfile(path, dst / path.name)
        copied.append(path.name)
    return copied


def deliver(text: str, ws: Path, meta: dict, current: dict | None, *, review_id: str,
            reviewer_ws: Path | None = None, library: Path | None = None, tail: str | None = None) -> dict:
    """Write the report into the player's workspace as review/<review_id>.md (a new file, never overwritten; the file name is
    the id the scheduler gave, and numbers are not filled in when a previous review was stopped -- the proxy decides "has it
    been opened" by review/<id>); with ``reviewer_ws``, bring its workspace (snapshot + review_work/) to review/Rnn/, where
    the scripts the report cites run as is. ``library``: the arcgame that the reviewer has and the player does not (v14 arms),
    also placed in review/Rnn/review_work/. Returns {"report", "frozen", "work"}.

    Fence ownership (posix: root-owned, readable by the player) is handled by the caller."""
    if not re.fullmatch(r"R\d\d", review_id):
        raise ValueError(f"review id {review_id!r} is not Rnn")
    if library is not None and (reviewer_ws is None or not any(library.glob("*.py"))):
        # Before writing anything: a half-built review/Rnn/ would make every later delivery in this game collide on the name
        raise ValueError(f"a library goes with the reviewer's workspace and needs a module: {library}")
    out = ws / DELIVERY_DIR
    out.mkdir(exist_ok=True)
    name = review_id
    target = out / f"{name}.md"
    if target.exists() or (out / name).exists():
        raise FileExistsError(target)
    work = None
    frozen = None
    if reviewer_ws is not None:
        work = _copy_work(reviewer_ws, out / name)
        if library is not None:
            work["library"] = _copy_library(library, out / name / "review_work")
        frozen = f"{DELIVERY_DIR}/{name}"
    tmp = out / f".{target.name}.tmp"
    # O_EXCL|O_NOFOLLOW: a same-name symlink the player put there is not written through
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(delivery_header(meta, current, frozen, library is not None,
                                     None if work is None else work["skipped"], tail) + text)
    os.replace(tmp, target)
    return {"report": target, "frozen": (out / name) if frozen else None, "work": work}


def notice(path: Path, meta: dict) -> str:
    """The sentence the proxy appends after the player's next tool output once delivered (the file itself survives compaction)."""
    return (f"[harness] A read-only reviewer has written {DELIVERY_DIR}/{path.name} about your play up to step "
            f"{meta['step']} (level {meta['level']}). Read it in full before your next game action; it lists "
            f"evidence from your record, and you decide what to do with it.")
