"""Prior-context seeds for ARC3 single-level RL: the agent's workspace at the start of level K (replayed from real whole
games by ``work/arc_code/trace_seeds.py``, ``seeds/LKK/ws``). This module hashes, verifies and loads seeds. Standard
library only: both run_arm (.venv) and arc3_levels.py (harness venv) import it.

The seed contract (the stub's ``seed*`` fields, produced by ``arc3_levels.py seed``):

- ``seed``: the seed workspace directory; ``seed_sha256``: its :func:`tree_sha256` (computed when the stub is produced,
  checked at load; a seed that changed is not loaded).
- ``seed_prefix``: the absolute path of the **replay** workspace hard-coded in the seed's files (the agent's code has
  ``sys.path.insert``, ``ROOT=``, ``cd``), replaced with the new workspace path at load. If any file still has it (or
  any ``arc3_trace_replay`` string) after the rewrite, raise: the jail mounts only the new workspace, a script still
  holding the old path cannot read its files in the jail, and training would get an episode with broken prior context.
- ``seed_actions``: the number of actions N counted before the cut = the step of the last row of
  ``record/history.jsonl``; ``level`` K: the last step's ``levels_completed`` must be K−1 (the last step is level K−1's
  clearing action), and ``current.txt`` must equal that step's ``final.txt``.

Loading (:func:`load`) = verify the hash -> deep copy (99% of a seed's files share inodes with other seeds, so no hard
links or mounts: the agent will modify them) -> delete ``__pycache__`` (compiled during replay, with the replay's paths
and mtimes) -> rewrite paths -> check for leftovers -> verify the record. Afterwards run_arm's ``workspace()`` writes
``AGENTS.md`` / ``act.py`` / ``act`` as usual, which replaces the source run's copies in the seed (fp8 seeds carry the
v11 briefing).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

#: A path string from the seed that must not appear in the new workspace (the replay tree's root dir name): still present after the rewrite = missed
REPLAY_MARKER = b"arc3_trace_replay"
#: Subtree skipped by the path rewrite: the per-step images / frames the middleware writes (binary and board text only, no paths). The leftover check still covers them
NO_REWRITE = ("record/steps",)
SPEC_KEYS = {"ws", "sha256", "prefix", "actions", "level", "label"}
#: act's note_gate marker directory (GATE_DIR in act_client NOTE_GATE_HOOK). g2seeds are taken when note_done passes (level_snapshot_at
#: note_done), and snapshots exclude this directory by rule (seed_snapshot.EXCLUDED), so after loading, level K−1 is waiting for note_done
#: again: v5k on 2026-10-27, measured over 15,640 episodes, spent 400 tokens on average from the opening to note_done, 0.46% edited the notes
#: first, 1.2% were first refused by act do. :func:`load`'s ``note_done`` puts the marker back.
GATE_DIR = ".note_gate"


class SeedError(RuntimeError):
    pass


def tree_sha256(root: Path) -> str:
    """Fingerprint of a directory tree: every directory's and file's relative path (sorted) + the sha256 of file contents.
    Special files such as symlinks and sockets raise: a seed may only hold regular files and directories (so that after the
    deep copy the agent cannot reach other seeds through a symlink)."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise SeedError(f"{root} is not a directory")
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        here = Path(dirpath)
        rel_dir = here.relative_to(root).as_posix()
        h.update(b"d\0" + rel_dir.encode() + b"\0")
        for name in sorted(dirnames + filenames):
            path = here / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise SeedError(f"{path}: not a regular file or directory (mode {stat.filemode(mode)})")
        for name in sorted(filenames):
            rel = (here / name).relative_to(root).as_posix()
            digest = hashlib.sha256((here / name).read_bytes()).hexdigest()
            h.update(b"f\0" + rel.encode() + b"\0" + digest.encode() + b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class SeedSpec:
    """What run_arm ``--level-seed`` reads (written out from the stub by run_arc3_level.py)."""
    ws: Path
    sha256: str
    prefix: str
    actions: int
    level: int
    label: str

    def __post_init__(self) -> None:
        if not (isinstance(self.level, int) and not isinstance(self.level, bool) and self.level >= 2):
            raise SeedError(f"a seed starts level K >= 2, not {self.level!r}")
        if not (isinstance(self.actions, int) and not isinstance(self.actions, bool) and self.actions >= 1):
            raise SeedError(f"seed actions must be a positive int (K-1 levels were played), not {self.actions!r}")
        if not (isinstance(self.prefix, str) and self.prefix.startswith("/") and not self.prefix.endswith("/")):
            raise SeedError(f"seed prefix must be an absolute path without a trailing slash, not {self.prefix!r}")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise SeedError(f"seed sha256 must be 64 hex digits, not {self.sha256!r}")

    @classmethod
    def from_json(cls, data: dict) -> "SeedSpec":
        if set(data) != SPEC_KEYS:
            raise SeedError(f"seed spec keys {sorted(data)}, want {sorted(SPEC_KEYS)}")
        return cls(ws=Path(data["ws"]), sha256=data["sha256"], prefix=data["prefix"], actions=data["actions"],
                   level=data["level"], label=data["label"])

    def to_json(self) -> dict:
        return {"ws": str(self.ws), "sha256": self.sha256, "prefix": self.prefix, "actions": self.actions,
                "level": self.level, "label": self.label}


def record_tail(ws: Path, level: int, actions: int) -> dict:
    """The last step of the seed's record: step = ``actions``, ``levels_completed`` = level−1, not finished; ``current.txt`` =
    that step's ``final.txt``, and that step has ``frames.npz`` (the middleware recovers the last board from it on restart).
    Returns the last row."""
    record = Path(ws) / "record"
    lines = [line for line in (record / "history.jsonl").read_text().splitlines() if line.strip()]
    if not lines:
        raise SeedError(f"{record}/history.jsonl is empty")
    last = json.loads(lines[-1])
    problems = []
    if last["step"] != actions:
        problems.append(f"last step {last['step']} != seed actions {actions}")
    if last["levels_completed"] != level - 1:
        problems.append(f"last levels_completed {last['levels_completed']} != {level - 1}")
    if last["state"] != "NOT_FINISHED":
        problems.append(f"last state {last['state']!r}")
    step_dir = record / "steps" / f"{last['step']:06d}"
    if not (step_dir / "frames.npz").is_file():
        problems.append(f"{step_dir}/frames.npz is missing")
    elif (record / "current.txt").read_bytes() != (step_dir / "final.txt").read_bytes():
        problems.append("current.txt is not the last step's final.txt")
    if problems:
        raise SeedError(f"{record}: " + "; ".join(problems))
    return last


def last_clear_step(ws: Path, level: int) -> int:
    """The step in the seed's record where level ``level``−1 was cleared: the same rule as act's ``_last_clear`` (the last step
    where levels_completed jumps up and that is not WIN). act's ``_waiting`` compares the marker's step with it and still
    blocks on a mismatch."""
    clear, done = None, 0
    for line in (Path(ws) / "record" / "history.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        now = row.get("levels_completed") or 0
        if now > done and row.get("state") != "WIN":
            clear = (now, row["step"])
        done = max(done, now)
    if clear is None or clear[0] != level - 1:
        raise SeedError(f"{ws}: the record's last clear is {clear} (level, step), not level {level - 1}")
    return clear[1]


def check(spec: SeedSpec) -> dict:
    """The seed is still the one the stub recorded, and the record agrees with what the stub says. Returns the record's last row."""
    got = tree_sha256(spec.ws)
    if got != spec.sha256:
        raise SeedError(f"{spec.ws}: tree sha256 {got} != the stub's {spec.sha256} (the seed changed)")
    return record_tail(spec.ws, spec.level, spec.actions)


def _files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def load(spec: SeedSpec, dest: Path, link_steps: bool = False, note_done: bool = False) -> dict:
    """Deep-copy the seed to ``dest`` (a new workspace, must not exist yet) and rewrite paths. Returns the load record (goes
    into ARM_STATE).

    ``link_steps``: files under ``record/steps/`` become hard links (same filesystem). The per-step records of levels
    1..K−1 are only read after loading, never written (the middleware only adds later steps, and the path rewrite skips them,
    NO_REWRITE), which saves about half the space per episode (the 2026-10-23 gen-2 seed difficulty eval was stopped by
    /data's floor). Linked files keep the seed store's read-only permissions.

    ``note_done``: write ``.note_gate/L(K-1).done`` = ``{"step": the clearing step, "time": load time}``, the very file act
    writes when note_done passes: act no longer blocks ``do``, and the gen-2 hook's status pack no longer says it is waiting
    for note_done. Raise if the seed already has ``.note_gate`` (snapshots should not carry it). The record gets an extra
    ``note_done`` (level, step); without this flag the key is absent."""
    dest = Path(dest)
    if not dest.is_absolute():
        # This exact string is written into the files: the jail mounts the workspace at the path run_arm gives, as is; a resolved symlink path does not exist in the jail
        raise SeedError(f"{dest}: the workspace path must be absolute")
    if dest.exists():
        raise SeedError(f"{dest} exists: a seed is loaded into a fresh workspace only")
    tail = check(spec)
    src_root = Path(spec.ws)
    linked = lambda rel: link_steps and any(rel == d or rel.startswith(d + "/") for d in NO_REWRITE)  # noqa: E731

    def copy_one(src: str, dst: str) -> str:
        if linked(Path(src).relative_to(src_root).as_posix()):
            os.link(src, dst)
            return dst
        return shutil.copy2(src, dst)

    shutil.copytree(spec.ws, dest, symlinks=True, copy_function=copy_one)
    for dirpath, dirnames, filenames in os.walk(dest):   # the seed store is read-only; the loaded workspace belongs to the agent
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or (stat.S_ISREG(mode) and linked(path.relative_to(dest).as_posix())):
                continue                                   # a linked file is the seed store's inode itself: leave permissions alone
            os.chmod(path, stat.S_IMODE(mode) | stat.S_IWUSR)
    for cache in sorted(dest.rglob("__pycache__"), reverse=True):
        if not cache.is_dir():
            raise SeedError(f"{cache}: __pycache__ that is not a directory")
        shutil.rmtree(cache)
    new = str(dest).encode()
    old = spec.prefix.encode()
    rewritten, total = [], 0
    for path in _files(dest):
        rel = path.relative_to(dest).as_posix()
        data = path.read_bytes()
        total += len(data)
        if old in data and not any(rel == d or rel.startswith(d + "/") for d in NO_REWRITE):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise SeedError(f"{path}: a binary file holds the replay path {spec.prefix}") from None
            mode = path.stat().st_mode
            path.write_text(text.replace(spec.prefix, new.decode()), encoding="utf-8")
            os.chmod(path, stat.S_IMODE(mode))
            data = path.read_bytes()
            rewritten.append(rel)
        if old in data or REPLAY_MARKER in data:
            raise SeedError(f"{path}: still names the replay tree after the rewrite ({spec.prefix})")
    (dest / "record").chmod(0o755)
    record_tail(dest, spec.level, spec.actions)
    out = {"seed": spec.to_json(), "files": sum(1 for _ in _files(dest)), "bytes": total,
           "rewritten": sorted(rewritten), "tail": tail}
    if note_done:
        step = last_clear_step(dest, spec.level)
        gate = dest / GATE_DIR
        if gate.exists():
            raise SeedError(f"{gate}: the seed brought a note gate (snapshots exclude it)")
        gate.mkdir()
        (gate / ("L%02d.done" % (spec.level - 1))).write_text(json.dumps({"step": step, "time": time.time()}))
        out["note_done"] = {"level": spec.level - 1, "step": step}
    return out
