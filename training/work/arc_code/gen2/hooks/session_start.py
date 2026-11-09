"""SessionStart: at the start of every window, hand the status packet to the model (a developer message, which codex
appends after the initial context).

- ``startup``: a brand-new game (no move made yet, notes.md missing or still the skeleton) only gets the state, and the
  skeleton notes.md is put in place; a game already played (a new conversation taking over an old workspace) gets the
  full status packet.
- ``resume`` (``codex exec resume``): the same thread's history is all there; only the state. The window-full flag stays:
  context is still full.
- ``compact`` (``new_context`` or a threshold window change; gen-2 never writes a summary): the full status packet.

``startup`` and ``compact`` are new windows: ``.context_full`` (created by PostToolUse when the previous window filled up)
is removed first, so act accepts ``do`` again.

Seed collection (budget's ``seed_snapshots``): a new window of a game that has already moved (``compact``, or a
``startup`` taking over an old workspace) asks the middleware to snapshot ``Wnnn`` before building the status packet;
n = which window of this game it is (number of earlier ``startup`` / ``compact`` + 1; the opening one is 1).

The number of level files put in full is the configured ``packet_levels`` (None = all, see common.packet); with
``packet_level_docs`` = latest only the newest level's file is put in (common.PACKET_LEVEL_DOCS, the same setting as RL
training and evaluation); the two are mutually exclusive.

The status packet's last paragraph is the configured ``tail``; if the config has ``stuck`` (the prompt set's stuck.md)
and this level has reached its ``stuck_windows``-th window, it becomes the stuck review instead. This level's window count
= the window it started in (where the level-clearing step was; for level 1, the opening one) + every later new window
that was still on this level at its start. The latter are counted from this hook's own log: ``startup`` / ``compact``,
same levels, moves made (the opening one and restarts before the first move don't count -- they are "the one it started
in"), not waiting for note_done (while waiting, the next level hasn't started yet).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def level_windows(source: str, st: dict) -> int:
    """Which window of this level this is (including this one); 0 if this one is not a new window (resume) or doesn't count."""
    def counts(e: dict) -> bool:
        return (e["source"] in ("startup", "compact") and e["levels"] == st["levels"] and e["actions"] > 0
                and e["waiting"] is None)
    log_path = common.codex_home() / "gen2_hooks.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
    if not counts({"source": source, **st}):
        return 0
    earlier = sum(1 for e in events if e["event"] == "SessionStart" and counts(e))
    return 1 + earlier + 1                  # the window it started in + earlier counted new windows + this one


def level_actions(rows: list[dict], levels: int) -> int:
    start = next((int(r.get("step") or 0) for r in rows if int(r.get("levels_completed") or 0) >= levels), 0) \
        if levels else 0
    return int(rows[-1].get("step") or 0) - start


inp = common.read_input()
cfg = common.budget()
ws = Path(inp["cwd"])
source = inp["source"]
if source not in ("startup", "resume", "compact"):
    raise RuntimeError(f"gen2 SessionStart: unexpected source {source!r}")
cleared = False
if source in ("startup", "compact") and (ws / common.CONTEXT_FULL).exists():
    (ws / common.CONTEXT_FULL).unlink()
    cleared = True
rows = common.history(ws)
st = common.status(ws, rows)
snap = None
if cfg["seed_snapshots"] and source in ("startup", "compact") and st["actions"] > 0:
    log_path = common.codex_home() / "gen2_hooks.jsonl"
    earlier = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
    window = 1 + sum(1 for e in earlier if e["event"] == "SessionStart" and e["source"] in ("startup", "compact"))
    snap = common.seed_snapshot(window, source)
notes = ws / "notes.md"
fresh = st["actions"] == 0 and (not notes.exists() or notes.read_text(encoding="utf-8") == cfg["skeleton"])
if source == "startup" and fresh and not notes.exists():
    notes.write_text(cfg["skeleton"], encoding="utf-8")
windows = level_windows(source, st)
stuck = cfg["stuck"] is not None and windows >= cfg["stuck_windows"] and st["state"] != "WIN"
if source == "resume" or (source == "startup" and fresh):
    text = common.where(st)
else:
    last = (cfg["stuck"].format(level=st["levels"] + 1, windows=windows, actions=level_actions(rows, st["levels"]))
            if stuck else cfg["tail"])
    text = common.packet(ws, st, source, last, cfg["packet_levels"], cfg.get("packet_level_docs", "all"))
common.log("SessionStart", source=source, session_id=inp.get("session_id"), chars=len(text),
           notes_chars=len(notes.read_text(encoding="utf-8", errors="replace")) if notes.exists() else None,
           context_full_cleared=cleared, level_windows=windows, stuck=stuck, seed_snapshot=snap, **st)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))
