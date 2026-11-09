"""Shared parts of the gen-2 codex hooks: read the game state, build the "status packet", write the log.

A hook is a child process of codex, running inside this game's fence as the game's user; stdlib only (the
agent's python, ``-I``). State is derived only from the server-written ``record/history.jsonl`` (which the
agent can't change) and the note_gate markers, without calling the API; the one exception is seed collection
(``seed_snapshots``), where SessionStart calls the middleware's ``/api/snapshot`` once (:func:`seed_snapshot`).
codex's ``--json`` doesn't show hooks, so every run writes ``$CODEX_HOME/gen2_hooks.jsonl`` -- the runner relies
on it to confirm the hooks really ran. When a hook fails codex only records it in the hook event and blocks
nothing, so uncaught exceptions are written to the same log too (``hook_error``), and the runner stops on seeing one.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

#: Status packet size cap (chars): notes.md always in full (not truncated since 10-21; in 776 window changes it was
#: truncated only 9 times, median 6.5k chars), levels/ from newest to oldest until the packet is full, the rest listed by name.
PACKET_CHARS = 64_000
#: Which levels/ files the status packet carries (budget's ``packet_level_docs``, a runner setting; moved over from
#: arc3-v6b 6675fecd3): all = newest to oldest until full, as above; latest = only the highest-numbered levels/LNN.md,
#: the rest listed by name (10-27: the opening packet of RL single-level games seeded at L08 has a median of 40k chars).
#: **latest is a setting shared by training, evaluation and deployment** (the user, 10-27): RL's harness-train /
#: harness-eval and the Kaggle deployment's arm config must switch together. Key absent = all; an unrecognized value
#: raises. Mutually exclusive with ``packet_levels`` (config ``gen2_packet_levels``): giving both raises.
PACKET_LEVEL_DOCS = ("all", "latest")
GATE_DIR = ".note_gate"
#: Window full: created by PostToolUse when context passes the window-change point, removed by the next window's
#: SessionStart (compact / startup); while it exists act refuses ``do`` (act_client's ``CONTEXT_FULL``, same file name).
CONTEXT_FULL = ".context_full"
#: Settings written by the runner (run_arm's WindowBudget): the window-change point, and the text sent to the model
#: (full / skeleton / tail / stuck, replaceable per prompt set).
BUDGET = "gen2_budget.json"


def read_input() -> dict:
    return json.load(sys.stdin)


def budget() -> dict:
    """Written by the runner before this game's first session (WindowBudget.occupancy); missing means the runner isn't wired up."""
    return json.loads((codex_home() / BUDGET).read_text())


def seed_snapshot(window: int, source: str) -> dict:
    """Seed collection (budget's ``seed_snapshots``): asks the middleware to snapshot the workspace at the start of this
    new window as ``Wnnn``. Uses this game's key (the jail's ``ARC_BASE_URL`` / ``ARC_API_KEY``); raises on failure
    (hook_error, the runner stops). The hook timeout is 60 seconds."""
    req = urllib.request.Request(os.environ["ARC_BASE_URL"].rstrip("/") + "/api/snapshot", method="POST",
                                 data=json.dumps({"window": window, "source": source}).encode())
    req.add_header("X-API-Key", os.environ["ARC_API_KEY"])
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=50) as r:
        return json.loads(r.read())


def codex_home() -> Path:
    home = os.environ.get("CODEX_HOME")
    if not home:
        raise RuntimeError("gen2 hook: CODEX_HOME is not set")
    return Path(home)


def log(event: str, **data) -> None:
    with open(codex_home() / "gen2_hooks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "event": event, **data}, ensure_ascii=False) + "\n")


def _logged_failure(kind, value, tb) -> None:
    try:
        log("hook_error", hook=Path(sys.argv[0]).name, error=repr(value))
    finally:
        sys.__excepthook__(kind, value, tb)


sys.excepthook = _logged_failure


def history(ws: Path) -> list[dict]:
    path = ws / "record" / "history.jsonl"
    if not path.is_file():
        raise RuntimeError(f"gen2 hook: {path} does not exist (the harness opens the game before the session)")
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:          # only the last line can be half-written
                raise
    if not rows:
        raise RuntimeError(f"gen2 hook: {path} is empty")
    return rows


def status(ws: Path, rows: list[dict]) -> dict:
    last = rows[-1]
    levels = int(last.get("levels_completed") or 0)
    clear = None
    before = 0
    for row in rows:
        done = int(row.get("levels_completed") or 0)
        if done > before and row.get("state") != "WIN":
            clear = done
        before = done
    waiting = clear is not None and not (ws / GATE_DIR / ("L%02d.done" % clear)).exists()
    return {"levels": levels, "win_levels": last.get("win_levels"), "actions": int(last.get("step") or 0),
            "state": last.get("state"), "waiting": clear if waiting else None,
            "context_full": (ws / CONTEXT_FULL).exists()}


def where(st: dict) -> str:
    text = ("Where you are (from the server's record): %d of %s level(s) cleared, %d action(s) spent, state %s."
            % (st["levels"], st["win_levels"], st["actions"], st["state"]))
    if st["waiting"] is not None:
        text += ("\nLevel %d was cleared and the next level waits for `./act note_done`: bring levels/L%02d.md "
                 "and notes.md up to date, then run it." % (st["waiting"], st["waiting"]))
    if st["state"] == "GAME_OVER":
        text += "\nThe game is over on this level: only RESET is accepted (`./act do RESET`, one action)."
    if st["context_full"]:
        text += ("\nThis context window is full: `./act do` is refused until the next window. Bring your notes up "
                 "to date, then call `new_context`.")
    return text


#: With ``packet_levels`` (config ``gen2_packet_levels``), earlier level files show only the first paragraph of their
#: Goal section, at most this many chars.
GOAL_CHARS = 300


def goal(text: str) -> str:
    """A level file's title (the first ``# ``) + the first paragraph of its ``## Goal`` section, squeezed onto one line;
    if there is no Goal section, it says so."""
    lines = text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
    body, inside = [], False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().lower().startswith("goal")
        elif inside:
            if line.strip():
                body.append(line.strip())
            elif body:
                break
    said = " ".join(body) if body else "(no Goal section)"
    if len(said) > GOAL_CHARS:
        said = said[:GOAL_CHARS].rstrip() + " ..."
    return "%s: %s" % (title, said) if title else said


def level_number(name: str) -> int | None:
    """``LNN.md`` in levels/ -> NN; other ``L*.md`` names the agent picked itself don't count as any level (None)."""
    m = re.fullmatch(r"L(\d+)\.md", name)
    return int(m.group(1)) if m else None


def packet(ws: Path, st: dict, source: str, last: str, full_levels: int | None = None, level_docs: str = "all") -> str:
    """New window (compact) and new conversation carrying progress (startup): state + full notes.md + levels/, with
    ``last`` as the final paragraph (the configured tail; the review when stuck). ``full_levels``: only the newest
    this-many level files go in full, earlier ones show only their Goal (:func:`goal`); None puts everything in until
    the packet is full (g2b). In v34g4 status packets past 3+ levels, level files took 73% of the chars. ``level_docs``:
    see PACKET_LEVEL_DOCS: with latest only the highest-numbered LNN.md is a candidate, and the others (including names
    that aren't LNN.md) are listed on the "Earlier level files" line (verbatim the same as arc3-v6b's RL training and
    evaluation); the candidate is still subject to PACKET_CHARS."""
    if level_docs not in PACKET_LEVEL_DOCS:
        raise RuntimeError(f"gen2 hook: packet_level_docs {level_docs!r} is not one of {PACKET_LEVEL_DOCS}")
    if level_docs != "all" and full_levels is not None:
        raise RuntimeError(f"gen2 hook: packet_level_docs {level_docs!r} and packet_levels {full_levels!r} both given")
    parts = []
    if source == "compact":
        parts.append("# New context window\n\nThe previous context window is gone. Nothing from it is here except "
                     "AGENTS.md and what follows: the game's state from the server, and your notes and level files "
                     "as they are on disk now.")
    else:
        parts.append("# Picking the game up\n\nA previous session played this game in this workspace. Here is the "
                     "game's state from the server, and your notes and level files as they are on disk now.")
    parts.append(where(st))
    notes = ws / "notes.md"
    if notes.is_file():
        parts.append("## notes.md\n\n" + notes.read_text(encoding="utf-8", errors="replace").rstrip())
    else:
        parts.append("## notes.md\n\n(notes.md does not exist yet.)")
    docs = sorted((ws / "levels").glob("L*.md"), reverse=True) if (ws / "levels").is_dir() else []
    earlier = []
    if level_docs == "latest":
        numbered = [d for d in docs if level_number(d.name) is not None]
        newest = max(numbered, key=lambda d: level_number(d.name)) if numbered else None
        earlier = [d.name for d in docs if d != newest]
        docs = [newest] if newest is not None else []
    older = docs[full_levels:] if full_levels is not None else []
    shown, left = [], []
    used = sum(len(p) for p in parts)
    for doc in docs[:full_levels]:
        text = doc.read_text(encoding="utf-8", errors="replace")
        if used + len(text) <= PACKET_CHARS:
            shown.append("## levels/%s\n\n%s" % (doc.name, text.rstrip()))
            used += len(text)
        else:
            left.append(doc.name)
    if older:
        parts.append("## Older levels\n\nOnly the goal of each is shown here; the whole file is in levels/. Read it "
                     "when a rule or a sequence from that level matters now.\n\n"
                     + "\n".join("- levels/%s -- %s" % (doc.name, goal(doc.read_text(encoding="utf-8",
                                                                                      errors="replace")))
                                 for doc in reversed(older)))
    parts.extend(reversed(shown))
    if earlier:
        parts.append("Earlier level files (read them from levels/ if you need them): " + ", ".join(sorted(earlier)))
    if left:
        parts.append("Not shown for space (read them from levels/ if you need them): " + ", ".join(sorted(left)))
    parts.append(last.strip())
    return "\n\n".join(parts) + "\n"
