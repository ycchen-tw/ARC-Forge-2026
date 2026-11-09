"""Evidence-only diagnosis of a finished Codex driver log.

This module never restarts a process, resets a game, writes a transcript, or
infers state from the model's prose. The opt-in launcher owns the recovery count,
same-workspace/GUID continuation, and the authoritative ledger checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

MAX_FRESH_RECOVERIES = 2

_CONTEXT = re.compile(
    r"The input \((?P<input>[0-9]{1,12}(?:,[0-9]{3}){0,3}) tokens\) is longer than the model's "
    r"context length \((?P<limit>[0-9]{1,12}(?:,[0-9]{3}){0,3}) tokens\)\.?", re.IGNORECASE)
_RECONNECT = re.compile(r"Reconnecting\.\.\. \d+/\d+ \((.*)\)", re.DOTALL)
_INCOMPLETE = re.compile(r"Incomplete response returned, reason: max_output_tokens\.?", re.IGNORECASE)
_EMPTY_COMPACT = re.compile(
    r"compaction response [A-Za-z0-9_-]+ has no assistant message \(tool calls dropped: \d+\)")
_HTTP_TRANSIENT = re.compile(
    r"(?:unexpected status(?: code)?|HTTP(?: error)?)\s*:?\s*"
    r"(?:429|500|502|503|504)(?:\b[^\n]*)", re.IGNORECASE)
_TRANSPORT = re.compile(
    r"(?:error sending request for url \(https?://[^\s)]+\)"
    r"|(?:tcp connect error: )?Connection refused \(os error \d+\)"
    r"|(?:connection error: )?Connection reset by peer \(os error \d+\)"
    r"|request timed out|operation timed out|stream disconnected before completion)\.?",
    re.IGNORECASE)
_CONTEXT_CODES = {"context_length_exceeded", "context_window_exceeded"}
_TRANSIENT_CODES = {"server_error", "internal_server_error", "service_unavailable",
                    "rate_limit_exceeded", "timeout", "connection_error"}


def _classify(error):
    """Only called on a top-level driver error envelope, never tool/model text."""
    if not isinstance(error, dict):
        return "other_error", "", None, None
    message = error.get("message", "")
    if not isinstance(message, str):
        message = ""
    code = error.get("code")
    text = message.strip()
    reconnect = _RECONNECT.fullmatch(text)
    if reconnect:
        text = reconnect[1].strip()
    prefix = "stream disconnected before completion: "
    if text.startswith(prefix):
        text = text[len(prefix):]
    matched = _CONTEXT.fullmatch(text)
    count = limit = None
    if matched:
        count, limit = (int(matched[k].replace(",", "")) for k in ("input", "limit"))
        if count > limit > 0:
            return "context_limit", message, count, limit
        # A purported numeric overflow that does not overflow is not evidence.
        return "other_error", message, count, limit
    if isinstance(code, str) and code in _CONTEXT_CODES:
        return "context_limit", message, None, None
    if _INCOMPLETE.fullmatch(text) or code == "max_output_tokens":
        return "output_incomplete", message, None, None
    if (isinstance(code, str) and code in _TRANSIENT_CODES
            or _EMPTY_COMPACT.fullmatch(text) or _HTTP_TRANSIENT.fullmatch(text)
            or _TRANSPORT.fullmatch(text)):
        return "transient_model_error", message, None, None
    return "other_error", message, None, None


def diagnosis(log_path):
    """Return a small JSON-compatible receipt for the latest unresolved error.

    Only an explicit terminal ``turn.failed`` with verified context-limit
    evidence recommends fresh recovery. A reconnect warning, output truncation,
    missing terminal event, later progress, or uncertain trailing JSON never does.
    Missing files raise; malformed JSON is recorded and cannot authorize recovery
    when it follows the selected failure. The file is read without modifying it.
    """
    path = Path(log_path).resolve()
    hasher = hashlib.sha256()
    selected = None
    observed = {}
    malformed_count, last_malformed = 0, None
    size = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for lineno, raw in enumerate(stream, 1):
            hasher.update(raw)
            size += len(raw)
            if not raw.lstrip().startswith(b"{"):
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed_count += 1
                last_malformed = lineno
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if not isinstance(kind, str):
                malformed_count += 1
                last_malformed = lineno
                continue
            if kind in {"thread.started", "turn.started", "turn.completed"}:
                selected = None
            elif kind in {"item.started", "item.updated", "item.completed"}:
                if selected is not None:
                    selected["terminal"] = False
            elif kind in {"error", "turn.failed"}:
                error = event.get("error") if kind == "turn.failed" else event
                category, message, count, limit = _classify(error)
                summary = observed.setdefault(category, {"count": 0, "first_line": lineno, "last_line": lineno})
                summary["count"] += 1
                summary["last_line"] = lineno
                selected = {"kind": category, "terminal": kind == "turn.failed",
                            "message": message[:1000], "input_tokens": count, "context_limit": limit,
                            "line": lineno, "line_sha256": hashlib.sha256(raw).hexdigest(), "event_type": kind}
        after = os.fstat(stream.fileno())
    current = path.stat()
    stable = ((before.st_ino, before.st_size, before.st_mtime_ns)
              == (after.st_ino, after.st_size, after.st_mtime_ns)
              == (current.st_ino, current.st_size, current.st_mtime_ns) and size == after.st_size)
    if selected is None:
        selected = {"kind": "none", "terminal": False, "message": "", "input_tokens": None,
                    "context_limit": None, "line": None, "line_sha256": None, "event_type": None}
    uncertain_tail = last_malformed is not None and (selected["line"] is None or last_malformed > selected["line"])
    source = {"path": str(path), "sha256": hasher.hexdigest(), "bytes": size,
              "line": selected.pop("line"), "line_sha256": selected.pop("line_sha256"),
              "event_type": selected.pop("event_type"), "stable_snapshot": stable}
    return {**selected, "fresh_recommended": bool(stable and not uncertain_tail and selected["terminal"]
                                                  and selected["kind"] == "context_limit"),
            "source": source, "observed": observed,
            "malformed_json_count": malformed_count, "last_malformed_json_line": last_malformed,
            "uncertain_tail": uncertain_tail}


def _integer(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def render_recovery_note(report, *, game_id, step, levels_completed, win_levels,
                         recovery_number, remaining_seconds=None):
    """Render a short fresh-thread handoff using a caller-supplied ledger checkpoint.

    Revalidates the pinned log before rendering. This does not reconstruct notes,
    verify the GUID, write files, or increment the recovery counter; the launcher
    must do those operations and preserve the same live game. The maximum is two.
    """
    recovery_number = _integer(recovery_number, "recovery_number", 1)
    if recovery_number > MAX_FRESH_RECOVERIES:
        raise ValueError(f"at most {MAX_FRESH_RECOVERIES} fresh recoveries are permitted")
    current = diagnosis(report["source"]["path"])
    keys = ("kind", "terminal", "fresh_recommended", "input_tokens", "context_limit", "source")
    if not current["fresh_recommended"] or any(report.get(k) != current[k] for k in keys):
        raise ValueError("fresh recovery requires unchanged, terminal context-limit evidence")
    if not isinstance(game_id, str) or not re.fullmatch(r"[A-Za-z0-9_/-]{1,128}", game_id):
        raise ValueError("game_id must be a bounded identifier without whitespace")
    step = _integer(step, "step")
    levels_completed = _integer(levels_completed, "levels_completed")
    win_levels = _integer(win_levels, "win_levels", 1)
    if levels_completed > win_levels:
        raise ValueError("levels_completed exceeds win_levels")
    remaining = ""
    if remaining_seconds is not None:
        if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                or not math.isfinite(remaining_seconds) or remaining_seconds < 0):
            raise ValueError("remaining_seconds must be finite and nonnegative")
        remaining = f" Remaining wall time: {int(remaining_seconds)} seconds."
    return (
        "The previous conversation exceeded the model context limit. Continue the existing game "
        "in this workspace; do not RESET as part of conversation recovery.\n"
        f"Server-ledger checkpoint: {game_id}, counted step {step}, "
        f"{levels_completed}/{win_levels} levels cleared.{remaining}\n"
        "Check ./act status and the current recorded board before acting. Read relevant notes and "
        "saved code as needed; verify their claims against the observation. Continue with a short "
        "experiment or a checked plan.\n"
    )
