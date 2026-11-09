"""Nudge after too long without acting (duckx c0i3): hooks into ``sampling_proxy.py``'s ``/v1/responses``
(``--act-nudge-every N``).

Why: in c0i-r2, stretches of more than 20k tokens between actions make up half the output (two thirds in c0-k2),
and they are followed by the lowest rate of clearing a level within the next 20 actions (>50k 19%, ≤5k 42%).
Gaps longer than 8 tool calls are only 2.3%, but the output after the 8th call is about 27%. Super Duck has no
such long tail: once its turn uses up its budget it is called back with "Act NOW with the best option you
already have".

How: walk the input counting tool calls since the last action; for the Nth, 2Nth, 3Nth... call, append a
one-line reminder (with the count) after its output. Looks only at the history itself, so the same history
always yields the same bytes -- the prefix cache only takes appends. assistant/reasoning are untouched; codex's
stored history lacks the line, and the proxy adds it back to every request by the same rule.
"Acted" = the command contains ``./act do`` (counts even if the agent filters the output), or the output
contains act_images' board marker (c0L acts from Python, invisible in the command) -- so this must run before
act_images replaces the marker with an image.
"""

from __future__ import annotations

import json
import re
import threading

ACT = re.compile(r"(^|[\s;&|(])\./act do\b")
BOARD = "[[arc-image:board:"
CALL_TYPES = ("function_call", "custom_tool_call")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
#: Shared by both arms: c0i acts via ./act do, c0L via Python's act(); the sentence doesn't say which
NOTE = ("[harness] {n} commands since your last game action. If you have a plausible next move, send it now -- "
        "one real action teaches more than another pass over the same board.")


def _command(item: dict) -> str:
    raw = item.get("arguments") if item.get("type") == "function_call" else item.get("input")
    if not isinstance(raw, str):
        return ""
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return args.get("cmd", "") if isinstance(args, dict) and isinstance(args.get("cmd"), str) else raw


def _text(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(p.get("text", "") for p in output if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


class ActNudge:
    def __init__(self, every: int) -> None:
        if every < 2:
            raise ValueError(f"--act-nudge-every {every}: a reminder after every command is not a nudge")
        self.every = every
        self.lock = threading.Lock()
        self.counts = {"requests": 0, "requests_with_nudges": 0, "nudges": 0}

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.counts)

    def apply(self, payload: dict) -> int:
        """Modify payload in place; return how many reminders this request carries."""
        items = payload.get("input")
        items = items if isinstance(items, list) else []
        boards = {item.get("call_id") for item in items
                  if isinstance(item, dict) and item.get("type") in OUTPUT_TYPES and BOARD in _text(item.get("output"))}
        marked: dict[str, int] = {}
        since = 0
        for item in items:
            if not isinstance(item, dict) or item.get("type") not in CALL_TYPES:
                continue
            if ACT.search(_command(item).replace("2>&1", "")) or item.get("call_id") in boards:
                since = 0
                continue
            since += 1
            if since % self.every == 0 and isinstance(item.get("call_id"), str):
                marked[item["call_id"]] = since
        n = 0
        for item in items if marked else ():
            if isinstance(item, dict) and item.get("type") in OUTPUT_TYPES and item.get("call_id") in marked:
                note = NOTE.format(n=marked[item["call_id"]])
                output = item.get("output")
                if isinstance(output, str):
                    item["output"] = f"{output}\n\n{note}"
                elif isinstance(output, list):
                    item["output"] = [*output, {"type": "input_text", "text": f"\n\n{note}"}]
                else:
                    continue
                n += 1
        with self.lock:
            self.counts["requests"] += 1
            self.counts["requests_with_nudges"] += bool(n)
            self.counts["nudges"] += n
        return n
