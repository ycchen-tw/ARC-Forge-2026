"""After a compaction, carry the last stretch of the previous window's raw text forward: an optional
sampling_proxy feature (``--carry-tail-tokens``).

Why: after a duckx c2 compaction the model re-reads the README, rewrites its object detection and
re-derives the board, spending 12k more tokens on average before it acts again (median summary 8.8k
characters, and its content is actually decent). Super Duck does not compact: it drops the oldest items,
so recent raw text is always there. This lets codex's new window still see, after the summary, the tool
calls and outputs of the last few turns before compaction (no reasoning; images replaced by one line).

How: codex (patch 0002) sends ``session_id`` = ``<thread>:<window number>`` with every request. When a
thread moves to its next window, the previous window's last request is the compaction request -- its
input is the whole pre-compaction history plus one final user message (the compact prompt). Drop that
message, walk back from the tail taking function_call/output pairs and assistant messages within a
budget of ``budget`` tokens (estimated as characters / ``chars_per_token``), and insert them after the
summary user message in every request of the new window. The position is fixed, so the prefix cache
still hits; the next compaction's summary request also sees this stretch.

Contract:
  - If the previous window's last request does not end with a user message (not a compaction request,
    e.g. a restart midway), nothing is carried; counted as ``no_compaction``.
  - If the new window's input has no summary (codex's fixed prefix ``SUMMARY_PREFIX``), nothing is
    carried; counted as ``no_summary`` and printed loudly.
  - A skipped number (window number not +1) is not carried; counted as ``window_gap``.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

#: start of codex 0.154 ``templates/compact/summary_prefix.md`` (verbatim message text of a compacted
#: record)
SUMMARY_PREFIX = "Another language model started to solve this problem and produced a summary of its thinking process."
IMAGE_NOTE = "[image from before the context was summarized; not carried over -- view it again if you need it]"
#: every other content part is treated as media and replaced by IMAGE_NOTE (codex images are input_image,
#: videos video_url)
TEXT_PARTS = ("input_text", "output_text", "text")


def _text_of_message(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(c.get("text") or "" for c in content or [] if isinstance(c, dict))


def is_user_message(item: dict) -> bool:
    return item.get("type") in (None, "message") and item.get("role") == "user"


def is_summary(item: dict) -> bool:
    return is_user_message(item) and _text_of_message(item).startswith(SUMMARY_PREFIX)


def _strip_media(item: dict) -> dict:
    output = item.get("output")
    if not isinstance(output, list):
        return item
    item = copy.deepcopy(item)
    item["output"] = [part if isinstance(part, dict) and part.get("type") in TEXT_PARTS
                      else {"type": "input_text", "text": IMAGE_NOTE} for part in item["output"]]
    return item


def _clip(text: str, keep: int) -> str:
    if len(text) <= keep:
        return text
    half = max(keep // 2, 1)
    return text[:half] + f"\n…[{len(text) - 2 * half} characters cut when the context was summarized]…\n" + text[-half:]


def _fit(unit: list[dict], budget_chars: int) -> list[dict] | None:
    """When the newest unit alone exceeds the budget, clip its output text from the middle until it fits;
    None if it cannot fit."""
    unit = copy.deepcopy(unit)
    out = unit[-1]
    if out.get("type") not in ("function_call_output", "custom_tool_call_output"):
        return None
    over = _chars(unit) - budget_chars
    if over <= 0:
        return unit
    if isinstance(out.get("output"), str):
        out["output"] = _clip(out["output"], max(len(out["output"]) - over - 200, 0))
    elif isinstance(out.get("output"), list):
        texts = [p for p in out["output"] if p.get("type") in TEXT_PARTS]
        total = sum(len(p.get("text") or "") for p in texts)
        for p in texts:
            share = (len(p.get("text") or "") / total) if total else 0
            p["text"] = _clip(p.get("text") or "", max(int((total - over - 200) * share), 0))
    return unit if _chars(unit) <= budget_chars else None


def _chars(items: list[dict]) -> int:
    return sum(len(json.dumps(it, ensure_ascii=False)) for it in items)


def select_tail(items: list[dict], budget_chars: int) -> list[dict]:
    """Input of a compaction request (last item is the compact prompt) -> the stretch to carry forward
    (original order)."""
    if not items or not is_user_message(items[-1]):
        raise ValueError("not a compaction request: the last item is not a user message")
    body = items[:-1]
    outputs_by_call = {it.get("call_id"): i for i, it in enumerate(body)
                       if it.get("type") in ("function_call_output", "custom_tool_call_output")}
    units: list[list[dict]] = []
    taken: set[int] = set()
    for i in range(len(body) - 1, -1, -1):
        it = body[i]
        kind = it.get("type")
        if i in taken or kind == "reasoning" or is_user_message(it) or it.get("role") in ("developer", "system"):
            continue
        if kind in ("function_call_output", "custom_tool_call_output"):
            call = next((j for j in range(i - 1, -1, -1)
                         if body[j].get("type") in ("function_call", "custom_tool_call")
                         and body[j].get("call_id") == it.get("call_id")), None)
            if call is None:
                continue                                   # call is in an earlier window: unpaired, not carried
            taken.update((i, call))
            units.append([body[call], _strip_media(it)])
        elif kind in ("function_call", "custom_tool_call"):
            if it.get("call_id") in outputs_by_call:
                continue                                   # carried together with its output
            taken.add(i)
            units.append([it])                             # call without output (interrupted): carried
        elif kind == "message" and it.get("role") == "assistant":
            taken.add(i)
            units.append([it])
    tail: list[dict] = []
    used = 0
    for unit in units:                                     # newest to oldest
        size = _chars(unit)
        if used + size > budget_chars:
            if not tail:                                   # newest unit too big: clip it, don't drop it
                fitted = _fit(unit, budget_chars)
                if fitted is not None:
                    tail = fitted
            break
        tail = unit + tail
        used += size
    return tail


def shape(items: list[dict], last: int = 8) -> list[list]:
    """Debug aid: (type, role, character count) of the last few items."""
    return [[it.get("type"), it.get("role"), len(json.dumps(it, ensure_ascii=False))] for it in items[-last:]]


@dataclass
class Thread:
    window: int | None = None
    last_input: list | None = None      # input of this window's last request
                                        # (after the tail insert, before the reasoning trim)
    tail: list = field(default_factory=list)


@dataclass
class CompactionTail:
    budget: int
    chars_per_token: float = 2.0        # tool output is mostly hex boards, denser than reasoning's 2.95
    log_path: Path | None = None
    threads: dict[str, Thread] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stats: dict[str, int] = field(default_factory=lambda: {
        "requests": 0, "carried_windows": 0, "inserted_requests": 0, "no_compaction": 0,
        "no_summary": 0, "window_gap": 0, "no_session": 0})

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"budget must be positive: {self.budget}")

    def _log(self, event: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), **event}, ensure_ascii=False) + "\n")

    def apply(self, payload: dict) -> dict:
        items = payload.get("input")
        session = payload.get("session_id")
        with self.lock:
            self.stats["requests"] += 1
            if not isinstance(items, list):
                return payload
            if not session or ":" not in session:
                self.stats["no_session"] += 1
                return payload
            thread_id, _, number = session.rpartition(":")
            window = int(number)
            th = self.threads.setdefault(thread_id, Thread())
            if th.window != window:
                previous, th.tail = th.last_input, []
                if th.window is not None and previous is not None:
                    if window != th.window + 1:
                        self.stats["window_gap"] += 1
                        self._log({"event": "window_gap", "thread": thread_id, "from": th.window, "to": window})
                    elif not is_user_message(previous[-1]):
                        self.stats["no_compaction"] += 1
                        self._log({"event": "no_compaction", "thread": thread_id, "window": window})
                    else:
                        th.tail = select_tail(previous, int(self.budget * self.chars_per_token))
                        self.stats["carried_windows"] += 1
                        self._log({"event": "carry", "thread": thread_id, "window": window,
                                   "items": len(th.tail), "chars": _chars(th.tail),
                                   "est_tokens": int(_chars(th.tail) / self.chars_per_token),
                                   "previous_shape": shape(previous)})
                th.window, th.last_input = window, None
            out = items
            if th.tail:
                at = next((i for i in range(len(items) - 1, -1, -1) if is_summary(items[i])), None)
                if at is None:
                    self.stats["no_summary"] += 1
                    print(f"[compaction_tail] NO SUMMARY in window {session}; tail not inserted", flush=True)
                    self._log({"event": "no_summary", "session": session})
                else:
                    out = items[:at + 1] + th.tail + items[at + 1:]
                    self.stats["inserted_requests"] += 1
            th.last_input = out
            payload["input"] = out
            return payload

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.stats, "threads": len(self.threads)}
