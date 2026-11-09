"""Trim old reasoning before compaction: an optional sampling_proxy feature (``--trim-reasoning-at``).

Why: codex has a single user turn per game, and Qwen3.8's chat template by default (``preserve_thinking``
unset) keeps every reasoning segment in context. Measured over 12 duckx c1 games: at the moment of
compaction the context (images excluded) is **57% reasoning**, 25% tool output. So compaction is mostly
triggered by old reasoning, and every compaction costs a summary plus re-orientation
(docs/research/super_duck_codex_roundtrips_20261008.md: 13-17% of each game's output) and also loses the
raw tool output.

How: each codex context window (``session_id`` in the request body, sent by codex patch 0002; a new one
after each compaction) keeps one **boundary** -- reasoning items among the first ``boundary`` items of the
input are never sent upstream. The boundary jumps forward only when the context reported by the previous
response (input + output tokens) is past ``trim_at`` and this jump would trim at least ``min_gain``
tokens; it jumps to "before the last ``keep_items`` items" and then stays put: the prefix before the
boundary is identical every time, the prefix cache keeps hitting, and only the jump itself costs a fresh
prefill. When nothing more can be trimmed (what remains is mostly tool output), codex compacts as usual.

Contract:
  - The items before the boundary must be exactly the same as last time (anchored on the hash of the last
    item before the boundary). A mismatch means codex rewrote history inside a window; no guessing here:
    count ``anchor_mismatch``, restart that window, and print it loudly.
  - Only ``type == "reasoning"`` items are touched; everything else is bit-identical. A function_call
    that followed a trimmed reasoning item becomes, on the sglang side, an assistant tool call without
    reasoning_content, which the template renders as an empty think block.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


def item_hash(item: dict) -> str:
    return hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def reasoning_chars(item: dict) -> int:
    """sglang uses ``summary`` first and ``content`` only when that is empty (serving_responses
    conversion rule)."""
    for key in ("summary", "content"):
        parts = item.get(key) or []
        chars = sum(len(p.get("text") or "") for p in parts if isinstance(p, dict))
        if chars:
            return chars
    return 0


@dataclass
class Window:
    boundary: int = 0
    anchor: str | None = None
    context: int | None = None          # input_tokens + output_tokens of the previous response
    trims: int = 0
    requests: int = 0


@dataclass
class ReasoningTrim:
    trim_at: int
    keep_items: int
    min_gain: int
    # Only used to estimate "how much this jump can trim"; the actual amount shows in the next usage.
    # Measured on 1,669 duckx c1 reasoning segments: 2.95 characters/token (many coordinates and numbers)
    chars_per_token: float = 3.0
    log_path: Path | None = None
    windows: dict[str, Window] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stats: dict[str, int] = field(default_factory=lambda: {
        "requests": 0, "trimmed_requests": 0, "trims": 0, "anchor_mismatch": 0,
        "reasoning_items_dropped": 0, "no_session": 0})

    def __post_init__(self) -> None:
        if not (self.trim_at > 0 and self.keep_items >= 0 and self.min_gain > 0):
            raise ValueError(f"trim_at/keep_items/min_gain must be positive: {self.trim_at}/{self.keep_items}/{self.min_gain}")

    def _log(self, event: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), **event}, ensure_ascii=False) + "\n")

    def apply(self, payload: dict) -> dict:
        """Rewrite one /v1/responses body (``input`` in place) and return the same dict."""
        items = payload.get("input")
        session = payload.get("session_id")
        with self.lock:
            self.stats["requests"] += 1
            if not isinstance(items, list):
                return payload
            if not session:
                # without a session_id the window cannot be identified; codex always sends one when
                # sglang_session_release is on
                self.stats["no_session"] += 1
                return payload
            w = self.windows.setdefault(session, Window())
            w.requests += 1
            if w.boundary:
                if len(items) < w.boundary or item_hash(items[w.boundary - 1]) != w.anchor:
                    self.stats["anchor_mismatch"] += 1
                    print(f"[reasoning_trim] ANCHOR MISMATCH session={session} boundary={w.boundary} "
                          f"items={len(items)}: history rewritten inside a window; restarting it", flush=True)
                    self._log({"event": "anchor_mismatch", "session": session, "boundary": w.boundary,
                               "items": len(items)})
                    w.boundary, w.anchor = 0, None
            target = len(items) - self.keep_items
            if w.context is not None and w.context >= self.trim_at and target > w.boundary:
                gain_chars = sum(reasoning_chars(it) for it in items[w.boundary:target]
                                 if it.get("type") == "reasoning")
                gain = int(gain_chars / self.chars_per_token)
                if gain >= self.min_gain:
                    old = w.boundary
                    w.boundary, w.anchor = target, item_hash(items[target - 1])
                    w.trims += 1
                    self.stats["trims"] += 1
                    self._log({"event": "trim", "session": session, "from": old, "to": target,
                               "items": len(items), "context": w.context, "est_gain_tokens": gain,
                               "window_trims": w.trims})
            if not w.boundary:
                return payload
            kept = [it for i, it in enumerate(items)
                    if not (i < w.boundary and isinstance(it, dict) and it.get("type") == "reasoning")]
            dropped = len(items) - len(kept)
            if dropped:
                self.stats["trimmed_requests"] += 1
                self.stats["reasoning_items_dropped"] += dropped
            payload["input"] = kept
            return payload

    def observe(self, session: str | None, usage: dict) -> None:
        """usage of an upstream response at completion (``response.completed``)."""
        if not session:
            return
        with self.lock:
            w = self.windows.get(session)
            if w is not None:
                w.context = int(usage["input_tokens"]) + int(usage["output_tokens"])

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.stats, "windows": len(self.windows)}


class UsageScanner:
    """Fish the usage of the terminal event out of an SSE byte stream (inspected while forwarding, the
    response is not buffered).

    ``response.incomplete`` must be recognised too: sglang sends incomplete for a response that hits the
    output cap, still with usage. Recognising only completed missed that response's usage entirely --
    in c0L4-r2 sk48, after a ~90k-token degenerate loop, context_window still used the previous 66k and
    let through a 166k-token request, above the server's 155k; every session after that game hit the same
    error with zero actions.
    The usage of ``response.failed`` may be None (generation aborted); that means no usage."""

    TERMINAL = ("response.completed", "response.incomplete", "response.failed")
    MARKS = tuple(f'"{t}"'.encode() for t in TERMINAL)

    def __init__(self) -> None:
        self.buffer = b""
        self.usage: dict | None = None
        self.status: str | None = None      # which terminal event

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        while b"\n\n" in self.buffer:
            event, self.buffer = self.buffer.split(b"\n\n", 1)
            if not any(mark in event for mark in self.MARKS):
                continue
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    data = json.loads(line[5:])
                    if data.get("type") in self.TERMINAL:
                        self.status = data["type"]
                        self.usage = data["response"]["usage"]
