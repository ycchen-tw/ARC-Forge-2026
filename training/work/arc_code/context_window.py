"""Keep the context from growing to the point where codex compacts it itself: an option of sampling_proxy
(``--window-at HIGH --window-keep LOW``, duckx c0i6).

Why: by 250k tokens per game in c0i-r2 / c0L-r1, action gaps that span a compaction already take 40% of the output
(gaps.py) -- the summary itself, re-orienting after compaction, re-reading record/. Super Duck doesn't compact: it
drops the oldest part, and the recent raw text is always there. codex decides whether to compact from the usage the
server reports (k2 with reasoning trim went from 170 to 80 compactions), so if the proxy shortens the input it sends,
codex doesn't compact.

How: each codex window (the body's ``session_id``) keeps one **cut point** = the ``call_id`` of some tool output. The
leading developer/user messages (AGENTS.md, kickoff; plus the summary if it was compacted) are sent as is; every item
after them up to the cut point is not sent and is replaced by one fixed user message (``NOTE``: how many tool calls
were cut + the content of ``notes.md`` at the time of the cut + which files levels/ has); everything after the cut
point is unchanged. When the context the previous response reported (input + output tokens) passes ``high``, the cut
point jumps forward once, to the earliest clean position where "the estimate of what remains ≤ ``keep`` tokens", and
then stays put: the prefix is the same every time, the prefix cache hits as usual, and only the jump re-prefills.
The cut point is a call_id rather than an index, so the reasoning trim that runs before this moving its boundary
(dropping reasoning shifts indices) doesn't affect it either.

Clean cut point: right after a tool output, and every call in the dropped part has its output in the dropped part
too (parallel calls aren't split from their outputs); the last tool output is always kept. Token estimate: text
chars ÷ ``chars_per_token``, ``IMAGE_TOKENS`` per image (input_image or an act_images marker), with the ratio then
calibrated by matching the previous actually-sent estimate against usage.

Contract:
  - When usage is missing or stale (upstream sent no usage, the request was rejected, before 2026-10-12 the proxy
    didn't recognise response.incomplete): if this request's own estimate (times the last measured input/estimate
    ratio) passes ``hard`` (``high`` if unset) while the last usage is still below it, cut on the estimate without
    waiting for notes, record ``stale_usage`` and print it loudly. When usage is fresh, this at most makes the
    notes-waiting cut one tool output earlier. The ratio exists only after this session has been measured once: the
    first request is the kickoff, which is short; the proxy starts only once per arm.
    c0L4-r2 sk48: a response that hit the output cap had no usage, the window let a 166k request through on 66k,
    above the server's 155k, and after that every session hit the same error with zero actions.
  - If the cut point's call_id isn't in the input (codex rewrote the history within the same window), no guessing:
    record ``anchor_mismatch``, restart this window from scratch, and print it loudly.
  - If the game can't be identified (AGENTS.md header) or notes.md can't be read, cut anyway; the message says "no
    snapshot, read it yourself"; count it and print it.
  - If the jump finds no clean cut point, don't cut; record ``no_cut``.

Waiting for notes (``notes_after``/``hard``; duckx 2026-10-11 audit: in 19% of c0i-r2's 195 compactions notes had not
been written since the previous compaction, and the reminder didn't help; 70% of the facts that were actually used
after compaction but missing from notes came from these "notes lagging" compactions): when context first passes
``notes_after``, record the time and send a reminder; when a cut is due at ``high`` and notes.md hasn't been written
since then, hold off (record ``postponed``) and cut once it is written, and always cut by ``hard`` at the latest. The
cut-point message also states how many actions notes.md lags behind the game (record/history.jsonl's time vs
notes.md's mtime).

The reminder is also sent from here (``REMINDER``, appended after the latest tool output of the first request past
``notes_after``, keyed by call_id, re-added every time) rather than through codex's pre_compact_reminder: that one is a
user message, and codex 0.154, when ``server_reasoning_included`` is false (always so for a local provider),
re-estimates every reasoning item with encrypted_content "before the last user message" (len×3/4−650 each) and adds
it to the server's usage (core/src/context_manager/history.rs ``get_total_token_usage``). Normally the only user
message is at the start and this term is 0; once the reminder is inserted it balloons and codex compacts -- 192 of
c0i-r2's 203 compactions came after the reminder, at a median of 2 responses, which is also why c0i6-r1's window
couldn't hold it off.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from reasoning_trim import reasoning_chars

CALLS = ("function_call", "custom_tool_call")
OUTPUTS = ("function_call_output", "custom_tool_call_output")
HEAD_ROLES = ("developer", "system", "user")
MARKER = re.compile(r"\[\[arc-image:(board|diff):[^\]\s]*\]\]")
#: a 640px board image is ~400 tokens on this model (preflight_images.py: two images +800)
IMAGE_TOKENS = 400
NOTE = ("[harness] Your context is kept short, so the oldest part of this conversation is no longer shown: "
        "{calls} of your earlier tool calls were dropped here. Nothing on disk changed -- notes.md, levels/, lib/ "
        "and scripts/ are as you left them, and record/ has every step. The calls after this message are your most "
        "recent ones, unchanged; carry on from them.\n\n{notes}")
NOTES = ("Your notes.md when this cut was made ({age}):\n\n--- notes.md ---\n{text}\n--- end of notes.md ---\n\n"
         "levels/: {levels}")
AGE = "last written {n} game action(s) ago, at step {step}; the game is now at step {now}"
FRESH = "written after your last game action"

NO_NOTES = "(No snapshot of notes.md could be taken: {why}. Read it yourself if you need it.)"
REMINDER = ("[harness] Your context is nearly full: soon the oldest part of this conversation will be dropped (your "
            "most recent commands stay). Before your next game action, bring notes.md up to date -- what you now "
            "believe, what you have ruled out, what you are in the middle of -- and correct anything in notes.md or "
            "levels/ that you now know is wrong.")


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(p.get("text", "") for p in value if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _images(value) -> int:
    if isinstance(value, list):
        return sum(1 for p in value if isinstance(p, dict) and p.get("type") in ("input_image", "image_url", "video_url"))
    return 0


def estimate(item: dict, chars_per_token: float) -> float:
    """Token estimate for one input item: text chars ÷ chars_per_token, images counted separately."""
    if not isinstance(item, dict):
        return 0.0
    kind = item.get("type", "message")
    if kind in OUTPUTS:
        body = item.get("output")
    elif kind == "message":
        body = item.get("content")
    elif kind == "reasoning":
        # codex's summary and content are the same text; sglang sends only one (c0i6-s1 summed both, cut at 40k not 60k)
        return reasoning_chars(item) / chars_per_token
    elif kind in CALLS:
        body = item.get("arguments") if kind == "function_call" else item.get("input")
    else:
        body = json.dumps(item)
    text = _text(body)
    return len(text) / chars_per_token + IMAGE_TOKENS * (_images(body) + len(MARKER.findall(text)))


def head_end(items: list) -> int:
    """Where the leading run of developer/system/user messages ends (AGENTS.md, kickoff, compaction summary)."""
    for i, item in enumerate(items):
        if not (isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") in HEAD_ROLES):
            return i
    return len(items)


@dataclass
class Window:
    cut_call: str | None = None
    note: str | None = None
    context: int | None = None      # the previous response's input_tokens + output_tokens
    sent: float | None = None       # estimate of the input actually sent last time (for the calibration ratio)
    calib: float | None = None      # the previous response's measured input_tokens ÷ the estimate sent then
    jumps: int = 0
    armed_at: float | None = None   # time context first passed notes_after (reset after the cut)
    postponed: bool = False         # postponement already recorded this round
    remind_pending: bool = False    # armed: the next request gets the reminder
    remind_calls: list = field(default_factory=list)   # tool outputs (call_id) carrying the reminder, re-added every time


@dataclass
class ContextWindow:
    run_dir: Path
    high: int
    keep: int
    notes_chars: int = 8000
    notes_after: int | None = None
    hard: int | None = None
    chars_per_token: float = 3.0
    log_path: Path | None = None
    windows: dict[str, Window] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stats: dict[str, int] = field(default_factory=lambda: {
        "requests": 0, "cut_requests": 0, "jumps": 0, "anchor_mismatch": 0, "no_cut": 0, "no_session": 0,
        "no_notes": 0, "items_dropped": 0, "postponed": 0, "forced": 0, "reminders": 0, "stale_usage": 0})

    def __post_init__(self) -> None:
        if not 0 < self.keep < self.high:
            raise ValueError(f"--window-keep {self.keep} must be positive and less than --window-at {self.high}")
        if (self.notes_after is None) != (self.hard is None):
            raise ValueError("--window-notes-after and --window-hard must be given together")
        if self.notes_after is not None and not self.keep < self.notes_after < self.high <= self.hard:
            raise ValueError(f"need keep {self.keep} < notes_after {self.notes_after} < high {self.high} <= hard {self.hard}")
        self.games = self.run_dir / "games"
        self.header = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(self.games)) + r"/([A-Za-z0-9_.-]+)")

    def _log(self, event: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), **event}, ensure_ascii=False) + "\n")

    def _game(self, head: list) -> str | None:
        for item in head:
            if item.get("role") == "user":
                m = self.header.search(_text(item.get("content")))
                if m:
                    return m.group(1)
        return None

    def _notes(self, game: str | None) -> tuple[str, str | None]:
        """Snapshot of notes.md at the moment of the cut; if it can't be read, returns (explanation, reason)."""
        if game is None:
            return NO_NOTES.format(why="the game was not recognised"), "unknown game"
        ws = self.games / game
        path = ws / "notes.md"
        if not path.is_file():
            return NO_NOTES.format(why="there was no notes.md yet"), "no notes.md"
        text = path.read_text(errors="replace")
        if len(text) > self.notes_chars:
            half = self.notes_chars // 2
            text = f"{text[:half]}\n[... {len(text) - 2 * half} characters of notes.md not shown ...]\n{text[-half:]}"
        levels = sorted(p.name for p in (ws / "levels").glob("*")) if (ws / "levels").is_dir() else []
        return NOTES.format(age=self._age(ws, path.stat().st_mtime), text=text.rstrip(),
                            levels=", ".join(levels) or "(none yet)"), None

    @staticmethod
    def _age(ws: Path, mtime: float) -> str:
        """How many actions notes.md lags behind the game: rows of record/history.jsonl timed after mtime (START
        excluded)."""
        history = ws / "record" / "history.jsonl"
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()] if history.is_file() else []
        rows = [r for r in rows if r.get("action") != "START"]
        later = [r for r in rows if r["time"] > mtime]
        if not later:
            return FRESH
        before = [r for r in rows if r["time"] <= mtime]
        return AGE.format(n=len(later), step=before[-1]["step"] if before else 0, now=rows[-1]["step"])

    def _notes_mtime(self, game: str | None) -> float | None:
        path = None if game is None else self.games / game / "notes.md"
        return path.stat().st_mtime if path is not None and path.is_file() else None

    def _cut(self, items: list, h: int, start: int, ratio: float, fixed: float) -> int | None:
        """Where to jump: items[h:p] is cut, items[p:] kept. p sits right after a tool output, doesn't split parallel
        calls and keeps the last output; take the earliest p where "the estimate of what remains ≤ keep"; failing
        that, the last clean position; if there is none at all, return None.
        ``fixed``: estimate of what is sent every time but isn't in input (instructions, tools, cut-point message)."""
        last_output = max((i for i in range(len(items)) if items[i].get("type") in OUTPUTS), default=None)
        if last_output is None:
            return None
        head_tokens = (sum(estimate(it, self.chars_per_token) for it in items[:h]) + fixed) * ratio
        suffix = [0.0] * (len(items) + 1)
        for i in range(len(items) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + estimate(items[i], self.chars_per_token) * ratio
        open_calls: set = set()
        best = None
        for p in range(h + 1, last_output + 1):
            item = items[p - 1]
            kind = item.get("type")
            if kind in CALLS:
                open_calls.add(item.get("call_id"))
            elif kind in OUTPUTS:
                open_calls.discard(item.get("call_id"))
            if p <= start or kind not in OUTPUTS or open_calls:
                continue
            best = p
            if head_tokens + suffix[p] <= self.keep:
                return p
        return best

    def apply(self, payload: dict) -> int:
        """Modify payload in place; return how many items this request cut."""
        items = payload.get("input")
        session = payload.get("session_id")
        with self.lock:
            self.stats["requests"] += 1
            if not isinstance(items, list):
                return 0
            if not session:
                self.stats["no_session"] += 1
                return 0
            w = self.windows.setdefault(session, Window())
            h = head_end(items)
            at = None
            if w.cut_call is not None:
                at = next((i for i in range(h, len(items)) if items[i].get("type") in OUTPUTS
                           and items[i].get("call_id") == w.cut_call), None)
                if at is None:
                    self.stats["anchor_mismatch"] += 1
                    print(f"[context_window] ANCHOR MISMATCH session={session} cut_call={w.cut_call} "
                          f"items={len(items)}: history rewritten inside a window; restarting it", flush=True)
                    self._log({"event": "anchor_mismatch", "session": session, "cut_call": w.cut_call,
                               "items": len(items)})
                    w.cut_call = w.note = None
            overhead = len(json.dumps([payload.get("instructions"), payload.get("tools")])) / self.chars_per_token
            context, ratio = w.context, (w.context / w.sent if w.context is not None and w.sent else None)
            # how big this request is when sent through the current window; relied on when usage is missing or stale
            # (the docstring's contract)
            sending = items if at is None else items[:h] + items[at + 1:]
            estimated = (sum(estimate(it, self.chars_per_token) for it in sending) + overhead
                         + (len(w.note) / self.chars_per_token if at is not None else 0.0)) * (w.calib or 1.0)
            ceiling = self.hard if self.hard is not None else self.high
            if w.calib is not None and estimated >= ceiling and (context is None or context < ceiling):
                self.stats["stale_usage"] += 1
                print(f"[context_window] STALE USAGE session={session}: this request is ~{estimated:.0f} tokens, "
                      f"over {ceiling}, but the last usage said {context}; cutting on the estimate", flush=True)
                self._log({"event": "stale_usage", "session": session, "items": len(items), "context": context,
                           "estimated": round(estimated)})
                context, ratio = int(estimated), w.calib
            if context is not None and context >= self.high:
                total = sum(estimate(it, self.chars_per_token) for it in items) + overhead
                ratio = ratio if ratio is not None else 1.0
                p = self._cut(items, h, (at + 1) if at is not None else h, ratio,
                              overhead + self.notes_chars / self.chars_per_token)
                game = self._game(items[:h])
                waiting = False
                if self.notes_after is not None and w.armed_at is not None:
                    mtime = self._notes_mtime(game)
                    waiting = mtime is None or mtime < w.armed_at
                if p is not None and waiting and context < self.hard:
                    if not w.postponed:
                        w.postponed = True
                        self.stats["postponed"] += 1
                        self._log({"event": "postponed", "session": session, "game": game, "context": context,
                                   "armed_at": w.armed_at})
                    p = None
                elif p is None:
                    self.stats["no_cut"] += 1
                    self._log({"event": "no_cut", "session": session, "items": len(items), "context": context})
                else:
                    if waiting:
                        self.stats["forced"] += 1
                    notes, why = self._notes(game)
                    if why is not None:
                        self.stats["no_notes"] += 1
                        print(f"[context_window] no notes.md snapshot for session={session} game={game}: {why}",
                              flush=True)
                    calls = sum(1 for it in items[h:p] if it.get("type") in CALLS)
                    w.cut_call, at = items[p - 1]["call_id"], p - 1
                    w.note = NOTE.format(calls=calls, notes=notes)
                    w.jumps += 1
                    # the old reminder says "about to cut", which is wrong if it stays in the window after the cut;
                    # everything after the cut point is re-prefilled anyway, so removing it costs nothing extra
                    w.armed_at, w.postponed = None, False
                    w.remind_calls.clear()
                    self.stats["jumps"] += 1
                    self._log({"event": "cut", "session": session, "game": game, "items": len(items), "head": h,
                               "to": p, "calls_dropped": calls, "context": context, "ratio": round(ratio, 3),
                               "est_total": round(total * ratio), "window_jumps": w.jumps,
                               "notes": why or "snapshot", "forced": waiting})
                # after one jump, wait for the next usage: context is from before the cut and can't drive another jump
                w.context = None
            if w.cut_call is None:
                self._remind(session, w, items)
                w.sent = sum(estimate(it, self.chars_per_token) for it in items) + overhead
                return 0
            note = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": w.note}]}
            kept = items[:h] + [note] + items[at + 1:]
            dropped = len(items) - len(kept) + 1
            self.stats["cut_requests"] += 1
            self.stats["items_dropped"] += dropped
            self._remind(session, w, kept)
            w.sent = sum(estimate(it, self.chars_per_token) for it in kept) + overhead
            payload["input"] = kept
            return dropped

    def _remind(self, session: str, w: Window, items: list) -> None:
        """First request after arming: the reminder is appended after the latest tool output; every later request
        re-adds the same sentence on the same call_ids."""
        if w.remind_pending:
            last = max((i for i, it in enumerate(items) if it.get("type") in OUTPUTS), default=None)
            if last is not None:
                w.remind_calls.append(items[last]["call_id"])
                w.remind_pending = False
                self.stats["reminders"] += 1
                self._log({"event": "reminder", "session": session, "call": items[last]["call_id"],
                           "context": w.context})
        if not w.remind_calls:
            return
        for item in items:
            if item.get("type") in OUTPUTS and item.get("call_id") in w.remind_calls:
                output = item.get("output")
                if isinstance(output, str):
                    item["output"] = f"{output}\n\n{REMINDER}"
                elif isinstance(output, list):
                    item["output"] = [*output, {"type": "input_text", "text": f"\n\n{REMINDER}"}]

    def observe(self, session: str | None, usage: dict) -> None:
        """usage at the end of the upstream response (completed or incomplete)."""
        if not session:
            return
        with self.lock:
            w = self.windows.get(session)
            if w is not None:
                w.context = int(usage["input_tokens"]) + int(usage["output_tokens"])
                if w.sent:
                    w.calib = int(usage["input_tokens"]) / w.sent
                if self.notes_after is not None and w.armed_at is None and w.context >= self.notes_after:
                    w.armed_at, w.remind_pending = time.time(), True

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.stats, "windows": len(self.windows)}
