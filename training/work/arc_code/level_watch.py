"""Remind once when a level costs too much (duckx c0i7 / c0L5): hooks into ``sampling_proxy.py``'s
``/v1/responses`` (``--level-watch FIRST,FLOOR,F1,F2 --level-watch-run-dir <run_arm's run-dir>``).

Why: on the 50 levels Super Duck clears within 20k, c0i-r2's codex is 2.3× at the median and 3.02× in total -- the
total is pulled up by a few easy levels where it got stuck (sc25 L3 Duck 7k → 254k, ar25 L6 18k → 123k). Across
three arms, getting stuck is mostly random and the same level goes normally in other arms; on hard levels codex is
already ≤1.1× Duck, and a correct long derivation must not be interrupted. Hence a high threshold, at most twice per
level, and neutral wording (no "you are wrong").

How: each request identifies the game from the AGENTS.md header in the input and the level from
``levels_completed`` on the last line of that game's ``record/history.jsonl`` (same method as effort_schedule.py);
``observe`` sums this level's output tokens, and a level clear moves the level's cost into "earlier levels".
Thresholds: on the first level ``FIRST`` and 2×``FIRST``; afterwards ``max(F×median of earlier levels, FLOOR)``,
with F = F1, F2. On the first request past a threshold the reminder is appended after the latest tool output
(keyed by call_id), and every later request re-adds the same sentence on the same call_ids: the same history
always yields the same bytes, and the prefix cache only takes appends. A request whose game can't be identified is
left untouched and counted.
"""

from __future__ import annotations

import json
import re
import statistics
import threading
from pathlib import Path

OUTPUTS = ("function_call_output", "custom_tool_call_output")
NOTE = ("[harness] This level has now cost {k}k output tokens{rel}. Before your next move, write down in notes.md what "
        "you know for certain about this level and which of your assumptions you have not tested yet; then decide "
        "what to do next.")
REL = ", {x:.0f}x the median of the levels you cleared before"


def parse(text: str) -> tuple[int, int, float, float]:
    """``60000,40000,3,6`` → (FIRST, FLOOR, F1, F2)."""
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"--level-watch {text!r}: want FIRST,FLOOR,F1,F2 such as 60000,40000,3,6")
    first, floor, f1, f2 = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
    if not (first > 0 and floor > 0 and 1 < f1 < f2):
        raise ValueError(f"--level-watch {text!r}: FIRST and FLOOR must be positive and 1 < F1 < F2")
    return first, floor, f1, f2


class LevelWatch:
    def __init__(self, run_dir: Path, first: int, floor: int, f1: float, f2: float) -> None:
        self.games = run_dir / "games"
        self.first, self.floor, self.factors = first, floor, (f1, f2)
        self.header = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(self.games)) + r"/([A-Za-z0-9_.-]+)")
        self.lock = threading.Lock()
        # game → {"level", "tokens" (this level), "done" (costs of earlier levels), "fired" (reminders on this level),
        #       "pending" (sentence to append on the next request), "notes" (call_id → sentence)}
        self.state: dict[str, dict] = {}
        self.counts = {"requests": 0, "unknown_game": 0, "notes": 0, "level_changes": 0}

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.counts, "games": {g: {k: s[k] for k in ("level", "tokens", "fired")}
                                             for g, s in sorted(self.state.items())}}

    def _game(self, items: list) -> str | None:
        for item in items:
            if not isinstance(item, dict) or item.get("type", "message") != "message" or item.get("role") != "user":
                continue
            content = item.get("content")
            parts = content if isinstance(content, list) else [{"text": content}]
            for part in parts:
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str):
                    m = self.header.search(text)
                    if m:
                        return m.group(1)
        return None

    def _level(self, game: str) -> int:
        """levels_completed from the last line of record/history.jsonl; 0 if the game hasn't started (no record)."""
        path = self.games / game / "record" / "history.jsonl"
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            lines = [line for line in handle.read().splitlines() if line.strip()]
        return int(json.loads(lines[-1]).get("levels_completed") or 0) if lines else 0

    def threshold(self, s: dict) -> tuple[int, float | None] | None:
        """Threshold for this level's next reminder (tokens, multiple of the earlier levels' median or None); None
        once both reminders have fired."""
        n = s["fired"]
        if n >= len(self.factors):
            return None
        if not s["done"]:
            return self.first * (n + 1), None
        median = statistics.median(s["done"])
        return max(int(self.factors[n] * median), self.floor * (n + 1)), self.factors[n]

    def apply(self, payload: dict) -> tuple[str | None, int]:
        """Modify payload in place; return (game, reminders appended to this request (re-added ones included)). The
        game goes to ``observe``."""
        items = payload.get("input")
        items = items if isinstance(items, list) else []
        game = self._game(items)
        with self.lock:
            self.counts["requests"] += 1
            if game is None:
                self.counts["unknown_game"] += 1
                return None, 0
        level = self._level(game)
        with self.lock:
            s = self.state.setdefault(game, {"level": level, "tokens": 0, "done": [], "fired": 0, "pending": None,
                                             "notes": {}})
            if s["level"] != level:
                if level > s["level"]:
                    s["done"].append(s["tokens"])
                s.update(level=level, tokens=0, fired=0, pending=None)
                self.counts["level_changes"] += 1
            if s["pending"] is not None:
                last = max((i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") in OUTPUTS),
                           default=None)
                if last is not None:
                    s["notes"][items[last]["call_id"]] = s["pending"]
                    s["pending"] = None
                    self.counts["notes"] += 1
            notes = dict(s["notes"])
        n = 0
        for item in items:
            if isinstance(item, dict) and item.get("type") in OUTPUTS and item.get("call_id") in notes:
                note = notes[item["call_id"]]
                output = item.get("output")
                if isinstance(output, str):
                    item["output"] = f"{output}\n\n{note}"
                elif isinstance(output, list):
                    item["output"] = [*output, {"type": "input_text", "text": f"\n\n{note}"}]
                else:
                    continue
                n += 1
        return game, n

    def observe(self, game: str | None, usage: dict) -> None:
        """Response finished: add to this level's output tokens; past a threshold, queue a reminder for the next
        request."""
        if game is None:
            return
        with self.lock:
            s = self.state.get(game)
            if s is None:
                return
            s["tokens"] += int(usage.get("output_tokens") or 0)
            limit = self.threshold(s)
            if limit is not None and s["tokens"] >= limit[0] and s["pending"] is None:
                rel = "" if limit[1] is None else REL.format(x=s["tokens"] / statistics.median(s["done"]))
                s["pending"] = NOTE.format(k=s["tokens"] // 1000, rel=rel)
                s["fired"] += 1


#: Stuck-level reminder (RSI round 3 rf, ``--level-watch-stuck TOKENS,ACTIONS``): speaks only on "many tokens, few
#: actions". 8 runs of 800k, 1,238 levels: 27 levels had ≤10 actions when the level's output reached 200k, and only 9
#: of them were cleared afterwards (150k/≤10: 52 levels, 30 cleared); the sentence almost never fires on normal levels
#: (duckx's act_nudge nudged every 8 calls: more actions, lower score). The k-th time (k=1,2) the threshold is
#: k×TOKENS and the action cap k×ACTIONS; if actions already exceed the cap when the threshold is crossed, nothing is
#: said, and it is not made up later. Read 15 stuck cases in full: the stuck runs write down the key probe but never
#: run it (analysis/R3-read).
STUCK_NOTE = ("[harness] This level has cost {k}k output tokens and {n} game action(s). Levels that end up cleared "
              "rarely look like this: what breaks them open is a test on the board, not another pass of analysis. Look "
              "in notes.md and your plans for tests you wrote down and never ran and for objects you have never tried, "
              "and run the cheapest of them now.")
STUCK_SLOTS = 2
#: Probes written down in notes.md but not yet run (of the 17 stuck levels in the finished R3 rd run, 11 had such
#: sentences left in notes.md; s5i5 L7 even wrote "NEXT ./act do (decided, NOT yet sent)"): the reminder quotes them
#: verbatim, at most STUCK_QUOTES sentences, each cut to STUCK_QUOTE_CHARS.
UNTESTED = re.compile(r"never (tried|tested|clicked|pressed|used|run)|untested|untried|not (yet )?(tried|tested|sent|run)"
                      r"|to test:|next probes?", re.I)
ACTION_TOKEN = re.compile(r"\bACTION\d|\bRESET\b")
STUCK_QUOTES = 4
STUCK_QUOTE_CHARS = 300
STUCK_QUOTED = ("[harness] This level has cost {k}k output tokens and {n} game action(s). Your notes.md lists things "
                "you have not tested yet:\n{lines}\nA level that has cost this much is usually broken open by a test "
                "like these, tried on the board, not by another pass of analysis: run the cheapest of them now.")


def parse_stuck(text: str) -> tuple[int, int]:
    """``200000,10`` → (TOKENS, ACTIONS); ACTIONS ``-1`` ignores the action count and looks only at tokens."""
    parts = text.split(",")
    if len(parts) != 2 or not parts[0].strip().isdigit() or not (parts[1].strip().isdigit() or parts[1].strip() == "-1"):
        raise ValueError(f"--level-watch-stuck {text!r}: want TOKENS,ACTIONS such as 200000,10 (ACTIONS -1: no gate)")
    tokens, actions = int(parts[0]), int(parts[1])
    if tokens <= 0:
        raise ValueError(f"--level-watch-stuck {text!r}: TOKENS must be positive")
    return tokens, actions


class StuckWatch(LevelWatch):
    """Reuses LevelWatch's game/level identification, sentence appending and re-adding; only the threshold differs:
    when a level's output reaches k×TOKENS, the reminder is queued only if the level's actions (rows of record's
    history.jsonl whose ``level`` is this level) are ≤ k×ACTIONS."""

    def __init__(self, run_dir: Path, tokens: int, actions: int) -> None:
        super().__init__(run_dir, first=tokens, floor=tokens, f1=2.0, f2=3.0)
        self.stuck = (tokens, actions)
        self.counts["stuck_skipped"] = 0

    def _untested(self, game: str) -> list[str]:
        """Sentences in notes.md that mention probes not yet run (verbatim, markdown prefix stripped), at most
        STUCK_QUOTES."""
        path = self.games / game / "notes.md"
        if not path.exists():
            return []
        lines = path.read_text(errors="replace").splitlines()
        out = []
        for i, line in enumerate(lines):
            text = line.strip().lstrip("-*#> ").strip()
            if not (text and UNTESTED.search(text)):
                continue
            if not ACTION_TOKEN.search(text):
                # a heading like "NEXT ./act do (decided, NOT yet sent)": the command is on the next few lines, quote it too
                cmd = next((l.strip().strip("`") for l in lines[i + 1:i + 5] if "./act do " in l and ACTION_TOKEN.search(l)),
                           None)
                text = f"{text} {cmd}" if cmd else text
            text = text if len(text) <= STUCK_QUOTE_CHARS else text[:STUCK_QUOTE_CHARS - 3] + "..."
            if text not in out:
                out.append(text)
            if len(out) == STUCK_QUOTES:
                break
        return out

    def _actions(self, game: str, level: int) -> int:
        """Actions so far on level ``level`` (0-based, i.e. levels_completed)."""
        path = self.games / game / "record" / "history.jsonl"
        if not path.exists():
            return 0
        n = 0
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    n += row.get("level") == level + 1 and row.get("action") != "START"
        return n

    def observe(self, game: str | None, usage: dict) -> None:
        if game is None:
            return
        with self.lock:
            s = self.state.get(game)
            if s is None:
                return
            s["tokens"] += int(usage.get("output_tokens") or 0)
            k = s["fired"]
            if k >= STUCK_SLOTS or s["pending"] is not None or s["tokens"] < self.stuck[0] * (k + 1):
                return
            level, tokens = s["level"], s["tokens"]
        n = self._actions(game, level)
        lines = self._untested(game)
        with self.lock:
            if s["level"] != level or s["fired"] != k:
                return
            s["fired"] += 1
            if self.stuck[1] < 0 or n <= self.stuck[1] * (k + 1):
                s["pending"] = (STUCK_QUOTED.format(k=tokens // 1000, n=n, lines="\n".join(f"  - {l}" for l in lines))
                                if lines else STUCK_NOTE.format(k=tokens // 1000, n=n))
                self.counts["stuck_quoted"] = self.counts.get("stuck_quoted", 0) + bool(lines)
            else:
                self.counts["stuck_skipped"] += 1

