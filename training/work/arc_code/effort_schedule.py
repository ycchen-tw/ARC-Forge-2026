"""Set reasoning effort from the output tokens already spent on the current level: hooks into
``sampling_proxy.py``'s ``/v1/responses`` (``--effort-schedule medium,xhigh,30000 --effort-run-dir <run_arm's
run-dir>``).

Why: c0L-r1 cut the calls between actions to 0.78×, yet tokens stayed at 0.99× -- total reasoning did not
change (78–80% of every response is reasoning). Super Duck is also xhigh; it saves elsewhere. To start fast on
the output-token clock, easy levels have to think shallow. Hard levels (the ones that spend more) go back up to
xhigh, so their later part runs at Codex's depth.

How: each request identifies the game from the AGENTS.md header in the input (``# AGENTS.md instructions for
<run>/games/<game>``, which codex re-injects after compaction) and reads ``levels_completed`` from the last line
of that game's ``record/history.jsonl`` -- when the level changes, the game's counter resets to zero. The proxy
keeps the counter itself: ``usage.output_tokens`` of each response on this level (``observe``). Counter <
threshold sends the low effort, otherwise the high one. A request whose game can't be identified is left
untouched and counted; no guessing.

Cost: Qwen3.8's template writes effort as a sentence at the start of system (medium = nothing written), so the
request that switches effort recomputes the whole prefix; that costs only prefill, not output tokens. At most two
switches per level (back to low after a level clear, up to high after spending a lot).

Result (2026-10-11, never run as a formal run): first measured whole-game fixed medium (c0Lm-p1, 8 games at 400k
each) against c0L-r1's xhigh -- 47 same-level pairs, per-level tokens medium/xhigh 1.22× (95% CI 0.97–1.54),
levels cleared within 200k tokens 30 vs 37. Thinking shallow does not save; the premise does not hold.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh")


def parse(text: str) -> tuple[str, str, int]:
    """``medium,xhigh,30000`` → (low, high, threshold)."""
    parts = text.split(",")
    if len(parts) != 3 or parts[0] not in EFFORTS or parts[1] not in EFFORTS or not parts[2].isdigit():
        raise ValueError(f"--effort-schedule {text!r}: want LOW,HIGH,TOKENS such as medium,xhigh,30000")
    return parts[0], parts[1], int(parts[2])


class EffortSchedule:
    def __init__(self, run_dir: Path, low: str, high: str, level_tokens: int) -> None:
        if level_tokens <= 0:
            raise ValueError(f"effort schedule threshold {level_tokens} must be positive")
        self.games = run_dir / "games"
        self.low, self.high, self.level_tokens = low, high, level_tokens
        self.header = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(self.games)) + r"/([A-Za-z0-9_.-]+)")
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}     # game → {"level": levels cleared, "tokens": output tokens on this level}
        self.counts = {"requests": 0, low: 0, high: 0, "unknown_game": 0, "level_resets": 0}

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.counts, "games": {g: dict(s) for g, s in sorted(self.state.items())}}

    def _game(self, payload: dict) -> str | None:
        items = payload.get("input")
        for item in items if isinstance(items, list) else []:
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

    def apply(self, payload: dict) -> tuple[str | None, str | None]:
        """Set ``payload["reasoning"]["effort"]`` in place; return (game, effort), or (None, None) if the game
        can't be identified."""
        game = self._game(payload)
        with self.lock:
            self.counts["requests"] += 1
            if game is None:
                self.counts["unknown_game"] += 1
                return None, None
        level = self._level(game)
        with self.lock:
            state = self.state.setdefault(game, {"level": level, "tokens": 0})
            if state["level"] != level:
                state.update(level=level, tokens=0)
                self.counts["level_resets"] += 1
            effort = self.low if state["tokens"] < self.level_tokens else self.high
            self.counts[effort] += 1
        reasoning = payload.get("reasoning")
        payload["reasoning"] = {**(reasoning if isinstance(reasoning, dict) else {}), "effort": effort}
        return game, effort

    def observe(self, game: str | None, usage: dict) -> None:
        if game is None:
            return
        with self.lock:
            self.state[game]["tokens"] += int(usage.get("output_tokens") or 0)
