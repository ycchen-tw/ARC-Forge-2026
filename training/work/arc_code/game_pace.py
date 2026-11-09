"""Whole-game pacing line (2026-10-27, arc3 RL v6 whole-game eval): hooks into ``sampling_proxy.py``'s
``/v1/responses`` (``--pace-run-dir <run_arm's run-dir> --pace-game-tokens <game_output_tokens>``) and writes
``.pace.json`` into each game's workspace, from which gen-2 act (run_arm's ``pace_line``, act_client's PACE_HOOK)
prints the last line "This level: X output tokens used of Y.".

In single-level RL (run_arc3_level's proxy, ``model_proxy.TaskProxy.write_pace``) X = the episode's cumulative
output and Y = the episode's output budget. A whole game has no episodes, so instead:
  - A level starts when the previous level's note_done passes (level 1 at game start): the same starting point as
    single-level RL (seeds are snapshotted when note_done passes, ``seed_note_done``). Output spent writing notes
    between the clear and note_done still counts toward the level just cleared (single-level RL stops at the clear).
  - Y = the game's remaining output budget when this level starts ÷ the levels left (this one included), fixed for
    the whole level; X = the game's output tokens since this level started.
The game is identified as in level_watch.py (the AGENTS.md header in the input); the level from the last row of
that game's ``record/history.jsonl`` and ``.note_gate/LNN.done`` (same as gen2 hooks/common.status). Writes happen
at the same points as TaskProxy: before each request goes upstream, and before its terminal event is forwarded to
codex (that write adds this response's output). Output tokens (usage.output_tokens, reasoning included) are
summed here: they are the same responses as run_arm's game_output_tokens (the rollout's token_usage_record); the
proxy lives and dies with the arm, so counting from 0 gives the whole game's count.
Proxy restarted midway (a game first seen not on level 1), game not identifiable, history missing or going
backwards, workspace not a plain directory: raise.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

#: act_client.PACE_FILE (the file name act reads)
PACE_FILE = ".pace.json"
#: gen2 hooks/common.GATE_DIR (act writes ``LNN.done`` when note_done passes)
GATE_DIR = ".note_gate"
#: the last row of history.jsonl is always within this many bytes of the end of the file
TAIL_BYTES = 1 << 16


class GamePace:
    def __init__(self, run_dir: Path, game_tokens: int, log_path: Path | None = None) -> None:
        if game_tokens <= 0:
            raise ValueError(f"--pace-game-tokens must be positive, not {game_tokens}")
        self.games = run_dir / "games"
        self.game_tokens = game_tokens
        self.log_path = log_path
        self.header = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(self.games)) + r"/([A-Za-z0-9_.-]+)")
        self.lock = threading.Lock()
        #: game → {"total" (whole-game output), "level" (level number, from 1), "start" (total when this level
        #: started), "budget" (Y)}
        self.state: dict[str, dict] = {}
        self.counts = {"requests": 0, "responses": 0, "writes": 0, "level_starts": 0}

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.counts, "games": {g: {"level": s["level"], "used": s["total"] - s["start"],
                                                 "budget": s["budget"], "total": s["total"]}
                                             for g, s in sorted(self.state.items())}}

    def _game(self, items: list) -> str:
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
        raise RuntimeError(f"pace: no 'AGENTS.md instructions for {self.games}/<game>' in the request input")

    def _level(self, game: str) -> tuple[int, int]:
        """(the level this game is on now, win_levels): after a clear it stays on the level just cleared until
        note_done passes; after WIN it stays on the last level."""
        ws = self.games / game
        path = ws / "record" / "history.jsonl"
        with path.open("rb") as handle:                 # missing = a request before the harness started the game: raise
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - TAIL_BYTES))
            lines = [line for line in handle.read().split(b"\n") if line.strip()]
        rows = []
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i not in (0, len(lines) - 1):        # first line read from mid-line, last line still being written
                    raise
        if not rows:
            raise RuntimeError(f"pace: {path} has no complete row")
        last = rows[-1]
        levels, win = int(last.get("levels_completed") or 0), last["win_levels"]
        if type(win) is not int or win < 1:
            raise RuntimeError(f"pace: {path} win_levels {win!r}")
        if last.get("state") == "WIN":
            return win, win
        if levels > 0 and not (ws / GATE_DIR / ("L%02d.done" % levels)).exists():
            return levels, win
        return levels + 1, win

    def _write(self, game: str, used: int, budget: int) -> None:
        """Same as model_proxy.TaskProxy.write_pace: O_EXCL|O_NOFOLLOW temp file + os.replace, never follows a
        symlink. Caller holds the lock."""
        ws = self.games / game
        if ws.is_symlink() or not ws.is_dir():
            raise NotADirectoryError(f"pace: {ws} is not a plain directory")
        body = json.dumps({"level_output_tokens": used, "level_output_budget": budget}).encode()
        tmp = ws / f".{PACE_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            if os.write(fd, body) != len(body):
                raise OSError(f"short write to {tmp}")
        finally:
            os.close(fd)
        os.replace(tmp, ws / PACE_FILE)
        self.counts["writes"] += 1

    def apply(self, payload: dict) -> str:
        """Before going upstream: identify the game and level (a level change fixes this level's Y), write the
        file. Returns the game, for ``observe``."""
        items = payload.get("input")
        game = self._game(items if isinstance(items, list) else [])
        level, win = self._level(game)
        with self.lock:
            self.counts["requests"] += 1
            s = self.state.get(game)
            if s is None and level != 1:
                raise RuntimeError(f"pace: first request of {game} is on level {level}: the proxy restarted mid-game "
                                   f"and lost the game's output count")
            if s is not None and level < s["level"]:
                raise RuntimeError(f"pace: {game} went from level {s['level']} back to {level}")
            if s is None or level != s["level"]:
                total = 0 if s is None else s["total"]
                left = win - level + 1
                # remaining budget ≤ 0: the game has spent the whole-game budget and run_arm's polling
                # (TOKEN_POLL_SECONDS) ends it right away; act needs Y > 0
                budget = max(1, (self.game_tokens - total) // left)
                s = self.state[game] = {"total": total, "level": level, "start": total, "budget": budget}
                self.counts["level_starts"] += 1
                if self.log_path is not None:
                    with self.log_path.open("a") as fh:
                        fh.write(json.dumps({"t": round(time.time(), 3), "game": game, "level": level,
                                             "win_levels": win, "start": total, "budget": budget}) + "\n")
            self._write(game, s["total"] - s["start"], s["budget"])
        return game

    def observe(self, game: str, usage: dict) -> None:
        """Before the response's terminal event is forwarded to codex: add its output, write the file (codex may
        run a tool call as soon as it receives it)."""
        with self.lock:
            s = self.state[game]
            s["total"] += int(usage["output_tokens"])
            self.counts["responses"] += 1
            self._write(game, s["total"] - s["start"], s["budget"])
