"""Append-only action record.

The definition is **not** "the sole scoring input" but "the most authoritative record on this
machine, labelled with which kind it is" -- in a competition rerun the authority is the organiser's
card and ours is merely an observation. A scorer must not silently mix different authorities.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

#: Written by our own gateway; usable for scoring.
LOCAL_GATEWAY = "local-gateway"
#: The card the organiser closed. The only score source in a competition rerun.
OFFICIAL_SCORECARD = "official-scorecard"
#: Only the agent's own logs.txt. Misses actions that bypass act.py; not usable for scoring.
AGENT_LOG = "agent-log"
#: Actions the middleware observed being delivered to the official gateway. A convenience record; the authority is the official card.
MIDDLEWARE = "middleware"

AUTHORITIES = (LOCAL_GATEWAY, OFFICIAL_SCORECARD, AGENT_LOG, MIDDLEWARE)


class Ledger:
    """One per run. Opened ``0600`` because under fence_posix all games share one filesystem."""

    def __init__(self, path: Path, *, authority: str = LOCAL_GATEWAY) -> None:
        if authority not in AUTHORITIES:
            raise ValueError(f"unknown authority {authority!r}")
        self.path = path
        self.authority = authority
        self._lock = threading.Lock()
        self._seq = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        os.chmod(path, 0o600)

    def record(self, *, game: str, action: str, counted: bool, play: int,
               guid: str | None, card_id: str | None, obs: Any = None,
               via: str = "unknown", x: int | None = None,
               y: int | None = None) -> dict[str, Any]:
        """``via`` is **informational only**: the agent can forge User-Agent, and that is fine --
        both act.py and direct connections are legitimate; this field only answers "how often does the agent write its own client"."""
        with self._lock:
            self._seq += 1
            row: dict[str, Any] = {
                "seq": self._seq,
                "unix": time.time(),
                "game": game,
                "card_id": card_id,
                "guid": guid,
                "action": action,
                "play": play,
                "counted": counted,
                "via": via,
                "authority": self.authority,
                "levels_completed": int(getattr(obs, "levels_completed", 0) or 0),
                "win_levels": int(getattr(obs, "win_levels", 0) or 0),
                "state": getattr(getattr(obs, "state", None), "name", None),
            }
            if x is not None or y is not None:
                row["x"], row["y"] = x, y
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return row

    def rows(self) -> list[dict[str, Any]]:
        text = self.path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
