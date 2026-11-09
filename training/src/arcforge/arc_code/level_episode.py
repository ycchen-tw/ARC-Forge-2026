"""Shared conventions for single-level episodes (ARC-AGI-3 RL, ``work/arc_code/run_arc3_level.py``).

An episode = play from the start of level k until it is cleared or a stop condition comes first. Stop
conditions are written by four parties into a host-side **stop directory** (not mounted into the jail, out of
the agent's reach), and readers only check "is there a file in the directory":

- middleware: ``cleared`` (level k cleared), ``action_cap`` (level k's action count reached the cap), ``start_failed``
  (the game did not open on level k);
- run_arc3_level.py: ``budget`` (output tokens counted by the proxy used up), ``context`` (harness config's
  ``context_token_limit``: one thread's context reached the cap and every later request to the proxy gets 503),
  ``window`` (harness config's ``stop_after_first_window``, RL v8: the first window-1 request reaches the proxy and is
  blocked with 503, so the episode ends at the first window change);
- run_arm.py: ``died`` (the codex session ended abnormally and it was not an infrastructure fault: the scope hit
  MemoryMax and was OOM-killed, or codex exited non-zero with no model-side error in the last turn). RL treats it as a
  failure: no resuming the same thread (resuming would leave a turn the server committed but codex did not record in
  the middle of the thread, which the rid join cannot handle); the episode ends here with reward 0.

Readers: the middleware (refuses all later actions), the model proxy (all later /responses get 503), run_arm (no
further sessions, the running session is ended), run_arc3_level.py (decides stop and reward). Each file is written
only once (O_EXCL): writing the same stop condition twice is a bug.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CLEARED = "cleared"
ACTION_CAP = "action_cap"
START_FAILED = "start_failed"
BUDGET = "budget"
CONTEXT = "context"
DIED = "died"
WINDOW = "window"
STOPS = (CLEARED, ACTION_CAP, START_FAILED, BUDGET, CONTEXT, DIED, WINDOW)


@dataclass(frozen=True)
class LevelEpisode:
    """The middleware's single-level mode. ``level`` is the 1-based level (at the opening ``levels_completed`` must be level−1);
    ``action_cap`` is the number of actions that may be counted on this level (including RESET); ``start_hash``, when given,
    must equal the opening screen's :func:`frame_hash`.

    ``prior_actions`` > 0 = the workspace was loaded with a prior-context seed (``level_seed.py``): levels 1..K−1 took this
    many actions in the real whole game, and the record already reaches that step. No step 0 is written at the opening (the
    opening screen must equal the seed's last step), and later actions are numbered on from prior_actions + 1; the
    ``actions_used`` the agent sees in ``/api/status`` is the total (as when a new session takes over in a whole game), while
    ``action_cap`` counts only this level.

    ``through`` = after opening at level K (usually with a prior-context seed) **keep playing to the end** (gen-2 seed
    collection, continued runs on hard levels, run_arm ``--game-seed``): no ``cleared`` on level clear, no action cap
    (``action_cap`` must be 0); ``start_failed`` is the only stop file left."""
    level: int
    action_cap: int
    stop_dir: Path
    start_hash: str | None = None
    prior_actions: int = 0
    through: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.level, int) and self.level >= 1):
            raise ValueError(f"level must be an int >= 1, got {self.level!r}")
        if not (isinstance(self.prior_actions, int) and not isinstance(self.prior_actions, bool)
                and self.prior_actions >= 0):
            raise ValueError(f"prior_actions must be an int >= 0, got {self.prior_actions!r}")
        if self.prior_actions and self.level < 2:
            raise ValueError(f"prior_actions {self.prior_actions} at level 1: there is nothing before level 1")
        if self.through:
            if self.action_cap != 0:
                raise ValueError(f"a through episode has no action cap (0), got {self.action_cap!r}")
        elif not (isinstance(self.action_cap, int) and self.action_cap >= 1):
            raise ValueError(f"action_cap must be an int >= 1, got {self.action_cap!r}")
        if not Path(self.stop_dir).is_dir():
            raise FileNotFoundError(f"stop_dir {self.stop_dir} is not a directory")


def frame_hash(frame) -> str:
    """Fingerprint of the official response's ``frame`` field (list of frames, int cells). The stub generator and the middleware use the same function."""
    return hashlib.sha256(json.dumps(frame, separators=(",", ":")).encode()).hexdigest()[:16]


def write_stop(stop_dir: Path, name: str, payload: dict) -> None:
    """Atomically write a stop file (temp file first, then link, so readers never see half a file). Raise if it already exists."""
    if name not in STOPS:
        raise ValueError(f"unknown stop {name!r}; one of {STOPS}")
    stop_dir = Path(stop_dir)
    tmp = stop_dir / f".{name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload) + "\n")
    try:
        os.link(tmp, stop_dir / name)      # FileExistsError if it was already written
    finally:
        tmp.unlink()


def stops(stop_dir: Path) -> dict[str, dict]:
    """Which stop files have been written so far (name -> content). Unknown file names raise: the stop directory holds only what is defined here."""
    out = {}
    for path in Path(stop_dir).iterdir():
        if path.name.startswith("."):
            continue                       # write_stop's temp file
        if path.name not in STOPS:
            raise RuntimeError(f"{path}: not a stop file ({STOPS})")
        out[path.name] = json.loads(path.read_text())
    return out
