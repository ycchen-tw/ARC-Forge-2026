"""The layer between the agent and the official gateway.

    agent ──(our interface)──► middleware ──(official API, with the competition key)──► official gateway

The official gateway owns game semantics and scoring and is **not changed by a single line**
(``official.py``). This layer only does what the official one lacks and we need, and **its interface to
the agent is ours to design** -- this layer is deployed both locally and on Kaggle, so what the agent sees
is the same in both places and no longer needs to match the official wire.

1. **Per-game keys**. The key the agent gets maps to a single game; the middleware swaps it for the
   competition key before forwarding, and a game_id mismatch gets 403. The official server has no such
   concept (one key owns the whole card), and with 25 jails in parallel agents really do hit other games
   (280 times measured in the four-arm run).
2. **The opening RESET, which is also the only one, is done by the middleware on the agent's behalf**; the
   middleware holds the guid and injects it into every action. The agent no longer needs to know the guid
   exists -- the whole class of "lost guid -> permanent 400" failures (three games in the four-arm run, arm
   b's 3.5-million-RESET storm) can no longer happen. The guid is persisted to ``--state``, so restarting
   the middleware does not kill a game.
3. **Illegal actions are blocked here**: an action not in ``available_actions``, or an ``ACTION6`` missing
   coordinates or outside 0..63, gets a 400 with a clear explanation instead of hitting the gateway.
4. **Rate limiting**: one token bucket per key; exceeding it gets 429 + ``Retry-After``.
5. ``GET /api/status`` returns the current state **for free** (level, available actions, actions used,
   board), so the agent need not spend a real action asking "where am I".
6. With ``--record-dirs``, every action's response is written into that game's ``record/`` (``record.py``):
   board text, images, all frames, animation. Every client's actions pass through here, so the record is complete.
7. With ``--image-ref-key`` (``act_image``): every step's ``record`` and ``/api/status`` carry an extra
   ``image_ref`` (``image_ref.py``); act.py prints it as a marker and the model proxy swaps in that step's image.
8. With ``--level-gates`` (prompts v9): after a level is cleared, the next action waits until the workspace has
   ``levels/Lnn.md`` (long enough, citing ``record/steps/``); then ``compact_request`` is placed in the driver
   home and codex (patch 0004) compacts at the next turn boundary. Both times the answer is 409 and no action
   is spent -- write the doc first, then compact, then play the next level; the order is fixed.
9. With ``--level-episode K`` (single-level RL episode, ``level_episode.py``): the game must open exactly on
   level K (the gateway places it with ``official.py --start-level``; on mismatch write ``start_failed`` and
   return 500); once level K is cleared write ``cleared``, once the action count reaches ``--action-cap`` write
   ``action_cap``, and after that every action gets 409 without spending an action. The stop files are in
   ``--stop-dir`` (host side); the proxy and run_arm read the same directory to end the whole episode.
   Additionally giving ``--prior-actions N`` = the workspace was loaded with a prior-context seed (the real
   workspace at the start of level K, ``level_seed.py``): the launcher opens the game on the agent's behalf
   (the agent's key gets 409 until then), the opening screen must equal the seed record's last board and the
   record must end exactly at step N (otherwise ``start_failed``), and no step 0 is written at the opening;
   later actions are numbered on from N+1, ``actions_used`` in ``/api/status`` is the total including the
   first K−1 levels, and the action cap counts only this level.
10. With ``--seed-snapshots DIR`` (gen-2 seed collection, ``seed_snapshot.py``): the level-change snapshot
   ``DIR/<game dir>/LKK`` (K≥2) is taken inside that game's lock, at the point set by ``--level-snapshot-at``:
   ``first_action`` (default, collections since 2026-10-21) = before level K's first counted action is sent
   to the official server; ``note_done`` (since 2026-10-24) = after act's ``note_done`` passes it calls
   ``POST /api/snapshot {"level_done": K-1}``, when level K has no actions yet.
   On 2026-10-23 we found 64% of first_action seeds carried prework on level K (after note_done the model can
   already see level K and can solve it offline), some carrying the answer outright; snapshotting at note_done
   leaves only what was glimpsed in passing while writing the previous level's notes. In note_done mode, if
   there is still no LKK at level K's first action (note_done was not used) it is taken then as before, with
   meta recording ``at: first_action_fallback``. ``POST /api/snapshot {"window": n, ...}`` (called with this
   game's key by the gen-2 SessionStart hook) takes the window-change ``Wnnn``. A failed snapshot raises: 500
   is returned (no action spent) and the error is written to ``DIR/ERRORS.jsonl`` (the collection monitor
   stops when it sees it).
11. ``--stop-dir`` alone (without ``--level-episode``; whole-game RL episode, run_arc3_level.py's arc3_game):
   once any stop file appears in that directory (the proxy's ``budget`` / ``context``, run_arm's ``died``),
   every action gets 409 without spending an action -- without waiting for run_arm's 15-second poll to end
   the session (model-free scripts running during that window could still send actions and change the score
   of a stop=budget game; 2026-10-24 review).

Scoring is **not here**: the score is the card the official ``close_scorecard()`` returns. The
``ledger.jsonl`` here is only a convenience record (timestamps, via act or direct), not the authority.

    python -m arcforge.arc_code.middleware \\
        --upstream http://127.0.0.1:18801 --upstream-key <competition key> \\
        --tokens <run>/gateway/tokens --port 18733 --state <run>/middleware_state.json \\
        --open-card --card-out <run>/card.json --scorecard-out <run>/scorecard_official.json
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import image_ref
from . import level_episode as episode_stops
from . import seed_snapshot
from .ledger import MIDDLEWARE, Ledger
from .level_episode import LevelEpisode
from .record import GameRecord, RecordError, frames_array
from .score import game_score
from .tokens import TokenTable

ACTIONS: dict[str, int] = {"RESET": 0, **{f"ACTION{i}": i for i in range(1, 8)}}


class _Server(ThreadingHTTPServer):
    """socketserver defaults to listen(5). The gen-2 harness opens every game at the same moment (Kaggle: 110 games): the backlog
    fills, syncookies make the client think it connected, and it only gets an RST after sending its request (2026-10-20 v29g: 5 games
    died of ConnectionResetError before play)."""
    request_queue_size = 1024
COORD_ACTION = "ACTION6"
GRID = 64
MAX_BODY = 1 << 20
MAX_PLAN = 4000
MAX_DOC = 1 << 20


class Fault(Exception):
    """An error to return to the agent. Keeps the official ``{error, message}`` shape, which the agent understands at a glance."""

    def __init__(self, status: int, error: str, message: str,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(f"{status} {error}: {message}")
        self.status, self.error, self.message = status, error, message
        self.headers = headers or {}

    def body(self) -> dict[str, str]:
        return {"error": self.error, "message": self.message}


class Upstream:
    """HTTP client for the official gateway. The key is the competition key; the agent never sees it."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # One cookie jar for the whole upstream connection. The official server never checks cookies, but the official client
        # (remote_wrapper.py) always carries a jar -- it is for AWS ALB stickiness -- and on Kaggle every submission that got a
        # score (the old arm's act.py, Vista) carried it. Nobody has measured whether anything in front of the rerun's sidecar
        # looks at cookies, so follow the path that got scores.
        self.cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies))

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any, str]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("X-API-Key", self.api_key)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                raw, ctype = r.read(), (r.headers.get("Content-Type") or "")
                status = r.status
        except urllib.error.HTTPError as exc:
            raw, ctype, status = exc.read(), (exc.headers.get("Content-Type") or ""), exc.code
        except OSError as exc:
            # A vanished upstream must be **loud**: the bridge layer used to print not a word in this case,
            # and the agent only saw RemoteDisconnected and retried frantically.
            raise Fault(502, "UPSTREAM_ERROR",
                        f"official gateway unreachable: {exc}") from exc
        if "json" in ctype:
            try:
                return status, json.loads(raw or b"null"), ctype
            except json.JSONDecodeError:
                pass
        return status, raw, ctype


class _Session:
    __slots__ = ("game_id", "guid", "started", "actions_used", "last", "lock",
                 "spans", "mark", "last_unix", "doc_pending")

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.guid: str | None = None
        self.started = False
        self.actions_used = 0
        self.last: dict[str, Any] | None = None
        self.lock = threading.Lock()
        # Actions spent on each level, cleared in order -- the same rule as score.plays_from_ledger, but kept in
        # memory so /api/progress need not reread the whole ledger each time (arm d last round: 400 MB).
        self.spans: list[int] = []
        self.mark = 0
        self.last_unix = 0.0
        # This level is cleared and its levels/Lnn.md has not arrived yet (--level-gates).
        self.doc_pending: int | None = None


class Middleware:
    def __init__(self, upstream: Upstream, tokens: TokenTable, *,
                 state_path: Path, ledger: Ledger | None = None,
                 card_id: str | None = None, rate: float = 10.0,
                 burst: float = 50.0,
                 baselines: dict[str, list[int]] | None = None,
                 short_names: bool = True,
                 discover_seconds: float = 0.0,
                 record_dirs: dict[str, Path] | None = None,
                 record_options: dict[str, Any] | None = None,
                 level_gates: dict[str, Path] | None = None,
                 level_gate_options: dict[str, int] | None = None,
                 image_ref_key: bytes | None = None,
                 level_episode: LevelEpisode | None = None,
                 game_stop_dir: Path | None = None,
                 seed_snapshots: Path | None = None,
                 level_snapshot_at: str = "first_action") -> None:
        self.upstream = upstream
        self.tokens = tokens
        # Local only (the corpus's baseline_actions); None on Kaggle, so /api/progress has no score.
        # **The agent never sees it** -- only the launcher key reaches that route.
        self.baselines = baselines
        self.state_path = state_path
        self.ledger = ledger
        self.card_id = card_id
        self.closed = False
        self.closed_payload: dict[str, Any] | None = None
        self.rate, self.burst = rate, burst
        self._guard = threading.Lock()
        # Open one game at a time. 2026-09-25 PC end-to-end: 8 games opened in the same second, the official server collided while
        # loading two game modules of the same name (the resampled cn04 and cn04…x2) in parallel, and answered 400 -- and in
        # competition mode a game can build its environment only once, so those two games could never open again: 0 for the round.
        # The hidden games all differ from each other, but nobody has measured how the organiser's gateway behaves when games are
        # built in parallel; opening one at a time costs only a fraction of a second, so always serialise.
        self._start_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._buckets: dict[str, list[float]] = {}
        self._games: list[dict[str, Any]] = []
        self._resolve: dict[str, str] = {}
        self.short_names = short_names
        self._load_games(discover_seconds)
        self.tokens = self._normalise_tokens(tokens)
        # Normalise game names to full names; an unknown game blows up here, rather than finding out at that game's opening that it has no record.
        self._record_dirs = {self.resolve(g): Path(d) for g, d in (record_dirs or {}).items()}
        self._record_options = record_options or {}
        self._records: dict[str, GameRecord] = {}
        self.record_errors = 0
        if image_ref_key is not None and not self._record_dirs:
            raise ValueError("image_ref_key needs record_dirs: an image ref names a recorded step")
        self._image_key = image_ref_key
        # game -> workspace. levels/ and .driver/ are both inside it.
        self._level_gates = {self.resolve(g): Path(w) for g, w in (level_gates or {}).items()}
        self._gate_options = level_gate_options or {}
        if self._level_gates and set(self._gate_options) != {"min_chars", "compact_min_tokens"}:
            raise ValueError(f"level_gate_options needs min_chars and compact_min_tokens, got {self._gate_options}")
        self.level_episode = level_episode
        if level_episode is not None:
            if self._level_gates:
                raise ValueError("level_episode plays one level: it cannot have level gates")
            owned = {r for roles in self.tokens._roles.values() for r in roles if r != "*"}
            if len(owned) != 1:
                raise ValueError(f"level_episode is one game, but the token table grants {sorted(owned)}")
            if level_episode.prior_actions and set(self._record_dirs) != owned:
                raise ValueError("a seeded level episode (prior_actions) continues the seed's record: it needs the "
                                 f"game's record dir, have {sorted(self._record_dirs)}")
        self.game_stop_dir = Path(game_stop_dir) if game_stop_dir is not None else None
        if self.game_stop_dir is not None:
            if level_episode is not None:
                raise ValueError("game_stop_dir is the whole-game episode's; a level episode has its own stop_dir")
            if not self.game_stop_dir.is_dir():
                raise FileNotFoundError(f"game_stop_dir {self.game_stop_dir} is not a directory")
        self.seed_snapshots = Path(seed_snapshots) if seed_snapshots is not None else None
        if self.seed_snapshots is not None:
            if not self._record_dirs:
                raise ValueError("seed_snapshots needs record_dirs: a snapshot is the game's workspace with its record")
            if level_episode is not None and not level_episode.through:
                raise ValueError("seed_snapshots collects from whole games, not from a single-level episode "
                                 "(a through episode plays on to the end)")
            self.seed_snapshots.mkdir(parents=True, exist_ok=True)
        if level_snapshot_at not in ("first_action", "note_done"):
            raise ValueError(f"level_snapshot_at must be first_action or note_done, not {level_snapshot_at!r}")
        if level_snapshot_at != "first_action" and self.seed_snapshots is None:
            raise ValueError("level_snapshot_at is about seed snapshots: it needs seed_snapshots")
        self.level_snapshot_at = level_snapshot_at
        self._load_state()

    # --- corpus -------------------------------------------------------------
    def _load_games(self, discover_seconds: float = 0.0) -> None:
        """``discover_seconds`` > 0: wait while the upstream is not ready (Kaggle's sidecar is not guaranteed to start before us;
        the old arm waited 600 s). 0: one shot -- locally the upstream is one we just started ourselves, so unreachable is an error.
        Failing to get a non-empty list raises all the same; never start with an empty list."""
        deadline = time.monotonic() + discover_seconds
        while True:
            try:
                status, payload, _ = self.upstream.request("GET", "/api/games")
                problem = None if status == 200 and isinstance(payload, list) and payload \
                    else f"upstream /api/games -> {status}: {payload!r}"
            except Fault as fault:
                problem = str(fault)
            if problem is None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(problem)
            time.sleep(2.0)
        self._games = payload
        ids = sorted(g["game_id"] for g in payload)
        for full in ids:
            self._resolve[full] = full
            if not self.short_names:
                # Kaggle: the token table, workspaces and ledger all use full names; short names only bring ambiguity --
                # the resampled public corpus (ar25-…x2) and multi-version games both make short names collide.
                continue
            short = full.split("-")[0]
            if short in self._resolve and self._resolve[short] != full:
                # Two versions of one game: which one is played decides the score. Not a case where we may just pick one.
                raise RuntimeError(
                    f"upstream has several versions of the same game: {short} -> "
                    f"{self._resolve[short]} / {full}")
            self._resolve[short] = full

    def _normalise_tokens(self, tokens: TokenTable) -> TokenTable:
        """Game names in the token table are always normalised to the upstream's full names.

        Caught by the 2026-09-18 smoke run: the launcher wrote `bp35=<key>` (short name), and the middleware
        sent `bp35` upstream as the game_id as is. The official `make()` accepted the short name, **opened the game
        successfully** and returned a guid, but every later action hit 400 on the **exact comparison**
        `environment_info.game_id == game_id` in `_get_or_create_environment` -- all three games had
        zero actions. The fake upstream uses full names, so the unit tests missed it; there is now a test with short names.
        """
        out = TokenTable()
        for key, roles in tokens._roles.items():
            for role in roles:
                out.grant(key, role if role == "*" else self.resolve(role))
        return out

    def resolve(self, game_id: Any) -> str:
        if not isinstance(game_id, str) or not game_id:
            raise Fault(400, "VALIDATION_ERROR", "`game_id` must be a non-empty string")
        try:
            return self._resolve[game_id]
        except KeyError:
            # The error name follows the official server (get_game_info returns SERVER_ERROR); caught by a probe.
            raise Fault(404, "SERVER_ERROR", f"game {game_id} not found") from None

    # --- state persistence -------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.card_id = data.get("card_id") or self.card_id
        self.closed = bool(data.get("closed"))
        for game_id, row in data.get("games", {}).items():
            s = _Session(game_id)
            s.guid, s.started = row.get("guid"), bool(row.get("started"))
            s.actions_used = int(row.get("actions_used") or 0)
            s.spans = list(row.get("spans") or [])
            s.mark = int(row.get("mark") or 0)
            s.last_unix = float(row.get("last_unix") or 0.0)
            s.doc_pending = row.get("doc_pending")
            self._sessions[game_id] = s

    def _save_state(self) -> None:
        payload = {
            "card_id": self.card_id,
            "closed": self.closed,
            "games": {g: {"guid": s.guid, "started": s.started,
                          "actions_used": s.actions_used, "spans": s.spans,
                          "mark": s.mark, "last_unix": s.last_unix,
                          "doc_pending": s.doc_pending}
                      for g, s in self._sessions.items()},
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.state_path)

    def _prior(self) -> int:
        """Actions counted before the prior-context seed (no seed = 0): record's step and the actions_used the agent sees both count up from here."""
        return self.level_episode.prior_actions if self.level_episode is not None else 0

    def _session(self, game_id: str) -> _Session:
        with self._guard:
            return self._sessions.setdefault(game_id, _Session(game_id))

    # --- auth and throttling -------------------------------------------------------
    def api_key(self, header: str | None) -> str:
        if not self.tokens.known(header):
            raise Fault(401, "VALIDATION_ERROR", "missing or invalid X-API-Key")
        return header  # type: ignore[return-value]

    def games_for(self, key: str) -> set[str] | None:
        """None = launcher, sees everything."""
        if self.tokens.is_launcher(key):
            return None
        return {r for r in self.tokens.roles(key) if r != "*"}

    def own_game(self, key: str, game_id: Any) -> str:
        """When a per-game key gives no game_id, use its only game."""
        allowed = self.games_for(key)
        if game_id in (None, ""):
            if allowed is not None and len(allowed) == 1:
                return self.resolve(next(iter(allowed)))
            raise Fault(400, "VALIDATION_ERROR", "missing `game_id` in action data")
        resolved = self.resolve(game_id)
        if allowed is not None and resolved not in allowed:
            raise Fault(403, "VALIDATION_ERROR",
                        f"key is not authorised for {resolved}")
        return resolved

    def throttle(self, key: str) -> None:
        now = time.monotonic()
        with self._guard:
            bucket = self._buckets.setdefault(key, [self.burst, now])
            bucket[0] = min(self.burst, bucket[0] + (now - bucket[1]) * self.rate)
            bucket[1] = now
            if bucket[0] < 1.0:
                wait = (1.0 - bucket[0]) / self.rate
                raise Fault(429, "RATE_LIMITED",
                            f"too many requests; retry in {wait:.1f}s "
                            f"(limit {self.rate:g}/s, burst {self.burst:g})",
                            {"Retry-After": str(max(1, int(wait + 0.999)))})
            bucket[0] -= 1.0

    # --- read-only routes -----------------------------------------------------------
    def games(self, key: str) -> list[dict[str, Any]]:
        allowed = self.games_for(key)
        if allowed is None:
            return self._games
        # A per-game key only sees its own game: listing other games would only tempt it to hit them and get 403.
        return [g for g in self._games if g["game_id"] in allowed]

    def game(self, key: str, game_id: str) -> dict[str, Any]:
        resolved = self.own_game(key, game_id)
        for g in self._games:
            if g["game_id"] == resolved:
                return g
        raise Fault(404, "SERVER_ERROR", f"game {game_id} not found")

    def status(self, key: str, game_id: Any) -> dict[str, Any]:
        """Free. Answered from cache; does not touch the upstream or spend an action."""
        resolved = self.own_game(key, game_id)
        s = self._session(resolved)
        with s.lock:
            out: dict[str, Any] = {
                "game_id": resolved, "started": s.started,
                "actions_used": self._prior() + s.actions_used, "closed": self.closed,
            }
            if s.last:
                for field in ("state", "levels_completed", "win_levels",
                              "available_actions", "frame"):
                    if field in s.last:
                        out[field] = s.last[field]
            if self._image_key is not None and s.started:
                out["image_ref"] = self._image_ref(resolved, self._prior() + s.actions_used)
            return out

    def _image_ref(self, game_id: str, step: int) -> str:
        """``record_dirs`` is ``<run>/games/<game>/record``: the proxy finds the step again by the game dir name."""
        return image_ref.sign(self._image_key, self._record_dirs[game_id].parent.name, step)

    def progress(self, key: str) -> dict[str, Any]:
        """Live progress across all games, **launcher-only**. The official server has nothing like it in competition mode (GET
        scorecard 403), and the agent should not have it either -- it may only know its own game's /api/status.

        With baselines (locally) it includes per-game RHAE and the mean; on Kaggle there are no baselines, so honestly no score.
        """
        self._launcher_only(key)
        now = time.time()
        games = []
        with self._guard:
            sessions = list(self._sessions.values())
        for s in sessions:
            with s.lock:
                row: dict[str, Any] = {
                    "game_id": s.game_id, "started": s.started,
                    "actions_used": s.actions_used,
                    "levels_completed": len(s.spans),
                    "state": (s.last or {}).get("state"),
                    "win_levels": (s.last or {}).get("win_levels"),
                    "last_action_age_s": round(now - s.last_unix, 1) if s.last_unix else None,
                }
                if self.baselines and s.game_id in self.baselines:
                    row["rhae"] = round(game_score(s.spans, self.baselines[s.game_id]), 4)
            games.append(row)
        games.sort(key=lambda r: r["game_id"])
        out: dict[str, Any] = {"card_id": self.card_id, "closed": self.closed,
                               "games": games}
        if self._record_dirs:
            out["record_errors"] = self.record_errors
        if self.baselines:
            scored = [r["rhae"] for r in games if "rhae" in r]
            out["rhae_mean_over_started"] = round(sum(scored) / len(scored), 4) if scored else 0.0
            out["note"] = "rhae is a local diagnostic from baselines; the score is the official card"
        return out

    # --- card ---------------------------------------------------------------
    def _launcher_only(self, key: str) -> None:
        if not self.tokens.is_launcher(key):
            raise Fault(403, "VALIDATION_ERROR", "scorecard routes are launcher-only")

    def open_card(self, key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        self._launcher_only(key)
        with self._guard:
            if self.card_id is not None:
                return {"card_id": self.card_id}
        status, payload, _ = self.upstream.request(
            "POST", "/api/scorecard/open", body or {"tags": ["arc-code"]})
        if status != 200 or not isinstance(payload, dict) or "card_id" not in payload:
            raise Fault(status if status >= 400 else 502, "UPSTREAM_ERROR",
                        f"open scorecard failed: {payload!r}")
        with self._guard:
            self.card_id = payload["card_id"]
            self._save_state()
        return {"card_id": self.card_id}

    def close_card(self, key: str) -> dict[str, Any]:
        self._launcher_only(key)
        if self.card_id is None:
            raise Fault(400, "VALIDATION_ERROR", "no scorecard is open")
        with self._guard:
            if self.closed_payload is not None:
                return self.closed_payload
        status, payload, _ = self.upstream.request(
            "POST", "/api/scorecard/close", {"card_id": self.card_id})
        if status != 200 or not isinstance(payload, dict):
            raise Fault(status if status >= 400 else 502, "UPSTREAM_ERROR",
                        f"close scorecard failed: {payload!r}")
        with self._guard:
            self.closed, self.closed_payload = True, payload
            self._save_state()
        return payload

    # --- actions -------------------------------------------------------------
    def _start(self, s: _Session, via: str) -> dict[str, Any]:
        """The opening RESET, which is also the only one. In competition mode the official server allows building the env only once."""
        if self.card_id is None:
            raise Fault(409, "VALIDATION_ERROR", "no scorecard is open yet")
        with self._start_lock:
            status, obs, _ = self.upstream.request(
                "POST", "/api/cmd/RESET", {"game_id": s.game_id, "card_id": self.card_id})
        if status != 200 or not isinstance(obs, dict) or not obs.get("guid"):
            raise Fault(status if status >= 400 else 502,
                        (obs or {}).get("error", "UPSTREAM_ERROR")
                        if isinstance(obs, dict) else "UPSTREAM_ERROR",
                        f"could not start {s.game_id}: "
                        f"{obs.get('message') if isinstance(obs, dict) else obs!r}")
        start_done = int(obs.get("levels_completed") or 0)
        if self.level_episode is not None:
            self._check_episode_start(s, obs, start_done)
            if self._prior():
                self._check_seed_start(s, obs)
            # Levels 1..K-1 were not played in this episode: their spans record 0 actions, so the next action is not spread over K-1 spans
            s.spans, s.mark = [0] * start_done, 0
        s.guid, s.started, s.last = obs["guid"], True, obs
        s.last_unix = time.time()
        if self.ledger:
            self.ledger.record(game=s.game_id, action="START", counted=False,
                               play=1, guid=s.guid, card_id=self.card_id,
                               obs=_Obs(obs), via=via)
        with self._guard:
            self._save_state()
        if self._prior():
            # The seed's record already reaches step N and the opening screen is its last step (checked by _check_seed_start): no step 0
            return {**obs, "record": {"step": self._prior(), "resumed_from_seed": True}}
        return self._recorded(s, obs, step=0, action="START", level=start_done + 1)

    def _check_seed_start(self, s: _Session, obs: dict[str, Any]) -> None:
        """Opening from a prior-context seed: the record must end exactly at step N (prior_actions) and the opening screen must equal
        its last board. set_level's starting point is not necessarily the state reached by playing up to this level in a whole game
        (sc25 carries state across levels); the gate before stub generation checked once, and here the seed itself is checked again
        in every episode. On mismatch write ``start_failed`` and return 500."""
        le = self.level_episode
        problems = []
        try:
            rec = self._records.get(s.game_id)
            if rec is None:
                rec = self._records[s.game_id] = GameRecord(self._record_dirs[s.game_id], **self._record_options)
            if rec.last_step != le.prior_actions:
                problems.append(f"the seed record ends at step {rec.last_step}, want {le.prior_actions}")
            final = frames_array(obs.get("frame"))
            if not len(final) or rec.last_final is None or not np.array_equal(final[-1], rec.last_final):
                problems.append("the start board is not the seed record's last board")
        except (RecordError, OSError, KeyError, ValueError) as error:
            problems.append(f"the seed record cannot be read: {type(error).__name__}: {error}")
        if problems:
            episode_stops.write_stop(le.stop_dir, episode_stops.START_FAILED,
                                     {"game": s.game_id, "level": le.level, "problems": problems,
                                      "prior_actions": le.prior_actions, "unix": time.time()})
            raise Fault(500, "SERVER_ERROR", f"{s.game_id}: the seeded level episode cannot start: {'; '.join(problems)}")

    def _check_episode_start(self, s: _Session, obs: dict[str, Any], start_done: int) -> None:
        """A single-level episode must open exactly on level K: if the gateway did not place it (hook not installed, different corpus version) the whole episode fails."""
        le = self.level_episode
        problems = []
        if start_done != le.level - 1:
            problems.append(f"levels_completed is {start_done} at the start, want {le.level - 1}")
        if obs.get("state") != "NOT_FINISHED":
            problems.append(f"state is {obs.get('state')!r} at the start")
        got = episode_stops.frame_hash(obs.get("frame"))
        if le.start_hash is not None and got != le.start_hash:
            problems.append(f"start frame hash {got} != expected {le.start_hash}")
        if problems:
            episode_stops.write_stop(le.stop_dir, episode_stops.START_FAILED,
                                     {"game": s.game_id, "level": le.level, "problems": problems,
                                      "frame_hash": got, "unix": time.time()})
            raise Fault(500, "SERVER_ERROR", f"{s.game_id}: the level episode cannot start: {'; '.join(problems)}")

    def _episode_gate(self, s: _Session) -> None:
        """Single-level episode: once a stop file exists (cleared, cap, budget, ...) no more actions are accepted; when the action count reaches the cap, write ``action_cap``. Neither spends an action."""
        le = self.level_episode
        over = episode_stops.stops(le.stop_dir)
        if over:
            raise Fault(409, "EPISODE_OVER",
                        f"This episode is over ({', '.join(sorted(over))}). No action was spent; stop now.")
        if le.through:
            return
        if s.actions_used >= le.action_cap:
            episode_stops.write_stop(le.stop_dir, episode_stops.ACTION_CAP,
                                     {"game": s.game_id, "level": le.level, "actions": s.actions_used,
                                      "unix": time.time()})
            raise Fault(409, "ACTION_CAP",
                        f"Level {le.level} has used its {le.action_cap} actions. This episode is over; "
                        "no action was spent.")

    def _recorded(self, s: _Session, obs: dict[str, Any], *, step: int, action: str,
                  level: int, x: int | None = None, y: int | None = None,
                  plan: str | None = None) -> dict[str, Any]:
        """Write the record; return the response for the client (with an extra ``record`` field).

        The upstream has already executed this action: if writing the record fails we **must not** return an error, which would
        lie to the agent that the action did not happen. So print the traceback, log it in ``record_errors`` (visible in the
        heartbeat), put ``error`` in the response, and the game goes on.
        """
        root = self._record_dirs.get(s.game_id)
        if root is None:
            return obs
        try:
            rec = self._records.get(s.game_id)
            if rec is None:
                rec = self._records[s.game_id] = GameRecord(root, **self._record_options)
            row = rec.write(step, action=action, obs=obs, level=level, x=x, y=y, plan=plan)
        except (RecordError, OSError, ValueError) as error:
            traceback.print_exc()
            self.record_errors += 1
            return {**obs, "record": {"step": step, "error": f"{type(error).__name__}: {error}"[:300]}}
        if "video_error" in row:
            print(f"RECORD video failed {s.game_id} step {step}: {row['video_error']}",
                  file=sys.stderr, flush=True)
            self.record_errors += 1
        out = {k: row[k] for k in ("step", "dir", "frames", "changed_cells",
                                   "video_frames", "video_error") if k in row}
        if self._image_key is not None:
            out["image_ref"] = self._image_ref(s.game_id, step)
        return {**obs, "record": out}

    def _level_gate(self, s: _Session, workspace: Path) -> None:
        """The next action after a level clear: block it if the doc fails the check; if it passes, ask codex to compact and block this time too."""
        level = s.doc_pending
        name = f"L{level:02d}.md"
        problem = self._doc_problem(workspace, name)
        if problem:
            raise Fault(409, "LEVEL_DOC_REQUIRED",
                        f"Level {level} is cleared. Before your next action, write levels/{name}: "
                        f"{problem}. No action was spent.")
        try:
            fd = _open_below(workspace, ".driver", "compact_request",
                             flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                f.write(f"{self._gate_options['compact_min_tokens']}\n")
        except FileExistsError:
            pass                                  # already requested (or placed by the agent itself): same thing
        s.doc_pending = None
        with self._guard:
            self._save_state()
        print(f"LEVEL_GATE {s.game_id} levels/{name} accepted; compaction requested",
              file=sys.stderr, flush=True)
        raise Fault(409, "COMPACTING",
                    f"levels/{name} is in place. Your context is being compacted now. Afterwards, "
                    "read notes.md and every file in levels/ in full, then send this action again. "
                    "No action was spent.")

    def _doc_problem(self, workspace: Path, name: str) -> str | None:
        try:
            fd = _open_below(workspace, "levels", name)
        except FileNotFoundError:
            return "the file does not exist yet"
        except OSError as error:
            return f"it cannot be read ({error.strerror})"
        with os.fdopen(fd, "rb") as f:
            text = f.read(MAX_DOC).decode("utf-8", errors="replace")
        need = self._gate_options["min_chars"]
        if len(text.strip()) < need:
            return f"it has {len(text.strip())} characters; write at least {need}"
        if "record/steps/" not in text:
            return ("it cites no record/steps/ path; cite the steps, frames or animations "
                    "that showed each rule")
        return None

    @staticmethod
    def _validate(action: str, body: dict[str, Any], last: dict[str, Any] | None
                  ) -> dict[str, Any]:
        if action not in ACTIONS:
            raise Fault(404, "VALIDATION_ERROR",
                        f"no such action {action}; one of {sorted(ACTIONS)}")
        extra: dict[str, Any] = {}
        if action == COORD_ACTION:
            x, y = body.get("x"), body.get("y")
            if not isinstance(x, int) or not isinstance(y, int) \
                    or isinstance(x, bool) or isinstance(y, bool):
                raise Fault(400, "VALIDATION_ERROR",
                            f"{COORD_ACTION} needs integer `x` and `y` at the top "
                            "level of the body")
            if not (0 <= x < GRID and 0 <= y < GRID):
                raise Fault(400, "VALIDATION_ERROR",
                            f"`x`/`y` must be within 0..{GRID - 1}, got {x},{y}")
            extra["x"], extra["y"] = x, y
        if action != "RESET" and last is not None:
            # 6 h live run, tu93: sending ACTION1 after GAME_OVER, the official server returns GAME_NOT_STARTED_ERROR
            # "send RESET to begin playing". Block here first with the same sentence and save a round trip; the lesson
            # the agent learns is word for word what the official server gives.
            if last.get("state") == "GAME_OVER":
                raise Fault(400, "GAME_NOT_STARTED_ERROR",
                            f"game {last.get('game_id', '')} is over (GAME_OVER); "
                            "send RESET to begin playing this level again")
            avail = last.get("available_actions")
            if isinstance(avail, list) and ACTIONS[action] not in avail:
                names = [f"ACTION{i}" for i in avail if isinstance(i, int)]
                raise Fault(400, "VALIDATION_ERROR",
                            f"{action} is not available in this game right now; "
                            f"available: {names}. RESET is always available.")
        if "reasoning" in body:
            extra["reasoning"] = body["reasoning"]
        return extra

    def _at_seed_start(self, s: _Session) -> bool:
        """A through episode still on the level the seed opened on: that level's starting point is the seed itself; no second snapshot."""
        le = self.level_episode
        return le is not None and le.through and len(s.spans) + 1 <= le.level

    def _workspace(self, s: _Session) -> Path:
        return self._record_dirs[s.game_id].parent

    def _seed_snapshot(self, s: _Session, name: str, extra: dict[str, Any]) -> dict[str, Any]:
        """Called inside ``s.lock``. If the snapshot fails, write the error to ERRORS.jsonl and raise (Fault 500, no action spent)."""
        assert self.seed_snapshots is not None
        workspace = self._workspace(s)
        meta = {**extra, "game": workspace.name, "game_id": s.game_id, "levels_completed": len(s.spans),
                "level": len(s.spans) + 1, "actions": self._prior() + s.actions_used}
        try:
            out = seed_snapshot.take(workspace, self.seed_snapshots, name, meta)
            if out["record_last"]["step"] != meta["actions"]:
                raise seed_snapshot.SnapshotError(
                    f"{workspace.name}/{name}: record ends at step {out['record_last']['step']} but the middleware "
                    f"counted {meta['actions']} actions")
        except Exception as error:
            traceback.print_exc()
            with self._guard, open(self.seed_snapshots / "ERRORS.jsonl", "a") as f:
                f.write(json.dumps({"unix": time.time(), "game": workspace.name, "name": name,
                                    "error": f"{type(error).__name__}: {error}"}) + "\n")
            raise Fault(500, "SERVER_ERROR", f"seed snapshot {workspace.name}/{name} failed: "
                        f"{type(error).__name__}: {error}. No action was spent.") from error
        print(f"SEED_SNAPSHOT {workspace.name}/{name} files={out['files']} skipped={len(out['skipped'])} "
              f"{out['copy_seconds']}s", file=sys.stderr, flush=True)
        return out

    def snapshot(self, key: str, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/snapshot``: window-change snapshot (gen-2 SessionStart hook). ``{"window": n, "source": ...}``; a second call
        for the same window returns ``existing``. ``{"level_done": L}``: the level-change snapshot of ``--level-snapshot-at note_done``
        (called by act's note_done), taken as ``L{L+1:02d}``."""
        if self.seed_snapshots is None:
            raise Fault(404, "SERVER_ERROR", "no route POST /api/snapshot (seed snapshots are off)")
        if "level_done" in body:
            return self._note_done_snapshot(key, body)
        window, source = body.get("window"), body.get("source")
        if not (isinstance(window, int) and not isinstance(window, bool) and 1 <= window <= 999):
            raise Fault(400, "VALIDATION_ERROR", f"`window` must be an int in 1..999, not {window!r}")
        if source not in ("startup", "compact"):
            raise Fault(400, "VALIDATION_ERROR", f"`source` must be startup or compact, not {source!r}")
        resolved = self.own_game(key, body.get("game_id"))
        self.throttle(key)
        s = self._session(resolved)
        name = seed_snapshot.window_name(window)
        with s.lock:
            if not s.started:
                raise Fault(409, "VALIDATION_ERROR", "the game is not open yet")
            if seed_snapshot.exists(self.seed_snapshots, self._workspace(s).name, name):
                return {"snapshot": name, "existing": True}
            if self.level_episode is not None and self.level_episode.through and s.actions_used == 0:
                return {"snapshot": None, "skipped": "the seed's own start (no action since the seed)"}
            out = self._seed_snapshot(s, name, {"kind": "window", "window": window, "source": source})
        return {"snapshot": name, "existing": False, "files": out["files"], "skipped": len(out["skipped"])}

    def _note_done_snapshot(self, key: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.level_snapshot_at != "note_done":
            raise Fault(409, "VALIDATION_ERROR", "level snapshots are taken at the first action here, not at note_done")
        done = body["level_done"]
        if not (isinstance(done, int) and not isinstance(done, bool) and 1 <= done <= 99):
            raise Fault(400, "VALIDATION_ERROR", f"`level_done` must be an int in 1..99, not {done!r}")
        resolved = self.own_game(key, body.get("game_id"))
        self.throttle(key)
        s = self._session(resolved)
        with s.lock:
            if not s.started:
                raise Fault(409, "VALIDATION_ERROR", "the game is not open yet")
            if len(s.spans) != done:
                raise Fault(409, "VALIDATION_ERROR", f"note_done for level {done}, but {len(s.spans)} levels are done")
            if s.actions_used != s.mark:
                raise Fault(409, "VALIDATION_ERROR", f"level {done + 1} already has {s.actions_used - s.mark} actions")
            name = seed_snapshot.level_name(done + 1)
            if seed_snapshot.exists(self.seed_snapshots, self._workspace(s).name, name):
                return {"snapshot": name, "existing": True}
            if self._at_seed_start(s):
                return {"snapshot": None, "skipped": "the seed's own start (no action since the seed)"}
            if seed_snapshot.record_last(self._workspace(s))["state"] == "WIN":
                return {"snapshot": None, "skipped": "the game is won"}
            out = self._seed_snapshot(s, name, {"kind": "level", "at": "note_done"})
        return {"snapshot": name, "existing": False, "files": out["files"], "skipped": len(out["skipped"])}

    def command(self, key: str, action: str, body: dict[str, Any], via: str
                ) -> dict[str, Any]:
        resolved = self.own_game(key, body.get("game_id"))
        plan = body.get("plan")
        if plan is not None and (not isinstance(plan, str) or len(plan) > MAX_PLAN):
            raise Fault(400, "VALIDATION_ERROR", f"`plan` must be a string of at most {MAX_PLAN} characters")
        self.throttle(key)
        s = self._session(resolved)
        with s.lock:
            if self.closed:
                raise Fault(400, "SERVER_ERROR",
                            "scorecard is closed; the run is over")
            if not s.started:
                if self._prior() and not self.tokens.is_launcher(key):
                    # A seeded game is opened by the launcher before codex starts (the opening message says "the game is open"): the agent getting here first is a harness bug
                    raise Fault(409, "VALIDATION_ERROR", "the game is not open yet (the harness opens a seeded "
                                "level episode before the session starts)")
                obs = self._start(s, via)
                if action == "RESET":
                    return obs          # opening: no action spent, return the initial board directly
            workspace = self._level_gates.get(resolved)
            if workspace is not None and s.doc_pending is not None:
                self._level_gate(s, workspace)
            if self.level_episode is not None:
                self._episode_gate(s)
            elif self.game_stop_dir is not None:
                over = episode_stops.stops(self.game_stop_dir)
                if over:
                    raise Fault(409, "EPISODE_OVER",
                                f"This episode is over ({', '.join(sorted(over))}). No action was spent; stop now.")
            extra = self._validate(action, body, s.last)
            if self.seed_snapshots is not None and len(s.spans) >= 1 and not self._at_seed_start(s):
                # Level K's first counted action (K = levels cleared + 1; spans are persisted, so this holds across middleware restarts):
                # snapshot before sending. If the agent keeps sending actions after beating the game (2026-10-22 s092 ar25) there is no
                # level K, so no snapshot (judged from the record's last step: after a restart s.last is empty)
                name = seed_snapshot.level_name(len(s.spans) + 1)
                if not seed_snapshot.exists(self.seed_snapshots, self._workspace(s).name, name) \
                        and seed_snapshot.record_last(self._workspace(s))["state"] != "WIN":
                    at = "first_action" if self.level_snapshot_at == "first_action" else "first_action_fallback"
                    self._seed_snapshot(s, name, {"kind": "level", "action": action, "at": at})
            level = int((s.last or {}).get("levels_completed") or 0) + 1
            payload = {"game_id": resolved, "guid": s.guid, "card_id": self.card_id,
                       **extra}
            status, obs, _ = self.upstream.request("POST", f"/api/cmd/{action}", payload)
            if status != 200 or not isinstance(obs, dict) or "state" not in obs:
                # The official error goes to the agent as is -- that is what it should learn.
                if isinstance(obs, dict) and "error" in obs:
                    raise Fault(status, str(obs["error"]), str(obs.get("message", "")))
                raise Fault(status if status >= 400 else 502, "UPSTREAM_ERROR",
                            f"official gateway returned {status}")
            s.last = obs
            s.actions_used += 1
            s.last_unix = time.time()
            done = int(obs.get("levels_completed") or 0)
            if workspace is not None and done >= level and obs.get("state") != "WIN":
                s.doc_pending = done
            while done > len(s.spans):
                s.spans.append(s.actions_used - s.mark)
                s.mark = s.actions_used
            if self.ledger:
                self.ledger.record(game=resolved, action=action, counted=True,
                                   play=1, guid=s.guid, card_id=self.card_id,
                                   obs=_Obs(obs), via=via,
                                   x=extra.get("x"), y=extra.get("y"))
            le = self.level_episode
            if le is not None and not le.through and done >= le.level:
                # The ledger line is written first (above), the stop file after: whoever reads cleared can always find that action in the ledger
                episode_stops.write_stop(le.stop_dir, episode_stops.CLEARED,
                                         {"game": resolved, "level": le.level, "actions": s.actions_used,
                                          "state": obs.get("state"), "unix": time.time()})
            with self._guard:
                self._save_state()
            return self._recorded(s, obs, step=self._prior() + s.actions_used, action=action, level=level,
                                  x=extra.get("x"), y=extra.get("y"), plan=plan)


def _open_below(workspace: Path, *names: str, flags: int = os.O_RDONLY, mode: int = 0o644) -> int:
    """A path under the workspace, opened O_NOFOLLOW at every component: the workspace belongs to the agent, and the middleware,
    running as root, must not be led by the agent's symlinks into reading or writing anywhere else."""
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names[:-1]:
            nxt = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(names[-1], flags | os.O_NOFOLLOW, mode, dir_fd=fd)
    finally:
        os.close(fd)


class _Obs:
    """Lets ``Ledger.record`` read ``levels_completed`` and other fields from the official response (a dict)."""

    class _State:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, obs: dict[str, Any]) -> None:
        self.levels_completed = obs.get("levels_completed")
        self.win_levels = obs.get("win_levels")
        self.state = self._State(str(obs.get("state")))


# --- HTTP shell --------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    mw: Middleware
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, payload: Any, *, ctype: str = "application/json",
              headers: dict[str, str] | None = None) -> None:
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise Fault(413, "VALIDATION_ERROR", "body too large")
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise Fault(400, "VALIDATION_ERROR", f"body is not valid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise Fault(400, "VALIDATION_ERROR", "request body must be a JSON object")
        return data

    def _via(self) -> str:
        agent = (self.headers.get("User-Agent") or "").lower()
        return "act" if "arc-code-act" in agent else "direct"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            path, _, query = self.path.partition("?")
            path = path.rstrip("/") or "/"
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/api/healthcheck" and method == "GET":
                # Exactly like the official server: plain-text okay, no key needed. The readiness probe relies on it.
                self._send(200, b"okay", ctype="text/plain")
                return
            key = self.mw.api_key(self.headers.get("X-API-Key"))
            body = self._body() if method == "POST" else {}
            result = self._route(method, path, params, body, key)
            if isinstance(result, tuple):
                payload, ctype = result
                self._send(200, payload, ctype=ctype)
            else:
                self._send(200, result)
        except Fault as fault:
            self._send(fault.status, fault.body(), headers=fault.headers)
        except Exception as error:  # noqa: BLE001 — bugs here must be visible, not a dropped socket
            import traceback
            traceback.print_exc()
            self._send(500, {"error": "SERVER_ERROR",
                             "message": f"{type(error).__name__}: {error}"})

    def _route(self, method: str, path: str, params: dict[str, str],
               body: dict[str, Any], key: str) -> Any:
        mw = self.mw
        if method == "GET":
            if path == "/api/games":
                return mw.games(key)
            if path == "/api/status":
                return mw.status(key, params.get("game_id"))
            if path == "/api/progress":
                return mw.progress(key)
            if path.startswith("/api/games/"):
                return mw.game(key, path[len("/api/games/"):])
            if path.startswith("/api/scorecard/"):
                # Passed through to the official server -- in competition mode it returns 403 itself, so what the agent learns is real.
                mw._launcher_only(key)
                status, payload, ctype = mw.upstream.request("GET", path)
                if status != 200:
                    raise Fault(status, (payload or {}).get("error", "SERVER_ERROR")
                                if isinstance(payload, dict) else "SERVER_ERROR",
                                (payload or {}).get("message", "") if isinstance(payload, dict) else "")
                return payload
        else:
            if path == "/api/scorecard/open":
                return mw.open_card(key, body or None)
            if path == "/api/scorecard/close":
                return mw.close_card(key)
            if path == "/api/snapshot":
                return mw.snapshot(key, body)
            if path.startswith("/api/cmd/"):
                return mw.command(key, path[len("/api/cmd/"):], body, self._via())
        raise Fault(404, "SERVER_ERROR", f"no route {method} {path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--upstream", required=True, help="base URL of the official gateway")
    key = p.add_mutually_exclusive_group(required=True)
    key.add_argument("--upstream-key", help="competition key; the agent never sees it")
    # Under a fence without a pid namespace (Kaggle's fence_posix), every user can read
    # everyone else's /proc/<pid>/cmdline -- a key in argv is a key handed to every agent.
    key.add_argument("--upstream-key-file", type=Path,
                     help="read the competition key from a file (0600); argv is visible to everyone on a host without a pid namespace")
    p.add_argument("--tokens", type=Path, required=True, help="<game|*>=<key>, one per line")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--state", type=Path, required=True,
                   help="persistence file for the guid and other state; a restart does not kill a game")
    p.add_argument("--ledger", type=Path, help="convenience record (not authoritative)")
    p.add_argument("--open-card", action="store_true", help="open a card upstream at start-up")
    p.add_argument("--card-id", help="use the card the organiser gives (Kaggle rerun)")
    p.add_argument("--card-out", type=Path, help="write card_id here for the launcher")
    p.add_argument("--baselines", type=Path, metavar="ENVIRONMENT_FILES",
                   help="local corpus; only with it does /api/progress include RHAE (not given on Kaggle)")
    p.add_argument("--rate", type=float, default=10.0, help="requests per second per key")
    p.add_argument("--burst", type=float, default=50.0)
    p.add_argument("--full-ids-only", action="store_true",
                   help="do not create short-name aliases (Kaggle: any name but the full one only brings ambiguity)")
    p.add_argument("--discover-seconds", type=float, default=0.0,
                   help="max seconds to wait while upstream /api/games is not reachable yet (0 = one shot)")
    p.add_argument("--record-dirs", type=Path,
                   help="JSON {game: record dir}: when given, every action is written into that game's record (record.py)")
    p.add_argument("--ffmpeg", help="ffmpeg used to encode animation.mp4 (required with --record-dirs)")
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument("--video-size", type=int, default=384)
    p.add_argument("--video-repeat", type=int, default=1,
                   help="how many video frames each frame is held for in animation.mp4 (record.py GameRecord video_repeat)")
    p.add_argument("--png-size", type=int, default=512,
                   help="side length of final.png / frame_NNN.png / current.png (a multiple of 64; size÷64 px per cell)")
    p.add_argument("--video-max-frames", type=int,
                   help="max frames in animation.mp4 (beyond that, dedupe and sample evenly); omitted = unlimited")
    p.add_argument("--record-tmp", type=Path, help="ffmpeg temp directory (somewhere the agent cannot reach)")
    p.add_argument("--level-gates", type=Path,
                   help="JSON {game: workspace}: after a level clear, require levels/Lnn.md before letting actions through, then ask codex to compact")
    p.add_argument("--level-doc-min-chars", type=int)
    p.add_argument("--compact-min-tokens", type=int)
    p.add_argument("--image-ref-key", type=Path,
                   help="act_image signing key file (somewhere the agent cannot read); when given, every step and /api/status carry image_ref")
    p.add_argument("--level-episode", type=int, metavar="K",
                   help="single-level episode: the game must open on level K (1-based); ends on level clear or at --action-cap (level_episode.py)")
    p.add_argument("--action-cap", type=int, help="max actions counted on the --level-episode level (including RESET)")
    p.add_argument("--stop-dir", type=Path, help="stop-file directory for --level-episode (host side, out of the agent's reach); given alone = "
                                                 "whole-game episode: every action after a stop file appears gets 409 and costs no action")
    p.add_argument("--expect-start-hash", help="frame_hash of the --level-episode opening screen; on mismatch the game is not opened")
    p.add_argument("--through", action="store_true",
                   help="after --level-episode keep playing to the end (gen-2 seed collection, continued runs on hard levels): no stop on level clear, no action cap (--action-cap 0)")
    p.add_argument("--prior-actions", type=int,
                   help="the --level-episode workspace was loaded with a prior-context seed: actions already counted on the first K-1 levels (the step the record reaches)")
    p.add_argument("--seed-snapshots", type=Path,
                   help="gen-2 seed collection: snapshot the workspace here at level changes (before level K's first action) and window changes (POST /api/snapshot)"
                        " (seed_snapshot.py); needs --record-dirs")
    p.add_argument("--level-snapshot-at", choices=("first_action", "note_done"), default="first_action",
                   help="when the level-change snapshot is taken: before level K's first action (default), or when act's note_done passes (POST /api/snapshot level_done)")
    a = p.parse_args(argv)
    episode_args = (a.level_episode, a.action_cap, a.stop_dir)
    if any(x is not None for x in (a.level_episode, a.action_cap, a.expect_start_hash, a.prior_actions)) \
            and any(x is None for x in episode_args):
        p.error("--level-episode, --action-cap and --stop-dir go together (--expect-start-hash / --prior-actions need them); "
                "--stop-dir alone is a whole-game episode")
    if a.record_dirs and not (a.ffmpeg and a.record_tmp):
        p.error("--record-dirs needs --ffmpeg and --record-tmp")
    if a.level_gates and (a.level_doc_min_chars is None or a.compact_min_tokens is None):
        p.error("--level-gates needs --level-doc-min-chars and --compact-min-tokens")
    if a.image_ref_key and not a.record_dirs:
        p.error("--image-ref-key needs --record-dirs")
    if a.seed_snapshots and not a.record_dirs:
        p.error("--seed-snapshots needs --record-dirs")

    if a.upstream_key_file is not None:
        upstream_key = a.upstream_key_file.read_text().strip()
        if not upstream_key:
            raise SystemExit(f"{a.upstream_key_file} is empty")
    else:
        upstream_key = a.upstream_key
    upstream = Upstream(a.upstream, upstream_key)
    base = None
    if a.baselines:
        from .score import baselines as load_baselines
        base = load_baselines(a.baselines)
    mw = Middleware(upstream, TokenTable.load(a.tokens), state_path=a.state,
                    ledger=Ledger(a.ledger, authority=MIDDLEWARE) if a.ledger else None,
                    card_id=a.card_id, rate=a.rate, burst=a.burst, baselines=base,
                    short_names=not a.full_ids_only, discover_seconds=a.discover_seconds,
                    record_dirs=json.loads(a.record_dirs.read_text()) if a.record_dirs else None,
                    record_options={"ffmpeg": a.ffmpeg, "video_fps": a.video_fps,
                                    "video_size": a.video_size, "video_max_frames": a.video_max_frames,
                                    "png_size": a.png_size, "video_repeat": a.video_repeat,
                                    "tmp_dir": a.record_tmp},
                    level_gates=json.loads(a.level_gates.read_text()) if a.level_gates else None,
                    level_gate_options={"min_chars": a.level_doc_min_chars,
                                        "compact_min_tokens": a.compact_min_tokens}
                    if a.level_gates else None,
                    image_ref_key=image_ref.load_key(a.image_ref_key) if a.image_ref_key else None,
                    level_episode=LevelEpisode(a.level_episode, a.action_cap, a.stop_dir, a.expect_start_hash,
                                               a.prior_actions or 0, a.through)
                    if a.level_episode is not None else None,
                    game_stop_dir=a.stop_dir if a.level_episode is None else None,
                    seed_snapshots=a.seed_snapshots, level_snapshot_at=a.level_snapshot_at)
    if a.open_card and mw.card_id is None:
        launcher_key = next(k for k, r in mw.tokens._roles.items() if "*" in r)
        mw.open_card(launcher_key)
    if a.card_out and mw.card_id:
        a.card_out.write_text(json.dumps({"card_id": mw.card_id}, indent=2))
    print(f"middleware on {a.host}:{a.port} upstream={a.upstream} "
          f"card={mw.card_id} games={len(mw._games)} rate={a.rate:g}/s burst={a.burst:g}"
          + (f" level_episode={mw.level_episode}" if mw.level_episode is not None else "")
          + (f" game_stop_dir={mw.game_stop_dir}" if mw.game_stop_dir is not None else ""),
          file=sys.stderr, flush=True)
    Handler.mw = mw
    _Server((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
