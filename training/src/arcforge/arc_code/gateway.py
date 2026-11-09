"""ARC-AGI-3 gateway: a faithful implementation of the official 15 routes.

**The real gateway cannot tell ``act.py`` from ``curl``, so neither should we.**
The agent may use the actuator or write its own client; the authoritative action record lives
on this side, so bypassing ``act.py`` is pointless, not merely "detectable".

Design rationale: ``docs/plans/arc_code_gateway.md``.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections import defaultdict
from typing import Any

from . import errors as E
from .engine import ACTIONS, COORD_ACTION, EngineAdapter
from .errors import ApiFault, ServerError
from .ledger import Ledger
from .tokens import TokenTable

#: Our **deliberate** differences from the official server. In competition mode there must be no
#: ``agent_visible`` entry -- the tests assert this. Diverging is fine; diverging silently is not.
DIVERGENCES: list[dict[str, Any]] = [
    {
        "what": "with auth='strict', an unknown key gets 401 and a key for another game gets 403",
        "why": "Defence in depth. The official server has no 401; a missing key defaults to '1234' (api.py:89 and four other places)",
        "agent_visible": True,
        "disabled_by": "auth='official'",
    },
    {
        "what": "no automatic reclaiming of idle scorecards",
        "why": "The official server sweeps every 60 s with a ~15 min timeout (api.py:445-470). Not needed locally",
        "agent_visible": False,
        "disabled_by": None,
    },
    {
        "what": "ONLY_RESET_LEVELS not implemented",
        "why": "The official server has this env-var switch (api.py:352-358). Whether Kaggle's gateway enables it was not probed",
        "agent_visible": False,
        "disabled_by": None,
    },
]


def competition_divergences(auth: str) -> list[dict[str, Any]]:
    """Differences still agent-visible in competition mode. Should be empty."""
    live = []
    for row in DIVERGENCES:
        if not row["agent_visible"]:
            continue
        if row["disabled_by"] == f"auth='{auth}'":
            continue
        live.append(row)
    return live


class _Game:
    """Server-side state of one game."""

    __slots__ = ("env", "guid", "play", "started")

    def __init__(self) -> None:
        self.env: Any = None
        self.guid: str | None = None
        self.play = 0
        self.started = False        # whether this play's first RESET has been taken


class Gateway:
    """Pure logic, no HTTP. The HTTP layer is in ``server.py``."""

    def __init__(self, adapter: EngineAdapter, tokens: TokenTable, ledger: Ledger,
                 *, competition_mode: bool = True, auth: str = "strict",
                 ) -> None:
        if auth not in ("strict", "official"):
            raise ServerError(f"auth must be strict or official, not {auth!r}")
        self.adapter = adapter
        self.tokens = tokens
        self.ledger = ledger
        self.competition_mode = competition_mode
        self.auth = auth

        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._games: dict[str, _Game] = defaultdict(_Game)
        self._by_guid: dict[str, str] = {}
        self.card_id: str | None = None
        self.card_closed = False
        self._closed_payload: dict[str, Any] | None = None
        # short id -> full id. act.py gets the full form, but short ids resolve too, so a typo
        # does not silently hit "no such game".
        self._resolve: dict[str, str] = {}
        for full in self.adapter.list_games():
            self._resolve[full] = full
            self._resolve.setdefault(full.split("-")[0], full)

    # --- auth ------------------------------------------------------------
    def api_key(self, header: str | None) -> str:
        """The official server has no 401: a missing header is treated as ``"1234"`` (``api.py:89``)."""
        if self.auth == "official":
            return header if header is not None else "1234"
        if not self.tokens.known(header):
            raise ApiFault(401, E.APIError.VALIDATION_ERROR,
                           "missing or invalid X-API-Key")
        return header  # type: ignore[return-value]

    def cookie_value(self) -> str | None:
        """The ``arc_session`` attached to responses. **Set only, never checked** -- as the official server does.

        Before 2026-09-18 there was also a ``check_cookie()`` here that returned
        ``400 game <id> not found`` without it. **I invented that.** The entire official server
        (``api.py`` / ``server.py``) has only one cookie-related thing, an ``add_cookie`` callback
        that *attaches* the cookie to 200 responses (``api.py:44-53``); **nothing anywhere checks
        it**. The client side (``base.py:914-919``) states its own reason for keeping a cookie jar:
        the AWS ALB stickiness cookie (``AWSALBAPP-*``). That is the load balancer's business, not
        the API contract.

        Cost: that invented check wrecked the two games `b/g50t` and `d/sc25` for a full six hours.
        The agent's first RESET had no cookie and was blocked inside ``command()``;
        ``adapter.make()`` was never called (neither game has a single file under recordings/).
        And ``act.py`` could not get that cookie -- ``jar.save()`` comes after the ``except``, the
        400 raises straight away, so the ``Set-Cookie`` the server put on the 400 was received and
        dropped. The error message also sent both agents in the wrong direction: `b/g50t` concluded
        "the fence deleted the server-side corpus" and then issued 3.5 million RESETs.

        This is the second time I invented a check the official server does not have (the first was
        the 401 error table in design doc v1). **If you cannot grep it in the official source, do not
        implement it.**

        No card means no session to pin; return None.
        """
        if self.card_id is None:
            return None
        return hashlib.sha256(f"arc-code:{self.card_id}".encode()).hexdigest()[:32]

    def _may_play(self, api_key: str, game_id: str) -> None:
        if self.auth == "official":
            return
        if not self.tokens.may_play(api_key, game_id):
            raise ApiFault(403, E.APIError.VALIDATION_ERROR,
                           f"key is not authorised for {game_id}")

    def _launcher_only(self, api_key: str) -> None:
        if self.auth == "official":
            return
        # close returns every game's progress -- a cross-game peek the player does not need.
        if not self.tokens.is_launcher(api_key):
            raise ApiFault(403, E.APIError.VALIDATION_ERROR,
                           "scorecard is launcher-only")

    def resolve(self, game_id: Any, *, info_route: bool = False) -> str:
        # Check the type first. In the 2026-09-18 four-arm comparison a self-written client sent
        # game_id as a list, so `self._resolve[game_id]` raised TypeError: unhashable type and
        # surfaced as a 500. **Nothing a client sends may be used directly as a dict key.**
        if not isinstance(game_id, str):
            raise ApiFault(
                400, E.APIError.VALIDATION_ERROR,
                "`game_id` must be a string")
        try:
            return self._resolve[game_id]
        except KeyError:
            raise (E.game_not_found_info(game_id) if info_route
                   else E.game_not_found_cmd(game_id)) from None

    # --- routes: games -----------------------------------------------------
    def games(self) -> list[dict[str, Any]]:
        return [self.adapter.game_info(g) for g in self.adapter.list_games()]

    def game(self, game_id: str) -> dict[str, Any]:
        return self.adapter.game_info(self.resolve(game_id, info_route=True))

    def healthcheck(self) -> dict[str, Any]:
        return {"status": "ok"}

    # --- routes: scorecard -------------------------------------------------
    def open_card(self, api_key: str) -> dict[str, Any]:
        # The competition check must come before launcher-only. The official server has **no**
        # launcher/per-game distinction (`api.py:132-141` only looks at competition_mode), so in
        # competition mode a per-game key opening a card gets 409 "there is already a card" from
        # the real gateway, not our 403 "this route is launcher-only". Run our check first and the
        # agent learns a lesson that does not hold on competition day -- exactly what this whole
        # design exists to eliminate.
        with self._guard:
            if self.card_id is not None and self.competition_mode:
                raise E.multiple_scorecards()
        self._launcher_only(api_key)
        with self._guard:
            if self.card_id is None:
                self.card_id = self.adapter.open_scorecard()
            return {"card_id": self.card_id}

    def get_card(self, card_id: str, api_key: str) -> dict[str, Any]:
        # Order follows the official `api.py:86-107`: card missing -> 404, competition_mode -> 403.
        # The official server has no key-role check, so in competition mode every key gets the same 403.
        if self.card_id is None or card_id != self.card_id:
            raise E.card_not_found(card_id)
        # This is the root of single-pass: act.py's alive() treats any non-200 as a dead card, and
        # run.py then does a whole-batch --again. The official answer is 403, so --resume is
        # structurally impossible on the real gateway, and locally we should not pretend otherwise.
        if self.competition_mode:
            raise E.scorecard_in_competition_mode()
        self._launcher_only(api_key)
        return {"card_id": self.card_id}

    def get_card_game(self, card_id: str, game_id: str,
                      api_key: str) -> dict[str, Any]:
        """Official `/api/scorecard/<card_id>/<game_id>`. Equally opaque in competition mode."""
        if self.card_id is None or card_id != self.card_id:
            raise E.card_not_found(card_id)
        if self.competition_mode:
            raise E.scorecard_in_competition_mode()
        self._launcher_only(api_key)
        resolved = self.resolve(game_id, info_route=True)
        state = self._games.get(resolved)
        if state is None:
            raise E.game_not_found_info(game_id)
        return {"card_id": card_id, "game_id": resolved, "plays": state.play}

    def close_card(self, card_id: str, api_key: str) -> dict[str, Any]:
        self._launcher_only(api_key)
        with self._guard:
            if self.card_id is None or card_id != self.card_id:
                raise E.card_not_found(card_id)
            if self._closed_payload is None:
                self._closed_payload = self.adapter.close_scorecard(card_id)
                self.card_closed = True
            return self._closed_payload

    # --- routes: cmd -------------------------------------------------------
    def command(self, action: str, body: dict[str, Any], api_key: str,
                *, cookie: str | None = None, via: str = "unknown") -> dict[str, Any]:
        if action not in ACTIONS:
            raise E.game_not_found_cmd(str(body.get("game_id", "")))
        raw_game = body.get("game_id")
        if not raw_game:
            raise E.missing_game_id()
        game_id = self.resolve(raw_game)
        self._may_play(api_key, game_id)

        guid = body.get("guid")
        if action != "RESET" and not guid:
            raise E.missing_guid()

        with self._locks[game_id]:
            state = self._games[game_id]
            if action == "RESET":
                return self._reset(game_id, state, guid, api_key, via)
            if state.guid is None or guid != state.guid:
                raise E.guid_key_mismatch(raw_game, str(guid), api_key)
            return self._act(game_id, state, action, body, via)

    def _reset(self, game_id: str, state: _Game, guid: str | None,
               api_key: str, via: str) -> dict[str, Any]:
        new_play = guid is None or guid != state.guid
        if new_play and state.env is not None and self.competition_mode:
            # Official api.py:424: once a game has been touched, competition mode never opens a
            # second environment for it. On the wire it looks the same as "no such game"; official behaviour.
            raise E.game_not_found_cmd(game_id)
        if state.env is None:
            if self.card_id is None:
                self.card_id = self.adapter.open_scorecard()
            state.env = self.adapter.make(game_id, self.card_id)
            state.guid = uuid.uuid4().hex
            state.play = 1
            state.started = False
            self._by_guid[state.guid] = game_id
        elif new_play:
            # Only non-competition mode reaches here.
            state.env = self.adapter.make(game_id, self.card_id or "")
            old, state.guid = state.guid, uuid.uuid4().hex
            self._by_guid.pop(old or "", None)
            self._by_guid[state.guid] = game_id
            state.play += 1
            state.started = False

        if not state.started:
            # A game's first RESET returns the observation available right at make, without another
            # step. This matches VISTA's runner, so the two lines' scores are comparable.
            state.started = True
            obs = self.adapter.initial_obs(state.env)
            self._record(game_id, "START", False, state, obs, via)
            return self.adapter.render(obs, state.guid or "")
        obs = self.adapter.step(state.env, "RESET", None, None)
        self._record(game_id, "RESET", True, state, obs, via)
        return self.adapter.render(obs, state.guid or "")

    def _act(self, game_id: str, state: _Game, action: str,
             body: dict[str, Any], via: str) -> dict[str, Any]:
        x = body.get("x") if action == COORD_ACTION else None
        y = body.get("y") if action == COORD_ACTION else None
        if action == COORD_ACTION and (x is None or y is None):
            # Self-written clients easily wrap the coordinates as {"data": {x, y}}. This used to go
            # all the way to the engine, return None, become a ServerError and drop the connection;
            # the agent only saw RemoteDisconnected. Say exactly what is wrong, in the official error shape.
            raise ApiFault(
                400, E.APIError.VALIDATION_ERROR,
                f"missing `x`/`y` for {COORD_ACTION}; they go at the top level "
                "of the body, not inside `data`")
        state.started = True
        obs = self.adapter.step(state.env, action, x, y)
        self._record(game_id, action, True, state, obs, via, x=x, y=y)
        return self.adapter.render(obs, state.guid or "")

    def _record(self, game_id: str, action: str, counted: bool, state: _Game,
                obs: Any, via: str, x: int | None = None,
                y: int | None = None) -> None:
        self.ledger.record(game=game_id, action=action, counted=counted,
                           play=state.play, guid=state.guid, card_id=self.card_id,
                           obs=obs, via=via, x=x, y=y)
