"""Error shapes of the official ARC-AGI-3 API.

The agent practises locally and runs in the competition; the error strings are its only channel
for learning "this path is closed". One character off and practice no longer equals the real
thing -- so these match the ``reference/polyphony-arc3-prep/vendor/arc/arc_agi/`` wheel verbatim.
"""

from __future__ import annotations

from enum import Enum


class APIError(str, Enum):
    """The four values of ``arc_agi/models.py:10-14``, no more, no fewer."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    GAME_NOT_AVAILABLE_ERROR = "GAME_NOT_AVAILABLE_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    GAME_NOT_STARTED_ERROR = "GAME_NOT_STARTED_ERROR"


class ApiFault(Exception):
    """An error to be returned to the client.

    ``ServerError`` (configuration error, fail-closed) is deliberately not one of these -- that
    kind must blow up the whole run, not return a 4xx and let the agent keep playing.
    """

    def __init__(self, status: int, error: APIError, message: str) -> None:
        super().__init__(f"{status} {error.value}: {message}")
        self.status = status
        self.error = error
        self.message = message

    def body(self) -> dict[str, str]:
        return {"error": self.error.value, "message": self.message}


class ServerError(RuntimeError):
    """Fail-closed configuration error. Never raised because a client sent a bad request."""


# --- Messages verbatim from the official server -----------------------------------
# Each one maps to a row of the error-code table in docs/plans/arc_code_gateway.md and to a test.

def missing_game_id() -> ApiFault:
    return ApiFault(400, APIError.VALIDATION_ERROR, "missing `game_id` in action data")


def missing_guid() -> ApiFault:
    return ApiFault(
        400, APIError.VALIDATION_ERROR,
        "missing `guid` for any action other than RESET")


def guid_key_mismatch(game_id: str, guid: str, api_key: str) -> ApiFault:
    return ApiFault(
        400, APIError.VALIDATION_ERROR,
        f"game {game_id} with guid {guid} does not match API key {api_key}")


def game_not_found_cmd(game_id: str) -> ApiFault:
    """``/api/cmd``: no such game.

    In competition mode, reopening a game that was already played also lands here -- ``api.py:424``
    makes ``_get_or_create_environment`` return ``(None, False)``, which falls through to
    ``api.py:398-404``. **A replay and a missing game are identical on the wire**; that is official
    behaviour, not laziness on our part.
    """
    return ApiFault(400, APIError.SERVER_ERROR, f"game {game_id} not found")


def game_not_found_info(game_id: str) -> ApiFault:
    """``/api/games/<id>``: no such game. Same message, **different code** (404); see ``api.py:66-85``."""
    return ApiFault(404, APIError.SERVER_ERROR, f"game {game_id} not found")


def card_not_found(card_id: str) -> ApiFault:
    return ApiFault(404, APIError.SERVER_ERROR, f"card_id `{card_id}` not found")


def scorecard_in_competition_mode() -> ApiFault:
    return ApiFault(
        403, APIError.SERVER_ERROR,
        "cannot get scorecard that is in competition mode")


def multiple_scorecards() -> ApiFault:
    return ApiFault(
        409, APIError.SERVER_ERROR,
        "cannot open multiple scorecards in competition mode")
