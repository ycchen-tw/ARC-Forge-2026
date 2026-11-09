"""The single seam between the gateway and the game engine.

`arc_agi` / `arcengine` live in an arm-specific venv outside the repo (see CONVENTIONS and the
runtime section of `docs/plans/arc_code_gateway.md`); **the repo's .venv does not have them**.
So the gateway does not import the engine directly, only this adapter -- which lets the
conformance tests run on machines without the competition wheel, exactly where we most need them to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .errors import ServerError

#: Exactly the actions the official `server.py:87-121` accepts, no more, no fewer.
ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3",
           "ACTION4", "ACTION5", "ACTION6", "ACTION7")
#: The only one that carries coordinates.
COORD_ACTION = "ACTION6"


class EngineAdapter(Protocol):
    """Everything the gateway needs the engine to do."""

    def list_games(self) -> list[str]:
        """Full list of game ids, e.g. ``ft09-0d8bbf25``."""

    def game_info(self, game_id: str) -> dict[str, Any]:
        """Per-game metadata. **Must exclude ``baseline_actions``**, as the official server does
        (``api.py:56-64``) -- dropping that field is the very mechanism of "no baseline during the competition"."""

    def open_scorecard(self) -> str: ...

    def close_scorecard(self, card_id: str) -> dict[str, Any]: ...

    def make(self, game_id: str, card_id: str) -> Any:
        """Build a new environment instance."""

    def initial_obs(self, env: Any) -> Any:
        """The observation available right at ``make``, without another step."""

    def step(self, env: Any, action: str, x: int | None, y: int | None) -> Any: ...

    def render(self, obs: Any, guid: str) -> dict[str, Any]:
        """Render an observation into the shape of the official ``FrameData``."""


class ArcEngineAdapter:
    """The real `arc_agi` / `arcengine`. Imports are deferred so module load cannot fail."""

    def __init__(self, corpus: Path, recordings_dir: Path | None) -> None:
        if corpus.name != "environment_files":
            raise ServerError(f"corpus must be named environment_files: {corpus}")
        from arc_agi import Arcade, OperationMode  # noqa: PLC0415 — deliberately deferred

        self._save_recording = recordings_dir is not None
        self.arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(corpus),
            recordings_dir=str(recordings_dir) if recordings_dir else None,
        )
        self._info: dict[str, Any] = {}
        for info in self.arcade.get_environments():
            full = getattr(info, "game_id", None) or str(info)
            self._info[full] = info

    def list_games(self) -> list[str]:
        return sorted(self._info)

    def game_info(self, game_id: str) -> dict[str, Any]:
        info = self._info[game_id]
        return info.model_dump(
            mode="json",
            exclude={"private_tags", "level_tags", "baseline_actions"})

    def open_scorecard(self) -> str:
        return self.arcade.open_scorecard()

    def close_scorecard(self, card_id: str) -> dict[str, Any]:
        result = self.arcade.close_scorecard(card_id)
        if result is None:
            raise ServerError(f"close_scorecard returned nothing for {card_id!r}")
        # arc_agi 0.9.8 returns a pydantic model offline; one version stringified it with
        # json.dumps(default=str), so every close returned 500. See offline_arc_server.py:104-124.
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        for _ in range(4):           # double-encoded in practice; unwrap until it is not a string
            if not isinstance(result, str):
                break
            import json  # noqa: PLC0415
            result = json.loads(result)
        if not isinstance(result, dict):
            raise ServerError(
                f"close_scorecard returned {type(result).__name__}, not a mapping")
        return result

    def make(self, game_id: str, card_id: str) -> Any:
        env = self.arcade.make(
            game_id, scorecard_id=card_id,
            save_recording=self._save_recording, include_frame_data=True)
        if env is None or env.observation_space is None:
            raise ServerError(f"failed to initialise {game_id}")
        return env

    def initial_obs(self, env: Any) -> Any:
        return env.observation_space

    def step(self, env: Any, action: str, x: int | None, y: int | None) -> Any:
        from arcengine import GameAction  # noqa: PLC0415

        member = GameAction[action]
        data = {"x": x, "y": y} if action == COORD_ACTION else None
        obs = env.step(member, data)
        if obs is None:
            raise ServerError("environment returned no observation")
        return obs

    def render(self, obs: Any, guid: str) -> dict[str, Any]:
        payload = obs.model_dump(mode="json")
        # frame holds the 64x64 grid, which model_dump excludes; without re-attaching it the agent sees an empty board.
        payload["frame"] = [[[int(v) for v in row] for row in layer]
                            for layer in obs.frame]
        payload["guid"] = guid
        return payload
