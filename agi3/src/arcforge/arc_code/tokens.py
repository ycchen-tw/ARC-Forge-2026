"""Token table: key -> **set of roles**.

The old version was ``tokens[value] = owner`` (``offline_arc_server.py:340``), keyed by token value.
In competition mode the organiser hands out a single key, so ``*=K`` and ``game_i=K`` land in the
same slot, the ``*`` entry is overwritten and the launcher role silently disappears -- preflight was
already working around this (``run_arm.py:355-358`` writes only ``*=preflight``).

One key can be both the launcher and a player of some game, so the value must be a set.
Without this fix it is **impossible to rehearse the competition's token shape locally**.
"""

from __future__ import annotations

from pathlib import Path

from .errors import ServerError

LAUNCHER = "*"


class TokenTable:
    """The result of parsing ``<game|*>=<token>`` line by line."""

    def __init__(self) -> None:
        self._roles: dict[str, set[str]] = {}

    def grant(self, key: str, role: str) -> None:
        if not key:
            raise ServerError("token must not be an empty string")
        self._roles.setdefault(key, set()).add(role)

    def roles(self, key: str | None) -> set[str]:
        if key is None:
            return set()
        return self._roles.get(key, set())

    def is_launcher(self, key: str | None) -> bool:
        return LAUNCHER in self.roles(key)

    def may_play(self, key: str | None, game_id: str) -> bool:
        """The launcher's key may also play -- in the competition it is the same key."""
        roles = self.roles(key)
        return bool(roles) and (game_id in roles or LAUNCHER in roles)

    def known(self, key: str | None) -> bool:
        return bool(self.roles(key))

    def __len__(self) -> int:
        return len(self._roles)

    @classmethod
    def parse(cls, text: str, *, resolve=None) -> "TokenTable":
        """``resolve`` normalises a short id to the full game id; with None, ids are taken as is."""
        table = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            owner, sep, value = line.partition("=")
            if not sep or not value:
                raise ServerError(f"token line needs '<game|*>=<token>': {line!r}")
            role = owner if owner == LAUNCHER or resolve is None else resolve(owner)
            table.grant(value, role)
        if not len(table):
            raise ServerError("token file is empty")
        return table

    @classmethod
    def load(cls, path: Path, *, resolve=None) -> "TokenTable":
        return cls.parse(path.read_text(encoding="utf-8"), resolve=resolve)
