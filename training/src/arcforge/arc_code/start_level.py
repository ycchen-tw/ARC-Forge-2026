"""Start from level k (single-level episode, ``level_episode.py``): after the official gateway loads the named game,
put the engine on level k.

The official server has no "start at level k" interface. arcengine's ``set_level(index)`` is a public @final method, but
it resets ``_action_count`` to 0, and ``LocalEnvironmentWrapper.__init__`` calls ``reset()`` right after: a RESET with
``_action_count == 0`` takes ``full_reset()`` and the whole game goes back to level 1 (``handle_reset`` in
``base_game.py``). So two things are needed together:

- ``ONLY_RESET_LEVELS=true``: every RESET is ``level_reset()``, back to the start of **the current level**. api.py reads
  it when ``create_app`` constructs and base_game reads it on every RESET, so official.py must set it before
  ``create_app`` (gateway_jail.sh uses ``--clearenv``; it cannot be passed in from outside).
- :func:`install` wraps ``LocalEnvironmentWrapper._load_game_class``: once the game object is built, ``set_level(k-1)``
  and ``_score = k-1`` (``levels_completed`` is ``_score``; ``set_level`` leaves it alone, only ``next_level()`` adds).

The ``reset()`` in ``__init__`` is then a level_reset: back to level k's initial board, ``levels_completed = k-1``. The
middleware checks these two things and the screen fingerprint at the opening (``--level-episode``), and on mismatch the
whole episode fails -- this module only places, it does not verify.

Only the named game is affected; other games in the same gateway start from level 1 as usual.
"""

from __future__ import annotations

import os


def parse(spec: str) -> tuple[str, int]:
    """``<game>:<k>`` (k is the 1-based level)."""
    game, sep, level = spec.rpartition(":")
    if not sep or not game or not level.isdigit() or int(level) < 1:
        raise ValueError(f"--start-level wants <game>:<level>, level >= 1; got {spec!r}")
    return game, int(level)


def install(game: str, level: int) -> None:
    """Wrap arc_agi's ``LocalEnvironmentWrapper._load_game_class`` (install only once). ``game`` is a full or short name."""
    if os.environ.get("ONLY_RESET_LEVELS") != "true":
        raise RuntimeError("start_level needs ONLY_RESET_LEVELS=true before create_app: otherwise the first RESET "
                           "is a full reset and the game starts at level 1")
    from arc_agi.local_wrapper import LocalEnvironmentWrapper as W

    if getattr(W, "_start_level", None) is not None:
        raise RuntimeError(f"start_level already installed: {W._start_level}")
    orig = W._load_game_class

    def wrapped(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        gid = self.environment_info.game_id
        if gid == game or gid.split("-")[0] == game:
            g = self._game
            if g is None:
                raise RuntimeError(f"{gid}: the game class did not load; cannot start at level {level}")
            if not 1 <= level <= len(g._levels):
                raise ValueError(f"{gid} has {len(g._levels)} levels; cannot start at level {level}")
            g.set_level(level - 1)
            g._score = level - 1
        return result

    wrapped.__name__ = orig.__name__
    wrapped.__doc__ = orig.__doc__
    W._load_game_class = wrapped
    W._start_level = (game, level)
