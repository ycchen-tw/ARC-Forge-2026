"""Per-step timeout for the official gateway: one game's step / reset / load hanging must not take down the other
games in the same gateway.

A round's 25 games share one official gateway process (``official.py``), and game code runs in Flask's request
threads. Community games are untrusted code -- the sandbox stops them from touching files and the network, but not
from entering an infinite loop on a later level: that thread never returns and keeps eating CPU, and the agent's game
hangs until the 6-hour wall.

Approach (the official game semantics and scoring are unchanged; this only wraps the outside):

- ``install()`` wraps ``LocalEnvironmentWrapper``'s ``step`` / ``reset`` / ``_load_game_class``: register (thread,
  deadline) on entry, unregister on exit.
- A watchdog thread checks every ``POLL`` seconds; for anything past ``timeout`` seconds it throws ``StepTimeout`` into
  that thread with ``PyThreadState_SetAsyncExc``. It is a ``BaseException``: the official ``step()`` only catches
  ``Exception`` (on catching it returns None, and the api then returns a misleading "game not started" 400), so it has
  to get past that and be caught by the wrapper here.
- Once caught, this game (this wrapper object) is **disabled**: every later request immediately raises
  ``GameDisabled`` (Flask returns 500) and never enters game code again -- the game state stopped halfway and can no
  longer be trusted. Other games are unaffected. Every timeout writes one line to ``log_path`` (jsonl) and prints to stderr.
- An async exception only takes effect on returning to Python bytecode; something stuck in C (a big numpy operation,
  sleep) cannot be interrupted. If it has not returned after ``hard_factor × timeout``, ``os._exit(HARD_EXIT)``: the
  whole round fails loudly, which beats the whole round quietly hanging until the wall.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import weakref
from pathlib import Path

POLL = 0.25
HARD_EXIT = 75


class StepTimeout(BaseException):
    """Thrown by the watchdog into a stuck thread. BaseException: gets past the ``except Exception`` of the game and the official wrapper."""


class GameDisabled(RuntimeError):
    """This game timed out earlier and is disabled."""


def _async_raise(tid: int, exc: type[BaseException] | None) -> int:
    return ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(exc) if exc else None)


class Guard:
    def __init__(self, timeout: float, log_path: Path | None, hard_factor: float = 5.0) -> None:
        if not timeout > 0:
            raise ValueError(f"step timeout must be > 0, got {timeout!r}")
        self.timeout = timeout
        self.hard = timeout * hard_factor
        self.log_path = log_path
        self.lock = threading.Lock()
        self.active: dict[int, dict] = {}      # token -> {tid, start, what, game, fired}
        self.disabled: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # wrapper -> reason
        self._next = 0
        self.thread = threading.Thread(target=self._watch, name="step-guard", daemon=True)
        self.thread.start()

    def _log(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        print(f"step_guard: {line}", file=sys.stderr, flush=True)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def _watch(self) -> None:
        while True:
            time.sleep(POLL)
            now = time.monotonic()
            with self.lock:
                for tok, a in self.active.items():
                    age = now - a["start"]
                    if age > self.hard:
                        self._log({"event": "hard_exit", "game": a["game"], "what": a["what"],
                                   "seconds": round(age, 1), "t": time.time()})
                        os._exit(HARD_EXIT)
                    if age > self.timeout:
                        # Throw again every poll: a bare except in game code swallows one each time, until hard is reached
                        a["fired"] += 1
                        n = _async_raise(a["tid"], StepTimeout)
                        if n != 1:
                            self._log({"event": "async_raise_failed", "game": a["game"], "returned": n})
                            os._exit(HARD_EXIT)

    def call(self, wrapper, what: str, fn, *args, **kwargs):
        game = getattr(getattr(wrapper, "environment_info", None), "game_id", None)
        with self.lock:
            reason = self.disabled.get(wrapper)
        if reason is not None:
            raise GameDisabled(f"{game}: disabled after {reason}")
        with self.lock:
            self._next += 1
            tok = self._next
            a = self.active[tok] = {"tid": threading.get_ident(), "start": time.monotonic(), "what": what,
                                    "game": game, "fired": 0}
        result, caught = None, False
        try:
            result = fn(wrapper, *args, **kwargs)
        except StepTimeout:
            caught = True
        finally:
            # Once unregistered the watchdog throws no more; withdraw one thrown earlier but not yet delivered. Delivery can
            # land right in these lines (an async exception takes effect at any bytecode), at most once -- catch it and redo.
            for _ in range(3):
                try:
                    with self.lock:
                        self.active.pop(tok, None)
                        if a["fired"]:
                            _async_raise(a["tid"], None)
                    break
                except StepTimeout:
                    caught = True
        if caught or a["fired"]:
            # Also counts if game code swallowed the StepTimeout and then returned: the state stopped halfway and cannot be trusted
            reason = f"{what} exceeded {self.timeout:g}s"
            with self.lock:
                self.disabled[wrapper] = reason
            self._log({"event": "timeout", "game": game, "what": what, "timeout": self.timeout,
                       "raised": a["fired"], "returned": not caught, "t": time.time()})
            raise GameDisabled(f"{game}: {reason}; this game is disabled")
        return result


def install(timeout: float, log_path: Path | None, hard_factor: float = 5.0) -> Guard:
    """Wrap arc_agi's LocalEnvironmentWrapper (install only once)."""
    from arc_agi.local_wrapper import LocalEnvironmentWrapper as W

    if getattr(W, "_step_guard", None) is not None:
        raise RuntimeError("step_guard already installed")
    guard = Guard(timeout, log_path, hard_factor)
    for name in ("step", "reset", "_load_game_class"):
        orig = getattr(W, name)

        def wrapped(self, *args, _orig=orig, _name=name, **kwargs):
            return guard.call(self, _name, _orig, *args, **kwargs)

        wrapped.__name__ = name
        wrapped.__doc__ = orig.__doc__
        setattr(W, name, wrapped)
    W._step_guard = guard
    return guard
