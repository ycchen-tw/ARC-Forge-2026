"""Driver adapter protocol. **One arm = one file, ``arms/<name>.py``.**

The four arms must measure differences in agent capability, not in the harness; so the gateway,
ledger, fence and prompt content are all shared, and drivers may differ only in the few surfaces
defined here:

Module attributes
  ``NAME``            matches the key in ``manifest.CLI_PINS``
  ``VERSION``         the pinned version; checked against ``installed_version()`` at launch
  ``CAPABILITIES``    ``frozenset`` ⊆ :data:`CAPABILITIES`. The launcher uses it for a start-up check
                      of the config: config asking for ``resume_mode=thread`` while the driver lacks
                      ``resume`` is a SystemExit, not a silent fallback to fresh.
  ``BRIEF_FILE``      the file name the driver auto-loads and re-injects after compaction
                      (``AGENTS.md`` for both codex and pi), or ``None`` -- the latter is recorded in
                      ARM_STATE as the named degradation ``brief_delivery=prompt-only``.

Functions
  ``prepare(driver_home, ws, knobs)``
      Writes the driver's own config file (config.toml / models.json / settings.json).
  ``env(driver_home, ws, paths) -> dict[str, str]``
      Variables to setenv inside the jail. jail.sh knows nothing about drivers; it passes these one
      per line as ``--setenv``, after its own defaults, so a ``PATH`` given here overrides the default.
  ``mounts(paths) -> list[Path]``
      Paths besides driver_bin to mount read-only into the jail (pi needs the Node 22 prefix).
  ``argv(task, knobs) -> list[str]``
      The command that starts the driver inside the jail. task is the full brief text.
  ``resume_argv(handle, nudge, knobs)`` (only when ``"resume" in CAPABILITIES``)
      Reattaches to the same conversation: handle is the id carried by the previous session's
      ``started`` event.
  ``events(log) -> Iterator[Event]``
      Reads the driver's output into the unified vocabulary. **An unrecognized envelope must raise**
      :class:`UnknownStream`, never yield nothing -- after a driver switch, the scanner failing to
      read a new shape and silently returning CLEAN has already happened once (2026-09-16).
  ``brief_addendum() -> str``
      Only the driver's **structural** differences (e.g. "each command runs in a new subshell").
      Counted into the prompt fingerprint.
  ``installed_version(paths) -> str``
      The actually installed version string.
  ``digest(driver_home) -> Iterator[dict]`` (optional)
      At the end of a game, extracts from the driver's own records what the ``--json`` event stream
      lacks: the raw compaction summaries, every tool call (including which image view_image looked
      at), token usage. Once the launcher has written ``driver_digest.jsonl`` the whole driver home
      can be deleted (codex is 30 MB per game, and oversized Kaggle output makes the submission end
      straight in system error).
  ``effective(knobs) -> dict`` (optional)
      The knobs the driver actually applies. pi does not read ``context_window`` and derives three
      budgets itself, so ARM_STATE recording only the config values would mislead (caught in another
      session's review on 2026-09-19). If present, recorded into ``knobs_effective``; if absent,
      marked ``source: config``. Not in REQUIRED_FUNCS.

Unified event vocabulary (:data:`KINDS`)
  ``started``     data.handle = this conversation's id (thread id / session id)
  ``command``     text = the shell command that ran
  ``tool``        text = ``<name> <json args>``; required for **every** tool call, shell included --
                  a reader that only picks payload keys misses ``read_file`` reading the corpus
  ``reasoning``   text = thinking
  ``message``     text = what was said to the user
  ``usage``       data = token usage reported by the driver (verbatim)
  ``compaction``  data.reason
  ``ended``       data.reason
"""

from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent

CAPABILITIES = frozenset({"resume", "auto_compact", "effort", "brief_file",
                          #: the config's `driver_config` can become the driver's own settings
                          #: verbatim (codex's `-c key=value`). For a driver without this
                          #: capability, the launcher refuses to run when config sets driver_config.
                          "driver_config",
                          #: the config's `driver_files`: driver setting key -> file inside the unit
                          #: (codex's base instructions, compaction prompt).
                          "driver_files"})
KINDS = ("started", "command", "tool", "reasoning", "message", "usage", "compaction", "ended")

REQUIRED_ATTRS = ("NAME", "VERSION", "CAPABILITIES", "BRIEF_FILE")
REQUIRED_FUNCS = ("prepare", "env", "mounts", "argv", "events", "brief_addendum",
                  "installed_version")


@dataclass(frozen=True)
class Knobs:
    """Every knob the launcher hands the adapter. One that can't be translated must raise, not be swallowed."""
    model: str
    base_url: str          #: model endpoint as seen inside the jail (127.0.0.1:<model_port>/v1)
    effort: str
    context_window: int    #: context size the driver uses for its own accounting (codex's model_context_window)
    auto_compact: int      #: compaction once cumulative usage passes this value
    server_context: int    #: the server's real context_length (host-side /get_server_info)
    #: Driver-specific extra settings (codex's ``-c key=value``). The config's ``driver_config``
    #: is passed through verbatim and the adapter decides how to translate it -- **if it can't, it
    #: must raise**, never silently ignore, or the whole round runs believing a setting is applied when it isn't.
    driver_config: tuple[tuple[str, str], ...] = ()
    #: driver setting key -> absolute path (the config's ``driver_files``; the launcher has verified each exists and is non-empty).
    driver_files: tuple[tuple[str, str], ...] = ()
    #: gen-2: codex hooks are not a managed source; under exec without ``--dangerously-bypass-hook-trust`` they are **silently skipped**.
    hook_trust_bypass: bool = False


@dataclass(frozen=True)
class Paths:
    agent_runtime: Path    #: the agent's python venv (first on PATH inside the jail)
    driver_bin: Path       #: the driver's install prefix (jail.sh always mounts it)
    runtime: Path | None   #: extra runtime prefix the driver needs (pi needs Node 22); None if not needed


@dataclass(frozen=True)
class Event:
    kind: str
    text: str = ""
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"event kind {self.kind!r} not in {KINDS}")


class UnknownStream(RuntimeError):
    """A JSON line in the driver output is an envelope this adapter does not recognize.

    Never catch it and treat it as "nothing happened". A record that can't be read is a check not done.
    """


def available() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.py") if not p.name.startswith("_"))


def check(module) -> None:
    """Check the protocol is complete at load time; anything missing is a start-up failure, not an AttributeError three hours in."""
    missing = [a for a in REQUIRED_ATTRS + REQUIRED_FUNCS if not hasattr(module, a)]
    if missing:
        raise TypeError(f"arms/{module.NAME if hasattr(module, 'NAME') else module.__name__}: missing {missing}")
    for f in REQUIRED_FUNCS:
        if not callable(getattr(module, f)):
            raise TypeError(f"arms/{module.NAME}.{f} is not a function")
    caps = module.CAPABILITIES
    if not isinstance(caps, frozenset) or not caps <= CAPABILITIES:
        raise TypeError(f"arms/{module.NAME}.CAPABILITIES={caps!r} must be a subset of {sorted(CAPABILITIES)}.")
    if "resume" in caps and not callable(getattr(module, "resume_argv", None)):
        raise TypeError(f"arms/{module.NAME} declares resume but has no resume_argv()")
    if module.BRIEF_FILE is not None and not isinstance(module.BRIEF_FILE, str):
        raise TypeError(f"arms/{module.NAME}.BRIEF_FILE must be a file name or None")


def load(name: str):
    if name not in available():
        raise SystemExit(f"no arms/{name}.py; available: {available()}")
    module = importlib.import_module(f"arms.{name}")
    check(module)
    if module.NAME != name:
        raise TypeError(f"NAME in arms/{name}.py is {module.NAME!r}")
    return module


def json_lines(log: Path) -> Iterator[dict]:
    """The driver's stdout is mixed with non-JSON stderr (codex's "Reading additional input...",
    the jail's ARCSEC messages); those are skipped. A line that **is** JSON must be a dict."""
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue          # truncated last line (driver killed mid-write)
        if not isinstance(event, dict):
            raise UnknownStream(f"{log}: JSON is not an object: {line[:120]}")
        yield event


def session_handle(driver, log: Path) -> str | None:
    """The previous session's conversation id. None only means the driver never started."""
    for event in driver.events(log):
        if event.kind == "started":
            handle = event.data.get("handle")
            if not handle:
                raise RuntimeError(f"{driver.NAME}: started event has no handle ({log})")
            return handle
    return None


def run_version(argv: list[str], env: dict[str, str]) -> str:
    out = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"{argv} exited {out.returncode}: {out.stderr.strip()[:300]}")
    return out.stdout.strip()
