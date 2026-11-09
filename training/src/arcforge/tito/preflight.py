"""Harness-side TITO preflight: before a run starts, confirm the server really keeps the ledger and codex really sends
session_id.

Spec: "harness (C1, minimal wiring)" in ``docs/plans/tito_session_ledger.md``. Enabled by the config's
``tito: {"required": true}``; a config without this key never loads this module.

Three things; if any one does not hold, raise (no degrading, no default values):

- the server has replay and on-disk ledger on: ``session_replay`` in ``/get_server_info`` is true and
  ``session_ledger_dir`` is non-empty (the field names are the sglang server_args names).
- the ledger instance: ``/session_ledger/status`` returns JSON with both ``instance`` and ``dir`` present.
- codex has patch 0002's ``features.sglang_session_release`` on: without it there is no session_id, and the ledger
  only receives one segment per request (or, without ``--session-ledger-all``, nothing).

At the end of the run ``check_server_end`` reads status again (a changed instance is recorded, not raised).

stdlib only, and no import of other modules in the package: arc2 loads it by file path (``.venv`` is shared across
worktrees, and an editable install may point at another worktree).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Mapping

#: feature flag of codex patch 0002 (a driver_config key). The value must be the string ``"true"``:
#: driver_config becomes ``-c key=value`` verbatim, and a JSON true would be written as ``True``.
RELEASE_FEATURE = "features.sglang_session_release"
#: the only currently defined shape of config ``tito``.
CONFIG = {"required": True}


class TitoPreflightError(RuntimeError):
    """The ledger cannot receive this run, or receives it but cannot map it back to the game / task."""


def check_config(value: object) -> dict:
    """The config's ``tito`` value. The spec defines only ``{"required": true}``; any other shape (including
    ``required: false``) has undefined meaning and is rejected outright, without inventing semantics for it."""
    if value != CONFIG or not isinstance(value, dict) or value["required"] is not True:
        raise TitoPreflightError(f"config tito accepts only {json.dumps(CONFIG)}, got {value!r}")
    return dict(value)


def check_codex_config(driver_config: Mapping[str, object]) -> None:
    """driver_config must enable patch 0002: without it requests carry no session_id and the ledger's sid cannot be
    mapped back to a thread."""
    value = driver_config.get(RELEASE_FEATURE)
    if value != "true":
        raise TitoPreflightError(
            f"tito needs {RELEASE_FEATURE}: \"true\" in driver_config (codex patch 0002; only then do requests carry "
            f"session_id), got {value!r}")


def server_root(base_url: str) -> str:
    """``http://host:port/v1`` → ``http://host:port``. Blows up when it does not end in ``/v1``; no guessing."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        raise TitoPreflightError(f"--model-base-url must end in /v1 to derive the server root: {base_url!r}")
    return url[:-len("/v1")]


def _get_json(url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
    except urllib.error.URLError as error:
        raise TitoPreflightError(f"{url} unreadable: {error}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise TitoPreflightError(f"{url} did not return JSON: {body[:200]!r}") from error
    if not isinstance(value, dict):
        raise TitoPreflightError(f"{url} did not return a JSON object: {value!r}")
    return value


def _nonempty_str(source: str, payload: dict, key: str) -> str:
    if key not in payload:
        raise TitoPreflightError(f"{source} has no {key!r} (server is not the TITO fork, or the field was renamed): "
                                 f"{sorted(payload)[:40]}")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TitoPreflightError(f"{source}: {key} must be a non-empty string, got {value!r}")
    return value


def ledger_status(root: str, timeout: float = 120) -> dict:
    """``/session_ledger/status``: raises when unreadable, not a JSON object, or missing ``instance`` / ``dir``."""
    url = root + "/session_ledger/status"
    status = _get_json(url, timeout)
    _nonempty_str(url, status, "instance")
    _nonempty_str(url, status, "dir")
    return status


def check_server(root: str, timeout: float = 120) -> dict:
    """Assert the server has replay + on-disk ledger on; return what goes into the run state / manifest."""
    info_url = root + "/get_server_info"
    info = _get_json(info_url, timeout)
    if "session_replay" not in info:
        raise TitoPreflightError(f"{info_url} has no session_replay (server is not the TITO fork)")
    if info["session_replay"] is not True:
        raise TitoPreflightError(f"server does not have --session-replay on (session_replay={info['session_replay']!r})")
    ledger_dir = _nonempty_str(info_url, info, "session_ledger_dir")
    status = ledger_status(root, timeout)
    return {"server_root": root, "instance": status["instance"], "ledger_dir": ledger_dir,
            "status_dir": status["dir"],
            # copy every ledger-related server_arg verbatim (--session-ledger-all / logprobs / routing / max-bytes...),
            # so one can tell afterwards what this run's ledger collected.
            "server_args": {k: info[k] for k in sorted(info) if k.startswith("session_")},
            "status": status}


def check_server_end(server: dict, timeout: float = 120) -> dict:
    """At the end of the run (before writing the manifest) read status again: ``status_end``, and ``instances`` (start,
    end; only one when equal).

    A changed instance = the server restarted midway (e.g. Kaggle's ServerWatchdog): recorded, not raised; the segments
    of the second half are in another instance directory. If it restarted more than once in between, only the first and
    last are visible. Raises only when status cannot be read."""
    status = ledger_status(server["server_root"], timeout)
    return {**server, "status_end": status,
            "instances": list(dict.fromkeys([server["instance"], status["instance"]]))}


def preflight(tito: object, base_url: str, driver_config: Mapping[str, object], timeout: float = 120) -> dict:
    """Check the config and codex settings first (no network), then ask the server."""
    check_config(tito)
    check_codex_config(driver_config)
    return check_server(server_root(base_url), timeout)
