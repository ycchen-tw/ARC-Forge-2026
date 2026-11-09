"""``tito_manifest.json``: written by the harness at the end of a run; maps the ledger's sessions / responses back to
games or tasks.

Spec: ``docs/plans/tito_session_ledger.md`` (harness C1; read by ``Manifest`` in ``join.py``). Written only when the
config has ``tito``, into the run directory (the reader uses the directory name as the run).

One manifest = the run level (harness, config path and sha256, the server instance and ledger directory preflight got;
``server.status_end`` is the status read again before writing the manifest, ``server.instances`` = [start, end], only
one when the server did not restart midway -- ``server.instance`` is always the starting one) + one item per game /
task. When one run uses several backends (``--sample-backends`` in arc2 group mode) there is also ``servers``: one
record in the same format per backend, ``servers[0]`` being ``server``; a manifest without ``servers`` = only one
backend used (readers use ``join.Manifest.servers``):

- ``threads``: codex thread ids. ``in_sessions`` = the harness's session records have it (arc3 ARM_STATE's
  ``plays[].sessions[].handle``, arc2 results' ``sessions[].handle``); ``in_digest`` = driver_digest has its
  rollout. When the two sides disagree it is recorded as is, nothing dropped or filled in -- reconciliation is
  ``audit``'s call. A thread belongs to exactly one item.
- ``windows``: per compaction window, ``window_number``, ``response_ids`` (the rollout's ``token_usage_record``,
  **excluding** compaction requests) and ``compaction_response_id`` (the compaction request that ends this window; its
  usage record in the rollout also falls in this window, and codex 0002's compaction request still carries the old
  window's session_id -- the spec lists this as unverified, so it is kept apart from ordinary requests).
  ``sglang_session_id`` = ``<thread>:<window_number>``, the session_id codex patch 0002 sends
  (``Session::current_window``).
- ``outcome``: summary given by the harness, always has ``status`` (arc2: score fields; arc3: levels, won, actions).
- optional ``task``: grouping key (``join.py``'s group / DFlash uid / GRPO grouping); absent = item id. Harnesses with
  several episodes per task (``run_arc2_seq.py``: item id ``<task>__s<k>``) write the task id, so that a task's s0 /
  s2 / refiner land in the same group; ``kind`` records the episode kind (seq's first / ind / refine_exp...). The item
  id is still the episode (results file name, workspace and outcome all go by episode).

Optional run-level ``meta`` (dict): the harness's own run description (seq: runner, task pool ``pool``, session kinds,
episodes not started).

Response ids come from driver_digest's ``usage`` rows (``digest()`` in ``arms/codex.py``), so runs with
``keep_driver_home: false`` map back too. stdlib only, no import of other modules in the package (arc2 loads it by
path).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
FILE = "tito_manifest.json"
HARNESSES = ("arc2", "arc3")


class TitoManifestError(RuntimeError):
    """The manifest input breaks the contract: even if written, it would not map back to the ledger."""


def config_identity(path: Path) -> dict:
    """Absolute path and sha256 of the config. Computed at run start (so an edit midway through the run shows)."""
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sglang_session_id(thread: str, window_number: int) -> str:
    """codex patch 0002: a request's ``session_id`` = ``<thread_id>:<window_number>``."""
    return f"{thread}:{window_number}"


def _rid(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TitoManifestError(f"{where}: response id must be a non-empty string, got {value!r}")
    return value


def digest_threads(paths: Iterable[Path]) -> dict[str, list[dict]]:
    """``usage`` rows of driver_digest*.jsonl → thread id → windows (by window_number).

    The same thread can appear in several digests (with ``tail_resume`` and the driver home kept, the later part's
    digest copies the earlier part's rollout again): response ids of the same window are unioned, in order of
    appearance; when both have ``compaction_response_id`` they must be equal (a window the earlier part had not
    compacted yet is None).
    """
    merged: dict[str, dict[int, dict]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            if row["kind"] != "usage":
                continue
            where = f"{path}:{number}"
            if "windows" not in row or "thread_id" not in row:
                raise TitoManifestError(f"{where}: usage row has no thread_id / windows -- the digest is a version from "
                                        f"before response ids were added")
            thread = row["thread_id"]
            if not isinstance(thread, str) or not thread:
                raise TitoManifestError(f"{where}: rollout has no thread id from session_meta ({thread!r})")
            windows = row["windows"]
            numbers = [w["window_number"] for w in windows]
            if numbers != list(range(len(windows))):
                raise TitoManifestError(f"{where}: window_number must count up by one from 0, got {numbers}")
            slots = merged.setdefault(thread, {})
            for window in windows:
                slot = slots.setdefault(window["window_number"], {"response_ids": [], "compaction_response_id": None})
                for rid in window["response_ids"]:
                    if _rid(rid, where) not in slot["response_ids"]:
                        slot["response_ids"].append(rid)
                closing = window["compaction_response_id"]
                if closing is not None:
                    _rid(closing, where)
                    if slot["compaction_response_id"] not in (None, closing):
                        raise TitoManifestError(f"{where}: thread {thread} window {window['window_number']} has two "
                                                f"compaction_response_ids: {slot['compaction_response_id']} / {closing}")
                    slot["compaction_response_id"] = closing
    return {thread: [{"window_number": n, "sglang_session_id": sglang_session_id(thread, n), **slot}
                     for n, slot in sorted(slots.items())]
            for thread, slots in merged.items()}


#: keys an item always has, and optional keys (an optional key, when given, must be a non-empty string)
ITEM_KEYS = {"id", "threads", "digests", "outcome"}
ITEM_OPTIONAL_KEYS = ("task", "kind")


def _check_server(server: dict) -> None:
    for key in ("instance", "ledger_dir"):
        if not isinstance(server.get(key), str) or not server[key]:
            raise TitoManifestError(f"server must be the result of preflight.check_server_end() (missing {key}): {server!r}")
    instances = server.get("instances")
    if not isinstance(server.get("status_end"), dict) or not isinstance(instances, list) \
            or not 1 <= len(instances) <= 2 or instances[0] != server["instance"]:
        raise TitoManifestError("server must carry status_end and instances from the end of the run (start first; "
                                f"preflight.check_server_end()): {server!r}")


def build(*, harness: str, config: dict, server: dict, items: list[dict], meta: dict | None = None,
          servers: list[dict] | None = None) -> dict:
    """``items``: ``{"id", "threads" (thread ids in the session records, in order), "digests" (list of Paths),
    "outcome"}``, optionally also ``task`` (grouping key) and ``kind``. ``meta``: run-level description (None = this
    key is not written). ``servers``: one per backend when several were used (``servers[0] is server``, instances all
    distinct); omitted when only one was used."""
    if harness not in HARNESSES:
        raise TitoManifestError(f"harness must be one of {HARNESSES}, got {harness!r}")
    if set(config) != {"path", "sha256"}:
        raise TitoManifestError(f"config must be the result of config_identity(), got {config!r}")
    _check_server(server)
    if servers is not None:
        if not isinstance(servers, list) or len(servers) < 2 or servers[0] != server:
            raise TitoManifestError("servers must be the records of ≥ 2 backends, the first being server (omit it when "
                                    "only one backend is used)")
        for srv in servers[1:]:
            _check_server(srv)
        firsts = [srv["instance"] for srv in servers]
        if len(set(firsts)) != len(firsts):
            raise TitoManifestError(f"duplicate instance in servers: {firsts}")
    if meta is not None and (not isinstance(meta, dict) or not all(isinstance(k, str) and k for k in meta)):
        raise TitoManifestError(f"meta must be a dict with non-empty string keys, got {meta!r}")
    out_items, seen_items, thread_owner, owner = [], set(), {}, {}
    for item in items:
        if not ITEM_KEYS <= set(item) <= ITEM_KEYS | set(ITEM_OPTIONAL_KEYS):
            raise TitoManifestError(f"item keys must be id/threads/digests/outcome "
                                    f"(optionally also {'/'.join(ITEM_OPTIONAL_KEYS)}), "
                                    f"got {sorted(item)}")
        iid = item["id"]
        if not isinstance(iid, str) or not iid or iid in seen_items:
            raise TitoManifestError(f"item id must be a unique non-empty string: {iid!r}")
        seen_items.add(iid)
        for key in ITEM_OPTIONAL_KEYS:
            if key in item and (not isinstance(item[key], str) or not item[key]):
                raise TitoManifestError(f"{iid}: {key} must be a non-empty string, got {item[key]!r}")
        if not isinstance(item["outcome"], dict) or "status" not in item["outcome"]:
            raise TitoManifestError(f"{iid}: outcome must be a dict with status, got {item['outcome']!r}")
        listed = list(dict.fromkeys(item["threads"]))
        for thread in listed:
            if not isinstance(thread, str) or not thread:
                raise TitoManifestError(f"{iid}: thread id must be a non-empty string, got {thread!r}")
        digests = [Path(p) for p in item["digests"]]
        found = digest_threads(digests)
        threads = []
        for thread in [*listed, *sorted(set(found) - set(listed))]:
            if thread_owner.setdefault(thread, iid) != iid:
                raise TitoManifestError(f"thread {thread} belongs to both {thread_owner[thread]} and {iid}")
            windows = []
            for window in found.get(thread, []):
                closing = window["compaction_response_id"]
                place = (iid, thread, window["window_number"])
                for rid in [*window["response_ids"], *([closing] if closing is not None else [])]:
                    if owner.setdefault(rid, place) != place:
                        raise TitoManifestError(f"response id {rid} belongs to both {owner[rid]} and {place}: "
                                                f"cannot map back to a single game / task")
                windows.append({**window, "response_ids": [r for r in window["response_ids"] if r != closing]})
            threads.append({"thread": thread, "in_sessions": thread in listed, "in_digest": thread in found,
                            "windows": windows})
        out_items.append({"id": iid, **{k: item[k] for k in ITEM_OPTIONAL_KEYS if k in item},
                          "outcome": item["outcome"], "digests": [str(p) for p in digests], "threads": threads})
    return {"schema_version": SCHEMA_VERSION, "harness": harness, "written": time.time(),
            "config": config, "server": server, **({} if servers is None else {"servers": servers}),
            **({"meta": meta} if meta is not None else {}), "items": out_items,
            "counts": {"items": len(out_items),
                       "threads": sum(len(i["threads"]) for i in out_items),
                       "windows": sum(len(t["windows"]) for i in out_items for t in i["threads"]),
                       "response_ids": len(owner)}}


def write(path: Path, **fields) -> dict:
    """``build()``, then write atomically (readers only ever see a complete file). Returns what was written."""
    manifest = build(**fields)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return manifest
