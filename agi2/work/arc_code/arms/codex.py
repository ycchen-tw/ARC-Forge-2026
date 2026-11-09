"""Driver adapter for the codex CLI. Protocol: see ``arms/__init__.py``.

This is all an arm should contain: how to start the driver and what its events look like.
Everything else (gateway, ledger, fence, scoring, prompt content) is shared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from arms import Event, Knobs, Paths, UnknownStream, json_lines, run_version

NAME = "codex"
#: manifest.CLI_PINS["codex"]
VERSION = "0.154.0"
#: codex 0.154 removed the chat wire_api; only responses is left.
WIRE_API = "responses"
CAPABILITIES = frozenset({"resume", "auto_compact", "effort", "brief_file",
                          "driver_config", "driver_files"})
#: Keys ``driver_files`` recognizes. All are codex settings of the form "path points to a file":
#: ``model_instructions_file`` replaces the base instructions codex gives a generic coding assistant;
#: ``experimental_compact_prompt_file`` is the last message of the compaction request.
#: For both, a missing or empty file makes codex refuse to start -- better than the inline
#: ``compact_prompt``, where an empty string is silently swapped back to the default.
#: ``tool_output_tokenizer`` (patch 0007) is the served model's ``tokenizer.json``:
#: ``tool_output_token_limit`` then counts real tokens. If it can't be read or parsed codex refuses
#: to start; a binary without 0007 also refuses under ``--strict-config``. The jail must be able to
#: see it (on Kaggle /kaggle is closed off before the game starts, so it has to be copied out first).
FILE_KEYS = frozenset({"model_instructions_file", "experimental_compact_prompt_file",
                       "tool_output_tokenizer"})
#: codex injects the cwd's AGENTS.md as user instructions every turn, and it survives compaction.
BRIEF_FILE = "AGENTS.md"

#: --json envelopes. An unrecognized one must raise, never be silently skipped.
_TOP = {"thread.started", "turn.started", "turn.completed", "turn.failed",
        "item.started", "item.updated", "item.completed", "error"}


def settings(knobs: Knobs) -> list[str]:
    """Settings go through ``-c key=value``, not a written config.toml.

    config.toml has an easy trap: once a ``[model_providers.local]`` table header appears,
    **every key after it belongs to that table**, including ``model_provider``, which should be
    top-level. So the provider setting has no effect, codex falls back to calling
    ``wss://api.openai.com``, and the jail has no network -- the error only says
    "failed to lookup address information", pointing nowhere near the real cause.
    ``-c`` is flat and has no such problem.

    ``model_context_window`` must be stated explicitly because codex can't read sglang's
    ``/v1/models`` metadata. ``model_auto_compact_token_limit`` is the compaction gate:
    if usage is past it at the start of a turn, compaction runs first.
    """
    return [
        "-c", "model_providers.local.name=sglang-local",
        "-c", f"model_providers.local.base_url={knobs.base_url}",
        "-c", f"model_providers.local.wire_api={WIRE_API}",
        "-c", "model_providers.local.env_key=OPENAI_API_KEY",
        "-c", "model_provider=local",
        "-c", f"model_reasoning_effort={knobs.effort}",
        "-c", f"model_context_window={knobs.context_window}",
        "-c", f"model_auto_compact_token_limit={knobs.auto_compact}",
        # Sub-agents (namespace tool multi_agent_v1) and OpenAI-hosted web_search are on by default;
        # sglang currently only forwards function/custom tools, so both are dropped "by luck". Turn
        # them off explicitly so the agent doesn't suddenly gain abilities (and a bigger prompt)
        # once sglang fixes that.
        "-c", "features.multi_agent=false",
        "-c", 'web_search="disabled"',
        # Same set as CODEX_ARC_FLAGS in arc_agi3/arc_code_qwen/run_arm.py (the part stock 0.154
        # recognizes): the three goals tools, the permissions block ("Network access is enabled"
        # contradicts the rules), the bundled skills directory, request_user_input (only useful in
        # Plan mode). view_video / video modality / session release / compaction_reuse_prefix /
        # compaction_strip_analysis need a self-built codex (patches/codex-0.154.0); stock refuses
        # them under --strict-config, so they go through driver_config.
        # Tool output longer than this many tokens is cut from the middle; 2500 is codex's implicit
        # default for an unknown model name (10,000 bytes / 4), written out so it is visible and a
        # change is one line. Without driver_files' tool_output_tokenizer this is still 10,000 bytes
        # (a board is about 1 byte/token, so ~10k tokens).
        "-c", "tool_output_token_limit=2500",
        "-c", "features.goals=false",
        "-c", "include_permissions_instructions=false",
        "-c", "skills.bundled.enabled=false",
        "-c", "tools.experimental_request_user_input.enabled=false",
        # The config's driver_config is passed down verbatim. Together with `--strict-config` (see
        # _common), a mistyped key makes codex refuse to start **at launch** instead of being quietly
        # ignored -- "thought it was set but it wasn't" for a four-hour run is the costliest failure
        # mode here.
        *[arg for key, value in knobs.driver_config
          for arg in ("-c", f"{key}={value}")],
        *[arg for key, path in _files(knobs)
          for arg in ("-c", f"{key}={json.dumps(path)}")],
    ]


def _files(knobs: Knobs) -> tuple[tuple[str, str], ...]:
    unknown = sorted(k for k, _ in knobs.driver_files if k not in FILE_KEYS)
    if unknown:
        raise ValueError(f"codex driver_files does not recognize {unknown}; recognized: {sorted(FILE_KEYS)}")
    return knobs.driver_files


def prepare(driver_home: Path, workspace: Path, knobs: Knobs) -> None:
    """codex stops to ask about an untrusted directory, and nobody in the jail can answer."""
    driver_home.mkdir(parents=True, exist_ok=True)
    (driver_home / "config.toml").write_text(
        f'[projects."{workspace}"]\ntrust_level = "trusted"\n')


def env(driver_home: Path, workspace: Path, paths: Paths) -> dict[str, str]:
    """``CODEX_HOME`` points at this game's driver home -- the thread's rollout lives there and
    persists across sessions, which is what lets ``exec resume`` reattach. The key is a placeholder
    string for ``env_key`` to see."""
    if paths.runtime is not None:
        raise ValueError("codex is a static binary and needs no --driver-runtime; passing one most likely means the wrong arm")
    return {"CODEX_HOME": str(driver_home), "OPENAI_API_KEY": "sglang-local"}


def mounts(paths: Paths) -> list[Path]:
    return []


def _common(knobs: Knobs) -> list[str]:
    """The part shared by ``argv`` and ``resume_argv``. Their settings must be identical word for
    word -- one flag of difference means measuring the harness, not the agent.

    - Inside the jail ``--dangerously-bypass-approvals-and-sandbox`` is required: codex's own
      sandbox can't nest inside bubblewrap, and keeping it would block the agent from the loopback
      bridged in. The fence is provided by bwrap; this flag is the honest way to say so.
    - The model goes through ``-m``, not ``-c model=``.
    """
    return [
        *settings(knobs),
        "-m", knobs.model,
        "-c", "approval_policy=never",
        "-c", "sandbox_workspace_write.network_access=true",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        # An unrecognized `-c` key must blow up on the spot, not be silently swallowed. codex ignores
        # them by default, so something like `-c model_temperature=0.7` that "looks set" quietly does nothing.
        "--strict-config",
    ]


def argv(task: str, knobs: Knobs) -> list[str]:
    """Exactly the shape of upstream rig/agents.py:Codex.argv, the one verified to work.

    **task is the last positional argument and must not be preceded by ``--``**. With it, codex
    treats the prompt as missing and stops at "Reading additional input from stdin..." waiting for
    input that never comes -- and nobody in the jail can type.
    """
    return ["codex", "exec", "--json", *_common(knobs), task]


def resume_argv(handle: str, nudge: str, knobs: Knobs) -> list[str]:
    """Continue the same thread: ``codex exec resume <thread_id> <nudge>``.

    Before a new turn starts codex looks at the usage reported by the previous one: past
    ``model_auto_compact_token_limit`` it runs compaction first and then handles the nudge;
    otherwise it carries the whole history verbatim (**including the previous step's truncated
    reasoning**) and continues -- it "keeps thinking", not re-thinks.
    Both paths were verified on 2026-09-19 against real driver homes with a fake model server:
    bp35 s2 cut off by context (usage 119,994) -> the first request is compaction; cn04 s1 killed by
    OOM mid tool call -> no compaction, straight continuation. ``exec resume`` has no
    ``-C``/``-s``/``-p``, which we don't use anyway; ``--json`` and the ``-c`` overrides are all there.
    """
    return ["codex", "exec", "resume", "--json", *_common(knobs), handle, nudge]


def events(log: Path) -> Iterator[Event]:
    """Reads ``codex exec --json`` into the unified vocabulary.

    Envelopes seen in two real runs on 2026-09-18/19: thread.started / turn.started / turn.completed /
    item.started / item.completed; item.type: command_execution / reasoning /
    agent_message / file_change / error. Compaction is **not** in --json (only in the
    rollout), so this never yields ``compaction`` -- to see compaction, read the rollout under
    ``$CODEX_HOME/sessions/``.
    """
    for event in json_lines(log):
        kind = event.get("type")
        if kind not in _TOP:
            raise UnknownStream(f"{log}: unrecognized codex event {kind!r}: {json.dumps(event)[:160]}")
        if kind == "thread.started":
            yield Event("started", data={"handle": event.get("thread_id")})
        elif kind == "turn.completed":
            yield Event("usage", data=dict(event.get("usage") or {}))
        elif kind == "turn.failed":
            yield Event("ended", data={"reason": "turn.failed",
                                       "error": json.dumps(event.get("error"), ensure_ascii=False)})
        elif kind == "error":
            yield Event("ended", data={"reason": "error", "error": str(event.get("message", ""))})
        elif kind == "item.completed":
            item = event.get("item") or {}
            itype = item.get("type")
            if itype == "command_execution":
                yield Event("command", text=str(item.get("command", "")),
                            data={"status": item.get("status"), "exit_code": item.get("exit_code")})
                yield Event("tool", text="command_execution " + json.dumps(
                    {"command": item.get("command", "")}, ensure_ascii=False))
            elif itype == "reasoning":
                yield Event("reasoning", text=str(item.get("text", "")))
            elif itype == "agent_message":
                yield Event("message", text=str(item.get("text", "")))
            elif itype == "error":
                yield Event("message", text=str(item.get("message", "")), data={"error": True})
            else:
                # file_change, mcp_tool_call, web_search, todo_list... deliberately over-inclusive:
                # non-shell tools must leave a trace too, or "what did it actually do" depends on us guessing the types right.
                yield Event("tool", text=f"{itype} " + json.dumps(item, ensure_ascii=False))
        # turn.started / item.started / item.updated: no new information, skipped.


def digest(driver_home: Path) -> Iterator[dict]:
    """The three things in the rollout that ``--json`` lacks, in time order.

    - ``compaction``: the raw summary (the message of the ``compacted`` item, the one that replaces
      the whole history). The 2026-09-24 Kaggle rehearsal deleted CODEX_HOME after every wave, so
      not one of its 98 compactions was kept.
    - ``tool``: name and arguments of every tool call (cut to 400 chars) -- what ``view_image`` /
      ``view_video`` looked at is only here; ``--json`` records only shell commands.
    - ``usage``: the last cumulative token count of each rollout.

    For TITO (ledger mapped back to the game; read by ``src/arcforge/tito/manifest.py``) the
    following are recorded additionally, old fields unchanged:

    - ``usage`` rows add ``thread_id`` (the ``session_meta`` id) and ``windows``: the rollout is split
      into windows at ``compacted``; each window has ``window_number`` (0 for the first, then
      ``compacted.payload.window_number``), ``response_ids`` (``token_usage_record.response_id``, in
      order, deduplicated per window), ``compaction_response_id`` (the compaction request that ended
      this window; None if not ended yet). The row is also emitted when there are response ids but
      no ``token_count`` (``total`` is None), so no id is lost.
    - ``compaction`` rows add ``window_number`` / ``window_id`` / ``compaction_response_id`` (payload verbatim).

    This only copies, never validates: digest runs for every round, validation lives in the manifest
    that only tito runs.

    Images themselves (base64 in function_call_output) are not copied; they are the main source of size.
    """
    rollouts = sorted((driver_home / "sessions").rglob("rollout-*.jsonl")) \
        if (driver_home / "sessions").is_dir() else []
    for rollout in rollouts:
        last_usage = None
        thread = None
        windows = [{"window_number": 0, "response_ids": [], "compaction_response_id": None}]
        for line in rollout.read_text(errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # last line killed mid-write
            kind, payload = row.get("type"), row.get("payload") or {}
            stamp = row.get("timestamp")
            if kind == "compacted":
                windows[-1]["compaction_response_id"] = payload.get("compaction_response_id")
                windows.append({"window_number": payload.get("window_number"), "response_ids": [],
                                "compaction_response_id": None})
                yield {"kind": "compaction", "t": stamp, "rollout": rollout.name,
                       "summary": str(payload.get("message", "")),
                       "window_number": payload.get("window_number"), "window_id": payload.get("window_id"),
                       "compaction_response_id": payload.get("compaction_response_id")}
            elif kind == "response_item" and payload.get("type") in ("function_call", "custom_tool_call"):
                args = payload.get("arguments", payload.get("input", ""))
                yield {"kind": "tool", "t": stamp, "name": payload.get("name"),
                       "arguments": str(args)[:400]}
            elif kind == "event_msg" and payload.get("type") == "token_count" and payload.get("info"):
                last_usage = {"kind": "usage", "t": stamp, "rollout": rollout.name,
                              "total": (payload["info"] or {}).get("total_token_usage")}
            elif kind == "session_meta" and thread is None:
                thread = payload.get("id")
            elif kind == "token_usage_record":
                rid = payload.get("response_id")
                if rid not in windows[-1]["response_ids"]:
                    windows[-1]["response_ids"].append(rid)
        if last_usage is None and any(w["response_ids"] for w in windows):
            last_usage = {"kind": "usage", "t": None, "rollout": rollout.name, "total": None}
        if last_usage:
            yield {**last_usage, "thread_id": thread, "windows": windows}


def brief_addendum() -> str:
    return ""


def effective(knobs: Knobs) -> dict:
    """codex takes the config values literally: whatever -c passes is what applies."""
    return {"model_reasoning_effort": knobs.effort,
            "model_context_window": knobs.context_window,
            "model_auto_compact_token_limit": knobs.auto_compact}


def installed_version(paths: Paths) -> str:
    """``codex --version`` -> ``codex-cli 0.154.0``."""
    out = run_version([str(paths.driver_bin / "bin" / "codex"), "--version"], {"PATH": "/usr/bin:/bin"})
    return out.split()[-1]
