"""Driver adapter for pi (``@earendil-works/pi-coding-agent``). Protocol: see ``arms/__init__.py``.

pi is a third event vocabulary -- codex is a flat ``item.type``, Claude Code / Qwen Code are
``assistant`` + ``tool_use`` blocks, pi is a lifecycle stream: ``session`` (header),
``agent_start``, ``turn_start``, ``message_start``/``message_update``/``message_end``,
``tool_execution_start``/``_update``/``_end``, ``turn_end``, ``compaction_start``/``_end``,
``agent_end``, ``agent_settled``.

Three things aligned with codex (from reading the pi 0.85.1 source on 2026-09-19):

**Compaction.** pi's threshold is ``contextTokens > contextWindow − reserveTokens``
(``shouldCompact``); a 2026-09-19 smoke test showed it fires **mid** agentic loop
(cd82 turn 11: ``compaction_start reason=threshold`` -> keeps running), the same shape as codex's
"check usage before every turn". Three numbers are derived by ``budgets()`` from the same set of knobs:

- ``maxTokens = server_context − auto_compact − TURN_MARGIN``: pi sends
  ``max_completion_tokens=maxTokens`` with every request, and sglang returns 400 when
  ``prompt + max_new_tokens > context_len``. With the threshold at ``auto_compact`` and a 32k margin
  for one turn's growth, the prompt won't hit 400 before compaction; at the same time the output
  budget (≈129k) is far above the single-step thinking measured on 09-18 (median 27k, max 44k), so
  thinking won't be cut as it was then.
- ``reserveTokens = maxTokens``, ``contextWindow = auto_compact + maxTokens``
  -> the threshold is exactly ``auto_compact`` (the same number as codex's
  ``model_auto_compact_token_limit``), and **one turn can push context at most to exactly
  contextWindow**, never past it to be rescued by the overflow path (the first version used 80k as
  the reserve with a 129k output cap, and in that pi_aligned round 37 of 51 compactions were overflow
  rather than threshold -- caught by another session). ``reserveTokens`` also sets the output budget
  of the summary request (``0.8 ×``): it can't simply be ``server_context − auto_compact`` (162k) --
  that makes the summary request's ``prompt + 130k`` hit 262k (the first smoke test really did, 400);
  now it is 0.8 × 129k ≈ 103k, and ``budgets()`` checks
  ``auto_compact + TURN_MARGIN + summary ≤ server_context``.
  In pi ``contextWindow`` is only used for the threshold and the ``usage > contextWindow`` overflow
  check; it is not something sent to the server.
- But the summary budget must not be ≥ the threshold: post-compaction context = summary +
  ``keepRecentTokens`` (pi default 20k); if the summary really grows to the budget cap, it lands
  above the threshold right after compaction -> compacts again immediately (pointed out by another
  session). So ``maxTokens`` must also be ≤ ``(auto_compact − 20k − 10k margin) / 0.8`` (= 87,500
  at 262144/100000), which ``budgets()`` checks as an invariant. 87.5k is still twice the longest
  single-step thinking measured on 09-18 (44k).

What sglang's 400 looks like to pi: the OpenAI SDK only recognizes ``{"error": {"message"}}``, while
sglang's error has a top-level ``message``, so pi sees ``400 status code (no body)`` -- which happens
to match the Cerebras entry of the overflow pattern table, and pi treats it as overflow: one
compaction + retry. This is a fallback, not a design dependency. ``keepRecentTokens`` stays at pi's
default 20000.

**Effort.** ``--thinking <level>`` becomes ``reasoning_effort`` via ``openai-completions``, and
sglang feeds it into the chat template. Qwen3.8's template only recognizes
``xhigh``/``medium``/``low`` and raises on anything else; and pi only treats xhigh as available if
``thinkingLevelMap`` has ``xhigh`` (otherwise it clamps to high -> the template blows up). So
models.json lists these three explicitly and the adapter rejects other values.

**Brief.** pi auto-loads the cwd's ``AGENTS.md`` (``resource-loader.js:33``), like codex; the old
arm turning that off with ``--no-context-files`` was a different rule, not carried over here -- every
arm does "brief as the first prompt + also written as BRIEF_FILE".

The wire is ``openai-completions`` (chat). Confirmed on 2026-09-17 that the degeneration is not
caused by the wire (see memory qwencode-chat-wire-degeneration); the old pi arm ran on this path too.

Event volume: ``message_update`` is a per-token delta, 6 MB within a few minutes of a game, 89% of
the total; ``tool_execution_update`` is partial command output. Neither carries information that
``_end`` doesn't, so they are dropped inside the jail with ``grep -v`` by prefix -- by prefix, not by
parsing: if pi changes the key order, the filter stops working and the file grows, rather than
quietly dropping something else.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Iterator, Sequence

from arms import Event, Knobs, Paths, UnknownStream, json_lines, run_version

NAME = "pi"
#: manifest.CLI_PINS["pi"]
VERSION = "0.85.1"
CAPABILITIES = frozenset({"resume", "auto_compact", "effort", "brief_file"})
BRIEF_FILE = "AGENTS.md"

PROVIDER = "sglang"
#: reasoning_effort values accepted by sglang's chat template (Qwen3.8).
EFFORTS = ("low", "medium", "xhigh")
#: Allow-list of built-in tools. No web tools: a tool that can't be called won't show up in the records either.
TOOLS = "read,bash,edit,write,grep,find,ls"
#: Margin for one turn's growth between the threshold and sglang's 400.
TURN_MARGIN = 32_768
#: How much raw text pi keeps by default (settings.json leaves it alone); summary + this must be below the threshold.
KEEP_RECENT_TOKENS = 20_000
SUMMARY_HEADROOM = 10_000
#: Stream events dropped by prefix (see the module docstring).
DROP_PREFIXES = ('{"type":"message_update"', '{"type":"tool_execution_update"')

#: --mode json envelopes. An unrecognized one must raise.
_TOP = {"session", "agent_start", "agent_end", "agent_settled", "turn_start", "turn_end",
        "message_start", "message_update", "message_end",
        "tool_execution_start", "tool_execution_update", "tool_execution_end",
        "compaction_start", "compaction_end", "auto_retry_start", "auto_retry_end",
        "queue_update",
        # Emitted when compaction's summary call hits a retryable error (agent-session.js).
        "summarization_retry_scheduled", "summarization_attempt_start", "summarization_finished"}
#: Those carrying no information beyond pi's session events: listed explicitly so new types get caught.
_IGNORE = {"agent_start", "agent_settled", "turn_start", "turn_end", "message_start",
           "message_update", "tool_execution_update", "tool_execution_end",
           "auto_retry_start", "auto_retry_end", "queue_update",
           "summarization_retry_scheduled", "summarization_attempt_start", "summarization_finished"}
SHELL_TOOL = "bash"


def _check_effort(effort: str) -> None:
    if effort not in EFFORTS:
        raise ValueError(f"pi: effort {effort!r} not in {EFFORTS} -- Qwen3.8's chat template only recognizes these three")


def budgets(knobs: Knobs) -> dict[str, int]:
    """Three budgets derived from the same set of numbers (see the module docstring)."""
    by_server = knobs.server_context - knobs.auto_compact - TURN_MARGIN
    by_summary = int((knobs.auto_compact - KEEP_RECENT_TOKENS - SUMMARY_HEADROOM) / 0.8)
    max_tokens = min(by_server, by_summary)
    if max_tokens < 32_768:
        raise ValueError(f"pi: output cap down to {max_tokens} (server {knobs.server_context}, auto_compact "
                         f"{knobs.auto_compact}); thinking would be cut, adjust the threshold")
    summary_budget = int(0.8 * max_tokens)
    if knobs.auto_compact + TURN_MARGIN + summary_budget > knobs.server_context:
        raise ValueError("pi: the summary request during compaction would hit context")
    if summary_budget + KEEP_RECENT_TOKENS >= knobs.auto_compact:
        raise ValueError("pi: post-compaction context may still be above the threshold, causing back-to-back compaction")
    return {"contextWindow": knobs.auto_compact + max_tokens,
            "maxTokens": max_tokens, "reserveTokens": max_tokens}


def models_json(knobs: Knobs) -> dict:
    b = budgets(knobs)
    return {"providers": {PROVIDER: {
        "baseUrl": knobs.base_url,
        "api": "openai-completions",
        "apiKey": "EMPTY",                       # non-empty, or pi hides the model
        "compat": {"supportsDeveloperRole": False},
        "models": [{
            "id": knobs.model,
            "reasoning": True,
            "thinkingLevelMap": {e: e for e in EFFORTS},
            "input": ["text"],
            "contextWindow": b["contextWindow"],
            "maxTokens": b["maxTokens"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }],
    }}}


def settings_json(knobs: Knobs) -> dict:
    return {"compaction": {"enabled": True, "reserveTokens": budgets(knobs)["reserveTokens"]},
            "quietStartup": True}


def prepare(driver_home: Path, workspace: Path, knobs: Knobs) -> None:
    """Writes models.json and settings.json into this game's ``PI_CODING_AGENT_DIR``.
    Rewritten every session: the driver home is inside the workspace and the agent owns it."""
    _check_effort(knobs.effort)
    driver_home.mkdir(parents=True, exist_ok=True)
    (driver_home / "models.json").write_text(json.dumps(models_json(knobs), indent=2) + "\n")
    (driver_home / "settings.json").write_text(json.dumps(settings_json(knobs), indent=2) + "\n")


def env(driver_home: Path, workspace: Path, paths: Paths) -> dict[str, str]:
    """pi is a Node program; the host's /usr/bin/node is 18 and pi needs ≥ 22.19: ``--driver-runtime``
    gives the Node 22 prefix, which is mounted into the jail and put on PATH."""
    if paths.runtime is None:
        raise ValueError("pi needs --driver-runtime <Node ≥ 22 prefix> (e.g. ~/opt/node-v22.23.2-linux-x64)")
    node = paths.runtime / "bin" / "node"
    if not node.is_file():
        raise FileNotFoundError(f"pi: {node} does not exist")
    return {
        "PI_CODING_AGENT_DIR": str(driver_home),
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
        "PATH": f"{paths.agent_runtime}/bin:{paths.driver_bin}/bin:{paths.runtime}/bin:/usr/bin:/bin",
    }


def mounts(paths: Paths) -> list[Path]:
    if paths.runtime is None:
        raise ValueError("pi needs --driver-runtime")
    return [paths.runtime]


def _common(knobs: Knobs, *, tools: str = TOOLS, extra: Sequence[str] = ()) -> list[str]:
    """Shared by ``argv`` and ``resume_argv``; settings identical word for word.

    ``tools``/``extra`` are only for ``arms/pi_aligned.py`` -- it is another set of knobs on the same
    CLI, and copying these eight lines would make the difference between the two arms "visible only
    with a line-by-line diff". The defaults are this arm.

    ``--no-extensions`` is a security flag: extensions are arbitrary TypeScript loaded from the host
    package directory. ``--no-skills``/``--no-prompt-templates``/``--no-themes`` close doors of the
    same kind, and ``--no-approve`` makes pi never trust the workspace's ``.pi/`` (written by the agent itself).
    """
    _check_effort(knobs.effort)
    return [
        "pi", "--mode", "json",
        "--model", f"{PROVIDER}/{knobs.model}",
        "--thinking", knobs.effort,
        "--tools", tools,
        "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
        "--no-approve",
        *extra,
    ]


def _piped(argv: list[str]) -> list[str]:
    """Drops delta events by prefix with grep inside the jail. pipefail makes the pipeline report
    pi's exit code, not grep's -- grep exits 1 when it filters everything, which would look like a
    session failure."""
    command = " ".join(shlex.quote(p) for p in argv)
    patterns = " ".join(f"-e {shlex.quote('^' + p)}" for p in DROP_PREFIXES)
    return ["/bin/bash", "-c", f"set -o pipefail; {command} | grep --line-buffered -v {patterns}"]


def argv(task: str, knobs: Knobs) -> list[str]:
    """task comes after ``--``, so a leading dash isn't taken as a flag. stdin must be closed (the
    launcher uses DEVNULL): print mode merges stdin into the prompt, and if it is open it waits forever."""
    return _piped([*_common(knobs), "--", task])


def resume_argv(handle: str, nudge: str, knobs: Knobs) -> list[str]:
    """``--session <id>`` reattaches to the same session (id from the header event) and sends the nudge.

    Before sending a new prompt pi first runs ``_checkCompaction(lastAssistant)``: if the last one was
    an overflow error -> compaction first; otherwise it carries the whole history verbatim and
    continues. Session files are under ``$PI_CODING_AGENT_DIR/sessions/``, looked up by cwd, so cwd
    must still be the same workspace -- the jail's ``--chdir`` guarantees that.
    """
    return _piped([*_common(knobs), "--session", handle, "--", nudge])


def events(log: Path) -> Iterator[Event]:
    for event in json_lines(log):
        kind = event.get("type")
        if kind not in _TOP:
            raise UnknownStream(f"{log}: unrecognized pi event {kind!r}: {json.dumps(event)[:160]}")
        if kind in _IGNORE:
            continue
        if kind == "session":
            yield Event("started", data={"handle": event.get("id"), "cwd": event.get("cwd")})
        elif kind == "tool_execution_start":
            name = str(event.get("toolName", ""))
            args = event.get("args")
            if not isinstance(args, dict):
                args = {"_args": args}
            yield Event("tool", text=f"{name} " + json.dumps(args, ensure_ascii=False, default=str))
            if name == SHELL_TOOL:
                yield Event("command", text=str(args.get("command", "")))
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            for block in message.get("content") or []:
                if block.get("type") == "thinking":
                    yield Event("reasoning", text=str(block.get("thinking", "")))
                elif block.get("type") == "text":
                    yield Event("message", text=str(block.get("text", "")))
            usage = message.get("usage")
            if usage:
                yield Event("usage", data=dict(usage))
            if message.get("stopReason") in ("error", "length"):
                yield Event("message", text=str(message.get("errorMessage", "")),
                            data={"error": True, "stop": message.get("stopReason")})
        elif kind in ("compaction_start", "compaction_end"):
            yield Event("compaction", data={"phase": kind.split("_")[1],
                                            "reason": event.get("reason")})
        elif kind == "agent_end":
            yield Event("ended", data={"reason": "agent_end"})


def brief_addendum() -> str:
    return ""


def effective(knobs: Knobs) -> dict:
    """The numbers actually written into models.json / settings.json. ``context_window`` has no counterpart in pi."""
    b = budgets(knobs)
    return {"thinking": knobs.effort, "contextWindow": b["contextWindow"],
            "reserveTokens": b["reserveTokens"], "maxTokens": b["maxTokens"],
            "compaction_threshold": b["contextWindow"] - b["reserveTokens"],
            "ignored": ["context_window"]}


def installed_version(paths: Paths) -> str:
    """``pi --version`` -> ``0.85.1``. Needs Node 22 on PATH."""
    if paths.runtime is None:
        raise ValueError("pi needs --driver-runtime")
    return run_version([str(paths.driver_bin / "bin" / "pi"), "--version"],
                       {"PATH": f"{paths.runtime}/bin:/usr/bin:/bin", "PI_OFFLINE": "1",
                        "PI_SKIP_VERSION_CHECK": "1", "HOME": "/tmp"})
