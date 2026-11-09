"""mini-swe-agent 2.4.6 with every knob that can be aligned to codex put on. Protocol: see ``arms/__init__.py``.

**Why mini_aligned and not mini.** This arm has no stock control (the user's decision), and its
difference from stock mini-swe-agent is not just settings: ``mini_stream.py`` adds compaction and
resume, capabilities that **do not exist at all** in the package. Calling it ``mini`` would make
ARM_STATE look as if stock mini-swe-agent was measured. It was not.

The flip side of the same rule: the four knobs of ``arms/pi_aligned.py`` are settings pi **already
has**, and turning them off gives back stock pi; turning off compaction and resume here does not
give "stock mini", it gives "an arm that can't last four hours". So the honest name for this arm is
"**mini-swe-agent, completed**", not "mini-swe-agent".

## With no control group, make the independent variables countable another way

pi_aligned keeps attribution intact with two things: the stock control still exists, and a test
that uses a symmetric difference to pin the difference between the two arms down to four items.
This arm has neither, so instead: **every alignment point is one named constant**,
``effective()`` writes all constants and fingerprints back into ARM_STATE, and the table below
states how to turn each one off on its own.

| # | Alignment | Constant | Basis | To test it alone, change to |
|---|---|---|---|---|
| 1 | compaction @ threshold, keep only the summary | ``COMPACT_AT`` (= ``auto_compact``) | codex checks usage before each turn; past ``model_auto_compact_token_limit`` it compacts first and keeps only the summary | set ``--compact-at`` above ``server_context`` = never compact (= stock behavior, dies on context) |
| 2 | output budget **not sent**, given by the server | ``budgets()["max_tokens"] is None`` | codex does not send ``max_output_tokens``; sglang passes the absence through as ``max_new_tokens=None``, the total-length check skips None, and the scheduler clamps it to ``context − prompt − 1`` | give ``--max-tokens`` a fixed value (v4 derived 129,376, and that killed two vc33 sessions) |
| 3 | tool output cap | ``OBSERVATION_MAX_CHARS`` | codex ``exec_command``'s ``max_output_tokens`` defaults to 10,000 tokens (binary string: ``max_output_tokens`` sets the token budget for direct ``exec`` results. Defaults to 10000 tokens.) | set it to a number larger than any observation |
| 4 | bash timeout | ``COMMAND_TIMEOUT_SECONDS`` | codex ``exec_command`` schema hardcodes 10,000 ms | back to mini's stock 30, or the old arm's 300 |
| 5 | instruction set | ``SYSTEM_FILE`` (``mini_system_v7.md``, 14.7k chars) | the v4 one plus the sections unique to codex (Personality / Preamble / plan examples / progress updates / presenting results). Measured input-side gap: 9,578 vs 5,435 tokens on each game's first request | back to ``mini_system.md`` (the v4 one, 6,828 chars) |

Item 5 points directly at ``pi_hook/append_system.md`` -- **the same file** pi_aligned uses, not a
copy: both arms must wear the same text, or "both arms added the same thing" is a claim nobody
has verified. The ``## Validating your work`` of codex ``base_instructions`` is **deliberately not
added**: pi_aligned's first round measured its effect as negative (see the comment on
``SYSTEM_APPEND_FILE``).

## Deliberately not aligned / can't be aligned (don't assume they were missed)

- **The bash timeout is only half aligned, and stricter on the mini side.** codex's 10 seconds is a
  **default**; the model can pass ``timeout_ms`` itself to override it (never once used in 15,289
  real calls). mini's ``LocalEnvironment(timeout=)`` is global, and the model **can neither see nor
  change it** -- the ``BASH_TOOL`` schema has only one parameter, ``command``
  (``models/utils/actions_toolcall.py``), and ``DefaultAgent.execute_actions()`` passes no per-call
  timeout when it calls ``env.execute(action)``. So with 10 seconds on, mini is stricter than
  codex; this is a residual difference, not an alignment.
- **The tool output caps use different units.** codex's cap is 10,000 **tokens**, this one is 40,000
  **characters**, converted at 4 chars/token. ARC board output is digit-dense and the real ratio is
  below 4, so ``OBSERVATION_MAX_CHARS`` converted back to tokens is likely **looser than codex**. The
  direction is recorded here; to tighten, change the constant directly.
- **Brief delivery.** codex re-injects ``AGENTS.md`` as user instructions every turn; mini has no
  equivalent -- ``BRIEF_FILE=None``, and ARM_STATE records the named degradation ``brief-delivery``.
  The brief is only in the first user message, and compaction keeps ``[system, instance, summary]``,
  so it is **still in context** after compaction (``instance`` is the brief). This happens to have the
  same effect as codex but by a different mechanism; don't write it up as "aligned".
- **The summary request sends ``tool_choice="none"``; codex doesn't.** This is **required** to get a
  summary at all, not added casually: measured on 2026-09-19, with tools still present the model
  just calls bash instead of writing the summary (``finish_reason=tool_calls``, 66-char content,
  same both times). codex gets prose from the same request because its system prompt is the
  20,751-char ``base_instructions``, while this arm's is three lines. **What can be aligned is
  effort, not the tool surface.**
- **wire**: mini goes through litellm chat completions, codex through responses. Confirmed on
  2026-09-17 that the degeneration is not caused by the wire.
- **Memory cap**: OOM kills the systemd scope; all four arms share the same mechanism. Any fix goes
  to all drivers together.

## This arm's launch acceptance criteria are three numbers we cannot fake

The events ``mini_stream.py`` prints itself (compaction pairs, session reattached) **cannot serve
as acceptance** -- that is checking itself. The hard criteria are:

1. **The server-reported ``prompt_tokens``** (``usage`` inside the ``usage`` event, verbatim from
   sglang). After every ``compaction end``, the next request's ``prompt_tokens`` must be below the
   threshold; after a reattach, the first request's ``prompt_tokens`` must be far above the size of
   the brief alone (proof the history was really loaded back).
2. **Unattended action runs in ``ledger.jsonl``** (written by the middleware side, the driver can't
   touch it). The measure: within one game, only an action gap >5 seconds counts as the model coming
   back to look. pi_aligned cut the longest run from 3,347 actions/330 s to 32/62 s; codex is 45/9.3 s.
3. **``maximum context`` appears 0 times in ``driver.log``**. litellm prints it, not us; the old arm
   on 2026-09-16, with 10 each in 12/25 games, is the counterexample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from arms import Event, Knobs, Paths, UnknownStream, json_lines, run_version
from arms import pi as _pi
from arms import pi_aligned as _pi_aligned

NAME = "mini_aligned"
#: manifest.CLI_PINS["mini-swe-agent"]
VERSION = "2.4.6"
#: ``brief_file`` is not in it: mini auto-loads no file at all (see the module docstring).
CAPABILITIES = frozenset({"resume", "auto_compact", "effort"})
BRIEF_FILE = None

#: litellm provider prefix. Goes through chat completions.
MODEL_PREFIX = "openai"
#: Qwen3.8's chat template only recognizes these three; same reason as pi.
EFFORTS = _pi.EFFORTS
#: `TURN_MARGIN` is **taken directly from pi**, not a copied number: both arms' margin must be the
#: same constant, or "all other settings identical" is an unverified claim.
#: (`SUMMARY_RESERVE` used to come from pi too; no longer needed after 2026-09-20: this arm now
#: sends neither `max_tokens` nor a summary budget, both are given by the server as `context − prompt`.)
TURN_MARGIN = _pi.TURN_MARGIN

#: ``work/arc_code/``. The shim and the instruction set are both here.
UNIT_DIR = Path(__file__).resolve().parents[1]
#: The shim. Mounted read-only into the jail (see the mini_stream module docstring).
SHIM = UNIT_DIR / "mini_stream.py"

#: **This item is not alignment, it is mitigation.** codex doesn't send ``top_p`` either, and under
#: the server default (``--sampling-defaults openai`` -> ``top_p=1.0``) it is at 0.9 ppm -- so top_p
#: does not explain why codex is clean. But it sets **the absolute level for non-codex drivers**:
#:
#: 2026-09-19, same server, same thinking-heavy prompt, xhigh, 6 samples each:
#:
#:     server default top_p=1.0     23.0 ppm (2 of 6 degenerated: 89.5, 56.7)
#:     top_p=0.95 (stock value)      0.0 ppm (0/6)
#:     top_p=0.95 + top_k=20         0.0 ppm (0/6)
#:
#: and **thinking did not get shorter** (72.5k -> 76.5k chars per sample). This is the combination the
#: v2 round didn't achieve: v2 cut degeneration 4.8x but did it by "thinking less" (thinking median
#: 778->346), and the score didn't move; top_p blocks the drift without reducing thinking.
#:
#: The reason the serve script passes ``--sampling-defaults openai`` (line 35) only covers ``top_k``
#: (20 candidates is too narrow for a 248k vocab); the side effect is ``top_p`` also becoming 1.0,
#: while the model's stock value is 0.95. Only ``top_p`` is patched here; ``top_k`` is not touched.
#:
#: Statistically: at the char level 10/435k vs 0/459k is large, but per sample 2/6 vs 0/6 gives
#: Fisher p≈0.45, **underpowered**. It is used because the effect is large and the mechanism clear,
#: not because it is proven.
TOP_P = 0.95

#: **Temperature, added 2026-09-20.** Before this, every mini round ran at temperature 1.0 --
#: ``mini_stream.py``'s ``--temperature`` defaults to 1.0 and this file used not to pass it, so it
#: was **explicitly sending 1.0**. codex was at 0.7 throughout. Neither was anyone's choice; the wire
#: classes have different defaults:
#: ``--sampling-defaults openai`` makes ``get_default_sampling_params()`` return ``{}``
#: (``model_config.py:1670``), so what applies is
#: ``ResponsesRequest._DEFAULT_SAMPLING_PARAMS`` = 0.7 (``protocol.py:1567``, codex's path)
#: vs ``ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS`` = 1.0 (``protocol.py:884``, this arm).
#:
#: Four-hour official cards 64.96 -> **73.35** (+8.39; the same change on pi was +6.42, and the
#: noise floor for the same code across time windows is 1.67). What it bought is the ceiling
#: (72.23 -> 82.01), not efficiency.
#: See docs/research/arc_code_temp07_sglang0520_20260920.md.
TEMPERATURE = 0.7

#: codex exec_command's default timeout is 10,000 ms.
COMMAND_TIMEOUT_SECONDS = 10
#: codex's default budget for exec results is 10,000 tokens; converted at 4 chars/token (see the
#: module docstring for the direction of the bias).
OBSERVATION_MAX_CHARS = 40_000
#: This arm's own instruction set. **Changed on 2026-09-19 from "sharing one Task execution line with
#: pi_aligned" to this**, because the largest measured input-side difference is the amount of
#: instructions:
#:
#:     layer             codex                    mini (old)
#:     system            23,750 chars             512 chars      46x
#:     brief             12,971 (two copies)      6,480 (one)    2x
#:     resident prefix   36,721                   7,752          4.7x
#:
#: Earlier, after reading codex's 20,751 chars section by section and finding most of it is about
#: ``apply_patch`` / ``update_plan`` / how to talk to the user, we concluded "nothing to port" --
#: **and then stopped treating it as a difference**. That mistook "can't be ported verbatim" for "not
#: a difference". This file is the other route: instead of copying text, write the **substance** of
#: those kinds of codex guidance (persistence, planning discipline, verifying against the record,
#: shell hygiene, the explore-vs-commit trade-off) in terms of things that really exist in this world.
#:
#: **Keep apart which parts are alignment and which are our own judgment:**
#: - ``## Task execution`` is verbatim from codex, the same text pi_aligned uses.
#: - In ``## Working with the shell``, ``rg`` and "don't print big files into context" come from
#:   codex's ``## Shell commands`` (346 chars).
#: - ``## Validating your work`` is a **rewrite**: pi_aligned's first round measured the codex
#:   original as negative (``stopReason=length`` 156 times / 23 games; the model spent its output on
#:   self-checking before acting). Here it becomes "verify against the record **after** acting", and
#:   says explicitly not to rehearse before there is anything to check.
#: - ``## Understand before you build``, ``## Planning``, ``## When you are stuck``
#:   are **written by us**. Based on observing codex's behavior (a predecessor's close reading of
#:   ``cd82``: it formalizes the rules on level one and then writes a general solver) and mini's
#:   failure shape (``ft09``: 425 actions stuck at 1/6, debugging its own solver at turn 208, still
#:   establishing level two's basic rules at turn 414).
#:   **This is not alignment, it is prompt engineering**, and is accounted for separately.
SYSTEM_FILE = UNIT_DIR / "mini_system_v7.md"


#: The handoff summary compaction asks for. Deliberately worded as "hand over to the next person"
#: rather than "summarize the conversation": the latter yields prose that reads plausibly but has no
#: coordinates and no failure list, and this passage is all the memory there will be from here on.
#:
#: **The summary uses the session's own effort (xhigh), not lowered.** Reading codex's real rollout
#: on 2026-09-19: its compaction turn at xhigh spent only 768 reasoning tokens and 1,055 content
#: tokens and produced a 3,567-char handoff document -- so xhigh is not the problem. Sampling this
#: prompt four times on the same server gives 2,007 / 2,497 / 4,613 / 3,218 chars, bracketing
#: codex's 3,567. Switching to codex's "write a complete handoff document" wording is actually
#: **worse**: 546 / 11,934 / 3,429 / 13,224 chars -- a 13k-char summary raises the post-compaction
#: floor so high that the very next step hits the threshold again.
#: content occasionally landing entirely in the thinking channel (empty string) is the tail of this
#: distribution, handled by ``_summarise()``'s retry, not worked around by lowering effort.
SUMMARY_PROMPT = """\
You are about to lose everything above this line. Write the handoff note that the
next session will wake up with -- it is all it will have besides the files on disk.

Write it as compact structured notes, not prose. Cover, in this order:

1. What this game's rules appear to be, and which of those you have actually
   confirmed by observation versus are still guessing.
2. Where you are right now: level, what the board looks like, what the last
   action did.
3. What you have already tried that did NOT work, specifically enough that the
   next session does not spend live actions re-discovering it.
4. Which files you wrote and what each one is for. Name them exactly.
5. The single next thing to do.

Do not apologise, do not describe this instruction, do not write a preamble.
Start with the notes.
"""


def _check_effort(effort: str) -> None:
    if effort not in EFFORTS:
        raise ValueError(
            f"mini_aligned: effort {effort!r} not in {EFFORTS} -- Qwen3.8's chat template only recognizes these three")


def budgets(knobs: Knobs) -> dict[str, int]:
    """Output budget and summary budget, derived item by item the same as ``budgets()`` in ``arms/pi.py``.

    **Both output budgets are ``None`` = not sent**; sglang gives ``context − prompt − 1``.
    That is exactly codex's behavior, and the basis item 2 of the alignment table above has always stated.

    Before v4 this was implemented as "derive a constant"
    (``server_context − auto_compact − TURN_MARGIN = 129,376``), and that derivation hides an
    assumption: **the prompt never exceeds ``auto_compact + TURN_MARGIN``**.
    It doesn't hold -- the compaction decision uses the ``prompt_tokens`` the server reported for
    the previous turn, so the threshold always lags by one turn, and the upper bound on how much one
    turn can grow is ``max_tokens`` itself (the full thinking stays in history and is resent; codex
    too, the medians of "growth − output" being 501 vs 482). On 2026-09-19 two vc33 sessions of the
    official round died exactly on ``133,070 + 129,376 = 262,446 > 262,144``, 302 tokens over.
    Any fixed margin only moves that line back; not sending is what removes it.

    Only one check remains: there must still be room to write after compaction. If the threshold is
    too close to context, the length the first request after compaction can generate isn't enough
    for one turn.
    """
    headroom = knobs.server_context - knobs.auto_compact
    if headroom < TURN_MARGIN:
        raise ValueError(
            f"mini_aligned: server context {knobs.server_context} − threshold "
            f"{knobs.auto_compact} leaves only {headroom} to generate (need ≥ {TURN_MARGIN}); thinking would be cut")
    return {"max_tokens": None, "summary_budget": None,
            "compact_at": knobs.auto_compact}


def _uv_python_roots(interpreter: Path) -> list[Path]:
    """The install root of every hop on the venv's ``bin/python`` symlink chain.

    jail.sh already resolves this chain for the **agent runtime**, but not for driver_bin -- and
    mini's driver_bin is a uv venv.

    **The hop on that chain is a directory symlink, not a file one.** Measured: ``venv/bin/python`` ->
    ``…/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12``, and the **directory**
    ``cpython-3.12-linux-x86_64-gnu`` is the symlink pointing to
    ``cpython-3.12.12-linux-x86_64-gnu``. So walking only file-level ``is_symlink()`` misses the
    versioned directory, while mounting only the versioned one leaves the unversioned path actually
    written in the venv symlink nonexistent inside the jail. **Both names must be mounted.**
    On 2026-09-16 the old arm hit **exit 127, zero actions** here, which reads exactly like the
    model refusing to play.
    """
    roots: list[Path] = []

    def add(root: Path) -> None:
        for candidate in (root, root.resolve()):
            if candidate not in roots:
                roots.append(candidate)

    link = interpreter
    for _ in range(5):
        add(link.parent.parent)
        if not link.is_symlink():
            break
        target = Path(link.readlink())
        link = target if target.is_absolute() else link.parent / target
    return roots


def prepare(driver_home: Path, workspace: Path, knobs: Knobs) -> None:
    """mini has no config file -- all knobs go through the command line (see ``argv()``). This only creates directories."""
    _check_effort(knobs.effort)
    budgets(knobs)                      # failing to derive budgets is a start-up failure, not a 400 three hours in
    (driver_home / "sessions").mkdir(parents=True, exist_ok=True)
    (driver_home / "mswea").mkdir(parents=True, exist_ok=True)


def env(driver_home: Path, workspace: Path, paths: Paths) -> dict[str, str]:
    """**No PATH given.** jail.sh's default is ``<agent runtime>/bin:<driver_bin>/bin:…``, and the
    agent's own ``python`` must keep pointing at the agent runtime; putting driver_bin first would
    swap the agent's python for mini's venv, an independent variable nobody declared. Which
    interpreter the shim uses goes through ``ARCSEC_MINI_PYTHON``.
    """
    if paths.runtime is not None:
        raise ValueError("mini_aligned needs no --driver-runtime; passing one most likely means the wrong arm")
    interpreter = paths.driver_bin / "bin" / "python"
    if not interpreter.exists():
        raise FileNotFoundError(f"mini_aligned: {interpreter} does not exist (--driver-bin must point at mini's venv)")
    return {
        "ARCSEC_MINI_PYTHON": str(interpreter),
        "ARCSEC_MINI_SHIM": str(SHIM),
        # At import time the package does ``global_config_dir.mkdir()`` and load_dotenv on the
        # ``.env`` under it. In the jail HOME is the workspace, so left alone it would create
        # ``.config/mini-swe-agent/`` in the agent's workspace; pointing it at driver_home keeps it
        # out of the agent's sight, and guarantees that keys or model defaults in the host's
        # ``~/.config/mini-swe-agent/.env`` **cannot** become a setting this arm did not declare.
        "MSWEA_GLOBAL_CONFIG_DIR": str(driver_home / "mswea"),
        "MSWEA_SILENT_STARTUP": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
    }


def mounts(paths: Paths) -> list[Path]:
    """The shim plus the interpreter. ``driver_bin`` itself is not listed: jail.sh always mounts it;
    mounting twice is harmless, but listing it here would suggest it is a special need of this arm."""
    roots = _uv_python_roots(paths.driver_bin / "bin" / "python")
    return [SHIM, *(r for r in roots if r != paths.driver_bin)]


def _common(knobs: Knobs) -> list[str]:
    """Shared by ``argv`` and ``resume_argv``; settings identical word for word. One flag of
    difference means measuring the harness, not the agent."""
    _check_effort(knobs.effort)
    b = budgets(knobs)
    return [
        "--model", knobs.model,
        "--model-prefix", MODEL_PREFIX,
        "--api-base", knobs.base_url,
        "--api-key", "sglang-local",
        "--effort", knobs.effort,
        # ``--max-tokens`` and ``--summary-budget`` are **deliberately not here**: not sending them
        # is the alignment (see budgets()). Add them back only to test item 2 alone.
        "--compact-at", str(b["compact_at"]),
        "--summary-prompt", SUMMARY_PROMPT,
        "--summary-effort", knobs.effort,
        "--top-p", str(TOP_P),
        "--temperature", str(TEMPERATURE),
        "--brief-twice",
        "--system-append", SYSTEM_FILE.read_text(),
        "--observation-chars", str(OBSERVATION_MAX_CHARS),
        "--command-timeout", str(COMMAND_TIMEOUT_SECONDS),
        "--session-dir", ".driver/sessions",
    ]


def _launch(args: list[str]) -> list[str]:
    """``bash -c 'exec "$ARCSEC_MINI_PYTHON" "$@"' mini <shim> …``

    The protocol gives ``argv()`` only task and knobs, not ``paths``, so the absolute paths of the
    interpreter and the shim can only come from environment variables. ``"$@"`` rather than splicing
    the command into a string: task is the whole brief, with quotes, newlines and ``$`` in it, and
    sooner or later some brief would break a spliced string.

    ``--session-dir`` is a relative path, the same for every game (the jail's ``--chdir`` is the
    workspace), and the 25 games run in parallel -- a module-level absolute path would collide.
    """
    return ["/bin/bash", "-c", 'exec "$ARCSEC_MINI_PYTHON" "$ARCSEC_MINI_SHIM" "$@"',
            "mini_aligned", *args]


def argv(task: str, knobs: Knobs) -> list[str]:
    return _launch([*_common(knobs), task])


def resume_argv(handle: str, nudge: str, knobs: Knobs) -> list[str]:
    """Reattach to the same history. handle is the id from the ``session`` event; the history is in
    ``.driver/sessions/<id>.json`` -- driver_home is inside the workspace and persists across
    sessions, which is how jail.sh binds it anyway."""
    return _launch([*_common(knobs), "--session", handle, nudge])


#: Envelopes the shim emits. An unrecognized one must raise.
_TOP = {"session", "assistant", "tool_call", "tool_result", "usage",
        "compaction", "ended", "repair", "preflight", "format_error"}
#: Those with no slot in the unified vocabulary: observation receipts (numbers only, content not in
#: the stream), dangling tool_calls patched on reattach, preflight receipts. All three stay in the
#: raw log for probes to read.
_IGNORE = {"tool_result", "repair", "preflight"}
SHELL_TOOL = "bash"


def events(log: Path) -> Iterator[Event]:
    for event in json_lines(log):
        kind = event.get("type")
        if kind not in _TOP:
            raise UnknownStream(
                f"{log}: unrecognized mini event {kind!r}: {json.dumps(event, ensure_ascii=False)[:160]}")
        if kind in _IGNORE:
            continue
        if kind == "session":
            yield Event("started", data={"handle": event.get("id"),
                                         "resumed": bool(event.get("resumed"))})
        elif kind == "assistant":
            if event.get("thinking"):
                yield Event("reasoning", text=str(event["thinking"]))
            if event.get("text"):
                yield Event("message", text=str(event["text"]))
        elif kind == "tool_call":
            name = str(event.get("name", ""))
            args = event.get("input")
            if not isinstance(args, dict):
                args = {"_args": args}
            yield Event("tool", text=f"{name} " + json.dumps(args, ensure_ascii=False, default=str))
            if name == SHELL_TOOL:
                yield Event("command", text=str(args.get("command", "")))
        elif kind == "usage":
            yield Event("usage", data=dict(event.get("usage") or {}))
        elif kind == "compaction":
            yield Event("compaction", data={"phase": event.get("phase"),
                                            "reason": event.get("reason"),
                                            "prompt_tokens": event.get("prompt_tokens")})
        elif kind == "format_error":
            # The model returned a message with no tool call. Three in a row end the session, so this
            # is a clue to "why it stopped", not noise.
            yield Event("message", text=str(event.get("text", "")),
                        data={"error": True, "kind": event.get("kind")})
        elif kind == "ended":
            yield Event("ended", data={"reason": event.get("reason"),
                                       "error": event.get("error", "")})


def brief_addendum() -> str:
    """Only the driver's **structural** differences. Counted into the prompt fingerprint, so the difference is visible.

    All three are facts about mini, not settings: ``LocalEnvironment._run`` does
    ``subprocess.Popen(shell=True)`` each time, the only tool is the hardcoded ``BASH_TOOL``, and
    observations are truncated by ``TruncatingEnvironment``. Stating the third is necessary --
    otherwise the model infers rules from a truncated board taken as the complete one.
    """
    return (
        "Each command runs in a NEW shell: `cd` and exported variables do NOT persist "
        "between commands. Prefix with `cd <dir> && ` when you need to, or keep state "
        "in files.\n"
        "Your only tool is bash. Write files with heredocs (`cat > f <<'EOF'`).\n"
        f"Command output longer than {OBSERVATION_MAX_CHARS:,} characters is truncated in "
        "the middle before you see it, and the cut is marked. If you see that marker, you "
        "are NOT looking at the whole output -- narrow the query or write it to a file and "
        "read it in slices."
    )


def effective(knobs: Knobs) -> dict:
    """What this arm actually wears, written back into ARM_STATE.

    There is no stock control, so this is the only machine-readable source of "what exactly was it
    wearing" (see the module docstring).
    """
    b = budgets(knobs)
    return {
        "compaction_threshold": b["compact_at"],
        # None = not sent on the wire. Whoever reads ARM_STATE must see at a glance who set the
        # budget, so the source is recorded on its own line instead of leaving an opaque null.
        "max_completion_tokens": b["max_tokens"],
        "summary_budget": b["summary_budget"],
        "output_budget_source": "server: context − prompt − 1",
        "summary_effort": knobs.effort,
        "top_p": TOP_P,                      # not alignment, mitigation (see the TOP_P comment)
        "temperature": TEMPERATURE,          # independent variable (see the TEMPERATURE comment)
        "observation_max_chars": OBSERVATION_MAX_CHARS,
        "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
        "compaction_keeps": "system+instance+summary",
        "brief_delivery": "prompt-only-twice",
        "system_sha256": hashlib.sha256(SYSTEM_FILE.read_bytes()).hexdigest()[:16],
        "system_chars": len(SYSTEM_FILE.read_text()),
        "summary_prompt_sha256": hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest()[:16],
        "shim_sha256": hashlib.sha256(SHIM.read_bytes()).hexdigest()[:16],
    }


def installed_version(paths: Paths) -> str:
    """mini is not a CLI, so the version is asked of the package, not via ``--version``.

    Ask with driver_bin's own interpreter: asking the wrong interpreter returns the version of
    another mini copy on the host, and that is exactly where "installed is not what runs" errors hide best.
    """
    return run_version(
        [str(paths.driver_bin / "bin" / "python"), "-c",
         "import minisweagent; print(minisweagent.__version__)"],
        {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "MSWEA_SILENT_STARTUP": "1"})
