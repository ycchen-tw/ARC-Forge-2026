"""pi 0.85.1 with every knob that can be aligned to codex put on. Protocol: see ``arms/__init__.py``.

The control is ``arms/pi.py`` (stock settings). **The diff between these two files is this arm's list
of independent variables** -- same CLI, same weights, same brief; the items below differ.

The divergence measured in the 2026-09-19 codex vs pi comparison (the two-card FP8 round, killed at
1h52m): over the same wall clock codex made 2,775 actions and pi 13,318 (4.8x);
**the longest run of actions without model involvement** was 45 / 9.3 s for codex and 3,347 / 330 s
for pi. codex's p95 is 7.5 s -- right under its ``exec_command`` default timeout of 10 s.
RHAE is quadratic in action count, so this line is the axis with the largest expected effect of the four.

## The four things put on

**1. bash default timeout = 10 s, overridable by the model** (``BASH_TIMEOUT_SECONDS``).
codex's ``exec_command`` schema hardcodes "Defaults to 10000 ms" -- **a default, overridable** -- so
the model never needs to think about it; in 15,289 real calls it never overrode it. pi's schema says
"no default timeout", which is what makes the model reason about it, experiment, and work around it
after hitting the wall. **The difference is in the tool description, not in capability**, so the
faithful mirror is "fill in only if not given, respect it if given".

pi has no settings key for this default; the only route is an extension's ``tool_call`` pre-hook;
``--no-extensions`` only disables **discovery**, an explicit ``--extension`` still loads.

**2026-09-19 took a detour: v2-v4 switched to clamping (``min(given ?? 10, 10)``), and it got worse.**
The reason at the time was "in v1 the model learned to pass its own timeout to get around it"
(bypass rate 1%->27%). But re-examined with "a single batch of >50 actions" as the blind-run
criterion (codex's largest batch in four hours is 32, never more):

    arm                 actions@25min  max batch  blind actions  blind share
    codex                    1,990         32            0          0.0%
    v1 (fill, overridable)   2,976         72          135          4.5%
    v2 (clamp)               4,832        841        1,605         33.2%
    v3 (clamp)               7,442      1,219        2,353         31.6%
    v4 (clamp)               3,024         96          407         13.5%
    stock pi                 8,480      1,253        4,597         54.2%

**Clamping plugged the small hole, so the model switched to the unbounded big hole of ``nohup &``** --
far worse than it asking for a 180 s timeout in the first place. Nor did it buy levels
(v1 35, v2 37, v4 34, all within 35±2).

**Can't be blocked, and deliberately not blocked**: ``nohup`` / ``setsid`` background execution. codex
has the same ability but chooses not to use it (max batch 32); adding a restriction codex doesn't have
would not be alignment, it would be inventing an advantage.

**2. Tool surface cut down to bash only** (``TOOLS``). In real runs codex only used ``exec_command``
(219) and ``write_stdin`` (6); stock pi has 7. ``--tools`` shrinks **both** the tool schema sent to the
model and the "Available tools" list in the system prompt (``agent-session.js:659-673`` rebuilds the
system prompt from the same list), so this is real alignment, not just blocking calls. The cost is
that writing files needs heredocs -- exactly what codex does (it never used ``apply_patch`` either).
The bash tool has its own 2000-line / 50 KB truncation, so it doesn't fall back to "unbounded tool output".

**3. What is kept after compaction** (``KEEP_RECENT_TOKENS``). **This item left "alignment"; it is now
decided by the mechanism.** codex keeps only the summary, so v2-v9 all set 0; but 0 makes pi's
mid-turn compaction **silently fail** -- ``findCutPoint``'s scan stops on the trailing ``toolResult``,
which is not a valid cut point, so ``prepareCompaction`` returns ``undefined`` every time and emits no
event. For the full mechanism, probe output and the computable lower bound, see the
``KEEP_RECENT_TOKENS`` comment. Now set to 20,000.

**4. The system prompt only adds codex's persistence clause** (``pi_hook/append_system.md``).
**codex's 20,751 chars are not ported wholesale**: they mention tools pi doesn't have
(``apply_patch``/``update_plan``), and 4,066 chars are about markdown formatting of the final reply.

The first round ported both ``## Task execution`` and ``## Validating your work``;
**the second one's effect was negative and it has been removed**: with it, ``stopReason=length``
appeared **156 times** (23/25 games; stock pi: **0**) -- the model used up its output tokens muttering
self-checks before acting and never produced the tool call; ``agent_end.willRetry`` rose from 3.8% to
12.2%. And the "don't quit early" it was meant to achieve **can't be measured**: the harness itself has
a "nudge when no action was sent this turn" fallback, and both pi variants used it about equally
(both ~1.1% of turns).

**5. Compaction happens mid-turn and doesn't end the session** (``COMPACT_THRESHOLD`` = 100,000,
= codex's ``model_auto_compact_token_limit``). It goes through pi's own
``_compactBeforeNextAssistantResponse``: hooked on ``agent.prepareNextTurnWithContext``, called on
every tool turn by the **inner** loop at ``agent-loop.js:90``, deciding on the usage-backed real
context, and going straight on to the next ``turn_start`` after compaction -- no ``abort()``, no
session restart. The only condition for this path to stay alive is a large enough
``KEEP_RECENT_TOKENS`` (see item 3).

The hook's ``message_end`` handler is still there, but demoted to a **fallback gate**
(``REAL_COMPACT_LIMIT`` = 180,000): it reads the server-reported ``usage`` and normally should never
fire; firing means the main gate failed, and its ``ctx.compact()`` ends the session in exchange for not
hitting the real 262,144 window. v9 mistook it for the main gate, at the cost of 91 session restarts in
56 minutes. The accounting and the four detours are in the ``REAL_COMPACT_LIMIT`` comment.

**6. Per-turn output cap lowered to 32,768** (``MAX_OUTPUT_TOKENS``, see ``models_json()``).
This is not "aligning to codex" but fixing a bug the first round exposed: the 129,376 given by
``budgets()`` in ``arms/pi.py`` lets one turn push context from below the threshold past the window, so
compaction degraded from threshold to overflow (37/51; stock is 253/253, all threshold).

## Deliberately not aligned (don't assume they were missed)

- **Where and how often the brief is injected**: codex re-injects ``AGENTS.md`` as user instructions
  every turn; pi puts it once into the system prompt's ``<project_context>``. pi has no equivalent.
- **wire**: responses vs chat. Confirmed on 2026-09-17 that the degeneration is not caused by the wire.
- **``max_completion_tokens``**: codex doesn't send it, sglang gives ``context_length − prompt``;
  pi sends ``budgets()["maxTokens"]`` with every request.
- **How many tools one turn can call in parallel**: codex almost always does one at a time; pi has no limit.

## Why the hook keeps its own receipts

``tool_execution_start`` carries ``toolCall.arguments`` (the model's **raw** arguments), and it is
emitted **before** ``prepareToolCall`` -> ``beforeToolCall`` (``agent-loop.js:264-269``,
``:410-418``). So **the injected timeout never shows up in ``driver.log``** -- using it as acceptance
evidence would judge "hook in effect" as "hook not in effect". And if the hook fails to **load**, pi
prints human-readable text to stderr via ``ExtensionErrorListener``, ``run_arm.py`` merges stderr into
``driver.log``, and ``json_lines()`` silently skips lines not starting with ``{`` -- **the whole arm
quietly degrades into stock pi**. So the hook appends a line to ``ARCSEC_PI_HOOK_RECEIPT`` every time;
no receipt means not in effect, and this is the arm's only launch acceptance criterion.

(If the hook throws inside a handler, pi treats it as ``block`` -- "Extension failed, blocking
execution" -- and every bash call is blocked. So the receipt-writing part is wrapped in try/except:
a bookkeeping failure must not block execution, but it must shout on stderr.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arms import HERE as UNIT_ARMS, Knobs, Paths
from arms import pi as _pi

UNIT = UNIT_ARMS.parent

NAME = "pi_aligned"
#: Same CLI; manifest.CLI_PINS["pi"]
VERSION = _pi.VERSION
CAPABILITIES = _pi.CAPABILITIES
BRIEF_FILE = _pi.BRIEF_FILE

#: codex exec_command's default. **Seconds** -- pi's schema is in seconds, codex's timeout_ms in ms.
#: Semantics are "fill in only if not given, respect it if given", like codex; clamping was tried and was worse, see the module docstring.
BASH_TIMEOUT_SECONDS = 10
#: In real runs codex only used exec_command / write_stdin.
TOOLS = "bash"
#: **This number is not a preference, it is a hard condition for pi's mid-turn compaction to work. 0 makes compaction silently fail.**
#:
#: Mechanism (confirmed on 2026-09-19 by layer-by-layer probes on an instrumented copy of pi):
#: ``findCutPoint`` (``compaction.js:293-330``) accumulates tokens backwards from the last entry,
#: stops at the entry where it reaches ``keepRecentTokens``, then looks for a valid cut point at a
#: position ≥ that one. With 0 the budget is reached at the very first entry -- and **the last entry
#: mid-turn is always a ``toolResult``**, for which ``isCutPointMessage`` (``compaction.js:227-240``)
#: returns ``False``. With no cut point found it falls back to ``cutPoints[0]``, then skips forward
#: past the two empty entries ``model_change`` / ``thinking_level_change``, ``cutIndex`` becomes **0**,
#: ``messagesToSummarize`` is therefore empty, ``prepareCompaction`` returns ``undefined``, and
#: ``_runAutoCompaction`` **returns False without emitting any event**. No trace at all.
#:
#: This is the real cause of "pi doesn't compact mid-turn" in the three rounds v7/v8/v9. **The
#: attribution written here before, "pi only checks compaction at the end of an agent run", was
#: wrong**: ``_compactBeforeNextAssistantResponse`` is hooked on ``agent.prepareNextTurnWithContext``,
#: the **inner** loop at ``agent-loop.js:90`` calls it on every tool turn, and the
#: ``estimateContextTokens`` it gets is the usage-backed real value. The decision was always right; it
#: just failed silently every time.
#:
#: Probe (``keepRecentTokens`` 0 and 20,000, the same task of 8 ``cat`` calls):
#:
#:     0       rac prep=false x3 (mid-turn) -> agent_end -> prep=true x1 (after the run ends)
#:     20,000  rac prep=true x4, each compaction_end followed directly by turn_end,
#:             only one agent_end overall -- compacted, but the session did not end
#:
#: **The lower bound is computable**: the scan must get past the trailing ``toolResult`` to land on an
#: assistant entry. pi's bash tool truncates output at 2000 lines / 50 KB, ≈12,800 tokens, so
#: ``KEEP_RECENT_TOKENS`` must be larger than that (see the guard in ``budgets()``).
#:
#: **Four-hour measurement (v9's 0 vs v10's 20,000, everything else identical):**
#:
#:     official score         55.63 -> 69.14     per-game paired +1.04 levels/game t=+3.32 p=0.006
#:     mid-turn compaction    0/312 -> 351/358
#:     session restarts       97/hour -> 9.8/hour
#:     ``abort``              169 -> 0
#:     levels                 135 -> 161 (codex v4: 156)
#:     games cleared          11 -> 17 (codex v4: 17)
#:
#: **The cost is that it traded for another problem**: once fixed, pi no longer dies from
#: compaction interruptions, so it can grind indefinitely on levels it can't clear -- actions
#: 12,369 -> 23,820, efficiency loss 9.36 -> 14.62 (codex: 2.37). The remaining 9.5-point gap of v10
#: vs codex is **entirely efficiency** (the ceiling, 83.77, already exceeds codex's 80.99).
#:
#: 20,000 was chosen because 0 can't be used, not because it is optimal.
#: **The 2026-09-19 v2/v3 comparison (0 vs 20,000, 9 levels apart) is void** --
#: neither of those rounds actually had mid-turn compaction, so the premise doesn't hold. The real
#: effect of the keep amount is to be re-measured with a three-point design (13,000 / 20,000 / 40,000).
KEEP_RECENT_TOKENS = 20_000
#: Conservative upper bound, in tokens, of pi's bash tool output truncation cap (2000 lines / 50 KB).
#: ``KEEP_RECENT_TOKENS`` must exceed it, or the mid-turn cut-point scan can't get past the trailing toolResult.
MAX_TOOL_RESULT_TOKENS = 12_800
#: Per-turn output cap. **Must not use arms/pi.py's budgets()["maxTokens"]** (see the module docstring).
MAX_OUTPUT_TOKENS = 32_768
#: **This item is not alignment, it is a deliberate deviation** (off by default, see ``SAMPLING_PARAMS``).
#: codex sends no sampling parameters at all, so it gets the server's **class defaults**.
#: ⚠️ **Correction 2026-09-20**: the old version here said "= temperature 1.0 / top_p 1.0 (sglang help
#: text)", taking the ``--sampling-defaults`` help string for the values codex actually got. The help is
#: a generic description; what really applies is **each wire's own** ``_DEFAULT_SAMPLING_PARAMS``
#: (``--sampling-defaults openai`` makes ``ModelConfig.get_default_sampling_params()`` simply
#: ``return {}``, ``model_config.py:1670``, so generation_config.json doesn't come in at all):
#:
#:     /v1/responses (codex)            temperature **0.7**   protocol.py:1567
#:     /v1/chat/completions (pi/mini)   temperature **1.0**   protocol.py:884
#:
#: That is, codex has always run at 0.7 and this arm at 1.0, **neither chosen by anyone**.
#: After changing to 0.7, four-hour official cards 67.47 -> **73.89** (+6.42; the noise floor for the
#: same code across time windows is 1.67; the same change on mini was +8.39). See
#: docs/research/arc_code_temp07_sglang0520_20260920.md.
#: Qwen3.8's
#: ``generation_config.json`` recommends ``top_p 0.95 / top_k 20``.
#: A 248k vocab + temperature 1.0 + no tail truncation at all is the most plausible recipe for leaking
#: rare tokens, and **output degeneration is the only dimension still two orders of magnitude off codex
#: after configuration alignment**: Cyrillic characters per million thinking chars, codex 0.9,
#: pi_aligned 108-228, mini 348, stock pi 793.
#: Sending ``top_p`` only affects this arm (client side), without touching the serving recipe shared by
#: the three arms. The path is verified live: ``models.json`` -> ``provider-composer`` ->
#: ``model.samplingParams`` -> ``buildBaseOptions`` -> ``Object.assign(params, samplingParams)`` in ``buildParams``.
SAMPLING_PARAMS: dict[str, float] = {"top_p": 0.95, "temperature": 0.7}

#: **Fallback gate; normally it should never fire.** The hook's ``message_end`` handler reads the
#: server-reported ``usage`` and calls ``ctx.compact()`` only when past the line. The main gate is
#: ``COMPACT_THRESHOLD`` (pi's own mid-turn compaction, see below); this 180,000 only guards the real
#: 262,144 window when the main gate fails.
#:
#: **Before v9 it was the main gate, a product of misdiagnosis.** The attribution then was "pi only
#: checks compaction at the end of an agent run" -- wrong; the real cause was
#: ``KEEP_RECENT_TOKENS = 0`` making ``prepareCompaction`` silently return undefined every time. Using
#: the hook as the main gate pays an unnecessary cost: the first thing ``ctx.compact()`` does is
#: ``await this.abort()``, so every compaction ends a session.
#: v9 measured 91 compactions in 56 minutes = 91 session restarts, each time the model having to reread
#: ``notes.md`` to reorient; levels over the same period 69 vs codex's 93.
#:
#: **The four detours taken on 2026-09-19, written down here so they aren't taken again:**
#:
#: 1. "Tune the threshold" -- v7 wrote 100,000, v8 38,000 (= the compensated 100,000/2.58).
#:    The threshold was always right; what failed was ``prepareCompaction`` downstream of it.
#: 2. "``estimateTokens`` misses thinking" -- **false**. The assistant branch of
#:    ``compaction.js:201-203`` explicitly has ``block.type === "thinking"``. What misses thinking is
#:    ``estimateTextAndImageContentChars``, which is only used for user / toolResult.
#: 3. "Feed usage back into pi's own mechanism" -- the ``context`` event can indeed change messages,
#:    but in ``agent-loop.js`` ``transformContext`` runs **after** ``prepareNextTurn``, and its return
#:    value only goes into the LLM request, not back into ``agent.state.messages``. It can't reach the decision.
#: 4. "pi only checks compaction at the end of an agent run" (v9's design premise) -- **false**.
#:    The inner loop at ``agent-loop.js:90`` calls ``prepareNextTurn`` on every tool turn.
#:    The common lesson of the four: three times we **inferred internal mechanism backwards from
#:    externally visible behavior**, and a silent early return is invisible from outside. Only the
#:    fourth time did we instrument a copy of pi, and it was located in ten minutes.
#: v10 four-hour measurement: **fired 0 times**, as designed (the main gate fires first).
REAL_COMPACT_LIMIT = 180_000
#: **Main gate.** pi's own ``_compactBeforeNextAssistantResponse``, run once per tool turn, deciding on
#: the usage-backed real context and compacting in place when past the line **without ending the session**.
#: Value = codex's ``model_auto_compact_token_limit``.
#: It was always alive, only blocked by ``KEEP_RECENT_TOKENS = 0`` (see that comment).
COMPACT_THRESHOLD = 100_000
#: The context window pi uses for accounting = **the server's real context_length**.
#: v1/v2 set it to `auto_compact + 80,000 = 180,000`, an artificial ceiling: as context approached it
#: each request's output budget got squeezed out (v2 had 33 `stopReason=length`, all with totalTokens
#: 175,9xx and output only 1~392), and compaction was also judged as overflow rather than threshold.
#: The real window is 262,144; there is no reason to report a smaller number to pi.

#: The hook and the append prompt are real files in the repo, mounted read-only into the jail -- not
#: generated into driver_home. driver_home is ``ws/.driver``, inside the agent's own workspace: in the
#: first round on 2026-09-19, ``lp85`` simply did ``cat .driver/timeout_hook.ts`` and read the whole
#: thing, design-rationale comments included.
#: The path still appears on pi's cmdline (visible by reading /proc inside the jail), so **the deployed
#: version carries no comments**; the rationale stays in this docstring, which doesn't go into the jail.
HOOK = UNIT / "pi_hook" / "timeout_hook.ts"
APPEND = UNIT / "pi_hook" / "append_system.md"
RECEIPT_NAME = "hook_receipt.jsonl"


def budgets(knobs: Knobs) -> dict[str, int]:
    """The three numbers this arm pins itself, not inherited from ``_pi.budgets()``.

    The stock formula changes with the control arm's needs (it already changed once on 2026-09-19:
    ``reserveTokens`` went from ``SUMMARY_RESERVE`` to ``maxTokens``); following it would make this
    arm irreproducible across rounds.

    - ``contextWindow`` = the server's real window, no artificial ceiling (see the ``CONTEXT_WINDOW`` comment).
    - ``reserveTokens`` is derived back from the main gate: ``contextWindow − COMPACT_THRESHOLD``.
      The criterion of ``shouldCompact`` (``compaction.js:160-164``) is exactly
      ``context > contextWindow − reserveTokens``; **there is no separate threshold knob**.
    - The summary's actual budget is ``min(0.8 × reserveTokens, maxTokens)`` = ``MAX_OUTPUT_TOKENS``,
      plus the ``KEEP_RECENT_TOKENS`` of kept raw text, so the post-compaction context is bounded by
      ``MAX_OUTPUT_TOKENS + KEEP_RECENT_TOKENS`` = 52,768, still far below the 100,000 threshold;
      the "already above the threshold right after compaction" loop can't occur.
    """
    reserve = knobs.server_context - COMPACT_THRESHOLD
    if KEEP_RECENT_TOKENS <= MAX_TOOL_RESULT_TOKENS:
        raise ValueError(
            "pi_aligned: keepRecentTokens is not above the upper bound of a single tool output; "
            "findCutPoint's scan can't get past the trailing toolResult and mid-turn compaction would silently fail")
    if MAX_OUTPUT_TOKENS + KEEP_RECENT_TOKENS >= COMPACT_THRESHOLD:
        raise ValueError("pi_aligned: post-compaction context would exceed the threshold, a compaction loop")
    # Worst-case prompt: **the gate that fires last** + the growth of the one turn after it.
    # The upper bound on "how much one turn can grow" is ``MAX_OUTPUT_TOKENS`` itself -- the full
    # thinking stays in history and is resent turn after turn, so prompt growth ≈ the previous turn's
    # completion (``Arc code swe align`` per-turn reconciliation: codex growth median 2,155 / output
    # median 1,095, mini 1,842 / 1,029). Plus the output budget that request itself asks for, that is
    # the ``input + max_new_tokens`` sent to sglang -- ``tokenizer_manager.py:1216`` returns 400 past context.
    last_gate = max(COMPACT_THRESHOLD, REAL_COMPACT_LIMIT)
    if last_gate + 2 * MAX_OUTPUT_TOKENS > knobs.server_context:
        raise ValueError("pi_aligned: the turn after the last gate would push input+max_new_tokens past sglang's context, returning 400")
    if REAL_COMPACT_LIMIT and REAL_COMPACT_LIMIT <= COMPACT_THRESHOLD:
        raise ValueError("pi_aligned: the fallback gate is not above the main gate and would end the session ahead of it")
    return {"contextWindow": knobs.server_context, "reserveTokens": reserve,
            "maxTokens": MAX_OUTPUT_TOKENS}


def settings_json(knobs: Knobs) -> dict:
    """Both the threshold and "what is kept after compaction" are pinned here."""
    return {"compaction": {"enabled": True,
                           "reserveTokens": budgets(knobs)["reserveTokens"],
                           "keepRecentTokens": KEEP_RECENT_TOKENS},
            "quietStartup": True}


def models_json(knobs: Knobs) -> dict:
    spec = _pi.models_json(knobs)
    model = spec["providers"][_pi.PROVIDER]["models"][0]
    b = budgets(knobs)
    model["contextWindow"] = b["contextWindow"]
    model["maxTokens"] = b["maxTokens"]
    if SAMPLING_PARAMS:
        model["samplingParams"] = dict(SAMPLING_PARAMS)
    return spec


def prepare(driver_home: Path, workspace: Path, knobs: Knobs) -> None:
    _pi._check_effort(knobs.effort)
    driver_home.mkdir(parents=True, exist_ok=True)
    (driver_home / "models.json").write_text(json.dumps(models_json(knobs), indent=2) + "\n")
    (driver_home / "settings.json").write_text(json.dumps(settings_json(knobs), indent=2) + "\n")
    # Rewritten every session: the receipt must tell whether "this session's hook ran",
    # and resume uses the same driver_home. Without clearing, all we'd know is "it ran at some point".
    (driver_home / RECEIPT_NAME).write_text("")


def env(driver_home: Path, workspace: Path, paths: Paths) -> dict[str, str]:
    base = _pi.env(driver_home, workspace, paths)
    base["ARCSEC_PI_HOOK_RECEIPT"] = str(driver_home / RECEIPT_NAME)
    # Unset or 0: the hook registers no message_end handler, falling back to pi's own estimate-based threshold.
    base["ARCSEC_PI_REAL_COMPACT_LIMIT"] = str(REAL_COMPACT_LIMIT)
    return base


def mounts(paths: Paths) -> list[Path]:
    """pi's Node runtime, plus the directory holding the hook and the append prompt (read-only)."""
    return [*_pi.mounts(paths), HOOK.parent]


def _common(knobs: Knobs) -> list[str]:
    return _pi._common(knobs, tools=TOOLS,
                       extra=["--extension", str(HOOK),
                              "--append-system-prompt", str(APPEND)])


def argv(task: str, knobs: Knobs) -> list[str]:
    return _pi._piped([*_common(knobs), "--", task])


def resume_argv(handle: str, nudge: str, knobs: Knobs) -> list[str]:
    return _pi._piped([*_common(knobs), "--session", handle, "--", nudge])


def effective(knobs: Knobs) -> dict:
    """What this arm actually wears, written back into ARM_STATE (``run_arm.py`` picks it up via getattr; optional).

    The ``model_context_window`` given by ``config`` is a dead key for pi -- ``budgets()`` derives its
    own from ``auto_compact``. Recording only the config value would make whoever reconciles later
    think 120000 was in effect. The four alignment constants are recorded here too: ARM_STATE must be
    able to say how this arm differs from stock pi, not just that it is called pi_aligned.
    """
    b = budgets(knobs)
    return {
        "context_window": b["contextWindow"],   # the config's model_context_window has no effect
        # Main gate (real usage, run by the hook) and fallback gate (pi's estimate). Both must be
        # recorded: with only one, whoever reconciles later would think compaction was decided by the other path.
        "real_compact_limit": REAL_COMPACT_LIMIT,
        "compaction_threshold": b["contextWindow"] - b["reserveTokens"],
        "reserve_tokens": b["reserveTokens"],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "keep_recent_tokens": KEEP_RECENT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "bash_timeout_seconds": BASH_TIMEOUT_SECONDS,
        "tools": TOOLS,
        "sampling_params": dict(SAMPLING_PARAMS),   # empty dict = server defaults, same as codex
        "system_append_sha256": hashlib.sha256(APPEND.read_bytes()).hexdigest()[:16],
        "hook_sha256": hashlib.sha256(HOOK.read_bytes()).hexdigest()[:16],
    }


def receipts(driver_home: Path, kind: str | None = None) -> list[dict]:
    """Launch acceptance: the hook leaves one line per action. Empty means this arm has degraded into stock pi.

    Two ``kind`` values: ``timeout`` (filled in bash's default timeout) and ``ctx`` (measured real usage
    once; ``fired`` says whether that triggered compaction). ``ctx`` rows are the only evidence for the
    ``REAL_COMPACT_LIMIT`` path -- ``compaction_start`` in ``driver.log`` only says compaction happened,
    not who asked for it; the main gate and the fallback gate look identical there.
    """
    path = driver_home / RECEIPT_NAME
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows if kind is None else [r for r in rows if r.get("kind") == kind]


events = _pi.events
brief_addendum = _pi.brief_addendum
installed_version = _pi.installed_version
