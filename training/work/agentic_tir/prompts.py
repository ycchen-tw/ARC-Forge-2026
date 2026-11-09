"""Agentic-TIR protocol: explorer / refiner system prompts, user prompt rendering, report parsing.

**Unrelated** to `arcforge.tir.protocol` (the frozen protocol the student model is trained on) -- this arm runs a
general model (Qwen3.8-Flash-Next); the prompts are this unit's own design and may change.

Design points
-------------
* **All test inputs of a task are given together**: the sandbox has `test_inputs` (a list), no `test_input`.
  The rule must produce a sensible output for every test input at once; this is the core signal of joint solving.
* **One final report format** (used by both explorer and refiner): RULE / FINDINGS / CODE / CONFIDENCE /
  ANSWER, six sections (v6 adds NEXT). An explorer that finishes normally writes one report; one forced to stop
  when the budget runs out writes the same report, only ANSWER may be empty. The refiner's input is these reports.
* **Each ANSWER attempt may be a literal JSON grid or a single sandbox expression**
  (`transform(test_inputs[0])`, `candidates[0]["A"]`). An expression saves tokens on large grids and avoids
  copying errors; the harness evaluates the expression to a grid while the sandbox is still alive.
* Parsing is "lenient extraction, strict validation": headers may use `#`/`##`/`###`/bold; a grid must be
  rectangular with values 0-9, otherwise it counts as absent.
* **Grids in the prompt are "one row per line, one digit per cell, no separators"** (`_grid_text`): 1.05 tok/cell
  under the Qwen tokenizer, vs 2.0 for compact JSON and 3.0 for the official `[[1, 1], ...]`. The model never needs
  to copy a grid from the prompt (the sandbox has `train_pairs` / `test_inputs`), so the Python literal can go.
* **State lives in the sandbox, not in the head**: v1's handoff report (thinking off, no tool calls allowed) had good
  RULE/FINDINGS, but CODE was never run and ANSWER was a hand-typed grid (132/185 handoff CODEs did not even pass the
  train pairs). v2 requires `transform` to be kept current in the sandbox at all times, CODE to be the version that
  was run, and ANSWER to prefer expressions; the handoff keeps thinking on and allows a few tool calls as a checkpoint.
* **One explorer may span several sessions (v4)**: the handoff report written when the budget runs out is not only
  for the refiner; the same explorer opens a fresh context on **the same sandbox** and continues
  (`render_continue_user`): prompt = task + previous session's report + the `transform` check the harness just ran
  + the sandbox's `notes`. This cuts the baseline's 40–100k of continuous thinking into several 50k windows, with
  state carried over by the sandbox and the report.
"""

from __future__ import annotations

import json
import re

REPORT_SECTIONS = ("RULE", "FINDINGS", "NEXT", "CODE", "ALTERNATIVE", "CONFIDENCE", "ANSWER")   # only the alt variant writes ALTERNATIVE
VARIANTS = ("default", "alt")   # --prompt-variant; every prompt string of default is unchanged

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SANDBOX_COMMON = """\
You have a stateful Python sandbox (the `python` tool). The puzzle's data is already loaded, so you never need to retype the grids:

    train_pairs   list of {{"input": grid, "output": grid}} -- the training pairs shown in the question
    test_inputs   list of grids -- ALL {n_test} test input(s) of this puzzle, in order ({test_ref})
    np            numpy, already imported
{extra_names}
A grid is a list of lists of integers 0-9 (in the question, grids are printed one row per line, one digit per cell, no separators -- the same grids are already in the sandbox, don't retype them). Variables you define persist across calls. The sandbox has no network and cannot write files; it is for computation only. Each call is cut off after {tool_timeout:g} seconds and its output is truncated at {tool_max_output_chars} characters, so print summaries rather than dumping large grids.
"""

_METHOD = """\
Keep your state in the sandbox, not in your head. As soon as you can state a hypothesis as code, write it as `def transform(grid): ...` in the sandbox and run it on every training input, comparing against the expected training output -- a mismatch tells you exactly where the hypothesis is wrong. `transform` should always hold your latest version, and its result on the training pairs (from the tool output, not from memory) is the only evidence that counts. `notes` (a Python list of short strings) is your ledger: every time you verify a fact or a hypothesis fails, `notes.append(...)` one line saying what and where (which training pair, which cells). The ledger is carried verbatim to whoever continues this work -- you in a later session, or a colleague -- while your reasoning is not, so anything that is only in your head is lost. Look at what `transform` does to every test input too: the same rule has to produce a sensible output for all of them, and a test input that your rule cannot handle usually means the rule is incomplete, not that the test is special.
"""

_BUDGET = """\
Working budget: this whole conversation (your reasoning, your replies and the tool outputs) must fit in about {budget_k}k tokens. After each tool result you are told how much is left. Plan for it: spend the budget on testing hypotheses in code, not on hand-simulating grids in your head. A single reply is also limited to about {turn_k}k tokens: think in short bursts and land each one as a tool call (run the check, update `transform` / `notes`) rather than reasoning for a long time without touching the sandbox -- if a reply is cut off you will be asked to checkpoint and continue. When the total budget is nearly gone you will be asked to stop and write the report below, whatever state you are in -- but do not wait for that: as soon as you have a rule that passes every training pair and gives outputs for every test input, write the report.
"""

_REPORT_FORMAT = """\
FINAL REPORT FORMAT -- your last reply (the one with no tool call) must be exactly these six sections, in this order, with these headers:

## RULE
One to three sentences stating the transformation, or your best partial understanding of it.

## FINDINGS
The current picture, synthesised: what is established, what has been ruled out, and what is still uncertain -- at most about 10 bullets. Do not re-list the ledger (`notes` is delivered alongside this report); do not repeat the puzzle.

## NEXT
If the puzzle is not solved: the single most promising direction you have NOT yet tried, and the first concrete check to run for it (which pair, what to look at). This is the first thing whoever continues will do, so make it specific. If you are done, write `none`.

## CODE
One ```python fenced block defining `def transform(grid)` (input grid -> output grid, list of lists): the exact code you last ran in the sandbox, even if it does not pass every training pair, followed by one line stating which training pairs it passes according to the tool output. Do not write code here that you have not run. If you have no code yet, write `none`.

## CONFIDENCE
An integer 0-100: your probability that your answer for every test input is exactly right.

## ANSWER
{answer_instructions}
"""

_EXPLORER_ANSWER = """\
One block per test input, in order:

=== TEST 1 ===
<the output grid>
=== TEST 2 ===
<the output grid>

Each <grid> is normally a single Python expression that evaluates to the grid in your sandbox, e.g. `transform(test_inputs[0])` -- the harness evaluates it for you, so it is exact and costs no tokens. A literal grid (JSON like [[0,1],[2,3]], or one row per line of digits) is accepted only when it is small and you have already produced and checked it in the sandbox; never type out a grid from memory. If you genuinely have no answer for a test input, write `none` for it.
"""

_REFINER_ANSWER = """\
For each test input give ATTEMPT 1 (your best answer) and, when a second interpretation is genuinely plausible, ATTEMPT 2 (a different grid). Scoring counts a test input as solved if either attempt is exactly right, so a distinct second attempt costs nothing -- but a duplicate is wasted, and a random one is worth nothing. Format:

=== TEST 1 ATTEMPT 1 ===
<grid>
=== TEST 1 ATTEMPT 2 ===
<grid>
=== TEST 2 ATTEMPT 1 ===
<grid>

Each <grid> is normally a single Python expression that evaluates to the grid in your sandbox, e.g. `transform(test_inputs[0])`, `transform_2(test_inputs[1])` or `candidates[0]["A"]` -- the harness evaluates it for you, so it is exact and costs no tokens. A literal grid (JSON like [[0,1],[2,3]], or one row per line of digits) is accepted only when it is small and you have already produced and checked it in the sandbox; never type out a grid from memory.
"""

_EXPLORER_HEAD = (
    "You are an expert puzzle solver working on an ARC-style grid puzzle. Your job in this "
    "session is to find the transformation rule, verify it in code against the training "
    "pairs, and apply it to every test input.\n\n"
)

_REFINER_HEAD = (
    "You are an expert puzzle solver working on an ARC-style grid puzzle. Several colleagues "
    "(explorers) have already worked on this puzzle independently, each within a limited "
    "budget, and handed you their reports and ledgers. Your job is to consolidate their work, settle "
    "the disagreements, verify in code, and produce the final answer for every test input.\n\n"
)

_REFINER_HOWTO = """\
How to use the reports: the explorers' `transform` functions are already defined in your sandbox as `transform_1`, `transform_2`, ... (only those that compiled), and the harness has already run each one on every training pair and every test input -- the check table in the question is trustworthy, you don't need to redo it. Distinct predicted test outputs are collected in `candidates` (a list, one dict per test input, keyed "A", "B", ...). Start from what agrees, examine where the candidates differ cell by cell (`np.argwhere(np.array(a) != np.array(b))`), and decide which interpretation the training pairs actually support. A high-confidence explorer can still be wrong; a rule that passes every training pair AND makes sense on every test input beats one that only fits the training pairs. If no explorer got it, use their failed hypotheses to avoid repeating them and attack the puzzle yourself.
"""

EXPLORER_SYSTEM = _EXPLORER_HEAD + _SANDBOX_COMMON + "\n" + _METHOD + "\n" + _BUDGET + "\n" + _REPORT_FORMAT

REFINER_SYSTEM = (_REFINER_HEAD + _SANDBOX_COMMON + "\n" + _METHOD + "\n" + _REFINER_HOWTO
                  + "\n" + _BUDGET + "\n" + _REPORT_FORMAT)

# --prompt-variant alt (v8 analysis: of the 19 wrong pairs, 14 had a CODE passing all train pairs, and for 12 of them the
# correct answer was never produced by any session; of the 102 "consistent and all-pass" pairs, 72 had an empty second
# attempt, and the refiner's self-made ATTEMPT 2 was right in only 1 of 36): after passing all train pairs the explorer
# spends one capped step looking for a counter-hypothesis `transform_alt` that "also passes train but gives a different
# test output", written into ALTERNATIVE and ATTEMPT 2; the refiner deliberately looks for counterexamples when all
# candidates are train-pass, and ATTEMPT 2 is left to it.
_ALT_STEP = """\
Before writing the report, when `transform` passes every training pair, spend ONE bounded step on a counter-hypothesis: at most 2 tool calls and about 3k tokens, and skip the step entirely if the last budget note shows less than 8k tokens left. A counter-hypothesis is a rule that ALSO passes every training pair but gives a different output on at least one test input -- usually a different reading of a detail the training pairs never pin down (tie-breaking, direction or orientation, what counts as an object, which colour is background, a case that only the test inputs exercise). Write it as `def transform_alt(grid)` in the sandbox and run it on the training pairs and the test inputs. Keep it only if it passes every training pair AND differs from `transform` on some test input; otherwise discard it. Do not revise `transform` during this step and do not spend a third tool call on it: report what you have.
"""

_ALTERNATIVE_SECTION = """\
## ALTERNATIVE
Only if the counter-hypothesis step produced a `transform_alt` that passes every training pair and differs from `transform` on a test input: one or two sentences saying how its reading differs and why the training pairs cannot decide between the two, then one ```python fenced block defining `def transform_alt(grid)` (the exact code you ran), then one line with its training result and which test inputs it differs on. Otherwise write `none`.

"""

_REPORT_FORMAT_ALT = (_REPORT_FORMAT.replace("exactly these six sections", "these sections")
                      .replace("## CONFIDENCE\n", _ALTERNATIVE_SECTION + "## CONFIDENCE\n"))
assert _REPORT_FORMAT_ALT.count("## ALTERNATIVE") == 1 and "six sections" not in _REPORT_FORMAT_ALT

_EXPLORER_ANSWER_ALT = """\
One block per test input, in order; ATTEMPT 1 is the output of `transform`:

=== TEST 1 ATTEMPT 1 ===
<the output grid>
=== TEST 1 ATTEMPT 2 ===
<the output grid of `transform_alt`>
=== TEST 2 ATTEMPT 1 ===
<the output grid>

Each <grid> is normally a single Python expression that evaluates to the grid in your sandbox, e.g. `transform(test_inputs[0])` -- the harness evaluates it for you, so it is exact and costs no tokens. A literal grid (JSON like [[0,1],[2,3]], or one row per line of digits) is accepted only when it is small and you have already produced and checked it in the sandbox; never type out a grid from memory. If you genuinely have no answer for a test input, write `none` for it. Give ATTEMPT 2 for a test input only when ALTERNATIVE is not `none` and `transform_alt` differs from `transform` on that test input; it must be `transform_alt(test_inputs[i])`, never a hand-edited copy of attempt 1.
"""

_REFINER_ALT_STEP = """\
Agreement is weaker evidence than it looks: passing every training pair does not identify the rule, and when every explorer read the same detail the same way, their agreement says nothing about that detail. Whenever every candidate for a test input comes from code that passes all training pairs -- and especially when there is only one candidate -- spend ONE bounded step (at most 3 tool calls, about 4k tokens) on a counter-hypothesis before answering: name the detail the training pairs never pin down (tie-breaking, orientation, what counts as an object, background, a case only the test inputs exercise), write it as `def transform_alt(grid)`, and keep it only if it passes every training pair and differs from the consensus on some test input. Do not re-derive or re-implement a rule the explorers already verified -- the check table is the harness's own run, not their claim. ATTEMPT 1 MUST stay the consensus answer (candidate A, or the candidate you would have chosen without this step) unless a training pair refutes it -- never put the counter-hypothesis in ATTEMPT 1, no matter how much you like it: the consensus has the better base rate. The counter-hypothesis goes in ATTEMPT 2 only, as `transform_alt(test_inputs[i])`.
"""

_REFINER_ANSWER_ALT = """\
For each test input give ATTEMPT 1 (your best answer) and ATTEMPT 2 (the output of the most plausible alternative reading: a rule that also passes every training pair but differs from ATTEMPT 1 on this test input -- normally `transform_alt(test_inputs[i])`, or an explorer's ALTERNATIVE / a rival candidate if that is the strongest alternative). Scoring counts a test input as solved if either attempt is exactly right, so ATTEMPT 2 is free -- but it only helps if it is a different rule's output: a copy of ATTEMPT 1, a hand-edited variant of it, or a grid that no rule produced is worth nothing. Omit ATTEMPT 2 only when you found no alternative that passes every training pair. Never swap the two: ATTEMPT 1 is the consensus reading, ATTEMPT 2 is the alternative. Format:

=== TEST 1 ATTEMPT 1 ===
<grid>
=== TEST 1 ATTEMPT 2 ===
<grid>
=== TEST 2 ATTEMPT 1 ===
<grid>

Each <grid> is normally a single Python expression that evaluates to the grid in your sandbox, e.g. `transform(test_inputs[0])`, `transform_alt(test_inputs[0])`, `transform_2(test_inputs[1])` or `candidates[0]["A"]` -- the harness evaluates it for you, so it is exact and costs no tokens. A literal grid (JSON like [[0,1],[2,3]], or one row per line of digits) is accepted only when it is small and you have already produced and checked it in the sandbox; never type out a grid from memory.
"""

EXPLORER_SYSTEM_ALT = (_EXPLORER_HEAD + _SANDBOX_COMMON + "\n" + _METHOD + "\n" + _BUDGET + "\n" + _ALT_STEP
                       + "\n" + _REPORT_FORMAT_ALT)

REFINER_SYSTEM_ALT = (_REFINER_HEAD + _SANDBOX_COMMON + "\n" + _METHOD + "\n" + _REFINER_HOWTO + "\n" + _REFINER_ALT_STEP
                      + "\n" + _BUDGET + "\n" + _REPORT_FORMAT)

HANDOFF_USER = """\
STOP -- your working budget is used up. You may make at most {n_calls} more tool call(s), only to checkpoint: make sure `transform` in the sandbox is your latest version and run it on the training pairs (and on the test inputs) so you can report real results, and bring the `notes` ledger up to date. Then write the FINAL REPORT, in exactly the five-section format from the instructions (RULE / FINDINGS / NEXT / CODE / CONFIDENCE / ANSWER), so that whoever continues from where you are (you in a fresh session with this same sandbox, or a colleague) does not have to redo it. CODE must be the code that is actually in the sandbox; in ANSWER use sandbox expressions (`transform(test_inputs[0])`) or `none` -- do not type grids by hand.
"""

NOTE_TAGS = ("RULE", "VERIFIED", "REFUTED", "TODO", "CODE")
_NOTE_TAG_RE = re.compile(r"^\s*(RULE|VERIFIED|REFUTED|TODO|CODE)\s*:\s*", re.IGNORECASE)

# --notes-structured: `notes` is still a list (the model-side API is unchanged), but each entry must carry a tag; the
# harness groups by tag when rendering, dedups, and keeps only the newest RULE. OpenHands' condenser summary
# (COMPLETED / PENDING / CURRENT_STATE / CODE_STATE / TESTS), Cursor's self-summary, and ARC Prize's refinement loop
# all carry structured state; v3's free-form appended notes repeat and are unordered.
NOTES_PROTOCOL = """

`notes` protocol: every entry you append starts with one of these tags, and says one thing:
  RULE: <your current best statement of the transformation> (append a new one whenever it changes; only the newest counts)
  VERIFIED: <a fact you confirmed by running code, with the evidence in a few words>
  REFUTED: <a hypothesis you tested and ruled out, and what killed it>
  TODO: <the next concrete thing to check, in order of promise>
  CODE: <what `transform` (or a helper) currently does / its known failure cases>
Keep entries short and never duplicate an existing one. The harness shows `notes` grouped by tag (newest RULE only) \
whenever earlier rounds are removed from view or another session takes over, so this is your durable memory."""


def render_notes(notes: list[str], structured: bool) -> str:
    """Render notes as prompt text. structured: grouped by tag, deduped, newest RULE only; untagged entries go to OTHER."""
    if not notes:
        return "(empty)"
    if not structured:
        return "\n".join(f"- {x}" for x in notes)
    groups: dict[str, list[str]] = {tag: [] for tag in NOTE_TAGS}
    groups["OTHER"] = []
    for x in notes:
        m = _NOTE_TAG_RE.match(x)
        tag, body = (m.group(1).upper(), x[m.end():].strip()) if m else ("OTHER", x.strip())
        if body and body not in groups[tag]:
            groups[tag].append(body)
    if groups["RULE"]:
        groups["RULE"] = groups["RULE"][-1:]
    out = []
    for tag in (*NOTE_TAGS, "OTHER"):
        if groups[tag]:
            out.append(f"{tag}:\n" + "\n".join(f"  - {b}" for b in groups[tag]))
    return "\n".join(out) if out else "(empty)"


# --delegate: an explorer may hand a self-contained subproblem to a helper with a clean context (same sandbox) and get
# back only a short conclusion. In the seven-method comparison of arXiv 2606.29718, sub-agent isolation (54.0%) is
# clearly above every summarization variant (≤48.2%); Anthropic's advice is that a subagent return only 1–2k tokens.
DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Hand one self-contained sub-question to a helper that starts with a CLEAN context but shares your sandbox "
            "(every function and variable you defined is available to it, and whatever it defines stays for you). "
            "It works on its own budget and returns only a short conclusion, so its exploration never uses your context. "
            "Good uses: an exhaustive scan or table over all pairs, testing one alternative hypothesis end to end, "
            "debugging why `transform` fails on a specific pair, or writing and validating a helper function. "
            "State precisely what to test, what is already known (or point it at `notes`), and what to return."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "the self-contained task for the helper, incl. what to return"},
            },
            "required": ["question"],
        },
    },
}

DELEGATE_SYSTEM_NOTE = """

Delegation: you also have a `delegate` tool (see its description). Use it when a sub-investigation would flood your \
context with output or would take many rounds -- the helper does the work in its own context and you get back only \
the conclusion. Keep the question self-contained; the helper knows the puzzle and the sandbox but not your reasoning."""

HELPER_SYSTEM = (
    """You are a helper spun up by the main solver of an ARC puzzle. The main solver has been working in a Python sandbox \
and hands you ONE self-contained sub-question. You share that sandbox: everything it defined (listed in the question) is \
available to you, and anything you define stays available to it. You do not see its reasoning -- only the question, \
the puzzle and `notes`.

"""
    + _SANDBOX_COMMON + "\n" + """\
Do exactly what the question asks, with code, and verify with code. Append tagged entries to `notes` (VERIFIED: / \
REFUTED: / CODE:) for anything you establish. Do not redefine `transform` unless the question asks you to; if you write \
a candidate, name it `transform_alt` or as the question says.

Working budget: about {budget_k}k generated tokens; after each tool result you are told how much is left. \
When you are done (or the budget is nearly gone), reply WITHOUT a tool call, in at most ~1500 tokens, with exactly:

## VERDICT
One to three sentences answering the question.
## EVIDENCE
The concrete results (numbers, which pairs, what failed) -- compact, no grids unless tiny.
## DEFINED
Names you defined in the sandbox and one line each on what they do (or `none`).
## SUGGESTION
What the main solver should do next given this (one or two sentences)."""
)

HELPER_HANDOFF = """\
STOP -- your budget is used up. No more tool calls: write your reply now in the VERDICT / EVIDENCE / DEFINED / \
SUGGESTION format with what you have."""


def helper_system(n_test: int, budget_tokens: int, tool_timeout: float, tool_max_output_chars: int) -> str:
    return HELPER_SYSTEM.format(
        n_test=n_test, test_ref=_test_ref(n_test),
        extra_names="    notes         the main solver's ledger (list of tagged strings) -- read it, append to it\n",
        tool_timeout=tool_timeout, tool_max_output_chars=tool_max_output_chars, budget_k=budget_tokens // 1000,
    )


def render_helper_user(train_pairs, test_inputs, names: str, notes: list[str], structured: bool, question: str) -> str:
    return (render_task(train_pairs, test_inputs)
            + "\n\nSandbox: functions already defined by the main solver: " + (names or "(none)")
            + "\n`notes` so far:\n" + render_notes(notes, structured)
            + "\n\nYOUR QUESTION from the main solver:\n" + question.strip()
            + "\n\nWork it through in the sandbox, then reply in the VERDICT / EVIDENCE / DEFINED / SUGGESTION format.")


COMPACT_SYSTEM_NOTE = """

Context management: when this conversation gets close to the context limit, the harness removes your oldest tool \
rounds from view (a note in the task message tells you what was removed and what the sandbox holds) and you simply \
continue in the same conversation -- you do not write a report at that point. The sandbox is never reset: every \
function and variable you defined stays available. Keep `notes` and `transform` up to date so nothing is lost when \
old rounds disappear. The working budget above is the total you may generate in this conversation."""

CHECKPOINT_USER = """\
[harness] The context is about to be compacted: everything before your last round (about {n_rounds} tool rounds) will \
disappear from view; the sandbox keeps all names, but the reasoning in those rounds will be gone. Make ONE short tool \
call now that appends to `notes` the things from those rounds you would otherwise lose -- verified facts, refuted \
hypotheses, the current best rule and where it fails, and what you were about to try (a few compact lines; do not \
re-list what is already in `notes`). Then continue exactly where you left off."""

CHECKPOINT_USER_STRUCTURED = """\
[harness] The context is about to be compacted: everything before your last round (about {n_rounds} tool rounds) will \
disappear from view; the sandbox keeps all names, but the reasoning in those rounds will be gone. Make ONE short tool \
call now that appends to `notes` what you would otherwise lose, one tagged entry per line (RULE: / VERIFIED: / REFUTED: / \
TODO: / CODE:), skipping anything already in `notes`. Then continue exactly where you left off."""

CONTINUE_USER = """\
Your reply was cut off by the per-reply limit (the total budget is not used up: about {left_k}k tokens remain). Checkpoint first -- one tool call that writes what you have established so far into the sandbox (update `transform`, append to `notes`) -- then continue from where you were.
"""


def explorer_system(n_test: int, budget_tokens: int, turn_tokens: int, tool_timeout: float, tool_max_output_chars: int,
                    variant: str = "default", notes_structured: bool = False) -> str:
    assert variant in VARIANTS, variant
    system, answer = (EXPLORER_SYSTEM, _EXPLORER_ANSWER) if variant == "default" else (EXPLORER_SYSTEM_ALT, _EXPLORER_ANSWER_ALT)
    if notes_structured:
        system += NOTES_PROTOCOL
    return system.format(
        n_test=n_test, test_ref=_test_ref(n_test),
        extra_names="    notes         empty list for your own verified facts / failed hypotheses\n",
        tool_timeout=tool_timeout, tool_max_output_chars=tool_max_output_chars,
        budget_k=budget_tokens // 1000, turn_k=turn_tokens // 1000, answer_instructions=answer,
    )


def refiner_system(n_test: int, budget_tokens: int, turn_tokens: int, tool_timeout: float, tool_max_output_chars: int,
                   transform_names: list[str], variant: str = "default", notes_structured: bool = False) -> str:
    assert variant in VARIANTS, variant
    system, answer = (REFINER_SYSTEM, _REFINER_ANSWER) if variant == "default" else (REFINER_SYSTEM_ALT, _REFINER_ANSWER_ALT)
    if notes_structured:
        system += NOTES_PROTOCOL
    extra = ""
    if transform_names:
        extra += "    " + ", ".join(transform_names) + "   the explorers' transform functions (see the question)\n"
    extra += "    candidates    list (one dict per test input) of the distinct predicted test outputs, keyed \"A\", \"B\", ...\n"
    extra += "    notes         empty list for your own verified facts / failed hypotheses\n"
    return system.format(
        n_test=n_test, test_ref=_test_ref(n_test), extra_names=extra,
        tool_timeout=tool_timeout, tool_max_output_chars=tool_max_output_chars,
        budget_k=budget_tokens // 1000, turn_k=turn_tokens // 1000, answer_instructions=answer,
    )


def _test_ref(n_test: int) -> str:
    return "test_inputs[0]" if n_test == 1 else f"test_inputs[0] .. test_inputs[{n_test - 1}]"


# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

def _grid_json(g) -> str:
    return json.dumps(g, separators=(",", ":"))


def _grid_text(g) -> str:
    """One row per line, one digit per cell, no separators."""
    return "\n".join("".join(str(v) for v in row) for row in g)


def render_task(train_pairs: list[dict], test_inputs: list) -> str:
    out = ["Below are the training pairs of the puzzle. Find the transformation that maps each "
           "input to its output, then apply it to every test input. Grids are printed one row per "
           "line, one digit (0-9) per cell.\n", "--Training Examples--"]
    for i, p in enumerate(train_pairs):
        gi, go = p["input"], p["output"]
        out.append(f"--Example {i + 1}--\nINPUT ({len(gi)}x{len(gi[0])}):\n{_grid_text(gi)}\n"
                   f"OUTPUT ({len(go)}x{len(go[0])}):\n{_grid_text(go)}\n")
    out.append("--End of Training Examples--\n")
    out.append(f"--Test Inputs-- ({len(test_inputs)} test input(s); solve them together, they share one rule)")
    for i, g in enumerate(test_inputs):
        out.append(f"--Test {i + 1}-- (test_inputs[{i}], {len(g)}x{len(g[0])}):\n{_grid_text(g)}\n")
    out.append("--End of Test Inputs--")
    return "\n".join(out)


def render_explorer_user(train_pairs, test_inputs) -> str:
    return render_task(train_pairs, test_inputs) + "\n\nWork through the puzzle with the sandbox, then write the final report."


def render_continue_user(train_pairs, test_inputs, session_no: int, max_sessions: int, prev: dict) -> str:
    """Next session of the same explorer. `prev`: the previous session's parsed report (parse_report fields) +
    'stop_reason', 'check' (the check dict of the `transform` the harness just ran in this sandbox, see
    agentic_tir._check_transform), 'notes' (contents of `notes` in the sandbox)."""
    parts = [render_task(train_pairs, test_inputs), "", "=" * 60,
             f"YOUR PREVIOUS SESSION ON THIS PUZZLE (session {session_no - 1} of at most {max_sessions}; "
             f"it ended: {prev['stop_reason']})", "=" * 60,
             "You already worked on this puzzle in a previous session whose working budget ran out. Its sandbox "
             "is the one you have now: `transform`, `notes`, and every helper function or variable you defined are "
             "still there (use dir() if unsure). This is what that session reported:", "",
             "RULE: " + (prev["rule"].strip() or "(none)"),
             "FINDINGS:\n" + (prev["findings"].strip() or "(none)"),
             "NEXT (what that session planned to do next): " + (prev["next"].strip() or "(none)")]
    chk = prev["check"]
    if prev["code"] and chk["defined"]:
        parts.append(f"CODE (this is `transform` in your sandbox now):\n```python\n{prev['code'].strip()}\n```")
    elif prev["code"]:
        parts.append(f"CODE (reported, but it does not load in the sandbox -- {chk['summary']}):\n"
                     f"```python\n{prev['code'].strip()}\n```")
    else:
        parts.append("CODE: none reported" + ("" if not chk["defined"] else " (but a `transform` is defined in the sandbox)"))
    if chk["defined"]:
        parts.append("Harness check of `transform`, run just now: " + chk["summary"])
    conf = prev["confidence"]
    parts.append("CONFIDENCE: " + ("not stated" if conf is None else str(conf)))
    notes = prev["notes"]
    if notes:
        parts.append("`notes` (the ledger, verbatim from the sandbox):\n" + render_notes(notes, prev.get("notes_structured", False)))
    else:
        parts.append("`notes` (the ledger): empty -- the previous session did not keep one")
    parts += ["", f"This is session {session_no} of at most {max_sessions}, with the same working budget as before. "
              "Continue from where you left off: do not redo what FINDINGS already established and do not retry "
              "hypotheses that already failed; start with NEXT unless the ledger shows it is already done. Work through the puzzle "
              "with the sandbox, then write the final report -- it must stand on its own, so carry forward the "
              "still-relevant findings from the previous session."]
    return "\n".join(parts)


def render_refiner_user(train_pairs, test_inputs, reports: list[dict], checks: list[dict],
                        candidates: list[dict], inline_cells: int, variant: str = "default") -> str:
    """`reports[k]`: explorer k's parsed report (see parse_report) + 'label'/'stop_reason'; the alt variant also has
    'alt_check' (the result of the harness checking `transform_alt` in its sandbox; None for the refiner's own report).
    `checks[k]`: the result of the harness running transform_k (see agentic_tir.verify_reports).
    `candidates[t]`: {"A": {"grid":..., "sources": [...]}, ...}."""
    assert variant in VARIANTS, variant
    parts = [render_task(train_pairs, test_inputs), "", "=" * 60, f"EXPLORER REPORTS ({len(reports)})", "=" * 60]
    for k, (rep, chk) in enumerate(zip(reports, checks), start=1):
        conf = rep["confidence"]
        conf_s = "not stated" if conf is None else str(conf)
        parts.append(f"\n### Explorer {k} -- {rep['label']} (ended: {rep['stop_reason']}; stated confidence: {conf_s})")
        parts.append("RULE: " + (rep["rule"].strip() or "(none)"))
        parts.append("FINDINGS:\n" + (rep["findings"].strip() or "(none)"))
        parts.append("NEXT (what it would have tried next): " + (rep["next"].strip() or "(none)"))
        notes = rep["notes"]
        parts.append("NOTES (its ledger, verbatim):\n" + render_notes(notes, rep.get("notes_structured", False)))
        if rep["code"]:
            where = (f"loaded in your sandbox as `transform_{k}`" if chk["defined"] else
                     "reported; it does NOT load in your sandbox -- its helpers lived in the explorer's sandbox")
            parts.append(f"CODE ({where}):\n```python\n{rep['code'].strip()}\n```")
            parts.append("Harness check of this code: " + chk["summary"])
        elif rep["home_check"]["defined"]:
            parts.append("CODE: none reported, but a `transform` existed in its sandbox -- harness check: " + chk["summary"])
        else:
            parts.append("CODE: none")
        if variant == "alt":
            alt_chk = rep["alt_check"]
            alt_txt = rep["alt"] or "none"
            if alt_chk is not None and alt_chk["defined"]:
                parts.append(f"ALTERNATIVE (its counter-hypothesis; harness check of its `transform_alt`: {alt_chk['summary']}):\n{alt_txt}")
            else:
                parts.append("ALTERNATIVE: " + alt_txt)
        ans_bits = []
        for t, grids in enumerate(rep["answers"]):
            keys = [key for g in grids for key, c in candidates[t].items() if c["grid"] == g]
            ans_bits.append(f"test {t + 1}: " + (", ".join(f"candidate {k}" for k in keys) if keys else "none"))
        parts.append("Stated ANSWER -> " + "; ".join(ans_bits))
    parts += ["", "=" * 60, "CANDIDATE TEST OUTPUTS (distinct grids; `candidates[t][key]` in your sandbox; "
              "rows of digits as in the question)", "=" * 60]
    for t, cands in enumerate(candidates):
        parts.append(f"\nTest {t + 1} ({len(test_inputs[t])}x{len(test_inputs[t][0])} input):")
        if not cands:
            parts.append("  (no candidate -- nobody produced an output for this test input)")
        for key, c in cands.items():
            g = c["grid"]
            shape = f"{len(g)}x{len(g[0])}"
            srcs = ", ".join(c["sources"])
            line = f"  {key}: {shape}, from {srcs}"
            if len(g) * len(g[0]) <= inline_cells:
                line += "\n" + _grid_text(g)
            parts.append(line)
        keys = list(cands)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                parts.append(f"  {keys[i]} vs {keys[j]}: " + diff_summary(cands[keys[i]]["grid"], cands[keys[j]]["grid"]))
    if variant == "alt" and all(_train_all_pass(c["train"]) for c in checks):
        # Every candidate comes from code that passes all train pairs: in v8, 72 of the 102 "consistent and all-pass"
        # pairs had an empty second attempt, and on the two wrong tasks both explorer and refiner wrote "no ambiguity".
        # State outright that train does not decide the rule, forcing it to take the counter-hypothesis step.
        assert checks
        disagree = [str(t + 1) for t in range(len(test_inputs)) if len(candidates[t]) > 1]
        agree_s = ("the explorers agree on every test input" if not disagree
                   else "they disagree on test " + ", ".join(disagree))
        parts += ["", f"NOTE: every candidate above comes from code that passes all {len(train_pairs)} training pairs, "
                  f"and {agree_s}. The training pairs therefore do not decide the rule -- the agreement or the votes tell "
                  "you which reading the explorers defaulted to, not which one is right. Do the counter-hypothesis step "
                  "from your instructions before answering, and use ATTEMPT 2 for it."]
    parts += ["", "Consolidate, verify, and write the final report with ATTEMPT 1 (and ATTEMPT 2 where a second "
              "interpretation is plausible) for every test input."]
    return "\n".join(parts)


def _train_all_pass(train: list) -> bool:
    return bool(train) and all(r == "pass" for r in train)


def diff_summary(a, b, max_cells: int = 12) -> str:
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        return f"different shapes {len(a)}x{len(a[0])} vs {len(b)}x{len(b[0])}"
    diffs = [(r, c, a[r][c], b[r][c]) for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c]]
    if not diffs:
        return "identical"
    shown = ", ".join(f"({r},{c}) {x}->{y}" for r, c, x, y in diffs[:max_cells])
    more = "" if len(diffs) <= max_cells else f", ... ({len(diffs) - max_cells} more)"
    return f"{len(diffs)} cells differ: {shown}{more}"


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^[ \t]*(?:#{1,4}[ \t]*|\*\*)?(RULE|FINDINGS|NEXT|CODE|ALTERNATIVE|CONFIDENCE|ANSWER)(?:\*\*)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_ANSWER_HDR_RE = re.compile(
    r"^[ \t]*=+[ \t]*TEST[ \t]*(\d+)(?:[ \t]*ATTEMPT[ \t]*(\d+))?[ \t]*=+[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text: str) -> dict[str, str]:
    """Split by the five section headers. On duplicate names take the last one (the model sometimes drafts, then rewrites)."""
    hits = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out[m.group(1).upper()] = text[m.end():end]
    return out


def extract_transform_code(text: str, name: str = "transform") -> str | None:
    """The last fenced block that defines `def <name>(` (`transform` does not match `transform_alt`)."""
    blocks = [b for b in _FENCE_RE.findall(text) if re.search(rf"^\s*def {name}\s*\(", b, re.MULTILINE)]
    return blocks[-1].rstrip() if blocks else None


def is_grid(g) -> bool:
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) == len(g[0]) and len(r) > 0 for r in g)
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


def _first_balanced(text: str, start: int = 0):
    """Find the first `[` from start and return the matching `[...]` substring (None if there is none)."""
    i = text.find("[", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        ch = text[j]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def parse_answer_blocks(text: str) -> dict[tuple[int, int], str]:
    """`=== TEST i [ATTEMPT j] ===` header → that block's raw content (whitespace stripped). Keys are 0-based (test, attempt)."""
    hits = list(_ANSWER_HDR_RE.finditer(text))
    out: dict[tuple[int, int], str] = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        t = int(m.group(1)) - 1
        a = int(m.group(2)) - 1 if m.group(2) else 0
        out[(t, a)] = text[m.end():end].strip()
    return out


def classify_answer_body(body: str) -> tuple[str, str | None]:
    """One ANSWER body → ("grid", json) | ("expr", code) | ("none", None).

    grid: a literal `[[...]]` that json.loads into a valid grid. expr: any other non-empty, non-`none` single line of
    text (backticks and fence markers stripped)."""
    s = body.strip()
    if not s or s.lower().rstrip(".") in ("none", "n/a", "no answer", "(none)"):
        return "none", None
    rows = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if rows and all(re.fullmatch(r"[0-9]+", r) for r in rows) and len({len(r) for r in rows}) == 1:
        return "grid", json.dumps([[int(ch) for ch in r] for r in rows], separators=(",", ":"))
    lit = _first_balanced(s)
    if lit is not None and s.lstrip().startswith("["):
        try:
            g = json.loads(lit)
        except json.JSONDecodeError:
            g = None
        if is_grid(g):
            return "grid", lit
    s = re.sub(r"^```(?:python|py)?\s*|\s*```$", "", s).strip()
    s = s.strip("`").strip()
    line = s.splitlines()[0].strip() if s else ""
    if not line or line.lower() == "none":
        return "none", None
    return "expr", line


def parse_report(text: str) -> dict:
    """Parse the final reply into a report. Missing sections get an empty string/None, but all fields are always returned."""
    sec = split_sections(text)
    code = extract_transform_code(sec.get("CODE", "")) or extract_transform_code(text)
    alt = sec.get("ALTERNATIVE", "").strip()   # the alt variant's counter-hypothesis; always empty for default
    conf = None
    m = re.search(r"\d{1,3}", sec.get("CONFIDENCE", ""))
    if m:
        conf = max(0, min(100, int(m.group(0))))
    answer_text = sec.get("ANSWER", "")
    blocks = parse_answer_blocks(answer_text) or parse_answer_blocks(text)
    return {
        "has_sections": bool(sec),
        "rule": sec.get("RULE", "").strip(),
        "findings": sec.get("FINDINGS", "").strip(),
        "next": sec.get("NEXT", "").strip(),
        "code": code,
        "alt": alt,
        "alt_code": extract_transform_code(alt, "transform_alt"),
        "confidence": conf,
        "answer_blocks": {f"{t}:{a}": classify_answer_body(b) for (t, a), b in blocks.items()},
        "answer_text": answer_text.strip(),
    }
