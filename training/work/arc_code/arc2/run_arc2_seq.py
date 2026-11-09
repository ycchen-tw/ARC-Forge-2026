"""codex solves ARC-AGI-2 with several sessions per task giving two attempts; one run compares several combinations, paired.

    python work/arc_code/arc2/run_arc2_seq.py --config work/arc_code/arc2/configs/<cfg>.json \\
        --run-dir <dir> --tasks-dir <ARC-AGI-2/data/evaluation> \\
        --agent-python <venv>/bin/python --driver-bin <codex prefix> \\
        --model-base-url http://127.0.0.1:<port>/v1 --model <served name> --tokenizer-json <tokenizer.json> \\
        [--arms ind,refine_exp] [--second-budget N] [--refine-budget N] \\
        [--reuse <old run> --reuse-kinds first,seq,ind]

Sessions per task (episode ``<task>__s<k>``, each one a run_arc2.solve(); workspace, budget and harvest as before):
  s0  first       the current v3 harness solving (the config's prompts); its answer is every arm's attempt_1 candidate
  s1  seq         starts after s0 ends: the workspace gets an extra read-only previous/ (s0's solution, notes, check
                  result, each test output); prompts v3seq ask it for a second answer "different from previous". When s0
                  passes all train, it does not count as done while its outputs equal s0's on every test (nudged, at most
                  max_sessions)
  s2  ind         a second, independent line started together with s0 (the old agentic_tir explorer design = K independent
                  explorers)
  s3  refine_seq  starts after both s0 and s1 end: candidates/A = s0, B = s1; prompts v3refine ask it to write
                  final/attempt_{1,2}/test_J.json
  s4  refine_exp  as above, A = s0, B = s2 (explorer → verify → refine)
  s5  refine_all  starts after s0, s1 and s2 all end: candidates/A = s0, B = s1 (saw A, deliberately different), C = s2
                  (independent), prompts v3refine3
  s6  verify      starts after s0 ends (10-12, the user): takes over previous/ (including s0's handoff report); prompts
                  v3verify ask it to verify s0; first line of VERDICT.md CONFIRMED = no problem, no second answer;
                  ALTERNATIVE = points out the problem and submits a corrected / alternative answer
  s7  rerefine    filler (10-13, the user: use up the time): after refine_exp finishes, one more independent refine_exp
                  (same A, B, same prompts, cannot see the first refiner). Uses only slots the main line will never use
                  again (Scheduler); while main-line episodes are unfinished it starts only with --filler-guard left
                  before the deadline (it takes no slots, but the extra rows slow every step, and the last few main-line
                  refiners would be dragged past the deadline), once the main line is done only with --filler-min-left;
                  one killed by the deadline is not used. B200 eval120-r2, the same s0 / s2 with three refiners voting
                  pairwise offline: none of the six orderings loses to using only the earlier one, +0.7 on average

Arms and scoring (official partial credit: a test counts as right if either of its two attempts is right, per task =
fraction of tests right, mean over 120 tasks; tasks without a result count 0):
  first        attempt_1 = s0 (pass@1, the run_arc2 control)
  seq          s0, s1
  ind          s0, s2 in harness order (all-train-passing first, then more pairs passed; identical outputs merged)
  refine_seq   s3's final/; missing attempts filled from seq (with the one that differs)
  refine_exp   s4's final/; missing ones filled from ind
  rank3        s0, s1, s2 in harness order, top two (the three-candidate control without a refiner)
  refine_all   s5's final/; missing ones filled from the order of the three candidates
  verify       s0 + s6's alternative answer (tests that are CONFIRMED or equal to s0 get no attempt_2)
  refine_vote  vote (vote2) over the [a1, a2] of refine_exp and s7; tasks without s7 = refine_exp
Arms under the same s0 are paired; the summary also records the tests the refiner rescued / lost and the share of
attempt_2 duplicating attempt_1.

``--reuse <old run>``: sessions of ``--reuse-kinds`` are not rerun; their workspaces (without .driver) and results are
copied from the old run (config and shared prompts must be identical) and only the rest run -- e.g. swapping only the
refiner on the same s0 / s1 / s2, for a tighter pairing at the cost of the refiner alone.

``--refine-agree-prompts P`` (+ ``--refine-agree-budget N``): for tasks where s0 and s2 both pass all train and have the
same output on every test, refine_exp switches to prompts P and budget N (default --refine-budget), and the workspace's
final/attempt_1 is pre-filled with that common output -- the refiner does only attempt_2.
10-25 eval120fix: such tasks are 103/120 and eat 80% of the refiner's tokens (~47k per task), yet rescue only 2 more
tests (both the other reading of an ambiguous task).

``--ind-gate-tokens T`` (+ ``--refine-solo-prompts P``): s2 waits for s0 to finish before deciding whether to start --
if s0 passes all train, wraps up normally, has an output for every test, and used no more than T output tokens, s2 is
not started (recorded as gated: result file ``gated: true``, outputs copied from s0, 0 tokens), and the ind arm has only
s0; for such tasks refine_exp has only candidate A and uses prompts P (attempt_1 pre-filled with s0's output, the
refiner does only attempt_2, budget = --refine-agree-budget). Only with --arms ind,refine_exp. 10-25 eval120fix offline:
at T = 60k s2 starts on only 48/120 tasks, s0 + s2 pass@2 −0.65 (mean of 12 paired groups), pair-base-01 with the
refiner is estimated not to lose score, and the whole pipeline uses −12% tokens.

``--handoff-file REPORT.md``: an s0 / s2 workspace without this file (or with under 200 bytes) does not count as done
(nudged); paired with the config's prompts v3doc (the brief asks for a handoff report). The later s1 / s6 / refiners can
read it from previous / candidates.
``--review-file REVIEW.md``: likewise for s6, which does not count as done without a review report; paired with
``--verify-prompts v3verify2``.

View text only, no cache_salt. fence none is how node1 runs it (--driver-bin points to a bwrap-wrapped codex).

TITO (config ``tito: {"required": true}``, needs ``--pool-label``): before starting, verify like run_arc2 that the server
is keeping the ledger (prepare_run); at the end (normal end, deadline, and SIGTERM / SIGINT cancellation all go through
the same wrap-up) write ``tito_manifest.json``: one item per episode, id = ``<task>__s<k>`` (the results file name and
workspace), ``task`` = the task id (the join's group: s0 / s2 / refiners of one task form one group), ``kind`` = the
session kind; the run-level ``meta`` records ``pool`` (--pool-label). Only episodes actually started in this run are
recorded (those brought over by --reuse belong to the old run's ledger).

Kaggle (kaggle_run.py calls ``run()``; fence posix, no netns): ``--challenges`` (+ ``--solutions`` in Save & Run)
replaces --tasks-dir, and a hidden test without outputs is not scored (correct / score recorded as None); once
``--deadline-unix`` arrives no new episodes start, and running ones are killed and harvested as usual; ``--dp-size``: each
episode, when it starts, picks the DP with the fewest running episodes at that moment (s0 / s2 / refiner of one task
need not share a DP); ``--solvers-first``: s0 > s2 > refiner (a freed slot goes to the highest-priority ready episode,
no stage barrier); ``--submission-out``: before starting, write an all-placeholder submission, then atomically rewrite it
with the ``--submit-arm`` combination after every finished episode -- when that arm's sessions are not all in
(deadline, failure), the ones present are used (submit_attempts).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arc2 as base  # noqa: E402
import arc2_render as render  # noqa: E402

KINDS = ("first", "seq", "ind", "refine_seq", "refine_exp", "refine_all", "verify", "rerefine")
SAMPLE = {k: i for i, k in enumerate(KINDS)}                  # episode <task>__s<k>
#: Filler kinds: not in the main-line queue (the Scheduler's filler queue), not counted in pending; their failing or
#: never starting drags no one else down
FILLERS = ("rerefine",)
#: Filler episodes stopped for these reasons are not used (final/ may be just the first version written early on)
UNUSABLE_FILLER_STOPS = ("deadline", "cancelled", "wall")
#: Filler failures (traceback) do not count toward the config's max_failures (an optional episode must not cancel the
#: whole run); filler is switched off once this many occur
FILLER_MAX_FAILURES = 3
#: refiner → its candidates (A is always first, then B, C in order)
REFINERS = {"refine_seq": ("seq",), "refine_exp": ("ind",), "refine_all": ("seq", "ind"), "rerefine": ("ind",)}
NEEDS = {"first": (), "seq": ("first",), "ind": (), "verify": ("first",), **{k: ("first", *v) for k, v in REFINERS.items()},
          "rerefine": ("first", "ind", "refine_exp")}          # rerefine votes with refine_exp: waits for it
PRIORITY = {"refine_seq": 0, "refine_exp": 0, "refine_all": 0, "seq": 1, "verify": 1, "first": 2, "ind": 2}  # lower first
#: --solvers-first (Kaggle, 10-12, the user): s0 > s2 > refiner. When the deadline cuts off, what is lost is refiners, and
#: attempts are filled from the s0 / s2 order.
PRIORITY_SOLVERS_FIRST = {"first": 0, "ind": 1, "seq": 1, "verify": 2, "refine_seq": 2, "refine_exp": 2, "refine_all": 2}
ARMS = ("first", "seq", "verify", "ind", "rank3", "refine_seq", "refine_exp", "refine_all", "refine_vote")
#: the sessions each arm uses (the arm's cost, which arms can be computed)
ARM_SESSIONS = {"first": ("first",), "seq": ("first", "seq"), "verify": ("first", "verify"), "ind": ("first", "ind"),
                "rank3": ("first", "seq", "ind"),
                **{k: ("first", *v, k) for k, v in REFINERS.items() if k not in FILLERS},
                "refine_vote": ("first", "ind", "refine_exp", "rerefine")}
#: which arm a refiner's missing attempts are filled from, and which arm its rescued / lost counts are compared against
RAW = {"refine_seq": "seq", "refine_exp": "ind", "refine_all": "rank3"}
#: the summary's rescued / lost: refine_vote is compared against refine_exp (under the same first refiner: the difference
#: one more re-refiner makes)
COMPARE = {**RAW, "refine_vote": "refine_exp"}
SEQ_PROMPTS = ("brief.md", "kickoff.md", "nudge.md", "resume_fresh.md", "previous_passed.md", "previous_failed.md")
REFINE_PROMPTS = ("brief.md", "kickoff.md", "nudge.md", "resume_fresh.md", "midpoint_reminder.md")
VERIFY_PROMPTS = ("brief.md", "kickoff.md", "nudge.md", "resume_fresh.md")
VERDICTS = ("CONFIRMED", "ALTERNATIVE")
HANDOFF_MIN_BYTES = 200
#: Files copied into previous / candidates: these at the workspace top level (check.py, arc.py and TOOLS.md, placed by
#: the harness itself, are not copied)
COPY_SUFFIXES = (".py", ".md", ".txt", ".json")
COPY_SKIP = {"check.py", "arc.py", "TOOLS.md", "AGENTS.md", "budget.txt", "overview.txt"}
COPY_MAX_BYTES = 256 * 1024


# --------------------------------------------------------------------------- prompts

def load_prompt_set(name: str, required: tuple[str, ...]) -> dict[str, str]:
    pdir = HERE / "prompts" / name
    out = {}
    for f in required:
        text = (pdir / f).read_text()
        if not text.strip():
            raise SystemExit(f"{pdir / f} is empty")
        out[f] = text
    return out


def escape(text: str) -> str:
    """For templates that will be .format()-ed later: braces must become literal."""
    return text.replace("{", "{{").replace("}", "}}")


# --------------------------------------------------------------------------- previous / candidates

def train_status(r: dict) -> str:
    passed = sum(v == "PASS" for v in r["train"])
    text = f"it passes {passed} of {len(r['train'])} training pairs"
    if r["load_error"]:
        text = f"its solution.py cannot be loaded ({r['load_error']})"
    return text


def train_all(r: dict) -> bool:
    return not r["load_error"] and all(v == "PASS" for v in r["train"])


def copy_solver(src_ws: Path, r: dict, dst: Path) -> None:
    """Copy one session's results to dst (read-only): the solution it was harvested with (the .best copy when the
    snapshot fell back), notes, top-level helpers, answer/, the check.txt the harness wrote and each test's final
    output."""
    dst.mkdir(parents=True)
    for f in sorted(src_ws.iterdir()):
        if f.is_file() and f.suffix in COPY_SUFFIXES and f.name not in COPY_SKIP and f.stat().st_size <= COPY_MAX_BYTES:
            shutil.copyfile(f, dst / f.name)
    fallback = r["snapshot_fallback"]
    if fallback is not None and fallback["used"]:
        shutil.copyfile(src_ws / ".best" / "solution.py", dst / "solution.py")
    if (src_ws / "answer").is_dir():
        answers = [f for f in sorted((src_ws / "answer").glob("test_*.json")) if f.stat().st_size <= COPY_MAX_BYTES]
        if answers:
            (dst / "answer").mkdir()
            for f in answers:
                shutil.copyfile(f, dst / "answer" / f.name)
    lines = [f"train {k}: {v}" for k, v in enumerate(r["train"], 1)]
    lines.append(f"{sum(v == 'PASS' for v in r['train'])}/{len(r['train'])} training pairs pass"
                 + (f" (solution.py cannot be loaded: {r['load_error']})" if r["load_error"] else ""))
    for j, (grid, src) in enumerate(zip(r["attempt_1"], r["source"], strict=True), 1):
        if grid is None:
            err = r["transform_errors"][j - 1] or r["answer_errors"][j - 1] or "no output"
            lines.append(f"test {j}: no output ({err})")
        else:
            lines.append(f"test {j}: output {len(grid)}x{len(grid[0])} (from {src.replace('_', ' ')})")
            (dst / f"test_{j}_output.json").write_text(json.dumps(grid))
            (dst / f"test_{j}_output.txt").write_text(render.grid_text(grid))
    (dst / "check.txt").write_text("\n".join(lines) + "\n")
    # Read-only files are only a hint (the agent is the same uid and can change them back); directories stay writable so
    # a later rm -r can delete them. The harness never reads anything back from here.
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


def candidates_summary(task: dict, ra: dict, rb: dict) -> str:
    """A paragraph of the refiner's brief: how many train pairs each of the two candidates passes, and whether each test's
    outputs agree."""
    lines = []
    for name, r in (("A", ra), ("B", rb)):
        lines.append(f"- {name}: {train_status(r)}.")
    for j in range(len(task["test"])):
        a, b = ra["attempt_1"][j], rb["attempt_1"][j]
        if a is None and b is None:
            what = "neither gives an output"
        elif a is None or b is None:
            what = f"only {'B' if a is None else 'A'} gives an output"
        elif a == b:
            what = "A and B give the same output"
        else:
            what = "A and B give different outputs"
        lines.append(f"- test {j + 1}: {what}.")
    return "\n".join(lines)


def candidates_summary_n(task: dict, cands: list[tuple[str, dict]]) -> str:
    """The same paragraph for three or more candidates: how many train pairs each passes; per test, which outputs are
    identical and which are missing."""
    lines = [f"- {name}: {train_status(r)}." for name, r in cands]
    for j in range(len(task["test"])):
        groups: dict[str, list[str]] = {}
        none = []
        for name, r in cands:
            grid = r["attempt_1"][j]
            if grid is None:
                none.append(name)
            else:
                groups.setdefault(json.dumps(grid), []).append(name)
        parts = []
        for names in groups.values():
            if len(names) > 1:
                parts.append(f"{' and '.join(names)} give the same output")
            else:
                parts.append(f"{names[0]} gives an output of its own")
        if none:
            parts.append(f"{' and '.join(none)} give{'s' if len(none) == 1 else ''} no output")
        lines.append(f"- test {j + 1}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


def read_final(ws: Path, n_test: int) -> tuple[list[list], list[str]]:
    """The refiner's final/attempt_{1,2}/test_J.json: [[a1, a2] per test] (missing or broken = None) and a list of
    problems. Grid rules (as_grid, valid_grid) come from check.py's copy."""
    out, problems = [], []
    for j in range(1, n_test + 1):
        pair = []
        for k in (1, 2):
            f = ws / "final" / f"attempt_{k}" / f"test_{j}.json"
            grid = None
            if f.is_file():
                try:
                    value = CHECK.as_grid(json.loads(f.read_text()))
                    problem = CHECK.valid_grid(value)
                except (ValueError, OSError) as error:
                    problem = f"{type(error).__name__}: {error}"
                if problem is None:
                    grid = value
                else:
                    problems.append(f"final/attempt_{k}/test_{j}.json: {problem}")
            pair.append(grid)
        out.append(pair)
    return out, problems


def _load_check():
    import importlib.util
    spec = importlib.util.spec_from_file_location("arc2_check_rules", HERE / "check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load_check()


# --------------------------------------------------------------------------- status hooks

def outputs(h: dict) -> list:
    return [t["answer_file"] if t["answer_file"] is not None else t["transform"] for t in h["test"]]


def seq_status(previous: dict):
    """s1's "done yet?": per status_text; when s0 passes all train, at least one test's output must also differ from
    s0's."""
    prev_all = train_all(previous)

    def status(h: dict, ws: Path) -> tuple[bool, str]:
        done, text = base.status_text(h)
        if done and prev_all and outputs(h) == previous["attempt_1"]:
            return False, (text + ", but its output for every test input is the same as previous/ "
                                  "(a copy adds nothing as the second answer)")
        return done, text
    return status


def read_verdict(ws: Path) -> str | None:
    """If the first line of VERDICT.md (markdown markup stripped) is CONFIRMED / ALTERNATIVE, return it, else None."""
    f = ws / "VERDICT.md"
    if not f.is_file():
        return None
    lines = [l.strip().strip("#*` ").upper() for l in f.read_text(errors="replace").splitlines() if l.strip()]
    return lines[0] if lines and lines[0] in VERDICTS else None


def verify_status(previous: dict, review: str | None = None):
    """s6's "done yet?": CONFIRMED (allowed only when s0 passes all train) is done; ALTERNATIVE must be done per
    status_text, with at least one test differing from s0. ``review`` (--review-file): also requires a proper review
    report in the workspace."""
    prev_all = train_all(previous)

    def status(h: dict, ws: Path) -> tuple[bool, str]:
        done, text = verdict_status(h, ws)
        f = ws / review if review else None
        if f is not None and (not f.is_file() or f.stat().st_size < HANDOFF_MIN_BYTES):
            return False, text + f", and {review} (your review) is missing or nearly empty"
        return done, text

    def verdict_status(h: dict, ws: Path) -> tuple[bool, str]:
        verdict = read_verdict(ws)
        if verdict == "CONFIRMED":
            if prev_all:
                return True, "VERDICT.md says CONFIRMED"
            return False, ("VERDICT.md says CONFIRMED, but the previous solution does not pass every training pair, "
                           "so it cannot be confirmed: write ALTERNATIVE and solve the puzzle")
        if verdict is None:
            return False, "VERDICT.md is missing or its first line is neither CONFIRMED nor ALTERNATIVE"
        done, text = base.status_text(h)
        if done and outputs(h) == previous["attempt_1"]:
            return False, (text + ", but VERDICT.md says ALTERNATIVE and your output for every test input is the same "
                                  "as previous/")
        return done, "VERDICT.md says ALTERNATIVE; " + text
    return status


def handoff_status(name: str):
    """--handoff-file: per status_text, and the workspace must also hold a proper handoff report."""
    def status(h: dict, ws: Path) -> tuple[bool, str]:
        done, text = base.status_text(h)
        f = ws / name
        if not f.is_file() or f.stat().st_size < HANDOFF_MIN_BYTES:
            return False, text + f", and {name} (the hand-off report) is missing or nearly empty"
        return done, text
    return status


def verify_offer(r0: dict, r: dict, verdict: str | None) -> list:
    """s6's second answer (per test): none for CONFIRMED; otherwise its output, except on tests where it equals s0's."""
    if verdict == "CONFIRMED":
        return [None] * len(r["attempt_1"])
    return [g if g is not None and g != a else None for g, a in zip(r["attempt_1"], r0["attempt_1"], strict=True)]


def refine_status(n_test: int):
    def status(h: dict, ws: Path) -> tuple[bool, str]:
        final, problems = read_final(ws, n_test)
        missing = [f"attempt_{k + 1} for test {j + 1}" for j, pair in enumerate(final) for k in (0, 1) if pair[k] is None]
        # attempt_2 equal to attempt_1 = an attempt thrown away (smoke 10-11: a refiner convinced the rule is unique copies
        # it); s1 follows the same rule
        same = [j + 1 for j, pair in enumerate(final) if pair[0] is not None and pair[0] == pair[1]]
        if not missing and not problems and not same:
            return True, "final/ holds two different attempts for every test input"
        parts = ["final/ is missing " + ", ".join(missing)] if missing else []
        if same:
            parts.append(f"attempt_2 is the same as attempt_1 for test input(s) {same} -- a copy adds nothing; "
                         "make it the most plausible different reading")
        if problems:
            parts.append("invalid: " + "; ".join(problems))
        return False, "; ".join(parts)
    return status


# --------------------------------------------------------------------------- attempts per arm

def rank_two(results: list[dict], j: int) -> list:
    """The top two of rank_all."""
    return rank_all(results, j)[:2]


def rank_all(results: list[dict], j: int) -> list:
    """Harness order: identical outputs merged; all-train-passing first, then more pairs passed, then more votes, then
    earlier run. Returns all distinct outputs."""
    groups: dict[str, dict] = {}
    for order, r in enumerate(results):
        grid = r["attempt_1"][j]
        if grid is None:
            continue
        key = json.dumps(grid)
        passed = sum(v == "PASS" for v in r["train"]) if not r["load_error"] else -1
        g = groups.setdefault(key, {"grid": grid, "all": False, "passed": -1, "votes": 0, "first": order})
        g["all"] |= train_all(r)
        g["passed"] = max(g["passed"], passed)
        g["votes"] += 1
    ranked = sorted(groups.values(), key=lambda g: (not g["all"], -g["passed"], -g["votes"], g["first"]))
    return [g["grid"] for g in ranked]


def fill(primary: list, backup: list) -> list:
    """The refiner's [a1, a2]: a missing a1 takes backup's first; a missing a2 takes the first in backup that differs
    from a1."""
    a1, a2 = primary
    if a1 is None:
        a1 = next((g for g in backup if g is not None), None)
    if a2 is None:
        a2 = next((g for g in backup if g is not None and g != a1), None)
    return [a1, a2]


def vote2(first: list, second: list) -> list:
    """Combine two refiners' [a1, a2] into two attempts: a1 gets two votes, a2 one, take the top two; ties go in the order
    first's a1, first's a2, second's a1, second's a2 (= when the two a1 agree, first's a2 is kept; when they differ,
    usually the two a1)."""
    score: dict[str, int] = {}
    order: dict[str, tuple] = {}
    for o, pair in enumerate((first, second)):
        for k, w in ((0, 2), (1, 1)):
            if pair[k] is None:
                continue
            key = json.dumps(pair[k])
            score[key] = score.get(key, 0) + w
            order.setdefault(key, (o, k))
    top = sorted(score, key=lambda key: (-score[key], order[key]))[:2]
    return ([json.loads(key) for key in top] + [None, None])[:2]


def arm_attempts(task: dict, rs: dict[str, dict], finals: dict[str, list]) -> dict[str, list | None]:
    """Each arm's [a1, a2] per test; an arm missing a session result is None (counts 0)."""
    n = len(task["test"])
    out: dict[str, list | None] = {arm: None for arm in ARMS}
    if "first" not in rs:
        return out
    r0 = rs["first"]
    out["first"] = [[r0["attempt_1"][j], None] for j in range(n)]
    if "seq" in rs:
        out["seq"] = [[r0["attempt_1"][j], rs["seq"]["attempt_1"][j]] for j in range(n)]
    if "verify" in rs:
        out["verify"] = [[r0["attempt_1"][j], rs["verify"]["offer"][j]] for j in range(n)]
    if "ind" in rs:
        out["ind"] = [(rank_two([r0, rs["ind"]], j) + [None, None])[:2] for j in range(n)]
    if "seq" in rs and "ind" in rs:
        three = [r0, rs["seq"], rs["ind"]]
        out["rank3"] = [(rank_two(three, j) + [None, None])[:2] for j in range(n)]
    for arm, raw in RAW.items():
        if arm in finals and out[raw] is not None:
            # what refine_all lacks is filled from the full order of the three candidates (not just the top two)
            backup = ([rank_all([r0, rs["seq"], rs["ind"]], j) for j in range(n)] if arm == "refine_all"
                      else out[raw])
            out[arm] = [fill(finals[arm][j], backup[j]) for j in range(n)]
    # refine_vote: the re-refiner (filler, present for only some tasks) votes with refine_exp; tasks without a re-refiner =
    # refine_exp
    if out["ind"] is not None:
        second = [fill(finals["rerefine"][j], out["ind"][j]) for j in range(n)] if "rerefine" in finals else None
        if out["refine_exp"] is not None and second is not None:
            out["refine_vote"] = [vote2(a, b) for a, b in zip(out["refine_exp"], second, strict=True)]
        else:
            out["refine_vote"] = out["refine_exp"] if out["refine_exp"] is not None else second
    return out


def pair_correct(pair: list | None, gt) -> bool:
    return pair is not None and any(a is not None and a == gt for a in pair)


# --------------------------------------------------------------------------- summary

def summarize(tasks: dict, by_task: dict[str, dict], finals: dict[str, dict], arms: list[str]) -> dict:
    total = {arm: 0.0 for arm in arms}
    pairs = {arm: 0 for arm in arms}
    a1_only = {arm: 0.0 for arm in arms}
    dup = {arm: 0 for arm in arms}
    oracle = 0.0
    refine = {arm: {"gained": 0, "lost": 0} for arm in arms if arm in COMPARE}
    first_all_pass = 0
    n_pairs = sum(len(t["test"]) for t in tasks.values())
    per_task = {}
    for tid, task in tasks.items():
        rs = by_task.get(tid, {})
        att = arm_attempts(task, rs, {a: finals[a][tid] for a in finals if tid in finals[a]})
        gts = [t["output"] for t in task["test"]]
        n = len(gts)
        row = {}
        for arm in arms:
            ok = [pair_correct(att[arm][j] if att[arm] else None, gts[j]) for j in range(n)]
            ok1 = [att[arm] is not None and att[arm][j][0] is not None and att[arm][j][0] == gts[j] for j in range(n)]
            total[arm] += sum(ok) / n
            a1_only[arm] += sum(ok1) / n
            pairs[arm] += sum(ok)
            if att[arm] is not None:
                dup[arm] += sum(p[1] is None or p[1] == p[0] for p in att[arm])
            row[arm] = ok
        for arm, raw in COMPARE.items():
            if arm in refine:
                refine[arm]["gained"] += sum(a and not b for a, b in zip(row[arm], row.get(raw, [False] * n)))
                refine[arm]["lost"] += sum(b and not a for a, b in zip(row[arm], row.get(raw, [False] * n)))
        seen = [r["attempt_1"] for k, r in rs.items() if k in ("first", "seq", "ind")]
        oracle += sum(any(s[j] is not None and s[j] == gts[j] for s in seen) for j in range(n)) / n
        if "first" in rs and train_all(rs["first"]):
            first_all_pass += 1
        per_task[tid] = {arm: row[arm] for arm in arms}
    n_tasks = len(tasks)
    # each arm's cost = output tokens of the sessions it uses (tasks missing a session are counted short, as they are)
    arm_tokens = {arm: sum(r["usage"]["proxy_output_tokens"] for rs in by_task.values() for k, r in rs.items()
                           if k in ARM_SESSIONS[arm]) for arm in arms}
    usage = {}
    for kind in KINDS:
        toks = [r["usage"]["proxy_output_tokens"] for rs in by_task.values() for k, r in rs.items()
                if k == kind and not r.get("gated")]
        if toks:
            usage[kind] = {"episodes": len(toks), "output_tokens_total": sum(toks),
                           "output_tokens_mean": round(sum(toks) / len(toks))}
    return {
        "tasks": n_tasks, "test_pairs": n_pairs, "arms": arms,
        "score_pass2_partial": {arm: round(100 * total[arm] / n_tasks, 2) for arm in arms},
        "score_attempt1_only": {arm: round(100 * a1_only[arm] / n_tasks, 2) for arm in arms},
        "test_pairs_correct": pairs,
        "attempt2_missing_or_duplicate": {arm: dup[arm] for arm in arms if arm != "first"},
        "oracle_any_solver_session": round(100 * oracle / n_tasks, 2),
        "refiner_vs_raw_test_pairs": refine,
        "first_train_all_pass": first_all_pass,
        "usage_by_session": usage,
        "output_tokens_by_arm": arm_tokens,
        **({"verify_verdicts": {v: sum(rs["verify"]["verdict"] == v for rs in by_task.values() if "verify" in rs)
                                for v in (*VERDICTS, None)}} if any("verify" in rs for rs in by_task.values()) else {}),
        "per_task": per_task,
        "scoring": "official ARC-AGI-2 partial score: a test pair is correct if attempt_1 or attempt_2 matches; per task = "
                   "correct test pairs / test pairs; mean over all tasks; a task whose sessions are missing counts 0. "
                   "score_attempt1_only scores attempt_1 alone. oracle = any of s0/s1/s2 right.",
    }


def submit_attempts(task: dict, rs: dict[str, dict], finals_tid: dict[str, list], arm: str) -> list:
    """[a1, a2] for submission (per test). The arm's solver sessions (those in ARM_SESSIONS that are not refiners) are
    used, however many are present, in rank_all order; if arm is a refiner and it has a final, fill (= that arm of
    arm_attempts). With all sessions present it equals arm_attempts.
    refine_vote: vote between refine_exp's (filled) attempts and the re-refiner's; if only one of them exists, use it."""
    if arm == "refine_vote":
        first = submit_attempts(task, rs, finals_tid, "refine_exp")
        if "rerefine" not in finals_tid:
            return first
        solvers = [rs[k] for k in ("first", "ind") if k in rs]
        second = [fill(finals_tid["rerefine"][j], rank_all(solvers, j)) for j in range(len(task["test"]))]
        if "refine_exp" not in finals_tid:
            return second
        return [vote2(a, b) for a, b in zip(first, second, strict=True)]
    solvers = [rs[k] for k in ARM_SESSIONS[arm] if k in rs and k not in REFINERS]
    out = []
    for j in range(len(task["test"])):
        backup = rank_all(solvers, j)
        out.append(fill(finals_tid[arm][j], backup) if arm in finals_tid else (backup + [None, None])[:2])
    return out


def write_submission(path: Path, tasks: dict, att_by_task: dict[str, list | None]) -> None:
    """Kaggle format. Attempts without an answer get the test input as a placeholder. A bad format is a harness fault:
    raise."""
    sub = {}
    for tid, task in tasks.items():
        att = att_by_task.get(tid)
        sub[tid] = [{"attempt_1": (att[j][0] if att and att[j][0] is not None else t["input"]),
                     "attempt_2": (att[j][1] if att and att[j][1] is not None else t["input"])}
                    for j, t in enumerate(task["test"])]
        for entry in sub[tid]:
            for grid in entry.values():
                problem = CHECK.valid_grid(grid)
                if problem is not None:
                    raise RuntimeError(f"{tid}: submission grid is invalid ({problem})")
    base.atomic_write(path, json.dumps(sub))


# --------------------------------------------------------------------------- scheduler

class Scheduler:
    """Work queue with dependencies + priorities: an episode enters the queue only once all its dependencies are done; an
    episode whose dependency failed is recorded as failed and not started.

    Fillers (FILLERS) use a separate queue and only slots the main line will never use again: they are dispatched only
    while unfinished main-line episodes (pending: queued, waiting on dependencies, running) plus running fillers number
    fewer than workers, so main-line episodes never have to wait for a slot. ``filler_gate(pending)`` returns "open"
    (dispatch), "wait" (not now, may open later) or "closed" (all queued fillers dropped, ``dropped``). pop returns
    ("main"|"filler", item), or None when there is nothing to do."""

    def __init__(self, workers: int, filler_gate=None) -> None:
        self.workers = workers
        self.cv = threading.Condition()
        self.heap: list = []
        self.fillers: list = []
        self.counter = 0
        self.pending = 0
        self.filler_running = 0
        self.filler_gate = filler_gate if filler_gate is not None else (lambda pending: "closed")
        self.dropped: list = []

    def push(self, prio: tuple, item) -> None:
        with self.cv:
            heapq.heappush(self.heap, (prio, self.counter, item))
            self.counter += 1
            self.cv.notify()

    def push_filler(self, prio: tuple, item) -> None:
        with self.cv:
            heapq.heappush(self.fillers, (prio, self.counter, item))
            self.counter += 1
            self.cv.notify_all()

    def add_pending(self, n: int) -> None:
        with self.cv:
            self.pending += n

    def done_one(self) -> None:
        with self.cv:
            self.pending -= 1
            self.cv.notify_all()

    def filler_done(self) -> None:
        with self.cv:
            self.filler_running -= 1
            self.cv.notify_all()

    def pop(self):
        with self.cv:
            while True:
                if self.heap:
                    return "main", heapq.heappop(self.heap)[2]
                if self.fillers and self.pending + self.filler_running < self.workers:
                    gate = self.filler_gate(self.pending)
                    if gate == "open":
                        self.filler_running += 1
                        return "filler", heapq.heappop(self.fillers)[2]
                    if gate == "closed":
                        self.dropped += [entry[2] for entry in self.fillers]
                        self.fillers.clear()
                    elif gate != "wait":
                        raise ValueError(f"filler_gate returned {gate!r}")
                if self.pending <= 0 and not self.fillers:
                    return None
                self.cv.wait(timeout=5)


def reuse_sessions(old: Path, kinds: list[str], cfg: dict, prompts: dict[str, str], tids: list[str],
                   episodes: dict, run_dir: Path) -> dict:
    """Bring these session kinds over from the old run: results read as is, workspaces copied (without .driver etc.; the
    refiner's candidates read only the top level, .best, answer/). The config and the prompts both sides have must be
    identical, otherwise the comparison is not of the same thing."""
    state = json.loads((old / "STATE.json").read_text())
    # concurrency is only scheduling (how many episodes at once) and does not change the episodes themselves; it often
    # changes between collections (lowered when KV overflows), so it is not compared
    same = lambda c: {k: v for k, v in c.items() if k != "concurrency"}
    if same(state["config"]) != same(cfg):
        raise SystemExit(f"--reuse {old}: its config differs from --config (beyond concurrency)")
    differ = sorted(n for n, h in state["prompt_sha256"].items()
                    if n in prompts and hashlib.sha256(prompts[n].encode()).hexdigest() != h)
    if differ:
        raise SystemExit(f"--reuse {old}: prompts changed since: {differ}")
    out = {}
    for tid in tids:
        for k in kinds:
            name = episodes[(tid, k)].name
            src_result = old / "results" / f"{name}.json"
            if not src_result.exists():
                raise SystemExit(f"--reuse {old}: no result for {name}")
            r = json.loads(src_result.read_text())
            if r["kind"] != k or r["tid"] != tid:
                raise SystemExit(f"--reuse {old}: {name} is {r['kind']} {r['tid']}, expected {k} {tid}")
            shutil.copytree(old / "tasks" / name, run_dir / "tasks" / name, symlinks=True,
                            ignore=shutil.ignore_patterns(".driver", ".tmp", "__pycache__"))
            shutil.copyfile(src_result, run_dir / "results" / f"{name}.json")
            out[(tid, k)] = r
    base.log(f"reused {len(out)} sessions ({kinds}) from {old}")
    return out


def tito_items(run_dir: Path, done: list[dict], failures: list[dict]) -> list[dict]:
    """Items of ``tito_manifest.json`` (called only when the config has ``tito``): one per episode as in
    run_arc2.tito_items (id = the results' ``task`` field = ``<task>__s<k>``; thread, digest, outcome as before), plus
    ``task`` = the task id (the join's group) and ``kind``. ``done`` / ``failures`` hold only episodes started in this
    run (those brought over by --reuse are absent: their segments are in the old run's ledger and manifest); an episode
    not started because a dependency failed is recorded as failed like run_arc2 does, with no thread."""
    rows = {}
    for row in [*done, *failures]:
        if row["task"] in rows:
            raise RuntimeError(f"{row['task']}: both a result and a failure (or twice) -- the manifest needs one each")
        if row["task"] != f"{row['tid']}__s{SAMPLE[row['kind']]}":
            raise RuntimeError(f"{row['task']}: is not episode {row['tid']} {row['kind']}")
        rows[row["task"]] = row
    return [{**item, "task": rows[item["id"]]["tid"], "kind": rows[item["id"]]["kind"]}
            for item in base.tito_items(run_dir, done, failures)]


def pick_dp(ctx) -> int | None:
    """The DP with the fewest running episodes right now (ties go to the smaller rank); one fixed DP per episode."""
    if ctx.dp_live is None:
        return None
    with base.LOCK:
        dp = min(range(len(ctx.dp_live)), key=lambda rank: (ctx.dp_live[rank], rank))
        ctx.dp_live[dp] += 1
    return dp


RATE_WINDOW_SECONDS = 600


def heartbeat(stop: threading.Event, ctx, state: dict, lock: threading.Lock) -> None:
    """One line: done / running counts per session kind, running per DP, cumulative output tokens (proxy count), tok/s
    over the last 10 minutes, deadline."""
    import collections
    samples: collections.deque = collections.deque()
    while not stop.wait(ctx.cfg["heartbeat_seconds"]):
        now = time.time()
        with lock:
            done = collections.Counter(r["kind"] for r in state["done"])
            running = collections.Counter(k for _, k in state["running"].values())
            not_started, failed = len(state["not_started"]), len(state["failed"])
        with base.LOCK:
            proxies = list(ctx.proxies.values())
            dp = list(ctx.dp_live) if ctx.dp_live is not None else None
        cum = sum(p.output_tokens for p in proxies)
        samples.append((now, cum))
        while len(samples) > 1 and now - samples[1][0] >= RATE_WINDOW_SECONDS:
            samples.popleft()
        t0, c0 = samples[0]
        rate = (cum - c0) / (now - t0) if now > t0 else 0.0
        parts = [f"HEARTBEAT done {sum(v for k, v in done.items() if k not in FILLERS)}/{state['total']} {dict(done)}",
                 f"running {sum(running.values())} {dict(running)}", f"dp_running {dp}", f"failed {failed}",
                 f"not_started {not_started}", f"out_tok {cum:,}",
                 f"tok_s_{int(min(RATE_WINDOW_SECONDS, now - t0) // 60)}min {rate:.1f}"]
        if ctx.deadline is not None:
            parts.append(f"deadline_in {max(0, ctx.deadline - now) / 60:.1f}min")
        try:
            metrics = base.sglang_metrics(ctx.args.model_base_url)
        except OSError as error:
            metrics = {"error": repr(error)}
        base.log(" ".join(parts) + " " + json.dumps(metrics))


def main(argv: list[str] | None = None) -> None:
    summary, failures, run_dir = run(argv)
    if failures:
        raise SystemExit(f"{len(failures)} episode(s) failed; see {run_dir / 'failures.jsonl'}")


def run(argv: list[str] | None = None, record: dict | None = None) -> tuple[dict, list[dict], Path]:
    """Run until done (or the deadline / cancellation); return (summary, failures, run_dir). Individual episode failures
    are only recorded, not raised (Kaggle: the submission is still written); configuration errors raise before the first
    episode starts. ``record``: extra records written into STATE.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks-dir", type=Path, help="one json per task with test outputs (public eval)")
    src.add_argument("--challenges", type=Path, help="Kaggle {task_id: {train, test:[{input}]}}")
    ap.add_argument("--solutions", type=Path, help="with --challenges: {task_id: [grid per test]} to score the run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--agent-python", type=Path, required=True)
    ap.add_argument("--driver-bin", type=Path, required=True)
    ap.add_argument("--model-base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--deny", default="", help="fence posix: colon-separated paths the episode users must not read")
    ap.add_argument("--concurrency", type=int, help="when the config's concurrency is null (Kaggle)")
    ap.add_argument("--dp-size", type=int, help="route each episode to the least busy of this many DPs")
    ap.add_argument("--deadline-unix", type=float, help="no new episode after this; running ones are stopped and harvested")
    ap.add_argument("--solvers-first", action="store_true", help="priority s0 > s2 > refiners (default: refiners first)")
    ap.add_argument("--submission-out", type=Path, help="also write the --submit-arm submission here, after every episode")
    ap.add_argument("--submit-arm", help="the arm written to submission.json / --submission-out "
                                         "(default: refine_exp when its sessions run, else first)")
    # default = the approach chosen by the two 10-11 / 10-12 runs (independent second line + refiner); experiments with
    # all arms use --arms seq,ind,refine_seq,refine_exp
    ap.add_argument("--arms", default="ind,refine_exp",
                    help="sessions besides s0 (first): any of seq, ind, refine_seq, refine_exp, refine_all, verify, "
                         "rerefine (filler: idle slots only)")
    ap.add_argument("--second-budget", type=int, help="output-token budget of s1/s2 (default: the config's)")
    ap.add_argument("--refine-budget", type=int, help="output-token budget of the refiners (default: half the config's)")
    ap.add_argument("--refine-wrapup", type=int,
                    help="the refiners' wrap-up allowance when the config has wrapup_tokens (default: half of it)")
    ap.add_argument("--filler-guard", type=float, default=7200.0,
                    help="while main episodes remain, filler episodes (rerefine) start only with at least this many "
                         "seconds to the deadline (they take no slot but slow every step of the main line's tail)")
    ap.add_argument("--filler-min-left", type=float, default=3600.0,
                    help="once the main line is done, filler episodes start only with at least this many seconds left")
    ap.add_argument("--seq-prompts", default="v3seq")
    ap.add_argument("--refine-prompts", default="v3refine", help="refiners over two candidates")
    ap.add_argument("--refine3-prompts", default="v3refine3", help="refine_all (three candidates)")
    ap.add_argument("--refine-agree-prompts",
                    help="refine_exp on a task whose s0 and s2 both pass every training pair with the same outputs: "
                         "these prompts, final/attempt_1 pre-filled with that output (default: --refine-prompts)")
    ap.add_argument("--ind-gate-tokens", type=int,
                    help="run s2 only when s0 failed a training pair, did not finish, left a test unanswered or used "
                         "more than this many output tokens (needs --refine-solo-prompts and --arms ind,refine_exp)")
    ap.add_argument("--refine-solo-prompts", help="refine_exp of a task whose s2 was gated off: one candidate (A = s0)")
    ap.add_argument("--refine-agree-budget", type=int,
                    help="those refiners' (and the --refine-solo-prompts ones') output-token budget "
                         "(default: --refine-budget)")
    ap.add_argument("--verify-prompts", default="v3verify", help="verify (s6)")
    ap.add_argument("--handoff-file", help="s0/s2 are not done until the workspace has this hand-off report (prompts v3doc)")
    ap.add_argument("--review-file", help="the verifier is not done until its workspace has this review (prompts v3verify2)")
    ap.add_argument("--reuse", type=Path, help="earlier run_arc2_seq run directory to take --reuse-kinds sessions from")
    ap.add_argument("--reuse-kinds", default="first,seq,ind")
    ap.add_argument("--pool-label", help="config tito only (required there): the task pool's name, "
                                         "written into tito_manifest.json's meta")
    args = ap.parse_args(argv)
    args.samples, args.salt_prefix = None, None          # prepare_run / Context read these two
    args.tasks_source = args.tasks_dir or args.challenges
    cfg = base.load_config(args.config, concurrency=args.concurrency)
    if cfg["view"] != "text" or "cache_salt" in cfg:
        raise SystemExit("run_arc2_seq supports view text, no cache_salt")
    # TITO: the config's tito was already validated by load_config (shape, codex's sglang_session_release); the server
    # is verified in prepare_run
    if "tito" in cfg and not args.pool_label:
        raise SystemExit("config tito needs a non-empty --pool-label (the task pool, written into tito_manifest.json)")
    if "tito" not in cfg and args.pool_label is not None:
        raise SystemExit("--pool-label given but the config has no tito: it would be written nowhere")
    tito = {"config": base.tito_module("manifest").config_identity(args.config)} if "tito" in cfg else None
    if args.solutions is not None and args.challenges is None:
        raise SystemExit("--solutions goes with --challenges")
    deny = [d for d in args.deny.split(":") if d]
    if cfg["fence"] == "posix":
        import os
        if os.geteuid() != 0:
            raise SystemExit("fence posix needs root")
        if not deny:
            raise SystemExit("fence posix needs --deny (the paths holding the answers)")
        hidden = [args.tasks_dir] if args.tasks_dir else [args.challenges, *([args.solutions] if args.solutions else [])]
        for path in hidden:
            if not any(path.resolve().is_relative_to(Path(d).resolve()) for d in deny):
                raise SystemExit(f"{path} is not under any --deny path")
    kinds = ["first"] + [k for k in args.arms.split(",") if k]
    unknown = set(kinds) - set(KINDS)
    if unknown or len(set(kinds)) != len(kinds):
        raise SystemExit(f"--arms: unknown or repeated {sorted(unknown) or kinds}")
    for k in kinds:
        if not set(NEEDS[k]) <= set(kinds):
            raise SystemExit(f"--arms: {k} needs {NEEDS[k]}")
    wrapup = cfg.get("wrapup_tokens")
    if wrapup is not None and "verify" in kinds:
        raise SystemExit("wrapup_tokens has no wrap-up message for the verifier (prompts v3verify*)")
    if args.submit_arm is None:
        args.submit_arm = ("refine_vote" if "rerefine" in kinds
                           else "refine_exp" if set(ARM_SESSIONS["refine_exp"]) <= set(kinds) else "first")
    if args.submit_arm not in ARMS or not set(ARM_SESSIONS[args.submit_arm]) <= set(kinds):
        raise SystemExit(f"--submit-arm {args.submit_arm} needs sessions {ARM_SESSIONS.get(args.submit_arm)} "
                         f"(running {kinds})")
    second = args.second_budget or cfg["output_token_budget"]
    refine_budget = args.refine_budget or cfg["output_token_budget"] // 2
    if args.refine_agree_budget is not None and not (args.refine_agree_prompts or args.refine_solo_prompts):
        raise SystemExit("--refine-agree-budget needs --refine-agree-prompts or --refine-solo-prompts")
    if args.refine_agree_prompts and "refine_exp" not in kinds:
        raise SystemExit("--refine-agree-prompts needs the refine_exp session")
    agree_budget = args.refine_agree_budget or refine_budget
    gate = args.ind_gate_tokens
    if (gate is None) != (args.refine_solo_prompts is None):
        raise SystemExit("--ind-gate-tokens and --refine-solo-prompts go together")
    if gate is not None and sorted(kinds) != ["first", "ind", "refine_exp"]:
        raise SystemExit(f"--ind-gate-tokens needs --arms ind,refine_exp (running {kinds})")
    for b in (second, refine_budget, agree_budget):
        if not cfg["min_resume_tokens"] < b:
            raise SystemExit(f"budget {b} must exceed min_resume_tokens {cfg['min_resume_tokens']}")
    if args.refine_wrapup is not None and wrapup is None:
        raise SystemExit("--refine-wrapup needs wrapup_tokens in the config")
    refine_wrapup = None if wrapup is None else (args.refine_wrapup or wrapup // 2)
    if refine_wrapup is not None and not cfg["min_resume_tokens"] < refine_wrapup:
        raise SystemExit(f"refiner wrap-up {refine_wrapup} must exceed min_resume_tokens {cfg['min_resume_tokens']}")
    if not 0 <= args.filler_min_left <= args.filler_guard:
        raise SystemExit("need 0 <= --filler-min-left <= --filler-guard")
    reuse_kinds = [k for k in args.reuse_kinds.split(",") if k] if args.reuse else []
    if not set(reuse_kinds) <= set(kinds):
        raise SystemExit(f"--reuse-kinds {reuse_kinds} must be among the sessions {kinds}")
    seq_prompts = load_prompt_set(args.seq_prompts, SEQ_PROMPTS) if "seq" in kinds else {}
    refine_names = REFINE_PROMPTS + (("wrapup.md",) if wrapup is not None else ())
    refine_prompts = (load_prompt_set(args.refine_prompts, refine_names)
                      if {"refine_seq", "refine_exp", "rerefine"} & set(kinds) else {})
    refine3_prompts = load_prompt_set(args.refine3_prompts, refine_names) if "refine_all" in kinds else {}
    agree_prompts = load_prompt_set(args.refine_agree_prompts, refine_names) if args.refine_agree_prompts else {}
    solo_prompts = load_prompt_set(args.refine_solo_prompts, refine_names) if args.refine_solo_prompts else {}
    verify_prompts = load_prompt_set(args.verify_prompts, VERIFY_PROMPTS) if "verify" in kinds else {}
    if args.tasks_dir is not None:
        tasks = base.load_tasks(args.tasks_dir, args.limit, args.only.split(",") if args.only else None)
        scored = True
    else:
        if args.only:
            raise SystemExit("--only goes with --tasks-dir")
        tasks, scored = base.load_challenges(args.challenges, args.solutions, args.limit)
    tids = list(tasks)
    # names (__s<k>) follow KINDS' numbering; uids use a compact index (Kaggle 240 tasks × 3 kinds = 720 uids, not 7 × 240)
    episodes = {(tid, k): base.Episode(len(kinds) * i + kinds.index(k), tid, SAMPLE[k])
                for i, tid in enumerate(tids) for k in kinds}
    ctx, info, version = base.prepare_run(args, cfg, list(episodes.values()), deny, tito, dp_size=args.dp_size,
                                          deadline=args.deadline_unix, scored=scored)
    run_dir = ctx.run_dir
    for name, texts in (("prompts_seq", seq_prompts), ("prompts_refine", refine_prompts),
                        ("prompts_refine3", refine3_prompts), ("prompts_verify", verify_prompts),
                        ("prompts_refine_agree", agree_prompts), ("prompts_refine_solo", solo_prompts)):
        if texts:
            (run_dir / "harness" / name).mkdir()
            for f, t in texts.items():
                (run_dir / "harness" / name / f).write_text(t)
    all_prompts = {**{f"{cfg['prompts']}/{n}": t for n, t in ctx.prompts.items()},
                   **{f"{args.seq_prompts}/{n}": t for n, t in seq_prompts.items()},
                   **{f"{args.refine_prompts}/{n}": t for n, t in refine_prompts.items()},
                   **{f"{args.refine3_prompts}/{n}": t for n, t in refine3_prompts.items()},
                   **{f"{args.verify_prompts}/{n}": t for n, t in verify_prompts.items()},
                   **{f"{args.refine_agree_prompts}/{n}": t for n, t in agree_prompts.items()},
                   **{f"{args.refine_solo_prompts}/{n}": t for n, t in solo_prompts.items()}}
    if gate is not None and "ind" in reuse_kinds:
        raise SystemExit("--ind-gate-tokens decides s2 from s0: reuse only first")
    reused = reuse_sessions(args.reuse, reuse_kinds, cfg, all_prompts, tids, episodes, run_dir) if args.reuse else {}
    state_file = {
        "config": cfg, "args": {k: str(v) for k, v in vars(args).items() if k not in ("samples", "salt_prefix")},
        "kinds": kinds, "budgets": {"first": cfg["output_token_budget"], "second": second, "refine": refine_budget,
                                    **({"refine_agree": agree_budget} if agree_prompts or solo_prompts else {}),
                                    **({"ind_gate_tokens": gate} if gate is not None else {}),
                                    **({"wrapup": {"solver": wrapup, "refine": refine_wrapup}} if wrapup else {})},
        "filler_gate_seconds": ({"guard": args.filler_guard, "min_left": args.filler_min_left}
                                if set(kinds) & set(FILLERS) else None),
        "codex_version": version,
        "prompt_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in all_prompts.items()},
        "server_info": {k: info.get(k) for k in ("context_length", "max_running_requests", "served_model_name",
                                                 "default_sampling_params", "preferred_sampling_params",
                                                 "sampling_defaults", "model_path", "version")},
        "tasks": tids, "started": time.strftime("%FT%T%z"), "scored": scored, "deadline_unix": args.deadline_unix,
        "dp_size": args.dp_size, "solvers_first": args.solvers_first, "submit_arm": args.submit_arm,
        **({"record": record} if record is not None else {}),
        **({"reuse": {"run": str(args.reuse.resolve()), "kinds": reuse_kinds}} if args.reuse else {}),
        **({"handoff_file": args.handoff_file} if args.handoff_file else {}),
        **({"review_file": args.review_file} if args.review_file else {}),
        **({"tito": tito, "pool_label": args.pool_label} if tito is not None else {}),
    }
    base.atomic_write(run_dir / "STATE.json", json.dumps(state_file, indent=1) + "\n")
    if scored:
        (run_dir / "results" / "solutions.json").write_text(
            json.dumps({tid: [t["output"] for t in task["test"]] for tid, task in tasks.items()}))
    deadline = args.deadline_unix
    base.log(f"{len(tids)} tasks x sessions {kinds}, concurrency {cfg['concurrency']}, budgets first "
             f"{cfg['output_token_budget']:,} second {second:,} refine {refine_budget:,}, fence {cfg['fence']}, "
             f"dp {args.dp_size}, scored {scored}, priority {'solvers first' if args.solvers_first else 'refiners first'}, "
             f"submit {args.submit_arm}, fillers {[k for k in kinds if k in FILLERS] or 'none'}"
             f"{f' (>= {args.filler_guard:.0f} s before the deadline while the main line runs, then >= {args.filler_min_left:.0f} s)' if set(kinds) & set(FILLERS) else ''}, "
             f"wrap-up {'none' if wrapup is None else f'+{wrapup:,} / refiners +{refine_wrapup:,}'}, deadline "
             f"{'none' if deadline is None else time.strftime('%FT%T%z', time.localtime(deadline))}")
    sub_paths = [run_dir / "submission.json", *([args.submission_out] if args.submission_out else [])]

    def on_signal(signum, frame):
        base.log(f"signal {signum}: cancelling; running sessions stop at their next poll")
        ctx.cancelled.set()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    lock = threading.Lock()
    results: dict[str, dict] = {tid: {} for tid in tids}          # tid -> kind -> result
    failed: dict[str, set] = {tid: set() for tid in tids}
    finals: dict[str, dict] = {k: {} for k in kinds if k in REFINERS}
    for (tid, k), r in reused.items():
        results[tid][k] = r
        if k in finals:
            finals[k][tid] = r["final"]
    to_run = [key for key in episodes if key not in reused]
    main_run = [key for key in to_run if key[1] not in FILLERS]
    failures: list[dict] = []
    state = {"done": [], "running": {}, "total": len(main_run), "not_started": [], "failed": [],
             "fillers": {"done": [], "not_started": [], "unusable": [], "failed": []}}
    priority = PRIORITY_SOLVERS_FIRST if args.solvers_first else PRIORITY

    def finals_of(tid: str) -> dict:
        return {a: finals[a][tid] for a in finals if tid in finals[a]}

    # --ind-gate-tokens: s2 follows s0, and s0 decides whether it starts
    needs = {**NEEDS, "ind": ("first",)} if gate is not None else NEEDS
    state["ind_gated"] = []

    def sure(r0: dict) -> bool:
        return (train_all(r0) and r0["stop"] == "finished" and all(g is not None for g in r0["attempt_1"])
                and r0["usage"]["output_tokens"] <= gate)

    sub_lock = threading.Lock()   # atomic_write's tmp name is fixed: two workers writing the same file would clobber the tmp

    def write_submissions() -> None:
        with sub_lock:
            with lock:
                att = {tid: submit_attempts(tasks[tid], dict(results[tid]), finals_of(tid), args.submit_arm)
                       for tid in tids}
            for path in sub_paths:
                write_submission(path, tasks, att)

    write_submissions()
    base.log(f"SUBMISSION placeholders for {len(tids)} tasks -> {[str(p) for p in sub_paths]}")
    def filler_gate(main_left: int) -> str:
        if ctx.cancelled.is_set() or len(state["fillers"]["failed"]) >= FILLER_MAX_FAILURES:
            return "closed"
        if deadline is None:
            return "open"
        left = deadline - time.time()
        if left < args.filler_min_left:
            return "closed"
        return "wait" if main_left > 0 and left < args.filler_guard else "open"

    sched = Scheduler(cfg["concurrency"], filler_gate)
    sched.add_pending(len(main_run))
    started = time.time()

    def ctx_for(kind: str, tid: str):
        task = tasks[tid]
        rs = results[tid]
        if kind in ("first", "ind"):
            c = ctx if kind == "first" or second == cfg["output_token_budget"] else \
                dataclasses.replace(ctx, cfg={**cfg, "output_token_budget": second})
            return c, ({"status_fn": handoff_status(args.handoff_file)} if args.handoff_file else {})
        if kind == "verify":
            r0 = rs["first"]
            prompts = {**ctx.prompts, **{n: verify_prompts[n] for n in VERIFY_PROMPTS if n != "brief.md"},
                       "brief.md": verify_prompts["brief.md"].replace("<<PREVIOUS>>", escape(train_status(r0)) + ".")}
            c = dataclasses.replace(ctx, prompts=prompts, cfg={**cfg, "output_token_budget": second})
            src = run_dir / "tasks" / episodes[(tid, "first")].name
            return c, {"extra_files": lambda ws: copy_solver(src, r0, ws / "previous"),
                       "status_fn": verify_status(r0, args.review_file)}
        if kind == "seq":
            r0 = rs["first"]
            situation = seq_prompts["previous_passed.md" if train_all(r0) else "previous_failed.md"]
            situation = situation.format(status=train_status(r0))
            prompts = {**ctx.prompts, **{n: seq_prompts[n] for n in ("kickoff.md", "nudge.md", "resume_fresh.md")},
                       "brief.md": seq_prompts["brief.md"].replace("<<PREVIOUS>>", escape(situation.strip()))}
            c = dataclasses.replace(ctx, prompts=prompts, cfg={**cfg, "output_token_budget": second})
            src = run_dir / "tasks" / episodes[(tid, "first")].name
            return c, {"extra_files": lambda ws: copy_solver(src, r0, ws / "previous"), "status_fn": seq_status(r0)}
        names = ("first", *REFINERS[kind])
        cands = [(chr(ord("A") + i), k) for i, k in enumerate(names)]
        # s0 and s2 both pass all train with the same outputs: --refine-agree-prompts / budget, attempt_1 pre-filled
        solo = kind == "refine_exp" and rs["ind"].get("gated", False)
        if solo:
            cands = cands[:1]
        agree = (kind == "refine_exp" and bool(agree_prompts) and not solo and not uncertain(tid)
                 and all(g is not None for g in rs["first"]["attempt_1"]))
        budget = agree_budget if agree or solo else refine_budget
        if solo:
            pset, text = solo_prompts, candidates_summary_n(task, [("A", rs["first"])])
        elif len(cands) == 2:
            pset, text = agree_prompts if agree else refine_prompts, candidates_summary(task, rs["first"], rs[names[1]])
        else:
            pset, text = refine3_prompts, candidates_summary_n(task, [(letter, rs[k]) for letter, k in cands])
        prompts = {**ctx.prompts, **{n: t for n, t in pset.items() if n != "brief.md"},
                   "brief.md": pset["brief.md"].replace("<<CANDIDATES>>", escape(text))}
        if cfg["midpoint_reminder_tokens"] is not None:
            reminder = min(cfg["midpoint_reminder_tokens"], budget // 2)
            c_cfg = {**cfg, "output_token_budget": budget, "midpoint_reminder_tokens": reminder}
        else:
            prompts.pop("midpoint_reminder.md")
            c_cfg = {**cfg, "output_token_budget": budget,
                     **({"wrapup_tokens": refine_wrapup} if refine_wrapup is not None else {})}
        c = dataclasses.replace(ctx, prompts=prompts, cfg=c_cfg)
        def extra(ws: Path) -> None:
            for letter, k in cands:
                copy_solver(run_dir / "tasks" / episodes[(tid, k)].name, rs[k], ws / "candidates" / letter)
            (ws / "final" / "attempt_1").mkdir(parents=True)
            (ws / "final" / "attempt_2").mkdir()
            if agree or solo:
                for j, grid in enumerate(rs["first"]["attempt_1"], 1):
                    (ws / "final" / "attempt_1" / f"test_{j}.json").write_text(json.dumps(grid))
        return c, {"extra_files": extra, "status_fn": refine_status(len(task["test"]))}

    def ready(tid: str, kind: str) -> bool:
        return all(n in results[tid] for n in needs[kind])

    def uncertain(tid: str) -> bool:
        """s0 and s2 do not both pass all train with identical outputs (offline: nearly all of the re-refiner's gain is on
        such tasks)."""
        r0, r2 = results[tid]["first"], results[tid]["ind"]
        return not (train_all(r0) and train_all(r2) and r0["attempt_1"] == r2["attempt_1"])

    def enqueue(tid: str, kind: str) -> None:
        if kind == "ind" and gate is not None and sure(results[tid]["first"]):
            # s2 not started: result = s0's outputs, 0 tokens (the ind arm degenerates to s0 alone); the refine_exp that
            # depends on it is scheduled as usual
            r0 = results[tid]["first"]
            ep = episodes[(tid, "ind")]
            g = {**r0, "task": ep.name, "sample": SAMPLE["ind"], "episode": ep.name, "kind": "ind", "gated": True,
                 "usage": {k: 0 for k in r0["usage"]}, "sessions": [], "seconds": 0.0,
                 "sessions_closed_by_harness": [], "requests_aborted_by_harness": [], "teardown_errors": []}
            base.atomic_write(run_dir / "results" / f"{ep.name}.json", json.dumps(g) + "\n")
            with lock:
                results[tid]["ind"] = g
                state["ind_gated"].append(ep.name)
                nxt = [k for k in kinds if "ind" in needs[k] and ready(tid, k)]
            base.log(f"{ep.name} [ind]: GATED (s0 passed every training pair, finished, "
                     f"{r0['usage']['output_tokens']:,} <= {gate:,} output tokens)")
            sched.done_one()
            for k in nxt:
                enqueue(tid, k)
            return
        if kind in FILLERS:
            sched.push_filler((0 if uncertain(tid) else 1, tids.index(tid)), (tid, kind))
        else:
            sched.push((priority[kind], tids.index(tid), SAMPLE[kind]), (tid, kind))

    def fail_dependents(tid: str, kind: str, why: str) -> None:
        for k in kinds:
            if k in FILLERS:
                continue                 # fillers are not in pending; it never enters the queue until its dependencies finish
            if kind in needs[k] and k not in failed[tid] and k not in results[tid]:
                failed[tid].add(k)
                failures.append({"task": episodes[(tid, k)].name, "tid": tid, "kind": k, "error": why})
                state["failed"].append(episodes[(tid, k)].name)
                sched.done_one()
                fail_dependents(tid, k, why)

    def skip(tid: str, kind: str) -> None:
        """deadline / cancellation: this episode is not started, nor are the ones depending on it (recorded as
        not_started, not as failed). Caller holds the lock."""
        failed[tid].add(kind)
        state["not_started"].append(episodes[(tid, kind)].name)
        for k in kinds:
            if k in FILLERS:
                continue
            if kind in needs[k] and k not in failed[tid] and k not in results[tid]:
                sched.done_one()
                skip(tid, k)

    def worker() -> None:
        while True:
            item = sched.pop()
            if item is None:
                return
            lane, (tid, kind) = item
            filler = lane == "filler"
            ep = episodes[(tid, kind)]
            dp = None
            try:
                cap = cfg["task_wall_seconds"]
                if deadline is not None:
                    cap = min(cap, deadline - time.time())
                if ctx.cancelled.is_set() or cap <= 0:
                    with lock:
                        if filler:
                            state["fillers"]["not_started"].append(ep.name)
                        else:
                            skip(tid, kind)
                    continue
                dp = pick_dp(ctx)
                with lock:
                    state["running"][ep.name] = (tid, kind)
                base.log(f"{ep.name} [{kind}]: START{' (filler)' if filler else ''} wall_cap {cap:.0f}s dp {dp}")
                c, hooks = ctx_for(kind, tid)
                result = base.solve(c, ep, tasks[tid], **hooks, dp=dp, wall_cap=cap)
                result = {**result, "tid": tid, "kind": kind, "episode": ep.name}
                if kind in REFINERS:
                    final, problems = read_final(run_dir / "tasks" / ep.name, len(tasks[tid]["test"]))
                    result["final"], result["final_problems"] = final, problems
                if kind == "verify":
                    result["verdict"] = read_verdict(run_dir / "tasks" / ep.name)
                    result["offer"] = verify_offer(results[tid]["first"], result, result["verdict"])
                base.atomic_write(run_dir / "results" / f"{ep.name}.json", json.dumps(result) + "\n")
                with lock:
                    results[tid][kind] = result
                    if filler:
                        state["fillers"]["done"].append(ep.name)
                    if filler and result["stop"] in UNUSABLE_FILLER_STOPS:
                        state["fillers"]["unusable"].append(ep.name)      # result file kept, but not used in the vote
                    elif kind in REFINERS:
                        finals[kind][tid] = result["final"]
                    state["done"].append(result)
                    nxt = [k for k in kinds if kind in needs[k] and ready(tid, k)]
                score = "" if result["score"] is None else f"score {result['score']:.2f} "
                base.log(f"{ep.name} [{kind}]: DONE {result['stop']} {score}train "
                         f"{sum(v == 'PASS' for v in result['train'])}/{len(result['train'])} "
                         f"answered {sum(a is not None for a in result['attempt_1'])}/{len(result['attempt_1'])} "
                         f"out {result['usage']['proxy_output_tokens']:,} sessions {len(result['sessions'])} "
                         f"{result['seconds']:.0f}s/{result['wall_cap_seconds']:.0f}s dp {dp} "
                         f"closed {len(result['sessions_closed_by_harness'])} "
                         f"[{sum(r['kind'] not in FILLERS for r in state['done'])}/{len(main_run)}"
                         f"{' filler' if filler else ''}]")
                write_submissions()
                for k in nxt:
                    enqueue(tid, k)
            except BaseException as error:  # noqa: BLE001 -- recorded; episodes depending on it are recorded as failed too
                row = {"task": ep.name, "tid": tid, "kind": kind, "error": repr(error),
                       "traceback": "".join(traceback.format_exception(error))}
                with lock:
                    failures.append(row)
                    failed[tid].add(kind)
                    state["failed"].append(ep.name)
                    if filler:
                        state["fillers"]["failed"].append(ep.name)
                    fail_dependents(tid, kind, f"dependency {kind} failed")
                with (run_dir / "failures.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                base.log(f"{ep.name} [{kind}]: FAILED {error!r}")
                if filler and len(state["fillers"]["failed"]) == FILLER_MAX_FAILURES:
                    base.log(f"FILLERS FAILED {FILLER_MAX_FAILURES} times {state['fillers']['failed']}: no more fillers "
                             f"(the main line goes on)")
                if sum(1 for r in failures if "traceback" in r and r["kind"] not in FILLERS) > cfg["max_failures"]:
                    base.log(f"FAILED more than max_failures {cfg['max_failures']} episodes: cancelling the run")
                    ctx.cancelled.set()
            finally:
                with lock:
                    state["running"].pop(ep.name, None)
                if dp is not None:
                    with base.LOCK:
                        ctx.dp_live[dp] -= 1
                if filler:
                    sched.filler_done()
                else:
                    sched.done_one()

    def watch_deadline() -> None:
        if not ctx.cancelled.wait(max(0.0, deadline - time.time())):
            ctx.deadline_hit.set()
            ctx.cancelled.set()
            with lock:
                running = len(state["running"])
            base.log(f"DEADLINE reached: no new episodes; stopping {running} running episode(s) and harvesting them")

    if deadline is not None:
        threading.Thread(target=watch_deadline, daemon=True, name="deadline").start()

    # Snapshot first: a gated s2 already scheduled refine_exp in enqueue, so the loop must not schedule it again
    for tid, k in [(tid, k) for tid in tids for k in kinds if (tid, k) not in reused and ready(tid, k)]:
        enqueue(tid, k)
    stop_hb = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_hb, ctx, state, lock), daemon=True).start()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(cfg["concurrency"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_hb.set()
    if ctx.watchdog is not None:
        ctx.watchdog.stop()
        base.atomic_write(run_dir / "watchdog_summary.json", json.dumps(ctx.watchdog.summary(), indent=1) + "\n")
    write_submissions()
    arms = [a for a in ARMS if set(ARM_SESSIONS[a]) <= set(kinds)]
    for arm in arms:
        att = {tid: arm_attempts(tasks[tid], results[tid], finals_of(tid))[arm] for tid in tids}
        write_submission(run_dir / f"submission_{arm}.json", tasks, att)
    submitted = {tid: submit_attempts(tasks[tid], results[tid], finals_of(tid), args.submit_arm) for tid in tids}
    done = state["done"]
    summary = {
        "tasks": len(tids), "test_pairs": sum(len(t["test"]) for t in tasks.values()), "kinds": kinds,
        "scored": scored, "submit_arm": args.submit_arm, "episodes": len(main_run),
        "episodes_done": {k: sum(r["kind"] == k for r in done) for k in kinds},
        "episodes_not_started": sorted(state["not_started"]),
        **({"ind_gated": len(state["ind_gated"])} if gate is not None else {}),
        "failed_episodes": sorted(f["task"] for f in failures),
        "deadline_hit": ctx.deadline_hit.is_set(),
        **({"fillers": {"done": len(state["fillers"]["done"]), "unusable": sorted(state["fillers"]["unusable"]),
                        "failed": sorted(state["fillers"]["failed"]),
                        "not_started": len(state["fillers"]["not_started"]), "dropped_at_the_gate": len(sched.dropped),
                        "never_queued": sum(1 for key in to_run if key[1] in FILLERS) - len(state["fillers"]["done"])
                        - len(state["fillers"]["failed"]) - len(state["fillers"]["not_started"]) - len(sched.dropped)}}
           if set(kinds) & set(FILLERS) else {}),
        "answered_test_pairs": {f"attempt_{k + 1}": sum(pair[k] is not None for att in submitted.values() for pair in att)
                                for k in (0, 1)},
        "tasks_with_refiner_final": {k: len(v) for k, v in finals.items()},
        "stops": {k: {s: sum(r["kind"] == k and r["stop"] == s for r in done) for s in sorted({r["stop"] for r in done})}
                  for k in kinds},
        "episodes_by_dp": {str(dp): sum(r["dp_rank"] == dp for r in done) for dp in sorted({r["dp_rank"] for r in done},
                                                                                           key=str)},
        "close_session_failures": len(ctx.close_failures),
        "sessions_closed_by_harness": sum(len(r["sessions_closed_by_harness"]) for r in done),
        "episodes_with_teardown_errors": sorted(r["episode"] for r in done if r["teardown_errors"]),
        "output_tokens_total": sum(p.output_tokens for p in ctx.proxies.values()),
        "wall_seconds": round(time.time() - started),
    }
    if scored:
        gts = {tid: [t["output"] for t in task["test"]] for tid, task in tasks.items()}
        summary["score_submission"] = round(100 * sum(
            sum(pair_correct(pair, gt) for pair, gt in zip(submitted[tid], gts[tid], strict=True)) / len(gts[tid])
            for tid in tids) / len(tids), 2)
        summary.update({k: v for k, v in summarize(tasks, results, finals, arms).items() if k not in summary})
    base.atomic_write(run_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    base.log("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "episodes_not_started")}))
    if tito is not None:
        # Cancellation (SIGTERM / SIGINT, max_failures) and the deadline all end up here: running episodes have already
        # been harvested into results (stop cancelled / deadline), and those never started are not in the manifest.
        # Read the ledger status again: a server restart midway is only recorded, not raised; unreadable raises.
        tito["server"] = base.tito_module("preflight").check_server_end(tito["server"])
        manifest = base.tito_module("manifest")
        written = manifest.write(
            run_dir / manifest.FILE, harness="arc2", config=tito["config"], server=tito["server"],
            items=tito_items(run_dir, state["done"], failures),
            meta={"runner": "run_arc2_seq", "pool": args.pool_label, "kinds": kinds,
                  "not_started": sorted(state["not_started"] + state["fillers"]["not_started"]),
                  **({"reused": {"run": str(args.reuse.resolve()), "kinds": reuse_kinds}} if args.reuse else {})})
        base.log(f"tito: {manifest.FILE} {written['counts']} instances {written['server']['instances']} "
                 f"pool {args.pool_label}")
    return summary, failures, run_dir


if __name__ == "__main__":
    main()
