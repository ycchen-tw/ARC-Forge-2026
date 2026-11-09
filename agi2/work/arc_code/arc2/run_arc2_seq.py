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
                  one killed by the deadline is not used. b200 eval120-r2, the same s0 / s2 with three refiners voting
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

``--handoff-file REPORT.md``: an s0 / s2 workspace without this file (or with under 200 bytes) does not count as done
(nudged); paired with the config's prompts v3doc (the brief asks for a handoff report). The later s1 / s6 / refiners can
read it from previous / candidates.
``--review-file REVIEW.md``: likewise for s6, which does not count as done without a review report; paired with
``--verify-prompts v3verify2``.

View text only, no tito / cache_salt. fence none is how b200b runs it (--driver-bin points to a bwrap-wrapped codex).

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
import os
import re
import shutil
import signal
import stat
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arc2 as base  # noqa: E402
import arc2_render as render  # noqa: E402

KINDS = ("first", "seq", "ind", "refine_seq", "refine_exp", "refine_all", "verify", "rerefine", "ind3", "ind4")
SAMPLE = {k: i for i, k in enumerate(KINDS)}                  # episode <task>__s<k>
#: Extra independent solvers (config extra_solvers, 10-29, the user: open s3 / s4 for hard tasks s0 / s2 cannot solve;
#: episode names are __s8 / __s9 because s3 / s4 are already the numbers of refine_seq / refine_exp). Not in --arms:
#: started only on live signals (see run()'s extra_*)
EXTRAS = ("ind3", "ind4")
#: G4 time-scaled simulation (branch arc2-harness-x34-g4sim, 2026-10-29): the Kaggle L4 C128 schedule replayed on one
#: Colab G4, whose per-stream output rate is K times the Kaggle one (x34: 3.0 from a calibration, 2.93 measured against
#: the Kaggle 12 h Save & Run). The G4 config (g4sim/scale_config.py) has every wall-clock second divided by K and
#: ``time_scale`` = K; the wall-clock constants below are the Kaggle values and the harness divides them by the config's
#: time_scale (absent = 1, real time). Before 10-30 K was a constant 3.0 here, so a Kaggle build of that branch would
#: have run with the G4 values.
EXTRA_KEYS = {"max_per_task", "trigger_tokens", "min_budget", "refine_reserve_seconds", "max_running", "poll_seconds",
              "wrapup_tokens"}
#: extra_solvers' optional ``"tail": true`` (10-30 improvement plan §2.4 b, c): when refine_exp finishes with the queue empty
#: (free slots in the run), the task still undecided (no two or more solvers passing all train with the same answer) and
#: extra solvers not used up, open one more; the re-refiner waits for it to finish and judges with it included
EXTRA_OPTIONAL = {"tail", "prompts", "informed", "agreement", "reserve_queued", "retry_hold_seconds", "retry_hold_on",
                  "undecided", "backlog", "estimator", "blocking_until_seconds"}
#: x37 RL version (10-31, x35+RL G4 sim and a schedule replay of codex):
#: ``"reserve_queued": true``: extra solvers queued but not yet started also count toward the backlog (their budget +
#: wrap-up at start); previously only running ones counted, so with many queued the backlog was underestimated; removed
#: from the reservation when started or SKIPped.
#: ``"retry_hold_seconds": N`` (wall clock, scaled by K on G4): when informed / agreement meet their conditions at the
#: moment s0 and s2 finish but are blocked by the budget or backlog, this task's refine_exp is not queued yet and the check
#: is repeated every poll; once granted it starts (refine_exp waits for it); if not granted within N seconds refine_exp is
#: released. Previously checked only once: x36+RL G4 at 90 G4 minutes into the run, 14 tasks met the informed condition,
#: 4 were blocked and never rechecked.
#: ``"agreement": {"min_output_tokens": M}``: when s0 and s2 both pass all train, their first answers agree, their combined
#: output is ≥ M, and this task has not opened an extra solver yet, open one independent extra solver before refine_exp
#: (it does not read others' reports, and counts as one vote in decided()). Of x35+RL's 6.81 lost points 1.88 are tasks
#: where both copies agree yet are both wrong, for which the current rules open no extra solver at all; the RL weights
#: talk briefly, so from about 3.5 h before D there are free slots (running drops from a full 128 → 56 on average → 31)
AGREEMENT_KEYS = {"min_output_tokens"}
#: x38 RL version (10-31, idle analysis of x35+RL / x36+RL G4: the RL weights use ~11M fewer tokens than base, and from
#: 200 min before D the average number running is only 75 → 4):
#: agreement's optional ``"max_budget": N``: budget cap for the agreement line (tokens; on RL tasks where both copies
#: agree, each solver's output is p50 35k, p90 110k, 90% ≤ 100k) -- it fills gaps, so both the reservation (backlog) and
#: its usage are held to this. ``"after_s2": true``: it is ordered after s2 and before refine_exp
#: (PRIORITY_AGREEMENT_AFTER_S2), not taking s2's slots (only once s2 is done is it known which tasks disagree and need
#: informed)
#: x39: agreement's optional ``"blocking": false``: refine_exp does not wait for this line (it starts as before once s0 and
#: s2 are done); the line is ordered after refine_exp (PRIORITY_AGREEMENT_NONBLOCKING, filler in nature), reserves nothing
#: while queued, is admitted on budget alone, and passes the full gate once right before starting; when done, if it
#: disagrees with s0 / s2 (or did not pass train) a re-refiner is queued (rerefine_skip_agreed does not skip it), if it
#: agrees that is the end. x38's after_s2 made refine_exp wait for it: at the same point in time x38 had started only 29
#: refine_exp while x37 had 41 done + 76 running, and queued extras count as pending, so the re-refiner fillers had to
#: wait for them.
#: ``"max_inflight": N``: at most N agreement extras queued plus running (the rest retry at the next poll), so they do not
#: eat all the re-refiners' slots.
AGREEMENT_OPTIONAL = {"max_budget", "after_s2", "blocking", "max_inflight"}
PRIORITY_AGREEMENT_AFTER_S2 = 1.5
PRIORITY_AGREEMENT_NONBLOCKING = 2.5
#: x39 s2 runway rule: every poll the extras monitor computes lead = "now + queued s2 count × L_s2 / R_full + one s2's
#: wall clock + refine_reserve ≤ deadline"; when true, ordinary / informed / undecided extras are ordered ahead of s2 as
#: designed since x34 (PRIORITY_EXTRA), when false they drop behind s2 (PRIORITY_EXTRA_BEHIND_S2): when the task pool is
#: harder than the public eval, s2 gets longer and must not be pushed later by extras. If it cannot be computed it counts
#: as false (Scheduler.lead)
PRIORITY_EXTRA_BEHIND_S2 = 1.5
#: x38: ``"undecided": true``: when an extra solver finishes (before refine_exp) and this task has a train-passing
#: candidate but not yet two independent solvers passing all train with the same answer (decided()), and extra solvers are
#: not used up, open the next independent one (the extra_solvers.prompts set); refine_exp waits for it.
#: Of x36+RL's 8.54 lost points 2.85 are tasks where s0 and s2 both pass train but answer differently, one extra solver was
#: already opened, and the candidates still lack the right answer; when the candidates contain the right answer the
#: refiner nearly always picks right (tasks of this kind with the right answer among the candidates: x35+RL 43.3 tasks,
#: 0.5 lost; x36+RL 37.5 tasks, 1.5 lost) -- what is missing is candidates, not selection
#: x38: ``"retry_hold_on": ["backlog"]``: which refusal reasons hold refine_exp for rechecking (omitted = x37's budget and
#: backlog). budget is the time left to the refiner cutoff × rate, which never recovers; holding refine_exp to recheck a
#: budget refusal only makes it wait in vain (a recheck that turns into this refusal also releases it)
RETRY_HOLD_REASONS = ("budget", "backlog")
#: x38: ``"backlog": {"s2": …, "refine": …, "running_solver": …, "running_refine": …}``: the four constants the backlog gate
#: uses to estimate the main line's remaining output (omitted = BACKLOG_S2, BACKLOG_REFINE, BACKLOG_RUNNING_SOLVER,
#: BACKLOG_RUNNING_REFINE, set for the base weights). Recomputed per time window on x35+RL / x36+RL G4: the defaults
#: overestimate RL's remaining main line 1.15× (D−560) → 1.45× (D−300) → 2.0× (D−200), so the gate refuses while slots
#: are free (x37+RL at D−430: backlog 37.8M vs room 36.5M); 70k / 40k / 100k / 50k give 1.00–1.07× over D−560..D−350
#: and stay ≥1 afterwards (base with the defaults: 0.87–1.27×). The floors (BACKLOG_RUNNING_*_MIN) are unchanged
BACKLOG_KEYS = {"s2", "refine", "running_solver", "running_refine"}
#: extra_solvers' optional ``"prompts"`` (10-31): which prompt set the extra solvers use, "first" (default: s0's
#: cfg["prompts"]) or "ind" (s2's ind_prompts; x36: v4ydual, which may submit solution2.py; the harness already harvests
#: extra solvers' second answers, only the prompt did not say they may write one)
#: extra_solvers' optional ``"informed"`` (10-31, x35-base G4 sim: of 17.33 lost points 11.67 are tasks where s0 and s2
#: both pass all train yet are both wrong, and such tasks never open an extra solver; in 17 of those 18 tasks s2 also had a
#: train-passing solution2, and the median s0 + s2 output is 278k versus 101k for correctly solved ones): when s0 and s2
#: are both done, both pass all train, s2's second answer passes train and differs from its first, and their combined
#: output tokens are ≥ min_output_tokens, open one extra solver that has read both reports: the brief of that prompt set
#: must contain <<PREVIOUS>>, and the workspace's previous/A, previous/B hold the two solver_reports (no code).
#: It does not count toward decided()'s consensus; the refiner's brief says it has read A's and B's reports. Optional
#: ``"disagree": true`` (x36 RL version, 10-31: the RL weights talk briefly and are 74% idle from 3 h before the deadline,
#: while of x35+RL's 6.67 lost points 3.82 come from the 53 tasks where s0 and s2 both pass train but answer
#: differently): when s0 and s2 both pass all train and their first answers differ, skip the solution2 and token
#: thresholds and open the informed line directly as a third opinion
INFORMED_KEYS = {"min_output_tokens", "prompts"}
INFORMED_OPTIONAL = {"disagree"}
#: Filler kinds: not in the main-line queue (the Scheduler's filler queue), not counted in pending; failing or never
#: starting does not drag anything else along
FILLERS = ("rerefine",)
#: Filler episodes stopped for these reasons are not used (final/ may be just the first version written early on)
UNUSABLE_FILLER_STOPS = ("deadline", "cancelled", "wall")
#: Filler failures (traceback) do not count toward the config's max_failures (an optional episode must not cancel the
#: whole run); filler is switched off once this many occur
FILLER_MAX_FAILURES = 3
#: refiner → its candidates (A is always first, then B, C in order)
REFINERS = {"refine_seq": ("seq",), "refine_exp": ("ind",), "refine_all": ("seq", "ind"), "rerefine": ("ind",)}
NEEDS = {"first": (), "seq": ("first",), "ind": (), "verify": ("first",), **{k: ("first", *v) for k, v in REFINERS.items()},
         "rerefine": ("first", "ind", "refine_exp"), "ind3": (), "ind4": ()}   # rerefine votes with refine_exp: waits for it
#: extra_solvers: the priority used when s2 is promoted (= s0's tier); extra solvers go ahead of s2 (hard tasks need
#: runway: their budget shrinks with the time left from their start to the refiner cutoff, so the later they start the
#: shorter it is; s2 of an easy task is short and finishes even if it starts a bit later)
PRIORITY_PROMOTED = 0
PRIORITY_EXTRA = 0.5
#: extra_solvers' backlog gate (10-29 reviewer's scheduling simulation: b200b v3y sessions, Kaggle C128 rate model, 240
#: tasks): before opening an extra solver, estimate the main line's remaining output -- 75k for each unstarted s2, 50k for
#: each unfinished refine_exp, at least 20k for each running s0 / s2 (120k − used), at least 10k for each running refiner
#: (60k − used), budget + wrap-up − used for each running extra solver -- which, plus this line's budget, must be
#: ≤ 0.9 × per-line rate × workers × (deadline − 1800 s − now). Without the gate, throughput 0.8× nets −0.6pp (refiners
#: squeezed past the deadline); with it, 1.0× +0.73, 0.8× / 0.65× +0.06 to +0.35.
BACKLOG_S2, BACKLOG_REFINE = 75000, 50000
BACKLOG_RUNNING_SOLVER, BACKLOG_RUNNING_SOLVER_MIN = 120000, 20000
BACKLOG_RUNNING_REFINE, BACKLOG_RUNNING_REFINE_MIN = 60000, 10000
BACKLOG_TAIL_SECONDS, BACKLOG_FILL = 1800, 0.9
#: x39 estimator (config extra_solvers.estimator {"priors": <JSON in the harness directory>, "shrink_n0": N,
#: "m_clip": [lo, hi]}, mutually exclusive with the backlog constants): the backlog's "unstarted s2" and "unstarted
#: refine_exp" use the prior table's mean (L), running s0 / s2 / refine_exp use the prior table's conditional remainder
#: E[T − u | T > u] (cond), all multiplied by this run's difficulty factor m(t): the Kaplan–Meier restricted mean of this
#: run's s0, s2 output tokens (done = fully observed, running = right-censored at used), up to the largest complete value
#: so far τ, divided by the prior's restricted mean over the same [0, τ] gives r; shrink m = (n0 + n·r)/(n0 + n) (n =
#: number finished), clip. The four constants of x35–x38 were hand-tuned per weights (calibrated on base they overestimate
#: RL by 1.2–2.0×) and would also underestimate when the task pool is harder than the public eval; m lets it track by
#: itself. room now uses the full-load total throughput R_full (EMA over polls with ≥ RATE_FULL_LOAD × C running): per-line
#: rate × C overestimates by 1.2–1.6× under partial load (x37: 35 tok/s per line at 128 lines, 52 at 48, 78 at 16); until
#: it is measured, RATE_FULL_FALLBACK × per-line rate × C is used. Admission also adds this line's wrap-up (consistent
#: with the reservation)
ESTIMATOR_KEYS = {"priors", "shrink_n0", "m_clip"}
ESTIMATOR_MIN_FINISHED = 20
PRIORS_KINDS = ("first", "ind", "refine_exp")
RATE_FULL_LOAD, RATE_FULL_ALPHA, RATE_FULL_FALLBACK = 0.9, 0.3, 0.8
#: live_pass reads at most this many bytes of .best/meta.json (what check.py writes is a few dozen bytes)
META_MAX_BYTES = 4096
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
#: Used only if the refiner's prompt set has this file (since v4refine, improvement plan §1.6): codex uses it instead of
#: s0's when compacting the refiner's context (s0's template asks about solution.py and answer/, while the refiner submits
#: final/; x34's refiner summaries after compaction lacked the state of the candidates and final/)
REFINE_COMPACT = "compact_prompt.md"
VERIFY_PROMPTS = ("brief.md", "kickoff.md", "nudge.md", "resume_fresh.md")
VERDICTS = ("CONFIRMED", "ALTERNATIVE")
HANDOFF_MIN_BYTES = 200
#: <<NAME>> placeholders in the prompts the harness substitutes (<<CANDIDATES>>, <<FINAL_FILES>>…): none may remain when
#: the text goes to codex
PLACEHOLDER = re.compile(r"<<[A-Z_]+>>")
#: Refiner prompts whose brief has <<SOLVERS>> (since v4refine, improvement plan §2.2) take any number of candidates (2–4):
#: tasks with extra solvers hand all of them over (A, B… in rank_candidates order), summarized with candidates_summary_all;
#: sets without it stay at two (pick_two when there are extra solvers)
SOLVERS_MARK = "<<SOLVERS>>"
#: brief of the informed extra solver: summary and report locations of the two solvers that finished first
#: (informed_previous)
PREVIOUS_MARK = "<<PREVIOUS>>"
#: <<BUDGET_END>>: what happens once the budget is used up, per config (whether there is a wrap-up message, and whether
#: one comes when time is nearly up)
BUDGET_END = {(True, False): "When it is used up you get one message asking you to wrap up, with a small extra allowance "
                             "for that; after the allowance you are stopped mid-step and `final/` is taken as it is.",
              (True, True): "When it is used up, or when your time is nearly up, you get one message asking you to wrap "
                            "up, with a small allowance for that; after it you are stopped mid-step and `final/` is "
                            "taken as it is.",
              (False, False): "When it runs out you are stopped mid-step without warning, and `final/` is taken as it is."}
#: the session line in check.txt (improvement plan §2.3 item 6: the refiner needs to know whether a candidate finished on
#: its own or was cut off by budget or time)
STOP_TEXT = {"finished": "finished on its own", "wrapped_up": "used up its budget and was asked to wrap up",
             "budget": "was stopped at its budget limit, possibly mid-step", "budget_low": "used up its budget",
             "max_sessions": "was stopped after its last allowed session", "wall": "was stopped by its time limit",
             "deadline": "was stopped by the run's deadline", "cancelled": "was cancelled",
             "context_overflow": "ran into the context limit too often"}
NUMBERS = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}
#: Files copied into previous / candidates: these at the workspace top level (check.py and arc.py, placed by the harness
#: itself, are not copied; likewise the workspace's arc.py)
COPY_SUFFIXES = (".py", ".md", ".txt", ".json")
COPY_SKIP = {"check.py", "arc.py", "TOOLS.md", "AGENTS.md", "budget.txt"}
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


#: the prompt files run_arc2.prepare_run hands to codex (knobs.driver_files): when s2 switches prompt set, these two
#: switch too
IND_DRIVER_FILES = {"model_instructions_file": "instructions.md", "experimental_compact_prompt_file": "compact_prompt.md"}


def compact_context(ctx, path: Path):
    """Point codex's compact prompt file at path (experimental_compact_prompt_file in knobs.driver_files). The directory
    and file of path must be readable by codex under every task uid in fence posix."""
    for f in (path.parent, path):
        f.chmod(0o755 if f.is_dir() else 0o644)
    files = dict(ctx.knobs.driver_files)
    key = "experimental_compact_prompt_file"
    if key not in files:
        raise RuntimeError(f"knobs.driver_files {sorted(files)} lacks {key}")
    files[key] = str(path)
    return dataclasses.replace(ctx, knobs=dataclasses.replace(ctx.knobs, driver_files=tuple(sorted(files.items()))))


def extension_decision(ext: dict, total: int, began: float, now: float, cap: float, deadline: float | None) -> dict:
    """budget_extension's time gate (10-25): at the moment output passes the budget (total), estimate from this episode's
    average output rate so far how long the extra extra_tokens take (need). Granted only if the remaining wall clock (cap,
    counted from began) fits need + margin_seconds and, before the deadline, there is still room for need + the refiner
    (refine_reserve_seconds) + margin_seconds. Without a deadline (b200) only the wall clock counts."""
    rate = total / max(now - began, 1.0)
    need = ext["extra_tokens"] / rate
    wall_left = cap - (now - began)
    out = {"rate": round(rate, 2), "need_s": round(need), "wall_left_s": round(wall_left)}
    if need + ext["margin_seconds"] > wall_left:
        return {**out, "granted": False, "reason": "wall"}
    if deadline is None:
        return {**out, "granted": True, "reason": "no deadline"}
    left = deadline - now
    ok = left >= need + ext["refine_reserve_seconds"] + ext["margin_seconds"]
    return {**out, "left_s": round(left), "granted": ok, "reason": "time" if ok else "deadline"}


def ind_context(ctx, cfg: dict, run_dir: Path):
    """s2 uses the config's ind_prompts (10-24: v3dual). The file set is exactly the one loaded for s0 (brief, wrap-up,
    resume_fresh…), written to harness/prompts_ind/; codex's developer_instructions and instructions / compact prompt
    files are switched along with it."""
    return named_context(ctx, cfg["ind_prompts"], run_dir, "prompts_ind")


def named_context(ctx, set_name: str, run_dir: Path, dirname: str):
    """ctx switched to another prompt set (prompts/<set_name>/, exactly the file set ctx loaded), written to
    harness/<dirname>/; codex's developer_instructions and instructions / compact prompt files are switched along with it.
    Used by s2 (ind_context) and the informed extra solver."""
    pdir = HERE / "prompts" / set_name
    out_dir = run_dir / "harness" / dirname
    out_dir.mkdir()
    prompts = {}
    for name in ctx.prompts:
        text = (pdir / name).read_text()
        if not text.strip():
            raise SystemExit(f"{pdir / name} is empty")
        prompts[name] = text
        (out_dir / name).write_text(text)
    for p in (out_dir, *out_dir.iterdir()):
        p.chmod(0o755 if p.is_dir() else 0o644)      # fence posix: codex under each task uid must be able to read it
    files = dict(ctx.knobs.driver_files)
    if not set(IND_DRIVER_FILES) <= set(files):
        raise RuntimeError(f"knobs.driver_files {sorted(files)} lacks {sorted(IND_DRIVER_FILES)}")
    files.update({key: str(out_dir / name) for key, name in IND_DRIVER_FILES.items()})
    config = dict(ctx.knobs.driver_config)
    if "developer_instructions" not in config:
        raise RuntimeError("knobs.driver_config has no developer_instructions")
    config["developer_instructions"] = json.dumps(prompts["developer.md"])
    knobs = dataclasses.replace(ctx.knobs, driver_files=tuple(sorted(files.items())),
                                driver_config=tuple(sorted(config.items())))
    return dataclasses.replace(ctx, prompts=prompts, knobs=knobs)


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


def refiner_agrees(r0: dict, r2: dict, rx: dict, n_test: int) -> bool:
    """config rerefine_skip_agreed: s0 and s2 both pass all train, and s0, s2 and the final attempt_1 refine_exp wrote
    itself are identical on every test (an a1 refine_exp did not write does not count as agreement: fill took it from the
    candidates, the refiner took no position)."""
    final = rx.get("final") or []
    return (train_all(r0) and train_all(r2) and len(final) == n_test
            and all(r0["attempt_1"][j] is not None and r0["attempt_1"][j] == r2["attempt_1"][j] == final[j][0]
                    for j in range(n_test)))


def second_outputs(r: dict) -> list:
    """A solver's second answer (config second_solution, solution2.py of prompts v3dual), one entry per test: counts only
    if it passes all train, loads, and differs from this session's first answer, else None. second_solution off or the
    file not written = all None."""
    rec = r.get("second")
    n = len(r["attempt_1"])
    if rec is None or rec["load_error"] or not rec["train"] or any(v != "PASS" for v in rec["train"]):
        return [None] * n
    return [g if g is not None and g != a else None for g, a in zip(rec["outputs"], r["attempt_1"], strict=True)]


def candidates_all(results: list[dict], j: int) -> list:
    """After rank_all's order (each session's first answer), append their second answers in session order (duplicates
    removed). A second answer enters the attempts only once the first answers are used up: when two solvers give
    different answers the attempts are unchanged."""
    out = rank_all(results, j)
    for r in results:
        g = second_outputs(r)[j]
        if g is not None and g not in out:
            out.append(g)
    return out


def second_answer_stats(tids: list[str], results: dict[str, dict], submitted: dict[str, list],
                        kinds: tuple[str, ...] = ("first", "ind")) -> dict:
    """Per solver kind (kinds: s0, s2 and any extra solvers opened): the number of episodes that wrote a second answer,
    that passed all train, and of tests where it provided a different output; plus the number of tests in the submission
    that use a second answer (some submitted attempt equals some solver's second answer and no solver's first answer)."""
    out = {}
    for kind in kinds:
        rs = [results[t][kind] for t in tids if kind in results[t]]
        recs = [r["second"] for r in rs if r.get("second") is not None]
        out[kind] = {"episodes": len(rs), "wrote_file": len(recs),
                     "passes_train": sum(not x["load_error"] and bool(x["train"]) and all(v == "PASS" for v in x["train"])
                                         for x in recs),
                     "tests_offered": sum(g is not None for r in rs for g in second_outputs(r))}
    used = 0
    for t in tids:
        rs = [results[t][k] for k in kinds if k in results[t]]
        for j, pair in enumerate(submitted[t]):
            firsts = [r["attempt_1"][j] for r in rs]
            seconds = [second_outputs(r)[j] for r in rs]
            used += any(g is not None and g in seconds and g not in firsts for g in pair)
    out["submitted_tests_using_a_second_answer"] = used
    return out


def copy_solver(src_ws: Path, r: dict, dst: Path) -> None:
    """Copy one session's results to dst (read-only): the solution it was harvested with (the .best one when the snapshot
    fallback applied), notes, top-level helpers, answer/, the check.txt the harness writes and each test's final output;
    with a second answer (second_outputs) test_J_output2 is written too."""
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
    rec = r.get("second")
    if rec is not None:
        if rec["load_error"]:
            lines.append(f"{rec['file']} (second answer): cannot be loaded ({rec['load_error']}); not used")
        else:
            passed = sum(v == "PASS" for v in rec["train"])
            lines.append(f"{rec['file']} (second answer): {passed}/{len(rec['train'])} training pairs pass"
                         + ("" if passed == len(rec["train"]) else "; not used"))
        for j, grid in enumerate(second_outputs(r), 1):
            if grid is not None:
                lines.append(f"test {j}: second answer {len(grid)}x{len(grid[0])}, different from the first "
                             f"(test_{j}_output2.json)")
                (dst / f"test_{j}_output2.json").write_text(json.dumps(grid))
                (dst / f"test_{j}_output2.txt").write_text(render.grid_text(grid))
    lines.append(session_line(r))
    (dst / "check.txt").write_text("\n".join(lines) + "\n")
    # Read-only files are only a hint (the agent is the same uid and can change them back); directories stay writable so
    # a later rm -r can delete them. The harness never reads anything back from here.
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


def copy_report(src_ws: Path, dst: Path, report_file: str) -> bool:
    """previous/<letter>/ of the informed extra solver: only that solver's report (no code, so it does not get copied).
    Without a report (or one too large to be a report) only the directory is created and False returned; the brief says
    it left no report."""
    dst.mkdir(parents=True)
    f = src_ws / report_file
    if not f.is_file() or f.stat().st_size > COPY_MAX_BYTES:
        return False
    shutil.copyfile(f, dst / report_file)
    (dst / report_file).chmod(0o444)
    return True


def session_line(r: dict) -> str:
    """The last line of check.txt: how this solver's session ended, how much output it used, how many compactions (the
    session_id the proxy sees is <thread>:<window>, window + 1 per compaction), how many times it hit the context edge and
    opened a new thread."""
    seen = r["sessions_seen"]
    bad = [x for x in seen if not re.fullmatch(r"[^:]+:\d+", x)]
    if bad:
        raise RuntimeError(f"session ids without a :<window> suffix: {bad}")
    compactions = len(seen) - len({x.rsplit(":", 1)[0] for x in seen})
    overflows = sum(bool(row["context_overflow"]) for row in r["sessions"])
    stop_text = STOP_TEXT[r["stop"]]
    if r["stop"] == "wrapped_up" and (r.get("wrapup") or {}).get("armed_by") == "time":
        stop_text = "ran out of time and was asked to wrap up"       # x39: a time wrap-up is not an exhausted budget (x35 to-do)
    text = (f"session: {stop_text}; {r['usage']['proxy_output_tokens'] // 1000}k output tokens; "
            f"context compacted {compactions} time{'' if compactions == 1 else 's'}")
    if overflows:
        text += f"; ran into the context limit {overflows} time{'' if overflows == 1 else 's'} and restarted from its files"
    return text


def join_names(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def candidates_intro(letters: list[str], informed: list[str] = (), read: list[str] | None = None) -> str:
    """<<SOLVERS>> of the refiner brief: how many solvers and where their directories are; the informed ones (extra
    solvers that read others' reports) are not independent of the others, and that must be said. ``read``: whose reports
    the informed one read (letters of s0, s2); omitted = all the others (wrong since x38 when there is a fourth
    candidate: the undecided line opens after informed, which never read it; refiner calls always pass it explicitly)."""
    dirs = [f"`candidates/{x}/`" for x in letters]
    where = ", ".join(dirs[:-1]) + " and " + dirs[-1]
    text = f"{NUMBERS[len(letters)]} solvers have already worked on this puzzle, independently of each other"
    if informed:
        others = [x for x in letters if x not in informed] if read is None else list(read)
        if not others or set(others) & set(informed) or not set(others) <= set(letters):
            raise RuntimeError(f"candidates_intro: informed {informed} read {others} among {letters}")
        text += (f", except {join_names(list(informed))}, which read {join_names(others)}'s report"
                 f"{'s' if len(others) > 1 else ''} (not the code) before starting")
    return text + f". Their work is in {where} (read-only)."


def informed_previous(task: dict, cands: list[tuple[str, dict]], report_file: str, has_report: dict[str, bool]) -> str:
    """<<PREVIOUS>> of the informed extra solver's brief: the two solvers that finished first (A, B, per rank_candidates)
    -- how many train pairs each passes, whether each test's outputs agree, who also has a second answer (the
    candidates_summary_all format), where in previous/ the reports are, and a note for any that wrote no report."""
    letters = [letter for letter, _ in cands]
    if len(letters) != 2:
        raise RuntimeError(f"informed_previous takes the two main solvers, got {letters}")
    paths = [f"`previous/{x}/{report_file}`" for x in letters if has_report[x]]
    missing = [x for x in letters if not has_report[x]]
    lines = [f"Two solvers, {letters[0]} and {letters[1]}, have already worked on this puzzle, independently of each other "
             f"and of you. In short:", "", candidates_summary_all(task, cands), ""]
    if paths:
        lines.append(f"Their reports -- the rule each found, the evidence, the readings each rejected -- are in "
                     f"{join_names(paths)} (read-only). Their code is not here on purpose.")
    if missing:
        lines.append(f"{join_names(missing)} left no report, so only the summary above is known about "
                     f"{'that solver' if len(missing) == 1 else 'them'}.")
    return "\n".join(lines)


def candidates_summary_all(task: dict, cands: list[tuple[str, dict]]) -> str:
    """<<CANDIDATES>> of the refiner brief (any number of candidates): how many train pairs each passes; per test, which
    outputs are identical, which differ from all others, which are missing, and who also has a second answer (identical to
    which first answer, or different from all)."""
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
        join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]  # noqa: E731
        shared = [names for names in groups.values() if len(names) > 1]
        alone = [names[0] for names in groups.values() if len(names) == 1]
        parts = [f"{join(names)} give the same output" for names in shared]
        if alone:
            if len(alone) == 1:
                parts.append(f"{alone[0]} gives an output of its own")
            elif shared:
                parts.append(f"{join(alone)} each give an output of their own")
            else:
                parts.append(f"{join(alone)} give {'different outputs' if len(alone) == 2 else f'{len(alone)} different outputs'}")
        if none:
            parts.append(f"{join(none)} give{'s' if len(none) == 1 else ''} no output")
        for name, r in cands:
            g = second_outputs(r)[j]
            if g is not None:
                owners = [n for key, names in groups.items() for n in names if json.loads(key) == g]
                parts.append(f"{name} also gives a second answer"
                             + (f" (the same as {join(owners)}'s output)" if owners else " (different from every first answer)"))
        lines.append(f"- test {j + 1}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


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
        firsts = [g for g in (a, b) if g is not None]
        for name, r in (("A", ra), ("B", rb)):
            g = second_outputs(r)[j]
            if g is not None:
                what += (f"; {name} also gives a second answer"
                         + (" (the same as the other candidate's output)" if g in firsts else " (different from both)"))
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


def final_names(n_test: int) -> list[str]:
    """The refiner deliverables the harness reads, ordered by test, then attempt."""
    return [f"final/attempt_{k}/test_{j}.json" for j in range(1, n_test + 1) for k in (1, 2)]


def final_files_text(n_test: int) -> str:
    """<<FINAL_FILES>> of the refiner prompts: every file name listed out (x34 b9e38dc0: the brief said `test_J.json`
    (J = 1..n), the refiner literally wrote final/attempt_1/test_J.json, which the harness cannot read)."""
    names = [f"`{n}`" for n in final_names(n_test)]
    return ", ".join(names[:-1]) + " and " + names[-1]


def final_strays(ws: Path, n_test: int) -> list[str]:
    """Files in final/ the harness does not read (wrong name, wrong level)."""
    expected = set(final_names(n_test))
    root = ws / "final"
    if not root.is_dir():
        return []
    return sorted(str(f.relative_to(ws)) for f in root.rglob("*") if f.is_file() and str(f.relative_to(ws)) not in expected)


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
        strays = final_strays(ws, n_test)
        if strays and missing:
            parts.append(f"final/ also has {', '.join(strays)}, which the harness does not read -- it reads only "
                         + ", ".join(final_names(n_test)))
        return False, "; ".join(parts)
    return status


# --------------------------------------------------------------------------- attempts per arm

def rank_candidates(rs: dict[str, dict], pool: list[str]) -> list[str]:
    """Candidate order (A, B…): all-train-passing first, then more pairs passed, then more tests answered, then earlier in
    pool."""
    def key(k: str):
        r = rs[k]
        passed = sum(v == "PASS" for v in r["train"]) if not r["load_error"] else -1
        return (not train_all(r), -passed, -sum(a is not None for a in r["attempt_1"]), pool.index(k))
    return sorted(pool, key=key)


def pick_two(rs: dict[str, dict], pool: list[str]) -> tuple[str, str]:
    """extra_solvers, refiner prompts with two candidates: A / B are picked from pool (s0, s2, finished extra solvers):
    A = rank_candidates' first; B = the first after A whose first answer differs from A's, else the second."""
    order = rank_candidates(rs, pool)
    a = order[0]
    b = next((k for k in order[1:] if rs[k]["attempt_1"] != rs[a]["attempt_1"]), order[1])
    return a, b


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
        # second answers (second_outputs) fill in only when the two solvers' first answers give fewer than two entries
        out["ind"] = [(candidates_all([r0, rs["ind"]], j) + [None, None])[:2] for j in range(n)]
    if "seq" in rs and "ind" in rs:
        three = [r0, rs["seq"], rs["ind"]]
        out["rank3"] = [(rank_two(three, j) + [None, None])[:2] for j in range(n)]
    extras = [rs[k] for k in EXTRAS if k in rs]
    for arm, raw in RAW.items():
        if arm in finals and out[raw] is not None:
            # refine_all's missing ones come from the full order of the three candidates (not just the top two);
            # refine_exp's also append the extra solvers (= submit_attempts)
            backup = ([rank_all([r0, rs["seq"], rs["ind"]], j) for j in range(n)] if arm == "refine_all"
                      else [candidates_all([r0, rs["ind"], *extras], j) for j in range(n)] if raw == "ind" and extras
                      else out[raw])
            out[arm] = [fill(finals[arm][j], backup[j]) for j in range(n)]
    # refine_vote: the re-refiner (filler, present for only some tasks) votes with refine_exp; tasks without a re-refiner =
    # refine_exp
    if out["ind"] is not None:
        second = ([fill(finals["rerefine"][j], candidates_all([r0, rs["ind"], *extras], j)) for j in range(n)]
                  if "rerefine" in finals else None)
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
        toks = [r["usage"]["proxy_output_tokens"] for rs in by_task.values() for k, r in rs.items() if k == kind]
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
    used, however many are present, in candidates_all order (first answers first, second answers after); if arm is a
    refiner and it has a final, fill (= that arm of arm_attempts). With all sessions present it equals arm_attempts.
    refine_vote: vote between refine_exp's and the re-refiner's (filled); if only one of them exists, use it."""
    if arm == "refine_vote":
        first = submit_attempts(task, rs, finals_tid, "refine_exp")
        if "rerefine" not in finals_tid:
            return first
        solvers = [rs[k] for k in ("first", "ind", *EXTRAS) if k in rs]
        second = [fill(finals_tid["rerefine"][j], candidates_all(solvers, j)) for j in range(len(task["test"]))]
        if "refine_exp" not in finals_tid:
            return second
        return [vote2(a, b) for a, b in zip(first, second, strict=True)]
    solvers = [rs[k] for k in ARM_SESSIONS[arm] if k in rs and k not in REFINERS]
    if "ind" in ARM_SESSIONS[arm]:
        solvers += [rs[k] for k in EXTRAS if k in rs]      # extra solvers' answers after s0 / s2 (candidates_all order)
    out = []
    for j in range(len(task["test"])):
        backup = candidates_all(solvers, j)
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

    def __init__(self, workers: int, filler_gate=None, extra_cap: int = 0) -> None:
        self.workers = workers
        #: extra_solvers: extra solvers have their own queue, at most extra_cap running at once; ranked by priority against
        #: the main line, counted in pending
        self.extras: list = []
        self.extra_cap, self.extra_running = extra_cap, 0
        #: x39 s2 runway rule: when false, extra solvers at PRIORITY_EXTRA count as PRIORITY_EXTRA_BEHIND_S2 (ordered after
        #: s2); the extras monitor updates it every poll via set_lead, False when it cannot be computed
        self.lead = True
        #: (tid, kind) already dispatched: a promoted s2 leaves an old copy in the heap, dropped when popped
        self.popped: set = set()
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

    def queued(self) -> int:
        """Number of main-line and extra solver episodes queued but not yet dispatched (old entries left by a promoted s2
        not counted)."""
        with self.cv:
            return sum(1 for e in self.heap if e[2] not in self.popped) + len(self.extras)

    def push_extra(self, prio: tuple, item) -> None:
        with self.cv:
            heapq.heappush(self.extras, (prio, self.counter, item))
            self.counter += 1
            self.cv.notify()

    def extra_done(self) -> None:
        with self.cv:
            self.extra_running -= 1
            self.cv.notify_all()

    def set_lead(self, lead: bool) -> None:
        with self.cv:
            self.lead = bool(lead)
            self.cv.notify_all()

    def extra_priority(self, prio: float) -> float:
        """An extra solver's effective priority in its lane (lower first): PRIORITY_EXTRA drops behind s2 when lead is
        false; others as is."""
        return PRIORITY_EXTRA_BEHIND_S2 if prio == PRIORITY_EXTRA and not self.lead else prio

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
                while self.heap and self.heap[0][2] in self.popped:
                    heapq.heappop(self.heap)                      # old entry left by a promoted s2
                if self.extras and self.extra_running < self.extra_cap and (
                        not self.heap or self.extra_priority(self.extras[0][0][0]) <= self.heap[0][0][0]):
                    item = heapq.heappop(self.extras)[2]
                    if item in self.popped:
                        raise RuntimeError(f"extra solver {item} queued twice")
                    self.popped.add(item)
                    self.extra_running += 1
                    return "extra", item
                if self.heap:
                    item = heapq.heappop(self.heap)[2]
                    self.popped.add(item)
                    return "main", item
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
    if state["config"] != cfg:
        raise SystemExit(f"--reuse {old}: its config differs from --config")
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


def pick_dp(ctx) -> int | None:
    """The DP with the fewest running episodes right now (ties go to the smaller rank); one fixed DP per episode."""
    if ctx.dp_live is None:
        return None
    with base.LOCK:
        dp = min(range(len(ctx.dp_live)), key=lambda rank: (ctx.dp_live[rank], rank))
        ctx.dp_live[dp] += 1
    return dp


RATE_WINDOW_SECONDS = 600
#: extra_solvers' per-stream rate: counted only once the samples span at least this long
RATE_MIN_SECONDS = 300


def time_scale(cfg: dict) -> float:
    """The config's time_scale (K of the G4 simulation); key absent = real time."""
    return float(cfg.get("time_scale", 1))


def load_priors(path: Path) -> dict:
    """Prior table of the x39 estimator (work/arc_code/arc2/priors/*.json, computed by
    probes/x39_estimator_20261101/build_priors.py from results/*.json of several G4 / Kaggle 12 h runs). A wrong schema is
    a SystemExit: stopped before the run starts."""
    try:
        p = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"estimator priors {path}: {error!r}")
    need = {"weights", "source", "bucket", "kinds", "L", "cond"}
    if not isinstance(p, dict) or not need <= set(p):
        raise SystemExit(f"estimator priors {path}: needs the keys {sorted(need)}")
    for kind in PRIORS_KINDS:
        k = p["kinds"].get(kind)
        if not (isinstance(k, dict) and isinstance(k.get("totals"), list) and len(k["totals"]) >= 50
                and all(type(v) is int and v >= 0 for v in k["totals"]) and k["totals"] == sorted(k["totals"])):
            raise SystemExit(f"estimator priors {path}: kinds.{kind}.totals must be >= 50 sorted non-negative ints")
    for key in ("s2", "refine"):
        if not (type(p["L"].get(key)) in (int, float) and p["L"][key] > 0):
            raise SystemExit(f"estimator priors {path}: L.{key} must be a positive number")
    for key in ("solver", "refine"):
        t = p["cond"].get(key)
        if not (isinstance(t, list) and len(t) >= 2 and all(isinstance(row, list) and len(row) == 2 for row in t)
                and [row[0] for row in t] == sorted(row[0] for row in t) and t[0][0] == 0
                and all(type(row[0]) is int and type(row[1]) in (int, float) and row[1] >= 0 for row in t)):
            raise SystemExit(f"estimator priors {path}: cond.{key} must be [[used_from, expected_remaining], ...] from 0")
    return p


def km_restricted_mean(obs: list[tuple[float, bool]], tau: float) -> float:
    """Kaplan–Meier restricted mean ∫₀^τ S(t) dt: obs is (value, fully observed?), False = right-censored (a running
    session censored at used). At tied times events are counted before censored ones are removed (the standard
    convention)."""
    rows = sorted(obs)
    at_risk, surv, last, area, i = len(rows), 1.0, 0.0, 0.0, 0
    while i < len(rows) and rows[i][0] <= tau:
        t = rows[i][0]
        j, events, censored = i, 0, 0
        while j < len(rows) and rows[j][0] == t:
            events += rows[j][1]
            censored += not rows[j][1]
            j += 1
        area += surv * (t - last)
        last = t
        if events and at_risk > 0:
            surv *= 1 - events / at_risk
        at_risk -= events + censored
        i = j
    return area + surv * (tau - last)


def prior_restricted_mean(totals: list[float], tau: float) -> float:
    """The prior distribution's restricted mean over the same [0, τ]: mean(min(T, τ))."""
    return sum(min(v, tau) for v in totals) / len(totals)


def difficulty_multiplier(obs: list[tuple[float, bool]], prior_totals: list[float], n0: int, clip: tuple[float, float],
                          min_finished: int = ESTIMATOR_MIN_FINISHED) -> dict:
    """This run's difficulty factor m: the ratio r of this run's s0 / s2 restricted mean (censored ones included) to the
    prior's, shrunk toward 1 (the prior weighted as n0 lines), clipped. 1 while fewer than min_finished have finished."""
    finished = [v for v, event in obs if event]
    if len(finished) < min_finished:
        return {"m": 1.0, "r": None, "tau": None, "finished": len(finished), "running": len(obs) - len(finished)}
    tau = max(finished)
    prior = prior_restricted_mean(prior_totals, tau)
    r = km_restricted_mean(obs, tau) / prior if prior > 0 else 1.0
    m = (n0 + len(finished) * r) / (n0 + len(finished))
    return {"m": min(max(m, clip[0]), clip[1]), "r": r, "tau": tau, "finished": len(finished),
            "running": len(obs) - len(finished)}


def cond_remaining(table: list[list[float]], used: float, floor: float) -> float:
    """The prior table's conditional remainder: E[T − u | T > u] of the bin that used falls in (past the last bin, the last
    bin's), no lower than floor."""
    value = table[0][1]
    for start, expected in table:
        if start > used:
            break
        value = expected
    return max(floor, value)


def heartbeat(stop: threading.Event, ctx, state: dict, lock: threading.Lock) -> None:
    """One line: done / running counts per session kind, running per DP, cumulative output tokens (proxy count), tok/s
    over the last 10 minutes, deadline."""
    import collections
    samples: collections.deque = collections.deque()
    window = RATE_WINDOW_SECONDS / time_scale(ctx.cfg)
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
        while len(samples) > 1 and now - samples[1][0] >= window:
            samples.popleft()
        t0, c0 = samples[0]
        rate = (cum - c0) / (now - t0) if now > t0 else 0.0
        parts = [f"HEARTBEAT done {sum(v for k, v in done.items() if k not in FILLERS)}/{state['total']} {dict(done)}",
                 f"running {sum(running.values())} {dict(running)}", f"dp_running {dp}", f"failed {failed}",
                 f"not_started {not_started}", f"out_tok {cum:,}",
                 f"tok_s_{int(min(window, now - t0) // 60)}min {rate:.1f}"]
        if ctx.deadline is not None:
            parts.append(f"deadline_in {max(0, ctx.deadline - now) / 60:.1f}min")
        with lock:
            ex = state.get("extra")
            if ex is not None and ex.get("estimator_on"):      # x39: difficulty factor, full-load throughput, s2 runway flag
                rf = ex["rate_full"]
                parts.append(f"m {ex['m']:.2f} rfull {'none' if rf is None else f'{rf:.0f}'} lead {ex['lead']}")
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
    ap.add_argument("--refine-prompts", help="refiners over two candidates (default: the config's refine_prompts, "
                                             "else v3refine)")
    ap.add_argument("--refine3-prompts", default="v3refine3", help="refine_all (three candidates)")
    ap.add_argument("--verify-prompts", default="v3verify", help="verify (s6)")
    ap.add_argument("--handoff-file", help="s0/s2 are not done until the workspace has this hand-off report (prompts v3doc)")
    ap.add_argument("--review-file", help="the verifier is not done until its workspace has this review (prompts v3verify2)")
    ap.add_argument("--reuse", type=Path, help="earlier run_arc2_seq run directory to take --reuse-kinds sessions from")
    ap.add_argument("--reuse-kinds", default="first,seq,ind")
    ap.add_argument("--extra-solvers-key", default="extra_solvers", choices=("extra_solvers", "extra_solvers_saverun"),
                    help="which config block sets the extra solvers (kaggle_run: extra_solvers_saverun in Save & Run, "
                         "whose 80 minutes never reach the competition thresholds)")
    args = ap.parse_args(argv)
    args.samples, args.salt_prefix = None, None          # prepare_run / Context read these two
    args.tasks_source = args.tasks_dir or args.challenges
    cfg = base.load_config(args.config, concurrency=args.concurrency)
    if args.refine_prompts is None:
        args.refine_prompts = cfg.get("refine_prompts", "v3refine")
    elif "refine_prompts" in cfg and cfg["refine_prompts"] != args.refine_prompts:
        raise SystemExit(f"--refine-prompts {args.refine_prompts} but the config says {cfg['refine_prompts']}")
    second_file = cfg.get("second_solution")
    if cfg["view"] != "text" or "tito" in cfg or "cache_salt" in cfg:
        raise SystemExit("run_arc2_seq supports view text, no tito, no cache_salt")
    if "solver_report" in cfg and args.handoff_file not in (None, cfg["solver_report"]):
        raise SystemExit(f"--handoff-file {args.handoff_file} but the config's solver_report is {cfg['solver_report']}")
    report_file = args.handoff_file or cfg.get("solver_report")
    if args.solutions is not None and args.challenges is None:
        raise SystemExit("--solutions goes with --challenges")
    deny = [d for d in args.deny.split(":") if d]
    if cfg["fence"] == "posix":
        if os.geteuid() != 0:
            raise SystemExit("fence posix needs root")
        if not deny:
            raise SystemExit("fence posix needs --deny (the paths holding the answers)")
        hidden = [args.tasks_dir] if args.tasks_dir else [args.challenges, *([args.solutions] if args.solutions else [])]
        for path in hidden:
            if not any(path.resolve().is_relative_to(Path(d).resolve()) for d in deny):
                raise SystemExit(f"{path} is not under any --deny path")
    kinds = ["first"] + [k for k in args.arms.split(",") if k]
    if set(kinds) & set(EXTRAS):
        raise SystemExit(f"--arms: {sorted(set(kinds) & set(EXTRAS))} come from the config's extra_solvers, not --arms")
    unknown = set(kinds) - set(KINDS)
    if unknown or len(set(kinds)) != len(kinds):
        raise SystemExit(f"--arms: unknown or repeated {sorted(unknown) or kinds}")
    for k in kinds:
        if not set(NEEDS[k]) <= set(kinds):
            raise SystemExit(f"--arms: {k} needs {NEEDS[k]}")
    wrapup = cfg.get("wrapup_tokens")
    if wrapup is not None and "verify" in kinds:
        raise SystemExit("wrapup_tokens has no wrap-up message for the verifier (prompts v3verify*)")
    extension = cfg.get("budget_extension")
    if extension is not None and not set(extension["kinds"]) <= set(kinds):
        raise SystemExit(f"budget_extension kinds {extension['kinds']} must be among the sessions {kinds}")
    if args.submit_arm is None:
        args.submit_arm = ("refine_vote" if "rerefine" in kinds
                           else "refine_exp" if set(ARM_SESSIONS["refine_exp"]) <= set(kinds) else "first")
    if args.submit_arm not in ARMS or not set(ARM_SESSIONS[args.submit_arm]) <= set(kinds):
        raise SystemExit(f"--submit-arm {args.submit_arm} needs sessions {ARM_SESSIONS.get(args.submit_arm)} "
                         f"(running {kinds})")
    second = args.second_budget or cfg["output_token_budget"]
    refine_budget = args.refine_budget or cfg["output_token_budget"] // 2
    for b in (second, refine_budget):
        if not cfg["min_resume_tokens"] < b:
            raise SystemExit(f"budget {b} must exceed min_resume_tokens {cfg['min_resume_tokens']}")
    if args.refine_wrapup is not None and wrapup is None:
        raise SystemExit("--refine-wrapup needs wrapup_tokens in the config")
    refine_wrapup = None if wrapup is None else (args.refine_wrapup or wrapup // 2)
    if refine_wrapup is not None and not cfg["min_resume_tokens"] < refine_wrapup:
        raise SystemExit(f"refiner wrap-up {refine_wrapup} must exceed min_resume_tokens {cfg['min_resume_tokens']}")
    if not 0 <= args.filler_min_left <= args.filler_guard:
        raise SystemExit("need 0 <= --filler-min-left <= --filler-guard")
    if "extra_solvers_saverun" in cfg and "extra_solvers" not in cfg:
        raise SystemExit("extra_solvers_saverun is the Save & Run variant of extra_solvers: the config needs both")
    if args.extra_solvers_key not in cfg and args.extra_solvers_key != "extra_solvers":
        raise SystemExit(f"--extra-solvers-key {args.extra_solvers_key}: not in the config")
    xs = cfg.get(args.extra_solvers_key)
    extra_kinds: list[str] = []
    if xs is not None:
        if not isinstance(xs, dict) or not EXTRA_KEYS <= set(xs) <= EXTRA_KEYS | EXTRA_OPTIONAL:
            raise SystemExit(f"extra_solvers must have exactly {sorted(EXTRA_KEYS)} (and maybe {sorted(EXTRA_OPTIONAL)}), "
                             f"got {xs!r}")
        if xs.get("tail", True) is not True or ("tail" in xs and "rerefine" not in kinds):
            raise SystemExit("extra_solvers tail must be true (leave it out to turn it off) and needs the rerefine session")
        if any(not isinstance(xs[k], int) or isinstance(xs[k], bool) or xs[k] <= 0 for k in EXTRA_KEYS):
            raise SystemExit(f"extra_solvers values must be positive integers: {xs}")
        if not args.solvers_first or not {"ind", "refine_exp"} <= set(kinds):
            raise SystemExit("extra_solvers needs --solvers-first and the ind / refine_exp sessions")
        if args.reuse:
            raise SystemExit("extra_solvers does not combine with --reuse")
        if not cfg["best_snapshot"]:
            raise SystemExit("extra_solvers reads the running sessions' .best/meta.json: needs best_snapshot true")
        if xs["max_per_task"] > len(EXTRAS):
            raise SystemExit(f"extra_solvers max_per_task is at most {len(EXTRAS)}")
        if not cfg["min_resume_tokens"] < xs["min_budget"] <= cfg["output_token_budget"]:
            raise SystemExit("extra_solvers min_budget must be in (min_resume_tokens, output_token_budget]")
        if not xs["trigger_tokens"] < cfg["output_token_budget"]:
            raise SystemExit("extra_solvers trigger_tokens must be below output_token_budget")
        if cfg.get("wrapup_tokens") is None or not cfg["min_resume_tokens"] < xs["wrapup_tokens"]:
            raise SystemExit("extra_solvers wrapup_tokens needs the config's wrapup_tokens and must exceed min_resume_tokens")
        extra_kinds = list(EXTRAS[:xs["max_per_task"]])
        if xs.get("prompts", "first") not in ("first", "ind"):
            raise SystemExit(f"extra_solvers prompts must be \"first\" or \"ind\", not {xs.get('prompts')!r}")
        if xs.get("prompts") == "ind" and not cfg.get("ind_prompts"):
            raise SystemExit("extra_solvers prompts \"ind\" needs the config's ind_prompts")
        inf_cfg = xs.get("informed")
        if inf_cfg is not None:
            if not isinstance(inf_cfg, dict) or not INFORMED_KEYS <= set(inf_cfg) <= INFORMED_KEYS | INFORMED_OPTIONAL:
                raise SystemExit(f"extra_solvers informed must have exactly {sorted(INFORMED_KEYS)} (and maybe "
                                 f"{sorted(INFORMED_OPTIONAL)}), got {inf_cfg!r}")
            if "disagree" in inf_cfg and not isinstance(inf_cfg["disagree"], bool):
                raise SystemExit(f"extra_solvers informed disagree must be true or false, not {inf_cfg['disagree']!r}")
            m = inf_cfg["min_output_tokens"]
            if not isinstance(m, int) or isinstance(m, bool) or m < 0:
                raise SystemExit(f"extra_solvers informed min_output_tokens must be a non-negative integer, not {m!r}")
            if not isinstance(inf_cfg["prompts"], str) or not (HERE / "prompts" / inf_cfg["prompts"]).is_dir():
                raise SystemExit(f"extra_solvers informed prompts {inf_cfg['prompts']!r} is not a prompt set")
            if not second_file or not cfg.get("ind_prompts"):
                raise SystemExit("extra_solvers informed fires on s2's second answer: needs second_solution and ind_prompts")
            if not report_file:
                raise SystemExit("extra_solvers informed hands s0's and s2's reports over: needs solver_report")
        if xs.get("reserve_queued", True) is not True:
            raise SystemExit("extra_solvers reserve_queued must be true (leave it out to turn it off)")
        agr_cfg = xs.get("agreement")
        if agr_cfg is not None:
            if not isinstance(agr_cfg, dict) or not AGREEMENT_KEYS <= set(agr_cfg) <= AGREEMENT_KEYS | AGREEMENT_OPTIONAL:
                raise SystemExit(f"extra_solvers agreement must have exactly {sorted(AGREEMENT_KEYS)} (and maybe "
                                 f"{sorted(AGREEMENT_OPTIONAL)}), got {agr_cfg!r}")
            m = agr_cfg["min_output_tokens"]
            if not isinstance(m, int) or isinstance(m, bool) or m < 0:
                raise SystemExit(f"extra_solvers agreement min_output_tokens must be a non-negative integer, not {m!r}")
            if "max_budget" in agr_cfg:
                mb = agr_cfg["max_budget"]
                if (not isinstance(mb, int) or isinstance(mb, bool)
                        or not xs["min_budget"] <= mb <= cfg["output_token_budget"]):
                    raise SystemExit(f"extra_solvers agreement max_budget must be an integer in [min_budget, "
                                     f"output_token_budget], not {mb!r}")
            if agr_cfg.get("after_s2", True) is not True:
                raise SystemExit("extra_solvers agreement after_s2 must be true (leave it out to turn it off)")
            if not PRIORITY_SOLVERS_FIRST["ind"] < PRIORITY_AGREEMENT_AFTER_S2 < PRIORITY_SOLVERS_FIRST["refine_exp"]:
                raise RuntimeError("PRIORITY_AGREEMENT_AFTER_S2 must sit between s2 and refine_exp")
            if not PRIORITY_SOLVERS_FIRST["refine_exp"] < PRIORITY_AGREEMENT_NONBLOCKING:
                raise RuntimeError("PRIORITY_AGREEMENT_NONBLOCKING must sit after refine_exp")
            if "blocking" in agr_cfg and not isinstance(agr_cfg["blocking"], bool):
                raise SystemExit(f"extra_solvers agreement blocking must be true or false, not {agr_cfg['blocking']!r}")
            if agr_cfg.get("blocking", True) is False:
                if agr_cfg.get("after_s2"):
                    raise SystemExit("extra_solvers agreement blocking false replaces after_s2: give one of them")
                if "rerefine" not in kinds:
                    raise SystemExit("extra_solvers agreement blocking false hands its result to the re-refiner: needs the "
                                     "rerefine session")
            if "max_inflight" in agr_cfg and (not isinstance(agr_cfg["max_inflight"], int)
                                              or isinstance(agr_cfg["max_inflight"], bool) or agr_cfg["max_inflight"] < 1):
                raise SystemExit(f"extra_solvers agreement max_inflight must be an int >= 1, not {agr_cfg['max_inflight']!r}")
        if xs.get("undecided", True) is not True:
            raise SystemExit("extra_solvers undecided must be true (leave it out to turn it off)")
        if xs.get("undecided") and xs["max_per_task"] < 2:
            raise SystemExit("extra_solvers undecided opens the next extra solver after one: needs max_per_task 2")
        bk = xs.get("backlog")
        if bk is not None:
            if not isinstance(bk, dict) or set(bk) != BACKLOG_KEYS:
                raise SystemExit(f"extra_solvers backlog must have exactly {sorted(BACKLOG_KEYS)}, got {bk!r}")
            if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in bk.values()):
                raise SystemExit(f"extra_solvers backlog values must be positive integers: {bk}")
            if bk["running_solver"] < BACKLOG_RUNNING_SOLVER_MIN or bk["running_refine"] < BACKLOG_RUNNING_REFINE_MIN:
                raise SystemExit(f"extra_solvers backlog running_solver / running_refine must be at least the floors "
                                 f"{BACKLOG_RUNNING_SOLVER_MIN} / {BACKLOG_RUNNING_REFINE_MIN}: {bk}")
        est_cfg, priors = xs.get("estimator"), None
        if est_cfg is not None:                 # x39
            if bk is not None:
                raise SystemExit("extra_solvers estimator and backlog are alternatives: give one of them")
            if not isinstance(est_cfg, dict) or set(est_cfg) != ESTIMATOR_KEYS:
                raise SystemExit(f"extra_solvers estimator must have exactly {sorted(ESTIMATOR_KEYS)}, got {est_cfg!r}")
            if not (isinstance(est_cfg["priors"], str) and est_cfg["priors"].endswith(".json") and ".." not in est_cfg["priors"]
                    and not est_cfg["priors"].startswith("/")):
                raise SystemExit(f"extra_solvers estimator priors must be a .json path under {HERE}, not {est_cfg['priors']!r}")
            if not isinstance(est_cfg["shrink_n0"], int) or isinstance(est_cfg["shrink_n0"], bool) or est_cfg["shrink_n0"] < 1:
                raise SystemExit(f"extra_solvers estimator shrink_n0 must be an int >= 1, not {est_cfg['shrink_n0']!r}")
            clip = est_cfg["m_clip"]
            if not (isinstance(clip, list) and len(clip) == 2 and all(type(v) in (int, float) for v in clip)
                    and 0 < clip[0] <= 1 <= clip[1]):
                raise SystemExit(f"extra_solvers estimator m_clip must be [lo <= 1, hi >= 1], not {clip!r}")
            priors = load_priors(HERE / est_cfg["priors"])
        blocking_until = xs.get("blocking_until_seconds")
        if blocking_until is not None and (not isinstance(blocking_until, int) or isinstance(blocking_until, bool)
                                           or blocking_until <= xs["refine_reserve_seconds"]):
            raise SystemExit("extra_solvers blocking_until_seconds must be an int above refine_reserve_seconds (an extra "
                             "admitted closer to the deadline than this does not hold up refine_exp)")
        hold = xs.get("retry_hold_seconds")
        if hold is not None:
            if not isinstance(hold, int) or isinstance(hold, bool) or hold <= 0:
                raise SystemExit(f"extra_solvers retry_hold_seconds must be a positive integer, not {hold!r}")
            if inf_cfg is None and agr_cfg is None and not xs.get("undecided"):
                raise SystemExit("extra_solvers retry_hold_seconds holds refine_exp for an informed, agreement or "
                                 "undecided extra solver: needs one of them")
        on = xs.get("retry_hold_on")
        if on is not None:
            if (not isinstance(on, list) or not on or len(set(on)) != len(on)
                    or not set(on) <= set(RETRY_HOLD_REASONS)):
                raise SystemExit(f"extra_solvers retry_hold_on must be a non-empty list of distinct reasons from "
                                 f"{list(RETRY_HOLD_REASONS)}, not {on!r}")
            if hold is None:
                raise SystemExit("extra_solvers retry_hold_on needs retry_hold_seconds")
        # x38 review: once extra solvers are shut down at the refiner cutoff, undecided is necessarily blocked by budget;
        # holding only makes refine_exp wait a further retry_hold_seconds in vain after the cutoff
        if xs.get("undecided") and hold is not None and "budget" in xs.get("retry_hold_on", RETRY_HOLD_REASONS):
            raise SystemExit("extra_solvers undecided with retry_hold_seconds needs retry_hold_on without \"budget\" "
                             "(a budget refusal does not recover; holding would delay refine_exp past the cutoff)")
    else:
        inf_cfg = agr_cfg = est_cfg = priors = blocking_until = None
    if extension is not None and "idle" in extension:
        solvers = {"first", "ind"} & set(kinds) | set(extra_kinds)
        if not set(extension["idle"]["kinds"]) <= solvers:
            raise SystemExit(f"budget_extension idle kinds {extension['idle']['kinds']} must be among the solvers "
                             f"{sorted(solvers)}")
    reuse_kinds = [k for k in args.reuse_kinds.split(",") if k] if args.reuse else []
    if not set(reuse_kinds) <= set(kinds):
        raise SystemExit(f"--reuse-kinds {reuse_kinds} must be among the sessions {kinds}")
    seq_prompts = load_prompt_set(args.seq_prompts, SEQ_PROMPTS) if "seq" in kinds else {}
    repair_tokens = cfg.get("refine_repair_tokens")
    # A repair is one resume of the same thread, after which solve stops (repaired only once; no resume with less than
    # min_resume_tokens left), so unlike the wrap-up allowance it need not exceed min_resume_tokens (x35 is 8,000 vs
    # 10,000; 10-31 Kaggle v2 was blocked by the old version of this check)
    if repair_tokens is not None and (not isinstance(repair_tokens, int) or isinstance(repair_tokens, bool)
                                      or repair_tokens <= 0):
        raise SystemExit(f"refine_repair_tokens must be a positive integer, not {repair_tokens!r}")
    refine_names = REFINE_PROMPTS + (("wrapup.md",) if wrapup is not None else ()) \
        + (("wrapup_time.md",) if cfg.get("wrapup_time_seconds") else ()) \
        + (("repair.md",) if repair_tokens is not None else ())
    def refine_set(name: str) -> dict[str, str]:
        has_compact = (HERE / "prompts" / name / REFINE_COMPACT).is_file()
        return load_prompt_set(name, refine_names + ((REFINE_COMPACT,) if has_compact else ()))
    refine_prompts = refine_set(args.refine_prompts) if {"refine_seq", "refine_exp", "rerefine"} & set(kinds) else {}
    refine3_prompts = refine_set(args.refine3_prompts) if "refine_all" in kinds else {}
    # <<SOLVERS>> prompts say the candidates are mutually independent: s1 (seq) works after seeing s0, so it cannot go in
    if SOLVERS_MARK in refine3_prompts.get("brief.md", "") or (SOLVERS_MARK in refine_prompts.get("brief.md", "")
                                                                and "refine_seq" in kinds):
        raise SystemExit(f"prompts with {SOLVERS_MARK} take independent candidates: refine_exp / rerefine only")
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
    # extra solvers also get their episodes (uid, workspace name) assigned up front; whether they start follows live
    # signals
    ep_kinds = kinds + extra_kinds
    episodes = {(tid, k): base.Episode(len(ep_kinds) * i + ep_kinds.index(k), tid, SAMPLE[k])
                for i, tid in enumerate(tids) for k in ep_kinds}
    ep_by_name = {ep.name: key for key, ep in episodes.items()}
    ctx, info, version = base.prepare_run(args, cfg, list(episodes.values()), deny, None, dp_size=args.dp_size,
                                          deadline=args.deadline_unix, scored=scored)
    run_dir = ctx.run_dir
    ind_ctx = ind_context(ctx, cfg, run_dir) if cfg.get("ind_prompts") and "ind" in kinds else ctx
    if cfg.get("ind_prompts") and "ind" not in kinds:
        raise SystemExit(f"the config's ind_prompts {cfg['ind_prompts']} needs the ind session")
    informed_ctx = named_context(ctx, inf_cfg["prompts"], run_dir, "prompts_informed") if inf_cfg is not None else None
    if informed_ctx is not None and PREVIOUS_MARK not in informed_ctx.prompts["brief.md"]:
        raise SystemExit(f"extra_solvers informed prompts {inf_cfg['prompts']}: brief.md has no {PREVIOUS_MARK}")
    for name, texts in (("prompts_seq", seq_prompts), ("prompts_refine", refine_prompts),
                        ("prompts_refine3", refine3_prompts), ("prompts_verify", verify_prompts)):
        if texts:
            (run_dir / "harness" / name).mkdir()
            for f, t in texts.items():
                (run_dir / "harness" / name / f).write_text(t)
    # the refiner's context: if its prompt set has its own compact prompt, switch to it (other driver files and
    # developer_instructions same as s0)
    refine_ctx, refine3_ctx = ((compact_context(ctx, run_dir / "harness" / name / REFINE_COMPACT)
                                if REFINE_COMPACT in texts else ctx)
                               for name, texts in (("prompts_refine", refine_prompts), ("prompts_refine3", refine3_prompts)))
    all_prompts = {**{f"{cfg['prompts']}/{n}": t for n, t in ctx.prompts.items()},
                   **({f"{cfg['ind_prompts']}/{n}": t for n, t in ind_ctx.prompts.items()} if ind_ctx is not ctx else {}),
                   **({f"{inf_cfg['prompts']}/{n}": t for n, t in informed_ctx.prompts.items()} if informed_ctx is not None else {}),
                   **{f"{args.seq_prompts}/{n}": t for n, t in seq_prompts.items()},
                   **{f"{args.refine_prompts}/{n}": t for n, t in refine_prompts.items()},
                   **{f"{args.refine3_prompts}/{n}": t for n, t in refine3_prompts.items()},
                   **{f"{args.verify_prompts}/{n}": t for n, t in verify_prompts.items()}}
    reused = reuse_sessions(args.reuse, reuse_kinds, cfg, all_prompts, tids, episodes, run_dir) if args.reuse else {}
    state_file = {
        "config": cfg, "args": {k: str(v) for k, v in vars(args).items() if k not in ("samples", "salt_prefix")},
        "kinds": kinds, "budgets": {"first": cfg["output_token_budget"], "second": second, "refine": refine_budget,
                                    **({"wrapup": {"solver": wrapup, "refine": refine_wrapup}} if wrapup else {}),
                                    **({"refine_repair": repair_tokens} if repair_tokens is not None else {}),
                                    **({"extension": extension} if extension else {})},
        "filler_gate_seconds": ({"guard": args.filler_guard, "min_left": args.filler_min_left}
                                if set(kinds) & set(FILLERS) else None),
        "codex_version": version,
        "prompt_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in all_prompts.items()},
        "server_info": {k: info.get(k) for k in ("context_length", "max_running_requests", "served_model_name",
                                                 "default_sampling_params", "preferred_sampling_params",
                                                 "sampling_defaults", "model_path", "version")},
        "tasks": tids, "started": time.strftime("%FT%T%z"), "scored": scored, "deadline_unix": args.deadline_unix,
        "dp_size": args.dp_size, "solvers_first": args.solvers_first, "submit_arm": args.submit_arm,
        **({"extra_solvers": xs} if xs is not None else {}),
        **({"record": record} if record is not None else {}),
        **({"reuse": {"run": str(args.reuse.resolve()), "kinds": reuse_kinds}} if args.reuse else {}),
        **({"handoff_file": report_file} if report_file else {}),
        **({"review_file": args.review_file} if args.review_file else {}),
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
    main_run = [key for key in to_run if key[1] not in FILLERS and key[1] not in extra_kinds]
    failures: list[dict] = []
    state = {"done": [], "running": {}, "total": len(main_run), "not_started": [], "failed": [],
             "fillers": {"done": [], "not_started": [], "unusable": [], "failed": [], "skipped_agreed": []},
             "extra": {"triggered": {tid: [] for tid in tids}, "flagged": {}, "promoted": [], "budget": {},
                       "skipped": [], "candidates": {}, "locked": set(), "rate": None, "backlog_refused": 0,
                       "monitor_error": None, "tail": [], "informed": [], "agreement": [], "undecided": [],
                       #: reserve_queued: (tid, kind) → tokens reserved while queued; held: tid → refine_exp waiting on
                       #: informed / agreement admission
                       "queued": {}, "held": {},
                       #: every admission decision: trigger → result → tasks; noted: (task, trigger, result) already logged
                       "admission": {}, "noted": set(),
                       #: x39: extra solvers that do not block refine_exp (episode names), agreement tasks retrying at the
                       #: next poll, tasks that had a re-refiner queued; the estimator's difficulty factor and its details,
                       #: full-load total throughput, the s2 runway flag and a record of its flips
                       "nonblocking": set(), "agreement_pending": set(), "rerefine_queued": set(), "estimator_on": False,
                       "m": 1.0, "m_info": None, "rate_full": None, "lead": None, "lead_changes": []}}
    priority = PRIORITY_SOLVERS_FIRST if args.solvers_first else PRIORITY

    def finals_of(tid: str) -> dict:
        return {a: finals[a][tid] for a in finals if tid in finals[a]}

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

    sched = Scheduler(cfg["concurrency"], filler_gate, extra_cap=xs["max_running"] if xs else 0)
    sched.add_pending(len(main_run))
    started = time.time()

    def ctx_for(kind: str, tid: str):
        task = tasks[tid]
        rs = results[tid]
        if kind in ("first", "ind"):
            own = ind_ctx if kind == "ind" else ctx
            c = own if kind == "first" or second == cfg["output_token_budget"] else \
                dataclasses.replace(own, cfg={**cfg, "output_token_budget": second})
            return c, ({"status_fn": handoff_status(report_file)} if report_file else {}) \
                | ({"second": second_file} if second_file else {})
        if kind in EXTRAS:
            # prompts per extra_solvers.prompts (s0's set, or s2's ind_prompts); the informed line uses the informed set,
            # with <<PREVIOUS>> filled in its brief and s0's, s2's reports in the workspace's previous/A, previous/B. The
            # budget is computed at start (extra_budget); no extension
            informed = episodes[(tid, kind)].name in state["extra"]["informed"]
            own = informed_ctx if informed else ind_ctx if xs.get("prompts") == "ind" else ctx
            c_cfg = {**cfg, "output_token_budget": state["extra"]["budget"][(tid, kind)], "wrapup_tokens": xs["wrapup_tokens"]}
            hooks = ({"second": second_file} if second_file else {}) \
                | ({"status_fn": handoff_status(report_file)} if report_file else {})
            if not informed:
                return dataclasses.replace(own, cfg=c_cfg), hooks
            pairs = [(chr(ord("A") + i), k) for i, k in enumerate(rank_candidates(rs, ["first", "ind"]))]
            has_report = {letter: (run_dir / "tasks" / episodes[(tid, k)].name / report_file).is_file() for letter, k in pairs}
            text = informed_previous(task, [(letter, rs[k]) for letter, k in pairs], report_file, has_report)
            prompts = {**own.prompts, "brief.md": own.prompts["brief.md"].replace(PREVIOUS_MARK, escape(text))}
            left = sorted(n for n, t in prompts.items() if PLACEHOLDER.search(t))
            if left:
                raise RuntimeError(f"{kind} {tid}: placeholders left in {left}")
            def previous(ws: Path) -> None:
                for letter, k in pairs:
                    copied = copy_report(run_dir / "tasks" / episodes[(tid, k)].name, ws / "previous" / letter, report_file)
                    if copied != has_report[letter]:
                        raise RuntimeError(f"{kind} {tid}: {letter}'s {report_file} changed between the brief and the copy")
            return dataclasses.replace(own, prompts=prompts, cfg=c_cfg), {**hooks, "extra_files": previous}
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
        generic = SOLVERS_MARK in refine_prompts.get("brief.md", "")      # run() already rejected refine_seq / refine_all
        if kind in ("refine_exp", "rerefine") and any(k in rs for k in extra_kinds):
            # tasks with extra solvers: the <<SOLVERS>> set takes s0, s2 and every finished extra solver; the
            # two-candidate set picks two with pick_two
            pool = [k for k in ("first", "ind", *extra_kinds) if k in rs]
            names = rank_candidates(rs, pool) if generic else pick_two(rs, pool)
            state["extra"]["candidates"][episodes[(tid, kind)].name] = list(names)
        elif generic:
            names = rank_candidates(rs, list(names))
        cands = [(chr(ord("A") + i), k) for i, k in enumerate(names)]
        if generic:
            own, pset = refine_ctx, refine_prompts
            text = candidates_summary_all(task, [(letter, rs[k]) for letter, k in cands])
        elif len(cands) == 2:
            own, pset, text = refine_ctx, refine_prompts, candidates_summary(task, rs[names[0]], rs[names[1]])
        else:
            own, pset = refine3_ctx, refine3_prompts
            text = candidates_summary_n(task, [(letter, rs[k]) for letter, k in cands])
        prompts = {**own.prompts, **{n: t for n, t in pset.items() if n != "brief.md"},
                   "brief.md": pset["brief.md"].replace("<<CANDIDATES>>", escape(text))}
        if cfg["midpoint_reminder_tokens"] is not None:
            reminder = min(cfg["midpoint_reminder_tokens"], refine_budget // 2)
            c_cfg = {**cfg, "output_token_budget": refine_budget, "midpoint_reminder_tokens": reminder}
        else:
            prompts.pop("midpoint_reminder.md")
            c_cfg = {**cfg, "output_token_budget": refine_budget,
                     **({"wrapup_tokens": refine_wrapup} if refine_wrapup is not None else {})}
        fills = {"<<FINAL_FILES>>": final_files_text(len(task["test"])),
                 "<<BUDGET_END>>": BUDGET_END[(c_cfg.get("wrapup_tokens") is not None,
                                               bool(c_cfg.get("wrapup_time_seconds")))]}
        if generic:     # <<SOLVERS>> only in the set that takes any number of candidates (pick_two's two may lack s0 / s2)
            fills[SOLVERS_MARK] = candidates_intro(
                [letter for letter, _ in cands],
                [letter for letter, k in cands if episodes[(tid, k)].name in state["extra"]["informed"]],
                # informed read s0's and s2's reports (ctx_for's pairs), not the extra solvers opened later
                [letter for letter, k in cands if k in ("first", "ind")])
        for mark, value in fills.items():
            prompts = {n: t.replace(mark, value) for n, t in prompts.items()}
        left = sorted(n for n, t in prompts.items() if PLACEHOLDER.search(t))
        if left:
            raise RuntimeError(f"{kind} {tid}: placeholders left in {left}")
        hooks = {"repair": {"tokens": repair_tokens, "text": prompts["repair.md"]}} if repair_tokens is not None else {}
        c = dataclasses.replace(own, prompts=prompts, cfg=c_cfg)
        def extra(ws: Path) -> None:
            for letter, k in cands:
                copy_solver(run_dir / "tasks" / episodes[(tid, k)].name, rs[k], ws / "candidates" / letter)
            (ws / "final" / "attempt_1").mkdir(parents=True)
            (ws / "final" / "attempt_2").mkdir()
        return c, {"extra_files": extra, "status_fn": refine_status(len(task["test"])), **hooks}

    def extension_gate(kind: str, cap: float, name: str):
        """Extend only when kind is among budget_extension's kinds or idle kinds (else None). For kinds the first
        extension looks only at the time gate (extension_decision); later ones (for idle kinds from the first on) also
        require idleness -- no main-line or extra-solver work queued, fewer running than concurrency -- and that this
        session has not passed train yet (.best/meta.json), at most idle max_grants times."""
        idle = None if extension is None else extension.get("idle")
        first = extension is not None and kind in extension["kinds"]
        repeat = idle is not None and kind in idle["kinds"]
        if not (first or repeat):
            return None
        began = time.time()
        grants = {"first": 0, "idle": 0}

        def gate(total: int) -> dict:
            d = extension_decision(extension, total, began, time.time(), cap, deadline)
            if first and grants["first"] == 0 and grants["idle"] == 0:
                grants["first"] += d["granted"]
                return {**d, "rule": "first"}
            if not d["granted"]:
                return {**d, "rule": "idle"}
            if not repeat or grants["idle"] >= idle["max_grants"]:
                return {**d, "granted": False, "rule": "idle", "reason": "max_grants"}
            with lock:
                running = len(state["running"])
            queued = sched.queued()
            out = {**d, "rule": "idle", "running": running, "queued": queued}
            if queued or running >= cfg["concurrency"]:
                return {**out, "granted": False, "reason": "busy"}
            if live_pass(name):
                return {**out, "granted": False, "reason": "passes train"}
            grants["idle"] += 1
            return {**out, "reason": "idle"}
        return gate

    def ready(tid: str, kind: str) -> bool:
        if not all(n in results[tid] for n in NEEDS[kind]):
            return False
        # extra_solvers: refine_exp waits until all "blocking" extra solvers this task opened are done (or failed, or never
        # opened); the re-refiner waits for all of them, non-blocking ones included (x39: agreement with blocking false,
        # and extras admitted with less than blocking_until_seconds left before the cutoff; their results go through the
        # re-refiner)
        if kind not in ("refine_exp", "rerefine"):
            return True
        waits = [k for k in state["extra"]["triggered"][tid]
                 if kind == "rerefine" or episodes[(tid, k)].name not in state["extra"]["nonblocking"]]
        return all(k in results[tid] or k in failed[tid] for k in waits)

    t_ref = None if xs is None or deadline is None else deadline - xs["refine_reserve_seconds"]
    #: constants the backlog gate uses to estimate the main line's remainder (configurable since x38; see BACKLOG_KEYS)
    backlog_est = (dict(xs["backlog"]) if xs is not None and "backlog" in xs else
                   {"s2": BACKLOG_S2, "refine": BACKLOG_REFINE, "running_solver": BACKLOG_RUNNING_SOLVER,
                    "running_refine": BACKLOG_RUNNING_REFINE})
    #: which refusal reasons hold refine_exp (configurable since x38; omitted = hold on both of x37's)
    hold_on = set(xs.get("retry_hold_on", RETRY_HOLD_REASONS)) if xs is not None else set()
    #: x39: agreement does not block refine_exp (blocking false); late gate = no reservation while queued, admission on
    #: budget alone, full gate right before starting (after_s2 or non-blocking)
    agr_nonblocking = agr_cfg is not None and agr_cfg.get("blocking", True) is False
    agr_late_gate = agr_cfg is not None and bool(agr_cfg.get("after_s2") or agr_nonblocking)
    #: x39 estimator: the prior table's L (unstarted s2 / refine_exp) and conditional-remainder table (running ones), both
    #: multiplied by state["extra"]["m"]
    est = None
    if priors is not None:
        est = {"cfg": est_cfg, "L": {"s2": float(priors["L"]["s2"]), "refine": float(priors["L"]["refine"])},
               "cond": priors["cond"],
               "prior_solver_totals": priors["kinds"]["first"]["totals"] + priors["kinds"]["ind"]["totals"]}
        state["extra"]["estimator_on"] = True
    if xs is not None:
        if est is None:
            base.log(f"EXTRA SOLVERS backlog estimates s2 {backlog_est['s2']:,} refine {backlog_est['refine']:,} "
                     f"running solver {backlog_est['running_solver']:,} running refiner {backlog_est['running_refine']:,} "
                     f"({'config' if 'backlog' in xs else 'defaults'}); hold on {sorted(hold_on)}; "
                     f"undecided {'on' if xs.get('undecided') else 'off'}")
        else:
            base.log(f"EXTRA SOLVERS estimator priors {est_cfg['priors']} ({priors['weights']}, from {priors['source']}): "
                     f"L s2 {est['L']['s2']:,.0f} refine {est['L']['refine']:,.0f}, cond tables solver {len(est['cond']['solver'])} "
                     f"refine {len(est['cond']['refine'])} rows, shrink_n0 {est_cfg['shrink_n0']}, m_clip {est_cfg['m_clip']}; "
                     f"room from the full-load throughput; hold on {sorted(hold_on)}; undecided {'on' if xs.get('undecided') else 'off'}; "
                     f"agreement {'non-blocking' if agr_nonblocking else 'blocking' if agr_cfg is not None else 'off'}"
                     f"{f' (max_inflight {agr_cfg['max_inflight']})' if agr_cfg is not None and agr_cfg.get('max_inflight') else ''}; "
                     f"extras admitted under {blocking_until} s to the deadline do not block refine_exp"
                     if blocking_until is not None else
                     f"EXTRA SOLVERS estimator priors {est_cfg['priors']} ({priors['weights']}): L s2 {est['L']['s2']:,.0f} "
                     f"refine {est['L']['refine']:,.0f}; room from the full-load throughput; hold on {sorted(hold_on)}; "
                     f"undecided {'on' if xs.get('undecided') else 'off'}; "
                     f"agreement {'non-blocking' if agr_nonblocking else 'blocking' if agr_cfg is not None else 'off'}")
    #: Mock e2e only: refuse the first few extra-solver admissions regardless (tests retry_hold_seconds / retry_hold_on):
    #: N = the first N are backlog refusals, or a comma list saying how each one is handled ("backlog", "budget" = refuse;
    #: "admit" = use the real gate, e.g. "admit,backlog"); when set, it is logged loudly
    raw_refusals = os.environ.get("ARC2_TEST_REFUSE_EXTRA_ADMISSIONS", "0")
    test_refusals = ["backlog"] * int(raw_refusals) if raw_refusals.isdigit() else raw_refusals.split(",")
    if not set(test_refusals) <= {*RETRY_HOLD_REASONS, "admit"}:
        raise SystemExit(f"ARC2_TEST_REFUSE_EXTRA_ADMISSIONS {raw_refusals!r}: a count or a list from "
                         f"{[*RETRY_HOLD_REASONS, 'admit']}")
    if test_refusals:
        base.log(f"TEST HOOK ARC2_TEST_REFUSE_EXTRA_ADMISSIONS={raw_refusals}: the first extra solver admissions "
                 f"are refused (mock e2e only)")

    def live_pass(name: str) -> bool:
        """Whether a running session has passed all train so far: check.py writes .best/meta.json when all pass
        (passed == total). The file is in the agent's workspace and writable by the agent: if it cannot be parsed it
        counts as not passed (this only affects whether to open more solvers), and a note is logged."""
        # The harness is root: no following symlinks (O_NOFOLLOW), no blocking on a FIFO (O_NONBLOCK), regular files
        # only, read at most META_MAX_BYTES; the content is agent-controlled input, so any parse failure counts as not
        # passed (Infinity, absurdly deep nesting are all stopped here)
        meta = run_dir / "tasks" / name / ".best" / "meta.json"
        try:
            fd = os.open(meta, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return False
        except OSError as error:
            base.log(f"{name}: .best/meta.json not opened ({error!r}); counted as not passing")
            return False
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                base.log(f"{name}: .best/meta.json is not a regular file; counted as not passing")
                return False
            raw = os.read(fd, META_MAX_BYTES + 1)
        finally:
            os.close(fd)
        try:
            if len(raw) > META_MAX_BYTES:
                raise ValueError(f"more than {META_MAX_BYTES} bytes")
            m = json.loads(raw)
            passed, total = m["passed"], m["total"]
            if type(passed) is not int or type(total) is not int:
                raise ValueError(f"passed / total are {type(passed).__name__} / {type(total).__name__}")
            return 0 < total == passed
        except Exception as error:  # noqa: BLE001 -- agent-written file: if broken it only affects opening more solvers; logged
            base.log(f"{name}: unreadable .best/meta.json ({error!r}); counted as not passing")
            return False

    def task_has_pass(tid: str) -> bool:
        """Whether this task has a candidate passing all train: finished solvers by their result, running ones by .best.
        Caller holds the lock."""
        solvers = ("first", "ind", *extra_kinds)
        if any(k in results[tid] and train_all(results[tid][k]) for k in solvers):
            return True
        return any(t == tid and k in solvers and live_pass(name) for name, (t, k) in state["running"].items())

    def extra_budget_now() -> int:
        """The soft budget for an extra solver opened now: the time to the refiner cutoff (deadline −
        refine_reserve_seconds) × the per-stream rate over the last 10 minutes − the wrap-up allowance, capped at the
        config's budget. No deadline or no rate yet = the config's budget."""
        full = cfg["output_token_budget"]
        rate = state["extra"]["rate"]
        if t_ref is None or rate is None:
            return full
        return int(min(full, rate * (t_ref - time.time()) - xs["wrapup_tokens"]))

    def admission_gate(cap: int | None = None, check_backlog: bool = True) -> tuple[str | None, int, str]:
        """The common threshold for opening an extra solver: the budget if opened now (min'd with cap if given,
        agreement's max_budget) ≥ min_budget, and (with a deadline and a rate, and check_backlog) the backlog fits.
        Returns (refusal reason "budget" / "backlog" or None, budget, numbers for the log). An after_s2 agreement looks
        only at the budget when queued (it opens only after all s2 have been dispatched, so the current backlog says
        nothing about that moment) and passes the full gate once more right before starting (worker). Caller holds the
        lock."""
        ex = state["extra"]
        budget = extra_budget_now() if cap is None else min(cap, extra_budget_now())
        if test_refusals:
            refusal = test_refusals.pop(0)
            if refusal != "admit":
                return refusal, budget, "test refusal (ARC2_TEST_REFUSE_EXTRA_ADMISSIONS)"
        if budget < xs["min_budget"]:
            return "budget", budget, f"budget {budget:,} < min_budget {xs['min_budget']:,}"
        rate = ex["rate"]
        if check_backlog and deadline is not None and rate is not None:
            back = backlog_tokens()
            # x39: room uses the full-load total throughput (until measured, conservatively RATE_FULL_FALLBACK × per-line
            # rate × C); this line's wrap-up allowance counts too (consistent with the reservation)
            full = ex["rate_full"] if ex["rate_full"] is not None else RATE_FULL_FALLBACK * rate * cfg["concurrency"]
            room = BACKLOG_FILL * full * (deadline - BACKLOG_TAIL_SECONDS / time_scale(cfg) - time.time())
            need = budget + xs["wrapup_tokens"]
            tail = f" (m {ex['m']:.2f})" if est is not None else ""
            if back + need > room:
                ex["backlog_refused"] += 1
                return "backlog", budget, f"backlog {back:,.0f} + budget {need:,} > room {room:,.0f}{tail}"
            return None, budget, f"backlog {back:,.0f} + budget {need:,} <= room {room:,.0f}{tail}"   # logged when admitted too (x38)
        return None, budget, ""

    def note(tid: str, trigger: str, outcome: str, detail: str | None = None) -> None:
        """One admission decision: the summary counts tasks by trigger and result; with detail it is logged, only once
        per task, trigger and result (flagged tasks retry every poll). Caller holds the lock."""
        ex = state["extra"]
        ex["admission"].setdefault(trigger, {}).setdefault(outcome, set()).add(tid)
        if detail is not None and (tid, trigger, outcome) not in ex["noted"]:
            ex["noted"].add((tid, trigger, outcome))
            base.log(f"{tid} [{trigger}]: NOT ADMITTED ({outcome}: {detail})")

    def push_extra(tid: str, budget: int, trigger: str, why: str, gate: str = "") -> None:
        """Admitted: open this task's next extra solver. With reserve_queued on, its budget + wrap-up allowance count
        toward the backlog while queued. The agreement line with after_s2 is ordered after s2 (PRIORITY_AGREEMENT_AFTER_S2)
        and reserves nothing while queued (0): it may wait hours until all s0 / s2 are dispatched, and a full reservation
        would crowd out informed / undecided lines decided later (x38 review); instead it passes the gate once more right
        before starting (worker). ``gate``: backlog / room at gate admission (log). Caller holds the lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        kind = extra_kinds[len(trig)]
        trig.append(kind)
        if trigger in ("tail", "informed", "agreement", "undecided"):
            ex[trigger].append(episodes[(tid, kind)].name)
        state["total"] += 1
        sched.add_pending(1)
        late_gate = trigger == "agreement" and agr_late_gate
        # x39: extra solvers that do not block refine_exp: agreement with blocking false, and ordinary / informed /
        # undecided admitted with less than blocking_until_seconds left before the cutoff (refine_exp opens as usual;
        # their results go through the re-refiner)
        nonblocking = ((trigger == "agreement" and agr_nonblocking)
                       or (trigger in ("ordinary", "informed", "undecided") and blocking_until is not None
                           and deadline is not None and deadline - time.time() < blocking_until))
        if nonblocking:
            ex["nonblocking"].add(episodes[(tid, kind)].name)
        if xs.get("reserve_queued"):
            if (tid, kind) in ex["queued"]:
                raise RuntimeError(f"{tid} {kind}: queued twice")
            ex["queued"][(tid, kind)] = 0 if late_gate else budget + xs["wrapup_tokens"]
        prio = (PRIORITY_AGREEMENT_NONBLOCKING if trigger == "agreement" and agr_nonblocking
                else PRIORITY_AGREEMENT_AFTER_S2 if late_gate else PRIORITY_EXTRA)
        sched.push_extra((prio, tids.index(tid), SAMPLE[kind]), (tid, kind))
        note(tid, trigger, "admitted")
        base.log(f"{episodes[(tid, kind)].name} [{kind}]: TRIGGER ({why}; budget now {budget:,})"
                 f"{' non-blocking' if nonblocking else ''}{f' gate: {gate}' if gate else ''}")
        return nonblocking

    def maybe_extra(tid: str, why: str) -> bool:
        """Open this task's next extra solver (ind3, then ind4). Conditions: refine_exp not queued yet, the task has no
        candidate passing all train yet, the previous extra solver is itself stuck too (flagged) or done, and the budget
        if opened now ≥ min_budget. Caller holds the lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        if (xs is None or tid in ex["locked"] or len(trig) >= len(extra_kinds) or ctx.cancelled.is_set()
                or task_has_pass(tid)):
            return False
        if trig and not (trig[-1] in results[tid] or trig[-1] in failed[tid]
                         or episodes[(tid, trig[-1])].name in ex["flagged"]):
            return False
        refused, budget, detail = admission_gate()
        if refused:
            note(tid, "ordinary", refused, detail)
            return False
        push_extra(tid, budget, "ordinary", why, detail)
        return True

    def decided(tid: str) -> bool:
        """tail: this task has two or more finished independent solvers passing all train, and their answers all agree
        (the informed extra solver read others' reports: not independent, no vote). Caller holds the lock."""
        passers = [results[tid][k] for k in ("first", "ind", *extra_kinds) if k in results[tid]
                   and train_all(results[tid][k]) and episodes[(tid, k)].name not in state["extra"]["informed"]]
        return len(passers) >= 2 and all(r["attempt_1"] == passers[0]["attempt_1"] for r in passers)

    def maybe_informed(tid: str) -> str | None:
        """extra_solvers informed: s0 and s2 are both done, both pass all train, and either (with disagree on) their first
        answers differ, or s2's second answer passes train and differs from its first and their combined output is
        ≥ min_output_tokens; if this task has not opened an extra solver yet, open the first extra solver as informed
        (reads both reports); the budget and backlog thresholds are as in maybe_extra. Returns None (conditions not
        met), "admitted" or the refusal reason. Caller holds the lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        if (inf_cfg is None or trig or tid in ex["locked"] or ctx.cancelled.is_set()
                or not {"first", "ind"} <= set(results[tid])):
            return None
        r0, r2 = results[tid]["first"], results[tid]["ind"]
        if not (train_all(r0) and train_all(r2)):
            return None
        out = r0["usage"]["proxy_output_tokens"] + r2["usage"]["proxy_output_tokens"]
        if inf_cfg.get("disagree") and r0["attempt_1"] != r2["attempt_1"]:
            why = f"s0 and s2 pass train but disagree, {out:,} output tokens together"
        elif any(g is not None for g in second_outputs(r2)) and out >= inf_cfg["min_output_tokens"]:
            why = f"s0 and s2 pass train, s2 has a second answer, {out:,} output tokens together"
        else:
            return None
        refused, budget, detail = admission_gate()
        if refused:
            note(tid, "informed", refused, detail)
            return refused
        push_extra(tid, budget, "informed", f"informed: {why}", detail)
        return "admitted"

    def agreement_inflight() -> int:
        """Number of agreement extras queued plus running (x39 max_inflight). Caller holds the lock."""
        n = 0
        for name in state["extra"]["agreement"]:
            t, k = ep_by_name[name]
            n += k not in results[t] and k not in failed[t]
        return n

    def maybe_agreement(tid: str) -> str | None:
        """extra_solvers agreement (x37 RL version): s0 and s2 both done, both pass all train, first answers agree,
        combined output ≥ min_output_tokens, and no extra solver opened for this task yet: open one independent extra
        solver before refine_exp (the extra_solvers.prompts set, not reading others' reports). informed is tried first
        (only one line per task). Returns None (conditions not met), "admitted" or the refusal reason. Caller holds the
        lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        # x39 non-blocking agreement: can still open after refine_exp is queued (locked), its result goes through the
        # re-refiner; no more once the re-refiner has been queued (or skipped)
        if (agr_cfg is None or trig or ctx.cancelled.is_set() or (tid in ex["locked"] and not agr_nonblocking)
                or (agr_nonblocking and (tid in ex["rerefine_queued"] or "rerefine" in results[tid] or "rerefine" in failed[tid]))
                or not {"first", "ind"} <= set(results[tid])):
            return None
        r0, r2 = results[tid]["first"], results[tid]["ind"]
        if not (train_all(r0) and train_all(r2) and r0["attempt_1"] == r2["attempt_1"]):
            return None
        out = r0["usage"]["proxy_output_tokens"] + r2["usage"]["proxy_output_tokens"]
        if out < agr_cfg["min_output_tokens"]:
            return None
        if agr_cfg.get("max_inflight") and agreement_inflight() >= agr_cfg["max_inflight"]:
            ex["agreement_pending"].add(tid)
            note(tid, "agreement", "inflight", f"{agreement_inflight()} agreement extras queued or running "
                                                f"(max_inflight {agr_cfg['max_inflight']}); retried every poll")
            return "inflight"
        refused, budget, detail = admission_gate(agr_cfg.get("max_budget"), check_backlog=not agr_late_gate)
        if refused:
            note(tid, "agreement", refused, detail)
            if agr_nonblocking:
                ex["agreement_pending"].add(tid)      # non-blocking: just retry later (every poll), refine_exp not held
            return refused
        ex["agreement_pending"].discard(tid)
        push_extra(tid, budget, "agreement", f"agreement: s0 and s2 pass train and agree, {out:,} output tokens together",
                   detail)
        return "admitted"

    def maybe_undecided(tid: str) -> str | None:
        """extra_solvers undecided (x38): all extra solvers this task opened have ended, refine_exp is not queued yet, s0
        and s2 are both done, the task has a train-passing candidate (if not, that is maybe_extra's job) but is not
        decided yet (decided(): two or more independent solvers passing all train with the same answer), and extra
        solvers are not used up → open the next independent one. Returns None (conditions not met), "admitted" or the
        refusal reason. Caller holds the lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        if (not xs.get("undecided") or not trig or len(trig) >= len(extra_kinds) or tid in ex["locked"]
                or ctx.cancelled.is_set() or not {"first", "ind"} <= set(results[tid])
                or not all(k in results[tid] or k in failed[tid] for k in trig)):
            return None
        if decided(tid) or not task_has_pass(tid):
            return None
        refused, budget, detail = admission_gate()
        if refused:
            note(tid, "undecided", refused, detail)
            return refused
        ind = [results[tid][k]["attempt_1"] for k in ("first", "ind", *extra_kinds) if k in results[tid]
               and train_all(results[tid][k]) and episodes[(tid, k)].name not in ex["informed"]]
        distinct = len({json.dumps(a) for a in ind})
        push_extra(tid, budget, "undecided", f"undecided: {len(ind)} independent train-passing solver(s) with "
                                             f"{distinct} different answer(s); {len(trig)} extra solver(s) done", detail)
        return "admitted"

    def can_hold(out: str | None) -> bool:
        """Whether this admission result holds refine_exp: the refusal reason is in retry_hold_on, retry_hold_seconds is
        set, and the monitor is still alive (x38 review: once the monitor dies nobody rechecks, a held refine_exp is
        never queued and the worker never ends). Caller holds the lock."""
        return (out in hold_on and bool(xs.get("retry_hold_seconds"))
                and state["extra"]["monitor_error"] is None)

    def hold(tid: str, trigger: str, reason: str) -> None:
        """retry_hold_seconds: refine_exp is not queued yet; every poll rechecks this task's informed / agreement /
        undecided. Caller holds the lock."""
        ex = state["extra"]
        if tid in ex["held"]:
            raise RuntimeError(f"{tid}: held twice")
        now = time.time()
        ex["held"][tid] = {"trigger": trigger, "since": now, "until": now + xs["retry_hold_seconds"]}
        note(tid, trigger, "held")
        base.log(f"{tid} [{trigger}]: HOLD refine_exp for up to {xs['retry_hold_seconds']} s ({reason}); "
                 f"retried every poll")

    def release(tid: str, why: str) -> bool:
        """End the hold; True means the caller enqueues refine_exp (if admitted, refine_exp waits for that extra solver
        and False is returned). Caller holds the lock."""
        h = state["extra"]["held"].pop(tid)
        note(tid, h["trigger"], why)
        base.log(f"{tid} [{h['trigger']}]: HOLD ended ({why}) after {time.time() - h['since']:.0f} s")
        return why != "admitted after hold" and refine_check(tid)

    def maybe_tail(tid: str) -> bool:
        """tail: right after refine_exp finishes, if no main-line or extra-solver work is queued, the task is not
        decided, extra solvers are not used up, the budget if opened now ≥ min_budget and the backlog fits, open the next
        extra solver (the re-refiner waits for it). Caller holds the lock."""
        ex = state["extra"]
        trig = ex["triggered"][tid]
        if not xs.get("tail") or len(trig) >= len(extra_kinds) or ctx.cancelled.is_set() or decided(tid):
            return False
        queued = sched.queued()
        if queued:
            note(tid, "tail", "queue busy", f"{queued} episodes queued")
            return False
        refused, budget, detail = admission_gate()
        if refused:
            note(tid, "tail", refused, detail)
            return False
        push_extra(tid, budget, "tail", "tail: refine_exp done, the queue empty, the task undecided", detail)
        return True

    def rerefine_open(tid: str) -> bool:
        """This task's re-refiner has not been queued (nor skipped) and is not done. Caller holds the lock."""
        return ("rerefine" in kinds and tid not in state["extra"]["rerefine_queued"]
                and "rerefine" not in results[tid] and "rerefine" not in failed[tid])

    def after_extra(tid: str, name: str) -> list[str]:
        """What to queue after an extra solver ends (done, failed or never opened): the next line opened by undecided
        (x38; refine_exp waits for it, held when blocked for a retry_hold_on reason), refine_exp (refine_check), or the
        re-refiner after tail / a non-blocking line (x39). Caller holds the lock."""
        ex = state["extra"]
        if name not in ex["nonblocking"]:
            out = maybe_undecided(tid)
            if out == "admitted" and episodes[(tid, ex["triggered"][tid][-1])].name not in ex["nonblocking"]:
                return []                       # blocking one opened: refine_exp waits for it (a non-blocking one does not)
            if out != "admitted" and can_hold(out):
                hold(tid, "undecided", out)
                return []
            if refine_check(tid):
                return ["refine_exp"]
        if (name in ex["tail"] or name in ex["nonblocking"]) and ready(tid, "rerefine") and rerefine_open(tid):
            return ["rerefine"]
        return []

    def backlog_tokens() -> float:
        """Output the main line still needs (estimate: the BACKLOG_* constants, or the x39 estimator's priors × m).
        Running refine_exp are counted once (before x38 the "unfinished" term also counted running ones, so x37 at D−200
        overestimated 1.67× → 1.01× once removed). Caller holds the lock."""
        ex = state["extra"]
        m = ex["m"] if est is not None else 1.0
        w = 0.0
        for t in tids:
            if "ind" not in results[t] and "ind" not in failed[t] and (t, "ind") not in sched.popped:
                w += (est["L"]["s2"] * m) if est is not None else backlog_est["s2"]
            if "refine_exp" not in results[t] and "refine_exp" not in failed[t] and (t, "refine_exp") not in sched.popped:
                w += (est["L"]["refine"] * m) if est is not None else backlog_est["refine"]
        for name, (t, k) in state["running"].items():
            with base.LOCK:
                proxy = ctx.proxies.get(name)
            used = 0 if proxy is None else proxy.output_tokens
            if k in ("first", "ind"):
                w += (cond_remaining(est["cond"]["solver"], used, BACKLOG_RUNNING_SOLVER_MIN / m) * m if est is not None
                      else max(BACKLOG_RUNNING_SOLVER_MIN, backlog_est["running_solver"] - used))
            elif k == "refine_exp":
                w += (cond_remaining(est["cond"]["refine"], used, BACKLOG_RUNNING_REFINE_MIN / m) * m if est is not None
                      else max(BACKLOG_RUNNING_REFINE_MIN, backlog_est["running_refine"] - used))
            elif k in extra_kinds:
                w += max(0, ex["budget"].get((t, k), 0) + xs["wrapup_tokens"] - used)
        if xs.get("reserve_queued"):
            w += sum(ex["queued"].values())        # queued extra solvers (removed when started or SKIPped)
        return w

    def update_estimator(now: float) -> None:
        """x39: every poll, update the difficulty factor m (KM restricted mean of this run's s0 / s2 relative to the
        prior's) and the s2 runway flag lead. Caller holds the lock."""
        ex = state["extra"]
        obs = []
        for t in tids:
            for k in ("first", "ind"):
                r = results[t].get(k)
                if r is not None:
                    obs.append((r["usage"]["proxy_output_tokens"] or 0, True))
        for name, (t, k) in state["running"].items():
            if k in ("first", "ind"):
                with base.LOCK:
                    proxy = ctx.proxies.get(name)
                if proxy is not None:
                    obs.append((proxy.output_tokens, False))
        info = difficulty_multiplier(obs, est["prior_solver_totals"], est["cfg"]["shrink_n0"], tuple(est["cfg"]["m_clip"]))
        if round(info["m"], 2) != round(ex["m"], 2):
            base.log(f"ESTIMATOR m {ex['m']:.2f} -> {info['m']:.2f} (r {info['r']:.3f}, tau {info['tau']:,}, "
                     f"{info['finished']} finished, {info['running']} running s0/s2)")
        ex["m"], ex["m_info"] = info["m"], info
        queued_s2 = sum(1 for t in tids if "ind" not in results[t] and "ind" not in failed[t] and (t, "ind") not in sched.popped)
        rate = ex["rate"]
        full = ex["rate_full"] if ex["rate_full"] is not None else (
            None if rate is None else RATE_FULL_FALLBACK * rate * cfg["concurrency"])   # same fallback as admission_gate
        if deadline is None or t_ref is None:
            lead, eta = True, None                          # no deadline: no refiner cutoff to protect
        elif full is None or full <= 0:
            lead, eta = False, None                         # no throughput measured yet: conservative
        else:
            length = est["L"]["s2"] * ex["m"]
            eta = queued_s2 * length / full + length / (full / cfg["concurrency"])
            lead = now + eta <= t_ref
        if lead != ex["lead"]:
            ex["lead_changes"].append({"t": round(now - started), "lead": lead, "queued_s2": queued_s2,
                                       "eta_s": None if eta is None else round(eta), "m": round(ex["m"], 3)})
            base.log(f"LEAD {'on' if lead else 'off'}: extras {'ahead of' if lead else 'behind'} the queued s2 "
                     f"({queued_s2} queued, eta {'n/a' if eta is None else f'{eta / 60:.0f} min'} to the refiner cutoff "
                     f"{'n/a' if t_ref is None else f'{(t_ref - now) / 60:.0f} min'} away, m {ex['m']:.2f}, "
                     f"R_full {'none' if full is None else f'{full:.0f}'} tok/s)")
        ex["lead"] = lead
        sched.set_lead(lead)

    def refine_check(tid: str) -> bool:
        """extra_solvers: whether refine_exp can be queued now. Before that, check whether to open another extra solver
        (s0, s2 and those already opened are all done and none passes all train); once it is to be queued, lock this task
        (no more extra solvers after that). Caller holds the lock; True means the caller enqueues."""
        ex = state["extra"]
        if tid in ex["locked"] or "refine_exp" in results[tid] or "refine_exp" in failed[tid]:
            return False
        if not ready(tid, "refine_exp"):
            return False
        maybe_extra(tid, "s0, s2 and the extra solvers so far finished without passing train")
        if not ready(tid, "refine_exp"):
            return False                  # a blocking extra solver was opened: wait (x39: non-blocking ones don't hold it)
        ex["locked"].add(tid)
        return True

    def watch_extras(stop: threading.Event) -> None:
        """Every poll_seconds: update each stream's rate; a running s0 / s2 / extra solver whose output has passed
        trigger_tokens without passing all train = flagged (when s0 is flagged, this task's s2 is promoted if not started
        yet); flagged tasks try to open the next extra solver."""
        import collections
        samples: collections.deque = collections.deque()
        solvers = ("first", "ind", *extra_kinds)
        while not stop.wait(xs["poll_seconds"]):
            try:
                poll(samples, solvers)
            except Exception as error:  # noqa: BLE001 -- monitor broke: log loudly, disable it; main line runs on (no extras)
                with lock:
                    state["extra"]["monitor_error"] = "".join(traceback.format_exception(error))
                base.log(f"EXTRA SOLVERS MONITOR FAILED, no more flags / triggers: "
                         f"{state['extra']['monitor_error']}")
                sched.set_lead(False)           # x39: runway rule no longer computed; queued extras drop behind s2
                with lock:                      # nobody rechecks held refine_exp anymore: release them all
                    for tid in sorted(state["extra"]["held"], key=tids.index):
                        if release(tid, "monitor failed"):
                            enqueue(tid, "refine_exp")
                return

    def poll(samples, solvers) -> None:
        """One round of watch_extras (the caller catches exceptions)."""
        now = time.time()
        with base.LOCK:
            cum = sum(p.output_tokens for p in ctx.proxies.values())
        with lock:
            running = len(state["running"])
        samples.append((now, cum, running))
        while len(samples) > 1 and now - samples[1][0] >= RATE_WINDOW_SECONDS / time_scale(cfg):
            samples.popleft()
        t0, c0, _ = samples[0]
        streams = sum(r for _, _, r in samples) / len(samples)
        with lock:
            ex = state["extra"]
            if now - t0 >= RATE_MIN_SECONDS / time_scale(cfg) and streams > 0 and cum > c0:
                ex["rate"] = (cum - c0) / (now - t0) / streams
                if streams >= RATE_FULL_LOAD * cfg["concurrency"]:      # x39: EMA of the full-load total throughput (for room)
                    agg = (cum - c0) / (now - t0)
                    ex["rate_full"] = agg if ex["rate_full"] is None else \
                        (1 - RATE_FULL_ALPHA) * ex["rate_full"] + RATE_FULL_ALPHA * agg
            if est is not None:
                update_estimator(now)
            hard = set()
            for name, (tid, kind) in list(state["running"].items()):
                if kind not in solvers or name in ex["flagged"]:
                    continue
                with base.LOCK:
                    proxy = ctx.proxies.get(name)
                if proxy is None or proxy.output_tokens < xs["trigger_tokens"] or live_pass(name):
                    continue
                ex["flagged"][name] = round(now - started)
                base.log(f"{name} [{kind}]: FLAG {proxy.output_tokens:,} output tokens without passing train")
                hard.add(tid)
                if kind == "first" and (tid, "ind") not in sched.popped and "ind" not in results[tid] \
                        and "ind" not in failed[tid] and tid not in ex["promoted"]:
                    ex["promoted"].append(tid)
                    sched.push((PRIORITY_PROMOTED, tids.index(tid), SAMPLE["ind"]), (tid, "ind"))
                    base.log(f"{episodes[(tid, 'ind')].name} [ind]: PROMOTED (s0 flagged)")
            hard |= {tid for name, (tid, kind) in state["running"].items() if name in ex["flagged"]}
            for tid in sorted(hard, key=tids.index):
                maybe_extra(tid, "a flagged session")
            # retry_hold_seconds: blocked informed / agreement / undecided are rechecked every poll (informed, undecided
            # first; agreement fills gaps, so it goes last); on expiry (or cancellation) refine_exp is released; it is also
            # released when the recheck's refusal reason is not in retry_hold_on (x38: budget)
            retry = {"informed": maybe_informed, "agreement": maybe_agreement, "undecided": maybe_undecided}
            rank = {"informed": 0, "undecided": 0, "agreement": 1}
            for tid in sorted(ex["held"], key=lambda t: (rank[ex["held"][t]["trigger"]], tids.index(t))):
                h = ex["held"][tid]
                if ctx.cancelled.is_set() or now >= h["until"]:
                    if release(tid, "cancelled" if ctx.cancelled.is_set() else "expired"):
                        enqueue(tid, "refine_exp")
                    continue
                out = retry[h["trigger"]](tid)
                if out == "admitted":
                    release(tid, "admitted after hold")
                elif out is None:
                    raise RuntimeError(f"{tid}: refine_exp held for {h['trigger']}, which no longer applies")
                elif out not in hold_on:
                    if release(tid, f"released ({out})"):
                        enqueue(tid, "refine_exp")
            # x39: tasks whose non-blocking agreement was blocked (backlog / budget / inflight) retry every poll; dropped
            # once they no longer qualify (e.g. the re-refiner is already queued)
            for tid in sorted(list(ex["agreement_pending"]), key=tids.index):
                if maybe_agreement(tid) is None:
                    ex["agreement_pending"].discard(tid)

    def extras_agree(tid: str) -> bool:
        """x39: every finished, all-train-passing extra solver of this task has a first answer equal to s0's (ones that did
        not pass train do not count as dissent); rerefine_skip_agreed also checks this, so the re-refiner runs when a
        non-blocking agreement extra gives a different answer. Caller holds the lock."""
        r0 = results[tid]["first"]
        return all(not train_all(r) or r["attempt_1"] == r0["attempt_1"]
                   for k in extra_kinds if (r := results[tid].get(k)) is not None)

    def uncertain(tid: str) -> bool:
        """s0 and s2 do not both pass all train with identical outputs (offline: nearly all of the re-refiner's gain is on
        such tasks)."""
        r0, r2 = results[tid]["first"], results[tid]["ind"]
        return not (train_all(r0) and train_all(r2) and r0["attempt_1"] == r2["attempt_1"])

    def enqueue(tid: str, kind: str) -> None:
        if kind in FILLERS:
            if tid in state["extra"]["rerefine_queued"]:
                return                                       # x39: a task's re-refiner is queued (or skipped) only once
            state["extra"]["rerefine_queued"].add(tid)
            if cfg.get("rerefine_skip_agreed") and refiner_agrees(results[tid]["first"], results[tid]["ind"],
                                                                  results[tid]["refine_exp"], len(tasks[tid]["test"])) \
                    and extras_agree(tid):
                state["fillers"]["skipped_agreed"].append(episodes[(tid, kind)].name)
                base.log(f"{episodes[(tid, kind)].name} [{kind}]: SKIP s0, s2 and refine_exp agree (rerefine_skip_agreed)")
                return
            sched.push_filler((0 if uncertain(tid) else 1, tids.index(tid)), (tid, kind))
        else:
            sched.push((priority[kind], tids.index(tid), SAMPLE[kind]), (tid, kind))

    def fail_dependents(tid: str, kind: str, why: str) -> None:
        for k in kinds:
            if k in FILLERS:
                continue                 # fillers are not in pending; queued only after their dependencies finish
            if kind in NEEDS[k] and k not in failed[tid] and k not in results[tid]:
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
            if kind in NEEDS[k] and k not in failed[tid] and k not in results[tid]:
                sched.done_one()
                skip(tid, k)

    def worker() -> None:
        while True:
            item = sched.pop()
            if item is None:
                return
            lane, (tid, kind) = item
            filler = lane == "filler"
            extra = lane == "extra"
            later: list[str] = []          # extra_solvers: refine_exp to queue once this episode has wrapped up
            ep = episodes[(tid, kind)]
            dp = None
            try:
                if extra and xs.get("reserve_queued"):
                    with lock:                   # queued reservation: counted as running (budget) once started; SKIP drops it
                        del state["extra"]["queued"][(tid, kind)]
                cap = cfg["task_wall_seconds"]
                if deadline is not None:
                    cap = min(cap, deadline - time.time())
                if extra and t_ref is not None:
                    cap = min(cap, t_ref - time.time())    # extras stop before the refiner cutoff (results still harvested)
                if ctx.cancelled.is_set() or cap <= 0:
                    with lock:
                        if filler:
                            state["fillers"]["not_started"].append(ep.name)
                        else:
                            skip(tid, kind)
                        if extra:
                            base.log(f"{ep.name} [{kind}]: SKIP {'cancelled' if ctx.cancelled.is_set() else 'past the refiner cutoff'}")
                            later += after_extra(tid, ep.name)
                    continue
                if extra:
                    with lock:
                        agreement = ep.name in state["extra"]["agreement"]
                        if agreement and agr_late_gate:
                            # x38 / x39: an agreement ordered after s2 (or refine_exp) reserved nothing while queued and its
                            # admission ignored the backlog; the full gate before starting
                            refused, budget, detail = admission_gate(agr_cfg.get("max_budget"))
                            why = None if refused is None else f"{refused} at start: {detail}"
                        else:
                            budget = extra_budget_now()
                            if agreement and agr_cfg.get("max_budget"):
                                budget = min(budget, agr_cfg["max_budget"])   # the same cap as at admission (admission_gate)
                            why = (f"budget {budget:,} < min_budget {xs['min_budget']:,} at start"
                                   if budget < xs["min_budget"] else None)
                        if why is not None:
                            failed[tid].add(kind)
                            state["extra"]["skipped"].append(ep.name)
                            if agreement and agr_late_gate:
                                note(tid, "agreement", "skipped at start")
                            base.log(f"{ep.name} [{kind}]: SKIP {why}")
                            later += after_extra(tid, ep.name)
                            continue
                        state["extra"]["budget"][(tid, kind)] = budget
                dp = pick_dp(ctx)
                with lock:
                    state["running"][ep.name] = (tid, kind)
                base.log(f"{ep.name} [{kind}]: START{' (filler)' if filler else ''} wall_cap {cap:.0f}s dp {dp}")
                c, hooks = ctx_for(kind, tid)
                gate = None if filler else extension_gate(kind, cap, ep.name)
                result = base.solve(c, ep, tasks[tid], **hooks, dp=dp, wall_cap=cap, extend_gate=gate)
                result = {**result, "tid": tid, "kind": kind, "episode": ep.name}
                with lock:
                    if (tid, kind) in state["extra"]["budget"]:
                        result["extra_budget"] = state["extra"]["budget"][(tid, kind)]
                    if ep.name in state["extra"]["candidates"]:
                        result["candidates_from"] = state["extra"]["candidates"][ep.name]
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
                    if xs is None:
                        nxt = [k for k in kinds if kind in NEEDS[k] and ready(tid, k)]
                    else:
                        if kind == "refine_exp":
                            maybe_tail(tid)          # if opened, the re-refiner is not ready yet: wait for it (after_extra)
                        nxt = [k for k in kinds if k != "refine_exp" and kind in NEEDS[k] and ready(tid, k)]
                        if kind in extra_kinds:
                            nxt += after_extra(tid, ep.name)
                        elif kind in ("first", "ind"):
                            # If one was opened, refine_exp is not ready yet; wait for that line (after_extra); when blocked
                            # with retry_hold_seconds set, hold first. undecided (x38 review): when a flagged ordinary extra
                            # solver finishes before s2, s2 is not done yet at after_extra and no judgment is made; it is
                            # made up here once s2 is done
                            trigger, out = "informed", maybe_informed(tid)
                            if out is None:
                                trigger, out = "agreement", maybe_agreement(tid)
                            if out is None:
                                trigger, out = "undecided", maybe_undecided(tid)
                            if trigger == "agreement" and agr_nonblocking:
                                # x39: for a non-blocking agreement, admitted or not makes no difference: refine_exp is
                                # queued as usual (a blocked one goes into agreement_pending and retries every poll)
                                if refine_check(tid):
                                    nxt.append("refine_exp")
                            elif can_hold(out):
                                hold(tid, trigger, out)
                            elif refine_check(tid):
                                nxt.append("refine_exp")
                score = "" if result["score"] is None else f"score {result['score']:.2f} "
                base.log(f"{ep.name} [{kind}]: DONE {result['stop']} {score}train "
                         f"{sum(v == 'PASS' for v in result['train'])}/{len(result['train'])} "
                         f"answered {sum(a is not None for a in result['attempt_1'])}/{len(result['attempt_1'])} "
                         f"out {result['usage']['proxy_output_tokens']:,} sessions {len(result['sessions'])} "
                         f"{result['seconds']:.0f}s/{result['wall_cap_seconds']:.0f}s dp {dp} "
                         f"closed {len(result['sessions_closed_by_harness'])} "
                         f"[{sum(r['kind'] not in FILLERS for r in state['done'])}/{state['total']}"
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
                    if extra:
                        later += after_extra(tid, ep.name)
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
                for k in later:
                    enqueue(tid, k)
                if filler:
                    sched.filler_done()
                else:
                    sched.done_one()
                if extra:
                    sched.extra_done()

    def watch_deadline() -> None:
        if not ctx.cancelled.wait(max(0.0, deadline - time.time())):
            ctx.deadline_hit.set()
            ctx.cancelled.set()
            with lock:
                running = len(state["running"])
            base.log(f"DEADLINE reached: no new episodes; stopping {running} running episode(s) and harvesting them")

    if deadline is not None:
        threading.Thread(target=watch_deadline, daemon=True, name="deadline").start()

    for tid in tids:
        for k in kinds:
            if (tid, k) not in reused and ready(tid, k):
                enqueue(tid, k)
    stop_hb = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_hb, ctx, state, lock), daemon=True).start()
    if xs is not None:
        threading.Thread(target=watch_extras, args=(stop_hb,), daemon=True, name="extras").start()
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
        "failed_episodes": sorted(f["task"] for f in failures),
        "deadline_hit": ctx.deadline_hit.is_set(),
        **({"fillers": {"done": len(state["fillers"]["done"]), "unusable": sorted(state["fillers"]["unusable"]),
                        "failed": sorted(state["fillers"]["failed"]),
                        "not_started": len(state["fillers"]["not_started"]), "dropped_at_the_gate": len(sched.dropped),
                        "skipped_agreed": len(state["fillers"]["skipped_agreed"]),
                        "never_queued": sum(1 for key in to_run if key[1] in FILLERS) - len(state["fillers"]["done"])
                        - len(state["fillers"]["failed"]) - len(state["fillers"]["not_started"]) - len(sched.dropped)
                        - len(state["fillers"]["skipped_agreed"])}}
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
        **({"extra_solvers": {
            "config": xs, "flagged_sessions": len(state["extra"]["flagged"]), "s2_promoted": len(state["extra"]["promoted"]),
            "triggered": {k: sum(k in v for v in state["extra"]["triggered"].values()) for k in extra_kinds},
            "done": {k: sum(r["kind"] == k for r in done) for k in extra_kinds},
            "skipped_at_start": sorted(state["extra"]["skipped"]),
            "train_pass": {k: sum(r["kind"] == k and train_all(r) for r in done) for k in extra_kinds},
            "output_tokens": {k: sum(r["usage"]["proxy_output_tokens"] or 0 for r in done if r["kind"] == k)
                              for k in extra_kinds},
            "refiners_with_extra_candidate": sum(any(k in extra_kinds for k in v)
                                                 for v in state["extra"]["candidates"].values()),
            "tasks_with_extra": sum(bool(v) for v in state["extra"]["triggered"].values()),
            "backlog_refused": state["extra"]["backlog_refused"],
            "tail": sorted(state["extra"]["tail"]),
            "informed": sorted(state["extra"]["informed"]),
            "agreement": sorted(state["extra"]["agreement"]),
            "undecided": sorted(state["extra"]["undecided"]),
            "backlog_estimates": backlog_est,
            # number of tasks per trigger (ordinary / informed / agreement / tail) and per result (admitted, budget, backlog,
            # queue busy, held, admitted after hold, expired…)
            "admission": {t: {o: len(v) for o, v in sorted(d.items())} for t, d in sorted(state["extra"]["admission"].items())},
            "held_at_the_end": sorted(state["extra"]["held"]),
            "queued_reservations_at_the_end": len(state["extra"]["queued"]),
            # x39
            "nonblocking": sorted(state["extra"]["nonblocking"]),
            "agreement_pending_at_the_end": sorted(state["extra"]["agreement_pending"]),
            "rerefine_queued": len(state["extra"]["rerefine_queued"]),
            "estimator": ({"priors": est_cfg["priors"], "L": est["L"], "m": state["extra"]["m"], "m_info": state["extra"]["m_info"],
                           "rate_full": state["extra"]["rate_full"], "lead": state["extra"]["lead"],
                           "lead_changes": state["extra"]["lead_changes"]} if est is not None else None),
            "monitor_error": state["extra"]["monitor_error"]}} if xs is not None else {}),
        "wall_seconds": round(time.time() - started),
        **({"second_answers": second_answer_stats(tids, results, submitted, ("first", "ind", *extra_kinds))}
           if second_file else {}),
    }
    if scored:
        gts = {tid: [t["output"] for t in task["test"]] for tid, task in tasks.items()}
        summary["score_submission"] = round(100 * sum(
            sum(pair_correct(pair, gt) for pair, gt in zip(submitted[tid], gts[tid], strict=True)) / len(gts[tid])
            for tid in tids) / len(tids), 2)
        summary.update({k: v for k, v in summarize(tasks, results, finals, arms).items() if k not in summary})
    base.atomic_write(run_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    base.log("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "episodes_not_started")}))
    return summary, failures, run_dir


if __name__ == "__main__":
    main()
