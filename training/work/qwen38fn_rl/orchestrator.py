"""Orchestrator for Qwen3.8-FN LoRA RL: synchronous rounds (spec in SPEC.md §4).

    python work/qwen38fn_rl/orchestrator.py --config work/qwen38fn_rl/configs/toy-b8k.json [--check]
        [--abandon-incomplete-round] [--set-aside-judge-groups gNNNNNN[,...]]

Each round r:
  0. gate     STOP file, disk, trainer health, server weight version and idleness
  1. pick     take tasks_per_round tasks from the training list in order, wrapping around at the end (train.shuffle=true
              reshuffles every epoch; no repeats within a round)
  2. held-out runs only when r % heldout_every == 0 (with heldout_at_round0=false, r=0 is skipped and heldout_baseline_file
              is recorded instead): a separate task set, same weight version, not trained on (runs first, while the server
              has no other traffic)
  3. rollout  samples_per_task ``run_arc2.py`` processes run concurrently (each in its own run dir ``rounds/rNNN/sK``); any rc≠0 stops
  4. audit    server idle + ledger fully written → ``python -m arcforge.tito audit``; FAIL stops. The only built-in exception: a
              task killed at the 8k budget leaves one rid that the ledger has and the harness did not record (the audit line
              "rid join incomplete: 0 missing, N extra, 0 window mismatches"); with audit.allow_budget_truncation on, it passes
              only when each run's N ≤ the number of tasks the harness stopped in that run (for arc3 every game counts,
              budget_stops); plus the substrings in audit.whitelist
  5. prepare  ``prepare_rl`` → ``rounds/rNNN/batch`` (any nonzero exit stops)
  6. train    one ``POST /train`` per minibatch
  7. publish  ``POST /publish {weight_version: r+1}``
  8. sync     server idle → ``POST /update_weights_from_disk`` → verify ``/model_info``
  9. record   one row into ``rounds.jsonl`` (authoritative: a restart resumes at the round after the last completed one)

Interruption and resume: each completed stage writes ``rounds/rNNN/state.json``. On restart, rounds already in rounds.jsonl are
skipped; an unfinished round continues from its first unfinished stage (a run dir that exists while the stage is not recorded as
done = the harness stopped halfway: refused, a human decides; ``--abandon-incomplete-round`` renames that round out of the way and
redoes it, but a round whose minibatches the trainer has already consumed may not be moved).

Touches only its own processes: harness, audit and prepare are all children of this process, and stopping signals only those PIDs.
STOP file ``<run_root>/STOP``: stop after the current round finishes. SIGTERM/SIGINT: terminate children, keep state, exit 3.
Standard library only.

``"pipeline": "async"`` (SPEC §8; default ``"sync"`` = the flow above): a rollout thread and a learner thread run in parallel.
  rollout round k: gate (server version ≥ k-1) → [held-out] → samples_per_task harness runs → record the ledger's enqueued_events
                  → hand to the learner
  learner round k: ledger written_events ≥ that value → audit → prepare (--weight-version k-1,k) → /train (trainer at v=k) → /publish k+1
                  → sync: if rollout is running, pause(in_place) → update(flush_cache:false) → continue whatever the outcome → then
                  check; if rollout is waiting at the gate, as in v1
  During held-out, sync waits for it to finish; the server version must be the same before and after held-out; if either thread
  dies the other stops.
  Resume only from a clean point: every started round is in rounds.jsonl, server version = trainer version = number of completed
  rounds; anything else raises (no guessing).
  ``"tail_overlap": {"remaining_episodes": N}`` (may be absent = off): once round k has ≤ N episodes still running (done =
  ``results/<task>.json``) and the gate is open, round k+1 starts; round k's tail finishes in its own thread before going to the
  learner. No early start when round k+1 has a held-out.

``"pipeline": "stream"`` (SPEC §9): no rounds. A dispatcher thread keeps up to max_groups_in_flight groups (one task × group_size
  samples = one group-mode harness process, ``groups/gNNNNNN``) running at once and launches the next as one finishes (tasks follow
  the (seed, epoch) shuffle continuously); finished groups are recorded in groups.jsonl and enter the pool. No new launches while the
  pool's unconsumed live groups ≥ max_pooled_groups (back-pressure).
  learner thread: once the pool has ≥ batch_live_episodes live episodes, take a batch in completion order (dead groups are consumed
  along with it, not counted) → ledger flushed to disk → audit → prepare (--current-version V --max-staleness S) → /train →
  /publish V+1 → always an in_place sync (no flush) → one row in batches.jsonl.
  STOP: no new groups are launched, running ones finish, and every batch that can still be assembled is trained. Disk < min_free_gb:
  the dispatcher pauses launching (waits); < stream.hard_min_free_gb: the learner stops the run. Resume: from a clean point, or from
  the one batch whose step the trainer has already taken (continue it); a group dir not in groups.jsonl = stopped halfway, raise.
  Quarantine (stream.quarantine): run-level invariants still stop the run; per-group anomalies (harness rc≠0, timeout (it exits on
  its own after SIGTERM), group_ready's prepare naming some groups' data as incomplete) quarantine only those groups
  (quarantine.jsonl, groups/quarantined/, groups.jsonl abandoned) and the run continues; when prepare's routing exceeds
  prepare.max_gb the packet is split in half; the run stops only on the Nth occurrence of the same error signature within an hour,
  or when the quarantined share among recent groups is too high (circuit breaker).
  In-loop held-out (stream.heldout_every E, default 0 = off): when the server version reaches a multiple of E (v0 per
  heldout_at_v0) a round opens: one group of heldout_samples samples per task in tasks.heldout (``heldout/vNNNNNN/<task>``), queued
  ahead of new training groups and sharing rollout_slots; results go only to heldout.jsonl (one row per group, one summary row per
  round: pass rate, share that hit the budget, output tokens, version span); failed groups are only recorded as failed and do not
  stop the run.
"""

from __future__ import annotations

import argparse
import ast
import calendar
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from arcforge import arc2_tooluse, arc3_hinge, arc3_note, arc3_official, arc3_rank
from arcforge.rl_storage import DEFAULT_ROUTING_GB, DEFAULT_RESERVE_GB, MAX_D_CROSS_FRACTION, required_free_bytes
from package_lineage import LineageError, check_lineage

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
ROLLOUT_FORMAT = "tito-rl-v1"
#: kind of the harness config for an arc3 single-level episode (work/arc_code/run_arc3_level.py)
ARC3_LEVEL_KIND = "arc3_level"
#: gen-2 single-level episode (opens level K from a gen-2 level-change seed; same run_arc3_level.py):
#: score = 1[level cleared] × min(1, (baseline / actions)²)
ARC3_G2_KIND = "arc3_g2seed"
#: stops with which the harness ends an arc3 single-level episode: the in-flight last turn may already be in the ledger while codex
#: has not recorded it. Same as the arc3 set in arcforge.tito.join.RL_KILL_STOPS (used by prepare_rl's rl_rid_join); join imports
#: numpy via tito.ledger and the orchestrator runs on the system python3, so this is a copy; test_orchestrator checks the two match.
#: hot-reload file (under control/) and the stream keys it may change (Orchestrator.dispatcher_overrides)
DISPATCHER_OVERRIDE_FILE = "dispatcher.json"
DISPATCHER_OVERRIDE_KEYS = frozenset({"max_groups_in_flight", "max_pooled_groups", "group_timeout_seconds", "rollout_slots"})
ARC3_KILL_STOPS = frozenset({"budget", "context", "cleared", "won", "action_cap", "wall", "died", "window"})
#: the same set for arc2 (arcforge.tito.join.RL_KILL_STOPS["arc2"]): budget hit, and the harness killing codex at wall / deadline /
#: cancelled
ARC2_KILL_STOPS = frozenset({"budget", "wall", "deadline", "cancelled"})
#: start of ``arcforge.tito.audit``'s rid-join failure line (the audit's own text, matched verbatim: "rid join incomplete")
RID_JOIN_FAILURE = "rid join 不完整"
#: start of arcforge.tito.audit's split-reason failure line (matched verbatim: "unexpected split reasons"); audit.allow_splits
#: judges by the report's split_reasons, not by this line's text
UNEXPECTED_SPLITS_FAILURE = "預期外的分岔原因"
LOADS_URL = "/v1/loads?include=core"
#: polling interval while waiting for a child process (audit / prepare / harness group) to exit: poll_seconds (5 s in the run
#: config) would round every audit and prepare up to a multiple of 5 s and delay a finished group's entry into the pool by up to
#: 5 s, a cost paid by every group_ready packet and by the last group's tail
CHILD_POLL_SECONDS = 0.5
# verbatim rejection message of the fork's rl/v1 weight_updater (keep_identity + flush_cache while the scheduler is not idle)
SYNC_REFUSED_BUSY = "keep_identity with flush_cache needs an idle scheduler"
SYNC_IDLE_RETRIES = 3
#: prepare_rl --error-json error kind → dedicated exit code (copy of prepare_rl.PREPARE_EXIT_CODES: prepare_rl imports numpy and
#: the orchestrator runs on the system python3; test_orchestrator checks the two match)
PREPARE_EXIT_CODES = {"group_data": 65, "routing_over_max_gb": 66}
#: quarantine log (append-only, one row per group); the dir and .out of a group quarantined at the harness level move into this
#: dir under groups/ (GROUP_DIR does not match it, so a resume does not take it for a half-run group)
QUARANTINE_FILE = "quarantine.jsonl"
QUARANTINE_DIR = "quarantined"
QUARANTINE_STAGES = ("harness", "prepare", "audit")
QUARANTINE_ROW_KEYS = {"utc", "unix", "gid", "group", "task", "stage", "signature", "error", "evidence", "rewards"}
#: since 2026-10-24 each row also carries ``event``: rows written by the same event (one prepare rejection naming several groups,
#: one audit failure) share one id, and the circuit breaker's same_signature_max counts events, not rows (old rows without the key
#: = one event each)
QUARANTINE_OPTIONAL_ROW_KEYS = {"event"}
#: stream (2026-10-24, SPEC §9 "adoption"): after launching a group the dispatcher writes ``groups/gNNNNNN.launch.json``, the state
#: that adopting an in-flight group needs and that used to live only in memory (pid and its /proc starttime, server version at
#: launch, which replicas it was sent to, ledger roster, salt). On resume: process still alive → adopt; dead with complete results
#: → finish it; dead and incomplete → abandon. Handled together with the group dir on completion or quarantine.
LAUNCH_SUFFIX = ".launch.json"
LAUNCH_KEYS = {"gid", "group", "task", "salt_prefix", "session_id", "pid", "proc_starttime", "launched_unix", "launched_utc",
               "server_version_launch", "backends", "ledger_instances"}
#: ``EXIT.json {"rc", "finished_unix"}`` that the harness (run_arc2 / run_arc3_level) writes atomically on exit: an adopted group
#: is not this orchestrator's child and waitpid cannot get its rc, so success / failure is judged from this file; no file = no
#: normal shutdown (SIGKILL, OOM) = failure, not inferred from the manifest
EXIT_FILE = "EXIT.json"
#: ``groups/gNNNNNN.terminated.json {"reason", …}`` that the dispatcher writes atomically **before** sending a group SIGTERM
#: (reason: cancel = the whole run was cancelled, timeout = group_timeout_seconds, abandon = --abandon-incomplete-round on resume).
#: run_arc2 does not die on SIGTERM: episodes stop at the next poll with stop=cancelled, and results / manifest / EXIT.json rc 0 are
#: written as usual; without this marker the resume triage would treat it as a cleanly finished group and train on it (2026-10-24
#: review). Marker seen: timeout → quarantine, cancel / abandon → move aside; still alive → wait for it to exit (SIGKILL after the
#: timeout grace).
TERMINATED_SUFFIX = ".terminated.json"
TERMINATED_REASONS = ("cancel", "timeout", "abandon")
KILL_WAIT_SECONDS = 60          # a SIGKILLed harness pid that is still there this long later stops the run (check ps)
MEDIAN_MIN_GROUPS = 5           # supply_in_flight: completed groups needed before stragglers are left out of the supply
#: SIGTERM / SIGINT while ``<run_root>/DETACH`` exists: in-flight groups are not terminated (left for the next orchestrator to
#: adopt); SIGUSR1 always behaves this way
DETACH_FILE = "DETACH"
#: in-loop held-out for stream (2026-10-25, stream.heldout_every, SPEC §9 "held-out"). One round = one group per task in
#: tasks.heldout at nominal version V (a multiple of heldout_every): ``<run>/heldout/vNNNNNN/<task>`` (+ ``.out``,
#: ``.launch.json``, ``.terminated.json``: the same adoption markers as groups/; not under groups/ and without a gid, so task_of's
#: gid → task mapping is unchanged). Round state lives in heldout_state.json (round opened; launched recorded before each group is
#: dispatched, so a restart neither re-dispatches nor misses one); results go only to heldout.jsonl (one row of kind group per
#: group, one row of kind round per finished round), never into groups.jsonl / pool / batch. routing_release treats held-out groups
#: that have a group row as consumed (their sessions are never trained on)
HELDOUT_DIR = "heldout"
HELDOUT_STATE_FILE = "heldout_state.json"
HELDOUT_FILE = "heldout.jsonl"
HELDOUT_ROUND_DIR = re.compile(r"v\d{6}")
HELDOUT_ROUND_KEYS = {"version", "launch_version", "opened_utc", "opened_unix", "session_id", "samples", "tasks", "launched"}
HELDOUT_LAUNCH_KEYS = {"version", "task", "group", "samples", "salt_prefix", "session_id", "pid", "proc_starttime",
                       "launched_unix", "launched_utc", "server_version_launch", "backends"}
#: episodes cancelled together with the rest, in the harness's failures.jsonl (run_arc3_level on SIGTERM records
#: ``<episode>: cancelled`` for each episode)
CANCELLED_FAILURE = re.compile(r": cancelled['\"]\)$")
#: error_signature: paths, UUIDs (codex thread ids), hex ids (at least 8 chars, both letters and digits) and numbers are replaced
#: by placeholders; maximum signature length in chars
SIGNATURE_PATH = re.compile(r"(?<![\w<>])(?:/[\w.@%+=~-]+)+/?")
SIGNATURE_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SIGNATURE_HEX = re.compile(r"(?<![A-Za-z0-9])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{8,}(?![A-Za-z0-9])")
SIGNATURE_NUMBER = re.compile(r"\d+(?:\.\d+)?")
SIGNATURE_MAX_CHARS = 300
# MAX_D_CROSS_FRACTION (arcforge.rl_storage): prepare_rl's --max-d-cross-fraction default (cap on the share of D_cross episodes,
# SPEC §5); config prepare.max_d_cross_fraction is passed to prepare_rl as is, and stream.group_ready applies the same value to the
# whole optimizer step
#: prepare_rl's MAX_M_DROP_FRACTION (default of --max-routing-m-fraction; cap on the share of episodes dropped by routing M, SPEC §5;
#: the orchestrator does not import prepare_rl, test_orchestrator compares against its source); config
#: prepare.max_routing_m_fraction is passed to prepare_rl as is, and stream.group_ready applies the same value to the whole
#: optimizer step
MAX_ROUTING_M_FRACTION = 0.005


class OrchestratorError(RuntimeError):
    """Anything that means this run should not continue. The message must say where it is stuck without anyone reading the logs."""


class Cancelled(Exception):
    """SIGTERM / SIGINT received (or the flag set by tests)."""


class Aborted(Exception):
    """async: the other thread has already died, so this one takes no further step (the real cause is the other thread's exception)."""


class PrepareRejected(OrchestratorError):
    """prepare_rl exited with one of PREPARE_EXIT_CODES: ``kind`` (group_data / routing_over_max_gb), the ``doc`` it wrote, that
    file's ``path``. stream.group_ready's materializer handles it (quarantine those groups / split the packet, gr_materialize);
    elsewhere it stops the run like any OrchestratorError."""

    def __init__(self, message: str, kind: str, doc: dict, path: Path):
        super().__init__(message)
        self.kind, self.doc, self.path = kind, doc, path


class AuditFailed(OrchestratorError):
    """audit's blocking failure lines (audit_with_status). For failures attributable to a single manifest (``bisectable``: rid join,
    a task / game with no segment, a manifest whose ledger instance was not loaded) stream.group_ready's materializer bisects the
    packet to find the offending group and quarantines it (gr_materialize); ledger-level failures (lossy, writer_error, invariants,
    media blob, continuation contract) and failures elsewhere stop the run like any OrchestratorError."""

    def __init__(self, message: str, blocking: list[str], packet_dir: Path, report_path: Path):
        super().__init__(message)
        self.blocking, self.packet_dir, self.report_path = blocking, packet_dir, report_path


#: the other audit failure line "attributable to a single manifest" (arcforge.tito.audit's string, matched verbatim: "N tasks /
#: games have no segment at all"; rid join is RID_JOIN_FAILURE; all others are ledger-level)
NO_SEGMENT_FAILURE = re.compile(r"^\d+ 題／局沒有任何段")


#: ledger invariants that always attach to a single segment (arcforge.tito.ledger's finish_segment fails that segment): with
#: --manifest-threads-only the audit loads only this packet's threads, so such a failure can only be on one group's own segment and
#: bisection finds the offending group. v5 2026-10-26 g000053: a request the client had abandoned but the server ran for 728 s
#: finished late and opened a concurrent segment; the next request attached in its middle (late split), and the media of the
#: attached image was recorded on the old segment → media_coverage
SEGMENT_INVARIANTS = frozenset({"media_coverage"})
# arcforge.tito.audit's invariant line, matched verbatim: "N invariant violations: {code: n, ...}"
INVARIANT_FAILURE = re.compile(r"^不變量違反 \d+ 處：(\{.*\})$")


def segment_invariants_only(line: str) -> bool:
    """audit's "N invariant violations: {code: n, ...}" line, with every kind in SEGMENT_INVARIANTS."""
    m = INVARIANT_FAILURE.match(line)
    if m is None:
        return False
    kinds = ast.literal_eval(m.group(1))
    if not (isinstance(kinds, dict) and kinds):
        raise OrchestratorError(f"audit invariant line {line!r}: not a non-empty dict of codes")
    return set(kinds) <= SEGMENT_INVARIANTS


def audit_bisectable(blocking: list[str], manifests: list[Path], threads_only: bool = False) -> bool:
    """Bisection to find the offending group is possible only when every blocking line is a manifest-level failure (rid join, no
    segment, starts with a manifest path in this packet, or only the single-segment invariants SEGMENT_INVARIANTS). With
    ``threads_only`` (audit run with --manifest-threads-only: split_reasons counts only this packet's threads), unexpected split
    reasons qualify too: on 2026-10-30 in arc2-hq741b an orphan request left by one codex idle timeout was recorded as open:evicted
    and stopped the whole run."""
    if not blocking:
        return False
    paths = tuple(str(m) for m in manifests)
    return all(line.startswith(RID_JOIN_FAILURE) or NO_SEGMENT_FAILURE.match(line) or line.startswith(paths)
               or segment_invariants_only(line) or (threads_only and line.startswith(UNEXPECTED_SPLITS_FAILURE))
               for line in blocking)


def proc_stat(pid: int) -> tuple[str, int] | None:
    """(state, starttime) from /proc/<pid>/stat: the state letter in field 3 and field 22 (clock ticks since boot; differs when the
    pid is reused). Returns None when there is no such process."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text[text.rindex(")") + 2:].split()     # comm is in parentheses and may contain spaces: count from after the last )
    return fields[0], int(fields[19])                # field 3 = 1st after the parenthesis; field 22 = 20th after it


def session_pids(sid: int) -> list[int]:
    """Processes in /proc whose session id (field 6 of /proc/<pid>/stat) is ``sid``. A sid is not reused while the session has
    members."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(text[text.rindex(")") + 2:].split()[3]) == sid:   # after the parenthesis: state ppid pgrp session
            pids.append(int(entry.name))
    return pids


def proc_starttime(pid: int) -> int | None:
    stat = proc_stat(pid)
    return None if stat is None else stat[1]


def process_alive(pid: int, starttime: int | None) -> bool:
    """A process with the same pid and the same starttime is still running. A zombie counts as dead: it has exited and its exit
    code is on disk (EXIT.json); an adopted group is not our child and is reaped by its parent (systemd under KillMode=process).
    starttime None (already gone at launch) = dead."""
    if starttime is None:
        return False
    stat = proc_stat(pid)
    return stat is not None and stat[1] == starttime and stat[0] != "Z"


def killpg_quietly(pid: int, sig: int) -> bool:
    """Signal the harness's process group (the harness is started with start_new_session, pgid = its pid; run_arm / gateway / proxy
    and the outer bwrap are all in it. codex inside the sandbox is not in this pgid because of bwrap ``--new-session``, but bwrap
    ``--die-with-parent --unshare-pid`` makes the whole pid namespace go when the outer bwrap dies); returns False when the group is
    already gone."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


class AdoptedProc:
    """An adopted harness group process (pid / starttime from launch.json), with the same poll / terminate / kill / wait as
    subprocess.Popen. Not our child: alive = /proc still has the pid with the same starttime; the exit code comes from
    ``<run_dir>/EXIT.json`` (the harness writes {"rc", …} atomically); without EXIT.json, tito_manifest.json present and no
    failures.jsonl counts as 0, anything else as 1 (SIGKILL / OOM are failures)."""

    def __init__(self, pid: int, starttime: int, run_dir: Path) -> None:
        self.pid, self.starttime, self.run_dir = pid, starttime, run_dir
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if process_alive(self.pid, self.starttime):
            return None
        self.returncode = exit_code_on_disk(self.run_dir)
        return self.returncode

    def terminate(self) -> None:
        if self.poll() is None:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def kill(self) -> None:
        if self.poll() is None:
            killpg_quietly(self.pid, signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(f"pid {self.pid}", timeout)
            time.sleep(0.05)
        return self.returncode


def exit_record(run_dir: Path) -> dict | None:
    """The harness's ``EXIT.json`` ({"rc": int, "finished_unix": …}); missing = None."""
    path = run_dir / EXIT_FILE
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    rc = doc["rc"]
    if not isinstance(rc, int) or isinstance(rc, bool):
        raise OrchestratorError(f"{path}: rc {rc!r} is not an int")
    return doc


def exit_code_on_disk(run_dir: Path) -> int:
    """Exit code of a harness group that is already gone: EXIT.json's rc; no EXIT.json = no normal shutdown (SIGKILL / OOM / old
    harness) = 1. Not inferred from the manifest: run_arc2 and run_arc3_level write EXIT.json on every exit path, so a missing one is a
    failure."""
    doc = exit_record(run_dir)
    return 1 if doc is None else doc["rc"]


def log_line(message: str) -> str:
    return f"[{time.strftime('%H:%M:%S')}] {message}"


# --------------------------------------------------------------------------- config

TOP_KEYS = {"schema_version", "run_root", "seed", "rounds", "tokenizer_dir", "poll_seconds", "min_free_gb",
            "disk_guard_paths", "legacy_sync", "rollout", "trainer", "harness", "tasks", "audit", "prepare"}
#: optional top-level keys. ``pipeline``: "sync" (default, v1 synchronous rounds) or "async" (SPEC §8)
OPTIONAL_TOP_KEYS = {"pipeline", "retention", "tail_overlap", "stream", "note_judge"}
#: ``note_judge`` (required with prepare.arc3_reward note, forbidden otherwise; RL v8,
#: docs/plans/arc3_rl_v8_window_note_judge_20261029.md §4): when a group finishes with games that have a stop window and changed
#: window at the window-change point (arcforge.arc3_note.judged_episode), write judge_input.json in the group dir and run
#: work/qwen38fn_rl/arc3_note_judge.py with ``python`` (``model`` / ``effort`` of ``backend``, seed = gid) to write judge.json; the
#: group enters the pool only after that. At most ``max_concurrent`` judge subprocesses at once (each its own process group: on
#: timeout / stop the whole group is signalled, so codex exec subprocesses go too), each for at most ``timeout_seconds``; each
#: backend request of the judge takes at most ``call_timeout_seconds`` (--call-timeout), and timeout_seconds must be ≥
#: arcforge.arc3_note.judge_attempt_seconds(backend, call_timeout_seconds) (the longest time if every pass × pass retry × request
#: times out). On failure wait JUDGE_BACKOFF_SECONDS and retry; if all JUDGE_ATTEMPTS fail: **stop the run** (OrchestratorError, no
#: quarantine, no default score); this group's harness has finished without a judge.json, and on restart stream_start_check hands
#: it back to the dispatcher for re-judging.
#: ``backend``: api = Anthropic API (``python`` is a separate venv with the anthropic SDK; ``env_file`` = the file
#: launch_orchestrator.sh sources before exec'ing the orchestrator, holding JUDGE_API_KEY_ENV; only its presence is checked here,
#: it is not read); codex = ``codex_bin`` exec (the binary pinned with readlink -f at launch; ``codex_version`` is its ``--version``
#: at that time and must match when the stream starts), with the default CODEX_HOME (~/.codex, the user's own ChatGPT login; no
#: CODEX_HOME override, no copying or linking auth.json, since a token refresh replacing the file would desync the two logins) plus
#: ``--ignore-user-config`` (does not read ~/.codex/config.toml); ~/.codex/AGENTS.md is not covered by that flag and codex may
#: still read it. No key, no env_file.
#: ``ensemble`` (RL v8b, docs/plans/arc3_rl_v8b_judge_ensemble_20261101.md): Claude Code (``claude_bin`` -p, the user's
#: subscription login; ANTHROPIC_API_KEY must not be in the environment, or claude switches to API billing) + codex (as above),
#: each judging once per round with the v8b prompt (answer key points + ranking with flexible quotas); judge.json's ``version`` is
#: v8b and its ``mapping`` must equal ``mapping`` here (arcforge.arc3_note.V8B_MAPPINGS); model / effort split into claude_* /
#: codex_*, no top-level model / effort.
#: ``claude-code`` (v8b as finalized, after the user reviewed the test on 2026-11-01): like ensemble but Claude Code only, once per
#: group, with only claude_* and mapping.
#: All required, no defaults.
NOTE_JUDGE_KEYS = {"backend", "python", "max_concurrent", "timeout_seconds", "call_timeout_seconds"}
NOTE_JUDGE_BACKENDS = ("api", "codex", "ensemble", "claude-code")
#: efforts each backend accepts (arc3_note_judge.API_EFFORTS / CODEX_EFFORTS / CLAUDE_EFFORTS) and its extra keys
NOTE_JUDGE_EFFORTS = {"api": ("low", "medium", "high", "xhigh", "max"), "codex": ("minimal", "low", "medium", "high", "xhigh"),
                      "claude": ("low", "medium", "high", "xhigh", "max")}
NOTE_JUDGE_BACKEND_KEYS = {"api": {"model", "effort", "env_file"}, "codex": {"model", "effort", "codex_bin", "codex_version"},
                           "ensemble": {"mapping", "codex_bin", "codex_version", "codex_model", "codex_effort", "claude_bin",
                                        "claude_version", "claude_model", "claude_effort"},
                           "claude-code": {"mapping", "claude_bin", "claude_version", "claude_model", "claude_effort"}}
#: v8b backends (judge.json has version v8b and mapping; config has claude_*)
V8B_JUDGE_BACKENDS = ("ensemble", "claude-code")
NOTE_JUDGE_SCRIPT = ROOT / "work" / "qwen38fn_rl" / "arc3_note_judge.py"
JUDGE_INPUT_FILE = "judge_input.json"
JUDGE_FILE = "judge.json"
JUDGE_ATTEMPTS = 4
#: wait after the Nth failure before attempt N+1 (judge_stop interrupts it); length = JUDGE_ATTEMPTS - 1
JUDGE_BACKOFF_SECONDS = (60.0, 300.0, 900.0)
#: the user, 2026-10-29: the judge uses only the second key
JUDGE_API_KEY_ENV = "ANTHROPIC_API_KEY_2"
#: ``stream`` (required with pipeline stream, forbidden with other pipelines; SPEC §9). max_pooled_groups counts only live groups
#: (dead groups do not block dispatch, otherwise a pool full of dead groups would never assemble a batch); eval_keep_every E: merged
#: weights whose version is a multiple of E are exempt from retention deletion (0 = off);
#: disk: free < min_free_gb → the dispatcher pauses launching (waits), < hard_min_free_gb (≤ min_free_gb) → the learner stops the run;
#: end_margin (≥ 1.0): while winding down, stop launching once the live episodes at hand (pool + running × recent live share) ≥
#: end_margin × what the remaining batches need;
#: prefetch_prepare (may be absent = false): while batch b trains, take batch b+1's groups and run ledger / audit / prepare (version
#: window of b+1; the pool is taken in completion order, so the batch composition is the same as without prefetch, only earlier)
#: group_ready (may be absent = false; mutually exclusive with prefetch_prepare, prepare.minibatches must be 1): each group is
#: reserved for the current batch as soon as it finishes, in completion order (same composition as take_batch), ledger / audit /
#: prepare run packet by packet, trainer /begin → /accumulate per packet → one /step when full (units = number of reserved groups
#: or episodes); the routing M threshold (prepare.max_routing_m_fraction, default MAX_ROUTING_M_FRACTION), the 0.5% unrecorded
#: version threshold and the D_cross threshold (prepare.max_d_cross_fraction, default MAX_D_CROSS_FRACTION) are judged over the
#: whole batch (stream_learner_group_ready)
#: rollout_slots (may be absent = old rule: launch while running groups < max_groups_in_flight): global number of rollout slots x
#: (in episodes). With it, the next group launches when running episodes (dispatched, results not yet written) + group_size ≤ x,
#: without waiting for a whole group to finish; max_groups_in_flight then only caps the number of harness processes. Requires
#: rollout.replica_urls and x a multiple of the replica count: each sample goes to the replica with the fewest running episodes at
#: that moment (ties to the lower index), so no replica exceeds x / replica count
#: ramp_seconds (may be absent = 0 = no ramp): after each dispatcher start (resume included), the cap on running episodes rises
#: linearly from group_size to min(rollout_slots, max_groups_in_flight × group_size), reaching it after this many seconds
#: (ramp_allowed), which staggers game starts so KV does not peak all at once
#: retire_tasks_file (optional): a file with one train task id per line; at stream start (resume included) these tasks retire: gids
#: that later map to them are skipped (skip_retired, the gid→task mapping is unchanged), while in-flight and pooled groups of the
#: same task run to completion and enter batches as usual. The list must be a subset of the train list, without duplicates, and may
#: not retire every task. Changing the list needs an orchestrator restart (v5 2026-10-26: dropped the first set of seeds with all
#: 24 games at 0)
#: quarantine (may be absent when reading a config, so tools such as heldout_offline can read old runs' configs; required when
#: running stream, quarantine_cfg): a per-group anomaly (harness rc≠0 / timeout, a per-group data problem reported by prepare)
#: quarantines only that group and the run continues, and a circuit breaker catches systematic errors: QUARANTINE_KEYS, see
#: validate_quarantine and quarantine_breaker
#: STREAM_HELDOUT_KEYS (2026-10-25, in-loop held-out, SPEC §9 "held-out"; all optional): heldout_every E (optimizer version; 0 = off,
#: the default: a config without these three keys behaves exactly as before); with E > 0, heldout_samples N (one group of N samples
#: per held-out task, required, ≥ 2: the group-mode harness needs at least 2 samples) and heldout_at_v0 (default true: also evaluate
#: v0). A round opens when the server version reaches a multiple of E: one group per task in tasks.heldout, dispatched ahead of new
#: training groups and sharing rollout_slots with training (so rollout_slots is required); results go only to heldout.jsonl
STREAM_KEYS = {"group_size", "max_groups_in_flight", "max_pooled_groups", "batch_live_episodes", "max_staleness",
               "group_timeout_seconds", "eval_keep_every", "hard_min_free_gb", "end_margin"}
STREAM_HELDOUT_KEYS = ("heldout_every", "heldout_samples", "heldout_at_v0")
STREAM_INT_MIN = {"group_size": 2, "max_groups_in_flight": 1, "max_pooled_groups": 1, "batch_live_episodes": 1,
                  "max_staleness": 0, "eval_keep_every": 0}
#: section keys pipeline stream does not use (and forbids): (section, key, reason)
STREAM_DROPPED_KEYS = (
    ("tasks", {"heldout_samples", "heldout_every", "heldout_at_round0", "heldout_baseline_file"},
     "the stream loop's held-out is configured in the stream section (stream.heldout_every / heldout_samples / "
     "heldout_at_v0, SPEC section 9); or evaluate the stream.eval_keep_every checkpoints offline"),
    ("tasks", {"samples_per_task", "tasks_per_round"},
     "a stream group is one task x stream.group_size samples; a batch is stream.batch_live_episodes live episodes"),
    ("harness", {"round_timeout_seconds"}, "each group process has stream.group_timeout_seconds"),
)
#: circuit breaker of stream.quarantine: stop the run on the same_signature_max-th quarantine with the same error signature within
#: same_signature_seconds; also stop when the quarantined share among the last fraction_window_groups finished groups (completed
#: ones and ones quarantined at the harness level, in finishing order) > max_fraction, counted only once there are at least
#: fraction_min_groups of them
QUARANTINE_KEYS = {"same_signature_max", "same_signature_seconds", "max_fraction", "fraction_window_groups",
                   "fraction_min_groups"}
GROUP_DIR = re.compile(r"g\d{6}")
BATCH_DIR = re.compile(r"b\d{6}")
#: ``retention`` (may be absent = delete nothing): ``keep_merged`` keeps only the latest few merged weights (≥1),
#: ``drop_batch_payload`` deletes a batch's large files after training
RETENTION_KEYS = {"keep_merged", "drop_batch_payload"}
#: ``tail_overlap`` (async only; may be absent = the next round waits until every harness run of this round has ended):
#: ``remaining_episodes``: the next round may start once this round has ≤ that many episodes still running (1 ≤ N < episodes per round)
TAIL_OVERLAP_KEYS = {"remaining_episodes"}
#: per-position binary arrays under tito-rl-v1's ``train/`` (index.jsonl, manifest, report and media are kept)
BATCH_PAYLOAD_SUFFIXES = (".i16", ".i32", ".f32", ".u8")
MERGED_DIR = re.compile(r"v\d{6}")
PIPELINES = ("sync", "async", "stream")
#: optional section keys. audit.allow_splits: {split reason: max segments}; unexpected splits pass when they have only these reasons
#: and stay within the caps; the segment count is the audit report's split_reasons: with --manifest-threads-only (when the command is
#: arcforge.tito) it counts only the threads of this audit's manifests, otherwise the cumulative count of the whole server instance
#: (e.g. text_differs: on a codex resend the rendering of the previous turn's start differs from the sampled tokens and the server
#: splits the segment; each segment is consistent on its own and prepare accepts both)
#: prepare.arc3_reward (prepare_rl --arc3-reward): score (default = no such key), eff (in-group relative level clear + efficiency) or
#: c0 (smooth efficiency term centered among level-clearing games); eff / c0 → its parameter keys (= prepare_rl.ARC3_MODE_PARAMS
#: with an arc3_ prefix, all required, each only with its own mode)
#: official (piecewise-linear official single-level score, arcforge.arc3_official) and rank (score times an in-group efficiency-rank
#: discount, arcforge.arc3_rank) are shaped R as well, but without efficiency weight keys (group_live handles them separately); hinge
#: (RL v7, arcforge.arc3_hinge: score plus a per-level frozen-threshold cost centered among level-clearing games) is likewise handled
#: separately, and its arc3_hinge_refs is a path string
#: note (RL v8, arcforge.arc3_note: the level-clear score of the first window, or the score the judge gives the note) has no
#: parameter keys; the judge's settings are in the top-level note_judge
ARC3_REWARDS = ("score", "eff", "c0", "official", "rank", "hinge", "note")
ARC3_MODE_KEYS = {"eff": ("arc3_beta_actions", "arc3_beta_tokens", "arc3_w_start", "arc3_eff_clip"),
                  "c0": ("arc3_lambda_actions", "arc3_kappa_actions", "arc3_lambda_tokens", "arc3_kappa_tokens",
                         "arc3_w_start", "arc3_success_floor"),
                  "official": tuple("arc3_" + k for k in arc3_official.PARAMS),
                  "rank": tuple("arc3_" + k for k in arc3_rank.PARAMS),
                  "hinge": tuple("arc3_" + k for k in arc3_hinge.PARAMS),
                  "note": ()}
#: log of stream.retire_full_groups (one row per retired task, append-only; recomputed from groups.jsonl on resume, this file is
#: only a human-readable record)
RETIRED_FILE = "retired.jsonl"
#: arc2 tool-use terms (prepare_rl --tool-*, arcforge.arc2_tooluse): all four together
TOOL_KEYS = tuple("tool_" + k for k in arc2_tooluse.PARAMS)
ARC3_KEYS = tuple(dict.fromkeys(k for keys in ARC3_MODE_KEYS.values() for k in keys))
#: weight keys of the action and token terms of eff / c0 (group_live: a term with weight 0 does not count as signal)
ARC3_WEIGHT_KEYS = {"eff": ("arc3_beta_actions", "arc3_beta_tokens"), "c0": ("arc3_lambda_actions", "arc3_lambda_tokens")}
#: prepare.algo (prepare_rl --algo, SPEC §1) → its parameter keys (= work/qwen38fn_lora/rl_loss.ALGOS, checked by test_orchestrator);
#: no algo key = bpo (configs from before mis was added). All parameters required, each only with its own algo
PREPARE_ALGOS = {"bpo": ("eps", "cap", "eps_low", "eps_high"), "mis": ("mis_low", "mis_high")}
#: prepare.wrapup_loss (prepare_rl --wrapup-loss, = prepare_rl.WRAPUP_LOSS_MODES, checked by test_orchestrator): no such key = train
WRAPUP_LOSS_MODES = ("train", "mask")
SECTION_OPTIONAL_KEYS = {"audit": frozenset({"allow_splits"}),
                         "prepare": frozenset({"max_gb", "reserve_gb", "adv_normalization", "loss_reduction",
                                               "allow_compaction", "wrapup_loss",
                                               "efficiency_beta", "efficiency_budget", "efficiency_guard",
                                               *TOOL_KEYS, "max_d_cross_fraction", "max_routing_m_fraction", "arc3_reward",
                                               *ARC3_KEYS, "algo",
                                               "top_p", "top_k",
                                               *(k for keys in PREPARE_ALGOS.values() for k in keys)}),
                         "rollout": frozenset({"replica_urls", "initial_weights_source", "trainer_base_path"})}
SECTION_KEYS = {
    "rollout": {"base_url", "served_name", "idle_timeout_seconds", "ledger_drain_timeout_seconds",
                "sync_timeout_seconds"},
    "trainer": {"endpoint_file", "train_timeout_seconds", "publish_timeout_seconds", "save_every_rounds"},
    "harness": {"script", "agent_python", "driver_bin", "config", "launch_stagger_seconds",
                "round_timeout_seconds", "terminate_grace_seconds"},
    "tasks": {"train", "heldout", "samples_per_task", "tasks_per_round", "heldout_samples", "heldout_every",
              "heldout_at_round0", "heldout_baseline_file"},
    "audit": {"command", "whitelist", "allow_budget_truncation", "per_run", "timeout_seconds"},
    "prepare": {"command", "timeout_seconds", "temperature", "minibatches", "world", "bin_tokens",
                "weight_version_recorded"},
}
TASK_SET_KEYS = {"train": {"dir", "ids_file", "shuffle"}, "heldout": {"dir", "ids_file"}}
POSITIVE = {
    "": ("rounds", "poll_seconds", "min_free_gb"),
    "rollout": ("idle_timeout_seconds", "ledger_drain_timeout_seconds", "sync_timeout_seconds"),
    "trainer": ("train_timeout_seconds", "publish_timeout_seconds"),
    "harness": ("round_timeout_seconds", "terminate_grace_seconds"),
    "tasks": ("samples_per_task", "tasks_per_round", "heldout_samples", "heldout_every"),
    "audit": ("timeout_seconds",),
    "prepare": ("timeout_seconds", "temperature", "minibatches", "world", "bin_tokens"),
}


def _expand(value, where: str):
    """Expand ``$VAR`` / ``${VAR}`` in a string; an unexpandable one (variable not set) raises instead of leaving the ``$``."""
    if isinstance(value, str):
        out = os.path.expandvars(value)
        if "$" in out:
            raise SystemExit(f"config {where}: unresolved environment variable in {value!r} (source env/env.sh first)")
        return out
    if isinstance(value, list):
        return [_expand(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: _expand(v, f"{where}.{k}") for k, v in value.items()}
    return value


def _keys(obj, want: set[str], where: str, optional: frozenset[str] | set[str] = frozenset()) -> None:
    if not isinstance(obj, dict):
        raise SystemExit(f"config {where}: must be an object")
    missing, unknown = want - set(obj), set(obj) - want - optional
    if missing or unknown:
        raise SystemExit(f"config {where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def load_config(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    whitelist = raw.get("audit", {}).get("whitelist")        # patterns are matched literally, never expanded
    cfg = _expand(raw, "")
    if whitelist is not None:
        cfg["audit"]["whitelist"] = whitelist
    _keys(cfg, TOP_KEYS, "", OPTIONAL_TOP_KEYS)
    cfg.setdefault("pipeline", "sync")
    if "retention" in cfg:
        _keys(cfg["retention"], RETENTION_KEYS, "retention", {"keep_eval"})
        keep = cfg["retention"]["keep_merged"]
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
            raise SystemExit(f"config retention.keep_merged must be an int >= 1, not {keep!r}")
        if not isinstance(cfg["retention"]["drop_batch_payload"], bool):
            raise SystemExit("config retention.drop_batch_payload must be a bool")
        if "keep_eval" in cfg["retention"]:
            count = cfg["retention"]["keep_eval"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise SystemExit("config retention.keep_eval must be an int >= 1 (latest eligible eval exports)")
    pipeline = cfg["pipeline"]
    if pipeline not in PIPELINES:
        raise SystemExit(f"config pipeline must be one of {PIPELINES}, not {pipeline!r}")
    stream = pipeline == "stream"
    if pipeline in ("async", "stream") and cfg["legacy_sync"] is not False:
        raise SystemExit(f"config pipeline {pipeline} needs legacy_sync false: a mid-flight sync cannot rename the model "
                         "under running harness processes")
    if pipeline in ("async", "stream") and cfg["prepare"]["weight_version_recorded"] is not True:
        why = "the version set is the gate's proof" if pipeline == "async" else "the staleness window reads the recorded versions"
        raise SystemExit(f"config pipeline {pipeline} needs prepare.weight_version_recorded true: {why}")
    if ("stream" in cfg) != stream:
        raise SystemExit("config stream section goes with pipeline stream: pipeline stream needs it, the others must not have it")
    if stream:
        validate_stream_section(cfg["stream"])
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise SystemExit(f"config schema_version {cfg['schema_version']} != {SCHEMA_VERSION}")
    dropped: dict[str, set[str]] = {}
    for name, keys, why in STREAM_DROPPED_KEYS if stream else ():
        present = sorted(keys & set(cfg[name])) if isinstance(cfg[name], dict) else []
        if present:
            raise SystemExit(f"config pipeline stream: remove {name}.{present}: {why}")
        dropped.setdefault(name, set()).update(keys)
    for name, want in SECTION_KEYS.items():
        _keys(cfg[name], want - dropped.get(name, set()), name, SECTION_OPTIONAL_KEYS.get(name, frozenset()))
    allow_splits = cfg["audit"].setdefault("allow_splits", {})
    for key, choices in (("adv_normalization", ("std", "none")), ("loss_reduction", ("sequence", "prompt"))):
        if key in cfg["prepare"] and cfg["prepare"][key] not in choices:
            raise SystemExit(f"config prepare.{key} must be one of {choices} (prepare_rl --{key.replace('_', '-')})")
    for key in ("max_d_cross_fraction", "max_routing_m_fraction"):
        if key in cfg["prepare"]:
            value = cfg["prepare"][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise SystemExit(f"config prepare.{key} must be a number in [0, 1], not {value!r}")
    eff_keys = {"efficiency_beta", "efficiency_budget", "efficiency_guard"} & set(cfg["prepare"])
    if eff_keys:
        p = cfg["prepare"]
        if eff_keys != {"efficiency_beta", "efficiency_budget", "efficiency_guard"}:
            raise SystemExit(f"config prepare: efficiency_beta/_budget/_guard go together (have {sorted(eff_keys)})")
        if not (isinstance(p["efficiency_beta"], (int, float)) and p["efficiency_beta"] > 0
                and isinstance(p["efficiency_budget"], int) and p["efficiency_budget"] > 0
                and isinstance(p["efficiency_guard"], (int, float)) and 0 < p["efficiency_guard"] < 1):
            raise SystemExit("config prepare: efficiency_beta > 0, efficiency_budget a positive int, efficiency_guard in (0, 1)")
        if p.get("adv_normalization") != "none":
            raise SystemExit("config prepare: the efficiency term needs adv_normalization none")
    validate_arc3_reward(cfg["prepare"])
    if (arc3_mode(cfg["prepare"]) == "note") != ("note_judge" in cfg):
        raise SystemExit("config note_judge goes with prepare.arc3_reward note: note needs it, the other modes must not have it")
    if "note_judge" in cfg:
        if not stream:
            raise SystemExit("config prepare.arc3_reward note needs pipeline stream (the judge runs when a group completes)")
        validate_note_judge(cfg["note_judge"])
    validate_tool_reward(cfg["prepare"])
    if not isinstance(cfg["prepare"].get("allow_compaction", False), bool):
        raise SystemExit(f"config prepare.allow_compaction must be true/false, not {cfg['prepare']['allow_compaction']!r}")
    if cfg["prepare"].get("wrapup_loss", "train") not in WRAPUP_LOSS_MODES:
        raise SystemExit(f"config prepare.wrapup_loss must be one of {WRAPUP_LOSS_MODES}, not "
                         f"{cfg['prepare']['wrapup_loss']!r}")
    if (((arc3_mode(cfg["prepare"]) == "rank" and cfg["prepare"]["arc3_rank_warmup_versions"] > 0)
         or arc3_mode(cfg["prepare"]) == "hinge")
            and not (cfg.get("stream") and cfg["stream"].get("group_ready", False))):
        raise SystemExit("config prepare.arc3_rank_warmup_versions > 0 and arc3_reward hinge need stream.group_ready (only "
                         "its packets know each group's launch version)")
    if cfg.get("stream") and cfg["stream"].get("retire_full_groups", False) and arc3_mode(cfg["prepare"]) != "official":
        raise SystemExit("config stream.retire_full_groups needs prepare.arc3_reward official (a task retires when one of "
                         "its groups has R = 1 on every episode)")
    validate_algo(cfg["prepare"])
    validate_sampling(cfg["prepare"])
    prepare_required_bytes = required_free_bytes(cfg["prepare"].get("max_gb", DEFAULT_ROUTING_GB),
                                                cfg["prepare"].get("reserve_gb", DEFAULT_RESERVE_GB))
    if not (isinstance(allow_splits, dict) and all(isinstance(k, str) and k and isinstance(v, int)
                                                   and not isinstance(v, bool) and v >= 1 for k, v in allow_splits.items())):
        raise SystemExit("config audit.allow_splits must map split reasons to a positive int (the most segments over the server instance)")
    for name, want in TASK_SET_KEYS.items():
        _keys(cfg["tasks"][name], want, f"tasks.{name}")
    for section, names in POSITIVE.items():
        scope = cfg if section == "" else cfg[section]
        for key in names:
            if key in dropped.get(section, set()):
                continue
            value = scope[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise SystemExit(f"config {section + '.' if section else ''}{key} must be positive, not {value!r}")
    if stream and not cfg["stream"]["hard_min_free_gb"] <= cfg["min_free_gb"]:
        raise SystemExit(f"config stream.hard_min_free_gb {cfg['stream']['hard_min_free_gb']} must be <= min_free_gb "
                         f"{cfg['min_free_gb']}: below min_free_gb the dispatcher waits, below the hard floor the learner stops")
    if stream and cfg["min_free_gb"] * 2 ** 30 < prepare_required_bytes:
        raise SystemExit("config min_free_gb is in GiB and must cover prepare.max_gb + prepare.reserve_gb "
                         f"(decimal GB): required_free_bytes={prepare_required_bytes}")
    for key in ("seed",):
        if not isinstance(cfg[key], int):
            raise SystemExit(f"config {key} must be an int")
    for key, section in (("legacy_sync", ""), ("per_run", "audit"), ("allow_budget_truncation", "audit"),
                         ("weight_version_recorded", "prepare"), ("heldout_at_round0", "tasks")):
        if key in dropped.get(section, set()):
            continue
        scope = cfg if section == "" else cfg[section]
        if not isinstance(scope[key], bool):
            raise SystemExit(f"config {section + '.' if section else ''}{key} must be a bool")
    if not isinstance(cfg["tasks"]["train"]["shuffle"], bool):
        raise SystemExit("config tasks.train.shuffle must be a bool")
    if not stream:
        baseline = cfg["tasks"]["heldout_baseline_file"]
        if baseline is not None and not isinstance(baseline, str):
            raise SystemExit("config tasks.heldout_baseline_file must be a path or null")
        if not cfg["tasks"]["heldout_at_round0"] and baseline is None:
            raise SystemExit("config tasks.heldout_at_round0 is false: tasks.heldout_baseline_file must name the base-policy "
                             "held-out summary")
    if not (isinstance(cfg["trainer"]["save_every_rounds"], int) and cfg["trainer"]["save_every_rounds"] >= 0):
        raise SystemExit("config trainer.save_every_rounds must be an int >= 0 (0 = never)")
    if not cfg["rollout"]["base_url"].endswith("/v1"):
        raise SystemExit(f"config rollout.base_url must end in /v1: {cfg['rollout']['base_url']!r}")
    replicas = cfg["rollout"].get("replica_urls")
    if replicas is not None and (not stream or not isinstance(replicas, list) or not replicas
                                or not all(isinstance(u, str) and u.endswith("/v1") for u in replicas)
                                or len(set(replicas)) != len(replicas)):
        raise SystemExit("config rollout.replica_urls must be distinct /v1 backend URLs, only for pipeline stream")
    if stream and cfg["stream"].get("group_ready", False) and cfg["prepare"]["minibatches"] != 1:
        raise SystemExit("config stream.group_ready accumulates every packet into one optimizer step: prepare.minibatches "
                         "must be 1")
    if stream and "rollout_slots" in cfg["stream"] and (not replicas or cfg["stream"]["rollout_slots"] % len(replicas)):
        raise SystemExit("config stream.rollout_slots needs rollout.replica_urls and a multiple of their count "
                         "(every replica holds at most rollout_slots / replicas episodes)")
    if "initial_weights_source" in cfg["rollout"] and not isinstance(cfg["rollout"]["initial_weights_source"], str):
        raise SystemExit("config rollout.initial_weights_source must be a source path string")
    if "trainer_base_path" in cfg["rollout"] and (not stream or not isinstance(cfg["rollout"]["trainer_base_path"], str)):
        raise SystemExit("config rollout.trainer_base_path must be a path string, only for pipeline stream")
    if not (isinstance(cfg["disk_guard_paths"], list) and cfg["disk_guard_paths"]):
        raise SystemExit("config disk_guard_paths must be a non-empty list")
    if not (isinstance(cfg["audit"]["whitelist"], list) and all(isinstance(w, str) and w for w in cfg["audit"]["whitelist"])):
        raise SystemExit("config audit.whitelist must be a list of non-empty strings (substrings of audit failure lines)")
    for name in ("audit", "prepare"):
        command = cfg[name]["command"]
        if not (isinstance(command, list) and command and all(isinstance(c, str) for c in command)):
            raise SystemExit(f"config {name}.command must be a non-empty list of strings")
    if "tail_overlap" in cfg:
        if cfg["pipeline"] != "async":
            raise SystemExit("config tail_overlap needs pipeline async: the sync pipeline trains between rollouts")
        _keys(cfg["tail_overlap"], TAIL_OVERLAP_KEYS, "tail_overlap")
        n = cfg["tail_overlap"]["remaining_episodes"]
        per_round = cfg["tasks"]["samples_per_task"] * cfg["tasks"]["tasks_per_round"]
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n < per_round):
            raise SystemExit(f"config tail_overlap.remaining_episodes must be an int in [1, {per_round}) "
                             f"(episodes per round = samples_per_task * tasks_per_round), not {n!r}")
    cfg.setdefault("tail_overlap", None)
    return cfg


def tool_params(prepare: dict) -> dict | None:
    """prepare's arc2 tool-use term parameters (names from arc2_tooluse.PARAMS); None when off."""
    if not any(k in prepare for k in TOOL_KEYS):
        return None
    return {k: prepare["tool_" + k] for k in arc2_tooluse.PARAMS}


def validate_tool_reward(p: dict) -> None:
    """tool_hand_weight / tool_toolkit_weight / tool_hand_scale / tool_hand_cap all given together (ranges per
    arc2_tooluse.check_params), adv_normalization none, not combined with arc3_reward."""
    given = [k for k in TOOL_KEYS if k in p]
    if not given:
        return
    if len(given) != len(TOOL_KEYS):
        raise SystemExit(f"config prepare: {list(TOOL_KEYS)} go together (have {given})")
    try:
        arc2_tooluse.check_params(**tool_params(p))
    except ValueError as error:
        raise SystemExit(f"config prepare: {error}") from error
    if p.get("adv_normalization") != "none" or arc3_mode(p) is not None:
        raise SystemExit("config prepare: the tool-use term needs adv_normalization none and no arc3_reward")


def arc3_mode(prepare: dict) -> str | None:
    """prepare's arc3 shaped-R mode: eff / c0 (prepare_rl.arc3_eff_rewards / arc3_c0_rewards), official (arcforge.arc3_official),
    rank (arcforge.arc3_rank); score / no key is None."""
    mode = prepare.get("arc3_reward", "score")
    return None if mode == "score" else mode


def validate_arc3_reward(p: dict) -> None:
    """prepare.arc3_reward ∈ ARC3_REWARDS; eff / c0 ⇒ its own parameters (ARC3_MODE_KEYS) all given, finite and in range (eff:
    β ≥ 0, clip > 0; c0: λ ≥ 0, κ > 0, 0 < success_floor < 1; both 0 ≤ w_start < 1), adv_normalization none, no efficiency_*;
    official / rank ranges are left to the shared modules' check_params (arcforge.arc3_official / arc3_rank); parameters of another
    mode raise (the same check as prepare_rl, caught early when the config is read)."""
    if "arc3_reward" in p and p["arc3_reward"] not in ARC3_REWARDS:
        raise SystemExit(f"config prepare.arc3_reward must be one of {ARC3_REWARDS}, not {p['arc3_reward']!r}")
    mode = arc3_mode(p)
    want = ARC3_MODE_KEYS[mode] if mode is not None else ()
    foreign = [k for k in ARC3_KEYS if k in p and k not in want]
    if foreign:
        raise SystemExit(f"config prepare: {foreign} are not parameters of arc3_reward {p.get('arc3_reward', 'score')}")
    if mode is None:
        return
    missing = [k for k in want if k not in p]
    if missing:
        raise SystemExit(f"config prepare: arc3_reward {mode} needs all of {list(want)} (missing {missing}; no defaults)")
    v = {k: p[k] for k in want}
    if mode == "note":                     # no parameters: the formulas are arcforge.arc3_note, the grades judge.json
        if p.get("adv_normalization") != "none" or "efficiency_beta" in p:
            raise SystemExit("config prepare: arc3_reward note needs adv_normalization none and no efficiency_beta")
        return
    if mode == "hinge":                    # arc3_hinge_refs is a path: the shared check_params checks every type
        try:
            arc3_hinge.check_params(**{k[len("arc3_"):]: x for k, x in v.items()})
        except ValueError as e:
            raise SystemExit(f"config prepare: {e}") from None
        if p.get("adv_normalization") != "none" or "efficiency_beta" in p:
            raise SystemExit("config prepare: arc3_reward hinge needs adv_normalization none and no efficiency_beta")
        return
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v.values()):
        raise SystemExit(f"config prepare: arc3 {mode} parameters must be finite numbers, not {v}")
    if mode in ("official", "rank"):
        try:
            (arc3_official if mode == "official" else arc3_rank).check_params(**{k[len("arc3_"):]: x for k, x in v.items()})
        except ValueError as e:
            raise SystemExit(f"config prepare: {e}") from None
        if p.get("adv_normalization") != "none" or "efficiency_beta" in p:
            raise SystemExit(f"config prepare: arc3_reward {mode} needs adv_normalization none and no efficiency_beta")
        return
    ok = 0 <= v["arc3_w_start"] < 1 and (
        v["arc3_beta_actions"] >= 0 and v["arc3_beta_tokens"] >= 0 and v["arc3_eff_clip"] > 0 if mode == "eff" else
        v["arc3_lambda_actions"] >= 0 and v["arc3_lambda_tokens"] >= 0 and v["arc3_kappa_actions"] > 0
        and v["arc3_kappa_tokens"] > 0 and 0 < v["arc3_success_floor"] < 1)
    if not ok:
        raise SystemExit(f"config prepare: arc3 {mode} parameters out of range (have {v}; see validate_arc3_reward)")
    if p.get("adv_normalization") != "none" or "efficiency_beta" in p:
        raise SystemExit(f"config prepare: arc3_reward {mode} needs adv_normalization none and no efficiency_beta "
                         "(the efficiency term assumes 0/1 rewards)")


def validate_note_judge(nj) -> None:
    """note_judge (NOTE_JUDGE_KEYS plus the backend's own NOTE_JUDGE_BACKEND_KEYS): backend ∈ NOTE_JUDGE_BACKENDS, python (and api's
    env_file) are existing files, model a non-empty string, effort ∈ this backend's NOTE_JUDGE_EFFORTS, max_concurrent an int ≥ 1,
    timeout_seconds and call_timeout_seconds finite positive numbers, timeout_seconds ≥ arc3_note.judge_attempt_seconds(backend,
    call_timeout_seconds); codex: codex_bin an executable file (absolute path), codex_version a non-empty string."""
    if not isinstance(nj, dict) or nj.get("backend") not in NOTE_JUDGE_BACKENDS:
        raise SystemExit(f"config note_judge.backend must be one of {NOTE_JUDGE_BACKENDS} "
                         f"(not {nj.get('backend') if isinstance(nj, dict) else nj!r})")
    backend = nj["backend"]
    _keys(nj, NOTE_JUDGE_KEYS | NOTE_JUDGE_BACKEND_KEYS[backend], "note_judge")
    for key in ("python", *(("env_file",) if backend == "api" else ())):
        if not (isinstance(nj[key], str) and Path(nj[key]).is_file()):
            raise SystemExit(f"config note_judge.{key} {nj[key]!r} is not an existing file")
    # (model key, effort key, which effort list) per judge of this backend
    judges = ([("claude_model", "claude_effort", "claude"), ("codex_model", "codex_effort", "codex")] if backend == "ensemble"
              else [("claude_model", "claude_effort", "claude")] if backend == "claude-code"
              else [("model", "effort", backend)])
    for mkey, ekey, kind in judges:
        if not (isinstance(nj[mkey], str) and nj[mkey]):
            raise SystemExit(f"config note_judge.{mkey} must be a non-empty string, not {nj[mkey]!r}")
        if nj[ekey] not in NOTE_JUDGE_EFFORTS[kind]:
            raise SystemExit(f"config note_judge.{ekey} must be one of {NOTE_JUDGE_EFFORTS[kind]} for backend {backend}, "
                             f"not {nj[ekey]!r}")
    if backend in V8B_JUDGE_BACKENDS and nj["mapping"] not in arc3_note.V8B_MAPPINGS:
        raise SystemExit(f"config note_judge.mapping must be one of {arc3_note.V8B_MAPPINGS}, not {nj['mapping']!r}")
    n = nj["max_concurrent"]
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 1):
        raise SystemExit(f"config note_judge.max_concurrent must be an int >= 1, not {n!r}")
    for key in ("timeout_seconds", "call_timeout_seconds"):
        t = nj[key]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not (math.isfinite(t) and t > 0):
            raise SystemExit(f"config note_judge.{key} must be a finite number > 0, not {t!r}")
    need = arc3_note.judge_attempt_seconds(backend, nj["call_timeout_seconds"])
    if nj["timeout_seconds"] < need:
        raise SystemExit(f"config note_judge.timeout_seconds {nj['timeout_seconds']:g} < {need:g}, the longest a {backend} judge "
                         f"attempt can take at call_timeout_seconds {nj['call_timeout_seconds']:g} "
                         f"(arcforge.arc3_note.judge_attempt_seconds: every call of every pass timing out, api's overload "
                         f"budget, x1.1 + 600 s)")
    tools = {"ensemble": ("codex", "claude"), "claude-code": ("claude",), "codex": ("codex",)}.get(backend, ())
    for tool in tools:
        exe = nj[f"{tool}_bin"]
        if not (isinstance(exe, str) and Path(exe).is_absolute() and Path(exe).is_file() and os.access(exe, os.X_OK)):
            raise SystemExit(f"config note_judge.{tool}_bin {exe!r} is not an absolute path to an executable file")
        if not (isinstance(nj[f"{tool}_version"], str) and nj[f"{tool}_version"]):
            raise SystemExit(f"config note_judge.{tool}_version must be a non-empty string, not {nj[f'{tool}_version']!r}")


def judge_backend_args(nj: dict) -> list[str]:
    """arc3_note_judge.py backend flags: all have --call-timeout; api is --model / --effort, codex is --backend codex
    --codex-model / --codex-effort / --codex-bin, ensemble is --backend ensemble --mapping plus claude's and codex's own model /
    effort / bin, claude-code is --backend claude-code --mapping plus claude's model / effort / bin."""
    call = ["--call-timeout", repr(float(nj["call_timeout_seconds"]))]
    if nj["backend"] == "api":
        return ["--model", nj["model"], "--effort", nj["effort"], *call]
    if nj["backend"] in V8B_JUDGE_BACKENDS:
        claude = ["--backend", nj["backend"], "--mapping", nj["mapping"], "--claude-model", nj["claude_model"],
                  "--claude-effort", nj["claude_effort"], "--claude-bin", nj["claude_bin"]]
        if nj["backend"] == "claude-code":
            return [*claude, *call]
        return [*claude, "--codex-model", nj["codex_model"], "--codex-effort", nj["codex_effort"], "--codex-bin",
                nj["codex_bin"], *call]
    return ["--backend", "codex", "--codex-model", nj["model"], "--codex-effort", nj["effort"], "--codex-bin", nj["codex_bin"],
            *call]


def validate_sampling(p: dict) -> None:
    """prepare.top_p ∈ (0, 1], prepare.top_k = -1 (off) or an integer ≥ 1 (both optional; absent = 1 / off). With truncated
    sampling, prepare_rl requires the ledger to record the support of every sampled token (server started with
    --session-ledger-sampling-mask), and the trainer replays it (SPEC §1)."""
    top_p, top_k = p.get("top_p", 1.0), p.get("top_k", -1)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise SystemExit(f"config prepare.top_p must be in (0, 1], not {top_p!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not (top_k == -1 or top_k >= 1):
        raise SystemExit(f"config prepare.top_k must be -1 (off) or an integer >= 1, not {top_k!r}")


def prepare_algo(prepare: dict) -> str:
    """prepare's loss (``PREPARE_ALGOS``); no algo key = bpo."""
    return prepare.get("algo", "bpo")


def validate_algo(p: dict) -> None:
    """prepare.algo ∈ PREPARE_ALGOS, its parameters all given and positive, no other algo's parameters (value ranges are checked
    again by prepare_rl and the trainer)."""
    algo = prepare_algo(p)
    if algo not in PREPARE_ALGOS:
        raise SystemExit(f"config prepare.algo must be one of {tuple(PREPARE_ALGOS)}, not {algo!r}")
    missing = [k for k in PREPARE_ALGOS[algo] if k not in p]
    foreign = [k for other, keys in PREPARE_ALGOS.items() if other != algo for k in keys if k in p]
    if missing or foreign:
        raise SystemExit(f"config prepare: algo {algo} needs {list(PREPARE_ALGOS[algo])} (missing {missing}) and none of "
                         f"another algo's parameters (have {foreign})")
    for key in PREPARE_ALGOS[algo]:
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise SystemExit(f"config prepare.{key} must be positive, not {value!r}")


def validate_stream_section(sc) -> None:
    _keys(sc, STREAM_KEYS, "stream", {"max_empty_batches", "task_phases", "prefetch_prepare", "rollout_slots", "group_ready",
                                         "ramp_seconds", "quarantine", "retire_full_groups", "keep_groups_on_failure",
                                         "retire_tasks_file", *STREAM_HELDOUT_KEYS})
    if "quarantine" in sc:
        validate_quarantine(sc["quarantine"])
    for key in ("prefetch_prepare", "group_ready", "retire_full_groups", "keep_groups_on_failure"):
        if not isinstance(sc.get(key, False), bool):
            raise SystemExit(f"config stream.{key} must be a bool")
    if sc.get("group_ready", False) and sc.get("prefetch_prepare", False):
        raise SystemExit("config stream.group_ready already materializes every group as it lands: drop prefetch_prepare")
    phases = sc.get("task_phases", [])
    if not isinstance(phases, list):
        raise SystemExit("config stream.task_phases must be a list")
    previous = -1
    for i, phase in enumerate(phases):
        where = f"stream.task_phases[{i}]"
        _keys(phase, {"start_gid", "priority_ids_file", "priority_draws", "cycle_groups"}, where)
        for key, low in (("start_gid", 0), ("priority_draws", 1), ("cycle_groups", 2)):
            value = phase[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise SystemExit(f"config {where}.{key} must be an int >= {low}")
        if phase["start_gid"] <= previous:
            raise SystemExit("config stream.task_phases start_gid must be sorted and distinct")
        previous = phase["start_gid"]
        if phase["priority_draws"] >= phase["cycle_groups"]:
            raise SystemExit(f"config {where}.priority_draws must be < cycle_groups (retain exploration)")
        if not isinstance(phase["priority_ids_file"], str) or not phase["priority_ids_file"]:
            raise SystemExit(f"config {where}.priority_ids_file must be a non-empty string")
    empty_limit = sc.get("max_empty_batches", 100)
    if not isinstance(empty_limit, int) or isinstance(empty_limit, bool) or empty_limit < 1:
        raise SystemExit("config stream.max_empty_batches must be a positive int")
    for key, low in STREAM_INT_MIN.items():
        value = sc[key]
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= low):
            raise SystemExit(f"config stream.{key} must be an int >= {low}, not {value!r}")
    for key in ("group_timeout_seconds", "hard_min_free_gb"):
        value = sc[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise SystemExit(f"config stream.{key} must be positive, not {value!r}")
    margin = sc["end_margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not margin >= 1.0:
        raise SystemExit(f"config stream.end_margin must be a number >= 1.0 (the end-of-run hold needs supply >= margin x "
                         f"demand; below 1 the learner could starve), not {margin!r}")
    if "rollout_slots" in sc:
        slots = sc["rollout_slots"]
        if not (isinstance(slots, int) and not isinstance(slots, bool) and slots >= sc["group_size"]):
            raise SystemExit(f"config stream.rollout_slots must be an int >= group_size {sc['group_size']}, not {slots!r}")
    if "ramp_seconds" in sc:
        ramp = sc["ramp_seconds"]
        if isinstance(ramp, bool) or not isinstance(ramp, (int, float)) or not (ramp >= 0 and math.isfinite(ramp)):
            raise SystemExit(f"config stream.ramp_seconds must be a finite number >= 0, not {ramp!r}")
    if "retire_tasks_file" in sc:
        read_retire_list(sc["retire_tasks_file"])
    if sc["max_pooled_groups"] * sc["group_size"] < sc["batch_live_episodes"]:
        raise SystemExit("config stream: max_pooled_groups * group_size must be >= batch_live_episodes, or back-pressure "
                         "stops the dispatcher before the pool can hold one batch (deadlock)")
    validate_stream_heldout(sc)


def read_retire_list(path) -> list[str]:
    """stream.retire_tasks_file: one task id per line (blank lines skipped), at least one, no duplicates."""
    if not isinstance(path, str) or not path:
        raise SystemExit(f"config stream.retire_tasks_file must be a non-empty string, not {path!r}")
    if not Path(path).is_file():
        raise SystemExit(f"config stream.retire_tasks_file {path}: no such file")
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids:
        raise SystemExit(f"config stream.retire_tasks_file {path}: lists no task")
    dup = sorted({t for t in ids if ids.count(t) > 1})
    if dup:
        raise SystemExit(f"config stream.retire_tasks_file {path}: duplicated task(s) {dup[:5]}")
    return ids


def validate_stream_heldout(sc: dict) -> None:
    """STREAM_HELDOUT_KEYS. heldout_every 0 (or absent) = off: giving either of the other two keys raises (a key that does nothing is
    worse than none). E > 0: heldout_samples required (≥ 2, the group mode of run_arc3_level / run_arc2 needs at least 2 samples),
    heldout_at_v0 a bool (default true); rollout_slots required and ≥ heldout_samples, since held-out groups share episode slots
    with training groups (replica_running counts both); with only the group-count rule there is no episode budget to share, and a
    held-out round would be dispatched all at once."""
    every = sc.get("heldout_every", 0)
    if isinstance(every, bool) or not isinstance(every, int) or every < 0:
        raise SystemExit(f"config stream.heldout_every must be an int >= 0 (optimizer versions; 0 = no held-out in the "
                         f"stream loop), not {every!r}")
    if every == 0:
        extra = [k for k in STREAM_HELDOUT_KEYS[1:] if k in sc]
        if extra:
            raise SystemExit(f"config stream.{extra} go with stream.heldout_every > 0 (heldout_every 0 = no held-out "
                             f"in the stream loop)")
        return
    if "heldout_samples" not in sc:
        raise SystemExit("config stream.heldout_every > 0 needs stream.heldout_samples (episodes per held-out task)")
    samples = sc["heldout_samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise SystemExit(f"config stream.heldout_samples must be an int >= 2 (a group-mode harness runs >= 2 samples), "
                         f"not {samples!r}")
    if not isinstance(sc.get("heldout_at_v0", True), bool):
        raise SystemExit(f"config stream.heldout_at_v0 must be a bool, not {sc['heldout_at_v0']!r}")
    if "rollout_slots" not in sc:
        raise SystemExit("config stream.heldout_every needs stream.rollout_slots: held-out groups share the episode slots "
                         "with the training groups (the group-count rule has no episode budget to share)")
    if samples > sc["rollout_slots"]:
        raise SystemExit(f"config stream.heldout_samples {samples} must be <= rollout_slots {sc['rollout_slots']}: a "
                         f"held-out group that never fits blocks every training group behind it")


def validate_quarantine(q) -> None:
    """stream.quarantine (QUARANTINE_KEYS, all required, no defaults): same_signature_max and fraction_window_groups ints ≥ 1,
    1 ≤ fraction_min_groups ≤ fraction_window_groups; same_signature_seconds a finite positive number; 0 ≤ max_fraction < 1.
    same_signature_max 1 = stop on every quarantine (the behavior before quarantine existed)."""
    _keys(q, QUARANTINE_KEYS, "stream.quarantine")
    for key in ("same_signature_max", "fraction_window_groups", "fraction_min_groups"):
        value = q[key]
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            raise SystemExit(f"config stream.quarantine.{key} must be an int >= 1, not {value!r}")
    if q["fraction_min_groups"] > q["fraction_window_groups"]:
        raise SystemExit(f"config stream.quarantine.fraction_min_groups {q['fraction_min_groups']} must be <= "
                         f"fraction_window_groups {q['fraction_window_groups']}")
    seconds = q["same_signature_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not (math.isfinite(seconds) and seconds > 0):
        raise SystemExit(f"config stream.quarantine.same_signature_seconds must be a finite number > 0, not {seconds!r}")
    fraction = q["max_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
        raise SystemExit(f"config stream.quarantine.max_fraction must be a number in [0, 1), not {fraction!r}")


def error_signature(error: str, literals) -> str:
    """Key that groups "the same error" together: paths become ``<path>`` (whole path; episode names inside are not replaced
    separately), UUIDs become ``<uuid>``, then this group's own words (``literals``: task id, episode, group name; longest first, not
    adjacent to alphanumerics) become ``<id>``, hex ids and numbers become ``<hex>`` / ``<n>``, whitespace collapses to one space,
    at most SIGNATURE_MAX_CHARS chars. The error type is in the string (failures.jsonl holds a repr, prepare gives
    ``<type>: <message>``)."""
    text = SIGNATURE_UUID.sub("<uuid>", SIGNATURE_PATH.sub("<path>", error))
    for literal in sorted({x for x in literals if x}, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])", "<id>", text)
    text = SIGNATURE_HEX.sub("<hex>", text)
    text = SIGNATURE_NUMBER.sub("<n>", text)
    return " ".join(text.split())[:SIGNATURE_MAX_CHARS]


def quarantine_breaker(rows: list[dict], new: list[dict], finished: list[int], q: dict, now: float) -> str | None:
    """Circuit breaker of stream.quarantine (``rows``: all rows of quarantine.jsonl including the just-written ``new``;
    ``finished``: gids in finishing order, completed ones and ones quarantined at the harness level). Returns the reason to stop
    the run, None = continue.
    1. for each signature in ``new``: the number of **events** with that signature within the last same_signature_seconds ≥
       same_signature_max (rows written by one event share ``event`` and count once; old rows without ``event`` count once each);
    2. among the last fraction_window_groups finished groups (at least fraction_min_groups), the quarantined share > max_fraction."""
    for signature in dict.fromkeys(r["signature"] for r in new):
        same = [r for r in rows if r["signature"] == signature and now - r["unix"] <= q["same_signature_seconds"]]
        events = len({r.get("event") or f"row:{i}" for i, r in enumerate(same)})
        if events >= q["same_signature_max"]:
            return (f"signature {signature!r} quarantined {len(same)} group(s) in {events} event(s) within "
                    f"{q['same_signature_seconds']:g}s (stream.quarantine.same_signature_max {q['same_signature_max']}): "
                    f"{[r['group'] for r in same][-8:]}")
    recent = finished[-q["fraction_window_groups"]:]
    if len(recent) >= q["fraction_min_groups"]:
        bad = {r["gid"] for r in rows} & set(recent)
        if len(bad) / len(recent) > q["max_fraction"]:
            return (f"{len(bad)} of the last {len(recent)} finished group(s) quarantined ({len(bad) / len(recent):.1%} > "
                    f"stream.quarantine.max_fraction {q['max_fraction']:.1%}): {sorted(f'g{g:06d}' for g in bad)[-8:]}")
    return None


def harness_failure(run_dir: Path, stdout: Path, rc: int) -> str:
    """Why a group with rc≠0 failed: the first entry in the harness's ``failures.jsonl`` (one repr row per failed episode) that is
    not a collateral cancellation; with no such row, the last non-empty line of .out."""
    path = run_dir / "failures.jsonl"
    if path.is_file():
        errors = [json.loads(line)["error"] for line in path.read_text().splitlines() if line.strip()]
        real = [e for e in errors if not CANCELLED_FAILURE.search(e)]
        if real:
            more = f" (+{len(real) - 1} more in failures.jsonl)" if len(real) > 1 else ""
            return f"harness exited {rc}: {real[0]}{more}"
    lines = [ln.strip() for ln in stdout.read_text(errors="replace").splitlines() if ln.strip()] if stdout.is_file() else []
    return f"harness exited {rc}: {lines[-1] if lines else '(no failures.jsonl and an empty log)'}"


def harness_rewards(run_dir: Path, task: str, group_size: int) -> list[float | None]:
    """Score of each sample in the group (those whose results are written; None for the rest): the "if available" rewards of a
    quarantine record."""
    results = run_dir / "results"
    return [float(json.loads((results / f"{e}.json").read_text())["score"]) if (results / f"{e}.json").is_file() else None
            for e in group_episode_ids(task, group_size)]


def cfg_path(value: str) -> Path:
    """Relative paths are relative to the repo root (subprocesses also run with cwd = repo root)."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- pure helpers

def read_ids(path: Path) -> list[str]:
    ids = [t for t in path.read_text().replace(",", " ").split() if t]
    if not ids:
        raise OrchestratorError(f"{path}: no task ids")
    dup = sorted(t for t, n in Counter(ids).items() if n > 1)
    if dup:
        raise OrchestratorError(f"{path}: duplicate task ids {dup[:5]}")
    return ids


def tasks_for_round(pool: list[str], k: int, seed: int, r: int, shuffle: bool) -> list[str]:
    """Tasks for round r. shuffle=False: take k tasks per round in file order, wrapping around at the end (the list is ordered
    externally). shuffle=True: shuffle with seed and reshuffle for the next epoch only once it is used up; no repeats within a
    round (duplicates across an epoch boundary are skipped).
    Pure function: after a restart the same r gets the same tasks."""
    n = len(pool)
    if k > n:
        raise OrchestratorError(f"tasks_per_round {k} > train pool {n}")
    if not shuffle:
        return [pool[(r * k + j) % n] for j in range(k)]

    epoch, order, pos, picked = 0, epoch_order(pool, seed, 0), 0, []
    for _ in range(r + 1):
        picked = []
        while len(picked) < k:
            if pos == n:
                epoch, order, pos = epoch + 1, epoch_order(pool, seed, epoch + 1), 0
            task = order[pos]
            pos += 1
            if task not in picked:
                picked.append(task)
    return picked


def epoch_order(pool: list[str], seed: int | str, epoch: int) -> list[str]:
    """Shuffle of the task list in epoch number ``epoch`` (shared by sync's shuffle and stream)."""
    order = list(pool)
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def stream_task(pool: list[str], seed: int | str, shuffle: bool, gid: int) -> str:
    """Which task stream group number gid runs: the list is walked epoch after epoch (shuffle: each epoch shuffled with (seed, epoch);
    otherwise cycling in file order). Pure function: after a resume the same gid gets the same task, so a resume only needs the next gid."""
    epoch, pos = divmod(gid, len(pool))
    return epoch_order(pool, seed, epoch)[pos] if shuffle else pool[pos]


def group_episode_ids(task: str, group_size: int) -> list[str]:
    """item / episode id of the group-mode harness (SPEC §9)."""
    return [f"{task}__s{k}" for k in range(group_size)]


def manifest_failed_ids(manifest: dict, run_dir: Path) -> set[str]:
    """Episodes the harness marked as failed in the manifest (run_arc3_level dd0ae223c: when ≤ 25% of a group's games break, the
    harness exits with rc 0, and those items have ``outcome.status == "failed"``, no thread and no results file; prepare_rl drops each
    of them as harness_failed)."""
    failed = set()
    for item in manifest["items"]:
        if item["outcome"]["status"] == "failed":
            if item["threads"]:
                raise OrchestratorError(f"{run_dir / 'tito_manifest.json'}: failed item {item['id']} has threads {item['threads'][:2]}")
            failed.add(item["id"])
    return failed


def group_results_complete(run_dir: Path, task: str, group_size: int) -> bool:
    """Whether a finished group's results are complete: the manifest exists and every episode the manifest does not mark as failed
    has a results file."""
    path = run_dir / "tito_manifest.json"
    if not path.is_file():
        return False
    failed = manifest_failed_ids(json.loads(path.read_text()), run_dir)
    have = {p.stem for p in (run_dir / "results").glob("*.json")} if (run_dir / "results").is_dir() else set()
    return set(group_episode_ids(task, group_size)) - failed <= have


def read_group_episodes(run_dir: Path, task: str, group_size: int, failed: set[str] = frozenset()) -> list[dict]:
    """``results/<task>__s<k>.json`` of a group-mode harness run (k = 0..group_size-1, none missing and none extra except ``failed``,
    the ones the manifest marks as failed) → episode rows (failed ones excluded, ``sample`` keeps the original k).
    The file's ``task`` is the item id (needed by the TITO join); the task id is in ``tid``. arc3 single level (``harness: arc3``,
    run_arc3_level.py) also carries ``actions`` (actions counted for this level) and ``tokens_at_clear`` (output tokens at the moment
    of clearing, None if not cleared), used by the live check of arc3_reward eff / c0; arc2 results have neither field and no
    ``harness``."""
    results = run_dir / "results"
    ids = group_episode_ids(task, group_size)
    if not failed <= set(ids):
        raise OrchestratorError(f"{run_dir}: manifest marks {sorted(failed - set(ids))[:3]} failed, not episodes of {task}")
    want = [e for e in ids if e not in failed]
    if not want:
        raise OrchestratorError(f"{run_dir}: every one of the {group_size} episodes of {task} is marked failed in the manifest, "
                                f"yet the harness exited 0")
    have = {p.stem for p in results.glob("*.json")} - {"solutions"}
    if have != set(want):
        raise OrchestratorError(f"{run_dir}: results {sorted(have)[:4]} != the {len(want)} non-failed episodes of {task} "
                                f"(missing {sorted(set(want) - have)[:3]}, extra {sorted(have - set(want))[:3]}; "
                                f"manifest marks {sorted(failed)[:3]} failed)")
    return [episode_result(results / f"{episode}.json", task, k, run_dir.name)
            for k, episode in enumerate(ids) if episode not in failed]


def episode_result(path: Path, task: str, k: int, group: str) -> dict:
    """One group-mode ``results/<task>__s<k>.json`` → episode row (read_group_episodes; held-out groups go through here too)."""
    episode = f"{task}__s{k}"
    doc = json.loads(path.read_text())
    if (doc["tid"], doc["sample"], doc["episode"], doc["task"]) != (task, k, episode, episode):
        raise OrchestratorError(f"{path}: tid/sample/episode/task are {doc['tid']!r}/{doc['sample']!r}/{doc['episode']!r}/"
                                f"{doc['task']!r}, want {task!r}/{k}/{episode!r}/{episode!r}")
    out = {"task": task, "group": group, "sample": k, "score": float(doc["score"]), "stop": doc["stop"],
           "tokens": int(doc["usage"]["proxy_output_tokens"])}
    if "tool_use" in doc:                   # run_arc2 tool_use_stats (the arc2 tool-use term, group_live)
        out["tool_use"] = doc["tool_use"]
    if "harness" in doc:
        if doc["harness"] != "arc3":
            raise OrchestratorError(f"{path}: harness {doc['harness']!r} (results with a harness field are arc3)")
        out.update(actions=doc["actions"], tokens_at_clear=doc["tokens_at_clear"], baseline_actions=doc["baseline_actions"])
        if "window_cause" in doc:           # g2seed results since RL v8 (run_arc3_level): arc3_reward note reads these
            out.update(window_cause=doc["window_cause"], resets=doc["resets"], game=doc["game"], level=doc["level"])
    return out


def official_rewards(episodes: list[dict], prepare: dict) -> list[float]:
    """arc3_reward official: R of each episode (arcforge.arc3_official.episode_reward, the same definition as prepare_rl)."""
    params = {k: prepare["arc3_" + k] for k in arc3_official.PARAMS}
    return [arc3_official.episode_reward(e["score"] == 1.0, e["actions"], e["baseline_actions"], **params)
            for e in episodes]


def rank_rewards(episodes: list[dict], prepare: dict, launch_version: int | None) -> list[float]:
    """arc3_reward rank: R of each episode in a group (arcforge.arc3_rank.group_rewards, the same definition as prepare_rl; rank and
    level-clear share ρ are computed over all of this group's episodes, prepare only over the ones it keeps). Level cleared = score > 0
    (κ score for g2seed). ``launch_version`` = the group's dispatch version (required when rank_warmup_versions > 0)."""
    params = {k: prepare["arc3_" + k] for k in arc3_rank.PARAMS}
    return arc3_rank.group_rewards([e["score"] for e in episodes], [e["score"] > 0 for e in episodes],
                                   [e["actions"] for e in episodes], [e["tokens_at_clear"] for e in episodes],
                                   **params, launch_version=launch_version)[0]


def hinge_rewards(episodes: list[dict], prepare: dict, launch_version: int, task: str, gid: int) -> list[float]:
    """arc3_reward hinge: R of each episode in a group (arcforge.arc3_hinge.group_rewards, the same definition as prepare_rl; ρ and
    the mean cost of level-clearing games are computed over all of this group's episodes, prepare only over the ones it keeps).
    Thresholds come from prepare.arc3_hinge_refs (not yet present = no level frozen); ``task`` / ``gid`` decide whether this group
    uses that level's frozen threshold."""
    params = {k: prepare["arc3_" + k] for k in arc3_hinge.PARAMS}
    return arc3_hinge.group_rewards([e["score"] for e in episodes], [e["score"] > 0 for e in episodes],
                                    [e["actions"] for e in episodes], [e["tokens_at_clear"] for e in episodes],
                                    **params, task=task, gid=gid, launch_version=launch_version)[0]


def note_rewards(episodes: list[dict], judge: dict | None) -> list[float]:
    """arc3_reward note: R of each episode in a group (arcforge.arc3_note.episode_reward, the same definition as prepare_rl).
    ``judge`` = this group's judge.json (None when no game needs judging): the games to judge (arc3_note.judged_episode: stop window,
    window changed at the window-change point) must be exactly the ones it judged, with the same action and RESET counts, and the
    reward is recomputed from the grade by arc3_note.judge_rewards. Level cleared = score > 0 (κ score for g2seed)."""
    want = {f"{e['task']}__s{e['sample']}": e for e in episodes if arc3_note.judged_episode(e["stop"], e["window_cause"])}
    if bool(want) != (judge is not None):
        raise OrchestratorError(f"arc3 note: episodes to judge {sorted(want)[:3]} but judge.json "
                                f"{'present' if judge is not None else 'missing'}")
    got = {}
    if want:
        level = {(e["game"], e["level"], e["baseline_actions"]) for e in want.values()}
        if len(level) != 1:
            raise OrchestratorError(f"arc3 note: one group is one level, its judged episodes are {sorted(level)}")
        got = arc3_note.judge_rewards(judge, *level.pop())
        if set(got) != set(want):
            raise OrchestratorError(f"arc3 note: judge.json grades {sorted(got)[:4]}, the episodes to judge are "
                                    f"{sorted(want)[:4]}")
        for name, e in want.items():
            if (got[name]["actions"], got[name]["resets"]) != (e["actions"], e["resets"]):
                raise OrchestratorError(f"arc3 note: judge.json has {name} at {got[name]['actions']} actions / "
                                        f"{got[name]['resets']} resets, its result {e['actions']} / {e['resets']}")
    return [arc3_note.episode_reward(e["stop"], e["score"] > 0, e["actions"], e["baseline_actions"], e["window_cause"],
                                     got[f"{e['task']}__s{e['sample']}"]["reward"]
                                     if f"{e['task']}__s{e['sample']}" in want else None)
            for e in episodes]


def check_judge_doc(nj: dict, doc: dict, path: Path) -> None:
    """judge.json must have been written by this config's judge: ensemble / claude-code = version v8b, same mapping, and judges exactly
    the config's claude (plus codex for ensemble) model and effort; api / codex = no version (v8 format)."""
    if nj["backend"] in V8B_JUDGE_BACKENDS:
        want = [{"backend": "claude", "model": nj["claude_model"], "effort": nj["claude_effort"]}]
        if nj["backend"] == "ensemble":
            want.append({"backend": "codex", "model": nj["codex_model"], "effort": nj["codex_effort"]})
        if (doc.get("version"), doc.get("mapping"), doc.get("judges")) != (arc3_note.V8B_VERSION, nj["mapping"], want):
            raise OrchestratorError(f"{path}: version {doc.get('version')!r} mapping {doc.get('mapping')!r} judges "
                                    f"{doc.get('judges')}, the config's {nj['backend']} judge is {arc3_note.V8B_VERSION} "
                                    f"{nj['mapping']} {want}")
    elif "version" in doc:
        raise OrchestratorError(f"{path}: version {doc['version']!r} from a {nj['backend']} judge (v8 files have no version)")


def note_judge_input(run_dir: Path, task: str, episodes: list[dict]) -> dict | None:
    """arc3_reward note: what a completed group hands to the judge (arc3_note_judge.py's group.json): {game, level, baseline, notes:
    [{id, text (handoff.md), actions, resets}]}, only the games to judge (arc3_note.judged_episode); none at all = None (no judge run).
    The sha256 of handoff.md must match the one recorded in the results file."""
    notes, levels = [], set()
    for e in episodes:
        if not arc3_note.judged_episode(e["stop"], e["window_cause"]):
            continue
        doc = json.loads((run_dir / "results" / f"{task}__s{e['sample']}.json").read_text())
        handoff = doc["handoff"]
        text = (run_dir / handoff["path"]).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != handoff["sha256"]:
            raise OrchestratorError(f"{run_dir / handoff['path']}: sha256 differs from the one in {doc['task']}'s result")
        levels.add((doc["game"], doc["level"], doc["baseline_actions"]))
        notes.append({"id": doc["task"], "text": text, "actions": doc["actions"], "resets": doc["resets"]})
    if not notes:
        return None
    if len(levels) != 1:
        raise OrchestratorError(f"{run_dir}: one group is one level, its notes are {sorted(levels)}")
    game, level, baseline = levels.pop()
    return {"game": game, "level": level, "baseline": baseline, "notes": notes}


def group_full(episodes: list[dict], prepare: dict) -> bool:
    """arc3_reward official: every episode has R = 1 (level cleared with official score ≥ top_score), the retirement condition of
    stream.retire_full_groups."""
    return all(r == 1.0 for r in official_rewards(episodes, prepare))


def group_live(episodes: list[dict], prepare: dict, launch_version: int | None, judge: dict | None = None) -> bool:
    """Whether a group has training signal (prepare_rl will not drop it whole as std0): rewards not all equal; or (arc2 tool-use terms
    tool_*) the per-episode q not all equal; or (prepare's efficiency term) every episode all tests correct with output tokens not all
    equal; or (arc3_reward eff / c0) every episode cleared the level and the term with nonzero weight (β / λ: actions,
    tokens_at_clear) not all equal; official: R converted from the official score not all equal; rank: R of score times the in-group
    rank discount not all equal (``launch_version`` = dispatch version, efficiency term 0 during warm-up; other modes do not read it);
    note: R from note_rewards (``judge`` = this group's judge.json, given only for note) not all equal. prepare also drops individual
    episodes (compaction, routing…), so this is an estimate; batches count live episodes by it."""
    if arc3_mode(prepare) == "note":     # judge: the group's judge.json (complete_group), None when nothing was judged
        return len(set(note_rewards(episodes, judge))) > 1
    if judge is not None:
        raise OrchestratorError(f"group_live: a judge.json for arc3_reward {prepare.get('arc3_reward', 'score')}")
    if arc3_mode(prepare) == "official":
        return len(set(official_rewards(episodes, prepare))) > 1
    if arc3_mode(prepare) == "rank":
        return len(set(rank_rewards(episodes, prepare, launch_version))) > 1
    if arc3_mode(prepare) == "hinge":    # read_group_episodes rows: task = the tid, group = the g<NNNNNN> run directory
        (task,), (group,) = {e["task"] for e in episodes}, {e["group"] for e in episodes}
        return len(set(hinge_rewards(episodes, prepare, launch_version, task, int(group.removeprefix("g"))))) > 1
    rewards = [e["score"] for e in episodes]
    if len(set(rewards)) > 1:
        return True
    tool = tool_params(prepare)
    if tool is not None:
        lack = [e["sample"] for e in episodes if "tool_use" not in e]
        if lack:
            raise OrchestratorError(f"prepare has the tool-use term but episodes {lack[:4]} have no tool_use (harness "
                                    "config tool_use_stats)")
        if len({arc2_tooluse.episode_score(e["tool_use"], **tool)[0] for e in episodes}) > 1:
            return True
    if min(rewards) != 1.0:
        return False
    if "efficiency_beta" in prepare:
        return len({e["tokens"] for e in episodes}) > 1
    mode = arc3_mode(prepare)
    if mode is not None:
        w_actions, w_tokens = ARC3_WEIGHT_KEYS[mode]
        return ((prepare[w_actions] > 0 and len({e["actions"] for e in episodes}) > 1)
                or (prepare[w_tokens] > 0 and len({e["tokens_at_clear"] for e in episodes}) > 1))
    return False


def manifest_servers(manifest: dict) -> list[dict]:
    """Each backend record of the tito manifest: ``servers`` (a group's samples hit several replicas); absent = only ``server``."""
    return manifest["servers"] if "servers" in manifest else [manifest["server"]]


def assign_samples(running: list[int], group_size: int) -> list[int]:
    """Assign each sample of a group in turn to the replica with the fewest running episodes at that moment (ties to the lower
    index); returns the list of replica indices."""
    load = list(running)
    out = []
    for _ in range(group_size):
        r = min(range(len(load)), key=lambda i: (load[i], i))
        out.append(r)
        load[r] += 1
    return out


def ramp_allowed(elapsed: float, cap: int, group_size: int, ramp_seconds: float) -> float:
    """stream.ramp_seconds: cap on running episodes ``elapsed`` seconds after the dispatcher starts, rising linearly from group_size
    (the first group launches immediately) to cap (the smaller of rollout_slots and max_groups_in_flight × group_size), reaching it
    after ramp_seconds; 0 = cap from the start (no ramp)."""
    if elapsed < 0 or ramp_seconds < 0 or cap < group_size:
        raise OrchestratorError(f"ramp: elapsed {elapsed}, ramp_seconds {ramp_seconds}, cap {cap} < group_size {group_size}")
    if elapsed >= ramp_seconds:
        return float(cap)
    return group_size + (cap - group_size) * elapsed / ramp_seconds


def group_episodes_done(run_dir: Path, task: str, group_size: int) -> int:
    """Number of episodes in a group whose results are written (the harness atomically writes results/<task>__s<k>.json as each
    episode ends; dir not yet created = 0)."""
    results = run_dir / "results"
    if not results.is_dir():
        return 0
    return len({p.stem for p in results.glob("*.json")} & set(group_episode_ids(task, group_size)))


def unfinished_episodes(entries) -> int:
    """Episodes of running groups (dispatcher entries: training or held-out) whose results are not yet written. Each entry carries its
    own ``samples``: group_size for a training group, that round's heldout_samples for a held-out group, so group_size times the
    process count would be wrong."""
    return sum(e["samples"] - group_episodes_done(e["run_dir"], e["task"], e["samples"]) for e in entries)


def heldout_round_summary(rnd: dict, rows: list[dict]) -> dict:
    """Summary row of one held-out round (``rnd``: the round in heldout_state.json); ``rows``: its kind-group rows, one per task
    (counted only once all are in). Only episodes of completed groups count: pass rate = share with score > 0 (run_arc3_level single
    level: score 1 = level cleared = correct; full game: official score ÷ 100, > 0 = some progress), mean score, share that hit the
    budget (stop == "budget"), mean / median output tokens, stop distribution, pass rate per task; failed groups (rc≠0, timeout,
    mismatched results) and games in a group that the harness marked as failed all count toward failed_episodes. Version span =
    smallest server version at group launch to largest at completion (an episode may span versions: for monitoring, not an exact
    single-version evaluation). No episodes at all: shares and means are None."""
    if sorted(r["task"] for r in rows) != sorted(rnd["tasks"]) or any(r["version"] != rnd["version"] for r in rows):
        raise OrchestratorError(f"held-out round v{rnd['version']:06d}: summary over rows {sorted(r['task'] for r in rows)[:5]}, "
                                f"want every task {rnd['tasks'][:5]}")
    done = [r for r in rows if r["status"] == "completed"]
    episodes = [e for r in done for e in r["episodes"]]
    n = len(episodes)
    tokens = [e["tokens"] for e in episodes]
    passed = lambda eps: sum(e["score"] > 0 for e in eps)  # noqa: E731
    launched = [r["server_version_launch"] for r in rows if r["server_version_launch"] is not None]
    out = {"kind": "round", "version": rnd["version"], "launch_version": rnd["launch_version"], "tasks": len(rows),
           "samples": rnd["samples"], "groups_failed": len(rows) - len(done), "episodes": n,
           "failed_episodes": sum(len(r["failed_episodes"]) for r in rows),
           "pass_rate": round(passed(episodes) / n, 6) if n else None,
           "score_mean": round(statistics.fmean(e["score"] for e in episodes), 6) if n else None,
           "budget_hit_rate": round(sum(e["stop"] == "budget" for e in episodes) / n, 6) if n else None,
           "output_tokens_mean": round(statistics.fmean(tokens), 1) if n else None,
           "output_tokens_p50": statistics.median(tokens) if n else None,
           "stops": dict(sorted(Counter(e["stop"] for e in episodes).items())),
           "per_task_pass_rate": {r["task"]: round(passed(r["episodes"]) / len(r["episodes"]), 6) for r in done},
           "server_versions": [min(launched) if launched else None, max(r["server_version_complete"] for r in rows)],
           "opened_utc": rnd["opened_utc"], "finished_utc": utc_now(),
           "seconds": round(max(r["completed_unix"] for r in rows) - rnd["opened_unix"], 1)}
    if episodes and all("official_reward" in e for e in episodes):     # prepare.arc3_reward official: the run's own R
        out["official_reward_mean"] = round(statistics.fmean(e["official_reward"] for e in episodes), 6)
    return out


def recent_live_fraction(live_history: list[bool], live_target: int) -> float | None:
    """Share of live groups among recently completed groups: count back from the newest until ``live_target`` live groups are seen
    (stream: two batches' worth of completions), or the whole run so far if there are not enough; no completed group yet = None (no
    estimate)."""
    seen = live = 0
    for flag in reversed(live_history):
        seen += 1
        live += flag
        if live >= live_target:
            break
    return live / seen if seen else None


def end_of_run_supply(pool_live_groups: int, in_flight: int, live_fraction: float | None, batches_left: int,
                      sc: dict) -> tuple[bool, float, int]:
    """Stream wind-down (SPEC §9). Demand: live episodes the remaining batches need; one batch actually consumes
    ceil(batch_live_episodes / group_size) live groups (taken until enough, the last one is not split). Supply: the pool's live
    episodes + running groups × group_size × the recent live share (no share yet: running ones do not count, conservatively).
    Returns (supply ≥ end_margin × demand = no more launches needed, supply, demand)."""
    if batches_left < 0:
        raise OrchestratorError(f"end-of-run supply: {batches_left} batches left (the learner took more than rounds)")
    gs = sc["group_size"]
    demand = batches_left * math.ceil(sc["batch_live_episodes"] / gs) * gs
    expected = 0.0 if live_fraction is None else in_flight * gs * live_fraction
    supply = pool_live_groups * gs + expected
    return supply >= sc["end_margin"] * demand, supply, demand


#: stream.group_ready: prepare summary keys summed over a batch's packets, and the counter dicts merged key by key
SUMMARY_SUMS = ("episodes", "episodes_kept", "groups", "groups_kept", "samples", "pad_samples", "tokens", "loss_tokens",
                "tokens_with_pad", "budget_extra_rids", "wrapup_cut_rids", "wrapup_masked_rids", "wrapup_masked_tokens",
                "unjoined_segments", "ledger_read_bytes", "seconds",
                "dropped_reward_sum", "groups_std0_after_drop")
SUMMARY_COUNTS = ("dropped_episodes", "dropped_groups", "routing_positions_joined", "staleness_loss_tokens",
                  "staleness_episodes", "staleness_episodes_stale", "staleness_stale_tokens", "weight_version_tokens",
                  "temperature_found", "quality", "reforward")


#: shares in prepare_rl's arc3_reward summary (arc3_reward_summary) that are recomputed from sums when merging (other numbers add up;
#: non-numbers must be identical in every packet)
ARC3_SHARE_DERIVED = ("unweighted", "loss_token_weighted")


def merge_arc3_reward(summaries: list[dict]) -> dict | None:
    """Packet ``arc3_reward`` summaries → the batch's: mode / parameters / rule must be identical, counts and Σ|·| add up, p̂
    histograms merge, and the efficiency term's share of the gradient and c0's token_share are recomputed from the summed totals
    (denominator 0 = None, as in prepare_rl). No packet has one = None (score mode); if some packets have it and some do not, the
    ones without it may only be packets that kept no group at all (EMPTY_ELIGIBLE)."""
    present = [s["arc3_reward"] for s in summaries if s.get("arc3_reward") is not None]
    if not present:
        return None
    missing = [s.get("groups_kept", 0) for s in summaries if s.get("arc3_reward") is None]
    if any(missing):
        raise OrchestratorError(f"merge_arc3_reward: packets that kept groups {missing} have no arc3_reward summary")
    first = present[0]
    nested = {"p_hat_histogram", "gradient_share", "token_share", "hinge_refs_snapshot", "judge_files"}
    params = {"mode", "rule"} | {k for k in first if k in {p[len("arc3_"):] for p in ARC3_KEYS}}
    counts = set(first) - nested - params
    for k in counts:
        if isinstance(first[k], bool) or not isinstance(first[k], (int, float)):
            raise OrchestratorError(f"merge_arc3_reward: {k} = {first[k]!r} is neither a parameter nor a number")
    out: dict = {}
    for a in present:
        if set(a) != set(first) or set(a["gradient_share"]) != set(first["gradient_share"]):
            raise OrchestratorError(f"merge_arc3_reward: packets carry different keys {sorted(set(a) ^ set(first))}")
        if any(a[k] != first[k] for k in params):
            raise OrchestratorError(f"merge_arc3_reward: packets of one batch have different modes or parameters "
                                    f"{ {k: (first[k], a[k]) for k in params if a[k] != first[k]} }")
    out.update({k: first[k] for k in sorted(params)})
    for k in sorted(counts):            # group counts, the mode's guard_hits / clip_hits, c0's |eff| sums
        out[k] = sum(a[k] for a in present)
    hist: Counter = Counter()
    for a in present:
        hist.update(a["p_hat_histogram"])
    out["p_hat_histogram"] = {k: hist[k] for k in sorted(hist)}
    sums = {k: sum(a["gradient_share"][k] for a in present) for k in first["gradient_share"]
            if k not in ARC3_SHARE_DERIVED}

    def share(eff: str, task: str) -> float | None:
        total = sums[task] + sums[eff]
        return sums[eff] / total if total > 0 else None

    out["gradient_share"] = {"unweighted": share("eff_abs", "task_abs"),
                             "loss_token_weighted": share("eff_abs_loss_tokens", "task_abs_loss_tokens"), **sums}
    if "hinge_refs_snapshot" in first:   # hinge: the refs file each packet read (levels freeze between packets)
        out["hinge_refs_snapshots"] = sorted({json.dumps(a["hinge_refs_snapshot"], sort_keys=True) for a in present})
        out["hinge_refs_snapshots"] = [json.loads(s) for s in out["hinge_refs_snapshots"]]
    if "judge_files" in first:          # note: {gid: judge.json sha256} per packet; a group is in one packet only
        files: dict = {}
        for a in present:
            dup = set(files) & set(a["judge_files"])
            if dup:
                raise OrchestratorError(f"merge_arc3_reward: groups {sorted(dup)} have a judge.json in two packets")
            files.update(a["judge_files"])
        out["judge_files"] = {k: files[k] for k in sorted(files)}
    if "token_share" in first:
        eff_total = out["eff_actions_abs"] + out["eff_tokens_abs"]
        out["token_share"] = out["eff_tokens_abs"] / eff_total if eff_total > 0 else None
    return out


def merge_prepare_summaries(summaries: list[dict]) -> dict:
    """group_ready: the prepare summaries of one batch's packets -> the batch's (what batches.jsonl and wandb_sync read):
    counts summed, counter dicts merged, the versions present united. Every packet has the batch's window (V, S). An
    EMPTY_ELIGIBLE packet has no token counts (0 here). The per-packet summaries stay in the packet directories."""
    if not summaries:
        raise OrchestratorError("merge_prepare_summaries: no packet")
    windows = {(s["current_version"], s["max_staleness"]) for s in summaries}
    if len(windows) != 1:
        raise OrchestratorError(f"merge_prepare_summaries: packets of one batch have windows {sorted(windows)}")
    out: dict = {"packets": len(summaries)}
    for key in SUMMARY_SUMS:
        out[key] = sum(s.get(key, 0) for s in summaries)
    for key in SUMMARY_COUNTS:
        merged: Counter = Counter()
        for s in summaries:
            merged.update(s.get(key) or {})
        out[key] = dict(merged)
    out["current_version"], out["max_staleness"] = windows.pop()
    versions = sorted({v for s in summaries if "weight_version" in s
                       for v in ([s["weight_version"]] if isinstance(s["weight_version"], str) else s["weight_version"])},
                      key=int)
    out["weight_version"] = versions[0] if len(versions) == 1 else versions
    out["live_group_fraction"] = round(out["groups_kept"] / out["groups"], 4) if out["groups"] else 0.0
    arc3 = merge_arc3_reward(summaries)
    if arc3 is not None:
        out["arc3_reward"] = arc3
    return out


def read_episodes(run_dirs: list[Path], tasks: list[str]) -> list[dict]:
    """Each run's ``results/<task>.json`` (run_arc2's solve() result) → episode rows. Raises if runs and tasks do not line up."""
    episodes = []
    for run_dir in run_dirs:
        sample = int(run_dir.name[1:])
        results = run_dir / "results"
        have = {p.stem for p in results.glob("*.json")} - {"solutions"}
        if have != set(tasks):
            raise OrchestratorError(f"{run_dir}: results for {len(have)} tasks, want {len(tasks)}; "
                                    f"missing {sorted(set(tasks) - have)[:3]} extra {sorted(have - set(tasks))[:3]}")
        for task in tasks:
            doc = json.loads((results / f"{task}.json").read_text())
            if doc["task"] != task:
                raise OrchestratorError(f"{results / (task + '.json')}: task field is {doc['task']!r}")
            episodes.append({"task": task, "sample": sample, "score": float(doc["score"]), "stop": doc["stop"],
                             "tokens": int(doc["usage"]["proxy_output_tokens"])})
    return episodes


def episodes_done(run_dirs: list[Path], tasks: list[str]) -> int:
    """Number of finished episodes: each run's ``results/<task>.json``. run_arc2 writes it with tmp + os.replace after a task's
    solve() returns (codex has exited, the answer is collected), so file present = that task is done and the content is complete;
    a task that crashed writes nothing (counted as "still running"; that round will have rc≠0 anyway).
    A run dir not yet created (staggered harness start) counts as 0."""
    want = set(tasks)
    done = 0
    for run_dir in run_dirs:
        results = run_dir / "results"
        if results.is_dir():
            done += len({p.stem for p in results.glob("*.json")} & want)
    return done


def summarize(episodes: list[dict], samples: int, by: str = "task") -> dict:
    """Mean reward, live-group share (samples of the same group with not all-equal scores), output tokens, share that hit the budget
    (stop == "budget"). Group key ``by``: the task (``task``) in rounds, the group dir (``group``) in stream (the same task can
    appear twice in one batch)."""
    by_task: dict[str, list[float]] = {}
    for e in episodes:
        by_task.setdefault(e[by], []).append(e["score"])
    # fewer than ``samples`` is a group whose harness marked some episodes failed (manifest_failed_ids); more is a bug
    short = {t: len(v) for t, v in by_task.items() if len(v) > samples}
    if short:
        raise OrchestratorError(f"groups with more than {samples} samples: {dict(list(short.items())[:3])}")
    live = sum(len(set(v)) > 1 for v in by_task.values())
    tokens = [e["tokens"] for e in episodes]
    return {"episodes": len(episodes), {"task": "tasks", "group": "groups"}[by]: len(by_task),
            "reward_mean": round(statistics.fmean(e["score"] for e in episodes), 6),
            "live_groups": live, "live_fraction": round(live / len(by_task), 6),
            "output_tokens_mean": round(statistics.fmean(tokens), 1), "output_tokens_p50": statistics.median(tokens),
            "budget_hit_rate": round(sum(e["stop"] == "budget" for e in episodes) / len(episodes), 6),
            "stops": dict(sorted(Counter(e["stop"] for e in episodes).items())),
            "per_task_mean": {t: round(statistics.fmean(v), 6) for t, v in sorted(by_task.items())}}


def run_threads(manifest_path: Path) -> set[str]:
    doc = json.loads(manifest_path.read_text())
    return {t["thread"] for item in doc["items"] for t in item["threads"]}


def budget_stops(run_dir: Path) -> int:
    """Upper bound on "last turn truncated when the harness wound down" for this run (cap on the audit's extra rids,
    budget_truncation_blockers): for arc2 the number of tasks whose stop is in ARC2_KILL_STOPS (budget equals the number of
    ``: budget score`` lines in the harness stdout); for arc3 (results carry ``harness: arc3``) every game counts as one: level
    cleared / cap / budget / wall / died are wound down by the harness, and for a gave_up game run_arm may also truncate the in-flight
    turn when it closes the session (v4b g094: one gave_up game had 1 extra rid and the whole group was quarantined, while an extra 1
    on a won game in the same batch passed). The exact per-thread condition (it is that thread's last ledger entry, at most one, the
    stop is a wind-down) is judged by prepare_rl's rl_rid_join; a mismatch drops only that game and does not take the group down."""
    n = 0
    for path in (run_dir / "results").glob("*.json"):
        if path.stem != "solutions":
            doc = json.loads(path.read_text())
            n += doc.get("harness") == "arc3" or doc["stop"] in ARC2_KILL_STOPS
    return n


def wrapup_cut_rids(run_dir: Path) -> set[str]:
    """rids of this run's turns truncated at the wrap-up line (results' ``wrapup.cut_rid``; harness config ``wrapup_cut``)."""
    out = set()
    for path in (run_dir / "results").glob("*.json"):
        if path.stem != "solutions":
            rid = (json.loads(path.read_text()).get("wrapup") or {}).get("cut_rid")
            if rid is not None:
                out.add(rid)
    return out


def zombie_rids(run_dir: Path) -> set[str]:
    """rids of requests this run's proxy log records as ``client_disconnected`` (zombies: tool call arguments are not streamed until
    complete, codex disconnects and resends after 300 s without an event; the server still generates to the end, so the ledger has
    it and the manifest does not). They do not count toward the audit's extra-rid cap (like wrapup_cut_rids); per game, prepare_rl's
    rl_rid_join judges, and a mismatch drops only that game. RL v7 smoke 2026-10-28: all 24 games of g000006 hit the budget, each
    with 1 extra wrap-up rid, plus 1 zombie = 25 > 24, and the whole group was quarantined (a hard-level group where every game hits
    the cap has no slack to absorb a zombie)."""
    out = set()
    for path in (run_dir / "proxy").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("client_disconnected"):
                out.add(rec["request_id"])
    return out


def budget_truncation_blockers(join: dict, threads: dict[str, set[str]], budget: dict[str, int],
                               cut: dict[str, set[str]] | None = None) -> list[str]:
    """The audit's rid-join-incomplete line passes only in this case: 0 missing, 0 window mismatches, and each run's number of "extra
    rids" ≤ that run's number of tasks that hit the budget (a task killed at the budget: the server recorded the truncated last turn,
    codex did not record its rid, and the session was not closed). The wrap-up-truncated turns named in ``cut`` (run → rid) (codex does
    not record the id of an incomplete response) and zombie requests (zombie_rids) do not count as extra. Returns the blocking reasons
    (empty = pass)."""
    out = []
    if join["missing"]:
        out.append(f"{len(join['missing'])} manifest rid(s) missing from the ledger")
    if join["wrong_window"]:
        out.append(f"{len(join['wrong_window'])} rid(s) in the wrong window")
    owner = {t: run for run, ts in threads.items() for t in ts}
    extra = Counter()
    for thread, rid in join["extra"]:
        if thread not in owner:
            out.append(f"extra rid on thread {thread}, which belongs to no audited run")
        elif rid not in (cut or {}).get(owner[thread], set()):
            extra[owner[thread]] += 1
    for run, n in sorted(extra.items()):
        if n > budget[run]:
            out.append(f"{run}: {n} extra rid(s) but only {budget[run]} harness-ended task(s) (budget_stops)")
    return out


def ledger_instance_dir(status: dict) -> Path:
    """The dir of ``/session_ledger/status`` may be the ledger root or already an instance dir (same rule as lane2.sh)."""
    root, instance = status["dir"].rstrip("/"), status["instance"]
    return Path(root) if root.endswith(instance) else Path(root) / instance


def ledger_statuses(status: dict) -> list[dict]:
    """A control endpoint retains the real status of every independent replica."""
    statuses = status.get("replica_statuses", [status])
    names = [row["instance"] for row in statuses]
    if not statuses or len(names) != len(set(names)):
        raise OrchestratorError(f"empty or duplicate replica ledger instances: {names}")
    return statuses


def ledger_count(status: dict, key: str) -> int:
    """``enqueued_events`` / ``written_events`` of ``/session_ledger/status`` (SPEC §8; missing = the fork is too old)."""
    if key not in status:
        raise OrchestratorError(f"/session_ledger/status has no {key!r}: the rollout server's fork is older than SPEC section 8 "
                                f"(async needs enqueued_events and written_events)")
    return status[key]


def splits_over_allowance(split_reasons: dict[str, int], expected: list[str], allow: dict[str, int]) -> list[str]:
    """Unexpected entries in the audit report's ``split_reasons`` (``open:<reason>`` / ``close:<reason>`` → segment count) that
    exceed ``allow``; empty = pass."""
    over = []
    for key, count in sorted(split_reasons.items()):
        side, reason = key.split(":", 1)
        if side not in ("open", "close"):
            raise OrchestratorError(f"audit split_reasons key {key!r} is not open:/close:<reason>")
        if reason in expected:
            continue
        if reason not in allow:
            over.append(f"{key}={count} not allowed")
        elif count > allow[reason]:
            over.append(f"{key}={count} > {allow[reason]}")
    return over


def assert_finite(value, where: str) -> None:
    """Stop if NaN / inf appears in the trainer's returned metrics (without guessing which field is the loss)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise OrchestratorError(f"non-finite value at {where}")
    if isinstance(value, dict):
        for k, v in value.items():
            assert_finite(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            assert_finite(v, f"{where}[{i}]")


def http_request(method: str, url: str, body: dict | None = None, timeout: float = 60) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise OrchestratorError(f"{method} {url} -> HTTP {error.code}: {error.read()[:2000]!r}") from error
    except OSError as error:                                   # URLError, refused, timeout
        raise OrchestratorError(f"{method} {url} failed: {error!r}") from error


def http_text(method: str, url: str, timeout: float = 60) -> str:
    """Endpoints whose response is not JSON (``/flush_cache`` returns plain text and fails with HTTP 400)."""
    return http_request(method, url, None, timeout).decode(errors="replace")


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 60) -> dict:
    raw = http_request(method, url, body, timeout)
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OrchestratorError(f"{method} {url}: response is not JSON: {raw[:300]!r}") from error
    if not isinstance(doc, (dict, list)):
        raise OrchestratorError(f"{method} {url}: unexpected JSON {doc!r}")
    return doc


def http_ok(url: str, timeout: float = 10) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


#: lock for append_jsonl: groups.jsonl and quarantine.jsonl are written by three threads (dispatcher, learner, materializer), and a
#: line may be larger than the buffer
_APPEND_LOCK = threading.Lock()


def append_jsonl(path: Path, doc: dict) -> None:
    """Write one JSON line to an append-only file and fsync (it counts only once written)."""
    line = json.dumps(doc, ensure_ascii=False) + "\n"
    with _APPEND_LOCK, path.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def utc_now() -> str:
    return time.strftime("%FT%TZ", time.gmtime())


# --------------------------------------------------------------------------- orchestrator

class Orchestrator:
    def __init__(self, cfg: dict, *, abandon_incomplete: bool = False, set_aside_judge: tuple[int, ...] = ()) -> None:
        self.cfg = cfg
        self.root = cfg_path(cfg["run_root"])
        self.poll = float(cfg["poll_seconds"])
        self.abandon_incomplete = abandon_incomplete
        # --set-aside-judge-groups: finished-but-unjudged note groups stream_start_check moves aside (never trained)
        self.set_aside_judge = tuple(set_aside_judge)
        if self.set_aside_judge and "note_judge" not in cfg:
            raise OrchestratorError("--set-aside-judge-groups is for a stream run with prepare.arc3_reward note (note_judge)")
        self.cancelled = threading.Event()
        self.log_file = None
        self.legacy = cfg["legacy_sync"]
        self.base_url = cfg["rollout"]["base_url"]
        self.server_root = self.base_url[:-len("/v1")]
        self.served_name: str | None = None
        self.harness_cfg: dict | None = None
        self.pools: dict[str, list[str]] = {}
        self.task_phases: list[dict] = []
        self.heldout_baseline: dict | None = None
        self.log_lock = threading.Lock()
        # ---- async pipeline state (SPEC §8). Everything below is guarded by ``cv``.
        self.aborted = threading.Event()              # set when one pipeline thread died; the other stops starting stages
        self.cv = threading.Condition()
        self.server_version = 0                       # the server's weight_version as verified by the learner's last sync
        self.rolled: dict[int, dict] = {}             # rollout k -> hand-off to the learner
        self.rollout_end: int | None = None           # the rollout thread will produce no round >= this
        # round -> claimed monotonic, while its harness runs are live. Two entries only under tail_overlap: round k's tail
        # and round k+1 (the gate keeps older rounds out: round k+1 needs version k, i.e. round k-1 fully learned).
        self.active_rollouts: dict[int, float] = {}
        self.rollout_ended: dict[int, float] = {}     # round -> monotonic when its last harness run exited
        self.heldout_running = False
        self.sync_in_progress = False
        # pipeline threads (rollout, learner, and one per round under tail_overlap) and what they returned / raised
        self.pipeline_threads: list[threading.Thread] = []
        self.pipeline_outcomes: dict[str, str] = {}
        self.pipeline_failures: list[tuple[str, BaseException]] = []
        # ---- stream pipeline (SPEC §9): rows are batches, not rounds. Shared state below is guarded by ``cv``.
        self.stream: dict | None = cfg["stream"] if cfg["pipeline"] == "stream" else None
        #: rollout_slots from config.json: hot reload (dispatcher_overrides) may only lower it to this value or below
        self.config_rollout_slots = None if self.stream is None else self.stream.get("rollout_slots")
        self.history_name, self.history_key = (("batches.jsonl", "batch") if self.stream is not None
                                               else ("rounds.jsonl", "round"))
        self.pool: list[dict] = []                    # completed, unconsumed groups in completion order (groups.jsonl records)
        self.inflight_groups: dict[int, float] = {}   # gid -> launch monotonic; the Popen handles live in the dispatcher only
        self.group_backends: dict[int, list[int]] = {}  # in-flight gid -> replica index (rollout.replica_urls) per sample
        self.dispatch_ended = False                   # the dispatcher exited normally: no group will complete any more
        self.learner_finished = False                 # the learner trained ``rounds`` batches: stop dispatching
        self.batches_taken = 0                        # batches whose groups left the pool (rows + the one in progress)
        self.retention_rows: list[dict] | None = None  # loaded once under the orchestrator lock; append incrementally
        self.live_history: list[bool] = []            # live flag of every completed group, completion order (dispatcher's)
        self.stream_stats: dict | None = None         # dispatcher counters since the last batch row
        self.session_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.urandom(3).hex()}"   # in every stream salt prefix
        # ---- stream.group_ready (guarded by ``cv``): batch -> reservation / packets (gr_open); the frontier batch takes
        # the pool's head next; the learner's batch bounds the frontier (one batch ahead, like prefetch_prepare)
        self.gr_batches: dict[int, dict] = {}
        self.gr_frontier = 0
        self.gr_learner = 0
        self.gr_closed = False                        # the learner returned: the materializer stops
        # ---- stream.quarantine (guarded by ``cv``): quarantine.jsonl's rows, and the gids in the order they finished
        # (completed, or quarantined at the harness stage) -- the circuit breaker's inputs (quarantine_breaker)
        self.quarantine_rows: list[dict] = []
        self.finish_order: list[int] = []
        # ---- stream adoption (SPEC §9): detaching = this stop does not terminate in-flight groups (SIGUSR1, the DETACH file, or the
        # other thread dying with stream.keep_groups_on_failure); group_seconds = seconds each completed group took (straggler test
        # during wind-down)
        self.detaching = False
        self.group_seconds: list[float] = []
        # ---- stream in-loop held-out (stream.heldout_every; read back by heldout_start_check, afterwards touched only by the
        # dispatcher thread)
        self.heldout_every = 0 if self.stream is None else self.stream.get("heldout_every", 0)    # 0 = off
        self.heldout_rounds: dict[int, dict] = {}           # nominal version -> its heldout_state.json round
        self.heldout_rows: dict[int, dict[str, dict]] = {}  # nominal version -> task -> its heldout.jsonl group row
        self.heldout_closed: set[int] = set()               # nominal versions with a kind round (summary) row
        self.heldout_queue: list[tuple[int, str]] = []      # (nominal version, task) not launched yet, launch order
        self.heldout_adopted: list[dict] = []               # stream_start_check -> dispatcher_loop: in flight, adopted
        # ---- note judge (prepare.arc3_reward note, config note_judge): at most max_concurrent judge processes; judge_stop
        # ends every judge worker (the dispatcher sets it when it exits); judge_resume: groups whose harness finished before a
        # restart and still need their judge (stream_start_check -> dispatcher_loop)
        self.note_judge: dict | None = cfg.get("note_judge")
        self.judge_slots = None if self.note_judge is None else threading.Semaphore(self.note_judge["max_concurrent"])
        self.judge_stop = threading.Event()
        self.judge_resume: list[tuple[dict, dict]] = []

    def request_detach(self) -> None:
        """Stop but leave in-flight groups running (the next orchestrator adopts them from launch.json): what SIGUSR1 does."""
        self.detaching = True
        self.cancelled.set()

    # ---- logging / cancellation
    def log(self, message: str) -> None:
        line = log_line(message)
        with self.log_lock:
            print(line, flush=True)
            if self.log_file is not None:
                self.log_file.write(line + "\n")
                self.log_file.flush()

    def check_cancel(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()
        if self.aborted.is_set():
            raise Aborted()

    def sleep(self, seconds: float) -> None:
        self.cancelled.wait(seconds)
        self.check_cancel()

    # ---- child processes (only ever signals Popen objects this process created)
    def terminate(self, procs: dict[str, subprocess.Popen]) -> list[str]:
        """SIGTERM, then SIGKILL the process group of any that has not exited within terminate_grace_seconds; returns the SIGKILLed ones."""
        live = {n: p for n, p in procs.items() if p.poll() is None}
        if not live:
            return []
        grace = self.cfg["harness"]["terminate_grace_seconds"]
        self.log(f"terminating {len(live)} child process(es) started by this orchestrator: {sorted(live)}")
        for p in live.values():
            p.terminate()
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(p.poll() is None for p in live.values()):
            time.sleep(min(1.0, self.poll))
        killed = []
        for name, p in live.items():
            if p.poll() is None:
                # every child is its own process group (popen: start_new_session; adopted harnesses alike): run_arm,
                # gateway, proxy and the outer bwrap go with it, as on the group-timeout path
                self.log(f"{name}: still running {grace}s after SIGTERM -> SIGKILL to process group {p.pid}")
                killpg_quietly(p.pid, signal.SIGKILL)
                p.wait(timeout=KILL_WAIT_SECONDS)
                killed.append(name)
        return killed

    @staticmethod
    def popen(spec: dict) -> tuple[subprocess.Popen, list]:
        """Start a child (argv/stdout[/stderr][/env]; no stderr = merged into stdout); returns (Popen, files to close)."""
        files = [open(spec["stdout"], "wb")]
        try:
            err = subprocess.STDOUT
            if spec.get("stderr") is not None:
                files.append(open(spec["stderr"], "wb"))
                err = files[-1]
            proc = subprocess.Popen(spec["argv"], cwd=ROOT, env=spec.get("env"), stdin=subprocess.DEVNULL,
                                    stdout=files[0], stderr=err, start_new_session=True)
        except BaseException:
            for f in files:
                f.close()
            raise
        return proc, files

    def run_procs(self, specs: list[dict], *, timeout: float, what: str, stagger: float = 0.0,
                  fail_fast: bool = False) -> dict[str, int]:
        """Start specs (name/argv/stdout[/stderr]), wait for all to finish, return {name: rc}.
        fail_fast: any rc≠0 immediately terminates the remaining children and raises. Timeout, cancellation and launch failure also
        terminate the children before raising."""
        procs: dict[str, subprocess.Popen] = {}
        files = []
        began = time.monotonic()
        try:
            for i, spec in enumerate(specs):
                if i and stagger:
                    self.sleep(stagger)
                self.check_cancel()
                procs[spec["name"]], opened = self.popen(spec)
                files += opened
            last_report = time.monotonic()
            while True:
                self.check_cancel()
                rcs = {n: p.poll() for n, p in procs.items()}
                bad = {n: rc for n, rc in rcs.items() if rc not in (None, 0)}
                if fail_fast and bad:
                    raise OrchestratorError(f"{what}: {bad} exited nonzero; logs: "
                                            f"{[str(s['stdout']) for s in specs if s['name'] in bad]}")
                if all(rc is not None for rc in rcs.values()):
                    return rcs
                if time.monotonic() - began > timeout:
                    raise OrchestratorError(f"{what}: not finished after {timeout:.0f}s "
                                            f"({sum(rc is None for rc in rcs.values())} still running)")
                if time.monotonic() - last_report > 600:
                    last_report = time.monotonic()
                    self.log(f"{what}: {sum(rc is not None for rc in rcs.values())}/{len(rcs)} finished, "
                             f"{time.monotonic() - began:.0f}s")
                self.sleep(min(self.poll, CHILD_POLL_SECONDS))
        except BaseException:
            self.terminate(procs)
            raise
        finally:
            for f in files:
                f.close()

    # ---- server / trainer
    def model_info(self) -> dict:
        return http_json("GET", self.server_root + "/model_info")

    def served_names(self) -> list[str]:
        return [m["id"] for m in http_json("GET", self.base_url + "/models")["data"]]

    def wait_idle(self, what: str, expected: Callable[[], int] | None = None) -> None:
        """running + waiting requests of every scheduler rank are 0 twice in a row (/v1/loads?include=core; a busy → idle transition
        forces a publish). ``expected`` (stream adopted groups that are still running): those groups' own requests are expected, so it
        counts when the sum of running + waiting over all ranks ≤ expected() twice in a row (recomputed each time: games of adopted
        groups with no results written yet, at most one request per game at a time; games the harness marked failed have no results
        file and are counted too, so the bound can only be loose); above it = someone else's traffic."""
        timeout = self.cfg["rollout"]["idle_timeout_seconds"]
        deadline = time.monotonic() + timeout
        streak = 0
        while True:
            loads = http_json("GET", self.server_root + LOADS_URL)["loads"]
            if not loads:
                raise OrchestratorError(f"{what}: /v1/loads returned no scheduler")
            busy = [(x["dp_rank"], x["num_running_reqs"], x["num_waiting_reqs"]) for x in loads
                    if x["num_running_reqs"] or x["num_waiting_reqs"]]
            total = sum(running + waiting for _, running, waiting in busy)
            allowed = 0 if expected is None else expected()
            streak = 0 if total > allowed else streak + 1
            if streak >= 2:
                if total:
                    self.log(f"{what}: {total} request(s) running or waiting, within the {allowed} unfinished episode(s) "
                             f"of the adopted group(s)")
                return
            if time.monotonic() > deadline:
                raise OrchestratorError(f"{what}: server not idle after {timeout}s; (dp_rank, running, waiting) = {busy}"
                                        + ("" if expected is None else
                                           f": {total} request(s) > the {allowed} unfinished episode(s) of the adopted "
                                           f"group(s) (someone else's traffic)"))
            self.sleep(self.poll)

    def drain_ledger(self) -> dict:
        """The ledger counts as flushed only when the writer queue is empty and the written count is unchanged twice in a row; returns
        the last /session_ledger/status."""
        timeout = self.cfg["rollout"]["ledger_drain_timeout_seconds"]
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            status = http_json("GET", self.server_root + "/session_ledger/status")
            key = (status["queue"], status["written_bytes"])
            if status["queue"] == 0 and key == previous:
                return status
            previous = key
            if time.monotonic() > deadline:
                raise OrchestratorError(f"ledger writer not drained after {timeout}s: queue={status['queue']}")
            self.sleep(self.poll)

    def trainer_url(self) -> str:
        doc = json.loads(cfg_path(self.cfg["trainer"]["endpoint_file"]).read_text())
        if "url" in doc:
            return doc["url"].rstrip("/")
        if "host" in doc and "port" in doc:
            return f"http://{doc['host']}:{doc['port']}"
        raise OrchestratorError(f"trainer endpoint file has neither url nor host+port: {doc}")

    def trainer(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
        return http_json(method, self.trainer_url() + path, body, timeout)

    def trainer_health(self) -> dict:
        health = self.trainer("GET", "/health")
        if health["ok"] is not True:
            raise OrchestratorError(f"trainer /health not ok: {health}")
        return health

    # ---- guards
    @staticmethod
    def free_gib(path: str) -> float:
        return shutil.disk_usage(path).free / 2 ** 30

    def disk_short(self, floor: float) -> list[str]:
        """disk_guard_paths with less than ``floor`` GiB free (``"<path>: N GiB free"``); empty = all have enough."""
        frees = [(path, self.free_gib(path)) for path in self.cfg["disk_guard_paths"]]
        return [f"{path}: {free:.0f} GiB free" for path, free in frees if free < floor]

    def guard_disk(self) -> None:
        short = self.disk_short(self.cfg["min_free_gb"])
        if short:
            raise OrchestratorError(f"{short[0]} < min_free_gb {self.cfg['min_free_gb']}")

    def guard_prepare_disk(self, rd: Path) -> None:
        """Same routing/workspace byte reservation as prepare_rl; include its output filesystem."""
        pcfg = self.cfg["prepare"]
        required = required_free_bytes(pcfg.get("max_gb", DEFAULT_ROUTING_GB),
                                       pcfg.get("reserve_gb", DEFAULT_RESERVE_GB))
        paths = list(dict.fromkeys([*self.cfg["disk_guard_paths"], str(rd)]))
        frees = [(p, self.free_gib(p)) for p in paths]
        short = [f"{p}: {free:.1f} GiB free" for p, free in frees if free * 2 ** 30 < required]
        if short:
            raise OrchestratorError(f"{'; '.join(short)} < prepare required_free_bytes={required} "
                                    f"({required / 2 ** 30:.3f} GiB): insufficient routing/workspace reservation")

    def static_checks(self, *, salted: bool | None = None) -> None:
        """No network: the files and task lists the config refers to exist, and the harness config has a shape this orchestrator can use.
        salted: the harness config must have ``cache_salt: "episode"`` (True) or no salt (False); None = required only when the
        pipeline is stream (heldout_offline.py swaps in an unsalted held-out config for a stream run and passes False)."""
        cfg, tasks = self.cfg, self.cfg["tasks"]
        want_salt = self.stream is not None if salted is None else salted
        for key in ("script", "agent_python", "driver_bin", "config"):
            if not cfg_path(cfg["harness"][key]).exists():
                raise OrchestratorError(f"harness.{key} {cfg['harness'][key]} does not exist")
        if not (cfg_path(cfg["tokenizer_dir"]) / "tokenizer.json").is_file():
            raise OrchestratorError(f"{cfg['tokenizer_dir']}/tokenizer.json does not exist")
        harness = json.loads(cfg_path(cfg["harness"]["config"]).read_text())
        # always bwrap: each episode gets its own jail (no network, sees only its own workspace), socket and port, and no uid (the
        # samples_per_task runs of a batch share this config, so posix fence's uid_base would collide). arc3 single level is
        # run_arc3_level.py + work/arc_code/jail.sh; arc2 is run_arc2.py + jail_arc2.sh (since 2026-10-22; before that, fence none + an
        # arc2-jail wrapper outside the repo had no network cut-off, audit P6 of 2026-10-18)
        if harness["fence"] != "bwrap":
            raise OrchestratorError(f"harness config fence must be 'bwrap' (kind {harness.get('kind')!r}), "
                                    f"not {harness['fence']!r}: the episode jail has to cut the network")
        if harness.get("tito") != {"required": True}:
            raise OrchestratorError("harness config needs tito {required: true}: the ledger is the training data")
        # eff / c0 / official shape a 0/1 level-clear reward; rank multiplies onto score, and also accepts g2seed's κ score (in 0–1)
        mode = arc3_mode(cfg["prepare"])
        kinds = ((ARC3_LEVEL_KIND, ARC3_G2_KIND) if mode == "rank" else (ARC3_G2_KIND,) if mode in ("hinge", "note")
                 else (ARC3_LEVEL_KIND,))
        if mode is not None and harness.get("kind") not in kinds:
            raise OrchestratorError(f"prepare.arc3_reward {mode} is for arc3 single-level episodes (harness config kind "
                                    f"{' or '.join(map(repr, kinds))}), not kind {harness.get('kind')!r}")
        # note: only games that end at the first window change have a note to judge (run_arc3_level's stop_after_first_window). Both
        # directions hold only for the training harness (want_salt: stream's harness-train); evaluation harnesses (unsalted:
        # baseline_b200, heldout_offline*, the harness-eval that parallel_eval_b200 swaps in) play the whole level as deployed and must
        # never have it (configure_b200 writes it only into harness-train)
        first_window = harness.get("stop_after_first_window")
        if want_salt and (mode == "note") != (first_window is True):
            raise OrchestratorError(f"prepare.arc3_reward {mode or 'score'} with harness stop_after_first_window "
                                    f"{first_window!r}: note goes with it (episodes end at their first window change and "
                                    f"their note is judged), and only note")
        if not want_salt and "stop_after_first_window" in harness:
            raise OrchestratorError(f"the evaluation harness config {cfg['harness']['config']} has stop_after_first_window "
                                    f"{first_window!r}: evaluation plays whole levels (it is harness-train only)")
        # run_arc2: cache_salt is optional ("none" when absent); "episode" requires --salt-prefix, which only stream passes
        if (harness.get("cache_salt", "none") == "episode") != want_salt:
            raise OrchestratorError(f"harness config cache_salt is {harness.get('cache_salt', 'none')!r}: pipeline stream needs "
                                    f"'episode' (each episode its own prefix-cache salt, so syncs never flush) and only it "
                                    f"passes --salt-prefix")
        self.harness_cfg = harness
        for name in ("train", "heldout"):
            spec = tasks[name]
            tdir = cfg_path(spec["dir"])
            ids = read_ids(cfg_path(spec["ids_file"]))
            absent = [t for t in ids if not (tdir / f"{t}.json").is_file()]
            if absent:
                raise OrchestratorError(f"tasks.{name}: {len(absent)} ids have no {tdir}/<id>.json, e.g. {absent[:3]}")
            self.pools[name] = ids
        if self.stream is None and tasks["heldout_baseline_file"] is not None:
            baseline = json.loads(cfg_path(tasks["heldout_baseline_file"]).read_text())
            self.heldout_baseline = {"file": tasks["heldout_baseline_file"],
                                     **{k: v for k, v in baseline.items() if k != "per_task_mean"}}
        overlap = set(self.pools["train"]) & set(self.pools["heldout"])
        if overlap:          # stream: no held-out in the loop, but its offline eval set must stay out of training
            raise OrchestratorError(f"held-out tasks are also in the train list: {sorted(overlap)[:5]}")
        # stream.retire_full_groups: tasks that had a group with R = 1 on every episode are not dispatched again
        # (recomputed from groups.jsonl at the stream start; retire_task adds the ones that finish later)
        self.retired: set[str] = set()
        self.task_phases = []
        for phase in self.stream.get("task_phases", []) if self.stream is not None else ():
            path = cfg_path(phase["priority_ids_file"])
            ids = read_ids(path)
            heldout = sorted(set(ids) & set(self.pools["heldout"]))
            outside = sorted(set(ids) - set(self.pools["train"]))
            if heldout or outside:
                raise OrchestratorError(f"stream task phase {phase['start_gid']}: priority ids must be a train subset "
                                        f"without held-out tasks: held-out {heldout[:5]}, nontrain {outside[:5]}")
            self.task_phases.append({**phase, "priority_ids": ids,
                                     "priority_ids_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if self.stream is None and len(self.pools["train"]) < tasks["tasks_per_round"]:
            raise OrchestratorError(f"train list has {len(self.pools['train'])} tasks < tasks_per_round")

    # ---- history
    def load_history(self) -> list[dict]:
        """Rows of rounds.jsonl (sync / async) or batches.jsonl (stream); numbering must be 0..n-1."""
        path, key = self.root / self.history_name, self.history_key
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [r[key] for r in rows] != list(range(len(rows))):
            raise OrchestratorError(f"{path}: {key}s are not 0..{len(rows) - 1} in order: {[r[key] for r in rows]}")
        return rows

    def row_dir(self, row: dict) -> Path:
        if self.stream is not None:
            return self.root / "batches" / f"b{row['batch']:06d}"
        return self.root / "rounds" / f"r{row['round']:03d}"

    def append_row(self, row: dict) -> None:
        """Write one row to rounds.jsonl (stream: batches.jsonl), then clean up per ``retention`` (a round counts as complete only once
        its row is written)."""
        if self.cfg.get("retention") is not None and self.retention_rows is None:
            self.retention_rows = self.load_history()
        append_jsonl(self.root / self.history_name, row)
        if self.retention_rows is not None:
            self.retention_rows.append(row)
        self.apply_retention(self.retention_rows)

    def apply_retention(self, rows: list[dict] | None = None) -> None:
        """Durable cursors clean newly committed payloads and expired exports only.

        keep_merged protects recent serving weights; optional keep_eval bounds evaluation exports.
        Only paths recorded in committed history are removed.
        """
        ret = self.cfg.get("retention")
        if ret is None:
            return
        began = time.monotonic()
        rows = self.load_history() if rows is None else rows
        keep_every = self.stream["eval_keep_every"] if self.stream is not None else 0
        policy = {**ret, "eval_keep_every": keep_every}
        state_path = self.root / "retention_state.json"
        cursor = json.loads(state_path.read_text()) if state_path.exists() else {}
        if cursor.get("policy") != policy:
            cursor = {"policy": policy, "payload_done": 0, "merged_done": 0, "eval_pending": []}
        old_end = max(len(rows) - ret["keep_merged"], 0)
        if not (0 <= cursor["payload_done"] <= len(rows) and 0 <= cursor["merged_done"] <= old_end):
            raise OrchestratorError(f"retention cursor exceeds committed history: {cursor}")
        freed, removed = 0, []
        if ret["drop_batch_payload"]:
            for row in rows[cursor["payload_done"]:]:
                # a group_ready row: every packet that prepared data (an EMPTY_ELIGIBLE packet wrote no train/)
                trains = ([Path(p["batch_dir"]) / "train" for p in row["group_ready"]["packets"] if not p["empty"]]
                          if "group_ready" in row else [self.row_dir(row) / "batch" / "train"])
                for train in trains:
                    if not train.is_dir():
                        raise OrchestratorError(f"retention: {train} is missing for a completed {self.history_key}")
                    for f in sorted(train.iterdir()):
                        if f.suffix in BATCH_PAYLOAD_SUFFIXES and f.is_file():
                            freed += f.stat().st_size
                            f.unlink()
                            removed.append(str(f))
        def remove_merged(row: dict) -> None:
            nonlocal freed
            merged = Path(row["publish"]["path"])
            if not MERGED_DIR.fullmatch(merged.name) or merged.parent.name != "merged":
                raise OrchestratorError(f"retention: {self.history_key} {row[self.history_key]} published {merged}, "
                                        f"not a <out>/merged/vNNNNNN directory")
            if merged.exists():
                freed += sum(f.stat().st_size for f in merged.rglob("*") if f.is_file())
                shutil.rmtree(merged)
                removed.append(str(merged))

        for row in rows[cursor["merged_done"]:old_end]:
            merged = Path(row["publish"]["path"])
            if not MERGED_DIR.fullmatch(merged.name) or merged.parent.name != "merged":
                raise OrchestratorError(f"retention: {self.history_key} {row[self.history_key]} published {merged}, "
                                        f"not a <out>/merged/vNNNNNN directory")
            if keep_every:
                version = int(row["weight_version_out"])
                if merged.name != f"v{version:06d}":
                    raise OrchestratorError(f"retention: {self.history_key} {row[self.history_key]} published version "
                                            f"{version} at {merged}")
                if version % keep_every == 0:
                    if "keep_eval" in ret:
                        cursor["eval_pending"].append(row[self.history_key])
                        while len(cursor["eval_pending"]) > ret["keep_eval"]:
                            remove_merged(rows[cursor["eval_pending"].pop(0)])
                    continue                # eval exports can optionally be bounded by retention.keep_eval
            remove_merged(row)
        if "keep_eval" in ret and keep_every:
            recent_eval = sum(int(row["weight_version_out"]) % keep_every == 0 for row in rows[old_end:])
            while len(cursor["eval_pending"]) > max(ret["keep_eval"] - recent_eval, 0):
                remove_merged(rows[cursor["eval_pending"].pop(0)])
        cursor.update(payload_done=len(rows), merged_done=old_end,
                      retention_seconds=round(time.monotonic() - began, 4))
        atomic_write(state_path, json.dumps(cursor, indent=1) + "\n")
        if removed:
            self.log(f"retention: removed {len(removed)} path(s), {freed / 2 ** 30:.1f} GiB "
                     f"(e.g. {removed[0]}{' ... ' + removed[-1] if len(removed) > 1 else ''})")

    # ---- top level
    def run(self) -> str:
        """Returns ``done`` (all rounds finished) or ``stopped`` (STOP file)."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock = open(self.root / "orchestrator.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OrchestratorError(f"{self.root}: another orchestrator holds orchestrator.lock") from error
        lock.write(str(os.getpid()))
        lock.flush()
        self.log_file = open(self.root / "orchestrator.log", "a")
        try:
            self.static_checks()
            history = self.load_history()
            r = len(history)
            self.log(f"run_root {self.root}: {len(history)} {self.history_key}(s) done, target {self.cfg['rounds']}, "
                     f"pipeline {self.cfg['pipeline']}")
            if self.cfg["pipeline"] == "async":
                return self.run_async(len(history))
            if self.cfg["pipeline"] == "stream":
                return self.run_stream(history)
            while r < self.cfg["rounds"]:
                if (self.root / "STOP").exists():
                    self.log(f"STOP file present: stopping before round {r}")
                    return "stopped"
                self.guard_disk()
                self.prepare_round_dir(r)
                row = self.run_round(r)
                self.append_row(row)
                self.log(f"round {r} done: reward {row['train']['reward_mean']:.4f} live {row['train']['live_fraction']:.2f} "
                         f"total {row['seconds']['total']:.0f}s")
                r += 1
            return "done"
        finally:
            self.log_file.close()
            self.log_file = None
            lock.close()

    def prepare_round_dir(self, r: int) -> None:
        """rounds/rNNN exists but is not in rounds.jsonl = last attempt did not finish; moving it aside to redo it requires
        --abandon-incomplete-round."""
        rd = self.root / "rounds" / f"r{r:03d}"
        if not rd.exists() or not self.abandon_incomplete:
            return
        state = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else {}
        if state.get("train", {}).get("metrics") or state.get("train", {}).get("inflight") or "publish" in state.get("stages", {}):
            raise OrchestratorError(f"{rd}: the trainer already took steps in this round; abandoning it would desync "
                                    f"trainer and server. Resume instead (or fix by hand).")
        target = rd.with_name(f"{rd.name}.aborted-{time.strftime('%Y%m%dT%H%M%S')}")
        rd.rename(target)
        self.log(f"abandoned incomplete round dir -> {target}")

    # ---- one round
    def heldout_due(self, r: int) -> bool:
        tcfg = self.cfg["tasks"]
        return r % tcfg["heldout_every"] == 0 and (r > 0 or tcfg["heldout_at_round0"])

    def open_round(self, r: int, *, fresh: bool = False) -> tuple[Path, list[str], bool, dict, Path]:
        """Create (or reattach to) rounds/rNNN and its state.json. fresh: the dir must not exist yet (async resumes only from a clean
        point)."""
        cfg = self.cfg
        tcfg = cfg["tasks"]
        rd = self.root / "rounds" / f"r{r:03d}"
        tasks = tasks_for_round(self.pools["train"], tcfg["tasks_per_round"], cfg["seed"], r, tcfg["train"]["shuffle"])
        heldout_due = self.heldout_due(r)
        state_path = rd / "state.json"
        if rd.exists():
            if fresh:
                raise OrchestratorError(f"{rd} already exists: a round that rounds.jsonl does not have was started before")
            if not state_path.exists():
                raise OrchestratorError(f"{rd} exists without state.json: not created by this orchestrator, "
                                        f"move it aside or pass --abandon-incomplete-round")
            state = json.loads(state_path.read_text())
            if state["round"] != r or state["tasks"] != tasks or state["heldout_due"] != heldout_due:
                raise OrchestratorError(f"{state_path}: does not match this config (round/tasks/held-out schedule changed)")
            self.log(f"round {r}: resuming; stages done: {sorted(state['stages'])}")
        else:
            (rd / "logs").mkdir(parents=True)
            state = {"round": r, "tasks": tasks, "heldout_due": heldout_due, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
                     "stages": {}, "timings": {}, "train": {"metrics": [], "inflight": None}}
            atomic_write(state_path, json.dumps(state, indent=1) + "\n")
        return rd, tasks, heldout_due, state, state_path

    def make_stage(self, r: int, state: dict, state_path: Path, label: str = "round"):
        """``stage(name, fn)``: skip if already in the state, otherwise run fn and write its result and seconds to state.json. label:
        round / batch in the log."""
        def stage(name: str, fn):
            if name in state["stages"]:
                self.log(f"{label} {r}: stage {name} already done")
                return state["stages"][name]
            self.check_cancel()
            self.log(f"{label} {r}: stage {name}")
            began = time.monotonic()
            result = fn()
            state["stages"][name] = result
            state["timings"][name] = round(time.monotonic() - began, 1)
            atomic_write(state_path, json.dumps(state, indent=1) + "\n")
            return result
        return stage

    def build_row(self, r: int, state: dict, tasks: list[str], pre: dict, heldout: dict | None, train: dict, audit: dict,
                  prepared: dict, published: dict, synced: dict, saved: dict | None, t_round: float) -> dict:
        timings = dict(state["timings"])
        timings["total"] = round(sum(timings.values()), 1)
        timings["wall_this_session"] = round(time.monotonic() - t_round, 1)
        return {"round": r, "started_utc": state["started_utc"], "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                "weight_version_in": pre["server_weight_version"], "weight_version_out": str(r + 1),
                "served_name": pre["served_name"], "served_name_after": synced["served_name"],
                "tasks": tasks, "train": {k: v for k, v in train["summary"].items() if k != "per_task_mean"},
                "per_task_mean": train["summary"]["per_task_mean"],
                "heldout": None if heldout is None else heldout["summary"],
                "heldout_baseline": self.heldout_baseline if r == 0 and heldout is None else None,
                "audit": {k: audit[k] for k in ("instance", "reports")},
                "prepare": prepared["summary"], "minibatches": prepared["minibatches"],
                "train_metrics": state["train"]["metrics"], "publish": published, "sync": synced, "save": saved,
                "seconds": timings}

    def maybe_save(self, r: int, stage):
        every = self.cfg["trainer"]["save_every_rounds"]
        if every and (r + 1) % every == 0:
            return stage("save", lambda: self.trainer("POST", "/save", {}, timeout=self.cfg["trainer"]["publish_timeout_seconds"]))
        return None

    def run_round(self, r: int) -> dict:
        rd, tasks, heldout_due, state, state_path = self.open_round(r)
        t_round = time.monotonic()
        stage = self.make_stage(r, state, state_path)
        pre = stage("preflight", lambda: self.preflight(r))
        self.served_name = pre["served_name"]
        heldout = None
        if heldout_due:
            heldout = stage("heldout", lambda: self.rollouts(rd, "heldout", self.pools["heldout"]))
        train = stage("rollout", lambda: self.rollouts(rd, "train", tasks))
        audit = stage("audit", lambda: self.audit(rd, [Path(p) for p in train["run_dirs"]]))
        prepared = stage("prepare", lambda: self.prepare_batch(
            r, rd, audit, [pre["server_weight_version"]] if self.cfg["prepare"]["weight_version_recorded"] else None))
        self.train_batch(r, rd, state, state_path, prepared)
        published = stage("publish", lambda: self.publish(r))
        synced = stage("sync", lambda: self.sync(r, published))
        saved = self.maybe_save(r, stage)
        return self.build_row(r, state, tasks, pre, heldout, train, audit, prepared, published, synced, saved, t_round)

    # ---- stages
    def preflight(self, r: int) -> dict:
        """Before a round starts: trainer idle at version r, server healthy and idle at version r (at r=0, record the version it started
        with), served model name correct."""
        cfg = self.cfg
        health = self.trainer_health()
        if health["busy"] is not False or str(health["weight_version"]) != str(r):
            raise OrchestratorError(f"round {r}: trainer must be idle at weight_version {r}, /health says {health}")
        if not http_ok(self.server_root + "/health"):
            raise OrchestratorError(f"rollout server {self.server_root} /health is not 200")
        info = self.model_info()
        names = self.served_names()
        if len(names) != 1:
            raise OrchestratorError(f"expected exactly one served model, got {names}")
        legacy_renamed = self.legacy and r > 0           # legacy sync renames the model to the merged path each round
        if not legacy_renamed and names != [cfg["rollout"]["served_name"]]:
            raise OrchestratorError(f"server serves {names}, config rollout.served_name is {cfg['rollout']['served_name']!r}")
        if info["served_model_name"] != names[0]:
            raise OrchestratorError(f"/model_info served_model_name {info['served_model_name']!r} != /v1/models {names}")
        version = str(info["weight_version"])
        if version != str(r):                            # round 0: the server starts with --weight-version 0 (launch script)
            raise OrchestratorError(f"round {r}: server weight_version is {version!r}, want {str(r)!r} "
                                    "(sync of the previous round lost, or the server was not launched with --weight-version)")
        self.wait_idle(f"round {r} start")
        self.guard_disk()
        return {"served_name": names[0], "server_weight_version": version, "trainer": health}

    def rollouts(self, rd: Path, kind: str, tasks: list[str]) -> dict:
        tcfg, hcfg = self.cfg["tasks"], self.cfg["harness"]
        base = rd if kind == "train" else rd / "heldout"
        samples = tcfg["samples_per_task"] if kind == "train" else tcfg["heldout_samples"]
        base.mkdir(exist_ok=True)
        for k in range(samples):
            if (base / f"s{k}").exists():
                raise OrchestratorError(f"{base / f's{k}'} exists but the {kind} rollout stage is not recorded as done "
                                        f"(harness died mid-run?). Inspect it; --abandon-incomplete-round moves the round aside.")
        # the round in the messages: under tail_overlap two rounds' harness runs are live at once
        return self.sample_runs(base, rd / "logs", "" if kind == "train" else "heldout-", cfg_path(tcfg[kind]["dir"]), tasks,
                                samples, timeout=hcfg["round_timeout_seconds"], what=f"{rd.name} {kind}")

    def sample_runs(self, base: Path, logs: Path, prefix: str, tasks_dir: Path, tasks: list[str], samples: int, *,
                    timeout: float, what: str) -> dict:
        """``samples`` classic-mode ``run_arc2.py`` processes (``base/sK``, stdout in ``logs/<prefix>sK.out``) run the same tasks
        concurrently, starts staggered by launch_stagger_seconds; any rc≠0 terminates the rest and raises. Returns the run dirs and
        summarize().
        Offline held-out (heldout_offline.py) also goes through here: the same argv as the in-loop held-out."""
        specs, run_dirs = [], []
        for k in range(samples):
            run_dir = base / f"s{k}"
            if run_dir.exists():
                raise OrchestratorError(f"{run_dir} already exists: a sample run directory is never reused")
            run_dirs.append(run_dir)
            specs.append({"name": f"{prefix}s{k}", "argv": self.harness_argv(run_dir, tasks_dir, tasks, []),
                          "stdout": logs / f"{prefix}s{k}.out"})
        began = time.monotonic()
        self.run_procs(specs, timeout=timeout, what=f"{what} rollouts", fail_fast=True,
                       stagger=self.cfg["harness"]["launch_stagger_seconds"])
        episodes = read_episodes(run_dirs, tasks)
        summary = summarize(episodes, samples)
        self.log(f"{what}: {summary['episodes']} episodes in {time.monotonic() - began:.0f}s, reward {summary['reward_mean']:.4f}, "
                 f"live {summary['live_fraction']:.2f}, budget-hit {summary['budget_hit_rate']:.3f}")
        return {"run_dirs": [str(d) for d in run_dirs], "summary": summary}

    def harness_argv(self, run_dir: Path, tasks_dir: Path, tasks: list[str], extra: list[str],
                     backends: list[str] | None = None) -> list[str]:
        """argv of one ``run_arc2.py`` (``extra`` goes after ``--only``; used by stream's group mode). Stream groups with
        replica_urls (training ``groups/gNNNNNN`` and in-loop held-out ``heldout/vNNNNNN/<task>``) must pass ``backends`` (one per
        sample, picked by launch_group / launch_heldout_group from each replica's running-episode count): ``--model-base-url`` is the
        first, ``--sample-backends`` all of them; otherwise base_url is used."""
        hcfg = self.cfg["harness"]
        replicas = self.cfg["rollout"].get("replica_urls")
        heldout_group = run_dir.parent.parent == self.root / HELDOUT_DIR and HELDOUT_ROUND_DIR.fullmatch(run_dir.parent.name)
        if replicas and (GROUP_DIR.fullmatch(run_dir.name) or heldout_group):
            if not backends or any(b not in replicas for b in backends):
                raise OrchestratorError(f"{run_dir.name}: stream group needs one backend per sample from "
                                        f"rollout.replica_urls, got {backends!r}")
            backend, per_sample = backends[0], ["--sample-backends", ",".join(backends)]
        elif backends is not None:
            raise OrchestratorError(f"{run_dir.name}: backends {backends!r} given outside a stream group with replica_urls")
        else:
            backend, per_sample = self.base_url, []
        return [str(cfg_path(hcfg["agent_python"])), str(cfg_path(hcfg["script"])),
                "--config", str(cfg_path(hcfg["config"])), "--run-dir", str(run_dir), "--tasks-dir", str(tasks_dir),
                "--only", ",".join(tasks), *extra, *per_sample, "--agent-python", str(cfg_path(hcfg["agent_python"])),
                "--driver-bin", str(cfg_path(hcfg["driver_bin"])), "--model-base-url", backend,
                "--model", self.served_name, "--tokenizer-json", str(cfg_path(self.cfg["tokenizer_dir"]) / "tokenizer.json")]

    def audit(self, rd: Path, run_dirs: list[Path]) -> dict:
        self.wait_idle("before audit")
        return self.audit_with_status(rd, run_dirs, self.drain_ledger())

    def audit_with_status(self, rd: Path, run_dirs: list[Path], status: dict) -> dict:
        """``status``: snapshot of ``/session_ledger/status`` (sync: after idle and an empty queue; async: after the ledger is flushed)."""
        cfg, acfg = self.cfg, self.cfg["audit"]
        statuses = ledger_statuses(status)
        instances = [ledger_instance_dir(row) for row in statuses]
        names = {row["instance"] for row in statuses}
        for instance in instances:
            if not instance.is_dir():
                raise OrchestratorError(f"ledger instance dir {instance} (from /session_ledger/status) does not exist")
        status_path = rd / "ledger_status.json"
        atomic_write(status_path, json.dumps(statuses if len(statuses) > 1 else status, indent=1) + "\n")
        manifests = []
        for run_dir in run_dirs:
            path = run_dir / "tito_manifest.json"
            if not path.is_file():
                raise OrchestratorError(f"{run_dir}: no tito_manifest.json (harness did not write it)")
            for server in manifest_servers(json.loads(path.read_text())):
                if server["instance"] not in names or server["instances"] != [server["instance"]]:
                    raise OrchestratorError(f"{path}: ledger instance {server['instances']} != the server's current "
                                            f"{sorted(names)} (server restarted during the run?)")
            manifests.append(path)
        groups = [[m] for m in manifests] if acfg["per_run"] else [manifests]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        reports = []
        for i, group in enumerate(groups):
            tag = group[0].parent.name if acfg["per_run"] else "all"
            report_path = rd / f"audit-{tag}.json"
            argv = [*acfg["command"], "audit", *map(str, instances), "--tokenizer", str(cfg_path(cfg["tokenizer_dir"])),
                    *[a for m in group for a in ("--manifest", str(m))],
                    "--ledger-status", str(status_path), "--json", str(report_path)]
            if acfg["command"][-2:] == ["-m", "arcforge.tito"]:
                argv.append("--manifest-threads-only")
            rcs = self.run_procs([{"name": f"audit-{tag}", "argv": argv, "env": env,
                                   "stdout": rd / "logs" / f"audit-{tag}.txt"}],
                                 timeout=acfg["timeout_seconds"], what=f"audit {tag}")
            rc = rcs[f"audit-{tag}"]
            if rc not in (0, 1) or not report_path.is_file():
                raise OrchestratorError(f"audit {tag}: rc={rc}, report {'present' if report_path.is_file() else 'missing'}; "
                                        f"see {rd / 'logs' / ('audit-' + tag + '.txt')}")
            report = json.loads(report_path.read_text())
            failures = report["failures"]
            if report["collection_passed"] != (rc == 0) or bool(failures) == (rc == 0):
                raise OrchestratorError(f"audit {tag}: rc={rc} contradicts the report "
                                        f"(collection_passed={report['collection_passed']}, {len(failures)} failures)")
            blocking = [f for f in failures if not any(w in f for w in acfg["whitelist"])]
            if any(f.startswith(UNEXPECTED_SPLITS_FAILURE) for f in blocking) and acfg["allow_splits"]:
                over = splits_over_allowance(report["split_reasons"], report["expected_splits"], acfg["allow_splits"])
                blocking = [f + ("" if not f.startswith(UNEXPECTED_SPLITS_FAILURE) else f" (audit.allow_splits: {over})")
                            for f in blocking if not (f.startswith(UNEXPECTED_SPLITS_FAILURE) and not over)]
            if blocking and acfg["allow_budget_truncation"]:
                rest = [f for f in blocking if not f.startswith(RID_JOIN_FAILURE)]
                if len(blocking) > len(rest):                 # the rid-join line is there: judge it by the structured rule
                    runs = [m.parent for m in group]
                    reasons = budget_truncation_blockers(report["rid_join"],
                                                         {d.name: run_threads(m) for d, m in zip(runs, group)},
                                                         {d.name: budget_stops(d) for d in runs},
                                                         {d.name: wrapup_cut_rids(d) | zombie_rids(d) for d in runs})
                    blocking = [f"{RID_JOIN_FAILURE}… not benign: " + "; ".join(reasons)] + rest if reasons else rest
            if blocking:
                raise AuditFailed(f"audit {tag} FAILED:\n  " + "\n  ".join(blocking)
                                  + f"\nreport {report_path}. Only audit.allow_budget_truncation and audit.whitelist can excuse a failure",
                                  blocking, rd, report_path)
            if failures:
                self.log(f"audit {tag}: excused failure line(s): {failures}")
            reports.append({"tag": tag, "report": str(report_path), "failures_whitelisted": failures,
                            "warnings": len(report["warnings"])})
        return {"instance": status["instance"], "instance_dir": str(instances[0]),
                "instance_dirs": list(map(str, instances)), "manifests": [str(m) for m in manifests],
                "ledger_status": {k: status[k] for k in ("queue", "pending", "lossy", "writer_error")}, "reports": reports}

    def prepare_batch(self, r: int, rd: Path, audit: dict, versions: list[str] | None, *,
                      window: tuple[int, int] | None = None, extra: tuple[str, ...] = (),
                      launch_versions: dict[str, int] | None = None) -> dict:
        """``versions``: weight versions allowed in this batch (one for sync; k-1,k for async); None = the ledger records no version
        (``unversioned``). ``window``: stream's (V, S): prepare accepts tokens of versions [V−S, V] and drops older episodes as
        ``stale`` (versions must be None). ``launch_versions``: {manifest: this group's dispatch version}, required during arc3 rank
        warm-up (arc3_rank_warmup_versions > 0) and must cover every manifest of the audit; written as
        ``<rd>/arc3_launch_versions.json`` for prepare_rl (only stream.group_ready can provide it)."""
        pcfg = self.cfg["prepare"]
        batch = rd / "batch"
        if batch.exists():
            raise OrchestratorError(f"{batch} exists but the prepare stage is not recorded as done; prepare_rl wants a new directory")
        self.guard_prepare_disk(rd)
        if window is None:
            version = "unversioned" if versions is None else ",".join(versions)
            version_args = ["--weight-version", version]
        else:
            if versions is not None:
                raise OrchestratorError(f"prepare_batch: a staleness window {window} and a version set {versions} are exclusive")
            version_args = ["--current-version", str(window[0]), "--max-staleness", str(window[1])]
        argv = [*pcfg["command"], "--ledger", *audit.get("instance_dirs", [audit["instance_dir"]]),
                *[a for m in audit["manifests"] for a in ("--manifest", m)], "--out", str(batch), "--round", str(r),
                *version_args, "--temperature", str(pcfg["temperature"]),
                "--tokenizer-dir", str(cfg_path(self.cfg["tokenizer_dir"])), "--minibatches", str(pcfg["minibatches"]),
                "--world", str(pcfg["world"]), "--bin-tokens", str(pcfg["bin_tokens"]), "--seed", str(self.cfg["seed"] + r),
                "--algo", prepare_algo(pcfg),
                *[a for k in PREPARE_ALGOS[prepare_algo(pcfg)] for a in ("--" + k.replace("_", "-"), str(pcfg[k]))]]
        for key in ("max_gb", "reserve_gb", "adv_normalization", "loss_reduction", "wrapup_loss", "efficiency_beta",
                    "efficiency_budget",
                    "efficiency_guard", *TOOL_KEYS, "max_d_cross_fraction", "max_routing_m_fraction", "arc3_reward", *ARC3_KEYS,
                    "top_p", "top_k"):
            if key in pcfg:
                argv.extend(["--" + key.replace("_", "-"), str(pcfg[key])])
        if pcfg.get("allow_compaction", False):    # whole-game episodes (harness kind arc3_game): windows are segments
            argv.append("--allow-compaction")
        if (arc3_mode(pcfg) == "rank" and pcfg["arc3_rank_warmup_versions"] > 0) or arc3_mode(pcfg) == "hinge":
            if launch_versions is None or set(launch_versions) != {str(m) for m in audit["manifests"]}:
                raise OrchestratorError(f"{rd}: arc3 rank warmup / hinge needs every manifest's launch version (have "
                                        f"{None if launch_versions is None else sorted(launch_versions)[:3]}, manifests "
                                        f"{[str(m) for m in audit['manifests']][:3]})")
            lv = rd / "arc3_launch_versions.json"
            lv.write_text(json.dumps(launch_versions, indent=1, sort_keys=True) + "\n")
            argv.extend(["--arc3-launch-versions", str(lv)])
        elif launch_versions is not None:
            raise OrchestratorError(f"{rd}: launch versions given without arc3 rank warmup or hinge")
        error_json = rd / "logs" / "prepare.error.json"
        if error_json.exists():                    # an earlier attempt of this stage (resume): keep it as evidence
            earlier = error_json.with_name(f"prepare.error.{time.strftime('%Y%m%dT%H%M%S')}.json")
            if earlier.exists():
                raise OrchestratorError(f"{earlier} already exists: two prepare attempts of {rd} in one second")
            error_json.rename(earlier)
            self.log(f"{rd}: an earlier prepare attempt's {error_json.name} -> {earlier.name}")
        argv.extend([*extra, "--error-json", str(error_json)])
        out, err = rd / "logs" / "prepare.stdout", rd / "logs" / "prepare.stderr"
        rc = self.run_procs([{"name": "prepare_rl", "argv": argv, "stdout": out, "stderr": err}],
                            timeout=pcfg["timeout_seconds"], what="prepare_rl")["prepare_rl"]
        kinds = {code: kind for kind, code in PREPARE_EXIT_CODES.items()}
        if rc in kinds:
            if not error_json.is_file():
                raise OrchestratorError(f"prepare_rl rc={rc} ({kinds[rc]}) but no {error_json}; see {err}")
            doc = json.loads(error_json.read_text())
            if doc["kind"] != kinds[rc]:
                raise OrchestratorError(f"prepare_rl rc={rc} ({kinds[rc]}) but {error_json} says kind {doc['kind']!r}")
            raise PrepareRejected(f"prepare_rl {doc['kind']}: {doc['message']} (see {error_json}, {err})", doc["kind"], doc,
                                  error_json)
        if rc != 0:
            raise OrchestratorError(f"prepare_rl rc={rc}; see {err}")
        if error_json.exists():
            raise OrchestratorError(f"prepare_rl rc=0 but it wrote {error_json}")
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        try:
            summary = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise OrchestratorError(f"prepare_rl: last stdout line is not a JSON summary ({out})") from error
        if not isinstance(summary, dict):
            raise OrchestratorError(f"prepare_rl: summary is not an object: {summary!r}")
        if summary.get("status") == "EMPTY_ELIGIBLE":
            if (window is None or (summary["current_version"], summary["max_staleness"]) != window
                    or summary["round"] != r or summary["episodes_kept"] != 0 or summary["groups_kept"] != 0):
                raise OrchestratorError(f"prepare_rl: invalid EMPTY_ELIGIBLE summary: {summary}")
            return {"status": "EMPTY_ELIGIBLE", "batch_dir": str(batch), "minibatches": 0, "summary": summary}
        manifest = json.loads((batch / "manifest.json").read_text())
        if manifest["format"] != ROLLOUT_FORMAT:
            raise OrchestratorError(f"{batch}: format {manifest['format']!r} != {ROLLOUT_FORMAT}")
        rl = manifest["rl"]
        if rl["round"] != r:
            raise OrchestratorError(f"{batch}: manifest rl.round {rl['round']} != {r}")
        got = rl["weight_version"] if isinstance(rl["weight_version"], list) else [rl["weight_version"]]
        if window is not None:
            current, staleness = window
            low = max(current - staleness, 0)
            outside = [g for g in got if not (str(g).isdigit() and low <= int(g) <= current)]
            if (rl["current_version"], rl["max_staleness"]) != window or not got or outside:
                raise OrchestratorError(f"{batch}: manifest rl current_version/max_staleness {rl['current_version']!r}/"
                                        f"{rl['max_staleness']!r} weight_version {rl['weight_version']!r}; want {current}/"
                                        f"{staleness} and versions in [{low}, {current}]")
        elif versions is not None:
            if sorted(str(g) for g in got) != sorted(versions):
                raise OrchestratorError(f"{batch}: manifest rl.weight_version {rl['weight_version']!r} != {version!r}")
        n = len(rl["minibatches"])
        if not 1 <= n <= pcfg["minibatches"]:
            raise OrchestratorError(f"{batch}: {n} minibatches (config minibatches={pcfg['minibatches']}); 0 means no live group "
                                    f"to learn from. Prepare summary: {summary}")
        return {"batch_dir": str(batch), "minibatches": n, "summary": summary}

    def train_batch(self, r: int, rd: Path, state: dict, state_path: Path, prepared: dict,
                    expect_version: int | None = None, label: str = "round") -> None:
        """One /train per minibatch. An in-flight call (inflight written but no result written back) cannot be judged as consumed or
        not, so a restart raises. expect_version: the trainer must be at this version (used by the async / stream learner; sync already
        checked it in the round-start preflight)."""
        train = state["train"]
        if "train" in state["stages"]:
            self.log(f"{label} {r}: stage train already done")
            return
        if train["inflight"] is not None:
            raise OrchestratorError(f"{state_path}: /train for minibatch {train['inflight']['minibatch']} was in flight when the "
                                    f"orchestrator died; compare trainer /health step with step_before="
                                    f"{train['inflight']['step_before']} and fix state.json by hand")
        timeout = self.cfg["trainer"]["train_timeout_seconds"]
        began = time.monotonic()
        for i in range(len(train["metrics"]), prepared["minibatches"]):
            self.check_cancel()
            health = self.trainer_health()
            if health["busy"] is not False:
                raise OrchestratorError(f"trainer busy before minibatch {i}: {health}")
            if expect_version is not None and str(health["weight_version"]) != str(expect_version):
                raise OrchestratorError(f"{label} {r}: trainer is at weight_version {health['weight_version']!r}, "
                                        f"want {expect_version} for minibatch {i}")
            train["inflight"] = {"minibatch": i, "step_before": health["step"]}
            atomic_write(state_path, json.dumps(state, indent=1) + "\n")
            metrics = self.trainer("POST", "/train", {"batch_dir": prepared["batch_dir"], "minibatch": i}, timeout=timeout)
            assert_finite(metrics, f"/train minibatch {i}")
            after = self.trainer_health()
            if after["step"] != health["step"] + 1:
                raise OrchestratorError(f"trainer step {health['step']} -> {after['step']} after minibatch {i}, want +1")
            train["metrics"].append(metrics)
            train["inflight"] = None
            atomic_write(state_path, json.dumps(state, indent=1) + "\n")
            self.log(f"{label} {r}: minibatch {i + 1}/{prepared['minibatches']} trained")
        state["stages"]["train"] = {"minibatches": prepared["minibatches"]}
        state["timings"]["train"] = round(state["timings"].get("train", 0) + time.monotonic() - began, 1)
        atomic_write(state_path, json.dumps(state, indent=1) + "\n")

    def publish(self, r: int) -> dict:
        reply = self.trainer("POST", "/publish", {"weight_version": r + 1}, timeout=self.cfg["trainer"]["publish_timeout_seconds"])
        path = Path(reply["path"])
        if not path.is_dir():
            raise OrchestratorError(f"/publish returned {path}, which is not a directory on this machine")
        health = self.trainer_health()
        if str(health["weight_version"]) != str(r + 1):
            raise OrchestratorError(f"trainer weight_version {health['weight_version']!r} after publish, want {r + 1}")
        return {"path": str(path), "manifest_sha256": reply["manifest_sha256"], "seconds": reply["seconds"]}

    def verify_sync(self, wv: str, published: dict, before: dict, after: dict, names: list[str]) -> None:
        """/model_info check after a sync (shared by idle and mid-flight): version, name, identity unchanged, no poison, weights_source."""
        if str(after["weight_version"]) != wv:
            raise OrchestratorError(f"after sync /model_info weight_version is {after['weight_version']!r}, want {wv!r}")
        if len(names) != 1 or after["served_model_name"] != names[0]:
            raise OrchestratorError(f"after sync /model_info served_model_name {after['served_model_name']!r} vs /v1/models {names}")
        if self.legacy:
            if after["model_path"] != published["path"]:
                raise OrchestratorError(f"legacy sync: model_path is {after['model_path']!r}, want {published['path']!r}")
            self.log(f"legacy sync: served name is now {names[0]}")
        else:
            if after["served_model_name"] != before["served_model_name"] or after["model_path"] != before["model_path"]:
                raise OrchestratorError(f"keep_identity changed the identity: {before['served_model_name']!r}/{before['model_path']!r}"
                                        f" -> {after['served_model_name']!r}/{after['model_path']!r}")
            if after["poisoned"] is not False:
                raise OrchestratorError(f"server is poisoned after sync: {after['poisoned']!r}")
            if after["weights_source"] != published["path"]:
                raise OrchestratorError(f"/model_info weights_source {after['weights_source']!r} != {published['path']!r}")

    def sync(self, r: int, published: dict) -> dict:
        """server idle → update_weights_from_disk → verify /model_info. Idempotent (resending the same path only reloads)."""
        wv = str(r + 1)
        self.wait_idle("before weight sync")
        before = self.model_info()
        body = {"model_path": published["path"], "weight_version": wv, "flush_cache": True}
        if not self.legacy:
            body["keep_identity"] = True
        for attempt in range(SYNC_IDLE_RETRIES + 1):
            reply = http_json("POST", self.server_root + "/update_weights_from_disk", body,
                              timeout=self.cfg["rollout"]["sync_timeout_seconds"])
            # fork rl/v1: with flush_cache the scheduler rejects verbatim when it judges itself not idle (nothing loaded, no poison);
            # /v1/loads and the scheduler's judgement are one event loop apart, so this one is retried; other failures (poisoned
            # included) still stop
            refused_busy = (not self.legacy and reply["success"] is False and reply.get("weights_poisoned") is None
                            and SYNC_REFUSED_BUSY in reply["message"])
            if not refused_busy or attempt == SYNC_IDLE_RETRIES:
                break
            self.log(f"sync refused (scheduler not idle yet), retry {attempt + 1}/{SYNC_IDLE_RETRIES}")
            self.wait_idle("before weight sync (retry)")
        if reply["success"] is not True:
            raise OrchestratorError(f"update_weights_from_disk failed: {reply}")
        if reply["num_paused_requests"] != 0:
            raise OrchestratorError(f"update_weights_from_disk paused {reply['num_paused_requests']} request(s): "
                                    f"the server was not idle")
        after = self.model_info()
        names = self.served_names()
        self.verify_sync(wv, published, before, after, names)
        self.wait_idle("after weight sync")
        return {"request": body, "reply": reply, "served_name": names[0], "weight_version": wv,
                "weights_source": after.get("weights_source")}

    # =========================================================================== async pipeline (SPEC §8)

    def run_async(self, done: int) -> str:
        """Rollout thread and learner thread in parallel; if either dies the other stops, and the real cause is raised at the end."""
        total = self.cfg["rounds"]
        if self.abandon_incomplete:
            raise OrchestratorError("--abandon-incomplete-round is for the sync pipeline: an async run only resumes from a clean "
                                    "point, and a half-finished round is for a person to inspect and move aside")
        if done >= total:
            return "done"
        self.async_start_check(done)
        self.server_version = done
        self.spawn("rollout", lambda: self.rollout_loop(done, total))
        self.spawn("learner", lambda: self.learner_loop(done, total))
        self.join_pipeline()
        return "done" if self.pipeline_outcomes["learner"] == "done" else "stopped"

    def join_pipeline(self) -> None:
        """Wait for all pipeline threads to finish; if a thread failed, raise the real cause (not the Aborted one; cancellation ranks
        after other errors)."""
        joined = 0
        while True:                    # the rollout thread may spawn per-round threads (tail_overlap): join those too
            with self.cv:
                if joined == len(self.pipeline_threads):
                    break              # the first thread is joined by now, so nothing is spawned any more
                thread = self.pipeline_threads[joined]
            thread.join()
            joined += 1
        failures = self.pipeline_failures
        real = [(n, e) for n, e in failures if not isinstance(e, Aborted)]
        for name, error in real:
            self.log(f"{name} thread failed: {error!r}")
        if real:
            raise next((e for _, e in real if not isinstance(e, Cancelled)), real[0][1])
        if failures:
            raise OrchestratorError(f"pipeline threads aborted without a recorded cause: {failures}")

    def spawn(self, name: str, fn) -> None:
        """async thread: the exception is recorded (the main thread raises it at the end, never swallowed); unless it is Aborted, tell
        the other threads to stop (their children are terminated by run_procs itself)."""
        def guarded() -> None:
            try:
                self.pipeline_outcomes[name] = fn()
            except BaseException as error:         # recorded and re-raised by the main thread; never swallowed
                with self.cv:
                    self.pipeline_failures.append((name, error))
                    if not isinstance(error, Aborted):
                        self.aborted.set()
                    self.cv.notify_all()

        thread = threading.Thread(target=guarded, name=name)
        thread.start()                 # started before it is listed: run_async never joins a thread that did not start
        with self.cv:
            self.pipeline_threads.append(thread)

    def async_start_check(self, n: int) -> None:
        """async starts only from a clean point: rounds.jsonl has n rows, no further started rounds, trainer and server both at version
        n and idle. No guessing."""
        rounds_dir = self.root / "rounds"
        started = sorted(int(p.name[1:]) for p in rounds_dir.iterdir() if re.fullmatch(r"r\d+", p.name)) if rounds_dir.is_dir() else []
        stray = [r for r in started if r >= n]
        if stray:
            raise OrchestratorError(f"async resume: round dir(s) {stray} exist but rounds.jsonl has only {n} round(s): a round was "
                                    f"started and not finished. The trainer and server may be out of step; inspect {rounds_dir} and "
                                    f"move those directories aside by hand (async never resumes mid-round)")
        health = self.trainer_health()
        if health["busy"] is not False or str(health["weight_version"]) != str(n):
            raise OrchestratorError(f"async resume: rounds.jsonl has {n} round(s) but the trainer is not idle at weight_version {n}: "
                                    f"/health says {health}")
        if not http_ok(self.server_root + "/health"):
            raise OrchestratorError(f"rollout server {self.server_root} /health is not 200")
        version = str(self.model_info()["weight_version"])
        if version != str(n):
            raise OrchestratorError(f"async resume: rounds.jsonl has {n} round(s) but the server weight_version is {version!r}, want "
                                    f"{str(n)!r} (a sync was lost or half-done, or the server was not launched with --weight-version)")
        status = http_json("GET", self.server_root + "/session_ledger/status")
        ledger_count(status, "enqueued_events")
        ledger_count(status, "written_events")
        self.wait_idle("async start")

    # ---- rollout thread
    def rollout_loop(self, start: int, total: int) -> str:
        """Round k: STOP / disk → gate → preflight → [held-out] → register slot → harness runs until all have ended → hand to the learner.
        With tail_overlap on, the harness part runs in each round's own thread (``finish_rollout``) and this thread only waits until
        the next round can start (``await_rollout_tail``); with it off, everything runs to completion in this thread as before."""
        tail = self.cfg["tail_overlap"]
        overlap: dict | None = None            # round k starts in round k-1's tail: {previous_round, previous_remaining_episodes}
        for k in range(start, total):
            self.check_cancel()
            if (self.root / "STOP").exists():
                self.log(f"STOP file present: no rollout for round {k} and later; the learner finishes the rounds already rolled out")
                self.end_rollouts(k)
                return "stopped"
            self.guard_disk()
            began = time.monotonic()
            with self.cv:                      # the gate: round k starts only when the server is at version >= k-1
                while self.server_version < k - 1:
                    self.cv.wait(self.poll)
                    self.check_cancel()
            gate_wait = time.monotonic() - began
            rd, tasks, heldout_due, state, state_path = self.open_round(k, fresh=True)
            t_round = time.monotonic()
            stage = self.make_stage(k, state, state_path)
            pre = stage("preflight", lambda: self.preflight_async(k))
            self.served_name = pre["served_name"]
            heldout = stage("heldout", lambda: self.heldout_async(rd, k)) if heldout_due else None
            version_start, claimed = self.claim_rollout(k)
            if overlap is not None:
                self.log(f"round {k}: tail overlap: harness runs start while {overlap['previous_remaining_episodes']} episode(s) "
                         f"of round {k - 1} still run (server version {version_start})")
            job = {"rd": rd, "tasks": tasks, "state": state, "state_path": state_path, "pre": pre, "heldout": heldout,
                   "t_round": t_round, "version_start": version_start, "claimed_mono": claimed, "gate_wait_seconds": gate_wait,
                   "tail_overlap": overlap}
            if tail is None:
                self.finish_rollout(k, job)    # the next round starts after every harness run of this one exited
                continue
            self.spawn(f"rollout-r{k:03d}", lambda k=k, job=job: self.finish_rollout(k, job))
            remaining = self.await_rollout_tail(k, rd, tasks, total)
            overlap = None if remaining is None else {"previous_round": k, "previous_remaining_episodes": remaining}
        self.end_rollouts(total)
        return "done"

    def finish_rollout(self, k: int, job: dict) -> str:
        """Round k's harness runs until all have ended → record the ledger's enqueued_events (events queued after the snapshot do not
        belong to this round) → release the slot → hand to the learner. On failure, mark aborted under the same lock that releases the
        slot, so the rollout thread waiting on this slot does not start the next round when it wakes."""
        stage = self.make_stage(k, job["state"], job["state_path"])
        completed = False
        try:
            train = stage("rollout", lambda: self.rollouts(job["rd"], "train", job["tasks"]))
            enqueued = ledger_count(http_json("GET", self.server_root + "/session_ledger/status"), "enqueued_events")
            completed = True
        finally:
            version_end, later = self.release_rollout(k, completed)
        note = "".join(f"; round {j} started {s:.0f}s before this round ended (tail overlap)" for j, s in later.items())
        self.log(f"round {k}: rollout done (server version {job['version_start']} -> {version_end}), "
                 f"ledger enqueued_events {enqueued}{note}")
        with self.cv:
            self.rolled[k] = {**{key: job[key] for key in ("rd", "tasks", "state", "state_path", "pre", "heldout", "t_round",
                                                           "version_start", "claimed_mono", "gate_wait_seconds", "tail_overlap")},
                              "train": train, "enqueued_events": enqueued, "version_end": version_end,
                              "rollout_seconds": job["state"]["timings"]["rollout"], "ready_mono": time.monotonic()}
            self.cv.notify_all()
        return "done"

    def await_rollout_tail(self, k: int, rd: Path, tasks: list[str], total: int) -> int | None:
        """tail_overlap: wait until round k's harness runs have all ended (returns None), or round k+1 can start within its tail (returns
        the number of round k's episodes still running): still running ≤ remaining_episodes, and round k+1's gate (server version ≥ k)
        is open. No early start when round k+1 has a held-out: it needs v=k+1, i.e. round k fully trained, so starting early would only
        idle before the held-out. STOP is not checked here (it is checked at the start of the next round)."""
        limit = self.cfg["tail_overlap"]["remaining_episodes"]
        samples = self.cfg["tasks"]["samples_per_task"]
        run_dirs = [rd / f"s{s}" for s in range(samples)]
        early = k + 1 < total and not self.heldout_due(k + 1)
        while True:
            with self.cv:
                self.check_cancel()
                if k not in self.active_rollouts:
                    return None
                gate = self.server_version >= k
            if early and gate:
                remaining = samples * len(tasks) - episodes_done(run_dirs, tasks)
                with self.cv:
                    live = k in self.active_rollouts
                if live and remaining <= limit:
                    self.log(f"round {k}: {remaining} of {samples * len(tasks)} episode(s) still running <= "
                             f"tail_overlap.remaining_episodes {limit} and the gate is open: round {k + 1} may start")
                    return remaining
            with self.cv:
                if k in self.active_rollouts and not (self.cancelled.is_set() or self.aborted.is_set()):
                    self.cv.wait(self.poll)

    def end_rollouts(self, k: int) -> None:
        with self.cv:
            self.rollout_end = k
            self.cv.notify_all()

    def preflight_async(self, k: int) -> dict:
        """Check before rollout starts. The trainer belongs to the learner; server version in [k-1, k] (the gate guarantees the lower
        bound, and round k is not trained yet so it cannot be higher)."""
        cfg = self.cfg
        if not http_ok(self.server_root + "/health"):
            raise OrchestratorError(f"rollout server {self.server_root} /health is not 200")
        info = self.model_info()
        names = self.served_names()
        if names != [cfg["rollout"]["served_name"]] or info["served_model_name"] != names[0]:
            raise OrchestratorError(f"server serves {names} (/model_info {info['served_model_name']!r}), "
                                    f"config rollout.served_name is {cfg['rollout']['served_name']!r}")
        version = int(info["weight_version"])
        if not max(k - 1, 0) <= version <= k:
            raise OrchestratorError(f"round {k}: server weight_version is {version}, the gate allows [{max(k - 1, 0)}, {k}]")
        return {"served_name": names[0], "server_weight_version": str(version)}

    def claim_rollout(self, k: int) -> tuple[int, float]:
        """Register "round k is running" before the harness runs start; if an idle-mode sync is in progress, wait for it to finish
        (during it the server is idle and swapping weights). At most two rounds run at once: round k and round k-1's tail
        (tail_overlap); the gate guarantees earlier rounds are trained, so one still present is a bug.
        Returns (server version, monotonic time of registration)."""
        with self.cv:
            while self.sync_in_progress:
                self.cv.wait(self.poll)
                self.check_cancel()
            if set(self.active_rollouts) - {k - 1}:
                raise OrchestratorError(f"round {k}: rounds {sorted(self.active_rollouts)} still have harness runs; only round "
                                        f"{k - 1}'s tail may overlap round {k}")
            self.active_rollouts[k] = time.monotonic()
            return self.server_version, self.active_rollouts[k]

    def release_rollout(self, k: int, completed: bool) -> tuple[int, dict[int, float]]:
        """Round k's harness runs have all ended (or failed). Returns (server version, {newer rounds running at the same time: how many
        seconds earlier than this round they started})."""
        with self.cv:
            if not completed:
                self.aborted.set()             # before anyone waiting on this release wakes up and starts the next step
            del self.active_rollouts[k]
            ended = time.monotonic()
            self.rollout_ended[k] = ended
            later = {j: round(ended - t, 2) for j, t in sorted(self.active_rollouts.items()) if j > k}
            self.cv.notify_all()
            return self.server_version, later

    def heldout_async(self, rd: Path, k: int) -> dict:
        """held-out: first wait for the server to reach v=k (evaluates the same version as sync's held-out in round k, so rounds compare
        one to one; rollout pauses for the learner once every heldout_every rounds), with sync blocked until it ends; the server is idle
        before it starts, so flush_cache runs then; the server version must be the same before and after."""
        began = time.monotonic()
        with self.cv:
            while self.server_version < k or self.sync_in_progress:
                self.cv.wait(self.poll)
                self.check_cancel()
            if self.active_rollouts:           # v=k means round k-1 was learned, so its harness runs are long gone
                raise OrchestratorError(f"round {k} held-out: rounds {sorted(self.active_rollouts)} still have harness runs; "
                                        f"the held-out needs the server to itself")
            self.heldout_running = True
            tracked = self.server_version
        waited = round(time.monotonic() - began, 2)
        completed = False
        try:
            self.wait_idle(f"round {k} held-out start")
            flushed = http_text("POST", self.server_root + "/flush_cache", timeout=self.cfg["rollout"]["sync_timeout_seconds"])
            before = int(self.model_info()["weight_version"])
            if before != tracked:
                raise OrchestratorError(f"round {k} held-out: server weight_version is {before}, the last verified sync left {tracked}")
            result = self.rollouts(rd, "heldout", self.pools["heldout"])
            after = int(self.model_info()["weight_version"])
            if after != before:
                raise OrchestratorError(f"round {k} held-out: server weight_version changed {before} -> {after} while it ran")
            completed = True
        finally:
            with self.cv:
                if not completed:
                    self.aborted.set()          # before the learner's deferred sync can wake up: no sync after a bad held-out
                self.heldout_running = False
                self.cv.notify_all()
        if before != k:
            raise OrchestratorError(f"round {k} held-out ran on weight_version {before}, want {k}")
        result["summary"]["weight_version"] = str(before)
        result["summary"]["version_wait_seconds"] = waited
        result["flush_cache"] = flushed.strip()
        return result

    # ---- learner thread
    def learner_loop(self, start: int, total: int) -> str:
        ready = time.monotonic()
        for k in range(start, total):
            with self.cv:
                while k not in self.rolled:
                    if self.rollout_end is not None and k >= self.rollout_end:
                        return "stopped"
                    self.cv.wait(self.poll)
                    self.check_cancel()
                rolled = self.rolled.pop(k)
            wait = max(rolled["ready_mono"] - ready, 0.0)
            self.learn_round(k, rolled, wait)
            ready = time.monotonic()
        return "done"

    def ledger_landing(self, rolled: dict) -> dict:
        """The ledger keeps being written, so "queue empty" never comes: wait for written_events to catch up with the enqueued_events
        recorded when the rollout ended."""
        target = rolled["enqueued_events"]
        timeout = self.cfg["rollout"]["ledger_drain_timeout_seconds"]
        deadline = time.monotonic() + timeout
        while True:
            status = http_json("GET", self.server_root + "/session_ledger/status")
            written = ledger_count(status, "written_events")
            replica_targets = rolled.get("replica_events")
            if replica_targets is not None:
                actual = {row["instance"]: ledger_count(row, "written_events") for row in ledger_statuses(status)}
                if set(actual) != set(replica_targets):
                    raise OrchestratorError("ledger instances changed while waiting for replica watermarks")
                landed = all(actual[name] >= value for name, value in replica_targets.items())
            else:
                landed = written >= target
            if landed:
                return {"enqueued_events": target, "written_events": written, "status": status}
            if time.monotonic() > deadline:
                raise OrchestratorError(f"ledger landing: written_events {written} < {target} after {timeout}s "
                                        f"(queue {status['queue']}, writer_error {status['writer_error']!r})")
            self.sleep(self.poll)

    def learn_round(self, k: int, rolled: dict, learner_wait: float) -> None:
        rd, state, state_path = rolled["rd"], rolled["state"], rolled["state_path"]
        stage = self.make_stage(k, state, state_path)
        train = rolled["train"]
        landing = stage("ledger", lambda: self.ledger_landing(rolled))
        audit = stage("audit", lambda: self.audit_with_status(rd, [Path(p) for p in train["run_dirs"]], landing["status"]))
        prepared = stage("prepare", lambda: self.prepare_batch(k, rd, audit, [str(v) for v in ([k - 1, k] if k else [0])]))
        self.train_batch(k, rd, state, state_path, prepared, expect_version=k)
        published = stage("publish", lambda: self.publish(k))
        synced = stage("sync", lambda: self.sync_pipeline(k, published))
        saved = self.maybe_save(k, stage)
        row = self.build_row(k, state, rolled["tasks"], rolled["pre"], rolled["heldout"], train, audit, prepared, published,
                             synced, saved, rolled["t_round"])
        row["weight_version_in"] = str(k)           # the version this update was applied on; the rollout's own versions are below
        row["pipeline"] = "async"
        row["async"] = {"rollout_server_version_start": rolled["version_start"], "rollout_server_version_end": rolled["version_end"],
                        "gate_wait_seconds": round(rolled["gate_wait_seconds"], 2), "learner_wait_seconds": round(learner_wait, 2),
                        "rollout_seconds": rolled["rollout_seconds"], "sync_mode": synced["mode"],
                        "sync_during_rollout_round": synced["during_round"],
                        "sync_seconds_into_rollout": synced["seconds_into_rollout"],
                        "sync_deferred_seconds": synced["deferred_seconds"], "pause_seconds": synced["pause_seconds"],
                        "ledger_landing_seconds": state["timings"]["ledger"], "tail_overlap": None}
        overlap = rolled["tail_overlap"]
        if overlap is not None:        # this round started in the previous round's tail; that round is learned, so it ended
            with self.cv:
                previous_end = self.rollout_ended[overlap["previous_round"]]
            # > 0: seconds both rounds had harness runs live (< 0 only if the tail ended during this round's preflight)
            row["async"]["tail_overlap"] = {**overlap,
                                            "seconds_before_previous_end": round(previous_end - rolled["claimed_mono"], 2)}
        self.append_row(row)
        self.log(f"round {k} done: reward {row['train']['reward_mean']:.4f} live {row['train']['live_fraction']:.2f} "
                 f"sync {synced['mode']} (rollout versions {rolled['version_start']}->{rolled['version_end']})")

    def sync_pipeline(self, k: int, published: dict) -> dict:
        """Sync only after held-out ends; if any round's rollout is running, mid-flight (pause in_place); if none is (rollout waiting at
        the gate, server idle), as in v1. The "rollout is running" check is mutually exclusive with a rollout starting
        (sync_in_progress): once judged idle, rollout must wait for the sync to finish before starting.
        (Under tail_overlap, round k has fully ended by the time of round k's sync, so at most round k+1 is running; all slots are still
        checked.)"""
        began = time.monotonic()
        with self.cv:
            while self.heldout_running:
                self.cv.wait(self.poll)
                self.check_cancel()
            self.sync_in_progress = True
            active = dict(self.active_rollouts)
        deferred = round(time.monotonic() - began, 2)
        completed = False
        try:
            if not active:
                result = {**self.sync(k, published), "mode": "idle", "pause_seconds": None}
            else:
                result = {**self.sync_midflight(k, published), "mode": "in_place"}
            newest = max(active) if active else None
            result["during_round"] = newest
            result["active_rounds"] = sorted(active)
            result["seconds_into_rollout"] = None if newest is None else round(began - active[newest], 2)
            result["deferred_seconds"] = deferred
            completed = True
        finally:
            with self.cv:
                if not completed:
                    self.aborted.set()          # before a rollout waiting for this sync can wake up and start
                else:
                    self.server_version = k + 1
                self.sync_in_progress = False
                self.cv.notify_all()
        return result

    def post_ok(self, path: str, body: dict) -> None:
        reply = http_json("POST", self.server_root + path, body, timeout=self.cfg["rollout"]["sync_timeout_seconds"])
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            raise OrchestratorError(f"POST {path} did not answer status ok: {reply!r}")

    def sync_midflight(self, k: int, published: dict) -> dict:
        """pause(in_place) → update(flush_cache:false) → continue whatever the outcome → only then judge success / poisoned /
        /model_info."""
        wv = str(k + 1)
        before = self.model_info()
        body = {"model_path": published["path"], "weight_version": wv, "flush_cache": False, "keep_identity": True}
        began = time.monotonic()
        try:
            self.post_ok("/pause_generation", {"mode": "in_place"})
            reply = http_json("POST", self.server_root + "/update_weights_from_disk", body,
                              timeout=self.cfg["rollout"]["sync_timeout_seconds"])
        finally:
            self.post_ok("/continue_generation", {})
        pause_seconds = round(time.monotonic() - began, 2)
        if reply["success"] is not True:
            raise OrchestratorError(f"mid-flight update_weights_from_disk failed (generation was continued): {reply}")
        after = self.model_info()
        names = self.served_names()
        self.verify_sync(wv, published, before, after, names)
        return {"request": body, "reply": reply, "served_name": names[0], "weight_version": wv,
                "weights_source": after.get("weights_source"), "pause_seconds": pause_seconds}

    # =========================================================================== stream pipeline (SPEC §9)

    def run_stream(self, history: list[dict]) -> str:
        """Dispatcher thread (sliding window of group processes) and learner thread (batch → train → sync) in parallel; if either dies
        the other stops (the dispatcher terminates every group process it started), and the real cause is raised at the end."""
        done, total = len(history), self.cfg["rounds"]
        batches = self.root / "batches"
        if done >= total:
            pending = sorted(p.name for p in batches.iterdir() if BATCH_DIR.fullmatch(p.name) and int(p.name[1:]) >= done) \
                if batches.is_dir() else []
            if pending:
                raise OrchestratorError(f"batches.jsonl has {done} row(s) = rounds, but {pending} is pending: raise rounds "
                                        f"to finish it")
            if self.set_aside_judge:            # never silently ignored
                raise OrchestratorError(f"--set-aside-judge-groups: the run is done ({done} batch(es) = rounds), nothing "
                                        f"starts that could set groups aside")
            return "done"
        self.quarantine_cfg()
        if self.note_judge is not None:
            self.check_judge_backend()
        (self.root / "groups").mkdir(exist_ok=True)
        batches.mkdir(exist_ok=True)
        next_gid, resume, adopted = self.stream_start_check(history)
        self.server_version = done + 1 if resume is not None and "sync" in resume["state"]["stages"] else done
        self.batches_taken = done + (1 if resume is not None else 0)
        self.reset_stream_stats(time.monotonic())
        self.gr_learner = self.gr_frontier = done + (1 if resume is not None else 0)
        self.spawn("dispatcher", lambda: self.dispatcher_loop(next_gid, adopted, self.heldout_adopted))
        if self.stream.get("group_ready", False):
            self.spawn("materializer", self.gr_materializer_loop)
        self.spawn("learner", lambda: self.stream_learner_loop(done, total, resume))
        self.join_pipeline()
        return "done" if self.pipeline_outcomes["learner"] == "done" else "stopped"

    def task_of(self, gid: int) -> str:
        for phase in reversed(self.task_phases):
            if gid < phase["start_gid"]:
                continue
            cycle, slot = divmod(gid - phase["start_gid"], phase["cycle_groups"])
            draws = phase["priority_draws"]
            seed = self.cfg["seed"]
            if slot < draws:
                return stream_task(phase["priority_ids"], f"{seed}:phase:{phase['start_gid']}:priority", True,
                                   cycle * draws + slot)
            return stream_task(self.pools["train"], f"{seed}:phase:{phase['start_gid']}:explore", True,
                               cycle * (phase["cycle_groups"] - draws) + slot - draws)
        return stream_task(self.pools["train"], self.cfg["seed"], self.cfg["tasks"]["train"]["shuffle"], gid)

    def pin_task_phases(self, next_gid: int) -> None:
        """A started phase is immutable; future phases may be appended or revised before dispatch."""
        path = self.root / "stream_task_phases.json"
        current = [{k: v for k, v in phase.items() if k != "priority_ids"} for phase in self.task_phases]
        if path.exists():
            previous = json.loads(path.read_text())
            started = lambda phases: [p for p in phases if p["start_gid"] < next_gid]
            if started(previous) != started(current):
                raise OrchestratorError("stream resume: a started task phase or priority ids SHA256 changed")
        if current or path.exists():
            atomic_write(path, json.dumps(current, indent=2) + "\n")

    def group_dir(self, gid: int) -> Path:
        return self.root / "groups" / f"g{gid:06d}"

    # ---- quarantine (stream.quarantine)
    def quarantine_cfg(self) -> dict:
        """stream.quarantine: may be absent when reading a config (other tools must be able to read old runs' configs), but is required
        to run stream; no defaults."""
        if "quarantine" not in self.stream:
            raise OrchestratorError(f"config stream.quarantine is missing: running the stream pipeline needs the "
                                    f"quarantine circuit breaker {sorted(QUARANTINE_KEYS)} (the configs/ templates and "
                                    f"configure_b200.py write it; add it to an older run's config by hand)")
        return self.stream["quarantine"]

    def quarantine_row(self, gid: int, group: str, task: str, stage: str, error: str, evidence: list[Path],
                       rewards: list, peers: list[dict] = ()) -> dict:
        """One quarantine row. ``peers``: the other groups in the same event (gr_quarantine); their task ids / episodes are also removed
        from the signature, so the same error gets the same signature whichever group it names (so the circuit breaker counts right)."""
        gs = self.stream["group_size"]
        literals = [group, task, *group_episode_ids(task, gs)]
        for r in peers:
            literals += [r["group"], r["task"], *group_episode_ids(r["task"], gs)]
        return {"utc": utc_now(), "unix": time.time(), "gid": gid, "group": group, "task": task, "stage": stage,
                "signature": error_signature(error, literals), "error": error, "evidence": [str(p) for p in evidence],
                "rewards": rewards}

    def record_quarantine(self, rows: list[dict]) -> None:
        """One row per group in quarantine.jsonl (append-only, counts only once written) + one prominent QUARANTINED line in
        orchestrator.log. The circuit breaker is judged separately (check_quarantine_breaker): the caller first settles the group's
        state (dir moved away, recorded as abandoned), so files are consistent if the run stops.
        One call = one event: every row carries the same ``event`` id, and the breaker counts events, not rows."""
        event = f"{int(time.time())}-{os.urandom(3).hex()}"
        for row in rows:
            row["event"] = event
            append_jsonl(self.root / QUARANTINE_FILE, row)
            with self.cv:
                self.quarantine_rows.append(row)
            self.log(f"QUARANTINED {row['group']} ({row['task']}) at the {row['stage']} stage: {row['signature']}; rewards "
                     f"{row['rewards']}; evidence {row['evidence']}")

    def check_quarantine_breaker(self, rows: list[dict]) -> None:
        with self.cv:
            reason = quarantine_breaker(self.quarantine_rows, rows, self.finish_order, self.quarantine_cfg(), time.time())
        if reason is not None:
            self.log(f"QUARANTINE CIRCUIT BREAKER: {reason}")
            raise OrchestratorError(f"quarantine circuit breaker (stream.quarantine): {reason}. This looks systematic, not "
                                    f"one bad group: see {self.root / QUARANTINE_FILE} and the evidence paths")

    def mark_terminated(self, name: str, reason: str, run_dir: Path | None = None) -> Path:
        """Before sending SIGTERM, persist "this group was taken down by us" (TERMINATED_SUFFIX, next to the group dir; ``run_dir`` None =
        training group ``groups/<name>``, held-out groups pass their own dir)."""
        if reason not in TERMINATED_REASONS:
            raise OrchestratorError(f"mark_terminated {name}: reason {reason!r} not in {TERMINATED_REASONS}")
        run_dir = self.root / "groups" / name if run_dir is None else run_dir
        path = run_dir.with_name(f"{run_dir.name}{TERMINATED_SUFFIX}")
        atomic_write(path, json.dumps({"group": name, "reason": reason, "utc": utc_now(), "unix": time.time(),
                                       "session_id": self.session_id}) + "\n")
        return path

    def read_terminated(self, gid: int) -> dict | None:
        return self.read_terminated_at(self.group_dir(gid))

    def read_terminated_at(self, run_dir: Path) -> dict | None:
        """The terminated marker next to the group dir (mark_terminated); missing = None."""
        path = run_dir.with_name(f"{run_dir.name}{TERMINATED_SUFFIX}")
        if not path.is_file():
            return None
        doc = json.loads(path.read_text())
        if doc.get("reason") not in TERMINATED_REASONS:
            raise OrchestratorError(f"{path}: reason {doc.get('reason')!r} not in {TERMINATED_REASONS}")
        return doc

    def quarantine_harness_group(self, gid: int, entry: dict, error: str, *, inflight: bool = True) -> None:
        """dispatcher: a group with harness rc≠0 (or timed out and exiting on its own after SIGTERM) does not stop the run but is
        quarantined: a row goes into quarantine.jsonl first, the group dir and .out move to groups/quarantined/ (never in groups.jsonl:
        not pooled, not trained), its running state is removed, then the circuit breaker is checked. If the run stops after the row is
        written but before the dir moves, settle_quarantined finishes the move on resume."""
        gs = self.stream["group_size"]
        run_dir, stdout = entry["run_dir"], entry["stdout"]
        aside = self.root / "groups" / QUARANTINE_DIR
        launch = run_dir.with_name(run_dir.name + LAUNCH_SUFFIX)
        terminated = run_dir.with_name(run_dir.name + TERMINATED_SUFFIX)
        moves = [(p, aside / p.name) for p in (run_dir, stdout, launch, terminated) if p.exists()]
        rewards = harness_rewards(run_dir, entry["task"], gs) if run_dir.exists() else [None] * gs
        evidence = [dest for _, dest in moves] + ([aside / run_dir.name / "failures.jsonl"]
                                                   if (run_dir / "failures.jsonl").is_file() else [])
        row = self.quarantine_row(gid, entry["name"], entry["task"], "harness", error, evidence, rewards)
        self.record_quarantine([row])
        aside.mkdir(exist_ok=True)
        for src, dest in moves:
            if dest.exists():
                raise OrchestratorError(f"{dest} already exists: group ids are never reused")
            src.rename(dest)
        with self.cv:
            if inflight:                     # False: a group of the previous orchestrator (triage_incomplete), never registered
                del self.inflight_groups[gid]
                if self.cfg["rollout"].get("replica_urls"):
                    del self.group_backends[gid]
            self.finish_order.append(gid)
            self.cv.notify_all()
        self.check_quarantine_breaker([row])

    def settle_quarantined(self, recorded: dict[int, dict], abandoned: dict[int, dict]) -> None:
        """Resume: read quarantine.jsonl. Harness-level rows: that group must never have completed in groups.jsonl; if its dir or .out is
        still under groups/ (row written, stopped before the move), finish the move to groups/quarantined/; everything there must have a
        row. Prepare-level rows: that group must already be recorded as abandoned (abandoned is recorded before the row is written).
        Finishing order (the breaker's share window) = completed groups and harness-level quarantined ones, ordered by time."""
        path = self.root / QUARANTINE_FILE
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        for i, row in enumerate(rows):
            if not (QUARANTINE_ROW_KEYS <= set(row) <= QUARANTINE_ROW_KEYS | QUARANTINE_OPTIONAL_ROW_KEYS) \
                    or row["stage"] not in QUARANTINE_STAGES:
                raise OrchestratorError(f"{path}:{i + 1}: not a quarantine row (keys {sorted(row)}, stage "
                                        f"{row.get('stage')!r})")
        harness = {r["gid"]: r for r in rows if r["stage"] == "harness"}
        prepare = {r["gid"]: r for r in rows if r["stage"] != "harness"}      # prepare / audit: recorded as abandoned
        groups, aside = self.root / "groups", self.root / "groups" / QUARANTINE_DIR
        wrong = sorted(set(harness) & set(recorded)) + sorted(set(prepare) - set(abandoned))
        if wrong:
            raise OrchestratorError(f"{path}: group(s) {wrong[:5]} quarantined at the harness stage but completed in "
                                    f"groups.jsonl, or at the prepare stage but not abandoned there: fix by hand")
        for gid, row in sorted(harness.items()):
            for name in (row["group"], f"{row['group']}.out", f"{row['group']}{LAUNCH_SUFFIX}", f"{row['group']}{TERMINATED_SUFFIX}"):
                if (groups / name).exists():
                    aside.mkdir(exist_ok=True)
                    if (aside / name).exists():
                        raise OrchestratorError(f"{groups / name} and {aside / name} both exist: inspect by hand")
                    (groups / name).rename(aside / name)
                    self.log(f"stream start: {name} was quarantined (harness) but not moved yet -> {aside / name}")
        if aside.is_dir():
            known = {r["group"] for r in harness.values()}
            stray = sorted(p.name for p in aside.iterdir()
                           if p.name.removesuffix(".out").removesuffix(LAUNCH_SUFFIX).removesuffix(TERMINATED_SUFFIX) not in known)
            if stray:
                raise OrchestratorError(f"{aside}: {stray[:5]} have no harness row in {path}: not moved there by this "
                                        f"orchestrator; inspect by hand")
        events = [(r["completed_unix"], r["gid"]) for r in recorded.values()] + [(r["unix"], g) for g, r in harness.items()]
        with self.cv:
            self.quarantine_rows = rows
            self.finish_order = [g for _, g in sorted(events)]
        if rows:
            self.log(f"stream start: {len(rows)} quarantined group(s) in {path} ({len(harness)} harness, {len(prepare)} "
                     f"prepare)")

    def load_groups(self) -> tuple[list[dict], dict[int, dict]]:
        """groups.jsonl (append-only): ``completed`` rows (completion order) and ``abandoned`` rows (pool groups a human decided not to
        train on at resume); ``set_aside`` rows (--set-aside-judge-groups: groups whose harness finished but whose judge kept failing,
        which a human decided neither to judge nor to train on; never completed) are only validated, not returned."""
        path = self.root / "groups.jsonl"
        if not path.exists():
            return [], {}
        completed, abandoned, set_aside = [], {}, set()
        for i, line in enumerate(path.read_text().splitlines()):
            doc = json.loads(line)
            if doc["event"] == "completed":
                completed.append(doc)
            elif doc["event"] == "abandoned":
                abandoned[doc["gid"]] = doc
            elif doc["event"] == "set_aside":       # --set-aside-judge-groups: finished, never judged, never trained
                if doc["gid"] in set_aside:
                    raise OrchestratorError(f"{path}:{i + 1}: group {doc['gid']} set aside twice")
                set_aside.add(doc["gid"])
            else:
                raise OrchestratorError(f"{path}:{i + 1}: unknown event {doc['event']!r}")
        gids = [r["gid"] for r in completed]
        if len(set(gids)) != len(gids):
            raise OrchestratorError(f"{path}: group(s) completed twice: {sorted(g for g, c in Counter(gids).items() if c > 1)}")
        if set(abandoned) - set(gids):
            raise OrchestratorError(f"{path}: abandoned group(s) {sorted(set(abandoned) - set(gids))} never completed")
        if set_aside & set(gids):
            raise OrchestratorError(f"{path}: set-aside group(s) {sorted(set_aside & set(gids))} also completed")
        return completed, abandoned

    def set_aside_judge_groups(self, to_finish: list[dict], aside: Path) -> list[dict]:
        """--set-aside-judge-groups: every gid must be a group whose "harness finished (rc 0, results complete, roster unchanged:
        triage_incomplete's to-finish entries), needs judging, has no judge.json" (the kind where the judge failed every time and
        stopped the run); otherwise raise and touch nothing. Their dirs (with .out, launch / terminated markers) move to ``aside``
        (groups/abandoned-<stamp>/, like abandoned groups: routing_release frees their routing), and groups.jsonl gets one
        ``set_aside`` row (with the reason spelled out; never completed, not pooled, not trained). Returns the remaining entries to
        finish."""
        finishing = {int(e["name"][1:]): e for e in to_finish}
        names = [f"g{g:06d}" for g in self.set_aside_judge]
        wrong = [n for g, n in zip(self.set_aside_judge, names)
                 if g not in finishing or self.judge_input_needed(finishing[g]) is None]
        if wrong:
            raise OrchestratorError(
                f"--set-aside-judge-groups {wrong}: not a group whose harness finished (rc 0, every result, current ledger "
                f"roster) and that waits for its note judge (episodes to judge, no {JUDGE_FILE}); nothing was moved. "
                f"Waiting for the judge now: {sorted(f'g{g:06d}' for g, e in finishing.items() if self.judge_input_needed(e))}")
        groups = self.root / "groups"
        aside.mkdir(exist_ok=True)
        for g, name in zip(self.set_aside_judge, names):
            for suffix in ("", ".out", LAUNCH_SUFFIX, TERMINATED_SUFFIX):
                if (groups / f"{name}{suffix}").exists():
                    (groups / f"{name}{suffix}").rename(aside / f"{name}{suffix}")
            append_jsonl(self.root / "groups.jsonl",
                         {"event": "set_aside", "gid": g, "group": name, "task": finishing[g]["task"], "utc": utc_now(),
                          "reason": "note judge failed every attempt; set aside by --set-aside-judge-groups (harness "
                                    f"finished, no {JUDGE_FILE}, never trained)", "moved_to": str(aside / name)})
        self.log(f"set aside {len(names)} finished but unjudged group(s) -> {aside} (--set-aside-judge-groups, never "
                 f"trained): {names}")
        return [e for g, e in finishing.items() if g not in self.set_aside_judge]

    def stream_start_check(self, rows: list[dict]) -> tuple[int, dict | None]:
        """stream starts only from a clean point or from one batch that can be continued (SPEC §9). n = number of rows in batches.jsonl.
        - ``batches/`` holds at most one batch not in batches.jsonl, and it is b = n. If the trainer has already taken a step from it
          (state.json has train metrics or publish): continue it (``resume_plan``; no flag needed, this is a determined action); its
          groups, trainer step and trainer / server versions must match state.json. If the trainer has not touched it, it is moved
          aside only with ``--abandon-incomplete-round`` (its groups return to the pool).
        - every ``groups/gNNNNNN`` is a completed group in groups.jsonl; those not consumed and not part of the batch being continued
          return to the pool (prepare drops too-old ones as stale). Half-run group dirs and pool groups whose ledger instance has changed
          are moved aside / recorded as abandoned only with ``--abandon-incomplete-round``.
        - no batch to continue: trainer and server both at version n, trainer step = trainer_step_after of the last row.
        Returns (next gid, the batch to continue or None). No guessing: anything else raises."""
        n = len(rows)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        pending = self.pending_batch(n, stamp)
        completed, abandoned = self.load_groups()
        recorded = {r["gid"]: r for r in completed}
        self.settle_quarantined(recorded, abandoned)     # groups/quarantined/ is not a gNNNNNN dir: never "incomplete"
        groups = self.root / "groups"
        present = {int(p.name[1:]) for p in groups.iterdir() if GROUP_DIR.fullmatch(p.name) and p.is_dir()}
        incomplete = sorted(present - set(recorded))
        adopted, to_finish, abandon = self.triage_incomplete(incomplete)
        self.live_history = [r["live"] for r in completed]
        if abandon:
            names = [f"g{g:06d}" for g in abandon]
            aside = groups / f"abandoned-{stamp}"
            aside.mkdir()
            for name in names:
                for suffix in ("", ".out", LAUNCH_SUFFIX, TERMINATED_SUFFIX):
                    if (groups / f"{name}{suffix}").exists():
                        (groups / f"{name}{suffix}").rename(aside / f"{name}{suffix}")
            self.log(f"abandoned {len(names)} incomplete group dir(s) -> {aside}: {names}")
        if self.set_aside_judge:
            to_finish = self.set_aside_judge_groups(to_finish, groups / f"abandoned-{stamp}")
        lost = sorted(set(recorded) - present)
        if lost:
            raise OrchestratorError(f"stream resume: groups.jsonl records group(s) {lost[:5]} whose directory is gone from {groups}")
        for r in completed:
            if r["task"] != self.task_of(r["gid"]) or r["run_dir"] != str(self.group_dir(r["gid"])):
                raise OrchestratorError(f"stream resume: group {r['gid']} was {r['task']} at {r['run_dir']}, this config maps it to "
                                        f"{self.task_of(r['gid'])} at {self.group_dir(r['gid'])}: the task stream (train ids, "
                                        f"seed, shuffle) or run_root changed")
        state_path = self.root / "stream_state.json"
        seen = set(recorded) | present | {r["gid"] for r in self.quarantine_rows if r["stage"] == "harness"}
        if state_path.exists():
            next_gid = json.loads(state_path.read_text())["next_gid"]
        elif seen:
            raise OrchestratorError(f"stream resume: {state_path} is missing but groups exist")
        else:
            next_gid = 0
        if seen and next_gid <= max(seen):
            raise OrchestratorError(f"stream resume: {state_path} next_gid {next_gid} <= a group that exists ({max(seen)})")
        self.pin_task_phases(next_gid)
        consumed = [g["gid"] for row in rows for g in row["groups"]]
        twice = sorted(g for g, c in Counter(consumed).items() if c > 1)
        unknown = sorted(set(consumed) - set(recorded))
        if twice or unknown or set(consumed) & set(abandoned):
            raise OrchestratorError(f"stream resume: batches.jsonl consumed group(s) twice {twice[:5]}, never completed "
                                    f"{unknown[:5]}, or abandoned {sorted(set(consumed) & set(abandoned))[:5]}")
        resume = None if pending is None else self.resume_plan(n, pending, recorded, set(consumed), abandoned)
        taken = set(consumed) | set(abandoned) | (set() if resume is None else set(resume["state"]["groups"]))
        unconsumed = [r for r in completed if r["gid"] not in taken]
        prev_step = rows[-1]["trainer_step_after"] if rows else None
        if resume is None:
            want_trainer = want_server = n
            where = f"batches.jsonl has {n} row(s)"
        else:
            stages, trained = resume["state"]["stages"], len(resume["state"]["train"]["metrics"])
            want_trainer = n + 1 if "publish" in stages else n
            want_server = n + 1 if "sync" in stages else n
            where = f"batch {n} is pending with stages {sorted(stages)} done"
        with self.cv:
            self.server_version = want_server      # the version the groups finished below are completed at
        for entry in to_finish:                    # dead with rc 0 and every result: nobody wrote their groups.jsonl row
            gid = int(entry["name"][1:])
            # triage_incomplete only finishes a group whose launch roster is the servers' current one, so this snapshot
            # (taken now, before any new group launches) covers it -- also for a group still waiting for its note judge.
            # One that finished on an old roster (a server restarted while it waited for its judge) was abandoned there
            # without any judge spend: its ledger instance is gone, it cannot be audited, a judge.json would not help
            self.snapshot_ledger(entry)
            doc = self.judge_input_needed(entry)
            if doc is not None:                    # arc3_reward note without its judge.json yet: the dispatcher judges it
                self.judge_resume.append((entry, doc))
                self.log(f"group {entry['name']} ({entry['task']}): harness finished (rc 0, every result) after its "
                         f"orchestrator stopped, but its note judge did not: judged again once the dispatcher starts")
                continue
            record = self.complete_group(gid, entry)
            completed.append(record)
            unconsumed.append(record)
            self.live_history.append(record["live"])
            self.log(f"group {entry['name']} ({entry['task']}): harness finished (rc 0, every result) after its orchestrator "
                     f"stopped: completed now at version {want_server}, rewards {record['rewards']}")
        self.heldout_adopted = self.heldout_start_check()     # held-out groups the previous orchestrator left (heldout/)
        health = self.trainer_health()
        if self.stream.get("group_ready", False):
            if "update" not in health:
                raise OrchestratorError(f"stream.group_ready: trainer /health has no 'update' field (a trainer without "
                                        f"/begin, /accumulate, /step): {health}")
            if health["update"] is not None and health["busy"] is False:
                # gradients of a batch the trainer never stepped from (that batch was set aside above): clear them
                self.log(f"stream start: trainer holds open update {health['update']!r}: /abort (no optimizer step)")
                self.trainer("POST", "/abort", {"update_id": health["update"]}, timeout=self.cfg["trainer"]["train_timeout_seconds"])
                health = self.trainer_health()
                if health["update"] is not None:
                    raise OrchestratorError(f"stream start: trainer still holds update {health['update']!r} after /abort")
        if health["busy"] is not False or str(health["weight_version"]) != str(want_trainer):
            raise OrchestratorError(f"stream resume: {where} but the trainer is not idle at weight_version {want_trainer}: "
                                    f"/health says {health}")
        if resume is not None and health["step"] != (prev_step or 0) + trained:
            raise OrchestratorError(f"stream resume: trainer step is {health['step']}, want {(prev_step or 0) + trained} = "
                                    f"{prev_step or 0} (the last batch row's trainer_step_after) + {trained} minibatch(es) recorded "
                                    f"in {resume['dir'] / 'state.json'}: the trainer and the batch state disagree; fix by hand")
        if resume is None and health["step"] != (prev_step or 0):
            raise OrchestratorError(f"stream resume: trainer step is {health['step']}, the last batch row left "
                                    f"{prev_step or 0}: the trainer took steps outside batches.jsonl")
        if not http_ok(self.server_root + "/health"):
            raise OrchestratorError(f"rollout server {self.server_root} /health is not 200")
        info = self.model_info()
        names = self.served_names()
        if names != [self.cfg["rollout"]["served_name"]] or info["served_model_name"] != names[0]:
            raise OrchestratorError(f"server serves {names} (/model_info {info['served_model_name']!r}), "
                                    f"config rollout.served_name is {self.cfg['rollout']['served_name']!r}")
        replicas = self.cfg["rollout"].get("replica_urls")
        if replicas is not None and info["replica_urls"] != replicas:
            raise OrchestratorError(f"stream startup: control replica_urls {info['replica_urls']!r} != "
                                    f"generation replica_urls {replicas!r}")
        if str(info["weight_version"]) != str(want_server) or info["poisoned"] is not False:
            raise OrchestratorError(f"stream resume: {where} but the server is at weight_version {info['weight_version']!r} "
                                    f"(poisoned {info['poisoned']!r}), want {str(want_server)!r} (a sync was lost, or the server "
                                    f"was not launched with --weight-version / synced to it)")
        identity = health["policy_identity"]
        # A QAT trainer keeps a BF16 master checkpoint that is not the deployed package the servers load
        # (INT3 W3A8: trainer --model is the BF16-dense variant). rollout.trainer_base_path then declares the
        # trainer's base explicitly; without it the two bases must be the same directory.
        trainer_base = self.cfg["rollout"].get("trainer_base_path", info["model_path"])
        if Path(identity["model_path"]).resolve() != Path(trainer_base).resolve():
            raise OrchestratorError(f"stream startup: trainer base {identity['model_path']!r} != "
                                    f"{'declared trainer base' if 'trainer_base_path' in self.cfg['rollout'] else 'serving base'} "
                                    f"{trainer_base!r}")
        if "trainer_base_path" in self.cfg["rollout"]:       # two directories: prove they are one model
            try:
                self.log(check_lineage(Path(info["model_path"]), Path(trainer_base)))
            except LineageError as e:
                raise OrchestratorError(f"stream startup: {e}") from None
        if want_server > 0:
            published = (resume["state"]["stages"]["publish"] if resume is not None
                         and "sync" in resume["state"]["stages"] else rows[-1]["publish"])
            if info["weights_source"] != published["path"]:
                raise OrchestratorError(f"stream resume: weights_source {info['weights_source']!r} != "
                                        f"the recorded policy {published['path']!r}")
        else:
            source = self.cfg["rollout"].get("initial_weights_source")
            if identity["init_lora"] is not None and source is None:
                raise OrchestratorError("stream fresh start: trainer init_lora requires rollout.initial_weights_source "
                                        "for an explicit initial policy handshake")
            if source is not None and Path(info["weights_source"]).resolve() != Path(source).resolve():
                raise OrchestratorError(f"stream fresh start: weights_source {info['weights_source']!r} != "
                                        f"declared initial policy {source!r}")
            if source is None and Path(info["weights_source"]).resolve() != Path(info["model_path"]).resolve():
                raise OrchestratorError(f"stream fresh start: unexpected source {info['weights_source']!r}; "
                                        f"zero-initialized trainer expects serving base {info['model_path']!r}")
        self.served_name = names[0]
        status = http_json("GET", self.server_root + "/session_ledger/status")
        ledger_count(status, "enqueued_events")
        ledger_count(status, "written_events")
        current_instances = {row["instance"] for row in ledger_statuses(status)}
        def ledger_matches(record):
            # Historical single-backend records have no replica watermarks. New records must
            # match the complete roster that ledger_landing checks, even if their own replica survived.
            events = record.get("replica_events")
            return (record["ledger_instance"] in current_instances
                    and (events is None or set(events) == current_instances))
        if resume is not None and "audit" not in resume["state"]["stages"]:      # audited batches keep their instance
            stray = [r["group"] for r in resume["groups"] if not ledger_matches(r)]
            if stray:
                raise OrchestratorError(f"stream resume: batch {n} is not audited yet and its group(s) {stray[:5]} were rolled "
                                        f"out on another ledger instance than the server's {status['instance']}")
        other = [r for r in unconsumed if not ledger_matches(r)]
        if other:
            if not self.abandon_incomplete:
                raise OrchestratorError(
                    f"stream resume: {len(other)} pooled group(s), e.g. {[r['group'] for r in other[:5]]}, were rolled out on "
                    f"ledger instance {other[0]['ledger_instance']}, the server now writes {status['instance']} (restarted); "
                    f"recorded replica roster {sorted(other[0].get('replica_events') or [other[0]['ledger_instance']])}, "
                    f"current roster {sorted(current_instances)}: "
                    f"they cannot be audited. Pass --abandon-incomplete-round to record them as abandoned (never trained)")
            for r in other:
                append_jsonl(self.root / "groups.jsonl",
                             {"event": "abandoned", "gid": r["gid"], "group": r["group"], "utc": utc_now(),
                              "reason": f"group ledger instance {r['ledger_instance']} / replica roster "
                                        f"{sorted(r.get('replica_events') or [r['ledger_instance']])} incompatible with "
                                        f"current roster {sorted(current_instances)}"})
            self.log(f"abandoned {len(other)} pooled group(s) of ledger instance(s) "
                     f"{sorted({r['ledger_instance'] for r in other})}: {[r['group'] for r in other]}")
            unconsumed = [r for r in unconsumed if ledger_matches(r)]
        in_flight = [*adopted, *self.heldout_adopted]           # their requests are on the servers: expected
        self.wait_idle("stream start", (lambda: unfinished_episodes(in_flight)) if in_flight else None)
        live = sum(r["live"] for r in unconsumed)
        with self.cv:
            self.pool = unconsumed
        if resume is not None:
            resume["pool_after"] = {"groups": len(unconsumed), "live_groups": live, "groups_in_flight": len(adopted)}
            self.log(f"resuming batch {n} from {resume['dir']}: stages done {resume['stages_done']}, {trained} minibatch(es) "
                     f"trained (trainer step {health['step']}, trainer version {want_trainer}, server version {want_server}); "
                     f"its {len(resume['groups'])} group(s) {[r['group'] for r in resume['groups']]}; the done stages are "
                     f"skipped")
        if self.stream.get("retire_full_groups", False):
            path = self.root / RETIRED_FILE
            logged = {json.loads(line)["task"] for line in path.read_text().splitlines()} if path.exists() else set()
            full = [r for r in completed if group_full(r["episodes"], self.cfg["prepare"])]
            if logged - {r["task"] for r in full}:
                raise OrchestratorError(f"stream resume: {RETIRED_FILE} retires {sorted(logged - {r['task'] for r in full})[:5]}, "
                                        f"which have no full group in groups.jsonl")
            for r in full:      # a crash between the groups.jsonl row and its retired.jsonl row: write the missing row now
                self.retire_task(r)
            self.log(f"retire_full_groups: {len(self.retired)} of {len(self.pools['train'])} train task(s) retired so far")
        if "retire_tasks_file" in self.stream:
            path = self.stream["retire_tasks_file"]
            listed = read_retire_list(path)
            outside = sorted(set(listed) - set(self.pools["train"]))
            if outside:
                raise OrchestratorError(f"stream.retire_tasks_file {path}: {len(outside)} task(s) not in the train list, "
                                        f"e.g. {outside[:5]}")
            self.retired |= set(listed)
            if set(self.pools["train"]) <= self.retired:
                raise OrchestratorError(f"stream.retire_tasks_file {path}: retires every train task: nothing left to dispatch")
            self.log(f"retire_tasks_file: {len(listed)} train task(s) retired by {path} "
                     f"(sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]}); "
                     f"{len(set(self.pools['train']) - self.retired)} of {len(self.pools['train'])} left to dispatch")
        self.log(f"stream start at version {want_server}: next group g{next_gid:06d}; {len(unconsumed)} completed group(s) back "
                 f"in the pool ({live} live); {len(adopted)} in-flight group(s) adopted"
                 + (f", {len(self.heldout_adopted)} held-out group(s) adopted" if self.heldout_adopted else ""))
        return next_gid, resume, adopted

    def triage_incomplete(self, incomplete: list[int]) -> tuple[list[dict], list[dict], list[int]]:
        """``groups/gNNNNNN`` not in groups.jsonl (in flight when the previous orchestrator stopped; SPEC §9 adoption). Those with a
        launch.json are judged by it:
        - harness still alive (same pid + starttime, not a zombie), ledger roster unchanged: **adopt** (the dispatcher manages it as its
          own: wait for it to finish, read results, put it in the pool). Flags do not affect this: in-flight work is not thrown away.
        - alive but the roster changed (the server restarted, so the results cannot be audited): raise; only with
          ``--abandon-incomplete-round`` is it sent SIGTERM, and if it has not left within terminate_grace_seconds, raise (no SIGKILL of
          someone else's process group: the user decides); once gone it is moved aside.
        - has terminated.json (the previous orchestrator sent SIGTERM): if still alive, wait for it to exit (grace; SIGKILL for timeout);
          reason timeout → quarantine (stage harness), cancel / abandon → move aside. run_arc2 still writes results / manifest /
          EXIT.json rc 0 after SIGTERM; without this rule it would be taken for a finished group (2026-10-24 review).
        - dead, rc 0 (EXIT.json), all group_size results present, roster unchanged: **finish it** (the caller calls complete_group once
          the server version is known: only the groups.jsonl entry is missing).
        - every other dead one (including those without EXIT.json): move aside (no flag needed: nobody is writing to it for sure).
        No launch.json (started by an old orchestrator): old rule, moved aside only with the flag. Returns (adopted entries, entries to
        finish, gids to move aside)."""
        adopted, finished, abandon = [], [], []
        if not incomplete:
            return adopted, finished, abandon
        roster = sorted(row["instance"] for row in ledger_statuses(http_json("GET", self.server_root + "/session_ledger/status")))
        grace = self.cfg["harness"]["terminate_grace_seconds"]
        gs = self.stream["group_size"]
        legacy = []
        for gid in incomplete:
            name = f"g{gid:06d}"
            doc = self.read_launch(gid)
            if doc is None:
                legacy.append(name)
                continue
            if doc["task"] != self.task_of(gid):
                raise OrchestratorError(f"stream resume: group {name} was launched as {doc['task']}, this config maps it to "
                                        f"{self.task_of(gid)}: the task stream (train ids, seed, shuffle) changed")
            entry = self.adopt_entry(doc)
            proc, pid = entry["proc"], doc["pid"]
            same_roster = sorted(doc["ledger_instances"]) == roster
            marker = self.read_terminated(gid)
            if proc.poll() is None and marker is not None:
                self.wait_told_to_stop(f"group {name}", doc["task"], proc, pid, marker)
            if proc.poll() is None:
                if same_roster:
                    adopted.append(entry)
                    self.log(f"group {name} ({doc['task']}): harness pid {pid} launched {doc['launched_utc']} by session "
                             f"{doc['session_id']} still runs: adopted")
                    continue
                why = f"its ledger roster {doc['ledger_instances']} != the servers' {roster} (server restarted: it cannot be audited)"
                if not self.abandon_incomplete:
                    raise OrchestratorError(f"stream resume: group {name} ({doc['task']}): harness pid {pid} still runs but {why}. "
                                            f"Pass --abandon-incomplete-round to terminate it and move the group aside")
                self.log(f"group {name} ({doc['task']}): harness pid {pid} still runs but {why}: SIGTERM, waiting up to {grace}s")
                self.mark_terminated(name, "abandon")
                proc.terminate()
                try:
                    proc.wait(grace)
                except subprocess.TimeoutExpired:
                    raise OrchestratorError(f"group {name}: harness pid {pid} still runs {grace}s after SIGTERM: stop it by hand "
                                            f"(kill -KILL -- -{pid} ends its process group) and start again") from None
                abandon.append(gid)
                continue
            rc = proc.poll()
            done = group_episodes_done(entry["run_dir"], doc["task"], gs)
            complete = group_results_complete(entry["run_dir"], doc["task"], gs)
            how = (f"rc {rc}{'' if exit_record(entry['run_dir']) is not None else ' (no EXIT.json)'}, {done}/{gs} results"
                   f"{'' if complete else ' (incomplete)'}")
            if marker is not None:
                if marker["reason"] == "timeout":
                    self.quarantine_harness_group(gid, entry, f"group timeout: terminated by the previous orchestrator (session "
                                                              f"{marker['session_id']}), {how}", inflight=False)
                    self.log(f"group {name} ({doc['task']}): terminated for timeout before the restart, {how}: quarantined")
                else:
                    self.log(f"group {name} ({doc['task']}): terminated ({marker['reason']}) before the restart, {how}: abandoned")
                    abandon.append(gid)
                continue
            if rc == 0 and complete and same_roster:
                finished.append(entry)
                continue
            self.log(f"group {name} ({doc['task']}): harness pid {pid} is gone with {how}"
                     f"{'' if same_roster else ', ledger roster changed'}: abandoned")
            abandon.append(gid)
        if legacy:
            if not self.abandon_incomplete:
                raise OrchestratorError(
                    f"stream resume: {len(legacy)} group dir(s) in {self.root / 'groups'} have no {LAUNCH_SUFFIX} record, e.g. "
                    f"{legacy[:5]}: an older orchestrator left them in flight. Make sure no harness still writes into them "
                    f"(check ps), then pass --abandon-incomplete-round or move them (and their .out) out by hand")
            abandon.extend(int(n[1:]) for n in legacy)
        return adopted, finished, sorted(abandon)

    def wait_told_to_stop(self, label: str, task: str, proc: AdoptedProc, pid: int, marker: dict) -> None:
        """Resume: the previous orchestrator sent this group SIGTERM (terminated marker ``marker``) and it is still alive: wait for it to
        exit (terminate_grace_seconds); still there: for reason timeout, SIGKILL its process group (like the dispatcher's timeout), for
        other reasons raise (no SIGKILL of someone else's process group: the user decides). Shared by training groups
        (triage_incomplete) and held-out groups (heldout_start_check)."""
        grace = self.cfg["harness"]["terminate_grace_seconds"]
        self.log(f"{label} ({task}): harness pid {pid} was told to stop ({marker['reason']}) by session "
                 f"{marker['session_id']} and still runs: waiting up to {grace}s")
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            if marker["reason"] != "timeout":
                raise OrchestratorError(f"{label}: harness pid {pid} still runs {grace}s after the earlier SIGTERM "
                                        f"({marker['reason']}): stop it by hand (kill -KILL -- -{pid}) and start again") from None
            self.log(f"{label}: still running: SIGKILL to process group {pid}")
            killpg_quietly(pid, signal.SIGKILL)
            try:
                proc.wait(KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                raise OrchestratorError(f"{label}: pid {pid} is still there {KILL_WAIT_SECONDS}s after SIGKILL; "
                                        f"check ps") from None

    def pending_batch(self, n: int, stamp: str) -> dict | None:
        """Batches in ``batches/`` not written to batches.jsonl: only one with b = n is allowed, plus (stream.prefetch_prepare) one
        prefetched b = n+1. If the trainer has taken a step from b = n (train metrics or publish) → return {dir, state} to resume_plan;
        died while /train was in flight → raise (unknown whether that step was consumed); trainer has not touched it → moved aside only
        with the flag. A prefetched batch (``prefetched`` in state.json) the trainer has not touched is moved aside without a flag: only
        ledger / audit / prepare were done, and its groups return to the pool."""
        batches = self.root / "batches"
        found = sorted(p for p in batches.iterdir() if BATCH_DIR.fullmatch(p.name) and int(p.name[1:]) >= n)
        if ([p.name for p in found] == [f"b{n:06d}", f"b{n + 1:06d}"] and (found[1] / "state.json").exists()
                and json.loads((found[1] / "state.json").read_text()).get("prefetched", False)):
            self.set_aside_prefetched(found[1], n + 1, stamp)
            found = found[:1]
        if [p.name for p in found] not in ([], [f"b{n:06d}"]):
            raise OrchestratorError(f"stream resume: batch dir(s) {[p.name for p in found]} exist past batches.jsonl's {n} "
                                    f"row(s); only b{n:06d} can be pending (the learner trains one batch at a time): inspect "
                                    f"{batches} by hand")
        if not found:
            return None
        path = found[0]
        state_path = path / "state.json"
        if not state_path.exists():
            raise OrchestratorError(f"{path} has no state.json: not created by this orchestrator; inspect it and move it aside by hand")
        state = json.loads(state_path.read_text())
        if state["batch"] != n:
            raise OrchestratorError(f"{state_path}: batch {state['batch']!r} != {n}")
        train = state["train"]
        if train["inflight"] is not None:
            raise OrchestratorError(f"{state_path}: /train for minibatch {train['inflight']['minibatch']} was in flight when the "
                                    f"orchestrator died; compare trainer /health step with step_before="
                                    f"{train['inflight']['step_before']} and fix state.json by hand")
        report_path = path / "batch" / "report.json"
        if "prepare" not in state["stages"] and report_path.exists():
            # prepare may finish its report before make_stage commits state.json; the report is an explicit no-data result.
            summary = json.loads(report_path.read_text())
            if summary.get("status") == "EMPTY_ELIGIBLE":
                if (summary["round"] != n or summary["current_version"] != n
                        or summary["max_staleness"] != self.stream["max_staleness"]
                        or summary["episodes_kept"] != 0 or summary["groups_kept"] != 0
                        or (path / "batch" / "manifest.json").exists()):
                    raise OrchestratorError(f"{report_path}: inconsistent EMPTY_ELIGIBLE report")
                state["stages"]["prepare"] = {"status": "EMPTY_ELIGIBLE", "summary": summary, "minibatches": 0}
                atomic_write(state_path, json.dumps(state, indent=1) + "\n")
        if state["stages"].get("prepare", {}).get("status") == "EMPTY_ELIGIBLE":
            self.discard_empty_batch(path, state)
            return None
        if train["metrics"] or "publish" in state["stages"]:
            return {"dir": path, "state": state}
        if state.get("prefetched", False):     # states written before prefetch existed have no key: not prefetched
            self.set_aside_prefetched(path, n, stamp)
            return None
        if not self.abandon_incomplete:
            raise OrchestratorError(
                f"stream resume: {path} exists but batches.jsonl has {n} row(s): a batch was started and the trainer took no "
                f"step from it (train.metrics empty, train.inflight null, no publish). Pass --abandon-incomplete-round to move "
                f"it aside (its groups go back to the pool)")
        target = path.with_name(f"{path.name}.abandoned-{stamp}")
        path.rename(target)
        self.log(f"abandoned half-done batch dir -> {target}; its {len(state['groups'])} group(s) go back to the pool")
        return None

    def set_aside_prefetched(self, path: Path, b: int, stamp: str) -> None:
        """Move prefetched batch b (untouched by the trainer: no train metrics, no in-flight /train, no publish) to
        ``<dir>.prefetched-<stamp>``; its groups return to the pool. Anything else raises."""
        state_path = path / "state.json"
        if not state_path.exists():
            raise OrchestratorError(f"{path} has no state.json: a batch dir past the pending one is only ever a prefetched batch "
                                    f"whose state.json is written right after the dir; inspect it and move it aside by hand")
        state = json.loads(state_path.read_text())
        train = state["train"]
        if (state["batch"] != b or not state.get("prefetched", False) or train["metrics"] or train["inflight"] is not None
                or "publish" in state["stages"] or "sync" in state["stages"]):
            raise OrchestratorError(f"stream resume: {path} (batch {state['batch']}, prefetched {state.get('prefetched')}, "
                                    f"stages {sorted(state['stages'])}, {len(train['metrics'])} minibatch(es) trained, "
                                    f"inflight {train['inflight']}) is not an untouched prefetched batch {b}: inspect it by hand")
        target = path.with_name(f"{path.name}.prefetched-{stamp}")
        path.rename(target)
        self.log(f"prefetched batch {b} the trainer never saw -> {target}; its {len(state['groups'])} group(s) go back to the pool")

    def resume_plan(self, n: int, pending: dict, recorded: dict[int, dict], consumed: set[int],
                    abandoned: dict[int, dict]) -> dict:
        """Continue batch n: its groups must all be completed, not consumed by any row and not recorded as abandoned; the stages must be
        consistent (sync ⇒ publish ⇒ train done ⇒ prepare; train metrics present ⇒ prepare, and no more than prepare's minibatch count)."""
        state, path = pending["state"], pending["dir"] / "state.json"
        gids, stages, trained = state["groups"], state["stages"], len(state["train"]["metrics"])
        dup = sorted(g for g, c in Counter(gids).items() if c > 1)
        missing = [g for g in gids if g not in recorded]
        eaten = [g for g in gids if g in consumed]
        dropped = [g for g in gids if g in abandoned]
        if not gids or dup or missing or eaten or dropped:
            raise OrchestratorError(f"stream resume: {path} lists groups {gids[:8]}: duplicated {dup[:5]}, never completed "
                                    f"{missing[:5]}, consumed by batches.jsonl {eaten[:5]}, abandoned {dropped[:5]}; it cannot "
                                    f"be resumed: fix by hand")
        consistent = (("sync" not in stages or "publish" in stages)
                      and ("publish" not in stages or ("train" in stages and stages["train"]["minibatches"] == trained))
                      and (not trained or ("prepare" in stages and trained <= stages["prepare"]["minibatches"])))
        if not consistent:
            raise OrchestratorError(f"stream resume: {path}: stages {sorted(stages)} with {trained} minibatch(es) trained are "
                                    f"not a state this orchestrator writes: fix by hand")
        return {"dir": pending["dir"], "state": state, "groups": [recorded[g] for g in gids],
                "stages_done": sorted(stages), "minibatches_done": trained,
                # states written before formed_unix was recorded: the second the batch dir was opened
                "formed_unix": state["formed_unix"] if "formed_unix" in state else
                calendar.timegm(time.strptime(state["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))}

    # ---- dispatcher thread
    def reset_stream_stats(self, now: float) -> None:
        carried = ("groups", "episodes", "heldout_groups", "heldout_episodes", "blocked", "disk_blocked", "end_hold")
        with self.cv:
            old = self.stream_stats
            self.stream_stats = {"since": now, "last": now, "launched": 0, "completed": 0, "group_seconds": 0.0,
                                 "episode_seconds": 0.0, "backpressure_seconds": 0.0, "disk_wait_seconds": 0.0,
                                 "end_hold_seconds": 0.0, "heldout_launched": 0, "heldout_finished": 0,
                                 "heldout_group_seconds": 0.0, "heldout_episode_seconds": 0.0,
                                 "judge_started": 0, "judge_attempts_failed": 0,
                                 **({"groups": 0, "episodes": 0, "heldout_groups": 0, "heldout_episodes": 0, "blocked": False,
                                     "disk_blocked": False, "end_hold": False}
                                    if old is None else {k: old[k] for k in carried})}

    def _integrate_stats(self, now: float) -> None:
        """Called inside cv: accumulate running groups / episodes (training and held-out separately), back-pressure and disk-wait time
        from the last tick to now (left endpoint)."""
        st = self.stream_stats
        dt = now - st["last"]
        st["group_seconds"] += st["groups"] * dt
        st["episode_seconds"] += st["episodes"] * dt
        st["heldout_group_seconds"] += st["heldout_groups"] * dt
        st["heldout_episode_seconds"] += st["heldout_episodes"] * dt
        st["backpressure_seconds"] += dt if st["blocked"] else 0.0
        st["disk_wait_seconds"] += dt if st["disk_blocked"] else 0.0
        st["end_hold_seconds"] += dt if st["end_hold"] else 0.0
        st["last"] = now

    def stream_tick(self, procs: dict[int, dict], hprocs: dict[str, dict], blocked: bool, disk_blocked: bool,
                    end_hold: bool) -> None:
        """Recorded once per dispatcher loop: running training groups and their episodes with no results written yet, the same two numbers
        for held-out (``hprocs``), and whether this loop was blocked by back-pressure / wind-down / disk."""
        running, heldout = unfinished_episodes(procs.values()), unfinished_episodes(hprocs.values())
        with self.cv:
            self._integrate_stats(time.monotonic())
            self.stream_stats.update(groups=len(procs), episodes=running, heldout_groups=len(hprocs),
                                     heldout_episodes=heldout, blocked=blocked, disk_blocked=disk_blocked, end_hold=end_hold)

    def take_stream_stats(self) -> dict:
        """Dispatcher statistics (time averages) since the last batch row, then reset (under the same lock: cv is an RLock).
        groups / episodes count only training groups; with in-loop held-out on (stream.heldout_every > 0) four heldout_* keys are
        added (with it off the row is exactly as before)."""
        with self.cv:
            now = time.monotonic()
            self._integrate_stats(now)
            st = self.stream_stats
            span = max(now - st["since"], 1e-9)
            out = {"seconds": round(span, 1), "groups_launched": st["launched"], "groups_completed": st["completed"],
                   "groups_in_flight_mean": round(st["group_seconds"] / span, 2),
                   "episodes_in_flight_mean": round(st["episode_seconds"] / span, 2),
                   "backpressure_seconds": round(st["backpressure_seconds"], 2),
                   "disk_wait_seconds": round(st["disk_wait_seconds"], 2),
                   "end_hold_seconds": round(st["end_hold_seconds"], 2)}
            if self.note_judge is not None:
                out.update(judge_groups_started=st["judge_started"], judge_attempts_failed=st["judge_attempts_failed"])
            if self.heldout_every:
                out.update(heldout_groups_launched=st["heldout_launched"], heldout_groups_finished=st["heldout_finished"],
                           heldout_groups_in_flight_mean=round(st["heldout_group_seconds"] / span, 2),
                           heldout_episodes_in_flight_mean=round(st["heldout_episode_seconds"] / span, 2))
            self.reset_stream_stats(now)
        return out

    def dispatcher_overrides(self, seen_mtime: float | None) -> tuple[float | None, bool]:
        """Hot reload (2026-10-24 arc2 RL review C1): when the mtime of ``<run>/control/dispatcher.json`` changes, re-read it and apply
        its keys to the stream section (only inside this orchestrator process; config.json is not changed and must be edited separately
        for a resume). Changeable keys: DISPATCHER_OVERRIDE_KEYS; rollout_slots may only be lowered, must stay a multiple of the replica
        count and ≥ group_size (raising it would exceed the server's MAXREQ), and ≥ one held-out group's sample count. A broken file or an
        invalid value raises (as loudly as a broken config.json). Returns (new mtime, whether anything was applied)."""
        path = self.root / "control" / DISPATCHER_OVERRIDE_FILE
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None, False
        if mtime == seen_mtime:
            return seen_mtime, False
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict) or not set(doc) <= DISPATCHER_OVERRIDE_KEYS:
            raise OrchestratorError(f"{path}: keys {sorted(doc) if isinstance(doc, dict) else type(doc).__name__} not a "
                                    f"subset of {sorted(DISPATCHER_OVERRIDE_KEYS)}")
        sc, gs = self.stream, self.stream["group_size"]
        for key, value in doc.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 \
                    or (key != "group_timeout_seconds" and not isinstance(value, int)):
                raise OrchestratorError(f"{path}: {key} = {value!r} must be a positive {'number' if key == 'group_timeout_seconds' else 'int'}")
        if "rollout_slots" in doc:
            slots, base = doc["rollout_slots"], self.config_rollout_slots
            replicas = len(self.cfg["rollout"].get("replica_urls") or [])
            if base is None or doc["rollout_slots"] > base or slots < gs or (replicas and slots % replicas):
                raise OrchestratorError(f"{path}: rollout_slots {slots} must be <= config.json's {base}, >= group_size {gs} "
                                        f"and a multiple of the {replicas} replica(s)")
            # a held-out group that no longer fits would block every training group behind it (held-out goes first)
            heldout = max([self.heldout_rounds[v]["samples"] for v, _ in self.heldout_queue]
                          + [sc["heldout_samples"] if self.heldout_every else 0])
            if slots < heldout:
                raise OrchestratorError(f"{path}: rollout_slots {slots} < {heldout} samples of a held-out group "
                                        f"(stream.heldout_samples / an open round): it would never launch")
        before = {k: sc.get(k) for k in doc}
        sc.update(doc)
        self.log(f"dispatcher: {path.name} applied {doc} (was {before}); config.json unchanged")
        return mtime, True

    # ---- note judge (prepare.arc3_reward note)
    def check_judge_backend(self) -> None:
        """Before stream starts, check that the judge backend is usable: the time-limit inequality (validate_note_judge checks it again);
        api needs JUDGE_API_KEY_ENV in the environment; codex needs no CODEX_HOME in the environment (the judge uses the default
        ~/.codex), ~/.codex/auth.json present (existence only, not read), and ``<codex_bin> --version`` succeeding and equal to the
        codex_version recorded in the config (the binary was not swapped). Otherwise a whole run would go by before the first group
        reveals it."""
        try:
            validate_note_judge(self.note_judge)
        except SystemExit as error:
            raise OrchestratorError(f"prepare.arc3_reward note: {error}") from None
        if self.note_judge["backend"] in V8B_JUDGE_BACKENDS:
            # claude -p on the subscription login: an ANTHROPIC_API_KEY in the environment would switch it to API billing
            if os.environ.get("ANTHROPIC_API_KEY"):
                raise OrchestratorError("prepare.arc3_reward note: ANTHROPIC_API_KEY is set: the claude judge runs on the "
                                        "Claude Code subscription login (user 11-01), unset it")
            exe = self.note_judge["claude_bin"]
            run = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
            if run.returncode != 0 or run.stdout.strip() != self.note_judge["claude_version"]:
                raise OrchestratorError(f"prepare.arc3_reward note: `{exe} --version` exited {run.returncode} with "
                                        f"{run.stdout.strip()!r}, the config recorded {self.note_judge['claude_version']!r}")
            if self.note_judge["backend"] == "claude-code":
                nj = self.note_judge
                self.log(f"note judge: claude-code mapping {nj['mapping']}: {exe} ({nj['claude_version']}), subscription "
                         f"login, model {nj['claude_model']} effort {nj['claude_effort']}, one pass; call timeout "
                         f"{nj['call_timeout_seconds']:g}s, attempt timeout {nj['timeout_seconds']:g}s")
                return
        if self.note_judge["backend"] == "api":
            if not os.environ.get(JUDGE_API_KEY_ENV):
                raise OrchestratorError(f"prepare.arc3_reward note: the api note judge needs {JUDGE_API_KEY_ENV} in the "
                                        f"orchestrator's environment (launch_orchestrator.sh sources note_judge.env_file "
                                        f"{self.note_judge['env_file']} before it execs the orchestrator)")
            return
        if "CODEX_HOME" in os.environ:
            raise OrchestratorError(f"prepare.arc3_reward note: CODEX_HOME={os.environ['CODEX_HOME']} is set: the codex "
                                    f"judge runs with the default ~/.codex (the user's login), unset it")
        auth = Path.home() / ".codex" / "auth.json"
        if not auth.is_file():
            raise OrchestratorError(f"prepare.arc3_reward note: the codex judge needs the ChatGPT login {auth} (codex login)")
        exe = self.note_judge["codex_bin"]
        run = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
        if run.returncode != 0:
            raise OrchestratorError(f"prepare.arc3_reward note: `{exe} --version` exited {run.returncode}: "
                                    f"{(run.stderr or run.stdout)[-500:]}")
        if run.stdout.strip() != self.note_judge["codex_version"]:
            raise OrchestratorError(f"prepare.arc3_reward note: `{exe} --version` is {run.stdout.strip()!r}, the config "
                                    f"recorded {self.note_judge['codex_version']!r} (the codex binary changed since configure)")
        nj = self.note_judge
        if nj["backend"] == "ensemble":
            self.log(f"note judge: ensemble mapping {nj['mapping']}: claude backend {nj['claude_bin']} "
                     f"({nj['claude_version']}), subscription login, model {nj['claude_model']} effort {nj['claude_effort']}; "
                     f"codex backend {exe} ({run.stdout.strip()}), login {auth}, --ignore-user-config, model "
                     f"{nj['codex_model']} effort {nj['codex_effort']}; call timeout {nj['call_timeout_seconds']:g}s, "
                     f"attempt timeout {nj['timeout_seconds']:g}s")
            return
        self.log(f"note judge: codex backend {exe} ({run.stdout.strip()}), login {auth}, --ignore-user-config, model "
                 f"{nj['model']} effort {nj['effort']}, call timeout "
                 f"{nj['call_timeout_seconds']:g}s, attempt timeout {nj['timeout_seconds']:g}s")

    def judge_input_needed(self, entry: dict) -> dict | None:
        """Whether a group with rc 0 still needs the note judge: arc3_reward note, it has games to judge (note_judge_input is not None),
        and the group dir has no judge.json yet (one already present on resume is reused, not re-judged). If so, returns the judge
        input, otherwise None."""
        if self.note_judge is None or (entry["run_dir"] / JUDGE_FILE).exists():
            return None
        run_dir, task = entry["run_dir"], entry["task"]
        manifest = json.loads((run_dir / "tito_manifest.json").read_text())
        episodes = read_group_episodes(run_dir, task, self.stream["group_size"], manifest_failed_ids(manifest, run_dir))
        return note_judge_input(run_dir, task, episodes)

    def start_judge(self, gid: int, entry: dict, doc: dict) -> dict:
        """Write judge_input.json in the group dir and start a worker thread running the judge (judge_worker); returns the job the
        dispatcher tracks."""
        atomic_write(entry["run_dir"] / JUDGE_INPUT_FILE, json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        job = {"gid": gid, "entry": entry, "done": threading.Event(), "ok": False, "stopped": False, "errors": [],
               "crash": None}
        job["thread"] = threading.Thread(target=self.judge_worker, args=(job,), name=f"judge-g{gid:06d}")
        job["thread"].start()
        with self.cv:
            self.stream_stats["judge_started"] += 1
        nj = self.note_judge
        who = (f"ensemble {nj['claude_model']}/{nj['claude_effort']} + {nj['codex_model']}/{nj['codex_effort']}"
               if nj["backend"] == "ensemble" else f"claude-code {nj['claude_model']}/{nj['claude_effort']}"
               if nj["backend"] == "claude-code" else f"{nj['backend']} {nj['model']}, effort {nj['effort']}")
        self.log(f"group {entry['name']} ({entry['task']}): {len(doc['notes'])} note(s) to the judge ({who}, seed {gid})")
        return job

    def kill_judge(self, gid: int, proc: subprocess.Popen) -> None:
        """Terminate a judge subprocess (popen: start_new_session, pgid = sid = its pid): SIGTERM to its process group (the judge's own
        handler kills the codex exec group and cleans its temp dir); if the judge has not exited within terminate_grace_seconds, SIGKILL
        the group; after that, in any case SIGKILL whatever is left in its whole session (each codex exec has its own process group, but
        all are in the judge's session). Raises only if something is still there KILL_WAIT_SECONDS after SIGKILL."""
        name = f"judge-g{gid:06d}"
        if proc.poll() is None:
            grace = self.cfg["harness"]["terminate_grace_seconds"]
            self.log(f"{name}: SIGTERM to its process group {proc.pid}")
            killpg_quietly(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.log(f"{name}: still running {grace}s after SIGTERM -> SIGKILL to process group {proc.pid}")
                killpg_quietly(proc.pid, signal.SIGKILL)
                proc.wait(timeout=KILL_WAIT_SECONDS)
        # the judge's session (start_new_session: sid = its pid) holds its codex exec groups (process_group=0) too
        killpg_quietly(proc.pid, signal.SIGKILL)
        left = session_pids(proc.pid)
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if left:
            self.log(f"{name}: session {proc.pid} still had process(es) {left} after the judge exited: SIGKILLed them")

    def judge_worker(self, job: dict) -> None:
        """One group's judge: start the subprocess (``note_judge.python`` arc3_note_judge.py judge_input.json judge.json) only once a
        judge_slots place is free, each attempt for at most timeout_seconds (timeout: kill_judge takes down the whole process group), log
        in the group dir as judge.<session>.attempt<N>.log; rc 0 with judge.json written = success, otherwise wait JUDGE_BACKOFF_SECONDS
        and retry, at most JUDGE_ATTEMPTS times (failure reasons go to job["errors"] and the dispatcher stops the run). On judge_stop
        (dispatcher ending): kill_judge, no more retries (job["stopped"]). An error in this thread itself (cannot start the subprocess…)
        is recorded as job["crash"], and the dispatcher raises it all the same."""
        nj, gid, run_dir = self.note_judge, job["gid"], job["entry"]["run_dir"]
        try:
            with self.judge_slots:
                for attempt in range(1, JUDGE_ATTEMPTS + 1):
                    if self.judge_stop.is_set():
                        job["stopped"] = True
                        return
                    out = run_dir / f"judge.{self.session_id}.attempt{attempt}.log"
                    argv = [nj["python"], str(NOTE_JUDGE_SCRIPT), str(run_dir / JUDGE_INPUT_FILE), str(run_dir / JUDGE_FILE),
                            *judge_backend_args(nj), "--seed", str(gid)]
                    proc, files = self.popen({"argv": argv, "stdout": out})
                    began, why = time.monotonic(), None
                    try:
                        while proc.poll() is None:
                            if self.judge_stop.is_set() or time.monotonic() - began > nj["timeout_seconds"]:
                                why = "stopped" if self.judge_stop.is_set() else f"timeout {nj['timeout_seconds']:g}s"
                                break
                            self.judge_stop.wait(min(1.0, self.poll))
                    finally:
                        self.kill_judge(gid, proc)    # also on an error here: the judge's process group never outlives this
                        for f in files:
                            f.close()
                    if why == "stopped":
                        job["stopped"] = True
                        return
                    if why is None and proc.returncode == 0 and (run_dir / JUDGE_FILE).is_file():
                        job["ok"] = True
                        return
                    error = (f"attempt {attempt}/{JUDGE_ATTEMPTS}: {why or f'rc {proc.returncode}'}"
                             + ("" if proc.returncode != 0 or why else f", no {JUDGE_FILE}") + f"; log {out}")
                    job["errors"].append(error)
                    with self.cv:
                        self.stream_stats["judge_attempts_failed"] += 1
                    if attempt < JUDGE_ATTEMPTS:
                        wait = JUDGE_BACKOFF_SECONDS[attempt - 1]
                        self.log(f"note judge g{gid:06d}: {error}; next attempt in {wait:g}s")
                        if self.judge_stop.wait(wait):
                            job["stopped"] = True
                            return
                    else:
                        self.log(f"note judge g{gid:06d}: {error}")
        except BaseException as error:            # recorded and re-raised by the dispatcher; never swallowed
            job["crash"] = error
        finally:
            job["done"].set()
            with self.cv:
                self.cv.notify_all()

    def finish_judged(self, judging: dict[int, dict]) -> None:
        """Each dispatcher loop: groups whose judge is done; successful ones complete and enter the pool as usual (finish_group). All
        JUDGE_ATTEMPTS failed: **stop the run** (OrchestratorError naming the group and the last error), no quarantine (a judge failure is
        not a problem of this group's rollout, and the circuit breaker does not count it); the group stays in its group dir (harness
        finished, no judge.json), and on restart stream_start_check hands it back to the dispatcher for re-judging. An error in the
        worker itself also raises."""
        for g in sorted(judging):
            job = judging[g]
            if not job["done"].is_set():
                continue
            job["thread"].join()
            if job["crash"] is not None:
                raise OrchestratorError(f"note judge worker of g{g:06d} failed: {job['crash']!r}") from job["crash"]
            if not job["ok"]:
                # stays in ``judging``: stop_judges joins it, nothing completes or quarantines it
                raise OrchestratorError(
                    f"NOTE JUDGE FAILED for {job['entry']['name']} ({job['entry']['task']}, {job['entry']['run_dir']}) "
                    f"{len(job['errors'])} time(s), last: {job['errors'][-1]}. The group's harness finished and it has no "
                    f"{JUDGE_FILE}: fix the judge and restart the orchestrator stage only (RL_STAGES=orchestrator; a full "
                    f"start restarts the servers and abandons unjudged groups): it judges the group again; or set it aside "
                    f"(never trained) with RL_ORCH_ARGS=\"--set-aside-judge-groups {job['entry']['name']}\"")
            del judging[g]
            self.finish_group(g, job["entry"], len(judging))

    def stop_judges(self, judging: dict[int, dict]) -> None:
        """Dispatcher ending: tell every judge worker to stop (kill_judge takes down the whole process group of its subprocess, no more
        retries) and wait for them to finish. Unfinished groups stay in their group dirs and are not in groups.jsonl: the next start's
        stream_start_check hands them back to the dispatcher for re-judging."""
        self.judge_stop.set()
        for job in judging.values():
            job["thread"].join()

    def finish_group(self, g: int, entry: dict, in_flight: int) -> None:
        """A group with harness rc 0 (for note, also with the judge done): complete_group, enter the pool."""
        record = self.complete_group(g, entry)
        self.live_history.append(record["live"])
        self.group_seconds.append(record["seconds"])
        if self.stream.get("retire_full_groups", False):
            self.retire_task(record)
        with self.cv:
            del self.inflight_groups[g]
            if self.cfg["rollout"].get("replica_urls"):
                del self.group_backends[g]
            self.finish_order.append(g)
            self.pool.append(record)
            self.stream_stats["completed"] += 1
            pooled, pooled_live = len(self.pool), sum(r["live"] for r in self.pool)
            self.cv.notify_all()
        self.log(f"group {record['group']} ({record['task']}) done in {record['seconds']:.0f}s: rewards "
                 f"{record['rewards']} {'live' if record['live'] else 'dead'}"
                 + (f" note R {[round(x, 4) for x in record['note_rewards']]}" if "note_rewards" in record else "")
                 + f"; pool {pooled} ({pooled_live} live), {in_flight} in flight")

    def dispatcher_loop(self, next_gid: int, adopted: list[dict] = (), heldout_adopted: list[dict] = ()) -> str:
        """At most max_groups_in_flight group processes; when one ends (rc 0), read its results, take a ledger snapshot, write
        groups.jsonl, put it in the pool, then launch another. No launch while the pool's live groups ≥ max_pooled_groups; none while
        winding down if what is at hand covers the remaining batches (end_of_run_supply), recomputed every loop, launching again if
        running groups come back dead and fall short; none while disk_guard_paths has less than min_free_gb free (wait, do not stop the
        run: running groups keep going, and the learner stops only below stream.hard_min_free_gb). STOP file or learner done: no more
        launches; return once running groups have finished.
        A group with rc≠0, or one that times out (SIGTERM, SIGKILL to its process group after terminate_grace_seconds, neither blocking
        dispatch): quarantine that group (quarantine_harness_group) and keep dispatching. If this thread itself fails (circuit breaker
        trips, reading results fails…), the other thread dies, or on detach (SIGUSR1 / DETACH file): in-flight groups **keep running**
        (stream.keep_groups_on_failure, default true; a restarted orchestrator adopts them from launch.json, SPEC §9); only cancellation
        (SIGTERM / SIGINT other than STOP) terminates every group process it started.
        ``adopted``: entries adopted by stream_start_check (AdoptedProc), managed as its own.
        In-loop held-out (stream.heldout_every, SPEC §9 "held-out"): a round opens when the server version reaches a multiple of
        heldout_every (open_due_heldout_round), and its not-yet-dispatched groups go **ahead** of new training groups: while that queue
        is not empty only held-out is dispatched, under the same rollout_slots / ramp / disk rules, with assign_samples distributing by
        each replica's running episodes (training and held-out counted together); back-pressure, wind-down supply and
        max_groups_in_flight do not apply to it (held-out does not enter the pool). Held-out groups live in a separate ``hprocs``
        (``heldout_adopted``: adopted on resume) and on finishing only write heldout.jsonl (finish_heldout_group: a failure or timeout
        just records a failed row, no quarantine, no stop); timeout, cancel, detach and adoption work as for training groups. No new
        rounds after STOP / learner done; when the learner is done (normal end of run), the open round's remaining groups are still
        dispatched and awaited before returning, while on STOP they are not (the round stays in heldout_state.json and the next start
        continues dispatching it)."""
        sc = self.stream
        window, max_pooled, timeout = sc["max_groups_in_flight"], sc["max_pooled_groups"], sc["group_timeout_seconds"]
        slots = sc.get("rollout_slots")       # None: the group-count rule only
        gs = sc["group_size"]
        ramp = sc.get("ramp_seconds", 0)      # 0: no ramp
        ramp_cap = window * gs if slots is None else min(slots, window * gs)
        stagger = self.cfg["harness"]["launch_stagger_seconds"]
        grace = self.cfg["harness"]["terminate_grace_seconds"]
        procs: dict[int, dict] = {}
        hprocs: dict[str, dict] = {e["name"]: e for e in heldout_adopted}
        judging: dict[int, dict] = {}         # gid -> note judge job (start_judge): harness done, not in the pool yet
        gid, last_launch, stopping, launched, disk_waiting, end_holding = next_gid, None, None, 0, False, False
        adopted_running = unfinished_episodes(adopted) + unfinished_episodes(heldout_adopted)
        for entry in adopted:
            g = int(entry["name"][1:])
            procs[g] = entry
            with self.cv:
                self.inflight_groups[g] = entry["started_mono"]
                if self.cfg["rollout"].get("replica_urls"):
                    if entry["backends"] is None:
                        raise OrchestratorError(f"group {entry['name']}: launch record has no sample backends, config has replica_urls")
                    self.group_backends[g] = entry["backends"]
        live_target = 2 * math.ceil(sc["batch_live_episodes"] / sc["group_size"])     # two batches' worth of live groups
        rule = "" if slots is None else f", a new one while running episodes + {sc['group_size']} <= rollout_slots {slots}"
        if ramp:
            rule += f", running episodes ramped from {gs} to {ramp_cap} over {ramp:.0f}s"
        if self.heldout_every:
            rule += (f"; held-out every {self.heldout_every} version(s) ({sc['heldout_samples']} sample(s) per task, "
                     f"{len(self.pools['heldout'])} task(s), v0 {'on' if sc.get('heldout_at_v0', True) else 'off'}) "
                     f"before any new training group")
        self.log(f"dispatcher: up to {window} group(s) of {sc['group_size']} in flight{rule}, back-pressure at "
                 f"{max_pooled} pooled live group(s); next group g{gid:06d}")
        ramp_start, ramping = time.monotonic(), ramp > 0
        if ramping and adopted_running > gs and ramp_cap > gs:    # the ramp starts where the adopted groups already are
            ramp_start -= ramp * min(1.0, (adopted_running - gs) / (ramp_cap - gs))
        if adopted or heldout_adopted:
            self.log(f"dispatcher: {len(adopted)} adopted group(s)"
                     + (f" and {len(heldout_adopted)} adopted held-out group(s)" if heldout_adopted else "")
                     + f" in flight ({adopted_running} running episode(s))")
        control_mtime = None
        try:
            for entry, doc in self.judge_resume:      # finished before the restart, judge missing (stream_start_check)
                g = int(entry["name"][1:])
                with self.cv:
                    self.inflight_groups[g] = entry["started_mono"]
                    if self.cfg["rollout"].get("replica_urls"):
                        self.group_backends[g] = entry["backends"]
                judging[g] = self.start_judge(g, entry, doc)
            self.judge_resume = []
            while True:
                self.check_cancel()
                control_mtime, changed = self.dispatcher_overrides(control_mtime)
                if changed:
                    window, max_pooled, timeout = sc["max_groups_in_flight"], sc["max_pooled_groups"], sc["group_timeout_seconds"]
                    slots = sc.get("rollout_slots")
                    ramp_cap = window * gs if slots is None else min(slots, window * gs)
                if ramping and time.monotonic() - ramp_start >= ramp:
                    ramping = False
                    self.log(f"dispatcher ramp done after {ramp:.0f}s: up to {ramp_cap} running episode(s) from now on")
                for g in sorted(procs):
                    entry = procs[g]
                    rc = self.poll_group(entry, timeout, grace)
                    if rc is None:
                        continue
                    for f in entry["files"]:
                        f.close()
                    del procs[g]
                    if "terminating_since" in entry:
                        how = "SIGKILLed" if "killed_at" in entry else "terminated"
                        self.quarantine_harness_group(g, entry, f"group timeout: not finished after {timeout:.0f}s "
                                                                f"(stream.group_timeout_seconds), {how}")
                        continue
                    if rc != 0:
                        self.quarantine_harness_group(g, entry, harness_failure(entry["run_dir"], entry["stdout"], rc))
                        continue
                    self.snapshot_ledger(entry)       # now, not after its judge: complete_group checks and records this
                    doc = self.judge_input_needed(entry)
                    if doc is not None:       # arc3_reward note: into the pool once its judge.json exists (finish_judged)
                        judging[g] = self.start_judge(g, entry, doc)
                        continue
                    self.finish_group(g, entry, len(procs))
                self.finish_judged(judging)
                for name in sorted(hprocs):
                    entry = hprocs[name]
                    rc = self.poll_group(entry, timeout, grace)
                    if rc is None:
                        continue
                    for f in entry["files"]:
                        f.close()
                    del hprocs[name]
                    self.finish_heldout_group(entry, rc)
                    with self.cv:
                        self.stream_stats["heldout_finished"] += 1
                if stopping is None:
                    with self.cv:
                        finished = self.learner_finished
                    if (self.root / "STOP").exists():
                        stopping = "stopped"
                        self.log(f"STOP file present: no new group; waiting for {len(procs) + len(hprocs)} in flight"
                                 + (f" and {len(judging)} at the note judge" if judging else ""))
                    elif finished:
                        stopping = "done"
                        self.log(f"the learner trained every batch: no new group; waiting for {len(procs) + len(hprocs)} "
                                 f"in flight" + (f" and launching the {len(self.heldout_queue)} held-out group(s) left of "
                                                 f"the open round" if self.heldout_queue else ""))
                # the run's normal end drains the open held-out round (no new round opens: heldout_round_due); STOP does not
                draining = stopping == "done" and bool(self.heldout_queue)
                if stopping is not None and not procs and not hprocs and not draining and not judging:
                    with self.cv:
                        self.dispatch_ended = True
                        unconsumed = len(self.pool)
                        self.cv.notify_all()
                    left = Counter(v for v, _ in self.heldout_queue)
                    self.log(f"dispatcher done: {launched} group(s) launched this session, {unconsumed} completed group(s) "
                             f"in the pool (unconsumed in groups.jsonl)"
                             + (f"; held-out task(s) never launched: {', '.join(f'v{v:06d} x{n}' for v, n in sorted(left.items()))}"
                                f" (their round stays open in {HELDOUT_STATE_FILE})" if left else ""))
                    return stopping
                if stopping is None:
                    self.open_due_heldout_round()
                blocked = disk_blocked = end_hold = False
                while stopping is None or (stopping == "done" and self.heldout_queue):
                    # held-out first: while a round has unlaunched groups no training group is launched (it would take the
                    # slot the held-out group waits for); back-pressure and the end-of-run hold are training-only
                    heldout = self.heldout_queue[0] if self.heldout_queue else None
                    if heldout is None:
                        if stopping is not None or len(procs) >= window:
                            break
                        gid = self.skip_retired(gid)
                        with self.cv:
                            pooled_live = self.pooled_live_groups()
                            batches_left = self.cfg["rounds"] - self.batches_taken
                        if pooled_live >= max_pooled:
                            blocked = True
                            break
                        fraction = recent_live_fraction(self.live_history, live_target)
                        counted = self.supply_in_flight(procs) + len(judging)   # judged groups are all but done
                        end_hold, supply, demand = end_of_run_supply(pooled_live, counted, fraction, batches_left, sc)
                        how = (f"supply {supply:.0f} live episode(s) (pool {pooled_live} live group(s) + {counted} of {len(procs)} in "
                               f"flight x {sc['group_size']} x recent live fraction "
                               f"{'n/a' if fraction is None else format(fraction, '.2f')}) vs end_margin {sc['end_margin']} x "
                               f"demand {demand} ({batches_left} batch(es) left)")
                        if end_hold:
                            if not end_holding:
                                end_holding = True
                                self.log(f"end of run: {how}: no new group while that holds")
                            break
                        if end_holding:
                            end_holding = False
                            self.log(f"end-of-run hold lifted: {how}: launching again")
                    elif slots is None:
                        raise OrchestratorError(f"internal: held-out {heldout} queued without stream.rollout_slots (the config "
                                                f"check requires it with stream.heldout_every)")
                    short = self.disk_short(self.cfg["min_free_gb"])
                    if short:
                        disk_blocked = True
                        if not disk_waiting:
                            disk_waiting = True
                            self.log(f"disk: {'; '.join(short)} < min_free_gb {self.cfg['min_free_gb']}: no new group until "
                                     f"space is freed ({len(procs) + len(hprocs)} in flight keep running; the learner stops "
                                     f"below stream.hard_min_free_gb {sc['hard_min_free_gb']})")
                        break
                    if disk_waiting:
                        disk_waiting = False
                        self.log(f"disk: every disk_guard_path has >= min_free_gb {self.cfg['min_free_gb']} again: launching")
                    size = gs if heldout is None else self.heldout_rounds[heldout[0]]["samples"]
                    if slots is not None or ramping:
                        running = (sum(self.replica_running(procs, hprocs)) if slots is not None else
                                   unfinished_episodes(procs.values()) + unfinished_episodes(hprocs.values()))
                        if slots is not None and running + size > slots:
                            break
                        if ramping and running + size > ramp_allowed(time.monotonic() - ramp_start, ramp_cap, gs, ramp):
                            break
                    if last_launch is not None and stagger:
                        self.sleep(max(0.0, last_launch + stagger - time.monotonic()))
                    if heldout is None:
                        procs[gid] = self.launch_group(gid, procs, hprocs)
                        gid += 1
                        launched += 1
                    else:
                        self.heldout_queue.pop(0)
                        entry = self.launch_heldout_group(*heldout, procs, hprocs)
                        hprocs[entry["name"]] = entry
                    last_launch = time.monotonic()
                self.stream_tick(procs, hprocs, blocked, disk_blocked, end_hold)
                with self.cv:
                    if not (self.cancelled.is_set() or self.aborted.is_set()):
                        self.cv.wait(min(self.poll, CHILD_POLL_SECONDS))
        except BaseException as error:
            keep = self.detaching or (not isinstance(error, Cancelled) and sc.get("keep_groups_on_failure", True))
            entries = [*procs.values(), *hprocs.values()]
            live = sorted(e["name"] for e in entries if e["proc"].poll() is None)
            if keep and live:
                self.log(f"dispatcher stops ({'detach' if self.detaching else type(error).__name__}) with {len(live)} group(s) "
                         f"left running: {live}. They finish on their own (results, EXIT.json); the next orchestrator start "
                         f"adopts or completes them from their {LAUNCH_SUFFIX} records")
            elif live:
                for entry in entries:        # run_arc2 finishes its results on SIGTERM: without the marker a restart would pool them
                    if entry["proc"].poll() is None:
                        self.mark_terminated(entry["name"], "cancel", entry["run_dir"])
                self.terminate({e["name"]: e["proc"] for e in entries})
            raise
        finally:
            self.stop_judges(judging)
            for entry in [*procs.values(), *hprocs.values()]:
                for f in entry["files"]:
                    f.close()

    def poll_group(self, entry: dict, timeout: float, grace: float) -> int | None:
        """dispatcher: returns rc when an in-flight group (training or held-out) has ended, None while it is still running. Past
        ``timeout`` (stream.group_timeout_seconds, clock from launch, adopted ones too): write the terminated marker first, then SIGTERM
        it alone (the harness winds down episodes, closes sessions and exits on its own; dispatch does not block), and SIGKILL its process
        group after terminate_grace_seconds; bwrap / codex / gateway are all in the harness's session, so nothing holds a rollout slot
        afterwards; raise if it is still there KILL_WAIT_SECONDS after SIGKILL.
        An ended entry with ``terminating_since`` = taken down by us (the caller treats it as a timeout, regardless of rc)."""
        rc = entry["proc"].poll()
        if rc is not None:
            return rc
        since = entry.get("terminating_since")
        if since is None:
            if time.monotonic() - entry["started_mono"] > timeout:
                self.log(f"group {entry['name']} ({entry['task']}): not finished after {timeout:.0f}s "
                         f"(stream.group_timeout_seconds): SIGTERM; its process group is SIGKILLed in {grace:.0f}s")
                self.mark_terminated(entry["name"], "timeout", entry["run_dir"])
                entry["proc"].terminate()
                entry["terminating_since"] = time.monotonic()
        elif "killed_at" not in entry and time.monotonic() - since > grace:
            self.log(f"group {entry['name']} ({entry['task']}): still running {grace:.0f}s after SIGTERM: "
                     f"SIGKILL to process group {entry['proc'].pid}")
            killpg_quietly(entry["proc"].pid, signal.SIGKILL)
            entry["killed_at"] = time.monotonic()
        elif "killed_at" in entry and time.monotonic() - entry["killed_at"] > KILL_WAIT_SECONDS:
            raise OrchestratorError(f"group {entry['name']} ({entry['task']}): pid {entry['proc'].pid} is still "
                                    f"there {KILL_WAIT_SECONDS}s after SIGKILL (unkillable: D state / zombie "
                                    f"nobody reaps?); check ps. Log {entry['stdout']}")
        return None

    def supply_in_flight(self, procs: dict[int, dict]) -> int:
        """In-flight groups that count toward the wind-down supply: age < 1.5 × the median seconds of groups completed in this session
        (estimated only with at least MEDIAN_MIN_GROUPS; otherwise all count). Those over it are mostly stragglers that hit wall /
        timeout; counting them in the supply would launch too few during wind-down and leave the last batch short (2026-10-24 review)."""
        if len(self.group_seconds) < MEDIAN_MIN_GROUPS:
            return len(procs)
        cutoff = 1.5 * sorted(self.group_seconds)[len(self.group_seconds) // 2]
        return sum(1 for e in procs.values() if time.monotonic() - e["started_mono"] < cutoff)

    def retire_task(self, record: dict) -> None:
        """stream.retire_full_groups: every episode of the just-completed group has R = 1 → this task retires (one row in retired.jsonl,
        no further dispatch). In-flight groups of the same task run to completion and enter the pool as usual (a group with all R = 1 is
        std0 in prepare and not trained)."""
        task = record["task"]
        if task in self.retired or not group_full(record["episodes"], self.cfg["prepare"]):
            return
        self.retired.add(task)
        path = self.root / RETIRED_FILE
        if path.exists() and task in {json.loads(line)["task"] for line in path.read_text().splitlines()}:
            return      # stream start re-reading a task retired before the restart
        with self.cv:
            version = self.server_version
        append_jsonl(self.root / RETIRED_FILE, {"task": task, "gid": record["gid"], "group": record["group"],
                                                "server_version_launch": record["server_version_launch"],
                                                "server_version": version, "utc": utc_now()})
        self.log(f"task {task} retired: group {record['group']} has R = 1 on all {len(record['episodes'])} episodes "
                 f"({len(self.retired)} of {len(self.pools['train'])} train task(s) retired)")

    def skip_retired(self, gid: int) -> int:
        """stream.retire_full_groups / retire_tasks_file: the first gid from ``gid`` on whose task is not retired (skipped gids are not
        dispatched or written to stream_state, leaving gaps in the ids; task_of's mapping is unchanged, so resume checks work as before).
        Every train task retired: raise (nothing to train)."""
        if not self.retired:
            return gid
        if set(self.pools["train"]) <= self.retired:
            raise OrchestratorError(f"retire: every one of the {len(self.pools['train'])} train tasks is retired: "
                                    f"nothing left to dispatch")
        while (task := self.task_of(gid)) in self.retired:
            self.log(f"group g{gid:06d} ({task}) skipped: task retired")
            gid += 1
        return gid

    def replica_running(self, procs: dict[int, dict], heldout: dict[str, dict] | None = None) -> list[int]:
        """Running episodes per replica (index in rollout.replica_urls): samples of running groups with no results written yet.
        Before 2026-10-15 a whole group went to one replica and load was counted in groups, so slots freed by finished samples were
        invisible. ``heldout``: in-flight held-out groups (the dispatcher's hprocs) are counted too; they occupy the same slots, with
        ``samples`` each (that round's heldout_samples, not group_size), and where they were sent is in the entry's ``assigned``."""
        gs = self.stream["group_size"]
        load = [0] * len(self.cfg["rollout"]["replica_urls"])
        entries = [(entry, self.group_backends[g], gs, "group_size") for g, entry in procs.items()]
        entries += [(entry, entry["assigned"], entry["samples"], "heldout_samples") for entry in (heldout or {}).values()]
        for entry, assigned, size, what in entries:
            if len(assigned) != size:
                raise OrchestratorError(f"group {entry['name']}: {len(assigned)} sample backends for {what} {size}")
            results = entry["run_dir"] / "results"
            for k, r in enumerate(assigned):
                if not (results / f"{entry['task']}__s{k}.json").exists():
                    load[r] += 1
        return load

    def launch_group(self, gid: int, procs: dict[int, dict], heldout: dict[str, dict] | None = None) -> dict:
        """One group-mode harness process (SPEC §9). The gid is written to stream_state.json before launch, an existing dir raises, and
        a failed group is not retried: a gid is never reused. Prefix-cache salt ``<run>/<session>/gNNNNNN/<episode>``: session is this
        start's time + random, so even if run_root is wiped and restarted without a server restart, the salt cannot hit KV computed
        with old weights."""
        name = f"g{gid:06d}"
        task = self.task_of(gid)
        run_dir = self.group_dir(gid)
        stdout = run_dir.with_name(f"{name}.out")
        if run_dir.exists() or stdout.exists():
            raise OrchestratorError(f"{run_dir} (or its .out) already exists: group ids are never reused (stream_state.json?)")
        atomic_write(self.root / "stream_state.json", json.dumps({"next_gid": gid + 1}) + "\n")
        gs = self.stream["group_size"]
        salt_prefix = f"{self.root.name}/{self.session_id}/{name}"
        replicas = self.cfg["rollout"].get("replica_urls") or []  # none: every group goes to base_url
        running = self.replica_running(procs, heldout) if replicas else []
        assigned = assign_samples(running, gs) if replicas else None
        argv = self.harness_argv(run_dir, cfg_path(self.cfg["tasks"]["train"]["dir"]), [task],
                                 ["--samples", str(gs), "--salt-prefix", salt_prefix],
                                 backends=None if assigned is None else [replicas[r] for r in assigned])
        roster = sorted(row["instance"] for row in ledger_statuses(http_json("GET", self.server_root + "/session_ledger/status")))
        proc, files = self.popen({"argv": argv, "stdout": stdout})
        started, launched_unix, launched_utc = time.monotonic(), time.time(), utc_now()
        with self.cv:
            version = self.server_version
            self.inflight_groups[gid] = started
            if assigned is not None:
                self.group_backends[gid] = assigned
            self.stream_stats["launched"] += 1
            in_flight = len(self.inflight_groups)
        starttime = proc_starttime(proc.pid)
        if starttime is None:                           # died before we looked: the dispatcher's poll will see its rc
            self.log(f"group {name} ({task}): harness pid {proc.pid} is already gone right after launch")
        # the hand-over record for a later orchestrator (SPEC §9 adoption): everything the dispatcher knows about this group
        atomic_write(run_dir.with_name(name + LAUNCH_SUFFIX), json.dumps(
            {"gid": gid, "group": name, "task": task, "salt_prefix": salt_prefix, "session_id": self.session_id,
             "pid": proc.pid, "proc_starttime": starttime, "launched_unix": launched_unix, "launched_utc": launched_utc,
             "server_version_launch": version, "backends": assigned, "ledger_instances": roster}, indent=1) + "\n")
        where = ""
        if assigned is not None:
            spread = " ".join(f"r{r}x{assigned.count(r)}" for r in sorted(set(assigned)))
            after = [n + assigned.count(r) for r, n in enumerate(running)]
            where = f" on {spread} (running {'/'.join(map(str, running))} -> {'/'.join(map(str, after))})"
        self.log(f"group {name} ({task}) launched at server version {version}{where}; {in_flight} in flight")
        return {"name": name, "task": task, "run_dir": run_dir, "stdout": stdout, "proc": proc, "files": files,
                "started_mono": started, "launched_utc": launched_utc, "version": version, "salt_prefix": salt_prefix,
                "samples": gs}

    def adopt_entry(self, doc: dict) -> dict:
        """launch.json → the dispatcher's procs entry (AdoptedProc; no files to close; the clock runs from the original launch time,
        so group_timeout applies to it as before)."""
        gid, name = doc["gid"], doc["group"]
        run_dir = self.group_dir(gid)
        age = max(0.0, time.time() - doc["launched_unix"])
        return {"name": name, "task": doc["task"], "run_dir": run_dir, "stdout": run_dir.with_name(f"{name}.out"),
                "proc": AdoptedProc(doc["pid"], doc["proc_starttime"], run_dir), "files": [],
                "started_mono": time.monotonic() - age, "launched_utc": doc["launched_utc"],
                "version": doc["server_version_launch"], "salt_prefix": doc["salt_prefix"], "adopted": True,
                "backends": doc["backends"], "samples": self.stream["group_size"]}

    def read_launch(self, gid: int) -> dict | None:
        """``groups/gNNNNNN.launch.json``; missing = started by an old orchestrator (cannot be adopted). Wrong keys raise."""
        path = self.group_dir(gid).with_name(f"g{gid:06d}{LAUNCH_SUFFIX}")
        if not path.is_file():
            return None
        doc = json.loads(path.read_text())
        if set(doc) != LAUNCH_KEYS or doc["gid"] != gid:
            raise OrchestratorError(f"{path}: keys {sorted(doc)} / gid {doc.get('gid')!r}, want {sorted(LAUNCH_KEYS)} / {gid}")
        return doc

    def snapshot_ledger(self, entry: dict) -> None:
        """At the moment of harness rc 0 (dispatcher; for groups finished on resume, at stream start) record ``/session_ledger/status``
        in entry["ledger_status"]: complete_group uses it to check the roster and record enqueued_events. A note group's judge may run
        for hours; taking the snapshot after the judge would attribute events queued by other groups in the meantime to this one, and a
        server restart in that window would also make it mismatch."""
        if "ledger_status" in entry:
            raise OrchestratorError(f"group {entry['name']}: ledger status snapshotted twice")
        entry["ledger_status"] = http_json("GET", self.server_root + "/session_ledger/status")

    def complete_group(self, gid: int, entry: dict) -> dict:
        """Group with rc 0: read its group_size results, check the manifest, check the roster against the ledger snapshot taken when the
        harness ended (snapshot_ledger), record enqueued_events (events queued later do not belong to it), and write one groups.jsonl
        row (complete only once written)."""
        gs = self.stream["group_size"]
        run_dir, task = entry["run_dir"], entry["task"]
        manifest_path = run_dir / "tito_manifest.json"
        if not manifest_path.is_file():
            raise OrchestratorError(f"{run_dir}: no tito_manifest.json (harness did not write it)")
        manifest = json.loads(manifest_path.read_text())
        items = sorted(item["id"] for item in manifest["items"])
        if items != sorted(group_episode_ids(task, gs)):
            raise OrchestratorError(f"{manifest_path}: items {items[:4]} are not the {gs} episodes of {task}")
        # a few failed episodes do not fail the group (run_arc3_level, <= 25%): they are status failed in the manifest,
        # have no results file and no thread; prepare_rl drops them as harness_failed. Here they are simply not episodes
        failed = manifest_failed_ids(manifest, run_dir)
        episodes = read_group_episodes(run_dir, task, gs, failed)
        if failed:
            self.log(f"group {entry['name']} ({task}): {len(failed)} of {gs} episode(s) failed in the harness "
                     f"({sorted(failed)[:4]}), the other {len(episodes)} count")
        if "ledger_status" not in entry:
            raise OrchestratorError(f"group {entry['name']}: complete_group without the ledger snapshot of its harness exit")
        status = entry["ledger_status"]
        enqueued = ledger_count(status, "enqueued_events")
        servers = manifest_servers(manifest)
        statuses = ledger_statuses(status)
        names = {row["instance"] for row in statuses}
        for server in servers:
            if server["instance"] not in names or server["instances"] != [server["instance"]]:
                raise OrchestratorError(f"{manifest_path}: ledger instance {server['instances']} is not one of the servers' "
                                        f"current {sorted(names)} (server restarted during the group?)")
        rewards = [e["score"] for e in episodes]
        note = None
        if arc3_mode(self.cfg["prepare"]) == "note":
            # the dispatcher completes a group with episodes to judge only once its judge.json exists (note_rewards checks
            # both ways: a group with nothing to judge must have none)
            judge_path = run_dir / JUDGE_FILE
            raw = judge_path.read_bytes() if judge_path.is_file() else None
            judge = None if raw is None else json.loads(raw)
            if judge is not None:
                check_judge_doc(self.note_judge, judge, judge_path)
            note = {"note_rewards": note_rewards(episodes, judge),
                    "judge_sha256": None if raw is None else hashlib.sha256(raw).hexdigest()}
        # live = the group has a training signal (group_live); batches count live episodes, so all-correct groups with
        # an efficiency term (efficiency_beta, arc3_reward eff / c0) count too
        live = group_live(episodes, self.cfg["prepare"], entry["version"], **({} if note is None else {"judge": judge}))
        with self.cv:
            version = self.server_version
        record = {"event": "completed", "gid": gid, "group": entry["name"], "task": task, "run_dir": str(run_dir),
                  "rewards": rewards, "live": live, **({} if note is None else note),
                  # sample / score / stop / tokens, and for arc3 also actions / tokens_at_clear (read_group_episodes)
                  "episodes": [{k: v for k, v in e.items() if k not in ("task", "group")} for e in episodes],
                  "failed_episodes": sorted(failed),
                  "enqueued_events": enqueued, "ledger_instance": servers[0]["instance"],
                  "ledger_instances": [srv["instance"] for srv in servers],
                  "replica_events": {row["instance"]: ledger_count(row, "enqueued_events") for row in statuses},
                  "server_version_launch": entry["version"], "server_version_complete": version,
                  "salt_prefix": entry["salt_prefix"], "launched_utc": entry["launched_utc"], "completed_utc": utc_now(), "completed_unix": time.time(),
                  "seconds": round(time.monotonic() - entry["started_mono"], 1),
                  **({"adopted": True} if entry.get("adopted") else {})}
        append_jsonl(self.root / "groups.jsonl", record)
        return record

    # ---- held-out in the stream loop (stream.heldout_every, SPEC §9 "held-out")
    def heldout_dir(self, version: int, task: str) -> Path:
        """Held-out group dir ``<run>/heldout/vNNNNNN/<task>`` for task ``task`` in the round of nominal version ``version``."""
        return self.root / HELDOUT_DIR / f"v{version:06d}" / task

    def write_heldout_state(self) -> None:
        atomic_write(self.root / HELDOUT_STATE_FILE,
                     json.dumps({"rounds": [self.heldout_rounds[v] for v in sorted(self.heldout_rounds)]}, indent=1) + "\n")

    def load_heldout(self) -> None:
        """heldout_state.json and heldout.jsonl → heldout_rounds / heldout_rows / heldout_closed. The two must agree: rows belong only to
        opened rounds and dispatched tasks, one row per task; a summary row comes only after every task of the round has a row, one per
        round; without heldout_state.json there may be no heldout.jsonl or heldout/. A mismatch = not written by this orchestrator,
        raise."""
        path, rows_path, root = self.root / HELDOUT_STATE_FILE, self.root / HELDOUT_FILE, self.root / HELDOUT_DIR
        rounds: dict[int, dict] = {}
        if path.exists():
            doc = json.loads(path.read_text())
            if set(doc) != {"rounds"}:
                raise OrchestratorError(f"{path}: keys {sorted(doc)}, want ['rounds']")
            for rnd in doc["rounds"]:
                if set(rnd) != HELDOUT_ROUND_KEYS:
                    raise OrchestratorError(f"{path}: a round has keys {sorted(rnd)}, want {sorted(HELDOUT_ROUND_KEYS)}")
                tasks, launched = rnd["tasks"], rnd["launched"]
                if (rnd["version"] in rounds or not tasks or len(set(tasks)) != len(tasks)
                        or len(set(launched)) != len(launched) or not set(launched) <= set(tasks)):
                    raise OrchestratorError(f"{path}: round v{rnd['version']:06d} is listed twice, or its tasks {tasks[:5]} / "
                                            f"launched {launched[:5]} are not distinct ids with launched a subset of tasks")
                rounds[rnd["version"]] = rnd
        elif rows_path.exists() or root.exists():
            raise OrchestratorError(f"{rows_path} or {root} exists without {path} (every held-out round is opened there "
                                    f"first): inspect by hand")
        rows: dict[int, dict[str, dict]] = {v: {} for v in rounds}
        closed: set[int] = set()
        for i, line in enumerate(rows_path.read_text().splitlines() if rows_path.exists() else []):
            row, where = json.loads(line), f"{rows_path}:{i + 1}"
            version = row["version"]
            if version not in rounds:
                raise OrchestratorError(f"{where}: round v{version:06d} is not in {path}")
            if row["kind"] == "group":
                if row["task"] not in rounds[version]["launched"] or row["task"] in rows[version]:
                    raise OrchestratorError(f"{where}: held-out {row['group']} was never launched, or has a row already")
                rows[version][row["task"]] = row
            elif row["kind"] == "round":
                if version in closed or set(rows[version]) != set(rounds[version]["tasks"]):
                    raise OrchestratorError(f"{where}: a second summary of round v{version:06d}, or one before every task "
                                            f"of it has a group row")
                closed.add(version)
            else:
                raise OrchestratorError(f"{where}: kind {row['kind']!r} is neither group nor round")
        self.heldout_rounds, self.heldout_rows, self.heldout_closed = rounds, rows, closed

    def heldout_start_check(self) -> list[dict]:
        """Resume (inside stream_start_check, once the server version is settled): read back the held-out rounds and rows
        (load_heldout) and handle groups the previous orchestrator left without a row, judged the same way as triage_incomplete for
        training groups, with only two outcomes (held-out is not trained or audited and needs no ledger roster; a group across a server
        restart fails on its own and is recorded as failed):
        - launch record present, harness still alive: **adopt** (return the entry; the dispatcher manages it as its own); with a
          terminated marker, first wait for it to exit (wait_told_to_stop), then record failed.
        - already gone: EXIT.json rc 0 with complete results = record completed now (server version = the reconciled version,
          adopted); no EXIT.json (hard-killed), rc≠0, incomplete results, or a terminated marker = record failed.
        - heldout_state.json records launched but there is no launch record (stopped between dispatch and the record): the harness may
          still be running with an unknown pid, so recording failed requires ``--abandon-incomplete-round`` (after making sure no such
          process exists).
        Undispatched tasks go back into the queue (not dispatched when stream.heldout_every is 0, with a log line); a round where every
        task has a row but no summary gets its summary written. A dir under heldout/ that belongs to no dispatched group: raise."""
        self.load_heldout()
        root = self.root / HELDOUT_DIR
        launched = {f"v{v:06d}": set(r["launched"]) for v, r in self.heldout_rounds.items()}
        for vdir in sorted(root.iterdir()) if root.is_dir() else ():
            stray = [vdir.name] if vdir.name not in launched or not vdir.is_dir() else \
                sorted(p.name for p in vdir.iterdir() if p.is_dir() and p.name not in launched[vdir.name])
            if stray:
                raise OrchestratorError(f"{vdir}: {stray[:5]} is not a held-out round / launched task in {HELDOUT_STATE_FILE}: "
                                        f"not created by this orchestrator; inspect by hand")
        adopted = []
        for nominal, rnd in sorted(self.heldout_rounds.items()):
            for task in rnd["launched"]:
                if task in self.heldout_rows[nominal]:
                    continue
                name, run_dir = f"v{nominal:06d}/{task}", self.heldout_dir(nominal, task)
                doc = self.read_heldout_launch(nominal, task)
                if doc is None:
                    if not self.abandon_incomplete:
                        raise OrchestratorError(
                            f"stream resume: held-out {name} is launched in {HELDOUT_STATE_FILE} but has no {LAUNCH_SUFFIX} "
                            f"record (the orchestrator stopped while launching it): its harness may still run (check ps for "
                            f"--run-dir {run_dir}); then pass --abandon-incomplete-round to record it failed")
                    entry = {"name": name, "task": task, "round": nominal, "samples": rnd["samples"], "run_dir": run_dir,
                             "stdout": run_dir.with_name(f"{task}.out"), "started_mono": None, "launched_utc": None,
                             "version": None, "salt_prefix": None}
                    self.finish_heldout_group(entry, None, f"no {LAUNCH_SUFFIX} record (the orchestrator stopped while "
                                                           f"launching it): recorded failed by --abandon-incomplete-round")
                    continue
                entry = self.heldout_adopt_entry(doc)
                proc, pid = entry["proc"], doc["pid"]
                marker = self.read_terminated_at(run_dir)
                if proc.poll() is None and marker is not None:
                    self.wait_told_to_stop(f"held-out {name}", task, proc, pid, marker)
                if proc.poll() is None:
                    adopted.append(entry)
                    self.log(f"held-out {name}: harness pid {pid} launched {doc['launched_utc']} by session "
                             f"{doc['session_id']} still runs: adopted")
                    continue
                rc, exited = proc.poll(), exit_record(run_dir) is not None
                how = (f"rc {rc}{'' if exited else f' (no {EXIT_FILE})'}, "
                       f"{group_episodes_done(run_dir, task, rnd['samples'])}/{rnd['samples']} results")
                error = (f"terminated ({marker['reason']}) by the previous orchestrator (session {marker['session_id']}), {how}"
                         if marker is not None else None if exited else
                         f"harness pid {pid} gone without {EXIT_FILE} after its orchestrator stopped (killed), {how}")
                self.log(f"held-out {name}: harness pid {pid} ended after its orchestrator stopped ({how}): recording it now")
                self.finish_heldout_group(entry, rc, error)
        pending = [(v, t) for v, r in sorted(self.heldout_rounds.items()) for t in r["tasks"] if t not in r["launched"]]
        if pending and not self.heldout_every:
            self.log(f"held-out is off (stream.heldout_every 0): {len(pending)} unlaunched task(s) of round(s) "
                     f"{sorted({f'v{v:06d}' for v, _ in pending})} stay unlaunched")
        self.heldout_queue = pending if self.heldout_every else []
        big = sorted({v for v, _ in self.heldout_queue if self.heldout_rounds[v]["samples"] > self.stream["rollout_slots"]})
        if big:          # held-out goes first: a group that never fits would block every training group behind it
            raise OrchestratorError(f"held-out round v{big[0]:06d} still has groups to launch with "
                                    f"{self.heldout_rounds[big[0]]['samples']} samples each > stream.rollout_slots "
                                    f"{self.stream['rollout_slots']}: they would never launch; give the slots back")
        for nominal in sorted(self.heldout_rounds):
            self.close_heldout_round(nominal)
        if self.heldout_rounds:
            self.log(f"stream start: held-out rounds {[f'v{v:06d}' for v in sorted(self.heldout_rounds)]}, "
                     f"{len(self.heldout_closed)} summarized; {len(adopted)} held-out group(s) adopted, "
                     f"{len(self.heldout_queue)} queued to launch")
        return adopted

    def read_heldout_launch(self, nominal: int, task: str) -> dict | None:
        """``heldout/vNNNNNN/<task>.launch.json`` (written by launch_heldout_group); missing = None. Wrong keys or content raise."""
        run_dir = self.heldout_dir(nominal, task)
        path = run_dir.with_name(task + LAUNCH_SUFFIX)
        if not path.is_file():
            return None
        doc = json.loads(path.read_text())
        want = (nominal, task, self.heldout_rounds[nominal]["samples"])
        if set(doc) != HELDOUT_LAUNCH_KEYS or (doc["version"], doc["task"], doc["samples"]) != want:
            raise OrchestratorError(f"{path}: keys {sorted(doc)} / (version, task, samples) "
                                    f"{tuple(doc.get(k) for k in ('version', 'task', 'samples'))}, want "
                                    f"{sorted(HELDOUT_LAUNCH_KEYS)} / {want}")
        return doc

    def heldout_adopt_entry(self, doc: dict) -> dict:
        """Held-out launch.json → the dispatcher's hprocs entry (like adopt_entry: AdoptedProc, clock from the original launch)."""
        run_dir = self.heldout_dir(doc["version"], doc["task"])
        age = max(0.0, time.time() - doc["launched_unix"])
        return {"name": doc["group"], "task": doc["task"], "round": doc["version"], "samples": doc["samples"],
                "run_dir": run_dir, "stdout": run_dir.with_name(f"{doc['task']}.out"),
                "proc": AdoptedProc(doc["pid"], doc["proc_starttime"], run_dir), "files": [],
                "started_mono": time.monotonic() - age, "launched_utc": doc["launched_utc"],
                "version": doc["server_version_launch"], "salt_prefix": doc["salt_prefix"], "adopted": True,
                "assigned": doc["backends"]}

    def heldout_round_due(self, version: int) -> int | None:
        """Round due at server version ``version``: nominal version = the largest multiple of heldout_every ≤ version (0 only with
        heldout_at_v0). Opens only if newer than every opened round: if the version skipped a multiple (e.g. a restart at 13), the latest
        multiple (10) is opened late and runs at the current version; earlier missed ones are not. Nominal version ≥ rounds (final
        weights: the learner is done) does not open: no in-loop evaluation, eval_keep_every weights are evaluated offline; if rounds is
        raised and a resume reaches it, it opens as usual."""
        if not self.heldout_every:
            return None
        nominal = version // self.heldout_every * self.heldout_every
        if nominal == 0 and not self.stream.get("heldout_at_v0", True):
            return None
        if nominal >= self.cfg["rounds"]:           # the final policy: nothing trains at it any more, evaluate it offline
            return None
        if self.heldout_rounds and nominal <= max(self.heldout_rounds):
            return None
        return nominal

    def open_due_heldout_round(self) -> None:
        """Each dispatcher loop (before STOP / learner done): when all groups of the previous round are dispatched and another round is
        due (heldout_round_due), open it: heldout_state.json first records the nominal version, the server version at opening, the task
        list (tasks.heldout order) and the sample count, and the tasks are queued."""
        if self.heldout_queue:
            return
        with self.cv:
            version = self.server_version
        nominal = self.heldout_round_due(version)
        if nominal is None:
            return
        tasks, samples = list(self.pools["heldout"]), self.stream["heldout_samples"]
        self.heldout_rounds[nominal] = {"version": nominal, "launch_version": version, "opened_utc": utc_now(),
                                        "opened_unix": time.time(), "session_id": self.session_id, "samples": samples,
                                        "tasks": tasks, "launched": []}
        self.heldout_rows[nominal] = {}
        self.write_heldout_state()
        self.heldout_queue.extend((nominal, t) for t in tasks)
        self.log(f"held-out round v{nominal:06d}: opened at server version {version}"
                 + (f" (no round was opened at version {nominal})" if version != nominal else "")
                 + f": {len(tasks)} task(s) x {samples} sample(s) = {len(tasks) * samples} episode(s), launched before any "
                   f"new training group")

    def launch_heldout_group(self, nominal: int, task: str, procs: dict[int, dict], heldout: dict[str, dict]) -> dict:
        """One held-out group: the same harness argv as training groups (same harness config: same episode budget), ``--samples`` is
        that round's heldout_samples, tasks from tasks.heldout.dir, dir ``heldout/vNNNNNN/<task>``, each sample dispatched by the
        replicas' running episodes (training + held-out). Before dispatch, launched is recorded in heldout_state.json (a restart will
        not dispatch it again; one without a launch record needs a human decision); salt ``<run>/<session>/hNNNNNN`` (the harness
        appends ``/<episode>``): different for every round and session, so it cannot hit the prefix cache of other rounds or old
        weights. Then ``<task>.launch.json`` is written (for adoption, like a training group's launch.json)."""
        rnd = self.heldout_rounds[nominal]
        samples, name, run_dir = rnd["samples"], f"v{nominal:06d}/{task}", self.heldout_dir(nominal, task)
        stdout, record = run_dir.with_name(f"{task}.out"), run_dir.with_name(task + LAUNCH_SUFFIX)
        if task in rnd["launched"] or run_dir.exists() or stdout.exists() or record.exists():
            raise OrchestratorError(f"held-out {name} was launched before ({run_dir}, its .out or launch record exists): a "
                                    f"held-out group is never launched twice")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        rnd["launched"].append(task)
        self.write_heldout_state()
        salt_prefix = f"{self.root.name}/{self.session_id}/h{nominal:06d}"
        replicas = self.cfg["rollout"]["replica_urls"]
        running = self.replica_running(procs, heldout)
        assigned = assign_samples(running, samples)
        argv = self.harness_argv(run_dir, cfg_path(self.cfg["tasks"]["heldout"]["dir"]), [task],
                                 ["--samples", str(samples), "--salt-prefix", salt_prefix],
                                 backends=[replicas[r] for r in assigned])
        proc, files = self.popen({"argv": argv, "stdout": stdout})
        started, launched_unix, launched_utc = time.monotonic(), time.time(), utc_now()
        with self.cv:
            version = self.server_version
            self.stream_stats["heldout_launched"] += 1
        starttime = proc_starttime(proc.pid)
        if starttime is None:                           # died before we looked: the dispatcher's poll will see its rc
            self.log(f"held-out {name}: harness pid {proc.pid} is already gone right after launch")
        atomic_write(record, json.dumps(
            {"version": nominal, "task": task, "group": name, "samples": samples, "salt_prefix": salt_prefix,
             "session_id": self.session_id, "pid": proc.pid, "proc_starttime": starttime, "launched_unix": launched_unix,
             "launched_utc": launched_utc, "server_version_launch": version, "backends": assigned}, indent=1) + "\n")
        spread = " ".join(f"r{r}x{assigned.count(r)}" for r in sorted(set(assigned)))
        after = [n + assigned.count(r) for r, n in enumerate(running)]
        self.log(f"held-out {name} launched at server version {version} on {spread} (running {'/'.join(map(str, running))} "
                 f"-> {'/'.join(map(str, after))}); round v{nominal:06d}: {len(rnd['launched'])}/{len(rnd['tasks'])} launched")
        return {"name": name, "task": task, "round": nominal, "samples": samples, "run_dir": run_dir, "stdout": stdout,
                "proc": proc, "files": files, "started_mono": started, "launched_utc": launched_utc, "version": version,
                "salt_prefix": salt_prefix, "assigned": assigned}

    def finish_heldout_group(self, entry: dict, rc: int | None, error: str | None = None) -> dict:
        """A held-out group finished (dispatcher; heldout_start_check on resume): one kind-group row in heldout.jsonl, and once every
        task of the round has a row, the summary (close_heldout_round). Results go only here: not into groups.jsonl, pool or batch, and
        not counted in the live share, completion seconds or retirement.
        rc 0 with complete results (the same read_group_episodes as training groups: none missing except games the manifest marks as
        failed) = completed, recording score / stop / output tokens per game (arc3 also actions and tokens_at_clear; with
        prepare.arc3_reward official also that game's R); rc≠0, timeout (``terminating_since``), ``error`` (cause of death determined
        on resume) or mismatched results = failed: record the error and the number of results written, no quarantine, no stop (data
        for monitoring should not stop training; the error is still logged loudly in the row and the log)."""
        nominal, task, samples, run_dir = entry["round"], entry["task"], entry["samples"], entry["run_dir"]
        rnd = self.heldout_rounds[nominal]
        if rc is None and error is None:
            raise OrchestratorError(f"held-out {entry['name']}: internal: no rc and no error to record")
        if error is None and "terminating_since" in entry:
            error = (f"group timeout: not finished after {self.stream['group_timeout_seconds']:.0f}s "
                     f"(stream.group_timeout_seconds), {'SIGKILLed' if 'killed_at' in entry else 'terminated'}")
        if error is None and rc != 0:
            error = harness_failure(run_dir, entry["stdout"], rc)
        episodes: list[dict] = []
        failed: set[str] = set()
        if error is None:
            manifest_path = run_dir / "tito_manifest.json"
            try:
                if not manifest_path.is_file():
                    raise OrchestratorError(f"{run_dir}: harness exited 0 without tito_manifest.json")
                failed = manifest_failed_ids(json.loads(manifest_path.read_text()), run_dir)
                episodes = read_group_episodes(run_dir, task, samples, failed)
            except OrchestratorError as e:
                # monitoring data: results that do not add up fail this held-out group only -- recorded in its row and logged
                # loudly below; the training run goes on
                error, episodes = f"results: {e}", []
        if error is not None:
            failed = set(group_episode_ids(task, samples))           # nothing of a failed group counts
        rewards = (official_rewards(episodes, self.cfg["prepare"])
                   if episodes and arc3_mode(self.cfg["prepare"]) == "official" else None)
        eps = [{**{k: v for k, v in e.items() if k not in ("task", "group")},
                **({} if rewards is None else {"official_reward": rewards[i]})} for i, e in enumerate(episodes)]
        with self.cv:
            version = self.server_version
        row = {"kind": "group", "version": nominal, "launch_version": rnd["launch_version"], "group": entry["name"],
               "task": task, "status": "completed" if error is None else "failed", "error": error, "rc": rc,
               "samples": samples, "episodes": eps, "failed_episodes": sorted(failed),
               "results_written": group_episodes_done(run_dir, task, samples),
               "server_version_launch": entry["version"], "server_version_complete": version,
               "salt_prefix": entry["salt_prefix"], "run_dir": str(run_dir), "launched_utc": entry["launched_utc"],
               "completed_utc": utc_now(), "completed_unix": time.time(),
               "seconds": None if entry["started_mono"] is None else round(time.monotonic() - entry["started_mono"], 1),
               **({"adopted": True} if entry.get("adopted") else {})}
        append_jsonl(self.root / HELDOUT_FILE, row)
        self.heldout_rows[nominal][task] = row
        progress = f"round v{nominal:06d}: {len(self.heldout_rows[nominal])}/{len(rnd['tasks'])} group(s) done"
        if error is None:
            passed, hit = sum(e["score"] > 0 for e in eps), sum(e["stop"] == "budget" for e in eps)
            self.log(f"held-out {entry['name']} done in {row['seconds']:.0f}s (server versions {row['server_version_launch']}"
                     f"->{version}): pass {passed}/{len(eps)}, budget-hit {hit}/{len(eps)}, output tokens mean "
                     f"{statistics.fmean(e['tokens'] for e in eps):.0f}"
                     + (f", {len(failed)} episode(s) failed in the harness" if failed else "") + f"; {progress}")
        else:
            self.log(f"HELD-OUT GROUP FAILED {entry['name']}: {error}; recorded in {HELDOUT_FILE} (not quarantined, the run "
                     f"goes on); {progress}")
        self.close_heldout_round(nominal)
        return row

    def close_heldout_round(self, nominal: int) -> None:
        """Every task of the round has a row but there is no summary yet: write the kind-round summary row (heldout_round_summary) and
        log one HELD-OUT line."""
        rnd, rows = self.heldout_rounds[nominal], self.heldout_rows[nominal]
        if nominal in self.heldout_closed or set(rows) != set(rnd["tasks"]):
            return
        summary = heldout_round_summary(rnd, [rows[t] for t in rnd["tasks"]])
        append_jsonl(self.root / HELDOUT_FILE, summary)
        self.heldout_closed.add(nominal)
        show = lambda x, spec: "n/a" if x is None else format(x, spec)  # noqa: E731
        lo, hi = summary["server_versions"]
        self.log(f"HELD-OUT v{nominal:06d}: pass rate {show(summary['pass_rate'], '.3f')} over {summary['episodes']} "
                 f"episode(s), score mean {show(summary['score_mean'], '.3f')}, budget-hit "
                 f"{show(summary['budget_hit_rate'], '.3f')}, output tokens mean {show(summary['output_tokens_mean'], '.0f')} "
                 f"(p50 {show(summary['output_tokens_p50'], '.0f')}); {summary['groups_failed']} group(s) / "
                 f"{summary['failed_episodes']} episode(s) failed; server versions {lo}-{hi} (opened at "
                 f"{rnd['launch_version']}), {summary['seconds']:.0f}s")

    # ---- learner thread
    def stream_learner_loop(self, start: int, total: int, resume: dict | None) -> str:
        """``resume`` (the batch to continue from stream_start_check): the first iteration does not take from the pool but continues
        with the groups and state recorded for it."""
        if self.stream.get("group_ready", False):
            return self.stream_learner_group_ready(start, total, resume)
        if self.stream.get("prefetch_prepare", False):
            return self.stream_learner_loop_prefetch(start, total, resume)
        b, consecutive_empty = start, 0
        while b < total:
            ready = time.monotonic()
            if resume is not None and b == start:
                pool = {**resume["pool_after"], "formed_mono": ready, "formed_unix": resume["formed_unix"]}
                self.learn_batch(b, resume["groups"], pool, ready, resumed=resume)
                resume = None
                b += 1
                continue
            taken = self.take_batch(b)
            if taken is None:
                return "stopped"
            if self.learn_batch(b, *taken, ready) is False:
                consecutive_empty += 1
                if consecutive_empty >= self.stream.get("max_empty_batches", 100):
                    raise OrchestratorError(f"batch {b}: {consecutive_empty} consecutive EMPTY_ELIGIBLE preparations; "
                                            "no optimizer step or publication; inspect discarded batch reports")
                continue
            consecutive_empty = 0
            b += 1
        with self.cv:
            self.learner_finished = True
            self.cv.notify_all()
        return "done"

    def stream_learner_loop_prefetch(self, start: int, total: int, resume: dict | None) -> str:
        """stream.prefetch_prepare: while batch b is in train / publish / sync, another thread takes batch b+1's groups and runs
        ledger → audit → prepare (version window V = b+1: it is certain to train at v(b+1)), after which the learner trains it directly.
        The pool is taken in completion order, so each batch's groups are the same as without prefetch; only prepare happens earlier. The
        prefetched batch dir's state.json carries ``prefetched``: the trainer has never seen it, and a resume moves it aside
        automatically (pending_batch). If batch b fails: tell the prefetch to stop (aborted), wait for it to take down its own children,
        then raise."""
        if resume is not None:
            ready = time.monotonic()
            pool = {**resume["pool_after"], "formed_mono": ready, "formed_unix": resume["formed_unix"]}
            current = self.form_batch(start, resume["groups"], pool, ready, resumed=resume)
            if current is None:                    # the resumed batch was EMPTY_ELIGIBLE: refill at the same version
                current = self.next_formed_batch(start, prefetch=False)
        else:
            current = self.next_formed_batch(start, prefetch=False)
        while current is not None:
            b = current["batch"]
            box: dict = {}
            ahead = None
            if b + 1 < total:
                ahead = threading.Thread(target=self._prefetch_into, args=(b + 1, box), name=f"prefetch-b{b + 1:06d}")
                ahead.start()
            try:
                self.finish_batch(current)
            except BaseException:
                if ahead is not None:
                    with self.cv:
                        self.aborted.set()
                        self.cv.notify_all()
                    ahead.join()
                raise
            if ahead is None:                      # b was the last batch
                with self.cv:
                    self.learner_finished = True
                    self.cv.notify_all()
                return "done"
            began = time.monotonic()
            ahead.join()
            if "error" in box:
                raise box["error"]
            current = box["formed"]
            if current is not None:
                current["prefetch_wait"] = round(time.monotonic() - began, 2)
                self.log(f"batch {b + 1}: prefetched; the learner waited {current['prefetch_wait']:.0f}s for it")
        return "stopped"

    def _prefetch_into(self, b: int, box: dict) -> None:
        try:
            box["formed"] = self.next_formed_batch(b, prefetch=True)
        except BaseException as error:            # re-raised by the learner thread once batch b-1 is done
            box["error"] = error

    def next_formed_batch(self, b: int, prefetch: bool) -> dict | None:
        """Take batch b's groups and run through prepare; on EMPTY_ELIGIBLE, take again at the same version. Dispatcher stopped and no
        batch can be assembled: None."""
        consecutive_empty = 0
        while True:
            ready = time.monotonic()
            taken = self.take_batch(b)
            if taken is None:
                return None
            formed = self.form_batch(b, *taken, ready, prefetch=prefetch)
            if formed is not None:
                return formed
            consecutive_empty += 1
            if consecutive_empty >= self.stream.get("max_empty_batches", 100):
                raise OrchestratorError(f"batch {b}: {consecutive_empty} consecutive EMPTY_ELIGIBLE preparations; "
                                        "no optimizer step or publication; inspect discarded batch reports")

    def take_batch(self, b: int) -> tuple[list[dict], dict] | None:
        """Wait until the pool's live episodes ≥ batch_live_episodes and take groups from the front in completion order until enough
        (dead groups in between are consumed too). Dispatcher already ended and no batch can be assembled: None. Returns (groups, pool
        state at the time of taking)."""
        gs, need = self.stream["group_size"], self.stream["batch_live_episodes"]
        with self.cv:
            while sum(gs for r in self.pool if r["live"]) < need:
                if self.dispatch_ended:
                    live = sum(gs for r in self.pool if r["live"])
                    self.log(f"batch {b}: the dispatcher stopped and the pool holds {live} live episode(s) < {need}: no more "
                             f"batches; {len(self.pool)} group(s) stay unconsumed in groups.jsonl")
                    return None
                self.cv.wait(self.poll)
                self.check_cancel()
            taken, live = [], 0
            while live < need:
                record = self.pool.pop(0)
                taken.append(record)
                live += gs if record["live"] else 0
            self.batches_taken += 1
            pool = {"groups": len(self.pool), "live_groups": sum(r["live"] for r in self.pool),
                    "groups_in_flight": len(self.inflight_groups), "formed_unix": time.time(), "formed_mono": time.monotonic()}
            self.cv.notify_all()       # back-pressure may lift
        return taken, pool

    def open_batch(self, b: int, rd: Path, groups: list[dict], formed_unix: float,
                   prefetched: bool = False) -> tuple[dict, Path]:
        if rd.exists():
            raise OrchestratorError(f"{rd} already exists: a batch that batches.jsonl does not have was started before")
        (rd / "logs").mkdir(parents=True)
        state = {"batch": b, "groups": [r["gid"] for r in groups], "started_utc": utc_now(), "formed_unix": formed_unix,
                 "prefetched": prefetched, "stages": {}, "timings": {}, "train": {"metrics": [], "inflight": None}}
        state_path = rd / "state.json"
        atomic_write(state_path, json.dumps(state, indent=1) + "\n")
        return state, state_path

    def discard_empty_batch(self, rd: Path, state: dict) -> None:
        """Idempotent recovery: record rejected groups durably, then archive the untouched batch."""
        if (state["train"]["metrics"] or state["train"]["inflight"] is not None
                or "publish" in state["stages"] or "sync" in state["stages"]):
            raise OrchestratorError(f"{rd}: EMPTY_ELIGIBLE batch already mutated trainer/server")
        completed, abandoned = self.load_groups()
        records = {r["gid"]: r for r in completed}
        # no group left: group_ready quarantined every one of them (gr_quarantine); their abandoned rows are written
        span = (f"g{min(state['groups']):06d}-g{max(state['groups']):06d}" if state["groups"]
                else f"quarantined-{time.strftime('%Y%m%dT%H%M%S')}-{os.urandom(3).hex()}")
        target = rd.with_name(f"{rd.name}.empty-{span}")
        for gid in state["groups"]:
            if gid not in records:
                raise OrchestratorError(f"{rd}: rejected group {gid} never completed")
            if gid not in abandoned:
                append_jsonl(self.root / "groups.jsonl", {"event": "abandoned", "gid": gid,
                             "group": records[gid]["group"], "utc": utc_now(), "reason": "EMPTY_ELIGIBLE",
                             "batch": state["batch"], "report": str(target / "batch" / "report.json")})
        rd.rename(target)
        self.log(f"batch {state['batch']}: EMPTY_ELIGIBLE; {len(state['groups'])} group(s) discarded -> {target}; "
                 "refill at the same policy version")

    def learn_batch(self, b: int, groups: list[dict], pool: dict, ready: float, resumed: dict | None = None) -> bool:
        """Batch b (trainer at V=b): ledger flushed (written ≥ the largest snapshot of this batch's groups) → audit (one manifest per
        group) → prepare (--current-version V --max-staleness S) → /train → /publish V+1 → in_place sync → one row in batches.jsonl.
        ``resumed``: continue this batch, which last time was not written to batches.jsonl; read back its state.json and skip the
        stages already done.
        False: prepare was EMPTY_ELIGIBLE (discarded, batches_taken rolled back)."""
        formed = self.form_batch(b, groups, pool, ready, resumed)
        if formed is None:
            return False
        self.finish_batch(formed)
        return True

    def form_batch(self, b: int, groups: list[dict], pool: dict, ready: float, resumed: dict | None = None,
                   prefetch: bool = False) -> dict | None:
        """First half of learn_batch: open the batch dir (or read back the resumed state) → ledger → audit → prepare. None:
        EMPTY_ELIGIBLE. ``prefetch``: done while batch b−1 is still training (stream.prefetch_prepare)."""
        sc = self.stream
        version = b                                    # V = batches trained so far (the start check and every row keep it)
        rd = self.root / "batches" / f"b{b:06d}"
        live = [r for r in groups if r["live"]]
        if resumed is None:
            self.log(f"batch {b}: {len(groups)} group(s) ({len(live)} live, {len(live) * sc['group_size']} live episodes) "
                     f"after {pool['formed_mono'] - ready:.0f}s; {pool['groups']} group(s) left in the pool"
                     + (f" (prefetched while batch {b - 1} trains)" if prefetch else ""))
            state, state_path = self.open_batch(b, rd, groups, pool["formed_unix"], prefetched=prefetch)
        else:
            if resumed["dir"] != rd or resumed["state"]["groups"] != [r["gid"] for r in groups]:
                raise OrchestratorError(f"batch {b}: the resume plan is for {resumed['dir']} with groups {resumed['state']['groups']}")
            state, state_path = resumed["state"], rd / "state.json"
            self.log(f"batch {b}: resuming {rd} ({len(groups)} group(s), {len(live)} live); stages done {resumed['stages_done']}")
        stage = self.make_stage(b, state, state_path, label="batch")
        short = self.disk_short(sc["hard_min_free_gb"])
        if short:
            raise OrchestratorError(f"{'; '.join(short)} < stream.hard_min_free_gb {sc['hard_min_free_gb']}: the learner stops "
                                    f"(prepare and publish write there; below min_free_gb {self.cfg['min_free_gb']} the "
                                    f"dispatcher only waits)")
        target = max(r["enqueued_events"] for r in groups)
        watermark = {"enqueued_events": target}
        if self.cfg["rollout"].get("replica_urls"):
            replica_events = {}
            for row in groups:
                for name, value in row["replica_events"].items():
                    replica_events[name] = max(value, replica_events.get(name, 0))
            watermark["replica_events"] = replica_events
        landing = stage("ledger", lambda: self.ledger_landing(watermark))
        audit = stage("audit", lambda: self.audit_with_status(rd, [Path(r["run_dir"]) for r in groups], landing["status"]))
        prepared = stage("prepare", lambda: self.prepare_batch(b, rd, audit, None, window=(version, sc["max_staleness"])))
        if prepared.get("status") == "EMPTY_ELIGIBLE":
            self.discard_empty_batch(rd, state)
            with self.cv:
                self.batches_taken -= 1
                self.cv.notify_all()
            return None
        return {"batch": b, "rd": rd, "groups": groups, "live": live, "pool": pool, "ready": ready, "resumed": resumed,
                "state": state, "state_path": state_path, "stage": stage, "audit": audit, "prepared": prepared,
                "prefetched": prefetch}

    def finish_batch(self, formed: dict) -> None:
        """Second half of learn_batch: /train → /publish V+1 → in_place sync → save → one row in batches.jsonl."""
        sc = self.stream
        b, rd, groups, live, pool, ready, resumed = (formed[k] for k in ("batch", "rd", "groups", "live", "pool", "ready",
                                                                         "resumed"))
        state, state_path, stage, audit, prepared = (formed[k] for k in ("state", "state_path", "stage", "audit", "prepared"))
        version = b
        self.train_batch(b, rd, state, state_path, prepared, expect_version=version, label="batch")
        short = self.disk_short(sc["hard_min_free_gb"])
        if short:
            raise OrchestratorError(f"{'; '.join(short)} below hard floor before publish; trainer state is retained")
        published = stage("publish", lambda: self.publish(version))
        synced = stage("sync", lambda: self.sync_stream(version, published))
        synced_unix = time.time()
        saved = self.maybe_save(b, stage)
        step_after = self.trainer_health()["step"]
        episodes = [{"task": r["task"], "group": r["group"], **e} for r in groups for e in r["episodes"]]
        summary = summarize(episodes, sc["group_size"], by="group")
        row = {"batch": b, "pipeline": "stream", "started_utc": state["started_utc"], "finished_utc": utc_now(),
               "weight_version_in": str(version), "weight_version_out": str(version + 1), "served_name": self.served_name,
               "groups": [{"gid": r["gid"], "group": r["group"], "task": r["task"], "live": r["live"],
                           "reward_mean": round(statistics.fmean(r["rewards"]), 6),
                           "versions": [r["server_version_launch"], r["server_version_complete"]],
                           "pool_seconds": round(pool["formed_unix"] - r["completed_unix"], 1)} for r in groups],
               "live_groups": len(live), "live_episodes": len(live) * sc["group_size"],
               "total_episodes": len(groups) * sc["group_size"],
               "pool_after": {k: pool[k] for k in ("groups", "live_groups", "groups_in_flight")},
               "train": {k: v for k, v in summary.items() if k != "per_task_mean"},
               "audit": {k: audit[k] for k in ("instance", "reports")},
               "prepare": prepared["summary"], "minibatches": prepared["minibatches"],
               "train_metrics": state["train"]["metrics"], "trainer_step_after": step_after,
               "publish": published, "sync": synced, "save": saved,
               "groups_in_flight_at_sync": synced["groups_in_flight"],
               "prefetched": formed["prefetched"],
               "seconds": {"pool_wait": round(pool["formed_mono"] - ready, 2), **state["timings"],
                           "pause": synced["pause_seconds"],
                           **({"prefetch_wait": formed["prefetch_wait"]} if "prefetch_wait" in formed else {}),
                           "total": round(time.monotonic() - ready if resumed is None else sum(state["timings"].values()), 1)},
               "resumed": None if resumed is None else {"stages_done": resumed["stages_done"],
                                                        "minibatches_done": resumed["minibatches_done"],
                                                        "session_seconds": round(time.monotonic() - ready, 1)},
               "dispatcher": self.take_stream_stats()}
        if "group_ready" in formed:
            row["group_ready"] = formed["group_ready"]
            if resumed is None:                     # what group_ready shortens: the batch's last group → its weights served
                row["seconds"]["last_group_to_sync"] = round(synced_unix - formed["group_ready"]["last_group_completed_unix"], 1)
        self.append_row(row)
        self.log(f"batch {b} done: v{version + 1} live {len(live)}/{len(groups)} reward {summary['reward_mean']:.4f}; "
                 f"{len(synced['groups_in_flight'])} group(s) in flight at the sync (pause {synced['pause_seconds']}s)")

    def sync_stream(self, version: int, published: dict) -> dict:
        """Always in_place: pause → update(flush_cache:false, keep_identity) → continue whatever the outcome → verify (sync_midflight).
        No flush: every episode has its own prefix-cache salt. Records the groups running when the sync started."""
        with self.cv:
            in_flight = sorted(self.inflight_groups)
        result = self.sync_midflight(version, published)
        with self.cv:
            self.server_version = version + 1
            self.cv.notify_all()
        return {**result, "mode": "in_place", "groups_in_flight": [f"g{g:06d}" for g in in_flight]}


    # =========================================================================== stream.group_ready

    def pooled_live_groups(self) -> int:
        """(cv) live group waiting for a batch: the pool's, plus those group_ready reserved for the unsealed frontier batch
        (they left the pool, that batch is not in batches_taken yet). Back-pressure and the end-of-run supply use it."""
        live = sum(r["live"] for r in self.pool)
        gb = self.gr_batches.get(self.gr_frontier)
        if gb is not None and not gb["sealed"]:
            live += sum(r["live"] for r in gb["members"])
        return live

    def gr_write_state(self, gb: dict) -> None:
        """(cv) state.json of a group_ready batch: the materializer (reservation) and the learner both write it."""
        atomic_write(gb["state_path"], json.dumps(gb["state"], indent=1) + "\n")

    def gr_open(self, b: int) -> dict:
        """(cv) group_ready batch b: its directory and state.json. ``prefetched`` stays true: until the trainer stepped from
        it, resume sets it aside (pending_batch) and its groups go back to the pool; a gradient the trainer accumulated for
        it is aborted at startup (stream_start_check)."""
        rd = self.root / "batches" / f"b{b:06d}"
        state, state_path = self.open_batch(b, rd, [], time.time(), prefetched=True)
        state.update(group_ready=True, formed_unix=None, update=None)
        gb = {"batch": b, "rd": rd, "state": state, "state_path": state_path, "members": [], "reserved_unix": [],
              "live": 0, "sealed": False, "sealed_unix": None, "sealed_mono": None,
              "materialized": 0,          # members already in a packet (formed or done)
              "formed": 0,                # packets formed; == len(packets) when none is being materialized
              "packets": [], "taken": 0, "units": 0,
              "rejected": 0}              # prepare attempts set aside (gr_materialize): <packet dir>.rejected-<n>
        self.gr_batches[b] = gb
        self.gr_write_state(gb)
        return gb

    def gr_reserve(self) -> None:
        """(cv) The pool's head goes to the frontier batch in completion order; at batch_live_episodes live episodes the
        batch is sealed (the same prefix take_batch takes). The frontier runs at most one batch ahead of the learner and
        never past the last batch (``rounds``): what is left stays in the pool, unconsumed in groups.jsonl."""
        gs, need = self.stream["group_size"], self.stream["batch_live_episodes"]
        moved = False
        while self.pool and self.gr_frontier <= self.gr_learner + 1 and self.gr_frontier < self.cfg["rounds"]:
            b = self.gr_frontier
            gb = self.gr_batches[b] if b in self.gr_batches else self.gr_open(b)
            record = self.pool.pop(0)
            gb["members"].append(record)
            gb["reserved_unix"].append(time.time())
            gb["live"] += gs if record["live"] else 0
            gb["state"]["groups"].append(record["gid"])
            if gb["live"] >= need:
                gb["sealed"], gb["sealed_unix"], gb["sealed_mono"] = True, time.time(), time.monotonic()
                gb["state"]["formed_unix"] = gb["sealed_unix"]
                self.batches_taken += 1
                self.gr_frontier += 1
                self.log(f"batch {b}: sealed with {len(gb['members'])} group(s) ({gb['live']} live episodes); "
                         f"{len(self.pool)} group(s) left in the pool")
            self.gr_write_state(gb)
            moved = True
        if moved:
            self.cv.notify_all()

    def gr_materializer_loop(self) -> str:
        """group_ready materializer thread: reserve, then pack the batch's not-yet-materialized groups into one packet ->
        ledger landing -> audit -> prepare (window (b, S); the 0.5% gates are the batch's, gr_learn). While a packet waits
        for the learner the next one is not formed (it grows instead: one packet holds every group that landed meanwhile);
        a sealed batch's last groups go at once. The next batch starts only once this one has a kept group (units > 0):
        an all-empty batch is refilled at the same version (gr_discard_empty), so nothing may be prepared for the next."""
        m = self.gr_learner
        while True:
            with self.cv:
                while True:
                    self.check_cancel()
                    if self.gr_closed:
                        return "done"
                    self.gr_reserve()
                    m = max(m, self.gr_learner)
                    gb = self.gr_batches.get(m)
                    if gb is not None:
                        pending = len(gb["members"]) - gb["materialized"]
                        if pending and (gb["sealed"] or gb["taken"] == len(gb["packets"])):
                            group = gb["members"][gb["materialized"]:]
                            gb["materialized"] = len(gb["members"])
                            index = gb["formed"]
                            gb["formed"] += 1
                            break
                        if gb["sealed"] and not pending and gb["formed"] == len(gb["packets"]) and gb["units"] > 0:
                            m += 1
                            continue
                    self.cv.wait(self.poll)
            packets = self.gr_materialize(gb, index, group)
            with self.cv:
                if self.gr_batches.get(m) is not gb or len(gb["packets"]) != index:
                    raise OrchestratorError(f"batch {m}: the batch changed while packet p{index:03d} was materialized")
                gb["packets"].extend(packets)
                gb["formed"] += len(packets) - 1     # one index was reserved: a split made more, a quarantine maybe none
                gb["units"] += sum(p["units"] for p in packets)
                self.cv.notify_all()

    def gr_materialize(self, gb: dict, index: int, group: list[dict]) -> list[dict]:
        """The reserved ``group`` (completion order) -> packets p<index>, p<index+1>, ... (gr_packet). prepare_rl's
        machine-readable rejections (PrepareRejected) do not stop the run: ``group_data`` -> the groups it names are
        quarantined (gr_quarantine) and the rest is prepared again; ``routing_over_max_gb`` -> the packet is split into two
        halves (completion order kept), recursively -- one group alone over prepare.max_gb stops the run. A rejected
        attempt's directory is renamed ``<dir>.rejected-<n>`` (the evidence). Returns the packets: none when every group
        was quarantined."""
        parts, packets, splits = [list(group)], [], []
        while parts:
            part = parts.pop(0)
            try:
                packet = self.gr_packet(gb, index + len(packets), part)
            except PrepareRejected as rejected:
                parts[:0] = self.gr_rejected(gb, part, rejected)
                continue
            except AuditFailed as failed:
                halves = self.gr_audit_failed(gb, part, failed)
                if halves:
                    splits.append({r["gid"] for r in part})
                parts[:0] = halves
                continue
            packets.append(packet)
        with self.cv:
            members = {r["gid"] for r in gb["members"]}
        innocent = [sorted(split) for split in splits if split <= members]     # nobody in the split got quarantined
        if innocent:
            raise OrchestratorError(f"batch {gb['batch']}: audit failed on group(s) {innocent} together but every half passed "
                                    f"alone: a cross-group interaction (e.g. two manifests claiming one rid), not one bad group; "
                                    f"refusing to train them. See the .rejected-* packet dirs")
        return packets

    def gr_audit_failed(self, gb: dict, part: list[dict], failed: AuditFailed) -> list[list[dict]]:
        """gr_materialize (2026-10-24 review C6): if every line of the audit's blocking failures can be attributed to a single manifest
        (audit_bisectable: rid join, no segment, a manifest whose ledger instance was not loaded; plus unexpected split reasons when the
        audit counts only the packet's threads), bisect the packet and re-audit to find the offending group, which is quarantined only
        if it fails on its own (stage audit, recorded as abandoned like prepare's group_data); ledger-level failures (lossy,
        writer_error, invariants, media, continuation contract) still stop the run. The packet dir of the failing attempt is renamed
        ``<packet>.rejected-<n>`` as evidence."""
        b = gb["batch"]
        manifests = [Path(r["run_dir"]) / "tito_manifest.json" for r in part]
        threads_only = self.cfg["audit"]["command"][-2:] == ["-m", "arcforge.tito"]   # same condition as audit_with_status
        if not audit_bisectable(failed.blocking, manifests, threads_only):
            raise failed
        pd = failed.packet_dir
        with self.cv:
            gb["rejected"] += 1
            aside = pd.with_name(f"{pd.name}.rejected-{gb['rejected']}")
        pd.rename(aside)
        evidence = aside / failed.report_path.relative_to(pd)
        if len(part) > 1:
            half = len(part) // 2
            self.log(f"batch {b}: audit of {len(part)} group(s) failed on manifest-level checks (attempt set aside: {aside}): "
                     f"splitting the packet into {half} + {len(part) - half} group(s) to find the culprit")
            return [part[:half], part[half:]]
        self.gr_quarantine(gb, part, "audit", "AuditFailed: " + "; ".join(failed.blocking), evidence)
        self.log(f"batch {b}: audit failed for group {part[0]['group']} alone (attempt set aside: {aside}): quarantined")
        return []

    def gr_rejected(self, gb: dict, part: list[dict], rejected: PrepareRejected) -> list[list[dict]]:
        """gr_materialize: set the rejected attempt's packet directory aside, then what to prepare instead of ``part``."""
        b, doc = gb["batch"], rejected.doc
        pd = rejected.path.parent.parent                       # <packet dir>/logs/prepare.error.json
        with self.cv:
            gb["rejected"] += 1
            aside = pd.with_name(f"{pd.name}.rejected-{gb['rejected']}")
        pd.rename(aside)
        evidence = aside / rejected.path.relative_to(pd)
        if rejected.kind == "group_data":
            by_manifest = {str(Path(r["run_dir"]) / "tito_manifest.json"): r for r in part}
            named = list(dict.fromkeys(doc["manifests"]))
            outside = [m for m in named if m not in by_manifest]
            if not named or outside:
                raise OrchestratorError(f"batch {b}: prepare_rl group_data names manifest(s) {named[:5]}, not in its packet "
                                        f"{outside[:5]}; see {evidence}")
            bad = self.gr_quarantine(gb, [by_manifest[m] for m in named], "prepare",
                                     f"{doc['error_type']}: {doc['message']}", evidence)
            rest = [r for r in part if r["gid"] not in bad]
            self.log(f"batch {b}: prepare_rl named {len(bad)} of {len(part)} group(s) with incomplete data (attempt set "
                     f"aside: {aside}); {f'preparing the other {len(rest)}' if rest else 'no group left in this packet'}")
            return [rest] if rest else []
        if rejected.kind != "routing_over_max_gb":
            raise OrchestratorError(f"batch {b}: prepare_rl rejection kind {rejected.kind!r} is not handled; see {evidence}")
        gigabytes = doc["routing_bytes"] / 1e9
        if len(part) == 1:
            raise OrchestratorError(f"batch {b}: group {part[0]['group']} ({part[0]['task']}) alone needs {gigabytes:.1f} GB "
                                    f"of routing > prepare.max_gb {doc['max_gb']}: one group cannot be split; raise "
                                    f"prepare.max_gb (min_free_gb must cover max_gb + reserve_gb). See {evidence}")
        half = len(part) // 2
        self.log(f"batch {b}: routing {gigabytes:.1f} GB of {len(part)} group(s) > prepare.max_gb {doc['max_gb']}: "
                 f"splitting the packet into {half} + {len(part) - half} group(s) (attempt set aside: {aside})")
        return [part[:half], part[half:]]

    def gr_quarantine(self, gb: dict, bad: list[dict], stage: str, error: str, evidence: Path) -> set[int]:
        """``bad`` (groups named by prepare_rl's group_data, or the one group that bisection of the audit left failing on its own):
        recorded as abandoned in groups.jsonl (reason quarantine: never trained), removed from the batch (members / the groups in
        state.json; ``live`` unchanged: the sealed batch keeps the estimate from reservation time, the missing group is like episodes
        dropped by prepare, and the batch is not topped up), one row each in quarantine.jsonl (same event), and finally the circuit
        breaker is checked. Returns their gids."""
        b = gb["batch"]
        if not bad:
            raise OrchestratorError(f"batch {b}: internal: gr_quarantine with no group ({stage}); see {evidence}")
        gids = {r["gid"] for r in bad}
        for r in bad:
            append_jsonl(self.root / "groups.jsonl", {"event": "abandoned", "gid": r["gid"], "group": r["group"],
                                                      "utc": utc_now(), "reason": "quarantine", "stage": stage,
                                                      "batch": b, "evidence": str(evidence)})
        with self.cv:
            keep = [i for i, r in enumerate(gb["members"]) if r["gid"] not in gids]
            if len(gb["members"]) - len(keep) != len(gids) or gb["materialized"] < len(gids):
                raise OrchestratorError(f"batch {b}: internal: quarantined group(s) {sorted(gids)} are not materialized "
                                        f"members of the batch")
            gb["members"] = [gb["members"][i] for i in keep]
            gb["reserved_unix"] = [gb["reserved_unix"][i] for i in keep]
            gb["state"]["groups"] = [g for g in gb["state"]["groups"] if g not in gids]
            gb["materialized"] -= len(gids)
            self.gr_write_state(gb)
            self.cv.notify_all()
        rows = [self.quarantine_row(r["gid"], r["group"], r["task"], stage, error, [evidence], r["rewards"], peers=bad)
                for r in bad]
        self.record_quarantine(rows)
        self.check_quarantine_breaker(rows)
        return gids

    def gr_packet(self, gb: dict, index: int, group: list[dict]) -> dict:
        """One packet (groups in completion order): ledger landing -> audit -> prepare_rl --defer-fraction-gates."""
        sc, b = self.stream, gb["batch"]
        # the session in the name: the trainer refuses a (batch_dir, minibatch) it saw before, aborted ones included, and
        # it outlives orchestrator restarts that set a batch aside and materialize its groups again
        pd = gb["rd"] / "packets" / f"p{index:03d}-{self.session_id}"
        (pd / "logs").mkdir(parents=True)
        short = self.disk_short(sc["hard_min_free_gb"])
        if short:
            raise OrchestratorError(f"{'; '.join(short)} < stream.hard_min_free_gb {sc['hard_min_free_gb']}: the learner stops "
                                    f"(prepare and publish write there)")
        began = time.monotonic()
        watermark = {"enqueued_events": max(r["enqueued_events"] for r in group)}
        if self.cfg["rollout"].get("replica_urls"):
            replica_events: dict[str, int] = {}
            for row in group:
                for name, value in row["replica_events"].items():
                    replica_events[name] = max(value, replica_events.get(name, 0))
            watermark["replica_events"] = replica_events
        landing = self.ledger_landing(watermark)
        t_ledger = time.monotonic()
        audit = self.audit_with_status(pd, [Path(r["run_dir"]) for r in group], landing["status"])
        t_audit = time.monotonic()
        warm = ((arc3_mode(self.cfg["prepare"]) == "rank" and self.cfg["prepare"]["arc3_rank_warmup_versions"] > 0)
                or arc3_mode(self.cfg["prepare"]) == "hinge")
        prepared = self.prepare_batch(b, pd, audit, None, window=(b, sc["max_staleness"]),
                                      extra=("--defer-fraction-gates",),
                                      launch_versions={str(Path(r["run_dir"]) / "tito_manifest.json"):
                                                       r["server_version_launch"] for r in group} if warm else None)
        t_prepare = time.monotonic()
        summary = prepared["summary"]
        empty = prepared.get("status") == "EMPTY_ELIGIBLE"
        unit = "groups_kept" if self.cfg["prepare"].get("loss_reduction", "sequence") == "prompt" else "episodes_kept"
        units = 0 if empty else summary[unit]
        if not empty and not (isinstance(units, int) and units >= 1):
            raise OrchestratorError(f"{pd}: prepare kept data but {unit}={units!r}")
        packet = {"index": index, "gids": [r["gid"] for r in group], "dir": str(pd), "batch_dir": prepared["batch_dir"],
                  "empty": empty, "units": units, "quality": summary["quality"], "summary": summary,
                  "audit": {k: audit[k] for k in ("instance", "reports")},
                  "seconds": {"ledger": round(t_ledger - began, 2), "audit": round(t_audit - t_ledger, 2),
                              "prepare": round(t_prepare - t_audit, 2)},
                  "materialized_unix": time.time()}
        append_jsonl(gb["rd"] / "packets.jsonl", packet)
        self.log(f"batch {b}: packet p{index:03d} of {len(group)} group(s) {[r['group'] for r in group]}: "
                 f"{'EMPTY_ELIGIBLE' if empty else f'{units} unit(s)'} in {t_prepare - began:.0f}s "
                 f"(ledger {t_ledger - began:.0f}s, audit {t_audit - t_ledger:.0f}s, prepare {t_prepare - t_audit:.0f}s)")
        return packet

    def stream_learner_group_ready(self, start: int, total: int, resume: dict | None) -> str:
        """stream.group_ready learner: batch b (trainer at V=b) /accumulate every packet as it is materialized; once the
        batch is sealed and every packet is in, the 0.5% gates over the whole batch -> /step (units = kept groups, or kept
        episodes for sequence reduction) -> /publish b+1 -> in_place sync -> batches.jsonl row. No kept group: refill at
        the same version (EMPTY_ELIGIBLE, like stream). ``resume``: a batch whose optimizer step is done; only publish /
        sync / save / row remain."""
        try:
            b = start
            if resume is not None:
                ready = time.monotonic()
                pool = {**resume["pool_after"], "formed_mono": ready, "formed_unix": resume["formed_unix"]}
                formed = self.form_batch(b, resume["groups"], pool, ready, resumed=resume)
                if formed is None:
                    raise OrchestratorError(f"batch {b}: a group_ready batch that took an optimizer step cannot be empty")
                formed["group_ready"] = resume["state"]["gr"]
                self.finish_batch(formed)
                b += 1
            consecutive_empty = 0
            while b < total:
                with self.cv:
                    self.gr_learner = b
                    self.gr_frontier = max(self.gr_frontier, b)
                    self.cv.notify_all()
                outcome = self.gr_learn(b)
                if outcome == "stopped":
                    return "stopped"
                if outcome == "empty":
                    consecutive_empty += 1
                    if consecutive_empty >= self.stream.get("max_empty_batches", 100):
                        raise OrchestratorError(f"batch {b}: {consecutive_empty} consecutive EMPTY_ELIGIBLE batches; no "
                                                "optimizer step or publication; inspect the discarded batch reports")
                    continue
                consecutive_empty = 0
                b += 1
            with self.cv:
                self.learner_finished = True
                self.cv.notify_all()
            return "done"
        finally:
            with self.cv:
                self.gr_closed = True
                self.cv.notify_all()

    def gr_learn(self, b: int) -> str:
        """One group_ready batch: ``done`` (stepped, published, synced, row written), ``empty`` (no kept group: discarded,
        refill at the same version) or ``stopped`` (the dispatcher ended and the pool cannot seal it: an open update is
        aborted; the batch dir stays for resume to set aside)."""
        tcfg = self.cfg["trainer"]
        ready = time.monotonic()
        update_id = f"{self.root.name}/b{b:06d}/{self.session_id}"
        begun: dict | None = None
        accumulated: list[dict] = []
        while True:
            with self.cv:
                while True:
                    self.check_cancel()
                    self.gr_reserve()
                    gb = self.gr_batches.get(b)
                    if gb is not None and gb["taken"] < len(gb["packets"]):
                        packet = gb["packets"][gb["taken"]]
                        gb["taken"] += 1
                        break
                    if (gb is not None and gb["sealed"] and gb["materialized"] == len(gb["members"])
                            and gb["formed"] == len(gb["packets"]) == gb["taken"]):
                        packet = None
                        break
                    if self.dispatch_ended and not self.pool and (gb is None or not gb["sealed"]):
                        live = 0 if gb is None else gb["live"]
                        self.log(f"batch {b}: the dispatcher stopped with {live} live episode(s) reserved < "
                                 f"{self.stream['batch_live_episodes']}: no more batches")
                        packet = "stop"
                        break
                    self.cv.wait(self.poll)
            if packet == "stop":
                if begun is not None:
                    self.trainer("POST", "/abort", {"update_id": update_id}, timeout=tcfg["train_timeout_seconds"])
                    self.log(f"batch {b}: aborted update {update_id} ({len(accumulated)} packet(s) accumulated)")
                return "stopped"
            if packet is None:
                break
            if packet["empty"]:
                continue
            if begun is None:
                with self.cv:
                    gb["state"]["update"] = {"id": update_id, "begin_utc": utc_now(), "packets": []}
                    self.gr_write_state(gb)
                begun = self.trainer("POST", "/begin", {"update_id": update_id}, timeout=60)
                if begun.get("update_id") != update_id or str(begun.get("weight_version")) != str(b):
                    raise OrchestratorError(f"batch {b}: /begin answered {begun}, want update {update_id} at weight_version {b}")
            t0 = time.monotonic()
            reply = self.trainer("POST", "/accumulate", {"update_id": update_id, "batch_dir": packet["batch_dir"],
                                                         "minibatch": 0}, timeout=tcfg["train_timeout_seconds"])
            assert_finite(reply, f"/accumulate packet p{packet['index']:03d}")
            if reply.get("units") != packet["units"] or reply.get("update_id") != update_id:
                raise OrchestratorError(f"batch {b}: /accumulate of p{packet['index']:03d} answered units "
                                        f"{reply.get('units')!r} update {reply.get('update_id')!r}, want {packet['units']} {update_id}")
            done = {"index": packet["index"], "units": packet["units"], "seconds": round(time.monotonic() - t0, 2),
                    "accumulated_unix": time.time(), "trainer": {k: reply.get(k) for k in ("micro_steps", "tokens",
                                                                                           "loss_tokens", "seconds")}}
            accumulated.append(done)
            with self.cv:
                gb["state"]["update"]["packets"].append(done)
                self.gr_write_state(gb)
            self.log(f"batch {b}: accumulated p{packet['index']:03d} ({packet['units']} unit(s)) in {done['seconds']:.0f}s")
        with self.cv:
            gb = self.gr_batches[b]
            packets, members = list(gb["packets"]), list(gb["members"])
        units = sum(p["units"] for p in packets)
        if units == 0:
            if begun is not None:
                raise OrchestratorError(f"batch {b}: internal: an update was begun but no packet kept a unit")
            self.gr_discard_empty(b)
            return "empty"
        quality = dict(sum((Counter(p["quality"]) for p in packets), Counter()))
        fractions = (("routing_M", self.cfg["prepare"].get("max_routing_m_fraction", MAX_ROUTING_M_FRACTION), "routing M"),
                     ("weight_version_unrecorded", 0.005, "weight_versions not recorded"),
                     ("routing_D_cross", self.cfg["prepare"].get("max_d_cross_fraction", MAX_D_CROSS_FRACTION),
                      "routing D_cross (forwarded twice across weight versions)"))
        failed = [f"{label} dropped {quality.get(key, 0)}/{quality['with_tokens']} episode(s) (> {limit:.1%})"
                  for key, limit, label in fractions if quality.get(key, 0) > limit * quality["with_tokens"]]
        if failed:
            if begun is not None:
                self.trainer("POST", "/abort", {"update_id": update_id}, timeout=tcfg["train_timeout_seconds"])
            raise OrchestratorError(f"batch {b}: {'; '.join(failed)} over its {len(packets)} packet(s) (the fraction gates "
                                    f"of prepare_rl, judged on the whole optimizer step); update {update_id} aborted")
        merged = merge_prepare_summaries([p["summary"] for p in packets])
        reports = [rep for p in packets for rep in p["audit"]["reports"]]
        last_completed = max(r["completed_unix"] for r in members)
        gr = {"packets": [{k: p[k] for k in ("index", "gids", "dir", "batch_dir", "empty", "units", "quality", "seconds",
                                              "materialized_unix")} for p in packets],
              "accumulated": accumulated, "units": units, "quality": quality, "update_id": update_id,
              "reserved_unix": dict(zip([r["gid"] for r in members], gb["reserved_unix"], strict=True)),
              "sealed_unix": gb["sealed_unix"], "last_group_completed_unix": last_completed,
              "accumulate_seconds": round(sum(a["seconds"] for a in accumulated), 2)}
        with self.cv:
            st = gb["state"]
            st["stages"]["ledger"] = {"group_ready": True, "packets": len(packets)}
            st["stages"]["audit"] = {"instance": packets[0]["audit"]["instance"], "reports": reports}
            st["stages"]["prepare"] = {"group_ready": True, "summary": merged, "minibatches": 1,
                                       "batch_dirs": [p["batch_dir"] for p in packets if not p["empty"]]}
            st["gr"] = gr
            st["train"]["inflight"] = {"minibatch": 0, "step_before": begun["step"]}
            self.gr_write_state(gb)
        t0 = time.monotonic()
        report = self.trainer("POST", "/step", {"update_id": update_id, "units": units}, timeout=tcfg["train_timeout_seconds"])
        assert_finite(report, f"batch {b} /step")
        after = self.trainer_health()
        if after["step"] != begun["step"] + 1 or after.get("update") is not None:
            raise OrchestratorError(f"batch {b}: trainer at step {after['step']} (update {after.get('update')!r}) after /step, "
                                    f"want {begun['step'] + 1} with no open update")
        gr["step_seconds"] = round(time.monotonic() - t0, 2)
        gr["step_done_unix"] = time.time()
        with self.cv:
            st["train"]["metrics"] = [report]
            st["train"]["inflight"] = None
            st["stages"]["train"] = {"minibatches": 1}
            st["timings"]["train"] = round(gr["accumulate_seconds"] + gr["step_seconds"], 1)
            self.gr_write_state(gb)
            pool = {"groups": len(self.pool), "live_groups": sum(r["live"] for r in self.pool),
                    "groups_in_flight": len(self.inflight_groups), "formed_unix": gb["sealed_unix"],
                    "formed_mono": max(gb["sealed_mono"], ready)}
        self.log(f"batch {b}: /step over {len(accumulated)} packet(s), {units} unit(s), "
                 f"{gr['step_done_unix'] - last_completed:.0f}s after its last group completed")
        formed = {"batch": b, "rd": gb["rd"], "groups": members, "live": [r for r in members if r["live"]], "pool": pool,
                  "ready": ready, "resumed": None, "state": st, "state_path": gb["state_path"],
                  "stage": self.make_stage(b, st, gb["state_path"], label="batch"),
                  "audit": st["stages"]["audit"], "prepared": st["stages"]["prepare"], "prefetched": False,
                  "group_ready": gr}
        self.finish_batch(formed)
        with self.cv:
            del self.gr_batches[b]
        return "done"

    def gr_discard_empty(self, b: int) -> None:
        """No packet of batch b kept a group: discard it like stream's EMPTY_ELIGIBLE (its groups are recorded abandoned)
        and refill at version b. The next batch only reserved groups (the materializer waits for units > 0): they go back
        to the head of the pool, in order, and its directory is set aside."""
        stamp = time.strftime("%Y%m%dT%H%M%S")
        with self.cv:
            gb = self.gr_batches.pop(b)
            nxt = self.gr_batches.pop(b + 1, None)
            if nxt is not None:
                if nxt["formed"] or nxt["packets"]:
                    raise OrchestratorError(f"batch {b + 1}: internal: packets formed before batch {b} kept a unit")
                self.pool[:0] = nxt["members"]
                self.batches_taken -= 1 if nxt["sealed"] else 0
                nxt["rd"].rename(nxt["rd"].with_name(f"{nxt['rd'].name}.refill-{stamp}"))
            self.batches_taken -= 1
            self.gr_frontier = b
            self.discard_empty_batch(gb["rd"], gb["state"])
            self.cv.notify_all()


# --------------------------------------------------------------------------- CLI

def signal_action(orch: "Orchestrator", signum: int) -> str:
    """SIGUSR1, or SIGTERM / SIGINT while the ``<run>/DETACH`` file exists: detach (in-flight groups keep running; the DETACH file
    is deleted once used, so the next SIGTERM without it really stops). Otherwise: cancel (terminate the children it started, state
    kept). Returns "detach" or "cancel"."""
    detach_file = orch.root / DETACH_FILE
    if signum == signal.SIGUSR1 or detach_file.exists():
        orch.log(f"signal {signum}: detaching; harness groups in flight are left running (the next start adopts them), "
                 f"state is kept")
        if detach_file.exists():
            detach_file.unlink()
            orch.log(f"{detach_file} consumed (removed): the next SIGTERM without it cancels")
        orch.request_detach()
        return "detach"
    orch.log(f"signal {signum}: cancelling; child processes are terminated, state is kept")
    orch.cancelled.set()
    return "cancel"


def group_ids(text: str) -> tuple[int, ...]:
    """``g000012,g000013`` -> (12, 13): every item a group name, no repeats."""
    names = text.split(",")
    if not names or any(not GROUP_DIR.fullmatch(n) for n in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"{text!r}: want distinct group names gNNNNNN, comma-separated")
    return tuple(int(n[1:]) for n in names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="validate config, files and task lists, then exit (no network)")
    ap.add_argument("--abandon-incomplete-round", action="store_true",
                    help="sync: move an unfinished rounds/rNNN aside and redo it (refused once the trainer took steps in it); "
                         "stream: move half-done groups/gNNNNNN and batches/bNNNNNN aside (refused once the trainer took steps) "
                         "and record pooled groups of an old ledger instance as abandoned")
    ap.add_argument("--set-aside-judge-groups", type=group_ids, default=(), metavar="gNNNNNN[,...]",
                    help="stream, arc3_reward note: at start, move these finished-but-unjudged groups (their note judge "
                         "failed every attempt and stopped the run) aside to groups/abandoned-<stamp>/ with a groups.jsonl "
                         "set_aside row; they are never trained. Refused for any group not in that state")
    args = ap.parse_args(argv)
    orch = Orchestrator(load_config(args.config), abandon_incomplete=args.abandon_incomplete_round,
                        set_aside_judge=args.set_aside_judge_groups)
    if args.check:
        orch.static_checks()
        if orch.stream is not None:
            orch.quarantine_cfg()
        print(f"ok: {len(orch.pools['train'])} train tasks, {len(orch.pools['heldout'])} held-out tasks, run_root {orch.root}")
        return 0

    def on_signal(signum, frame):
        signal_action(orch, signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    try:
        outcome = orch.run()
    except Cancelled:
        if orch.detaching:
            print(log_line("detached: in-flight groups left running; start again to adopt them"), flush=True)
            return 4
        print(log_line("cancelled"), flush=True)
        return 3
    except OrchestratorError as error:
        print(log_line(f"FAILED: {error}"), file=sys.stderr, flush=True)
        if orch.log_file is None and orch.root.exists():
            with (orch.root / "orchestrator.log").open("a") as f:
                f.write(log_line(f"FAILED: {error}") + "\n")
        return 1
    print(log_line(outcome), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
