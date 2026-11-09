"""Feature pool for train.py's fast engine: capture servers stream jobs into their own queue roots; rank 0 hands the
jobs to ranks so that every rank does the same work every step.

Layout: ``<pool>/<root>/train-<n:09d>/{features.safetensors,record.json,READY}``, one root per capture server
(capture_tito.py --queue), and ``<root>/train.producer_done`` once that root's producer has finished.

A rank works through up to ``jobs_per_rank`` jobs (captured sequences) at once: each one's eligible anchors in
ascending position order, a contiguous run of them per step. The step runs that run's clean pass under autograd, from
the receptive field before its first anchor (fast_train.clean_start) to its last anchor, so a job's slice costs its
anchors and those clean rows: ``anchors * ANCHOR_SECONDS + clean_rows * CLEAN_ROW_SECONDS``. Each step every rank gets
the same budget and spreads the anchors evenly over its jobs (water-filling: a job with fewer left gives its share to
the others), taking as many as the budget pays for, so a step mixes up to jobs_per_rank x world sequences and no long
sequence monopolizes a rank for many consecutive steps. A job that runs out is replaced in the same step. One job ahead
is reserved per rank, so the rank can load it while it works on the current ones.
"""

import json
import math
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from data import eligible_anchors
from safetensors import safe_open

# Fast engine, dense10240, one B200: forward and backward per anchor, and per row of a clean pass under autograd
# (2026-10-07 pool smoke, 4 jobs a rank: ~11k anchors and ~34k clean rows a rank-step, 1.19 s and 0.29 s).
ANCHOR_SECONDS = 0.108e-3
CLEAN_ROW_SECONDS = 8.8e-6
# Under target-KV injection (Config.context "inject") a clean row is the input fusion and the layers' K/V projections,
# no trunk: 0.93e-6 s a row on one B200, ~9x cheaper (2026-10-19 plain resume check, ~44k clean rows a rank-step).
# Costed at the trunk's rate, a step packed 66.6k of its 89k anchors and the ranks' real costs went 1.30 out of balance.
INJECTED_CLEAN_ROW_SECONDS = 0.93e-6


def clean_row_seconds(config):
    """A clean row's cost for this drafter: the trunk's under context "recompute", the injection's under "inject"."""
    if config.context not in ("recompute", "inject"):
        raise ValueError(f"unknown context {config.context!r}")
    return INJECTED_CLEAN_ROW_SECONDS if config.context == "inject" else CLEAN_ROW_SECONDS


# A segment's clean pass keeps its activations for backward (~a few hundred KB a row over the layers): a slice whose
# segment would run more rows than this is not planned (a single anchor's segment runs at most overlap + 1).
MAX_SEGMENT_ROWS = 65536


def segments(positions, lo, hi, overlap):
    """The anchors positions[lo:hi] (ascending, an array) as index ranges that each get their own clean pass: split
    where consecutive anchors are more than the receptive field apart (their clean rows do not overlap, and a pass over
    the gap between them -- a long tool output -- would cost more than the second one's own receptive field)."""
    if hi <= lo:
        return []
    cuts = [lo, *(np.flatnonzero(np.diff(positions[lo:hi]) > overlap) + lo + 1).tolist(), hi]
    return list(zip(cuts, cuts[1:], strict=False))


def _segment_rows(positions, lo, hi, overlap):
    """Each segment's clean rows (fast_train.clean_pass from clean_start through its last anchor - 1)."""
    bounds = np.array(segments(positions, lo, hi, overlap), dtype=np.int64).reshape(-1, 2)
    first = np.maximum(1, positions[bounds[:, 0]] - overlap)
    return np.maximum(positions[bounds[:, 1] - 1] - 1, first) - first + 1


def clean_rows(positions, lo, hi, overlap):
    """The clean rows a step runs for a job's anchors positions[lo:hi], over its segments."""
    return int(_segment_rows(positions, lo, hi, overlap).sum()) if hi > lo else 0


def slice_cost(positions, lo, hi, overlap, row_seconds=CLEAN_ROW_SECONDS):
    if hi <= lo:
        return 0.0
    rows = _segment_rows(positions, lo, hi, overlap)
    if rows.max() > MAX_SEGMENT_ROWS:
        return math.inf
    return (hi - lo) * ANCHOR_SECONDS + int(rows.sum()) * row_seconds


def planned_steps(anchors, tokens, step_anchors):
    """About the steps the Planner takes over jobs totalling these anchors and tokens: every step hands each rank
    step_anchors / world anchors' worth of budget, and the clean rows cost about the tokens (more when a job's anchors
    span several steps: each slice also runs the receptive field before it). An estimate for progress displays; the
    learning-rate schedule follows the plan's exact anchor total (train.py --total-anchors), and the run drains the
    pool either way."""
    if anchors < 0 or tokens < 0 or step_anchors <= 0:
        raise ValueError("planned_steps needs nonnegative totals and a positive step")
    return math.ceil((anchors * ANCHOR_SECONDS + tokens * CLEAN_ROW_SECONDS) / (step_anchors * ANCHOR_SECONDS))


def water_fill(remaining, total):
    """Split `total` anchors over jobs with `remaining` anchors left: equal shares, a job that has fewer left than its
    share takes all it has and the rest is shared by the others. Integer; hands out min(total, sum(remaining))."""
    shares = [0] * len(remaining)
    left = total
    order = sorted(range(len(remaining)), key=lambda i: remaining[i])
    for position, i in enumerate(order):
        share = min(remaining[i], left // (len(order) - position))
        shares[i] = share
        left -= share
    return shares


@dataclass
class Job:
    path: Path
    tokens: int
    positions: np.ndarray  # its anchors, ascending
    handed: int = 0  # anchors already planned

    @property
    def anchors(self):
        return len(self.positions)


@dataclass(frozen=True)
class Unit:
    """One rank's slice [lo, hi) of a job's ascending anchors in one step."""

    path: str
    lo: int
    hi: int
    start: bool  # the job's first slice: load it and run its clean pass
    last: bool  # the job's last slice: release it after the step


class Planner:
    """Rank 0's view of the pool. plan() blocks while capture has nothing ready and is not done.

    opd_fraction: OPD jobs (opd_assemble.py, recorded verify cycles) get that fraction of every rank's budget in every
    step, over up to opd_jobs_per_rank of them at once, and the captured jobs the rest -- so the cycles spread evenly
    over the run at a fixed share. Without it every job shares one queue in arrival order, and OPD jobs, all ready at
    the start, would fill the first steps. What the OPD jobs leave unused goes to the captured ones, and once the
    captured jobs are drained the OPD jobs take the whole budget. opd_fraction 1: the OPD jobs first, the captured
    ones only what they leave (an OPD-only run has no captured root)."""

    def __init__(self, pool, world, step_anchors, overlap, poll_seconds=0.5, stop_file=None, jobs_per_rank=1,
                 max_rank_tokens=None, opd_fraction=None, opd_jobs_per_rank=None, row_seconds=CLEAN_ROW_SECONDS):
        self.roots = sorted(p for p in Path(pool).iterdir() if p.is_dir())
        if not self.roots:
            raise ValueError(f"{pool} holds no queue roots")
        for root in self.roots:
            if not (root / "capture_server.json").is_file():
                raise ValueError(f"{root} is not a capture queue root (no capture_server.json)")
        if step_anchors < world:
            raise ValueError("--step-anchors must give every rank at least one anchor")
        if jobs_per_rank < 1 or (max_rank_tokens is not None and max_rank_tokens < 1):
            raise ValueError("jobs_per_rank and max_rank_tokens must be positive")
        if (opd_fraction is None) != (opd_jobs_per_rank is None):
            raise ValueError("opd_fraction and opd_jobs_per_rank go together")
        if opd_fraction is not None and not (0 < opd_fraction <= 1 and opd_jobs_per_rank >= 1):
            raise ValueError(f"opd_fraction {opd_fraction} must be in (0, 1], opd_jobs_per_rank positive")
        self.opd_fraction = opd_fraction
        # "all": one queue (no OPD share); "opd" / "captured": the two shares. Each kind has its own queue, slots per
        # rank, jobs in progress and reserved job.
        self.kinds = ("all",) if opd_fraction is None else ("opd", "captured")
        self.slots = {"all": jobs_per_rank} if opd_fraction is None else {"opd": opd_jobs_per_rank,
                                                                         "captured": jobs_per_rank}
        # A job in progress keeps its features on its rank's GPU (~20 KB a token, the GR), so a rank starts another
        # job only while its jobs' tokens stay within this; a rank with no job takes any job.
        self.max_rank_tokens = max_rank_tokens
        self.overlap = overlap  # fast_train.receptive_field
        self.row_seconds = row_seconds  # clean_row_seconds(config)
        self.world = world
        self.budget = step_anchors / world * ANCHOR_SECONDS
        self.poll = poll_seconds
        self.seen = set()
        self.pending = {kind: deque() for kind in self.kinds}
        self.released_empty = 0  # jobs without an eligible anchor, deleted unread
        self.current = {kind: [[] for _ in range(world)] for kind in self.kinds}  # each rank's jobs in progress
        self.reserved = {kind: [None] * world for kind in self.kinds}
        self.starved_seconds = 0.0
        # While it waits for capture, plan() also watches this file (train.py's STOP): a producer that died never marks
        # its root done, and the trainer has to be able to checkpoint and exit then (PR #6 F4).
        self.stop_file = None if stop_file is None else Path(stop_file)
        self.stopping = False

    def _scan(self):
        found = []
        for root in self.roots:
            for directory in root.glob("train-*"):
                if directory in self.seen or not (directory / "READY").is_file():
                    continue
                record = json.loads((directory / "record.json").read_text())
                if record["split"] != "train":
                    raise ValueError(f"{directory}: held-out features cannot enter training")
                opd = (directory / "cycles.safetensors").is_file()
                if opd:
                    # An OPD job (opd_assemble.py): its units are recorded verify cycles, each costs what an anchor does.
                    with safe_open(str(directory / "cycles.safetensors"), "pt") as f:
                        positions = f.get_tensor("anchor").numpy()
                    if len(positions) != record["cycles"]:
                        raise ValueError(f"{directory}: {len(positions)} cycles, record.json says {record['cycles']}")
                    if (np.diff(positions) <= 0).any():
                        raise ValueError(f"{directory}: cycle anchors do not increase")
                else:
                    with safe_open(str(directory / "features.safetensors"), "pt") as f:
                        positions = eligible_anchors(f.get_tensor("loss_mask")).numpy()
                kind = "all" if self.opd_fraction is None else "opd" if opd else "captured"
                found.append((record["sequence_index"], str(directory), kind,
                              Job(directory, record["num_tokens"], positions)))
                self.seen.add(directory)
        for _, _, kind, job in sorted(found, key=lambda x: x[:2]):  # roots interleaved by job number
            if job.anchors:
                self.pending[kind].append(job)
            else:  # no rank ever sees it
                shutil.rmtree(job.path)
                self.released_empty += 1

    def _producers_done(self):
        return all((root / "train.producer_done").is_file() for root in self.roots)

    def _next_job(self, kind, block):
        """The next `kind` job with anchors; None when the pool is drained of them (or, unless block, nothing is ready
        yet)."""
        while True:
            if self.pending[kind]:
                return self.pending[kind].popleft()
            # Producers mark done after their last READY, so a scan after seeing done misses nothing.
            done = self._producers_done()
            self._scan()
            if self.pending[kind]:
                continue
            if done or not block:
                return None
            if self.stop_file is not None and self.stop_file.exists():
                self.stopping = True
                return None
            began = time.monotonic()
            time.sleep(self.poll)
            self.starved_seconds += time.monotonic() - began

    def plan(self):
        """{"units": per-rank [Unit], "reserve": per-rank [path] (the jobs to load next), "finished": [path]}, None when
        done, or {"stop": True} when the stop file appeared while it waited for capture (this call's partial plan is
        dropped: the run checkpoints and exits, and a resume captures every job it has not finished again)."""
        units = [[] for _ in range(self.world)]
        finished = []
        # Wait for capture only while this call holds nothing: a job it has started keeps its directory (queue space)
        # until its rank loads it after plan() returns, so waiting then can deadlock against producers blocked on
        # their --max-queue-gib. A rank whose slots stay empty does less this step and fills them later. OPD jobs are
        # assembled before the run and hold no producer's queue space: they neither wait nor count here.
        self._started_any = False
        for rank in range(self.world):
            # slices: job -> [lo, hi, start] this step (a job's slices in one step are contiguous); finishing: tokens of
            # jobs finished earlier in this step -- their features are freed only after the step, so they still count
            step = {"slices": {}, "finishing": 0, "units": units[rank], "finished": finished}
            if self.opd_fraction is None:
                self._fill(rank, "all", self.budget, step)
            else:
                left = self.budget - self._fill(rank, "opd", self.budget * self.opd_fraction, step)
                left -= self._fill(rank, "captured", left, step)
                if self._drained("captured", rank):
                    self._fill(rank, "opd", left, step)
            for kind in self.kinds:
                for job in self.current[kind][rank]:
                    if id(job) in step["slices"]:
                        lo, hi, start = step["slices"][id(job)]
                        units[rank].append(Unit(str(job.path), lo, hi, start, False))
                if self.reserved[kind][rank] is None and not self.stopping:
                    self.reserved[kind][rank] = self._next_job(kind, block=False)
        if self.stopping:
            return {"stop": True}
        if not any(units):
            return None
        return {
            "units": units,
            "reserve": [[str(self.reserved[kind][rank].path) for kind in self.kinds
                         if self.reserved[kind][rank] is not None] for rank in range(self.world)],
            "finished": finished,
        }

    def _drained(self, kind, rank):
        """No `kind` job left for this rank, now or later."""
        if self.current[kind][rank] or self.reserved[kind][rank] is not None:
            return False
        done = self._producers_done()  # before the scan, as in _next_job
        self._scan()
        return done and not self.pending[kind]

    def _fill(self, rank, kind, budget, step):
        """Hand out this rank's `kind` anchors for up to `budget` (planned seconds) this step; returns what it spent."""
        current = self.current[kind][rank]
        spent_total = 0.0
        # jobs whose next anchor alone costs more than the budget left (it lies past a long stretch of prompt or tool
        # output whose clean rows the slice would have to run): the others share what is left
        blocked = set()
        while True:
            while len(current) < self.slots[kind] and not self.stopping:
                block = kind != "opd" and not self._started_any and not current
                job = self.reserved[kind][rank] or self._next_job(kind, block=block)
                self.reserved[kind][rank] = None
                if job is None:
                    break
                held = sum(j.tokens for k in self.kinds for j in self.current[k][rank]) + step["finishing"]
                if held and self.max_rank_tokens is not None and held + job.tokens > self.max_rank_tokens:
                    self.reserved[kind][rank] = job  # starts once this rank's jobs have room for it
                    break
                current.append(job)
                if kind != "opd":
                    self._started_any = True
                step["slices"][id(job)] = [job.handed, job.handed, True]
            active = [job for job in current if id(job) not in blocked]
            if not active:
                break
            slices = step["slices"]
            starts = {id(job): slices[id(job)][0] if id(job) in slices else job.handed for job in active}

            def grows(job, take, starts=starts):
                """What `take` more anchors add to the cost of the job's slice this step."""
                lo = starts[id(job)]
                return (slice_cost(job.positions, lo, job.handed + take, self.overlap, self.row_seconds)
                        - slice_cost(job.positions, lo, job.handed, self.overlap, self.row_seconds))

            def cost(total, active=active):
                shares = water_fill([job.anchors - job.handed for job in active], total)
                return sum(grows(job, share) for job, share in zip(active, shares, strict=True)), shares

            low, high = 0, sum(job.anchors - job.handed for job in active)  # the most the budget pays for
            while low < high:
                middle = (low + high + 1) // 2
                if cost(middle)[0] <= budget:
                    low = middle
                else:
                    high = middle - 1
            spent, shares = cost(low)
            for job, take in zip(active, shares, strict=True):
                piece = slices.setdefault(id(job), [job.handed, job.handed, False])
                job.handed += take
                piece[1] = job.handed
            budget -= spent
            spent_total += spent
            done = [job for job in current if job.handed == job.anchors]
            if not done:
                stuck = {id(job) for job in active if grows(job, 1) > budget}
                if stuck == {id(job) for job in active}:
                    break  # the budget is spent
                blocked |= stuck
                if not stuck:
                    break  # cannot happen: the search would have handed out one more anchor
                continue
            for job in done:
                current.remove(job)
                step["finishing"] += job.tokens
                step["finished"].append(str(job.path))
                lo, hi, start = slices.pop(id(job))
                step["units"].append(Unit(str(job.path), lo, hi, start, True))
        return spent_total

    def pending_jobs(self):
        return sum(len(queue) for queue in self.pending.values())
