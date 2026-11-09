"""wandb sidecar for an RL run: reads the run root's append-only files and logs them to wandb. Writes nothing the orchestrator reads.

    $HOME/work/venv-fn-train/bin/python work/qwen38fn_rl/wandb_sync.py --run-root <run_root> [--config <run config>] \\
        [--project qwen38fn-rl] [--name <run root dir name>] [--follow [--poll 30] [--orchestrator-pid PID]] \\
        [--baseline <sample run dir> ...] [--wandb-dir <dir>]

Reads ``rounds.jsonl`` (sync / async) or ``batches.jsonl`` + ``groups.jsonl`` (stream), and
``heldout_offline/vNNNNNN/summary.json`` (heldout_offline.py). Without --config, it looks in ``configs/`` for the one whose run_root
is this one.
- One wandb step per row, x axis ``version`` = the trainer's version when the row was trained (weight_version_in = row number).
  Prefixes train / optim (mean and max over minibatches) / prepare (drop reasons; stream also a staleness histogram) / rollout
  (generated tokens per second) / sync / async / dispatcher / pool / time. ``train/live_groups`` are groups whose scores are not all
  equal; stream also has ``train/trained_live_groups`` = the ones the learner actually trained on (under arc3's efficiency reward,
  groups that all cleared but differ in efficiency count too).
- stream also has ``train/quarantined_groups`` (cumulative) and ``train/quarantined_groups_batch`` (since the previous row): groups
  the orchestrator quarantined (``quarantine.jsonl``).
- arc3_level stream runs get two more sets: ``arc3/*`` = level-clear rate and efficiency of this batch's episodes (groups.jsonl):
  official single-level score min((baseline / actions)², 1.15) (not cleared = 0), actions / baseline of cleared games and output
  tokens at clearing; ``arc3/reward_mean`` is the training R (prepare.arc3_reward rank: recomputed per group by
  orchestrator.rank_rewards; otherwise: score), ``arc3/score_mean`` is always score; ``reward/*`` = prepare's arc3_reward summary
  (the efficiency term's share of the advantage, c0's token-component share, guard / clip counts, mean p̂; rows from before 10-20
  have none).
- ``train_log.jsonl`` in the same dir as the trainer's ``endpoint_file`` has one entry per optimizer step, prefix ``learner/*``, x
  axis ``learner/step``; it does not wait for the whole batch's publish. An old cursor backfills this new curve on first use without
  re-logging the per-version aggregates; the source has no timestamps, so none are fabricated.
- Optional low-frequency reward check from ``analysis/reward-watch/latest.json``, prefix ``learning_health/*``, its own x axis
  ``learning_health/optimizer_step``; an alert only means a regression was observed, with no causal or held-out conclusion.
- held-out has its own x axis ``heldout/version`` (offline results arrive late, and old versions are still drawn at the right
  place); in-loop and offline use the same keys, with ``heldout/offline`` 0/1. Paired per task against base (heldout_compare):
  offline uses summary.json's compare_base; in-loop compares against ``--baseline`` (default base24, as in heldout_offline.py),
  which has only per_task_mean, so only reward gets se / z.
- Resume: ``<run_root>/wandb_sync.json`` records the wandb run id, the number of rows logged, the held-out versions logged, the line
  reached in groups.jsonl, and the number of steps logged. Running again reattaches to the same wandb run (resume="allow", same id)
  without re-logging; a project / name different from the file's raises. A wandb step count that does not match the file (logged
  but not recorded in the file, or the run was deleted) also raises.
- ``--follow``: read every --poll seconds. When the orchestrator is gone (the /proc/<pid>/cmdline of the PID in
  ``orchestrator.lock``, or of --orchestrator-pid, does not contain orchestrator.py), read one last time and exit. The lock is only
  read, never taken or touched. SIGTERM / SIGINT: finish logging the current entry and the wandb flush before exiting. Only one
  sidecar per run root (flock on the run root dir).
- Unparseable rows, missing required keys, NaN / inf: raise, with file name and line number in the message. Fields that may be
  absent (keys missing from rows written by older orchestrators, no pause in an idle sync) are each spelled out explicitly.
- wandb's local files go to --wandb-dir (default ``$ARC_RUNS/qwen38fn_rl/wandb``), not the run root. System monitoring (GPU etc.) is
  off: this process uses no GPU, and whole-machine numbers on a shared machine would only mislead.
"""

from __future__ import annotations

import argparse
import calendar
import collections
import fcntl
import json
import math
import os
import re
import secrets
import signal
import statistics
import string
import sys
import time
from pathlib import Path

import wandb

from heldout_compare import paired, per_task
from heldout_offline import base24_dirs
from orchestrator import (ARC3_G2_KIND, ARC3_LEVEL_KIND, QUARANTINE_FILE, QUARANTINE_STAGES, TOP_KEYS, _expand, atomic_write, cfg_path,
                          hinge_rewards, load_config, rank_rewards, read_episodes, utc_now)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CURSOR_NAME = "wandb_sync.json"
CURSOR_FORMAT = "qwen38fn-rl-wandb-sync-v1"
DEFAULT_PROJECT = "qwen38fn-rl"
DEFAULT_POLL_SECONDS = 30.0
#: pipeline → (history file, row-number key)
HISTORY = {"sync": ("rounds.jsonl", "round"), "async": ("rounds.jsonl", "round"), "stream": ("batches.jsonl", "batch")}
#: prefixes whose x axis is ``version``; the x axis of ``heldout/*`` is ``heldout/version``
VERSION_PREFIXES = ("train", "optim", "prepare", "rollout", "sync", "async", "dispatcher", "pool", "time", "arc3", "reward")
#: cap on the official single-level score (×100 is the official score: min(100 (b / a)², 115))
ARC3_LEVEL_SCORE_CAP = 1.15
#: counts in prepare's arc3_reward summary logged as reward/<key> (mode-specific: c0 guard_hits, eff clip_hits)
ARC3_REWARD_COUNTS = ("groups_kept", "groups_eff_nonzero", "all_cleared_groups_kept")
ARC3_REWARD_MODE_COUNTS = {"c0": ("guard_hits",), "eff": ("clip_hits",),
                           "official": ("full_groups", "cleared_zero_episodes", "partial_episodes"),
                           "rank": ("ranked_groups", "warmup_groups"), "hinge": ("hinged_groups", "unfrozen_groups"),
                           "note": ("window_episodes", "early_window_episodes")}
HELDOUT_PREFIX = "heldout"
LEARNER_PREFIX = "learner"
HEALTH_PREFIX = "learning_health"
TRAIN_KEYS = ("reward_mean", "live_fraction", "budget_hit_rate", "output_tokens_mean", "output_tokens_p50", "live_groups",
              "episodes")
#: keys of train_metrics (one entry per minibatch) taken as mean and max; tokens are summed separately. OPTIM_ALGO_KEYS: extra required
#: keys depending on the row's algo (the trainer report's ``algo``; no such key = bpo, rows from before mis was added)
OPTIM_KEYS = ("loss", "grad_norm", "abs_delta_mean", "abs_delta_p95", "kl_k3", "masked_frac", "lr")
OPTIM_ALGO_KEYS = {"bpo": ("omega_mean", "omega_p95"), "mis": ("is_weight_mean",)}
LEARNER_OPTIONAL_KEYS = ("round", "micro_steps", "units", "accumulate_seconds", "temperature", "sequence_tokens",
                         "routing_equal_rate", "routing_mean_shared", "abs_delta_p50", "abs_delta_p99", "delta_mean", "omega_over_cap_frac",
                         "masked_frac_adv_pos", "masked_frac_adv_neg", "tokens_adv_pos", "tokens_adv_neg",
                         "ratio_outside_0.8_1.28", "ratio_outside_0.5_2.0", "ratio_outside_0.2_5.0", "pi_mass_outside_0.2_5.0",
                         "delta_p01", "delta_p99", "coef_kept_frac", "masked_below_frac", "masked_above_frac",
                         "entropy_mean", "entropy_p10", "entropy_p50", "entropy_p90", "entropy_mean_adv_pos",
                         "entropy_mean_adv_neg", "nll_mean")
PREPARE_KEYS = ("episodes", "episodes_kept", "groups", "groups_kept", "tokens", "loss_tokens")
#: the ``async`` section of async rows: always-numeric keys and nullable keys (an idle sync has no pause; sync not during rollout)
ASYNC_NUMBERS = ("rollout_server_version_start", "rollout_server_version_end", "gate_wait_seconds", "learner_wait_seconds",
                 "rollout_seconds", "sync_deferred_seconds", "ledger_landing_seconds")
ASYNC_NULLABLE = ("pause_seconds", "sync_seconds_into_rollout")
OFFLINE_DIR = re.compile(r"v(\d{6})")
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
#: some argument of /proc/<pid>/cmdline is orchestrator.py: python's argv[1], or, before launch_orchestrator.sh's unit execs, the
#: ``bash -c`` string (does not mistake test_orchestrator.py for it)
ORCHESTRATOR_ARG = re.compile(r"(^|[\s/])orchestrator\.py(\s|$)")


class SyncError(RuntimeError):
    """Something unparseable was read, or the resume state does not match: stop for a human to look, no guessing."""


class StopFlag:
    """Flag set by SIGTERM / SIGINT. No lock is taken in the signal handler: threading.Event.set() takes one, and a signal arriving
    exactly while Event.wait() holds the same lock would deadlock."""

    def __init__(self) -> None:
        self.raised = False

    def set(self) -> None:
        self.raised = True

    def is_set(self) -> bool:
        return self.raised

    def sleep(self, seconds: float) -> bool:
        """Sleep ``seconds`` or until the flag is set (checked every 0.2 s); returns the flag."""
        end = time.monotonic() + seconds
        while not self.raised and time.monotonic() < end:
            time.sleep(min(0.2, max(end - time.monotonic(), 0.0)))
        return self.raised


# --------------------------------------------------------------------------- reading

def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def parse_json(text: str | bytes, where: str):
    """JSON; NaN / Infinity raise right away (Python's json accepts them by default)."""
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as error:                       # json.JSONDecodeError is a ValueError
        raise SyncError(f"{where}: {error}") from error


def read_jsonl(path: Path, *, tail_ok: bool) -> list[tuple[str, dict]]:
    """Complete lines of an append-only JSONL → [(``file:line``, object)]. A last line without a newline = the writer is mid-write: if
    the writer is still alive (tail_ok) skip it this time, otherwise (the file was truncated) raise."""
    *lines, tail = path.read_bytes().split(b"\n")
    if tail and not tail_ok:
        raise SyncError(f"{path}: the last line ({len(tail)} bytes) has no newline and the orchestrator is gone: the file "
                        f"is truncated")
    rows = []
    for i, line in enumerate(lines, 1):
        where = f"{path}:{i}"
        doc = parse_json(line, where)
        if not isinstance(doc, dict):
            raise SyncError(f"{where}: a row must be a JSON object, not {type(doc).__name__}")
        rows.append((where, doc))
    return rows


def req(doc, key: str, where: str, kind: type | tuple[type, ...] = object):
    if not isinstance(doc, dict):
        raise SyncError(f"{where}: expected an object with {key!r}, got {type(doc).__name__}")
    if key not in doc:
        raise SyncError(f"{where}: missing {key!r} (has {sorted(doc)[:15]})")
    value = doc[key]
    if not isinstance(value, kind):
        raise SyncError(f"{where}: {key} is {value!r}, want {kind}")
    return value


def number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SyncError(f"{where}: {value!r} is not a number")
    return value


def num(doc, key: str, where: str) -> float:
    return number(req(doc, key, where), f"{where} {key}")


def counts(doc, key: str, where: str) -> dict[str, float]:
    """A ``{name: number}`` field (stops, drop reasons, dispatcher statistics…)."""
    return {k: number(v, f"{where} {key}.{k}") for k, v in req(doc, key, where, dict).items()}


def utc_seconds(doc, key: str, where: str) -> float:
    text = req(doc, key, where, str)
    try:
        return float(calendar.timegm(time.strptime(text, UTC_FORMAT)))
    except ValueError as error:
        raise SyncError(f"{where}: {key} {text!r} is not {UTC_FORMAT}") from error


# --------------------------------------------------------------------------- row sections (pure functions of one row)

def train_section(row: dict, where: str, unit: str) -> dict:
    """``unit``: a group is a task (tasks) for sync / async and a group dir (groups) for stream."""
    train, w = req(row, "train", where, dict), f"{where} train"
    out = {f"train/{k}": num(train, k, w) for k in TRAIN_KEYS + (unit,)}
    out.update({f"train/stops/{k}": v for k, v in counts(train, "stops", w).items()})
    return out


def optim_keys(step: dict, where: str) -> tuple[str, ...]:
    """Required optim keys of a trainer report: OPTIM_KEYS plus OPTIM_ALGO_KEYS of its algo."""
    algo = step.get("algo", "bpo") if isinstance(step, dict) else None
    if algo not in OPTIM_ALGO_KEYS:
        raise SyncError(f"{where}: algo {algo!r} is not one of {sorted(OPTIM_ALGO_KEYS)}")
    return OPTIM_KEYS + OPTIM_ALGO_KEYS[algo]


def optim_section(row: dict, where: str) -> dict:
    steps, n = req(row, "train_metrics", where, list), req(row, "minibatches", where, int)
    if len(steps) != n or n == 0:
        raise SyncError(f"{where}: {len(steps)} train_metrics entries for minibatches {n}")
    places = [f"{where} train_metrics[{i}]" for i in range(n)]
    keys = {optim_keys(s, w) for s, w in zip(steps, places)}
    if len(keys) != 1:
        raise SyncError(f"{where}: the train_metrics entries mix loss algorithms")
    out = {}
    for key in keys.pop():
        values = [num(s, key, w) for s, w in zip(steps, places)]
        out[f"optim/{key}_mean"] = statistics.fmean(values)
        out[f"optim/{key}_max"] = max(values)
    out["optim/tokens"] = sum(num(s, "tokens", w) for s, w in zip(steps, places))
    out["optim/minibatches"] = n
    out["optim/trainer_step"] = num(steps[-1], "step", places[-1])
    return out


def lag_histogram(prep: dict, key: str, where: str, low: int, high: int) -> dict[int, float]:
    """prepare's lag histogram ``{"lag": count}`` (sparse: lags that did not occur are absent); lags must be in [low, high]."""
    hist = {}
    for lag, n in counts(prep, key, where).items():
        if not (lag.isdigit() and low <= int(lag) <= high):
            raise SyncError(f"{where} {key}: lag {lag!r} is outside [{low}, {high}]")
        hist[int(lag)] = n
    return hist


def staleness_section(prep: dict, where: str, version: int) -> dict:
    """A stream batch: shares of loss tokens and episodes by lag (V − the token's version), every lag 0..max_staleness listed (absent =
    0), mean lag, number of episodes dropped as stale (lag > max_staleness)."""
    if req(prep, "current_version", where, int) != version:
        raise SyncError(f"{where}: current_version {prep['current_version']} != the row's version {version}")
    window = req(prep, "max_staleness", where, int)
    out = {}
    for key, name in (("staleness_loss_tokens", "loss_tokens"), ("staleness_episodes", "episodes")):
        hist = lag_histogram(prep, key, where, 0, window)
        total = sum(hist.values())
        if total <= 0:
            raise SyncError(f"{where} {key}: empty histogram {hist}")
        for lag in range(window + 1):
            out[f"prepare/staleness_{name}/lag{lag}"] = (hist[lag] if lag in hist else 0) / total
        out[f"prepare/lag_mean_{name}"] = sum(lag * n for lag, n in hist.items()) / total
    out["prepare/stale_episodes"] = sum(lag_histogram(prep, "staleness_episodes_stale", where, window + 1, sys.maxsize).values())
    return out


def prepare_section(row: dict, where: str, version: int, stream: bool) -> dict:
    prep, w = req(row, "prepare", where, dict), f"{where} prepare"
    out = {f"prepare/{k}": num(prep, k, w) for k in PREPARE_KEYS}
    for kind in ("dropped_episodes", "dropped_groups"):
        out.update({f"prepare/{kind}/{reason}": n for reason, n in counts(prep, kind, w).items()})
    out.update({f"prepare/routing/{state}": n for state, n in counts(prep, "routing_positions_joined", w).items()})
    if stream:
        out.update(staleness_section(prep, w, version))
    return out


def time_section(row: dict, where: str, first: tuple[str, dict]) -> dict:
    """Each item of ``seconds``, plus wall-clock hours from the start of the first row to the end of this one."""
    out = {f"time/{k}": number(v, f"{where} seconds.{k}") for k, v in req(row, "seconds", where, dict).items()}
    began = utc_seconds(first[1], "started_utc", first[0])
    out["time/wall_hours"] = (utc_seconds(row, "finished_utc", where) - began) / 3600
    return out


def round_rollout_section(row: dict, where: str) -> dict:
    """sync / async: this round's total output tokens ÷ rollout seconds."""
    train = req(row, "train", where, dict)
    tokens = num(train, "output_tokens_mean", f"{where} train") * num(train, "episodes", f"{where} train")
    seconds = num(req(row, "seconds", where, dict), "rollout", f"{where} seconds")
    if seconds <= 0:
        raise SyncError(f"{where}: seconds.rollout {seconds} is not positive")
    return {"rollout/output_tokens": tokens, "rollout/seconds": seconds, "rollout/tokens_per_second": tokens / seconds}


def async_section(row: dict, where: str) -> dict:
    sec, w = req(row, "async", where, dict), f"{where} async"
    out = {f"async/{k}": num(sec, k, w) for k in ASYNC_NUMBERS}
    for key in ASYNC_NULLABLE:
        value = req(sec, key, w)
        if value is not None:
            out[f"async/{key}"] = number(value, f"{w} {key}")
    # tail_overlap: rows written before tail overlap existed (10-08 async rounds 0-13) do not have the key; null = this
    # round did not start in the previous round's tail
    overlap = sec["tail_overlap"] if "tail_overlap" in sec else None
    out["async/tail_overlap"] = 0 if overlap is None else 1
    if overlap is not None:
        out["async/tail_seconds_before_previous_end"] = num(overlap, "seconds_before_previous_end", f"{w} tail_overlap")
        out["async/tail_previous_remaining_episodes"] = num(overlap, "previous_remaining_episodes", f"{w} tail_overlap")
    return out


def round_sync_section(row: dict, where: str, pipeline: str) -> dict:
    sync, w = req(row, "sync", where, dict), f"{where} sync"
    if pipeline == "sync":
        if "mode" in sync:
            raise SyncError(f"{w}: a v1 sync row has no mode (it only syncs an idle server, SPEC section 4), got "
                            f"{sync['mode']!r}")
        return {"sync/in_place": 0}
    mode, pause = req(sync, "mode", w, str), req(sync, "pause_seconds", w)
    if (mode, pause is None) not in (("idle", True), ("in_place", False)):
        raise SyncError(f"{w}: mode {mode!r} with pause_seconds {pause!r} (idle has none, in_place has one)")
    if "active_rounds" in sync:
        active = req(sync, "active_rounds", w, list)
    else:
        # rows written before tail overlap (10-08 async rounds 0-13): at most one round was live, the one in during_round
        during = req(sync, "during_round", w, (int, type(None)))
        active = [] if during is None else [during]
    if bool(active) != (mode == "in_place"):
        raise SyncError(f"{w}: mode {mode!r} with rounds in flight {active}")
    out = {"sync/in_place": int(mode == "in_place"), "sync/rounds_in_flight": len(active),
           "sync/deferred_seconds": num(sync, "deferred_seconds", w)}
    if pause is not None:
        out["sync/pause_seconds"] = number(pause, f"{w} pause_seconds")
    out.update(async_section(row, where))
    return out


def stream_section(row: dict, where: str) -> dict:
    sync, w = req(row, "sync", where, dict), f"{where} sync"
    if req(sync, "mode", w, str) != "in_place":
        raise SyncError(f"{w}: mode {sync['mode']!r}: a stream sync is always in_place (SPEC section 9)")
    out = {"train/live_episodes": num(row, "live_episodes", where), "train/total_episodes": num(row, "total_episodes", where),
           "train/trained_live_groups": num(row, "live_groups", where),
           "sync/in_place": 1, "sync/pause_seconds": num(sync, "pause_seconds", w),
           "sync/groups_in_flight": len(req(sync, "groups_in_flight", w, list))}
    out.update({f"dispatcher/{k}": v for k, v in counts(row, "dispatcher", where).items()})
    out.update({f"pool/{k}": v for k, v in counts(row, "pool_after", where).items()})
    return out


def quarantine_section(rows: list, idx: int, quarantine: list[tuple[str, dict]], where: str) -> dict:
    """stream: quarantine.jsonl (groups the orchestrator quarantined, one row per group, with unix time): the cumulative count up to
    the end of this row, and the count between the end of the previous row (first row: from the start) and the end of this one."""
    end = utc_seconds(rows[idx][1], "finished_utc", where)
    start = utc_seconds(rows[idx - 1][1], "finished_utc", rows[idx - 1][0]) if idx else -math.inf
    times = [num(q, "unix", qw) for qw, q in quarantine]
    return {"train/quarantined_groups": sum(t <= end for t in times),
            "train/quarantined_groups_batch": sum(start < t <= end for t in times)}


def arc3_section(row: dict, where: str, completed: dict[int, tuple[str, dict]], baseline_of, g2: bool = False,
                 kappa: float = 1.0, rank: dict | None = None, note: bool = False) -> dict:
    """A stream row of arc3_level: each gid in row.groups → the episodes of its completed record in groups.jsonl.
    ``baseline_of(task)`` = the stub's baseline_actions. Cleared = score 1 (anything but 0 / 1 raises); cleared games have actions ≥ 1
    and a numeric tokens_at_clear, uncleared ones None.
    ``g2`` (kind arc3_g2seed): score = 1[cleared] × min(1, (κ·baseline / actions)²), cleared = score > 0, and it must equal that
    formula (otherwise raise); ``kappa`` = the harness config's efficiency_free_fraction (no such key = 1; 0.7 for v5k 10-26).
    ``rank`` = the prepare section when prepare.arc3_reward is rank: ``arc3/reward_mean`` becomes the R recomputed per group
    (orchestrator.rank_rewards, dispatch version from groups.jsonl's server_version_launch; rank and clear share are computed over the
    whole group, prepare only over the episodes it keeps, so it may differ slightly from prepare's R). ``note``
    (prepare.arc3_reward note, v8): ``arc3/reward_mean`` is groups.jsonl's ``note_rewards`` (the training R the orchestrator computes
    from judge.json, one per game). ``arc3/score_mean`` is always score.
    Every kind logs stop shares (``arc3/stop/<stop>``; budget = hit the output budget) and mean output tokens over all games."""
    cleared_scores, ratio, at_clear, failed_tokens, kinds, n = [], [], [], [], {"all": 0, "none": 0, "mixed": 0}, 0
    stops, all_tokens, rewards, scores = collections.Counter(), [], [], []
    for i, g in enumerate(req(row, "groups", where, list)):
        gw = f"{where} groups[{i}]"
        gid = req(g, "gid", gw, int)
        if gid not in completed:
            raise SyncError(f"{gw}: gid {gid} has no completed record in groups.jsonl")
        rw, rec = completed[gid]
        base = baseline_of(req(rec, "task", rw, str))
        cleared = 0
        episodes = req(rec, "episodes", rw, list)
        for k, e in enumerate(episodes):
            ew = f"{rw} episodes[{k}]"
            score, n = num(e, "score", ew), n + 1
            stops[req(e, "stop", ew, str)] += 1
            all_tokens.append(num(e, "tokens", ew))
            scores.append(score)
            if score == 1.0 or (g2 and score > 0):
                actions = num(e, "actions", ew)
                if actions < 1:
                    raise SyncError(f"{ew}: a cleared episode with {actions} actions")
                if g2 and abs(score - min(1.0, (kappa * base / actions) ** 2)) > 1e-9:
                    raise SyncError(f"{ew}: g2seed score {score} != min(1, ({kappa} x {base} / {actions})^2)")
                cleared += 1
                cleared_scores.append(min((base / actions) ** 2, ARC3_LEVEL_SCORE_CAP))
                ratio.append(actions / base)
                at_clear.append(num(e, "tokens_at_clear", ew))
            elif score == 0.0:
                failed_tokens.append(num(e, "tokens", ew))
            else:
                raise SyncError(f"{ew}: arc3 score {score} is not 0 or 1" + (" (g2seed: not in [0, 1])" if g2 else ""))
        kinds["all" if cleared == len(episodes) else "none" if cleared == 0 else "mixed"] += 1
        launch = None if rank is None else req(rec, "server_version_launch", rw, int)
        if note:
            got = req(rec, "note_rewards", rw, list)
            if len(got) != len(episodes):
                raise SyncError(f"{rw}: {len(got)} note_rewards for {len(episodes)} episodes")
            rewards.extend(number(x, f"{rw} note_rewards[{k}]") for k, x in enumerate(got))
            continue
        rewards.extend(scores[len(rewards):] if rank is None else
                       rank_rewards(episodes, rank, launch) if rank["arc3_reward"] == "rank" else
                       hinge_rewards(episodes, rank, launch, req(rec, "task", rw, str), req(rec, "gid", rw, int)))
    if n == 0:
        raise SyncError(f"{where}: no episode in the batch's groups")
    out = {"arc3/clear_rate": len(ratio) / n, "arc3/level_score_mean": sum(cleared_scores) / n,
           "arc3/cleared_episodes": len(ratio), **{f"arc3/groups_{k}_cleared": v for k, v in kinds.items()},
           "arc3/output_tokens_mean": statistics.fmean(all_tokens), "arc3/reward_mean": statistics.fmean(rewards),
           "arc3/score_mean": statistics.fmean(scores),
           **{f"arc3/stop/{k}": v / n for k, v in sorted(stops.items())}}
    if ratio:
        out.update({"arc3/cleared/level_score_mean": statistics.fmean(cleared_scores),
                    "arc3/cleared/actions_over_baseline_mean": statistics.fmean(ratio),
                    "arc3/cleared/actions_over_baseline_p50": statistics.median(ratio),
                    "arc3/cleared/tokens_at_clear_mean": statistics.fmean(at_clear),
                    "arc3/cleared/tokens_at_clear_p50": statistics.median(at_clear)})
    if failed_tokens:
        out["arc3/failed/output_tokens_mean"] = statistics.fmean(failed_tokens)
    return out


def arc3_reward_section(prep: dict, where: str) -> dict:
    """prepare's ``arc3_reward`` summary (for group_ready rows the orchestrator merges it across packets; rows from before 10-20 have
    none = {}). When a share's denominator is 0 (no group kept) prepare records None, and that item is not logged here."""
    a = prep.get("arc3_reward")
    if a is None:
        return {}
    w = f"{where} arc3_reward"
    mode = req(a, "mode", w, str)
    if mode not in ARC3_REWARD_MODE_COUNTS:
        raise SyncError(f"{w}: mode {mode!r} is not one of {sorted(ARC3_REWARD_MODE_COUNTS)}")
    out = {f"reward/{k}": num(a, k, w) for k in ARC3_REWARD_COUNTS + ARC3_REWARD_MODE_COUNTS[mode]}
    share = req(a, "gradient_share", w, dict)
    for key, name in (("unweighted", "eff_share_of_advantage"), ("loss_token_weighted", "eff_share_of_advantage_tokens")):
        if req(share, key, f"{w} gradient_share", (int, float, type(None))) is not None:
            out[f"reward/{name}"] = num(share, key, f"{w} gradient_share")
    if mode == "c0" and req(a, "token_share", w, (int, float, type(None))) is not None:
        out["reward/token_share_of_eff"] = num(a, "token_share", w)
    hist = counts(a, "p_hat_histogram", w)
    total = sum(hist.values())
    if total:
        out["reward/p_hat_mean"] = sum(float(p) * c for p, c in hist.items()) / total
    return out


def offline_heldout_section(version: int, where: str, doc: dict) -> dict:
    """heldout_offline.py's summary.json: summary (this version) + compare_base (paired per task against base)."""
    summary, compare = req(doc, "summary", where, dict), req(doc, "compare_base", where, dict)
    ws, wc = f"{where} summary", f"{where} compare_base"
    out = {"heldout/version": version, "heldout/offline": 1,
           "heldout/reward": num(summary, "reward_mean", ws), "heldout/live_fraction": num(summary, "live_fraction", ws),
           "heldout/budget_hit": num(summary, "budget_hit_rate", ws),
           "heldout/output_tokens": num(summary, "output_tokens_mean", ws),
           "heldout/base_samples": number(req(compare, "samples", wc, list)[0], f"{wc} samples[0]")}
    for name in ("reward", "budget_hit", "output_tokens"):
        part, w = req(compare, name, wc, dict), f"{wc} {name}"
        out[f"heldout/{name}_diff"], out[f"heldout/{name}_se"] = num(part, "diff", w), num(part, "se", w)
        z = req(part, "z", w)
        if z is not None:                              # null when se == 0
            out[f"heldout/{name}_z"] = number(z, f"{w} z")
    reward = compare["reward"]
    out.update({"heldout/base_reward": num(reward, "a", f"{wc} reward"),
                "heldout/tasks_up": num(reward, "tasks_up", f"{wc} reward"),
                "heldout/tasks_down": num(reward, "tasks_down", f"{wc} reward")})
    return out


# --------------------------------------------------------------------------- process / locks / config

def orchestrator_pid(run_root: Path) -> int:
    """The PID in ``<run_root>/orchestrator.lock`` (written by the orchestrator after it gets the flock). Read only: not opened for
    writing, no flock."""
    lock = run_root / "orchestrator.lock"
    if not lock.is_file():
        raise SyncError(f"{lock} does not exist: no orchestrator has run in {run_root}")
    text = lock.read_text().strip()
    if not text.isdigit():
        raise SyncError(f"{lock} holds {text!r}, not a PID (an orchestrator is starting, or a refused second start "
                        f"truncated it): cannot tell whether the orchestrator lives; retry, or pass --orchestrator-pid")
    return int(text)


def process_is_orchestrator(pid: int) -> bool:
    """/proc/<pid>/cmdline contains orchestrator.py. Empty cmdline: mid-exec (the new argv is not up yet, as at the moment
    launch_orchestrator.sh's unit execs from bash into python), a zombie, or a kernel thread; read again a moment later, and if it is
    still empty after 1 s it is not a live orchestrator."""
    for _ in range(20):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):  # no such process, or it exited while we read
            return False
        if raw:
            return any(ORCHESTRATOR_ARG.search(arg) for arg in raw.decode(errors="replace").split("\0"))
        time.sleep(0.05)
    return False


def hold_run_root(run_root: Path, wait_seconds: float = 0.0) -> int:
    """flock the run root dir (not orchestrator.lock): only one sidecar per run, otherwise both log = duplicates. Returns the fd.
    ``wait_seconds``: how long to wait at most while the lock is held (retrying every second). On an orchestrator restart the new
    sidecar overlaps the old one, which reads once more after its orchestrator ends and releases the lock only after the wandb flush
    (10-30 arc2-hq741b: the new sidecar failed immediately and wandb logging was down for 70 minutes unnoticed)."""
    fd = os.open(run_root, os.O_RDONLY | os.O_DIRECTORY)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError as error:
            if time.monotonic() < deadline:
                time.sleep(1.0)
                continue
            os.close(fd)
            raise SyncError(f"{run_root}: another wandb_sync is attached to this run root (it holds a flock on the "
                            f"directory{f', still after {wait_seconds:g}s' if wait_seconds else ''})") from error


def find_config(run_root: Path) -> Path:
    """The orchestrator config in ``configs/`` whose expanded run_root is this one (harness configs are also there: they have no
    orchestrator top-level keys and are skipped)."""
    hits = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        raw = parse_json(path.read_text(), str(path))
        if not TOP_KEYS <= set(raw):
            continue
        if cfg_path(_expand(raw["run_root"], f"{path.name} run_root")).resolve() == run_root:
            hits.append(path)
    if len(hits) != 1:
        raise SyncError(f"{run_root}: {len(hits)} config(s) in {CONFIG_DIR} have it as run_root "
                        f"{[p.name for p in hits]}: pass --config")
    return hits[0]


def new_run_id() -> str:
    """Shape of a wandb run id (8 base-36 chars); wandb 0.28 has no public generator (wandb.util.generate_id was moved away)."""
    return "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))


def default_wandb_dir() -> Path:
    if "ARC_RUNS" not in os.environ:
        raise SyncError("ARC_RUNS is not set (source env/env.sh) and --wandb-dir was not given")
    return Path(os.environ["ARC_RUNS"]) / "qwen38fn_rl" / "wandb"


# --------------------------------------------------------------------------- the sidecar

class Syncer:
    def __init__(self, args: argparse.Namespace, cfg: dict, config_path: Path, stop: StopFlag) -> None:
        self.args, self.cfg, self.config_path, self.stop = args, cfg, config_path, stop
        self.root: Path = args.run_root
        self.pipeline = cfg["pipeline"]
        self.stream = self.pipeline == "stream"
        self.history_name, self.key = HISTORY[self.pipeline]
        self.history = self.root / self.history_name
        self.groups_path = self.root / "groups.jsonl"
        self.quarantine_path = self.root / QUARANTINE_FILE
        harness = json.loads(cfg_path(cfg["harness"]["config"]).read_text())
        self.arc3 = self.stream and harness.get("kind") in (ARC3_LEVEL_KIND, ARC3_G2_KIND)
        self.g2 = harness.get("kind") == ARC3_G2_KIND
        # optional key of run_arc3_level.py (default 1 = v5's reward): checking score needs the same κ (v5k 10-26 15:59 stopped on the
        # very first row because it was missing)
        self.kappa = harness.get("efficiency_free_fraction", 1.0)
        # prepare.arc3_reward rank: arc3/reward_mean logs the training R (in-group rank discount), not score
        # hinge (v7) likewise: recomputed per group with orchestrator.hinge_rewards
        self.rank = cfg["prepare"] if cfg["prepare"].get("arc3_reward") in ("rank", "hinge") else None
        # note (v8): arc3/reward_mean logs groups.jsonl's note_rewards
        self.note = cfg["prepare"].get("arc3_reward") == "note"
        self.baselines: dict[str, float] = {}
        self.trainer_path = cfg_path(cfg["trainer"]["endpoint_file"]).resolve().parent / "train_log.jsonl"
        self.cursor_path = self.root / CURSOR_NAME
        self.base: dict | None = None                 # baseline per-task stats, read at the first in-loop held-out
        self.run = None
        self.cursor: dict = {}

    # ---- cursor / wandb run
    def identity(self) -> dict:
        return {"format": CURSOR_FORMAT, "run_root": str(self.root), "project": self.args.project, "name": self.args.name,
                "history": self.history_name}

    def load_cursor(self) -> dict | None:
        if not self.cursor_path.exists():
            return None
        doc = parse_json(self.cursor_path.read_text(), str(self.cursor_path))
        where = str(self.cursor_path)
        differs = {k: (req(doc, k, where), v) for k, v in self.identity().items() if req(doc, k, where) != v}
        if differs:
            raise SyncError(f"{self.cursor_path}: wandb run {req(doc, 'run_id', where, str)} was recorded with different "
                            f"(recorded, now): {differs}. Use the recorded project/name, or move the file aside to start "
                            f"a new wandb run")
        for key, kind in (("run_id", str), ("rows_logged", int), ("wandb_steps", int), ("heldout_logged", dict)):
            req(doc, key, where, kind)
        req(doc, "groups_lines", where, int if self.stream else type(None))
        for kind in ("inloop", "offline"):
            req(doc["heldout_logged"], kind, f"{where} heldout_logged", list)
        # v1 cursors predate per-step logging; backfill only the new learner namespace.
        if "trainer_lines" not in doc and "trainer_step" not in doc:
            doc.update(trainer_lines=0, trainer_step=0)
        for key in ("trainer_lines", "trainer_step"):
            value = req(doc, key, where, int)
            if isinstance(value, bool) or value < 0:
                raise SyncError(f"{where}: {key} {value!r} must be a nonnegative integer")
        if "trainer_path" not in doc:
            # Earlier per-step cursors only read root/trainer; original v1 cursors have no logged trainer lines.
            doc["trainer_path"] = str(self.root / "trainer" / "train_log.jsonl") if doc["trainer_lines"] else str(self.trainer_path)
        recorded_path = req(doc, "trainer_path", where, str)
        if doc["trainer_lines"] and recorded_path != str(self.trainer_path):
            raise SyncError(f"{where}: trainer log source changed from {recorded_path!r} to {str(self.trainer_path)!r} "
                            f"after {doc['trainer_lines']} line(s) were logged")
        doc["trainer_path"] = str(self.trainer_path)
        if "learning_health_receipt_id" not in doc:
            doc["learning_health_receipt_id"] = None
        req(doc, "learning_health_receipt_id", where, (str, type(None)))
        return doc

    def save_cursor(self) -> None:
        self.cursor["updated_utc"] = utc_now()
        atomic_write(self.cursor_path, json.dumps(self.cursor, indent=1) + "\n")

    def open(self) -> None:
        cursor = self.load_cursor()
        fresh = cursor is None
        run_id = new_run_id() if fresh else cursor["run_id"]
        wandb_dir = self.args.wandb_dir if self.args.wandb_dir is not None else default_wandb_dir()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        provenance = {"run_root": str(self.root), "config_path": str(self.config_path.resolve()),
                      "pipeline": self.pipeline, "history": self.history_name, "run_config": self.cfg}
        run = wandb.init(project=self.args.project, name=self.args.name, id=run_id, resume="allow", dir=str(wandb_dir),
                         tags=[self.pipeline], allow_val_change=True,
                         config=provenance,
                         # System stats stay on: they are the run's only periodic upload between batches. With
                         # them disabled, an ~8-minute optimizer step left W&B without a heartbeat and it marked
                         # the run crashed (and emailed "run failed") until the next batch arrived.
                         settings=wandb.Settings(x_stats_sampling_interval=60.0, console="off"))
        self.run = run
        if run.id != run_id:
            raise SyncError(f"wandb.init returned run {run.id}, asked for {run_id}")
        if fresh:
            cursor = {**self.identity(), "run_id": run.id, "entity": run.entity, "url": run.url, "rows_logged": 0,
                      "groups_lines": 0 if self.stream else None, "heldout_logged": {"inloop": [], "offline": []},
                      "wandb_steps": 0, "trainer_lines": 0, "trainer_step": 0, "trainer_path": str(self.trainer_path),
                      "learning_health_receipt_id": None}
        elif run.entity != req(cursor, "entity", str(self.cursor_path)):
            raise SyncError(f"{self.cursor_path}: run {run_id} was logged under entity {cursor['entity']!r}, wandb now "
                            f"resumed it under {run.entity!r}")
        # starting_step: the step the next log gets (0 for a new run, the server's history length on resume). Not run.step:
        # wandb 0.28 answers 0 for it right after a resume
        if run.starting_step != cursor["wandb_steps"]:
            raise SyncError(f"wandb run {run_id} resumed at step {run.starting_step}, {self.cursor_path} recorded "
                            f"{cursor['wandb_steps']} "
                            f"log calls: a log was not recorded in the cursor (killed between the two?) or the run was "
                            f"deleted. Compare the run with the cursor and fix the cursor by hand (or move it aside to "
                            f"start a new wandb run)")
        # The SDK installs the server's config on resume after init's config, so explicitly refresh current provenance.
        run.config.update(provenance, allow_val_change=True)
        self.cursor = cursor
        self.save_cursor()
        run.define_metric("version")
        for prefix in VERSION_PREFIXES:
            run.define_metric(f"{prefix}/*", step_metric="version")
        run.define_metric(f"{HELDOUT_PREFIX}/version")
        run.define_metric(f"{HELDOUT_PREFIX}/*", step_metric=f"{HELDOUT_PREFIX}/version")
        run.define_metric(f"{LEARNER_PREFIX}/step")
        run.define_metric(f"{LEARNER_PREFIX}/*", step_metric=f"{LEARNER_PREFIX}/step")
        run.define_metric(f"{HEALTH_PREFIX}/optimizer_step")
        run.define_metric(f"{HEALTH_PREFIX}/*", step_metric=f"{HEALTH_PREFIX}/optimizer_step")
        self.say(f"wandb run {run.id} ({'new' if fresh else 'resumed'}) {run.url}; "
                 f"{self.cursor['rows_logged']} {self.key}(s) and offline held-out {self.cursor['heldout_logged']['offline']} "
                 f"already logged")

    def log(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if key != "version" and key.split("/", 1)[0] not in VERSION_PREFIXES + (HELDOUT_PREFIX, LEARNER_PREFIX, HEALTH_PREFIX):
                raise SyncError(f"metric {key!r} has no known prefix")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SyncError(f"metric {key!r} = {value!r} is not a finite number")
        self.run.log(metrics)
        self.cursor["wandb_steps"] += 1

    def say(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] wandb_sync {self.root.name}: {message}", flush=True)

    # ---- liveness
    def orchestrator_alive(self) -> bool:
        pid = self.args.orchestrator_pid if self.args.orchestrator_pid is not None else orchestrator_pid(self.root)
        return process_is_orchestrator(pid)

    # ---- files
    def read_trainer(self, tail_ok: bool) -> list[tuple[str, dict]]:
        logged = self.cursor["trainer_lines"]
        if not self.trainer_path.exists():
            if logged:
                raise SyncError(f"{self.trainer_path} is gone, {logged} line(s) were logged from it")
            return []                                    # normal before trainer startup
        rows = read_jsonl(self.trainer_path, tail_ok=tail_ok)
        if len(rows) < logged:
            raise SyncError(f"{self.trainer_path} has {len(rows)} line(s), {logged} were logged: it is append-only")
        previous = 0
        for where, row in rows:
            step = req(row, "step", where, int)
            if isinstance(step, bool) or step <= previous:
                raise SyncError(f"{where}: step {step!r} must exceed the previous optimizer step {previous}")
            previous = step
        recorded = rows[logged - 1][1]["step"] if logged else 0
        if recorded != self.cursor["trainer_step"]:
            raise SyncError(f"{self.trainer_path}: optimizer step {recorded} at cursor line {logged}, cursor recorded "
                            f"{self.cursor['trainer_step']}: it is append-only")
        return rows

    def learner_metrics(self, row: dict, where: str) -> dict:
        # /train rows carry the minibatch index; a stream.group_ready /step row has none (it sums its packets)
        keys = ("step", "weight_version", "tokens", "seconds") + optim_keys(row, where)
        out = {f"learner/{key}": num(row, key, where) for key in keys}
        if "minibatch" in row or "update_id" not in row:
            out["learner/minibatch"] = num(row, "minibatch", where)
        out.update({f"learner/{key}": num(row, key, where) for key in LEARNER_OPTIONAL_KEYS if key in row})
        return out

    def read_rows(self, tail_ok: bool) -> list[tuple[str, dict]]:
        logged = self.cursor["rows_logged"]
        if not self.history.exists():
            if logged:
                raise SyncError(f"{self.history} is gone, {logged} row(s) were logged from it")
            return []
        rows = read_jsonl(self.history, tail_ok=tail_ok)
        for i, (where, row) in enumerate(rows):
            if req(row, self.key, where, int) != i:
                raise SyncError(f"{where}: {self.key} {row[self.key]} at row {i}: rows must be 0..n-1 in order")
        if len(rows) < logged:
            raise SyncError(f"{self.history} has {len(rows)} row(s), {logged} were logged: it is append-only")
        return rows

    def read_groups(self, tail_ok: bool, rows: list) -> list[tuple[str, dict]]:
        if not self.groups_path.exists():
            if rows:
                raise SyncError(f"{self.groups_path} does not exist, {self.history} has {len(rows)} row(s)")
            return []
        groups = read_jsonl(self.groups_path, tail_ok=tail_ok)
        if len(groups) < self.cursor["groups_lines"]:
            raise SyncError(f"{self.groups_path} has {len(groups)} line(s), {self.cursor['groups_lines']} were read: it is "
                            f"append-only")
        return groups

    def read_quarantine(self, tail_ok: bool) -> list[tuple[str, dict]]:
        """stream's quarantine.jsonl (created only at the orchestrator's first quarantine: missing = nothing quarantined yet); each row
        must have gid, stage, unix."""
        if not self.quarantine_path.exists():
            return []
        rows = read_jsonl(self.quarantine_path, tail_ok=tail_ok)
        for where, row in rows:
            req(row, "gid", where, int)
            num(row, "unix", where)
            if req(row, "stage", where, str) not in QUARANTINE_STAGES:
                raise SyncError(f"{where}: stage {row['stage']!r} is not one of {QUARANTINE_STAGES}")
        return rows

    # ---- one row
    def row_metrics(self, rows: list[tuple[str, dict]], idx: int, groups: list | None,
                    quarantine: list | None) -> tuple[dict, int | None]:
        """wandb metrics of row idx, and the line reached in stream's groups.jsonl (None for other pipelines). ``quarantine``: stream's
        quarantine.jsonl (None for other pipelines)."""
        where, row = rows[idx]
        where = f"{where} ({self.key} {idx})"
        if self.pipeline == "sync":
            if "pipeline" in row or "async" in row:
                raise SyncError(f"{where}: an async/stream row in a sync run ({self.config_path} says pipeline sync)")
        elif req(row, "pipeline", where, str) != self.pipeline:
            raise SyncError(f"{where}: pipeline {row['pipeline']!r}, {self.config_path} says {self.pipeline!r}")
        for key, want in (("weight_version_in", idx), ("weight_version_out", idx + 1)):
            if req(row, key, where, str) != str(want):
                raise SyncError(f"{where}: {key} {row[key]!r}, want {str(want)!r} ({self.key} {idx} trains version {idx} "
                                f"into {idx + 1})")
        out = {"version": idx}
        out.update(train_section(row, where, "groups" if self.stream else "tasks"))
        out.update(optim_section(row, where))
        out.update(prepare_section(row, where, idx, self.stream))
        out.update(time_section(row, where, rows[0]))
        if self.stream:
            rollout, groups_end = self.stream_rollout(rows, idx, groups, where)
            out.update(rollout)
            out.update(stream_section(row, where))
            out.update(quarantine_section(rows, idx, quarantine, where))
            if self.arc3:
                completed = {req(g, "gid", gw, int): (gw, g) for gw, g in groups if g.get("event") == "completed"}
                out.update(arc3_section(row, where, completed, self.baseline_actions, g2=self.g2, kappa=self.kappa,
                                        rank=self.rank, note=self.note))
                out.update(arc3_reward_section(req(row, "prepare", where, dict), f"{where} prepare"))
            return out, groups_end
        out.update(round_rollout_section(row, where))
        out.update(round_sync_section(row, where, self.pipeline))
        heldout = req(row, "heldout", where, (dict, type(None)))   # null on rounds without a held-out
        if heldout is not None:
            out.update(self.inloop_heldout(heldout, idx, f"{where} heldout"))
        return out, None

    def stream_rollout(self, rows: list, idx: int, groups: list, where: str) -> tuple[dict, int]:
        """Generated tokens per second: the summed episode tokens of groups completed between the end of the previous row and the end of
        this one (groups.jsonl is appended in completion order) ÷ those seconds; the first row counts from the launch time of the first
        group it consumed. Reading continues from the cursor: a group written exactly across a row's end time goes to the next row,
        never missed or counted twice."""
        end = utc_seconds(rows[idx][1], "finished_utc", where)
        line, tokens, completed, first_launch = self.cursor["groups_lines"], 0, 0, None
        while line < len(groups):
            gwhere, group = groups[line]
            event = req(group, "event", gwhere, str)
            if event == "completed":
                if num(group, "completed_unix", gwhere) > end:
                    break
                episodes = req(group, "episodes", gwhere, list)
                tokens += sum(num(e, "tokens", f"{gwhere} episodes[{i}]") for i, e in enumerate(episodes))
                launched = utc_seconds(group, "launched_utc", gwhere)
                first_launch = launched if first_launch is None else min(first_launch, launched)
                completed += 1
            elif event not in ("abandoned", "set_aside"):  # never trained: a pooled group a resume dropped, or a finished
                # group whose note judge kept failing (orchestrator --set-aside-judge-groups)
                raise SyncError(f"{gwhere}: unknown event {event!r}")
            line += 1
        if idx > 0:
            start = utc_seconds(rows[idx - 1][1], "finished_utc", rows[idx - 1][0])
        elif first_launch is None:
            raise SyncError(f"{where}: no group in {self.groups_path} completed before the first batch finished")
        else:
            start = first_launch
        seconds = end - start
        if seconds <= 0:
            raise SyncError(f"{where}: rollout window {start}..{end} is not positive")
        return {"rollout/output_tokens": tokens, "rollout/window_seconds": seconds,
                "rollout/tokens_per_second": tokens / seconds, "rollout/groups_completed": completed}, line

    def baseline_actions(self, task: str) -> float:
        """The stub's baseline_actions (config tasks.train.dir/<task>.json), read once and cached."""
        if task not in self.baselines:
            path = cfg_path(self.cfg["tasks"]["train"]["dir"]) / f"{task}.json"
            base = num(parse_json(path.read_text(), str(path)), "baseline_actions", str(path))
            if base < 1:
                raise SyncError(f"{path}: baseline_actions {base}")
            self.baselines[task] = base
        return self.baselines[task]

    # ---- held-out
    def baseline(self, tasks: list[str], where: str) -> dict:
        if self.base is None:
            dirs = self.args.baseline if self.args.baseline is not None else base24_dirs()
            episodes = read_episodes(dirs, tasks)
            self.base = {"tasks": tasks, "samples": len(dirs), "reward": per_task(episodes, lambda e: e["score"]),
                         "budget_hit": statistics.fmean(per_task(episodes, lambda e: float(e["stop"] == "budget")).values()),
                         "output_tokens": statistics.fmean(per_task(episodes, lambda e: e["tokens"]).values())}
        elif self.base["tasks"] != tasks:
            raise SyncError(f"{where}: held-out tasks differ from the first held-out's ({len(tasks)} vs "
                            f"{len(self.base['tasks'])}): one baseline cannot pair both")
        return self.base

    def inloop_heldout(self, heldout: dict, version: int, where: str) -> dict:
        """held-out of a sync / async row (weights at the start of this round = version ``version``): the same keys as offline; pairing
        against base works only for reward (the row has only per_task_mean), so budget hits and output tokens report only differences."""
        if self.pipeline == "async" and req(heldout, "weight_version", where, str) != str(version):
            raise SyncError(f"{where}: weight_version {heldout['weight_version']!r}, the row's version is {version}")
        per = counts(heldout, "per_task_mean", where)
        base = self.baseline(sorted(per), where)
        reward = paired(base["reward"], per)
        out = {"heldout/version": version, "heldout/offline": 0,
               "heldout/reward": num(heldout, "reward_mean", where), "heldout/live_fraction": num(heldout, "live_fraction", where),
               "heldout/budget_hit": num(heldout, "budget_hit_rate", where),
               "heldout/output_tokens": num(heldout, "output_tokens_mean", where), "heldout/base_samples": base["samples"],
               "heldout/base_reward": reward["a"], "heldout/reward_diff": reward["diff"], "heldout/reward_se": reward["se"],
               "heldout/tasks_up": reward["tasks_up"], "heldout/tasks_down": reward["tasks_down"]}
        if reward["z"] is not None:                    # None when se == 0
            out["heldout/reward_z"] = reward["z"]
        out["heldout/budget_hit_diff"] = out["heldout/budget_hit"] - base["budget_hit"]
        out["heldout/output_tokens_diff"] = out["heldout/output_tokens"] - base["output_tokens"]
        return out

    def offline_heldout(self) -> list[tuple[int, str, dict]]:
        """``heldout_offline/vNNNNNN/summary.json`` files not yet logged (sorted by version). A version dir without summary.json = still
        being evaluated (heldout_offline.py writes it atomically at the very end), look again next time; dirs not named vNNNNNN (logs,
        dirs moved aside by hand) are ignored."""
        out_dir = self.root / "heldout_offline"
        if not out_dir.is_dir():
            return []
        logged = set(self.cursor["heldout_logged"]["offline"])
        found = []
        for sub in sorted(out_dir.iterdir()):
            match = OFFLINE_DIR.fullmatch(sub.name)
            path = sub / "summary.json"
            if match is None or not sub.is_dir() or int(match.group(1)) in logged or not path.is_file():
                continue
            version, where = int(match.group(1)), str(path)
            doc = parse_json(path.read_text(), where)
            for key, want in (("version", version), ("weight_version", str(version)), ("pipeline", self.pipeline)):
                if req(doc, key, where) != want:
                    raise SyncError(f"{where}: {key} {doc[key]!r}, want {want!r}")
            if Path(req(doc, "run_root", where, str)).resolve() != self.root:
                raise SyncError(f"{where}: run_root {doc['run_root']!r} is not {self.root}")
            found.append((version, where, doc))
        return found

    # ---- passes
    def sync_learning_health(self) -> None:
        path = self.root / "analysis/reward-watch/latest.json"
        if not path.exists() or self.stop.is_set():
            return
        where = str(path)
        doc = parse_json(path.read_text(), where)
        if req(doc, "format", where, str) != "qwen38fn-rl-reward-watch-v1" or Path(req(doc, "run_root", where, str)).resolve() != self.root:
            raise SyncError(f"{where}: wrong learning-health format or run_root")
        receipt = req(doc, "receipt_id", where, str)
        if receipt == self.cursor["learning_health_receipt_id"]:
            return
        paired = req(doc, "paired_tasks", where, dict)
        out = {"learning_health/optimizer_step": num(doc, "optimizer_step", where),
               "learning_health/paired_tasks": num(paired, "tasks", where),
               "learning_health/alarm": int(req(doc, "alarm", where, bool)),
               "learning_health/final": int(req(doc, "final", where, bool))}
        if paired["tasks"]:
            out["learning_health/reward_difference_pp"] = num(paired, "reward_difference_pp", where)
            interval = req(req(paired, "bootstrap", where, dict), "interval_pp", where, list)
            if len(interval) != 2:
                raise SyncError(f"{where}: expected two bootstrap endpoints")
            out.update({"learning_health/ci95_low_pp": number(interval[0], where),
                        "learning_health/ci95_high_pp": number(interval[1], where)})
            for side in ("first", "latest"):
                summary = req(paired, side, where, dict)
                for key in ("reward", "budget_rate", "mean_output_tokens"):
                    out[f"learning_health/{key}_{side}"] = num(summary, key, where)
        for size, window in req(doc, "batch_windows", where, dict).items():
            out[f"learning_health/batch{size}_difference_pp"] = num(window, "reward_difference_pp", where)
            for side in ("first", "last"):
                out[f"learning_health/batch{size}_reward_{side}"] = num(req(window, side, where, dict), "reward", where)
        self.log(out)
        self.cursor["learning_health_receipt_id"] = receipt
        self.save_cursor()
        self.say(f"reward receipt {receipt}: alarm={doc['alarm']}, paired tasks={paired['tasks']}")

    def sync_once(self, *, tail_ok: bool) -> tuple[int, int, int]:
        """Write the cursor after every logged entry. Returns (rows, held-out entries, optimizer steps)."""
        trainer = self.read_trainer(tail_ok)
        rows = self.read_rows(tail_ok)
        groups = self.read_groups(tail_ok, rows) if self.stream else None
        quarantine = self.read_quarantine(tail_ok) if self.stream else None
        new_rows = new_heldout = new_steps = 0
        for idx in range(self.cursor["trainer_lines"], len(trainer)):
            if self.stop.is_set():
                return new_rows, new_heldout, new_steps
            where, row = trainer[idx]
            self.log(self.learner_metrics(row, where))
            self.cursor.update(trainer_lines=idx + 1, trainer_step=row["step"])
            self.save_cursor()
            new_steps += 1
        for idx in range(self.cursor["rows_logged"], len(rows)):
            if self.stop.is_set():
                return new_rows, new_heldout, new_steps
            metrics, groups_end = self.row_metrics(rows, idx, groups, quarantine)
            self.log(metrics)
            self.cursor["rows_logged"] = idx + 1
            if groups_end is not None:
                self.cursor["groups_lines"] = groups_end
            if "heldout/version" in metrics:
                self.cursor["heldout_logged"]["inloop"].append(idx)
            self.save_cursor()
            new_rows += 1
        for version, where, doc in self.offline_heldout():
            if self.stop.is_set():
                break
            self.log(offline_heldout_section(version, where, doc))
            self.cursor["heldout_logged"]["offline"].append(version)
            self.save_cursor()
            new_heldout += 1
        self.sync_learning_health()
        return new_rows, new_heldout, new_steps

    def loop(self) -> None:
        """Once (without --follow) or repeatedly until the orchestrator is gone: liveness is checked before reading, so the last read
        before exiting always happens after it died."""
        while True:
            alive = self.orchestrator_alive()
            rows, heldout, steps = self.sync_once(tail_ok=alive)
            if rows or heldout or steps:
                self.say(f"+{rows} {self.key}(s) (logged {self.cursor['rows_logged']}), +{heldout} offline held-out, "
                         f"+{steps} optimizer step(s) (latest {self.cursor['trainer_step']})")
            if self.stop.is_set():
                self.say("signal: stopping after the last complete log")
                return
            if not self.args.follow:
                return
            if not alive:
                self.say("the orchestrator is gone: final pass done")
                return
            if self.stop.sleep(self.args.poll):
                self.say("signal: stopping")
                return


# --------------------------------------------------------------------------- CLI

def positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive, not {text!r}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="the run's orchestrator config (default: the configs/*.json whose run_root "
                                                "is --run-root)")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name", help="wandb run name (default: the run root's directory name)")
    ap.add_argument("--follow", action="store_true", help="keep reading until the orchestrator is gone, then a final pass")
    ap.add_argument("--poll", type=positive_float, default=DEFAULT_POLL_SECONDS, help="--follow: seconds between reads")
    ap.add_argument("--orchestrator-pid", type=int,
                    help="the orchestrator's PID (launch_orchestrator.sh passes its unit's MainPID); default: the PID in "
                         "<run_root>/orchestrator.lock")
    ap.add_argument("--baseline", type=Path, nargs="+",
                    help="in-loop held-out: baseline sample run dirs (s<K>, results/<task>.json); default base24 like "
                         "heldout_offline.py")
    ap.add_argument("--wandb-dir", type=Path, help="wandb's local files (default $ARC_RUNS/qwen38fn_rl/wandb)")
    args = ap.parse_args(argv)
    if not args.run_root.is_dir():
        ap.error(f"--run-root {args.run_root} is not a directory")
    args.run_root = args.run_root.resolve()
    if args.name is None:
        args.name = args.run_root.name
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config is not None else find_config(args.run_root)
    cfg = load_config(config_path)
    if cfg_path(cfg["run_root"]).resolve() != args.run_root:
        raise SyncError(f"{config_path}: run_root is {cfg['run_root']}, not --run-root {args.run_root}")
    stop = StopFlag()

    def on_signal(signum, frame):
        stop.set()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    # --follow is a sidecar next to the orchestrator: the previous sidecar releases the lock at most one poll plus a wandb flush after
    # its orchestrator ends
    fd = hold_run_root(args.run_root, wait_seconds=2 * args.poll + 120 if args.follow else 0.0)
    syncer = Syncer(args, cfg, config_path, stop)
    try:
        try:
            syncer.open()
            syncer.loop()
        except BaseException:
            if syncer.run is not None:                 # None: wandb.init itself failed
                syncer.run.finish(exit_code=1)
            raise
        syncer.run.finish()
    finally:
        os.close(fd)
        for s, handler in previous.items():
            signal.signal(s, handler)
    syncer.say(f"done: {syncer.cursor['rows_logged']} {syncer.key}(s), offline held-out "
               f"{syncer.cursor['heldout_logged']['offline']}, {syncer.cursor['wandb_steps']} wandb steps; {syncer.run.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
