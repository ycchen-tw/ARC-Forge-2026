"""Read-only native DP census and owned ARC2 in-flight status; no generation calls."""
from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PATHS = ("/v1/loads?include=all", "/server_info")
COUNTERS = ("logical_capacity_tokens", "logical_available_tokens", "host_used_tokens",
            "mamba_available_slots", "snapshot_buffers_free", "snapshot_pending", "parked_records")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def get_json(base_url, path, timeout):
    started = time.monotonic()
    try:
        request = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        return {"status": "ok", "duration_s": time.monotonic() - started, "body": body}
    except Exception as exc:
        return {"status": "failed", "duration_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}"}


def physical_loads(payload, dp_size):
    rows = {}
    for row in payload["loads"]:
        dp = row["dp_rank"]
        require(type(dp) is int and 0 <= dp < dp_size, "unexpected load DP rank")
        for key in ("num_running_reqs", "num_waiting_reqs", "total_prefill_uncached_tokens", "total_prefill_busy_us"):
            require(type(row[key]) is int and row[key] >= 0, f"invalid native {key}")
        require(isinstance(row["timestamp"], (int, float)) and math.isfinite(row["timestamp"]), "invalid load timestamp")
        require(isinstance(row["queues"], dict), "missing native queues")
        moments = row["decode_moments"]
        require(moments is None or isinstance(moments, list) and len(moments) == 6
                and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in moments),
                "invalid native decode moments")
        if dp in rows:
            require(row == rows[dp], "conflicting duplicate physical DP load")
        rows[dp] = row
    require(set(rows) == set(range(dp_size)), "missing physical DP loads")
    return rows


def physical_ledgers(info, dp_size):
    rows = {}
    for state in info["internal_states"]:
        row = state["qsa_hisparse"]
        if row["attn_tp_rank"] != 0:
            continue
        dp = row["dp_rank"]
        require(type(dp) is int and 0 <= dp < dp_size and dp not in rows, "duplicate/invalid HiSparse DP leader")
        requests = row["running_requests"]
        require(len({req["rid"] for req in requests}) == len(requests), "duplicate native running RID")
        require(all(isinstance(req["rid"], str) and type(req["input_tokens"]) is int
                    and type(req["committed_tokens"]) is int for req in requests), "invalid running ledger")
        rows[dp] = {**{key: row[key] for key in COUNTERS}, "running_requests": requests}
    require(set(rows) == set(range(dp_size)), "missing HiSparse DP leader ledger")
    return rows


def owned_states(directory, now):
    rows, errors = [], []
    for path in sorted(directory.glob("*/inflight.json")):
        try:
            row = json.loads(path.read_text())
            if not row["active"]:
                continue
            require(row["sid"] == path.parent.name, "SID/path mismatch")
            started = row.get("call_started_monotonic_s") if row["phase"] == "model_call" else None
            if started is None:
                started = row["started_monotonic_s"]
            row["observed_elapsed_s"] = max(0., now - started)
            row["producer_event_age_s"] = max(0., now - row["monotonic_s"])
            rows.append(row)
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return rows, errors


class Observer:
    def __init__(self, args):
        self.args = args
        self.previous = {}
        self.progress_at = {}
        self.last_success = {}

    def snapshot(self, fetch=get_json):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {path: pool.submit(fetch, self.args.base_url, path, self.args.http_timeout) for path in PATHS}
            replies = {path: future.result() for path, future in pending.items()}
        at, epoch = time.monotonic(), time.time()
        loads, ledgers = {}, {}
        for path, parse in ((PATHS[0], physical_loads), (PATHS[1], physical_ledgers)):
            reply = replies[path]
            if reply["status"] == "ok":
                try:
                    parsed = parse(reply["body"], self.args.dp_size)
                    if path == PATHS[0]:
                        loads = parsed
                    else:
                        ledgers = parsed
                    self.last_success[path] = at
                except Exception as exc:
                    reply.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            reply["last_success_age_s"] = None if path not in self.last_success else at - self.last_success[path]
        owned, errors = owned_states(self.args.transport_dir, at)
        dps = {}
        for dp, row in loads.items():
            age = epoch - row["timestamp"]
            fresh = 0 <= age <= 2 * self.args.interval
            moments = row["decode_moments"]
            signature = (row["total_prefill_uncached_tokens"], None if moments is None else tuple(moments))
            previous = self.previous.get(dp)
            delta = None
            if fresh:
                if previous is not None:
                    before = previous[0]
                    old_steps = 0 if previous[1] is None else previous[1][0]
                    new_steps = 0 if moments is None else moments[0]
                    delta = {"prefill_uncached_tokens": signature[0] - before, "decode_steps": new_steps - old_steps}
                    if any(value < 0 for value in delta.values()):
                        delta = {"counter_reset": True}
                if previous != signature:
                    self.progress_at[dp] = at
                self.previous[dp] = signature
            dps[str(dp)] = {"running": row["num_running_reqs"], "waiting": row["num_waiting_reqs"],
                "queues": row["queues"], "native_timestamp_age_s": age, "stale": not fresh,
                "prefill_uncached_tokens": row["total_prefill_uncached_tokens"],
                "prefill_busy_us": row["total_prefill_busy_us"], "decode_moments": moments,
                "delta_since_previous_fresh_poll": delta,
                "native_counters_unchanged_for_s": None if not fresh or dp not in self.progress_at else at - self.progress_at[dp],
                "hisparse": ledgers.get(dp)}
        for row in owned:
            dp, rid = row["dp_rank"], row.get("rid")
            ledger = ledgers.get(dp)
            match = [] if ledger is None or not rid else [req for req in ledger["running_requests"] if req["rid"] == rid]
            row["native_running_request"] = match[0] if match else None
            row["correlation"] = ("exact-owned-rid-and-dp" if match else "ledger-unavailable" if ledger is None
                                  else "rid-unavailable" if not rid else "not-in-running-ledger")
        healthy = all(r["status"] == "ok" for r in replies.values()) and not errors and not any(dp["stale"] for dp in dps.values())
        return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "capture_monotonic_s": at,
            "poll_duration_s": at - started, "status": "ok" if healthy else "degraded",
            "endpoints": {path: {k: v for k, v in reply.items() if k != "body"} for path, reply in replies.items()},
            "physical_dp": dps, "owned_inflight": owned, "inflight_file_errors": errors,
            "producer_directory_exists": self.args.transport_dir.is_dir(),
            "native_loads": replies[PATHS[0]].get("body"),
            "scope": "sampled native/client observation; pending HTTP is not a server queue; unchanged counters do not prove a stall"}


def concise(row):
    dps = {}
    for dp, state in row["physical_dp"].items():
        ledger = state["hisparse"]
        dps[dp] = {**{key: value for key, value in state.items() if key != "hisparse"},
                   "hisparse": None if ledger is None else {key: ledger[key] for key in COUNTERS},
                   "running_ledger_count": None if ledger is None else len(ledger["running_requests"])}
    counts = {}
    for state in row["owned_inflight"]:
        key = f"dp{state['dp_rank']}:{state['phase']}"
        counts[key] = counts.get(key, 0) + 1
    return {**{key: row[key] for key in ("timestamp_utc", "status", "endpoints", "inflight_file_errors")},
            "physical_dp": dps, "owned_active_sessions": len(row["owned_inflight"]), "owned_phases": counts,
            "oldest_owned_sessions": sorted(row["owned_inflight"], key=lambda state: state["observed_elapsed_s"], reverse=True)[:6]}


def run(args):
    require(0 < args.http_timeout <= 2 and math.isfinite(args.http_timeout), "HTTP timeout must be <=2s")
    require(args.interval > 0 and math.isfinite(args.interval), "interval must be positive")
    require(args.dp_size > 0 and not args.base_url.rstrip("/").endswith("/v1"), "use native server root and positive DP size")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    observer = Observer(args)
    with (args.output_dir / "snapshots.jsonl").open("x") as output:
        while not stop.is_set() and (args.stop_file is None or not args.stop_file.exists()):
            row = observer.snapshot()
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            output.flush()
            print("ARC2_LIVE=" + json.dumps(concise(row), separators=(",", ":")), flush=True)
            if args.once:
                break
            deadline = time.monotonic() + args.interval
            while not stop.is_set() and time.monotonic() < deadline:
                if args.stop_file is not None and args.stop_file.exists():
                    stop.set()
                stop.wait(min(1., max(0., deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--transport-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--interval", type=float, default=30.)
    parser.add_argument("--http-timeout", type=float, default=2.)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    raise SystemExit(run(parser.parse_args()))
