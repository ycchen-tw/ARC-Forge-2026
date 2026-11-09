"""One-case long-history HiSparse acceptance probe; no model-quality claim.

Run C8, C16, C32, C64 separately, increasing only after the previous case passes.
Uses local tokenizer IDs, independent early namespaces, retained sessions, a
continuation barrier and fresh /v1/loads DP census. Saves native streaming data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def unique_loads(payload, dp_size):
    """Count a physical attention-DP scheduler once, never once per TP rank."""
    fields = ("num_running_reqs", "num_waiting_reqs", "num_used_tokens", "num_active_tokens",
              "max_total_num_tokens", "max_running_requests")
    unique = {}
    for row in payload["loads"]:
        dp = int(row["dp_rank"])
        require(0 <= dp < dp_size, f"unexpected physical DP rank {dp}")
        require(all(isinstance(row[key], int) and row[key] >= 0 for key in fields), "invalid DP census")
        if dp in unique:
            require(all(row[key] == unique[dp][key] for key in fields), f"conflicting TP receipts for DP {dp}")
        else:
            unique[dp] = row
    require(set(unique) == set(range(dp_size)), "missing physical DP census")
    return unique


HISPARSE_COUNTERS = ("logical_available_tokens", "host_used_tokens", "mamba_available_slots",
                    "snapshot_buffers_free", "snapshot_pending", "parked_records")


def unique_hisparse_states(info, dp_size):
    """Require direct allocator/lifecycle receipts; admission-view gauges are insufficient."""
    unique = {}
    for state in info["internal_states"]:
        require("qsa_hisparse" in state, "server lacks direct HiSparse capacity telemetry")
        row = state["qsa_hisparse"]
        require(isinstance(row["attn_tp_rank"], int) and row["attn_tp_rank"] >= 0,
                "missing attention-TP identity in HiSparse receipt")
        if row["attn_tp_rank"] != 0:
            continue
        dp = int(row["dp_rank"])
        require(0 <= dp < dp_size, "invalid HiSparse telemetry DP rank")
        require(all(isinstance(row[key], int) and row[key] >= 0
                    for key in HISPARSE_COUNTERS + ("logical_capacity_tokens",)), "invalid HiSparse counters")
        require(len({request["rid"] for request in row["running_requests"]}) == len(row["running_requests"]),
                "duplicated running RID in physical DP")
        require(all(isinstance(request["req_pool_idx"], int) and request["req_pool_idx"] > 0
                    and isinstance(request["input_tokens"], int) and request["input_tokens"] > 0
                    and isinstance(request["committed_tokens"], int) and request["committed_tokens"] >= 0
                    for request in row["running_requests"]), "invalid running request ledger")
        if dp in unique:
            keys = HISPARSE_COUNTERS + ("logical_capacity_tokens", "running_requests")
            require(all(row[key] == unique[dp][key] for key in keys), "conflicting HiSparse TP receipts")
        else:
            unique[dp] = row
    require(set(unique) == set(range(dp_size)), "missing direct HiSparse DP receipt")
    return unique


def counter_signature(loads, hisparse):
    return {str(dp): {"native": [row[key] for key in ("num_running_reqs", "num_waiting_reqs", "num_used_tokens")],
                      "hisparse": [hisparse[dp][key] for key in HISPARSE_COUNTERS]}
            for dp, row in loads.items()}


class ActiveWindow:
    def __init__(self, minimum, maximum_gap):
        self.minimum, self.maximum_gap = minimum, maximum_gap
        self.start = self.previous = None
        self.longest = 0.
        self.samples = 0

    def observe(self, at, qualifies):
        if not qualifies:
            self.start = self.previous = None
            return
        if self.previous is None or at - self.previous > self.maximum_gap:
            self.start = at
        self.previous = at
        self.samples += 1
        self.longest = max(self.longest, at - self.start)

    @property
    def passed(self):
        return self.longest >= self.minimum


class Job:
    def __init__(self, index, dp, sid, phase, ids, budget):
        self.index, self.dp, self.sid, self.phase = index, dp, sid, phase
        self.ids, self.budget = ids, budget
        self.rid = f"{sid}-{phase}"
        self.done = threading.Event()
        self.terminal = False
        self.error = None
        self.result = None


class Probe:
    def __init__(self, args):
        self.args = args
        self.root = args.output_dir
        self.epoch = time.monotonic()
        self.active = set()
        self.started_jobs = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.metrics_available = True

    def request(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.args.base_url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.args.control_timeout) as response:
            raw = response.read()
            if not raw:
                require(path in {"/abort_request", "/close_session"}, f"{path}: empty data response")
                return None  # Successful void control acknowledgement.
            return json.loads(raw)

    def snapshot(self, phase, timeout=None):
        started = time.monotonic()
        payload = self.request("/v1/loads?include=all", timeout=timeout)
        loads = unique_loads(payload, self.args.dp_size)
        info = self.request("/server_info", timeout=timeout)
        record = {"phase": phase, "loads": payload, "server_info": info}
        if self.metrics_available:
            req = urllib.request.Request(self.args.base_url.rstrip("/") + "/metrics")
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.args.control_timeout) as response:
                    record["metrics"] = response.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code not in (404, 405):
                    raise
                self.metrics_available = False
                record["metrics_unavailable"] = {"http_status": exc.code, "body": exc.read().decode()}
        record["elapsed_s"] = time.monotonic() - self.epoch
        record["poll_duration_s"] = time.monotonic() - started
        with (self.root / "census.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return loads, info, record

    def generate(self, job, barrier=None):
        directory = self.root / "requests" / f"{job.phase}-{job.index:04d}"
        body = {"rid": job.rid, "session_id": job.sid, "routed_dp_rank": job.dp,
                "input_ids": job.ids, "stream": True,
                "sampling_params": {"max_new_tokens": job.budget, "temperature": self.args.temperature,
                                    "ignore_eos": self.args.ignore_eos, "stream_interval": self.args.stream_interval,
                                    **({"top_p": self.args.top_p} if self.args.top_p is not None else {})}}
        response_ids, arrivals = [], []
        first = final_meta = None
        started = None
        try:
            write_json(directory / "request.json", body)
            payload = json.dumps(body).encode()  # serialize before the start barrier
            if barrier is not None:
                barrier.wait(timeout=self.args.control_timeout + 30)
            started = time.monotonic()
            req = urllib.request.Request(self.args.base_url.rstrip("/") + "/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
            with self.lock:
                require(not self.stop.is_set(), "probe cancelled before request dispatch")
                self.active.add(job.rid)
                self.started_jobs[job.rid] = job
            with urllib.request.urlopen(req, timeout=self.args.request_timeout) as response, \
                    (directory / "response.jsonl").open("w") as trace:
                saw_done = False
                for raw in response:
                    line = raw.decode().strip()
                    if not line:
                        continue
                    require(line.startswith("data:"), f"unexpected native stream line: {line[:100]}")
                    data = line[5:].strip()
                    at = time.monotonic() - started
                    if data == "[DONE]":
                        trace.write(json.dumps({"elapsed_s": at, "done": True}) + "\n")
                        saw_done = True
                        break
                    message = json.loads(data)
                    trace.write(json.dumps({"elapsed_s": at, "response": message}) + "\n")
                    trace.flush()
                    meta = message["meta_info"]
                    require(meta["prompt_tokens"] == len(job.ids), f"{job.rid}: input length changed")
                    require(meta["dp_rank"] == job.dp or (self.args.dp_size == 1 and meta["dp_rank"] is None),
                            f"{job.rid}: wrong DP affinity")
                    require(meta["num_retractions"] == 0, f"{job.rid}: request retracted")
                    count, chunk = meta["completion_tokens"], message["output_ids"]
                    previous = len(response_ids)
                    if len(chunk) == count:
                        require(chunk[:previous] == response_ids, f"{job.rid}: generated IDs changed")
                        response_ids = list(chunk)
                    elif len(chunk) + previous == count:
                        response_ids.extend(chunk)
                    else:
                        raise RuntimeError(f"{job.rid}: stream IDs/count disagree")
                    require(count <= job.budget, f"{job.rid}: exceeded output budget")
                    if count > previous:
                        first = at if first is None else first
                        arrivals.append([at, count - previous])
                    if meta["finish_reason"] is not None:
                        job.terminal = True
                        final_meta = meta
                require(saw_done and final_meta is not None, f"{job.rid}: incomplete native stream")
            require(len(response_ids) == job.budget, f"{job.rid}: output {len(response_ids)}/{job.budget}")
            require(final_meta["finish_reason"]["type"] == "length", f"{job.rid}: unexpected stop")
            cached = final_meta["cached_tokens"]
            if job.phase == "cold":
                require(cached == 0, f"{job.rid}: independent cold session shared a prefix")
            else:
                require(cached / len(job.ids) >= self.args.min_cache_ratio,
                        f"{job.rid}: cache reuse {cached}/{len(job.ids)} below threshold")
            elapsed = time.monotonic() - started
            gaps = [later[0] - earlier[0] for earlier, later in zip(arrivals, arrivals[1:])]
            after_first = len(response_ids) - arrivals[0][1]
            job.result = {"index": job.index, "sid": job.sid, "rid": job.rid, "dp": job.dp,
                          "phase": job.phase, "input_tokens": len(job.ids), "input_sha256": ids_digest(job.ids),
                          "output_ids": response_ids, "meta_info": final_meta, "elapsed_s": elapsed,
                          "ttft_s": first, "tokens_per_s": len(response_ids) / elapsed,
                          "mean_delivery_itl_s": (arrivals[-1][0] - first) / after_first if after_first else None,
                          "chunk_gap_p50_s": statistics.median(gaps) if gaps else None,
                          "chunk_gap_max_s": max(gaps) if gaps else None, "arrivals": arrivals}
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.result = {"rid": job.rid, "phase": job.phase, "error": job.error,
                          "received_output_ids": response_ids,
                          "elapsed_s": None if started is None else time.monotonic() - started}
            if isinstance(exc, urllib.error.HTTPError):
                job.result["http_status"] = exc.code
                job.result["http_body"] = exc.read().decode()
        finally:
            with self.lock:
                self.active.discard(job.rid)
            try:
                write_json(directory / "result.json", job.result)
            except Exception as exc:
                message = f"result logging failed: {type(exc).__name__}: {exc}"
                job.error = message if job.error is None else f"{job.error}; {message}"
                job.result["error"] = job.error
            finally:
                job.done.set()

    def cleanup(self, sessions, baseline):
        errors = []
        deadline = time.monotonic() + self.args.cleanup_timeout
        with self.lock:
            self.stop.set()
            active = list(self.active)
            started = list(self.started_jobs.values())
        # A timed-out HTTP request can still be executing. Also wait for already
        # dispatched clients to finish, so close cannot race a delayed POST.
        abort = {job.rid for job in started if job.error or not job.done.is_set()}
        first = True
        while first or not all(job.done.is_set() for job in started):
            first = False
            abort.update(job.rid for job in started if not job.done.is_set())
            for rid in sorted(abort):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    response = self.request("/abort_request", {"rid": rid},
                                            timeout=min(self.args.control_timeout, remaining))
                    with (self.root / "cleanup.jsonl").open("a") as output:
                        output.write(json.dumps({"operation": "abort", "request": {"rid": rid},
                                                 "response": response}) + "\n")
                except Exception as exc:
                    errors.append(f"abort {rid}: {type(exc).__name__}: {exc}")
            if all(job.done.is_set() for job in started) or time.monotonic() >= deadline:
                break
            time.sleep(min(self.args.poll_interval, max(0., deadline - time.monotonic())))
        if not all(job.done.is_set() for job in started):
            errors.append("dispatched clients did not finish before cleanup deadline")
        unattempted = []
        for _, _, sid in sessions:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                unattempted.append(sid)
                continue
            try:
                response = self.request("/close_session", {"session_id": sid},
                                        timeout=min(self.args.control_timeout, remaining))
                with (self.root / "cleanup.jsonl").open("a") as output:
                    output.write(json.dumps({"operation": "close", "request": {"session_id": sid},
                                             "response": response}) + "\n")
            except Exception as exc:
                errors.append(f"close {sid}: {type(exc).__name__}: {exc}")
        restored = False
        while time.monotonic() < deadline:
            try:
                timeout = min(self.args.control_timeout, max(.1, (deadline - time.monotonic()) / 3))
                loads, info, _ = self.snapshot("cleanup", timeout=timeout)
                hisparse = unique_hisparse_states(info, self.args.dp_size)
                restored = counter_signature(loads, hisparse) == baseline and all(job.done.is_set() for job in started)
                if restored:
                    break
            except Exception as exc:
                errors.append(f"cleanup census: {type(exc).__name__}: {exc}")
                break
            time.sleep(min(self.args.poll_interval, max(0., deadline - time.monotonic())))
        return {"owned_active_rids_aborted": active, "errors": errors, "counters_restored": restored,
                "unattempted_session_closes": unattempted, "baseline": baseline}


def make_fixtures(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    block = tokenizer.encode(f"Study note seed {args.seed}: A river crosses a quiet valley. "
                             "The notes describe seasons, tools, gardens and measured observations.\n",
                             add_special_tokens=False)
    tail = tokenizer.encode("\nContinue these notes in detail.\n", add_special_tokens=False)
    require(block and tail, "empty tokenizer fixture segments")
    sessions, fixtures = [], []
    with (args.output_dir / "fixtures.jsonl").open("w") as output:
        for index in range(args.concurrency):
            dp = index % args.dp_size
            sid = f"{args.namespace}-c{args.concurrency}-s{index:04d}"
            header = tokenizer.encode(f"{index:04d} | {sid} | Independent reading notes.\n", add_special_tokens=False)
            needed = args.input_tokens - len(header) - len(tail)
            require(needed > 0, "input budget too short for independent namespace")
            ids = header + (block * math.ceil(needed / len(block)))[:needed] + tail
            require(len(ids) == args.input_tokens, "fixture length differs from requested input")
            sessions.append((index, dp, sid))
            fixtures.append(ids)
            output.write(json.dumps({"index": index, "dp": dp, "session_id": sid,
                                     "input_sha256": ids_digest(ids), "input_ids": ids}) + "\n")
    require(len({tuple(ids[:128]) for ids in fixtures}) == args.concurrency,
            "early namespace token prefixes are not independent")
    meta = {"model": args.model, "tokenizer_class": type(tokenizer).__name__,
            "vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "input_format": "plain-text native token IDs", "namespace_prefix_tokens_checked": 128,
            "fixture_sha256": hashlib.sha256((args.output_dir / "fixtures.jsonl").read_bytes()).hexdigest()}
    return sessions, fixtures, tail, meta


def run(args):
    require(not args.output_dir.exists(), "refusing to overwrite capacity artifacts")
    args.output_dir.mkdir(parents=True)
    probe = Probe(args)
    result = {"status": "running", "scope": "long-history capacity and transport; no quality claim",
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "validation_limits": ["sampled scheduler census; unobserved sub-poll transitions",
                                    "delivery ITL from stream chunks, not individual GPU token timestamps",
                                    "native num_used_tokens records the admission view, not true HiSparse logical use",
                                    "host/GDN idle invariants need the preserved server log"]}
    sessions, baseline = [], None
    jobs = []
    try:
        sessions, fixtures, suffix, tokenizer_meta = make_fixtures(args)
        result["tokenizer"] = tokenizer_meta
        loads, info, _ = probe.snapshot("baseline")
        hisparse = unique_hisparse_states(info, args.dp_size)
        baseline = counter_signature(loads, hisparse)
        require(all(row["num_running_reqs"] == row["num_waiting_reqs"] == 0 for row in loads.values()),
                "server has active requests before this case")
        require(info["dp_size"] == args.dp_size and bool(info["enable_dp_attention"]) == (args.dp_size > 1),
                "server DP topology differs")
        require(info["tp_size"] == 4 and info["ep_size"] == 4, "server TP/EP topology differs from L4x4 arm")
        require(info["radix_cache_backend"] == "qsa_hisparse_session", "retained HiSparse session backend required")
        require(info["speculative_algorithm"] is None, "this capacity arm excludes speculative decoding")
        hisparse_config = info["hisparse_config"]
        hisparse_config = json.loads(hisparse_config) if isinstance(hisparse_config, str) else hisparse_config
        require(hisparse_config["qsa_session_resume"] and hisparse_config["qsa_session_idle_timeout_s"] == 0,
                "sessions must remain retained while later independent contexts warm")
        require(args.warm_tokens > hisparse_config["qsa_snapshot_interval"], "warm output must cross a snapshot boundary")
        maximum_length = args.input_tokens + args.warm_tokens + len(suffix) + args.output_tokens
        require(info["context_length"] is not None and maximum_length + 64 <= info["context_length"],
                "server context cap is too short for the complete continuation budget")
        per_dp = args.concurrency // args.dp_size
        page = info["page_size"]
        for dp, row in loads.items():
            needed = per_dp * (math.ceil(maximum_length / page) * page + page)
            require(row["max_running_requests"] >= per_dp, f"DP {dp} request cap is too small")
            require(hisparse[dp]["logical_available_tokens"] >= needed,
                    f"DP {dp} logical pool unsafe for {needed} independent live history/output slots")
        result["baseline_counters"] = baseline
        cold = [Job(index, dp, sid, "cold", fixtures[index], args.warm_tokens) for index, dp, sid in sessions]

        def warm_dp(dp):
            for job in cold:
                if job.dp == dp and not probe.stop.is_set():
                    probe.generate(job)
                    if job.error:
                        probe.stop.set()
                        return

        warm_threads = [threading.Thread(target=warm_dp, args=(dp,), daemon=True) for dp in range(args.dp_size)]
        warm_started = time.monotonic()
        for thread in warm_threads:
            thread.start()
        while any(thread.is_alive() for thread in warm_threads):
            require(not any(job.error for job in cold), "cold prefill or snapshot warmup failed")
            require(time.monotonic() - probe.epoch < args.max_runtime_seconds, "case runtime limit reached")
            probe.snapshot("cold-prefill")
            time.sleep(args.poll_interval)
        require(all(job.done.is_set() and job.error is None for job in cold), "not every independent session warmed")
        result["cold_wall_s"] = time.monotonic() - warm_started
        result["cold_input_tokens_per_wall_s"] = args.concurrency * args.input_tokens / result["cold_wall_s"]
        result["cold_results"] = [job.result for job in cold]
        jobs = [Job(index, dp, sid, "continuation", fixtures[index] + cold[index].result["output_ids"] + suffix,
                    args.output_tokens) for index, dp, sid in sessions]
        barrier = threading.Barrier(args.concurrency + 1)
        threads = [threading.Thread(target=probe.generate, args=(job, barrier), daemon=True) for job in jobs]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=args.control_timeout + 30)
        continuation_started = time.monotonic()
        window = ActiveWindow(args.min_active_seconds, args.poll_interval * 2.5 + .5)
        while not all(job.done.is_set() for job in jobs):
            require(not any(job.error for job in jobs), "continuation request failed")
            require(time.monotonic() - probe.epoch < args.max_runtime_seconds, "case runtime limit reached")
            loads, info, census = probe.snapshot("continuation")
            hisparse = unique_hisparse_states(info, args.dp_size)
            expected = {dp: {job.rid: job for job in jobs if job.dp == dp} for dp in range(args.dp_size)}
            rosters = {dp: {request["rid"]: request for request in row["running_requests"]}
                       for dp, row in hisparse.items()}
            history = sum(request["committed_tokens"] for row in rosters.values() for request in row.values())
            roster_matches = all(
                set(rosters[dp]) == set(expected[dp])
                and len({request["req_pool_idx"] for request in rosters[dp].values()}) == per_dp
                and all(request["input_tokens"] == len(expected[dp][rid].ids)
                        and request["committed_tokens"] >= args.input_tokens
                        for rid, request in rosters[dp].items())
                for dp in range(args.dp_size)
            )
            qualifies = (
                all(not job.terminal and not job.done.is_set() for job in jobs)
                and roster_matches
                and all(row["num_running_reqs"] == per_dp and row["num_waiting_reqs"] == 0
                        and row["queues"]["retracted"] == 0
                        for dp, row in loads.items())
                and history >= args.concurrency * args.input_tokens
                and census["poll_duration_s"] <= args.poll_interval * 2.5 + .5
            )
            window.observe(time.monotonic() - continuation_started, qualifies)
            result["active_acceptance"] = {"requested_concurrency": args.concurrency,
                                           "required_active_history_tokens": args.concurrency * args.input_tokens,
                                           "minimum_active_seconds": args.min_active_seconds,
                                           "longest_qualified_span_s": window.longest,
                                           "qualifying_samples": window.samples}
            with (args.output_dir / "acceptance_samples.jsonl").open("a") as output:
                output.write(json.dumps({"elapsed_s": time.monotonic() - continuation_started,
                                         "qualified": qualifies, "unique_dp_count": len(loads),
                                         "owned_running_roster_matches": roster_matches,
                                         "running_requests": sum(row["num_running_reqs"] for row in loads.values()),
                                         "active_owned_committed_tokens": history,
                                         "physical_dp_owners": {str(dp): {"attn_tp_rank": row["attn_tp_rank"],
                                                                         "running_rids": sorted(rosters[dp])}
                                                                for dp, row in hisparse.items()},
                                         "longest_qualified_span_s": window.longest}) + "\n")
            time.sleep(args.poll_interval)
        require(all(job.error is None for job in jobs), "one or more continuations failed")
        elapsed = time.monotonic() - continuation_started
        result["continuation_results"] = [job.result for job in jobs]
        result["continuation_wall_s"] = elapsed
        result["aggregate_output_tokens_per_s"] = sum(len(job.result["output_ids"]) for job in jobs) / elapsed
        require(window.passed, f"requested C{args.concurrency} never met the history/census gate for {args.min_active_seconds}s")
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if baseline is not None:
            result["cleanup"] = probe.cleanup(sessions, baseline)
            if result["cleanup"]["errors"] or not result["cleanup"]["counters_restored"]:
                result["status"] = "failed"
                result["cleanup_failure"] = "owned sessions did not restore counter baseline"
        result["continuation_job_states"] = [{"rid": job.rid, "done": job.done.is_set(), "error": job.error}
                                             for job in jobs]
        result["elapsed_s"] = time.monotonic() - probe.epoch
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"], "concurrency": args.concurrency,
                          "elapsed_s": result["elapsed_s"]}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, default=100000)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--warm-tokens", type=int, default=320)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="local/cached tokenizer for the actually served checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="capacity-" + uuid.uuid4().hex[:16])
    parser.add_argument("--seed", type=int, default=1838)
    parser.add_argument("--ignore-eos", action="store_true", help="explicit fixed-budget capacity workload; not quality")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-p", type=float, help="production ARC sampling uses 0.95 (flashinfer top-p path)")
    parser.add_argument("--min-active-seconds", type=float, default=30.)
    parser.add_argument("--min-cache-ratio", type=float, default=.99)
    parser.add_argument("--stream-interval", type=int, default=16)
    parser.add_argument("--poll-interval", type=float, default=1.)
    parser.add_argument("--control-timeout", type=float, default=10.)
    parser.add_argument("--request-timeout", type=float, default=600.)
    parser.add_argument("--cleanup-timeout", type=float, default=60.)
    parser.add_argument("--max-runtime-seconds", type=float, default=7200.)
    args = parser.parse_args()
    require(args.dp_size > 0 and args.concurrency > 0 and args.concurrency % args.dp_size == 0,
            "concurrency must divide evenly across fixed DP affinity")
    require(min(args.input_tokens, args.output_tokens, args.warm_tokens, args.stream_interval) > 0,
            "token budgets must be positive")
    require(args.min_active_seconds > 0 and args.poll_interval > 0 and 0 < args.min_cache_ratio <= 1,
            "invalid acceptance thresholds")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
