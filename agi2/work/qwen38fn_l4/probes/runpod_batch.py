"""Short-prompt parallel serving smoke; no output equality or quality gate.

Invented text, local tokenizer, owned sessions, fixed attention DP0/DP1 routes.
Raw receipts and client timing are saved; scheduler occupancy and actual graph
execution require separate server evidence.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TEXT = ("A traveler visits a quiet library beside a garden. The librarian puts three "
        "books on a table and returns two books to a shelf. The traveler reads notes "
        "about rivers, seasons, tools, and trees. Continue this invented passage.")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def make_jobs(args, namespace):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    require(args.model.is_dir(), "--model must be a local tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    vocabulary_size = len(tokenizer)
    jobs = []
    for index in range(args.concurrency):
        sid = f"{namespace}-job{index:04d}"
        text = f"Request {index:04d}, namespace {namespace}. {TEXT}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(ids and all(type(token) is int and 0 <= token < vocabulary_size for token in ids),
                "tokenizer produced invalid prompt IDs")
        jobs.append({"index": index, "sid": sid, "rid": sid + "-generate", "dp": index % 2,
                     "input_ids": ids, "text": text})
    require(len({tuple(job["input_ids"][:32]) for job in jobs}) == args.concurrency,
            "early prompt namespaces are not distinct")
    files = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
             for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
             if (args.model / name).is_file()}
    return jobs, {"class": type(tokenizer).__name__, "file_sha256": files,
                  "literal_text_sha256": hashlib.sha256(TEXT.encode()).hexdigest()}


def exchange(args, directory, path, body=None, *, barrier=None, epoch=None, timeout=None):
    directory.mkdir(parents=True)
    payload = None if body is None else json.dumps(body).encode()
    write_json(directory / "request.json", {"path": path, "body": body})
    if payload is not None:
        (directory / "request.body").write_bytes(payload)
    if barrier is not None:
        barrier.wait(timeout=args.control_timeout + 30)
    started = time.monotonic()
    receipt = {"path": path, "started_monotonic_s": started,
               "dispatch_offset_s": None if epoch is None else started - epoch}
    try:
        req = urllib.request.Request(args.base_url.rstrip("/") + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or args.timeout) as response:
            raw, status = response.read(), response.status
        receipt["response_received_monotonic_s"] = time.monotonic()
        receipt["http_elapsed_s"] = receipt["response_received_monotonic_s"] - started
        (directory / "response.body").write_bytes(raw)
        receipt["http_status"] = status
        require(200 <= status < 300, f"{path}: HTTP {status}")
        if not raw and path in ("/close_session", "/abort_request"):
            return None, receipt
        return json.loads(raw), receipt
    except urllib.error.HTTPError as exc:
        receipt.update(http_status=exc.code, error=f"{type(exc).__name__}: {exc}")
        (directory / "response.body").write_bytes(exc.read())
        raise
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_s"] = time.monotonic() - started
        write_json(directory / "transport.json", receipt)


def run(args):
    require(args.concurrency > 0 and args.new_tokens > 0, "concurrency and output budget must be positive")
    require(args.cleanup_timeout > 0, "cleanup timeout must be positive")
    require(not args.output_dir.exists(), "refusing to overwrite batch evidence")
    args.output_dir.mkdir(parents=True)
    namespace = "batch-" + uuid.uuid4().hex[:16]
    result = {"status": "running", "scope": "short-prompt parallel serving smoke",
              "namespace": namespace, "requested_concurrency": args.concurrency,
              "sampling_params": {"temperature": 0, "max_new_tokens": args.new_tokens, "ignore_eos": True},
              "limits": ["client parallelism is not scheduler occupancy proof",
                         "graph execution needs the preserved server log",
                         "no output equality, model quality, or long-context claim"],
              "jobs": [], "cleanup_errors": [], "unattempted_cleanup": []}
    jobs = []
    overall_started = time.monotonic()
    try:
        info, _ = exchange(args, args.output_dir / "server-before", "/server_info", timeout=args.control_timeout)
        result["server_info"] = info
        require(info["dp_size"] == 2 and info["enable_dp_attention"] and info["tp_size"] == info["ep_size"] == 4,
                "batch smoke requires the TP4/EP4 attention-DP2 server")
        jobs, result["tokenizer"] = make_jobs(args, namespace)
        barrier = threading.Barrier(args.concurrency + 1)
        dispatch_epoch = time.monotonic()

        def generate(job):
            directory = args.output_dir / "requests" / f"job-{job['index']:04d}"
            record = {key: job[key] for key in ("index", "rid", "sid", "dp")}
            record.update(input_tokens=len(job["input_ids"]), input_ids_sha256=digest(job["input_ids"]))
            body = {"rid": job["rid"], "session_id": job["sid"], "routed_dp_rank": job["dp"],
                    "input_ids": job["input_ids"], "stream": False, "sampling_params": result["sampling_params"]}
            try:
                response, timing = exchange(args, directory, "/generate", body, barrier=barrier, epoch=dispatch_epoch)
                ids, meta = response["output_ids"], response["meta_info"]
                require(isinstance(ids, list) and len(ids) == args.new_tokens and
                        all(type(token) is int and token >= 0 for token in ids), f"{job['rid']}: output IDs/count mismatch")
                require(meta["prompt_tokens"] == len(job["input_ids"]) and meta["completion_tokens"] == len(ids),
                        f"{job['rid']}: native token counts differ")
                require(meta["dp_rank"] == job["dp"], f"{job['rid']}: wrong DP receipt")
                require(meta["num_retractions"] == 0, f"{job['rid']}: request retracted")
                require(meta["finish_reason"]["type"] == "length", f"{job['rid']}: unexpected finish reason")
                record.update(status="passed", output_id_count=len(ids), output_ids_sha256=digest(ids),
                              meta_info=meta, timing=timing)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(directory / "result.json", record)
            return record

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            pending = [executor.submit(generate, job) for job in jobs]
            barrier.wait(timeout=args.control_timeout + 30)
            dispatch_started = time.monotonic()
            result["jobs"] = [future.result() for future in pending]
        result["batch_wall_s"] = time.monotonic() - dispatch_started
        result["completed_requests"] = sum(job["status"] == "passed" for job in result["jobs"])
        result["completed_output_tokens"] = sum(job.get("output_id_count", 0) for job in result["jobs"])
        result["completed_requests_by_dp"] = {str(dp): sum(job["dp"] == dp and job["status"] == "passed"
                                                            for job in result["jobs"]) for dp in (0, 1)}
        result["status"] = "passed" if result["completed_requests"] == args.concurrency else "failed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_started = time.monotonic()
        deadline = cleanup_started + args.cleanup_timeout
        failed = {job["rid"] for job in result["jobs"] if job["status"] == "failed"}
        if len(result["jobs"]) != len(jobs):
            failed.update(job["rid"] for job in jobs)
        for job in jobs:
            operations = [("abort", "/abort_request", {"rid": job["rid"]})] if job["rid"] in failed else []
            operations.append(("close", "/close_session", {"session_id": job["sid"]}))
            for operation, path, body in operations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result["unattempted_cleanup"].append({"operation": operation, "rid": job["rid"], "sid": job["sid"]})
                    continue
                try:
                    exchange(args, args.output_dir / "cleanup" / f"{operation}-{job['index']:04d}",
                             path, body, timeout=min(args.control_timeout, remaining))
                except Exception as exc:
                    result["cleanup_errors"].append(f"{operation} {job['rid']}: {type(exc).__name__}: {exc}")
        result["cleanup_elapsed_s"] = time.monotonic() - cleanup_started
        result["cleanup_timeout_s"] = args.cleanup_timeout
        if result["cleanup_errors"] or result["unattempted_cleanup"]:
            result["status"] = "failed"
        result["elapsed_s"] = time.monotonic() - overall_started
        write_json(args.output_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--control-timeout", type=float, default=10)
    parser.add_argument("--cleanup-timeout", type=float, default=60)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "requested_concurrency", "elapsed_s")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
