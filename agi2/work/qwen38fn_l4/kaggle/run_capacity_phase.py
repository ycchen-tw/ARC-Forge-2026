"""Kaggle-only entry for the verified C64 x100k retained-history workload.

This is a capacity phase, separate from ARC2 inference. It starts no server,
changes no profile, and never substitutes a short batch for the long-context gate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "probes/runpod_capacity.py"
PROBE_SHA256 = "e36b58ba3527f2ca24690f3155652f2b513a80ce281bd91e9396ffdc8919fc1c"
REFERENCE_RESULT_SHA256 = "5a098f6c71e508e82e32d2b68a0b488d8b35ddee043e42634ca01aa007cd557b"
WORKLOAD = {"concurrency": 64, "input_tokens": 100000, "warm_tokens": 128, "output_tokens": 512,
            "dp_size": 2, "seed": 1838, "ignore_eos": True, "temperature": 0.,
            "min_active_seconds": 30., "min_cache_ratio": .99, "stream_interval": 16,
            "poll_interval": 1., "control_timeout": 10., "request_timeout": 600.,
            "cleanup_timeout": 90., "max_runtime_seconds": 7200.}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(base_url, model, case, python=sys.executable):
    require(not base_url.rstrip("/").endswith("/v1"), "capacity base-url is the native server root, without /v1")
    require(hashlib.sha256(PROBE.read_bytes()).hexdigest() == PROBE_SHA256, "capacity probe differs from verified c3485dae source")
    argv = [python, str(PROBE), "--base-url", base_url, "--model", str(model), "--output-dir", str(case)]
    for key, value in WORKLOAD.items():
        argv += ["--" + key.replace("_", "-")]
        if type(value) is not bool:
            argv.append(str(value))
    return argv


def validate_result(result, case):
    require(result["status"] == "passed", "probe did not pass")
    require(all(result["config"][key] == value for key, value in WORKLOAD.items()), "capacity workload changed")
    cold, continued = result["cold_results"], result["continuation_results"]
    require(len(cold) == len(continued) == 64, "not all64 independent contexts completed both phases")
    require(len({row["sid"] for row in cold}) == 64, "cold sessions are not independent")
    require(collections.Counter(row["dp"] for row in cold) == collections.Counter({0: 32, 1: 32})
            and collections.Counter(row["dp"] for row in continued) == collections.Counter({0: 32, 1: 32}),
            "both phases need32 actual requests per DP")
    by_sid = {row["sid"]: row for row in cold}
    require({row["sid"] for row in continued} == set(by_sid), "continuation changed session ownership")
    for row in cold + continued:
        size = 128 if row["phase"] == "cold" else 512
        ids, meta = row["output_ids"], row["meta_info"]
        require(len(ids) == size and all(type(token) is int and token >= 0 for token in ids), "missing native output IDs")
        require(meta["completion_tokens"] == size and meta["dp_rank"] == row["dp"]
                and meta["num_retractions"] == 0, "native count/routing/retraction receipt differs")
        require(meta["cached_tokens"] == 0 if row["phase"] == "cold"
                else meta["cached_tokens"] / row["input_tokens"] >= .99, "cold independence or reuse failed")
    for row in continued:
        first = by_sid[row["sid"]]
        require(first["dp"] == row["dp"] and first["input_tokens"] == 100000, "session history/routing changed")
        start = json.loads((case / "requests" / f"cold-{first['index']:04d}" / "request.json").read_text())
        resume = json.loads((case / "requests" / f"continuation-{row['index']:04d}" / "request.json").read_text())
        expected = start["input_ids"] + first["output_ids"]
        require(len(start["input_ids"]) == 100000 and resume["input_ids"][:len(expected)] == expected,
                "continuation did not append its actual prior generated IDs")
        require(len(resume["input_ids"]) == row["input_tokens"] > len(expected), "continuation suffix/count invalid")
    active = result["active_acceptance"]
    require(active["requested_concurrency"] == 64 and active["required_active_history_tokens"] == 6400000
            and active["minimum_active_seconds"] == 30 and active["longest_qualified_span_s"] >= 30,
            "C64 x100k never met the actual30s history/census gate")
    cleanup = result["cleanup"]
    require(cleanup["counters_restored"] is True and not cleanup["errors"]
            and not cleanup["unattempted_session_closes"], "owned session cleanup did not restore baseline")
    require(len(result["continuation_job_states"]) == 64
            and all(job["done"] and job["error"] is None for job in result["continuation_job_states"]),
            "unfinished/failed continuation clients")
    return {"cold_calls": 64, "continuation_calls": 64, "cold_output_ids": 8192,
            "continuation_output_ids": 32768, "active_acceptance": active,
            "continuation_output_tokens_per_s": result["aggregate_output_tokens_per_s"],
            "cleanup_counters_restored": True}


def run(args):
    fast, output = args.fast_dir.resolve(), args.output_dir.resolve()
    require(fast.is_relative_to(Path("/tmp")), "capacity logging must use a fresh /tmp fast-disk directory")
    require(fast != output and not fast.is_relative_to(output) and not output.is_relative_to(fast), "fast/output directories overlap")
    require(not fast.exists() and not output.exists(), "refusing to overwrite prior phase evidence")
    argv = command(args.base_url, args.model, fast / "case")
    require(args.model.is_dir(), "--model must be the mounted local tokenizer directory")
    fast.mkdir(parents=True)
    started = time.monotonic()
    record = {"status": "running", "scope": "C64 x100k capacity; Kaggle long context pending until this phase passes",
              "command": argv, "probe_source_sha256": PROBE_SHA256, "reference_git": "c3485dae",
              "reference_result_sha256": REFERENCE_RESULT_SHA256, "workload": WORKLOAD,
              "fast_directory": str(fast), "persistent_output": str(output)}
    write_json(fast / "phase-launch.json", record)
    try:
        with (fast / "probe.log").open("w") as log:
            child = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
        record["probe_return_code"] = child.returncode
        require(child.returncode == 0, f"capacity probe exited{child.returncode}; inspect preserved probe.log")
        case = fast / "case"
        result = json.loads((case / "result.json").read_text())
        record["verification"] = validate_result(result, case)
        record["result_sha256"] = hashlib.sha256((case / "result.json").read_bytes()).hexdigest()
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["elapsed_s"] = time.monotonic() - started
        write_json(fast / "phase-result.json", record)
        try:
            shutil.copytree(fast, output)
            require(hashlib.sha256((output / "phase-result.json").read_bytes()).hexdigest()
                    == hashlib.sha256((fast / "phase-result.json").read_bytes()).hexdigest(), "phase receipt mirror differs")
            record["final_mirror_completed"] = True
        except Exception as exc:
            record.update(status="failed", final_mirror_completed=False, mirror_error=f"{type(exc).__name__}: {exc}")
        write_json(fast / "phase-result.json", record)
        if output.is_dir():
            write_json(output / "phase-result.json", record)
    print(json.dumps({key: record[key] for key in ("status", "elapsed_s", "final_mirror_completed")}))
    return 0 if record["status"] == "passed" and record["final_mirror_completed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
