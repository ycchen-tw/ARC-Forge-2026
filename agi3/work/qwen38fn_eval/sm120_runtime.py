"""Offline SM120 preparation and owned-server lifecycle for readable notebooks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from sm120_host_guard import HostMemoryGuard, available_gib, owned_tree

RUNTIME_NAME = "sglang-qwen4-73a255206f-sm120-cp312.tar.zst"
RUNTIME_SHA = "e7c03ad7300669fab9d96f9d70080671b0df198f7ab40b99e0adff9162309481"
INDEX_SHA = "02ad3caa5976d7d57f12672ffb44bcf8bccc9d033c7a8566e62ccd573e3e1347"
SEQUENTIAL_INDEX_SHA = "c1d694c0fc0b6383e9f9a726e5675456f77d7d34df54987de9be12686afd4f16"
SEQUENTIAL_RELEASE_SHA = "9485bee4b390d402792624219cc2a18ef16fc73fc00941a8312e5f47c6213cb9"
NVFP4_MARKER = "hf_quant_config.json"


def checkpoint_format(config: dict) -> str:
    fmt = config.get("checkpoint_format")
    if fmt:
        return fmt
    return "sequential_mxfp4" if config.get("prequantized_checkpoint") else "mxfp4_convert"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}: {matches}")
    return matches[0]


def prefetch_checkpoint(source: Path, workers: int = 16,
                        chunk_bytes: int = 256 << 20) -> dict:
    """Warm input-mount page cache without duplicating 95 GB on the root disk.

    Explicit reads finish before the loader starts. Per-range SHA256 records
    exactly what was prefetched; residency is best-effort under kernel eviction.
    """
    started = time.monotonic()
    shards = sorted(source.glob("*.safetensors"))
    if not shards or workers < 1:
        raise ValueError("Expected checkpoint shards and a positive worker count")
    total = sum(p.stat().st_size for p in shards)
    jobs = [(p, offset, min(chunk_bytes, p.stat().st_size - offset))
            for p in shards for offset in range(0, p.stat().st_size, chunk_bytes)]

    def read_range(job):
        path, offset, length = job
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(offset)
            remaining = length
            while remaining:
                block = stream.read(min(remaining, 8 << 20))
                if not block:
                    raise EOFError(f"Short prefetch read: {path}:{offset}")
                digest.update(block)
                remaining -= len(block)
        return {"file": path.name, "offset": offset, "bytes": length,
                "sha256": digest.hexdigest()}

    ranges, loaded, reported = [], 0, started
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(read_range, job) for job in jobs]):
            row = future.result()
            ranges.append(row)
            loaded += row["bytes"]
            if time.monotonic() - reported >= 15:
                reported = time.monotonic()
                print(f"CHECKPOINT_PREFETCH {loaded / 1e9:.1f}/{total / 1e9:.1f} GB "
                      f"elapsed={reported - started:.1f}s threads={workers}", flush=True)
    if loaded != total:
        raise RuntimeError("Prefetch coverage is incomplete")
    seconds = time.monotonic() - started
    return {"mode": "pagecache", "source": str(source), "workers": workers,
            "bytes": total, "seconds": seconds, "GB_per_second": total / seconds / 1e9,
            "ranges": sorted(ranges, key=lambda r: (r["file"], r["offset"]))}


def verify_ready_checkpoint(source: Path, inventory_path: Path, workers: int) -> dict:
    """Hash every pinned release file and warm the page cache in one pass."""
    started = time.monotonic()
    inventory = json.loads(inventory_path.read_text())
    rows = inventory["files"]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Invalid checkpoint inventory")
    if {p.name for p in source.iterdir() if p.name != "dataset-metadata.json"} != set(names):
        raise RuntimeError("Checkpoint file inventory mismatch")

    def check(row):
        path = source / row["name"]
        if path.stat().st_size != row["totalBytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("Checkpoint hash mismatch: " + row["name"])
        print("CHECKPOINT_VERIFIED", row["name"], flush=True)
        return row

    with ThreadPoolExecutor(workers) as pool:
        verified = list(pool.map(check, rows))
    if inventory.get("format") == "nvidia_nvfp4":
        if not (source / NVFP4_MARKER).is_file():
            raise RuntimeError("NVIDIA NVFP4 checkpoint is missing hf_quant_config.json")
    else:
        if sha256(source / "release_manifest.json") != SEQUENTIAL_RELEASE_SHA:
            raise RuntimeError("Sequential release manifest mismatch")
        nonmoe = json.loads((source / "nonmoe_fp8_manifest.json").read_text())
        mtp = json.loads((source / "mtp_rtn_manifest.json").read_text())
        if (not nonmoe["complete"] or nonmoe["converted"] != 320
                or not mtp["complete"] or mtp["converted_weights"] != 768):
            raise RuntimeError("Incomplete prequantized checkpoint")
    return {"mode": "verified_pagecache", "source": str(source), "workers": workers,
            "bytes": sum(row["totalBytes"] for row in rows),
            "seconds": time.monotonic() - started, "files": verified}


def prepare(config: dict, source_hashes: dict) -> dict:
    scratch, output, repo = map(Path, (config["scratch"], config["output"], config["repo"]))
    for name, expected in source_hashes.items():
        if sha256(repo / name) != expected:
            raise RuntimeError(f"Embedded source changed: {name}")
    write_json(output / "source-sha.json", source_hashes)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader"], text=True)
    if len(gpu.strip().splitlines()) != 1 or "RTX PRO 6000" not in gpu:
        raise RuntimeError(f"Wrong accelerator: {gpu}")
    facts = {"python": sys.version, "gpu": gpu,
             "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
             "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
             "disk": subprocess.check_output(["df", "-h", str(scratch), str(output)], text=True)}
    write_json(output / "env-facts.json", facts)
    root = Path(config["input"])
    archive = unique_file(root, RUNTIME_NAME)
    fmt = checkpoint_format(config)
    ready_checkpoint = fmt in ("sequential_mxfp4", "nvidia_nvfp4")
    marker = NVFP4_MARKER if fmt == "nvidia_nvfp4" else "mxfp4_manifest.json"
    base = unique_file(root, marker).parent
    index_sha = sha256(base / "model.safetensors.index.json")
    if fmt == "nvidia_nvfp4":
        inventory = json.loads((repo / config["checkpoint_inventory"]).read_text())
        pinned = next(row for row in inventory["files"]
                      if row["name"] == "model.safetensors.index.json")
        if index_sha != pinned["sha256"]:
            raise RuntimeError("Model index does not match pinned dataset")
    else:
        expected_index = SEQUENTIAL_INDEX_SHA if ready_checkpoint else INDEX_SHA
        if index_sha != expected_index:
            raise RuntimeError("Model index does not match pinned dataset")
    zstd = scratch / "zstd"
    shutil.copyfile(unique_file(root, "zstd"), zstd)
    zstd.chmod(0o755)

    def extract_runtime():
        started = time.monotonic()
        if sha256(archive) != RUNTIME_SHA:
            raise RuntimeError("Runtime archive SHA mismatch")
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(scratch)],
                       env=dict(os.environ, PATH=str(scratch) + ":" + os.environ["PATH"]), check=True)
        return time.monotonic() - started

    started = time.monotonic()
    # Extract the runtime while warming the read-only model page cache.
    with ThreadPoolExecutor(max_workers=2) as pool:
        extraction = pool.submit(extract_runtime)
        if ready_checkpoint:
            staging = pool.submit(verify_ready_checkpoint, base,
                                  repo / config["checkpoint_inventory"], config["copy_threads"])
        else:
            staging = pool.submit(prefetch_checkpoint, base, config["copy_threads"])
        copy_manifest = staging.result()
        extract_seconds = extraction.result()
    write_json(output / "staging.json", copy_manifest)
    python = sys.executable if sys.version_info[:2] == (3, 12) else shutil.which("python3.12")
    if python is None:
        raise RuntimeError("The pinned runtime requires Python 3.12")
    runtime = scratch / "package"
    conversion_started = time.monotonic()
    if ready_checkpoint:
        model = base
    else:
        model = scratch / "model-mtp-rtn"
        subprocess.run([python, "-u", str(repo / "work/qwen38fn_eval/quantize_mtp_rtn.py"),
                        "--src", str(base), "--out", str(model), "--link-mode", "symlink",
                        "--threads", "8"], check=True,
                       env=dict(os.environ, PYTHONPATH=str(runtime / "lib/python3.12/site-packages"),
                                ARC_MXFP4_W4A8="0"))
    if fmt != "nvidia_nvfp4":
        shutil.copyfile(model / "mtp_rtn_manifest.json", output / "mtp_rtn_manifest.json")
    result = {"runtime": str(runtime), "model": str(model), "python": python,
              "prepare_seconds": time.monotonic() - started,
              "cache_mode": "pagecache", "prefetch_seconds": copy_manifest["seconds"],
              "extract_seconds": extract_seconds,
              "rtn_seconds": 0.0 if ready_checkpoint else time.monotonic() - conversion_started,
              "prequantized_checkpoint": ready_checkpoint,
              "checkpoint_format": fmt}
    write_json(output / "prepare.json", result)
    print("PREPARED", json.dumps(result), flush=True)
    return result


def stop_server(server: dict) -> None:
    process = server["process"]
    # The process group was created solely for this launch, including children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already exited; still wait/reap and close the owned log.
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    # SGLang's parent can exit before its children; remove only its own group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server["log_handle"].close()
    print("SERVER_STOPPED", process.pid, process.returncode, flush=True)


def start_server(config: dict, prepared: dict) -> dict:
    output, repo = Path(config["output"]), Path(config["repo"])
    log_path = output / "server.log"
    env = dict(os.environ, ARC_ROOT=str(repo), ARC_RUNS=str(output),
               MAX_JOBS=str(config.get("compile_jobs", 2)), FLASHINFER_NVCC_THREADS="1",
               TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONFAULTHANDLER="1",
               ARC_MODELS=config["scratch"], ARC_DATA=str(Path(config["scratch"]) / "data"),
               RUNTIME=prepared["runtime"], MODEL=prepared["model"], PYTHON=prepared["python"],
               SPEC="nextn", PORT=str(config["port"]),
               MAXREQ=str(config["max_requests"]), MAMBA_SLOTS=str(config["mamba_slots"]),
               MAMBA_STRATEGY=config.get("mamba_strategy", "auto"),
               MAMBA_MAX_STATES=str(config.get("mamba_max_states", -1)),
               MEMFRAC=str(config["mem_fraction"]), SSM_DTYPE=config["ssm_dtype"],
               LINEAR_DECODE=config["linear_decode"], LINEAR_VERIFY=config["linear_verify"],
               REPLAY_SSM="1" if config["replay_ssm"] else "0", CTXLEN=str(config["context_length"]),
               PREFILL_CHUNK=str(config["prefill_chunk"]),
               KV_DTYPE=config.get("kv_cache_dtype", "fp8_e4m3"))
    fmt = checkpoint_format(config)
    env["CHECKPOINT_FORMAT"] = fmt
    if fmt == "nvidia_nvfp4":
        env["MTP_MXFP4"] = "0"
        env["MOE_BACKEND"] = "flashinfer_cutlass"
    else:
        env["MTP_MXFP4"] = "1"
        env["MOE_BACKEND"] = "flashinfer_mxfp4"
    log_handle = log_path.open("w")
    started = time.monotonic()
    process = subprocess.Popen(["bash", str(repo / "work/qwen38fn_eval/serve_sm120.sh")],
                               cwd=repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    server = {"process": process, "log_handle": log_handle, "log_path": log_path,
              "base_url": f"http://127.0.0.1:{config['port']}", "config": config}
    write_json(output / "launch.json", {"pid": process.pid, "config": config})
    guard = HostMemoryGuard(process.pid, output / "host-memory-guard.json",
                            config.get("min_host_available_gib", 0))
    guard.start()
    try:
        next_report = started
        while time.monotonic() - started < config["startup_timeout"]:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited {process.returncode}\n{log_path.read_text()[-8000:]}")
            try:
                ready = requests.get(server["base_url"] + "/health", timeout=3).status_code == 200
            except (requests.ConnectionError, requests.Timeout):
                ready = False
            if ready:
                server["startup_seconds"] = time.monotonic() - started
                write_json(output / "startup.json", {"seconds": server["startup_seconds"]})
                print("SERVER_READY", server["startup_seconds"], flush=True)
                return server
            if time.monotonic() >= next_report:
                next_report = time.monotonic() + 30
                resource_sample = {"elapsed_s": time.monotonic() - started,
                                   "host_available_gib": available_gib(),
                                   "cgroup_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
                                   "owned_processes": owned_tree(process.pid)}
                try:
                    resource_sample["gpu"] = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                except subprocess.SubprocessError as exc:
                    resource_sample["gpu_error"] = type(exc).__name__
                with (output / "startup-resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(resource_sample) + "\n")
                print("STARTUP_RESOURCES", json.dumps(resource_sample), flush=True)
                tail = log_path.read_text(errors="replace").splitlines()[-3:]
                print("STARTUP", round(time.monotonic() - started), "s", "\n".join(tail)[-1200:], flush=True)
            time.sleep(3)
        raise TimeoutError("Server startup deadline exceeded")
    except BaseException:
        stop_server(server)
        raise
    finally:
        guard.close()


def capacity(server: dict, output: Path) -> dict:
    log = server["log_path"].read_text(errors="replace")
    matches = re.findall(r"max_total_num_tokens=(\d+).*?available_gpu_mem=([\d.]+) GB", log)
    if not matches:
        raise RuntimeError("Missing actual scheduler capacity in startup log")
    tokens, free_gib = int(matches[-1][0]), float(matches[-1][1])
    caps = re.findall(r"max_running_requests is capped to (\d+) by the mamba state cache", log)
    effective_requests = min([server["config"]["max_requests"]] + [int(n) for n in caps])
    result = {"kv_token_slots": tokens, "post_graph_available_gib": free_gib,
              "effective_max_requests": effective_requests,
              "allocation_log": [line for line in log.splitlines()
                                 if "Cache is allocated" in line or "ReplaySSM" in line],
              "equal_share_tokens_per_request": {str(c): tokens // c for c in (32, 48, 64)},
              "note": "Target and MTP share the logical token count; do not add their slots."}
    response = requests.get(server["base_url"] + "/get_server_info", timeout=15)
    response.raise_for_status()
    info = response.json()
    write_json(output / "server-info.json", info)
    config = server["config"]
    expected = {"enable_linear_replayssm_spec": config["replay_ssm"],
                "mamba_ssm_dtype": config["ssm_dtype"],
                "linear_attn_decode_backend": config["linear_decode"],
                "linear_attn_verify_backend": config["linear_verify"],
                "max_running_requests": config["max_requests"],
                "mem_fraction_static": config["mem_fraction"]}
    for name, value in expected.items():
        if info[name] != value:
            raise RuntimeError(f"Unexpected actual server setting {name}: {info[name]!r}")
    if info["max_total_num_tokens"] != tokens:
        raise RuntimeError("Server-info and scheduler KV counts disagree")
    write_json(output / "capacity.json", result)
    print("CAPACITY", json.dumps(result), flush=True)
    if server["config"].get("require_requested_concurrency", False) and effective_requests < server["config"]["max_requests"]:
        raise RuntimeError(f"Mamba pool caps concurrency at {effective_requests}; requested {server['config']['max_requests']}")
    return result
