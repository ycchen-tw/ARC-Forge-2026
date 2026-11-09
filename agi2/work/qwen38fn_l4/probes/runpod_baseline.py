"""Launch the audited Kaggle MTP/ReplaySSM-spec baseline on one L4x4 Pod.

Leaves the owned server running after readiness; on a startup failure, stops only its
process group. Readiness and capacity checks are not throughput/quality evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from runpod_runtime import runtime_env, verify_model, write_json


def server_args(python: str, model: Path, port: int) -> list[str]:
    # Same argv as audited notebook cell 15; only model path, Python, and port relocate.
    return [python, "-m", "sglang.launch_server", "--model-path", str(model),
            "--tokenizer-path", str(model), "--served-model-name", "tir",
            "--host", "127.0.0.1", "--port", str(port), "--tp", "4", "--dp-size", "2",
            "--enable-dp-attention", "--ep-size", "4", "--quantization", "modelopt_mixed",
            "--fp4-gemm-backend", "marlin", "--moe-runner-backend", "marlin",
            "--attention-backend", "triton", "--linear-attn-prefill-backend", "triton",
            "--linear-attn-decode-backend", "triton", "--ple-offload-embedding", "--language-only",
            "--json-model-override-args", json.dumps({"text_config": {
                "ple_embedding_dtype": "float8_e4m3fn", "num_nextn_predict_layers": 1}}),
            "--dtype", "bfloat16", "--kv-cache-dtype", "nvfp4", "--page-size", "64",
            "--context-length", "65536", "--chunked-prefill-size", "1024",
            "--max-running-requests", "16", "--max-mamba-cache-size", "80",
            "--mamba-ssm-dtype", "float16", "--mamba-radix-cache-strategy", "extra_buffer_lazy",
            "--mamba-max-states-per-path", "1", "--enable-cache-report", "--mem-fraction-static", "0.94",
            "--cuda-graph-max-bs-decode", "8", "--disable-prefill-cuda-graph", "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_coder", "--model-loader-extra-config",
            json.dumps({"enable_multithread_load": True, "num_threads": 2}), "--disable-custom-all-reduce",
            "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
            "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
            "--enable-linear-replayssm-spec"]


def validate_server_info(info: dict) -> list[dict]:
    required = {"tp_size": 4, "dp_size": 2, "ep_size": 4, "enable_dp_attention": True,
                "kv_cache_dtype": "nvfp4", "context_length": 65536, "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4,
                "enable_linear_replayssm_spec": True, "enable_linear_replayssm": False,
                "disable_custom_all_reduce": True, "disable_radix_cache": False,
                "mamba_radix_cache_strategy": "extra_buffer_lazy", "uses_mamba_radix_cache": True}
    for name, expected in required.items():
        if info.get(name) != expected:
            raise RuntimeError(f"server profile mismatch: {name}={info.get(name)!r}, expected {expected!r}")
    states = info["internal_states"]
    if len(states) != 2:
        raise RuntimeError("expected two independent attention DP pools")
    workers = []
    for rank, state in enumerate(states):
        worker = {"index": rank, "pool": int(state["memory_usage"]["token_capacity"]),
                  "running": int(state["effective_max_running_requests_per_dp"]),
                  "mamba": int(state["max_mamba_cache_size"]), "memory": state["memory_usage"]}
        if worker["pool"] < 200000 or worker["running"] < 8 or worker["mamba"] < 40:
            raise RuntimeError(f"baseline capacity gate failed: {worker}")
        # The notebook uses a 20k average admission budget, not 8 * 65536 residency.
        if 8 * 20000 > worker["pool"] * 0.80:
            raise RuntimeError(f"baseline admission headroom failed: {worker}")
        workers.append(worker)
    return workers


def gpu_probe(python: str, env: dict, run_dir: Path) -> dict:
    code = '''import json, platform, sys, torch, sglang
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
caps = [torch.cuda.get_device_capability(i) for i in range(len(names))]
assert len(names) == 4 and all("L4" in name for name in names), names
assert all(tuple(cap) == (8, 9) for cap in caps), caps
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda.startswith("13."), torch.version.cuda
for i in range(4):
    x = torch.arange(16, device=f"cuda:{i}", dtype=torch.float32)
    assert x.sum().item() == 120
    torch.cuda.synchronize(i)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
    "python": sys.version, "python_executable": sys.executable, "libc": platform.libc_ver(),
    "sglang": sglang.__version__, "torch_file": torch.__file__, "sglang_file": sglang.__file__,
    "gpus": names, "capabilities": caps,
    "p2p": [[i == j or torch.cuda.can_device_access_peer(i,j) for j in range(4)] for i in range(4)]}))
'''
    probe = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    (run_dir / "gpu_probe.log").write_text(probe.stdout + probe.stderr)
    if probe.returncode:
        raise RuntimeError("runtime CUDA probe failed; see gpu_probe.log")
    record = json.loads(probe.stdout.strip().splitlines()[-1])
    record["topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    record["driver"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"], text=True)
    record["cuda_compat_dir"] = env.get("CUDA_COMPAT_DIR")
    write_json(run_dir / "gpu_probe.json", record)
    return record


def require_free_port(port: int) -> None:
    with socket.socket() as sock:
        # Match the HTTP server's reuse policy: closed connections in TIME_WAIT
        # may remain after an owned stop; an existing listener still rejects bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def stop_owned(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)


def recorded_env(env: dict) -> dict:
    """Relevant runtime and collective settings, without inherited application credentials."""
    record = {k: env[k] for k in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LIBRARY_PATH",
                                 "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST", "SGLANG_QSA_DIRECT_DECODE",
                                 "SGLANG_QSA_DIRECT_MIN_ROWS", "ARC_QSA_FP4_FUSED")}
    if env.get("CUDA_COMPAT_DIR"):
        record["CUDA_COMPAT_DIR"] = env["CUDA_COMPAT_DIR"]
    record.update({k: v for k, v in env.items() if k.startswith("NCCL_")})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True, help="directory with baseline RUNTIME.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="new directory under ARC_RUNS")
    parser.add_argument("--port", type=int, default=30020)
    parser.add_argument("--python", help="Python 3.12 executable (or RUNPOD_PYTHON)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="validate assets and print launch manifest; no CUDA call")
    parser.add_argument("--wait", action="store_true", help="stay attached after readiness; default leaves server running")
    args = parser.parse_args()
    python, env, runtime = runtime_env(args.runtime.resolve(), "baseline", args.python)
    model = args.model.resolve()
    model_identity = verify_model(model, {"MANIFEST": runtime["target_manifest"]})
    command = server_args(python, model, args.port)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise RuntimeError("run directory is not empty; use a new run tag")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Keep secrets inherited by child processes out of run artifacts.
    env_record = recorded_env(env)
    launch = {"args": command, "environment": env_record, "runtime": runtime,
              "model": model_identity, "port": args.port, "status": "dry-run" if args.dry_run else "starting"}
    write_json(args.run_dir / "launch.json", launch)
    if args.dry_run:
        print(json.dumps(launch, indent=2))
        return
    require_free_port(args.port)
    gpu_probe(python, env, args.run_dir)
    log_path = args.run_dir / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (args.run_dir / "server.pid").write_text(f"{server.pid}\n")
    launch.update(pid=server.pid, pgid=server.pid)
    write_json(args.run_dir / "launch.json", launch)
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.timeout:
            if server.poll() is not None:
                raise RuntimeError(f"owned server exited during startup: {server.returncode}; see server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/get_server_info", timeout=5) as response:
                    info = json.load(response)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(5)
                continue
            workers = validate_server_info(info)
            bad = [line for line in log_path.read_text(errors="replace").splitlines()
                   if "nvfp4" in line.lower() and any(w in line.lower() for w in ("fallback", "not supported", "unavailable"))]
            if bad:
                raise RuntimeError("NVFP4 fallback detected: " + "\n".join(bad[-10:]))
            write_json(args.run_dir / "deployment_start.json", {"server_info": info, "workers": workers, "args": command})
            launch.update(status="ready", startup_seconds=time.monotonic() - started, workers=workers)
            write_json(args.run_dir / "launch.json", launch)
            print(f"BASELINE_READY pid={server.pid} port={args.port} run={args.run_dir}", flush=True)
            if args.wait:
                server.wait()
            return
        raise TimeoutError("baseline server startup timeout; see server.log")
    except BaseException as exc:
        stop_owned(server)
        launch.update(status="failed", error=str(exc), server_returncode=server.returncode)
        write_json(args.run_dir / "launch.json", launch)
        raise


if __name__ == "__main__":
    main()
