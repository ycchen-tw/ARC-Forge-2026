"""Durable one-shot GPTQ: same bounded MM replay for every independent layer window.

Run in a dedicated supervisor, not an interactive tool cell. The root stages the
FP8 model, original configs and ledgers first. Only stage-bf16 reads its explicit
source via direct_checkpoint.py. No sequential activation replay is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def require_local(path):
    """Check each symlink component before traversing a mounted NFS directory."""
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4])
        mounts.append((mount, right.split()[0]))
    mounts.sort(key=lambda item: len(item[0]), reverse=True)
    pending = list(Path(os.path.abspath(path)).parts[1:])
    current, links = Path("/"), 0
    while pending:
        part = pending.pop(0)
        if part == "..":
            current = current.parent
            continue
        if part == ".":
            continue
        candidate = current / part
        fs = next(kind for mount, kind in mounts if str(candidate) == mount or
                  str(candidate).startswith(mount.rstrip("/") + "/"))
        if fs in ("nfs", "nfs4"):
            raise RuntimeError(f"stage local input/output before calibration: {candidate}")
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise RuntimeError("local input/output symlink loop")
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                current = Path("/")
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        else:
            current = candidate
    return current


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("nonfinite quantization error metric")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def proc_identity(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
            "session": int(fields[3]), "start": int(fields[19])}


def group_members(pgid):
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            rec = proc_identity(int(p.name))
            raw = (p / "stat").read_text()
            if raw[raw.rfind(")") + 2:].split()[0] == "Z":
                continue  # zombies hold no resources; their parent must reap them
        except (FileNotFoundError, ProcessLookupError):
            continue
        if rec["pgid"] == pgid:
            if rec["session"] != pgid:
                raise RuntimeError("owned process group lost its dedicated session")
            out.append(rec)
    return out


def owned_group(identity):
    try:
        now = proc_identity(identity["pid"])
    except FileNotFoundError:
        now = None
    if now is not None and now != identity:
        raise RuntimeError("owned launcher PID was reused; refusing signal")
    return group_members(identity["pgid"])


def gpu_apps():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=20)
    return [(int(pid.strip()), uuid.strip()) for line in result.stdout.splitlines() if line.strip()
            for pid, uuid in [line.split(",", 1)]]


def gpu_uuids(gpus):
    result = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-gpu=uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=20)
    uuids = result.stdout.splitlines()
    if len(uuids) != len(gpus):
        raise RuntimeError("GPU UUID inventory does not match requested devices")
    return set(uuids)


def validate_resume(receipt, identity, parts, layers):
    if not receipt["complete"] or receipt["identity"] != identity or receipt["layers"] != layers:
        raise RuntimeError("resume receipt is incomplete or belongs to different source/run/mask")
    if {r["layer"] for r in receipt["parts"]} != set(layers):
        raise RuntimeError("resume receipt omits a quantized layer")
    for rec in receipt["parts"]:
        for suffix, key in (("safetensors", "tensor_sha256"), ("json", "report_sha256")):
            name = f"int3-L{rec['layer']:02d}.safetensors" if suffix == "safetensors" else f"layer-{rec['layer']:02d}.json"
            if digest(parts / name) != rec[key]:
                raise RuntimeError("resume quantization artifact SHA256 differs from complete receipt")


class Runner:
    def __init__(self, args, identity):
        self.args, self.identity = args, identity
        self.children = []
        self.state = {"identity": identity, "status": "running"}
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=f"{args.fork}/python:{args.source_root}/src", PYTHONDONTWRITEBYTECODE="1",
                        ARC_ROOT=str(args.source_root), OMP_NUM_THREADS="4",
                        PATH=str(args.python.parent) + os.pathsep + self.env["PATH"],
                        SGLANG_RUST_SERVER="0", SGLANG_RUST_BUILD_MODE="never",
                        SGLANG_QWEN4_MNNVL_HC="0", SGLANG_OPT_LORA_FUSED_TOPK_PACK="0",
                        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    def event(self, phase, **fields):
        row = {"time": time.time(), "phase": phase, **fields}
        with (self.args.run_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.state.update(phase=phase, **fields)
        atomic(self.args.run_dir / "state.json", self.state)
        print(json.dumps(row), flush=True)

    def start(self, name, cmd, env=None):
        log = self.args.run_dir / "logs" / (name + ".log")
        child_env = {**self.env, **(env or {})}
        with log.open("ab", buffering=0) as f:
            proc = subprocess.Popen(list(map(str, cmd)), cwd=self.args.source_root, env=child_env,
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        ident = proc_identity(proc.pid)
        if ident["pgid"] != proc.pid or ident["session"] != proc.pid:
            raise RuntimeError("child did not create its dedicated process group")
        child = {"name": name, "proc": proc, "identity": ident, "started": time.monotonic()}
        self.children.append(child)
        self.event("start", child=name, process=ident, command=list(map(str, cmd)), log=str(log))
        return child

    def stop(self, child):
        identity, proc = child["identity"], child["proc"]
        members = owned_group(identity)
        owned_pids = {identity["pid"], *(r["pid"] for r in members)}
        if members:
            os.killpg(identity["pgid"], signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and owned_group(identity):
            proc.poll()  # reap the session leader
            time.sleep(1)
        if owned_group(identity):
            os.killpg(identity["pgid"], signal.SIGKILL)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and owned_group(identity):
                proc.poll()
                time.sleep(1)
        if owned_group(identity):
            raise RuntimeError("owned process group did not stop (possible uninterruptible I/O)")
        proc.wait(timeout=10)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            remaining = [pid for pid, _ in gpu_apps() if pid in owned_pids]
            if not remaining:
                self.event("stopped", child=child["name"], gpu_owned_pids_released=sorted(owned_pids))
                self.children.remove(child)
                return
            time.sleep(2)
        raise RuntimeError("owned GPU processes did not release resources")

    def wait(self, child, timeout):
        deadline, heartbeat = child["started"] + timeout, 0.0
        while child["proc"].poll() is None:
            if time.monotonic() >= deadline:
                self.stop(child)
                raise TimeoutError(f"phase {child['name']} exceeded {timeout}s")
            if time.monotonic() >= heartbeat:
                self.event("waiting", child=child["name"])
                heartbeat = time.monotonic() + 30
            time.sleep(2)
        code = child["proc"].returncode
        self.stop(child)  # also reject surviving descendants
        if code != 0:
            raise RuntimeError(f"phase {child['name']} exited {code}")

    def server(self, lo, hi, window):
        a = self.args
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, a.gpus)), "TITO_CAPTURE_ROOT": str(window / "capture"),
               "ARC_HESS_RUN_ID": a.run_id, "ARC_HESS_LAYERS": f"{lo}:{hi}", "ARC_HESS_N_LAYERS": "48",
               "ARC_SALIENCY_OUT": str(window / "hessian"), "ARC_HESS_SHM": str(window / "hessian"),
               "ARC_SALIENCY_MODEL": str(a.fp8_model), "ARC_HESS_DEVICE": "cuda", "ARC_HESS_CPU_THREADS": "4",
               "ARC_HESS_MAX_NSAMP": "2147483647", "ARC_HESS_GEMM_BATCH": "8",
               "ARC_HESS_MASK_SHA256": self.identity["mask_sha256"],
               "ARC_HESS_MTP_MASK_SHA256": self.identity["mtp_mask_sha256"],
               "ARC_SALIENCY_HOOK_PY": str(a.source_root / "work/qwen38fn_eval/gptq_hess_hook.py"),
               "SGLANG_CI_DISABLE_MOE_FUSED_FUNC": "1", "MNNVL_HC": "0",
               "ARC_MM_CALIB_EXPECT_SGLANG_SHA": a.fork_sha, "SGLANG_VLM_CACHE_SIZE_MB": "1024"}
        name = f"gean-int3-{a.run_id}"
        cmd = [a.python, a.source_root / "work/qwen38fn_dflash/int3_mm_calib_runtime.py",
               "--model-path", a.fp8_model, "--served-model-name", name, "--tp", "4", "--ep", "4",
               "--host", "127.0.0.1", "--port", str(a.port), "--context-length", "131072",
               "--max-total-tokens", "131072", "--max-mamba-cache-size", "16", "--mem-fraction-static", ".32",
               "--max-running-requests", "1", "--chunked-prefill-size", "4096", "--kv-cache-dtype", "fp8_e4m3",
               "--mamba-ssm-dtype", "bfloat16", "--linear-attn-prefill-backend", "triton",
               "--linear-attn-decode-backend", "triton", "--moe-runner-backend", "triton",
               "--ple-offload-embedding", "--ple-mm-placeholder-ids", "--mm-feature-transport", "cpu",
               "--disable-cuda-graph", "--disable-prefill-cuda-graph", "--disable-radix-cache",
               "--disable-overlap-schedule", "--skip-server-warmup", "--watchdog-timeout", "1800"]
        child = self.start(f"w{lo:02d}-server", cmd, env)
        deadline = time.monotonic() + a.server_timeout
        url = f"http://127.0.0.1:{a.port}"
        while time.monotonic() < deadline:
            if child["proc"].poll() is not None:
                raise RuntimeError("calibration server exited before readiness")
            try:
                with urllib.request.urlopen(url + "/server_info", timeout=5) as response:
                    info = json.load(response)
                if info["served_model_name"] != name:
                    raise RuntimeError("calibration port belongs to another server")
                pids = []
                for rec in owned_group(child["identity"]):
                    if rec["ppid"] == child["proc"].pid:
                        path = Path("/proc") / str(rec["pid"])
                        if b"scheduler" in (path / "cmdline").read_bytes() or "scheduler" in (path / "comm").read_text():
                            pids.append(rec["pid"])
                if len(pids) == 4:
                    self.event("server_ready", layers=[lo, hi], scheduler_pids=pids)
                    return child, pids, url
            except (OSError, TimeoutError):
                pass  # only readiness connection failures retry; identity failures raise
            time.sleep(2)
        raise TimeoutError("calibration server did not become ready")


def identity_for(args):
    files = ["work/qwen38fn_dflash/feed_int3_mm_calib.py", "work/qwen38fn_dflash/int3_mm_calib_runtime.py",
             "work/qwen38fn_dflash/capture_tito.py", "work/qwen38fn_dflash/capture_tito_runtime.py",
             "work/qwen38fn_eval/gptq_hess_hook.py", "work/qwen38fn_eval/direct_checkpoint.py",
             "work/qwen38fn_eval/quantize_int3_gptq.py", "work/qwen38fn_eval/gptq_int3.py",
             "work/qwen38fn_eval/run_gean_int3_mm.py"]
    mask_sha, mtp_sha = digest(args.mask), digest(args.mtp_mask)
    config = json.loads((args.fp8_model / "config.json").read_text())["text_config"]
    manifest = json.loads((args.fp8_model / "prune_manifest.json").read_text())
    if (config["num_hidden_layers"], config["num_experts"], config["num_experts_per_tok"]) != (48, 320, 10):
        raise RuntimeError("this orchestrator requires gEAN320 / top-10 / 48 layers")
    if manifest["mask_sha256"] != mask_sha or manifest["mtp_mask_sha256"] != mtp_sha:
        raise RuntimeError("staged FP8 model's masks differ from this quantization run")
    commit = subprocess.run(["git", "-C", str(args.fork), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True, timeout=20).stdout.strip()
    if commit != args.fork_sha:
        raise RuntimeError("fork HEAD differs from the requested immutable SHA")
    return {"run_id": args.run_id, "mask_sha256": mask_sha, "mtp_mask_sha256": mtp_sha,
            "plan_sha256": digest(args.plan), "fork_sha": args.fork_sha, "bf16_source": str(args.bf16_source),
            "fp8_model": str(args.fp8_model), "fp8_config_sha256": digest(args.fp8_model / "config.json"),
            "prune_manifest_sha256": digest(args.fp8_model / "prune_manifest.json"),
            "code": {name: digest(args.source_root / name) for name in files},
            "window": args.window, "target_tokens": args.target_tokens, "gpus": args.gpus,
            "resident_cache_only": args.resident_cache_only,
            "ple_mm_placeholder_ids": True, "method": "one-shot"}


def run(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.ram_dir.mkdir(parents=True, exist_ok=True)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_for(args)
    state_path = args.run_dir / "state.json"
    if args.resume:
        if not state_path.exists() or json.loads(state_path.read_text())["identity"] != identity:
            raise RuntimeError("resume requires this exact run identity")
    elif state_path.exists() or list(args.parts_dir.iterdir()):
        raise RuntimeError("existing run / quantization parts require explicit --resume")
    (args.run_dir / "logs").mkdir(exist_ok=True)
    atomic(args.run_dir / "identity.json", identity)
    runner = Runner(args, identity)
    uuids = gpu_uuids(args.gpus)
    processed = 0
    try:
        for lo in range(0, 48, args.window):
            hi = min(48, lo + args.window)
            layers = list(range(lo, hi))
            name = f"w{lo:02d}"
            receipt_path = args.run_dir / f"{name}-complete.json"
            window = args.ram_dir / name
            if receipt_path.exists():
                prior = json.loads(receipt_path.read_text())
                validate_resume(prior, identity, args.parts_dir, layers)
                if digest(args.run_dir / f"{name}-calib/complete.json") != prior["calibration_sha256"]:
                    raise RuntimeError("resume calibration complete receipt differs from the verified window")
                if window.exists():
                    cleanup_window(window, args.run_id, name)
                runner.event("resume_verified", layers=layers)
                continue
            if window.exists() or any((args.parts_dir / f"layer-{layer:02d}.json").exists() or
                                      (args.parts_dir / f"int3-L{layer:02d}.safetensors").exists() for layer in layers):
                raise RuntimeError("incomplete window exists; preserve and inspect it before retrying")
            busy = [(pid, uuid) for pid, uuid in gpu_apps() if uuid in uuids]
            if busy:
                raise RuntimeError(f"requested GPU has an existing process: {busy}")
            window.mkdir()
            atomic(window / "owner.json", {"run_id": args.run_id, "window": name})
            (window / "hessian").mkdir()
            runner.event("calibration_start", layers=layers)
            server, pids, url = runner.server(lo, hi, window)
            feed_out = args.run_dir / f"{name}-calib"
            feed = runner.start(f"{name}-feed", [args.python, args.source_root / "work/qwen38fn_dflash/feed_int3_mm_calib.py",
                "--plan", args.plan, "--ledger-root", args.ledger_root, "--model-dir", args.fp8_model,
                "--recorded-config-dir", args.recorded_config_dir, "--capture-root", window / "capture",
                "--hess-dir", window / "hessian", "--out", feed_out, "--url", url, "--run-id", args.run_id,
                "--scheduler-pids", *map(str, pids), "--target-tokens", str(args.target_tokens),
                "--timeout", str(args.phase_timeout)])
            runner.wait(feed, args.phase_timeout)
            calibration = json.loads((feed_out / "complete.json").read_text())
            if not calibration["complete"] or calibration["run_id"] != args.run_id:
                raise RuntimeError("MM calibration did not produce this run's complete receipt")
            runner.stop(server)
            runner.event("stage_bf16", layers=layers)
            staged = window / "bf16"
            stage = runner.start(f"{name}-stage", [args.python, args.source_root / "work/qwen38fn_eval/direct_checkpoint.py",
                "stage-bf16", "--src", args.bf16_source, "--out", staged, "--mask", args.mask,
                "--layers", f"{lo}:{hi}", "--stall-seconds", "120", "--free-floor-bytes", str(20 << 30)]
                + (["--resident-cache-only"] if args.resident_cache_only else []))
            runner.wait(stage, args.phase_timeout)
            staging = json.loads((staged / "manifest.json").read_text())
            if (not staging["complete"] or staging["mask_sha256"] != identity["mask_sha256"] or
                    {p["layer"] for p in staging["parts"]} != set(layers)):
                raise RuntimeError("BF16 window staging did not complete with this mask/layer set")
            runner.event("quantization", layers=layers)
            workers = [runner.start(f"{name}-quant-gpu{gpu}", [args.python,
                args.source_root / "work/qwen38fn_eval/quantize_int3_gptq.py", "--src", args.bf16_source,
                "--source-parts", staged, "--mask", args.mask, "--hess-dir", window / "hessian",
                "--parts", args.parts_dir, "--run-id", args.run_id, "--layers", f"{lo}:{hi}",
                "--worker-id", str(worker), "--world-size", "4"], {"CUDA_VISIBLE_DEVICES": str(gpu)})
                for worker, gpu in enumerate(args.gpus)]
            for worker in workers:
                runner.wait(worker, args.phase_timeout)
            verified = args.run_dir / f"{name}-verified.json"
            check = runner.start(f"{name}-verify", [args.python, Path(__file__).resolve(), "verify",
                "--mask", args.mask, "--parts-dir", args.parts_dir, "--run-id", args.run_id,
                "--layers", f"{lo}:{hi}", "--out", verified])
            runner.wait(check, args.phase_timeout)
            verification = json.loads(verified.read_text())
            if not verification["complete"] or verification["run_id"] != args.run_id:
                raise RuntimeError("quantization verification is incomplete")
            meta_dir = args.run_dir / f"{name}-hessian-meta"
            meta_dir.mkdir()
            for file in (window / "hessian").glob("hess_rank*_*.json"):
                shutil.copyfile(file, meta_dir / file.name)
            receipt = {"complete": True, "identity": identity, "layers": layers,
                       "calibration_sha256": digest(feed_out / "complete.json"), "parts": verification["parts"]}
            atomic(receipt_path, receipt)
            cleanup_window(window, args.run_id, name)
            processed += 1
            runner.event("window_complete", layers=layers)
            if args.stop_after_windows and processed >= args.stop_after_windows and hi < 48:
                runner.state["status"] = "checkpoint"
                runner.event("stop_after_windows", completed_this_invocation=processed)
                return
        runner.state["status"] = "complete"
        runner.event("all_windows_complete", layers=48)
    except BaseException as exc:
        runner.state["status"] = "failed"
        runner.event("error", error=f"{type(exc).__name__}: {exc}")
        cleanup_errors = []
        for child in list(runner.children):
            try:
                runner.stop(child)
            except Exception as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if cleanup_errors:
            runner.event("cleanup_failed", errors=cleanup_errors)
        raise


def verify(args):
    import torch
    from safetensors import safe_open

    drop = torch.load(args.mask, map_location="cpu", weights_only=False)["drop"]
    lo, hi = map(int, args.layers.split(":"))
    records = []
    for layer in range(lo, hi):
        path = args.parts_dir / f"int3-L{layer:02d}.safetensors"
        report_path = args.parts_dir / f"layer-{layer:02d}.json"
        report = json.loads(report_path.read_text())
        if report["layer"] != layer or report["run_id"] != args.run_id or report["format"] != "humming-uint3-g128-nozp":
            raise RuntimeError("quantization report identity / format differs from requested run")
        finite(report)
        ids = (~drop[layer]).nonzero().flatten().to(torch.int32)
        with safe_open(path, framework="pt", device="cpu") as f:
            if f.metadata()["run_id"] != args.run_id or not torch.equal(f.get_tensor("orig_expert_ids"), ids):
                raise RuntimeError("quantized original expert IDs / run ID differ from mask")
            for bank in ("gate_up", "down"):
                scale = f.get_tensor(f"{bank}.scale")
                codes_shape = f.get_slice(f"{bank}.codes").get_shape()
                if scale.dtype != torch.bfloat16 or not torch.isfinite(scale).all() or scale.shape[0] != len(ids):
                    raise RuntimeError("invalid quantization scales")
                if codes_shape[:2] != list(scale.shape[:2]) or codes_shape[2] * 2 != scale.shape[2] * 128:
                    raise RuntimeError("quantization code/scale shapes disagree")
        records.append({"layer": layer, "tensor_sha256": digest(path), "report_sha256": digest(report_path),
                        "orig_expert_ids": ids.tolist(), "bytes": path.stat().st_size})
    atomic(args.out, {"complete": True, "run_id": args.run_id, "parts": records})


def cleanup_window(path, run_id, name):
    if path.is_symlink() or json.loads((path / "owner.json").read_text()) != {"run_id": run_id, "window": name}:
        raise RuntimeError("window ownership changed; refusing cleanup")
    shutil.rmtree(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    for name in ("source-root", "fork", "python", "fp8-model", "bf16-source", "mask", "mtp-mask", "plan",
                 "ledger-root", "recorded-config-dir", "run-dir", "ram-dir", "parts-dir"):
        run_p.add_argument("--" + name, type=Path, required=True)
    run_p.add_argument("--fork-sha", required=True)
    run_p.add_argument("--run-id", required=True)
    run_p.add_argument("--port", type=int, required=True)
    run_p.add_argument("--gpus", type=lambda raw: [int(x) for x in raw.split(",")], default=[0, 1, 2, 3])
    run_p.add_argument("--window", type=int, default=6)
    run_p.add_argument("--target-tokens", type=int, default=2_000_000)
    run_p.add_argument("--server-timeout", type=float, default=1800)
    run_p.add_argument("--phase-timeout", type=float, default=7200)
    run_p.add_argument("--stop-after-windows", type=int, default=0)
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--resident-cache-only", action="store_true",
                       help="stage BF16 only from ranges mincore verifies are already resident")
    verify_p = sub.add_parser("verify")
    for name in ("mask", "parts-dir", "out"):
        verify_p.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "layers"):
        verify_p.add_argument("--" + name, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "verify":
        verify(args)
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or not re.fullmatch(r"[0-9a-f]{40}", args.fork_sha):
        raise ValueError("valid run id and full immutable fork SHA are required")
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4 or args.window <= 0 or 48 % args.window:
        raise ValueError("four distinct GPUs and a window dividing 48 are required")
    if not 0 < args.port < 65536 or args.target_tokens <= 0 or min(args.server_timeout, args.phase_timeout) <= 0:
        raise ValueError("invalid port / budget / timeout")
    if args.stop_after_windows < 0:
        raise ValueError("stop-after-windows cannot be negative")
    for name in ("source_root", "fork", "fp8_model", "mask", "mtp_mask", "plan", "ledger_root", "recorded_config_dir"):
        path = require_local(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    # Keep the virtualenv entry path; resolving its symlink loses pyvenv.cfg.
    args.python = Path(os.path.abspath(args.python))
    require_local(args.python)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    args.bf16_source = Path(os.path.abspath(args.bf16_source))  # no buffered/stat NFS access
    for name in ("run_dir", "ram_dir", "parts_dir"):
        setattr(args, name, require_local(getattr(args, name)))
    def interrupted(signum, _frame):
        raise InterruptedError(f"orchestrator received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
