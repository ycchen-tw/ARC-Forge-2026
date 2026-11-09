"""Own the producer/consumer lifetime, bounded queue and one continuous epoch.

The run uses frozen copies of this unit and SGLang's Python source. Failures
stop only the process groups this supervisor created. Resume starts from the
last durable optimizer checkpoint and regenerates disposable GR features.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def atomic_json(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def tail_json(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 16384))
        raw = stream.read()
    if b"\n" not in raw:
        return None
    complete = raw.rsplit(b"\n", 1)[0]
    return json.loads(complete.rsplit(b"\n", 1)[-1]) if complete else None


def start_owned(command, log, env):
    with log.open("ab") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    process.arc_process_group = process.pid
    return process


def stop_owned(process, grace=30):
    # Child group is created with start_new_session=True, never inherited from
    # an interactive shell/tmux or another user's job.
    group = process.arc_process_group
    if group != process.pid:
        raise RuntimeError("invalid owned process group")
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        process.wait()
        return
    if process.poll() is None and os.getpgid(process.pid) != group:
        raise RuntimeError("refusing to signal a group not owned by this launcher")
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
            return
    process.wait()


def check_checkpoint_room(out, init):
    """Fail the launch, not the first checkpoint hours later.

    `save_state` needs `trainable * 12` bytes free with the previous generation
    still on disk -- two generations coexist until the new one is published --
    so a run needs room for both plus the 8 GiB reserve. Compression takes the
    written size to roughly 0.9x, which is why this is the conservative bound.
    """
    trainable = json.loads((init / "initialization_manifest.json").read_text())["trainable_parameters"]
    need = trainable * 12
    required = 2 * need + 8 * 2**30
    # A resume already holds one published generation, and `save_state` reclaims
    # it once the next one lands, so that space is part of the two-generation
    # budget rather than something the filesystem still has to find.
    trainer = out / "trainer"
    published = (
        sum(f.stat().st_size for d in trainer.glob("step-*") if d.is_dir() for f in d.iterdir() if f.is_file())
        if trainer.is_dir()
        else 0
    )
    free = shutil.disk_usage(out).free
    if free + published < required:
        raise RuntimeError(
            f"checkpoints need {required / 2**30:.1f} GiB for two generations of "
            f"{trainable / 1e9:.2f}B FP32 masters and Adam moments; {out} has "
            f"{free / 2**30:.1f} GiB free plus {published / 2**30:.1f} GiB already published"
        )


def check_resources(port, devices):
    for device in devices:
        used = int(
            subprocess.check_output(
                ["nvidia-smi", "-i", str(device), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
            ).strip()
        )
        if used > 0:
            raise RuntimeError(f"GPU{device} occupied: {used} MiB")
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    finally:
        sock.close()
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    if available < 256 * 2**30:
        raise RuntimeError("need 256 GiB available host RAM for lossless checkpoint resume")


def validate_source(unit, gates):
    for name, digest in gates["source_hashes"].items():
        if hashlib.sha256((unit / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed after correctness gates: {name}")


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "init", "data", "gates", "out", "server-python", "server-source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=30648)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    # Shared machine: the split is explicit rather than wired in, because the
    # trainer width follows the draft depth. Three layers fit two ranks; five
    # need four, and the target TP size is just how many cards it was given.
    p.add_argument("--target-gpus", default="0,1,2,3")
    p.add_argument("--trainer-gpus", default="4,5,6,7")
    args = p.parse_args()
    target_gpus = args.target_gpus.split(",")
    trainer_gpus = args.trainer_gpus.split(",")
    if set(target_gpus) & set(trainer_gpus):
        raise RuntimeError("the target and the trainer cannot share a GPU")
    args.out = args.out.resolve()
    gates = json.loads(args.gates.read_text())
    if not gates["production_ready"] or gates["attention_backend"] != "stable" or gates["compute_dtype"] != "bfloat16":
        raise RuntimeError("production requires the completed stable BF16 correctness gates")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != gates["dataset_manifest_sha256"]:
        raise RuntimeError("dataset changed after gate")
    if (
        hashlib.sha256((args.init / "initialization_manifest.json").read_bytes()).hexdigest()
        != gates["init_manifest_sha256"]
    ):
        raise RuntimeError("initialization changed after gate")
    unit = Path(__file__).resolve().parent
    if not args.resume:
        validate_source(unit, gates)
        args.out.mkdir(parents=True, exist_ok=False)
    elif not args.out.is_dir():
        raise FileNotFoundError(args.out)
    lock = (args.out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_resources(args.port, target_gpus + trainer_gpus)
    check_checkpoint_room(args.out, args.init)
    source = args.out / "source"
    server_source = args.out / "server_python"
    queue = args.out / "queue"
    if not args.resume:
        shutil.copytree(unit, source, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "results"))
        shutil.copytree(args.server_source, server_source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        queue.mkdir()
        shutil.copyfile(args.gates, args.out / "gates.json")
    validate_source(source, gates)
    trainer = args.out / "trainer"
    start_step = 0
    resume = None
    if args.resume:
        latest = (trainer / "latest").read_text().strip()
        if Path(latest).name != latest or not latest.startswith("step-"):
            raise ValueError("invalid checkpoint pointer")
        resume = trainer / latest
        start_step = json.loads((resume / "manifest.json").read_text())["step"]
        for partial in trainer.glob("step-*.partial"):
            shutil.rmtree(partial)
        if (trainer / "STOP").exists():
            (trainer / "STOP").unlink()
        for directory in queue.glob("train-*"):
            shutil.rmtree(directory)
        for name in ("active.json", "active.partial", "train.producer_done"):
            path = queue / name
            if path.exists():
                path.unlink()
    rows = gates["train_rows"]
    # One epoch is ceil(rows / world): every rank consumes its own row per step.
    # train.py asserts this exact relation, so a stale divisor here fails the
    # launch rather than silently training a fraction of the data.
    total_steps = math.ceil(rows / len(trainer_gpus))
    if start_step >= total_steps:
        raise ValueError("epoch already complete")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = args.out / f"logs-{attempt}"
    logs.mkdir()
    children = {}
    state = {
        "stage": "starting",
        "started_utc": attempt,
        "start_step": start_step,
        "total_steps": total_steps,
        "gpus": {"target": [int(g) for g in target_gpus], "trainer": [int(g) for g in trainer_gpus]},
        "queue_limit_gib": 4,
        "continuous_epochs": 1,
        "target_checkpoint_sha256": gates["target_checkpoint_sha256"],
    }
    atomic_json(args.out / "status.json", state)

    stopping_since = None

    def interrupted(signum, frame):
        nonlocal stopping_since
        if "trainer" in children and trainer.is_dir():
            (trainer / "STOP").touch()
            stopping_since = time.monotonic()
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    try:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=args.target_gpus,
            PYTHONPATH=str(server_source),
            MTP3_CAPTURE_ROOT=str(queue),
            SGLANG_QWEN4_MNNVL_HC="0",
            PATH=str(args.server_python.parent) + os.pathsep + os.environ["PATH"],
        )
        command = [
            str(args.server_python),
            str(source / "capture_runtime.py"),
            "--model-path",
            str(args.model),
            "--tp",
            str(len(target_gpus)),
            "--ep",
            str(len(target_gpus)),
            "--context-length",
            "65536",
            "--mem-fraction-static",
            ".78",
            "--max-running-requests",
            "1",
            "--max-mamba-cache-size",
            "8",
            "--chunked-prefill-size",
            "4096",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--linear-attn-prefill-backend",
            "flashinfer",
            "--linear-attn-decode-backend",
            "triton",
            "--mamba-ssm-dtype",
            "float32",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        children["target"] = start_owned(command, logs / "target.log", env)
        state["children"] = {"target": {"pid": children["target"].pid, "returncode": None}}
        atomic_json(args.out / "status.json", state)
        url = f"http://127.0.0.1:{args.port}"
        deadline = time.monotonic() + 600
        while True:
            if children["target"].poll() is not None:
                raise RuntimeError("target exited during startup; inspect target.log")
            try:
                ready = requests.get(url + "/health", timeout=2).status_code == 200
            except (requests.ConnectionError, requests.Timeout):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("target readiness deadline")
            time.sleep(1)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
        children["producer"] = start_owned(
            [
                sys.executable,
                str(source / "capture.py"),
                "--index",
                str(args.data.parent / "rows.sqlite"),
                "--queue",
                str(queue),
                "--url",
                url,
                "--split",
                "train",
                "--offset",
                # One row per rank per step, so a resume skips step * world rows.
                str(start_step * len(trainer_gpus)),
                "--max-queue-gib",
                "4",
            ],
            logs / "producer.log",
            env,
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.trainer_gpus, OMP_NUM_THREADS="8")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(trainer_gpus)}",
            str(source / "train.py"),
            "--checkpoint",
            str(args.init),
            "--queue",
            str(queue),
            "--out",
            str(trainer),
            "--dataset-manifest",
            str(args.data),
            "--steps",
            str(total_steps),
            "--total-rows",
            str(rows),
            "--anchors",
            "512",
            "--attention-backend",
            "stable",
            "--gate-report",
            str(args.out / "gates.json"),
            "--save-every",
            str(args.save_every),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        children["trainer"] = start_owned(command, logs / "trainer.log", env)
        state["stage"] = "training"
        while True:
            codes = {name: process.poll() for name, process in children.items()}
            state["children"] = {
                name: {"pid": process.pid, "returncode": codes[name]} for name, process in children.items()
            }
            state["last_metric"] = tail_json(trainer / "metrics.jsonl")
            state["elapsed_seconds"] = time.monotonic() - started
            state["free_disk_gib"] = shutil.disk_usage(args.out).free / 2**30
            atomic_json(args.out / "status.json", state)
            if codes["trainer"] == 0:
                result = json.loads((trainer / "run-result.json").read_text())
                state["stage"] = "complete" if result["completed"] else "paused"
                state["run_result"] = result
                break
            if stopping_since is not None and time.monotonic() - stopping_since > 300:
                raise KeyboardInterrupt
            if codes["trainer"] is not None or codes["target"] is not None or codes["producer"] not in (None, 0):
                raise RuntimeError(f"pipeline child failure: {codes}; retained durable checkpoint and logs")
            time.sleep(10)
    except KeyboardInterrupt:
        state["stage"] = "interrupted"
    except BaseException as exc:
        state.update(stage="failed", error=repr(exc))
        raise
    finally:
        for name in ("trainer", "producer", "target"):
            if name in children:
                stop_owned(children[name])
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        state["children"] = {
            name: {"pid": process.pid, "returncode": process.poll()} for name, process in children.items()
        }
        atomic_json(args.out / "status.json", state)


if __name__ == "__main__":
    main()
