"""Offline held-out for an RL run: sync each merged weight version the run published onto one rollout server, run the same tasks as
the async loop's held-out, and compare per task, paired, against base24. The stream loop (SPEC §9) has no held-out; this evaluates
the versions kept by ``stream.eval_keep_every``; sync / async runs work too.

    python3 work/qwen38fn_rl/heldout_offline.py --config <run config> --versions 3,6,9 --base-url http://127.0.0.1:PORT/v1 \\
        [--harness-config work/qwen38fn_rl/configs/harness-b8k-c64.json] [--samples 8] [--timeout-seconds 3600] \\
        [--out <run_root>/heldout_offline] [--baseline <sample run dir> ...]

server: started from the FP8 base (``launch_rollout_server.sh``, ledger on, ``RL_SERVE_ENV=MAXREQ=512``). INT3 QAT run (config has
``rollout.trainer_base_path``): the server is the gEAN320 deployment package (``RL_MODEL_KIND=int3``, TP1), merge format v2, with the
merge base checked against ``trainer_base_path``; to run several servers in parallel and rerun base, use
``heldout_offline_multi.py``. Before starting, without touching the server: hold ``<run_root>/orchestrator.lock`` (not obtainable =
this run's orchestrator is still alive, do not run; once obtained it is held until the end, and no orchestrator can start meanwhile),
config and task lists, the held-out harness config (unsalted), a publish record in rounds.jsonl / batches.jsonl and a complete merged
dir for every version, base24 results for every held-out task in every run, and no output dir existing yet. Then check the server: it
serves only the config's ``rollout.served_name``, no poison, weights are base (version 0) or some version of this run, base has the
same name as the base the trainer merged against, ledger writer healthy.
For each version k: idle → ``update_weights_from_disk`` (flush_cache, keep_identity; exactly the orchestrator's sync, checks
/model_info) → ``/flush_cache`` → ``samples`` run_arc2.py processes (``<out>/v{k:06d}/sK``; argv, stagger and timeout identical to
the async held-out) → version, weight source and ledger instance all unchanged → ``<out>/v{k:06d}/summary.json`` (summarize() and
heldout_compare's paired comparison).
Finally one line per version is printed. Anything wrong stops: no retry, no overwrite; a half-run version dir is left in place
(without summary.json) for a human to move aside. Standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from heldout_compare import compare
from orchestrator import (
    Cancelled,
    Orchestrator,
    OrchestratorError,
    _expand,
    atomic_write,
    cfg_path,
    http_json,
    http_ok,
    http_text,
    ledger_count,
    load_config,
    log_line,
    read_episodes,
    summarize,
    utc_now,
)

#: base24: 24 samples of the base weights on the same 64 held-out tasks (the calibration round and round-0 held-out of the sync / async
#: LR 1e-4 toys, s0..s7 each)
BASE24_ROOTS = ("$ARC_RUNS/qwen38fn_rl/20261007-calib-b8k/runs",
                "$ARC_RUNS/qwen38fn_rl/20261008-toy-b8k-lr1e4/rounds/r000/heldout",
                "$ARC_RUNS/qwen38fn_rl/20261008-toy-b8k-lr1e4-async/rounds/r000/heldout")
BASE24_SAMPLES = 8
#: harness config used by the async toy's held-out (all 64 tasks start at once, unsalted) and harness.round_timeout_seconds (stream
#: configs have neither)
DEFAULT_HARNESS_CONFIG = "work/qwen38fn_rl/configs/harness-b8k-c64.json"
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_SAMPLES = 8
#: work/qwen38fn_lora/merged_export.py: v1 = LoRA projections merged to BF16; v2 (trainer --dense-qat fp8) = the deployed-FP8 projections
#: written as FP8 + weight_scale_inv, the others BF16. One run's trainer writes one of them; the sync is the same call.
MERGED_FORMATS = ("fn-lora-merged-v1", "fn-lora-merged-v2")
LEDGER_KEYS = ("instance", "queue", "pending", "lossy", "writer_error", "enqueued_events", "written_events")


def parse_versions(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",")]
    if not all(p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError(f"--versions must be comma-separated integers, not {text!r}")
    versions = [int(p) for p in parts]
    if len(set(versions)) != len(versions) or min(versions) < 1:
        raise argparse.ArgumentTypeError(f"--versions must be distinct versions >= 1 (0 is the base: use base24), not {text!r}")
    return versions


def positive(kind):
    def parse(text: str):
        value = kind(text)
        if not value > 0:
            raise argparse.ArgumentTypeError(f"must be positive, not {text!r}")
        return value
    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the RL run's config (sync, async or stream)")
    ap.add_argument("--versions", type=parse_versions, required=True, help="weight versions to evaluate, in this order (3,6,9)")
    ap.add_argument("--base-url", required=True, help="the eval rollout server, ending in /v1 (FP8 base, ledger on)")
    ap.add_argument("--harness-config", default=DEFAULT_HARNESS_CONFIG,
                    help="held-out harness config, relative to the repo root like the config's harness.config (default: the "
                         "async toy's held-out config)")
    ap.add_argument("--samples", type=positive(int), default=DEFAULT_SAMPLES, help="sample runs per version")
    ap.add_argument("--timeout-seconds", type=positive(float), default=DEFAULT_TIMEOUT_SECONDS,
                    help="per version, for all sample runs (default: the async toy's harness.round_timeout_seconds)")
    ap.add_argument("--out", type=Path, help="output root (default <run_root>/heldout_offline); v<NNNNNN> must not exist")
    ap.add_argument("--baseline", type=Path, nargs="+", help="baseline sample run dirs (s<K>, results/<task>.json); default base24")
    args = ap.parse_args(argv)
    if not args.base_url.endswith("/v1"):
        ap.error(f"--base-url must end in /v1: {args.base_url!r}")
    return args


def base24_dirs() -> list[Path]:
    return [Path(_expand(root, "BASE24_ROOTS")) / f"s{k}" for root in BASE24_ROOTS for k in range(BASE24_SAMPLES)]


def build(args: argparse.Namespace) -> Orchestrator:
    """The run config's orchestrator, pointed at the offline evaluation server and switched to the held-out harness config; sync always
    uses keep_identity (the evaluation server is the rl fork, whether or not the run's server was legacy at the time)."""
    cfg = load_config(args.config)
    cfg["rollout"]["base_url"] = args.base_url
    cfg["harness"]["config"] = args.harness_config
    cfg["legacy_sync"] = False
    return Orchestrator(cfg)


def hold_run_lock(root: Path, *, shared: bool = False):
    """At start-up the orchestrator takes ``flock(LOCK_EX | LOCK_NB)`` on ``<run_root>/orchestrator.lock`` and holds it until it ends.
    The same flock is used here: not obtainable = it is still alive, raise; once obtained it is held (returns the open file; closing
    it releases). Opened for reading so the PID it wrote is not truncated.
    ``shared``: ``LOCK_SH``, so several evaluations that only read published weights, each with its own server (the two temperatures
    of arc3_version_eval.py), can hold it at once; the orchestrator's LOCK_EX still cannot be obtained."""
    path = root / "orchestrator.lock"
    if not path.is_file():
        raise OrchestratorError(f"{path} does not exist: {root} is not a run root the orchestrator has run")
    lock = open(path)
    try:
        fcntl.flock(lock, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise OrchestratorError(f"{path} is held (pid {path.read_text().strip()!r}): the orchestrator of this run is still "
                                f"live (or another evaluation holds it). Evaluate after it stops") from error
    return lock


def check_merged(version: int, publish: dict) -> tuple[Path, dict]:
    """The merged dir published for version ``version`` (the history row's ``publish``): ``merged/vNNNNNN``, ``DONE`` written
    (merged_export: readable only after DONE), manifest sha256 equal to the one reported at publish, format and version match.
    Returns (dir, manifest)."""
    path = Path(publish["path"])
    if path.name != f"v{version:06d}" or path.parent.name != "merged":
        raise OrchestratorError(f"version {version} was published at {path}, not a <trainer out>/merged/v{version:06d} directory")
    if not path.is_dir():
        raise OrchestratorError(f"{path}: version {version}'s merged weights are gone (retention keeps the last keep_merged "
                                f"versions and, under pipeline stream, the multiples of stream.eval_keep_every)")
    if not (path / "DONE").is_file():
        raise OrchestratorError(f"{path}: no DONE marker, the publish did not finish")
    raw = (path / "merged_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != publish["manifest_sha256"]:
        raise OrchestratorError(f"{path}/merged_manifest.json: sha256 differs from the one /publish returned for version "
                                f"{version} ({publish['manifest_sha256']}): the directory was replaced")
    manifest = json.loads(raw)
    if manifest["format"] not in MERGED_FORMATS or manifest["weight_version"] != str(version):
        raise OrchestratorError(f"{path}/merged_manifest.json: format {manifest['format']!r} weight_version "
                                f"{manifest['weight_version']!r}, want format in {MERGED_FORMATS} weight_version {str(version)!r}")
    return path, manifest


def server_identity(orch: Orchestrator, served: str, what: str) -> dict:
    """/v1/models has only ``served``, /model_info has the same name, no poison. Returns /model_info."""
    info = orch.model_info()
    names = orch.served_names()
    if names != [served] or info["served_model_name"] != served:
        raise OrchestratorError(f"{what}: server serves {names} (/model_info {info['served_model_name']!r}), config "
                                f"rollout.served_name is {served!r}")
    if info["poisoned"] is not False:
        raise OrchestratorError(f"{what}: server is poisoned (poisoned {info['poisoned']!r}); /model_info {info}")
    return info


def expect_weights(info: dict, version: int, merged: Path, what: str) -> None:
    if (str(info["weight_version"]), info["weights_source"]) != (str(version), str(merged)):
        raise OrchestratorError(f"{what}: server weight_version {info['weight_version']!r} from {info['weights_source']!r}, "
                                f"want {str(version)!r} from {str(merged)!r}")


def ledger_status(orch: Orchestrator, instance: str | None, what: str) -> dict:
    """/session_ledger/status: SPEC §8 counters present (rl fork), writer alive, no lost events, instance unchanged (server not
    restarted)."""
    status = http_json("GET", orch.server_root + "/session_ledger/status")
    ledger_count(status, "enqueued_events")
    ledger_count(status, "written_events")
    if status["writer_error"] is not None or status["lossy"] != 0:
        raise OrchestratorError(f"{what}: ledger writer_error {status['writer_error']!r}, lossy {status['lossy']!r}: the "
                                f"server's ledger is broken; restart the server")
    if instance is not None and status["instance"] != instance:
        raise OrchestratorError(f"{what}: ledger instance {status['instance']!r} != {instance!r}: the server restarted")
    return status


def check_start_state(info: dict, published: dict[int, dict], manifests: dict[int, dict],
                      trainer_base: str | None = None) -> None:
    """Before evaluating, the server's weights may only be base (version 0, source = model_path) or some version this run published
    (the merged tensor set is the same, and switching versions overwrites only those tensors); base has the same name as the base the
    trainer read when merging (merged weights replace only the LoRA projections, the rest is the server's base).
    ``trainer_base`` (the config's ``rollout.trainer_base_path``; QAT run: the trainer's BF16 master is not the deployment package the
    server loads): when given, the merge base must instead be this dir (the same as the orchestrator's trainer identity check), and
    the server's model_path name is not compared."""
    version, source = str(info["weight_version"]), info["weights_source"]
    from_base = version == "0" and source == info["model_path"]
    from_run = version.isdigit() and int(version) in published and source == published[int(version)]["path"]
    if not (from_base or from_run):
        raise OrchestratorError(f"server is at weight_version {version!r} from {source!r} (model_path {info['model_path']!r}): "
                                f"neither the base at version 0 nor a version this run published. Restart it from the base")
    for k, manifest in manifests.items():
        if trainer_base is not None:
            if Path(manifest["base_model"]).resolve() != Path(trainer_base).resolve():
                raise OrchestratorError(f"version {k} was merged onto {manifest['base_model']!r}, the run's declared trainer "
                                        f"base (rollout.trainer_base_path) is {trainer_base!r}")
        elif Path(manifest["base_model"]).name != Path(info["model_path"]).name:
            raise OrchestratorError(f"version {k} was merged onto {manifest['base_model']!r}, the server's base is "
                                    f"{info['model_path']!r}: serve the run's base (FP8)")


def evaluate(orch: Orchestrator, args: argparse.Namespace) -> list[dict]:
    lock = hold_run_lock(orch.root)
    try:
        return _evaluate(orch, args)
    finally:
        if orch.log_file is not None:
            orch.log_file.close()
            orch.log_file = None
        lock.close()


def _evaluate(orch: Orchestrator, args: argparse.Namespace) -> list[dict]:
    cfg = orch.cfg
    # ---- no network
    orch.static_checks(salted=False)                       # the held-out harness config: no cache salt, like async's held-out
    heldout = orch.pools["heldout"]
    published = {int(row["weight_version_out"]): row["publish"] for row in orch.load_history()}
    absent = [k for k in args.versions if k not in published]
    if absent:
        raise OrchestratorError(f"version(s) {absent}: no {orch.history_name} row published them "
                                f"(published: {sorted(published)})")
    merged = {k: check_merged(k, published[k]) for k in args.versions}
    # absolute: the harness runs with cwd = the repo root (Orchestrator.popen), this process reads its results from here
    out = args.out.absolute() if args.out is not None else orch.root / "heldout_offline"
    taken = [str(out / f"v{k:06d}") for k in args.versions if (out / f"v{k:06d}").exists()]
    if taken:
        raise OrchestratorError(f"output dir(s) {taken} exist; never overwritten (move them aside)")
    base_dirs = args.baseline if args.baseline is not None else base24_dirs()
    base_episodes = read_episodes(base_dirs, heldout)
    base_summary = summarize(base_episodes, len(base_dirs))
    # ---- the server
    served = cfg["rollout"]["served_name"]
    if not http_ok(orch.server_root + "/health"):
        raise OrchestratorError(f"rollout server {orch.server_root} /health is not 200")
    info = server_identity(orch, served, "start")
    check_start_state(info, published, {k: m for k, (_, m) in merged.items()}, cfg["rollout"].get("trainer_base_path"))
    instance = ledger_status(orch, None, "start")["instance"]
    orch.served_name = served
    out.mkdir(parents=True, exist_ok=True)
    orch.log_file = open(out / "heldout_offline.log", "a")
    orch.log(f"held-out offline: run {orch.root} ({cfg['pipeline']}), versions {args.versions}, server {orch.base_url} at "
             f"weight_version {info['weight_version']} (ledger {instance}), {len(heldout)} tasks x {args.samples} samples, "
             f"baseline {len(base_dirs)} samples, out {out}")
    base = {"run_dirs": [str(d) for d in base_dirs], "samples": len(base_dirs), "episodes": base_episodes,
            "summary": {k: v for k, v in base_summary.items() if k != "per_task_mean"}}
    docs = []
    for k in args.versions:
        docs.append(eval_version(orch, args, k, merged[k][0], published[k], heldout, out / f"v{k:06d}", instance, base))
    return docs


def eval_version(orch: Orchestrator, args: argparse.Namespace, k: int, merged: Path, publish: dict, heldout: list[str],
                 vd: Path, instance: str, base: dict | Callable[[], dict]) -> dict:
    cfg, served = orch.cfg, orch.served_name
    began, started_utc = time.monotonic(), utc_now()
    if not (merged / "DONE").is_file():
        raise OrchestratorError(f"{merged}: version {k}'s merged weights are gone or incomplete")
    ledger_before = ledger_status(orch, instance, f"v{k} start")
    # sync(r) loads version r + 1: idle -> update_weights_from_disk(flush_cache, keep_identity) -> verify /model_info -> idle
    synced = orch.sync(k - 1, {"path": str(merged)})
    expect_weights(server_identity(orch, served, f"v{k} after sync"), k, merged, f"v{k} after sync")
    flushed = http_text("POST", orch.server_root + "/flush_cache", timeout=cfg["rollout"]["sync_timeout_seconds"])
    sync_seconds = round(time.monotonic() - began, 1)
    vd.mkdir()                                              # FileExistsError if it appeared since the check: never overwrite
    (vd / "logs").mkdir()
    result = orch.sample_runs(vd, vd / "logs", "", cfg_path(cfg["tasks"]["heldout"]["dir"]), heldout, args.samples,
                              timeout=args.timeout_seconds, what=f"v{k:06d} held-out")
    after = server_identity(orch, served, f"v{k} after held-out")
    expect_weights(after, k, merged, f"v{k} after held-out (it changed while the held-out ran)")
    ledger_after = http_json("GET", orch.server_root + "/session_ledger/status")
    if ledger_after["instance"] != instance:
        raise OrchestratorError(f"v{k} after held-out: ledger instance {ledger_after['instance']!r} != {instance!r}: "
                                f"the server restarted while the held-out ran")
    episodes = read_episodes([Path(d) for d in result["run_dirs"]], heldout)
    rollout_finished_utc = utc_now()
    rollout_seconds = round(time.monotonic() - began - sync_seconds, 1)
    rollout_total_seconds = round(time.monotonic() - began, 1)
    if callable(base):
        base = base()  # parallel coordinator waits for the fresh baseline, with cancellation checks
    doc = {"version": k, "weight_version": str(k), "config": str(args.config.absolute()), "pipeline": cfg["pipeline"],
           "run_root": str(orch.root), "merged": {"path": str(merged), "manifest_sha256": publish["manifest_sha256"]},
           "harness_config": str(cfg_path(cfg["harness"]["config"])), "tasks_dir": str(cfg_path(cfg["tasks"]["heldout"]["dir"])),
           "tasks": len(heldout), "samples": args.samples, "timeout_seconds": args.timeout_seconds,
           "launch_stagger_seconds": cfg["harness"]["launch_stagger_seconds"], "run_dirs": result["run_dirs"],
           "summary": result["summary"],
           "baseline": {key: base[key] for key in ("run_dirs", "samples", "summary")},
           "compare_base": compare(base["episodes"], episodes, base["samples"], args.samples),
           "server": {"base_url": orch.base_url, "served_name": served, "model_path": after["model_path"], "sync": synced,
                      "flush_cache": flushed.strip(), "ledger_before": {key: ledger_before[key] for key in LEDGER_KEYS},
                      "ledger_after": {key: ledger_after[key] for key in LEDGER_KEYS}},
           "started_utc": started_utc, "finished_utc": rollout_finished_utc,
           "seconds": {"sync": sync_seconds, "heldout": rollout_seconds, "total": rollout_total_seconds}}
    atomic_write(vd / "summary.json", json.dumps(doc, indent=1) + "\n")
    orch.log(table_line(doc))
    return doc


def table_line(doc: dict) -> str:
    r, b, t = (doc["compare_base"][key] for key in ("reward", "budget_hit", "output_tokens"))
    z = "-" if r["z"] is None else f"{r['z']:+.2f}"
    return (f"v{doc['version']:06d}  reward {r['b']:.4f} (base {r['a']:.4f})  diff {r['diff']:+.4f} se {r['se']:.4f} z {z}  "
            f"tasks up/down {r['tasks_up']}/{r['tasks_down']}  budget-hit {b['b']:.3f} ({b['diff']:+.3f})  "
            f"output tokens {t['b']:.0f} ({t['diff']:+.0f})  live {doc['summary']['live_groups']}/{doc['tasks']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    orch = build(args)

    def on_signal(signum, frame):
        orch.log(f"signal {signum}: cancelling; the sample runs this process started are terminated")
        orch.cancelled.set()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        docs = evaluate(orch, args)
    except Cancelled:
        print(log_line("cancelled"), flush=True)
        return 3
    except OrchestratorError as error:
        print(log_line(f"FAILED: {error}"), file=sys.stderr, flush=True)
        return 1
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    print(f"held-out offline vs baseline ({docs[0]['baseline']['samples']} samples per task), {docs[0]['tasks']} tasks x "
          f"{args.samples} samples per version:")
    for doc in docs:
        print(table_line(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
