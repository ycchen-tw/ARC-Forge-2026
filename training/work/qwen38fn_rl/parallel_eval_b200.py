"""Run the planned fresh base and retained checkpoints on up to four exclusive TP2 backends."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
from urllib.parse import urlsplit

import baseline_b200 as baseline
import heldout_offline as offline
from orchestrator import Cancelled, OrchestratorError, atomic_write, http_ok, read_episodes, summarize, utc_now


def sorted_versions(text):
    versions = offline.parse_versions(text)
    if versions != sorted(versions):
        raise argparse.ArgumentTypeError("versions must be sorted and unique")
    return versions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", action="append", required=True, help="direct native /v1 URL; primary first")
    parser.add_argument("--versions", type=sorted_versions, required=True)
    parser.add_argument("--samples", type=offline.positive(int), default=4)
    parser.add_argument("--harness-config", required=True)
    parser.add_argument("--parent-lock-fd", type=int, help="inherited run-lock descriptor from postrun's direct parent")
    args = parser.parse_args(argv)
    args.timeout_seconds = 14400  # the existing matched baseline/evaluation limit
    if not 1 <= len(args.backend) <= 4 or len(set(args.backend)) != len(args.backend):
        parser.error("provide 1–4 distinct direct backend URLs")
    for backend in args.backend:
        url = urlsplit(backend)
        if (url.scheme not in {"http", "https"} or not url.netloc or url.path != "/v1"
                or url.query or url.fragment or url.username or url.password):
            parser.error(f"invalid direct backend URL: {backend!r}")
    return args


def schedule(versions, backends):
    queues = [list(versions) if backends == 1 else [versions[-1]]] + [[] for _ in range(backends - 1)]
    if backends > 1:
        for index, version in enumerate(versions[:-1]):
            queues[1 + index % (backends - 1)].append(version)
    return queues


def run_lock(root, inherited_fd):
    if inherited_fd is None:
        return offline.hold_run_lock(root)
    expected, actual = (root / "orchestrator.lock").stat(), os.fstat(inherited_fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise OrchestratorError("inherited descriptor does not refer to this run's orchestrator.lock")
    lock = os.fdopen(os.dup(inherited_fd))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


class BaselineReady:
    """One completion shared by the baseline worker and cancellation handler."""
    def __init__(self):
        self.future = Future()
        self.lock = threading.RLock()

    def finish(self, value=None, error=None):
        with self.lock:
            if not self.future.done():
                if error is None:
                    self.future.set_result(value)
                else:
                    self.future.set_exception(error)

    def wait(self, orch):
        while True:
            orch.check_cancel()
            try:
                return self.future.result(timeout=0.2)
            except TimeoutError:
                if self.future.done():
                    return self.future.result()  # propagate a baseline operation's own TimeoutError


def run(args, *, install_signals=False):
    workers = [offline.build(argparse.Namespace(config=args.config, base_url=url,
                                               harness_config=args.harness_config)) for url in args.backend]
    root, out = workers[0].root, workers[0].root / "heldout_offline"
    queues = schedule(args.versions, len(workers))
    ready = BaselineReady()
    failures, failure_lock = [], threading.RLock()
    previous = {}
    report = {"status": "preflight", "started_utc": utc_now(), "config": str(args.config.absolute()),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "samples": args.samples,
              "schedule": [{"backend": url, "baseline": index == 0, "versions": queue}
                           for index, (url, queue) in enumerate(zip(args.backend, queues, strict=True))]}

    def cancel(error):
        with failure_lock:
            if not failures:
                failures.append(error)
        for orch in workers:
            orch.cancelled.set()
        ready.finish(error=error)

    def on_signal(signum, frame):
        cancel(Cancelled(f"signal {signum}"))

    def record(status, **details):
        report.update(status=status, **details)
        atomic_write(root / "parallel_evaluation.json", json.dumps(report, indent=1) + "\n")

    with run_lock(root, args.parent_lock_fd):
        if install_signals:
            previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        started_outputs = False
        try:
            if any(path.exists() for path in (root / "baseline-b32k", out, root / "parallel_evaluation.json")):
                raise OrchestratorError("parallel evaluation outputs already exist; no reuse or retry")
            for orch in workers:
                orch.static_checks(salted=False)
            published = {int(row["weight_version_out"]): row["publish"] for row in workers[0].load_history()}
            if any(version not in published for version in args.versions):
                raise OrchestratorError("a requested version has no publication record")
            merged = {version: offline.check_merged(version, published[version]) for version in args.versions}
            condition = baseline.condition_fingerprint(workers[0], args.samples)
            instances, base_paths = [], []
            for index, orch in enumerate(workers):
                if baseline.condition_fingerprint(orch, args.samples) != condition:
                    raise OrchestratorError("workers have different evaluation conditions")
                if not http_ok(orch.server_root + "/health"):
                    raise OrchestratorError(f"backend {orch.base_url} is not healthy")
                info = offline.server_identity(orch, orch.cfg["rollout"]["served_name"], "parallel preflight")
                if "replicas" in info or "replica_urls" in info:
                    raise OrchestratorError("parallel workers require direct native backends, not a control facade")
                offline.check_start_state(info, published, {v: m for v, (_, m) in merged.items()})
                if index == 0 and (str(info["weight_version"]) != "0" or info["weights_source"] != info["model_path"]):
                    raise OrchestratorError("primary backend must serve fresh base weights at version 0")
                orch.wait_idle("parallel preflight")
                instances.append(offline.ledger_status(orch, None, "parallel preflight")["instance"])
                base_paths.append(Path(info["model_path"]).resolve())
                orch.served_name = orch.cfg["rollout"]["served_name"]
            if len(set(instances)) != len(instances) or len(set(base_paths)) != 1:
                raise OrchestratorError("backends alias the same ledger or use different base models")
            report.update(condition_sha256=condition, ledger_instances=instances,
                          manifests={str(v): published[v]["manifest_sha256"] for v in args.versions})
            out.mkdir()
            started_outputs = True
            record("running")

            def check_condition(orch):
                if (baseline.condition_fingerprint(orch, args.samples) != condition
                        or hashlib.sha256(args.config.read_bytes()).hexdigest() != report["config_sha256"]):
                    raise OrchestratorError("evaluation condition/config changed during collection")

            def baseline_value(orch):
                value = ready.wait(orch)
                check_condition(orch)
                doc = json.loads((root / "baseline-b32k/summary.json").read_text())
                if doc["condition_sha256"] != condition or doc["run_dirs"] != value["run_dirs"]:
                    raise OrchestratorError("baseline provenance/condition changed before comparison")
                return value

            def worker(index):
                orch = workers[index]
                try:
                    if index == 0:
                        doc = baseline.collect(orch, args.samples)
                        check_condition(orch)
                        if doc["condition_sha256"] != condition:
                            raise OrchestratorError("fresh baseline condition differs")
                        episodes = read_episodes([Path(d) for d in doc["run_dirs"]], orch.pools["heldout"])
                        summary = summarize(episodes, args.samples)
                        ready.finish({"run_dirs": doc["run_dirs"], "samples": args.samples, "episodes": episodes,
                                      "summary": {k: v for k, v in summary.items() if k != "per_task_mean"}})
                    orch.log_file = (out / f"worker-{index}.log").open("a")
                    for version in queues[index]:
                        orch.check_cancel()
                        offline.check_merged(version, published[version])
                        offline.eval_version(orch, args, version, merged[version][0], published[version],
                                             orch.pools["heldout"], out / f"v{version:06d}", instances[index],
                                             lambda: baseline_value(orch))
                except BaseException as error:
                    cancel(error)
                    raise
                finally:
                    if orch.log_file is not None:
                        orch.log_file.close()
                        orch.log_file = None

            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                try:
                    futures = [pool.submit(worker, index) for index in range(len(workers))]
                    for future in as_completed(futures):
                        future.result()  # executor joins every cancelled worker before run-lock release
                except BaseException as error:
                    cancel(error)
                    raise
            check_condition(workers[0])
            for version in args.versions:
                doc = json.loads((out / f"v{version:06d}/summary.json").read_text())
                if (doc["version"] != version or doc["merged"]["manifest_sha256"] != published[version]["manifest_sha256"]
                        or doc["samples"] != args.samples or Path(doc["run_root"]).resolve() != root.resolve()):
                    raise OrchestratorError(f"version {version}: committed summary provenance mismatch")
            record("done", finished_utc=utc_now(), versions=args.versions)
            return report
        except BaseException as error:
            cancel(error)
            error = failures[0]
            if started_outputs:
                record("failed", finished_utc=utc_now(), error=repr(error))
            raise error
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    try:
        run(parse_args(argv), install_signals=True)
    except Cancelled:
        return 3
    except (OrchestratorError, OSError, ValueError) as error:
        print(f"parallel evaluation failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
