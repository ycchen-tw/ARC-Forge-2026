"""Delete local live-ledger routing shards no trainable episode can read (runs without an NFS offloader).

Routing is ~99% of a ledger (960 B/token) and grows 40-60 GB/hour on the /dev/shm replicas; nothing keeps it after
training except the RL batches prepare_rl already wrote. Every --interval seconds this runs routing_release.release()
(covered + consumed, see its docstring: trained in batches.jsonl, abandoned in groups.jsonl, set aside under
groups/abandoned-* or groups/quarantined/, or a held-out group of the stream loop with its row in heldout.jsonl; pooled
groups, prefetched/pending batches and groups returned to the pool by STOP/resume are not consumed) and unlinks the shards
it lists. Finished groups whose sessions nothing names (release's ``unnamed``) keep their shards: every pass line and
status.json count them and status.json lists them -- look at them, they are not released by waiting. Who reads routing, and why a released shard is never read again:
  * prepare_rl (live, prefetch, resume re-prepare) loads only the threads of its batch's manifests
    (load_instance(threads=...)): every one of those groups is unconsumed, so every shard holding any of its appends
    is kept;
  * the orchestrator's ledger audit reconstructs only the manifest threads when audit.command is ``-m arcforge.tito``
    (it then passes --manifest-threads-only); a full-ledger audit would check every append's routing file and fail
    on a released shard, so this script refuses to start unless config.json's audit.command takes that path;
  * nothing else in the run reads routing (heldout/baseline use only /session_ledger/status counters; the stream loop's
    held-out groups are never audited or prepared).
A released shard is gone for good: a later full-ledger reader (audit without --manifest-threads-only, export-rl /
export-dflash over the whole instance) reports routing_file for those appends; <run>/routing_gc/deleted.jsonl names
every removed shard. A shard that is a symlink was moved by the NFS offloader: the two must not run together (raise).

  routing_gc.py --run-root R [--interval 60] [--once]
Writes R/routing_gc/: routing_release.json (+ its state, read incrementally), deleted.jsonl (a "pass" line with the
planned shards before unlinking, one "deleted" line per shard after), status.json. Stops when R/routing_gc/STOP exists.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_release import release  # noqa: E402

THREADS_ONLY_AUDIT = ["-m", "arcforge.tito"]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, doc: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(doc) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_config(run_root: Path) -> None:
    """The orchestrator's audit must be the manifest-threads-only one (orchestrator.audit_with_status adds the flag
    exactly when audit.command ends with ``-m arcforge.tito``)."""
    command = json.loads((run_root / "config.json").read_text())["audit"]["command"]
    if command[-2:] != THREADS_ONLY_AUDIT:
        raise SystemExit(f"{run_root / 'config.json'}: audit.command {command!r} does not end with {THREADS_ONLY_AUDIT}: "
                         f"the orchestrator would audit the whole ledger and read the routing of every append, "
                         f"including shards this script deletes")


def gc_pass(run_root: Path, out: Path, totals: dict) -> dict:
    report = release(run_root, out / "routing_release_state.json")
    write_json(out / "routing_release.json", report)
    planned = []
    for inst, shards in sorted(report["instances"].items()):
        for n in shards:
            path = Path(report["dirs"][inst]) / f"routing.{n}.bin"
            if path.is_symlink():
                raise RuntimeError(f"{path} is a symlink (moved by the NFS offloader): do not run routing_gc and the "
                                   f"offload stage on the same run")
            if not path.is_file():
                raise RuntimeError(f"{path}: released by routing_release but not a regular file")
            planned.append((inst, n, path))
    log = out / "deleted.jsonl"
    if planned:
        append_jsonl(log, {"event": "pass", "utc": utc(), "planned": [f"{inst}/routing.{n}.bin" for inst, n, _ in planned]})
    freed = 0
    for inst, n, path in planned:
        size = path.stat().st_size
        path.unlink()
        freed += size
        append_jsonl(log, {"event": "deleted", "utc": utc(), "instance": inst, "shard": n, "path": str(path),
                           "bytes": size})
    totals["files"] += len(planned)
    totals["bytes"] += freed
    totals["passes"] += 1
    pending = {inst: len(why) for inst, why in report["pending"].items()}
    status = {"state": "pass_complete", "utc": utc(), "time": time.time(), "deleted_this_pass": len(planned),
              "bytes_this_pass": freed, "totals": dict(totals), "consumed_groups": report["consumed_groups"],
              "aside_groups": report["aside_groups"], "heldout_groups": report["heldout_groups"],
              "unnamed_groups": len(report["unnamed"]), "unnamed": report["unnamed"], "covered": report["covered"],
              "kept_shards": pending}
    write_json(out / "status.json", status)
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval must be positive")
    run_root = args.run_root
    check_config(run_root)
    out = run_root / "routing_gc"
    out.mkdir(exist_ok=True)
    lock = (out / "loop.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out / 'loop.lock'} is held: another routing_gc runs on this run") from None
    totals = {"files": 0, "bytes": 0, "passes": 0}
    while not (out / "STOP").exists():
        try:
            status = gc_pass(run_root, out, totals)
        except BaseException as e:
            write_json(out / "status.json", {"state": "failed", "utc": utc(), "error": repr(e), "totals": totals})
            raise
        print(json.dumps({k: status[k] for k in ("utc", "deleted_this_pass", "bytes_this_pass", "totals", "unnamed_groups")}),
              flush=True)
        if args.once:
            return
        time.sleep(args.interval)
    write_json(out / "status.json", {"state": "stopped", "utc": utc(), "totals": totals})
    print(f"{out / 'STOP'} exists: stopped", flush=True)


if __name__ == "__main__":
    main()
