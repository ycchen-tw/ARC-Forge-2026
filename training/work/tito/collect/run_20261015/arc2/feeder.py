#!/usr/bin/env python3
"""Batch feeder for arc2 collection (slimmed version until the outer tooling is done): keeps ≤ MAX_ACTIVE batches on one
replica (r0), and opens the next batch only when running episodes plus the new batch's CONC stay within TARGET_RUNNING and
the server's KV usage < KV_MAX. No new batches while the STOP file exists.

Queue (built on first start and saved to queue.json; afterwards only the cursor in state.json is read):
  unrun independent1000 → quality300 → nvarc_training (excluding those used on 10-06, family round-robin order) with 1 batch
  of training1000 / synth108 interleaved every 8 batches → nvarc_full (family round-robin order)
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

C = Path(os.environ["ARC_RUNS"]) / "tito/20261015-fp8-dflash-collect"   # source env/env.sh
A = C / "arc2"
POOLS = Path(os.environ["ARC_DATA"]) / "arc2_pools"
OLD_DONE = Path(os.environ["ARC_RUNS"]) / "tito/20261006-node1-collect/queue/arc2/done"
SERVER_LOG = C / "gate/server.log"
PORT = 30811
BATCH = 50
CONC = 32
MAX_ACTIVE = 3
TARGET_RUNNING = 64               # cap on running episodes after opening a new batch (a mature episode is about 0.01 KV; 88 of them reach 0.98)
KV_MAX = 0.70
MIN_FREE_GB = 120
LIMITS = A / "limits.json"        # if present, reread every loop {"target_running": .., "max_active": .., "kv_max": ..} (10-15 concurrency testing)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def ids(pool):
    return sorted(p.stem for p in (POOLS / pool).glob("*.json"))


def order(pool):
    return [line.strip() for line in (POOLS / f"{pool}.order.txt").read_text().splitlines() if line.strip()]


def build_queue():
    used_here = set()
    for d in A.iterdir():
        if d.is_dir() and (d / "STATE.json").exists():
            st = json.loads((d / "STATE.json").read_text())
            only = st["args"]["only"]
            if only in (None, "None"):
                raise RuntimeError(f"{d}: batch without --only")
            used_here |= set(only.split(","))
    old_nvarc = set()
    for f in OLD_DONE.glob("*.txt"):
        kv = dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)
        if kv["pool"] == "nvarc_training":
            old_nvarc |= set(kv["ids"].split(","))
    if len(old_nvarc) != 720:
        raise RuntimeError(f"expected 720 nvarc_training ids used on 10-06, got {len(old_nvarc)}")
    chunks = lambda pool, xs: [(pool, xs[i:i + BATCH]) for i in range(0, len(xs), BATCH)]
    q = chunks("independent1000", [t for t in ids("independent1000") if t not in used_here])
    q += chunks("quality300", ids("quality300"))
    nv = chunks("nvarc_training", [t for t in order("nvarc_training") if t not in old_nvarc])
    side = chunks("training1000", ids("training1000")) + chunks("synth108", ids("synth108"))
    mixed = []
    for i, b in enumerate(nv):
        mixed.append(b)
        if i % 8 == 7 and side:
            mixed.append(side.pop(0))
    q += mixed + side
    q += chunks("nvarc_full", order("nvarc_full"))
    return [{"pool": p, "ids": xs} for p, xs in q]


def active_units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "collect-arc2-*"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split()[0].removesuffix(".service") for line in out.splitlines() if line.strip()]


def running_episodes(unit):
    tag = unit.removeprefix("collect-arc2-")
    logf = A / f"{tag}.log"
    if not logf.exists():
        return CONC
    last = None
    for line in logf.read_text(errors="replace").splitlines():
        if "HEARTBEAT done" in line:
            last = line
    if last is None:
        return CONC                       # just opened, no heartbeat yet: treat as full
    m = re.search(r" running (\d+) ", last)
    if m is None:
        raise RuntimeError(f"{logf}: cannot parse HEARTBEAT: {last}")
    return int(m.group(1))


def kv_usage():
    with open(SERVER_LOG, "rb") as f:
        f.seek(max(0, os.path.getsize(SERVER_LOG) - 2_000_000))
        tail = f.read().decode(errors="replace")
    vals = re.findall(r"Decode batch.*?full token usage: ([0-9.]+)", tail)
    if not vals:
        raise RuntimeError("no Decode batch line in the last 2 MB of the server log")
    return float(vals[-1])


def main():
    qf, sf = A / "queue.json", A / "state.json"
    if not qf.exists():
        q = build_queue()
        qf.write_text(json.dumps(q))
        sf.write_text(json.dumps({"next": 0}))
        log(f"queue built: {len(q)} batches, {sum(len(b['ids']) for b in q)} tasks")
    q = json.loads(qf.read_text())
    while True:
        st = json.loads(sf.read_text())
        if st["next"] >= len(q):
            log("queue exhausted")
            return
        if (A / "STOP").exists():
            log("STOP present: not starting new batches")
            time.sleep(300)
            continue
        lim = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
        free_gb = os.statvfs("/data").f_bavail * os.statvfs("/data").f_frsize / 2**30
        units = active_units()
        running = sum(running_episodes(u) for u in units)
        kv = kv_usage()
        if free_gb < MIN_FREE_GB:
            log(f"/data free {free_gb:.0f} GiB < {MIN_FREE_GB}: holding")
        elif len(units) < lim.get("max_active", MAX_ACTIVE) and running + CONC <= lim.get("target_running", TARGET_RUNNING) \
                and kv < lim.get("kv_max", KV_MAX):
            b = q[st["next"]]
            tag = f"{b['pool'][:4]}-q{st['next']:05d}"
            cmd = ["systemd-run", "--user", f"--unit=collect-arc2-{tag}", "--collect",
                   "-p", f"StandardOutput=append:{A}/{tag}.log", "-p", f"StandardError=append:{A}/{tag}.log",
                   str(A / "run_batch.sh"), b["pool"], ",".join(b["ids"]), tag, str(PORT), str(CONC)]
            subprocess.run(cmd, check=True, capture_output=True)
            st["next"] += 1
            sf.write_text(json.dumps(st))
            log(f"launched {tag} ({b['pool']}, {len(b['ids'])} tasks); active {len(units) + 1}, "
                f"running before {running}, kv {kv:.2f}, next {st['next']}/{len(q)}, limits {lim}")
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
