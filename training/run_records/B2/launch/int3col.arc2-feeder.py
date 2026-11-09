#!/usr/bin/env python3
"""Batch feeder for the arc2 collection (the two-server version of the 10-15 feeder.py). Each arc2 replica keeps
≤ max_active batches; the next batch starts on a replica only if its running episodes plus the new batch's CONC stay
within target_running and KV usage < kv_max. If the STOP file exists, no new batches start.

The queue (built at first start and saved as queue.json; afterwards only the cursor in state.json is read) holds only tasks
not collected before:
  quality300 (300 tasks, never collected) -> nvarc_training (minus the 720 used on 10-06, families in turn) -> nvarc_full
  (families in turn; contaminated, labeled separately).
  training1000 / independent1000 (all collected on 10-06; independent1000 again on 10-15) and synth108 (105/108 collected
  on 10-06) are not queued.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

R = Path(os.environ["ARC_RUNS"]) / "tito/20261019-int3-collect"   # source env/env.sh
A = R / "arc2"
POOLS = Path(os.environ["ARC_DATA"]) / "arc2_pools"
OLD_DONE = Path(os.environ["ARC_RUNS"]) / "tito/20261006-node1-collect/queue/arc2/done"
PORTS = (30931, 30932)
UNIT = "int3col-arc2-"
BATCH = 50
CONC = 32
MAX_ACTIVE = 3                    # per replica
TARGET_RUNNING = 64               # per replica: cap on running episodes after a new batch starts
KV_MAX = 0.70
MIN_FREE_GB = 35                  # /data (arc2-jail only accepts workspaces under $ARC_RUNS); ~0.5G per batch, moved when done, not stopped together with arc3
LIMITS = A / "limits.json"        # if present, reread every loop {"target_running": .., "max_active": .., "kv_max": .., "ports": [..]}
                                  # ports: after arc3 collection ends, handoff.py restarts the GPU 2/3 servers without think budget and adds them


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def ids(pool):
    return sorted(p.stem for p in (POOLS / pool).glob("*.json"))


def order(pool):
    return [line.strip() for line in (POOLS / f"{pool}.order.txt").read_text().splitlines() if line.strip()]


def build_queue():
    old = {}
    for f in OLD_DONE.glob("*.txt"):
        kv = dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)
        old.setdefault(kv["pool"], set()).update(kv["ids"].split(","))
    if len(old.get("nvarc_training", ())) != 720:
        raise RuntimeError(f"expected 720 nvarc_training ids used on 10-06, got {len(old.get('nvarc_training', ()))}")
    if old.get("quality300"):
        raise RuntimeError("quality300 was collected on 10-06; the queue assumes it never was")
    chunks = lambda pool, xs: [(pool, xs[i:i + BATCH]) for i in range(0, len(xs), BATCH)]
    q = chunks("quality300", ids("quality300"))
    q += chunks("nvarc_training", [t for t in order("nvarc_training") if t not in old["nvarc_training"]])
    q += chunks("nvarc_full", order("nvarc_full"))
    return [{"pool": p, "ids": xs} for p, xs in q]


def active_units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", f"{UNIT}*"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split()[0].removesuffix(".service").removeprefix(UNIT) for line in out.splitlines() if line.strip()]


def running_episodes(tag):
    logf = A / f"{tag}.log"
    if not logf.exists():
        return CONC
    last = None
    for line in logf.read_text(errors="replace").splitlines():
        if "HEARTBEAT done" in line:
            last = line
    if last is None:
        return CONC                       # just started, no heartbeat yet: count it as full
    m = re.search(r" running (\d+) ", last)
    if m is None:
        raise RuntimeError(f"{logf}: cannot parse HEARTBEAT: {last}")
    return int(m.group(1))


def kv_usage(port):
    path = R / f"serve/p{port}.log"
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 2_000_000))
        tail = f.read().decode(errors="replace")
    vals = re.findall(r"Decode batch.*?full token usage: ([0-9.]+)", tail)
    if not vals:
        return 0.0 if "The server is fired up" in tail or "Uvicorn running" in tail else None
    return float(vals[-1])


def main():
    qf, sf = A / "queue.json", A / "state.json"
    if not qf.exists():
        q = build_queue()
        qf.write_text(json.dumps(q))
        sf.write_text(json.dumps({"next": 0, "port_of": {}}))
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
        vfs = os.statvfs("/data")
        free_gb = vfs.f_bavail * vfs.f_frsize / 2**30
        units = active_units()
        ports = tuple(lim.get("ports", PORTS))
        load = {p: [0, 0] for p in ports}            # port → [batches, running episodes]
        for tag in units:
            port = st["port_of"].get(tag)
            if port is None:
                raise RuntimeError(f"{UNIT}{tag}: not launched by this feeder")
            load[port][0] += 1
            load[port][1] += running_episodes(tag)
        port = min(ports, key=lambda p: (load[p][1], load[p][0]))
        kv = kv_usage(port)
        if free_gb < MIN_FREE_GB:
            log(f"/data free {free_gb:.0f} GiB < {MIN_FREE_GB}: holding")
        elif kv is None:
            log(f"p{port}: no Decode batch line and not fired up yet: holding")
        elif load[port][0] < lim.get("max_active", MAX_ACTIVE) \
                and load[port][1] + CONC <= lim.get("target_running", TARGET_RUNNING) and kv < lim.get("kv_max", KV_MAX):
            b = q[st["next"]]
            tag = f"{b['pool'][:4]}-q{st['next']:05d}"
            cmd = ["systemd-run", "--user", f"--unit={UNIT}{tag}", "--collect",
                   "-p", f"StandardOutput=append:{A}/{tag}.log", "-p", f"StandardError=append:{A}/{tag}.log",
                   str(A / "run_batch.sh"), b["pool"], ",".join(b["ids"]), tag, str(port), str(CONC)]
            subprocess.run(cmd, check=True, capture_output=True)
            st["next"] += 1
            st["port_of"][tag] = port
            sf.write_text(json.dumps(st))
            log(f"launched {tag} on p{port} ({b['pool']}, {len(b['ids'])} tasks); load {load}, kv {kv:.2f}, "
                f"next {st['next']}/{len(q)}, limits {lim}")
            time.sleep(60)
            continue
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
