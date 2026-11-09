#!/usr/bin/env python3
"""Batch feeder for arc2 student-trajectory collection (for QAD reverse KL, 10-22): adapted from the 10-19 feeder.py.
Each arc2 replica keeps ≤ max_active batches; the next batch starts only if that replica's running episodes plus the new
batch's CONC stay within target_running and KV usage < kv_max. If the STOP file exists, no new batches start.

The queue (built at first start, saved as queue.json) aims for variety: pools take turns supplying batches
(training1000, independent1000, synth108, largegrid30, quality300, nvarc_training minus what 10-06 / 10-19 collected,
nvarc_full), and batches in turn rotate through prompt / effort / view variants (configs/*.json).
The user, 10-22: QAD data does not exclude contaminated tasks, but the ARC-AGI-2 evaluation 120 themselves are not
collected (they were never in the pools).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

R = Path(os.environ["ARC_RUNS"]) / "tito/20261022-int3-qad-collect"   # source env/env.sh
A = R / "arc2"
POOLS = Path(os.environ["ARC_DATA"]) / "arc2_pools"
OLD_DONE = Path(os.environ["ARC_RUNS"]) / "tito/20261006-node1-collect/queue/arc2/done"
Q1019 = Path(os.environ["ARC_RUNS"]) / "tito/20261019-int3-collect/arc2/queue.json"
VARIANTS = ("v3x", "altm", "alt", "v3m", "doc", "v3l")   # img: run_arc2_seq supports only view text, so altm instead
PORTS = (30941, 30942)
UNIT = "qadcol-arc2-"
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
    used = set()
    for f in OLD_DONE.glob("*.txt"):
        kv = dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)
        if kv["pool"] == "nvarc_training":
            used.update(kv["ids"].split(","))
    q19 = json.loads(Q1019.read_text())[:39]          # the 39 batches 10-19 actually finished
    used.update(t for b in q19 if b["pool"] == "nvarc_training" for t in b["ids"])
    if len(used) != 720 + 1650:
        raise RuntimeError(f"expected 2370 nvarc_training ids collected before, got {len(used)}")
    pools = {"training1000": ids("training1000"), "independent1000": ids("independent1000"), "synth108": ids("synth108"),
             "largegrid30": sorted(p.stem for p in (POOLS / "largegrid30-20261022/tasks").glob("*.json")),
             "quality300": ids("quality300"),
             "nvarc_training": [t for t in order("nvarc_training") if t not in used],
             "nvarc_full": order("nvarc_full")}
    for name, xs in pools.items():
        if not xs:
            raise RuntimeError(f"pool {name} is empty")
    chunks = {name: [xs[i:i + BATCH] for i in range(0, len(xs), BATCH)] for name, xs in pools.items()}
    q, k = [], 0
    while any(chunks.values()):
        for name in pools:
            if chunks[name]:
                q.append({"pool": name, "ids": chunks[name].pop(0), "variant": VARIANTS[k % len(VARIANTS)]})
                k += 1
    return q


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
            tag = f"{b['pool'][:4]}-{b['variant']}-q{st['next']:05d}"
            cmd = ["systemd-run", "--user", f"--unit={UNIT}{tag}", "--collect",
                   "-p", f"StandardOutput=append:{A}/{tag}.log", "-p", f"StandardError=append:{A}/{tag}.log",
                   str(A / "run_batch.sh"), b["pool"], ",".join(b["ids"]), tag, str(port), str(CONC), b["variant"]]
            subprocess.run(cmd, check=True, capture_output=True)
            st["next"] += 1
            st["port_of"][tag] = port
            sf.write_text(json.dumps(st))
            log(f"launched {tag} on p{port} ({b['pool']}, {b['variant']}, {len(b['ids'])} tasks); load {load}, kv {kv:.2f}, "
                f"next {st['next']}/{len(q)}, limits {lim}")
            time.sleep(60)
            continue
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
