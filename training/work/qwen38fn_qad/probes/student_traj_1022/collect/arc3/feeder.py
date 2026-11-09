#!/usr/bin/env python3
"""Round feeder for arc3 student-trajectory collection (for QAD reverse KL, 10-22): the 10-19 feeder.py changed to three
servers and ten arms in rotation.

Corpus: first the community batches the student has not played yet, A001–A019 and B001–B005 (on 10-15 only the teacher
ran them), then rerunning with different arms the B006–B025 and C001–C010 that 10-19 ran. The arm (configs/, the
original duckx config plus the 10-15 stopgap) changes every round: c0, rsi-rx, c0L3, c0i8, c5, c2, rsi-rb, c0L6, c9,
rsi-rac2. Everything else as on 10-19: run directories on /data (runs/), no new round when /data has < DATA_HOLD_GB
left, stop if the STOP file exists, non-conflicting ports per round.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

R = Path(os.environ["ARC_RUNS"]) / "tito/20261022-int3-qad-collect"
A = R / "arc3"
RUNS = A / "runs"
PORTS = (30943, 30944, 30945)
UNIT = "qadcol-arc3-"
GAMES = Path(os.environ["ARC_DATA"]) / "arc3_games_v2/collect-20261015/batches"
NEW = [f"A-{i:03d}" for i in range(1, 20)] + [f"B-{i:03d}" for i in range(1, 6)]
B = [f"B-{i:03d}" for i in range(6, 26)]
C = [f"C-{i:03d}" for i in range(1, 11)]
KV_TOKENS_PER_GAME = 55_000      # 10-15: a game that has warmed up holds about this many KV tokens
ROUND_GAMES = 25
KV_TARGET = 0.80
MAX_GAMES = 75                   # per server: cap on games after this round starts
DATA_HOLD_GB = 38
LIMITS = A / "limits.json"       # if present, reread every loop {"max_games": .., "kv_target": ..}
ARMS = ("c0", "rsi-rx", "c0L3", "c0i8", "c5", "c2", "rsi-rb", "c0L6", "c9", "rsi-rac2")
TOKENIZER = R / "tokenizer.json"                     # a copy of the deployment's tokenizer.json (checked with cmp)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def schedule():
    """A001–A019 and B001–B005, which the student has not played, first; then B, B, C rerun in rotation."""
    out, b, c = list(NEW), list(B), list(C)
    while b or c:
        for _ in range(2):
            if b:
                out.append(b.pop(0))
        if c:
            out.append(c.pop(0))
    return out


def units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", f"{UNIT}*"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split()[0].removesuffix(".service").removeprefix(UNIT) for line in out.splitlines() if line.strip()]


def running_games(tag):
    logf = RUNS / tag / "launcher.log"
    if not logf.exists():
        return ROUND_GAMES
    last = None
    for line in logf.read_text(errors="replace").splitlines():
        if "HEARTBEAT games" in line:
            last = line
    if last is None:
        return ROUND_GAMES                # just started: count it as full
    m = re.search(r"games run (\d+) ", last)
    if m is None:
        raise RuntimeError(f"{logf}: cannot parse: {last}")
    return int(m.group(1))


def kv_usage(port):
    path = R / f"serve/p{port}.log"
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 2_000_000))
        tail = f.read().decode(errors="replace")
    vals = re.findall(r"Decode batch.*?full token usage: ([0-9.]+)", tail)
    if not vals:
        return 0.0 if "The server is fired up" in tail else None
    return float(vals[-1])


_POOL = {}


def pool_tokens(port):
    if port not in _POOL:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=10) as r:
            _POOL[port] = int(json.loads(r.read())["max_total_num_tokens"])
    return _POOL[port]


def port_free(p):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


def pick_ports(used):
    for k in range(0, 40):
        for slot in range(10):
            off = 20 * k
            ps = (18907 + slot + off, 18917 + slot + off, 31467 + slot + off)
            if (slot, off) not in used and all(port_free(p) for p in ps):
                return slot, off
    raise RuntimeError("no free port block")


def main():
    sf = A / "feeder_state.json"
    sched = schedule()
    st = json.loads(sf.read_text()) if sf.exists() else {"next": 0, "port_of": {}, "used_ports": []}
    while True:
        if st["next"] >= len(sched):
            log("schedule exhausted (A001-A019, B001-B025, C001-C010 all launched)")
            return
        if (A / "STOP").exists():
            log("STOP present")
            time.sleep(300)
            continue
        vfs = os.statvfs("/data")
        data_gb = vfs.f_bavail * vfs.f_frsize / 2**30
        load = {p: 0 for p in PORTS}
        for tag in units():
            port = st["port_of"].get(tag)
            if port is None:
                raise RuntimeError(f"{UNIT}{tag}: not launched by this feeder")
            load[port] += running_games(tag)
        port = min(load, key=load.get)
        lim = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
        max_games, kv_target = lim.get("max_games", MAX_GAMES), lim.get("kv_target", KV_TARGET)
        kv = kv_usage(port)
        if data_gb < DATA_HOLD_GB:
            log(f"/data free {data_gb:.0f} GiB < {DATA_HOLD_GB}: holding")
        elif kv is None:
            log(f"p{port}: not fired up yet: holding")
        elif load[port] + ROUND_GAMES <= max_games and kv + ROUND_GAMES * KV_TOKENS_PER_GAME / pool_tokens(port) < kv_target:
            batch = sched[st["next"]]
            arm = ARMS[st["next"] % len(ARMS)]
            tag = f"comm-{batch.replace('-', '')}-{arm}"
            if (RUNS / tag).exists():
                raise RuntimeError(f"{tag} already exists")
            slot, off = pick_ports({tuple(x) for x in st["used_ports"][-40:]})
            env = {"STOPGAP_CFG_DIR": A / "configs", "CORPUS": GAMES / batch / "environment_files", "ARM": arm,
                   "SLOT": slot, "PORT_OFFSET": off, "TAG": tag, "OUT_ROOT": RUNS,
                   "SERVER": f"http://127.0.0.1:{port}", "TOKENIZER": TOKENIZER}
            cmd = ["systemd-run", "--user", f"--unit={UNIT}{tag}", "--collect",
                   *[f"--setenv={k}={v}" for k, v in env.items()],
                   "-p", f"StandardOutput=append:{A}/{tag}.unit.log", "-p", f"StandardError=append:{A}/{tag}.unit.log",
                   "/bin/bash", str(A / "launch_arm2.sh")]
            subprocess.run(cmd, check=True, capture_output=True)
            st["next"] += 1
            st["port_of"][tag] = port
            st["used_ports"].append([slot, off])
            sf.write_text(json.dumps(st))
            log(f"launched {tag} on p{port} (load {load}, kv {kv:.2f}, /data free {data_gb:.0f} GiB, max_games {max_games}) "
                f"slot {slot} offset {off}, next {st['next']}/{len(sched)}")
            time.sleep(120)                   # let it emit its first heartbeat before measuring
            continue
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
