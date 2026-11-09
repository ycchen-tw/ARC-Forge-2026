#!/usr/bin/env python3
"""Scans the collection state every INTERVAL seconds and writes a line to anomalies.log on anomalies (Monitor watches
this file). Revised from 10-15 watch_anomalies.py: five INT3 replicas; arc2 / arc3 runs are all on /data, and the brake
watches /data and per-round size. Signals go only to run_arm.py inside its own units' (qadcol-*) cgroups. State is kept
in watch_state.json (log offsets seen).
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

R = Path(os.environ["ARC_RUNS"]) / "tito/20261022-int3-qad-collect"
RUNS3 = R / "arc3/runs"
PORTS = (30941, 30942, 30943, 30944, 30945)
SF = R / "watch_state.json"
OUT = R / "anomalies.log"
INTERVAL = 300
ROUND_GB_WARN = 3                  # report when one arc3 round's run directory exceeds this (a normal round is 0.5–1.7G within 6 hours)
ROUND_GB_KILL = 4                  # SIGTERM that round beyond this (10-19 10:48 at /data ~35G it was 3G; 14:33 /data 61G and the 3G cap had already killed 4 B-tier rounds, so raised to 4G)


def gib(path):
    v = os.statvfs(path)
    return v.f_bavail * v.f_frsize / 2**30


def units(prefix):
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", f"{prefix}*"],
                         capture_output=True, text=True).stdout.split()
    return [u for u in out if u.startswith(prefix) and u.endswith(".service")]


def dir_gb(p: Path) -> float:
    r = subprocess.run(["du", "-s", "--block-size=1M", str(p)], capture_output=True, text=True)   # 1G would round up into a tie
    return float(r.stdout.split()[0]) / 1024 if r.returncode == 0 and r.stdout else -1.0


def scan(st):
    out = []

    def new_text(p: Path) -> str:
        key = str(p)
        size = p.stat().st_size
        off = st["off"].get(key, 0)
        if size < off:
            off = 0
        with open(p, "rb") as f:
            f.seek(off)
            txt = f.read().decode(errors="replace")
        st["off"][key] = size
        return txt

    for port in PORTS:
        logp = R / f"serve/p{port}.log"
        # with load near 500 the HTTP loop answers a few seconds late (14:05 two servers timed out at -m 5, logs
        # unbroken): 15 s timeout, report only after two in a row
        ok = subprocess.run(["curl", "-sf", "-m", "15", f"http://127.0.0.1:{port}/v1/models"], capture_output=True).returncode == 0
        miss = st.setdefault("down", {})
        miss[str(port)] = 0 if ok else miss.get(str(port), 0) + 1
        if miss[str(port)] >= 2:
            out.append(f"SERVER_DOWN p{port} ({miss[str(port)]} checks)")
        txt = new_text(logp)
        kv = re.findall(r"Decode batch.*?full token usage: ([0-9.]+).*?#queue-req: (\d+)", txt)
        if kv:
            hot = float(kv[-1][0]) > 0.92 or int(kv[-1][1]) > 8
            n = st["kv_hot"].get(str(port), 0) + 1 if hot else 0
            st["kv_hot"][str(port)] = n
            if n >= 2:
                out.append(f"KV_HOT p{port} usage {kv[-1][0]} queue {kv[-1][1]} ({n} checks)")
        if ok:
            try:
                ls = json.loads(subprocess.run(["curl", "-sf", "-m", "5", f"http://127.0.0.1:{port}/session_ledger/status"],
                                               capture_output=True, text=True).stdout or "{}")
            except json.JSONDecodeError:
                ls = {}
            if ls:
                bad = {"lossy": ls.get("lossy", 0), "writer_error": ls.get("writer_error")}
                key = f"ledger_{port}_{ls.get('instance')}"
                if bad != st.setdefault("ledger", {}).get(key, {"lossy": 0, "writer_error": None}):
                    out.append(f"LEDGER_ISSUE p{port} {bad}")
                st["ledger"][key] = bad
        # Responses requests with images whose max_output_tokens exceeds the context: the proxy shrinks and resends
        # automatically (expected), each leaving one Traceback + ValueError
        # when a round ends (wall / SIGTERM) the harness stops codex and in-flight streaming requests are aborted: each
        # is also a Traceback (type 1 disconnect)
        benign = ("exceeds the model's maximum context length", "Request is disconnected from the client side")
        ctx = sum(any(b in l for b in benign) for l in txt.splitlines())
        tbs = sum("Traceback (most recent call last)" in l for l in txt.splitlines()) \
            - sum("The above exception was the direct cause" in l or "During handling of the above exception" in l
                  for l in txt.splitlines())
        errs = [l for l in txt.splitlines() if "Error:" in l and not any(b in l for b in benign)]
        if tbs > ctx or errs:
            out.append(f"SERVER_ERR p{port} tracebacks {tbs} (context-length {ctx}), other errors {len(errs)}: "
                       f"{(errs or ['-'])[-1][:200]}")

    finished = {"qadcol-feeder-arc3": (R / "arc3/feeder.log", "schedule exhausted")}
    for unit in ("qadcol-feeder-arc2", "qadcol-feeder-arc3", "qadcol-offload"):
        if subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode != 0 and st.get("started", {}).get(unit):
            logf, marker = finished.get(unit, (None, None))
            if logf is not None and logf.exists() and marker in logf.read_text(errors="replace")[-2000:]:
                continue                             # exited only after finishing normally
            out.append(f"FEEDER_DOWN {unit}")

    logs = [p for p in (R / "arc2").glob("*.log") if p.name != "feeder.log"]
    logs += list((R / "arc3").glob("*.unit.log")) + list(RUNS3.glob("*/launcher.log"))
    seen = st.setdefault("failed_seen", {})
    for logp in logs:
        txt = new_text(logp)
        for line in txt.splitlines():
            m = re.search(r"failed ([1-9]\d*) ", line) or re.search(r" fail ([1-9]\d*)", line)
            if m:                                    # every heartbeat carries the cumulative failure count: report only when it grows
                if int(m.group(1)) <= seen.get(str(logp), 0):
                    continue
                seen[str(logp)] = int(m.group(1))
            if m or "Traceback" in line or "max_failures" in line:
                out.append(f"RUN_ISSUE {logp.parent.name}/{logp.name}: {line[:220]}")
                break

    for tl in RUNS3.glob("*/run/gateway/official_out/step_timeouts.jsonl"):
        for line in new_text(tl).splitlines():
            if line.strip():
                out.append(f"STEP_TIMEOUT {tl.parts[-5]}: {line[:220]}")

    def arm_pid(u):
        """The only run_arm.py in its own unit's cgroup (signals go only to it; its SIGTERM handler wraps up like a normal
        end and writes the manifest)."""
        cg = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", u],
                            capture_output=True, text=True).stdout.strip()
        pids = [int(x) for x in Path(f"/sys/fs/cgroup{cg}/cgroup.procs").read_text().split()]
        arm = [pid for pid in pids if b"work/arc_code/run_arm.py" in Path(f"/proc/{pid}/cmdline").read_bytes()]
        return arm[0] if len(arm) == 1 else arm

    done = st.setdefault("brake_terminated", [])
    # Button-mashing games (10-15 community A, late part: one game 1.6 GB): measure the run directory of each running
    # arc3 round every 10 minutes; beyond ROUND_GB_KILL that round is killed
    if time.time() - st.get("du_at", 0) > 600:
        st["du_at"] = time.time()
        for u in units("qadcol-arc3-"):
            tag = u.removeprefix("qadcol-arc3-").removesuffix(".service")
            g = dir_gb(RUNS3 / tag)
            if g >= ROUND_GB_KILL and u not in done:
                pid = arm_pid(u)
                if isinstance(pid, list):
                    out.append(f"ROUND_KILL cannot pick run_arm in {u}: {pid}")
                else:
                    os.kill(pid, 15)
                    done.append(u)
                    out.append(f"ROUND_KILL {tag} {g:.0f} GiB >= {ROUND_GB_KILL}: SIGTERM run_arm pid {pid}")
            elif g >= ROUND_GB_WARN and g > st.setdefault("big_round", {}).get(tag, 0) + 3:
                st["big_round"][tag] = g
                out.append(f"ROUND_BIG {tag} {g:.0f} GiB")

    # /data: arc2 and arc3 runs are all here (arc2-jail only accepts $ARC_RUNS, jail.sh masks /dev). Leave 30G for
    # others; ENOSPC would take down the whole machine
    data = gib("/data")
    step = int(data // 4)                            # report only when crossing a 4 GiB step (reporting every 5 minutes would flood)
    if data < 40 and step < st.get("disk_step", 99):
        out.append(f"DISK_LOW /data free {data:.0f} GiB (this collection {dir_gb(R):.0f} GiB)")
    st["disk_step"] = step
    # Agreed with the other sessions to leave 30G on /data, and the RSI chain (GPU 6/7) stops a whole wave at <30G: this
    # side holds 33G. arc3 (~45 MB per game per hour, moved only after the whole round ends) stops first; arc2 (~0.5G per
    # batch, moved when done) stops a bit later.
    # The brake has hysteresis: once space comes back, the STOP it placed itself is removed (a STOP placed by a person
    # is left alone)
    for sub, stop_below, release_above in (("arc3", 36, 40), ("arc2", 34, 37)):
        stop = R / sub / "STOP"
        if data > release_above and stop.exists() and stop.read_text().startswith("disk brake:"):
            stop.unlink()
            out.append(f"DISK_BRAKE_RELEASED {sub}/STOP (/data free {data:.0f} GiB)")
        if data < stop_below and not stop.exists():
            stop.write_text(f"disk brake: /data free {data:.0f} GiB at {time.strftime('%FT%TZ', time.gmtime())}\n")
            out.append(f"DISK_BRAKE touched {sub}/STOP (/data free {data:.0f} GiB)")
    # Last line of defense: below 33 GiB, SIGTERM the arc3 round with the largest run directory (most likely one with a
    # game button-mashing: the lowest data value, and moving it reclaims the most).
    # Space from a killed round comes back only after offload (a scan every 15 minutes) packs and moves it: at most one
    # round per 20 minutes, otherwise it would kill every round in a row
    if data < 33 and time.time() - st.get("emergency_at", 0) > 1200:
        st["emergency_at"] = time.time()
        size = {u: dir_gb(RUNS3 / u.removeprefix("qadcol-arc3-").removesuffix(".service"))
                for u in units("qadcol-arc3-") if u not in done}
        live = sorted(size, key=size.get)
        if live:
            u = live[-1]
            pid = arm_pid(u)
            if isinstance(pid, list):
                out.append(f"DISK_EMERGENCY cannot pick run_arm in {u}: {pid}")
            else:
                os.kill(pid, 15)
                done.append(u)
                out.append(f"DISK_EMERGENCY /data free {data:.0f} GiB: SIGTERM run_arm pid {pid} of {u}")
    shm = gib("/dev/shm")
    if shm < 250:                                    # the ledgers are here; /dev/shm must leave 200G for others
        out.append(f"SHM_LOW /dev/shm free {shm:.0f} GiB")
    return out


def main():
    while True:
        st = json.loads(SF.read_text()) if SF.exists() else {"off": {}, "kv_hot": {}}
        st["started"] = {u: True for u in ("qadcol-feeder-arc2", "qadcol-feeder-arc3", "qadcol-offload")
                         if st.get("started", {}).get(u)
                         or subprocess.run(["systemctl", "--user", "is-active", "-q", u]).returncode == 0}
        try:
            out = scan(st)
        except Exception as e:                       # the watcher itself breaking must be reported loudly too
            out = [f"WATCH_ERROR {type(e).__name__}: {e}"]
        SF.write_text(json.dumps(st))
        if out:
            with open(OUT, "a") as f:
                for line in out:
                    f.write(time.strftime("%FT%TZ ", time.gmtime()) + line + "\n")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
