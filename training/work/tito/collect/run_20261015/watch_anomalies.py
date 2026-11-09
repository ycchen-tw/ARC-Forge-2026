#!/usr/bin/env python3
"""Scan the collection state once per loop and print one line only on anomalies (as events for Monitor). State is kept in watch_state.json (log offsets already seen)."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

C = Path(os.environ["ARC_RUNS"]) / "tito/20261015-fp8-dflash-collect"   # source env/env.sh
SERVERS = {30811: C / "gate/server.log", 30812: C / "replicas/r1/server.log"}   # from 10-15 03:20Z only GPUs 0–3
SF = C / "watch_state.json"
st = json.loads(SF.read_text()) if SF.exists() else {"off": {}, "kv_hot": {}}
out = []


def new_text(p: Path) -> str:
    key = str(p)
    size = p.stat().st_size
    off = st["off"].get(key, size if key not in st["off"] and p.name == "server.log" else 0)
    if size < off:
        off = 0
    with open(p, "rb") as f:
        f.seek(off)
        txt = f.read().decode(errors="replace")
    st["off"][key] = size
    return txt


for port, logp in SERVERS.items():
    ok = subprocess.run(["curl", "-sf", "-m", "5", f"http://127.0.0.1:{port}/v1/models"], capture_output=True).returncode == 0
    if not ok:
        out.append(f"SERVER_DOWN p{port}")
    txt = new_text(logp)
    kv = re.findall(r"Decode batch.*?full token usage: ([0-9.]+).*?#queue-req: (\d+)", txt)
    if kv:
        hot = float(kv[-1][0]) > 0.92 or int(kv[-1][1]) > 8
        n = st["kv_hot"].get(str(port), 0) + 1 if hot else 0
        st["kv_hot"][str(port)] = n
        if n >= 2:
            out.append(f"KV_HOT p{port} usage {kv[-1][0]} queue {kv[-1][1]} ({n} checks)")
    # Ledgers (top-32 from 10-15 06Z): lossy / writer_error / missing top-k entries (reported only when new)
    try:
        ls = json.loads(subprocess.run(["curl", "-sf", "-m", "5", f"http://127.0.0.1:{port}/session_ledger/status"],
                                       capture_output=True, text=True).stdout or "{}")
    except json.JSONDecodeError:
        ls = {}
    if ok and ls:
        bad = {"lossy": ls.get("lossy", 0), "writer_error": ls.get("writer_error"),
               "top_logprobs_error": ls.get("counters", {}).get("top_logprobs_error", 0),
               "logprobs_len_mismatch": ls.get("counters", {}).get("logprobs_len_mismatch", 0)}
        key = f"ledger_{port}_{ls.get('instance')}"
        if bad != st.setdefault("ledger", {}).get(key, {"lossy": 0, "writer_error": None, "top_logprobs_error": 0,
                                                       "logprobs_len_mismatch": 0}):
            out.append(f"LEDGER_ISSUE p{port} {bad}")
        st["ledger"][key] = bad
    tb = [l for l in txt.splitlines() if "Error" in l and "exceeds the model's maximum context length" not in l
          and ("Traceback" in l or "Error:" in l)]
    if tb:
        out.append(f"SERVER_ERR p{port} {len(tb)}: {tb[-1][:200]}")

for unit in ("collect-feeder-arc2", "collect-feeder-arc3", "collect-offload"):
    if subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode != 0:
        out.append(f"FEEDER_DOWN {unit}")

for logp in list((C / "arc2").glob("*.log")) + list((C / "arc3").glob("*/launcher.log")) + list((C / "arc3").glob("*.unit.log")):
    if logp.name == "feeder.log":
        continue
    txt = new_text(logp)
    seen = st.setdefault("failed_seen", {})
    for line in txt.splitlines():
        m = re.search(r"failed ([1-9]\d*) ", line) or re.search(r" fail ([1-9]\d*)", line)
        if m:                                    # the heartbeat always carries the cumulative failure count: report only when it grows
            if int(m.group(1)) <= seen.get(str(logp), 0):
                continue
            seen[str(logp)] = int(m.group(1))
        if m or "Traceback" in line or "max_failures" in line:
            out.append(f"RUN_ISSUE {logp.parent.name}/{logp.name}: {line[:220]}")
            break

# Per-step timeout (step_guard): a stuck game is disabled; hard_exit = stuck in C, the whole gateway exits (that round will fail)
for tl in (C / "arc3").glob("*/run/gateway/official_out/step_timeouts.jsonl"):
    for line in new_text(tl).splitlines():
        if line.strip():
            out.append(f"STEP_TIMEOUT {tl.parts[-5]}: {line[:220]}")

free = os.statvfs("/data").f_bavail * os.statvfs("/data").f_frsize / 2**30
if free < 70:                      # 10-15 15:00 Taiwan: QAD asked to be notified before approaching 50G (it moves the pilot checkpoint-10 to tmpfs)
    out.append(f"DISK_LOW /data free {free:.0f} GiB")
if free < 50:                      # brake: both feeders stop opening new batches / rounds (running ones finish; ENOSPC would kill the whole machine)
    for sub in ("arc2", "arc3"):
        stop = C / sub / "STOP"
        if not stop.exists():
            stop.write_text(f"disk brake: /data free {free:.0f} GiB at {time.strftime('%FT%TZ', time.gmtime())}\n")
            out.append(f"DISK_BRAKE touched {sub}/STOP (/data free {free:.0f} GiB)")

# Last line of defence (10-15 19:00 Taiwan): running arc3 rounds grow about 32 GB per hour in total, which STOP cannot stop. Below 30 GiB, each check
# SIGTERMs the youngest arc3 round (sent only to run_arm.py in its own unit cgroup; its handler wraps up as on a normal finish)
if free < 30:
    units = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "collect-arc3-*"],
                           capture_output=True, text=True).stdout.split()
    units = [u for u in units if u.startswith("collect-arc3-") and u.endswith(".service")]
    def started(u):
        return int(subprocess.run(["systemctl", "--user", "show", "-p", "ActiveEnterTimestampMonotonic", "--value", u],
                                  capture_output=True, text=True).stdout.strip() or 0)
    done = set(st.setdefault("brake_terminated", []))
    live = sorted((u for u in units if u not in done), key=started)
    if live:
        u = live[-1]
        cg = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", u],
                            capture_output=True, text=True).stdout.strip()
        pids = [int(x) for x in Path(f"/sys/fs/cgroup{cg}/cgroup.procs").read_text().split()]
        arm = [pid for pid in pids
               if b"work/arc_code/run_arm.py" in Path(f"/proc/{pid}/cmdline").read_bytes()]
        if len(arm) != 1:
            out.append(f"DISK_EMERGENCY cannot pick run_arm in {u}: {arm}")
        else:
            os.kill(arm[0], 15)
            st["brake_terminated"].append(u)
            out.append(f"DISK_EMERGENCY /data free {free:.0f} GiB: SIGTERM run_arm pid {arm[0]} of {u}")
SF.write_text(json.dumps(st))
print("\n".join(out) if out else "", end="\n" if out else "")
