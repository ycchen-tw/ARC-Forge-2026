#!/usr/bin/env python3
"""Move finished run directories to NFS (work/tito/collect/nfs_move.py dirs: pack .tar.zst, verify with tar -d, O_DIRECT
write + read back sha256, delete local; small key files stay in place). Scans every INTERVAL seconds, one directory at a time.
offload2 (from the 10-15 top-32 ledger): each round also runs nfs_move.py ledger-live (one scan) on every ledger instance
under LEDGERS: rolled-and-closed topk.<n>.bin (4 GiB each) are moved to NFS and replaced locally by symlinks. Everything
runs in this one process, so nfs_move's flocks never collide.
Finished means: arc2 = systemd unit gone and tito_manifest.json present; arc3 = unit gone and ARM_DONE in launcher.log.
Directories referenced via --reuse by a run still in progress are not moved.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

C = Path(os.environ["ARC_RUNS"]) / "tito/20261015-fp8-dflash-collect"   # source env/env.sh
DST = Path(os.environ["ARC_NFS"]) / "runs/tito/20261015-fp8-dflash-collect"
WT = Path(__file__).resolve().parents[4]  # repo root
INTERVAL = 900
LEDGERS = {"gate": C / "gate/ledger_tk", "r1": C / "replicas/r1/ledger_tk"}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def active(prefix):
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", f"{prefix}*"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[0].removesuffix(".service").removeprefix(prefix) for line in out.splitlines() if line.strip()}


def reused_by_active(a2):
    keep = set()
    for tag in a2:
        st = C / "arc2" / tag / "STATE.json"
        if st.exists():
            r = json.loads(st.read_text()).get("reuse")
            if r is not None:
                keep.add(Path(r["run"]).name)
    return keep


def candidates():
    a2, a3 = active("collect-arc2-"), active("collect-arc3-")
    keep = reused_by_active(a2)
    out = []
    for d in sorted((C / "arc2").iterdir()):
        if d.is_dir() and d.name not in a2 and d.name not in keep and (d / "tito_manifest.json").exists() \
                and (d / "tasks").exists():
            out.append(("arc2", d.name))
    for d in sorted((C / "arc3").iterdir()):
        lg = d / "launcher.log"
        if d.is_dir() and d.name not in a3 and lg.exists() and "ARM_DONE" in lg.read_text(errors="replace") \
                and (d / "run").exists():
            out.append(("arc3", d.name))
    return out


def main():
    while True:
        for sub, name in candidates():
            log(f"moving {sub}/{name}")
            r = subprocess.run([sys.executable, str(WT / "work/tito/collect/nfs_move.py"), "--run-dir", str(C),
                                "dirs", "--src-root", str(C / sub), "--dst-root", str(DST / sub), "--delete", name],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log(f"nfs_move FAILED for {sub}/{name} rc={r.returncode}: {r.stderr[-800:]}")
                raise SystemExit(1)          # stop loudly: when NFS misbehaves, push no further
            log(f"moved {sub}/{name}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
        for tag, root in LEDGERS.items():
            if not root.is_dir():
                continue
            for inst in sorted(d for d in root.iterdir() if (d / "server.json").is_file()):
                r = subprocess.run([sys.executable, str(WT / "work/tito/collect/nfs_move.py"), "--run-dir", str(C),
                                    "ledger-live", "--instance", str(inst), "--dst-root", str(DST / "ledger_tk" / tag)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    log(f"ledger-live FAILED for {tag}/{inst.name} rc={r.returncode}: {r.stderr[-800:]}")
                    raise SystemExit(1)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
