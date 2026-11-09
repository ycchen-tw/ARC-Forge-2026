#!/usr/bin/env python3
"""Moves finished run directories to NFS (the dirs part of 10-15 offload2.py; work/tito/collect/nfs_move.py dirs: pack
.tar.zst, verify with tar -d, O_DIRECT write + read back sha256, delete locally; small key files stay in place). Scans
every INTERVAL seconds, one directory at a time, single process.
Finished means: arc2 = unit gone and tito_manifest.json present; arc3 = unit gone and launcher.log has ARM_DONE.
Directories referenced via --reuse by arc2 batches still running are not moved. Ledgers (ids-only, small) are not moved
here; they move all at once at wrap-up.
nfs_move's lanes / packing scratch go to $S on /dev/shm (/data space is tight).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(os.environ["ARC_RUNS"]) / "tito/20261022-int3-qad-collect"       # source env/env.sh
S = Path(f"/dev/shm/uid-{os.getuid()}/arcforge-runs/tito/20261022-int3-qad-collect")
DST = Path(os.environ["ARC_NFS"]) / "runs/tito/20261022-int3-qad-collect"
W = Path.home() / "work/arc-forge-collect1022"
INTERVAL = 900
SRC = {"arc2": R / "arc2", "arc3": R / "arc3/runs"}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def active(prefix):
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", f"{prefix}*"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[0].removesuffix(".service").removeprefix(prefix) for line in out.splitlines() if line.strip()}


def candidates():
    a2, a3 = active("qadcol-arc2-"), active("qadcol-arc3-")
    keep = set()
    for tag in a2:
        st = SRC["arc2"] / tag / "STATE.json"
        if st.exists():
            r = json.loads(st.read_text()).get("reuse")
            if r is not None:
                keep.add(Path(r["run"]).name)
    out = []
    for d in sorted(SRC["arc2"].iterdir()):
        if d.is_dir() and d.name not in a2 and d.name not in keep and (d / "tito_manifest.json").exists() \
                and (d / "tasks").exists():
            out.append(("arc2", d.name))
    for d in sorted(SRC["arc3"].iterdir()):
        lg = d / "launcher.log"
        # ARM_DONE, or run_arm crashed in tito_manifest only after writing FINAL (10-19 B016: the codex digest missed a
        # window that the ledger has) -- such runs are also finished and are moved; the manifest is filled in offline
        # later (one entry in MANIFEST_MISSING.txt)
        done = lg.exists() and "ARM_DONE" in lg.read_text(errors="replace")
        if not done and d.is_dir() and d.name not in a3 and (d / "run/FINAL.json").exists():
            (d / "MANIFEST_MISSING.txt").write_text("run_arm wrote FINAL but failed at tito_manifest; see launcher.log\n")
            done = True
        if d.is_dir() and d.name not in a3 and done and (d / "run").exists():
            out.append(("arc3", d.name))
    return out


def main():
    while True:
        for sub, name in candidates():
            log(f"moving {sub}/{name}")
            r = subprocess.run([sys.executable, str(W / "work/tito/collect/nfs_move.py"), "--run-dir", str(S),
                                "dirs", "--src-root", str(SRC[sub]), "--dst-root", str(DST / sub), "--delete", name],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log(f"nfs_move FAILED for {sub}/{name} rc={r.returncode}: {r.stderr[-800:]}")
                raise SystemExit(1)          # stop loudly: if NFS has a problem, push nothing more
            log(f"moved {sub}/{name}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
