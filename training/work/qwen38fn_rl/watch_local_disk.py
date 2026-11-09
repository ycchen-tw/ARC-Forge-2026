"""Emergency STOP only when low local space persists while the NFS offloader has failed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--orchestrator-pid", type=int, required=True)
    ap.add_argument("--orchestrator-unit", required=True)
    ap.add_argument("--offload-unit", required=True)
    ap.add_argument("--floor-gib", type=float, default=120)
    ap.add_argument("--poll-seconds", type=float, default=30)
    args = ap.parse_args()
    if args.orchestrator_pid < 1 or args.floor_gib <= 0 or args.poll_seconds <= 0:
        ap.error("PID, floor and poll interval must be positive")
    path = args.run_root / "disk_guard.json"
    unhealthy_low_polls = 0
    while True:
        free = shutil.disk_usage(args.run_root).free
        pid_live = Path(f"/proc/{args.orchestrator_pid}").exists()
        result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.orchestrator_unit], check=False)
        offload = subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.offload_unit], check=False)
        state = "watching"
        if not pid_live or result.returncode != 0:
            state = "orchestrator_exited"
        else:
            unhealthy_low_polls = unhealthy_low_polls + 1 if free < args.floor_gib * 2**30 and offload.returncode != 0 else 0
            if unhealthy_low_polls >= 3:
                (args.run_root / "STOP").touch()
                state = "STOP_written_offload_failed_low_space"
        doc = dict(state=state, free_bytes=free, free_gib=round(free / 2**30, 3), floor_gib=args.floor_gib,
                   orchestrator_pid=args.orchestrator_pid, orchestrator_unit=args.orchestrator_unit,
                   offload_unit=args.offload_unit, offload_active=offload.returncode==0,
                   unhealthy_low_polls=unhealthy_low_polls,
                   utc=time.strftime("%FT%TZ", time.gmtime()))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, path)
        print(json.dumps(doc), flush=True)
        if state != "watching":
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
