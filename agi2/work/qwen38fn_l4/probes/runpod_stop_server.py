"""Stop only the model process group recorded by a Runpod launcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads((args.run_dir / "launch.json").read_text())
    pid, pgid, port = int(launch["pid"]), int(launch["pgid"]), str(launch["port"])
    proc = Path(f"/proc/{pid}")
    receipt = {"pid": pid, "pgid": pgid, "port": port,
               "began": datetime.now(timezone.utc).isoformat()}
    if not proc.exists():
        receipt["status"] = "already-exited"
    else:
        command = proc.joinpath("cmdline").read_bytes().decode().split("\0")
        if "sglang.launch_server" not in command or "--port" not in command:
            raise RuntimeError("PID is not the recorded SGLang server")
        if command[command.index("--port") + 1] != port or os.getpgid(pid) != pgid:
            raise RuntimeError("server port/process-group ownership mismatch")
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while proc.exists() and time.monotonic() < deadline:
            time.sleep(.25)
        if proc.exists():
            os.killpg(pgid, signal.SIGKILL)
            receipt["forced"] = True
        receipt["status"] = "stop-issued"
    receipt["ended"] = datetime.now(timezone.utc).isoformat()
    (args.run_dir / "stop.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
