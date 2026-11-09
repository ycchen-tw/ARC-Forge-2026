#!/usr/bin/env python3
"""Round feeder for arc3 collection (slimmed version until the outer tooling is done). Opens a new round on each arc3
replica (r1–r3) when the KV estimate leaves room (see KV_TARGET): arms alternate c0L / v17; corpora follow ROTATION
(mostly synth, official 25 one round in 3; the community corpus is added via corpora.json once ready).
"community" in corpora.json: community batches (batches/<tier>-NNN/environment_files from collect_corpus.py assemble, tier A first);
when "community" comes up, take the next batch (community_next in state); after the whole list is used, start over (tag gets -pK).
From 10-15 01:10Z newly opened rounds all put the gateway in bwrap (configs c0Lj / c0j, launch_arm2.sh); rounds opened earlier finish with launch_arm.sh as before.
No new rounds while the STOP file exists. Each round uses non-conflicting ports (slot 0..9 + an offset in multiples of 20, checked free before start).
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

C = Path(os.environ["ARC_RUNS"]) / "tito/20261015-fp8-dflash-collect"   # source env/env.sh
A = C / "arc3"
S = C / "arc3_stopgap"
REPLICAS = {30812: "r1"}                 # from 10-15 03:20Z only GPUs 0–3: r0 arc2, r1 arc3
# Before opening a round (up to 25 games), estimate: a mature game is about 0.011 KV (88 games reach 0.97); open only if current KV + 25×0.011 < KV_TARGET.
# Before 10-15 02:00Z it was "games < 45 and KV < 0.70"; with half the games near their end KV only reached 0.5 and throughput fell to 4k.
KV_TOKENS_PER_GAME = 55_000      # 0.011 × old pool 4,996,480: KV tokens a mature game takes, roughly (each replica reads its own pool size)
ROUND_GAMES = 25
KV_TARGET = 0.85
MAX_GAMES = 80                    # cap on games after opening this round; a just-opened round's KV has not grown yet, so only the game count prevents back-to-back opens
LIMITS = A / "limits.json"        # if present, reread every loop {"max_games": .., "kv_target": ..} (tuned in stages during concurrency tests, no restart)
SERVER_LOGS = {30812: C / "replicas/r1/server.log"}
CORPORA = {"synth": os.environ["ARC_DATA"] + "/arc3_synth/environment_files",
           "off25": os.environ["ARC_ARC3_CORPUS"]}
ROTATION = ["synth", "synth", "off25"]
ARMS = {"c0L": "c0Lj", "v17": "c0j"}     # name → stopgap config (j = gateway inside bwrap)
LAUNCHER = S / "launch_arm2.sh"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def units():
    out = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "collect-arc3-*"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split()[0].removesuffix(".service").removeprefix("collect-arc3-")
            for line in out.splitlines() if line.strip()]


def run_server(tag):
    cfg = json.loads((A / tag / "config.json").read_text())
    info = A / tag / "server_info.json"
    if not info.exists():
        raise RuntimeError(f"{tag}: no server_info.json")
    port = json.loads(info.read_text())["port"]
    return port, cfg


def running_games(tag):
    logf = A / tag / "launcher.log"
    if not logf.exists():
        return 25
    last = None
    for line in logf.read_text(errors="replace").splitlines():
        if "HEARTBEAT games" in line:
            last = line
    if last is None:
        return 25                         # just opened: treat as full
    m = re.search(r"games run (\d+) ", last)
    if m is None:
        raise RuntimeError(f"{logf}: cannot parse: {last}")
    return int(m.group(1))


def kv_usage(port):
    log_ = SERVER_LOGS[port]
    with open(log_, "rb") as f:
        f.seek(max(0, log_.stat().st_size - 2_000_000))
        vals = re.findall(r"Decode batch.*?full token usage: ([0-9.]+)", f.read().decode(errors="replace"))
    if not vals:
        raise RuntimeError(f"{log_}: no Decode batch line in the last 2 MB")
    return float(vals[-1])


_POOL = {}


def pool_tokens(port):
    if port not in _POOL:
        import urllib.request
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
    st = json.loads(sf.read_text()) if sf.exists() else {"n": 0, "per_corpus": {}, "used_ports": []}
    while True:
        extra = A / "corpora.json"            # written once the community corpus is ready: {"name": path, ...} and "rotation": [...]
        corpora, rotation, e = dict(CORPORA), list(ROTATION), {}
        if extra.exists():
            e = json.loads(extra.read_text())
            corpora.update(e["corpora"])
            rotation = e["rotation"]
        if (A / "STOP").exists():
            log("STOP present")
            time.sleep(300)
            continue
        load = {p: 0 for p in REPLICAS}
        for tag in units():
            port, _ = run_server(tag)
            if port not in load:
                raise RuntimeError(f"{tag}: server port {port} is not an arc3 replica")
            load[port] += running_games(tag)
        port = min(load, key=load.get)
        lim = json.loads(LIMITS.read_text()) if LIMITS.exists() else {}
        max_games, kv_target = lim.get("max_games", MAX_GAMES), lim.get("kv_target", KV_TARGET)
        kv = kv_usage(port)
        per_game = KV_TOKENS_PER_GAME / pool_tokens(port)
        if load[port] + ROUND_GAMES <= max_games and kv + ROUND_GAMES * per_game < kv_target:
            name = rotation[st["n"] % len(rotation)]
            if name == "community":
                batches = e["community"]
                i = st.get("community_next", 0)
                path = batches[i % len(batches)]
                arm = "c0L" if i % 2 == 0 else "v17"
                bname = Path(path).parent.name.replace("-", "")
                tag = f"comm-{bname}-{arm}" + (f"-p{i // len(batches) + 1}" if i >= len(batches) else "")
                corpus_path, k = path, None
            else:
                arm = "c0L" if (st["n"] // len(rotation)) % 2 == 0 else "v17"
                k = st["per_corpus"].get(f"{name}-{arm}", 0) + 1
                tag = f"{name}-{arm}-f{k:03d}"
                corpus_path = corpora[name]
            if (A / tag).exists():
                raise RuntimeError(f"{tag} already exists")
            slot, off = pick_ports({tuple(x) for x in st["used_ports"][-40:]})
            env = {"STOPGAP_CFG_DIR": S / "configs", "CORPUS": corpus_path, "ARM": ARMS[arm], "SLOT": slot,
                   "PORT_OFFSET": off, "TAG": tag, "OUT_ROOT": A, "SERVER": f"http://127.0.0.1:{port}",
                   "TOKENIZER": os.environ["ARC_NFS"] + "/models/Qwen3.8-Flash-Next-FP8/tokenizer.json"}
            cmd = ["systemd-run", "--user", f"--unit=collect-arc3-{tag}", "--collect",
                   *[f"--setenv={k_}={v}" for k_, v in env.items()],
                   "-p", f"StandardOutput=append:{A}/{tag}.unit.log", "-p", f"StandardError=append:{A}/{tag}.unit.log",
                   "/bin/bash", str(LAUNCHER)]
            subprocess.run(cmd, check=True, capture_output=True)
            st["n"] += 1
            if name == "community":
                st["community_next"] = st.get("community_next", 0) + 1
            else:
                st["per_corpus"][f"{name}-{arm}"] = k
            st["used_ports"].append([slot, off])
            sf.write_text(json.dumps(st))
            log(f"launched {tag} on {REPLICAS[port]} (load {load}, kv {kv:.2f}, max_games {max_games}) slot {slot} offset {off}")
            time.sleep(120)                   # let it emit its first heartbeat before measuring
            continue
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
