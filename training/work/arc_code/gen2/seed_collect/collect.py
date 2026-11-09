#!/usr/bin/env python3
"""Scheduler for gen-2 seed collection: keeps ``--target`` games running on one server (serve.sh);
whenever ``--min-batch`` or more slots are free it starts a new run (``launch_codex.sh``, at most 25 games per run, 1M
output tokens per game); finished runs are left alone (gate / offload collect them later).

- **Game choice** (10-22, the user: if hard levels yield no seeds, dispatch more runs): for each of the 25 public games
  (the RL dataset's lists.json) compute the "gap" = the sum, over level 2 through the last level, of how far each
  level's level-change seeds fall short of ``--level-target`` (those in the seed-store index that passed the gate, plus
  those taken by runs still on the local machine). Larger gaps first; on equal gaps, fewer copies currently running
  first, then fewer starts so far, then random. One game can be in several runs at once. Games with gap 0 are added
  only when a run cannot otherwise be filled. ``--skip-levels`` (default sc25 L4–L6: set_level's opening board cannot be
  reproduced, so the gate always rejects them, old replay seeds included) do not count toward the gap; otherwise that
  game would always rank first while its snapshots never get into the seed store.
- **Hard-level continuation** (10-22, the user: start rollouts from the furthest seed of that level to collect):
  ``--seeded-share`` of the slots is reserved for one-game-per-run continuations (run_arm ``--game-seed``: the workspace
  loads the seed, the official server opens the game at its level K, and it plays to the end, still taking level-change /
  window-change snapshots). 10-22, the user: fill in and multiply the levels with no or few records -- for every level
  (not in ``--skip-levels``) the shortfall below ``--level-target`` is credited to "the highest level below it that has
  seeds in the store" (usually the previous level; later levels without seeds all go to the frontier); start points
  (game, level) are sorted ascending by "(running continuations + 1) / the shortfall credited to it", and within a
  level the least-used seed is picked. Level 2 has no lower seeds and relies on ordinary runs only. A failed
  continuation run (e.g. an opening board mismatch) is only recorded in ``collect/seed_failures.jsonl`` and that seed is
  not used again; collection does not stop.
- **Stop at the target** (10-22, the user: once everything has 10, stop; ``--stop-at-target``): ordinary runs pick only
  games that still have a gap. Once every level (not in ``--skip-levels``) reaches ``--level-target`` (seed store + local
  ones not yet promoted), no new runs start, and each running run's run_arm (the only ``run_arm.py`` child under that
  unit's main bash) gets SIGTERM: it opens no new session, stops the running ones, scores and writes ARM_STATE as usual,
  and the unit wraps up as usual whatever the result (the gate only collects finished games). Once everything has
  wrapped up: if gate rejections pushed some level below the target again, collection resumes automatically; if all are
  there, everything left unarchived in the seed store is archived to NFS (under finish.lock,
  ``seeds.py archive --min-seeds 1``), and the collector exits.
  ``--at-target`` (10-24: when raising the target 10→15→20 step by step, stopping at 10 killed 50 running runs and threw
  away their progress on hard levels): ``stop`` = as above; ``wait`` = no new runs and no SIGTERM, wait for the running
  ones to end naturally, then archive; ``next`` = end the collector right away and leave the running runs to the next
  (higher-target) collector to wrap up (it reads them from runs.jsonl).
  Level 2 can only be filled by ordinary runs from the start, so in this mode ``--min-batch`` is not waited for: if
  level 2's count + the ordinary runs still running that have not yet taken L02 fall short of the target, an ordinary
  run is started for those games first.
- **Sampling**: a new run uses whichever arm has started fewer games (``g2seed-t07`` T0.7 / ``g2seed-t10`` T1.0 top_p
  0.95 top_k 20).
- **Running games**: plays in a run's ``ARM_STATE.json`` that are not finished / failed, plus games not yet scheduled
  in; 0 once the unit has stopped.
- **Stop starting new runs** (running ones continue): ``<root>/collect/STOP`` exists, ``/data`` has less than
  ``--disk-floor-gib`` left, any run's ``seeds/ERRORS.jsonl`` is non-empty, the server's /health fails -- it resumes
  automatically once these conditions clear; a failed run unit writes ``collect/PAUSED`` instead, which needs a person to
  look and delete it before resuming. Every reason goes to the log.

- **Second round (10-24, g2seed2)**: the level-change snapshot is taken at act's note_done instead (arms
  ``g2seed2-t07`` / ``g2seed2-t10``, ``level_snapshot_at: note_done``). ``--count clean`` counts toward each level only
  "taken at note_done" seeds + the old seeds listed in ``--clean-seeds`` (taken at first_action, but with no workspace
  file changed after note_done, so equal to the state at note_done); continuation start points still use every
  level-change seed in the store (old ones too: playing on from a level K−1 seed takes a clean level K snapshot on
  clearing). ``--server URL@SLOT`` can be given for several servers (GPU 6, 7), and ``--target`` is per server; a new run
  goes to the one with the most free slots and a working /health. ``--tag-prefix`` keeps new run names from colliding
  with old runs in the seed store.

- **Lineage** (10-24, the user: seeds of a hard level should not all come from the same ancestor): a seed's ancestor =
  follow "which seed this run continued from" (``launch``'s ``seed``; old collections' runs.jsonl given by
  ``--lineage-runs``) all the way back to the run that played from the start. Continuation start points (game, level j)
  are still chosen by shortfall; among that level's seeds, pick the one "whose ancestor is rarest among the clean seeds
  of this game from level j on, and least used by running continuations too", ties broken by fewest uses.

When a run ends normally, the wrap-up unit ``g2seed-fin-<tag>`` starts (``finish.sh``: gate → promote into the seed store
→ move the whole run directory to NFS, delete it locally); every round also runs ``seeds.py dedupe`` on running runs
(files unchanged between a game's successive snapshots become hard links). A failed wrap-up is only logged and does not
stop collection (the local run directory stays; collection stops naturally when /data falls below the floor).

Each run is a systemd user unit ``g2seed-<tag>``, running in the pinned worktree (``--worktree``, whose HEAD must be
``--commit``). Events are logged in ``<root>/collect/runs.jsonl`` (launch / done); a restarted scheduler picks up from
there.

    bash -c 'source env/env.sh && exec .venv/bin/python work/arc_code/gen2/seed_collect/collect.py --root ... --worktree ... --commit ...'
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeds as seed_tools  # noqa: E402

TERMINAL = {"finished", "failed"}
MAX_PER_RUN = 25


def log(root: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(root / "collect" / "collect.log", "a") as f:
        f.write(line + "\n")


def events(root: Path) -> list[dict]:
    path = root / "collect" / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def record(root: Path, event: dict) -> None:
    with open(root / "collect" / "runs.jsonl", "a") as f:
        f.write(json.dumps({"unix": time.time(), **event}) + "\n")


def unit_state(unit: str) -> tuple[str, str]:
    """(ActiveState, Result). Run units are not started with --collect: failed ones must stay so Result can be read."""
    out = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "Result"],
                         capture_output=True, text=True, check=True).stdout
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return props["ActiveState"], props["Result"]


def busy_games(out: Path, games: list[str]) -> int:
    """How many games this run still occupies. Games without a play in ARM_STATE yet count as running (queued or
    opening)."""
    state = out / "run" / "ARM_STATE.json"
    if not state.exists():
        return len(games)
    plays = {p["game"]: p["status"] for p in json.loads(state.read_text()).get("plays", [])}
    return sum(1 for g in games if plays.get(g) not in TERMINAL)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_offset(active: list[dict], slot: int) -> int:
    """launch_codex's PORT_OFFSET. Counts downwards: with a positive offset the proxy (31467+slot+offset) falls into
    Linux's ephemeral port range (32768–60999), where someone else's outgoing socket can grab it first, the proxy fails
    to bind and the run dies at start (10-22 s165). Negative offsets keep all three ports below 32768 and above 1024. The
    three ports of the same offset on different cards (slot 0–7) never collide (they differ by the slot difference, or
    by 10 / 12560 plus a multiple of 100), so offsets only need to be unique among runs on the same card -- on 10-26,
    with three cards sharing one pool, 168 offsets were not enough for 3 × 64 games, pick_offset raised and the whole
    eval went down."""
    used = {r["offset"] for r in active if r["slot"] == slot}
    for k in range(11, 179):
        off = -k * 100
        if off in used:
            continue
        ports = (18907 + slot + off, 18917 + slot + off, 31467 + slot + off)
        if all(1024 < p < 32768 for p in ports) and all(port_free(p) for p in ports):
            return off
    raise RuntimeError("no free port offset in -1100..-17800")


def pause_reasons(root: Path, args) -> list[str]:
    reasons = []
    if (root / "collect" / "STOP").exists():
        reasons.append("collect/STOP exists")
    free = shutil.disk_usage("/data").free / 2**30
    if free < args.disk_floor_gib:
        reasons.append(f"/data free {free:.1f} GiB < {args.disk_floor_gib}")
    for path in sorted((root / "runs").glob("*/run/seeds/ERRORS.jsonl")):
        if path.stat().st_size:
            reasons.append(f"snapshot errors in {path}")
    paused = root / "collect" / "PAUSED"
    if paused.exists():
        reasons.append("collect/PAUSED: " + paused.read_text().strip().replace("\n", "; "))
    return reasons


def server_health(url: str) -> str | None:
    """None = usable; otherwise the reason."""
    try:
        with urllib.request.urlopen(url + "/health", timeout=30):
            return None
    except Exception as error:  # noqa: BLE001 -- any failure means "server unusable"; the reason is written as is
        return f"server {url}/health: {type(error).__name__}: {error}"


def parse_servers(specs: list[str]) -> list[tuple[str, int]]:
    out = []
    for spec in specs:
        url, sep, slot = spec.rpartition("@")
        if not sep or not url.startswith("http") or not slot.isdigit():
            raise SystemExit(f"--server {spec!r}: want URL@SLOT, e.g. http://127.0.0.1:30474@7")
        out.append((url, int(slot)))
    if len({u for u, _ in out}) != len(out) or len({s for _, s in out}) != len(out):
        raise SystemExit(f"--server: each server and slot once, got {specs}")
    return out


def counts_as_clean(row: dict, clean: set[str]) -> bool:
    """``--count clean``: taken at note_done, or old seeds confirmed by ``--clean-seeds`` to equal the state at
    note_done. Rows before 10-24 have no ``at`` (back then all were taken before the first action)."""
    return row.get("at") == "note_done" or row["seed"] in clean


def level_coverage(root: Path, store: Path, clean: set[str] | None = None) -> dict[str, collections.Counter]:
    """Number of level-change seeds per game per level: the seed-store index (entered only after the gate) + those
    taken by runs still on the local machine and not yet promoted. When ``clean`` is not None (``--count clean``),
    only clean ones are counted (``counts_as_clean``)."""
    cov: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    promoted = set()
    index = store / "index.jsonl"
    if index.exists():
        for line in index.read_text().splitlines():
            r = json.loads(line)
            promoted.add(r["tag"])
            if r["kind"] == "level" and (clean is None or counts_as_clean(r, clean)):
                cov[r["game"]][r["level"]] += 1
    for meta in (root / "runs").glob("*/run/seeds/*/L*/meta.json"):
        if meta.parents[4].name in promoted:
            continue
        m = json.loads(meta.read_text())
        if clean is None or m.get("at") == "note_done":
            cov[m["game"]][m["level"]] += 1
    return cov


def ancestor(tag: str, parent: dict[str, str | None]) -> str:
    """A run's ancestor: follow parent (run → the seed it continued from, ``<tag>/<game>/<level>``; None for a run
    played from the start) back to the run that played from the start."""
    seen = [tag]
    while parent[seen[-1]] is not None:
        seen.append(parent[seen[-1]].split("/")[0])
        if len(seen) > 100:
            raise RuntimeError(f"lineage of {tag} does not end: {seen[:5]}...")
    return seen[-1]


def lineage_parents(paths: list[Path], rows: list[dict]) -> dict[str, str | None]:
    parent = {}
    for e in [json.loads(line) for p in paths for line in p.read_text().splitlines()] + rows:
        if e["event"] == "launch":
            parent[e["tag"]] = e["seed"]["seed"] if e.get("seed") else None
    return parent


def level_ancestors(root: Path, store: Path, clean: set[str] | None, parent: dict) -> dict:
    """(game, level) → Counter(ancestor): over the same level-change seeds as level_coverage."""
    out: dict = collections.defaultdict(collections.Counter)
    promoted = set()
    index = store / "index.jsonl"
    for r in ([json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []):
        promoted.add(r["tag"])
        if r["kind"] == "level" and (clean is None or counts_as_clean(r, clean)):
            out[(r["game"], r["level"])][ancestor(r["tag"], parent)] += 1
    for meta in (root / "runs").glob("*/run/seeds/*/L*/meta.json"):
        tag = meta.parents[4].name
        if tag in promoted:
            continue
        m = json.loads(meta.read_text())
        if clean is None or m.get("at") == "note_done":
            out[(m["game"], m["level"])][ancestor(tag, parent)] += 1
    return out


def seeded_candidates(store: Path, win_levels: dict, skip: set, cov: dict, target: int, launched: list[dict],
                      active: list[dict], failed: set, parent: dict | None = None,
                      ancestors: dict | None = None) -> list[tuple[str, dict]]:
    """(game, the seed-store index row) in priority order. Each level's shortfall (``target`` − ``cov``) is credited to
    the highest level below it in the seed store; each start point (game, level) scores (running continuations + 1) /
    the shortfall credited to it, lower first; ties go to the larger shortfall. Seeds of the start level: given
    ``parent`` / ``ancestors``, pick the one whose ancestor is rarest in this game's later levels (see "Lineage" in the
    module docstring), otherwise the least used."""
    index = store / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    used = collections.Counter(r["seed"]["seed"] for r in launched if r.get("seed"))
    now = collections.Counter((r["games"][0], r["seed"]["level"]) for r in active if r.get("seed"))
    out = []
    for g, win in win_levels.items():
        seeds = collections.defaultdict(list)
        for r in rows:
            if r["game"] == g and r["kind"] == "level" and r["seed"] not in failed:
                seeds[r["level"]].append(r)
        need = collections.Counter()
        for k in range(3, win + 1):
            below = [j for j in seeds if j < k]
            if f"{g}:{k}" in skip or not below:
                continue
            need[max(below)] += max(0, target - cov[g][k])
        flying = collections.Counter(ancestor(r["seed"]["seed"].split("/")[0], parent)
                                     for r in active if r.get("seed") and r["games"] == [g]) if parent else None
        for j, n in need.items():
            if n == 0:
                continue
            if parent is None:
                row = min(seeds[j], key=lambda r: (used[r["seed"]], r["seed"]))
            else:
                def common(r, j=j):
                    a = ancestor(r["tag"], parent)
                    return sum(ancestors[(g, k)][a] for k in range(j + 1, win + 1)) + flying[a]
                row = min(seeds[j], key=lambda r: (common(r), used[r["seed"]], r["seed"]))
            out.append(((now[(g, j)] + 1) / n, -n, g, j, row))
    out.sort(key=lambda t: t[:4])
    return [(g, row) for _, _, g, _, row in out]


def launch(root: Path, args, server: tuple[str, int], arm: str, games: list[str], seq: int, offset: int,
           seed: dict | None = None) -> dict:
    tag = f"{args.tag_prefix}{seq:03d}-{arm.split('-')[1]}"
    extra = []
    if seed is not None:
        tag += f"-{games[0]}L{seed['level']:02d}"
        spec = root / "collect" / "seed_specs" / f"{tag}.json"
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(json.dumps({"ws": seed["ws"], "sha256": seed["tree_sha256"], "prefix": seed["prefix"],
                                    "actions": seed["actions"], "level": seed["level"], "label": seed["seed"]},
                                   indent=1) + "\n")
        extra = ["--game-seed", str(spec)]
    unit = f"g2seed-{tag}"
    out_root = root / "runs"
    url, slot = server
    env = {"ARM": arm, "SLOT": str(slot), "TAG": tag, "OUT_ROOT": str(out_root), "SERVER": url,
           "GAME_OUTPUT_TOKENS": str(args.game_output_tokens), "PORT_OFFSET": str(offset),
           "DRIVER_BIN": os.environ["ARC_CODEX_P8_PREFIX"], "TOKENIZER": str(args.tokenizer)}
    argv = ["systemd-run", "--user", f"--unit={unit}", f"--working-directory={args.worktree}",  # no --collect
            "-p", f"StandardOutput=append:{root / 'logs' / (tag + '.out')}",
            "-p", f"StandardError=append:{root / 'logs' / (tag + '.out')}",
            *[f"--setenv={k}={v}" for k, v in env.items()],
            "/bin/bash", "work/arc_code/duckx/launch_codex.sh", "--games", *games,
            "--wall-seconds", str(args.wall_seconds), "--server-log", str(root / f"server-g{slot}" / "server.log"),
            *extra]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    row = {"event": "launch", "tag": tag, "unit": unit, "arm": arm, "games": games, "offset": offset,
           "server": url, "slot": slot,
           "out": str(out_root / tag), "commit": args.commit,
           **({"seed": {"seed": seed["seed"], "level": seed["level"], "actions": seed["actions"]}} if seed else {})}
    record(root, row)
    return row


def launch_finish(root: Path, args, tag: str) -> None:
    unit = f"g2seed-fin-{tag}"
    log_path = root / "logs" / f"finish-{tag}.out"
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", f"--working-directory={args.worktree}",
                    "-p", f"StandardOutput=append:{log_path}", "-p", f"StandardError=append:{log_path}",
                    "/bin/bash", "work/arc_code/gen2/seed_collect/finish.sh", str(root), tag],
                   check=True, capture_output=True, text=True)
    record(root, {"event": "finish_launch", "tag": tag, "unit": unit})


def stop_run(root: Path, r: dict) -> None:
    """Send SIGTERM to the run's run_arm (graceful wrap-up). Only a run_arm under this unit's main process (launch_codex.sh's
    bash) is recognized."""
    main = subprocess.run(["systemctl", "--user", "show", r["unit"], "-p", "MainPID", "--value"],
                          capture_output=True, text=True, check=True).stdout.strip()
    kids = subprocess.run(["pgrep", "-P", main, "-f", "work/arc_code/run_arm.py"],
                          capture_output=True, text=True).stdout.split()
    if main in ("", "0") or len(kids) != 1:
        log(root, f"stop {r['tag']}: main pid {main!r}, run_arm children {kids}: not signalled (will retry)")
        return
    os.kill(int(kids[0]), signal.SIGTERM)
    record(root, {"event": "stop_sent", "tag": r["tag"], "pid": int(kids[0])})
    log(root, f"stop {r['tag']}: SIGTERM to run_arm pid {kids[0]}")


def check_finishes(root: Path, rows: list[dict]) -> None:
    ended = {r["tag"] for r in rows if r["event"] == "finish_done"}
    for r in rows:
        if r["event"] != "finish_launch" or r["tag"] in ended:
            continue
        state, result = unit_state(r["unit"])
        if state in ("active", "activating", "deactivating", "reloading"):
            continue
        record(root, {"event": "finish_done", "tag": r["tag"], "active_state": state, "result": result})
        ok = state == "inactive" and result == "success"
        log(root, f"finish {r['tag']}: {state}/{result}" + ("" if ok else f" -- see logs/finish-{r['tag']}.out"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--server", action="append", required=True,
                   help="URL@SLOT (a serve.sh server and its GPU); may be given several times, --target is per server")
    p.add_argument("--target", type=int, default=64, help="games kept running per server")
    p.add_argument("--min-batch", type=int, default=8)
    p.add_argument("--game-output-tokens", type=int, default=1_000_000)
    p.add_argument("--wall-seconds", type=int, default=50400)
    p.add_argument("--disk-floor-gib", type=float, default=40.0)
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--lists", type=Path,
                   default=Path.home() / "work/data/arc3_levels/round2-x2-20261020/lists/lists.json")
    p.add_argument("--tokenizer", type=Path, required=True,
                   help="a copy of the model's tokenizer.json (codex truncates tool output with it); must not be under /dev, /tmp, /run, /proc: "
                        "the jail masks those directories, run_arm's jail_ro_paths raises SystemExit in the slot thread, that game silently vanishes and the whole run hangs")
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--level-target", type=int, default=10, help="level-change seeds wanted per level (the gap used to pick games and continuation start points)")
    p.add_argument("--seeded-share", type=float, default=0.75,
                   help="share of --target reserved for hard-level continuations (one game per run, starting from the furthest seed in the store)")
    p.add_argument("--skip-levels", default="sc25:4,sc25:5,sc25:6",
                   help="<game>:<level> pairs not counted toward the gap, comma-separated (levels the gate always rejects)")
    p.add_argument("--store", type=Path, default=Path.home() / "work/data/arc3_seeds/g2seed-20261021",
                   help="seed store (finish.sh's STORE)")
    p.add_argument("--nfs", type=Path, default=Path(os.environ["ARC_NFS"]) / "arc3_seeds/g2seed-20261021",
                   help="the collection directory on NFS (finish.sh's NFS)")
    p.add_argument("--stop-at-target", action="store_true",
                   help="stop once every level reaches --level-target (see \"Stop at the target\" above)")
    p.add_argument("--at-target", choices=("stop", "wait", "next"), default="stop",
                   help="once --stop-at-target is reached: stop = stop the running runs, wait = wait for them to finish, next = leave them to the next collector")
    p.add_argument("--arms", default="g2seed-t07,g2seed-t10", help="arms used in rotation (duckx configs)")
    p.add_argument("--tag-prefix", default="s", help="prefix of run names (seed names are <tag>/<game>/<level> and must not collide with those in the store)")
    p.add_argument("--count", choices=("all", "clean"), default="all",
                   help="which level-change seeds count toward each level: all, or only clean ones (taken at note_done + --clean-seeds)")
    p.add_argument("--clean-seeds", type=Path, help="--count clean: old seeds confirmed equal to the state at note_done (jsonl, a seed per row)")
    p.add_argument("--lineage-runs", type=Path, action="append", default=[],
                   help="pick continuation start points with rare ancestors (see \"Lineage\" in the module docstring): collect/runs.jsonl of other collections in the store (for tracing ancestors)")
    args = p.parse_args()
    if args.at_target != "stop" and not args.stop_at_target:
        raise SystemExit("--at-target needs --stop-at-target")
    servers = parse_servers(args.server)
    arms = tuple(args.arms.split(","))
    if args.count == "clean":
        if args.clean_seeds is None:
            raise SystemExit("--count clean needs --clean-seeds (an empty file if no old seed counts)")
        clean = {json.loads(line)["seed"] for line in args.clean_seeds.read_text().splitlines() if line.strip()}
    elif args.clean_seeds is not None:
        raise SystemExit("--clean-seeds goes with --count clean")
    else:
        clean = None
    root = args.root
    (root / "collect").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    head = subprocess.run(["git", "-C", str(args.worktree), "rev-parse", "--short=9", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != args.commit:
        raise SystemExit(f"{args.worktree} is at {head}, not {args.commit}")
    if subprocess.run(["git", "-C", str(args.worktree), "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit(f"{args.worktree} has local changes")
    if any(str(args.tokenizer.resolve()).startswith(d + "/") for d in ("/dev", "/tmp", "/run", "/proc")) \
            or not args.tokenizer.is_file():
        raise SystemExit(f"--tokenizer {args.tokenizer}: not a file, or under a directory the jail hides")
    if not 0 < args.min_batch <= MAX_PER_RUN or args.target < args.min_batch:
        raise SystemExit("need 0 < --min-batch <= 25 and --target >= --min-batch")
    win_levels = json.loads(args.lists.read_text())["win_levels"]
    all_games = sorted(win_levels)
    if len(all_games) != 25:
        raise SystemExit(f"{args.lists}: {len(all_games)} games, expected the 25 public ones")
    rng = random.Random(args.seed)
    skip = {x for x in args.skip_levels.split(",") if x}
    bad = [x for x in skip if x.split(":")[0] not in win_levels]
    if bad:
        raise SystemExit(f"--skip-levels names games not in {args.lists}: {bad}")
    index = args.store / "index.jsonl"
    taken = {json.loads(line)["tag"] for line in index.read_text().splitlines()} if index.exists() else set()
    if any(t.startswith(args.tag_prefix) and t[len(args.tag_prefix):len(args.tag_prefix) + 3].isdigit() for t in taken) \
            and not events(root):
        raise SystemExit(f"--tag-prefix {args.tag_prefix!r}: the store already has runs named like that")
    log(root, f"collect start: servers {servers}, target {args.target} each, min batch {args.min_batch}, arms {arms}, "
              f"count {args.count}" + (f" ({len(clean)} old clean seeds)" if clean is not None else "")
        + f", worktree {args.worktree} @ {head}")
    paused_said = None
    reached_said = None
    while True:
        rows = events(root)
        launched = [r for r in rows if r["event"] == "launch"]
        done = {r["tag"] for r in rows if r["event"] == "done"}
        stopped = {r["tag"] for r in rows if r["event"] == "stop_sent"}
        active, running = [], collections.Counter()
        for r in launched:
            if r["tag"] in done:
                continue
            state, result = unit_state(r["unit"])
            if state in ("active", "activating", "deactivating", "reloading"):
                n = busy_games(Path(r["out"]), r["games"])
                running[r["server"]] += max(n, 1) if r.get("seed") else n   # a just-started continuation holds a slot even before its ARM_STATE appears
                active.append({**r, "busy": n})
                continue
            record(root, {"event": "done", "tag": r["tag"], "active_state": state, "result": result})
            log(root, f"run {r['tag']} ended ({state}/{result})")
            if r["tag"] in stopped:              # stopped on reaching the target: wrap up as usual whatever the result
                launch_finish(root, args, r["tag"])
            elif (state != "inactive" or result != "success") and r.get("seed"):
                with open(root / "collect" / "seed_failures.jsonl", "a") as f:
                    f.write(json.dumps({"unix": time.time(), "tag": r["tag"], "seed": r["seed"]["seed"],
                                        "state": state, "result": result}) + "\n")
                log(root, f"seeded run {r['tag']} failed ({state}/{result}): seed {r['seed']['seed']} not used again; "
                          f"see logs/{r['tag']}.out")
            elif state != "inactive" or result != "success":
                with open(root / "collect" / "PAUSED", "a") as f:
                    f.write(f"run unit {r['unit']} ended {state}/{result}\n")
            else:
                launch_finish(root, args, r["tag"])
        check_finishes(root, events(root))
        for r in active:                      # files unchanged across a game's successive snapshots become hard links (the agent's big caches, seeds.py dedupe)
            n, b = seed_tools.dedupe_run(Path(r["out"]) / "run")
            if b >= 2**28:
                log(root, f"dedupe {r['tag']}: linked {n} files, {b / 2**30:.2f} GiB")
        reasons = pause_reasons(root, args)
        health = {url: server_health(url) for url, _ in servers}
        up = [(url, slot) for url, slot in servers if health[url] is None]
        if not up:
            reasons += [h for h in health.values() if h]
        server = max(up, key=lambda x: (args.target - running[x[0]], -x[1])) if up else servers[0]
        free = args.target - running[server[0]]
        busy = sum(running.values())
        cov = level_coverage(root, args.store, clean)
        gap = {g: sum(max(0, args.level_target - cov[g][k]) for k in range(2, win_levels[g] + 1)
                      if f"{g}:{k}" not in skip) for g in all_games}
        if args.stop_at_target and not any(gap.values()):
            if args.at_target == "next":
                log(root, f"every level has {args.level_target}: {len(active)} run(s) left running for the next "
                          f"collector")
                return
            for r in active:
                if args.at_target == "stop" and r["tag"] not in stopped:
                    stop_run(root, r)
            rows = events(root)
            finishing = ({r["tag"] for r in rows if r["event"] == "finish_launch"}
                         - {r["tag"] for r in rows if r["event"] == "finish_done"})
            if active or finishing:
                if reached_said != (len(active), len(finishing)):
                    log(root, f"every level has {args.level_target}: no new runs; waiting for {len(active)} run(s) "
                              f"and {len(finishing)} finish(es)")
                    reached_said = (len(active), len(finishing))
                time.sleep(args.poll)
                continue
            log(root, "target reached and every run is finished: archiving what is left of the store")
            with open(root / "collect" / "finish.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run([sys.executable, str(Path(__file__).with_name("seeds.py")), "archive", "--store",
                                str(args.store), "--nfs", str(args.nfs), "--nfs-run-dir", str(root / "collect"),
                                "--min-seeds", "1", "--max-batches", "100"], check=True)
            log(root, "collection complete")
            return
        reached_said = None
        failures = root / "collect" / "seed_failures.jsonl"
        failed = {json.loads(line)["seed"] for line in failures.read_text().splitlines()} if failures.exists() else set()
        if args.lineage_runs:
            parent = lineage_parents(args.lineage_runs, rows)
            cands = seeded_candidates(args.store, win_levels, skip, cov, args.level_target, launched, active, failed,
                                      parent, level_ancestors(root, args.store, clean, parent))
        else:
            cands = seeded_candidates(args.store, win_levels, skip, cov, args.level_target, launched, active, failed)
        seeded_busy = sum(1 for r in active if r.get("seed") and r["server"] == server[0])   # this server's
        seeded_target = round(args.target * args.seeded_share)
        reserve = max(0, seeded_target - seeded_busy) if cands else 0
        per_arm = {a: sum(len(r["games"]) for r in launched if r["arm"] == a) for a in arms}
        arm = min(arms, key=lambda a: (per_arm[a], arms.index(a)))
        if reasons:
            if reasons != paused_said:
                log(root, f"not launching ({busy} busy): " + " | ".join(reasons))
                paused_said = reasons
        elif args.stop_at_target and free >= 1 and (l2 := [
                g for g in all_games if f"{g}:2" not in skip and cov[g][2] + sum(
                    1 for r in active if not r.get("seed") and g in r["games"]
                    and not (Path(r["out"]) / "run" / "seeds" / g / "L02").exists()) < args.level_target]):
            # Level 2 has no earlier seed to continue from and can only be filled by ordinary runs from the start; in the
            # wrap-up phase continuations fill the slots and --min-batch is never reached, so start directly: the games
            # whose level 2 count + ordinary runs still running that have not taken L02 fall short of the target
            paused_said = None
            l2.sort(key=lambda g: (cov[g][2], g))                 # largest shortfall first; only as many as free slots, keeping --target
            row = launch(root, args, server, arm, sorted(l2[:min(free, MAX_PER_RUN)]), len(launched) + 1,
                         pick_offset(active, server[1]))
            log(root, f"launched {row['tag']} (GPU {server[1]}, {arm}, {len(row['games'])} games short on level 2, "
                      f"offset {row['offset']}): {busy} busy before, {' '.join(row['games'])}")
            time.sleep(120)
            continue
        elif cands and seeded_busy < seeded_target and free >= 1:
            paused_said = None
            g, row = cands[0]
            launched_row = launch(root, args, server, arm, [g], len(launched) + 1, pick_offset(active, server[1]),
                                  seed=row)
            log(root, f"launched {launched_row['tag']} (GPU {server[1]}, {arm}, seeded from {row['seed']} "
                      f"L{row['level']:02d}, {row['actions']} prior actions): {busy} busy before, {seeded_busy} seeded")
            time.sleep(20)
            continue
        elif free - reserve >= args.min_batch:
            free -= reserve
            paused_said = None
            count = {g: 0 for g in all_games}
            for r in launched:
                for g in r["games"]:
                    count[g] += 1
            now = {g: 0 for g in all_games}
            for r in active:
                for g in r["games"]:
                    now[g] += 1
            order = sorted(all_games, key=lambda g: (-gap[g], now[g], count[g], rng.random()))
            if args.stop_at_target:
                order = [g for g in order if gap[g]]
            games = sorted(order[:min(free, MAX_PER_RUN)])
            log(root, "level gaps: " + " ".join(f"{g}:{gap[g]}" for g in order if gap[g]))
            row = launch(root, args, server, arm, games, len(launched) + 1, pick_offset(active, server[1]))
            log(root, f"launched {row['tag']} (GPU {server[1]}, {arm}, {len(games)} games, offset {row['offset']}): "
                      f"{busy} busy before, {' '.join(games)}")
            time.sleep(120)                   # let the new run's ARM_STATE appear before computing the next round
            continue
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
