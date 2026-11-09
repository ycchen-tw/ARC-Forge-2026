#!/usr/bin/env python3
"""gen-2 seed difficulty eval (10-22, the user: after collection, measure pass rate / actions / tokens running 100k tokens from a seed, similar to RL).

Each level-change seed (taken before the first action of level K) in the seed store (``index.jsonl``) gets one
single-level episode: ``launch_codex.sh`` with one game per run, ``run_arm --game-seed <spec> --episode-action-cap
<5×baseline>``, ``GAME_OUTPUT_TOKENS=100000``, arm ``g2eval-t10`` (g2b, T1.0 / top_p 0.95 / top_k 20, no seeds taken).
Clearing the level (the middleware writes ``cleared``), hitting the action cap on level K (``action_cap``), or using up
the output budget -- whichever comes first closes the session. The agent does not know "clearing ends it" (it plays
naturally, as during collection).

- **Dispatch**: ``--backend URL@SLOT`` can be given for several servers (GPU 7's 30474@7, GPU 6's 30473@6); each keeps
  ``--per-backend`` episodes running, and a new one goes to the server with the fewest running. Episodes are ordered
  round-robin across levels (each level's seeds spread over the whole sequence), so stopping midway still leaves samples
  for every level. ``--backend-file`` is reread every round (once GPU 6's server is up, add it there; no scheduler
  restart needed).
- **Results**: when an episode ends (its unit no longer active) one row goes into ``<root>/results.jsonl``: the stop
  file, actions used on level K, the time of the clearing step, output tokens in the codex rollout (summed over distinct
  response_ids; ``tokens_at_clear`` counts only responses completed before the clearing action), input tokens (cached
  included), the session count. The whole run directory is then queued for moving to NFS (``nfs_move.py dirs
  --delete``, one unit at a time, several runs together).
- **Repeats** (10-23, the user: at least 4 per seed, 16 as the goal): ``--repeats R`` runs each seed R times, scheduled
  round by round (every seed's r-th run is dispatched before any r+1-th), so wherever it stops each seed has about the
  same count. An episode's key is (seed, which repeat); old records without ``rep`` count as the first. The summary also
  writes ``seeds.jsonl`` with one row per seed (count, clears, median tokens at clear).
- **Alongside collection** (10-23: GPU 6 freed up first): the seed store is reread every round to schedule episodes
  (level-change seeds added late by collection are scheduled too); if ``--collect-log`` is given, dispatch counts as
  complete only once ``collection complete`` appears in it.
- **Invalidate and rerun** (10-25: when the eval program crashed, eval.sh stopped both servers and 122 running games
  lost their connection): a row ``{"event": "invalidate", "tags": [...], "why": ...}`` in ``events.jsonl`` makes those
  games count neither as results nor as dispatched (that repeat of that seed is dispatched again under a new tag); their
  run directories still move to NFS.
- **Evaluating only some seeds** (10-24, the user: 10 per level; clean old seeds use the results of eval-20261023, any
  shortfall is run with new seeds taken at note_done): ``--seeds-file`` (one ``{"seed": ...}`` per line) schedules only
  those seeds; a listed seed missing from the store is an error.
- **Stop starting new ones** (running ones continue): ``<root>/STOP``, ``/data`` below ``--disk-floor-gib``,
  ``start_failed`` accumulating to ``--max-start-failed`` (most likely a bug; writes ``PAUSED``), a server's /health
  failing (stops only that server). Once everything is dispatched and finished, it writes ``summary`` (``--summarize``
  can also run alone) and exits.

    bash -c 'source env/env.sh && exec .venv/bin/python work/arc_code/gen2/seed_collect/evaluate.py --root ... \\
        --worktree ... --commit ... --tokenizer ... --backend-file <root>/backends.txt'
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import pick_offset, port_free, unit_state  # noqa: E402,F401

ARM = "g2eval-t10"
CORPUS_META = os.environ["ARC_RUNS"] + "/arc_agi3_harness/20260831-qwen38/shared/environment_files"
LIVE = ("active", "activating", "deactivating", "reloading")
STOP_NAMES = ("cleared", "action_cap", "start_failed", "died")
QUOTAS = (16000, 32000, 48000, 64000, 80000, 100000)


def log(root: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(root / "eval.log", "a") as f:
        f.write(line + "\n")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"unix": time.time(), **row}, ensure_ascii=False) + "\n")


def baselines() -> dict[str, list[int]]:
    out = {}
    for meta in glob.glob(f"{CORPUS_META}/*/*/metadata.json"):
        m = json.load(open(meta))
        out[m["game_id"].split("-")[0]] = m["baseline_actions"]
    if len(out) != 25:
        raise SystemExit(f"{CORPUS_META}: {len(out)} games with baseline_actions, expected 25")
    return out


def plan(store: Path, seed: int, repeats: int = 1, only: set[str] | None = None) -> list[dict]:
    """The store's level-change seeds (only these if ``only`` is given) × repeats: every seed's first repeat is scheduled
    before any second, and so on; within a repeat, levels take turns (one seed per level per round, the level order
    shuffled each round). Each element is the index row plus ``rep``."""
    # Collection's wrap-up may be appending: read only completed rows (ending in a newline)
    text = (store / "index.jsonl").read_text()
    rows = [json.loads(line) for line in text[:text.rfind("\n") + 1].splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] == "level"]
    if only is not None:
        missing = only - {r["seed"] for r in rows}
        if missing:
            raise SystemExit(f"--seeds-file: {len(missing)} seeds not in {store}/index.jsonl, e.g. {sorted(missing)[:3]}")
        rows = [r for r in rows if r["seed"] in only]
    by_level = collections.defaultdict(list)
    for r in rows:
        if not Path(r["ws"]).is_dir():
            raise SystemExit(f"{r['seed']}: {r['ws']} is gone (level seeds stay local; see seeds.py archive)")
        by_level[(r["game"], r["level"])].append(r)
    order = []
    for r in range(1, repeats + 1):
        rng = random.Random(seed * 1000 + r)
        groups = {k: sorted(v, key=lambda x: x["seed"]) for k, v in by_level.items()}
        for v in groups.values():
            rng.shuffle(v)
        keys = sorted(groups)
        for i in range(max(len(v) for v in groups.values())):
            rng.shuffle(keys)
            order += [{**groups[k][i], "rep": r} for k in keys if i < len(groups[k])]
    return order


def backends(args) -> list[tuple[str, int]]:
    lines = [x.split("#")[0].strip() for x in args.backend_file.read_text().splitlines()]
    out = []
    for x in lines:
        if x:
            url, slot = x.rsplit("@", 1)
            out.append((url, int(slot)))
    return out


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=30):
            return True
    except Exception:  # noqa: BLE001 -- any failure means "this server is unusable right now"
        return False


def launch(root: Path, args, ep: dict, n: int, url: str, slot: int, offset: int, cap: int) -> dict:
    tag = f"e{n:05d}-{ep['game']}L{ep['level']:02d}"
    spec = root / "specs" / f"{tag}.json"
    spec.parent.mkdir(exist_ok=True)
    spec.write_text(json.dumps({"ws": ep["ws"], "sha256": ep["tree_sha256"], "prefix": ep["prefix"],
                                "actions": ep["actions"], "level": ep["level"], "label": ep["seed"]}, indent=1) + "\n")
    unit = f"g2eval-{tag}"
    env = {"ARM": ARM, "SLOT": str(slot), "TAG": tag, "OUT_ROOT": str(root / "runs"), "SERVER": url,
           "GAME_OUTPUT_TOKENS": str(args.output_tokens), "PORT_OFFSET": str(offset),
           "DRIVER_BIN": os.environ["ARC_CODEX_P8_PREFIX"], "TOKENIZER": str(args.tokenizer)}
    argv = ["systemd-run", "--user", f"--unit={unit}", f"--working-directory={args.worktree}",
            "-p", f"StandardOutput=append:{root / 'logs' / (tag + '.out')}",
            "-p", f"StandardError=append:{root / 'logs' / (tag + '.out')}",
            *[f"--setenv={k}={v}" for k, v in env.items()],
            "/bin/bash", "work/arc_code/duckx/launch_codex.sh", "--games", ep["game"],
            "--wall-seconds", str(args.wall_seconds), "--game-seed", str(spec),
            "--episode-action-cap", str(cap)]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    row = {"event": "launch", "tag": tag, "unit": unit, "seed": ep["seed"], "rep": ep["rep"], "game": ep["game"],
           "level": ep["level"],
           "prior_actions": ep["actions"], "action_cap": cap, "backend": url, "slot": slot, "offset": offset,
           "out": str(root / "runs" / tag), "commit": args.commit}
    append(root / "events.jsonl", row)
    return row


def iso_unix(stamp: str) -> float:
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def outcome(r: dict, state: str, result: str, output_tokens: int) -> dict:
    """The result of a finished episode."""
    out = Path(r["out"])
    run, g, k, prior = out / "run", r["game"], r["level"], r["prior_actions"]
    stops = sorted(p.name for p in (run / "seed_stops").iterdir()) if (run / "seed_stops").is_dir() else []
    hist = run / "games" / g / "record" / "history.jsonl"
    rows = [x for x in jsonl(hist) if x["step"] > prior] if hist.exists() else []
    acts, clear_time = 0, None
    for x in rows:
        acts += 1
        if x["levels_completed"] >= k:
            clear_time = x["time"]
            break
    cleared = "cleared" in stops
    clear_from = "record" if clear_time is not None else None
    if cleared and clear_time is None:
        # 10-25 e01398 (sb26's last level): at the clearing step the middleware wrote the cleared stop file (with action
        # count and time), but the session was closed before record wrote that step into history. Only "exactly the
        # last step missing" is accepted: the stop file's level is k and its action count is this level's step count in
        # history plus 1
        stop = json.loads((run / "seed_stops" / "cleared").read_text())
        if stop.get("level") != k or stop.get("actions") != acts + 1:
            raise RuntimeError(f"{r['tag']}: stop file cleared ({stop}) but the record never passes level {k} "
                               f"({acts} actions after the seed)")
        acts, clear_time, clear_from = stop["actions"], float(stop["unix"]), "stop_file"
    out_tok = in_tok = cached = at_clear = 0
    seen = set()
    for path in sorted((run / "games" / g / ".driver" / "sessions").glob("**/rollout*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if '"token_usage_record"' not in line:
                continue
            x = json.loads(line)
            if x.get("type") != "token_usage_record" or x["payload"]["response_id"] in seen:
                continue
            seen.add(x["payload"]["response_id"])
            u = x["payload"]["usage"]
            out_tok += int(u["output_tokens"])
            in_tok += int(u["input_tokens"])
            cached += int(u.get("cached_input_tokens") or 0)
            if clear_time is not None and iso_unix(x["timestamp"]) <= clear_time:
                at_clear += int(u["output_tokens"])
    state_path = run / "ARM_STATE.json"
    plays = json.loads(state_path.read_text()).get("plays", []) if state_path.exists() else []
    sessions = len(plays[0]["sessions"]) if plays else 0
    if cleared:
        end = "cleared"
    elif stops:
        end = stops[0]
    elif out_tok >= output_tokens:
        end = "budget"
    else:
        end = "other"
    return {"tag": r["tag"], "seed": r["seed"], "rep": r.get("rep", 1), "game": g, "level": k, "end": end,
            "stops": stops,
            "cleared": cleared, "actions": acts, "action_cap": r["action_cap"], "output_tokens": out_tok,
            "tokens_at_clear": at_clear if cleared else None, "input_tokens": in_tok, "cached_input_tokens": cached,
            "sessions": sessions, "unit": [state, result], "backend": r["backend"],
            **({"clear_from": clear_from} if cleared else {})}


def offload(root: Path, args, tags: list[str]) -> None:
    unit = f"g2eval-offload-{int(time.time())}"
    # Collection's finish.lock: shared with seed collection's wrap-up / archiving; one NFS write stream at a time
    cmd = (f"exec 9>'{args.nfs_lock}'; flock 9; source env/env.sh >/dev/null; "
           f".venv/bin/python work/tito/collect/nfs_move.py --run-dir '{root}' --pause 1 dirs --src-root '{root}/runs' "
           f"--dst-root '{args.nfs}/runs' --delete " + " ".join(tags))
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", f"--working-directory={args.worktree}",
                    "-p", f"StandardOutput=append:{root / 'logs' / 'offload.out'}",
                    "-p", f"StandardError=append:{root / 'logs' / 'offload.out'}", "/bin/bash", "-c", cmd],
                   check=True, capture_output=True, text=True)
    append(root / "events.jsonl", {"event": "offload", "unit": unit, "tags": tags})


def invalidated(root: Path) -> set[str]:
    return {t for r in jsonl(root / "events.jsonl") if r["event"] == "invalidate" for t in r["tags"]}


def summarize(root: Path, base: dict[str, list[int]]) -> None:
    bad = invalidated(root)
    res = [r for r in jsonl(root / "results.jsonl") if r["tag"] not in bad]
    by = collections.defaultdict(list)
    for r in res:
        by[(r["game"], r["level"])].append(r)
    table = []
    for (g, k), rs in sorted(by.items()):
        b = base[g][k - 1]
        # The budget is checked every 15 seconds, so a game may be closed slightly over; a clear counts only within the
        # budget (tokens_at_clear <= QUOTAS[-1])
        ok = [r for r in rs if r["cleared"] and r["tokens_at_clear"] <= QUOTAS[-1]]
        valid = [r for r in rs if r["end"] != "start_failed"]
        n = len(valid)
        acts = [r["actions"] for r in ok]
        toks = [r["tokens_at_clear"] for r in ok]
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None  # noqa: E731
        table.append({
            "game": g, "level": k, "baseline": b, "n": n, "cleared": len(ok),
            "pass_rate": len(ok) / n if n else None,
            "pass_within": {str(t): (sum(1 for r in ok if r["tokens_at_clear"] <= t) / n if n else None) for t in QUOTAS},
            "ends": dict(collections.Counter(r["end"] for r in rs)),
            "actions_median": statistics.median(acts) if acts else None, "actions_p90": q(acts, 0.9),
            "actions_x_median": statistics.median(a / b for a in acts) if acts else None,
            "p_actions_over": {str(m): (sum(1 for a in acts if a > m * b) / len(acts) if acts else None) for m in (1, 2, 4)},
            "tokens_at_clear_median": statistics.median(toks) if toks else None, "tokens_at_clear_p90": q(toks, 0.9),
            "output_tokens_median_all": statistics.median(r["output_tokens"] for r in valid) if valid else None,
            "input_tokens_median_all": statistics.median(r["input_tokens"] for r in valid) if valid else None,
        })
    (root / "summary.json").write_text(json.dumps(table, indent=1) + "\n")
    per_seed = collections.defaultdict(list)
    for r in res:
        per_seed[r["seed"]].append(r)
    with open(root / "seeds.jsonl", "w") as f:
        for sd, rs in sorted(per_seed.items()):
            ok = [r for r in rs if r["cleared"] and r["tokens_at_clear"] <= QUOTAS[-1]]
            valid = [r for r in rs if r["end"] != "start_failed"]
            f.write(json.dumps({"seed": sd, "game": rs[0]["game"], "level": rs[0]["level"], "n": len(valid),
                                "cleared": len(ok), "pass_rate": len(ok) / len(valid) if valid else None,
                                "tokens_at_clear_median": statistics.median(r["tokens_at_clear"] for r in ok) if ok else None,
                                "actions_median": statistics.median(r["actions"] for r in ok) if ok else None}) + "\n")
    lines = [f"{'level':10s} {'base':>4s} {'n':>3s} {'pass':>5s} {'<=32k':>5s} {'<=64k':>5s} {'act_med':>7s} {'x_med':>5s} "
             f"{'>1x':>5s} {'>2x':>5s} {'tok_med':>7s} {'tok_p90':>7s} ends"]
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    for t in table:
        lines.append(f"{t['game']}:L{t['level']:<4d} {t['baseline']:>4d} {t['n']:>3d} {f(t['pass_rate']):>5s} "
                     f"{f(t['pass_within']['32000']):>5s} {f(t['pass_within']['64000']):>5s} "
                     f"{f(t['actions_median'], 0):>7s} {f(t['actions_x_median']):>5s} {f(t['p_actions_over']['1']):>5s} "
                     f"{f(t['p_actions_over']['2']):>5s} {f(t['tokens_at_clear_median'], 0):>7s} "
                     f"{f(t['tokens_at_clear_p90'], 0):>7s} {t['ends']}")
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    log(root, f"summary: {len(res)} episodes, {len(table)} levels -> {root}/summary.txt")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worktree", type=Path)
    p.add_argument("--commit")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--backend-file", type=Path, help="one server per line: <url>@<slot>; reread every round")
    p.add_argument("--store", type=Path, default=Path.home() / "work/data/arc3_seeds/g2seed-20261021")
    p.add_argument("--nfs", type=Path, default=Path(os.environ["ARC_NFS"]) / "arc3_seeds/g2seed-20261021/eval-20261023")
    p.add_argument("--nfs-lock", type=Path,
                   default=Path.home() / "work/runs/arc_code/20261021-g2seed/collect/finish.lock",
                   help="NFS write lock shared with seed collection")
    p.add_argument("--per-backend", type=int, default=64)
    p.add_argument("--output-tokens", type=int, default=100_000)
    p.add_argument("--cap-factor", type=int, default=5, help="level K action cap = baseline_actions[K] × this (RL uses 5)")
    p.add_argument("--wall-seconds", type=int, default=7200)
    p.add_argument("--disk-floor-gib", type=float, default=30.0)
    p.add_argument("--max-start-failed", type=int, default=5)
    p.add_argument("--limit", type=int, help="run only the first N episodes (small test)")
    p.add_argument("--repeats", type=int, default=1, help="runs per seed (dispatched round by round)")
    p.add_argument("--collect-log", type=Path, help="seed collection's collect.log: dispatch is not complete until collection complete appears")
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--seeds-file", type=Path, help="evaluate only these seeds (jsonl, one {\"seed\": name} per line)")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--summarize", action="store_true", help="only recompute the summary")
    args = p.parse_args()
    root = args.root
    base = baselines()
    if args.summarize:
        summarize(root, base)
        return
    for need in ("worktree", "commit", "tokenizer", "backend_file"):
        if getattr(args, need) is None:
            raise SystemExit(f"--{need.replace('_', '-')} is required")
    head = subprocess.run(["git", "-C", str(args.worktree), "rev-parse", "--short=9", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != args.commit:
        raise SystemExit(f"{args.worktree} is at {head}, not {args.commit}")
    if any(str(args.tokenizer.resolve()).startswith(d + "/") for d in ("/dev", "/tmp", "/run", "/proc")):
        raise SystemExit(f"--tokenizer {args.tokenizer}: under a directory the jail hides")
    for d in ("runs", "logs", "specs", "lanes"):
        (root / d).mkdir(parents=True, exist_ok=True)
    only = {r["seed"] for r in jsonl(args.seeds_file)} if args.seeds_file else None
    if only is not None and not only:
        raise SystemExit(f"--seeds-file {args.seeds_file}: empty")
    order = plan(args.store, args.seed, args.repeats, only)
    log(root, f"eval start: {len(order)} episodes now, worktree {args.worktree} @ {head}"
              + (f", {len(only)} seeds from {args.seeds_file}" if only is not None else ""))
    said = None
    while True:
        order = plan(args.store, args.seed, args.repeats, only)
        if args.limit:
            order = order[:args.limit]
        collecting = args.collect_log is not None and "collection complete" not in args.collect_log.read_text()
        ev = jsonl(root / "events.jsonl")
        launched = [r for r in ev if r["event"] == "launch"]       # invalidated ones included: tag numbering continues, never repeats
        bad = {t for r in ev if r["event"] == "invalidate" for t in r["tags"]}
        done = {r["tag"] for r in jsonl(root / "results.jsonl")}
        offloaded = {t for r in ev if r["event"] == "offload" for t in r["tags"]}
        active = []
        for r in launched:
            if r["tag"] in done:
                continue
            if r["tag"] in bad:
                if unit_state(r["unit"])[0] in LIVE:
                    raise RuntimeError(f"{r['tag']} is invalidated but its unit {r['unit']} is still running: stop it first")
                continue
            state, result = unit_state(r["unit"])
            if state in LIVE:
                active.append(r)
                continue
            res = outcome(r, state, result, args.output_tokens)
            append(root / "results.jsonl", res)
            log(root, f"{r['tag']}: {res['end']} actions {res['actions']}/{r['action_cap']} output {res['output_tokens']:,}"
                      + (f" (at clear {res['tokens_at_clear']:,})" if res["cleared"] else ""))
        results = jsonl(root / "results.jsonl")
        pending_off = [t for t in [*(r["tag"] for r in results), *sorted(bad)]
                       if t not in offloaded and (root / "runs" / t).is_dir()]
        offloading = [r for r in ev if r["event"] == "offload" and unit_state(r["unit"])[0] in LIVE]
        if pending_off and not offloading and (len(pending_off) >= 5 or not active):   # move early: the /data floor would block dispatch
            offload(root, args, pending_off[:200])
        started = {(r["seed"], r.get("rep", 1)) for r in launched if r["tag"] not in bad}
        todo = [ep for ep in order if (ep["seed"], ep["rep"]) not in started]
        if not todo and not active:
            if offloading or pending_off or collecting:
                time.sleep(args.poll)
                continue
            summarize(root, base)
            log(root, "eval complete")
            return
        reasons = []
        if (root / "STOP").exists():
            reasons.append("STOP exists")
        if (root / "PAUSED").exists():
            reasons.append("PAUSED: " + (root / "PAUSED").read_text().strip())
        free = shutil.disk_usage("/data").free / 2**30
        if free < args.disk_floor_gib:
            reasons.append(f"/data free {free:.1f} GiB < {args.disk_floor_gib}")
        failed_start = sum(1 for r in results if r["end"] == "start_failed")
        if failed_start >= args.max_start_failed:
            (root / "PAUSED").write_text(f"{failed_start} start_failed episodes; look at them\n")
        if reasons:
            if reasons != said:
                log(root, f"not launching ({len(active)} running): " + " | ".join(reasons))
                said = reasons
            time.sleep(args.poll)
            continue
        said = None
        launched_now = False
        for url, slot in backends(args):
            if not todo:
                break
            mine = sum(1 for r in active if r["backend"] == url)
            if mine >= args.per_backend or not healthy(url):
                continue
            ep = todo.pop(0)
            cap = args.cap_factor * base[ep["game"]][ep["level"] - 1]
            r = launch(root, args, ep, len(launched) + 1, url, slot, pick_offset(active, slot), cap)
            launched.append(r)
            active.append(r)
            launched_now = True
        time.sleep(5 if launched_now else args.poll)


if __name__ == "__main__":
    main()
