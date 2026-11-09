#!/usr/bin/env python3
"""Pre-screening / evaluation of ARC3 single-level episodes: one group per level (work/arc_code/run_arc3_level.py, the same harness
as RL), dispatched to several replicas, summarized per level as the level-clear rate p and the quota curve p(q) (share cleared within
q, using the output tokens at the moment of clearing as recorded by the proxy; q in 4k steps up to this level's quota) and ``q_half``
(the smallest quota at which at least half the samples cleared, uncleared ones counting as infinite; clear rate below half = None).
The agent cannot see the quota, so p(q) from a 64k pre-screen is the clear rate with quota = q: set each level's quota from values
near q_half (arc3_levels.py quota).

Weights and temperature are not managed here: it evaluates whatever the backend serves (temperature = the server's
ARC_RESPONSES_DEFAULT_TEMPERATURE, see launch_b200_stream.sh).

    source env/env.sh
    python3 work/qwen38fn_rl/arc3_level_eval.py --harness-config work/qwen38fn_rl/configs/harness-arc3-level-q64k.json \\
        --tasks-dir $ARC_DATA/arc3_levels/public25 --ids-file work/qwen38fn_rl/task_lists/arc3_public25_l1.txt \\
        --samples 16 --backends http://127.0.0.1:30571/v1,http://127.0.0.1:30572/v1 --model qwen38fn \\
        --tokenizer-json <tokenizer.json> --out $ARC_RUNS/arc3_level/<run>

Dispatch: a group's episodes start together (harness concurrency), and the next group is dispatched only when running episodes +
group size ≤ ``--slots``; each sample goes to the replica with the fewest episodes at that moment. Once a group fails no new ones are
dispatched, running ones finish, the summary is written, and it exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from arcforge.arc_code.score import level_score  # noqa: E402  (official scorecard level score, squared action penalty)

HARNESS = ROOT / "work" / "arc_code" / "run_arc3_level.py"
QUOTA_STEP = 4000
#: the columns printed in each level's line (only those ≤ that level's quota)
SHOWN = (16000, 32000, 48000, 64000)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def assign(load: list[int], n: int) -> list[int]:
    """Assign n samples in turn to the replica with the fewest episodes (ties to the lower index)."""
    load, out = list(load), []
    for _ in range(n):
        r = min(range(len(load)), key=lambda i: (load[i], i))
        out.append(r)
        load[r] += 1
    return out


def running_load(out: Path, running: dict[str, dict], n_backends: int) -> list[int]:
    """Running episodes per backend: samples of running groups that have not yet written results/<tid>__s<k>.json (the same rule as
    orchestrator.replica_running). Previously a slot was freed only when the whole group finished, so a group's long tail held slots
    and 96 slots actually ran only 45–60 episodes."""
    load = [0] * n_backends
    for tid, job in running.items():
        results = out / "groups" / tid / "results"
        for k, r in enumerate(job["assigned"]):
            if not (results / f"{tid}__s{k}.json").exists():
                load[r] += 1
    return load


def expand(value: str) -> Path:
    text = os.path.expandvars(value)
    if "$" in text:
        raise SystemExit(f"{value!r}: unset environment variable; source env/env.sh")
    return Path(text)


def level_summary(tid: str, results: list[dict]) -> dict:
    n = len(results)
    cleared = [r for r in results if r["correct"]]
    at_clear = sorted(r["tokens_at_clear"] for r in cleared)
    budgets = {r["output_token_budget"] for r in results}
    if len(budgets) != 1:
        raise RuntimeError(f"{tid}: episodes ran with different quotas {sorted(budgets)}")
    budget = budgets.pop()
    half = -(-n // 2)
    stops: dict[str, int] = {}
    for r in results:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    return {"tid": tid, "game": results[0]["game"], "level": results[0]["level"], "n": n, "cleared": len(cleared),
            "p": round(len(cleared) / n, 4), "output_token_budget": budget,
            "q_half": at_clear[half - 1] if len(at_clear) >= half else None,
            "p_at": {str(q): round(sum(t <= q for t in at_clear) / n, 4)
                     for q in range(QUOTA_STEP, budget + 1, QUOTA_STEP)},
            "stops": dict(sorted(stops.items())), "tokens_at_clear": at_clear,
            "actions_over_baseline": sorted(round(r["actions"] / r["baseline_actions"], 2) for r in cleared),
            # official single-level score (scorecard.py: min(100·(baseline/actions)², 115), not cleared = 0) averaged over the n
            # samples: shows clear rate and action efficiency together
            "official_level_score": round(sum(level_score(r["actions"], r["baseline_actions"]) for r in cleared) / n, 2),
            "seeded": sorted({r["seed"]["label"] for r in results if r.get("seed")}),
            "output_tokens_median": statistics.median(r["usage"]["proxy_output_tokens"] for r in results),
            "seconds_median": statistics.median(r["seconds"] for r in results),
            "context_overflows": sum(r["context_overflows"] for r in results),
            "forbidden": sum(r["usage"]["proxy_forbidden"] for r in results)}


def read_results(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "results").glob("*.json"))]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--harness-config", type=Path, required=True)
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--ids-file", type=Path, required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--backends", required=True, help="comma-separated http://127.0.0.1:<port>/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slots", type=int, default=None, help="cap on running episodes (default 32 × number of backends)")
    p.add_argument("--launch-until", type=float, default=None,
                   help="unix seconds: no new groups are dispatched after this, running ones finish; levels not yet dispatched are"
                        " recorded in the summary's not_launched and do not count as failures (RL-style continuous dispatch: give a"
                        " level list too long to finish + this deadline)")
    p.add_argument("--agent-python", default="$ARC_AGENT_RUNTIME/bin/python")
    p.add_argument("--driver-bin", default="$ARC_CODEX_PREFIX")
    a = p.parse_args(argv)
    backends = a.backends.split(",")
    slots = 32 * len(backends) if a.slots is None else a.slots
    ids = a.ids_file.read_text().split()
    if not ids or len(set(ids)) != len(ids):
        p.error(f"{a.ids_file}: empty or duplicate ids")
    absent = [t for t in ids if not (a.tasks_dir / f"{t}.json").is_file()]
    if absent:
        p.error(f"no stub in {a.tasks_dir} for {absent}")
    if a.samples < 2 or slots < a.samples or len(set(backends)) != len(backends):
        p.error("need samples >= 2, slots >= samples, distinct backends")
    cfg = json.loads(a.harness_config.read_text())
    salted = cfg.get("cache_salt", "none") == "episode"
    out = a.out.absolute()
    out.mkdir(parents=True)                         # an evaluation dir is used only once
    (out / "groups").mkdir()
    agent_python, driver_bin = expand(a.agent_python), expand(a.driver_bin)
    atomic = out / "condition.json"
    atomic.write_text(json.dumps({"harness_config": str(a.harness_config.absolute()), "harness": cfg,
                                  "tasks_dir": str(a.tasks_dir.absolute()), "ids": ids, "samples": a.samples,
                                  "backends": backends, "model": a.model, "slots": slots,
                                  "agent_python": str(agent_python), "driver_bin": str(driver_bin),
                                  "started_unix": time.time()}, indent=1) + "\n")
    queue = list(ids)
    running: dict[str, dict] = {}                   # tid -> {proc, assigned, file}
    failed: list[dict] = []
    not_launched: list[str] = []                    # levels still in the queue at the --launch-until deadline
    stopping = False

    def on_signal(signum, _frame) -> None:
        nonlocal stopping
        log(f"signal {signum}: no new groups; SIGTERM to {len(running)} running group(s)")
        stopping = True
        for job in running.values():
            job["proc"].terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log(f"{len(ids)} levels x {a.samples} samples on {len(backends)} backend(s), {slots} slots -> {out}")
    while queue or running:
        for tid, job in list(running.items()):
            rc = job["proc"].poll()
            if rc is None:
                continue
            job["file"].close()
            del running[tid]
            if rc != 0:
                failed.append({"tid": tid, "rc": rc, "out": str(out / "groups" / f"{tid}.out")})
                stopping = True
                log(f"{tid}: group exited {rc}; no new groups (see {failed[-1]['out']})")
            else:
                s = level_summary(tid, read_results(out / "groups" / tid))
                log(f"{tid}: p {s['p']:.2f}  q_half {s['q_half']}  stops {s['stops']}")
        if queue and not stopping and a.launch_until is not None and time.time() >= a.launch_until:
            not_launched, queue = queue, []
            log(f"launch deadline: {len(not_launched)} level(s) not launched; waiting for {len(running)} running group(s)")
        load = running_load(out, running, len(backends))
        while queue and not stopping and sum(load) + a.samples <= slots:
            tid = queue.pop(0)
            assigned = assign(load, a.samples)
            per_sample = [backends[r] for r in assigned]
            argv = [str(agent_python), str(HARNESS), "--config", str(a.harness_config.absolute()),
                    "--run-dir", str(out / "groups" / tid), "--tasks-dir", str(a.tasks_dir.absolute()), "--only", tid,
                    "--samples", str(a.samples), "--sample-backends", ",".join(per_sample),
                    "--agent-python", str(agent_python), "--driver-bin", str(driver_bin),
                    "--model-base-url", per_sample[0], "--model", a.model, "--tokenizer-json", str(a.tokenizer_json),
                    *(["--salt-prefix", f"{out.name}/{tid}"] if salted else [])]
            f = (out / "groups" / f"{tid}.out").open("w")
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
            for r in assigned:
                load[r] += 1
            running[tid] = {"proc": proc, "assigned": assigned, "file": f}
            log(f"{tid}: launched on {'/'.join(map(str, load))} episodes per backend; {len(queue)} queued")
        if (stopping or not queue) and not running:
            break
        time.sleep(5)
    levels = []
    for tid in ids:
        run_dir = out / "groups" / tid
        if (run_dir / "summary.json").is_file() and tid not in {f["tid"] for f in failed}:
            levels.append(level_summary(tid, read_results(run_dir)))
    summary = {"levels": levels, "failed": failed, "not_launched": not_launched,
               "not_run": [t for t in ids if not (out / "groups" / t).exists() and t not in not_launched],
               "finished_unix": time.time()}
    tmp = out / ".summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    os.replace(tmp, out / "summary.json")
    for s in sorted(levels, key=lambda s: s["p"]):
        shown = "  ".join(f"@{q // 1000}k {s['p_at'][str(q)]:.2f}" for q in SHOWN if str(q) in s["p_at"])
        print(f"{s['tid']:10s} p {s['p']:.2f}  q_half {s['q_half']}  {shown}  stops {s['stops']}")
    if failed or summary["not_run"]:
        raise SystemExit(f"{len(failed)} group(s) failed, {len(summary['not_run'])} not run; see {out / 'summary.json'}")


if __name__ == "__main__":
    main()
