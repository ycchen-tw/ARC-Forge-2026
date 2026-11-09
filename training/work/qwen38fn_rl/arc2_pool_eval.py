#!/usr/bin/env python3
"""Pre-screening / evaluation of the ARC-AGI-2 task pool: one group per task (work/arc_code/arc2/run_arc2.py --samples, the same
harness and the same harness config as RL), dispatched to several replicas, summarized per task as the all-tests-correct rate p (a
sample counts only if all its tests are correct), mean partial credit, stop reasons and output tokens; ``select`` picks RL training
tasks by p.

Task quality (lesson of 2026-10-22): "occasionally solved" does not mean the task is correct. Of the 132 nvarc_training tasks picked
at 1..5/8, 498 of 716 wrong samples passed all train pairs and failed only test, and in 60 tasks several independent samples
submitted **the same** train-passing answer, outvoting the official answer (often a whole-region color swap): either the test label
is wrong or train does not determine the rule, and RL there rewards guessing the label. So each task records ``train_pass_wrong``
(samples that pass all train pairs but are wrong) and ``wrong_consensus`` (the largest number of them submitting the same answer);
``suspect`` = wrong_consensus >= 2 and > the number of all-correct samples; ``select`` excludes suspect tasks by default (only
``--keep-suspect`` includes them). This only catches tasks that are wrong consistently, not those wrong in scattered ways; the
selection still needs spot checks.

Weights and temperature are not managed here: it evaluates whatever the backend serves (temperature = the server's
ARC_RESPONSES_DEFAULT_TEMPERATURE, set by launch_b200_stream.sh from plan.json). Dispatch is the same as arc3_level_eval.py (the same
functions): a group's samples start together, the next group is dispatched only when running episodes + group size ≤ ``--slots``,
and each sample goes to the replica with the fewest episodes at that moment; once a group fails no new ones are dispatched, running
ones finish, the summary is written, and it exits nonzero.

    source env/env.sh
    python3 work/qwen38fn_rl/arc2_pool_eval.py run --harness-config <run>/harness-train.json \\
        --tasks-dir $ARC_DATA/arc2_pools/largegrid30-20261022/tasks --ids-file $ARC_DATA/arc2_pools/largegrid30-20261022/train.txt \\
        --samples 8 --backends http://127.0.0.1:30571/v1,... --model qwen38fn --tokenizer-json <tokenizer.json> --out <dir>
    python3 work/qwen38fn_rl/arc2_pool_eval.py select --summary <dir>/summary.json --min-correct 1 --max-correct 5 --out <list>
    python3 work/qwen38fn_rl/arc2_pool_eval.py summarize --out <dir>     # recompute summary.json (adds the suspect fields to old dirs)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from arc3_level_eval import assign, expand, log, running_load  # noqa: E402  (same dispatch logic)

HARNESS = ROOT / "work" / "arc_code" / "arc2" / "run_arc2.py"


def task_summary(tid: str, results: list[dict]) -> dict:
    n = len(results)
    full = [all(r["correct"]) for r in results]
    # samples that pass all train pairs and have an answer that is not the official one; if their answers agree and outnumber the
    # all-correct ones, the task is suspect (module docstring)
    train_pass_wrong = [r for r, f in zip(results, full)
                        if not f and r["train"] and all(x == "PASS" for x in r["train"]) and r["attempt_1"] is not None]
    votes = collections.Counter(json.dumps(r["attempt_1"]) for r in train_pass_wrong)
    consensus = max(votes.values(), default=0)
    stops: dict[str, int] = {}
    for r in results:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    return {"tid": tid, "n": n, "correct": sum(full), "p": round(sum(full) / n, 4),
            "score_mean": round(sum(r["score"] for r in results) / n, 4),
            "stops": dict(sorted(stops.items())),
            "output_tokens_median": statistics.median(r["usage"]["proxy_output_tokens"] for r in results),
            "output_tokens_correct": sorted(r["usage"]["proxy_output_tokens"] for r, f in zip(results, full) if f),
            "seconds_median": statistics.median(r["seconds"] for r in results),
            "context_overflows": sum(s["context_overflow"] for r in results for s in r["sessions"]),
            "train_pass_wrong": len(train_pass_wrong), "wrong_consensus": consensus,
            "suspect": consensus >= 2 and consensus > sum(full)}


def read_results(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "results").glob("*__s*.json"))]


def run(a: argparse.Namespace) -> None:
    backends = a.backends.split(",")
    slots = 32 * len(backends) if a.slots is None else a.slots
    ids = a.ids_file.read_text().split()
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit(f"{a.ids_file}: empty or duplicate ids")
    absent = [t for t in ids if not (a.tasks_dir / f"{t}.json").is_file()]
    if absent:
        raise SystemExit(f"no task in {a.tasks_dir} for {absent[:5]}...")
    if a.samples < 2 or slots < a.samples or len(set(backends)) != len(backends):
        raise SystemExit("need samples >= 2, slots >= samples, distinct backends")
    cfg = json.loads(a.harness_config.read_text())
    salted = cfg.get("cache_salt", "none") == "episode"
    out = a.out.absolute()
    out.mkdir(parents=True)                         # an evaluation dir is used only once
    (out / "groups").mkdir()
    agent_python, driver_bin = expand(a.agent_python), expand(a.driver_bin)
    (out / "condition.json").write_text(json.dumps(
        {"harness_config": str(a.harness_config.absolute()), "harness": cfg, "tasks_dir": str(a.tasks_dir.absolute()),
         "ids": ids, "samples": a.samples, "backends": backends, "model": a.model, "slots": slots,
         "agent_python": str(agent_python), "driver_bin": str(driver_bin), "started_unix": time.time()}, indent=1) + "\n")
    queue = list(ids)
    running: dict[str, dict] = {}
    failed: list[dict] = []
    stopping = False

    def on_signal(signum, _frame) -> None:
        nonlocal stopping
        log(f"signal {signum}: no new groups; SIGTERM to {len(running)} running group(s)")
        stopping = True
        for job in running.values():
            job["proc"].terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log(f"{len(ids)} tasks x {a.samples} samples on {len(backends)} backend(s), {slots} slots"
        f"{f', ramp {a.ramp_seconds:g} s per group' if a.ramp_seconds else ''} -> {out}")
    launched, next_launch = 0, 0.0
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
                s = task_summary(tid, read_results(out / "groups" / tid))
                log(f"{tid}: {s['correct']}/{s['n']} correct  score {s['score_mean']:.2f}  stops {s['stops']}  "
                    f"[{len(ids) - len(queue) - len(running)}/{len(ids)}]")
        load = running_load(out, running, len(backends))
        while queue and not stopping and sum(load) + a.samples <= slots:
            # --ramp-seconds: the first wave (the first `slots` episodes) starts groups at intervals, so they do not all reach
            # compaction / full context at once and fill the KV pool
            if a.ramp_seconds and launched < slots and time.monotonic() < next_launch:
                break
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
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            for r in assigned:
                load[r] += 1
            running[tid] = {"proc": proc, "assigned": assigned, "file": f}
            launched += a.samples
            next_launch = time.monotonic() + (a.ramp_seconds or 0)
        if (stopping or not queue) and not running:
            break
        time.sleep(5)
    write_summary(out, ids, a.samples, failed)


def write_summary(out: Path, ids: list[str], samples: int, failed: list[dict]) -> None:
    tasks = [task_summary(t, read_results(out / "groups" / t)) for t in ids
             if (out / "groups" / t / "summary.json").is_file() and t not in {f["tid"] for f in failed}]
    summary = {"samples": samples, "tasks": tasks, "failed": failed,
               "not_run": [t for t in ids if not (out / "groups" / t).exists()], "finished_unix": time.time()}
    tmp = out / ".summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    os.replace(tmp, out / "summary.json")
    hist: dict[int, int] = {}
    for s in tasks:
        hist[s["correct"]] = hist.get(s["correct"], 0) + 1
    print("correct-of-%d histogram: %s; suspect %d" % (samples, dict(sorted(hist.items())), sum(s["suspect"] for s in tasks)))
    if failed or summary["not_run"]:
        raise SystemExit(f"{len(failed)} group(s) failed, {len(summary['not_run'])} not run; see {out / 'summary.json'}")


def summarize(a: argparse.Namespace) -> None:
    """Recompute summary.json of a finished evaluation dir (e.g. to add fields introduced later); only reads the groups/<tid>/ written
    by run, and failed groups keep run's original record."""
    out = a.out.absolute()
    cond = json.loads((out / "condition.json").read_text())
    old = json.loads((out / "summary.json").read_text())
    write_summary(out, cond["ids"], cond["samples"], old["failed"])


def select(a: argparse.Namespace) -> None:
    """Tasks whose number of all-correct samples is in [min, max] (on the hard side but with both right and wrong samples in the group,
    so RL has signal); writes the task id list + a .json of the same name recording the criteria and distribution."""
    summary = json.loads(a.summary.read_text())
    if summary["failed"] or summary["not_run"]:
        raise SystemExit(f"{a.summary}: {len(summary['failed'])} failed / {len(summary['not_run'])} not run; finish it first")
    if not 0 <= a.min_correct <= a.max_correct <= summary["samples"]:
        raise SystemExit(f"need 0 <= min-correct <= max-correct <= samples ({summary['samples']})")
    if any("suspect" not in s for s in summary["tasks"]):
        raise SystemExit(f"{a.summary}: no suspect audit (written before it existed); rerun: arc2_pool_eval.py summarize --out <dir>")
    band = [s for s in summary["tasks"] if a.min_correct <= s["correct"] <= a.max_correct]
    suspect = sorted(s["tid"] for s in band if s["suspect"])
    picked = sorted(s["tid"] for s in band if a.keep_suspect or not s["suspect"])
    if not picked:
        raise SystemExit("no task in the band")
    a.out.write_text("".join(f"{t}\n" for t in picked))
    hist: dict[int, int] = {}
    for s in summary["tasks"]:
        hist[s["correct"]] = hist.get(s["correct"], 0) + 1
    a.out.with_suffix(".json").write_text(json.dumps(
        {"summary": str(a.summary.absolute()), "samples": summary["samples"], "min_correct": a.min_correct,
         "max_correct": a.max_correct, "picked": len(picked), "of": len(summary["tasks"]),
         "keep_suspect": a.keep_suspect, "suspect_in_band": suspect,
         "correct_histogram": dict(sorted(hist.items()))}, indent=1) + "\n")
    print(f"{len(picked)} of {len(summary['tasks'])} tasks with {a.min_correct}..{a.max_correct} of {summary['samples']} "
          f"correct ({len(suspect)} suspect in the band, {'kept' if a.keep_suspect else 'dropped'}) -> {a.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--harness-config", type=Path, required=True)
    r.add_argument("--tasks-dir", type=Path, required=True)
    r.add_argument("--ids-file", type=Path, required=True)
    r.add_argument("--samples", type=int, required=True)
    r.add_argument("--backends", required=True, help="comma-separated http://127.0.0.1:<port>/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--tokenizer-json", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--slots", type=int, default=None, help="cap on running episodes (default 32 × number of backends)")
    r.add_argument("--ramp-seconds", type=float, default=None, help="seconds between group starts in the first wave (default: no stagger)")
    r.add_argument("--agent-python", default="$ARC_AGENT_RUNTIME/bin/python")
    r.add_argument("--driver-bin", default="$ARC_CODEX_PREFIX")
    s = sub.add_parser("select")
    s.add_argument("--summary", type=Path, required=True)
    s.add_argument("--min-correct", type=int, required=True)
    s.add_argument("--max-correct", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--keep-suspect", action="store_true", help="also include suspect tasks (excluded by default, see module docstring)")
    m = sub.add_parser("summarize")
    m.add_argument("--out", type=Path, required=True, help="the --out of arc2_pool_eval.py run")
    a = p.parse_args(argv)
    {"run": run, "select": select, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
