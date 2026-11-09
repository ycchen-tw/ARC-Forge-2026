"""Failure-mode breakdown of one run_arc2.py run.

    python work/arc_code/arc2/analyze_run.py <run dir> [<run dir> ...]

Per run it prints: score (official partial credit, attempt_1), stop reasons, four quadrants (train
all passed × test correct), token distribution, per-task counts of sessions / compactions / image
views / txt reads / check runs (from driver_digest.jsonl); with several runs it also prints a
per-task pairing (tasks both runs finished).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(run: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted((run / "results").glob("*.json")):
        if path.name == "solutions.json":
            continue
        r = json.loads(path.read_text())
        digest = run / "tasks" / r["task"] / "driver_digest.jsonl"
        tools = Counter()
        compactions = 0
        if digest.exists():
            for line in digest.read_text().splitlines():
                d = json.loads(line)
                if d["kind"] == "compaction":
                    compactions += 1
                elif d["kind"] == "tool":
                    name, args = d["name"], d["arguments"]
                    tools[name] += 1
                    if name == "exec_command":
                        tools["check.py"] += "check.py" in args
                        tools["read_txt"] += "_input.txt" in args or "_output.txt" in args
                        tools["task.json"] += "task.json" in args
        r["tools"], r["compactions"] = tools, compactions
        rows[r["task"]] = r
    return rows


def tok(r: dict) -> int:
    """The proxy's count (authoritative since v1); old runs only have the rollout's count."""
    return r["usage"].get("proxy_output_tokens", r["usage"]["output_tokens"])


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def report(run: Path, rows: dict[str, dict]) -> None:
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    n_tasks = summary.get("tasks", len(rows))
    print(f"== {run.name}: {len(rows)}/{n_tasks} tasks done")
    print(f"score (partial, attempt_1, unfinished = 0): {100 * sum(r['score'] for r in rows.values()) / n_tasks:.2f}")
    print("stop:", dict(Counter(r["stop"] for r in rows.values())))
    quad = Counter()
    for r in rows.values():
        train_ok = all(v == "PASS" for v in r["train"])
        quad[("train_pass" if train_ok else "train_fail", "test_ok" if r["score"] == 1 else
              "test_partial" if r["score"] > 0 else "test_wrong")] += 1
    for k in sorted(quad):
        print(f"  {k[0]:10s} {k[1]:12s} {quad[k]}")
    out = [tok(r) for r in rows.values()]
    if out:
        print(f"output tokens p10/p50/p90/max: {pct(out, .1):,}/{pct(out, .5):,}/{pct(out, .9):,}/{max(out):,}  "
              f"mean {statistics.mean(out):,.0f}")
        solved = [tok(r) for r in rows.values() if r["score"] == 1]
        if solved:
            print(f"  solved tasks: p50 {pct(solved, .5):,}  max {max(solved):,}")
    secs = [r["seconds"] for r in rows.values()]
    if secs:
        print(f"seconds p50/p90/max: {pct(secs, .5):.0f}/{pct(secs, .9):.0f}/{max(secs):.0f}")
    for key in ("view_image", "check.py", "read_txt", "task.json", "exec_command"):
        vals = [r["tools"].get(key, 0) for r in rows.values()]
        print(f"  {key:12s} per task mean {statistics.mean(vals):.1f}  tasks using it {sum(v > 0 for v in vals)}")
    comp = [r["compactions"] for r in rows.values()]
    sess = [len(r["sessions"]) for r in rows.values()]
    print(f"  compactions per task mean {statistics.mean(comp):.2f} (tasks with any: {sum(c > 0 for c in comp)})"
          f"  sessions mean {statistics.mean(sess):.2f}")
    print("  source:", dict(Counter(s for r in rows.values() for s in r["source"])))
    errs = Counter((r["load_error"] or "").split(":")[0] for r in rows.values() if r["load_error"])
    if errs:
        print("  load errors:", dict(errs))


def paired(a: tuple[str, dict], b: tuple[str, dict]) -> None:
    common = sorted(set(a[1]) & set(b[1]))
    if not common:
        return
    sa = sum(a[1][t]["score"] for t in common)
    sb = sum(b[1][t]["score"] for t in common)
    win = sum(b[1][t]["score"] > a[1][t]["score"] for t in common)
    loss = sum(b[1][t]["score"] < a[1][t]["score"] for t in common)
    print(f"== paired {a[0]} -> {b[0]} on {len(common)} tasks: {100 * sa / len(common):.2f} -> "
          f"{100 * sb / len(common):.2f}  (+{win} / -{loss} tasks)")


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    loaded = [(r.name, load(r)) for r in runs]
    for run, (_, rows) in zip(runs, loaded):
        report(run, rows)
    for i in range(1, len(loaded)):
        paired(loaded[0], loaded[i])


if __name__ == "__main__":
    main()
