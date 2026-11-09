"""Per-task paired comparison of two held-out rollouts on the same tasks (e.g. the calibration round = base weights vs round 5's
held-out).

    python3 work/qwen38fn_rl/heldout_compare.py --a <run_dir> [...] --b <run_dir> [...] [--tasks <tasks.txt>]

Each side gets its sample run dirs (``s0``..``s7``, each with ``results/<task>.json``). Both sides must have the same tasks.
Takes the per-task difference d of mean reward and reports mean(d) and its standard error sd(d)/sqrt(number of tasks) (paired on the
same task: task difficulty cancels out, leaving only sampling noise); also reports the share that hit the budget, output tokens, and
each side's number of live groups. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from orchestrator import OrchestratorError, read_episodes, summarize


def load(run_dirs: list[Path], tasks: list[str] | None) -> tuple[list[str], list[dict]]:
    if tasks is None:
        tasks = sorted(p.stem for p in (run_dirs[0] / "results").glob("*.json") if p.stem != "solutions")
    return tasks, read_episodes(run_dirs, tasks)


def per_task(episodes: list[dict], key) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for e in episodes:
        by.setdefault(e["task"], []).append(key(e))
    return {t: statistics.fmean(v) for t, v in by.items()}


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    d = [b[t] - a[t] for t in sorted(a)]
    se = statistics.stdev(d) / math.sqrt(len(d))
    mean = statistics.fmean(d)
    return {"a": statistics.fmean(a.values()), "b": statistics.fmean(b.values()), "diff": mean, "se": se,
            "z": mean / se if se > 0 else None, "tasks_up": sum(x > 0 for x in d), "tasks_down": sum(x < 0 for x in d)}


def compare(ea: list[dict], eb: list[dict], samples_a: int, samples_b: int) -> dict:
    """Pair two sets of episodes (rows from ``read_episodes``) per task: reward, budget hits and output tokens each reported as b − a
    (``paired``), plus live groups and stop counts. Both sides must have the same tasks (heldout_offline.py also uses it against
    base24)."""
    sa, sb = summarize(ea, samples_a), summarize(eb, samples_b)
    ta, tb = set(sa["per_task_mean"]), set(sb["per_task_mean"])
    if ta != tb:
        raise OrchestratorError(f"paired comparison needs the same tasks: only in a {sorted(ta - tb)[:3]}, "
                                f"only in b {sorted(tb - ta)[:3]}")
    return {
        "tasks": len(ta), "samples": [samples_a, samples_b],
        "reward": paired(per_task(ea, lambda e: e["score"]), per_task(eb, lambda e: e["score"])),
        "budget_hit": paired(per_task(ea, lambda e: float(e["stop"] == "budget")),
                             per_task(eb, lambda e: float(e["stop"] == "budget"))),
        "output_tokens": paired(per_task(ea, lambda e: e["tokens"]), per_task(eb, lambda e: e["tokens"])),
        "live_groups": [sa["live_groups"], sb["live_groups"]],
        "stops": [sa["stops"], sb["stops"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, nargs="+", required=True, help="baseline sample run dirs")
    ap.add_argument("--b", type=Path, nargs="+", required=True, help="compared sample run dirs")
    ap.add_argument("--tasks", type=Path, help="task list (ids separated by commas or whitespace); default: the first --a run's results")
    args = ap.parse_args()
    tasks = [t for t in re.split(r"[,\s]+", args.tasks.read_text()) if t] if args.tasks else None
    tasks, ea = load(args.a, tasks)
    _, eb = load(args.b, tasks)
    print(json.dumps(compare(ea, eb, len(args.a), len(args.b)), indent=1))


if __name__ == "__main__":
    main()
