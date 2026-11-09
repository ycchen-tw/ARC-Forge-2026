"""Paired comparison within one run_arc2_seq run: score difference for each pair of arms, task-level
bootstrap 95% CI, and how many tasks each side gets right where the other does not.

    python work/arc_code/arc2/analyze_seq.py <run_dir> [--pairs seq:first,ind:first,seq:ind,refine_seq:seq,refine_exp:ind]

Arms under the same s0 are paired: the difference comes only from the extra sessions, so this is far
more sensitive than the MDE of two independent runs (8.4 points for a single run of A).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_PAIRS = ("seq:first,ind:first,seq:ind,refine_seq:seq,refine_exp:ind,refine_seq:refine_exp,"
                 "rank3:ind,refine_all:rank3,refine_all:refine_exp,verify:first,verify:ind,verify:seq,refine_exp:verify,"
                 "refine_vote:refine_exp")


def task_scores(per_task: dict, arm: str) -> dict[str, float]:
    return {tid: sum(row[arm]) / len(row[arm]) for tid, row in per_task.items()}


def compare(per_task: dict, a: str, b: str, boot: int, seed: int) -> dict:
    sa, sb = task_scores(per_task, a), task_scores(per_task, b)
    tids = sorted(per_task)
    diffs = [sa[t] - sb[t] for t in tids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(boot))
    return {"a": a, "b": b, "diff_points": round(100 * sum(diffs) / len(diffs), 2),
            "ci95": [round(100 * means[int(0.025 * boot)], 2), round(100 * means[int(0.975 * boot) - 1], 2)],
            "a_better_tasks": sorted(t for t, d in zip(tids, diffs) if d > 0),
            "b_better_tasks": sorted(t for t, d in zip(tids, diffs) if d < 0)}


def pattern(grids: list) -> str:
    """Agreement pattern of the candidate outputs: with two, agree / differ / missing; with three or
    more, something like "AC|B" (letters with identical output are joined, "-" means no output)."""
    if len(grids) == 2:
        a, b = grids
        return "missing" if a is None or b is None else ("agree" if a == b else "differ")
    groups: dict[str, str] = {}
    for i, g in enumerate(grids):
        key = "-" if g is None else json.dumps(g)
        groups[key] = groups.get(key, "") + chr(ord("A") + i)
    return "|".join(sorted(v if k != "-" else "-" + v for k, v in groups.items()))


def refiner_breakdown(run_dir: Path, kind: str, others: list[str]) -> dict:
    """Per test pair: the agreement pattern of the candidates (A = s0, then `others`), which candidate
    each of the refiner's two attempts came from (or whether it made a new one), and whether it is right.
    Shows which cell the refiner's gains and losses come from."""
    sols = json.loads((run_dir / "results" / "solutions.json").read_text())
    sample = {"first": 0, "seq": 1, "ind": 2, "refine_seq": 3, "refine_exp": 4, "refine_all": 5}
    cells: dict[str, dict] = {}
    for f in sorted((run_dir / "results").glob(f"*__s{sample[kind]}.json")):
        r = json.loads(f.read_text())
        tid = r["tid"]
        cands = [json.loads((run_dir / "results" / f"{tid}__s{sample[k]}.json").read_text())
                 for k in ("first", *others)]
        for j, gt in enumerate(sols[tid]):
            grids = [c["attempt_1"][j] for c in cands]
            src = []
            for g in r["final"][j]:
                hit = next((chr(ord("A") + i) for i, x in enumerate(grids) if g is not None and x == g), None)
                src.append("none" if g is None else hit or "new")
            key = f"{pattern(grids)} a1={src[0]} a2={src[1]}"
            c = cells.setdefault(key, {"n": 0, "raw_right": 0, "final_right": 0})
            c["n"] += 1
            c["raw_right"] += any(x is not None and x == gt for x in grids)
            c["final_right"] += any(g is not None and g == gt for g in r["final"][j])
    return dict(sorted(cells.items(), key=lambda kv: -kv[1]["n"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    s = json.loads((args.run_dir / "summary.json").read_text())
    arms = set(s["arms"])
    print("pass@2:", json.dumps(s["score_pass2_partial"]), " attempt_1 only:", json.dumps(s["score_attempt1_only"]))
    print("output tokens by arm (M):", {a: round(t / 1e6, 2) for a, t in s["output_tokens_by_arm"].items()})
    print("attempt_2 missing/duplicate (test pairs):", s["attempt2_missing_or_duplicate"],
          " refiner vs raw:", s["refiner_vs_raw_test_pairs"])
    rows = []
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        if not {a, b} <= arms:
            continue
        c = compare(s["per_task"], a, b, args.boot, args.seed)
        rows.append(c)
        print(f"{a:>11} - {b:<11} {c['diff_points']:+6.2f}  CI95 [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  "
              f"{a} better on {len(c['a_better_tasks'])} tasks, {b} better on {len(c['b_better_tasks'])}")
    breakdown = {}
    for kind, others in (("refine_seq", ["seq"]), ("refine_exp", ["ind"]), ("refine_all", ["seq", "ind"])):
        if kind in arms:
            breakdown[kind] = refiner_breakdown(args.run_dir, kind, others)
            print(f"{kind}: candidates / where the final attempts came from -> test pairs, right before, right after")
            for key, c in breakdown[kind].items():
                print(f"  {key:<32} n={c['n']:<4} raw {c['raw_right']:<4} final {c['final_right']}")
    (args.run_dir / "paired.json").write_text(json.dumps({"pairs": rows, "refiner": breakdown}, indent=1) + "\n")


if __name__ == "__main__":
    main()
