#!/usr/bin/env python3
"""Aggregate the quality grading of training seeds as "parents for new tasks".

    .venv/bin/python work/seed_bank/tools/seed_quality.py <crosscheck_dir> --out work/seed_bank/results/training_seed_quality.json

Input is crosscheck_arcgen.py's JSON output per task (one file <id>.json per task). Grades:
- avoid: seed.json verification.status is FAIL (unresolved differences on the official pairs).
- caution: in ARC-GEN sampling, mismatch ≥ 10% or raise ≥ 10% (the rule or generation domain clearly differs from the ARC-GEN family).
- note: a few mismatches, or rule_status is bounded_hypothesis.
- ok: everything else (including tasks with no ARC-GEN generator, verified only on the official pairs).
A mismatch is only a lead: ARC-GEN itself may have bugs or not reproduce the original task, see crosscheck_arcgen.py.
"""
import argparse, glob, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SEEDS = os.path.join(ROOT, "work/seed_bank/seeds/training")


def last_json(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"{path} is empty (crosscheck timed out or crashed)")
    return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crosscheck_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.10)
    a = ap.parse_args()
    ids = sorted(os.listdir(SEEDS))
    tasks = {}
    for tid in ids:
        seed = json.load(open(os.path.join(SEEDS, tid, "seed.json")))
        cc = last_json(os.path.join(a.crosscheck_dir, tid + ".json"))
        assert cc["task_id"] == tid, (tid, cc["task_id"])
        reasons = []
        if seed["verification"]["status"] != "PASS":
            tier = "avoid"
            reasons.append("official pairs not all passed (unresolved)")
        else:
            tier = "ok"
        if cc["status"] != "NO_ARCGEN":
            n = cc["samples"]
            if n == 0:
                raise RuntimeError(f"{tid} ARC-GEN has no valid samples")
            mis, rz = cc["mismatch"] / n, cc["raise"] / n
            if mis or rz:
                reasons.append(f"ARC-GEN {n} samples: mismatch {cc['mismatch']}, raise {cc['raise']}")
            if tier == "ok" and (mis >= a.threshold or rz >= a.threshold):
                tier = "caution"
            elif tier == "ok" and cc["mismatch"]:
                tier = "note"
        if seed.get("rule_status") == "bounded_hypothesis":
            reasons.append("bounded_hypothesis: holds only within the stated generation domain")
            if tier == "ok":
                tier = "note"
        tasks[tid] = {"tier": tier, "arcgen": cc["status"] != "NO_ARCGEN", "reasons": reasons}
    counts = {}
    for t in tasks.values():
        counts[t["tier"]] = counts.get(t["tier"], 0) + 1
    out = {
        "purpose": "quality grading of training seeds as parents for new tasks; avoid is not recommended as a parent, caution needs a manual look at the rule and the ARC-GEN differences first",
        "crosscheck": {"tool": "work/seed_bank/tools/crosscheck_arcgen.py", "n": 100, "rng_seed": 7, "gen_timeout_s": 3},
        "threshold": a.threshold,
        "counts": counts,
        "tasks": tasks,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
