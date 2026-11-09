"""Training pool of 90 tasks for arc2-r4 (context 64k): 60 random eval120fix tasks + 30 random true_hard tasks after review
(seed 20261029).

  build_pool.py <out_dir>

Ambiguous tasks are excluded only by the written per-task review verdicts, not by guesses such as "several samples submit the same
wrong answer" (that would drop genuinely hard tasks where the model often converges on another reading):
- eval120fix ($ARC_DATA/arc2_eval120fix/evaluation) minus the 5 tasks in alternates.json (test has another answer train cannot rule
  out), and abc82100 and faa9f03d, judged test-ambiguous in the 2026-10-28 follow-up review and not yet in alternates.json (per-task
  review page https://claude.ai/artifact/GFZdvxeMsnheVYsQ8yqkVw).
- true_hard ($ARC_DATA/arc2_pools/truehard558-20261028) takes only no_counterexample_found from audit549/audit_549.csv (excluding
  confirmed_training_ambiguity 11 and uncertain 12; the 9 restricted116 tasks were not reviewed and are not taken).
- each side is sorted, then drawn in order with the same random.Random(SEED); writes <out_dir>/<id>.json, <out_dir>.train.txt
  (90 tasks) and <out_dir>.selection.json (candidates, exclusions and reasons). Raises if out_dir already exists.
"""
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEED = 20261029
N_EVAL, N_HARD = 60, 30
#: eval120fix tasks judged test-ambiguous in the 2026-10-28 follow-up review and not yet in alternates.json
AUDITED_AMBIGUOUS = {"abc82100": "audit_ambiguous", "faa9f03d": "audit_ambiguous"}


def main(out: Path) -> None:
    data = Path(os.environ["ARC_DATA"])
    if out.exists():
        raise SystemExit(f"{out} exists")
    eval_dir, hard_dir = data / "arc2_eval120fix/evaluation", data / "arc2_pools/truehard558-20261028"
    alternates = set(json.loads((ROOT / "work/arc_code/arc2/eval120fix/alternates.json").read_text())["tasks"])
    eval_ids = sorted(p.stem for p in eval_dir.glob("*.json"))
    if len(eval_ids) != 120 or not (alternates | set(AUDITED_AMBIGUOUS)) <= set(eval_ids) or alternates & set(AUDITED_AMBIGUOUS):
        raise SystemExit("eval120fix: expected 120 tasks containing alternates.json and AUDITED_AMBIGUOUS (disjoint)")
    eval_out = {**{t: "alternates" for t in sorted(alternates)}, **AUDITED_AMBIGUOUS}
    audit = list(csv.DictReader((HERE / "audit549/audit_549.csv").open(encoding="utf-8-sig")))
    hard_out = {r["task"]: r["outcome"] for r in audit if r["outcome"] != "no_counterexample_found"}
    rng = random.Random(SEED)
    eval_pick = sorted(rng.sample([t for t in eval_ids if t not in eval_out], N_EVAL))
    hard_pick = sorted(rng.sample(sorted(r["task"] for r in audit if r["task"] not in hard_out), N_HARD))
    out.mkdir(parents=True)
    for src, ids in ((eval_dir, eval_pick), (hard_dir, hard_pick)):
        for t in ids:
            d = json.loads((src / f"{t}.json").read_text())
            if not (d["train"] and d["test"] and all("output" in p for p in d["test"])):
                raise SystemExit(f"{src / t}.json: needs train pairs and test outputs")
            shutil.copyfile(src / f"{t}.json", out / f"{t}.json")
    ids = eval_pick + hard_pick
    if len(set(ids)) != N_EVAL + N_HARD:
        raise SystemExit("duplicate ids across the two sources")
    out.with_name(out.name + ".train.txt").write_text("\n".join(ids) + "\n")
    out.with_name(out.name + ".selection.json").write_text(json.dumps(
        {"seed": SEED, "eval120fix": {"candidates": len(eval_ids) - len(eval_out), "excluded": eval_out, "picked": eval_pick},
         "true_hard": {"audited": len(audit), "candidates": len(audit) - len(hard_out), "excluded": hard_out,
                       "picked": hard_pick}}, indent=1) + "\n")
    print(f"eval120fix {len(eval_ids) - len(eval_out)} candidates -> {N_EVAL}; true_hard {len(audit) - len(hard_out)} "
          f"candidates -> {N_HARD}; {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
