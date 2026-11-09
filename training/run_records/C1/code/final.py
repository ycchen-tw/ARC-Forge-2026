"""Final per-task table for the synthetic-task quality check: this run (QAD-R INT3 + arc_v10/v3y, 200k) joined with earlier
collections (prior/<pool>.jsonl) and the audit verdicts. A wrong test whose answer equals an earlier collection's answer
(not a copy of the input, not one colour) while this run's program passes every train pair = "agreed_wrong".
python final.py <R> <prior out dir> <out dir>"""
import csv, json, sys, collections
from pathlib import Path
R, P, OUT = map(Path, sys.argv[1:4]); NFS = Path("$ARC_NFS/runs/tito")
prior_best = collections.defaultdict(float); prior_n = collections.Counter()
for f in P.glob("*.jsonl"):
    for l in open(f):
        r = json.loads(l); prior_best[r["task"]] = max(prior_best[r["task"]], r["score"], r.get("any_of_two", 0))
        prior_n[r["task"]] += 2 if "any_of_two" in r else 1
answers = collections.defaultdict(list)   # earlier answers per task, per test index
for d in list((NFS / "20261006-node1-collect/arc2").glob("*-independent1000-b*")) + list((NFS / "20261006-node1-collect/arc2").glob("*-synth108-b*")) \
        + list((NFS / "20261019-int3-collect/arc2").glob("qual-q*")) + list((NFS / "20261022-int3-qad-collect/arc2").glob("*-q*")):
    if not d.is_dir() or not (d / "submission.json").exists(): continue
    for t, att in json.loads((d / "submission.json").read_text()).items():
        for k, a in enumerate(att):
            for key in ("attempt_1", "attempt_2"):
                if a.get(key) is not None: answers[(t, k)].append(a[key])
verdicts = {}
for f in (R / "analysis/phase1_suspects").glob("verdicts_*.json"):
    for v in json.loads(f.read_text()): verdicts[v["task"]] = v["verdict"]
rows = []
for r in csv.DictReader(open(R / "analysis/all_runs.csv")):
    run = R / "runs" / r["run"]
    res = json.loads((run / "results" / f"{r['task']}.json").read_text())
    task = json.loads((Path(json.loads((run / "STATE.json").read_text())["args"]["tasks_dir"]) / f"{r['task']}.json").read_text())
    agreed = 0
    for k, (a, t) in enumerate(zip(res["attempt_1"], task["test"])):
        if a is None or a == t["output"] or a == t["input"] or len({v for row in a for v in row}) == 1: continue
        if a in answers[(r["task"], k)]: agreed += 1
    tp = "train-pass" in r["category"]
    rows.append({**{k: r[k] for k in ("set", "task", "run", "score", "category", "stop", "n_test", "output_tokens")},
                 "prior_sessions": prior_n[r["task"]], "prior_best": prior_best.get(r["task"], ""),
                 "ever_solved": r["category"] == "solved" or prior_best.get(r["task"]) == 1,
                 "agreed_wrong_tests": agreed if tp else 0, "audit_verdict": verdicts.get(r["task"], "")})
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "tasks.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by = collections.defaultdict(list)
for x in rows: by[x["set"]].append(x)
for s, xs in sorted(by.items()):
    print(f"{s:16s} run {len(xs):4d}: solved now {sum(x['category']=='solved' for x in xs):4d}, train-pass/test-wrong {sum('train-pass' in x['category'] for x in xs):3d} "
          f"(agreed with an earlier answer {sum(x['agreed_wrong_tests']>0 for x in xs):3d}), train-fail {sum('train-fail' in x['category'] for x in xs):3d}")
