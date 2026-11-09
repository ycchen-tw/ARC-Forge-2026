"""True-hard list for training: rule determined by the demos + reference right + hard for the model.
audited_hard = Sol VALID_HARD (audit_sol, audit_sol2) or phase-1 MODEL_WRONG; solved_hard = solved at least once (so the demos
lead to the reference) but either mixed across attempts or solved only with >=100k output tokens (this run or FP8 10-06).
true_hard.py <run dir> <tasks.csv> <prior sessions dir> <FP8 per-task csv> -> <run dir>/analysis/final/{true_hard.csv,verdicts_all.json}"""
import csv, json, sys, collections
from pathlib import Path
R, TASKS, PRIOR, FP8 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
rows = {r["task"]: r for r in csv.DictReader(open(TASKS))}
verd = {}
for f in ("verdicts_q200_cp.json", "verdicts_hs_rc.json"):
    for v in json.loads((R / "analysis/phase1_suspects" / f).read_text()):
        assert v["task"] in rows, v["task"]
        verd[v["task"]] = {"source": "phase1", "verdict": {"MODEL_WRONG": "VALID_HARD"}.get(v["verdict"], v["verdict"]),
                           "fix": v["fix"], "rule_quality": "", "human_difficulty": "", "summary_zh": ""}
pending = []
for a in ("audit_sol", "audit_sol2"):
    for m in json.loads((R / a / "manifest.json").read_text()):
        t = m["task"]; f = R / a / t / "verdict.json"
        if not f.exists(): pending.append(t); continue
        v = json.loads(f.read_text())
        # Sol sometimes writes the package id (n334) instead of the run id (q2_n334): the packet directory is authoritative
        assert v["task"] == t or (t.startswith("q2_") and v["task"] == t[3:]), (t, v["task"])
        assert t not in verd, t
        verd[t] = {"source": a, **{k: v[k] for k in ("verdict", "fix", "rule_quality", "human_difficulty", "summary_zh")}}
att = collections.defaultdict(list); pool = {}
for f in PRIOR.glob("*.jsonl"):
    for l in open(f):
        # seq batches (10-19 / 10-22) carry any_of_two = s0 or the independent s2 per test pair: s0 failing while the pair
        # is right counts as one failed and one solved attempt
        x = json.loads(l); att[x["task"]].append(x["score"] == 1); pool[x["task"]] = x["pool"]
        if x.get("any_of_two") == 1 and x["score"] < 1: att[x["task"]].append(True)
for t, r in rows.items():
    if r["run"].startswith("p2-"): att[t].append(r["category"] == "solved")
fp8 = {r["id"]: r for r in csv.DictReader(open(FP8))}
# universe: this run's 781 tasks plus the pool tasks fully solved before (not rerun this time)
for t in set(att) | set(fp8):
    if t in rows: continue
    xs = att.get(t, []); s = pool[t] if t in pool else "independent1000"
    assert any(xs) or (t in fp8 and float(fp8[t]["score"]) == 1), t
    rows[t] = {"set": s, "run": "p2-prior-only", "category": "not rerun", "output_tokens": "0", "ever_solved": "True"}
out = []
for t, r in rows.items():
    v = verd.get(t, {}); xs = att.get(t, []) if r["run"].startswith("p2-") else [r["category"] == "solved"]
    mixed = any(xs) and not all(xs) and len(xs) >= 2
    heavy = r["category"] == "solved" and int(r["output_tokens"]) >= 100000
    fheavy = t in fp8 and float(fp8[t]["score"]) == 1 and int(fp8[t]["tokens"]) >= 100000
    if v.get("verdict") == "VALID_HARD": tier = "audited_hard"
    elif r["ever_solved"] == "True" and (mixed or heavy or fheavy): tier = "solved_hard"
    else: continue
    out.append({"set": r["set"], "task": t, "tier": tier, "audit": v.get("source", ""), "rule_quality": v.get("rule_quality", ""),
                "human_difficulty": v.get("human_difficulty", ""), "attempts": len(xs), "solves": sum(xs),
                "mixed": int(mixed), "this_run_category": r["category"], "this_run_tokens": r["output_tokens"],
                "fp8_tokens_if_solved": fp8[t]["tokens"] if t in fp8 and float(fp8[t]["score"]) == 1 else "",
                "summary_zh": v.get("summary_zh", "")})
D = R / "analysis/final"
with open(D / "true_hard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(sorted(out, key=lambda x: (x["tier"], x["set"], x["task"])))
(D / "verdicts_all.json").write_text(json.dumps({t: verd[t] for t in sorted(verd)}, ensure_ascii=False, indent=1))
print("pending audits:", len(pending))
print("verdicts:", collections.Counter((v["source"], v["verdict"]) for v in verd.values()))
print("true hard:", collections.Counter((x["tier"], x["set"]) for x in out))
print("audited_hard with rule_quality <= 2:", [x["task"] for x in out if x["rule_quality"] != "" and int(x["rule_quality"]) <= 2])
