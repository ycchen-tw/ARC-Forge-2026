"""Four PR synthetic sets -> run_arc2 task files ({train, test with output}), one dir per set; grids over 30 are left out
(listed in skipped.json). Every task is checked like run_arc2.check_tasks. python convert.py <synthsrc dir> <out dir>"""
import json, re, sys, zipfile
from pathlib import Path
X, OUT = Path(sys.argv[1]), Path(sys.argv[2])

def ok_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and len(r) == len(g[0]) for r in g)
            and 1 <= len(g[0]) <= 30 and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))

def grids(task):
    for p in task["train"]: yield p["input"]; yield p["output"]
    for p in task["test"]: yield p["input"]; yield p["output"]

def put(set_name, tid, task, origin, skipped, written):
    assert task["train"] and task["test"] and all("output" in t for t in task["test"]), (set_name, tid)
    task = {"train": [{"input": p["input"], "output": p["output"]} for p in task["train"]],
            "test": [{"input": p["input"], "output": p["output"]} for p in task["test"]]}
    if not all(ok_grid(g) for g in grids(task)):
        side = max(max(len(g), len(g[0])) for g in grids(task))
        skipped.append({"task": tid, "origin": origin, "max_side": side}); return
    d = OUT / set_name / "tasks"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"{tid}.json"
    assert not f.exists(), f
    f.write_text(json.dumps(task, separators=(",", ":")))
    written.append({"task": tid, "origin": origin})

def finish(set_name, commit, skipped, written):
    (OUT / set_name / "manifest.json").write_text(json.dumps(
        {"source_commit": commit, "written": written, "skipped_over_30": skipped}, indent=1))
    print(f"{set_name}: {len(written)} tasks, {len(skipped)} skipped (side > 30)")

# PR8 quality200: task.json is standard ARC JSON
src = X / "arc2-quality200-20261009"; sk, wr = [], []
for f in sorted((src / "work/seed_bank/quality200/tasks").glob("*/task.json")):
    put("quality200", f"q2_{f.parent.name}", json.loads(f.read_text()), str(f.relative_to(src)), sk, wr)
finish("quality200", (src / "COMMIT").read_text().strip(), sk, wr)

# PR10 restricted checkpoint: dataset.py export seed 0 ordinary -> solver.json + answers.private.json joined by episode id
src = X / "research_arc2-restricted-checkpoint-20261015"; sk, wr = [], []
for d in sorted((X / "pr10_export").iterdir()):
    if not d.is_dir(): continue
    sol = json.loads((d / "solver.json").read_text())["episodes"]; ans = json.loads((d / "answers.private.json").read_text())["episodes"]
    assert sorted(sol) == sorted(ans) and len(sol) == 1, d
    eid = next(iter(sol)); ep = sol[eid]
    assert len(ans[eid]) == len(ep["test"]), d
    task = {"train": ep["train"], "test": [{"input": t["input"], "output": o} for t, o in zip(ep["test"], ans[eid])]}
    put("restricted116", f"rc_{d.name}_s0", task, f"dataset.py export --task {d.name} --seed 0 --kind ordinary", sk, wr)
finish("restricted116", (src / "COMMIT").read_text().strip(), sk, wr)

# PR11 hard-seed size156: public.zip (test input only) + answers.private.zip joined by file name
src = X / "research_hard-seed-size156-20261016"; base = src / "work/seed_bank/hard_seed_size156/archives"; sk, wr = [], []
with zipfile.ZipFile(base / "public.zip") as zp, zipfile.ZipFile(base / "answers.private.zip") as za:
    pub = {Path(n).name: n for n in zp.namelist() if n.endswith(".json")}
    ansn = {Path(n).name: n for n in za.namelist() if n.endswith(".json")}
    assert sorted(pub) == sorted(ansn) and len(pub) == 156, (len(pub), len(ansn))
    for name in sorted(pub):
        p = json.loads(zp.read(pub[name])); a = json.loads(za.read(ansn[name]))
        assert len(a["test"]) == len(p["test"]), name
        task = {"train": p["train"], "test": [{"input": t["input"], "output": o["output"]} for t, o in zip(p["test"], a["test"])]}
        put("size156", f"hs_{Path(name).stem}", task, f"{a['episode_id']}", sk, wr)
finish("size156", (src / "COMMIT").read_text().strip(), sk, wr)

# PR12 compositions: public/tasks/composition_NNN.json + evaluator/composition_NNN.answers.json (role query, in order)
src = X / "research_compositions105-frozen-20261020"; base = src / "work/seed_bank/compositions106_frozen"; sk, wr = [], []
for f in sorted((base / "public/tasks").glob("composition_*.json")):
    p = json.loads(f.read_text()); a = json.loads((base / "evaluator" / f"{f.stem}.answers.json").read_text())
    q = [x for x in a if x["role"] == "query"]
    # public tests carry no id: queries pair with them by order. The ids come in 26 shapes (query_00, Q0, 16b_C02_Q00_E2, ...);
    # the one number that differs between a task's queries must count up by one in file order (from 0, or from 1: QV3_01).
    nums = [[int(n) for n in re.findall(r"\d+", x["id"])] for x in q]
    assert len(q) == len(p["test"]) >= 2 and len({len(n) for n in nums}) == 1, f
    vary = [k for k in range(len(nums[0])) if len({n[k] for n in nums}) > 1]
    seq = [n[vary[0]] for n in nums] if len(vary) == 1 else None
    assert seq is not None and seq[0] in (0, 1) and seq == list(range(seq[0], seq[0] + len(q))), (f, [x["id"] for x in q])
    task = {"train": p["train"], "test": [{"input": t["input"], "output": x["output"]} for t, x in zip(p["test"], q)]}
    put("compositions105", f"cp_{f.stem.split('_')[1]}", task, str(f.relative_to(src)), sk, wr)
finish("compositions105", (src / "COMMIT").read_text().strip(), sk, wr)
