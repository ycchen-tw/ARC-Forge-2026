"""Audit packets for the Sol review: 40 sampled tasks, one directory each under <R>/audit_sol/<task>/.
  A  15  train passes, a wrong test equals an earlier model's answer (13 sampled + controls ind_0256, ind_0618)
  B  15  train fails, a wrong test equals an earlier model's answer
  C  10  quality200, train fails
Each packet: task.json (with reference test outputs), generation/ (the task's whole package: rule, transform, generator,
acceptance), parents/ (official parent tasks + their seed rule), qadr/ (the solver's solution.py, notes.md, result.json and an
arc.py + task/ so its program can be run), earlier_answers.json. python build_audit.py <R> <scratch> <repo>"""
import csv, io, json, random, re, shutil, subprocess, sys, tarfile, collections
from pathlib import Path
R, S, REPO = map(Path, sys.argv[1:4])
POOLS = Path("$ARC_DATA/arc2_pools"); ARC = Path("$ARC_DATA/ARC-AGI-2/data")
NFS = Path("$ARC_NFS/runs/tito"); Q200 = S / "synthsrc/arc2-quality200-20261009/work/seed_bank/quality200/tasks"
OUT = R / "audit_sol"; assert not OUT.exists(), OUT
rows = {r["task"]: r for r in csv.DictReader(open(R / "analysis/final/tasks.csv"))}
earlier = collections.defaultdict(list)
for d in list((NFS / "20261006-node1-collect/arc2").glob("*-independent1000-b*")) + list((NFS / "20261019-int3-collect/arc2").glob("qual-q*")) \
        + list((NFS / "20261022-int3-qad-collect/arc2").glob("*-q*")):
    if not d.is_dir() or not (d / "submission.json").exists(): continue
    model = json.loads((d / "STATE.json").read_text())["server_info"]["model_path"].split("/")[-1]
    for t, att in json.loads((d / "submission.json").read_text()).items():
        for k, a in enumerate(att):
            for key in ("attempt_1", "attempt_2"):
                if a.get(key) is not None: earlier[t].append({"test": k, "source": f"{d.parent.parent.name}/{d.name} {key}", "model": model, "output": a[key]})
def task_file(t):
    return R / "sets/quality200/tasks" / f"{t}.json" if t.startswith("q2_") else POOLS / ("independent1000" if t.startswith("ind_") else "quality300") / f"{t}.json"
def run_result(t):
    r = rows[t]; return json.loads((R / "runs" / r["run"] / "results" / f"{t}.json").read_text())
def agreed(t):
    task = json.loads(task_file(t).read_text()); res = run_result(t)
    return any(a is not None and a != x["output"] and a != x["input"] and len({v for row in a for v in row}) > 1
               and any(e["test"] == k and e["output"] == a for e in earlier[t])
               for k, (a, x) in enumerate(zip(res["attempt_1"], task["test"])))
pool_sets = ("independent1000", "quality300")
A = sorted(t for t, r in rows.items() if r["set"] in pool_sets and "train-pass" in r["category"] and int(r["agreed_wrong_tests"]) > 0)
B = sorted(t for t, r in rows.items() if r["set"] in pool_sets and "train-fail" in r["category"] and agreed(t))
C = sorted(t for t, r in rows.items() if r["set"] == "quality200" and "train-fail" in r["category"])
rng = random.Random(20261027)
controls = ["ind_0256", "ind_0618"]; assert all(c in A for c in controls)
pick = {"A": controls + rng.sample([t for t in A if t not in controls], 13), "B": rng.sample(B, 15), "C": rng.sample(C, 10)}
print({k: len(v) for k, v in (("A pool", A), ("B pool", B), ("C pool", C))})
def git_pkg(path, dest):
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "origin/main", path], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                rel = Path(m.name).relative_to(path); (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_bytes(tf.extractfile(m).read())
tar_cache = {}
def qadr_files(t, dest):
    r = rows[t]
    for base in (R / "analysis/phase2_suspects" / t, R / "analysis/phase1_unsolved" / t, R / "analysis/phase1_suspects" / t):
        if base.exists():
            for n in ("solution.py", "notes.md"):
                if (base / n).exists(): shutil.copyfile(base / n, dest / n)
            return
    arch = R / "runs" / r["run"] / "tasks.tar.zst"           # train-fail tasks of phase 2: pull from the compressed run
    if arch not in tar_cache:
        raw = subprocess.run(["zstd", "-q", "-dc", str(arch)], check=True, capture_output=True).stdout
        tar_cache[arch] = tarfile.open(fileobj=io.BytesIO(raw))
    tf = tar_cache[arch]
    for n in ("solution.py", "notes.md"):
        try: m = tf.getmember(f"tasks/{t}/{n}")
        except KeyError: continue
        (dest / n).write_bytes(tf.extractfile(m).read())
manifest = []
for group, ts in pick.items():
    for t in ts:
        d = OUT / t; (d / "generation").mkdir(parents=True); (d / "qadr" / "task").mkdir(parents=True); (d / "parents").mkdir()
        task = json.loads(task_file(t).read_text()); (d / "task.json").write_text(json.dumps(task))
        if t.startswith("q2_"): shutil.copytree(Q200 / t[3:], d / "generation", dirs_exist_ok=True)
        elif t.startswith("ind_"): git_pkg(f"work/seed_bank/independent1000/packages/{t}", d / "generation")
        else: git_pkg(f"work/seed_bank/quality300/packages/{t}", d / "generation")
        text = " ".join(p.read_text(errors="replace") for p in (d / "generation").glob("*.json"))
        for pid in sorted(set(re.findall(r"\b[0-9a-f]{8}\b", text))):
            for split in ("training", "evaluation"):
                if (ARC / split / f"{pid}.json").exists():
                    shutil.copyfile(ARC / split / f"{pid}.json", d / "parents" / f"{split}_{pid}.json")
                    seed = REPO / "work/seed_bank/seeds" / split / pid / "seed.json"
                    if seed.exists(): shutil.copyfile(seed, d / "parents" / f"{split}_{pid}.seed.json")
        res = run_result(t); (d / "qadr" / "result.json").write_text(json.dumps({k: res[k] for k in ("attempt_1", "train", "stop", "score")}))
        qadr_files(t, d / "qadr")
        shutil.copyfile(REPO / "work/arc_code/arc2/arc_v10.py", d / "qadr" / "arc.py")
        (d / "qadr" / "task" / "task.json").write_text(json.dumps({"train": task["train"], "test": [{"input": x["input"]} for x in task["test"]]}))
        (d / "earlier_answers.json").write_text(json.dumps(earlier[t]))
        manifest.append({"task": t, "group": group, "set": rows[t]["set"], "category": rows[t]["category"]})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest), "packets;", {g: v for g, v in pick.items()})
