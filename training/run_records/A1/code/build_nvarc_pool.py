"""NVARC nvarc_training (Kaggle sorokin/nvarc-synthetic-puzzles, 47,337 tasks, every parent an ARC-AGI-2 training task;
each file is [{"input","output"}, ...]) -> run_arc2's --tasks-dir layout (<uid>.json, {"train":[...],"test":[{input,output}]}).

Pairs are chosen exactly as work/agentic_tir/agentic_tir.py::choose_pairs (= qwen38fn_distill's naive corpus, attempt 0,
n_train 4, n_test 1), so a uid gets the same pairs in both corpora. Tasks whose grids fail run_arc2.load_tasks' conditions
(rectangular, 0-9, sides 1-30) are skipped and counted. Also writes order.txt: parent1 families in turn (within a family by
the uid's sha256), so the first batches have the most varied rules.

    python build_nvarc_pool.py <zip> <out_dir>
"""
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def choose_pairs(uid: str, attempt: int, n_pairs: int, n_train: int, n_test: int = 1):
    h = hashlib.sha256(f"{uid}:{attempt}".encode()).digest()
    order = sorted(range(n_pairs), key=lambda i: hashlib.sha256(h + i.to_bytes(2, "big")).digest())
    chosen = order[: n_train + n_test]
    return sorted(chosen[:n_train]), chosen[n_train:n_train + n_test]


def grid_ok(g) -> bool:
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) for r in g)
            and 1 <= len(g[0]) <= 30 and all(len(r) == len(g[0]) for r in g)
            and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))


def main() -> None:
    zpath, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    fam = defaultdict(list)
    n_ok = n_bad = n_short = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not (name.startswith("nvarc_training/") and name.endswith(".json")):
                continue
            uid = Path(name).stem
            pairs = json.loads(z.read(name))
            if len(pairs) < 5:
                n_short += 1
                continue
            tr, te = choose_pairs(uid, 0, len(pairs), 4, 1)
            task = {"train": [{"input": pairs[i]["input"], "output": pairs[i]["output"]} for i in tr],
                    "test": [{"input": pairs[i]["input"], "output": pairs[i]["output"]} for i in te]}
            if not all(grid_ok(p["input"]) and grid_ok(p["output"]) for p in task["train"] + task["test"]):
                n_bad += 1
                continue
            (out / f"{uid}.json").write_text(json.dumps(task, separators=(",", ":")))
            fam[uid.split("_")[0]].append(uid)
            n_ok += 1
    for f in fam.values():
        f.sort(key=lambda u: hashlib.sha256(u.encode()).digest())
    order, k = [], 0
    while len(order) < n_ok:
        for p in sorted(fam):
            if k < len(fam[p]):
                order.append(fam[p][k])
        k += 1
    (out.parent / f"{out.name}.order.txt").write_text("\n".join(order) + "\n")
    print(json.dumps({"written": n_ok, "bad_grid": n_bad, "too_few_pairs": n_short, "families": len(fam)}))


if __name__ == "__main__":
    main()
