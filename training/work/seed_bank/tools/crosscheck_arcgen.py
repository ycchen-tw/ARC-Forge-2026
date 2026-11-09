#!/usr/bin/env python3
"""Draw new tasks with ARC-GEN's native generator and check whether the seed's transform and check_input generalize beyond the original task.

    .venv/bin/python work/seed_bank/tools/crosscheck_arcgen.py work/seed_bank/seeds/training/<id> [...] [-n 300]

ARC-GEN's generate() draws input and output together, so its output can serve as an independent reference answer.
match = transform(input) identical to the ARC-GEN output; raise = transform raised an exception;
check_input_true = number of those ARC-GEN inputs the seed's check_input accepts.
A context_required seed first calls fit() with the original task's train pairs.
Note that ARC-GEN does not always reproduce the original task (e.g. 23b5c85d) and has its own bugs (34cfa167 line 73: bcol should be brow),
so a mismatch is a lead to investigate, not automatically a seed error.
gen_timeout = ARC-GEN generate() exceeded --gen-timeout seconds (some generators' rejection loops may never end);
that sample is skipped; skipped samples skew towards large boards that are hard to lay out, so check the fraction.
Tasks without an ARC-GEN generator print NO_ARCGEN. Any mismatch exits 1.
"""
import argparse, importlib.util, json, os, random, signal, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GEN = os.path.join(ROOT, "data/ARC-GEN")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GenTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise GenTimeout()


def check(seed_dir, n, seed, gen_timeout):
    tid = os.path.basename(os.path.normpath(seed_dir))
    gpath = os.path.join(GEN, "tasks", f"task_{tid}.py")
    if not os.path.exists(gpath):
        return {"task_id": tid, "status": "NO_ARCGEN"}
    sys.path.insert(0, seed_dir)
    for k in ("transform", "generate_input"):
        sys.modules.pop(k, None)
    try:
        tr = load(os.path.join(seed_dir, "transform.py"), "transform")
        gi = load(os.path.join(seed_dir, "generate_input.py"), "generate_input")
        g = load(gpath, f"arcgen_{tid}")
        if hasattr(tr, "fit"):
            meta = json.load(open(os.path.join(seed_dir, "seed.json")))
            task = json.load(open(os.path.join(ROOT, "data/ARC-AGI-2/data", meta["split"], tid + ".json")))
            tr.fit(task["train"])
        random.seed(seed)
        r = {"task_id": tid, "samples": 0, "match": 0, "mismatch": 0, "raise": 0, "gen_fail": 0,
             "gen_timeout": 0, "check_input_true": 0, "first_mismatch": None, "first_raise": None}
        for i in range(n):
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, gen_timeout)
            try:
                e = g.generate()
            except GenTimeout:
                r["gen_timeout"] += 1
                continue
            except Exception:
                r["gen_fail"] += 1
                continue
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            r["samples"] += 1
            inp, out = np.array(e["input"]), np.array(e["output"])
            if gi.check_input(inp.copy()):
                r["check_input_true"] += 1
            try:
                pred = np.asarray(tr.transform(inp.copy()))
            except Exception as x:
                r["raise"] += 1
                r["first_raise"] = r["first_raise"] or f"#{i}: {type(x).__name__}: {str(x)[:120]}"
                continue
            if pred.shape == out.shape and (pred == out).all():
                r["match"] += 1
            else:
                r["mismatch"] += 1
                r["first_mismatch"] = r["first_mismatch"] or f"#{i}: pred {pred.shape} vs ref {out.shape}"
        r["status"] = "MISMATCH" if r["mismatch"] else "OK"
        return r
    finally:
        sys.path.remove(seed_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_dirs", nargs="+")
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--gen-timeout", type=float, default=3.0)
    a = ap.parse_args()
    sys.path.insert(0, GEN)
    sys.path.insert(0, os.path.join(GEN, "tasks"))
    bad = False
    for d in a.seed_dirs:
        r = check(os.path.abspath(d), a.n, a.rng_seed, a.gen_timeout)
        bad |= r["status"] == "MISMATCH"
        print(json.dumps(r, ensure_ascii=False))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
