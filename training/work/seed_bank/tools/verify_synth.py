#!/usr/bin/env python3
"""Verify one synthetic seed directory (no original task; pairs come from task.json in the directory).

    .venv/bin/python work/seed_bank/tools/verify_synth.py work/seed_bank/synth/<id>

Checks: schema / taxonomy / task.json structure and consistency (output == transform(input)) / check_input on every task.json input /
four checks on 30 generated inputs. The result is written back to seed.json's verification field; any failure exits 1.
"""
import importlib.util, json, sys, os, re
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TAX = yaml.safe_load(open(os.path.join(ROOT, "work/concept_extract/taxonomy.yaml")))
TAX_IDS = {c["id"] for c in TAX["concepts"]}
REQ_TOP = ["task_id", "split", "origin", "rule", "concepts", "concepts_free", "difficulty", "provenance", "verification"]
REQ_RULE = ["intent", "input", "entities", "procedure", "output_size", "decided_by"]
REQ_PROV = ["parents", "method"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def valid_grid(g):
    g = np.asarray(g)
    return g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30 and np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9


def main(seed_dir):
    errs = []
    seed_path = os.path.join(seed_dir, "seed.json")
    seed = json.load(open(seed_path))
    for k in REQ_TOP:
        if k not in seed: errs.append(f"seed.json missing field {k}")
    if seed.get("split") != "synth" or seed.get("origin") != "synth": errs.append("split and origin must both be 'synth'")
    prov = seed.get("provenance", {})
    for k in REQ_PROV:
        if k not in prov: errs.append(f"provenance missing field {k}")
    if not (isinstance(prov.get("parents"), list) and prov["parents"]
            and all(isinstance(parent, str) for parent in prov["parents"])
            and len(set(prov["parents"])) == len(prov["parents"])):
        errs.append("provenance.parents must be a non-empty list of distinct task_ids")
    rule = seed.get("rule", {})
    for k in REQ_RULE:
        if k not in rule: errs.append(f"rule missing field {k}")
    inp = rule.get("input", {})
    for k in ("invariants", "varies", "distractors"):
        if not isinstance(inp.get(k), list) or not inp.get(k): errs.append(f"rule.input.{k} must be a non-empty list")
    if not isinstance(rule.get("procedure"), list) or not rule.get("procedure"): errs.append("procedure must be a non-empty list")
    ops = []
    for i, st in enumerate(rule.get("procedure") or []):
        if not isinstance(st, dict) or "op" not in st or "text" not in st: errs.append(f"procedure[{i}] must have op and text"); continue
        ops.append(st["op"])
        if st["op"] not in TAX_IDS: errs.append(f"procedure[{i}].op={st['op']} not in taxonomy")
    for c in seed.get("concepts", []):
        if c not in TAX_IDS: errs.append(f"concepts contains non-taxonomy id: {c}")
    for op in ops:
        if op in TAX_IDS and op not in seed.get("concepts", []): errs.append(f"procedure uses {op} but concepts does not list it")
    if "object_detection" in seed.get("concepts", []) and not rule.get("entities"): errs.append("concepts contains object_detection but entities is empty")
    db = rule.get("decided_by", {})
    if not db.get("decided"): errs.append("decided_by.decided must not be empty: a synthetic task must explain how the task.json pairs decide the rule")
    for i, d in enumerate(db.get("decided", []) or []):
        ev = d.get("evidence", "") if isinstance(d, dict) else ""
        if not re.search(r"(train|test)\s*\d", ev): errs.append(f"decided[{i}] evidence cites no pair: {ev!r}")
    if not isinstance(db.get("undecided"), list): errs.append("decided_by.undecided must be a list (may be empty)")
    def has_zh(s): return bool(re.search(r"[一-鿿]", s or ""))
    if not has_zh(rule.get("intent", "")): errs.append("intent is not Chinese")
    if not all(has_zh(s) for s in inp.get("invariants", [])): errs.append("invariants has non-Chinese entries")
    # ---- task.json ----
    task_path = os.path.join(seed_dir, "task.json")
    if not os.path.exists(task_path):
        errs.append("missing task.json"); task = {"train": [], "test": []}
    else:
        task = json.load(open(task_path))
    if len(task.get("train", [])) < 3: errs.append(f"task.json needs at least 3 train pairs, has {len(task.get('train', []))}")
    if len(task.get("test", [])) < 1: errs.append("task.json needs at least 1 test pair")
    pairs = task.get("train", []) + task.get("test", [])
    for i, p in enumerate(pairs):
        if not (valid_grid(np.array(p["input"])) and valid_grid(np.array(p["output"]))): errs.append(f"task.json pair {i} has an invalid grid")
    n_uniq_pairs = len({np.array(p["input"]).tobytes() + bytes(str(np.array(p["input"]).shape), "ascii") for p in pairs})
    if n_uniq_pairs < len(pairs): errs.append("task.json has duplicate inputs")
    sys.path.insert(0, os.path.abspath(seed_dir))
    for m in ("transform", "generate_input"): sys.modules.pop(m, None)
    T = load(os.path.join(seed_dir, "transform.py"), "transform")
    sys.modules["transform"] = T
    G = load(os.path.join(seed_dir, "generate_input.py"), "generate_input")
    ok = 0; fails = []
    for i, p in enumerate(pairs):
        tag = f"train {i}" if i < len(task["train"]) else f"test {i - len(task['train'])}"
        try:
            out = np.asarray(T.transform(np.array(p["input"])))
            if out.shape == np.array(p["output"]).shape and np.array_equal(out, np.array(p["output"])): ok += 1
            else: fails.append(tag)
            if np.array_equal(np.array(p["input"]), np.array(p["output"])): fails.append(f"{tag}: output == input")
        except Exception as e:
            fails.append(f"{tag}: {type(e).__name__}: {e}")
    if fails: errs.append(f"task.json and transform disagree: {fails}")
    bad = [i for i, p in enumerate(pairs) if not G.check_input(np.array(p["input"]))]
    if bad: errs.append(f"task.json inputs fail check_input: pairs {bad}")
    rng = np.random.default_rng(0)
    gens = []
    for k in range(30):
        try: gens.append(np.asarray(G.generate_input(rng)))
        except Exception as e: errs.append(f"generate_input raised on attempt {k}: {type(e).__name__}: {e}"); break
    n_valid = sum(valid_grid(g) for g in gens)
    n_uniq = len({g.tobytes() + bytes(str(g.shape), "ascii") for g in gens})
    n_chk = sum(bool(G.check_input(g)) for g in gens)
    n_run = 0; n_changed = 0; n_multicolor = 0
    for i, g in enumerate(gens):
        try:
            o = np.asarray(T.transform(g.copy()))
            if valid_grid(o): n_run += 1
            if o.shape != g.shape or not np.array_equal(o, g): n_changed += 1
            if len(np.unique(o)) >= 2: n_multicolor += 1
        except Exception as exc:
            errs.append(f"transform raised on generated input {i}: {type(exc).__name__}: {exc}")
    if len(gens) == 30:
        if n_valid < 30: errs.append(f"only {n_valid}/30 generated grids are valid")
        if n_uniq < 28: errs.append(f"only {n_uniq}/30 generated inputs are unique")
        if n_chk < 30: errs.append(f"only {n_chk}/30 generated inputs pass check_input")
        if n_run < 30: errs.append(f"transform runs with a valid output on only {n_run}/30 generated inputs")
        if n_changed < 30: errs.append(f"transform output equals input on {30 - n_changed}/30")
        if n_multicolor < 30: errs.append(f"output degenerates to a single colour on {30 - n_multicolor}/30")
    seed["verification"] = {
        "task_pairs": f"{ok}/{len(pairs)}", "check_input_on_task": not bad,
        "gen_valid": f"{n_valid}/30", "gen_unique": f"{n_uniq}/30", "gen_check_input": f"{n_chk}/30",
        "gen_transform_runs": f"{n_run}/30", "gen_output_differs": f"{n_changed}/30", "gen_output_multicolor": f"{n_multicolor}/30",
        "blind_solve": None, "status": "PASS" if not errs else "FAIL", "errors": errs,
    }
    json.dump(seed, open(seed_path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(seed["verification"], ensure_ascii=False, indent=1))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
