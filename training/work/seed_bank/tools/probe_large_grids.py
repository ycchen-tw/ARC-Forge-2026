#!/usr/bin/env python3
"""CPU opportunity audit, not a corpus acceptance gate.

Scan all official seed families, then probe native ARC-GEN size arguments.
Original seeds and upstream generators are never rewritten. Run artifacts and
sample grids go to ARC_RUNS; compact receipts can be published separately.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from arcforge import paths

BANK = paths.ROOT / "work/seed_bank"
DIM_ARGS = {"width", "height", "size", "length", "extended"}
CANVAS_NAMES = {"h", "w", "H", "W", "n", "size", "side", "height", "width", "N", "R", "C"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def evidence(path):
    tree = ast.parse(path.read_text())
    assignments, caps, allocations = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = set().union(*(names(t) for t in node.targets))
            if targets & CANVAS_NAMES:
                assignments.append({"line": node.lineno, "code": ast.unparse(node)})
        if isinstance(node, ast.Compare) and any(
                isinstance(n, ast.Constant) and n.value == 30 for n in ast.walk(node)):
            caps.append({"line": node.lineno, "code": ast.unparse(node)})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"zeros", "ones", "full", "empty", "grid", "grids"}:
                allocations.append({"line": node.lineno, "code": ast.unparse(node)})
    return {"sha256": digest(path), "canvas_assignments": assignments,
            "literal_30_comparisons": caps, "allocations": allocations}


def inventory(out):
    arcgen = paths.data() / "ARC-GEN"
    official = paths.data() / "ARC-AGI-2/data"
    quality = json.loads((BANK / "results/training_seed_quality.json").read_text())["tasks"]
    rows = []
    for split in ("training", "evaluation"):
        expected = {p.stem for p in (official / split).glob("*.json")}
        directories = sorted((BANK / "seeds" / split).glob("*/seed.json"))
        assert {p.parent.name for p in directories} == expected
        for meta_path in directories:
            d = meta_path.parent
            meta = json.loads(meta_path.read_text())
            gp, tp = d / "generate_input.py", d / "transform.py"
            fn = functions(ast.parse(gp.read_text()))["generate_input"]
            native = arcgen / "tasks" / f"task_{d.name}.py"
            native_info = None
            if native.exists():
                f = functions(ast.parse(native.read_text()))["generate"]
                args = [a.arg for a in f.args.args]
                reassigned = set()
                for n in ast.walk(f):
                    if isinstance(n, ast.Assign):
                        reassigned.update(set().union(*(names(t) for t in n.targets)))
                native_info = {"sha256": digest(native), "signature": ast.unparse(f.args),
                               "size_args": sorted(DIM_ARGS & set(args)),
                               "size_args_reassigned": sorted(DIM_ARGS & set(args) & reassigned),
                               "source_evidence": evidence(native)}
            rows.append({
                "split": split, "task_id": d.name,
                "seed_source": d.relative_to(paths.ROOT).as_posix(),
                "seed_sha256": digest(meta_path),
                "official_sha256": digest(official / split / f"{d.name}.json"),
                "historical_status": meta["verification"]["status"],
                "quality": quality[d.name]["tier"] if split == "training" else None,
                "context_required": meta.get("context_required", False),
                "finite_domain": meta.get("finite_domain"),
                "intent": meta["rule"]["intent"], "output_size": meta["rule"]["output_size"],
                "varies": meta["rule"]["input"]["varies"],
                "undecided": meta["rule"]["decided_by"]["undecided"],
                "size_hint_read": any(isinstance(n, ast.Name) and n.id == "size_hint"
                                      and isinstance(n.ctx, ast.Load) for n in ast.walk(fn)),
                "generator": evidence(gp), "transform": evidence(tp), "native": native_info,
            })
    receipt = {"repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "arcgen_head": subprocess.check_output(["git", "-C", str(arcgen), "rev-parse", "HEAD"], text=True).strip(),
               "arcgen_common_sha256": digest(arcgen / "common.py"),
               "tool_sha256": digest(__file__), "tasks": rows}
    write_json(out / "inventory.json", receipt)
    print(json.dumps({"tasks": len(rows), "native": sum(r["native"] is not None for r in rows),
                      "native_size_args": sum(bool(r["native"] and r["native"]["size_args"]) for r in rows)}))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ProbeTimeout(Exception):
    pass


class AllocationLimit(Exception):
    pass


def timed(seconds, fn):
    def alarm(signum, frame):
        raise ProbeTimeout(f"exceeded {seconds}s")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def grid_info(grid):
    import numpy as np
    a = np.asarray(grid)
    if (a.ndim != 2 or not a.size or not np.issubdtype(a.dtype, np.integer)
            or a.min() < 0 or a.max() > 9):
        raise ValueError("invalid nonempty integer/color grid")
    counts = np.unique(a, return_counts=True)[1]
    return {"shape": list(a.shape), "cells": int(a.size),
            "nonmodal_cells": int(a.size - counts.max()), "colors": len(counts)}


def adapt_sampler(source, target):
    """Keep the random-construction branch; change dimension assignments only.

    Passing width=64 often skips `if width is None`, leaving objects unset.
    These edits are explicit *candidate extensions*, not native API support.
    """
    tree = ast.parse(source)
    fn = functions(tree)["generate"]
    controls = {a.arg for a in fn.args.args} & DIM_ARGS
    edits = []
    offset = len(fn.args.args) - len(fn.args.defaults)
    for i, default in enumerate(fn.args.defaults):
        name = fn.args.args[offset + i].arg
        if name in controls and isinstance(default, ast.Constant) and isinstance(default.value, int):
            edits.append({"line": fn.lineno, "before": f"default {name}={default.value}", "after": f"default {name}={target}"})
            fn.args.defaults[i] = ast.Constant(target)
    def replacement(lhs, rhs):
        if isinstance(lhs, ast.Name) and lhs.id in controls and lhs.id not in names(rhs):
            return ast.copy_location(ast.Constant(target), rhs)
        if isinstance(lhs, (ast.Tuple, ast.List)) and isinstance(rhs, (ast.Tuple, ast.List)) and len(lhs.elts) == len(rhs.elts):
            rhs.elts = [replacement(a, b) for a, b in zip(lhs.elts, rhs.elts)]
        return rhs
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            before = ast.unparse(node)
            node.value = replacement(node.targets[0], node.value)
            after = ast.unparse(node)
            if before != after:
                edits.append({"line": node.lineno, "before": before, "after": after})
    return ast.fix_missing_locations(tree), edits


def native_probe(task_id, out, sizes, samples, adapt=False):
    import numpy as np
    root = paths.data() / "ARC-GEN"
    sys.path.insert(0, str(root))
    common = load(root / "common.py", "common")
    # Probe-only allocation guard: avoid size**2 outputs allocating gigabytes.
    original_grid = common.grid
    def bounded_grid(width, height, *args, **kwargs):
        if width > 512 or height > 512 or width * height > 262144:
            raise AllocationLimit(f"requested {height}x{width}; probe limit 512 per axis")
        return original_grid(width, height, *args, **kwargs)
    common.grid = bounded_grid
    g = load(root / "tasks" / f"task_{task_id}.py", "native_generator")
    d = BANK / "seeds/training" / task_id
    sys.path.insert(0, str(d))
    tr = load(d / "transform.py", "transform")
    gi = load(d / "generate_input.py", "generate_input")
    meta = json.loads((d / "seed.json").read_text())
    if meta.get("context_required"):
        official = json.loads((paths.data() / "ARC-AGI-2/data/training" / f"{task_id}.json").read_text())
        timed(2, lambda: tr.fit(official["train"]))
    args = set(inspect.signature(g.generate).parameters) & DIM_ARGS
    rows = []
    for target in [0] + (sizes if args else []):
        kw = {k: target for k in args} if target else {}
        group = {"target": target or "default", "kwargs": kw, "samples": []}
        active = g
        if adapt and target:
            tree, edits = adapt_sampler((root / "tasks" / f"task_{task_id}.py").read_text(), target)
            active = types.ModuleType("adapted_native_generator")
            exec(compile(tree, "<in-memory-native-sampler>", "exec"), active.__dict__)
            kw = {}
            group.update(kwargs=kw, sampler_edits=edits)
        for seed in range(samples if target else 1):
            row = {"rng_seed": 20261021 + seed}
            random.seed(row["rng_seed"])
            try:
                pair = timed(1.5, lambda: active.generate(**kw))
                x, y = np.asarray(pair["input"]), np.asarray(pair["output"])
                row.update(input=grid_info(x), output=grid_info(y), generated=True)
                random.seed(row["rng_seed"])
                replay = timed(1.5, lambda: active.generate(**kw))
                row["rng_replay"] = bool(np.array_equal(x, replay["input"]) and np.array_equal(y, replay["output"]))
                row["identity"] = bool(np.array_equal(x, y))
                row["pair_sha256"] = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
                for key, fn in [("check_input", lambda: bool(gi.check_input(x.copy()))),
                                ("transform_match", lambda: bool(np.array_equal(tr.transform(x.copy()), y)))]:
                    try:
                        row[key] = timed(2, fn)
                    except Exception as exc:
                        row[key + "_error"] = f"{type(exc).__name__}: {exc}"[:350]
                if target and max(*x.shape, *y.shape) >= target:
                    sample_mode = "native_sampler" if adapt else "native"
                    write_json(out / "samples" / sample_mode / task_id / f"{target}-{seed}.json", pair)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:350]
            group["samples"].append(row)
            if "error" in row:
                break
        rows.append(group)
    return {"task_id": task_id, "mode": "native_sampler" if adapt else "native", "size_args": sorted(args), "groups": rows,
            "meaning": "construction vs seed transform agreement is not independent rule correctness"}


def run_native(out, sizes, samples, workers, selected, adapt=False):
    inv = json.loads((out / "inventory.json").read_text())
    task_ids = [r["task_id"] for r in inv["tasks"] if r["native"] is not None
                and (not selected or r["task_id"] in selected)]
    target = out / ("native_sampler" if adapt else "native")
    target.mkdir(exist_ok=True)
    def one(tid):
        path = target / f"{tid}.json"
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "child", "--task", tid,
               "--out", str(out), "--sizes", *map(str, sizes), "--samples", str(samples)]
        if adapt:
            cmd.append("--adapt-sampler")
        started = time.monotonic()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=45,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
            if p.returncode:
                result = {"task_id": tid, "process_error": p.stderr[-1600:]}
            else:
                result = json.loads(p.stdout)
        except subprocess.TimeoutExpired:
            result = {"task_id": tid, "process_error": "subprocess timeout 45s"}
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["tool_sha256"] = digest(__file__)
        write_json(path, result)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(one, tid) for tid in task_ids]
        for i, future in enumerate(as_completed(pending), 1):
            result = future.result()
            if i % 50 == 0 or i == len(pending):
                print(f"native completed {i}/{len(pending)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["scan", "native", "child"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tag", default="20261021-large-grid-opportunities")
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--task", nargs="*")
    ap.add_argument("--adapt-sampler", action="store_true")
    args = ap.parse_args()
    out = args.out or paths.run_dir("seed_bank", args.tag, create=True)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "scan":
        inventory(out)
    elif args.stage == "native":
        run_native(out, args.sizes, args.samples, args.workers, args.task, args.adapt_sampler)
    else:
        assert len(args.task) == 1
        incidental = io.StringIO()
        with contextlib.redirect_stdout(incidental):
            result = native_probe(args.task[0], out, args.sizes, args.samples, args.adapt_sampler)
        result["incidental_stdout"] = incidental.getvalue()[:2000]
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
