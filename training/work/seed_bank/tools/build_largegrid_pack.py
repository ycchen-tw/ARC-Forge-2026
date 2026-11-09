#!/usr/bin/env python3
"""ARC-AGI-2 large-grid variant task pack (for the RL training pool, CPU only).

Source: the training families in ``results/size30_64_20261022/opportunities.csv`` (official ARC-AGI-2 training, **no eval parents**;
the 4 reviewed_eval tasks from eval are never included, so the held-out stays clean). Each family redraws pairs with the profile it passed at the 30 tier (native = ARC-GEN's native
size parameters; native_sampler = AST-rewritten size assignments, ``probe_large_grids.adapt_sampler``), assembled into one task
``{"train": [...], "test": [{"input", "output"}]}``:

  * train pair count = the official original task's train pair count, 1 test pair;
  * every pair must: be generated successfully by ARC-GEN, replay cell-identical with the same seed, have the output of ``seeds/training/<id>/transform.py`` (the solution written independently in this repo)
    equal ARC-GEN's output, be non-identity, have input and output sides within 1..30 (the arc2 harness limit), have input area larger than the family's
    native default sample (otherwise it is not a "large grid"), and have an input not repeated within the task;
  * sampling seed = ``seed_base + 1000 * variant + i`` (global ``random.seed``, same as probe_large_grids), i counting up from 0;
    failing seeds are skipped with the reason recorded, at most ``--max-attempts``; if it cannot be filled the task is not emitted (recorded in the manifest, not silent).

``transform_match`` means "two independent constructions agree", not proof of rule correctness (the meaning column of probe_large_grids); ARC-GEN itself
can also be wrong (README: 34cfa167, 23b5c85d), so families with quality caution/avoid are excluded.
    python work/seed_bank/tools/build_largegrid_pack.py build --out $ARC_DATA/arc2_pools/largegrid30-20261022

Outputs: ``tasks/<family>-lg30v<k>.json``, ``train.txt`` (task IDs), ``manifest.jsonl`` (one row per family: profile, per-pair seed / shape /
sha256, rejected seeds and reasons), ``receipt.json`` (ARC-GEN head, sha256 of each source, parameters, counts). A rerun with the same parameters is bit-identical.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import io
import json
import os
import random
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_large_grids as plg  # noqa: E402  (BANK, load, timed, adapt_sampler, grid_info, AllocationLimit)

paths = plg.paths
BANK = plg.BANK
OPPORTUNITIES = BANK / "results/size30_64_20261022/opportunities.csv"
#: arcgen/transform sha256 follow the 64-tier file (compare_size30_64 checks the same way when reusing 64)
SOURCE_SHAS = BANK / "results/large_grid_20261021/opportunities.csv"
SIZE = 30
MAX_SIDE = 30             # work/arc_code/arc2/check.py valid_grid, arc2_render
QUALITIES = ("ok", "note")
CHILD_TIMEOUT = 120


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(size: int) -> list[dict]:
    """Training families that pass at the 30 tier, with input area larger than the native default and quality ok/note, and the profile each should use (native first)."""
    shas = {r["task_id"]: r for r in csv.DictReader(SOURCE_SHAS.open()) if r["split"] == "training"}
    out = []
    for r in csv.DictReader(OPPORTUNITIES.open()):
        if r["split"] != "training" or r["quality"] not in QUALITIES:
            continue
        profiles = [p for p in json.loads(r["eligible_modes_and_shapes"])
                    if p["size"] == size and p["input_area_increased_first_seed_cap64"]
                    and p["mode"] in ("native", "native_sampler")]
        if not profiles:
            continue
        s = shas[r["task_id"]]
        # native (ARC-GEN's own size parameters) first; if it cannot fill (e.g. the input is two 30-wide blocks forming 30×61), try native_sampler
        out.append({"family": r["task_id"], "quality": r["quality"], "size": size,
                    "profiles": [{"mode": p["mode"], "baseline_shape": p["baseline_shape"]}
                                 for p in sorted(profiles, key=lambda p: p["mode"] != "native")],
                    "arcgen_sha256": s["arcgen_sha256"], "transform_sha256": s["transform_sha256"],
                    "seed_sha256": s["seed_sha256"]})
    return out


def child(spec: dict) -> dict:
    """One family, one variant: load the ARC-GEN generator and the seed's transform in this process and draw until filled or out of attempts."""
    import numpy as np
    fam = spec["family"]
    root = paths.data() / "ARC-GEN"
    gen_path = root / "tasks" / f"task_{fam}.py"
    seed_dir = BANK / "seeds/training" / fam
    for path, want in ((gen_path, spec["arcgen_sha256"]), (seed_dir / "transform.py", spec["transform_sha256"]),
                       (seed_dir / "seed.json", spec["seed_sha256"])):
        if digest(path) != want:
            raise RuntimeError(f"{path}: sha256 {digest(path)} != {want} (source changed since the size audit)")
    sys.path.insert(0, str(root))
    common = plg.load(root / "common.py", "common")
    original_grid = common.grid

    def bounded_grid(width, height, *args, **kwargs):
        if width > 512 or height > 512 or width * height > 262144:
            raise plg.AllocationLimit(f"requested {height}x{width}")
        return original_grid(width, height, *args, **kwargs)
    common.grid = bounded_grid
    native = plg.load(gen_path, "native_generator")
    sys.path.insert(0, str(seed_dir))
    tr = plg.load(seed_dir / "transform.py", "transform")
    meta = json.loads((seed_dir / "seed.json").read_text())
    official = json.loads((paths.data() / "ARC-AGI-2/data/training" / f"{fam}.json").read_text())
    if meta.get("context_required"):
        plg.timed(2, lambda: tr.fit(official["train"]))
    size_args = sorted(set(inspect.signature(native.generate).parameters) & plg.DIM_ARGS)
    if spec["mode"] == "native":
        if not size_args:
            raise RuntimeError(f"{fam}: profile native but generate() has no size argument")
        active, kw, edits = native, {k: spec["size"] for k in size_args}, []
    else:
        tree, edits = plg.adapt_sampler(gen_path.read_text(), spec["size"])
        active = types.ModuleType("adapted_native_generator")
        exec(compile(tree, "<in-memory-native-sampler>", "exec"), active.__dict__)
        kw = {}
    need = max(2, len(official["train"])) + 1
    base_area = spec["baseline_shape"][0] * spec["baseline_shape"][1]
    pairs, rejects, seen = [], [], set()
    for i in range(spec["max_attempts"]):
        if len(pairs) == need:
            break
        seed = spec["seed_base"] + 1000 * spec["variant"] + i
        reason = None
        try:
            random.seed(seed)
            pair = plg.timed(1.5, lambda: active.generate(**kw))
            x, y = np.asarray(pair["input"]), np.asarray(pair["output"])
            plg.grid_info(x), plg.grid_info(y)
            random.seed(seed)
            again = plg.timed(1.5, lambda: active.generate(**kw))
            key = json.dumps(pair["input"])
            if not (np.array_equal(x, again["input"]) and np.array_equal(y, again["output"])):
                reason = "rng_replay"
            elif max(*x.shape, *y.shape) > MAX_SIDE:
                reason = f"side>{MAX_SIDE} {list(x.shape)}->{list(y.shape)}"
            elif x.size <= base_area:
                reason = f"input area {x.size} <= native default {base_area}"
            elif np.array_equal(x, y):
                reason = "identity"
            elif key in seen:
                reason = "duplicate input"
            elif not np.array_equal(np.asarray(plg.timed(2, lambda: tr.transform(x.copy()))), y):
                reason = "transform mismatch"
        except Exception as exc:                       # an exception from generator/transform means this seed fails, not a harness error
            reason = f"{type(exc).__name__}: {exc}"[:200]
        if reason is not None:
            rejects.append({"seed": seed, "reason": reason})
            continue
        seen.add(key)
        pairs.append({"seed": seed, "input": x.tolist(), "output": y.tolist()})
    return {"family": fam, "variant": spec["variant"], "mode": spec["mode"], "size_args": size_args,
            "sampler_edits": edits, "need": need, "pairs": pairs, "rejects": rejects,
            "complete": len(pairs) == need}


def build(args: argparse.Namespace) -> None:
    out: Path = args.out
    if out.exists():
        raise SystemExit(f"{out} exists; a pack is written once")
    arcgen = paths.data() / "ARC-GEN"
    head = subprocess.check_output(["git", "-C", str(arcgen), "rev-parse", "HEAD"], text=True).strip()
    if head != args.arcgen_head:
        raise SystemExit(f"ARC-GEN at {arcgen} is {head}, the size audit used {args.arcgen_head}")
    families = select(SIZE)
    if args.limit:
        families = families[:args.limit]
    specs = [{**f, "variant": v, "seed_base": args.seed_base, "max_attempts": args.max_attempts}
             for f in families for v in range(args.variants)]

    def run(spec: dict) -> dict:
        tried = []
        for profile in spec["profiles"]:
            one = {k: v for k, v in spec.items() if k != "profiles"} | profile
            p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "child", "--spec", json.dumps(one)],
                               capture_output=True, text=True, timeout=CHILD_TIMEOUT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
            if p.returncode != 0:
                raise RuntimeError(f"{spec['family']} v{spec['variant']} {profile['mode']}: child exited "
                                   f"{p.returncode}: {p.stderr[-1500:]}")
            res = json.loads(p.stdout) | {"baseline_shape": profile["baseline_shape"]}
            tried.append({"mode": res["mode"], "complete": res["complete"], "rejects": len(res["rejects"])})
            if res["complete"]:
                break
        return res | {"tried": tried}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, specs))
    tmp = out.with_name(out.name + ".partial")
    (tmp / "tasks").mkdir(parents=True)
    ids, rows = [], []
    for spec, res in zip(specs, results, strict=True):
        tid = f"{res['family']}-lg{SIZE}v{res['variant']}"
        row = {"id": tid if res["complete"] else None, "family": res["family"], "variant": res["variant"],
               "quality": spec["quality"], "mode": res["mode"], "size_args": res["size_args"],
               "sampler_edits": res["sampler_edits"], "baseline_shape": res["baseline_shape"], "tried": res["tried"],
               "need_pairs": res["need"], "complete": res["complete"], "rejects": res["rejects"],
               "pairs": [{"seed": p["seed"], "role": "test" if k == res["need"] - 1 else "train",
                          "input_shape": [len(p["input"]), len(p["input"][0])],
                          "output_shape": [len(p["output"]), len(p["output"][0])],
                          "sha256": hashlib.sha256(json.dumps({"input": p["input"], "output": p["output"]},
                                                              sort_keys=True).encode()).hexdigest()}
                         for k, p in enumerate(res["pairs"])]}
        rows.append(row)
        if res["complete"]:
            pairs = [{"input": p["input"], "output": p["output"]} for p in res["pairs"]]
            (tmp / "tasks" / f"{tid}.json").write_text(json.dumps({"train": pairs[:-1], "test": pairs[-1:]}) + "\n")
            ids.append(tid)
    (tmp / "train.txt").write_text("".join(f"{t}\n" for t in ids))
    (tmp / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {
        "what": "ARC-AGI-2 training families regenerated at size 30 (ARC-GEN), one task per family x variant",
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True).strip(),
        "tool_sha256": digest(Path(__file__)), "probe_tool_sha256": digest(HERE / "probe_large_grids.py"),
        "opportunities_sha256": digest(OPPORTUNITIES), "arcgen_head": head, "arcgen_license": "Apache-2.0 "
        "(work/seed_bank/licenses/ARC-GEN-APACHE-2.0.txt)",
        "params": {"size": SIZE, "max_side": MAX_SIDE, "qualities": list(QUALITIES), "variants": args.variants,
                   "seed_base": args.seed_base, "max_attempts": args.max_attempts, "limit": args.limit},
        "families_selected": len(families), "tasks_written": len(ids),
        "incomplete": [f"{r['family']} v{r['variant']}" for r in rows if not r["complete"]],
        "modes": {m: sum(r["mode"] == m and r["complete"] for r in rows) for m in ("native", "native_sampler")},
        "held_out_note": "no ARC-AGI-2 evaluation parent: the 4 reviewed_eval families are excluded, ARC-GEN has no eval "
                         "generators; the public eval 120 stays a clean held-out for anything trained on this pack",
    }
    (tmp / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    tmp.rename(out)
    print(json.dumps({k: receipt[k] for k in ("families_selected", "tasks_written", "modes")}
                     | {"incomplete": len(receipt["incomplete"])}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--variants", type=int, default=1)
    b.add_argument("--seed-base", type=int, default=20261101)
    b.add_argument("--max-attempts", type=int, default=60)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--limit", type=int, help="first N families only (smoke)")
    b.add_argument("--arcgen-head", default="a15cbdb44c776610aeeb9f487a06af875d3d0878")
    c = sub.add_parser("child")
    c.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.stage == "build":
        build(args)
    else:
        with contextlib.redirect_stdout(io.StringIO()):     # some generators print
            result = child(json.loads(args.spec))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
