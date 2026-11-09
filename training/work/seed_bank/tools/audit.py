#!/usr/bin/env python3
"""Replay every official pair and stress generators without changing seed files.

Default: 200 inputs/family, RNG 0, 1, 17, 20261003, each replayed with the same
supplied RNG. Reports bind results to source, validator, taxonomy and task hashes.
PASS establishes executable consistency, not semantic completeness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "work/seed_bank"
sys.path.insert(0, str(ROOT / "src"))
from arcforge import paths

DEFAULT_RNG_SEEDS = [0, 1, 17, 20261003]
REQUIRED_FILES = ("seed.json", "transform.py", "generate_input.py")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = load(BANK / "verify.py", "seed_bank_validator")
valid = validator.valid_grid


def key(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def structural_key(grid):
    """D4 / colour-index normalization is only a novelty description, not evidence of semantic completeness."""
    variants = []
    for k in range(4):
        a = np.rot90(grid, k)
        for b in (a, np.fliplr(a)):
            labels = {}
            encoded = bytes(labels.setdefault(int(v), len(labels)) for v in b.flat)
            variants.append((b.shape, encoded))
    return min(variants)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(directory):
    # Helpers in the same directory are program sources too; after a helper changes, an old PASS cannot be reused.
    names = set(REQUIRED_FILES)
    names.update(p.relative_to(directory).as_posix() for p in directory.rglob("*.py")
                 if "__pycache__" not in p.parts)
    if (directory / "task.json").exists():
        names.add("task.json")
    return {name: sha256(directory / name) for name in sorted(names) if (directory / name).is_file()}


def validator_hashes():
    return {"audit.py": sha256(Path(__file__)), "verify.py": sha256(Path(validator.__file__)),
            "taxonomy.yaml": sha256(validator.TAXONOMY)}


def seed_directory(kind, bank=None):
    if kind not in ("evaluation", "training", "synth"):
        raise ValueError(f"unsupported kind: {kind!r}")
    bank = BANK if bank is None else Path(bank)
    return bank / "synth" if kind == "synth" else bank / "seeds" / kind


def audit_one(task_id, samples=50, rng_seeds=None, kind="evaluation", *, seed_dir=None, dataset_root=None):
    rng_seeds = list(DEFAULT_RNG_SEEDS if rng_seeds is None else rng_seeds)
    if samples < 1 or not rng_seeds or len(set(rng_seeds)) != len(rng_seeds):
        raise ValueError("samples must be positive and RNG seeds nonempty and unique")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ValueError("task_id must be a single safe directory name")
    directory = seed_directory(kind) / task_id if seed_dir is None else Path(seed_dir)
    before = hashes(directory)
    result = {
        "task_id": task_id, "kind": kind, "status": "FAIL", "source_sha256": before,
        "validator_sha256": validator_hashes(), "rng_seeds": rng_seeds,
        "samples_per_rng": samples, "generated_checked": 0, "generator_rng_replays": 0,
        "official_pairs_passed": 0, "official_pairs_total": 0, "check_input_on_originals": True,
        "generated_valid": 0, "generated_check_input": 0,
        "generated_transform_runs": 0, "generated_output_differs": 0,
        "semantic_completeness": "not established by executable checks", "errors": [],
    }
    errors = result["errors"]
    task_path = None
    old_path = list(sys.path)
    old_modules = {name: sys.modules.get(name) for name in ("transform", "generate_input")}
    try:
        missing = [name for name in REQUIRED_FILES if name not in before]
        if missing:
            raise ValueError(f"missing required files: {', '.join(missing)}")
        metadata = json.loads((directory / "seed.json").read_text(encoding="utf-8"))
        errors.extend(validator.validate_schema(metadata, expected_split=kind, expected_task_id=task_id))
        if errors:
            return result
        if kind == "synth":
            task_path = directory / "task.json"
        else:
            dataset_root = paths.data() / "ARC-AGI-2" if dataset_root is None else Path(dataset_root)
            task_path = dataset_root / "data" / kind / f"{task_id}.json"
        task_bytes = task_path.read_bytes()
        result["official_task_sha256"] = hashlib.sha256(task_bytes).hexdigest()
        task = json.loads(task_bytes)
        errors.extend(validator.validate_schema(metadata, task, kind, task_id))
        if errors:
            return result
        result["official_pairs_total"] = sum(len(task[k]) for k in ("train", "test"))
        sys.path.insert(0, str(directory.resolve()))
        module = load(directory / "transform.py", "transform")
        if metadata.get("context_required", False):
            module.fit(copy.deepcopy(task["train"]))
        transform = module.transform
        generator = load(directory / "generate_input.py", "generate_input")
        finite_keys = None
        if "finite_domain" in metadata:
            cardinality = metadata["finite_domain"]["cardinality"]
            declared_identities = metadata["finite_domain"].get("identity_members", 0)
            official_identity = any(pair["input"] == pair["output"]
                                    for split in ("train", "test") for pair in task[split])
            if declared_identities and not official_identity:
                raise ValueError("finite identity members require a documented official identity pair")
            enumerated = [np.asarray(x) for x in itertools.islice(generator.enumerate_inputs(), cardinality + 1)]
            replay = [np.asarray(x) for x in itertools.islice(generator.enumerate_inputs(), cardinality + 1)]
            if len(enumerated) != cardinality or len(replay) != cardinality:
                raise ValueError("finite enumeration count differs from declared cardinality")
            finite_keys = set()
            identity_members = 0
            for x, repeated in zip(enumerated, replay):
                if not valid(x) or not valid(repeated) or not np.array_equal(x, repeated):
                    raise ValueError("finite enumeration is illegal or nondeterministic")
                if not generator.check_input(x.copy()):
                    raise ValueError("check_input rejected finite-domain member")
                a, b = np.asarray(transform(x.copy())), np.asarray(transform(x.copy()))
                if not valid(a) or not np.array_equal(a, b):
                    raise ValueError("finite-domain transform is illegal or nondeterministic")
                identity_members += int(np.array_equal(a, x))
                finite_keys.add(key(x))
            if len(finite_keys) != cardinality:
                raise ValueError("finite enumeration contains duplicate inputs")
            if identity_members != declared_identities:
                raise ValueError("finite enumeration identity count differs from declaration")
            result["finite_domain"] = {"cardinality": cardinality, "exhaustive_checked": cardinality,
                                       "enumeration_replays": cardinality, "official_inputs_in_domain": True,
                                       "generated_inputs_in_domain": True, "identity_members": identity_members,
                                       "identity_members_supported_by_official": not identity_members or official_identity}
        original, unique, structures, dimensions, output_colors = set(), set(), set(), set(), set()
        uniqueness_by_rng = {}
        for split in ("train", "test"):
            for i, pair in enumerate(task[split]):
                x = np.asarray(pair["input"])
                original.add(structural_key(x))
                try:
                    if finite_keys is not None and key(x) not in finite_keys:
                        result["finite_domain"]["official_inputs_in_domain"] = False
                        raise ValueError("official input is outside declared finite domain")
                    try:
                        accepted = bool(generator.check_input(x.copy()))
                    except Exception:
                        result["check_input_on_originals"] = False
                        raise
                    if not accepted:
                        result["check_input_on_originals"] = False
                        raise ValueError("check_input rejected original")
                    a = np.asarray(transform(x.copy()))
                    b = np.asarray(transform(x.copy()))
                    if not valid(a):
                        raise ValueError("illegal output")
                    if not np.array_equal(a, b):
                        raise ValueError("nondeterministic transform")
                    if not np.array_equal(a, np.asarray(pair["output"])):
                        raise ValueError("wrong output")
                    result["official_pairs_passed"] += 1
                except Exception as exc:
                    errors.append(f"{split} {i}: {type(exc).__name__}: {exc}")
        for seed in rng_seeds:
            left, right = np.random.default_rng(seed), np.random.default_rng(seed)
            first_thirty = set()
            for i in range(samples):
                try:
                    x = np.asarray(generator.generate_input(left))
                    repeat = np.asarray(generator.generate_input(right))
                    if not valid(x) or not valid(repeat):
                        raise ValueError("illegal generated input")
                    result["generated_valid"] += 1
                    if finite_keys is not None and (key(x) not in finite_keys or key(repeat) not in finite_keys):
                        result["finite_domain"]["generated_inputs_in_domain"] = False
                        raise ValueError("generated input is outside declared finite domain")
                    if not np.array_equal(x, repeat):
                        raise ValueError("generator does not reproduce supplied RNG")
                    result["generator_rng_replays"] += 1
                    if not generator.check_input(x.copy()):
                        raise ValueError("check_input rejected generated input")
                    result["generated_check_input"] += 1
                    a = np.asarray(transform(x.copy()))
                    b = np.asarray(transform(x.copy()))
                    if not valid(a):
                        raise ValueError("illegal generated output")
                    result["generated_transform_runs"] += 1
                    if not np.array_equal(a, b):
                        raise ValueError("nondeterministic generated output")
                    if np.array_equal(a, x):
                        raise ValueError("identity output")
                    result["generated_output_differs"] += 1
                    unique.add(key(x))
                    structures.add(structural_key(x))
                    dimensions.add(x.shape)
                    output_colors.add(len(np.unique(a)))
                    if i < 30:
                        first_thirty.add(key(x))
                    result["generated_checked"] += 1
                except Exception as exc:
                    errors.append(f"rng={seed} index={i}: {type(exc).__name__}: {exc}")
                    break
            uniqueness_by_rng[str(seed)] = len(first_thirty)
            if finite_keys is None and len(first_thirty) < min(28, samples):
                errors.append(f"rng={seed}: only {len(first_thirty)} unique inputs in first {min(30, samples)}")
        result.update({
            "unique_inputs": len(unique), "unique_first_30_by_rng": uniqueness_by_rng,
            "unique_structures_mod_d4_color": len(structures),
            "novel_structures_vs_official": len(structures - original),
            "input_dimensions": sorted(dimensions), "output_color_counts": sorted(output_colors),
        })
    except Exception as exc:
        errors.append(f"setup: {type(exc).__name__}: {exc}")
    finally:
        sys.path[:] = old_path
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if hashes(directory) != before:
            errors.append("source changed during audit")
        if task_path is not None and "official_task_sha256" in result and (
                not task_path.is_file() or sha256(task_path) != result["official_task_sha256"]):
            errors.append("task changed during audit")
        if kind == "synth":
            for old, new in (("official_task_sha256", "task_sha256"),
                             ("official_pairs_passed", "task_pairs_passed"),
                             ("official_pairs_total", "task_pairs_total")):
                if old in result:
                    result[new] = result.pop(old)
        result["status"] = "FAIL" if errors else "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="*")
    parser.add_argument("--kind", choices=("evaluation", "training", "synth"), default="evaluation")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--rng-seeds", default=",".join(map(str, DEFAULT_RNG_SEEDS)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        seeds = [int(s) for s in args.rng_seeds.split(",")]
    except ValueError:
        parser.error("RNG seeds must be comma-separated integers")
    if args.samples < 1 or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("samples must be positive and RNG seeds nonnegative and unique")
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    if args.child:
        if len(args.task_ids) != 1:
            parser.error("child mode requires exactly one task id")
        print(json.dumps(audit_one(args.task_ids[0], args.samples, seeds, args.kind), ensure_ascii=False))
        return
    if args.out is None:
        parser.error("--out is required")
    directory = seed_directory(args.kind)
    if args.kind == "training":
        inventory = load(Path(__file__).with_name("training_inventory.py"), "training_inventory")
        official = inventory.official_tasks(paths.data() / "ARC-AGI-2")
        ids = args.task_ids or sorted(official)
        if set(ids) - set(official):
            parser.error("training task IDs must belong to the pinned official training split")
    else:
        ids = args.task_ids or sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not ids or len(set(ids)) != len(ids):
        parser.error("task IDs must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", tid) for tid in ids):
        parser.error("task IDs must be safe directory names")
    args.out.mkdir(parents=True, exist_ok=True)

    def run(tid):
        started = time.monotonic()
        cmd = [sys.executable, str(Path(__file__).resolve()), tid, "--child",
               "--samples", str(args.samples), "--rng-seeds", args.rng_seeds, "--kind", args.kind]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            if process.returncode:
                row = {"task_id": tid, "kind": args.kind, "status": "ERROR", "errors": [process.stderr[-6000:]]}
            else:
                row = json.loads(process.stdout)
        except subprocess.TimeoutExpired:
            row = {"task_id": tid, "kind": args.kind, "status": "TIMEOUT", "errors": [f"exceeded {args.timeout}s"]}
        except ValueError as exc:
            row = {"task_id": tid, "kind": args.kind, "status": "ERROR", "errors": [f"invalid child report: {exc}"]}
        row["wall_seconds"] = round(time.monotonic() - started, 3)
        (args.out / f"{tid}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(tid, row["status"], row.get("generated_checked", 0), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, ids))
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "kind": args.kind,
               "validator_sha256": validator_hashes(), "audit_sha256": sha256(Path(__file__)),
               "task_count": len(rows), "passed": sum(r["status"] == "PASS" for r in rows),
               "generated_checked": sum(r.get("generated_checked", 0) for r in rows), "tasks": rows}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary["passed"] != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
