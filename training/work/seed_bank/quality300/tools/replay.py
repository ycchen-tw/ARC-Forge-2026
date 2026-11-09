#!/usr/bin/env python3
"""Portable mechanical replay. No model calls or new blind evaluation."""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = "55a85ef8a3e782419ac6d2c0c1c1b24474b194ae"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(value):
    assert isinstance(value, list) and 1 <= len(value) <= 30, "invalid grid height"
    assert isinstance(value[0], list) and 1 <= len(value[0]) <= 30, "invalid grid width"
    assert all(isinstance(row, list) and len(row) == len(value[0]) for row in value), "ragged grid"
    assert all(type(cell) is int and 0 <= cell <= 9 for row in value for cell in row), "invalid grid cell"


def closure(package):
    """Discover declared flat local imports, requiring all others to be stdlib.

    Dynamic imports and external runtime data are unsupported. This static check
    is a portability check, not a security sandbox.
    """
    files = {p.stem: p for p in package.glob("*.py")}
    pending = ["transform", "generate_input"]
    reached = set()
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        assert module in files, "missing runtime module: " + module
        reached.add(module)
        tree = ast.parse(files[module].read_bytes(), filename=files[module].name)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative imports are not supported by flat task packages"
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name in files:
                    pending.append(name)
                else:
                    assert name in sys.stdlib_module_names, "non-stdlib or missing local import: " + name
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                assert name not in {"__import__", "import_module", "open", "read_text", "read_bytes", "write_text", "write_bytes"}, "unsupported runtime import/file I/O: " + name
    return sorted(name + ".py" for name in reached)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def one(package, samples):
    runtime = closure(package)
    record = json.loads((package / "acceptance.json").read_text())
    expected = record["runtime_sha256"]
    assert set(runtime) == set(expected), "runtime import closure changed"
    for name in runtime:
        assert sha(package / name) == expected[name], "runtime bytes changed: " + name
    assert sha(package / "task.json") == record["task_sha256"], "task bytes changed"
    seed = json.loads((package / "seed.json").read_text())
    parents = seed["provenance"]["parents"]
    assert len({p["task_id"] for p in parents}) >= 2
    assert any(p["split"] == "evaluation" for p in parents)
    assert all(p["source_commit"] == PIN for p in parents)
    assert record["status"] == "accepted" and all(record["gates"].values())
    sys.path.insert(0, str(package))
    transform = load(package / "transform.py", "transform").transform
    generator = load(package / "generate_input.py", "generate_input")
    task = json.loads((package / "task.json").read_text())
    assert 3 <= len(task["train"]) <= 5 and len(task["test"]) == 2
    seen = set()
    shown = 0
    for split in ("train", "test"):
        for pair in task[split]:
            grid(pair["input"])
            grid(pair["output"])
            frozen = copy.deepcopy(pair["input"])
            output = transform(pair["input"])
            grid(output)
            assert output == pair["output"], "shown-output mismatch"
            assert pair["input"] == frozen, "shown input mutated"
            fingerprint = json.dumps(frozen, separators=(",", ":"))
            assert fingerprint not in seen, "duplicate shown input"
            seen.add(fingerprint)
            shown += 1
    oracle = 0
    for index in range(samples):
        # Reuse the production mechanical-check seed sequence. These are not a
        # fresh random pool, and neither author code nor this replay is blind.
        number = 710003 + index * 7919
        inp = generator.generate_input(number)
        grid(inp)
        frozen = copy.deepcopy(inp)
        again = generator.generate_input(number)
        grid(again)
        assert inp == again, "nondeterministic generation"
        output = transform(inp)
        grid(output)
        assert inp == frozen, "generated input mutated"
        assert transform(copy.deepcopy(frozen)) == output, "nondeterministic transform"
        if hasattr(generator, "generate_case"):
            case = generator.generate_case(number)
            ci, co = (case["input"], case["output"]) if isinstance(case, dict) else case
            grid(ci)
            grid(co)
            assert ci == frozen and co == output, "latent oracle mismatch"
            oracle += 1
    return {"task_id": package.name, "status": "PASS", "shown_pairs_replayed": shown,
            "deterministic_generator_cases": samples, "latent_oracle_matches": oracle,
            "runtime_files": runtime, "nonmutation": "PASS", "import_closure": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--samples", type=int, default=20, help="fixed mechanical seeds per task, not blind trials")
    parser.add_argument("--output", type=Path, help="optional JSON report; default does not write to the source tree")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 1 <= args.samples <= 100, "choose 1..100 existing mechanical seed positions"
    root = args.root.resolve()
    if args.one:
        try:
            result = one(root / "packages" / args.one, args.samples)
        except Exception as exc:
            print(json.dumps({"task_id": args.one, "status": "FAIL", "error": str(exc)}))
            raise SystemExit(1)
        print(json.dumps(result))
        return
    summary = json.loads((root / "reports" / "checkpoint_summary.json").read_text())
    ids = sorted(p.name for p in (root / "packages").iterdir() if p.is_dir())
    assert len(ids) == summary["accepted_concepts"]
    if summary["publication_status"] == "FINAL_SOURCE_PACKAGE":
        assert len(ids) == 300
    else:
        assert (root / "NOT_FOR_PUBLICATION.txt").exists()
    for rel, expected in json.loads((root / "FILE_SHA256.json").read_text()).items():
        assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
        assert sha(root / rel) == expected, "integrity mismatch: " + rel
    records = []
    for task_id in ids:
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(root),
                               "--one", task_id, "--samples", str(args.samples)],
                              text=True, capture_output=True, timeout=180)
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            raise AssertionError("task subprocess failed: " + task_id + " " + proc.stderr[-500:])
        assert proc.returncode == 0 and result["status"] == "PASS", result
        records.append(result)
    report = {"status": "PASS", "scope": "Mechanical replay only; no model call, fresh blind context, or new random blind pool",
              "tasks": len(records), "shown_pairs_replayed": sum(r["shown_pairs_replayed"] for r in records),
              "deterministic_generator_cases": sum(r["deterministic_generator_cases"] for r in records),
              "latent_oracle_matches": sum(r["latent_oracle_matches"] for r in records), "records": records}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
