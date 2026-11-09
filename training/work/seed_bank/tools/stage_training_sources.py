#!/usr/bin/env python3
"""After pinning the sources, build candidate programs awaiting manual extraction; this tool does not produce seeds or claim them complete.

Candidates keep the source licenses. ARC-GEN's per-task validate() is not copied; all randomness is taken instead from
the NumPy Generator passed in by the caller, and per task only the common functions actually used are kept.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess


RNG_ADAPTERS = '''
def _rng_choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]

def _rng_choices(rng, sequence, k=1):
    return [_rng_choice(rng, sequence) for _ in range(k)]

def _rng_randint(rng, start, stop):
    return int(rng.integers(start, stop + 1))

def _rng_sample(rng, sequence, k):
    return [sequence[int(i)] for i in rng.choice(len(sequence), k, replace=False)]

def _rng_random(rng):
    return float(rng.random())
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def source_notice(path):
    lines = path.read_text().splitlines()
    header = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            break
    return "\n".join(header).rstrip() + "\n"


class CommonAdapter(ast.NodeTransformer):
    def __init__(self, functions):
        self.functions = functions

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            node.func.id = "_arcgen_" + node.func.id
            node.keywords.append(ast.keyword(arg="_rng", value=ast.Name(id="_rng", ctx=ast.Load())))
        elif (isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "random"):
            if node.func.attr not in {"choice", "choices", "randint", "sample", "random"}:
                raise ValueError(f"unsupported upstream random function: {node.func.attr}")
            node.func = ast.Name(id="_rng_" + node.func.attr, ctx=ast.Load())
            node.args.insert(0, ast.Name(id="_rng", ctx=ast.Load()))
        return node

    def visit_Name(self, node):
        if node.id == "internal_colors" and isinstance(node.ctx, ast.Load):
            return ast.Tuple(elts=[ast.Constant(i) for i in range(10)], ctx=ast.Load())
        return node


class TaskAdapter(ast.NodeTransformer):
    def __init__(self, local_functions):
        self.local_functions = local_functions

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "common"):
            node.func = ast.Name(id="_arcgen_" + node.func.attr, ctx=ast.Load())
            node.keywords.append(ast.keyword(arg="_rng", value=ast.Name(id="_rng", ctx=ast.Load())))
        elif isinstance(node.func, ast.Name) and node.func.id in self.local_functions:
            node.keywords.append(ast.keyword(arg="_rng", value=ast.Name(id="_rng", ctx=ast.Load())))
        return node


class AtlasAdapter(ast.NodeTransformer):
    """Keep an upstream helper named transform, so the public interface does not overwrite it."""
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "transform":
            node.name = "_atlas_transform"
        return node

    def visit_Name(self, node):
        if node.id == "transform":
            node.id = "_atlas_transform"
        return node


def adapt_solver(content):
    tree = ast.parse(content)
    if any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        return ast.unparse(ast.fix_missing_locations(AtlasAdapter().visit(tree))) + "\n"
    return content


def add_rng_argument(function):
    if any(arg.arg == "_rng" for arg in function.args.args + function.args.kwonlyargs):
        raise ValueError("source already uses the reserved parameter _rng")
    function.args.kwonlyargs.append(ast.arg(arg="_rng"))
    function.args.kw_defaults.append(None)
    return function


def adapt_generator(task_path, common_path):
    common_tree = ast.parse(common_path.read_text())
    functions = {n.name: n for n in common_tree.body if isinstance(n, ast.FunctionDef)}
    task_tree = ast.parse(task_path.read_text())
    local = {n.name for n in task_tree.body
             if isinstance(n, ast.FunctionDef) and n.name != "validate"}
    if "generate" not in local:
        raise ValueError("source lacks generate")
    selected = [n for n in task_tree.body
                if not (isinstance(n, ast.FunctionDef) and n.name == "validate")]
    needed = {n.attr for item in selected for n in ast.walk(item)
              if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
              and n.value.id == "common"}
    pending = list(needed)
    while pending:
        name = pending.pop()
        if name not in functions:
            raise ValueError(f"unknown common dependency: {name}")
        for n in ast.walk(functions[name]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions:
                if n.func.id not in needed:
                    needed.add(n.func.id)
                    pending.append(n.func.id)
    if "set_colors" in needed:
        raise ValueError("the shared colour palette must not be modified")
    output = ast.parse("import copy\nimport math\nimport numpy as np\n" + RNG_ADAPTERS).body
    adapter = CommonAdapter(set(functions))
    for name in sorted(needed):
        function = adapter.visit(copy.deepcopy(functions[name]))
        function.name = "_arcgen_" + name
        output.append(add_rng_argument(function))
    adapter = TaskAdapter(local)
    for item in selected:
        if isinstance(item, ast.Import):
            remaining = [n for n in item.names if n.name not in {"common", "os", "sys"}]
            if remaining:
                raise ValueError(f"unreviewed import: {ast.unparse(item)}")
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        if isinstance(item, ast.FunctionDef):
            output.append(add_rng_argument(adapter.visit(copy.deepcopy(item))))
        elif isinstance(item, ast.ImportFrom) and item.module == "typing":
            output.append(copy.deepcopy(item))
        else:
            raise ValueError(f"unreviewed module-level syntax: {type(item).__name__}")
    output.extend(ast.parse('''
def generate_pair(rng, **parameters):
    return generate(_rng=rng, **parameters)

def generate_input(rng, size_hint=None):
    return np.asarray(generate_pair(rng)["input"], dtype=int)
''').body)
    module = ast.fix_missing_locations(ast.Module(body=output, type_ignores=[]))
    # Text written into the generated file, kept verbatim: "# Modified: removed the original validate, use only the passed-in RNG, fixed semantic palette."
    return source_notice(task_path) + "\n# 修改：移除原題 validate，僅用傳入 RNG，固定語義色盤。\n" + ast.unparse(module) + "\n", sorted(needed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--arcgen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ids = sorted(p.stem for p in (args.official / "data/training").glob("*.json"))
    if len(ids) != 1000:
        raise ValueError(f"official training should have 1000 tasks, got {len(ids)}")
    pins = {"official": revision(args.official), "atlas": revision(args.atlas), "arcgen": revision(args.arcgen)}
    args.out.mkdir(parents=True, exist_ok=True)
    for source in ("atlas", "arcgen"):
        (args.out / f"LICENSE-{source}").write_bytes((getattr(args, source) / "LICENSE").read_bytes())
    rows = []
    for task_id in ids:
        row = {"task_id": task_id, "status": "CANDIDATE_ONLY", "pins": pins}
        directory = args.out / task_id
        directory.mkdir(exist_ok=True)
        solver = args.atlas / "solutions" / f"{task_id}.py"
        content = solver.read_text()
        tree = ast.parse(content)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"open", "exec", "eval", "compile", "__import__"}:
                raise ValueError(f"{task_id}: unreviewed solver I/O or dynamic code")
        # Header written into the generated transform.py, kept verbatim; last line: "# Modified: added a NumPy
        # interface; candidate rules still await manual per-task review."
        notice = ("# Adapted from yhay81/arc-agi-2-solution-atlas (Apache-2.0).\n"
                  f"# Source commit: {pins['atlas']}; license: ../LICENSE-atlas.\n"
                  "# 修改：增加 NumPy 介面；候選規則仍待逐題人工審查。\n")
        wrapper = '\n\nimport numpy as np\n\ndef transform(grid):\n    return np.asarray(solve(np.asarray(grid).tolist()), dtype=int)\n'
        (directory / "transform.py").write_text(notice + adapt_solver(content) + wrapper)
        row["solver_source_sha256"] = digest(solver)
        generator = args.arcgen / "tasks" / f"task_{task_id}.py"
        if generator.exists():
            candidate, helpers = adapt_generator(generator, args.arcgen / "common.py")
            (directory / "generate_candidate.py").write_text(candidate)
            row.update(generator_source_sha256=digest(generator), helper_names=helpers,
                       generator_common_sha256=digest(args.arcgen / "common.py"))
        else:
            row["generator_missing"] = True
        (directory / "provenance.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        rows.append(row)
    (args.out / "manifest.json").write_text(json.dumps({"pins": pins, "tasks": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_solvers": len(rows), "candidate_generators": sum(not x.get("generator_missing") for x in rows), "completed_seeds": 0}))


if __name__ == "__main__":
    main()
