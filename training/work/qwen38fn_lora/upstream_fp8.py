"""Unmodified, hash-pinned SGLang FP8 GEMM/quantizer without serving imports.

Only select existing @triton.jit definitions from their original source AST;
no body rewriting. Full Apache-2.0 sources/license live in the pinned overlay.
"""

import argparse
import ast
import hashlib
import sys
import types
from functools import lru_cache

import triton
import triton.language as tl

from arcforge import paths

REVISION = "73a255206f916366c8d26d4022f82ddfb0ab558d"
FILES = {
    "fused_moe_triton_kernels.py": (
        "python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py",
        "9c3342d3147e7d60a78a2c934111f0fc1becbb8df1d2d32aafc82e6c8a0b2e70",
        ["write_zeros_to_output", "fused_moe_kernel"],
    ),
    "fp8_kernel.py": (
        "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        "bf4d819c35f35e58be94085c081425284de4d1efd640ff4b000219e1bfef1169",
        ["_per_token_group_quant_8bit"],
    ),
}


def root():
    return paths.runs() / "qwen38fn_lora/20260908-sglang-fp8-kernels-v1"


def bootstrap():
    import requests

    directory = root()
    directory.mkdir(parents=True, exist_ok=True)
    for name, (relative, digest, _) in FILES.items():
        response = requests.get(
            f"https://raw.githubusercontent.com/sgl-project/sglang/{REVISION}/{relative}", timeout=60
        )
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != digest:
            raise RuntimeError("upstream source hash mismatch")
        (directory / name).write_bytes(response.content)
    response = requests.get(f"https://raw.githubusercontent.com/sgl-project/sglang/{REVISION}/LICENSE", timeout=60)
    response.raise_for_status()
    (directory / "LICENSE").write_bytes(response.content)


@lru_cache(maxsize=1)
def kernels():
    module = types.ModuleType("_arc_sglang_fp8_kernel")
    module.__dict__.update(triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    for name, (_, digest, names) in FILES.items():
        path = root() / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError("FP8 kernel changed; re-audit upstream source")
        tree = ast.parse(source, filename=str(path))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in selected} != set(names):
            raise RuntimeError("upstream kernel definitions missing")
        # Preserve original AST nodes, line numbers and filename for Triton source inspection.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.fused_moe_kernel, module._per_token_group_quant_8bit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap()
    kernels()
    print("unmodified SGLang FP8 kernels loaded:", REVISION)
