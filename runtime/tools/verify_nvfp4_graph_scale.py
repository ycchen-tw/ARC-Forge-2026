#!/usr/bin/env python3
"""Fail before serving if HybridLinearKVPool hides NVFP4's device scales."""
import ast
import hashlib
import sys
from pathlib import Path


def main() -> None:
    source = (
        Path(sys.argv[1])
        / "python/sglang/srt/mem_cache/memory_pool.py"
    )
    raw = source.read_bytes()
    tree = ast.parse(raw, filename=str(source))
    pool = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HybridLinearKVPool"
    )
    method = next(
        node for node in pool.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_kv_buffer"
    )
    defaults = dict(zip(
        (arg.arg for arg in method.args.args[-len(method.args.defaults):]),
        method.args.defaults,
    ))
    for name in ("k_scale", "v_scale"):
        value = defaults.get(name)
        if not isinstance(value, ast.Constant) or value.value is not None:
            raise SystemExit(
                f"{source}: {name} must default to None so the full KV pool "
                "uses its preloaded GPU scale during CUDA graph capture"
            )
    print(f"NVFP4 graph-scale patch OK: {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
