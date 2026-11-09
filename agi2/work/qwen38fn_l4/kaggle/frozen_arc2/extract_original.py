"""Extract unchanged %%writefile bodies from the audited ARC2 notebook."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

NOTEBOOK_SHA256 = "4738fa3a764817442f94049823a9ee24a21ef3e205d78eeb25ce0cd5b1b84ef2"
PREFIX = "/tmp/arcforge_qwen38fn_nvfp4/"
ROOT = Path(__file__).resolve().parent


def extract(notebook: Path):
    previous = ROOT / "ORIGINAL_MANIFEST.json"
    if previous.is_file():
        patched = [name for name, identity in json.loads(previous.read_text())["files"].items()
                   if "patched_from_sha256" in identity]
        if patched:
            # 重新抽取會把補丁蓋回原版（v15 的 agentic_tir.py）；要重抽就先手動確認並移除 manifest 裡的補丁記錄。
            raise RuntimeError(f"refusing to overwrite patched frozen sources: {patched}")
    raw = notebook.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise RuntimeError("notebook differs from the audited original")
    document = json.loads(raw)
    files = {}
    for index, cell in enumerate(document["cells"]):
        source = "".join(cell["source"])
        if not source.startswith("%%writefile "):
            continue
        header, body = source.split("\n", 1)
        name = header.removeprefix("%%writefile ")
        if not name.startswith(PREFIX):
            raise RuntimeError(f"unexpected embedded path: {name}")
        relative = PurePosixPath(name.removeprefix(PREFIX))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in files:
            raise RuntimeError("unsafe or duplicated embedded path")
        target = ROOT / relative
        encoded = body.encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        files[str(relative)] = {"cell": index, "sha256": hashlib.sha256(encoded).hexdigest(),
                                "bytes": len(encoded), "characters": len(body)}
    shell = None
    for node in ast.walk(ast.parse("".join(document["cells"][1]["source"]))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
            if "src/arcforge/__init__.py" in ast.unparse(node.func.value):
                shell = ast.literal_eval(node.args[0]).encode()
    if shell is None or len(files) != 9:
        raise RuntimeError("original package shell or one of the nine writefile cells is missing")
    target = ROOT / "src/arcforge/__init__.py"
    target.write_bytes(shell)
    files["src/arcforge/__init__.py"] = {"cell": 1, "sha256": hashlib.sha256(shell).hexdigest(), "bytes": len(shell)}
    manifest = {"notebook_sha256": NOTEBOOK_SHA256, "scope": "unchanged original embedded ARC2 source", "files": files}
    (ROOT / "ORIGINAL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    print(json.dumps(extract(parser.parse_args().notebook), indent=2))
