"""Unpack the exact code a final submission notebook carried, byte for byte.

    python submissions/extract_code.py submissions/<final>/notebook.ipynb <out dir>

ARC-AGI-2 notebooks (L4x4) embed the harness as `BUNDLE_B64` (a tar, listed in
`BUNDLE_FILES`, its git origin in `BUNDLE_GIT`); the serving-side code ships in the payload
archive of the Kaggle dataset named in `ASSET_MANIFEST["source_dataset"]`.
ARC-AGI-3 notebooks embed every file they run as `PAYLOAD` (a tar), with per-file sha256
prefixes in `BUILD["carried"]`.

The script checks the unpacked files against those manifests and fails loudly on any mismatch.
"""
import ast
import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path


def literals(source: str, names: set[str]) -> dict:
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def untar(blob: bytes) -> dict[str, bytes]:
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        return {m.name.removeprefix("./"): tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    notebook, out = Path(sys.argv[1]), Path(sys.argv[2])
    cells = json.loads(notebook.read_text())["cells"]
    source = "".join(cells[1]["source"])
    lit = literals(source, {"BUNDLE_B64", "BUNDLE_FILES", "BUNDLE_GIT", "PAYLOAD", "BUILD"})
    if "BUNDLE_B64" in lit:
        files = untar(base64.b64decode(lit["BUNDLE_B64"]))
        listed = {x if isinstance(x, str) else x[0] for x in lit["BUNDLE_FILES"]}
        if set(files) != listed:
            raise SystemExit(f"bundle does not match BUNDLE_FILES: {sorted(set(files) ^ listed)[:10]}")
        origin = f"arc-forge {lit['BUNDLE_GIT']['head']} (dirty: {lit['BUNDLE_GIT']['dirty']})"
    elif "PAYLOAD" in lit:
        files = untar(base64.b64decode(lit["PAYLOAD"]))
        carried = lit["BUILD"]["carried"]
        if set(files) != set(carried):
            raise SystemExit(f"payload does not match BUILD['carried']: {sorted(set(files) ^ set(carried))[:10]}")
        bad = [p for p, h in carried.items() if not hashlib.sha256(files[p]).hexdigest().startswith(h)]
        if bad:
            raise SystemExit(f"sha256 mismatch against BUILD['carried']: {bad[:10]}")
        origin = f"arc-forge {lit['BUILD']['commit']} (working tree dirty: {lit['BUILD']['dirty']})"
    else:
        raise SystemExit(f"{notebook}: neither BUNDLE_B64 nor PAYLOAD found in cell 1")
    for rel, data in files.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    print(f"{len(files)} files -> {out}  (built from {origin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
