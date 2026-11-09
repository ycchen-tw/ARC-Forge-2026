"""eval120fix: applies fixes.json's cell-by-cell fixes to the 120 ARC-AGI-2 public evaluation tasks, written as a --tasks-dir run_arc2 can use directly.

    python work/arc_code/arc2/eval120fix/build.py [--src <evaluation directory>] [--out <output directory>]

Defaults: src = $ARC_DATA/ARC-AGI-2/data/evaluation, out = $ARC_DATA/arc2_eval120fix/evaluation.
Tasks not listed in fixes.json are copied as is (byte for byte). A task to fix is first checked against upstream_sha256,
then each cell against its pre-fix value; any mismatch stops: neither an upstream change nor a mistake in fixes.json may
be applied silently. An existing out with different content also stops (delete it yourself to rebuild).
Writes MANIFEST.json alongside: each task's sha256 and which cells were changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
from arcforge import paths  # noqa: E402


def apply(task: dict, edits: list[dict]) -> dict:
    task = json.loads(json.dumps(task))
    for e in edits:
        grid = task[e["split"]][e["pair"]][e["grid"]]
        for r, c, old, new in e["cells"]:
            if grid[r][c] != old:
                raise SystemExit(f"{e['split']}[{e['pair']}].{e['grid']} ({r},{c}) is {grid[r][c]}, fixes.json says {old}")
            grid[r][c] = new
    return task


def build(src: Path, out: Path) -> dict:
    spec = json.loads((HERE / "fixes.json").read_text())
    fixes = {f["task"]: f for f in spec["fixes"]}
    files = sorted(src.glob("*.json"))
    if len(files) != 120:
        raise SystemExit(f"{src}: {len(files)} tasks, expected 120")
    missing = set(fixes) - {f.stem for f in files}
    if missing:
        raise SystemExit(f"fixes.json names tasks not in {src}: {sorted(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(src), "upstream": spec["upstream"], "tasks": {}}
    for f in files:
        raw = f.read_bytes()
        entry = {"upstream_sha256": hashlib.sha256(raw).hexdigest()}
        if f.stem in fixes:
            fx = fixes[f.stem]
            if entry["upstream_sha256"] != fx["upstream_sha256"]:
                raise SystemExit(f"{f.stem}: upstream sha256 {entry['upstream_sha256']} != fixes.json {fx['upstream_sha256']}")
            raw = (json.dumps(apply(json.loads(raw), fx["edits"]), separators=(",", ":")) + "\n").encode()
            entry.update(kind=fx["kind"], cells=sum(len(e["cells"]) for e in fx["edits"]))
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        dst = out / f.name
        if dst.exists() and dst.read_bytes() != raw:
            raise SystemExit(f"{dst} exists with different content; delete {out} to rebuild")
        dst.write_bytes(raw)
        manifest["tasks"][f.stem] = entry
    (out.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    src = args.src or paths.data() / "ARC-AGI-2" / "data" / "evaluation"
    out = args.out or paths.data() / "arc2_eval120fix" / "evaluation"
    m = build(src, out)
    changed = {k: v for k, v in m["tasks"].items() if "kind" in v}
    print(f"{out}: 120 tasks, {len(changed)} edited: " + ", ".join(f"{k} ({v['kind']}, {v['cells']} cells)" for k, v in changed.items()))


if __name__ == "__main__":
    main()
