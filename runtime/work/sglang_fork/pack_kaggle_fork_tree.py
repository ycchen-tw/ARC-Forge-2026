"""Stage the Kaggle dataset holding sglang-arc-2 trees for the arc-code notebook (config `sglang_tree`, no upload).

The notebook (work/arc_code/kaggle/notebook/03_serve.py install_sglang_tree) deletes the runtime archive's own sglang
source and extracts a tree in its place; it refuses members outside python/sglang/ and asserts the whole-package
digest (arcforge.sglang_tree, symlinks as `symlink:<target>`). This packs a deploy/sglang.lock.json target exactly as
pack_kaggle_tree.py packs a patch-built tree -- python/sglang/** of the fork commit plus the generated _version.py,
deterministic tar, zstd -- and checks the unpacked digest against the lock. Staged files:

  sglang-<sha12>.tar.zst      the tree
  SOURCE_SHA256-<sha12>       the fork commit's h05 manifest (export_tree.py)
  TREE-<sha12>.json           lock target, commit, digest, files, archive and manifest sha256
  dataset-metadata.json       threerabbits/sglang-arc-2-trees, private

    python work/sglang_fork/pack_kaggle_fork_tree.py <lock target> --out-root <staging dir> [--carry <earlier staging>]

Prints the notebook config's `sglang_tree` block.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "work/qsa_hisparse/probes"))
sys.path.insert(0, str(HERE))
import export_tree  # noqa: E402
import pack_kaggle_tree as pk  # noqa: E402
from arcforge import sglang_lock  # noqa: E402

SLUG = "sglang-arc-2-trees"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--carry", type=Path, help="an earlier staging dir of this dataset: its trees are kept")
    a = ap.parse_args()
    t = sglang_lock.target(a.target)
    tag = t["commit"][:12]
    stage = a.out_root / SLUG
    stage.mkdir(parents=True, exist_ok=False)
    carried = pk.carry(a.carry, stage) if a.carry else []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar = tmp / "src.tar"
        tar.write_bytes(subprocess.run(["git", "-C", str(export_tree.paths.sglang_fork()), "archive", "--format=tar", t["commit"],
                                        "python/sglang"], check=True, capture_output=True).stdout)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(tmp)], check=True)
        pkg = tmp / "python" / "sglang"
        (pkg / "_version.py").write_text(pk.VERSION_PY)
        archive = stage / f"sglang-{tag}.tar.zst"
        pk.pack(tmp / "python", archive)
        check = tmp / "check"
        check.mkdir()
        got = pk.unpack_and_digest(archive, check)
        if (got["digest"], got["files"]) != (t["digest_symlink"], t["files"]):
            raise SystemExit(f"packed tree {got['digest'][:12]} ({got['files']} files) != lock {a.target} "
                             f"{t['digest_symlink'][:12]} ({t['files']})")
        # the h05 manifest: the commit's changed files (export_tree), each hash checked against the packed tree
        manifest_text = subprocess.run(["git", "-C", str(export_tree.paths.sglang_fork()), "diff", "--name-only", export_tree.UPSTREAM,
                                        t["commit"], "--", "python/sglang"], check=True, capture_output=True,
                                       text=True).stdout.split()
        lines = []
        for p in sorted(manifest_text):
            rel = p.removeprefix("python/sglang/")
            f = got["root"] / "python" / "sglang" / rel
            if f.is_file():
                lines.append(f"{pk.sha256(f)}  {rel}\n")
        manifest = stage / f"SOURCE_SHA256-{tag}"
        manifest.write_text("".join(lines))
        if pk.sha256(manifest) != t["source_manifest_sha256"]:
            raise SystemExit(f"{manifest.name} differs from the lock's source manifest")
    record = {"lock_target": a.target, "commit": t["commit"], "tree_digest": t["digest_symlink"],
              "tree_files": t["files"], "archive": archive.name, "archive_sha256": pk.sha256(archive),
              "manifest": manifest.name, "manifest_sha256": pk.sha256(manifest)}
    (stage / f"TREE-{tag}.json").write_text(json.dumps(record, indent=1) + "\n")
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": SLUG, "id": f"{pk.OWNER}/{SLUG}", "licenses": [{"name": "other"}]}, indent=1) + "\n")
    pk.scan([stage])
    block = {"dataset": f"{pk.OWNER}/{SLUG}", "lock_target": a.target, "commit": t["commit"],
             "archive": archive.name, "archive_sha256": record["archive_sha256"],
             "tree_digest": record["tree_digest"], "tree_files": record["tree_files"],
             "manifest": manifest.name, "manifest_sha256": record["manifest_sha256"]}
    print(json.dumps({"staged": str(stage), "carried": carried, "sglang_tree": block}, indent=1))


if __name__ == "__main__":
    main()
