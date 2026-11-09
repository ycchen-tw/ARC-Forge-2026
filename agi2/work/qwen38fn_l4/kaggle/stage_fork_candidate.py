"""Stage an sglang-arc-2 fork commit as an L4 Kaggle source candidate (the r9-r2 candidate directory format).

    python work/qwen38fn_l4/kaggle/stage_fork_candidate.py --lock-target arc-main --out-dir <new dir>

The L4 installer (runtime_setup.py) and packer (pack_asset.py --candidate-dir) read the layout that
work/qwen38fn_l4/probes/stage_source_candidates.py writes for the patch-built r9-r2 tree:

  source/sglang/            the package, symlinks stored as their target's bytes (the L4 "deref" convention)
  source.tar.gz             deterministic tar.gz, top level ``sglang/`` (deterministic_archive)
  SOURCE_SHA256             h05's ARC_HISPARSE_QSA_MANIFEST: every python/sglang file the commit changes against
                            upstream v0.5.20 (work/sglang_fork/export_tree.py), ``<sha256>  <relpath>``
  PACKAGE_SHA256(.json)     whole-package manifest (arcforge.sglang_tree.package_manifest)
  ARCHIVE_VALIDATION.json   CPU roundtrip of source.tar.gz
  CANDIDATE.json            tree digest, file count, manifest and archive sha256, fork commit

The fork commit comes from deploy/sglang.lock.json; the tree is read with ``git archive`` from a local fork clone
(SGLANG_ARC_REPO, never cloned on a remote machine) and must reproduce the lock's ``digest_deref``, ``files`` and
``source_manifest_sha256`` exactly, or nothing is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "work/qwen38fn_l4/probes"))
from arcforge import sglang_lock, sglang_tree  # noqa: E402
from stage_source_candidates import deterministic_archive, verify_archive, write_json  # noqa: E402

FORK = Path(os.environ.get("SGLANG_ARC_REPO") or os.environ["ARC_SGLANG_FORK"])
UPSTREAM = "v0.5.20"


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(FORK), *args], check=True, capture_output=True).stdout


def stage(lock_target: str, out: Path) -> dict:
    pin = sglang_lock.target(lock_target)
    commit = git("rev-parse", pin["commit"] + "^{commit}").decode().strip()
    if commit != pin["commit"]:
        raise RuntimeError(f"fork clone resolves {pin['commit']} to {commit}")
    if out.exists():
        raise FileExistsError("preserve existing candidates; choose a fresh output directory")
    with tempfile.TemporaryDirectory(prefix="l4-fork-candidate-") as tmp:
        tmp = Path(tmp)
        (tmp / "t.tar").write_bytes(git("archive", "--format=tar", commit, "python/sglang"))
        subprocess.run(["tar", "-xf", str(tmp / "t.tar"), "-C", str(tmp)], check=True)
        package = tmp / "python/sglang"
        symlinks = sorted(p for p in package.rglob("*") if p.is_symlink())
        for link in symlinks:
            content = link.resolve(strict=True).read_bytes()
            if not link.resolve().is_relative_to(package.resolve()):
                raise RuntimeError(f"symlink escapes the package: {link}")
            link.unlink()
            link.write_bytes(content)
        entries = sglang_tree.package_manifest(package)
        digest = sglang_tree.manifest_digest(entries)
        if (digest, len(entries)) != (pin["digest_deref"], pin["files"]):
            raise RuntimeError(f"fork tree {digest[:12]} ({len(entries)} files) != lock {lock_target} "
                               f"deref {pin['digest_deref'][:12]} ({pin['files']} files)")
        changed = git("diff", "--name-only", UPSTREAM, commit, "--", "python/sglang").decode().split()
        rows = []
        for path in sorted(changed):
            rel = path.removeprefix("python/sglang/")
            if rel in entries:  # files deleted against upstream are not in the package
                rows.append(f"{entries[rel]}  {rel}\n")
        rendered = "".join(rows)
        manifest_sha = hashlib.sha256(rendered.encode()).hexdigest()
        if manifest_sha != pin["source_manifest_sha256"]:
            raise RuntimeError(f"SOURCE_SHA256 {manifest_sha[:12]} differs from the lock's {pin['source_manifest_sha256'][:12]}")
        if "srt/mem_cache/qsa_hisparse_distributed.py" not in entries or not any(
                row.endswith("  srt/mem_cache/qsa_hisparse_distributed.py\n") for row in rows):
            raise RuntimeError("L4 launcher needs the h25 distributed helper in SOURCE_SHA256")
        staging = out.parent / (out.name + ".partial")
        if staging.exists():
            raise FileExistsError(f"stale partial candidate: {staging}")
        (staging / "source").mkdir(parents=True)
        shutil.copytree(package, staging / "source/sglang")
    package = staging / "source/sglang"
    if sglang_tree.package_manifest(package) != entries:
        raise RuntimeError("copied candidate package differs from the archived commit")
    (staging / "SOURCE_SHA256").write_text(rendered)
    (staging / "PACKAGE_SHA256").write_text("".join(f"{sha}  {name}\n" for name, sha in sorted(entries.items())))
    write_json(staging / "PACKAGE_SHA256.json", entries)
    archive = staging / "source.tar.gz"
    deterministic_archive(package, archive)
    validation = verify_archive(archive, entries)
    write_json(staging / "ARCHIVE_VALIDATION.json", validation)
    record = {"status": "FORK_EXPORT_CPU_VALIDATED", "arm": "fork", "lock_target": lock_target,
              "fork": "ycchen-tw/sglang-arc-2", "fork_ref": pin.get("ref"), "fork_commit": commit,
              "composition": f"sglang-arc-2 {pin.get('ref')} {commit[:12]}",
              "tree_sha256": digest, "tree_digest_symlink": pin["digest_symlink"], "files": len(entries),
              "dereferenced_symlinks": [p.relative_to(tmp / "python/sglang").as_posix() for p in symlinks],
              "source_manifest_sha256": manifest_sha, "source_manifest_files": len(rows),
              "source_archive": archive.name, "source_archive_sha256": sglang_tree.sha256_file(archive),
              "source_archive_bytes": archive.stat().st_size, "archive_roundtrip_verified": True,
              "gpu_execution": False, "runtime_archive": "reuse pinned SGLang 0.5.20 SM89 binary runtime"}
    write_json(staging / "CANDIDATE.json", record)
    staging.rename(out)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock-target", required=True, help="deploy/sglang.lock.json target, e.g. arc-main")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.lock_target, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
