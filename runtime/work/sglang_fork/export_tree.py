"""Export one sglang-arc-2 commit as a deployable tree: what a remote machine gets instead of a clone.

  python export_tree.py <ref> <out dir>

Writes <out>/sglang-<sha12>.tar.zst (python/ of the commit, as `python/...`), <out>/SOURCE_SHA256-<sha12> (every
python/sglang file the commit changes against upstream v0.5.20, in h05's `<sha256>  <relpath>` format, for
ARC_HISPARSE_QSA_MANIFEST) and <out>/TREE-<sha12>.json: commit, package digest under both symlink conventions
(`symlink` = arcforge.sglang_tree.package_manifest, as G4 TREE_DIGEST rows; `deref` = symlinks stored as their
target's content, as the L4 tarball rows), file count, tarball and manifest sha256.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM = "v0.5.20"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arcforge.sglang_tree import manifest_digest, package_manifest  # noqa: E402
from arcforge import paths  # noqa: E402


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(paths.sglang_fork()), *args], check=True, capture_output=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def export(ref: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    commit = git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    tag = commit[:12]
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "t.tar"
        tar.write_bytes(git("archive", "--format=tar", commit, "python"))
        subprocess.run(["tar", "-xf", str(tar), "-C", tmp], check=True)
        root = Path(tmp) / "python" / "sglang"
        m = package_manifest(root)
        deref = dict(m)
        for rel in m:
            if (root / rel).is_symlink():
                deref[rel] = sha((root / rel).resolve(strict=True).read_bytes())
        archive = out / f"sglang-{tag}.tar.zst"
        subprocess.run(["zstd", "-q", "-T0", "-19", "-f", str(tar), "-o", str(archive)], check=True)
    changed = [p for p in git("diff", "--name-only", UPSTREAM, commit, "--", "python/sglang").decode().split()]
    lines = []
    for p in sorted(changed):
        rel = p.removeprefix("python/sglang/")
        if rel not in m:  # deleted upstream file
            continue
        lines.append(f"{m[rel]}  {rel}\n")
    man = out / f"SOURCE_SHA256-{tag}"
    man.write_text("".join(lines))
    info = {
        "commit": commit, "ref": ref, "files": len(m),
        "digest_symlink": manifest_digest(m), "digest_deref": manifest_digest(deref),
        "archive": archive.name, "archive_sha256": sha(archive.read_bytes()),
        "source_manifest": man.name, "source_manifest_sha256": sha(man.read_bytes()), "source_files": len(lines),
    }
    (out / f"TREE-{tag}.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    print(json.dumps(export(sys.argv[1], Path(sys.argv[2])), indent=2))


if __name__ == "__main__":
    main()
