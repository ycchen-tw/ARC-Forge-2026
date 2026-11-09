"""deploy/sglang.lock.json: which sglang-arc-2 commit each deployment target runs, and how to recognise its tree.

Every sglang change lives in the private fork ycchen-tw/sglang-arc-2 (v0.5.20 + one commit per change). A target
(the Kaggle AGI-3 G4 notebook, the Kaggle ARC-AGI-2 L4 notebook, a Colab G4 or Runpod L4 experiment) names one fork
commit. Remote machines never clone the private fork: work/sglang_fork/export_tree.py turns the commit into
`sglang-<sha12>.tar.zst` (python/ of the commit) plus `SOURCE_SHA256-<sha12>` (h05's ARC_HISPARSE_QSA_MANIFEST: the
files the commit changes against v0.5.20), and this module checks an installed tree against the lock.

The package digest is arcforge.sglang_tree's whole-package digest. Trees unpacked from a tar that stored symlinks as
plain files (the L4 Kaggle/Runpod path) hash differently from a git checkout, so the lock keeps both:
`digest_symlink` (symlinks hashed as `symlink:<target>`) and `digest_deref` (as their target's bytes).
Standard library only (the Kaggle notebook imports it before any runtime is on the path).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arcforge.sglang_tree import manifest_digest, package_manifest

LOCK = Path(__file__).resolve().parents[2] / "deploy" / "sglang.lock.json"
FIELDS = ("commit", "digest_symlink", "digest_deref", "files", "archive", "archive_sha256", "source_manifest",
          "source_manifest_sha256")


def read_lock(path: Path = LOCK) -> dict:
    lock = json.loads(Path(path).read_text())
    for name, target in lock["targets"].items():
        missing = [f for f in FIELDS if f not in target]
        if missing:
            raise ValueError(f"{path}: target {name} lacks {missing}")
    return lock


def target(name: str, path: Path = LOCK) -> dict:
    targets = read_lock(path)["targets"]
    if name not in targets:
        raise KeyError(f"{path} has no target {name!r} (has {sorted(targets)})")
    return targets[name]


def tree_digests(package_root: Path) -> tuple[str, str, int]:
    """(digest_symlink, digest_deref, files) of an installed sglang package directory."""
    root = Path(package_root)
    m = package_manifest(root)
    deref = dict(m)
    for rel in m:
        if (root / rel).is_symlink():
            deref[rel] = hashlib.sha256((root / rel).resolve(strict=True).read_bytes()).hexdigest()
    return manifest_digest(m), manifest_digest(deref), len(m)


def check_tree(name: str, package_root: Path, path: Path = LOCK) -> dict:
    """Raise unless the sglang package at package_root is the lock's tree for `name` (either symlink convention)."""
    t = target(name, path)
    sym, deref, files = tree_digests(package_root)
    if files != t["files"] or (sym != t["digest_symlink"] and deref != t["digest_deref"]):
        raise RuntimeError(f"{package_root}: tree {sym[:12]}/{deref[:12]} files={files} is not lock target {name} "
                           f"({t['commit'][:12]}: {t['digest_symlink'][:12]}/{t['digest_deref'][:12]} files={t['files']})")
    return t
