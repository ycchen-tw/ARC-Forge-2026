"""The whole-sglang-package digest, shared by the G4 runtime checks and the Kaggle notebook.

A patched sglang tree is identified by one sha256 over every file of the installed package
(``python/sglang``) except caches and the generated ``_version.py``, written per arm in
``patches/sglang-0.5.20/TREE_DIGEST`` by ``work/qsa_hisparse/g4_runtime.py tree-write``.
``work/qsa_hisparse/g4_runtime.py`` asserts it on a G4; ``work/arc_code/kaggle`` asserts it on
Kaggle after laying the HiSparse tree over the runtime archive. Standard library only: the
notebook imports this from the carried bundle before any runtime is on the path.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

#: generated at install time (setuptools_scm for an editable build, the packager for the runtime): not source
GENERATED = ("_version.py",)
TREE_ROW = re.compile(r"(base|hisparse) ([0-9a-f]{64}) files=(\d+) patches=([0-9a-f]{64})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_manifest(package_root: Path) -> dict[str, str]:
    """relpath -> sha256 over every file under the sglang package directory but caches and generated files."""
    root = Path(package_root)
    if not (root / "__init__.py").is_file() or not (root / "srt").is_dir():
        raise ValueError(f"{root} is not an sglang package directory")
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc" or rel in GENERATED:
            continue
        if path.is_symlink():  # upstream ships a few (.clang-format): the target is the content
            out[rel] = hashlib.sha256(b"symlink:" + os.readlink(path).encode()).hexdigest()
        elif path.is_file():
            out[rel] = sha256_file(path)
    return out


def manifest_digest(entries: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{d}  {r}\n" for r, d in sorted(entries.items())).encode()).hexdigest()


def patch_set(patches_dir: Path, arm: str) -> list[Path]:
    """The patches an arm's tree is v0.5.20 plus, in order: base 0*.patch, then (hisparse) its h*.patch."""
    base = sorted(Path(patches_dir).glob("0*.patch"))
    if not base:
        raise ValueError(f"no base patches under {patches_dir}")
    if arm == "base":
        return base
    if arm != "hisparse":
        raise ValueError(f"unknown arm {arm!r}")
    hs = sorted((Path(patches_dir) / "experimental-hisparse").glob("h*.patch"))
    if not hs:
        raise ValueError(f"no experimental-hisparse patches under {patches_dir}")
    return base + hs


def patches_digest(patches: list[Path]) -> str:
    return hashlib.sha256("".join(f"{sha256_file(p)}  {p.name}\n" for p in patches).encode()).hexdigest()


def read_tree_digest(path: Path) -> dict[str, dict]:
    """TREE_DIGEST rows: arm -> {digest, files, patches}. Both arms must be there."""
    return parse_tree_digest(Path(path).read_text(), str(path))


def parse_tree_digest(text: str, where: str) -> dict[str, dict]:
    """TREE_DIGEST text (a file, or ``git show <commit>:...``) -> rows; ``where`` names it in errors."""
    rows = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        m = TREE_ROW.fullmatch(line)
        if m is None:
            raise ValueError(f"{where}:{n} is not '<arm> <digest> files=<n> patches=<sha>': {line!r}")
        rows[m.group(1)] = {"digest": m.group(2), "files": int(m.group(3)), "patches": m.group(4)}
    missing = {"base", "hisparse"} - set(rows)
    if missing:
        raise ValueError(f"{where} has no row for {sorted(missing)}")
    return rows
