"""A pinned humming-kernels wheel as a PYTHONPATH overlay on venv-0520, so a B200 serve runs another Humming than the
shared venv's (0.1.12) without touching the venv. serve_b200.sh lays it with HUMMING_VERSION=<version>.

  python work/sglang_fork/humming_overlay.py build 0.1.18 $ARC_DATA/humming-overlays/0.1.18   # download, unzip, chmod
  python work/sglang_fork/humming_overlay.py check 0.1.18 $ARC_DATA/humming-overlays/0.1.18   # before every serve

Unzipped, not pip-installed (as the Kaggle notebooks' install_humming): the .dist-info the INT3 loader reads the version
from comes along, and the bundled NVRTC tools need their exec bit back (zipfile drops it). build writes MANIFEST.json
(sha256 of every file); check refuses an overlay that differs from it in any file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

#: version -> (x86_64 wheel on PyPI, its sha256 there)
WHEELS = {
    "0.1.18": ("https://files.pythonhosted.org/packages/c4/5d/692bb0ac2d62dc5bbaf291f9ac5da125334ba6a6647709b6021822a6a48e/"
               "humming_kernels-0.1.18-cp310-abi3-manylinux_2_28_x86_64.whl",
               "3eee5f7365457708a8962c852a4c797d99439a450dfe16fee47cc39664d080d4"),
}
MANIFEST = "MANIFEST.json"
NATIVE = Path("humming/_native/x86_64")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_of(root: Path) -> dict[str, str]:
    """sha256 of the wheel's files; the __pycache__ that importing the overlay writes is not the wheel's."""
    return {str(p.relative_to(root)): sha256_of(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != MANIFEST and "__pycache__" not in p.relative_to(root).parts}


def check(version: str, overlay: Path) -> None:
    if version not in WHEELS:
        raise SystemExit(f"humming {version}: no pinned wheel (have {sorted(WHEELS)})")
    if not (overlay / MANIFEST).is_file():
        raise SystemExit(f"{overlay}: no {MANIFEST}; build it: python {Path(__file__).name} build {version} {overlay}")
    manifest = json.loads((overlay / MANIFEST).read_text())
    if (manifest["version"], manifest["wheel_sha256"]) != (version, WHEELS[version][1]):
        raise SystemExit(f"{overlay}: built from humming {manifest['version']} {manifest['wheel_sha256'][:12]}, "
                         f"want {version} {WHEELS[version][1][:12]}")
    dist = sorted(p.name for p in overlay.glob("humming_kernels-*.dist-info"))
    if dist != [f"humming_kernels-{version}.dist-info"]:
        raise SystemExit(f"{overlay}: dist-info {dist}, want humming_kernels-{version}.dist-info")
    native = sorted((overlay / NATIVE).glob("*"))
    if not native or any(not p.stat().st_mode & 0o100 for p in native):
        raise SystemExit(f"{overlay / NATIVE}: native tools missing or not executable")
    have = files_of(overlay)
    if have != manifest["files"]:
        changed = sorted(set(have) ^ set(manifest["files"]) | {k for k in have if manifest["files"].get(k) != have[k]})
        raise SystemExit(f"{overlay}: {len(changed)} files differ from {MANIFEST}: {changed[:5]}")
    print(f"· humming overlay {version} ({len(have)} files, wheel {WHEELS[version][1][:12]}): {overlay}")


def build(version: str, overlay: Path) -> None:
    if version not in WHEELS:
        raise SystemExit(f"humming {version}: no pinned wheel (have {sorted(WHEELS)})")
    if overlay.exists():
        raise SystemExit(f"{overlay} exists; remove it first to rebuild")
    url, want = WHEELS[version]
    overlay.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".humming-{version}-", dir=overlay.parent))
    wheel = tmp / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=120) as r, open(wheel, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256_of(wheel)
    if got != want:
        raise SystemExit(f"{url}: sha256 {got}, pinned {want}")
    lay = tmp / "overlay"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(lay)
    for path in (lay / NATIVE).glob("*"):
        path.chmod(0o755)
    (lay / MANIFEST).write_text(json.dumps({"version": version, "wheel": url, "wheel_sha256": got,
                                            "files": files_of(lay)}, indent=1) + "\n")
    lay.rename(overlay)
    shutil.rmtree(tmp)
    check(version, overlay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("version")
    ap.add_argument("overlay", type=Path)
    a = ap.parse_args()
    (build if a.cmd == "build" else check)(a.version, a.overlay.absolute())


if __name__ == "__main__":
    sys.exit(main())
