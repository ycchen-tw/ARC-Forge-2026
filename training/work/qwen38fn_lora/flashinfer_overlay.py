"""Pinned flashinfer wheels as a PYTHONPATH overlay on venv-fn-train (which has no flashinfer), for the trainer's
--dense-qat-gemm cutlass (fp8_dense_qat.py: flashinfer's SM100 CUTLASS groupwise-FP8 GEMM). The venv is not touched.

  python work/qwen38fn_lora/flashinfer_overlay.py build $ARC_DATA/flashinfer-overlays/0.6.18   # download, sha256, unzip
  python work/qwen38fn_lora/flashinfer_overlay.py check $ARC_DATA/flashinfer-overlays/0.6.18   # what the trainer runs

flashinfer-python 0.6.18 (the serving venv's version) plus the two pure-Python dependencies venv-fn-train lacks
(nvidia-ml-py for ``import pynvml``, tabulate); the rest it needs (torch 2.11 cu130, apache-tvm-ffi 0.1.12, ninja,
cuda-python, einops, click, requests, packaging, numpy, tqdm) is the venv's own. The CUTLASS kernel is JIT-compiled by
ninja + nvcc (the venv's bin/ on PATH, $CUDA_HOME) into $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/0.6.18/100a on
first use (~minutes, once per cache), against the venv's tvm-ffi (flashinfer 0.6 binds through tvm-ffi, not torch, so
the module does not depend on the torch build). A trainer launch therefore sets PYTHONPATH=<venv paths>:<overlay>,
FLASHINFER_WORKSPACE_BASE=<a cache of its own> and PATH=<venv>/bin:$PATH.
Unzipped as pip would install them (top-level files included); build writes MANIFEST.json (sha256 of every file) and
check refuses an overlay that differs from it in any file. ``check_imported()`` checks the flashinfer a process
actually imported: the overlay it came from must pass check.
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

VERSION = "0.6.18"
#: wheel file -> (PyPI URL, sha256)
WHEELS = {
    "flashinfer_python-0.6.18-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/31/7d/3dae45fc293cd5f16cefe4dcf59683d964ca8faf2303ed7fff54f4bca490/"
        "flashinfer_python-0.6.18-py3-none-any.whl",
        "d5d26edb48f8def0bf28b2c94415aef7f0754e2cd9e3cb3d6e7081262d068aad"),
    "nvidia_ml_py-13.615.71-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/53/a1/1681dfa1c904d4e3e72e51b55a0ff012d50b766843ef832d584abe2113c6/"
        "nvidia_ml_py-13.615.71-py3-none-any.whl",
        "959bf4adf6fe1308e4bd739e722236b0d1ec8392e2cefad33ff70c311380b9b6"),
    "tabulate-0.10.0-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/99/55/db07de81b5c630da5cbf5c7df646580ca26dfaefa593667fc6f2fe016d2e/"
        "tabulate-0.10.0-py3-none-any.whl",
        "f0b0622e567335c8fabaaa659f1b33bcb6ddfe2e496071b743aa113f8774f2d3"),
}
MANIFEST = "MANIFEST.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_of(root: Path) -> dict[str, str]:
    """sha256 of the wheels' files; the __pycache__ that importing the overlay writes is not theirs."""
    return {str(p.relative_to(root)): sha256_of(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != MANIFEST and "__pycache__" not in p.relative_to(root).parts}


def check(overlay: Path) -> None:
    if not (overlay / MANIFEST).is_file():
        raise SystemExit(f"{overlay}: no {MANIFEST}; build it: python {Path(__file__).name} build {overlay}")
    manifest = json.loads((overlay / MANIFEST).read_text())
    want = {name: sha for name, (_, sha) in WHEELS.items()}
    if (manifest["version"], manifest["wheels"]) != (VERSION, want):
        raise SystemExit(f"{overlay}: built from {manifest['version']} {sorted(manifest['wheels'])}, "
                         f"want {VERSION} {sorted(want)}")
    dist = sorted(p.name for p in overlay.glob("flashinfer_python-*.dist-info"))
    if dist != [f"flashinfer_python-{VERSION}.dist-info"]:
        raise SystemExit(f"{overlay}: dist-info {dist}, want flashinfer_python-{VERSION}.dist-info")
    have = files_of(overlay)
    if have != manifest["files"]:
        changed = sorted(set(have) ^ set(manifest["files"]) | {k for k in have if manifest["files"].get(k) != have[k]})
        raise SystemExit(f"{overlay}: {len(changed)} files differ from {MANIFEST}: {changed[:5]}")
    print(f"· flashinfer overlay {VERSION} ({len(have)} files): {overlay}", flush=True)


def check_imported() -> Path:
    """The overlay the imported ``flashinfer`` lives in, after check(); raises for a flashinfer from anywhere else."""
    import flashinfer

    root = Path(flashinfer.__file__).resolve().parents[1]
    if flashinfer.__version__ != VERSION:
        raise RuntimeError(f"imported flashinfer {flashinfer.__version__} from {root}, the trainer pins {VERSION}")
    try:
        check(root)
    except SystemExit as exc:  # check() speaks to a shell; a trainer gets an exception with the same text
        raise RuntimeError(f"flashinfer imported from {root} is not a checked overlay: {exc}") from None
    return root


def build(overlay: Path) -> None:
    if overlay.exists():
        raise SystemExit(f"{overlay} exists; remove it first to rebuild")
    overlay.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".flashinfer-{VERSION}-", dir=overlay.parent))
    lay = tmp / "overlay"
    got = {}
    for name, (url, want) in WHEELS.items():
        wheel = tmp / name
        with urllib.request.urlopen(url, timeout=120) as r, open(wheel, "wb") as f:
            shutil.copyfileobj(r, f)
        got[name] = sha256_of(wheel)
        if got[name] != want:
            raise SystemExit(f"{url}: sha256 {got[name]}, pinned {want}")
        with zipfile.ZipFile(wheel) as archive:
            clash = [n for n in archive.namelist() if (lay / n).exists()]
            if clash:
                raise SystemExit(f"{name} overwrites files of another wheel: {clash[:3]}")
            archive.extractall(lay)
    (lay / MANIFEST).write_text(json.dumps({"version": VERSION, "wheels": got, "files": files_of(lay)}, indent=1) + "\n")
    lay.rename(overlay)
    shutil.rmtree(tmp)
    check(overlay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("overlay", type=Path)
    a = ap.parse_args()
    (build if a.cmd == "build" else check)(a.overlay.absolute())


if __name__ == "__main__":
    sys.exit(main())
