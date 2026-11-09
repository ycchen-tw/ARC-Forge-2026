#!/bin/bash
# Packages the sglang 0.5.20 line as a relocatable runtime for Kaggle RTX PRO 6000 (sm_120).
# The old line's package_runtime_sm120.sh stays as it is — the v22 submission must stay reproducible; the two are independent.
#
#   source env/env.sh; bash work/qwen38fn_kaggle/package_runtime_sm120_0520.sh      # on the 8×B200 build node
#
# Three differences from the old line:
#
# 1. **The source tree is rebuilt by this script itself, not taken from $HOME/work/sglang-0520.**
#    That shared tree carries other sessions' uncommitted work — measured on 2026-09-20: the new file
#    host_kv_pool.py, plus changes to six files memory_pool / pool_host/mha / pool_configurator /
#    hybrid_pool_assembler / kv_cache_configurator (a sparse KV offload
#    prototype). Packaging from it would ship someone else's experiment in the scored runtime, and the notebook
#    could not tell afterwards what it ran. So this clones fresh from tag v0.5.20 and applies only the
#    **committed** patches in patches/sglang-0.5.20; by construction the result equals main.
#
# 2. **Patches are applied at packaging time; the notebook no longer applies them.** The old line ran
#    install_qsa_sm120 + install_qsa_nvfp4_kaggle on Kaggle, so "which patch did not apply" only surfaced
#    where nobody could see it (and it happened: piping the installer output into | head made a patch take a SIGPIPE and half-apply).
#    Now the notebook only reads package/runtime-provenance.json and asserts.
#
# 3. **AOT comes from the flashinfer_jit_cache package, not flashinfer/data/aot.**
#    The latter matches only module names, has no version check, and silently falls back to JIT on a load
#    failure; with the former, flashinfer itself checks the version and raises. Details in the docstring of install_flashinfer_jit_cache_sm120.py.
set -euo pipefail

ARC_ROOT=${ARC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$HOME/work/dsv4/venv-0520}
CUDA_SRC=${CUDA_SRC:-/usr/local/cuda-13.1}
OUT_ROOT=${OUT_ROOT:-$HOME/work/qwen38fn_kaggle_0520}
SRC=${SRC:-$OUT_ROOT/src/sglang-0520}
PATCHES=${PATCHES:-$ARC_ROOT/patches/sglang-0.5.20}
PYVER=${PYVER:-python3.12}
DATASET_SLUG=${DATASET_SLUG:-threerabbits/qwen38fn-sglang-0520-sm120-cp312}
# The archive is GB-sized and goes to NFS; staging must be on the same filesystem as site-packages for hardlinks to work.
ARCHIVE_DIR=${ARCHIVE_DIR:-$ARC_NFS/models/kaggle-runtime}
FA2_WHEEL=${FA2_WHEEL:-$(ls "$HOME"/work/qwen38fn_kaggle/fa2/wheels/flash_attn-2.*-cp312-*.whl 2>/dev/null | head -1)}

REF=$(sed -n 's/^REF=\([0-9a-f]\{40\}\).*/\1/p' "$PATCHES/BASE")
[ -n "$REF" ] || { echo "✗ cannot read REF from $PATCHES/BASE" >&2; exit 1; }

echo "== $(date -u +%FT%TZ) preflight checks"
[ -x "$VENV/bin/python" ]   || { echo "✗ no $VENV/bin/python" >&2; exit 1; }
[ -x "$VENV/bin/ninja" ]    || { echo "✗ venv has no ninja" >&2; exit 1; }
[ -x "$CUDA_SRC/bin/nvcc" ] || { echo "✗ no $CUDA_SRC/bin/nvcc" >&2; exit 1; }
command -v zstd >/dev/null  || { echo "✗ zstd is required" >&2; exit 1; }
[ -n "$FA2_WHEEL" ] && [ -f "$FA2_WHEEL" ] || { echo "✗ no FA2 wheel (FA2_WHEEL=)" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$ARCHIVE_DIR"

# ---- 1. rebuild the source tree (private; the shared tree is not touched) ----------------
echo "== $(date -u +%FT%TZ) source tree $SRC"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --branch v0.5.20 --depth 1 https://github.com/sgl-project/sglang "$SRC"
fi
got=$(git -C "$SRC" rev-parse HEAD)
[ "$got" = "$REF" ] || { echo "✗ HEAD $got != $REF from BASE" >&2; exit 1; }
# The rebuild must be repeatable: restore tracked files to pristine, then apply everything again. Untracked files
# (added by patches) stay — git apply fails on a new file that already exists, so the loop uses a reverse check to skip applied ones.
git -C "$SRC" checkout -- .
n=0
for p in "$PATCHES"/*.patch; do
  if git -C "$SRC" apply --check --reverse "$p" 2>/dev/null; then
    echo "   $(basename "$p"): already applied"; continue
  fi
  git -C "$SRC" apply "$p" || { echo "✗ $(basename "$p") does not apply" >&2; exit 1; }
  echo "   $(basename "$p"): applied"; n=$((n + 1))
done
echo "   applied $n this run; $(ls "$PATCHES"/*.patch | wc -l) in the directory"

# ---- 2. marker gate: the patch stack must not silently miss a piece ----------------------
echo "== $(date -u +%FT%TZ) marker gate"
PYTHONPATH="$ARC_ROOT/work/qwen38fn_eval" "$VENV/bin/python" - "$SRC" <<'PY'
import sys
from pathlib import Path

import sglang_0520_markers as gate

root = Path(sys.argv[1])
absent = gate.missing(root) + gate.missing(root, gate.SM120_MARKERS)
for row in absent:
    print("✗", row)
if absent:
    raise SystemExit(f"patch stack incomplete: {len(absent)} markers missing")
print(f"✓ markers present in all {len(gate.MARKERS) + len(gate.SM120_MARKERS)} files")
PY

# ---- 3. staging (hardlinks, same fs as site-packages) ------------------------------------
SITE_SRC=$VENV/lib/$PYVER/site-packages
SGL_COMMIT=$(git -C "$SRC" rev-parse --short HEAD)
NAME=${NAME:-sglang-0520-${SGL_COMMIT}-sm120-cp312}
STAGE=$(mktemp -d "$OUT_ROOT/stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
PKG=$STAGE/package
mkdir -p "$PKG/lib/$PYVER" "$PKG/bin" "$PKG/python" "$PKG/cuda-13.1"

echo "== $(date -u +%FT%TZ) site-packages (hardlinks preferred)"
cp -al "$SITE_SRC" "$PKG/lib/$PYVER/site-packages" 2>/dev/null \
  || cp -a "$SITE_SRC" "$PKG/lib/$PYVER/site-packages"
SITE=$PKG/lib/$PYVER/site-packages

echo "== $(date -u +%FT%TZ) sglang source (the rebuilt tree, without __pycache__)"
rsync -a --exclude '__pycache__' --exclude '*.egg-info' "$SRC/python/sglang/" "$PKG/python/sglang/"
# _version.py is generated at install time and is not in the source tree; without it sglang.version falls back to
# importlib.metadata, whose dist-info points at the editable install, which no longer holds on Kaggle.
"$VENV/bin/python" - "$PKG/python/sglang/_version.py" <<'PY'
import sys
from importlib.metadata import version
v = version("sglang")
open(sys.argv[1], "w").write(
    "# generated by package_runtime_sm120_0520.sh; matches the packaged source\n"
    "from __future__ import annotations\n"
    f'__version__ = "{v}"\n'
    f"__version_tuple__ = {tuple(v.split('.'))!r}\n")
print("   _version.py ->", v)
PY

echo "== $(date -u +%FT%TZ) removing editable pointers (those paths do not exist on Kaggle)"
rm -fv "$SITE"/__editable__*.pth "$SITE"/__editable___*_finder.py "$SITE"/_virtualenv.pth
rm -rf "$SITE"/arc_forge-*.dist-info "$SITE"/arcforge
cp -a "$VENV/bin/ninja" "$PKG/bin/ninja"

# ---- 4. overlay FA2 onto staging ---------------------------------------------------------
# Under ARC_QSA_NVFP4=1 the 0.5.20 QSA decode returns earlier and in theory never reaches FA2; on G4 both arms'
# venvs also passed smoke without FA2. But smoke is not arc-code; carry it until a real load proves otherwise —
# without FA2 the 0.5.20 fallback goes to flash_attn_4's cute, whose MLIR graph build always fails on sm_120.
echo "== $(date -u +%FT%TZ) FA2 overlay: $(basename "$FA2_WHEEL")"
"$VENV/bin/python" - "$FA2_WHEEL" "$SITE" <<'PY'
import pathlib
import sys
import zipfile

whl, site = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(whl) as z:
    names = z.namelist()
    assert any(n.startswith("flash_attn/__init__.py") for n in names), names[:5]
    # The v2.8.3 repo carries a snapshot of FA4's cute/; it must not overwrite flash_attn_4's own cute.
    keep = [n for n in names if not n.startswith("flash_attn/cute/")]
    # Staging's site-packages are hardlinks. extractall overwrites an existing file with
    # open(mode="wb"), which **writes through to the venv itself** — a production environment another
    # session is using. In theory FA2 and flash_attn_4 have no name clash (the latter has only cute/, not even
    # an __init__.py), but "no clash in theory" is not something to bet a shared venv on.
    shared = [n for n in keep
              if (site / n).is_file() and (site / n).stat().st_nlink > 1]
    if shared:
        raise SystemExit(f"✗ FA2 would overwrite {len(shared)} hardlinks, which would write through to the venv: {shared[:5]}")
    z.extractall(site, members=keep)
    for n in keep:
        if n.endswith(".so"):
            (site / n).chmod(0o755)
print(f"   overlaid {len(keep)} files (skipped {len(names) - len(keep)} flash_attn/cute/*)")
PY
FA2_SO=$(ls "$SITE"/flash_attn_2_cuda*.so | head -1)
ARCHS=$("$CUDA_SRC/bin/cuobjdump" -lelf "$FA2_SO" | grep -o "sm_[0-9a-z]*" | sort -u | tr '\n' ' ')
case "$ARCHS" in *sm_120*) echo "   FA2 cubins: $ARCHS" ;;
                 *) echo "✗ FA2 .so has no sm_120 cubin ($ARCHS)" >&2; exit 1 ;; esac

# ---- 5. official SM120 AOT (flashinfer_jit_cache package) --------------------------------
echo "== $(date -u +%FT%TZ) flashinfer_jit_cache SM120 subset"
"$VENV/bin/python" "$ARC_ROOT/work/qwen38fn_eval/install_flashinfer_jit_cache_sm120.py" \
  --runtime "$PKG" ${JIT_CACHE_WHEEL:+--wheel "$JIT_CACHE_WHEEL"}

# ---- 6. CUDA 13.1 toolchain subset -------------------------------------------------------
# JIT kernels are compiled with nvcc at run time and must emit sm_120a/sm_120f for sm_120 (nvcc >= 12.9).
echo "== $(date -u +%FT%TZ) CUDA 13.1 subset"
C=$PKG/cuda-13.1
mkdir -p "$C/bin" "$C/targets/x86_64-linux/lib"
for b in nvcc ptxas fatbinary nvlink cudafe++ bin2c nvcc.profile __nvcc_device_query; do
  cp -a "$CUDA_SRC/bin/$b" "$C/bin/$b"
done
cp -a "$CUDA_SRC/bin/crt" "$C/bin/crt"
cp -a "$CUDA_SRC/nvvm" "$C/nvvm"
cp -a "$CUDA_SRC/targets/x86_64-linux/include" "$C/targets/x86_64-linux/include"
cp -a "$CUDA_SRC/targets/x86_64-linux/lib/stubs" "$C/targets/x86_64-linux/lib/stubs"
cp -a "$CUDA_SRC"/targets/x86_64-linux/lib/libcudart.so* "$C/targets/x86_64-linux/lib/"
cp -a "$CUDA_SRC/targets/x86_64-linux/lib/libcudadevrt.a" "$C/targets/x86_64-linux/lib/"
[ -f "$CUDA_SRC/version.json" ] && cp -a "$CUDA_SRC/version.json" "$C/version.json"
ln -s targets/x86_64-linux/include "$C/include"
ln -s targets/x86_64-linux/lib "$C/lib64"
"$C/bin/nvcc" --version | tail -1

# ---- 7. self-check from staging (CPU only, no GPU) ---------------------------------------
echo "== $(date -u +%FT%TZ) self-check"
PYTHONPATH="$PKG/python:$SITE:$SITE/nvidia_cutlass_dsl/dsl_packages" \
  "$VENV/bin/python" - "$PKG" "$SITE" <<'PY'
import pathlib
import sys

pkg, site = sys.argv[1], sys.argv[2]
bad = [p.name for p in pathlib.Path(site).glob("arc_forge*")] + \
      [p.name for p in pathlib.Path(site).glob("__editable__*")]
if bad:
    raise SystemExit(f"✗ leftovers: {bad}")
for pth in pathlib.Path(site).glob("*.pth"):
    if "/dsv4/venv-0520" in pth.read_text(errors="ignore"):
        raise SystemExit(f"✗ {pth.name} still points to the local venv")

import sglang
assert sglang.__file__.startswith(pkg), sglang.__file__
assert sglang.__version__ == "0.5.20", sglang.__version__
import flashinfer
import flashinfer_jit_cache
from flashinfer.jit import env as jit_env
aot = pathlib.Path(jit_env.FLASHINFER_AOT_DIR)
assert aot.is_relative_to(pathlib.Path(site) / "flashinfer_jit_cache"), aot
assert (aot / "fused_moe_120/fused_moe_120.so").exists(), sorted(p.name for p in aot.iterdir())
import flash_attn
import flash_attn.cute.interface  # noqa: F401  FA2 and the FA4 cute must both be importable
assert flash_attn.__file__.startswith(site), flash_attn.__file__
print(f"   ✓ sglang {sglang.__version__} / flashinfer {flashinfer.__version__} / "
      f"jit_cache {flashinfer_jit_cache.__version__} / FA2 {flash_attn.__version__}")
print(f"   ✓ AOT dir = {aot}")
PY

# ---- 8. provenance: what the notebook asserts against ------------------------------------
echo "== $(date -u +%FT%TZ) provenance"
PYTHONPATH="$ARC_ROOT/work/qwen38fn_eval" "$VENV/bin/python" - \
    "$PKG" "$SRC" "$ARC_ROOT" "$PATCHES" "$PYVER" "$DATASET_SLUG" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

import sglang_0520_markers as gate

pkg, src, arc_root, patches, pyver, slug = sys.argv[1:7]
pkg, src, patches = pathlib.Path(pkg), pathlib.Path(src), pathlib.Path(patches)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = {
    "line": "sglang-0.5.20",
    "base_ref": subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "arc_forge_commit": subprocess.run(["git", "-C", arc_root, "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
    "patches": {p.name: sha(p) for p in sorted(patches.glob("*.patch"))},
    # Source hashes after applying: this is what the notebook compares, not the patch files themselves.
    "source_sha256": {rel: sha(pkg / "python" / pathlib.Path(rel).relative_to("python"))
                      for rel in sorted({**gate.MARKERS, **gate.SM120_MARKERS})},
    "kaggle_dataset": slug,
    "python": pyver,
    "layout": {"site_packages": f"lib/{pyver}/site-packages", "sglang_source": "python",
               "bin": "bin", "cuda": "cuda-13.1"},
    "aot": json.loads((pkg / "flashinfer-jit-cache-sm120.json").read_text()),
    "source_tree": "rebuilt from tag v0.5.20 by package_runtime_sm120_0520.sh; "
                   "NOT $HOME/work/sglang-0520, which carries uncommitted work",
}
(pkg / "runtime-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"   ✓ {len(record['patches'])} patches, {len(record['source_sha256'])} source hashes")
PY

du -sh "$PKG"/* | sed 's/^/   /'

# ---- 9. compress + upload directory ------------------------------------------------------
ARCHIVE=$ARCHIVE_DIR/$NAME.tar.zst
echo "== $(date -u +%FT%TZ) compress -> $ARCHIVE (zstd -10 -T0)"
tar -C "$STAGE" -cf - package | zstd -10 -T0 -o "$ARCHIVE" -f
SHA=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
SIZE=$(stat -c%s "$ARCHIVE")

UP=$ARCHIVE_DIR/upload-$NAME
mkdir -p "$UP"
ln -f "$ARCHIVE" "$UP/$NAME.tar.zst" 2>/dev/null || cp -f "$ARCHIVE" "$UP/$NAME.tar.zst"
cp -f "$PKG/runtime-provenance.json" "$UP/"
cat > "$UP/dataset-metadata.json" <<JSON
{"title": "qwen38fn sglang 0520 sm120 cp312 runtime", "id": "$DATASET_SLUG", "licenses": [{"name": "Apache 2.0"}]}
JSON
cat > "$ARCHIVE_DIR/$NAME.manifest.json" <<JSON
{"name": "$NAME", "archive": "$NAME.tar.zst", "bytes": $SIZE, "sha256": "$SHA",
 "kaggle_dataset": "$DATASET_SLUG", "sglang_commit": "$SGL_COMMIT", "python": "$PYVER"}
JSON

echo
echo "wrote $ARCHIVE  ($(numfmt --to=iec "$SIZE"))"
echo "  sha256 $SHA"
echo "upload dir: $UP   →  uvx --from kaggle kaggle datasets create -p $UP"
echo "PACKAGE_RUNTIME_SM120_0520_OK"
