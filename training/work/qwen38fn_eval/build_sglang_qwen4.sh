#!/bin/bash
# Build a **separate** sglang venv with PR #36497 installed (Qwen3.8-Flash-Next / qwen4_exp model support).
#
#   bash work/qwen38fn_eval/build_sglang_qwen4.sh
#
# Why $HOME/work/dsv4/venv cannot be upgraded in place:
#   the PR needs torch 2.13.0 / sglang-kernel 0.4.6.post1 / flashinfer 0.6.17,
#   the existing venv is 2.11.0 / 0.4.5 / 0.6.15.post1 and carries the four in-place modifications of
#   patches/sglang-0.5.17/, which dsv4 and Qwen3.5/3.6/3.8-27B all depend on. Upgrading it would tear down those lines.
#
# Why not the official docker image (lmsysorg/sglang:qwen38flashnext): this machine has no docker group.
#
# The new qwen4_exp kernels are all in python/sglang/kernels/jit/csrc/, JIT-compiled at run time,
# no wheel rebuild needed -- so `uv pip install -e python` is enough.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

SRC=${SRC:-$HOME/work/sglang-qwen4}
VENV=${VENV:-$HOME/work/dsv4/venv-qwen4}
PR=${PR:-36497}
# The verified version the cookbook names; changing the sha means rerunning the pre-deployment probes.
PR_SHA=${PR_SHA:-73a255206f}

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" remote add origin https://github.com/sgl-project/sglang.git
fi
git -C "$SRC" fetch --depth 1 origin "pull/$PR/head"
git -C "$SRC" checkout -q FETCH_HEAD
head_sha=$(git -C "$SRC" rev-parse HEAD)
case "$head_sha" in "$PR_SHA"*) ;; *) echo "✗ head of PR #$PR changed: $head_sha ≠ $PR_SHA. Rerun the probe, then update PR_SHA" >&2; exit 1;; esac

# A fresh checkout applies this unit's full patch stack in order. On a rerun, if the source already
# has local changes, there is no reset / overwrite; the marker gate below verifies that the required fixes are all there.
PATCH_DIR=$PWD/patches/sglang-qwen4-pr36497
patches=("$PATCH_DIR"/*.patch)
if [ ! -f "${patches[0]}" ]; then
    echo "✗ no $PATCH_DIR/*.patch found" >&2
    exit 1
fi
if git -C "$SRC" diff --quiet && git -C "$SRC" diff --cached --quiet; then
    for patch in "${patches[@]}"; do
        git -C "$SRC" apply --check "$patch"
        git -C "$SRC" apply "$patch"
        echo "· apply $(basename "$patch")"
    done
else
    echo "· $SRC already has local changes; not overwriting, leaving verification to the full marker gate"
fi

[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# **No Rust extension**: this machine has no cargo, and this path uses none of the three crates' consumers --
#   _server      → only the alternative HTTP server of SGLANG_RUST_SERVER (environ.py default False)
#   _grpc        → only the gRPC entrypoint
#   _multimodal  → only the image preprocessing of the Inkling model family
# The default sglang.launch_server HTTP path is pure Python. If something really needs one,
# `load_rust_extension` calls cargo and fails loudly (not a silent downgrade),
# and serve_qwen38fn.sh pins SGLANG_RUST_BUILD_MODE to never,
# so "silently compiling for ten minutes in the middle of a run" cannot happen.
SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV="$VENV" \
    uv pip install --python "$VENV/bin/python" -e "$SRC/python"

# --- CUDA toolkit pin ---
# sglang's dependencies do not pin CUDA: torch brings the cu130 runtime, but nvcc / crt / nvvm / cccl resolve to
# 13.3, and tilelang JIT dies with "CUDA compiler and CUDA toolkit headers are incompatible".
# env/README.md records this; the dsv4 venv is pinned the same way. The check below is a hard gate.
rt_mm=$("$VENV/bin/python" -c "from importlib.metadata import version; print('.'.join(version('nvidia-cuda-runtime').split('.')[:2]))")
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    "nvidia-cuda-nvcc==$rt_mm.*" "nvidia-cuda-crt==$rt_mm.*" \
    "nvidia-cuda-nvrtc==$rt_mm.*" "nvidia-nvvm==$rt_mm.*" "nvidia-cuda-cccl==$rt_mm.*"

# --- static checks before going live: none of these three errors would show up at start-up ---
"$VENV/bin/python" - <<'PY'
import importlib.util, sys, torch
ok = True
# 1. qwen4_exp really got installed (not the tagged release from PyPI)
for mod in ("sglang.srt.models.qwen4_exp", "sglang.srt.configs.qwen4_exp",
            "sglang.srt.layers.attention.qsa.config"):
    if importlib.util.find_spec(mod) is None:
        print(f"✗ missing {mod} -- the installed tree is not PR #36497"); ok = False
# 1b. The Rust extension is deliberately not built, on the premise that the alternative server is not turned on.
#     If the default ever flips, it must show up right here, not be dug out of the start-up log.
from sglang.srt.environ import envs
if envs.SGLANG_RUST_SERVER.get():
    print("✗ SGLANG_RUST_SERVER is on, but this venv has no _server extension"); ok = False
# 2. the CUDA toolkit components must match torch's runtime version, otherwise tilelang JIT dies with
#    "CUDA compiler and CUDA toolkit headers are incompatible" (env/README.md)
from importlib.metadata import version, PackageNotFoundError
rt = version("nvidia-cuda-runtime")
mm = lambda v: ".".join(v.split(".")[:2])
for pkg in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-cuda-nvrtc", "nvidia-nvvm", "nvidia-cuda-cccl"):
    try:
        v = version(pkg)
    except PackageNotFoundError:
        continue
    if mm(v) != mm(rt):
        print(f"✗ {pkg} {v} ≠ cuda-runtime {rt}; pin it back to the same version: "
              f"uv pip install '{pkg}=={mm(rt)}.*'"); ok = False

# 3. the patch stack must not silently lose pieces. These markers each correspond to the main production path of
# 0001…0017; experimental switches may be off, but the source must be the same tree we verified.
import pathlib
repo = pathlib.Path(__import__("sglang").__file__).resolve().parents[2]
markers = {
    "python/sglang/srt/models/qwen4_exp.py": (
        "ARC_ABLATE_PLE", "ARC_PLE_HOT", "_qwen4_mnnvl_hc_precompute_gate"),
    "python/sglang/srt/layers/moe/moe_runner/runner.py": ("ARC_SALIENCY_HOOK_PY",),
    "python/sglang/srt/layers/moe/fused_moe_triton/layer.py": (
        "arc-forge saliency: FusedMoE.layer_name",),
    "python/sglang/srt/layers/attention/qsa/sparse_attn.py": (
        "_validate_sparse_gqa_dtypes", "_sparse_gqa_decode_selected",
        "ZERO_PAGE_TAIL"),
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": (
        "row_valid_counts=valid_counts",),
    "python/sglang/srt/layers/attention/qsa/mqa.py": (
        "groups_per_cta = 1 if rows <= 8 else 2 if rows <= 24 else 4",
        "_tilelang_qsa_mqa_grouped_verify_kernel",
        "_grouped_qsa_effective_min_rows",
    ),
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py": (
        "_qsa_grouped_queries_per_request",),
    "python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "_fused_mamba_state_scatter_with_mask_strided_kernel",),
    "python/sglang/kernels/ops/speculative/topk1.py": (
        "draft_topk1_postprocess_distributed",),
    "python/sglang/srt/models/qwen2_moe.py": ("_SGLANG_QWEN_MOE_ROUTED_MAIN",),
    "python/sglang/kernels/ops/elementwise/fast_topk.py": (
        "fast_topk_zero_starts",),
    "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py": (
        "use_selected_rows",),
    "python/sglang/srt/speculative/eagle_worker_v2.py": (
        "_bind_native_qwen_mtp_before_pool",),
    "python/sglang/srt/layers/logits_processor.py": (
        "keep_bf16_input_logprobs",),
    "python/sglang/srt/layers/logprob_processor.py": (
        "ARC_TOPK_NUMPY_ROWS",),
    "python/sglang/srt/managers/io_struct.py": (
        "row is not None and len(row) != 0",),
}
for rel, needles in markers.items():
    path = repo / rel
    try:
        text = path.read_text()
    except OSError as exc:
        print(f"✗ patch marker file missing: {path}: {exc}"); ok = False
        continue
    for needle in needles:
        if needle not in text:
            print(f"✗ patch stack incomplete: {rel} lacks {needle!r}"); ok = False
print(f"torch {torch.__version__} / sglang {version('sglang')} / "
      f"sglang-kernel {version('sglang-kernel')} / flashinfer {version('flashinfer-python')}")
sys.exit(0 if ok else 1)
PY
echo "✓ $VENV"
