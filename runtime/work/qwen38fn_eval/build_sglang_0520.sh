#!/bin/bash
# Builds the **0.5.20 main-line** venv for Qwen3.8-Flash-Next: sglang tag v0.5.20 + patches/sglang-0.5.20/.
#
#   bash work/qwen38fn_eval/build_sglang_0520.sh
#
# Sibling of build_sglang_qwen4.sh: that one is pinned to the PR #36497 snapshot (frozen), this one to the
# tagged release. The two lines have independent source checkouts and venvs and don't affect each other:
#   old line  $HOME/work/sglang-qwen4  →  $HOME/work/dsv4/venv-qwen4   (patches/sglang-qwen4-pr36497/, 44 patches)
#   new line  $HOME/work/sglang-0520   →  $HOME/work/dsv4/venv-0520    (patches/sglang-0.5.20/, 29 patches)
#
# Why a source checkout and not a pip wheel: patches touch kernels/jit/csrc/*.cuh (JIT at run time,
# no wheel rebuild needed), the marker gate reads the source, and BASE's git-sha comparison needs .git.
#
# Differences from the old line, each with its source (docs/research/sglang_0520_upgrade.md):
#   - the initial forward port went from 36→23 patches (later grown to 44/29): upstream absorbed 4 (0004/0013/0015/0016), the rest are research instruments or discontinued lines;
#     the markers below for those four "absorbed upstream" assert that **the upstream behavior exists** — pointing at an old checkout stops right there.
#   - 0011's .cuh hunk merges "cleanly" three-way, but upstream deleted the null check on row_starts; restored, and pinned by a marker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

SRC=${SRC:-$HOME/work/sglang-0520}
VENV=${VENV:-$HOME/work/dsv4/venv-0520}
TAG=${TAG:-v0.5.20}
PATCH_DIR=$PWD/patches/sglang-0.5.20

# The base sha is read from BASE, not copied into the script again — two copies drift.
REF=$(sed -n 's/^REF=//p' "$PATCH_DIR/BASE")
[ -n "$REF" ] || { echo "✗ $PATCH_DIR/BASE has no REF=" >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" remote add origin https://github.com/sgl-project/sglang.git
fi
git -C "$SRC" fetch --depth 1 origin "tag" "$TAG" --no-tags
git -C "$SRC" checkout -q "$TAG"
head_sha=$(git -C "$SRC" rev-parse HEAD)
case "$head_sha" in "$REF"*) ;; *) echo "✗ $TAG resolves to $head_sha but BASE says $REF. Was the tag moved? Stopping." >&2; exit 1;; esac

# A fresh checkout gets the whole patch stack applied in order. On a rerun where the source already has local changes,
# there is no reset/overwrite; the full marker gate below verifies that every required fix is present instead.
patches=("$PATCH_DIR"/*.patch)
[ -f "${patches[0]}" ] || { echo "✗ no $PATCH_DIR/*.patch found" >&2; exit 1; }
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
# No Rust extension: this machine has no cargo, and this HTTP path doesn't need it (reason in build_sglang_qwen4.sh).
SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV="$VENV" \
    uv pip install --python "$VENV/bin/python" -e "$SRC/python"

# --- CUDA toolkit pin (same as the old line): nvcc/crt/nvrtc/nvvm/cccl must match cuda-runtime's major.minor,
# otherwise tilelang JIT dies with "CUDA compiler and CUDA toolkit headers are incompatible".
rt_mm=$("$VENV/bin/python" -c "from importlib.metadata import version; print('.'.join(version('nvidia-cuda-runtime').split('.')[:2]))")
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    "nvidia-cuda-nvcc==$rt_mm.*" "nvidia-cuda-crt==$rt_mm.*" \
    "nvidia-cuda-nvrtc==$rt_mm.*" "nvidia-nvvm==$rt_mm.*" "nvidia-cuda-cccl==$rt_mm.*"

# --- static checks before going live: none of these errors is reported at start-up ---
"$VENV/bin/python" - "$REF" <<'PY'
import importlib.util, pathlib, subprocess, sys, torch
from importlib.metadata import version, PackageNotFoundError
ok = True
ref = sys.argv[1]

# 0. the installed sglang really is BASE's commit (editable install → trace back to the source's git HEAD)
repo = pathlib.Path(__import__("sglang").__file__).resolve().parents[2]
head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
if not head.startswith(ref):
    print(f"✗ installed sglang source HEAD={head[:12]} ≠ BASE REF={ref[:12]}"); ok = False

# 1. qwen4_exp is present (native in v0.5.20)
for mod in ("sglang.srt.models.qwen4_exp", "sglang.srt.configs.qwen4_exp",
            "sglang.srt.layers.attention.qsa.config", "sglang.srt.models.qwen4_exp_ple_table"):
    if importlib.util.find_spec(mod) is None:
        print(f"✗ missing {mod} -- this is not v0.5.20"); ok = False
from sglang.srt.environ import envs
if envs.SGLANG_RUST_SERVER.get():
    print("✗ SGLANG_RUST_SERVER is on, but this venv has no _server extension"); ok = False

# 2. every CUDA toolkit piece at the same version
rt = version("nvidia-cuda-runtime")
mm = lambda v: ".".join(v.split(".")[:2])
for pkg in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-cuda-nvrtc", "nvidia-nvvm", "nvidia-cuda-cccl"):
    try:
        v = version(pkg)
    except PackageNotFoundError:
        continue
    if mm(v) != mm(rt):
        print(f"✗ {pkg} {v} ≠ cuda-runtime {rt}; pin it back to the same version: uv pip install '{pkg}=={mm(rt)}.*'"); ok = False

# 3. the patch stack must not silently miss a piece. dict keys are files; **markers for the same file go in one tuple** — a duplicate dict key silently overwrites the earlier entry.
#    The last 4 entries are the four "absorbed upstream, no longer carried locally": they assert the upstream behavior exists, so an old checkout is stopped.
markers = {
    "python/sglang/srt/layers/attention/qsa/mqa.py": (
        "groups_per_cta = 1 if rows <= 8 else 2 if rows <= 24 else 4",   # 0001 (old 0006)
        "_grouped_qsa_effective_min_rows",                                  # 0002 (old 0007)
        "_tilelang_qsa_mqa_grouped_verify_kernel",
        "_qsa_verify_group_size",                                          # 0025
        "if use_grouped_verify and (grouped_size * query_heads) % 8 == 0:",  # 0026
        "ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM",                                # 0027
        "_GROUPED_QSA_MIN_ROWS_SCALE = {2: 2.0, 4: 1.0, 8: 2.0}"),          # 0028
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py": ("_qsa_grouped_queries_per_request",),
    "python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "_fused_mamba_state_scatter_with_mask_strided_kernel",),             # 0003 (old 0008)
    "python/sglang/kernels/ops/speculative/topk1.py": ("draft_topk1_postprocess_distributed",),  # 0004 (old 0009)
    "python/sglang/kernels/ops/elementwise/fast_topk.py": ("fast_topk_zero_starts",),           # 0005 (old 0011)
    # 0005 passes row_starts=nullptr; v0.5.20 deleted the null check in the kernel and the patch restores it. Without it every decode step hits an illegal address.
    "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh": ("params.row_starts == nullptr",),
    "python/sglang/srt/models/qwen4_exp.py": ("_qwen4_mnnvl_hc_precompute_gate",),            # 0006 (old 0012)
    "python/sglang/srt/layers/flashinfer_comm_fusion.py": ("can_use_qwen4_mnnvl_hc",),
    "python/sglang/kernels/ops/communication/qwen4_mnnvl_hc.py": ("compile_qwen4_mnnvl_hc",),
    "python/sglang/kernels/ops/elementwise/hc_gate.py": ("compile_hc_gate",),
    "python/sglang/srt/speculative/eagle_worker_v2.py": ("_bind_native_qwen_mtp_before_pool",),  # 0007 (old 0014)
    # --- 0008–0014: NVFP4 target-KV arm (inert when ARC_QSA_NVFP4 is unset; the markers must still be there) ---
    "python/sglang/srt/layers/attention/flashinfer_backend.py": (
        "ARC_FULL_NVFP4_PREFILL=fa2_packed", "arc_nvfp4_paged_forward"),                  # 0008/0010 (old 0023/0025)
    "python/sglang/srt/model_executor/pool_configurator.py": (
        "Native packed prefill allocates no pool-sized dequant workspace",),                # 0009 (old 0024)
    "python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py": ("_arc_native_fp4_prefill",),  # 0011 (old 0029)
    "python/sglang/srt/layers/attention/qsa/sparse_attn.py": (
        "qwen_sparse_nvfp4_expand_triton",                                                  # 0012 (old 0030)
        "DST_V_GROUP",                                                                      # 0014 (old 0035)
        "_compact_kv_nvfp4_four_plane",                                                     # 0024
        # --- absorbed upstream (old 0004/0015): assert the upstream behavior exists ---
        "ZERO_FILL", "keys = keys.to(q_values.dtype)", "slots.to(tl.int64)"),
    # 0013 (old 0034): in v0.5.20 the field moved to arg_groups/fields/spec.py, no longer server_args.py
    "python/sglang/srt/arg_groups/fields/spec.py": ('"bfloat16", "nvfp4"',),
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": (
        "_trtllm_sparse_nvfp4_sm100",                                                       # 0014
        "qwen_sparse_nvfp4_gather4_triton",                                                 # 0024
        "zero_fill_cols=stride"),                                                           # upstream (old 0015)
    # --- 0016: commit of the PLE side state after ReplaySSM verify (re-cut of old 0018) ---
    "python/sglang/srt/speculative/spec_utils.py": (
        "_commit_ple_side_states_after_replayssm(",  # 0016
    ),
    "python/sglang/srt/entrypoints/openai/serving_responses.py": (
        "ARC_0017_MERGE_ASSISTANT_TURN",  # 0017
        "ARC_0018_TOOL_OUTPUT_MEDIA",     # 0018
    ),
    # --- absorbed upstream (old 0013/0016) ---
    "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py": (
        "out.hidden_states[:raw_bs].clone()",),                                                # old 0013 (PR #35546)
    "python/sglang/srt/models/qwen4_exp_mtp.py": ("hc_hidden_states = hc_hidden_states[select_index]",),  # old 0016
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
print(f"torch {torch.__version__} / sglang {version('sglang')} @ {head[:12]} / "
      f"sglang-kernel {version('sglang-kernel')} / flashinfer {version('flashinfer-python')}")
sys.exit(0 if ok else 1)
PY
"$VENV/bin/python" tools/verify_nvfp4_graph_scale.py "$SRC"
echo "✓ $VENV"
