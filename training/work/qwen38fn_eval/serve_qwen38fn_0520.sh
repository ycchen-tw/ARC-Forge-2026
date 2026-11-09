#!/bin/bash
# Serve Qwen/Qwen3.8-Flash-Next-FP8 with SGLang **v0.5.20 mainline** for ARC-AGI-2 TIR eval, on an 8×B200 node.
# Sister of serve_qwen38fn.sh (old line: PR #36497 snapshot + 36 patches, $HOME/work/dsv4/venv-qwen4);
# the two can run at the same time: different venv, different source, **different default port** (this one 30032,
# the old line 30031); the same served-model-name (qwen38fn), so the harness settings need no change.
#
#   GPUS=4,5,6,7 bash work/qwen38fn_eval/serve_qwen38fn_0520.sh      # MTP on (default)
#   SPEC=none GPUS=... bash work/qwen38fn_eval/serve_qwen38fn_0520.sh # MTP off
#   SPEC=dflash DFLASH_DIR=<dense drafter deploy dir> GPUS=... bash ...  # the fork's DFlash drafter (needs the sglang-arc-2 fork)
#
# Differences from the old-line script (each in docs/research/sglang_0520_upgrade.md):
#   - venv → $HOME/work/dsv4/venv-0520, built by build_sglang_0520.sh; patches/sglang-0.5.20/ has only 7.
#   - removed four gate sections and their flags: ABLATE_PLE/PLE_HOT (research instruments 0001/0003), QSA_DIRECT
#     (experiment 0005), EAGLE_SELECTED_ROWS (0013/0016 absorbed upstream), SALIENCY (0002/0022b; REAP calibration
#     stays on the old line).
#   - PYCHK4 (fp8 KV) now asserts that the **upstream** ZERO_FILL / int64 offset exist (old 0004/0015 deleted).
#   - new PYCHK15: the upstream-absorbed 0013/0016 behavior must be present -- pointing at the old venv is stopped on
#     the spot.
#   - upstream's TRT paged-decode scratch went from FP8 to BF16 (~+0.5 GB @256 rows); KV pool numbers need remeasuring.
#
# Everything else (GPU occupancy check, cookbook parameters, MTP draft source, Mamba slots, KV dtype resolution) is
# verbatim the same as the old line; the nine sourced reasons are at the top of serve_qwen38fn.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

# The experimental TP1 supervisor owns this process-local allocation policy.
# node2's machine env file resets the CUDA alias; restore both spellings before any Python.
if [ "${ARC_STAGE_DEPLOYMENT_PROFILE:-}" = tp1-reap256-nvfp4 ]; then
    : "${ARC_STAGE_ALLOCATOR_CONF:?TP1 stage requires its explicit allocator policy}"
    export PYTORCH_ALLOC_CONF="$ARC_STAGE_ALLOCATOR_CONF"
    export PYTORCH_CUDA_ALLOC_CONF="$ARC_STAGE_ALLOCATOR_CONF"
fi

# GPU allocation. **This is a shared machine**; the sister script qwen38_eval/serve_qwen38.sh uses another set.
#
# This used to be a hard-coded whitelist (only 0-3). It is now a **dynamic check for other people's processes on the
# target GPUs**: a whitelist stops nobody from putting work on someone else's GPUs (it only restrains itself), and
# allocations change anyway; what really needs guarding is "GPUs carrying someone else's work at start-up time".
# The 2026-08-28 prompt ablation batch got 4-7 while another session's Qwen3.6 SFT was running on 0-3.
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=$GPUS
if [ "${SKIP_GPU_CHECK:-0}" != 1 ]; then
    # **Must not fail open**: `|| true` would turn "nvidia-smi died" into "nobody uses these GPUs",
    # exactly what the removed whitelist used to guard against. If it cannot be queried, stop right here.
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) \
        || { echo "✗ nvidia-smi cannot report GPU occupancy, so there is no telling whether the GPUs are in use by others -- stopping." >&2; exit 1; }
    for g in ${GPUS//,/ }; do
        uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
        while IFS=, read -r u pid; do
            [ "${u// /}" = "$uuid" ] || continue
            pid=${pid// /}
            stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
            # **Zombies do not count as occupying.** A server just killed leaves its scheduler child processes in
            # nvidia-smi's list for a few more seconds (the parent has not reaped them yet), state Z/<defunct>,
            # but the GPU memory is already released. Without excluding them, "stop the old server → start a new
            # one" gets blocked by its own leftovers, with an error message identical to "this is someone else's
            # work". Hit on 2026-08-28.
            case "$stat" in
                Z*) echo "· pid $pid on GPU $g is a zombie (just exited, not yet reaped); skipping" >&2; continue;;
                "") echo "· pid $pid on GPU $g no longer exists; skipping" >&2; continue;;
            esac
            owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
            cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-90)
            echo "✗ refused: GPU $g already has a process (pid $pid, owner ${owner:-?})" >&2
            echo "    $cmd" >&2
            echo "  This is a shared machine. To share the GPU anyway set SKIP_GPU_CHECK=1, but first make sure that process does not belong to someone else." >&2
            exit 1
        done <<< "$busy"
    done
fi
TP=${TP:-4}
EP=${EP:-4}
# DP (default 1): sglang --dp-size, each DP rank gets its own set of TP GPUs (INT3 TP1 on two GPUs = DP 2); the caller
# pins one session to one rank with routed_dp_rank (run_arc2's --dp-size). No --enable-dp-attention; ranks do not
# share KV.
DP=${DP:-1}
case "$DP" in ""|*[!0-9]*|0) echo "✗ DP must be a positive integer, got '$DP'" >&2; exit 1;; esac
DP_ARGS=()
[ "$DP" -eq 1 ] || DP_ARGS=(--dp-size "$DP")
# With the hard-coded 0-3 whitelist gone, GPUS is free-form while TP/EP still default to 4 --
# `GPUS=4,5` would pass the occupancy check and then run --tp 4 on 2 GPUs. The count has to match: TP × DP.
n_gpus=$(printf '%s' "${GPUS//,/ }" | wc -w)
[ "$n_gpus" -eq $((TP * DP)) ] || { echo "✗ GPUS lists $n_gpus GPUs ($GPUS) but TP=$TP × DP=$DP; they do not match" >&2; exit 1; }

SGLANG_VENV=${SGLANG_VENV:-$HOME/work/dsv4/venv-0520}
[ -x "$SGLANG_VENV/bin/python" ] || { echo "✗ no $SGLANG_VENV; run build_sglang_0520.sh first" >&2; exit 1; }
# flashinfer's runtime JIT shells out to find ninja; when python is called by absolute path, PATH does not carry it.
export PATH="$SGLANG_VENV/bin:$PATH"
# The venv has no arcforge, so the mrope plugin is not loaded; set anyway so that a venv switch does not trip on it.
export ARC_MROPE_ENABLED=0
# This venv deliberately has no Rust extension (see build_sglang_0520.sh). `never` makes any path that
# wants it error out on the spot instead of shelling out to cargo build in the middle of start-up.
export SGLANG_RUST_SERVER=0 SGLANG_RUST_BUILD_MODE=never

# ── decoding libraries for video input ───────────────────────────────────────
# sglang decodes video with torchcodec, whose bundled libtorchcodec_core*.so link against the system
# libav*. This machine has no FFmpeg, so torchcodec **is installed but cannot load** (decord is not installed either)
# -- without this section the symptom is requests blowing up only at decode, and the message never says "FFmpeg
# missing". Text-only serving does not need it, so a missing one only warns; when present it is checked right here,
# rather than discovering a wrong version at the first video request (torchcodec only takes libavutil.so.56–60).
FFMPEG_DIR=${FFMPEG_DIR:-$HOME/work/dsv4/ffmpeg-n8.1}
if [ -d "$FFMPEG_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
    "$SGLANG_VENV/bin/python" - <<'PYCHKFFMPEG' || exit 1
import sys
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
except Exception as exc:
    print(f"✗ FFMPEG_DIR exists but torchcodec fails to load: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
    print("  Wrong version? torchcodec only accepts libavutil.so.56-60 (FFmpeg 4-8).", file=sys.stderr)
    print("  Rerun bash work/qwen38fn_eval/setup_ffmpeg.sh", file=sys.stderr)
    sys.exit(1)
PYCHKFFMPEG
else
    echo "· no $FFMPEG_DIR: video input will fail at decode (bash work/qwen38fn_eval/setup_ffmpeg.sh)" >&2
fi




# The CUDA graph of the QSA index score launches at a fixed size for the 262k context cap; short contexts
# leave many CTAs doing only bounds checks. 0 = choose gpc 1/2/4 automatically by rows, keeping small-batch
# low latency while shrinking the empty grid of large batches. QSA_MQA_GPC=1 restores the baseline.
export SGLANG_QSA_MQA_GROUPS_PER_CTA=${QSA_MQA_GPC:-0}
"$SGLANG_VENV/bin/python" - <<'PYCHK6' || exit 1
import importlib.util, pathlib, sys
src = pathlib.Path(importlib.util.find_spec(
    "sglang.srt.layers.attention.qsa.mqa").origin).read_text()
if "groups_per_cta = 1 if rows <= 8 else 2 if rows <= 24 else 4" not in src:
    sys.exit("✗ mqa.py lacks the arc-forge QSA auto-gpc patch: "
             "cd $HOME/work/sglang-0520 && git apply "
             "<arc-forge>/patches/sglang-0.5.20/0001-qsa-mqa-auto-gpc.patch")
PYCHK6
echo "· QSA MQA groups/CTA: ${SGLANG_QSA_MQA_GROUPS_PER_CTA} (0=auto 1/2/4)"

# TARGET_VERIFY: the four rows of a request share a K tile; QSA_GROUPED_VERIFY=0 restores.
export SGLANG_QSA_GROUPED_VERIFY=${QSA_GROUPED_VERIFY:-1}
export SGLANG_QSA_GROUPED_VERIFY_MIN_ROWS=${QSA_GROUPED_VERIFY_MIN_ROWS:-64}

# Greedy MTP all-gathers only each rank's (max, global-id); 0 restores the full vocab.
export SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1=${MTP_DISTRIBUTED_TOPK1:-1}
echo "· grouped QSA verify: $SGLANG_QSA_GROUPED_VERIFY (min rows=$SGLANG_QSA_GROUPED_VERIFY_MIN_ROWS)"
echo "· distributed MTP top-1: $SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1"

# Default-on/unconditional patches must fail loud after an editable reinstall.
"$SGLANG_VENV/bin/python" - <<'PYCHK14' || exit 1
import importlib.util
import pathlib
import sys

required = {
    "sglang.srt.layers.attention.qsa.mqa": "_grouped_qsa_effective_min_rows",
    "sglang.srt.layers.attention.qsa.qsa_indexer": "_qsa_grouped_queries_per_request",
    "sglang.kernels.ops.mamba.mamba_state_scatter_triton": "_fused_mamba_state_scatter_with_mask_strided_kernel",
    "sglang.kernels.ops.speculative.topk1": "draft_topk1_postprocess_distributed",
    "sglang.kernels.ops.elementwise.fast_topk": "fast_topk_zero_starts",
    "sglang.srt.speculative.eagle_worker_v2": "_bind_native_qwen_mtp_before_pool",
}
for module, marker in required.items():
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        sys.exit(f"✗ production patch stack is missing module: {module}")
    if marker not in pathlib.Path(spec.origin).read_text():
        sys.exit(f"✗ production patch stack incomplete: {module} lacks {marker}")
PYCHK14

# 0016: with GDN + --enable-linear-replayssm-spec, upstream's ReplaySSM ring branch returns right after verify,
# skipping attn_backend.update_mamba_state_after_mtp_verify -- PLE's short-conv / n-gram side state then stays forever
# where it was before the verify window, drifting without an error. Defects of this kind are completely silent, hence a
# hard gate.
"$SGLANG_VENV/bin/python" - <<'PYCHK16' || exit 1
import importlib.util
import pathlib
import sys

src = pathlib.Path(
    importlib.util.find_spec("sglang.srt.speculative.spec_utils").origin
).read_text()
if "_commit_ple_side_states_after_replayssm(" not in src or src.count(
    "_commit_ple_side_states_after_replayssm("
) < 2:
    sys.exit(
        "✗ spec_utils lacks 0016 (commit of the PLE side state after ReplaySSM verify); "
        "with --enable-linear-replayssm-spec the PLE state drifts silently"
    )
PYCHK16

# 0017: 0.5.20's Responses API splits one assistant turn's reasoning / message / function_call by ``phase`` into
# three rendered assistant blocks (two of them empty <think>\n\n</think>\n\n followed by <|im_end|>), and Qwen3.8
# imitates that shape: after ~100 reasoning tokens it emits \n\n and ends the turn without a tool call (arc-code: 8 of 25
# games judged as giving up within 90 seconds). Silent, hence a hard gate.
"$SGLANG_VENV/bin/python" - <<'PYCHK17' || exit 1
import importlib.util
import pathlib
import sys

src = pathlib.Path(
    importlib.util.find_spec("sglang.srt.entrypoints.openai.serving_responses").origin
).read_text()
if "ARC_0017_MERGE_ASSISTANT_TURN" not in src or (
    'merged[-1].get("phase") == msg.get("phase")' in src
):
    sys.exit(
        "✗ serving_responses lacks 0017 (assistant items of one turn must be merged into one block, not split by phase); "
        "with Responses clients such as codex the model ends its turn after ~100 tokens"
    )
PYCHK17

# The old line's 0013 (EAGLE draft-extend selected rows, backport of upstream PR #35546) and 0016 (the fix for the
# hidden-row defect it introduced) are both absorbed upstream in v0.5.20, and upstream does one extra thing for each:
# `.clone()` so the graph output is no longer a view overwritten by the next step; the root cause is fixed by the model
# wrapper selecting the right row. Both strings match exactly once in the whole v0.5.20 repo and are absent from the
# old snapshot -- so this section doubles as a "wrong venv" detector.
# The defect itself is completely silent (accepted tokens are decided by the target, only accept_len drops), so it
# must be gated.
"$SGLANG_VENV/bin/python" - <<'PYCHK15' || exit 1
import importlib.util
import pathlib
import sys

required = [
    ("sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner", "out.hidden_states[:raw_bs].clone()"),
    ("sglang.srt.layers.logits_processor", "draft_extend_select_index"),
    ("sglang.srt.models.qwen4_exp_mtp", "hc_hidden_states = hc_hidden_states[select_index]"),
]
for module, marker in required:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        sys.exit(f"✗ missing module: {module}")
    if marker not in pathlib.Path(spec.origin).read_text():
        sys.exit(f"✗ upstream draft-extend selected-rows behavior absent: {module} lacks {marker} (source not v0.5.20?)")
PYCHK15


# Qwen4 attention TP all-reduce + hyper-connection apply epilogue. Default on:
# the production route is restricted to TP4 B200 BF16, H=2560, HC=4 and
# decode/target-verify rows <=512. Every other shape/backend uses the original
# all-reduce + combine path. MNNVL_HC=0 restores baseline; enabled mode compiles
# before server launch so CUDA-graph capture never becomes the first compile.
export SGLANG_QWEN4_MNNVL_HC=${MNNVL_HC:-1}
# Keep the current production split path by default. HC_COMBINE_SPLIT=0 is only
# an isolated-kernel experiment; paired production traces did not show an e2e win.
export SGLANG_HC_COMBINE_SPLIT=${HC_COMBINE_SPLIT:-1}
if [ "$SGLANG_QWEN4_MNNVL_HC" = 1 ]; then
    [ "$TP" -eq 4 ] || { echo "✗ MNNVL_HC supports only TP4" >&2; exit 1; }
    "$SGLANG_VENV/bin/python" - <<'PYCHK9' || exit 1
import importlib.util
import pathlib
import sys
import torch

required = {
    "sglang.srt.models.qwen4_exp": "_qwen4_mnnvl_hc_precompute_gate",
    "sglang.srt.layers.flashinfer_comm_fusion": "can_use_qwen4_mnnvl_hc",
    "sglang.kernels.ops.communication.qwen4_mnnvl_hc": "compile_qwen4_mnnvl_hc",
    "sglang.kernels.ops.elementwise.hc_gate": "compile_hc_gate",
}
for module, marker in required.items():
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        sys.exit(f"✗ MNNVL_HC is missing module: {module}")
    if marker not in pathlib.Path(spec.origin).read_text():
        sys.exit(f"✗ MNNVL_HC patch incomplete: {module} lacks {marker}")
if torch.cuda.device_count() != 4:
    sys.exit(f"✗ MNNVL_HC needs 4 visible GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(4)]
if names != ["NVIDIA B200"] * 4:
    sys.exit(f"✗ MNNVL_HC strict B200 guard failed: {names}")
PYCHK9
    "$SGLANG_VENV/bin/python" work/qwen38fn_eval/probe_mnnvl_hc_prod.py --compile-only
    echo "· Qwen4 MNNVL+HC: attention branch enabled (TP4/B200/BF16, rows<=512)"
fi
echo "· HC combine split: $SGLANG_HC_COMBINE_SPLIT (1=current production; 0=single-CTA micro experiment)"


MODEL=${MODEL:-$ARC_MODELS/Qwen3.8-Flash-Next-FP8}
[ -f "$MODEL/config.json" ] || { echo "✗ no $MODEL; run download_model.sh first" >&2; exit 1; }
PORT=${PORT:-30032}
NAME=${NAME:-qwen38fn}
CTXLEN=${CTXLEN:-262144}

SPEC=${SPEC:-nextn}
# The three capacity values of SPEC=dflash have no defaults; the caller must give them (10-28). The old default
# MEMFRAC 0.75 + mamba auto was set on 10-08 for 512 requests × K=8 (verify scratch ~13 GiB: 0.85 scheduler OOM, 0.78
# image preprocessing OOM) and got copied all the way to K=4, ≤96-request TP1 INT3 evals: mamba was allocated 798 slots
# and used 32%, KV was only 3.8M and often full, and 44 GB per GPU sat idle. The measured setup for single-B200 TP1
# INT3 K4 is MAXREQ 96, MEMFRAC 0.96, MAMBA_CACHE_SLOTS 6×MAXREQ; reasons and measurements in
# work/sglang_fork/README.md (section on B200 single-GPU server start-up capacity).
if [ "$SPEC" = dflash ]; then
    for v in MAXREQ MEMFRAC MAMBA_CACHE_SLOTS; do
        [ -n "${!v:-}" ] || { echo "✗ SPEC=dflash requires an explicit $v (single-B200 TP1 INT3 K4: MAXREQ 96, MEMFRAC 0.96, MAMBA_CACHE_SLOTS 6×MAXREQ; work/sglang_fork/README.md)" >&2; exit 1; }
    done
fi
MAXREQ=${MAXREQ:-256}
MEMFRAC=${MEMFRAC:-0.85}
# ARC_DFLASH_DRAFTER makes the fork's EAGLEWorkerV2 swap the MTP draft for DFlash: only SPEC=dflash may set it.
if [ "$SPEC" != dflash ] && [ -n "${ARC_DFLASH_DRAFTER:-}" ]; then
    echo "✗ the environment has ARC_DFLASH_DRAFTER=$ARC_DFLASH_DRAFTER but SPEC=$SPEC; for DFlash use SPEC=dflash DFLASH_DIR=..." >&2
    exit 1
fi

# The two official Qwen3.8 Flash Next checkpoints passed FP8 KV + MTP stress tests, so target
# and draft default to E4M3. Custom models keep SGLang's own default unless the user specifies it explicitly via env
# vars or the CLI. The CLI is parsed first and the launcher fills a value only when the option is absent, so the
# same flag never appears twice; duplicates passed by the user are rejected too.
EXTRA_ARGS=("$@")
KV_CLI_COUNT=0
MTP_KV_CLI_COUNT=0
KV_CLI_VALUE=""
MTP_KV_CLI_VALUE=""
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    arg=${EXTRA_ARGS[$i]}
    case "$arg" in
        --kv-cache-dtype)
            KV_CLI_COUNT=$((KV_CLI_COUNT + 1))
            if [ $((i + 1)) -ge ${#EXTRA_ARGS[@]} ] || [[ ${EXTRA_ARGS[$((i + 1))]} == --* ]]; then
                echo "✗ --kv-cache-dtype is missing its value" >&2
                exit 1
            fi
            KV_CLI_VALUE=${EXTRA_ARGS[$((i + 1))]}
            ;;
        --kv-cache-dtype=*)
            KV_CLI_COUNT=$((KV_CLI_COUNT + 1))
            KV_CLI_VALUE=${arg#*=}
            [ -n "$KV_CLI_VALUE" ] || { echo "✗ --kv-cache-dtype is missing its value" >&2; exit 1; }
            ;;
        --speculative-draft-kv-cache-dtype)
            MTP_KV_CLI_COUNT=$((MTP_KV_CLI_COUNT + 1))
            if [ $((i + 1)) -ge ${#EXTRA_ARGS[@]} ] || [[ ${EXTRA_ARGS[$((i + 1))]} == --* ]]; then
                echo "✗ --speculative-draft-kv-cache-dtype is missing its value" >&2
                exit 1
            fi
            MTP_KV_CLI_VALUE=${EXTRA_ARGS[$((i + 1))]}
            ;;
        --speculative-draft-kv-cache-dtype=*)
            MTP_KV_CLI_COUNT=$((MTP_KV_CLI_COUNT + 1))
            MTP_KV_CLI_VALUE=${arg#*=}
            [ -n "$MTP_KV_CLI_VALUE" ] || {
                echo "✗ --speculative-draft-kv-cache-dtype is missing its value" >&2
                exit 1
            }
            ;;
    esac
done
[ "$KV_CLI_COUNT" -le 1 ] || { echo "✗ --kv-cache-dtype must not be repeated" >&2; exit 1; }
[ "$MTP_KV_CLI_COUNT" -le 1 ] || {
    echo "✗ --speculative-draft-kv-cache-dtype must not be repeated" >&2
    exit 1
}

OFFICIAL_QWEN38FN=0
case "$(basename "${MODEL%/}")" in
    Qwen3.8-Flash-Next-FP8|Qwen3.8-Flash-Next-NVFP4) OFFICIAL_QWEN38FN=1 ;;
esac

KV_ARGS=()
if [ "$KV_CLI_COUNT" -eq 1 ]; then
    EFFECTIVE_KV_CACHE_DTYPE=$KV_CLI_VALUE
elif [ -n "${KV_CACHE_DTYPE:-}" ]; then
    EFFECTIVE_KV_CACHE_DTYPE=$KV_CACHE_DTYPE
    KV_ARGS=(--kv-cache-dtype "$EFFECTIVE_KV_CACHE_DTYPE")
elif [ "$OFFICIAL_QWEN38FN" -eq 1 ]; then
    EFFECTIVE_KV_CACHE_DTYPE=fp8_e4m3
    KV_ARGS=(--kv-cache-dtype "$EFFECTIVE_KV_CACHE_DTYPE")
else
    EFFECTIVE_KV_CACHE_DTYPE=""
fi
echo "· target KV dtype: ${EFFECTIVE_KV_CACHE_DTYPE:-SGLang auto}"

# Hybrid Mamba extra_buffer uses a fixed admission ratio: three base slots plus
# two overlap slots per running request, independent of speculative step count.
# Pinning the actual operating point (256×5 by default) returned 12.43 GiB/card
# to KV while preserving 256-way admission and LRU prefix reuse. SPEC=none stays
# on SGLang auto sizing unless explicitly overridden because that arm was not
# part of the capped-MTP stress gate.
# SPEC=dflash may not use auto (required explicitly above): auto at MEMFRAC 0.75 allocated 798 slots (42 GB, 12.5 per
# request), all taken from KV.
if [ "$SPEC" = none ]; then
    default_mamba_slots=auto
else
    default_mamba_slots=$((MAXREQ * 5))
fi
MAMBA_CACHE_SLOTS=${MAMBA_CACHE_SLOTS:-$default_mamba_slots}
MAMBA_CACHE_ARGS=""
case "$MAMBA_CACHE_SLOTS" in
    auto) [ "$SPEC" != dflash ] || { echo "✗ SPEC=dflash does not allow MAMBA_CACHE_SLOTS=auto; give 6×MAXREQ ($((6 * MAXREQ)))" >&2; exit 1; };;
    ""|*[!0-9]*) echo "✗ MAMBA_CACHE_SLOTS must be auto or a positive integer, got $MAMBA_CACHE_SLOTS" >&2; exit 1;;
    0) echo "✗ MAMBA_CACHE_SLOTS must be > 0" >&2; exit 1;;
    *)
        # admission needs 5 slots per running request (the section above; 1 with --disable-radix-cache): below that,
        # SGLang prints a single line "max_running_requests is capped to N" and lowers the concurrency (10-28: the
        # 96 / 384 arc3 evals actually ran only 76).
        mamba_ratio=5
        for a in "${EXTRA_ARGS[@]}"; do [ "$a" != --disable-radix-cache ] || mamba_ratio=1; done
        [ "$MAMBA_CACHE_SLOTS" -ge $((mamba_ratio * MAXREQ)) ] || {
            echo "✗ MAMBA_CACHE_SLOTS $MAMBA_CACHE_SLOTS < ${mamba_ratio}×MAXREQ ($((mamba_ratio * MAXREQ))): SGLang would silently cut MAXREQ to $((MAMBA_CACHE_SLOTS / mamba_ratio))" >&2; exit 1; }
        MAMBA_CACHE_ARGS="--max-mamba-cache-size $MAMBA_CACHE_SLOTS";;
esac
echo "· Mamba persistent cache slots: $MAMBA_CACHE_SLOTS"

SPEC_ARGS=()
if [ "$SPEC" = dflash ]; then
    # The fork's (sglang-arc-2) DFlash: NEXTN's EAGLEWorkerV2 is replaced by the drafter of speculative/arc_dflash;
    # target / verify unchanged. Only dense drafter deploy directories are accepted (config ffn_type dense_shared). An
    # FP8 deploy on B200 needs the fork's sm_100 block-FP8 path (from exp/dflash-fp8-b200 on, FlashInfer TRTLLM /
    # Triton; the fork refuses DeepGEMM); without it, use deploy-bf16.
    #   DFLASH_K           verify width K (anchor + K−1 drafts, 2..8, default 8)
    #   DFLASH_REPLAYSSM   1 (default) = --enable-linear-replayssm-spec: GDN allocates no per-draft intermediate SSM
    #                      (K=8 with 512 requests needs ~54 GiB/GPU), the whole mamba budget goes to persistent slots;
    #                      0 = the original intermediate (SGLang cuts max_running accordingly)
    #   DFLASH_RS          1 (default) = rejection sampling (the drafter's q); 0 for diagnostics only (target-only verify)
    #   ARC_DFLASH_ATTN_TP default 0 when TP>1 (the drafter's 2 KV heads cannot be split 4 ways: attention is replicated
    #                      on every rank)
    # Other ARC_DFLASH_* (EXACT_PREFILL, SESSION_PARK, SPARSE_Q, CHECK_AGREE…) are passed to sglang from the environment.
    : "${DFLASH_DIR:?SPEC=dflash needs DFLASH_DIR=<drafter deploy dir>}"
    "$SGLANG_VENV/bin/python" - "$DFLASH_DIR" <<'PYDFLASH' || exit 1
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
# dense: plan B's plain / inject drafter (convert_plain.py); the fork's build_dense_drafter checks residual / context itself
if cfg.get("ffn_type") not in ("dense_shared", "dense"):
    sys.exit(f"✗ {d}: not a dense drafter (ffn_type {cfg.get('ffn_type')!r}); MoE drafters go through work/sglang_fork/serve_b200_dflash.sh")
if (cfg.get("quantization") or {}).get("modules"):
    import importlib.util
    fp8 = Path(importlib.util.find_spec("sglang.srt.speculative.arc_dflash.fp8").origin).read_text()
    if "(10, 0): sm100" not in fp8:
        sys.exit(f"✗ {d}: FP8 deploy, but arc_dflash in this fork tree has no block-FP8 path on sm_100; use deploy-bf16 on B200")
if not json.loads((d / "initialization_manifest.json").read_text()).get("complete"):
    sys.exit(f"✗ {d}: export incomplete")
PYDFLASH
    "$SGLANG_VENV/bin/python" -c 'import sglang.srt.speculative.arc_dflash' 2>/dev/null \
        || { echo "✗ this sglang tree has no speculative/arc_dflash (needs the sglang-arc-2 fork)" >&2; exit 1; }
    DFLASH_K=${DFLASH_K:-8}
    case "$DFLASH_K" in 2|3|4|5|6|7|8) ;; *) echo "✗ DFLASH_K must be 2..8, got $DFLASH_K" >&2; exit 1;; esac
    DFLASH_REPLAYSSM=${DFLASH_REPLAYSSM:-1}
    case "$DFLASH_REPLAYSSM" in 0|1) ;; *) echo "✗ DFLASH_REPLAYSSM must be 0 or 1" >&2; exit 1;; esac
    # DFLASH_RS=0 is for diagnostics only: verify uses target-only sampling (a draft is accepted only if it is the token
    # the target itself drew), the drafter's q is not read
    DFLASH_RS=${DFLASH_RS:-1}
    case "$DFLASH_RS" in 0|1) ;; *) echo "✗ DFLASH_RS must be 0 or 1" >&2; exit 1;; esac
    for k in ARC_DFLASH_KEEP ARC_DFLASH_CALIB ARC_DFLASH_MOE; do
        [ -z "${!k:-}" ] || { echo "✗ $k is a MoE drafter flag; it must not be set for a dense drafter" >&2; exit 1; }
    done
    export ARC_DFLASH_DRAFTER=$DFLASH_DIR
    if [ "$TP" -gt 1 ]; then export ARC_DFLASH_ATTN_TP=${ARC_DFLASH_ATTN_TP:-0}; fi
    # drafter CUDA graph sizes: the fork's default has only 1..96 + max_running, so 97..MAXREQ−1 requests all pad to the
    # MAXREQ graph.
    if [ -z "${ARC_DFLASH_GRAPH_SIZES:-}" ]; then
        sizes=""
        for s in 1 2 4 8 12 16 24 32 40 48 56 64 80 96; do [ "$s" -ge "$MAXREQ" ] || sizes+="${sizes:+,}$s"; done
        for ((s = 128; s < MAXREQ; s += 32)); do sizes+=",$s"; done
        [ -z "$sizes" ] || export ARC_DFLASH_GRAPH_SIZES=$sizes
    fi
    SPEC_ARGS=(
        --speculative-algorithm NEXTN
        --speculative-num-steps $((DFLASH_K - 1))
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens "$DFLASH_K"
    )
    [ "$DFLASH_RS" = 0 ] || SPEC_ARGS+=(--speculative-use-rejection-sampling)
    [ "$DFLASH_REPLAYSSM" = 0 ] || SPEC_ARGS+=(--enable-linear-replayssm-spec)
    EFFECTIVE_MTP_KV_CACHE_DTYPE=""
    echo "· DFlash: $DFLASH_DIR K=$DFLASH_K ReplaySSM=$DFLASH_REPLAYSSM RS=$DFLASH_RS ATTN_TP=${ARC_DFLASH_ATTN_TP:-1}"
    env | grep -E '^ARC_DFLASH_' | sort | sed 's/^/· env /'
elif [ "$SPEC" != none ]; then
    # NEXTN is a draft chain: eagle-topk can only be 1, num_draft_tokens = num_steps + 1.
    # QSA compression ratio=4 supports only 1..3 steps; 3 is both the cap and the officially recommended value.
    #
    # **Default changed back from 1 to 3 (2026-08-30).** The earlier choice of 1 rested on the 08-29 matched sweep,
    # "steps=1 beats steps=2/3 at every bs1..96", but that batch was measured after the
    # `SGLANG_EAGLE_DRAFT_EXTEND_SELECTED_ROWS` hidden-row defect had landed,
    # and **steps=1 is the only setting that does not trigger that defect** -- it "won" because its opponents were
    # crippled. After the fix (patch 0016) the accept_len at steps=3 went from 1.998 back to 3.036.
    # See docs/research/qwen38fn_mtp_draft_hidden_bug.md.
    # (On the 0.5.20 line neither this flag nor 0013/0016 exists: absorbed upstream and fixed at the root cause;
    # PYCHK15 asserts it.)
    #
    # End-to-end throughput was measured on 2026-09-20 with a real workload (arc-code 25 games in parallel, codex arm,
    # NVFP4 TP1, sampling 1.0/0.95/20, 60 minutes per cell, median of decode batch lines n=3000, batch range 21–24):
    #   steps=1 draft 2  accept 1.73  aggregate 2,307 tok/s  per request 102.8
    #   steps=2 draft 3  accept 2.23  aggregate 2,469 tok/s  per request 110.1
    #   steps=3 draft 4  accept 2.56  aggregate 2,561 tok/s  per request 112.1   ← 9.0% faster than steps=1
    # The score cannot decide this cell (speculative decoding is mathematically exact; the sd of reruns with the same
    # settings is 3.34 points); compare throughput within the same batch range (at the tail, after most games finish,
    # batch drops to 8–9 with aggregate 1,580 -- not the ceiling).
    # raw: $ARC_RUNS/arc_code/20260919-sampling-screen/serve/{s1spec1,s1spec2,s7}/serve.log.
    # The MTP weights are in the checkpoint's own mtp.* tensors; no --speculative-draft-model-path needed.
    MTP_STEPS=${SPEC_STEPS:-3}
    case "$MTP_STEPS" in
        1|2|3) ;;
        *) echo "✗ SPEC_STEPS must be 1..3; QSA compression ratio=4 does not support more" >&2; exit 1;;
    esac
    MTP_DRAFT_TOK=${SPEC_DRAFT_TOK:-$((MTP_STEPS + 1))}
    if [ "$MTP_DRAFT_TOK" != "$((MTP_STEPS + 1))" ]; then
        echo "✗ SPEC_DRAFT_TOK must equal SPEC_STEPS+1 (now $MTP_STEPS/$MTP_DRAFT_TOK)" >&2
        exit 1
    fi
    SPEC_ARGS=(
        --speculative-algorithm NEXTN
        --speculative-num-steps "$MTP_STEPS"
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens "$MTP_DRAFT_TOK"
    )
    echo "· MTP NEXTN: steps=$MTP_STEPS draft_tokens=$MTP_DRAFT_TOK"

    if [ "$MTP_KV_CLI_COUNT" -eq 1 ]; then
        EFFECTIVE_MTP_KV_CACHE_DTYPE=$MTP_KV_CLI_VALUE
    elif [ -n "${MTP_KV_CACHE_DTYPE:-}" ]; then
        EFFECTIVE_MTP_KV_CACHE_DTYPE=$MTP_KV_CACHE_DTYPE
        SPEC_ARGS+=(--speculative-draft-kv-cache-dtype "$EFFECTIVE_MTP_KV_CACHE_DTYPE")
    elif [ "$OFFICIAL_QWEN38FN" -eq 1 ]; then
        EFFECTIVE_MTP_KV_CACHE_DTYPE=fp8_e4m3
        SPEC_ARGS+=(--speculative-draft-kv-cache-dtype "$EFFECTIVE_MTP_KV_CACHE_DTYPE")
    else
        EFFECTIVE_MTP_KV_CACHE_DTYPE=""
    fi
    echo "· MTP draft KV dtype: ${EFFECTIVE_MTP_KV_CACHE_DTYPE:-SGLang auto}"

    # NVIDIA's NVFP4 release explicitly excludes mtp.*, so the checkpoint-native draft is
    # BF16; the official FP8 release already has serialized FP8 weight+scale for the same MTP's routed experts.
    # For the official NVFP4 target, when both checkpoints exist on this machine, the draft is loaded directly from
    # the FP8 checkpoint, with no online quant and no copy of the target embed/head.
    # Other / custom targets do not cross checkpoints automatically; with an explicit path the structure gate below
    # verifies it.
    MTP_DRAFT_MODEL=${MTP_DRAFT_MODEL:-auto}
    if [ "$MTP_DRAFT_MODEL" = auto ]; then
        if [ "$(basename "${MODEL%/}")" = Qwen3.8-Flash-Next-NVFP4 ] \
                && [ -f "$ARC_MODELS/Qwen3.8-Flash-Next-FP8/config.json" ]; then
            MTP_DRAFT_MODEL="$ARC_MODELS/Qwen3.8-Flash-Next-FP8"
        else
            MTP_DRAFT_MODEL=embedded
        fi
    fi

    if [ "$MTP_DRAFT_MODEL" != embedded ] && [ -n "$MTP_DRAFT_MODEL" ]; then
        for arg in "$@"; do
            case "$arg" in
                --speculative-draft-model-path|--speculative-draft-model-path=*|\
                --speculative-draft-model-quantization|--speculative-draft-model-quantization=*|\
                --speculative-draft-load-format|--speculative-draft-load-format=*)
                    echo "✗ MTP_DRAFT_MODEL is set; do not also pass draft path/quant/load-format in the extra args" >&2
                    exit 1
                    ;;
            esac
        done
        "$SGLANG_VENV/bin/python" - "$MODEL" "$MTP_DRAFT_MODEL" <<'PYMTP' || exit 1
import json
import pathlib
import sys

target_path, draft_path = map(pathlib.Path, sys.argv[1:])
for label, path in (("target", target_path), ("FP8 MTP draft", draft_path)):
    if not (path / "config.json").is_file():
        sys.exit(f"✗ {label} is missing config.json: {path}")

target = json.loads((target_path / "config.json").read_text())
draft = json.loads((draft_path / "config.json").read_text())
expected_arch = ["Qwen4ExpForConditionalGeneration"]
if draft.get("architectures") != expected_arch:
    sys.exit(f"✗ FP8 MTP draft architecture must be {expected_arch}, got {draft.get('architectures')}")
quant = draft.get("quantization_config") or {}
if quant.get("quant_method") != "fp8":
    sys.exit(f"✗ FP8 MTP draft quant_method must be fp8, got {quant.get('quant_method')!r}")

target_text = target.get("text_config", target)
draft_text = draft.get("text_config", draft)
structural = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "hc_count",
    "mtp_num_hidden_layers",
    "layer_types",
)
mismatch = [key for key in structural if target_text.get(key) != draft_text.get(key)]
if mismatch:
    detail = ", ".join(f"{key}={target_text.get(key)!r}/{draft_text.get(key)!r}" for key in mismatch)
    sys.exit(f"✗ target / FP8 MTP draft structures differ: {detail}")

index_path = draft_path / "model.safetensors.index.json"
if not index_path.is_file():
    sys.exit(f"✗ FP8 MTP draft is missing model.safetensors.index.json: {draft_path}")
weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
prefix = "mtp.layers.0.mlp.experts."
expert_weights = sorted(
    key for key in weight_map if key.startswith(prefix) and key.endswith(".weight")
)
expected_weights = 3 * int(draft_text["num_experts"])
if len(expert_weights) != expected_weights:
    sys.exit(f"✗ FP8 MTP expert weights: expected {expected_weights}, got {len(expert_weights)}")
missing_scales = [
    key[:-7] + ".weight_scale_inv"
    for key in expert_weights
    if key[:-7] + ".weight_scale_inv" not in weight_map
]
if missing_scales:
    sys.exit(f"✗ FP8 MTP experts lack weight_scale_inv: {missing_scales[:3]}")
PYMTP
        SPEC_ARGS+=(
            --speculative-draft-model-path "$MTP_DRAFT_MODEL"
            --speculative-draft-model-quantization fp8
        )
        echo "· MTP draft weights: official serialized FP8 ($MTP_DRAFT_MODEL)"
    else
        echo "· MTP draft weights: checkpoint-native"
    fi
else
    EFFECTIVE_MTP_KV_CACHE_DTYPE=$MTP_KV_CLI_VALUE
fi

# fp8 KV: the old line relied on patches 0004/0015; v0.5.20 upstream fixed the same set of bugs itself (chunk-prefill's
# FP8→bf16 widen, the pack kernel's ZERO_FILL page-aligned store, **int64 offset** -- int32 overflows once the pool
# exceeds ~4.2M tokens, which the local patch never covered). Whenever the target or the MTP draft uses FP8 KV, these
# are asserted. Missing means the source is not v0.5.20 (or was rolled back to the old snapshot), and high rows / long
# contexts get polluted with NaN.
FP8_KV_ENABLED=0
case "$EFFECTIVE_KV_CACHE_DTYPE $EFFECTIVE_MTP_KV_CACHE_DTYPE" in
    *fp8*) FP8_KV_ENABLED=1 ;;
esac
if [ "$FP8_KV_ENABLED" -eq 1 ]; then
    "$SGLANG_VENV/bin/python" - <<'PYCHK4' || exit 1
import importlib.util
import pathlib
import sys

sparse = pathlib.Path(
    importlib.util.find_spec("sglang.srt.layers.attention.qsa.sparse_attn").origin
).read_text()
backend = pathlib.Path(
    importlib.util.find_spec(
        "sglang.srt.layers.attention.qwen_sparse_attn_backend"
    ).origin
).read_text()
sparse_markers = (
    "ZERO_FILL",
    "keys = keys.to(q_values.dtype)",
    "slots.to(tl.int64)",
)
backend_markers = ("zero_fill_cols=stride",)
missing = [marker for marker in sparse_markers if marker not in sparse]
missing += [marker for marker in backend_markers if marker not in backend]
if missing:
    sys.exit(
        "✗ QSA lacks the v0.5.20 upstream fp8-KV fix (ZERO_FILL / int64 offset); "
        f"missing markers: {missing}. Does $SGLANG_VENV point to a v0.5.20 checkout?"
    )
PYCHK4
fi

launch=("$SGLANG_VENV/bin/python" -m sglang.launch_server)
# Opt-in independent launch-epoch instrumentation for RL sync validation.
if [ -n "${RL_SYNC_PROBE_DIR:-}" ]; then
    launch=("$SGLANG_VENV/bin/python" -c
        'from work.sglang_fork.probes import rl_sync_probe; rl_sync_probe.main()')
fi
# gEAN320 INT3 W3A8 (RL rollout, launch_rollout_server.sh RL_MODEL_KIND=int3): the packed-expert plugin owns the load
# path (Humming experts, FP8/BF16 dense, PLE INT5 via ARC_PLE_INT5). TP1/EP1 only; the caller passes
# --moe-runner-backend humming (the plugin refuses to start without it).
# INT3_PROBE_OUT=<dir>: work/qwen38fn_rl/probes/int3_sync_probe.py wraps the same launch and records weight-sync reports.
if [ "${INT3_DEPLOY:-0}" = 1 ]; then
    [ "$TP" -eq 1 ] && [ "$EP" -eq 1 ] || { echo "✗ INT3_DEPLOY=1 is TP1/EP1 only (TP=$TP EP=$EP)" >&2; exit 1; }
    [ -z "${RL_SYNC_PROBE_DIR:-}" ] || { echo "✗ INT3_DEPLOY=1 does not combine with RL_SYNC_PROBE_DIR (use INT3_PROBE_OUT)" >&2; exit 1; }
    # ARC_INT3_TUNING (opt-in): a Humming schedule table is measured on one GPU model (int3_humming_tune.py records
    # it); int3_deploy_runtime also refuses a table whose default schedules are not this device's heuristics
    if [ -n "${ARC_INT3_TUNING:-}" ]; then
        tuned_on=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["device"])' "$ARC_INT3_TUNING")
        serving_on=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS%%,*}")
        [ "$tuned_on" = "$serving_on" ] || { echo "✗ ARC_INT3_TUNING was measured on $tuned_on; this server runs on $serving_on" >&2; exit 1; }
    fi
    [ -z "${ARC_ONLINE_FP8:-}" ] || { echo "✗ ARC_ONLINE_FP8 would requantize the BF16 dense modules the package keeps in BF16" >&2; exit 1; }
    launch=("$SGLANG_VENV/bin/python" work/qwen38fn_eval/int3_deploy_runtime.py launch)
    if [ -n "${INT3_PROBE_OUT:-}" ]; then
        launch=("$SGLANG_VENV/bin/python" work/qwen38fn_rl/probes/int3_sync_probe.py launch)
    fi
fi
exec "${launch[@]}" \
    --model-path "$MODEL" \
    --served-model-name "$NAME" \
    --tp "$TP" --ep "$EP" "${DP_ARGS[@]}" \
    --context-length "$CTXLEN" \
    --mem-fraction-static "$MEMFRAC" \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend "${LINEAR_PREFILL_BACKEND:-flashinfer}" \
    --linear-attn-decode-backend "${LINEAR_DECODE_BACKEND:-flashinfer}" \
    --mamba-ssm-dtype "${SSM_DTYPE:-bfloat16}" \
    $MAMBA_CACHE_ARGS \
    "${KV_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --max-running-requests "$MAXREQ" \
    --cuda-graph-max-bs-decode "$MAXREQ" \
    --mm-feature-transport cpu \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --sampling-defaults openai \
    --enable-cache-report \
    --host 127.0.0.1 --port "$PORT" "$@"
