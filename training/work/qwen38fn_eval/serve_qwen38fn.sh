#!/bin/bash
# Serve Qwen/Qwen3.8-Flash-Next-FP8 with SGLang for ARC-AGI-2 TIR eval, on an 8×B200 node.
# GPUs are chosen with GPUS= (default 0-3); start-up refuses GPUs that already run someone else's process.
#
#   bash work/qwen38fn_eval/serve_qwen38fn.sh              # MTP on (default)
#   SPEC=none bash work/qwen38fn_eval/serve_qwen38fn.sh    # MTP off
#   MTP_DRAFT_MODEL=embedded ...                            # force the embedded BF16 MTP for an NVFP4 target
#
# The baseline is the SGLang cookbook's **verified cell** (hw=b200 / quant=fp8 / low-latency /
# single), item by item below, plus the must-set items of this repo's `docs/research/serving_sglang_notes.md`.
# The control arm is work/qwen38_eval/serve_qwen38.sh (Qwen3.8-27B-FP8).
#
# Differences from the 27B script, each with its source:
#
# 1. **A different venv**. qwen4_exp is in no tagged release (0.5.18 included); it needs PR #36497.
#    See build_sglang_qwen4.sh. The symptom of the wrong venv is "unsupported architecture" at start-up.
# 2. **No --attention-backend**. This model's full-attention layers are QSA (Qwen Sparse
#    Attention); the PR has a dedicated override for Qwen4ExpForConditionalGeneration in arg_groups/overrides.py;
#    the 27B script's trtllm_mha is for dense full attention and does not apply.
# 3. **No --json-model-override-args num_nextn_predict_layers**. On the 27B that line is required for MTP
#    (KV cells get inflated 5x), but the PR's model_config._config_draft_model writes
#    text_config.num_nextn_predict_layers=1 for Qwen4Exp directly; the pitfall is fixed upstream.
#    **Still probe max_total_num_tokens after start-up** -- this is a "symptomless" class of defect.
# 4. **--linear-attn-{prefill,decode}-backend flashinfer + --mamba-ssm-dtype bfloat16**
#    per the cookbook. This is the opposite of this repo's "GDN state is fp32" policy (see serving_sglang_notes.md),
#    but the flashinfer linear-attn backend was validated with bf16, and the vendor validated only this combination.
#    **The cost is a drift risk in the recurrent state, so probes/preflight.sh must be run.**
# 5. **No --enable-linear-replayssm-spec**. It requires --disable-radix-cache or the
#    mamba no_buffer strategy, while QSA compressed forces page_size=64 and only works with mamba
#    extra_buffer. Between the two, TIR's multi-turn prefix cache matters more (on the 27B it was 2.2x).
# 6. **--tool-call-parser must be given**. The harness uses OpenAI tool calling; without the parser
#    tool_calls never arrive and the model is treated as "cannot use tools" -- the score collapses with no error.
#    This model's template uses the XML form <tool_call><function=…><parameter=…>;
#    `auto` lets sglang resolve it from the chat template (the start-up log prints which one it resolved to).
# 7. **--sampling-defaults openai is still required**. generation_config.json says top_k=20,
#    leaving only 20 candidates of the 248k vocabulary, and the harness does not send top_k.
# 8. **--mm-feature-transport cpu is still required**. Declared as *ForConditionalGeneration,
#    the default cuda_ipc gets blocked by the yama ptrace scope on the first request.
# 9. **--ep 4**: 512 experts / top-10; the cookbook's fp8 cells all enable EP.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

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
# With the hard-coded 0-3 whitelist gone, GPUS is free-form while TP/EP still default to 4 --
# `GPUS=4,5` would pass the occupancy check and then run --tp 4 on 2 GPUs. The count has to match.
n_gpus=$(printf '%s' "${GPUS//,/ }" | wc -w)
[ "$n_gpus" -eq "$TP" ] || { echo "✗ GPUS lists $n_gpus GPUs ($GPUS) but TP=$TP; they do not match" >&2; exit 1; }

SGLANG_VENV=${SGLANG_VENV:-$HOME/work/dsv4/venv-qwen4}
[ -x "$SGLANG_VENV/bin/python" ] || { echo "✗ no $SGLANG_VENV; run build_sglang_qwen4.sh first" >&2; exit 1; }
# flashinfer's runtime JIT shells out to find ninja; when python is called by absolute path, PATH does not carry it.
export PATH="$SGLANG_VENV/bin:$PATH"
# The venv has no arcforge, so the mrope plugin is not loaded; set anyway so that a venv switch does not trip on it.
export ARC_MROPE_ENABLED=0
# This venv deliberately has no Rust extension (see build_sglang_qwen4.sh). `never` makes any path that
# wants it error out on the spot instead of shelling out to cargo build in the middle of start-up.
export SGLANG_RUST_SERVER=0 SGLANG_RUST_BUILD_MODE=never

# PLE / n-gram ablation. ABLATE_PLE=1 zeroes PLE's additive term on the residual = the whole 51B n-gram path
# contributes nothing, the rest of the model is bit-identical. The in-place modification lives in
# patches/sglang-qwen4-pr36497/0001-ablate-ple-ngram.patch.
# Below is a **self-check at start-up**: when the patch is silently reverted by `uv cache prune` or a reinstall,
# without this check an arm that "we thought was ablated but was not" would run quietly.
export ARC_ABLATE_PLE=${ABLATE_PLE:-0}
"$SGLANG_VENV/bin/python" - <<'PYCHK' || exit 1
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("sglang.srt.models.qwen4_exp")
src = pathlib.Path(spec.origin).read_text()
if "_ARC_ABLATE_PLE" not in src:
    sys.exit("✗ qwen4_exp.py lacks the arc-forge ablation patch (reverted?): "
             "cd $HOME/work/sglang-qwen4 && git apply <arc-forge>/patches/sglang-qwen4-pr36497/0001-ablate-ple-ngram.patch")
PYCHK

# PLE_HOT=<npz> starts a "pruned-table simulation" server: before lookup, ids landing in cold buckets are folded back
# into the hot table, equivalent to running a pruned table on the full 51 GB table -- **no new checkpoint needed**.
# The npz is produced by ple_ngram.py build-hot and carries the build / holdout task lists.
# **Acceptance may only run the npz's holdout tasks**: validating on tasks the table was built from gives a false
# "zero loss".
export ARC_PLE_HOT=${PLE_HOT:-}
export ARC_PLE_HOT_FALLBACK=${PLE_HOT_FALLBACK:-fold}
if [ -n "$ARC_PLE_HOT" ]; then
    [ -f "$ARC_PLE_HOT" ] || { echo "✗ no $ARC_PLE_HOT; run ple_ngram.py build-hot first" >&2; exit 1; }
    "$SGLANG_VENV/bin/python" - <<'PYCHK2' || exit 1
import importlib.util, pathlib, sys
src = pathlib.Path(importlib.util.find_spec("sglang.srt.models.qwen4_exp").origin).read_text()
if "_ARC_PLE_HOT" not in src:
    sys.exit("✗ qwen4_exp.py lacks the arc-forge hot-table patch: "
             "cd $HOME/work/sglang-qwen4 && git apply <arc-forge>/patches/sglang-qwen4-pr36497/0003-ple-hot-mask.patch")
PYCHK2
    echo "· PLE hot-table simulation: $(basename "$ARC_PLE_HOT") fallback=$ARC_PLE_HOT_FALLBACK"
fi


# QSA decode large-batch experiment: sparse attention directly from the paged KV by the selected logical indices,
# saving about 256 MiB of compact K/V materialization per layer. rows < threshold still go through
# compact+TRT-LLM; off by default, because the attention reduction order changes and it must pass the score gate
# first.
export SGLANG_QSA_DIRECT_DECODE=${QSA_DIRECT:-0}
export SGLANG_QSA_DIRECT_MIN_ROWS=${QSA_DIRECT_MIN_ROWS:-64}
if [ "$SGLANG_QSA_DIRECT_DECODE" = 1 ]; then
    "$SGLANG_VENV/bin/python" - <<'PYCHK5' || exit 1
import importlib.util, pathlib, sys
src = pathlib.Path(importlib.util.find_spec(
    "sglang.srt.layers.attention.qsa.sparse_attn").origin).read_text()
if "qwen_sparse_gqa_decode_selected_triton" not in src:
    sys.exit("✗ sparse_attn.py lacks the arc-forge direct-QSA patch: "
             "cd $HOME/work/sglang-qwen4 && git apply "
             "<arc-forge>/patches/sglang-qwen4-pr36497/0005-qsa-direct-decode.patch")
PYCHK5
    echo "· direct QSA decode: rows >= $SGLANG_QSA_DIRECT_MIN_ROWS (experimental, has not passed the full score gate)"
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
             "cd $HOME/work/sglang-qwen4 && git apply "
             "<arc-forge>/patches/sglang-qwen4-pr36497/0006-qsa-mqa-auto-gpc.patch")
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

# Backport upstream PR #35546: select one accepted row per request before the
# DRAFT_EXTEND_V2 lm_head and LAST-hidden capture. TP4 parity is bitwise and
# graph capture saves ~1 GiB/rank; EAGLE_SELECTED_ROWS=0 restores baseline.
export SGLANG_EAGLE_DRAFT_EXTEND_SELECTED_ROWS=${EAGLE_SELECTED_ROWS:-1}
if [ "$SGLANG_EAGLE_DRAFT_EXTEND_SELECTED_ROWS" = 1 ]; then
    "$SGLANG_VENV/bin/python" - <<'PYCHK13' || exit 1
import importlib.util
import pathlib
import sys

# **A list, not a dict**: one module needs two markers verified; with a dict the later entry would
# silently overwrite the earlier one, one check fewer -- exactly what this kind of gate must never do.
required = [
    ("sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner", "use_selected_rows"),
    ("sglang.srt.layers.logits_processor", "draft_extend_select_index"),
    ("sglang.srt.model_executor.input_buffers", "exclude: Collection[str]"),
    # patch 0016: without it selected-rows returns the wrong hidden row and every MTP draft position from the 2nd on
    # collapses (measured p2 0.84→0.12, accept_len at steps=3 3.04→2.00).
    # **This defect is completely silent** -- no error, no garbled output (accepted tokens are always decided by the
    # target), only accept_len drops, so nobody would notice a revert. That is why it must be gated.
    # The marker must be **unique to this change**. `buffers.select_index[:raw_bs]` does not work --
    # line 559 of the original source already has the identical string, so the gate would pass with the patch removed.
    # Verified by experiment: patch removed → the gate must block; if it does not, the protection is fake.
    ("sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner",
     "hidden = hidden[buffers.select_index"),
]
for module, marker in required:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        sys.exit(f"✗ EAGLE selected-row patch is missing module: {module}")
    if marker not in pathlib.Path(spec.origin).read_text():
        sys.exit(f"✗ EAGLE selected-row patch incomplete: {module} lacks {marker}")
PYCHK13
fi
echo "· EAGLE draft-extend selected rows: $SGLANG_EAGLE_DRAFT_EXTEND_SELECTED_ROWS"

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

# SALIENCY=<dir> starts a "measurement" server: REAP saliency needs per-expert ‖f‖,
# while the default flashinfer_trtllm does even top-k inside the kernel (BypassedTopKOutput),
# so nothing is available at the Python level. It takes three things together:
#   1. --moe-runner-backend triton   → topk goes STANDARD, ids/weights are available
#   2. SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1 → disables the fused_func shortcut that bypasses the hook
#   3. patch 0002 makes _maybe_build_lora_hooks return the saliency hook
#      (attaching a hook materializes intermediate_cache3, which is g·f)
# MTP **must be on**: draft and main model are accumulated separately by (target|mtp, layer_id), dumping two blobs.
# MTP is collected from actual draft rollouts (not SPEC=none followed by prefilling finished trajectories).
# **EP required**: under pure TP down_proj is split along the intermediate dimension, intermediate_cache3 is only a
# partial sum and the norm is wrong. --ep 4 is kept here.
SALIENCY=${SALIENCY:-}
SALIENCY_ARGS=""
if [ -n "$SALIENCY" ]; then
    export ARC_SALIENCY_OUT="$SALIENCY"
    export ARC_SALIENCY_HOOK_PY="${ARC_SALIENCY_HOOK_PY:-$ARC_ROOT/work/qwen38fn_eval/saliency_hook.py}"
    export ARC_SALIENCY_MODEL="${MODEL:-$ARC_MODELS/Qwen3.8-Flash-Next-FP8}"
    export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
    # Prefix cache skips the MoE hook on cached tokens. Calib must disable it.
    SALIENCY_ARGS="--moe-runner-backend triton --disable-radix-cache"
    mkdir -p "$SALIENCY"
    if [ -f "$ARC_SALIENCY_MODEL/prune_manifest.json" ]; then
        eval "$(
            "$SGLANG_VENV/bin/python" - "$ARC_SALIENCY_MODEL" <<'PY'
import hashlib, json, os, sys
src = sys.argv[1]
man = json.load(open(os.path.join(src, "prune_manifest.json")))

def sha256(path):
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

print(f'export ARC_HESS_MASK_SHA256={sha256(man.get("mask") or "")}')
print(f'export ARC_HESS_MTP_MASK_SHA256={sha256(man.get("mtp_mask") or "")}')
PY
        )"
    fi
    echo "· saliency mode: triton runner + hook, radix cache off, output to $SALIENCY"
    if [ -n "${ARC_HESS_RUN_ID:-}" ]; then
        echo "· hessian identity run_id=$ARC_HESS_RUN_ID mask=${ARC_HESS_MASK_SHA256:-none}"
    fi
fi

MODEL=${MODEL:-$ARC_MODELS/Qwen3.8-Flash-Next-FP8}
[ -f "$MODEL/config.json" ] || { echo "✗ no $MODEL; run download_model.sh first" >&2; exit 1; }
PORT=${PORT:-30031}
NAME=${NAME:-qwen38fn}
CTXLEN=${CTXLEN:-262144}
MEMFRAC=${MEMFRAC:-0.85}
MAXREQ=${MAXREQ:-256}

SPEC=${SPEC:-nextn}

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
if [ "$SPEC" = none ]; then
    default_mamba_slots=auto
else
    default_mamba_slots=$((MAXREQ * 5))
fi
MAMBA_CACHE_SLOTS=${MAMBA_CACHE_SLOTS:-$default_mamba_slots}
MAMBA_CACHE_ARGS=""
case "$MAMBA_CACHE_SLOTS" in
    auto) ;;
    ""|*[!0-9]*) echo "✗ MAMBA_CACHE_SLOTS must be auto or a positive integer, got $MAMBA_CACHE_SLOTS" >&2; exit 1;;
    0) echo "✗ MAMBA_CACHE_SLOTS must be > 0" >&2; exit 1;;
    *) MAMBA_CACHE_ARGS="--max-mamba-cache-size $MAMBA_CACHE_SLOTS";;
esac
echo "· Mamba persistent cache slots: $MAMBA_CACHE_SLOTS"

SPEC_ARGS=()
if [ "$SPEC" != none ]; then
    # NEXTN is a draft chain: eagle-topk can only be 1, num_draft_tokens = num_steps + 1.
    # QSA compression ratio=4 supports only 1..3 steps; 3 is both the cap and the officially recommended value.
    #
    # **Default changed back from 1 to 3 (2026-08-30).** The earlier choice of 1 rested on the 08-29 matched sweep,
    # "steps=1 beats steps=2/3 at every bs1..96", but that batch was measured after the
    # `SGLANG_EAGLE_DRAFT_EXTEND_SELECTED_ROWS` hidden-row defect had landed,
    # and **steps=1 is the only setting that does not trigger that defect** -- it "won" because its opponents were
    # crippled. After the fix (patch 0016) the accept_len at steps=3 went from 1.998 back to 3.036.
    # See docs/research/qwen38fn_mtp_draft_hidden_bug.md.
    #
    # ⚠️ **The end-to-end throughput matched sweep has not been rerun**: a restored accept_len does not necessarily
    # mean faster (each step verifies 4 tokens instead of 2). Throughput numbers need a new measurement.
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

# patch 0004 (the complete QSA FP8-KV path) + 0015 (TRT packed page-tail sanitize):
# verified whenever the target or the MTP draft explicitly uses FP8 KV. 0015 writes FP8 zeros into the final partial
# page padding that TRT may read; without it, high rows / long contexts get polluted with NaN.
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
    "is_fp8_kv_dtype",
    "ZERO_PAGE_TAIL=zero_page_tail",
    "copy_width = packed_stride",
)
backend_markers = (
    "row_valid_counts=valid_counts",
    "packed_stride=stride",
    "page_size=page",
)
missing = [marker for marker in sparse_markers if marker not in sparse]
missing += [marker for marker in backend_markers if marker not in backend]
if missing:
    sys.exit(
        "✗ QSA lacks the complete arc-forge fp8-KV + 0015 page-tail patch; "
        f"missing markers: {missing}. cd $HOME/work/sglang-qwen4 && git apply "
        "<arc-forge>/patches/sglang-qwen4-pr36497/{0004,0015}-*.patch"
    )
PYCHK4
fi

LAUNCH_ARGS=(-m sglang.launch_server)
if [ -n "${ARC_SGLANG_LAUNCH_PY:-}" ]; then
    [ -f "$ARC_SGLANG_LAUNCH_PY" ] || { echo "missing launcher: $ARC_SGLANG_LAUNCH_PY" >&2; exit 1; }
    LAUNCH_ARGS=("$ARC_SGLANG_LAUNCH_PY")
fi
exec "$SGLANG_VENV/bin/python" "${LAUNCH_ARGS[@]}" \
    --model-path "$MODEL" \
    --served-model-name "$NAME" \
    --tp "$TP" --ep "$EP" \
    --context-length "$CTXLEN" \
    --mem-fraction-static "$MEMFRAC" \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend "${LINEAR_PREFILL_BACKEND:-flashinfer}" \
    --linear-attn-decode-backend "${LINEAR_DECODE_BACKEND:-flashinfer}" \
    --mamba-ssm-dtype "${SSM_DTYPE:-bfloat16}" \
    $MAMBA_CACHE_ARGS \
    "${KV_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    $SALIENCY_ARGS \
    --max-running-requests "$MAXREQ" \
    --cuda-graph-max-bs-decode "$MAXREQ" \
    --mm-feature-transport cpu \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --sampling-defaults openai \
    --enable-cache-report \
    --host 127.0.0.1 --port "$PORT" "$@"
