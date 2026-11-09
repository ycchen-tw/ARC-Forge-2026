#!/bin/bash
# Launch the RL trainer service (train_rl_service.py) on explicitly assigned, idle GPUs (physical indices -> UUIDs).
#   GPUS=4,5,6,7 ARC_PYTHON=<venv>/bin/python MODEL=<FP8 checkpoint> OUT=<new dir> PORT=<port> \
#     [INIT_LORA=<checkpoint> | RESUME=<checkpoint> [RESUME_EXISTING_OUT=1]] [LR=2e-5] [EXTRA="--dump-logp"] \
#     bash work/qwen38fn_lora/train_rl_service.sh
# The service is ready once $OUT/trainer_endpoint.json exists; log in $OUT.log. A PYTHONPATH given by the caller is
# searched after this unit (e.g. a directory holding rl_batch.py before that file is in the unit).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
PY="${ARC_PYTHON:?set ARC_PYTHON to the training venv python}"
GPUS="${GPUS:?set GPUS, e.g. 4,5,6,7}"
MODEL="${MODEL:?set MODEL}"
OUT="${OUT:?set a fresh OUT directory}"
PORT="${PORT:?set PORT}"
existing=${RESUME_EXISTING_OUT:-0}
case "$existing" in
    0) [ ! -e "$OUT" ] || { echo "output exists: $OUT" >&2; exit 1; } ;;
    1) [ -n "${RESUME:-}" ] && [ -z "${INIT_LORA:-}" ] && [ -d "$OUT" ] \
        || { echo "RESUME_EXISTING_OUT=1 needs RESUME, existing OUT and no INIT_LORA" >&2; exit 1; } ;;
    *) echo "RESUME_EXISTING_OUT must be 0 or 1" >&2; exit 1 ;;
esac
IFS=, read -ra LIST <<< "$GPUS"
for gpu in "${LIST[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 100 ] || { echo "GPU $gpu has ${used} MiB in use; refusing to start" >&2; exit 1; }
done
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPUS" --query-gpu=uuid --format=csv,noheader | paste -sd,)
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ARC_ROOT/src:$ARC_ROOT/work/qwen38fn_lora${PYTHONPATH:+:$PYTHONPATH}"
# --dense-qat-gemm cutlass (fp8_dense_qat.set_gemm): flashinfer from the pinned overlay (flashinfer_overlay.py; the
# training venv has none), its own JIT cache, and the venv's ninja on PATH for the first build (~6 min per cache).
if [[ " ${EXTRA:-} " == *" --dense-qat-gemm cutlass "* ]]; then
    overlay="$ARC_DATA/flashinfer-overlays/0.6.18"
    "$PY" work/qwen38fn_lora/flashinfer_overlay.py check "$overlay"
    export PYTHONPATH="$PYTHONPATH:$overlay" FLASHINFER_WORKSPACE_BASE="$ARC_RUNS/qwen38fn_lora/flashinfer-ws"
    export PATH="$(dirname "$PY"):$PATH"
fi
export TORCHINDUCTOR_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-inductor-cache"
export TRITON_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-triton-cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"
# FlashQLA AutoCP backward, as train_tito_mm.sh (GDN_VERIFY.md).
export FLASH_QLA_AUTO_CP_BACKWARD="${GDN_CP_BACKWARD:-1}"
# Hang evidence (hang_watch.py; 10-21 a16t08: two FSDP all-gather timeouts): at NCCL's watchdog timeout the flight
# recorder writes each rank's last collectives with scheduled/started/completed state to $OUT/nccl_trace_rank_<r>, and
# faulthandler prints every thread's Python stack on the abort that follows.
export PYTHONFAULTHANDLER=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4000 TORCH_NCCL_ENABLE_TIMING=1 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_DUMP_TEMP_FILE="$OUT/nccl_trace_rank_"
ARGS=(--model "$MODEL" --out "$OUT" --port "$PORT" --lr "${LR:-2e-5}")
[ -z "${INIT_LORA:-}" ] || ARGS+=(--init-lora "$INIT_LORA")
[ -z "${RESUME:-}" ] || ARGS+=(--resume "$RESUME")
[ "$existing" != 1 ] || ARGS+=(--resume-existing-out)
# shellcheck disable=SC2206
[ -z "${EXTRA:-}" ] || ARGS+=($EXTRA)
mkdir -p "$(dirname "$OUT")"
tee_args=()
[ "$existing" != 1 ] || tee_args+=(-a)
"$(dirname "$PY")/torchrun" --standalone --nproc_per_node="${#LIST[@]}" \
    work/qwen38fn_lora/train_rl_service.py "${ARGS[@]}" 2>&1 | tee "${tee_args[@]}" "$OUT.log"
