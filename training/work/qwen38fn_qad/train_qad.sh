#!/bin/bash
# Launch train_qad.py on explicitly assigned idle GPUs (physical indices -> UUIDs). Same environment as
# work/qwen38fn_lora/train_tito_mm.sh (training venv, FlashQLA AutoCP backward, shared inductor/triton caches).
#   GPUS=4,5,6,7 OUT=<fresh dir> bash work/qwen38fn_qad/train_qad.sh --mode teacher|train|score --model ... --data ... [...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
PY="${ARC_PYTHON:-$HOME/work/venv-fn-train/bin/python}"
GPUS="${GPUS:?set GPUS, e.g. 4,5,6,7}"
OUT="${OUT:?set OUT (must not exist)}"
[ ! -e "$OUT" ] || { echo "output exists: $OUT" >&2; exit 1; }
IFS=, read -ra LIST <<< "$GPUS"
for gpu in "${LIST[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 100 ] || { echo "GPU $gpu has ${used} MiB in use; refusing to start" >&2; exit 1; }
done
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$GPUS" --query-gpu=uuid --format=csv,noheader | paste -sd,)
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ARC_ROOT/src:$ARC_ROOT/work/qwen38fn_lora:$ARC_ROOT/work/qwen38fn_qad${PYTHONPATH:+:$PYTHONPATH}"
# --dense-qat-gemm cutlass: the pinned flashinfer overlay, as work/qwen38fn_lora/train_rl_service.sh
if [[ " $* " == *" --dense-qat-gemm cutlass "* ]]; then
    overlay="$ARC_DATA/flashinfer-overlays/0.6.18"
    "$PY" work/qwen38fn_lora/flashinfer_overlay.py check "$overlay"
    export PYTHONPATH="$PYTHONPATH:$overlay" FLASHINFER_WORKSPACE_BASE="$ARC_RUNS/qwen38fn_lora/flashinfer-ws"
    export PATH="$(dirname "$PY"):$PATH"
fi
export TORCHINDUCTOR_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-inductor-cache"
export TRITON_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-triton-cache"
export QUACK_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-quack-cache"      # --grad-kernels quack (CuTe DSL JIT)
export CUTE_DSL_CACHE_DIR="$ARC_RUNS/qwen38fn_lora/node1-cute-dsl-cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"
export FLASH_QLA_AUTO_CP_BACKWARD="${GDN_CP_BACKWARD:-1}"
mkdir -p "$(dirname "$OUT")"
exec "$(dirname "$PY")/torchrun" --standalone --master-port "${MASTER_PORT:-29611}" --nproc_per_node="${#LIST[@]}" \
    work/qwen38fn_qad/train_qad.py --out "$OUT" "$@"
