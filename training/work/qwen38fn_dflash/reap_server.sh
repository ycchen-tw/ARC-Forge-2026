#!/bin/bash
# REAP saliency server (reap_runtime.py): the recording's model, env and flags in capture isolation, MoE on Triton,
# TP = EP. Feed it with reap_capture.py.
#
#   GPUS=4,5 PORT=30690 ROOT=$ARC_RAM_RUNS/<task>/root PROFILE=<ledger instance>/server.json \
#       MODEL_PATH=<copy of the recorded model> PIDFILE=<file> [CONTEXT_LENGTH=<n>] bash work/qwen38fn_dflash/reap_server.sh
#
# The fork tree must be the LOCK_TARGET (default b200-qwen38fn) commit. Stop it by the PID in PIDFILE (the server's own
# process; kill its process group), never by pattern.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

: "${GPUS:?}" "${PORT:?}" "${ROOT:?}" "${PROFILE:?}" "${MODEL_PATH:?}" "${PIDFILE:?}"
: "${ARC_SGLANG_FORK:?env/<machine>.sh does not set ARC_SGLANG_FORK}"
LOCK_TARGET=${LOCK_TARGET:-b200-qwen38fn}
SGLANG_VENV=${SGLANG_VENV:-$HOME/work/dsv4/venv-0520}
PY=$SGLANG_VENV/bin/python
TP=$(awk -F, '{print NF}' <<< "$GPUS")

python3 - "$LOCK_TARGET" "$ARC_SGLANG_FORK/python/sglang" <<'PYLOCK'
import sys

sys.path.insert(0, "src")
from arcforge.sglang_lock import check_tree

t = check_tree(sys.argv[1], sys.argv[2])
print(f"· sglang-arc-2 {t['commit'][:12]} (lock target {sys.argv[1]})")
PYLOCK

busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
    if grep -q "$uuid" <<< "$busy"; then
        echo "✗ GPU $g already runs a process; this is a shared machine -- check whose it is" >&2
        exit 1
    fi
done

FFMPEG_DIR=${FFMPEG_DIR:-$HOME/work/dsv4/ffmpeg-n8.1}
[ -d "$FFMPEG_DIR/lib" ] && export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$ARC_SGLANG_FORK/python REAP_CAPTURE_ROOT=$ROOT
export PATH="$SGLANG_VENV/bin:$PATH"  # ninja for the flashinfer JIT
origin=$("$PY" -c 'import sglang; print(sglang.__file__)')
[[ $origin == "$ARC_SGLANG_FORK"/python/* ]] || { echo "✗ sglang resolves to $origin, not the fork" >&2; exit 1; }
mkdir -p "$ROOT"

EXTRA=()
[ -n "${CONTEXT_LENGTH:-}" ] && EXTRA+=(--context-length "$CONTEXT_LENGTH")
echo $$ > "$PIDFILE"  # exec keeps the PID: this is the server's
cd work/qwen38fn_dflash
exec "$PY" reap_runtime.py --target-profile "$PROFILE" --port "$PORT" --tp "$TP" --model-path "$MODEL_PATH" \
    --vlm-cache-mb "${VLM_CACHE_MB:-16384}" "${EXTRA[@]}"
