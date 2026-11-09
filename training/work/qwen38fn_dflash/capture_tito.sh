#!/bin/bash
# Capture server for one TITO ledger instance (capture_tito_runtime.py): the recording server's model, SGLANG_* env and
# flags, isolated for single-request prefill replay. Feed it with capture_tito.py.
#
#   GPUS=0,1,2,3 PORT=30650 RECORDED=<ledger>/<instance>/server.json QUEUE=$ARC_RAM_RUNS/<task>/queue \
#       [PIDFILE=<file>] [TARGET_PROFILE=<server.json of the deployment to capture with>] [CONTEXT_LENGTH=<n>] \
#       bash work/qwen38fn_dflash/capture_tito.sh [--model-path <copy of the target model>] [--tp <n>]
#
# Without RECORDED, TARGET_PROFILE alone starts the profile-mode server of a data plan (every ledger and token row;
# feed it with capture_plan.py).
#
# The fork tree must be the LOCK_TARGET (default b200-qwen38fn) commit, as for serve_b200.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

: "${GPUS:?}" "${PORT:?}" "${QUEUE:?capture queue; the hook writes here}"
RECORDED=${RECORDED:-} TARGET_PROFILE=${TARGET_PROFILE:-}
[ -n "$RECORDED$TARGET_PROFILE" ] || { echo "✗ RECORDED=<ledger instance>/server.json or TARGET_PROFILE needed" >&2; exit 1; }
: "${ARC_SGLANG_FORK:?env/<machine>.sh does not set ARC_SGLANG_FORK}"
LOCK_TARGET=${LOCK_TARGET:-b200-qwen38fn}
SGLANG_VENV=${SGLANG_VENV:-$HOME/work/dsv4/venv-0520}
PY=$SGLANG_VENV/bin/python

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
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$ARC_SGLANG_FORK/python TITO_CAPTURE_ROOT=$QUEUE
export PATH="$SGLANG_VENV/bin:$PATH"  # ninja for the flashinfer JIT
origin=$("$PY" -c 'import sglang; print(sglang.__file__)')
[[ $origin == "$ARC_SGLANG_FORK"/python/* ]] || { echo "✗ sglang resolves to $origin, not the fork" >&2; exit 1; }
mkdir -p "$QUEUE"

TARGET=${TARGET_PROFILE:-$RECORDED}  # whose runtime env and flags the server takes (capture_tito_runtime.py)
EXTRA=()
[ -n "${CAPTURE_MODEL_PATH:-}" ] && EXTRA+=(--model-path "$CAPTURE_MODEL_PATH")
[ -n "$TARGET_PROFILE" ] && EXTRA+=(--target-profile "$TARGET_PROFILE")
[ -n "$RECORDED" ] && EXTRA+=(--recorded "$RECORDED")
[ -n "${CONTEXT_LENGTH:-}" ] && EXTRA+=(--context-length "$CONTEXT_LENGTH")
if [ "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["env"].get("SGLANG_QWEN4_MNNVL_HC", "0"))' "$TARGET")" = 1 ]; then
    # Same JIT self-check the production serve script runs before a MNNVL+HC server.
    "$PY" work/qwen38fn_eval/probe_mnnvl_hc_prod.py --compile-only
fi
[ -n "${PIDFILE:-}" ] && echo $$ > "$PIDFILE"  # exec keeps the PID: this is the server's
exec "$PY" work/qwen38fn_dflash/capture_tito_runtime.py --port "$PORT" \
    --vlm-cache-mb "${VLM_CACHE_MB:-16384}" "${EXTRA[@]}" "$@"
