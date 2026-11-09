#!/bin/bash
# fast_gates.py's pool resume evidence end to end: pipeline.sh SMOKE=1 on a small plan, STOP at STOP_AT steps, RESUME=1
# to the drain, then probes/pool_resume_check.py on the run.
#
#   TAG=<fresh tag> PLAN=<small plan dir> INIT=<init dir> CAPTURE_GPUS=5,6,7 TRAINER_GPUS=0,1,2,3,4 \
#       [STOP_AT=120] [STEP_ANCHORS=8192] [BASE_PORT=30741] bash work/qwen38fn_dflash/probes/pool_resume_check.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env/env.sh

: "${TAG:?}" "${PLAN:?}" "${INIT:?}" "${CAPTURE_GPUS:?}" "${TRAINER_GPUS:?}"
STOP_AT=${STOP_AT:-120}
export TAG PLAN INIT CAPTURE_GPUS TRAINER_GPUS SMOKE=1 STEPS=${STEPS:-100000} STEP_ANCHORS=${STEP_ANCHORS:-8192} \
    EVAL_EVERY=${EVAL_EVERY:-100} EVAL_ANCHORS=${EVAL_ANCHORS:-256} SAVE_EVERY=${SAVE_EVERY:-1000000} \
    BASE_PORT=${BASE_PORT:-30741}
OUT=$ARC_RUNS/qwen38fn_dflash/$TAG
LOG=$ARC_RUNS/qwen38fn_dflash/$TAG.check.log

RESUME=0 bash work/qwen38fn_dflash/pipeline.sh >> "$LOG" 2>&1 &
first=$!
until [ -f "$OUT/trainer/metrics.jsonl" ] && (( $(wc -l < "$OUT/trainer/metrics.jsonl") >= STOP_AT )); do
    kill -0 "$first" 2>/dev/null || { echo "✗ the first segment ended before step $STOP_AT; see $LOG" >&2; exit 1; }
    sleep 2
done
touch "$OUT/trainer/STOP"
echo "· STOP at $(wc -l < "$OUT/trainer/metrics.jsonl") steps"
wait "$first" || { echo "✗ the first segment failed; see $LOG" >&2; exit 1; }
RESUME=1 bash work/qwen38fn_dflash/pipeline.sh >> "$LOG" 2>&1 || { echo "✗ the resumed segment failed; see $LOG" >&2; exit 1; }
"${PYTHON:-.venv/bin/python}" work/qwen38fn_dflash/probes/pool_resume_check.py --plan "$PLAN" --run "$OUT" --objective acceptance \
    --out "$OUT/resume-evidence.json"
