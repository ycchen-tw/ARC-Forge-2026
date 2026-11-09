#!/bin/bash
set -euo pipefail
R=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-mm-01a1206d
R2=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-mm-2a34605e
EV=$ARC_RAM_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy
W=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_01HRqRkzQ4R6eLTHUm69WRXs/work/qwen38fn_eval
PY=$HOME/work/dsv4/venv-0520/bin/python
OUT=${OUT:-$EV/model}
cd $W
echo "[$(date -u +%H:%M:%S)] build -> $OUT"
$PY build_int3_deploy.py --sidecar $EV/sidecar --source /dev/shm/uid-$UID/models/Qwen3.8-FN-NVFP4-EAN320-v1 \
  --dense-fp8 $EV/sidecar/dense_fp8.safetensors --dense-manifest $EV/sidecar/int3_manifest.json --out $OUT \
  --gptq-manifest $R2/gptq_calibration.json --pruning-calibration-manifest $R/inputs/pruning_calibration.json \
  --drop-mtp "$@"
echo "[$(date -u +%H:%M:%S)] audit"
$PY int3_deploy_runtime.py check --model $OUT
du -sb $OUT
echo "[$(date -u +%H:%M:%S)] BUILD_OK"
