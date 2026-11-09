#!/bin/bash
# Rebuild the uploaded package locally (export from parts -> build) and verify against HF's deploy_manifest.json.
set -euo pipefail
NEW=20261009-gean320-int3-mm-2a34605e
R=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-mm-01a1206d
R2=$ARC_RUNS/qwen38fn_eval/$NEW
EV=$ARC_RAM_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy
L=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy
W=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_01HRqRkzQ4R6eLTHUm69WRXs/work/qwen38fn_eval
PY=$HOME/work/dsv4/venv-0520/bin/python
sel=$(python3 -c "import json;print(json.load(open('$R2/gptq_calibration.json'))['manifest_sha256'])")
cd $W
echo "[$(date -u +%T)] export"
CUDA_VISIBLE_DEVICES=0 $PY export_int3_sidecar.py --parts $R2/parts --deployed /dev/shm/uid-$UID/models/Qwen3.8-FN-NVFP4-EAN320-v1 \
  --bf16 $ARC_NFS/models/Qwen3.8-Flash-Next --dense-source $EV/dense_fp8.safetensors \
  --run-id $NEW --calib-sha256 $sel --hessian one-shot --out $EV/sidecar \
  --calibration-manifest $R2/gptq_calibration.json --pruning-manifest $R/inputs/pruning_calibration.json > $L/rebuild_export.log 2>&1
OUT=$EV/model-kaggle bash $L/build.sh --validation-notes $L/validation_notes.md --shard-bytes $((5 << 30))
rm -rf $EV/sidecar $EV/hfmeta
