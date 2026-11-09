#!/bin/bash
# gEAN320 one-shot INT3 GPTQ, fresh identity (O_DIRECT BF16 staging, no resident-cache-only).
# Reuses the stopped run's FP8 calibration model, masks, recorded configs and sglang checkout.
set -euo pipefail
NEW=20261009-gean320-int3-mm-2a34605e
OLD=20261009-gean320-int3-mm-01a1206d
R=$ARC_RUNS/qwen38fn_eval/$OLD
M=$ARC_RAM_RUNS/qwen38fn_eval/$OLD
R2=$ARC_RUNS/qwen38fn_eval/$NEW
M2=$ARC_RAM_RUNS/qwen38fn_eval/$NEW
PY=$HOME/work/dsv4/venv-0520/bin/python
exec $PY $R2/source/work/qwen38fn_eval/run_gean_int3_mm.py run \
  --source-root $R2/source --fork $R/sglang-src --fork-sha cd15559a287232cb350a6c1cbec62004a570eb03 \
  --python $PY --fp8-model $M/calib-fp8 --bf16-source $ARC_NFS/models/Qwen3.8-Flash-Next \
  --mask $R/inputs/gean320.pt --mtp-mask $R/inputs/mtp320.pt \
  --plan $ARC_RUNS/qwen38fn_eval/20261009-reap-mm/plan-v1/plan.jsonl \
  --ledger-root $ARC_RAM_RUNS/tito/20261006-node1-collect/ledger \
  --recorded-config-dir $R/inputs/recorded-configs \
  --run-dir $R2/orchestrator --ram-dir $M2/windows --parts-dir $R2/parts \
  --run-id $NEW --port 31073 --gpus 0,1,2,3 --window 6 --target-tokens 2000000 "$@"
