#!/bin/bash
# After the gEAN320 GPTQ orchestrator completes: export sidecar -> Humming decode check -> build package -> header audit.
set -euo pipefail
NEW=20261009-gean320-int3-mm-2a34605e
R=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-mm-01a1206d
R2=$ARC_RUNS/qwen38fn_eval/$NEW
EV=$ARC_RAM_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy
L=$ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy
W=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_01HRqRkzQ4R6eLTHUm69WRXs/work/qwen38fn_eval
PY=$HOME/work/dsv4/venv-0520/bin/python
NVFP4=/dev/shm/uid-$UID/models/Qwen3.8-FN-NVFP4-EAN320-v1
step() { echo "[$(date -u +%H:%M:%S)] $*"; }
until [ "$(python3 -c "import json;print(json.load(open('$R2/orchestrator/state.json'))['status'])")" != running ] \
      && ! kill -0 "$(cat $R2/orchestrator.pid)" 2>/dev/null; do sleep 20; done
status=$(python3 -c "import json;print(json.load(open('$R2/orchestrator/state.json'))['status'])")
[ "$status" = complete ] || { step "GPTQ status=$status; abort"; exit 1; }
for lo in 00 06 12 18 24 30 36 42; do test -f $R2/orchestrator/w$lo-complete.json || { step "missing w$lo receipt"; exit 1; }; done
[ "$(ls $R2/parts/int3-L*.safetensors | wc -l)" = 48 ] || { step "parts != 48"; exit 1; }
sel=$(python3 -c "import json;print(json.load(open('$R2/gptq_calibration.json'))['manifest_sha256'])")
python3 - <<PYEOF
import json,glob,hashlib
shas={hashlib.sha256(json.dumps(json.load(open(f))["selection"],sort_keys=True,separators=(",",":")).encode()).hexdigest() for f in glob.glob("$R2/orchestrator/w*-calib/complete.json")}
assert shas=={"$sel"}, shas
print("calibration selection identical across 8 windows")
PYEOF
cd $W
step export
CUDA_VISIBLE_DEVICES=0 $PY export_int3_sidecar.py --parts $R2/parts --deployed $NVFP4 \
  --bf16 $ARC_NFS/models/Qwen3.8-Flash-Next --dense-source $EV/dense_fp8.safetensors \
  --run-id $NEW --calib-sha256 $sel --hessian one-shot --out $EV/sidecar \
  --calibration-manifest $R2/gptq_calibration.json --pruning-manifest $R/inputs/pruning_calibration.json
step check_layer
CUDA_VISIBLE_DEVICES=0 $PY int3_sidecar.py --sidecar $EV/sidecar --layers 0,6,12,18,24,30,36,42,47
step build
$PY build_int3_deploy.py --sidecar $EV/sidecar --source $NVFP4 --dense-fp8 $EV/sidecar/dense_fp8.safetensors \
  --dense-manifest $EV/sidecar/int3_manifest.json --out $EV/model \
  --gptq-manifest $R2/gptq_calibration.json --pruning-calibration-manifest $R/inputs/pruning_calibration.json
step audit
$PY int3_deploy_runtime.py check --model $EV/model
du -sb $EV/sidecar $EV/model
step POST_GPTQ_OK
