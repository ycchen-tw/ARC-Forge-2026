#!/bin/bash
# prune the chosen REAP320 mask out of the NVFP4 512E source (5 GiB shards like the incumbent) and verify it
#   bash run_final.sh <name> <mask.pt>
set -euo pipefail
NAME=$1 MASK=$2
R=$ARC_RUNS/qwen38fn_eval/20261009-reap-mm
SRC=/dev/shm/uid-$UID/models/Qwen3.8-Flash-Next-NVFP4-nvidia
INC=$ARC_NFS/models/Qwen3.8-Flash-Next-NVFP4-nvidia-reap320-arccode
OUT=/dev/shm/uid-$UID/models/Qwen3.8-FN-NVFP4-$NAME
[ -e "$OUT" ] && { echo "✗ $OUT exists"; exit 1; }
cd $HOME/work/arc-forge/.claude/worktrees/reap-mm && source env/env.sh
cd work/qwen38fn_eval
PY=$HOME/work/arc-forge/.venv/bin/python
sha256sum "$MASK"
env PYTHONPATH=$ARC_ROOT/src $PY prune_experts.py --src $SRC --mask "$MASK" \
  --mtp-mask $R/select-v1/masks/reap320_incumbent_mtp.pt --out $OUT --shard-bytes $((5*1024**3)) > $R/prune_$NAME.log 2>&1
tail -2 $R/prune_$NAME.log
env PYTHONPATH=$ARC_ROOT/src $PY probes/verify_pruned_vs_incumbent.py --src $SRC --out $OUT --incumbent $INC --samples 64 \
  > $R/verify_$NAME.log 2>&1
cat $R/verify_$NAME.log
echo FINAL_DONE
