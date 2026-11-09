#!/bin/bash
# materialize.sh <R|F>: export v001000 of qadmix-<arm> + the gEAN320 INT3 package -> servable package (CPU only).
set -euo pipefail
ARM=${1:?R or F}
M=/dev/shm/uid-$UID/arcforge-runs/qwen38fn_qad/models
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_01Jn8ZjGJLPupy44aACbH3yT
export OMP_NUM_THREADS=8
exec $HOME/work/venv-fn-train/bin/python work/qwen38fn_qad/materialize_package.py --source $M/gean320-int3-gptq \
  --delta /dev/shm/uid-$UID/arcforge-runs/qwen38fn_qad/qadmix-$ARM-store/export/v001000 --out $M/qadmix-$ARM-v1000 \
  > $HOME/work/runs/qwen38fn_qad/20261023-qadmix/materialize-$ARM.log 2>&1
