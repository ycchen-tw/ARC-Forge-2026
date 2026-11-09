#!/bin/bash
# 10-22 student ledger → tito-mm (cpu-exact, no GPU). fork = sglang-arc-2-allmerge-b200 as used during collection
# (ed837597).
set -euo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_01Jn8ZjGJLPupy44aACbH3yT
source env/env.sh >/dev/null
S=$ARC_RAM_RUNS/tito/20261022-int3-qad-collect
export ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-allmerge-b200 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/work/dsv4/ffmpeg-n8.1/lib PYTHONPATH=$HOME/work/sglang-arc-2-allmerge-b200/python:$ARC_ROOT/src
mans=( $S/manifests1022/arc2/*/tito_manifest.json $S/manifests1022/arc3/*/run/tito_manifest.json )
exec $HOME/work/dsv4/venv-0520/bin/python work/qwen38fn_lora/prepare_tito_mm.py --device cpu-exact \
  --ledger $S/ledger/p309* --model /dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq \
  --out $S/data1022 --manifests "${mans[@]}" --workers 24 --max-gb 200
