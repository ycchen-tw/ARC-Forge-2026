#!/bin/bash
# One arc2 batch (codex2wr = the Kaggle AGI-2 harness): $1=pool $2=comma-separated ids $3=batch tag $4=port
# $5=concurrency $6=variant (configs/<variant>.json)
# Adapted from the 10-19 run_batch.sh: worktree collect1022 (e6d9c538), output directory, prompt / effort / view
# variants, largegrid30 tasks in a tasks/ subdirectory.
set -euo pipefail
cd $HOME/work/arc-forge-collect1022
source env/env.sh >/dev/null
export PYTHONPATH=src
R=$ARC_RUNS/tito/20261022-int3-qad-collect
case "$1" in largegrid30) dir=$ARC_DATA/arc2_pools/largegrid30-20261022/tasks ;; *) dir=$ARC_DATA/arc2_pools/$1 ;; esac
exec "$HOME/work/venv-fn-train/bin/python" work/arc_code/arc2/run_arc2_seq.py \
  --config "$R/arc2/configs/$6.json" \
  --run-dir "$R/arc2/$3" --tasks-dir "$dir" \
  --only "$2" --pool-label "$1" \
  --agent-python "$ARC_RUNS/arc_code/agent-runtime/bin/python" \
  --driver-bin "$(dirname "$ARC_CODEX_PREFIX")/arc2-jail" \
  --model-base-url http://127.0.0.1:$4/v1 --model qwen38fn \
  --tokenizer-json "$R/tokenizer.json" \
  --concurrency $5 --arms ind,refine_exp,rerefine --solvers-first
