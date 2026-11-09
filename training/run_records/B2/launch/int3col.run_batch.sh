#!/bin/bash
# One arc2 batch (codex2wr = the Kaggle AGI-2 harness, B200 collect config): $1=pool $2=comma-separated ids $3=batch tag
# $4=port $5=concurrency
# The 10-15 collection's run_batch.sh with only the repo (the pinned collect worktree), the output directory and the
# tokenizer (the INT3 deploy package's) changed.
set -euo pipefail
cd $HOME/work/arc-forge-collect1019
source env/env.sh >/dev/null
export PYTHONPATH=src
R=$ARC_RUNS/tito/20261019-int3-collect
exec "$HOME/work/venv-fn-train/bin/python" work/arc_code/arc2/run_arc2_seq.py \
  --config work/arc_code/arc2/configs/b200-fp8-collect-seq-wrap.json \
  --run-dir "$R/arc2/$3" --tasks-dir "$ARC_DATA/arc2_pools/$1" \
  --only "$2" --pool-label "$1" \
  --agent-python "$ARC_RUNS/arc_code/agent-runtime/bin/python" \
  --driver-bin "$(dirname "$ARC_CODEX_PREFIX")/arc2-jail" \
  --model-base-url http://127.0.0.1:$4/v1 --model qwen38fn \
  --tokenizer-json /dev/shm/uid-$UID/arcforge-runs/tito/20261019-int3-collect/model/qwen38fn-gean320-int3-gptq/tokenizer.json \
  --concurrency $5 --arms ind,refine_exp,rerefine --solvers-first "${@:6}"
