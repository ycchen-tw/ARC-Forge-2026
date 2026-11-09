#!/bin/bash
# Manual batch (until the outer tooling is done): $1=pool $2=comma-separated ids $3=batch tag $4=port $5=concurrency
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source env/env.sh >/dev/null
export PYTHONPATH=src
exec "$HOME/work/venv-fn-train/bin/python" work/arc_code/arc2/run_arc2_seq.py \
  --config work/arc_code/arc2/configs/b200-fp8-collect-seq-wrap.json \
  --run-dir "$ARC_RUNS/tito/20261015-fp8-dflash-collect/arc2/$3" --tasks-dir "$ARC_DATA/arc2_pools/$1" \
  --only "$2" --pool-label "$1" \
  --agent-python "$ARC_RUNS/arc_code/agent-runtime/bin/python" \
  --driver-bin "$(dirname "$ARC_CODEX_PREFIX")/arc2-jail" \
  --model-base-url http://127.0.0.1:$4/v1 --model qwen38fn \
  --tokenizer-json $ARC_NFS/models/Qwen3.8-Flash-Next-FP8/tokenizer.json \
  --concurrency $5 --arms ind,refine_exp,rerefine --solvers-first "${@:6}"
