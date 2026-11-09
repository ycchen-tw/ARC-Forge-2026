#!/bin/bash
# plan B pool resume evidence again, after pool.py costs clean rows by context (inject 0.93e-6 s): same recipe
set -euo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_012v6VfWuCWC2aCEdDxWL17H
export TAG=20261019-plain-resume-check2 PLAN=$ARC_RUNS/qwen38fn_dflash/20261019-codex1015-small50-plan INIT=$ARC_RUNS/qwen38fn_dflash/20261019-plain-convert/step9425-plain-inject
export CAPTURE_GPUS=5 TRAINER_GPUS=0,3 BASE_PORT=30921 STOP_AT=12
export STEP_ANCHORS=89000 JOBS_PER_RANK=8 MAX_RANK_TOKENS=500000 MICRO_ANCHORS=4096 EVAL_EVERY=10 EVAL_ANCHORS=256
export LR_SCALE=2 PREFIX_GATE_ALPHA=0.01 PREFIX_GATE_TEMPERATURE=0.7 LR_SCHEDULE=wsd LR_DECAY_FRACTION=0.3
export COV_WEIGHT=0.05 COV_WEIGHT_FINAL=0.05
export ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-rl-int3 LOCK_TARGET=b200-rl-int3
export PYTHON=$HOME/work/venv-fn-train/bin/python TORCHRUN=$HOME/work/venv-fn-train/bin/torchrun
exec bash work/qwen38fn_dflash/probes/pool_resume_check.sh
