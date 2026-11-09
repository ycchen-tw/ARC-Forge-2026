#!/bin/bash
# Plan B formal run (2026-10-19): the plain-residual target-KV-injection drafter converted from D-line ext step 9425
# (convert_plain.py, step9425-plain-inject), trained on the codex1015 plan (arc2 + arc3, INT3 W3A8 capture).
# Recipe: prefix gate v3 (alpha 0.01, T 0.7), wsd with the last 30% decayed, LR_SCALE 2 (a fresh run, not a warm
# start), coverage weight 0.3 -> 0.05. GPUs 0-5: 3 trainer ranks + 3 capture servers -- the balanced split (2026-10-19
# resume checks: a rank ~9.7k anchors/s, a capture server ~9k anchors/s); the held-out is the one 9425 was scored on.
# RESUME=1 resumes (same settings; the rank count is fixed by the checkpoints).
set -euo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_012v6VfWuCWC2aCEdDxWL17H
R=$ARC_RUNS/qwen38fn_dflash
export TAG=20261019-plain-inject-codex1015
export PLAN=$R/20261017-codex1015-plan INIT=$R/20261019-plain-convert/step9425-plain-inject
export GATE=$R/20261019-plain-inject-gate/gate-acceptance.json
export HELDOUT_FROM=$ARC_RAM_RUNS/qwen38fn_dflash/20261019-codex1015-heldout
export TRAINER_GPUS=0,1,2 CAPTURE_GPUS=3,4,5 BASE_PORT=30931
export LR_SCALE=2 COV_WEIGHT=0.3 COV_WEIGHT_FINAL=0.05 LR_SCHEDULE=wsd LR_DECAY_FRACTION=0.3
export PREFIX_GATE_ALPHA=0.01 PREFIX_GATE_TEMPERATURE=0.7
export STEP_ANCHORS=89000 JOBS_PER_RANK=8 MAX_RANK_TOKENS=500000 MICRO_ANCHORS=4096
export SAVE_EVERY=1000 EVAL_EVERY=1000 EVAL_ANCHORS=1024
export RESUME=${RESUME:-0}
exec bash work/qwen38fn_dflash/probes/native_mtp_dense/launch_dline_int3.sh
