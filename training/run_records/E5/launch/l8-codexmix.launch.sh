#!/bin/bash
# L8 main run (2026-10-20, plan docs/plans/dflash_l8_expand_20261020.md): the init from plan B 8251 deepened to 8 layers
# (copied, L8-copy), trained further on 20261020-l8-plan (codex1015 + 1019 INT3 + bench base / PLE + node1, 2.248 B
# anchors, one pass). Recipe as in the 9425 extension: LR_SCALE 1, wsd with the last 30% decayed, coverage 0.05, prefix
# gate v3 (α 0.01, T 0.7). The held-out set is the 105 jobs plan B / 9425 were scored on (--heldout-from), features reused.
# TRAINER_GPUS / CAPTURE_GPUS are filled in once the user grants cards (split decided from the resume-check measurements).
# RESUME=1 resumes.
set -euo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_017j1AQDBo8pGaN4Xzc733yh
R=$ARC_RUNS/qwen38fn_dflash
export TAG=20261020-l8-codexmix
export PLAN=$R/20261020-l8-plan INIT=$R/20261020-l8-expand/step8251-L8-copy
export GATE=$R/20261020-l8-gate/gate-acceptance.json
export HELDOUT_FROM=$ARC_RAM_RUNS/qwen38fn_dflash/20261019-codex1015-heldout
: "${TRAINER_GPUS:?}" "${CAPTURE_GPUS:?}"
export TRAINER_GPUS CAPTURE_GPUS BASE_PORT=30931
export LR_SCALE=1 COV_WEIGHT=0.05 COV_WEIGHT_FINAL=0.05 LR_SCHEDULE=wsd LR_DECAY_FRACTION=0.3
export PREFIX_GATE_ALPHA=0.01 PREFIX_GATE_TEMPERATURE=0.7
export STEP_ANCHORS=89000 JOBS_PER_RANK=8 MAX_RANK_TOKENS=500000 MICRO_ANCHORS=4096
export SAVE_EVERY=1000 EVAL_EVERY=1000 EVAL_ANCHORS=1024
export RESUME=${RESUME:-0}
exec bash work/qwen38fn_dflash/probes/native_mtp_dense/launch_dline_int3.sh
