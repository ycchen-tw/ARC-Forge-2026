#!/bin/bash
# Resume 20261017-dline-ext-codex1015 (RESUME=1, the run's own settings) once GPU 0-3 are free: every one at 0 MiB and
# either the go file exists (touched when the session using them says it is done) or they have stayed at 0 MiB for
# 15 minutes (a gap between another job's phases is shorter).
set -euo pipefail
R=$ARC_RUNS/qwen38fn_dflash
T=20261017-dline-ext-codex1015
WT=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_01PPZmfivQs1NA5YdZg7G9V4
GO=$R/$T.resume-now
free_since=
while true; do
    used=$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')
    if [ "$used" = 0 ]; then
        [ -n "$free_since" ] || free_since=$SECONDS
        if [ -f "$GO" ] || ((SECONDS - free_since >= 900)); then
            break
        fi
    else
        free_since=
    fi
    sleep 20
done
avail=$(($(df --output=avail -k /data | tail -1) / 1024 / 1024))
((avail >= 30)) || { echo "✗ /data has $avail GiB free (< 30): not resuming"; exit 1; }
echo "$(date -u +%FT%TZ) GPU 0-3 free (go file: $([ -f "$GO" ] && echo yes || echo no)); /data $avail GiB free; resuming"
cd "$WT"
RESUME=1 TAG=$T PLAN=$R/20261017-codex1015-plan GATE=$R/20261017-codex1015-fast-gate-2/gate-acceptance.json \
    CAPTURE_GPUS=2,3 TRAINER_GPUS=0,1 LR_SCALE=1 COV_WEIGHT=0.05 COV_WEIGHT_FINAL=0.05 LR_DECAY_FRACTION=0.3 \
    WARM_START=$ARC_RAM_RUNS/qwen38fn_dflash/20261017-codex1015/parent-dline-v3wsd-step-000034404 \
    WARM_NEW_DATA=1 REWARM_STEPS=500 BASE_PORT=30801 \
    bash work/qwen38fn_dflash/probes/native_mtp_dense/launch_dline_int3.sh
