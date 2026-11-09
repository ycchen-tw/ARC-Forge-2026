#!/bin/bash
set -uo pipefail
cd $ARC_RUNS/qwen38fn_qad/20261023-kaggle-upload-R
echo "start $(date -u +%FT%TZ)"
~/.local/bin/uvx --from kaggle python upload_version_parallel.py threerabbits/qwen38fn-gean320-int3-gptq/transformers/default $ARC_RAM_RUNS/qwen38fn_qad/kaggle-upload-qadmix-R 6 "v3: student-trajectory QAD, reverse KL32_tail, 1000 steps (gEAN320 INT3 W3A8 trained on its own arc2/arc3 trajectories, 512E FP8 teacher top-32+tail). Same file names, dtypes, shapes and config.json as v1/v2; 708 tensors changed vs v1 (shards 1,2,11-18). PLE untouched. deploy_manifest.json files.sha256 updated, qad section records the delta. B200 arc2 eval120 pass@1 85.0 vs same-day base 81.6 and forward-KL 82.3 (5/4/5 rounds; ~20% of QAD data from eval-parent pools); refine_vote 94.0 vs base 90.7 (1 round); arc3 public-25 within noise of non-QAD."
rc=$?
echo "exit $rc $(date -u +%FT%TZ)"
