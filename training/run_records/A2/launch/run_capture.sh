#!/bin/bash
# full REAP capture of plan-v1 against the server on GPU 4,5 (port 30690); one split after another
set -euo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/reap-mm/work/qwen38fn_dflash
for split in diag calib heldout; do
  check=0; [ $split = calib ] && check=20
  df -BG --output=avail /data | tail -1 | awk "{ if (\$1+0 < 40) { print \"✗ /data below 40G\"; exit 1 } }"
  PYTHONPATH=$HOME/work/arc-forge/.claude/worktrees/reap-mm/src ~/work/arc-forge/.venv/bin/python reap_capture.py --plan $ARC_RUNS/qwen38fn_eval/20261009-reap-mm/plan-v1/plan.jsonl --split $split \
    --url http://127.0.0.1:30690 --root $ARC_RAM_RUNS/qwen38fn_eval/20261009-reap-mm/root --out $ARC_RUNS/qwen38fn_eval/20261009-reap-mm/jobs/$split --ledger-root $ARC_RAM_RUNS/qwen38fn_eval/20261009-reap-mm/ledger --image-detail high \
    --check-routing $check
done
echo ALL_DONE
