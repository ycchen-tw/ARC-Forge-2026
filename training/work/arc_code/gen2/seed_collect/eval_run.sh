#!/bin/bash
# The whole seed difficulty eval (run inside a systemd unit): evaluate.py runs until everything is dispatched, finished
# and summarized, then the model servers used for this eval (g2seed-srv-g6 / g2seed-srv-g7, both started by this line of
# work itself) are stopped to free the GPUs.
#   bash eval_run.sh <eval root> <commit> [runs per seed, default 1]       # run in the pinned worktree
set -euo pipefail
E=${1:?eval root} C=${2:?commit} REPEATS=${3:-1}
R=$HOME/work/runs/arc_code/20261021-g2seed
source env/env.sh >/dev/null
.venv/bin/python work/arc_code/gen2/seed_collect/evaluate.py --root "$E" --worktree "$PWD" --commit "$C" \
    --tokenizer "$R/tokenizer.json" --backend-file "$E/backends.txt" --collect-log "$R/collect/collect.log" --repeats "$REPEATS"
echo "$(date -u +%FT%TZ) eval done: stopping g2seed-srv-g6 g2seed-srv-g7"
systemctl --user stop g2seed-srv-g6 g2seed-srv-g7
