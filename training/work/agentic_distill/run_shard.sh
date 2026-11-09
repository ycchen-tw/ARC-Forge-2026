#!/bin/bash
# Collect a distillation corpus on NVARC (nvarc_training) with agentic_tir's compact v2 harness; one shard per server.
#
#   TAG=pilot SHARD=a PORT=30031 bash work/agentic_distill/run_shard.sh            # reads lists/pilot_a.txt
#   TAG=full  SHARD=b PORT=30032 LIVE=96 TLIVE=48 bash work/agentic_distill/run_shard.sh
#
# - Each task's effort / ctx_cap / K / target_frac / gen_budget is drawn deterministically by --hparam-menu
#   (agentic_tir.HPARAM_MENU); everything else is fixed to the compact v2 settings (--compact --max-hops 1;
#   checkpoint / rich / structured notes / delegate off).
# - --save-segments: the full messages before each compaction are written to segments/<uid>.jsonl.gz; render_segments.py
#   later converts them to parquet.
# - --resume: a rerun skips uids already in tasks/, so the watchdog can restart safely.
# - Server: work/qwen38fn_eval/serve_qwen38fn.sh (GPUS=0-3 or 4-7, CTXLEN=131072); run preflight once it is up.
# - LIVE is the number of episodes in flight (each has one request at a time); must be ≤ the server's MAXREQ (default 256).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

D=$HOME/work/dsv4
PY=${PY:-$D/venv/bin/python}
PORT=${PORT:?}
SHARD=${SHARD:?"a or b"}
TAG=${TAG:?"pilot / full"}
LIVE=${LIVE:-96}
TLIVE=${TLIVE:-48}
SERVED=${SERVED:-qwen38fn}
TOKENIZER=${TOKENIZER:-$ARC_NFS/models/Qwen3.8-Flash-Next-FP8}
CORPUS=${CORPUS:-"$D/nvarc_distill/nvarc_training.jsonl.zst $HOME/work/data/nvarc_synthetic/nvarc_full.jsonl.zst"}   # may be several, space-separated; each task records its source
BACKEND=${BACKEND:-tokens}      # token-in/token-out (work/agentic_tir/token_backend.py); openai = chat API (comparison only)
R=$ARC_RUNS/agentic_distill
LIST=${LIST:-$R/lists/${TAG}_${SHARD}.txt}
OUT=${OUT:-$R/$TAG-$SHARD}

[ -f "$LIST" ] || { echo "✗ no $LIST" >&2; exit 1; }
curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "✗ no server on port $PORT" >&2; exit 1; }
mkdir -p "$OUT"
PYTHONPATH="$ARC_ROOT/src" exec "$PY" work/agentic_tir/agentic_tir.py \
    --task-source nvarc --corpus $CORPUS --task-list "$LIST" --n-train 4 --n-test 1 \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED" --tokenizer "$TOKENIZER" \
    --backend "$BACKEND" --compact --max-hops 1 --max-turns 200 --base-explorers 2 \
    --hparam-menu --save-segments --resume \
    --max-live "$LIVE" --task-live "$TLIVE" \
    --out "$OUT" "$@"
