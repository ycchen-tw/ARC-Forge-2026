#!/bin/bash
# sglang for the node1 TITO collection (Qwen3.8-FN FP8 TP4 + MTP, fork main 5c99db5c, lock target b200-qwen38fn; in sync
# with main since 10-06).
#   bash launch_server.sh <tag> <gpus> <port>      e.g. bash launch_server.sh a 0,1,2,3 30241
# Ledger: ids + routing (the user decided on 2026-10-06 not to record logprobs). Both servers share one ledger root, each
# with its own instance directory.
set -euo pipefail
TAG=$1 GPUS=$2 PORT=$3
RUN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEDGER=$RUN/ledger
mkdir -p "$RUN/serve" "$LEDGER"
REPO=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_012e3ydKyiZL2YKdz5DAbzWn
cd "$REPO"
export LOCK_TARGET=b200-qwen38fn GPUS PORT
# For claiming: /proc/<pid>/environ carries this tag, a second ownership check besides pgrep
export ARC_TITO_SERVER_TAG=tito-node1-$TAG-p$PORT
echo $$ > "$RUN/serve/server-$TAG.pid"
exec bash work/sglang_fork/serve_b200.sh \
    --session-replay \
    --session-ledger-dir "$LEDGER" \
    --session-ledger-all \
    --session-ledger-routing \
    --enable-return-routed-experts \
    --no-mm-literal-image-sentinel
