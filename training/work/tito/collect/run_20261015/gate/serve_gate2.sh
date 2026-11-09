#!/bin/bash
# Formal collection from 10-15 06Z: serve_gate.sh + top-32 ledger (the top32 logprobs QAD wants, T=1 original distribution).
# The fork is locked to b200-tito-topk in deploy/sglang.lock.json (sglang-arc-2 arc/ledger-topk 2da3f25f).
# Ledger layout ids_i4,lp_f2,tail_f4,flags_u1, 208 B per generated token (topk 197 B); rolled-and-closed topk.N.bin are moved to NFS by offload2.py.
# Arguments: $1=GPU pair (e.g. 0,1) $2=port $3=ledger dir $4=think budget (0 = off)
set -euo pipefail
WT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$WT"
source env/env.sh >/dev/null
export ARC_SGLANG_FORK=$ARC_SGLANG_FORK-ed837597 LOCK_TARGET=b200-tito-topk   # the sibling ed837597 checkout
export GPUS=$1 PORT=$2 TP=2 EP=2 MNNVL_HC=0 SPEC=dflash DFLASH_K=4 MEMFRAC=${MEMFRAC:-0.86} CTXLEN=262144 MAXREQ=${MAXREQ:-256}
export DFLASH_DIR=$ARC_RUNS/qwen38fn_rl/dflash-rollout/drafter/acc-v4/deploy-bf16
export MODEL=$ARC_NFS/models/Qwen3.8-Flash-Next-FP8
export ARC_TITO_SERVER_TAG=collect-p$2
export SGLANG_RETURN_ORIGINAL_LOGPROB=1
[ "$4" = 0 ] || export ARC_THINK_BUDGET=$4
mkdir -p "$3"
exec bash work/sglang_fork/serve_b200.sh \
    --session-replay --session-replay-empty-think \
    --session-ledger-dir "$3" --session-ledger-all \
    --session-ledger-logprobs --session-ledger-top-logprobs 32 \
    --no-mm-literal-image-sentinel --enable-metrics \
    --ple-mm-placeholder-ids \
    --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.05 \
    --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20 \
    --stream-interval 8 --scheduler-recv-interval 8
