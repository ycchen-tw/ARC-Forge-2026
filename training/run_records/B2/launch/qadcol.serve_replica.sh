#!/bin/bash
# One replica for the 20261022 QAD student-trajectory collection (node1 GPU 0–4, authorized by the user 10-22): gEAN320
# INT3 GPTQ W3A8 (HF @279402030, non-QAD) TP1, serve profile perf-main-fp16 (configure_b200.py: fork main ed837597 =
# lock main, no exact resume / own-last-turn), DFlash K4 acc-v4, CTXLEN 262144 (the arc2 collect config's note: with
# images the proxy's cap must fit), MAXREQ 96, memfrac 0.86, 640 mamba slots, Responses default temperature 0.7 (=
# sglang's original default; codex sends no sampling). Ledger as in the 10-15 collection: ids + media (no routing /
# logprobs).
#   bash serve_replica.sh <gpu> <port> <think budget, 0=off>      (systemd user unit qadcol-p<port>)
set -euo pipefail
gpu=${1:?gpu} port=${2:?port} tb=${3:?think budget}
case "$gpu" in 0|1|2|3|4) ;; *) echo "gpu $gpu is not in the granted 0-4" >&2; exit 64 ;; esac
W=$HOME/work/arc-forge-collect1022          # detached worktree pinned at e6d9c538 (same as 10-19)
source "$W/env/env.sh" >/dev/null
R=$ARC_RUNS/tito/20261022-int3-qad-collect
S=/dev/shm/uid-$UID/arcforge-runs/tito/20261022-int3-qad-collect
MAXREQ=${MAXREQ:-96}
unit=qadcol-p$port
log=$R/serve/p$port.log
ledger=$S/ledger/p$port
mkdir -p "$R/serve" "$ledger"
systemctl --user is-active --quiet "$unit" && { echo "unit $unit already running" >&2; exit 1; }
if ss -ltn "sport = :$port" | grep -q LISTEN; then echo "port $port taken" >&2; exit 1; fi
sizes=""
for s in 1 2 4 8 12 16 24 32 40 48 56 64 72 80 88 96; do [ "$s" -gt "$MAXREQ" ] || sizes+="${sizes:+,}$s"; done
env_kv=(
  # configure_b200.py SERVE_PROFILES perf-main-fp16 (graph sizes cut at MAXREQ)
  ARC_RS_FUSED_SOFTMAX=1 ARC_DFLASH_SPARSE_Q=1 ARC_QSA_FP8_DIRECT=1 ARC_GDN_FOLD_IN_VERIFY=1 ARC_GDN_FOLD_COMMIT_COMPACT=1
  ARC_HC_GATE_PRECOMPUTE=1 ARC_DFLASH_RING_OUT_T=1 ARC_DFLASH_RING_TILE=auto ARC_DFLASH_BATCHED_COMMIT=1
  ARC_DFLASH_FUSED_PROPOSE=1 ARC_DFLASH_FAST_TOPK=1 ARC_DFLASH_GRAPH_SIZES=$sizes ARC_QWEN4_EXP_PREFILL_BCG=1
  ARC_MAMBA_TRACK_HOST=1 ARC_MROPE_EXTEND_PINNED=1 ARC_QSA_INDEXER_LOCAL=1 ARC_QSA_UNION_PREFILL=1
  SSM_DTYPE=float16 LINEAR_DECODE_BACKEND=triton
  TP=1 EP=1 MNNVL_HC=0 MAXREQ=$MAXREQ DFLASH_K=4 MEMFRAC=0.86 MAMBA_CACHE_SLOTS=640 CTXLEN=262144
  ARC_RESPONSES_DEFAULT_TEMPERATURE=0.7 ARC_LOGITS_VOCAB_LIMIT=248077
  ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-allmerge-b200 LOCK_TARGET=main GPUS=$gpu PORT=$port SPEC=dflash
  DFLASH_DIR=$ARC_RUNS/qwen38fn_rl/dflash-rollout/drafter/acc-v4/deploy-bf16
  MODEL=/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq INT3_DEPLOY=1 KV_CACHE_DTYPE=fp8_e4m3 ARC_INT3_ACT=fp8 ARC_PLE_INT5=1
  ARC_TITO_SERVER_TAG=qadcol-p$port
)
[ "$tb" = 0 ] || env_kv+=(ARC_THINK_BUDGET=$tb)
setenv=()
for kv in "${env_kv[@]}"; do setenv+=(--setenv="$kv"); done
systemd-run --user --unit="$unit" --collect --working-directory="$W" "${setenv[@]}" \
  -p StandardOutput=append:"$log" -p StandardError=append:"$log" \
  /bin/bash "$W/work/sglang_fork/serve_b200.sh" \
    --session-replay --session-replay-empty-think --no-mm-literal-image-sentinel --enable-metrics \
    --dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend flashinfer_trtllm --ple-offload-embedding \
    --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \
    --stream-interval 8 --scheduler-recv-interval 8 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.1 \
    --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20 --ple-mm-placeholder-ids \
    --mamba-radix-cache-strategy extra_buffer \
    --session-ledger-dir "$ledger" --session-ledger-all
echo "started $unit (gpu $gpu, think budget $tb, MAXREQ $MAXREQ); log $log"
