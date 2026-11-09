#!/bin/bash
# serve_dline.sh <name> <port>: the gEAN320 INT3 W3A8 TP1 target on GPU $GPU (default 4) with the D-line drafter
# ${DLINE_DIR:-$MTPD_RUN/dline}/<name>-deploy-bf16 (make_dline_init.py) served by the fork's speculative/arc_dflash: SPEC=dflash of
# work/qwen38fn_eval/serve_qwen38fn_0520.sh, verify width DFLASH_K (default 4, the native MTP arms' 3 drafts + bonus).
# Target settings are the MTP arms' (serve.sh): INT3_DEPLOY, W3A8, PLE INT5, KV fp8, MAXREQ 32, MEMFRAC 0.75,
# CTXLEN 65536; no runtime patch (no fakevenv). ARC_DFLASH_EXACT_PREFILL=1 (default here): the drafter's window
# state after a prefill is computed from the prompt's last 5 x (1024 + 3) positions, not the default 2048-token tail.
# Fork: DLINE_FORK / DLINE_LOCK, default exp/dline-maskgr (lock b200-dline-maskgr: b200-rl-int3 + arc_dflash's opt-in
# Config.mask_gr, which a drafter config with "mask_gr" needs; one without it serves as on b200-rl-int3).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
name=${1:?name} port=${2:?port}
draft=${DLINE_DIR:-$T/dline}/$name-deploy-bf16
[ -f "$draft/config.json" ] || { echo "✗ no D-line deploy dir $draft" >&2; exit 1; }
export ARC_SGLANG_FORK=${DLINE_FORK:-$HOME/work/sglang-arc-2-dline-maskgr} LOCK_TARGET=${DLINE_LOCK:-b200-dline-maskgr}
export GPUS=${GPU:-4} PORT=$port NAME=dline-$name MODEL=$ARC_RAM_RUNS/qwen38fn_mtp_dense/qwen38fn-gean320-int3-gptq
export TP=1 EP=1 MNNVL_HC=0 INT3_DEPLOY=1 KV_CACHE_DTYPE=fp8_e4m3 ARC_INT3_ACT=fp8 ARC_PLE_INT5=1
export MAXREQ=32 MEMFRAC=0.75 CTXLEN=65536
export SPEC=dflash DFLASH_K=${DFLASH_K:-4} DFLASH_DIR=$draft ARC_DFLASH_EXACT_PREFILL=${ARC_DFLASH_EXACT_PREFILL:-1}
unset ARC_MTPD SGLANG_VENV
cd "$ROOT"
exec bash work/sglang_fork/serve_b200.sh \
    --dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend flashinfer_trtllm \
    --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}'
