#!/bin/bash
# serve.sh <arm> <port>: one gEAN320 INT3 W3A8 TP1 server on GPU $GPU (default 4) with native MTP (NEXTN steps 3 /
# topk 1 / 4 draft tokens, rejection sampling) from the arm's draft directory. Goes through the official chain
# (work/sglang_fork/serve_b200.sh -> work/qwen38fn_eval/serve_qwen38fn_0520.sh INT3_DEPLOY=1, the RL replica settings of
# launch_rollout_server.sh RL_MODEL_KIND=int3); the only change is the final launcher, swapped for patch/mtpd_launch.py
# by the fakevenv python shim (make_fakevenv.sh), which installs patch/mtpd_runtime.py in the workers after the INT3 plugin.
#   collect[-<x>]  mtp320-fp8[-<x>] draft, CUDA graphs off, the draft MoE inputs sampled (arms.sh arm_collect_dir)
#   moe320         mtp320-fp8 draft (baseline: the MTP MoE pruned to the target's 320 experts)
#   dense-ls       dense10240 draft, least-squares alpha
#   dense-uniform  dense10240 draft, alpha = 1/16
#   moe320-<x>     stacked drafts (copies, or a layer spec with target layers), dense-ls-<x>: see arms.sh
# Needs MTPD_RUN=<run dir> (drafts/, fakevenv/). Run it under systemd-run --user (run_arm.sh does).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
MODEL=$ARC_RAM_RUNS/qwen38fn_mtp_dense/qwen38fn-gean320-int3-gptq
arm=${1:?arm} port=${2:?port}
source "$HERE/arms.sh"
draft=$(arm_draft "$arm")
extra=()
collect_dir=$(arm_collect_dir "$arm")
if [ -n "$collect_dir" ]; then
    extra=(--disable-cuda-graph)
    export ARC_MTPD_COLLECT_DIR=$collect_dir ARC_MTPD_RATE_EXTEND=0.01 ARC_MTPD_MAX_ROWS=1000000
    [ ! -e "$ARC_MTPD_COLLECT_DIR" ] || { echo "✗ $ARC_MTPD_COLLECT_DIR exists" >&2; exit 1; }
fi
[ -f "$draft/config.json" ] || { echo "✗ no draft dir $draft" >&2; exit 1; }
[ -x "$T/fakevenv/bin/python" ] || { echo "✗ run make_fakevenv.sh first" >&2; exit 1; }
export ARC_MTPD=1 ARC_MTPD_DRAFT_DIR=$draft MTPD_LAUNCH=$HERE/patch/mtpd_launch.py
export SGLANG_VENV=$T/fakevenv ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-rl-int3 LOCK_TARGET=b200-rl-int3
export GPUS=${GPU:-4} PORT=$port NAME=mtpd-$arm MODEL=$MODEL
export TP=1 EP=1 MNNVL_HC=0 INT3_DEPLOY=1 KV_CACHE_DTYPE=fp8_e4m3 ARC_INT3_ACT=fp8 ARC_PLE_INT5=1
export SPEC=nextn SPEC_STEPS=3 MTP_DRAFT_MODEL=embedded MAXREQ=32 MEMFRAC=0.75 CTXLEN=65536
export MTP_DISTRIBUTED_TOPK1=0   # serve_qwen38fn_0520.sh maps it to SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1 (1 is incompatible with rejection sampling)
cd "$ROOT"
exec bash work/sglang_fork/serve_b200.sh \
    --dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend flashinfer_trtllm \
    --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \
    --speculative-draft-model-path "$draft" --speculative-draft-model-quantization fp8 \
    --speculative-moe-runner-backend flashinfer_trtllm --speculative-use-rejection-sampling \
    "${extra[@]}"
