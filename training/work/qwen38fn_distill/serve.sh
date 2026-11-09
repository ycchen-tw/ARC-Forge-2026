#!/bin/bash
# Qwen3.8-Flash-Next server for distillation. **Thin wrapper**; does not copy serve_qwen38fn.sh -- that script carries
# the whole patch-marker gate (0001…0015), and a copy would just be a chance for it to drift.
#
#   bash work/qwen38fn_distill/serve.sh 2>&1 | tee $ARC_RUNS/qwen38fn_distill/serve.log
#
# The only differences from the qwen38fn_eval script are four environment variables, each with a reason:
#
# 1. GPUS=4,5,6,7 -- GPUs 0-3 run Qwen3.6 SFT (shared machine). serve_qwen38fn.sh itself refuses to start if the
#    target GPUs have someone else's process; this only points the default at the right GPUs.
# 2. CTXLEN=45056 -- the runner's hard budget is 40,960 (prompt + generation + tool). The server's context-length only
#    needs to exceed it; 4k of margin for MTP draft tokens and template overhead.
#    **Lowering it from 262144 pays off**: QSA's index CUDA graph launches at a size fixed by the context cap, and with
#    short contexts many CTAs only do bounds checks (see the decode-step dissection §10).
# 3. SPEC_STEPS=3 -- specified value. QSA compression ratio=4 only supports 1..3, so 3 is the max;
#    num_draft_tokens is automatically steps+1=4.
#    ⚠️ This repo's matched TIR sweep measured steps=1 beating steps=2/3 at every bs1..96
#    (4k/32k, bs64/96 faster by 11~34%). 3 is used here as specified; smoke prints the measured tok/s for comparison.
# 4. MAXREQ=320 -- the runner's concurrency is **300 rollouts**, each with one request in flight at a time, so
#    max_running_requests must be ≥300, otherwise the excess silently sits in the queue
#    (symptom: "set to 300 but only 256 actually run", no error). 320 leaves margin.
#    The Mamba persistent cache follows at 320×5=1600 slots (serve_qwen38fn.sh's default rule), which takes some
#    away from the KV pool, so **re-check max_total_num_tokens after server start-up**.
#    At conc 200 the measured KV usage was steady at 0.21-0.39 (about 19.6k tokens per rollout on average, not the
#    cap's 40,960), so the expected usage of 300 rollouts is about 5.9M, still well below the pool.
#    The worst case (all at the cap) is 300×40,960=12.3M, which would overflow -- but measured usage never came close.
# 5. SPEC_STEPS=3 is serve_qwen38fn.sh's new default (since 2026-08-30) and is no longer overridden here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

export GPUS=${GPUS:-4,5,6,7}
export CTXLEN=${CTXLEN:-45056}
export SPEC_STEPS=${SPEC_STEPS:-3}
export MAXREQ=${MAXREQ:-320}
export PORT=${PORT:-30041}
export NAME=${NAME:-qwen38fn-distill}

echo "· distill server: GPUS=$GPUS PORT=$PORT CTXLEN=$CTXLEN SPEC_STEPS=$SPEC_STEPS MAXREQ=$MAXREQ"
exec bash work/qwen38fn_eval/serve_qwen38fn.sh "$@"
