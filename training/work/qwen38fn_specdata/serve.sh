#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
export MODEL=${MODEL:?set MODEL to the full Qwen3.8-Flash-Next-FP8 checkpoint}
TAG=${TAG:?set a unique YYYYMMDD-tag}
RUN=$ARC_RUNS/qwen38fn_specdata/$TAG
mkdir -p "$RUN"
export GPUS=${GPUS:-4,5,6,7} TP=${TP:-4} EP=${EP:-4}
export PORT=${PORT:-30471} NAME=qwen38fn-specdata
export CTXLEN=${CTXLEN:-45056} MAXREQ=300 MEMFRAC=${MEMFRAC:-0.85}
export SSM_DTYPE=float32 LINEAR_DECODE_BACKEND=triton
export SPEC=nextn SPEC_STEPS=3 SPEC_DRAFT_TOK=4 MAMBA_CACHE_SLOTS=1500
export SGLANG_VENV=${SGLANG_VENV:-$HOME/work/dsv4/venv-qwen4}
if ss -lntH "sport = :$PORT" | rg -q .; then
    echo "port $PORT occupied" >&2
    exit 1
fi
"$SGLANG_VENV/bin/python" - <<'PY'
import inspect
from sglang.srt.speculative import spec_utils
source = inspect.getsource(spec_utils.commit_mamba_states_after_verify)
assert 'commit_ple_states_after_replayssm(' in source, 'apply patch 0018 before ReplaySSM'
PY
exec bash work/qwen38fn_eval/serve_qwen38fn.sh \
    --enable-linear-replayssm-spec --linear-attn-verify-backend triton \
    --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0 \
    "$@" >> "$RUN/serve.log" 2>&1
