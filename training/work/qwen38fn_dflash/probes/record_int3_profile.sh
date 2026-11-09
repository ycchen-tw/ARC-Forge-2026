#!/bin/bash
# A capture target profile of the gEAN320 INT3 W3A8 deployment: the server.json SGLang's session ledger writes at
# startup, from a server started with work/arc_code/duckx/serve_int3.sh's target flags (= the RL int3 rollout's) minus
# the drafter, plus a session ledger. Use it as make_data_plan.py --target-profile / capture_tito.sh TARGET_PROFILE
# (with ARC_SGLANG_FORK=<the rl-int3 fork> LOCK_TARGET=b200-rl-int3).
#
#   OUT=<new dir> MODEL=<INT3 package> [GPU=4] [PORT=31749] bash work/qwen38fn_dflash/probes/record_int3_profile.sh
#
# Runs the server as systemd user unit int3-profile-$PORT, waits for /server_info (the flags really start a server),
# copies the server.json to $OUT/profile-int3-w3a8-tp1.json, stops the unit (only that one).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
ROOT=$PWD
export ARC_ROOT=$ROOT
source env/env.sh
: "${OUT:?OUT=<new dir>}" "${MODEL:?MODEL=<gEAN320 INT3 package>}"
GPU=${GPU:-4} PORT=${PORT:-31749}
FORK=${INT3_FORK:-$HOME/work/sglang-arc-2-rl-int3}
grep -q '"arc_int3"' "$MODEL/config.json" || { echo "✗ $MODEL/config.json has no arc_int3 (not an INT3 package)" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "✗ $OUT exists" >&2; exit 1; }
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { echo "✗ GPU $GPU busy" >&2; exit 1; }
! curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "✗ port $PORT answers" >&2; exit 1; }
mkdir -p "$OUT/ledger"
unit=int3-profile-$PORT
systemctl --user reset-failed "$unit" 2>/dev/null || true
systemd-run --user --unit "$unit" -p StandardOutput=append:"$OUT/server.log" -p StandardError=append:"$OUT/server.log" \
    --setenv=ARC_SGLANG_FORK="$FORK" --setenv=LOCK_TARGET=b200-rl-int3 --setenv=GPUS="$GPU" --setenv=PORT="$PORT" \
    --setenv=NAME=int3-profile --setenv=MODEL="$MODEL" --setenv=TP=1 --setenv=EP=1 --setenv=MNNVL_HC=0 \
    --setenv=CTXLEN=262144 --setenv=MAXREQ=32 --setenv=MEMFRAC=0.75 --setenv=INT3_DEPLOY=1 --setenv=ARC_INT3_ACT=fp8 \
    --setenv=ARC_PLE_INT5=1 --setenv=KV_CACHE_DTYPE=fp8_e4m3 --setenv=SPEC=none \
    bash "$ROOT/work/sglang_fork/serve_b200.sh" \
    --dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend flashinfer_trtllm \
    --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \
    --ple-mm-placeholder-ids --no-mm-literal-image-sentinel --image-processor-backend pil \
    --session-replay --session-ledger-dir "$OUT/ledger" --session-ledger-all
trap 'systemctl --user stop "$unit" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 3600))
until curl -fsS -m 5 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json" 2>/dev/null; do
    systemctl --user is-active --quiet "$unit" || { echo "✗ server died"; tail -40 "$OUT/server.log"; exit 1; }
    [ $SECONDS -lt $deadline ] || { echo "✗ not ready in 1 h"; exit 1; }
    sleep 10
done
recorded=$(ls "$OUT"/ledger/*/server.json)
[ "$(wc -l <<< "$recorded")" = 1 ] || { echo "✗ expected one ledger instance: $recorded" >&2; exit 1; }
cp "$recorded" "$OUT/profile-int3-w3a8-tp1.json"
echo "· profile $OUT/profile-int3-w3a8-tp1.json (instance $(basename "$(dirname "$recorded")"))"
