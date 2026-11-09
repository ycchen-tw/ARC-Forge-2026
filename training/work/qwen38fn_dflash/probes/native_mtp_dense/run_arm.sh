#!/bin/bash
# run_arm.sh <arm> [tag]: start serve.sh <arm> as systemd user unit mtpd-<tag> on GPU $GPU (default 4), wait until
# /server_info shows this arm's model and draft, run the accept client, stop the unit (only this unit).
#   collect[-<x>]: fit prompts (128), 1 sample, 1024 new tokens, conc 32, accept_probe.py (accept numbers are eager-mode, informational)
#   others:  eval prompts (64), seeds 0,1, 512 new tokens, conc 16, T=1 top_p=1, accept_probe_tol.py
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
PROBES=$ROOT/work/qsa_hisparse/probes
PY=$ARC_SGLANG_VENV/bin/python
MODEL=$ARC_RAM_RUNS/qwen38fn_mtp_dense/qwen38fn-gean320-int3-gptq
GPU=${GPU:-4}
arm=${1:?arm} tag=${2:-$1}
port=31747
unit=mtpd-$tag
out=$T/serve/$tag
mkdir -p "$out"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { log "GPU $GPU busy"; nvidia-smi -i "$GPU"; exit 1; }
! curl -s -m 3 http://127.0.0.1:$port/health >/dev/null 2>&1 || { log "port $port already answers"; exit 1; }
systemctl --user reset-failed "$unit" 2>/dev/null || true
systemd-run --user --unit "$unit" --setenv=MTPD_RUN="$T" --setenv=GPU="$GPU" \
    -p StandardOutput=append:"$out/server.log" -p StandardError=append:"$out/server.log" \
    bash "$HERE/serve.sh" "$arm" $port
trap 'systemctl --user stop $unit 2>/dev/null || true' EXIT
pid=$(systemctl --user show -p MainPID --value "$unit")
echo "$pid" > "$out/unit.pid"
log "unit $unit main pid $pid"
source "$HERE/arms.sh"
draft=$(arm_draft "$arm")
deadline=$((SECONDS + 3600))
until curl -fsS -m 5 http://127.0.0.1:$port/server_info 2>/dev/null > "$out/server_info.json" \
        && "$PY" -c "
import json
s=json.load(open('$out/server_info.json'))
assert s['model_path']=='$MODEL', s['model_path']
assert s['speculative_draft_model_path']=='$draft', s['speculative_draft_model_path']
assert s['speculative_algorithm'] in ('EAGLE','NEXTN'), s['speculative_algorithm']" 2>/dev/null; do
    systemctl --user is-active --quiet "$unit" || { log "unit died"; tail -40 "$out/server.log"; exit 1; }
    [ $SECONDS -lt $deadline ] || { log "not ready in 1 h"; exit 1; }
    sleep 10
done
log "ready"
grep -E 'ARC_MTPD|ARC_INT3_DIRECT_LOADED|fakevenv' "$out/server.log" | cut -c1-300 | sort | uniq -c | head -20
cd "$PROBES"
if [ -n "$(arm_collect_dir "$arm")" ]; then
    ACCEPT_SAMPLES=1 ACCEPT_MAX_NEW=1024 ACCEPT_CONC=32 "$PY" accept_probe.py $port "$pid" "$T/prompts/fit.json" "$out/accept.json"
else
    ACCEPT_SAMPLES=2 ACCEPT_MAX_NEW=512 ACCEPT_CONC=16 "$PY" "$HERE/accept_probe_tol.py" $port "$pid" "$T/prompts/eval.json" "$out/accept.json"
fi
curl -fsS -m 10 http://127.0.0.1:$port/server_info > "$out/server_info_end.json" || true
log "done"
