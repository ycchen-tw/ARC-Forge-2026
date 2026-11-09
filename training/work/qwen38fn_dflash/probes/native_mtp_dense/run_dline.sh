#!/bin/bash
# run_dline.sh <name> [tag]: (DLINE_DIR: where the <name>-deploy-bf16 exports are, default $MTPD_RUN/dline;
# DLINE_FORK / DLINE_LOCK: passed to serve_dline.sh, which has the defaults)
# serve_dline.sh <name> as systemd user unit mtpd-<tag> (default tag dline-<name>-k$K),
# wait until /server_info shows the INT3 model with NEXTN and the server log has arc_dflash's "DFlash drafter <deploy
# dir>" line, run the same accept client as run_arm.sh's eval arms (eval prompts, seeds 0/1, 512 new tokens, conc 16,
# T=1 top_p=1), stop the unit (only this unit).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
PROBES=$ROOT/work/qsa_hisparse/probes
PY=$ARC_SGLANG_VENV/bin/python
MODEL=$ARC_RAM_RUNS/qwen38fn_mtp_dense/qwen38fn-gean320-int3-gptq
GPU=${GPU:-4} K=${DFLASH_K:-4}
name=${1:?name} tag=${2:-dline-$1-k$K}
DLINE=${DLINE_DIR:-$T/dline}
draft=$DLINE/$name-deploy-bf16
port=31747
unit=mtpd-$tag
out=$T/serve/$tag
mkdir -p "$out"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { log "GPU $GPU busy"; nvidia-smi -i "$GPU"; exit 1; }
! curl -s -m 3 http://127.0.0.1:$port/health >/dev/null 2>&1 || { log "port $port already answers"; exit 1; }
systemctl --user reset-failed "$unit" 2>/dev/null || true
systemd-run --user --unit "$unit" --setenv=MTPD_RUN="$T" --setenv=GPU="$GPU" --setenv=DFLASH_K="$K" --setenv=DLINE_DIR="$DLINE" \
    --setenv=DLINE_FORK="${DLINE_FORK:-}" --setenv=DLINE_LOCK="${DLINE_LOCK:-}" \
    -p StandardOutput=append:"$out/server.log" -p StandardError=append:"$out/server.log" \
    bash "$HERE/serve_dline.sh" "$name" $port
trap 'systemctl --user stop $unit 2>/dev/null || true' EXIT
pid=$(systemctl --user show -p MainPID --value "$unit")
echo "$pid" > "$out/unit.pid"
log "unit $unit main pid $pid"
deadline=$((SECONDS + 3600))
until curl -fsS -m 5 http://127.0.0.1:$port/server_info 2>/dev/null > "$out/server_info.json" \
        && "$PY" -c "
import json
s=json.load(open('$out/server_info.json'))
assert s['model_path']=='$MODEL', s['model_path']
assert s['speculative_algorithm'] in ('EAGLE','NEXTN'), s['speculative_algorithm']
assert s['speculative_num_draft_tokens']==$K, s['speculative_num_draft_tokens']" 2>/dev/null; do
    systemctl --user is-active --quiet "$unit" || { log "unit died"; tail -40 "$out/server.log"; exit 1; }
    [ $SECONDS -lt $deadline ] || { log "not ready in 1 h"; exit 1; }
    sleep 10
done
grep -q "DFlash drafter $draft " "$out/server.log" || { log "✗ server is up but arc_dflash did not load $draft"; exit 1; }
log "ready"
grep -E "DFlash drafter|DFlash exact prefill|DFlash draft CUDA graphs|ARC_INT3_DIRECT_LOADED" "$out/server.log" | cut -c1-400 | sort | uniq -c
cd "$PROBES"
ACCEPT_SAMPLES=2 ACCEPT_MAX_NEW=512 ACCEPT_CONC=16 "$PY" "$HERE/accept_probe_tol.py" $port "$pid" "$T/prompts/eval.json" "$out/accept.json"
curl -fsS -m 10 http://127.0.0.1:$port/server_info > "$out/server_info_end.json" || true
log "done"
