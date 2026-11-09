#!/bin/bash
# width_sweep.sh <spec> <experts> ...: dense drafts of one layer spec (M for the one-layer MTP, or e.g. M,T35,T39,T43,T47)
# at other widths (experts x 640), each layer refitted with fit_dense.py --experts N (fits in $MTPD_RUN/width-<spec>/)
# on the rows collected on that spec's MoE draft (WIDTH_ROWS, default $MTPD_RUN/collect-<spec with _>/rows: the same
# positions, so no fit is reused across layers), built with build_drafts.py dense, one accept arm each
# (dense<W>-ls-<spec>), then accept_ci.py against WIDTH_REF (default dense-ls-<spec>, the width-10240 arm) into
# accept_ci_width_<spec>.json. GPU $GPU (default 4) must be free for the fits.
set -euo pipefail
main() {  # the whole body is a function, so an edit to this file cannot change a running sweep
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
export MTPD_RUN=$T GPU=${GPU:-4}
PY=$ARC_SGLANG_VENV/bin/python
DRAFTS=$ARC_RAM_RUNS/qwen38fn_mtp_dense/$(basename "$T")-drafts
spec=${1:?spec}; shift
[ $# -gt 0 ] || { echo "✗ no expert counts" >&2; exit 2; }
name=${spec//,/_}
ROWS=${WIDTH_ROWS:-$T/collect-$name/rows}
[ -d "$ROWS" ] || { echo "✗ no collection rows $ROWS" >&2; exit 1; }
C=$T/width-$name
mkdir -p "$C"
IFS=, read -r -a items <<< "$spec"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
arms=()
for n in "$@"; do
    w=$((n * 640))
    fits=()
    for i in "${!items[@]}"; do
        fit=$C/fit-L$i-e$n.json
        if [ ! -f "$fit" ]; then
            busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
            [ "$busy" = 0 ] || { log "✗ GPU $GPU busy before the fit"; exit 1; }
            (cd "$HERE" && CUDA_VISIBLE_DEVICES=$GPU "$PY" fit_dense.py --rows "$ROWS" --layer "$i" --source "${items[$i]}" \
                --experts "$n" --out "$fit") > "$C/fit-L$i-e$n.log" 2>&1
            log "fit layer $i (${items[$i]}) experts $n"
        fi
        fits+=("$fit")
    done
    d=dense$w-ls-$name
    if [ ! -e "$DRAFTS/$d" ]; then
        (cd "$HERE" && "$PY" build_drafts.py dense --spec "$spec" --fits "$(IFS=,; echo "${fits[*]}")" \
            --alpha least_squares --out "$DRAFTS/$d") > "$DRAFTS/$d.build.log"
        log "built $d"
    fi
    [ -e "$T/drafts/$d" ] || ln -s "$DRAFTS/$d" "$T/drafts/$d"
    arms+=("$d")
done
for a in "${arms[@]}"; do
    if [ -f "$T/serve/$a/accept.json" ]; then log "skip $a (done)"; continue; fi
    log "start $a"
    bash "$HERE/run_arm.sh" "$a" "$a"
    log "done $a"
done
cd "$ROOT/work/qsa_hisparse/probes"
ref=${WIDTH_REF:-dense-ls-$name}
"$PY" accept_ci.py --ref "$ref" $(for t in "$ref" "${arms[@]}"; do echo "$t=$T/serve/$t/accept.json"; done) \
    --json-out "$T/accept_ci_width_$name.json"
echo WIDTH-DONE
}
main "$@"
exit
