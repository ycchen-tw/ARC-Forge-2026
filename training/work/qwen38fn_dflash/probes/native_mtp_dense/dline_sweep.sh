#!/bin/bash
# dline_sweep.sh <spec> ...: D-line slot-0 check of dense init candidates (specs with _, e.g. M_T35_T39_T43_T47).
# For each: make_dline_init.py on $MTPD_RUN/drafts/dense10240-ls-<spec> into $DLINE_DIR/<spec>-{init,deploy-bf16}
# (default DLINE_DIR $MTPD_RUN/dline; skipped if the deploy dir exists), run_dline.sh <spec> (skipped if its accept.json
# exists), then accept_ci.py over the D-line arms against the first one (accept_ci_dline_$DLINE_TAG.json).
set -euo pipefail
main() {  # the whole body is a function, so an edit to this file cannot change a running sweep
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
T=${MTPD_RUN:?MTPD_RUN=<run dir>}
export MTPD_RUN=$T GPU=${GPU:-4} DLINE_DIR=${DLINE_DIR:-$T/dline} DFLASH_K=${DFLASH_K:-4}
PY=$ARC_SGLANG_VENV/bin/python
[ $# -gt 0 ] || { echo "✗ no specs" >&2; exit 2; }
mkdir -p "$DLINE_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
tags=()
for s in "$@"; do
    if [ ! -d "$DLINE_DIR/$s-deploy-bf16" ]; then
        (cd "$HERE" && "$PY" make_dline_init.py --draft "$T/drafts/dense10240-ls-$s" --init "$DLINE_DIR/$s-init" \
            --deploy "$DLINE_DIR/$s-deploy-bf16") > "$DLINE_DIR/$s.convert.log"
        log "converted $s"
    fi
    tag=dline-$s-k$DFLASH_K
    tags+=("$tag")
    if [ -f "$T/serve/$tag/accept.json" ]; then log "skip $tag (done)"; continue; fi
    log "start $tag"
    bash "$HERE/run_dline.sh" "$s" "$tag"
    log "done $tag"
done
cd "$ROOT/work/qsa_hisparse/probes"
"$PY" accept_ci.py --ref "${tags[0]}" $(for t in "${tags[@]}"; do echo "$t=$T/serve/$t/accept.json"; done) \
    --json-out "$T/accept_ci_dline_${DLINE_TAG:-sweep}.json"
echo SWEEP-DONE
}
main "$@"
exit
