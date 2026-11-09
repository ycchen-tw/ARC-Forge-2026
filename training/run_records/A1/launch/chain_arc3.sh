#!/bin/bash
# Waits for the arc3 smoke run to finish (it has its own 1500 s wall), checks that it wrote tito_manifest.json, then starts
# the arc3 lane (3 rounds x 3 h). If the smoke run wrote no manifest, the lane is not started and lanes/arc3.log says why.
set -uo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$R/lanes"
log() { echo "$(date -u +%FT%TZ) [arc3] $*" >> "$R/lanes/arc3.log"; }
log "chain: waiting for arc3 smoke"
while pgrep -f -- "--run-dir $R/smoke/arc3-smoke" > /dev/null; do sleep 30; done
if [ ! -f "$R/smoke/arc3-smoke/tito_manifest.json" ]; then log "stop: arc3 smoke wrote no tito_manifest.json -> not starting the arc3 lane"; exit 1; fi
log "chain: arc3 smoke done ($(tail -1 "$R/smoke/arc3-smoke.out" | cut -c1-160)) -> starting lane"
exec bash "$R/lane.sh" arc3 arc3 30241 3
