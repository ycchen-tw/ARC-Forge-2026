#!/bin/bash
# Starts the feeders (arc2 / arc3), offload (finished runs → NFS) and watch (anomalies and the /data brake), each as its
# own systemd user unit.
set -euo pipefail
R=$ARC_RUNS/tito/20261022-int3-qad-collect
W=$HOME/work/arc-forge-collect1022
mkdir -p /dev/shm/uid-$UID/arcforge-runs/tito/20261022-int3-qad-collect/lanes
for spec in "feeder-arc2:arc2/feeder.py:arc2/feeder.log" "feeder-arc3:arc3/feeder.py:arc3/feeder.log" \
            "offload:offload.py:offload.log" "watch:watch.py:watch.log"; do
  IFS=: read -r name py logf <<< "$spec"
  systemctl --user is-active -q "qadcol-$name" && { echo "qadcol-$name already running"; continue; }
  systemd-run --user --unit="qadcol-$name" --collect --working-directory="$R" \
    -p StandardOutput=append:"$R/$logf" -p StandardError=append:"$R/$logf" \
    /bin/bash -c "source $W/env/env.sh >/dev/null && exec python3 $R/$py"
done
