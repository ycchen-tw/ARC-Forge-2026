#!/bin/bash
# Adds GPU 7 as an eval backend once seed collection has ended (collect.log shows collection complete and the collector
# unit has stopped).
#   bash eval_join_g7.sh <eval root>
set -euo pipefail
E=${1:?eval root}
R=$HOME/work/runs/arc_code/20261021-g2seed
# The collector is also briefly absent when restarted midway (version change): wait until the log says collection
# complete and the unit has stopped
until grep -q "collection complete" "$R/collect/collect.log" && ! systemctl --user is-active --quiet g2seed-collect-g7; do
    sleep 60
done
curl -fsS --max-time 30 http://127.0.0.1:30474/health > /dev/null
echo "http://127.0.0.1:30474@7" >> "$E/backends.txt"
echo "$(date -u +%FT%TZ) GPU 7 joined the eval"
