#!/bin/bash
# Keep a hardlink copy (<run>/pre-decay) of the last trainer checkpoint saved before the WSD decay: the trainer keeps
# only its latest checkpoint. Exits once a checkpoint is past EDGE (progress = trained anchors over the plan's) or the
# pipeline unit ends.
#
#   bash keep_predecay.sh <run dir> <pipeline unit> <edge, e.g. 0.7>
set -euo pipefail
OUT=$1 UNIT=$2 EDGE=$3
T=$OUT/trainer KEEP=$OUT/pre-decay
last=
while systemctl --user is-active --quiet "$UNIT"; do
    if [ -f "$T/latest" ]; then
        cur=$(cat "$T/latest")
        if [ "$cur" != "$last" ] && [ -d "$T/$cur" ]; then
            n=$((10#${cur#step-}))
            p=$(python3 - "$T/metrics.jsonl" "$n" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
hit = [r for r in rows if r["step"] == int(sys.argv[2])]
if len(hit) != 1:
    raise SystemExit(f"metrics.jsonl has {len(hit)} rows for step {sys.argv[2]}")
print(hit[0]["progress"])
PY
)
            if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' "$p" "$EDGE"; then
                rm -rf "$KEEP.new"
                cp -al "$T/$cur" "$KEEP.new"
                rm -rf "$KEEP"
                mv "$KEEP.new" "$KEEP"
                echo "$(date -u +%FT%TZ) kept $cur (progress $p)"
            else
                echo "$(date -u +%FT%TZ) $cur is past the decay edge (progress $p >= $EDGE): done"
                exit 0
            fi
            last=$cur
        fi
    fi
    sleep 60
done
echo "$(date -u +%FT%TZ) $UNIT ended"
