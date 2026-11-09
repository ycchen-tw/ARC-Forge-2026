#!/bin/bash
# Download Qwen/Qwen3.8-Flash-Next-FP8 (185.5 GB, 131 shards) into $ARC_MODELS.
#
#   bash work/qwen38fn_eval/download_model.sh
#
# **FP8 only.** BF16 is 360 GB and /data has only 317 GB left; it does not fit.
# `local_dir` makes huggingface_hub write straight to the destination instead of keeping a second copy
# under $HF_HOME -- at this size there is no room for a second copy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

REPO=${REPO:-Qwen/Qwen3.8-Flash-Next-FP8}
DEST=${DEST:-$ARC_MODELS/$(basename "$REPO")}
PY=${PY:-$HOME/work/dsv4/venv/bin/python}

avail=$(df -BG --output=avail "$ARC_MODELS" | tail -1 | tr -dc '0-9')
[ "$avail" -ge 210 ] || { echo "✗ $ARC_MODELS has only ${avail}G free; FP8 needs 186G + headroom" >&2; exit 1; }

exec "$PY" - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=8), flush=True)
PY
