#!/bin/bash
# One round's rollout ledger + tito_manifest.json -> RL batch (tito-rl-v1). Batches with media need the deployed sglang
# processor: uses the sglang venv + fork (read-only, installs nothing), the same environment as prepare_tito_mm.sh.
# Arguments are passed through to prepare_rl.py (SPEC §7) unchanged.
#
#   MEDIA_DEVICE=cuda CUDA_VISIBLE_DEVICES=<UUID of one GPU> \
#       bash work/qwen38fn_lora/prepare_rl.sh --ledger <instance> --manifest <run>/tito_manifest.json [...] --out <dir> ...
#
# MEDIA_DEVICE=cuda is the deployed setup (HF processor on GPU); needs GPU authorization, exactly one GPU visible.
# MEDIA_DEVICE=cpu-exact does not touch the GPU: bicubic is a bit-exact reproduction of the CUDA kernel
# (cuda_bicubic_aa.py), frames identical to cuda.
# MEDIA_DEVICE=cpu-emulated does not touch the GPU; if any image is bicubic-resized prepare_rl raises unless
# --allow-emulated-media is also passed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

: "${MEDIA_DEVICE:?set MEDIA_DEVICE=cuda (as deployed, needs one GPU), cpu-exact or cpu-emulated}"
: "${ARC_SGLANG_FORK:?env/<machine>.sh does not set ARC_SGLANG_FORK}"
SGLANG_VENV=${SGLANG_VENV:-${ARC_SGLANG_VENV:?env/<machine>.sh does not set ARC_SGLANG_VENV}}
FFMPEG_DIR=${FFMPEG_DIR:-$(dirname "$(dirname "${ARC_FFMPEG:?env/<machine>.sh does not set ARC_FFMPEG}")")}

case $MEDIA_DEVICE in
    cuda)
        [[ -n ${CUDA_VISIBLE_DEVICES:-} && $CUDA_VISIBLE_DEVICES != *,* ]] \
            || { echo "✗ MEDIA_DEVICE=cuda needs CUDA_VISIBLE_DEVICES to be exactly one GPU (now '${CUDA_VISIBLE_DEVICES:-}')" >&2; exit 1; }
        nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" ;;
    cpu-exact|cpu-emulated) export CUDA_VISIBLE_DEVICES= ;;
    *) echo "✗ MEDIA_DEVICE=$MEDIA_DEVICE (only cuda/cpu-exact/cpu-emulated)" >&2; exit 1 ;;
esac
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ARC_SGLANG_FORK/python:$ARC_ROOT/src"

"$SGLANG_VENV/bin/python" - "$ARC_SGLANG_FORK" "$ARC_ROOT" <<'PYCHK'
import sys
from pathlib import Path

import arcforge
import sglang

fork, root = (Path(p).resolve() for p in sys.argv[1:])
if Path(sglang.__file__).resolve().parents[2] != fork:
    sys.exit(f"✗ sglang resolves to {sglang.__file__}, not {fork}")
if Path(arcforge.__file__).resolve().parents[2] != root:
    sys.exit(f"✗ arcforge resolves to {arcforge.__file__}, not {root}/src (editable install taking precedence?)")
PYCHK

exec "$SGLANG_VENV/bin/python" work/qwen38fn_lora/prepare_rl.py --media-device "$MEDIA_DEVICE" "$@"
