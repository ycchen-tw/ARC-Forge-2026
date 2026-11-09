#!/bin/bash
# B200 deployment entry point for sglang-arc-2: first confirm that $ARC_SGLANG_FORK/python/sglang is the tree of
# LOCK_TARGET (default b200-qwen38fn) in deploy/sglang.lock.json, then layer it over venv-0520 via PYTHONPATH and hand
# off to the actual server start-up script.
#
#   GPUS=4,5,6,7 MODEL=<Qwen3.8-Flash-Next-FP8> bash work/sglang_fork/serve_b200.sh                  # FP8 KV, port 30032
#   SERVE=work/qwen38fn_eval/serve_qwen38fn_nvfp4_0520.sh GPUS=... MODEL=... bash work/sglang_fork/serve_b200.sh  # NVFP4 KV
#
# No separate venv: flashinfer's JIT command line contains the venv path, so switching venvs rebuilds all of
# $HOME/.cache/sglang (44 minutes the first time).
# Changing versions = `git -C $ARC_SGLANG_FORK checkout <commit>` plus `python work/sglang_fork/lock.py set b200-qwen38fn
# <ref>`; if the two disagree it will not start. Validation: work/sglang_fork/b200_validate.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

: "${ARC_SGLANG_FORK:?env/<machine>.sh does not set ARC_SGLANG_FORK}"
LOCK_TARGET=${LOCK_TARGET:-b200-qwen38fn}
SERVE=${SERVE:-work/qwen38fn_eval/serve_qwen38fn_0520.sh}
PY=${SGLANG_VENV:-$HOME/work/dsv4/venv-0520}/bin/python

python3 - "$LOCK_TARGET" "$ARC_SGLANG_FORK/python/sglang" <<'PYLOCK'
import sys

sys.path.insert(0, "src")
from arcforge.sglang_lock import check_tree

t = check_tree(sys.argv[1], sys.argv[2])
print(f"· sglang-arc-2 {t['commit'][:12]} (lock target {sys.argv[1]})")
PYLOCK

export PYTHONPATH=$ARC_SGLANG_FORK/python
# HUMMING_VERSION=<v>: layer $ARC_DATA/humming-overlays/<v> (a pinned wheel built by work/sglang_fork/humming_overlay.py
# build) without touching the venv's humming-kernels (0.1.12).
if [ -n "${HUMMING_VERSION:-}" ]; then
    overlay=$ARC_DATA/humming-overlays/$HUMMING_VERSION
    python3 work/sglang_fork/humming_overlay.py check "$HUMMING_VERSION" "$overlay"
    export PYTHONPATH=$overlay:$PYTHONPATH
    got=$("$PY" -c 'import importlib.metadata as m, humming; print(m.version("humming-kernels"), humming.__file__)')
    [[ $got == "$HUMMING_VERSION $overlay"/* ]] \
        || { echo "✗ HUMMING_VERSION=$HUMMING_VERSION but humming resolves to $got" >&2; exit 1; }
fi
origin=$("$PY" -c 'import sglang; print(sglang.__file__)')
[[ $origin == "$ARC_SGLANG_FORK"/python/* ]] \
    || { echo "✗ PYTHONPATH did not take precedence over the venv editable install: sglang resolves to $origin" >&2; exit 1; }
exec bash "$SERVE" "$@"
