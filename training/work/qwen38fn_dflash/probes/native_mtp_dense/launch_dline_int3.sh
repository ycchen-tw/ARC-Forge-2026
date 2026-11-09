#!/bin/bash
# The D-line training run from the native-MTP init (dense10240, mask_gr none, mask embed[248070]) with INT3 W3A8
# capture: pipeline.sh as a systemd user unit.
#
#   TAG=<run tag> PLAN=<build_dline_int3_plan.sh dir> GATE=<fast_gates.py report> CAPTURE_GPUS=4,5,6 \
#       TRAINER_GPUS=0,1,2,3 [INIT=<init dir>] [RESUME=1] bash work/qwen38fn_dflash/probes/native_mtp_dense/launch_dline_int3.sh
#
# INIT default: the NFS copy of 20261011-int3-target47/dline/M_T35_T39_T43_T47-m70-none-init. Plan B's plain-residual
# KV-injection init (convert_plain.py: residual plain, ffn_type dense) is accepted too; REDUCER_LR then has no group.
# Settings are v5's (20261008-dense10240-rand-v5.launch.sh) except REDUCER_LR 3e-5 (this init's output reducer is the
# native MTP's; v5's random one took 1e-4) and the 2026-10-13 recipe: prefix gate v3 (alpha 0.01, T 0.7, on from the
# first step), LR wsd (3% warm-up, linear to 0 over the last 20%), coverage weight 0.3 -> 0.05 linear over the run.
# Every one can be overridden from the environment. GATE must come from fast_gates.py on a prefix-gate resume check.
# Capture servers serve the INT3 package: ARC_SGLANG_FORK=${INT3_FORK:-$HOME/work/sglang-arc-2-rl-int3},
# LOCK_TARGET=b200-rl-int3 (capture_tito.sh checks the tree). PYTHON / TORCHRUN: a venv with torch (default node1's
# $HOME/work/venv-fn-train). Checkpoints record the world size: a resume needs the same TRAINER_GPUS count.
# WARM_START=<checkpoint dir> [WARM_NEW_DATA=1] [REWARM_STEPS=500]: pipeline.sh's warm start (an extension run).
# HELDOUT_FROM=<a finished held-out capture of the same PLAN>: pipeline.sh reuses it instead of capturing again.
# Stop: touch $ARC_RUNS/qwen38fn_dflash/$TAG/trainer/STOP. Resume: RESUME=1, same TAG and settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
ROOT=$PWD
export ARC_ROOT=$ROOT
source env/env.sh
: "${TAG:?}" "${PLAN:?}" "${GATE:?}" "${CAPTURE_GPUS:?}" "${TRAINER_GPUS:?}"
INIT=${INIT:-$ARC_NFS/runs/qwen38fn_mtp_dense/20261011-int3-target47/dline/M_T35_T39_T43_T47-m70-none-init}
FORK=${INT3_FORK:-$HOME/work/sglang-arc-2-rl-int3}
PYTHON=${PYTHON:-$HOME/work/venv-fn-train/bin/python} TORCHRUN=${TORCHRUN:-$HOME/work/venv-fn-train/bin/torchrun}
RESUME=${RESUME:-0}
python3 - "$INIT" "$PLAN" <<'PY'
import json, sys
from pathlib import Path
init, plan = Path(sys.argv[1]), Path(sys.argv[2])
c = json.loads((init / "config.json").read_text())
dense = c.get("ffn_type") == "dense_shared" and c.get("residual", "hc") == "hc"
plain = c.get("ffn_type") == "dense" and c.get("residual") == "plain"
assert c.get("mask_gr") == "none" and (dense or plain), f"{init}: not a mask_gr none dense (HC or plain) init"
meta = json.loads((plan / "plan_meta.json").read_text())
profile = json.loads(Path(meta["target_profile"]["path"]).read_text())
assert profile["env"].get("ARC_INT3_DEPLOY") == "1" and profile["env"].get("ARC_INT3_ACT") == "fp8", \
    f"{plan}: the plan's target profile is not the INT3 W3A8 deployment"
print(f"· init {init} (mask_gr none, residual {c.get('residual', 'hc')}, context {c.get('context', 'recompute')}), "
      f"plan {plan} (target {profile['instance']})")
PY
python3 - "$GATE" <<'PY'
import json, sys
gate = json.loads(open(sys.argv[1]).read())
assert gate["prefix_gate_version"] == "prefix_gate_v3", f"{sys.argv[1]}: not a prefix-gate v3 gate report"
print(f"· gate {sys.argv[1]}: prefix gate {gate['prefix_gate']}")
PY
[ -d "$FORK/python/sglang" ] || { echo "✗ no sglang tree at $FORK" >&2; exit 1; }
"$PYTHON" -c 'import torch' || { echo "✗ $PYTHON has no torch" >&2; exit 1; }
R=$ARC_RUNS/qwen38fn_dflash
if [ -f "$R/$TAG.unit" ] && systemctl --user is-active --quiet "$(cat "$R/$TAG.unit")"; then
    echo "✗ $(cat "$R/$TAG.unit") is still running" >&2; exit 1
fi
UNIT=dflash-$TAG-$(date +%s)
systemd-run --user --unit="$UNIT" --collect -p WorkingDirectory="$ROOT" \
    -p StandardOutput=append:"$R/$TAG.pipeline.log" -p StandardError=append:"$R/$TAG.pipeline.log" \
    --setenv=RESUME="$RESUME" --setenv=TAG="$TAG" --setenv=PLAN="$PLAN" --setenv=INIT="$INIT" --setenv=GATE="$GATE" \
    --setenv=CAPTURE_GPUS="$CAPTURE_GPUS" --setenv=TRAINER_GPUS="$TRAINER_GPUS" \
    --setenv=ARC_SGLANG_FORK="$FORK" --setenv=LOCK_TARGET=b200-rl-int3 \
    --setenv=STEP_ANCHORS="${STEP_ANCHORS:-89000}" --setenv=JOBS_PER_RANK="${JOBS_PER_RANK:-8}" \
    --setenv=MAX_RANK_TOKENS="${MAX_RANK_TOKENS:-500000}" --setenv=LR_SCALE="${LR_SCALE:-2}" \
    --setenv=NEW_LR_MULT="${NEW_LR_MULT:-3}" --setenv=REDUCER_LR="${REDUCER_LR:-3e-5}" \
    --setenv=TEMPERATURE="${TEMPERATURE:-0.7}" --setenv=COV_WEIGHT="${COV_WEIGHT:-0.3}" \
    --setenv=COV_WEIGHT_FINAL="${COV_WEIGHT_FINAL:-0.05}" --setenv=LR_SCHEDULE="${LR_SCHEDULE:-wsd}" \
    --setenv=LR_DECAY_FRACTION="${LR_DECAY_FRACTION:-0.2}" --setenv=PREFIX_GATE_ALPHA="${PREFIX_GATE_ALPHA:-0.01}" \
    --setenv=PREFIX_GATE_TEMPERATURE="${PREFIX_GATE_TEMPERATURE:-0.7}" \
    --setenv=MICRO_ANCHORS="${MICRO_ANCHORS:-4096}" --setenv=SAVE_EVERY="${SAVE_EVERY:-2000}" \
    --setenv=EVAL_EVERY="${EVAL_EVERY:-1000}" --setenv=EVAL_ANCHORS="${EVAL_ANCHORS:-1024}" \
    --setenv=QUEUE_GIB="${QUEUE_GIB:-48}" --setenv=BASE_PORT="${BASE_PORT:-30751}" \
    --setenv=WARM_START="${WARM_START:-}" --setenv=WARM_NEW_DATA="${WARM_NEW_DATA:-0}" \
    --setenv=REWARM_STEPS="${REWARM_STEPS:-500}" --setenv=HELDOUT_FROM="${HELDOUT_FROM:-}" \
    --setenv=PYTHON="$PYTHON" --setenv=TORCHRUN="$TORCHRUN" \
    bash "$ROOT/work/qwen38fn_dflash/pipeline.sh"
echo "$UNIT" > "$R/$TAG.unit"
systemctl --user show -p MainPID --value "$UNIT" > "$R/$TAG.pipeline.pid"
echo "· $UNIT, pipeline $(cat "$R/$TAG.pipeline.pid"); log $R/$TAG.pipeline.log"
