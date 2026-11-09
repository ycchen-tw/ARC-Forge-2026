#!/bin/bash
# L8 preparation (2026-10-20): the three steps for deepening plan B 8251 to 8 layers (0,1,2,3,1,2,3,4, copied, not
# zeroed: the user chose L8-copy on 10-20, expand_layers.py), in order:
#   1. pool resume check (trainer GPUs $T0,$T1 + capture GPUs $C0,$C1, STOP_AT 12, small50 plan): the resume evidence for
#      fast_gates, and a measurement of per-rank speed and GPU memory with 8 layers
#      ($R/20261020-l8-resume-check/trainer/metrics.jsonl, gpu.csv);
#   2. fast_gates.py (GPU $E0): make_data_plan / data_plan / expand_layers changed, so plan B's gate source hash is stale;
#   3. step-0 eval (GPU $E0, probes/heldout_eval.py --inits): E / renewal of plan B 8251, L8-zero and L8-copy on the same
#      105-job held-out set.
# Usage: T0=0 T1=1 C0=2 C1=3 E0=0 bash $0. GPUs are granted by the user; pipeline.sh checks the cards are free first.
set -euo pipefail
main() {
: "${T0:?}" "${T1:?}" "${C0:?}" "${C1:?}" "${E0:?}"
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_017j1AQDBo8pGaN4Xzc733yh
R=$ARC_RUNS/qwen38fn_dflash
PY=$HOME/work/venv-fn-train/bin/python TR=$HOME/work/venv-fn-train/bin/torchrun
export TAG=20261020-l8-resume-check PLAN=$R/20261019-codex1015-small50-plan INIT=$R/20261020-l8-expand/step8251-L8-copy
export CAPTURE_GPUS=$C0,$C1 TRAINER_GPUS=$T0,$T1 BASE_PORT=30921 STOP_AT=12
export STEP_ANCHORS=89000 JOBS_PER_RANK=8 MAX_RANK_TOKENS=500000 MICRO_ANCHORS=4096 EVAL_EVERY=10 EVAL_ANCHORS=256
export LR_SCALE=1 PREFIX_GATE_ALPHA=0.01 PREFIX_GATE_TEMPERATURE=0.7 LR_SCHEDULE=wsd LR_DECAY_FRACTION=0.3
export COV_WEIGHT=0.05 COV_WEIGHT_FINAL=0.05
export ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-rl-int3 LOCK_TARGET=b200-rl-int3 PYTHON=$PY TORCHRUN=$TR
bash work/qwen38fn_dflash/probes/pool_resume_check.sh
mkdir -p $R/20261020-l8-gate
CUDA_VISIBLE_DEVICES=$E0 $PY work/qwen38fn_dflash/fast_gates.py --features $ARC_RAM_RUNS/qwen38fn_dflash/20261019-codex1015-heldout/shard-0/heldout-000000014 \
    --resume-evidence $R/20261020-l8-resume-check/resume-evidence.json --objective acceptance \
    --out $R/20261020-l8-gate/gate-acceptance.json > $R/20261020-l8-gate/fast_gates.log 2>&1
CUDA_VISIBLE_DEVICES=$E0 $TR --standalone --nproc-per-node 1 work/qwen38fn_dflash/probes/heldout_eval.py \
    --heldout $ARC_RAM_RUNS/qwen38fn_dflash/20261019-codex1015-heldout/shard-* --out $R/20261020-l8-expand/eval-step0.json \
    --inits planB-8251=$R/20261019-plain-inject-codex1015/exports/step-8251-bf16 \
    L8-zero=$R/20261020-l8-expand/step8251-L8-zero L8-copy=$R/20261020-l8-expand/step8251-L8-copy \
    > $R/20261020-l8-expand/eval-step0.log 2>&1
echo "· prep done: gate $R/20261020-l8-gate/gate-acceptance.json, step-0 eval $R/20261020-l8-expand/eval-step0.json"
}
main "$@"
