#!/bin/bash
# QAD on the student trajectories (qadmix-1019-1022 + FP8 teacher targets), GPUs 0-3 (world 4), after the teacher
# extraction (unit qadcol-teacher) succeeds: R = --kl reverse, then F = --kl forward. Recipe: 1000 steps cosine (user 10-22), warmup 30,
# 1x LR, min ratio 0.01 (user: lower final LR) + accel-C kernels (W3A16, cutlass, qsa_union, keep 8). Dev eval every 50.
{
set -uo pipefail
cd $HOME/work/arc-forge/.claude/worktrees/bridge-cse_01Jn8ZjGJLPupy44aACbH3yT
R=$HOME/work/runs/qwen38fn_qad
T=$R/20261023-qadmix
S=/dev/shm/uid-$UID/arcforge-runs/tito/20261022-int3-qad-collect
M=/dev/shm/uid-$UID/arcforge-runs/qwen38fn_qad/models
while systemctl --user is-active -q qadcol-teacher; do sleep 60; done
res=$(systemctl --user show -p Result --value qadcol-teacher)
echo "$(date -u +%FT%TZ) teacher result=$res"
[ "$res" = success ] || exit 1
for sp in train validation; do [ -f $S/teacher-qadmix/targets/$sp/DONE ] || { echo "no DONE for $sp"; exit 1; }; done
sleep 30
COMMON="--mode train --model $M/gean320-int3-bf16dense --dense-qat-reference $M/gean320-int3-gptq
  --data $S/qadmix-1019-1022 --targets $S/teacher-qadmix/targets
  --bin-tokens 120000 --ple-codec int5 --int3-act bf16 --dense-qat-gemm cutlass
  --fast-kernels hc,hc_inject,gdn_norm,act_quant,qsa_select,qsa_dkdv,gdn_gva,qsa_union --offload-keep-gib 8
  --steps 1000 --lr-schedule cosine --lr-min-ratio 0.01 --warmup-steps 30
  --eval-every 50 --save-every 50 --keep-checkpoints 1 --export-every 1000"
for arm in R F; do
  kl=$([ $arm = R ] && echo reverse || echo forward)
  echo "$(date -u +%FT%TZ) start $arm ($kl)"
  GPUS=0,1,2,3 OUT=$T/$arm MASTER_PORT=29661 bash work/qwen38fn_qad/train_qad.sh $COMMON --kl $kl \
    --store /dev/shm/uid-$UID/arcforge-runs/qwen38fn_qad/qadmix-$arm-store --wandb-name qad-qadmix-$arm-cos1000 \
    > $T/$arm.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end $arm rc=$rc"
  sleep 30
done
}
