#!/bin/bash
# ARC-AGI-2 RL on the large-grid pool (10-22, node1 GPUs 0-5): the ARC3 RL v3.2 stack carried over to run_arc2.py episodes.
#   pool     $ARC_DATA/arc2_pools/largegrid30-20261022/tasks (work/seed_bank/tools/build_largegrid_pack.py: ARC-AGI-2 training
#            families regenerated at 30x30 by ARC-GEN, no evaluation parent), the train list from the prescreen
#            (arc2_pool_eval.py); held-out = the public evaluation 120 (configure_b200.py --dataset arc2_dir), evaluated offline
#   harness  harness-b100k-group.json: fence bwrap (jail_arc2.sh, no network), 100k output + 8k wrap-up, window 180k / compact
#            170k (prepare_rl drops compacted episodes), codex p7 as the Kaggle deployment, run_arc2.py --jail-selftest first
#   serving  v3.2's: profile perf-main-fp16-ser-opt-lazyfix-l8-a16 (W3A16 Humming 0.1.18, exact resume + own-last-turn,
#            LAZY_MEDIA / RELEASE_AFTER_PREFILL / NO_SYNC), fork h0118 c61e351b7 (lock b200-rl-main-1021), the 8-layer plan B
#            FP8 drafter s25562; tp1x3 = 3 replicas on GPUs 0-2, a 3-rank trainer on 3-5 (v3.1 / a16t08 on node1)
#   trainer  v3.2's: A16, cutlass dense, bf16 QSA indexer, qsa_union, offload keep 8 GiB, memory fraction 0.92, prepare cpu-exact
#   target   v3's: MIS [0.2, 5], sequence mean, lr 1e-4, AdamW (0.9, 0.95) eps 1e-12, adv none, no efficiency term
#            (ARC-AGI-2 scores no efficiency), group 16, 256 episodes per update, ramp 1200 s, group_ready; rollout T 0.8
#            (v3.2; the deployment samples at 0.7, T < 1 learns slower: docs/research/qwen38fn_rl_reward_flat_review_20261018.md)
#   disk     node1's /data is shared and ~94% full, and the trainer writes 3.2 GB of merged weights a version: the whole run
#            directory lives on tmpfs ($RL_TMPFS) and $RL_RUN is a symlink to it (train_rl_service.sh wants a fresh trainer/,
#            so trainer/ alone cannot be a symlink). Everything is lost on a reboot: copy trainer/merged/<kept>, the
#            checkpoints and groups-packed/ to NFS when the run ends (work/tito/collect/nfs_move.py, one file at a time).
#
# Knobs (env, configure time only unless noted; plan.json / config.json record what was used):
#   RL_TRAIN_IDS (required: the prescreened list)   RL_TASKS_DIR   RL_HELDOUT_IDS   RL_GROUP_SIZE 16   RL_BATCH 256
#   RL_SESSIONS 80 (per replica; 10-28 run 3 at 48 used KV 0.47 / mamba 0.94 of an 8.88M pool)   RL_MAMBA_SLOTS 6 x sessions
#   RL_MEMFRAC 0.96   RL_STEPS 100   RL_SAVE_EVERY 10 (trainer checkpoint)   RL_EVAL_EVERY (kept merged weights, default RL_SAVE_EVERY)
#   RL_ROLLOUT_TEMPERATURE 0.8   RL_LR 1e-4   RL_LR_WARMUP_STEPS 0 (step k <= N at lr*k/(N+1))   RL_LOSS_REDUCTION sequence
#   RL_ALGO mis   RL_OUTPUT_BUDGET (template 100k)   RL_EFFICIENCY_BETA (unset = off; configure_b200.py --efficiency-beta)
#   RL_KEEP_TRAIN_ORDER=1 (train in the RL_TRAIN_IDS order, no shuffle)   RL_HELDOUT_DIR (configure_b200.py --heldout-dir)
#   RL_INIT_CHECKPOINT / RL_INIT_MERGED (configure_b200.py --init-checkpoint / --init-merged: start from another run's
#            trainer checkpoint-<N> and its merged/v<N>; the init-policy stage syncs the replicas to it)
#   RL_TOOL_HAND_WEIGHT / RL_TOOL_TOOLKIT_WEIGHT / RL_TOOL_HAND_SCALE / RL_TOOL_HAND_CAP (unset = off: the arc2 tool-use term,
#            configure_b200.py --tool-*; the harness config needs tool_use_stats, e.g. harness-b110k-v3y-tooluse-group.json)
#   RL_ROLLOUT_TOP_P / RL_ROLLOUT_TOP_K (unset = off; e.g. 0.95 / 20: the ledger records each sampled token's support and
#            the trainer replays it, replay: mask): needs RL_SERVE_PROFILE=perf-main-fp16-ser-opt-lazyfix-l8-a16-topp,
#            RL_LOCK_TARGET=b200-rl-topp-1024, RL_FORK_TREE=$HOME/work/sglang-arc-2-topp (fork rl-topp-mask bdf43e509)
#   RL_TEMPLATE (configs/arc2-largegrid-int3qat.json; arc2-eval120hard-int3qat.json = the 200k-budget harness)
#   RL_PREPARE_DROP_STOPS wall,deadline,cancelled,proxy_fatal (episodes the harness stopped for infra reasons are dropped, not trained as R = 0)
#   RL_MAX_GROUPS_IN_FLIGHT (harness process cap; default 2 x slots / group: long-tail groups then hold it, 10-24 arc2 ran ~73 of 144)
#   RL_AGENT_RUNTIME (the agent's python venv in the jail: configure_b200.py --agent-runtime -> harness.agent_python; default
#            $ARC_AGENT_RUNTIME, which has no scipy). The selftest reads it back from config.json, so a resume runs the same one
#   RL_REPLICA_PORT_BASE / RL_CONTROL_PORT / RL_TRAINER_PORT (default 30571.. / 30570 / 30591; a second stack on the same
#            machine, e.g. RL_TOPOLOGY=tp1x1 RL_GPUS=6,7, needs all three of its own)
#   RL_RAMP_SECONDS 1200   RL_TOPOLOGY tp1x3   RL_GPUS 0,...,5   RL_SERVE_PROFILE / RL_LOCK_TARGET / RL_FORK_TREE / RL_DFLASH_DIR
#   RL_TRAINER_TRAINABLE full (configure_b200.py --trainer-trainable: the projections as FP32 full weights instead of the LoRA)
#            with RL_TRAINER_FREEZE_FIRST_LAYERS (e.g. 2) / RL_TRAINER_KEEP_CHECKPOINTS (each full checkpoint ~35 GB on tmpfs)
#   RL_TRAINER_*   RL_MIN_FREE_GB 30   RL_HARD_MIN_FREE_GB 15   RL_PREPARE_MAX_GB 25   RL_PREPARE_RESERVE_GB 3
#   RL_DISK_FLOOR_GIB 15 (launch_b200_stream.sh)   RL_WANDB_PROJECT qwen38fn-rl   RL_UNIT_PREFIX
#
#   RL_RUN=$ARC_RUNS/qwen38fn_rl/<YYYYMMDD>-arc2lg-... RL_TRAIN_IDS=<list> bash work/qwen38fn_rl/launch_arc2_largegrid_int3qat.sh
#   RL_DRY_RUN=1 ...    print every command, start nothing        RL_STAGES="check servers" ...    a subset, in this order
# Stop: touch $RL_RUN/STOP (drains), then systemctl --user stop <unit prefix>-{orchestrator,routing-gc,group-pack,control,trainer,r0..r2}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
: "${RL_RUN:?set RL_RUN=\$ARC_RUNS/qwen38fn_rl/<YYYYMMDD>-arc2lg-...}"
: "${RL_TRAIN_IDS:?set RL_TRAIN_IDS to the prescreened train list (arc2_pool_eval.py select)}"
export RL_RUN
export RL_TOPOLOGY=${RL_TOPOLOGY:-tp1x3} RL_GPUS=${RL_GPUS:-0,1,2,3,4,5}
export RL_SERVE_PROFILE=${RL_SERVE_PROFILE:-perf-main-fp16-ser-opt-lazyfix-l8-a16}
export RL_LOCK_TARGET=${RL_LOCK_TARGET:-b200-rl-main-1021}
export RL_FORK_TREE=${RL_FORK_TREE:-$HOME/work/sglang-arc-2-h0118}
export RL_DFLASH_DIR=${RL_DFLASH_DIR:-$ARC_RUNS/qwen38fn_rl/dflash-rollout/drafter/planb-l8-s25562/deploy-fp8}
export RL_FP8_GEMM=${RL_FP8_GEMM:-flashinfer_cutlass}
export RL_SERVE_MODEL=${RL_SERVE_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq}
export RL_TRAIN_MODEL=${RL_TRAIN_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-bf16dense}
export RL_DISK_FLOOR_GIB=${RL_DISK_FLOOR_GIB:-15}
export RL_WANDB_PROJECT=${RL_WANDB_PROJECT:-qwen38fn-rl}
export RL_LEDGER_ROUTING=1                                   # prepare_rl reads the routing
RL_TMPFS=${RL_TMPFS:-/dev/shm/uid-$UID/arcforge-runs/qwen38fn_rl/$(basename "$RL_RUN")}
dry=${RL_DRY_RUN:-0}
stages=${RL_STAGES-check servers control init-policy trainer orchestrator routing-gc}  # init-policy: a no-op without RL_INIT_*
py=${RL_PYTHON:-python3}
tasks=${RL_TASKS_DIR:-$ARC_DATA/arc2_pools/largegrid30-20261022/tasks}
sessions=${RL_SESSIONS:-80}
gs=${RL_GROUP_SIZE:-16}
read -r n_replicas < <("$py" - "$RL_TOPOLOGY" <<'PY'
import sys
sys.path.insert(0, "work/qwen38fn_rl")
from configure_b200 import TOPOLOGIES
print(len(TOPOLOGIES[sys.argv[1]]["replica_gpus"]))
PY
)
(cd "$RL_DFLASH_DIR" && sha256sum --quiet -c SHA256SUMS)
python3 work/sglang_fork/humming_overlay.py check 0.1.18 "$ARC_DATA/humming-overlays/0.1.18" >/dev/null

# the run directory on tmpfs, $RL_RUN a symlink to it
if [ -L "$RL_RUN" ]; then
    [ "$(readlink "$RL_RUN")" = "$RL_TMPFS" ] || { echo "$RL_RUN points to $(readlink "$RL_RUN"), not $RL_TMPFS" >&2; exit 1; }
elif [ -e "$RL_RUN" ]; then
    echo "$RL_RUN exists and is not the tmpfs symlink" >&2; exit 1
elif [ "$dry" = 1 ]; then
    echo "mkdir -p $RL_TMPFS && ln -s $RL_TMPFS $RL_RUN"
else
    test ! -e "$RL_TMPFS" || { echo "$RL_TMPFS exists without $RL_RUN" >&2; exit 1; }
    mkdir -p "$RL_TMPFS" "$(dirname "$RL_RUN")"
    ln -s "$RL_TMPFS" "$RL_RUN"
fi
if [ ! -f "$RL_RUN/config.json" ]; then
    cmd=("$py" work/qwen38fn_rl/configure_b200.py --run-root "$RL_RUN"
         --template "${RL_TEMPLATE:-work/qwen38fn_rl/configs/arc2-largegrid-int3qat.json}" --topology "$RL_TOPOLOGY" --model-kind int3
         --gpus "$RL_GPUS" --prepare-media-device cpu-exact
         ${RL_REPLICA_PORT_BASE:+--replica-port-base "$RL_REPLICA_PORT_BASE"} ${RL_CONTROL_PORT:+--control-port "$RL_CONTROL_PORT"}
         ${RL_TRAINER_PORT:+--trainer-port "$RL_TRAINER_PORT"}
         --fork "$RL_FORK_TREE" --dataset arc2_dir --tasks-dir "$tasks" --train-ids-file "$RL_TRAIN_IDS"
         --heldout-ids-file "${RL_HELDOUT_IDS:-$PWD/work/qwen38fn_rl/task_lists/arc2_eval120.txt}"
         ${RL_HELDOUT_DIR:+--heldout-dir "$RL_HELDOUT_DIR"}
         ${RL_INIT_CHECKPOINT:+--init-checkpoint "$RL_INIT_CHECKPOINT"} ${RL_INIT_MERGED:+--init-merged "$RL_INIT_MERGED"}
         --sessions-per-replica "$sessions" --group-size "$gs" --rollout-slots $((n_replicas * sessions))
         --prepare-drop-stops "${RL_PREPARE_DROP_STOPS-wall,deadline,cancelled,proxy_fatal}"
         --ramp-seconds "${RL_RAMP_SECONDS:-1200}" ${RL_MAX_GROUPS_IN_FLIGHT:+--max-groups-in-flight "$RL_MAX_GROUPS_IN_FLIGHT"} --trainer-memory-fraction "${RL_TRAINER_MEMORY_FRACTION:-0.92}"
         ${RL_OUTPUT_BUDGET:+--output-token-budget "$RL_OUTPUT_BUDGET"}
         ${RL_EFFICIENCY_BETA:+--efficiency-beta "$RL_EFFICIENCY_BETA"} ${RL_KEEP_TRAIN_ORDER:+--keep-train-order}
         ${RL_TOOL_HAND_WEIGHT:+--tool-hand-weight "$RL_TOOL_HAND_WEIGHT"} ${RL_TOOL_TOOLKIT_WEIGHT:+--tool-toolkit-weight "$RL_TOOL_TOOLKIT_WEIGHT"}
         ${RL_TOOL_HAND_SCALE:+--tool-hand-scale "$RL_TOOL_HAND_SCALE"} ${RL_TOOL_HAND_CAP:+--tool-hand-cap "$RL_TOOL_HAND_CAP"}
         --memfrac "${RL_MEMFRAC:-0.96}" --mamba-slots "${RL_MAMBA_SLOTS:-$((6 * sessions))}"
         --optimizer-steps "${RL_STEPS:-100}" --save-every-steps "${RL_SAVE_EVERY:-10}"
         --eval-every-steps "${RL_EVAL_EVERY:-${RL_SAVE_EVERY:-10}}" --global-optimizer-batch "${RL_BATCH:-256}"
         --serve-profile "$RL_SERVE_PROFILE" --dflash-dir "$RL_DFLASH_DIR" --group-ready
         --rollout-temperature "${RL_ROLLOUT_TEMPERATURE:-0.8}" --lr "${RL_LR:-1e-4}" --lr-warmup-steps "${RL_LR_WARMUP_STEPS:-0}"
         ${RL_ROLLOUT_TOP_P:+--rollout-top-p "$RL_ROLLOUT_TOP_P"} ${RL_ROLLOUT_TOP_K:+--rollout-top-k "$RL_ROLLOUT_TOP_K"}
         ${RL_AGENT_RUNTIME:+--agent-runtime "$RL_AGENT_RUNTIME"}
         --adam-beta2 0.95 --adam-eps 1e-12 --adv-normalization none --loss-reduction "${RL_LOSS_REDUCTION:-sequence}"
         --trainer-fast-kernels "${RL_TRAINER_FAST_KERNELS:-all,qsa_union}"
         --trainer-dense-qat-gemm "${RL_TRAINER_DENSE_QAT_GEMM:-cutlass}"
         --trainer-qsa-indexer "${RL_TRAINER_QSA_INDEXER:-bf16}" --trainer-offload-keep-gib "${RL_TRAINER_OFFLOAD_KEEP_GIB:-8}"
         ${RL_TRAINER_TRAINABLE:+--trainer-trainable "$RL_TRAINER_TRAINABLE"}
         ${RL_TRAINER_FREEZE_FIRST_LAYERS:+--trainer-freeze-first-layers "$RL_TRAINER_FREEZE_FIRST_LAYERS"}
         ${RL_TRAINER_KEEP_CHECKPOINTS:+--trainer-keep-checkpoints "$RL_TRAINER_KEEP_CHECKPOINTS"}
         --algo "${RL_ALGO:-mis}"
         --min-free-gb "${RL_MIN_FREE_GB:-30}" --hard-min-free-gb "${RL_HARD_MIN_FREE_GB:-15}"
         --prepare-max-gb "${RL_PREPARE_MAX_GB:-25}" --prepare-reserve-gb "${RL_PREPARE_RESERVE_GB:-3}"
         --tokenizer-dir "$RL_SERVE_MODEL" --trainer-base-path "$RL_TRAIN_MODEL")
    if [ "$dry" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
fi
if [ "$dry" = 1 ] && [ ! -f "$RL_RUN/plan.json" ]; then
    echo "(dry run: $RL_RUN/plan.json does not exist yet; run the configure command above, then repeat for the stages)"
    exit 0
fi
# Anti-cheat controls (no model, no GPU) in the jail the episodes use: every launch, a failure stops here.
if [ -n "$stages" ]; then
    # the agent python the orchestrator will run (config.json harness.agent_python, expanded in this shell's env.sh)
    agent_py=$("$py" -c 'import json, os, sys; print(os.path.expandvars(json.load(open(sys.argv[1]))["harness"]["agent_python"]))' \
               "$RL_RUN/config.json")
    case "$agent_py" in *'$'*) echo "config.json harness.agent_python $agent_py does not expand" >&2; exit 1 ;; esac
    # and its codex (harness.driver_bin: the template's $ARC_CODEX_PREFIX = p7, or $ARC_CODEX_P8_PREFIX for the Kaggle x39 harness)
    driver_bin=$("$py" -c 'import json, os, sys; print(os.path.expandvars(json.load(open(sys.argv[1]))["harness"]["driver_bin"]))' \
                 "$RL_RUN/config.json")
    case "$driver_bin" in *'$'*) echo "config.json harness.driver_bin $driver_bin does not expand" >&2; exit 1 ;; esac
    selftest=(env PYTHONPATH="$PWD/src" "$agent_py" work/arc_code/arc2/run_arc2.py --jail-selftest
              --config "$RL_RUN/harness-train.json" --run-dir "$RL_RUN/selftest/$(date +%Y%m%d-%H%M%S)"
              --tasks-dir "$tasks" --agent-python "$agent_py" --driver-bin "$driver_bin")
    if [ "$dry" = 1 ]; then printf '%q ' "${selftest[@]}"; echo; else "${selftest[@]}"; fi
fi
for stage in $stages; do
    echo "=== stage $stage"
    bash work/qwen38fn_rl/launch_b200_stream.sh "$stage"
done
case " $stages " in
    *" orchestrator "*)
        pack=(systemd-run --user --unit="${RL_UNIT_PREFIX:-rl-b200-$(basename "$RL_RUN")}-group-pack" --collect
              --working-directory="$PWD" -p Nice=10 -p StandardOutput=append:"$RL_RUN/group_pack.log"
              -p StandardError=append:"$RL_RUN/group_pack.log"
              "$py" work/qwen38fn_rl/group_pack.py --run-root "$RL_RUN" --archive-dir "$RL_RUN/groups-packed")
        if [ "$dry" = 1 ]; then printf '%q ' "${pack[@]}"; echo; else "${pack[@]}"; fi ;;
esac
