#!/bin/bash
# RL rollout server: Qwen3.8-Flash-Next FP8 TP4, SPEC=none (non-spec; RL_SPEC=dflash switches to DFlash), sglang-arc-2 fork, TITO
# ledger records ids + logprob + routing.
# Starts in a systemd user unit (stop: systemctl --user stop <unit>, which kills only this unit's cgroup).
#
#   RL_RUN=$ARC_RUNS/qwen38fn_rl/20261007-toy-b8k RL_PORT=30471 RL_GPUS=0,1,2,3 \
#   RL_FORK_TREE=<worktree of the RL fork> bash work/qwen38fn_rl/launch_rollout_server.sh
#
# Required  RL_RUN        run root dir (ledger in $RL_RUN/ledger, server.pid / server.log in $RL_RUN/serve/)
#       RL_PORT       server port
#       RL_GPUS       comma-separated GPUs (must match TP; the serve script checks those cards for other people's processes)
#       RL_FORK_TREE  checkout of the fork (with python/sglang); deliberately no default, to avoid using the shared sglang-arc-2 by
#                     mistake
# Optional  RL_LOCK_TARGET  target in deploy/sglang.lock.json (default b200-qwen38fn-rl, SPEC §6)
#       RL_UNIT         systemd unit name (default rl-rollout-p$RL_PORT)
#       RL_LEDGER       ledger dir (default $RL_RUN/ledger)
#       RL_MODEL        model dir to load (default: the serve script's FP8 base). The trainer's merged/v00000r holds only the
#                       LoRA-merged tensors (not a full model): resuming from round r = start from base, then POST
#                       /update_weights_from_disk
#                       {model_path: merged/v00000r, weight_version: "r", flush_cache: true, keep_identity: true}
#       RL_WEIGHT_VERSION  weight_version at server start-up (default 0). sglang's default is "default"; the orchestrator requires
#                       it to equal r at the start of round r
#       RL_SERVE_ENV    extra environment variables for the serve script, space-separated K=V (e.g. "MAXREQ=256 CTXLEN=262144")
#       RL_SERVER_ARGS  extra sglang arguments, split on whitespace and appended after the fixed flags (e.g.
#                       "--stream-interval 8 --scheduler-recv-interval 8")
#       RL_SPEC         none (default) | dflash: the fork's DFlash speculative decoding (SPEC=dflash of serve_qwen38fn_0520.sh;
#                       K=8, rejection sampling, ReplaySSM, prefix reuse still goes through radix). Needs RL_DFLASH_DIR = the
#                       dense drafter's deploy-bf16 dir, and the fork needs keep_identity under spec (from exp/rl-dflash on).
#                       K / ReplaySSM / ARC_DFLASH_* go through RL_SERVE_ENV
#                       (configuration measured 10-08: "MAXREQ=256 DFLASH_K=4", with 32 groups / 256 sessions).
#                       T=0.7 distribution: no difference against another non-spec server (+0.008 nats per token, z +1.3;
#                       T 0.70→0.75 is −0.02)
#       RL_MODEL_KIND   fp8 (default, paths above unchanged) | int3: gEAN320 INT3 W3A8 deployment package (expert INT3 Humming,
#                       expert activations per-token FP8, dense 144 block-FP8 + 84 BF16, PLE converted to INT5 at load). Started
#                       via work/qwen38fn_eval/int3_deploy_runtime.py, TP1 / EP1, KV fp8_e4m3, no --quantization. Needs
#                       RL_MODEL = the package dir; RL_LOCK_TARGET defaults to b200-rl-int3.
#                       RL_INT3_ACT (default fp8; bf16 = W3A16), RL_FP8_GEMM (GEMM for dense block-FP8: flashinfer_trtllm = fp32
#                       scale, the same math as G4's cutlass; auto = DeepGEMM, which requantizes the scales at load and packs them
#                       as int32 UE8M0, unlike G4, and poisons on the first FP8+scale weight sync (split_with_sizes mismatch,
#                       measured 10-10). flashinfer_trtllm is slower than DeepGEMM: prefill −24%, decode −9%; triton is as fast
#                       as it).
#                       RL_INT3_PROBE_OUT=<dir>: start via work/qwen38fn_rl/probes/int3_sync_probe.py instead (weight-sync probe
#                       report).
#       SKIP_GPU_CHECK=1 passed through to the serve script as is: start even on cards that already have other people's processes
#                       (ask the card's owner before sharing; 10-21: an idle jupyter kernel of 0.6 GiB on GPU 0)
#       RL_LEDGER_ROUTING  1 (default) | 0: the ledger does not record MoE routing (no --session-ledger-routing;
#                       --enable-return-routed-experts stays on). Only for evaluation / pre-screening without training: routing is
#                       about 1 KB per token (26 GiB for a step of 256 games), and a 3000-game pre-screen would fill tmpfs;
#                       RL replicas always record it (prepare_rl reads it)
#       RL_DRY_RUN=1    only print the systemd-run command, do not start
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RL_RUN:?}" "${RL_PORT:?}" "${RL_GPUS:?}"
RL_LEDGER=${RL_LEDGER:-$RL_RUN/ledger}
RL_MODEL_KIND=${RL_MODEL_KIND:-fp8}
case $RL_MODEL_KIND in
    fp8) RL_LOCK_TARGET=${RL_LOCK_TARGET:-b200-qwen38fn-rl} ;;
    int3)
        RL_LOCK_TARGET=${RL_LOCK_TARGET:-b200-rl-int3}
        [ -f "${RL_MODEL:?RL_MODEL_KIND=int3 needs RL_MODEL=<gEAN320 INT3 deploy package dir>}/config.json" ] \
            || { echo "✗ $RL_MODEL has no config.json" >&2; exit 1; }
        grep -q '"arc_int3"' "$RL_MODEL/config.json" || { echo "✗ $RL_MODEL/config.json has no arc_int3 (not an INT3 deploy package)" >&2; exit 1; } ;;
    *) echo "✗ RL_MODEL_KIND must be fp8 or int3" >&2; exit 1 ;;
esac
RL_WEIGHT_VERSION=${RL_WEIGHT_VERSION:-0}
RL_LEDGER_ROUTING=${RL_LEDGER_ROUTING:-1}
case $RL_LEDGER_ROUTING in
    1) ledger_routing=(--session-ledger-routing) ;;
    0) ledger_routing=() ;;
    *) echo "✗ RL_LEDGER_ROUTING must be 1 or 0" >&2; exit 1 ;;
esac
RL_SPEC=${RL_SPEC:-none}
case $RL_SPEC in
    none) [ -z "${RL_DFLASH_DIR:-}" ] || { echo "✗ RL_DFLASH_DIR goes only with RL_SPEC=dflash" >&2; exit 1; } ;;
    dflash) [ -f "${RL_DFLASH_DIR:?RL_SPEC=dflash needs RL_DFLASH_DIR=<drafter deploy-bf16 dir>}/config.json" ] \
                || { echo "✗ $RL_DFLASH_DIR has no config.json" >&2; exit 1; } ;;
    *) echo "✗ RL_SPEC must be none or dflash" >&2; exit 1 ;;
esac

if [ "${RL_INNER:-0}" != 1 ]; then
    : "${RL_FORK_TREE:?}"
    [ -d "$RL_FORK_TREE/python/sglang" ] || { echo "✗ $RL_FORK_TREE has no python/sglang" >&2; exit 1; }
    unit=${RL_UNIT:-rl-rollout-p$RL_PORT}
    systemctl --user is-active --quiet "$unit" && { echo "✗ unit $unit is already running" >&2; exit 1; }
    if ss -ltn "sport = :$RL_PORT" | grep -q LISTEN; then echo "✗ port $RL_PORT already has a listener" >&2; exit 1; fi
    setenv=(--setenv=RL_INNER=1 --setenv=RL_RUN="$RL_RUN" --setenv=RL_PORT="$RL_PORT" --setenv=RL_GPUS="$RL_GPUS"
            --setenv=RL_FORK_TREE="$RL_FORK_TREE" --setenv=RL_LOCK_TARGET="$RL_LOCK_TARGET" --setenv=RL_LEDGER="$RL_LEDGER"
            --setenv=RL_WEIGHT_VERSION="$RL_WEIGHT_VERSION" --setenv=RL_SERVER_ARGS="${RL_SERVER_ARGS:-}"
            --setenv=RL_SPEC="$RL_SPEC" --setenv=RL_MODEL_KIND="$RL_MODEL_KIND"
            --setenv=RL_LEDGER_ROUTING="$RL_LEDGER_ROUTING")
    for v in RL_INT3_ACT RL_FP8_GEMM RL_INT3_PROBE_OUT SKIP_GPU_CHECK; do
        [ -z "${!v:-}" ] || setenv+=(--setenv="$v=${!v}")
    done
    [ -z "${RL_DFLASH_DIR:-}" ] || setenv+=(--setenv=RL_DFLASH_DIR="$RL_DFLASH_DIR")
    [ -z "${RL_MODEL:-}" ] || setenv+=(--setenv=RL_MODEL="$RL_MODEL")
    for kv in ${RL_SERVE_ENV:-}; do setenv+=(--setenv="$kv"); done
    cmd=(systemd-run --user --unit="$unit" --collect --working-directory="$PWD" "${setenv[@]}"
         -p StandardOutput=append:"$RL_RUN/serve/server.log" -p StandardError=append:"$RL_RUN/serve/server.log"
         /bin/bash "$PWD/work/qwen38fn_rl/launch_rollout_server.sh")
    if [ "${RL_DRY_RUN:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
    mkdir -p "$RL_RUN/serve" "$RL_LEDGER"
    "${cmd[@]}"
    echo "· unit $unit started; log: $RL_RUN/serve/server.log; stop: systemctl --user stop $unit"
    exit 0
fi

# ---- everything below runs inside the systemd unit ----
export ARC_SGLANG_FORK=$RL_FORK_TREE LOCK_TARGET=$RL_LOCK_TARGET GPUS=$RL_GPUS PORT=$RL_PORT SPEC=$RL_SPEC
[ "$RL_SPEC" != dflash ] || export DFLASH_DIR=$RL_DFLASH_DIR
[ -z "${RL_MODEL:-}" ] || export MODEL=$RL_MODEL
export ARC_TITO_SERVER_TAG=rl-p$RL_PORT          # for claiming: /proc/<pid>/environ carries this tag
kind_args=()
if [ "$RL_MODEL_KIND" = int3 ]; then
    # one rank = one card (the Humming plugin supports only TP1 / EP1); DP=N in RL_SERVE_ENV gives N TP1 ranks with one card each
    # (the serve script's --dp-size, requests split by routed_dp_rank). Any other TP / EP in RL_SERVE_ENV is a configuration
    # error and is not silently overridden.
    [ "${TP:-1}" = 1 ] && [ "${EP:-1}" = 1 ] && [ "$(printf '%s' "${RL_GPUS//,/ }" | wc -w)" = "${DP:-1}" ] \
        || { echo "✗ RL_MODEL_KIND=int3 allows only TP1/EP1, and the GPU count must equal DP (TP=${TP:-} EP=${EP:-} DP=${DP:-1} GPUS=$RL_GPUS)" >&2; exit 1; }
    # target KV defaults to fp8_e4m3; a serve profile can pass KV_CACHE_DTYPE=nvfp4 via RL_SERVE_ENV (with ARC_QSA_NVFP4=1)
    export TP=1 EP=1 MNNVL_HC=0 INT3_DEPLOY=1 KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
    export ARC_INT3_ACT=${RL_INT3_ACT:-fp8} ARC_PLE_INT5=1
    [ -z "${RL_INT3_PROBE_OUT:-}" ] || export INT3_PROBE_OUT=$RL_INT3_PROBE_OUT
    # PLE: the INT5 encoding takes only the host-offloaded FP8 table (ple_embedding_dtype override as in G4's v13 server start-up).
    kind_args=(--dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend "${RL_FP8_GEMM:-flashinfer_trtllm}"
               --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}')
fi
echo $$ > "$RL_RUN/serve/server.pid"
exec bash work/sglang_fork/serve_b200.sh \
    --weight-version "$RL_WEIGHT_VERSION" \
    --session-replay \
    --session-replay-empty-think \
    --session-ledger-dir "$RL_LEDGER" \
    --session-ledger-all \
    "${ledger_routing[@]}" \
    --session-ledger-logprobs \
    --enable-return-routed-experts \
    --no-mm-literal-image-sentinel \
    --enable-metrics \
    "${kind_args[@]}" \
    ${RL_SERVER_ARGS:-}
