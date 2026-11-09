#!/bin/bash
# Staged local B200 stream run. Set RL_RUN, RL_FORK_TREE, RL_LOCK_TARGET; RL_DRY_RUN=1 prints commands.
# configure_b200.py materializes config.json first. stages: check servers control trainer baseline offload orchestrator evaluate evaluate-parallel stop
# Offload uses the existing O_DIRECT mover, one file and one replica at a time, in its own user unit.
# RL_LEDGER_PORTS is a space-separated roster (default: the replica ports of plan.json, 30571 30572 without one);
# evaluate-parallel runs synchronously. RL_TRAINER_RESUME (trainer stage): resume the trainer from <run>/trainer/checkpoint-<step>. RL_DISK_FLOOR_GIB (default 120): local free space below which the disk guard writes STOP when the offloader is down.
# Topology and model kind come from the run's plan.json (configure_b200.py --topology tp2x2|tp1x4|tp1x2 --model-kind fp8|int3):
#   tp2x2 / fp8   two TP2 FP8 replicas (GPUs 0-1, 2-3), trainer on the Qwen3.8-Flash-Next-FP8 checkpoint (the 10-09 line)
#   tp1x4 / int3  four TP1 gEAN320 INT3 W3A8 replicas (GPUs 0, 1, 2, 3; ports 30571-30574), QAT trainer on the BF16-dense variant:
#                 RL_SERVE_MODEL (deployment package, default /dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq),
#                 RL_TRAIN_MODEL (trainer --model / BF16 master, default .../qwen38fn-gean320-int3-bf16dense),
#                 RL_TRAIN_EXTRA (trainer flags, default: --base-storage int3 --dense-qat fp8 --dense-qat-reference
#                 $RL_SERVE_MODEL --ple-codec int5 --int3-act <the serve profile's RL_INT3_ACT, default fp8>; then
#                 --dense-qat-gemm <plan.json trainer_dense_qat_gemm> is appended), RL_FP8_GEMM (dense FP8 GEMM backend of the servers).
#   tp1x2 / int3  the tp1x4 setup on half the machine: TP1 replicas on GPUs 0 and 1 (ports 30571-30572), a 2-rank trainer
#                 on GPUs 2-3 (plan.json trainer_gpus; prepare bins for world 2)
#   tp1x3 / int3  three TP1 replicas on GPUs 0-2 and a 3-rank trainer on GPUs 3-5 (tp1x5: five replicas, trainer on 5-7)
# The drafter is plan.json dflash_dir when the plan has one (configure_b200.py --dflash-dir), else RL_DFLASH_DIR or acc-v4.
# plan.json ple_media (configure_b200.py --ple-media) sets the PLE media ids of both sides: semantic = servers with
# --ple-mm-placeholder-ids and trainer --ple-media semantic; deployed = neither (the clamp).
# plan.json serve_profile (configure_b200.py --serve-profile) adds serving env/args to every replica; a non-base profile
# names the lock target its code needs, and RL_LOCK_TARGET must be that one (check / servers refuse otherwise). Once a
# replica is healthy, its process environment and /server_info are checked against the profile.
# config.json prepare.temperature (configure_b200.py --rollout-temperature) is the rollouts' sampling temperature: prepare_rl
# checks every ledger append against it, and the servers get ARC_RESPONSES_DEFAULT_TEMPERATURE=<it> (codex sends no
# temperature; the /v1/responses class default is 0.7). A fork tree without that opt-in can only serve 0.7.
# When the fork tree has ARC_LOGITS_VOCAB_LIMIT, the servers get it set to the tokenizer's id bound (config.json tokenizer_dir,
# the bound the ledger audit's ids_in_vocab checks): the lm_head can be padded past the tokenizer, and an untruncated sample
# of a padded id fails the audit and stops the run (10-11: id 248134 >= 248077 once in ~8.9e7 tokens at T=1.0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
: "${RL_RUN:?}" "${RL_FORK_TREE:?}" "${RL_LOCK_TARGET:?}"
stage=${1:?check|servers|control|init-policy|trainer|baseline|offload|orchestrator|evaluate|evaluate-parallel|stop}
dry=${RL_DRY_RUN:-0}
tag=${RL_UNIT_PREFIX:-rl-b200-$(basename "$RL_RUN")}
py=${RL_PYTHON:-python3}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
train_py=${RL_TRAIN_PYTHON:-$HOME/work/venv-fn-train/bin/python}
config=$RL_RUN/config.json
run_cmd() { if [ "$dry" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
# plan.json: model_kind tp sessions delayer-ratio ports(comma) gpus(semicolon). Plans written before the topology option
# are the 2 x TP2 FP8 line.
plan_fields() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
t=p.get("topology") or {"tp":2,"replica_gpus":["0,1","2,3"],"replica_ports":[30571,30572]}
print(p.get("model_kind","fp8"),t["tp"],p["sessions_per_replica"],p["prefill_delayer_queue_min_ratio"],
      ",".join(map(str,t["replica_ports"])),";".join(t["replica_gpus"]))
PY
}
# control proxy and trainer ports (configure_b200.py --control-port / --trainer-port). Plans written before 10-25 have
# neither field: those runs used the fixed 30570 / 30591.
plan_service_port() {
    "$py" -c 'import json,sys;p=json.load(open(sys.argv[1]));print(int(p.get(sys.argv[2],int(sys.argv[3]))))' \
        "$RL_RUN/plan.json" "$1" "$2"
}
plan_ports() {
    local kind tp maxreq ratio ports gpus
    read -r kind tp maxreq ratio ports gpus < <(plan_fields)
    echo "${ports//,/ }"
}
# PLE ids at media pads, one setting for servers and trainer (configure_b200.py --ple-media). Plans written before 10-11 have
# none: those runs served the clamp (deployed); add the field rather than guess.
plan_ple_media() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("ple_media") not in ("semantic","deployed"):
    raise SystemExit(f"{sys.argv[1]}: ple_media {p.get('ple_media')!r} is not semantic or deployed (plans before 10-11 have none: "
                     "those runs served the clamp, add \"ple_media\": \"deployed\")")
print(p["ple_media"])
PY
}
# trainer GPUs (configure_b200.py topology: 4,5,6,7, 3,4,5 for tp1x3, 2,3 for tp1x2). Plans without the field ran the
# trainer on 4-7.
# plan.json "init" (configure_b200.py --init-checkpoint / --init-merged): "<checkpoint>\t<merged>", or nothing for a run
# that starts from the base (plans before 10-23 have no "init")
plan_init() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
i=json.load(open(sys.argv[1])).get("init")
if i is not None:
    print(i["checkpoint"]+"\t"+i["merged"])
PY
}
plan_trainer_gpus() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,re,sys
g=json.load(open(sys.argv[1])).get("trainer_gpus","4,5,6,7")
if not re.fullmatch(r"[0-7](,[0-7])*",g) or len(g.split(",")) not in (1,2,3,4):   # 1: the tp1x1 toy (the trainer allows one rank since 33c091ca0)
    raise SystemExit(f"{sys.argv[1]}: trainer_gpus {g!r} is not 1-4 GPU indices")
print(g)
PY
}
# The DFlash drafter: plan.json dflash_dir (configure_b200.py --dflash-dir, 10-21) when the plan has one, and then
# RL_DFLASH_DIR must be unset or the same directory; plans without it serve RL_DFLASH_DIR or acc-v4, as before.
drafter_setup() {
    local planned
    planned=$("$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("dflash_dir") or "")
PY
)
    if [ -n "$planned" ]; then
        if [ -n "${RL_DFLASH_DIR:-}" ] && [ "$(realpath "$RL_DFLASH_DIR")" != "$(realpath "$planned")" ]; then
            echo "RL_DFLASH_DIR=$RL_DFLASH_DIR, but plan.json dflash_dir is $planned" >&2; exit 1
        fi
        drafter=$planned
    else
        drafter=${RL_DFLASH_DIR:-$ARC_RUNS/qwen38fn_rl/dflash-rollout/drafter/acc-v4/deploy-bf16}
    fi
}
# trainer --fast-kernels (configure_b200.py --trainer-fast-kernels). Plans written before 10-12 have no
# trainer_fast_kernels: they ran the reference kernels, which is "none".
# Plans written before 10-15 have no serve_memfrac: those servers ran --mem-fraction-static 0.75.
plan_memfrac() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(repr(float(json.load(open(sys.argv[1])).get("serve_memfrac", 0.75))))
PY
}
# Plans written before 10-15 (4r4t) have no serve_mamba_slots: those servers let SGLang size the mamba cache (auto).
plan_mamba_slots() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("serve_mamba_slots", "auto"))
PY
}
# Server context length = the harness serve_context_length when it has one (run_arc3_level.py: >= the window; the
# deployment serves 240000 under a 150000 window), else its model_context_window (the window codex plans and compacts
# against). Every harness config before 10-17 says 150000, the value the servers used to hard-code.
serve_ctxlen() {
    "$py" - "$RL_RUN/harness-train.json" <<'PY'
import json,sys
h=json.load(open(sys.argv[1]))
served=int(h.get("serve_context_length",h["model_context_window"]))
assert served>=int(h["model_context_window"]),(served,h["model_context_window"])
print(served)
PY
}
# The trainer's largest bin (sizes its offload slots) = the cap prepare_rl packs bins to (config.json prepare.bin_tokens).
config_bin_tokens() {
    "$py" - "$RL_RUN/config.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["prepare"]["bin_tokens"]))
PY
}
# Plans written before 10-15 have no trainer_adam_beta2: those trainers ran AdamW betas (0.9, 0.98).
plan_adam_beta2() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(repr(float(json.load(open(sys.argv[1])).get("trainer_adam_beta2", 0.98))))
PY
}
# Plans written before 10-18 have no trainer_adam_eps: those trainers ran the torch default 1e-8.
plan_adam_eps() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(repr(float(json.load(open(sys.argv[1])).get("trainer_adam_eps", 1e-8))))
PY
}
# Plans written before 10-24 have no trainer_lr_warmup_steps: those trainers ran a constant LR (0).
plan_lr_warmup_steps() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])).get("trainer_lr_warmup_steps", 0)
if isinstance(v, bool) or not isinstance(v, int) or v < 0:
    raise SystemExit(f"plan.json trainer_lr_warmup_steps {v!r} is not an integer >= 0")
print(v)
PY
}
plan_trainer_memory_fraction() {   # plans before 10-20 have none: no cap
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])).get("trainer_memory_fraction")
print("" if v is None else repr(float(v)))
PY
}
plan_trainer_offload_keep_gib() {   # configure_b200.py --trainer-offload-keep-gib; plans before 10-21 have none: the trainer's 0
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])).get("trainer_offload_keep_gib")
print("" if v is None else repr(float(v)))
PY
}
# trainer --dense-qat-gemm (configure_b200.py --trainer-dense-qat-gemm; int3 only). Plans written before 10-21 have no
# trainer_dense_qat_gemm: those trainers ran the Triton GEMM, which is "triton" (cutlass gives the same bits, faster).
plan_dense_qat_gemm() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])).get("trainer_dense_qat_gemm","triton")
if v not in ("triton","cutlass"):
    raise SystemExit(f"plan.json trainer_dense_qat_gemm {v!r} is not triton or cutlass")
print(v)
PY
}
# trainer --qsa-indexer (configure_b200.py --trainer-qsa-indexer). Plans written before 10-21 have no trainer_qsa_indexer:
# those trainers scored the QSA index in fp32.
plan_qsa_indexer() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])).get("trainer_qsa_indexer","fp32")
if v not in ("fp32","bf16","fp8"):
    raise SystemExit(f"plan.json trainer_qsa_indexer {v!r} is not fp32, bf16 or fp8")
print(v)
PY
}
# trainer --trainable full and its options (configure_b200.py --trainer-trainable ...). Plans written before 10-29
# have none of these keys: those trainers were LoRA. Prints the trainer flags (empty for a LoRA trainer).
plan_full_flags() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("trainer_trainable","lora")=="lora":
    print("")
else:
    flags=["--trainable","full","--freeze-first-layers",str(int(p["trainer_freeze_first_layers"])),
           "--keep-checkpoints",str(int(p["trainer_keep_checkpoints"]))]
    if p["trainer_checkpoint_store"] is not None:
        flags+=["--checkpoint-store",p["trainer_checkpoint_store"]]
    print(" ".join(flags))
PY
}
plan_fast_kernels() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("trainer_fast_kernels", "none"))
PY
}
# The serving profile: name, lock target, env (K=V ...), args. Plans written before 10-11 have no serve_profile: they ran
# the base flags (the field arrived with the perf profile; base did not change).
plan_serve_profile() {
    "$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
prof=p.get("serve_profile",{"name":"base"})
if prof["name"]=="base":
    if set(prof)!={"name"}:
        raise SystemExit(f"{sys.argv[1]}: serve_profile base must not carry {sorted(set(prof)-{'name'})}")
    print("base\t-\t-\t-")
    raise SystemExit(0)
missing=[k for k in ("lock_target","env","args") if k not in prof]
if missing:
    raise SystemExit(f"{sys.argv[1]}: serve_profile {prof['name']} lacks {missing}")
bad=[x for x in prof["env"]+prof["args"] if not x or any(c.isspace() for c in x)]+[x for x in prof["env"] if "=" not in x]
if bad or not prof["env"]:
    raise SystemExit(f"{sys.argv[1]}: serve_profile {prof['name']}: malformed env/args {bad or 'no env'}")
print("\t".join([prof["name"],prof["lock_target"]," ".join(prof["env"])," ".join(prof["args"]) or "-"]))
PY
}
# grep for an ARC opt-in in a fork tree source file; a missing file means a wrong fork tree, not "no opt-in"
tree_has() {
    local file="$RL_FORK_TREE/python/sglang/srt/$2"
    [ -f "$file" ] || { echo "$file does not exist: is RL_FORK_TREE=$RL_FORK_TREE a sglang fork tree?" >&2; exit 1; }
    grep -q "$1" "$file"
}
# sets temp_env: the env that makes the servers sample at config.json prepare.temperature
rollout_temperature_setup() {
    local t
    t=$("$py" -c 'import json,sys;t=json.load(open(sys.argv[1]))["prepare"]["temperature"];print(repr(float(t)))' "$config")
    if tree_has ARC_RESPONSES_DEFAULT_TEMPERATURE entrypoints/openai/protocol.py; then
        temp_env="ARC_RESPONSES_DEFAULT_TEMPERATURE=$t"
    elif [ "$t" = 0.7 ]; then
        temp_env=""                      # this tree has no opt-in: the /v1/responses class default 0.7 is the rollout temperature
    else
        echo "config.json prepare.temperature $t: $RL_FORK_TREE has no ARC_RESPONSES_DEFAULT_TEMPERATURE, so it can only serve 0.7" >&2
        exit 1
    fi
    # top-k / top-p (config.json prepare.top_p / top_k, configure_b200.py --rollout-top-p / --rollout-top-k): served
    # through ARC_RESPONSES_DEFAULT_TOP_P / _TOP_K (codex sends neither), the ledger records each sampled token's support
    # (--session-ledger-sampling-mask, packed width SAMPLING_MASK_MAX_TOKENS) and the trainer replays it (SPEC §1)
    local p k
    read -r p k < <("$py" -c 'import json,sys;q=json.load(open(sys.argv[1]))["prepare"];print(repr(float(q.get("top_p",1.0))),int(q.get("top_k",-1)))' "$config")
    samp_args=""
    if [ "$p" != 1.0 ] || [ "$k" != -1 ]; then
        tree_has ARC_RESPONSES_DEFAULT_TOP_P entrypoints/openai/protocol.py \
            && tree_has session_ledger_sampling_mask arg_groups/fields/serving.py \
            || { echo "config.json prepare.top_p $p / top_k $k: $RL_FORK_TREE has no ARC_RESPONSES_DEFAULT_TOP_P or --session-ledger-sampling-mask" >&2; exit 1; }
        temp_env="$temp_env ARC_RESPONSES_DEFAULT_TOP_P=$p ARC_RESPONSES_DEFAULT_TOP_K=$k"
        local cap
        cap=$("$py" -c 'import sys;sys.path.insert(0,"work/qwen38fn_rl");import configure_b200 as c;print(c.SAMPLING_MASK_MAX_TOKENS)')
        samp_args=" --session-ledger-sampling-mask --sampling-mask-max-tokens $cap"
    fi
}
# sets vocab_env: ARC_LOGITS_VOCAB_LIMIT=<tokenizer id bound> when the fork tree has that opt-in, else empty
vocab_limit_setup() {
    if ! tree_has ARC_LOGITS_VOCAB_LIMIT layers/logits_processor.py; then
        vocab_env=""
        return 0
    fi
    local bound
    bound=$("$py" - "$config" <<'PY'
import json,sys
from pathlib import Path
tok=Path(json.load(open(sys.argv[1]))["tokenizer_dir"])/"tokenizer.json"
t=json.load(open(tok))
ids=list(t["model"]["vocab"].values())+[a["id"] for a in t.get("added_tokens",[])]
if not ids:
    raise SystemExit(f"{tok}: no token ids")
print(max(ids)+1)  # = arcforge.tito.ledger Vocab.id_bound (max id with added tokens + 1)
PY
)
    [ -n "$bound" ] || { echo "could not read the tokenizer id bound for ARC_LOGITS_VOCAB_LIMIT" >&2; exit 1; }
    vocab_env="ARC_LOGITS_VOCAB_LIMIT=$bound"
}
# sets prof_name prof_lock prof_env prof_args; a non-base profile must be served from its own lock target
serve_profile_setup() {
    IFS=$'\t' read -r prof_name prof_lock prof_env prof_args < <(plan_serve_profile)
    [ -n "${prof_name:-}" ] || { echo "plan.json: could not read serve_profile" >&2; exit 1; }
    [ "$prof_name" != base ] || { prof_lock=""; prof_env=""; prof_args=""; }
    [ "$prof_args" != - ] || prof_args=""
    if [ "$prof_name" != base ] && [ "$RL_LOCK_TARGET" != "$prof_lock" ]; then
        echo "plan.json serve_profile $prof_name needs RL_LOCK_TARGET=$prof_lock (and RL_FORK_TREE on that tree); got $RL_LOCK_TARGET" >&2
        exit 1
    fi
}
# after a replica is healthy: every profile / temperature K=V is in its main process environment, every profile arg in
# /server_info
# Image preprocessing stays on sglang's fast processor (auto), pinned and verified: prepare_tito_mm rebuilds the pixels from
# the ledger and needs them bit-identical to (u8 - 127.5) / 127.5, which PIL's rescale-then-normalize is not (10-29 arc2-r4:
# --image-processor-backend pil failed the first packet). The fast processor runs on the replica's GPU in the tokenizer
# process, outside the static pool: leave it room with RL_MEMFRAC (10-29 arc2-r4 at 128 sessions / mamba 768 / graphs to
# 128 and MEMFRAC 0.96 had ~20 MiB free, every image request hit torch.OutOfMemoryError from 23 min in, the run stalled).
IMAGE_ARGS="--image-processor-backend auto"
verify_serve_profile() {
    local port=$1 unit=$2 pid env_all="$prof_env${temp_env:+ $temp_env}${vocab_env:+ $vocab_env}"
    [ "$dry" != 1 ] && [ -n "${env_all// /}" ] || return 0
    pid=$(systemctl --user show -p MainPID --value "$unit")
    [ -n "$pid" ] && [ "$pid" != 0 ] || { echo "$unit has no MainPID" >&2; exit 1; }
    "$py" - "$pid" "$port" "$prof_name" "$env_all" "$prof_args${samp_args:-} $IMAGE_ARGS" <<'PY'
import json,sys,urllib.request
pid,port,name=sys.argv[1:4]
env,args=sys.argv[4].split(),sys.argv[5].split()
have=set(open(f"/proc/{pid}/environ","rb").read().decode().split("\0"))
missing=[kv for kv in env if kv not in have]
if missing:
    raise SystemExit(f"replica :{port} (pid {pid}): serve profile {name} env not in the process: {missing}")
info=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info",timeout=60))
i=0
while i<len(args):
    key=args[i].removeprefix("--").replace("-","_")
    if i+1<len(args) and not args[i+1].startswith("--"):
        want,i=args[i+1],i+2
    else:
        want,i=True,i+1
    if key not in info or (info[key] is not True if want is True else str(info[key])!=want):
        raise SystemExit(f"replica :{port}: serve profile {name} arg {key}={want!r}, server_info has {info.get(key,'<absent>')!r}")
print(f"replica :{port}: serve profile {name} env ({len(env)}) and args ({len(args)}) verified")
PY
}
plan_kind() {
    local kind rest
    read -r kind rest < <(plan_fields)
    echo "$kind"
}
# trainer base, serving package and trainer flags of the run's model kind
model_setup() {
    kind=$(plan_kind)
    dense_gemm=$(plan_dense_qat_gemm)
    case "$kind" in
        fp8)
            serve_model=""
            train_model=${RL_TRAIN_MODEL:-$ARC_MODELS/Qwen3.8-Flash-Next-FP8}
            train_extra=${RL_TRAIN_EXTRA:-}
            [ "$dense_gemm" = triton ] || { echo "plan.json trainer_dense_qat_gemm $dense_gemm is for the int3 dense QAT trainer" >&2; exit 1; } ;;
        int3)
            serve_model=${RL_SERVE_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq}
            train_model=${RL_TRAIN_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-bf16dense}
            int3_act_setup
            train_extra=${RL_TRAIN_EXTRA:---base-storage int3 --dense-qat fp8 --dense-qat-reference $serve_model --ple-codec int5 --int3-act $int3_act}
            # the trainer simulates the routed experts with the activations the replicas serve
            [[ " $train_extra " == *" --int3-act $int3_act "* ]] && [ "$(grep -o -- '--int3-act' <<< "$train_extra" | wc -l)" = 1 ] \
                || { echo "trainer flags must carry exactly one --int3-act $int3_act (the replicas serve RL_INT3_ACT=$int3_act): $train_extra" >&2; exit 1; }
            [[ $train_extra != *--dense-qat-gemm* ]] \
                || { echo "--dense-qat-gemm comes from plan.json trainer_dense_qat_gemm (configure_b200.py --trainer-dense-qat-gemm): $train_extra" >&2; exit 1; }
            train_extra+=" --dense-qat-gemm $dense_gemm" ;;
        *) echo "plan.json model_kind $kind is not fp8 or int3" >&2; exit 1 ;;
    esac
}
# sets int3_act: the routed-expert activations the replicas serve, RL_INT3_ACT of plan.json's serve profile env (the
# rollout server's default, fp8, when the profile does not set it). Not from the caller's environment: one source for both.
int3_act_setup() {
    [ -z "${RL_INT3_ACT:-}" ] || { echo "RL_INT3_ACT comes from plan.json's serve profile; unset it (got $RL_INT3_ACT)" >&2; exit 1; }
    int3_act=$("$py" - "$RL_RUN/plan.json" <<'PY'
import json,sys
prof=json.load(open(sys.argv[1])).get("serve_profile",{"name":"base"})
acts=[kv.split("=",1)[1] for kv in prof.get("env",[]) if kv.startswith("RL_INT3_ACT=")]
if len(acts)>1 or not set(acts)<={"fp8","bf16"}:
    raise SystemExit(f"{sys.argv[1]}: serve profile {prof['name']} RL_INT3_ACT {acts}")
print(acts[0] if acts else "fp8")
PY
)
}
read_ledger_ports() {
    local roster=${RL_LEDGER_PORTS:-}
    if [ -z "$roster" ]; then
        if [ -f "$RL_RUN/plan.json" ]; then roster=$(plan_ports); else roster="30571 30572"; fi
    fi
    read -r -a ledger_ports <<< "$roster"
    local port seen=" "
    for port in "${ledger_ports[@]}"; do
        case "$port" in ''|*[!0-9]*) echo "invalid ledger port $port" >&2; exit 1;; esac
        [ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || { echo "invalid ledger port $port" >&2; exit 1; }
        case "$seen" in *" $port "*) echo "duplicate ledger port $port" >&2; exit 1;; esac
        seen+="$port "
    done
    [ "${#ledger_ports[@]}" -gt 0 ] || { echo "empty ledger port roster" >&2; exit 1; }
}
wait_health() {
    local port=$1 unit=$2 deadline=$((SECONDS + 3600))
    [ "$dry" != 1 ] || return 0
    while [ "$SECONDS" -lt "$deadline" ]; do
        systemctl --user is-active --quiet "$unit" || { echo "unit $unit failed; inspect its log" >&2; exit 1; }
        if curl -fsS -m 30 "http://127.0.0.1:$port/health" >/dev/null; then return 0; fi
        sleep 10
    done
    echo "server $port did not become healthy in 1 hour" >&2; exit 1
}
case "$stage" in
    check)
        "$py" work/qwen38fn_rl/orchestrator.py --config "$config" --check
        "$py" work/qwen38fn_rl/baseline_b200.py --config "$config" --check
        "$py" - "$RL_LOCK_TARGET" "$RL_FORK_TREE/python/sglang" <<'PY'
from arcforge.sglang_lock import check_tree
import sys
print("fork lock:", check_tree(sys.argv[1], sys.argv[2])["commit"])
PY
        serve_profile_setup
        echo "serve profile: $prof_name${prof_env:+ (lock $prof_lock)}"
        rollout_temperature_setup
        echo "rollout temperature: ${temp_env:-0.7 (the /v1/responses class default; this tree has no opt-in)}"
        vocab_limit_setup
        echo "vocab limit: ${vocab_env:-none (this tree has no ARC_LOGITS_VOCAB_LIMIT; a padded id would fail the audit)}"
        drafter_setup
        echo "drafter: $drafter"
        [ -x "$train_py" ] && [ -f "$drafter/config.json" ]
        model_setup
        # int3: the servers load $serve_model, the trainer $train_model; both default to the gEAN320 pair, so one of
        # RL_SERVE_MODEL / RL_TRAIN_MODEL left unset crosses two checkpoints (the orchestrator checks again at start)
        [ "$kind" != int3 ] || "$py" work/qwen38fn_rl/package_lineage.py "$serve_model" "$train_model"
        nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
        ;;
    servers)
        # Session count and delayer ratio come from the run's plan.json (configure_b200.py).
        read -r kind tp maxreq ratio ports_csv gpus_list < <(plan_fields)
        IFS=, read -r -a ports <<< "$ports_csv"
        IFS=';' read -r -a gpus <<< "$gpus_list"
        [ "${#ports[@]}" = "${#gpus[@]}" ] || { echo "plan.json: ${#ports[@]} ports but ${#gpus[@]} GPU groups" >&2; exit 1; }
        model_setup
        ple=$(plan_ple_media)
        ple_args=""
        [ "$ple" = deployed ] || ple_args=" --ple-mm-placeholder-ids"
        serve_profile_setup
        rollout_temperature_setup
        vocab_limit_setup
        memfrac=$(plan_memfrac)
        mamba_slots=$(plan_mamba_slots)
        ctxlen=$(serve_ctxlen)
        drafter_setup
        profile_env="${temp_env:+ $temp_env}${vocab_env:+ $vocab_env}" profile_args=""
        if [ "$prof_name" != base ]; then profile_env=" $prof_env$profile_env"; profile_args="${prof_args:+ $prof_args}"; fi
        profile_args="$profile_args$samp_args"
        start_replica() {
            local i=$1 port=${ports[$1]} replica_gpus=${gpus[$1]}
            local -a kind_env=()
            if [ "$kind" = int3 ]; then
                kind_env=(RL_MODEL_KIND=int3 RL_MODEL="$serve_model" RL_FP8_GEMM="${RL_FP8_GEMM:-flashinfer_trtllm}")
            fi
            env "${kind_env[@]}" RL_RUN="$RL_RUN/replicas/r$i" RL_PORT="$port" RL_GPUS="$replica_gpus" \
                RL_UNIT="$tag-r$i" RL_DFLASH_DIR="$drafter" RL_SPEC=dflash \
                RL_SERVE_ENV="TP=$tp EP=$tp MNNVL_HC=0 MAXREQ=$maxreq DFLASH_K=4 MEMFRAC=$memfrac MAMBA_CACHE_SLOTS=$mamba_slots CTXLEN=$ctxlen$profile_env" \
                RL_SERVER_ARGS="$IMAGE_ARGS --stream-interval 8 --scheduler-recv-interval 8 --enable-prefill-delayer --prefill-delayer-queue-min-ratio $ratio --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20$ple_args$profile_args" \
                RL_DRY_RUN="$dry" bash work/qwen38fn_rl/launch_rollout_server.sh
        }
        if [ "$kind" = int3 ]; then
            # One card per replica. The first server warms the JIT caches (Humming, flashinfer) alone; the rest load together.
            start_replica 0
            wait_health "${ports[0]}" "$tag-r0"
            verify_serve_profile "${ports[0]}" "$tag-r0"
            for ((i = 1; i < ${#ports[@]}; i++)); do start_replica "$i"; done
            for ((i = 1; i < ${#ports[@]}; i++)); do
                wait_health "${ports[$i]}" "$tag-r$i"
                verify_serve_profile "${ports[$i]}" "$tag-r$i"
            done
        else
            for ((i = 0; i < ${#ports[@]}; i++)); do
                start_replica "$i"
                wait_health "${ports[$i]}" "$tag-r$i"
                verify_serve_profile "${ports[$i]}" "$tag-r$i"
            done
        fi
        ;;
    control)
        control_port=$(plan_service_port control_port 30570)
        mkdir -p "$RL_RUN/control"
        backends=()
        for port in $(plan_ports); do backends+=(--backend "http://127.0.0.1:$port"); done
        run_cmd systemd-run --user --unit="$tag-control" --collect --working-directory="$PWD" \
            --setenv=PYTHONPATH="$PYTHONPATH" \
            -p StandardOutput=append:"$RL_RUN/control/control.log" -p StandardError=append:"$RL_RUN/control/control.log" \
            "$py" work/qwen38fn_rl/replica_control.py --listen 127.0.0.1 --port "$control_port" "${backends[@]}"
        wait_health "$control_port" "$tag-control"
        ;;
    init-policy)
        # A run started from another run's checkpoint: before the orchestrator's fresh start, every replica serves that
        # checkpoint's merged export as weight_version 0 (the orchestrator checks weights_source against
        # rollout.initial_weights_source). Nothing to do for a run from the base.
        control_port=$(plan_service_port control_port 30570)
        init_line=$(plan_init)
        init_ckpt=${init_line%%$'\t'*} init_merged=${init_line#*$'\t'}
        if [ -z "$init_line" ]; then
            echo "plan.json has no init: the replicas serve the base"
        else
            [ ! -s "$RL_RUN/batches.jsonl" ] || { echo "init-policy is for a fresh start; $RL_RUN/batches.jsonl has rows" >&2; exit 1; }
            body=$("$py" -c 'import json,sys;print(json.dumps({"model_path":sys.argv[1],"weight_version":"0","flush_cache":True,"keep_identity":True}))' "$init_merged")
            if [ "$dry" = 1 ]; then
                echo "curl -X POST http://127.0.0.1:$control_port/update_weights_from_disk -d '$body'"
            else
                curl -fsS -m 1800 -X POST -H 'Content-Type: application/json' -d "$body" \
                    "http://127.0.0.1:$control_port/update_weights_from_disk" >/dev/null
                curl -fsS -m 30 "http://127.0.0.1:$control_port/model_info" | "$py" -c '
import json,sys
i=json.load(sys.stdin)
if str(i["weight_version"])!="0" or i["weights_source"]!=sys.argv[1] or i["poisoned"] is not False:
    raise SystemExit(f"replicas did not load the init policy: {i}")
print("replicas serve "+i["weights_source"]+" as weight_version 0")' "$init_merged"
            fi
        fi
        ;;
    trainer)
        # One source of truth: plan.json (configure_b200.py --lr). Runs configured before 10-11 have no trainer_lr: fail.
        lr=$("$py" -c 'import json,sys;print(repr(float(json.load(open(sys.argv[1]))["trainer_lr"])))' "$RL_RUN/plan.json")
        trainer_port=$(plan_service_port trainer_port 30591)
        model_setup
        ple=$(plan_ple_media)
        fast=$(plan_fast_kernels)
        beta2=$(plan_adam_beta2)
        eps=$(plan_adam_eps)
        warmup=$(plan_lr_warmup_steps)
        trainer_gpus=$(plan_trainer_gpus)
        bin_tokens=$(config_bin_tokens)
        memcap=$(plan_trainer_memory_fraction)
        keep=$(plan_trainer_offload_keep_gib)
        qsa_indexer=$(plan_qsa_indexer)
        full_flags=$(plan_full_flags)
        [[ $train_extra != *--trainable* && $train_extra != *--freeze-first-layers* && $train_extra != *--checkpoint-store* \
            && $train_extra != *--keep-checkpoints* ]] \
            || { echo "--trainable and its options come from plan.json trainer_trainable ... (configure_b200.py --trainer-trainable): $train_extra" >&2; exit 1; }
        [[ $train_extra != *--qsa-indexer* ]] \
            || { echo "--qsa-indexer comes from plan.json trainer_qsa_indexer (configure_b200.py --trainer-qsa-indexer): $train_extra" >&2; exit 1; }
        [[ $train_extra != *--offload-keep-gib* ]] \
            || { echo "--offload-keep-gib comes from plan.json trainer_offload_keep_gib (configure_b200.py --trainer-offload-keep-gib): $train_extra" >&2; exit 1; }
        [[ $train_extra != *--lr-warmup-steps* ]] \
            || { echo "--lr-warmup-steps comes from plan.json trainer_lr_warmup_steps (configure_b200.py --lr-warmup-steps): $train_extra" >&2; exit 1; }
        init_line=$(plan_init)
        init_ckpt=${init_line%%$'\t'*}
        init_env=()
        [ -z "$init_line" ] || init_env=(--setenv=INIT_LORA="$init_ckpt")
        extra_env=(--setenv=EXTRA="${train_extra:+$train_extra }--adam-beta2 $beta2 --adam-eps $eps --lr-warmup-steps $warmup --fast-kernels $fast --ple-media $ple --bin-tokens $bin_tokens --qsa-indexer $qsa_indexer${memcap:+ --gpu-memory-fraction $memcap}${keep:+ --offload-keep-gib $keep}${full_flags:+ $full_flags}")
        # RL_TRAINER_RESUME=<trainer>/checkpoint-<step>: continue this run's trainer in place (train_rl_service.sh RESUME +
        # RESUME_EXISTING_OUT=1; the checkpoint must be the latest step, then rerun the orchestrator stage). A plan init
        # (configure_b200.py --init-checkpoint) is already inside this run's checkpoint, and the service refuses
        # INIT_LORA together with RESUME_EXISTING_OUT, so a resume drops it.
        # RL_TRAINER_RESUME_NEW_LR=1 (with RL_TRAINER_RESUME): a deliberate LR change, plan.json trainer_lr edited first;
        # the service's --resume-new-lr lets only the LR differ from the checkpoint and the original config.
        if [ -n "${RL_TRAINER_RESUME:-}" ]; then
            [ "$(dirname "$RL_TRAINER_RESUME")" = "$RL_RUN/trainer" ] && [ -f "$RL_TRAINER_RESUME/trainer_state.pt" ] \
                || { echo "RL_TRAINER_RESUME must be a checkpoint under $RL_RUN/trainer: $RL_TRAINER_RESUME" >&2; exit 1; }
            extra_env+=(--setenv=RESUME="$RL_TRAINER_RESUME" --setenv=RESUME_EXISTING_OUT=1)
            init_env=()
            case ${RL_TRAINER_RESUME_NEW_LR:-0} in
                1) extra_env[0]+=" --resume-new-lr" ;;   # extra_env[0] is --setenv=EXTRA=...
                0) ;;
                *) echo "RL_TRAINER_RESUME_NEW_LR must be 0 or 1: $RL_TRAINER_RESUME_NEW_LR" >&2; exit 1 ;;
            esac
        elif [ "${RL_TRAINER_RESUME_NEW_LR:-0}" != 0 ]; then
            echo "RL_TRAINER_RESUME_NEW_LR needs RL_TRAINER_RESUME" >&2; exit 1
        fi
        run_cmd systemd-run --user --unit="$tag-trainer" --collect --working-directory="$PWD" \
            --setenv=ARC_ROOT="$PWD" --setenv=ARC_PYTHON="$train_py" --setenv=GPUS="$trainer_gpus" \
            --setenv=MODEL="$train_model" --setenv=OUT="$RL_RUN/trainer" "${extra_env[@]}" "${init_env[@]}" \
            --setenv=PORT="$trainer_port" --setenv=LR="$lr" /bin/bash work/qwen38fn_lora/train_rl_service.sh
        if [ "$dry" != 1 ]; then
            deadline=$((SECONDS + 3600))
            while [ "$SECONDS" -lt "$deadline" ]; do
                systemctl --user is-active --quiet "$tag-trainer" || { echo "trainer unit failed" >&2; exit 1; }
                if [ -f "$RL_RUN/trainer/trainer_endpoint.json" ] && \
                    curl -fsS -m 5 "http://127.0.0.1:$trainer_port/health" | "$py" -c \
                    'import json,sys;s=json.load(sys.stdin);sys.exit(0 if s["ok"] is True and s["busy"] is False else 1)'; then
                    break
                fi
                sleep 5
            done
            [ "$SECONDS" -lt "$deadline" ] || { echo "trainer did not become ready in 1 hour" >&2; exit 1; }
            "$py" - "$RL_RUN/trainer/config.json" "$lr" "$ple" "$fast" "$beta2" "$eps" "$dense_gemm" "$qsa_indexer" "$init_ckpt" "$warmup" "$RL_RUN/plan.json" "${RL_TRAINER_RESUME:-}" <<'PY'
import json,sys
from pathlib import Path
cfg=json.load(open(sys.argv[1]))
plan=json.load(open(sys.argv[11]))
for key in ("trainable","freeze_first_layers","keep_checkpoints","checkpoint_store"):
    want=plan.get("trainer_"+key,{"trainable":"lora","freeze_first_layers":0,"keep_checkpoints":0,"checkpoint_store":None}[key])
    if cfg.get(key,{"trainable":"lora","freeze_first_layers":0,"keep_checkpoints":0,"checkpoint_store":None}[key])!=want:
        raise SystemExit(f"trainer {key} {cfg.get(key)!r} != plan.json trainer_{key} {want!r}")
want_init=sys.argv[9] or None
got_init=cfg.get("init_lora")
if (want_init is None) != (got_init is None) or (want_init is not None and Path(got_init).resolve()!=Path(want_init).resolve()):
    raise SystemExit(f"trainer init_lora {got_init!r} != plan.json init checkpoint {want_init!r}")
lr=cfg["lr"]
if sys.argv[12]:   # a resume keeps the original config.json; the LR it runs at is in its resume_history.jsonl record
    last=json.loads((Path(sys.argv[1]).parent/"resume_history.jsonl").read_text().splitlines()[-1])
    if Path(last["checkpoint"]).resolve()!=Path(sys.argv[12]).resolve():
        raise SystemExit(f"latest resume_history.jsonl record is {last['checkpoint']}, not this resume {sys.argv[12]}")
    lr=last["args"]["lr"]
if float(lr)!=float(sys.argv[2]):
    raise SystemExit(f"trainer resolved lr {lr} != plan.json trainer_lr {sys.argv[2]}")
if float(cfg["adam_beta2"])!=float(sys.argv[5]):
    raise SystemExit(f"trainer adam_beta2 {cfg['adam_beta2']} != plan.json trainer_adam_beta2 {sys.argv[5]}")
if float(cfg.get("adam_eps",1e-8))!=float(sys.argv[6]):
    raise SystemExit(f"trainer adam_eps {cfg.get('adam_eps')} != plan.json trainer_adam_eps {sys.argv[6]}")
if cfg.get("lr_warmup_steps",0)!=int(sys.argv[10]):   # trainers started before 10-24 ran a constant LR
    raise SystemExit(f"trainer lr_warmup_steps {cfg.get('lr_warmup_steps')} != plan.json trainer_lr_warmup_steps {sys.argv[10]}")
if cfg["ple_media"]!=sys.argv[3]:
    raise SystemExit(f"trainer ple_media {cfg['ple_media']} != plan.json ple_media {sys.argv[3]}")
if cfg["fast_kernels"]!=sys.argv[4]:
    raise SystemExit(f"trainer fast_kernels {cfg['fast_kernels']} != plan.json trainer_fast_kernels {sys.argv[4]}")
if cfg["dense_qat_gemm"]!=sys.argv[7]:
    raise SystemExit(f"trainer dense_qat_gemm {cfg['dense_qat_gemm']} != plan.json trainer_dense_qat_gemm {sys.argv[7]}")
if cfg["qsa_indexer"]!=sys.argv[8]:
    raise SystemExit(f"trainer qsa_indexer {cfg['qsa_indexer']} != plan.json trainer_qsa_indexer {sys.argv[8]}")
PY
        fi
        ;;
    baseline)
        run_cmd systemd-run --user --unit="$tag-baseline" --collect --working-directory="$PWD" \
            --setenv=PYTHONPATH="$PYTHONPATH" \
            /bin/bash -c 'source env/env.sh; exec "$0" work/qwen38fn_rl/baseline_b200.py --config "$1"' "$py" "$config"
        ;;
    offload)
        : "${RL_NFS_ROOT:?set the unique NFS ledger destination for this run}"
        read_ledger_ports
        mkdir -p "$RL_RUN/offload/lanes"
        run_cmd systemd-run --user --unit="$tag-offload" --collect --working-directory="$PWD" \
            --setenv=RL_RUN="$RL_RUN" --setenv=RL_NFS_ROOT="$RL_NFS_ROOT" \
            --setenv=RL_FORK_TREE="$RL_FORK_TREE" --setenv=RL_LOCK_TARGET="$RL_LOCK_TARGET" \
            --setenv=RL_LEDGER_PORTS="${ledger_ports[*]}" --setenv=RL_ROUTING_RELEASE="${RL_ROUTING_RELEASE:-0}" \
            -p StandardOutput=append:"$RL_RUN/offload/unit.log" -p StandardError=append:"$RL_RUN/offload/unit.log" \
            /bin/bash work/qwen38fn_rl/launch_b200_stream.sh offload-inner
        ;;
    offload-inner)
        : "${RL_NFS_ROOT:?}"
        read_ledger_ports
        mkdir -p "$RL_RUN/offload/lanes"
        exec 9>"$RL_RUN/offload/loop.lock"
        flock -n 9
        release=()
        while [ ! -f "$RL_RUN/offload/OFFLOAD_STOP" ]; do
            # RL_ROUTING_RELEASE=1: move only covered shards whose episodes are all trained/abandoned (routing_release.py),
            # so prepare never reads routing back over NFS.
            if [ "${RL_ROUTING_RELEASE:-0}" = 1 ]; then
                "$py" work/qwen38fn_rl/routing_release.py --run-root "$RL_RUN" --out "$RL_RUN/offload/routing_release.json"
                release=(--release "$RL_RUN/offload/routing_release.json")
            fi
            for port in "${ledger_ports[@]}"; do
                instance=$("$py" - "$port" <<'PY'
import json,sys,urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/session_ledger/status",timeout=10) as f:
    s=json.load(f);root=s["dir"].rstrip("/");name=s["instance"]
    print(root if root.rsplit("/",1)[-1]==name else root+"/"+name)
PY
)
                "$py" work/tito/collect/nfs_move.py --run-dir "$RL_RUN/offload" --chunk-mib 16 \
                    --pause 1 ledger-live --instance "$instance" --dst-root "$RL_NFS_ROOT" "${release[@]}"
            done
            "$py" - "$RL_RUN/offload/status.json" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix(".tmp")
t.write_text(json.dumps({"state":"pass_complete","utc":time.strftime("%FT%TZ",time.gmtime()),"time":time.time()})+"\n")
os.replace(t,p)
PY
            sleep 30
        done
        ;;
    routing-gc)
        # Runs without NFS offload: delete covered routing shards whose episodes are all trained/abandoned
        # (routing_gc.py, routing_release.py); log/status in $RL_RUN/routing_gc/, stop with touch $RL_RUN/routing_gc/STOP.
        mkdir -p "$RL_RUN/routing_gc"
        run_cmd systemd-run --user --unit="$tag-routing-gc" --collect --working-directory="$PWD" \
            --setenv=PYTHONPATH="$PYTHONPATH" \
            -p StandardOutput=append:"$RL_RUN/routing_gc/unit.log" -p StandardError=append:"$RL_RUN/routing_gc/unit.log" \
            "$py" work/qwen38fn_rl/routing_gc.py --run-root "$RL_RUN" --interval "${RL_ROUTING_GC_INTERVAL:-60}"
        ;;
    orchestrator)
        [ -f "$RL_RUN/trainer/trainer_endpoint.json" ] || { echo "trainer not ready" >&2; exit 1; }
        model_setup
        trainer_port=$(plan_service_port trainer_port 30591)
        if [ "$dry" != 1 ]; then
            "$py" - "$RL_RUN" "$train_model" "$trainer_port" <<'PY'
import json,sys,urllib.request
from pathlib import Path
r=Path(sys.argv[1]);p=r/"trainer/trainer_endpoint.json";e=json.loads(p.read_text())
if e["url"]!=f"http://127.0.0.1:{sys.argv[3]}" or Path(e["out"]).resolve()!=p.parent.resolve():
    raise RuntimeError("trainer endpoint URL/output does not match this run")
with urllib.request.urlopen(e["url"]+"/health",timeout=10) as f: h=json.load(f)
if h["ok"] is not True or h["busy"] is not False:
    raise RuntimeError("trainer endpoint is not ready")
if Path(h["policy_identity"]["model_path"]).resolve()!=Path(sys.argv[2]).resolve():
    raise RuntimeError("trainer endpoint uses a different base model")
PY
        fi
        # prepare's media device (configure_b200.py --prepare-media-device): cpu-exact uses no GPU (empty
        # CUDA_VISIBLE_DEVICES); cuda runs on the configured GPU (--prepare-gpu, a physical index), pinned here by its
        # UUID; a relaunch finds the UUID already written and keeps it.
        prep_gpu=$("$py" -c 'import json,sys
c=json.load(open(sys.argv[1]))["prepare"]["command"]
(mk,_,m),(k,_,v)=c[1].partition("="),c[2].partition("=")
assert mk=="MEDIA_DEVICE" and k=="CUDA_VISIBLE_DEVICES", c[:3]
assert (m=="cpu-exact" and v=="") or (m=="cuda" and v and "," not in v), c[:3]
print(v)' "$config")
        if [ -n "$prep_gpu" ]; then
            uuid=$(nvidia-smi -i "$prep_gpu" --query-gpu=uuid --format=csv,noheader)
            case "$uuid" in GPU-*) ;; *) echo "prepare GPU $prep_gpu: no UUID ($uuid)" >&2; exit 1;; esac
            "$py" - "$config" "$uuid" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads(p.read_text())
c["prepare"]["command"][2]="CUDA_VISIBLE_DEVICES="+sys.argv[2]
t=p.with_suffix(".tmp");t.write_text(json.dumps(c,indent=1)+"\n");os.replace(t,p)
PY
        fi
        # launch_orchestrator.sh takes <tag>-orchestrator-N while an earlier unit still holds detached groups (a restart
        # after DETACH): the unit it used is read back, never assumed. $RL_RUN/orchestrator_unit keeps it for watchers.
        unit_file=$RL_RUN/orchestrator_unit
        [ "$dry" != 1 ] || unit_file=$(mktemp "${TMPDIR:-/tmp}/orchestrator_unit.XXXXXX")
        rm -f "$unit_file"
        RL_CONFIG="$config" RL_UNIT="$tag-orchestrator" RL_UNIT_FILE="$unit_file" RL_DRY_RUN="$dry" \
            bash work/qwen38fn_rl/launch_orchestrator.sh
        [ -s "$unit_file" ] || { echo "launch_orchestrator.sh wrote no unit name to $unit_file (RL_WANDB_ONLY=1 starts no orchestrator)" >&2; exit 1; }
        orch_unit=$(cat "$unit_file")
        [ "$dry" != 1 ] || rm -f "$unit_file"
        case "$orch_unit" in
            "$tag-orchestrator"|"$tag-orchestrator"-[0-9]*) ;;
            *) echo "launch_orchestrator.sh wrote unit '$orch_unit', not $tag-orchestrator[-N]" >&2; exit 1 ;;
        esac
        if [ "$dry" = 1 ]; then
            pid='<orchestrator MainPID>'
        else
            pid=$(systemctl --user show --property=MainPID --value "$orch_unit")
            case "$pid" in ''|0|*[!0-9]*) echo "orchestrator unit $orch_unit already exited" >&2; exit 1;; esac
            # the previous orchestrator's disk guard ends within one poll (30 s) after that orchestrator: let the name free
            for _ in $(seq 90); do systemctl --user is-active --quiet "$tag-disk-guard" || break; sleep 1; done
            if systemctl --user is-active --quiet "$tag-disk-guard"; then
                echo "$tag-disk-guard still runs 90 s after $orch_unit started (see $RL_RUN/disk_guard.json): stop it, then" \
                     "rerun this stage's disk guard by hand -- the orchestrator runs without one until then" >&2
                exit 1
            fi
        fi
        run_cmd systemd-run --user --unit="$tag-disk-guard" --collect --working-directory="$PWD" \
            -p StandardOutput=append:"$RL_RUN/disk_guard.log" -p StandardError=append:"$RL_RUN/disk_guard.log" \
            "$py" work/qwen38fn_rl/watch_local_disk.py --run-root "$RL_RUN" --orchestrator-pid "$pid" \
            --orchestrator-unit "$orch_unit" --offload-unit "$tag-offload" --floor-gib "${RL_DISK_FLOOR_GIB:-120}" --poll-seconds 30
        ;;
    stop)
        touch "$RL_RUN/STOP"
        echo "STOP written; dispatcher drains current groups. Units remain for inspection/evaluation."
        ;;
    evaluate)
        : "${RL_EVAL_VERSIONS:?comma-separated retained versions, e.g. 10,20,30,40,50}"
        baseline=()
        "$py" - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"work/qwen38fn_rl"))
from orchestrator import load_config,read_ids
c=load_config(Path(sys.argv[1]));r=Path(c["run_root"])
condition={"harness":json.loads((r/"harness-eval.json").read_text()),
           "heldout":read_ids(Path(c["tasks"]["heldout"]["ids_file"])),"samples":4,"tokenizer_dir":c["tokenizer_dir"]}
digest=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
if json.loads((r/"baseline-b32k/summary.json").read_text())["condition_sha256"]!=digest:
    raise RuntimeError("baseline/evaluation condition differs")
PY
        for k in {0..3}; do baseline+=("$RL_RUN/baseline-b32k/s$k"); done
        run_cmd systemd-run --user --unit="$tag-evaluate" --collect --working-directory="$PWD" \
            --setenv=PYTHONPATH="$PYTHONPATH" /bin/bash -c \
            'source env/env.sh; exec "$0" work/qwen38fn_rl/heldout_offline.py --config "$1" --versions "$2" --base-url http://127.0.0.1:30571/v1 --harness-config "$3" --samples 4 --timeout-seconds 14400 --baseline "${@:4}"' \
            "$py" "$config" "$RL_EVAL_VERSIONS" "$RL_RUN/harness-eval.json" "${baseline[@]}"
        ;;
    evaluate-parallel)
        : "${RL_EVAL_VERSIONS:?comma-separated retained versions}"
        : "${RL_EVAL_BACKENDS:?space-separated backend URLs, primary first}"
        backend_args=()
        read -r -a backend_urls <<< "$RL_EVAL_BACKENDS"
        for url in "${backend_urls[@]}"; do backend_args+=(--backend "$url"); done
        lock_args=()
        if [ -n "${RL_EVAL_PARENT_LOCK_FD:-}" ]; then
            lock_args+=(--parent-lock-fd "$RL_EVAL_PARENT_LOCK_FD")
        fi
        # Direct child: the parent's flock FD cannot travel through systemd-run's D-Bus handoff.
        run_cmd "$py" work/qwen38fn_rl/parallel_eval_b200.py --config "$config" \
            "${backend_args[@]}" --versions "$RL_EVAL_VERSIONS" --samples 4 \
            --harness-config "$RL_RUN/harness-eval.json" "${lock_args[@]}"
        ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
esac
