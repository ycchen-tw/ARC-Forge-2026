#!/bin/bash
# Production drafter training on one machine: capture servers replay a data plan (capture_plan.py) into a RAM pool,
# the fast engine trains from it (train.py --pool, acceptance objective). Held-out jobs are captured first, once.
#
#   TAG=<run tag> PLAN=<make_data_plan.py dir> INIT=<init dir> GATE=<fast_gates.py report> \
#       CAPTURE_GPUS=5,6,7 TRAINER_GPUS=0,1,2,3,4 \
#       [STEP_ANCHORS=65536 JOBS_PER_RANK=4 MAX_RANK_TOKENS=500000 LR_SCALE=2 NEW_LR_MULT=3] [RESUME=1] \
#       systemd-run --user --scope --unit=dflash-<tag> bash work/qwen38fn_dflash/pipeline.sh
#
# The learning rate's schedule runs over the plan's train anchors (summary.json, every pass: --total-anchors); --steps is
# pool.planned_steps of the plan, an estimate for progress displays.
#
# WARM_START=<another pool run's checkpoint dir> [REWARM_STEPS=500]: this run starts from its trained parameters with a
# fresh optimizer (train.py --warm-start); its consumed jobs are skipped and count toward the schedule. The plan must hold
# them as train jobs (the same split as the run it continues).
#
# WARM_NEW_DATA=1 (with WARM_START): the parameters only (train.py --warm-start-new-data) -- nothing is skipped, the
# schedule starts at 0: another pass over the plan after the warm start drained it.
# STEP_SCHEDULE=1 STEPS=<n>: the run is STEPS steps and the learning rate's schedule runs over them (train.py
# --step-schedule) -- the arms of an A/B at equal steps.
# OPD_ROOT=<opd_assemble.py queue root, --done> OPD_WEIGHT=<w> OPD_FRACTION=<f> [OPD_LOSS=reached|wtv|importance
# OPD_JOBS_PER_RANK=2]:
# the root joins the pool (symlinked) and its recorded cycles train with weight w at a share f of every step's budget.
# OFFLINE_ACCEPT_WEIGHT=<w> (default 1): the captured anchors' WTV term (0: the design doc's section 23 form).
# WANDB_PROJECT=<wandb project> (default qwen38fn-dflash).
# REDUCER_LR=<rate before LR_SCALE> (default 3e-5, train.py --reducer-lr): 1e-4 for a random init.
# PREFIX_GATE_ALPHA=<tail threshold> PREFIX_GATE_TEMPERATURE=<T> (train.py --prefix-gate-*): prefix gate v3, the WTV
# and coverage terms weighted after low-trust tokens, on from the first step (offline data only).
# LR_SCHEDULE=cosine|wsd [LR_DECAY_FRACTION=<share>] (train.py --lr-schedule / --lr-decay-fraction).
# COV_WEIGHT_FINAL=<w> (train.py --cov-weight-final): the coverage weight linear from COV_WEIGHT to it over the run.
# PYTHON=<interpreter> TORCHRUN=<torchrun> (default .venv/bin/{python,torchrun}): a checkout whose .venv lacks torch
# (on node1: $HOME/work/venv-fn-train/bin/...).
# OPD_HELDOUT=<opd_assemble.py --heldout root>: every eval adds the importance estimate of the drafter's serving
#   acceptance on those held-out rollouts (train.py --opd-heldout).
# HELDOUT_FROM=<another run's $RAM/heldout>: reuse its captured held-out (same PLAN) instead of capturing again.
# OFFLINE=0 (with OPD_ROOT, HELDOUT_FROM, OPD_FRACTION=1, no CAPTURE_GPUS): OPD cycles only, no capture server -- the
# Draft-OPD paper's OPD stage.
#
# SMOKE=1 (no GATE, STEPS=<n>): train.py --smoke --checkpoint-smoke -- constant learning rate, no wandb, checkpoints with
# the consumed list, resumable; probes/pool_resume_check.py runs it to produce fast_gates.py's resume evidence.
#
# The servers are the plan's profile-mode capture servers (capture_tito.sh with TARGET_PROFILE only); each one's capture
# hook writes into one queue root, so held-out jobs are captured there first and then renamed into $RAM/heldout (they
# would otherwise count against the producers' --max-queue-gib); training jobs follow (capture_plan.py --split train).
# Resume (RESUME=1, same TAG and settings): the trainer continues from trainer/latest; each producer keeps the READY
# jobs in its root and skips the checkpoint's consumed ones (train.py pool_loop, capture_plan.py); held-out is kept.
# Stop gracefully: touch $ARC_RUNS/qwen38fn_dflash/<tag>/trainer/STOP (checkpoints, then everything exits).
# Logs and the trainer's output in $ARC_RUNS/qwen38fn_dflash/$TAG, features in $ARC_RAM_RUNS; NFS is only read (the
# target model). On exit it stops only what it started, by PID file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

OFFLINE=${OFFLINE:-1}
: "${TAG:?run tag}" "${PLAN:?plan dir}" "${INIT:?init dir}" "${TRAINER_GPUS:?}"
if [ "$OFFLINE" = 1 ]; then
    : "${CAPTURE_GPUS:?}"
else  # OPD only: no capture server, the held-out of another run of the same plan
    : "${OPD_ROOT:?OFFLINE=0 trains on OPD_ROOT only}" "${HELDOUT_FROM:?OFFLINE=0 has no capture server: HELDOUT_FROM}"
    [ -z "${CAPTURE_GPUS:-}" ] || { echo "✗ OFFLINE=0 starts no capture server: unset CAPTURE_GPUS" >&2; exit 1; }
    CAPTURE_GPUS=
fi
SMOKE=${SMOKE:-0}
[ "$SMOKE" = 1 ] || : "${GATE:?fast_gates.py report}"
[ "$SMOKE" = 1 ] && : "${STEPS:?a smoke runs STEPS steps}"
RESUME=${RESUME:-0}
ROBUST_TRANSITION=${ROBUST_TRANSITION:-}
if [ -n "$ROBUST_TRANSITION" ]; then
    [ "$RESUME" = 1 ] && [ "$SMOKE" = 0 ] || { echo "robust transition needs prepared production RESUME=1" >&2; exit 1; }
fi
BASE_PORT=${BASE_PORT:-30731}
QUEUE_GIB=${QUEUE_GIB:-48}
STEP_ANCHORS=${STEP_ANCHORS:-16384}
JOBS_PER_RANK=${JOBS_PER_RANK:-1}
MAX_RANK_TOKENS=${MAX_RANK_TOKENS:-}
LR_SCALE=${LR_SCALE:-1}
NEW_LR_MULT=${NEW_LR_MULT:-3}
REDUCER_LR=${REDUCER_LR:-3e-5}
PYTHON=${PYTHON:-.venv/bin/python}
TORCHRUN=${TORCHRUN:-.venv/bin/torchrun}
WARM_START=${WARM_START:-}
REWARM_STEPS=${REWARM_STEPS:-500}
MICRO_ANCHORS=${MICRO_ANCHORS:-4096}
SAVE_EVERY=${SAVE_EVERY:-5000}
EVAL_EVERY=${EVAL_EVERY:-2500}
EVAL_ANCHORS=${EVAL_ANCHORS:-2048}
TEMPERATURE=${TEMPERATURE:-0.7}
COV_WEIGHT=${COV_WEIGHT:-0.3}
WARM_NEW_DATA=${WARM_NEW_DATA:-0}
STEP_SCHEDULE=${STEP_SCHEDULE:-0}
OPD_ROOT=${OPD_ROOT:-}
OPD_LOSS=${OPD_LOSS:-reached}
OPD_JOBS_PER_RANK=${OPD_JOBS_PER_RANK:-2}
OFFLINE_ACCEPT_WEIGHT=${OFFLINE_ACCEPT_WEIGHT:-1}
HELDOUT_FROM=${HELDOUT_FROM:-}
OPD_HELDOUT=${OPD_HELDOUT:-}
WANDB_PROJECT=${WANDB_PROJECT:-qwen38fn-dflash}
PREFIX_GATE_ALPHA=${PREFIX_GATE_ALPHA:-}
PREFIX_GATE_TEMPERATURE=${PREFIX_GATE_TEMPERATURE:-}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
LR_DECAY_FRACTION=${LR_DECAY_FRACTION:-}
COV_WEIGHT_FINAL=${COV_WEIGHT_FINAL:-}
if [ -n "$ROBUST_TRANSITION" ]; then
    export PLAN INIT TRAINER_GPUS CAPTURE_GPUS STEP_ANCHORS MICRO_ANCHORS JOBS_PER_RANK MAX_RANK_TOKENS
    export LR_SCALE NEW_LR_MULT REDUCER_LR TEMPERATURE COV_WEIGHT OFFLINE_ACCEPT_WEIGHT WARM_START OPD_ROOT
    export STEP_SCHEDULE RESUME SMOKE HELDOUT_FROM
    : "${CAPTURE_MODEL_PATH:?robust capture requires a staged local target}"
    export CAPTURE_MODEL_PATH
    "$PYTHON" work/qwen38fn_dflash/robust_transition.py --check-launch "$ROBUST_TRANSITION"
fi
[ "$WARM_NEW_DATA" = 1 ] && : "${WARM_START:?WARM_NEW_DATA qualifies WARM_START}"
[ "$STEP_SCHEDULE" = 1 ] && : "${STEPS:?STEP_SCHEDULE runs STEPS steps}"
if [ -n "$OPD_ROOT" ]; then
    : "${OPD_WEIGHT:?OPD_ROOT trains with OPD_WEIGHT}" "${OPD_FRACTION:?OPD_ROOT needs OPD_FRACTION}"
    [ -f "$OPD_ROOT/train.producer_done" ] || { echo "✗ $OPD_ROOT is not done (opd_assemble.py --done)" >&2; exit 1; }
fi
meta() {  # dotted path into the plan's plan_meta.json
    python3 -c 'import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    v = v[k]
print(v)' "$PLAN/plan_meta.json" "$1"
}
PROFILE=$(meta target_profile.path)
SERVED=$(meta target_profile.served_model_name)
CTX=$(meta context_length)
OUT=$ARC_RUNS/qwen38fn_dflash/$TAG
RAM=$ARC_RAM_RUNS/qwen38fn_dflash/$TAG
read -ra CAP <<< "${CAPTURE_GPUS//,/ }"
read -ra TRN <<< "${TRAINER_GPUS//,/ }"
N=${#CAP[@]}

if [ "$RESUME" = 1 ]; then
    [ -f "$OUT/trainer/latest" ] || { echo "✗ nothing to resume in $OUT/trainer" >&2; exit 1; }
    LATEST=$OUT/trainer/$(cat "$OUT/trainer/latest")
    [ -f "$LATEST/consumed.jsonl" ] || { echo "✗ $LATEST has no consumed.jsonl" >&2; exit 1; }
    # Every process of the run is gone (this script owns them all), so an active.json is a dead capture's.
    rm -f "$RAM"/pool/shard-*/active.json "$RAM"/pool/shard-*/active.partial "$OUT/trainer/STOP"
    echo "· resuming from $LATEST ($(wc -l < "$LATEST/consumed.jsonl") jobs consumed)"
else
    [ -e "$OUT" ] && { echo "✗ $OUT exists: pick a fresh TAG or RESUME=1" >&2; exit 1; }
    [ -e "$RAM" ] && { echo "✗ $RAM exists" >&2; exit 1; }
    mkdir -p "$OUT"
fi
mkdir -p "$RAM/pool"
if [ -n "$HELDOUT_FROM" ]; then
    if [ ! -e "$RAM/heldout" ] && [ ! -L "$RAM/heldout" ]; then
        ln -s "$(realpath "$HELDOUT_FROM")" "$RAM/heldout"
    fi
    [ "$(realpath "$RAM/heldout")" = "$(realpath "$HELDOUT_FROM")" ] || { echo "✗ $RAM/heldout is not $HELDOUT_FROM" >&2; exit 1; }
else
    mkdir -p "$RAM/heldout"
fi
if [ -n "$OPD_ROOT" ] && [ ! -e "$RAM/pool/opd" ]; then
    ln -s "$(realpath "$OPD_ROOT")" "$RAM/pool/opd"
fi

for g in "${CAP[@]}" "${TRN[@]}"; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    if ((used > 0)); then
        echo "✗ GPU $g is in use ($used MiB): shared machine, check whose it is" >&2
        exit 1
    fi
done

OWNED=()  # PID files of everything this script started
stop_owned() {
    local f p
    for f in "${OWNED[@]}"; do
        [ -f "$f" ] || continue
        p=$(cat "$f")
        kill -0 "$p" 2>/dev/null && kill -TERM "$p" && echo "· stopping $(basename "$f" .pid) ($p)"
    done
    for _ in $(seq 1 120); do
        local alive=0
        for f in "${OWNED[@]}"; do [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && alive=1; done
        ((alive)) || return 0
        sleep 1
    done
    for f in "${OWNED[@]}"; do
        [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && echo "✗ $(basename "$f" .pid) ($(cat "$f")) still alive" >&2
    done
}
trap stop_owned EXIT

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 30 >> "$OUT/gpu.csv" &
echo $! > "$OUT/sampler.pid"
OWNED+=("$OUT/sampler.pid")

# One profile-mode capture server per capture GPU, every source through it (capture_plan.py checks each ledger).
for i in "${!CAP[@]}"; do
    rm -f "$OUT/server-$i.pid"
    OWNED+=("$OUT/server-$i.pid")
    GPUS=${CAP[$i]} PORT=$((BASE_PORT + i)) TARGET_PROFILE=$PROFILE CONTEXT_LENGTH=$CTX QUEUE=$RAM/pool/shard-$i \
        PIDFILE=$OUT/server-$i.pid setsid bash work/qwen38fn_dflash/capture_tito.sh \
        >> "$OUT/server-$i.log" 2>&1 < /dev/null &
done
deadline=$((SECONDS + 1800))
for i in "${!CAP[@]}"; do
    until [ -f "$OUT/server-$i.pid" ] && curl -sf "http://127.0.0.1:$((BASE_PORT + i))/v1/models" 2>/dev/null \
            | grep -q "\"$SERVED\""; do
        if [ -f "$OUT/server-$i.pid" ] && ! kill -0 "$(cat "$OUT/server-$i.pid")" 2>/dev/null; then
            echo "✗ capture server $i died; see $OUT/server-$i.log" >&2
            exit 1
        fi
        ((SECONDS < deadline)) || { echo "✗ capture server $i not ready in 30 min" >&2; exit 1; }
        sleep 5
    done
    echo "· capture server $i ready on GPU ${CAP[$i]} (port $((BASE_PORT + i)))"
done

produce() {  # produce <split> <i> <queue> <max queue GiB> [extra args...]
    local split=$1 i=$2 queue=$3 gib=$4
    shift 4
    PYTHONPATH=src "$PYTHON" work/qwen38fn_dflash/capture_plan.py --plan "$PLAN/plan.jsonl" --split "$split" \
        --shard "$i/$N" --url "http://127.0.0.1:$((BASE_PORT + i))" --queue "$queue" --max-queue-gib "$gib" "$@" \
        >> "$OUT/capture-$split-$i.log" 2>&1 &
    echo $! > "$OUT/capture-$split-$i.pid"
    OWNED+=("$OUT/capture-$split-$i.pid")
}

HELDOUT=()
if [ -n "$HELDOUT_FROM" ]; then
    HELDOUT=("$RAM"/heldout/shard-*)
else
    for i in "${!CAP[@]}"; do HELDOUT+=("$RAM/heldout/shard-$i"); done
fi
heldout_done() {
    local q
    for q in "${HELDOUT[@]}"; do [ -f "$q/heldout.producer_done" ] || return 1; done
}
if [ -n "$HELDOUT_FROM" ] && ! heldout_done; then
    echo "✗ $HELDOUT_FROM holds no finished held-out capture (shard-*/heldout.producer_done)" >&2
    exit 1
fi
if ! heldout_done; then
    # Nothing consumes held-out until it is moved out, so each producer's bound must hold its whole shard
    # (data_plan.split_queue_gib); check that RAM holds that plus the training pool's bound first (PR #6 F1).
    mapfile -t HELDOUT_GIB < <(PYTHONPATH=src "$PYTHON" - "$PLAN/plan.jsonl" "$N" <<'PY'
import sys
sys.path.insert(0, "work/qwen38fn_dflash")
from capture_tito import GR_BYTES_PER_TOKEN
from data_plan import load_plan, split_queue_gib
_, jobs = load_plan(sys.argv[1])
for gib in split_queue_gib(jobs, "heldout", int(sys.argv[2]), GR_BYTES_PER_TOKEN):
    print(f"{gib:.1f}")
PY
)
    ((${#HELDOUT_GIB[@]} == N)) || { echo "✗ held-out capacity: ${HELDOUT_GIB[*]}" >&2; exit 1; }
    need=$(python3 -c 'import sys; print(int(sum(map(float, sys.argv[2:])) + int(sys.argv[1]) + 32))' \
        "$((QUEUE_GIB * N))" "${HELDOUT_GIB[@]}")
    avail=$(( $(df --output=avail -k "$RAM" | tail -1) / 1024 / 1024 ))
    ((avail >= need)) || { echo "✗ $RAM has $avail GiB free; held-out + pool need $need GiB" >&2; exit 1; }
    echo "· held-out bounds per shard (GiB): ${HELDOUT_GIB[*]}; RAM free $avail GiB, need $need GiB"
    for i in "${!CAP[@]}"; do
        rm -rf "${HELDOUT[$i]}" "$RAM"/pool/shard-$i/heldout-* "$RAM/pool/shard-$i/heldout.producer_done"
        produce heldout "$i" "$RAM/pool/shard-$i" "${HELDOUT_GIB[$i]}"
    done
    for i in "${!CAP[@]}"; do
        wait "$(cat "$OUT/capture-heldout-$i.pid")" || { echo "✗ held-out producer $i failed" >&2; exit 1; }
        mkdir -p "${HELDOUT[$i]}"
        mv "$RAM"/pool/shard-$i/heldout-* "${HELDOUT[$i]}/"
        mv "$RAM/pool/shard-$i/heldout.producer_done" "${HELDOUT[$i]}/"
        rm "$RAM/pool/shard-$i/producer.json"  # it names the held-out split; the root's train producer writes its own
    done
    echo "· held-out captured: $(ls -d "$RAM"/heldout/shard-*/heldout-* | wc -l) jobs, $(du -sh "$RAM/heldout" | cut -f1)"
fi

SKIP=()
[ -n "$WARM_START" ] && [ "$WARM_NEW_DATA" != 1 ] && SKIP=(--skip-consumed "$WARM_START/consumed.jsonl")
[ "$RESUME" = 1 ] && SKIP=(--skip-consumed "$LATEST/consumed.jsonl")  # carries the warm start's on
for i in "${!CAP[@]}"; do
    produce train "$i" "$RAM/pool/shard-$i" "$QUEUE_GIB" "${SKIP[@]}"
done

python3 - "$OUT/launch.json" <<PY
import hashlib, json, subprocess, sys
from pathlib import Path
plan = Path("$PLAN")
launch = {
    "tag": "$TAG", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "work/qwen38fn_dflash"], text=True).strip()),
    "plan": str(plan), "plan_sha256": hashlib.sha256((plan / "plan.jsonl").read_bytes()).hexdigest(),
    "plan_summary": json.loads((plan / "summary.json").read_text()),
    "plan_config": {k: json.loads((plan / "plan_meta.json").read_text())[k] for k in ("argv", "seed", "corpora", "code")},
    "init": "$INIT", "smoke": "$SMOKE" == "1",
    "gate": "${GATE:-}", "gate_sha256": hashlib.sha256(Path("${GATE:-}").read_bytes()).hexdigest() if "${GATE:-}" else None,
    "capture_gpus": "$CAPTURE_GPUS", "trainer_gpus": "$TRAINER_GPUS", "target_profile": "$PROFILE",
    "capture_queue_gib": $QUEUE_GIB, "resume": "$RESUME" == "1", "warm_start": "$WARM_START" or None,
    "warm_new_data": "$WARM_NEW_DATA" == "1", "step_schedule": "$STEP_SCHEDULE" == "1", "opd_root": "$OPD_ROOT" or None,
    "opd_weight": "${OPD_WEIGHT:-}" or None, "opd_fraction": "${OPD_FRACTION:-}" or None, "opd_loss": "$OPD_LOSS",
    "offline_accept_weight": "$OFFLINE_ACCEPT_WEIGHT", "heldout_from": "$HELDOUT_FROM" or None,
    "opd_heldout": "$OPD_HELDOUT" or None,
    "prefix_gate_alpha": "$PREFIX_GATE_ALPHA" or None, "prefix_gate_temperature": "$PREFIX_GATE_TEMPERATURE" or None,
    "lr_schedule": "$LR_SCHEDULE", "lr_decay_fraction": "$LR_DECAY_FRACTION" or None,
    "cov_weight": "$COV_WEIGHT", "cov_weight_final": "$COV_WEIGHT_FINAL" or None,
}
Path(sys.argv[1]).write_text(json.dumps(launch, indent=2) + "\n")
PY

RESUME_ARGS=()
[ "$RESUME" = 1 ] && RESUME_ARGS=(--resume "$LATEST")
[ -n "$ROBUST_TRANSITION" ] && RESUME_ARGS+=(--robust-transition "$ROBUST_TRANSITION")
[ -n "$WARM_START" ] && RESUME_ARGS+=(--warm-start "$WARM_START" --rewarm-steps "$REWARM_STEPS")
[ "$WARM_NEW_DATA" = 1 ] && RESUME_ARGS+=(--warm-start-new-data)
OPD_ARGS=(--offline-accept-weight "$OFFLINE_ACCEPT_WEIGHT")
if [ -n "$OPD_HELDOUT" ]; then
    [ -f "$OPD_HELDOUT/heldout.producer_done" ] || { echo "✗ $OPD_HELDOUT is not done (opd_assemble.py --heldout --done)" >&2; exit 1; }
    OPD_ARGS+=(--opd-heldout "$OPD_HELDOUT")
fi
[ -n "$OPD_ROOT" ] && OPD_ARGS+=(--opd-weight "$OPD_WEIGHT" --opd-fraction "$OPD_FRACTION" --opd-loss "$OPD_LOSS"
                               --opd-jobs-per-rank "$OPD_JOBS_PER_RANK")
[ -n "$MAX_RANK_TOKENS" ] && RESUME_ARGS+=(--max-rank-tokens "$MAX_RANK_TOKENS")
RECIPE_ARGS=(--lr-schedule "$LR_SCHEDULE")
[ -n "$LR_DECAY_FRACTION" ] && RECIPE_ARGS+=(--lr-decay-fraction "$LR_DECAY_FRACTION")
[ -n "$COV_WEIGHT_FINAL" ] && RECIPE_ARGS+=(--cov-weight-final "$COV_WEIGHT_FINAL")
if [ -n "$PREFIX_GATE_ALPHA$PREFIX_GATE_TEMPERATURE" ]; then  # train.py refuses one without the other
    RECIPE_ARGS+=(--prefix-gate-alpha "$PREFIX_GATE_ALPHA" --prefix-gate-temperature "$PREFIX_GATE_TEMPERATURE")
fi
if [ "$SMOKE" = 1 ]; then
    MODE_ARGS=(--smoke --checkpoint-smoke)
elif [ "$STEP_SCHEDULE" = 1 ]; then
    MODE_ARGS=(--gate-report "$GATE" --step-schedule)
else
    # the plan's train anchors over every pass, and the step estimate (a plan counted by another anchor rule fails
    # capture_plan's per-job check)
    TOTALS=$(PYTHONPATH=work/qwen38fn_dflash "$PYTHON" - "$PLAN" "$STEP_ANCHORS" <<'PY'
import json, sys
from pathlib import Path
from data_plan import ANCHOR_RULE
from pool import planned_steps
plan = Path(sys.argv[1])
if json.loads((plan / "plan_meta.json").read_text()).get("anchor_rule") != ANCHOR_RULE:
    raise SystemExit(f"{plan} counts anchors by another rule than data.eligible_anchors ({ANCHOR_RULE}): rebuild it")
train = [split["train"] for split in json.loads((plan / "summary.json").read_text())["by_split"].values()]
anchors, tokens = sum(t["eligible_anchors"] for t in train), sum(t["tokens"] for t in train)
print(anchors, planned_steps(anchors, tokens, int(sys.argv[2])))
PY
)  # a failure here stops the script (set -e); a process substitution's would not
    read -r TOTAL_ANCHORS STEPS <<< "$TOTALS"
    MODE_ARGS=(--gate-report "$GATE" --total-anchors "$TOTAL_ANCHORS")
fi
WANDB_DIR=$OUT CUDA_VISIBLE_DEVICES=$TRAINER_GPUS OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$TORCHRUN" --standalone --nproc-per-node="${#TRN[@]}" work/qwen38fn_dflash/train.py \
    --engine fast --objective acceptance --temperature "$TEMPERATURE" --cov-weight "$COV_WEIGHT" --selector-grad joint \
    --checkpoint "$INIT" --pool "$RAM/pool" --heldout "${HELDOUT[@]}" --out "$OUT/trainer" \
    --dataset-manifest "$PLAN/summary.json" "${MODE_ARGS[@]}" --steps "$STEPS" --step-anchors "$STEP_ANCHORS" \
    --micro-anchors "$MICRO_ANCHORS" --jobs-per-rank "$JOBS_PER_RANK" --lr-scale "$LR_SCALE" --new-lr-mult "$NEW_LR_MULT" \
    --reducer-lr "$REDUCER_LR" \
    --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" \
    --eval-anchors "$EVAL_ANCHORS" --eval-by-corpus --wandb-name "$TAG" --wandb-extra "$OUT/launch.json" "${RESUME_ARGS[@]}" \
    --wandb-project "$WANDB_PROJECT" "${OPD_ARGS[@]}" "${RECIPE_ARGS[@]}" \
    >> "$OUT/trainer.log" 2>&1 &
echo $! > "$OUT/trainer.pid"
OWNED+=("$OUT/trainer.pid")
echo "· trainer on GPUs $TRAINER_GPUS; following $OUT/trainer.log"

declare -A finished=()
while kill -0 "$(cat "$OUT/trainer.pid")" 2>/dev/null; do
    for i in "${!CAP[@]}"; do
        [ -n "${finished[$i]:-}" ] && continue
        p=$(cat "$OUT/capture-train-$i.pid")
        kill -0 "$p" 2>/dev/null && continue
        if wait "$p"; then
            finished[$i]=1
            echo "· producer $i done: $(tail -1 "$OUT/capture-train-$i.log")"
        else
            echo "✗ producer $i failed; see $OUT/capture-train-$i.log" >&2
            touch "$OUT/trainer/STOP"  # checkpoint, then exit; resume with RESUME=1 after fixing the cause
            wait "$(cat "$OUT/trainer.pid")" || true
            exit 1
        fi
    done
    sleep 10
done
wait "$(cat "$OUT/trainer.pid")" || { echo "✗ trainer failed; see $OUT/trainer.log" >&2; exit 1; }
echo "· trainer finished: $(cat "$OUT/trainer/run-result.json" | tr -d '\n')"
