#!/bin/bash
# Model server for the gen-2 seed baseline (from 10-26, node2): qadR1000 INT3 (QAD-R, the starting point of ARC3 gen-2
# single-level RL) v0.
# Adapted from serve.sh; differences:
# - model qwen38fn-qadR1000-int3-gptq; the fork is RL's lock target b200-rl-topp-inldev-1026 (rl-topp-mask f9cf529b0 =
#   bdf43e509 + ARC_RESPONSES_INLINE_DEVELOPER) with that env turned on -- the same tree as the later RL rollouts.
# - no ARC_THINK_BUDGET: RL rollouts have no think budget (a forced </think> is not sampled), and the baseline must be
#   measured the same way as RL.
# - drafter same as v4b (planb-l8 FP8, DFlash K4). unit g2base-srv-g<GPU>, port 30467+GPU.
#   G2SEED_GPU=<0-7> G2SEED_SRV_DIR=<dir> bash serve_qadr.sh   # stop: systemctl --user stop g2base-srv-g<GPU>
set -euo pipefail
G=${G2SEED_GPU:-7}
PORT=$((30467 + G))
UNIT=g2base-srv-g$G
PROFILE=perf-main-fp16-ser-opt-lazyfix-l8-a16-topp
LOCK=b200-rl-topp-inldev-1026
MODEL=/dev/shm/uid-$UID/models/qwen38fn-qadR1000-int3-gptq
FORK=$HOME/work/sglang-arc-2-topp
DRAFTER=$HOME/work/runs/qwen38fn_rl/dflash-rollout/drafter/planb-l8-s25562/deploy-fp8
MAXREQ=80
R=$HOME/work/runs/arc_code/20261021-g2seed
SRV=${G2SEED_SRV_DIR:?G2SEED_SRV_DIR=<server log dir>}
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
if [ "${G2SEED_INNER:-0}" != 1 ]; then
    systemctl --user is-active --quiet "$UNIT" && { echo "✗ unit $UNIT is already running" >&2; exit 1; }
    if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "✗ port $PORT already has a listener" >&2; exit 1; fi
    case "$(git -C "$FORK" rev-parse HEAD)" in f9cf529b0*) ;; *) echo "✗ $FORK is not at f9cf529b0" >&2; exit 1;; esac
    [ -z "$(git -C "$FORK" status --short)" ] || { echo "✗ $FORK has uncommitted changes" >&2; exit 1; }
    mkdir -p "$SRV"
    # profile env (graph sizes cut to MAXREQ) + drafter kind check; ser's SESSION_PARK removed; RL_INT3_ACT → ARC_INT3_ACT
    (cd "$W" && PYTHONPATH=src:work/qwen38fn_rl .venv/bin/python - "$PROFILE" "$MAXREQ" "$DRAFTER" "$SRV/profile_env.txt" <<'PY'
import sys
from pathlib import Path
import configure_b200 as c
name, sessions, drafter, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
p = c.SERVE_PROFILES[name]
if p["model_kind"] != "int3" or p["tp"] != 1:
    raise SystemExit(f"{name}: {p['model_kind']} tp{p['tp']}")
if p["args"] != ["--mamba-radix-cache-strategy", "extra_buffer", "--session-exact-resume", "--session-replay-own-last-turn"]:
    raise SystemExit(f"{name}: args {p['args']} (serve.sh keeps only the mamba radix strategy)")
c.check_drafter(drafter, p["drafter"])
env = []
for item in c.profile_env(p, sessions):
    key, value = item.split("=", 1)
    if key == "ARC_DFLASH_SESSION_PARK":
        continue                                  # part of exact resume (see the file header)
    if key == "RL_INT3_ACT":
        if value != "bf16":
            raise SystemExit(f"{item}: want bf16")
        item = "ARC_INT3_ACT=bf16"
    env.append(item)
if "HUMMING_VERSION=0.1.18" not in env or "ARC_DFLASH_EXACT_PREFILL=1" not in env:
    raise SystemExit(f"profile env lacks HUMMING_VERSION=0.1.18 / ARC_DFLASH_EXACT_PREFILL=1: {env}")
if any(e.startswith("ARC_INT3_TUNING=") for e in env):
    raise SystemExit("a16 profile must not carry the W3A8 tuning table")
out.write_text("\n".join(env) + "\n")
print(f"profile {name} minus ser: {len(env)} env -> {out}")
PY
    )
    systemd-run --user --unit="$UNIT" --collect --working-directory="$W" --setenv=G2SEED_INNER=1 --setenv=G2SEED_GPU="$G" --setenv=G2SEED_SRV_DIR="$SRV" \
        -p StandardOutput=append:"$SRV/server.log" -p StandardError=append:"$SRV/server.log" \
        /bin/bash "$W/work/arc_code/gen2/seed_collect/serve_qadr.sh"
    echo "· unit $UNIT started; port $PORT; log: $SRV/server.log"
    exit 0
fi
# ---- below runs inside the unit ----
cd "$W"
echo $$ > "$SRV/server.pid"
export ARC_SGLANG_FORK=$FORK LOCK_TARGET=$LOCK GPUS=$G PORT=$PORT SPEC=dflash NAME=duckx    # launch_codex.sh calls --model duckx
export DFLASH_DIR=$DRAFTER MODEL=$MODEL
export TP=1 EP=1 MNNVL_HC=0 INT3_DEPLOY=1 KV_CACHE_DTYPE=fp8_e4m3 ARC_PLE_INT5=1
export MAXREQ DFLASH_K=4 MEMFRAC=0.86 MAMBA_CACHE_SLOTS=384 CTXLEN=262144
export ARC_RESPONSES_INLINE_DEVELOPER=1
while IFS= read -r kv; do [ -n "$kv" ] && export "${kv?}"; done < "$SRV/profile_env.txt"
[ "$ARC_INT3_ACT" = bf16 ] && [ "$HUMMING_VERSION" = 0.1.18 ] || { echo "✗ ARC_INT3_ACT=$ARC_INT3_ACT HUMMING_VERSION=$HUMMING_VERSION" >&2; exit 1; }
[ -z "${ARC_DFLASH_SESSION_PARK:-}" ] || { echo "✗ ARC_DFLASH_SESSION_PARK must not be set" >&2; exit 1; }
# lm_head rows padded beyond the tokenizer must never be sampled (launch_b200_stream.sh vocab_limit_setup)
ARC_LOGITS_VOCAB_LIMIT=$(python3 - "$MODEL/tokenizer.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
print(max(list(t["model"]["vocab"].values()) + [a["id"] for a in t.get("added_tokens", [])]) + 1)
PY
)
[ "$ARC_LOGITS_VOCAB_LIMIT" = 248077 ] || { echo "✗ ARC_LOGITS_VOCAB_LIMIT $ARC_LOGITS_VOCAB_LIMIT, want 248077" >&2; exit 1; }
export ARC_LOGITS_VOCAB_LIMIT
env | sort > "$SRV/serve_env.txt"
exec bash work/sglang_fork/serve_b200.sh \
    --weight-version 0 \
    --no-mm-literal-image-sentinel \
    --enable-metrics \
    --dtype bfloat16 --moe-runner-backend humming --fp8-gemm-backend flashinfer_cutlass \
    --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \
    --stream-interval 8 --scheduler-recv-interval 8 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.1 \
    --prefill-delayer-max-delay-ms 1000 --prefill-delayer-max-delay-passes 20 --ple-mm-placeholder-ids \
    --mamba-radix-cache-strategy extra_buffer
