#!/bin/bash
# ARC3 RL v8 (v8.env).   bash launch-v8.sh configure|start   (RL_DRY_RUN=1 prints, starts nothing)
#   configure:  configure_b200 (RL_STAGES="") -> then python3 apply-v8-config.py <run> <worktree> <train ids> [smoke]
#               (<train ids> = the number of lines of $RL_TASKS_DIR/lists/train.txt)
#   start:      the launcher's stages (check servers control init-policy trainer orchestrator routing-gc), or RL_STAGES
# Runs from a clean checkout of branch arc3-v8.
set -euo pipefail
mode=${1:?configure or start}
NOTES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # this checkout
cd "$WT"
test -z "$(git status --porcelain)" || { echo "worktree not clean" >&2; exit 1; }
test -x .venv/bin/python && test -d reference/ && test -f .env
grep -q '"perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev"' work/qwen38fn_rl/configure_b200.py     # v7's FP8-KV serve profile
unset ARC_ROOT; source env/env.sh >/dev/null 2>&1; test "$ARC_ROOT" = "$PWD"
topology=${RL_TOPOLOGY:-} gpus=${RL_GPUS:-}
source "$NOTES/v8.env"
[ "${RL_V8_SMOKE:-0}" != 1 ] || source "$NOTES/smoke.env"    # smoke: window 100k / change 50k, 4 ids, 2 steps
ntrain=$(wc -l < "$RL_TASKS_DIR/lists/train.txt")
ngames=$(cut -d_ -f1 "$RL_TASKS_DIR/lists/train.txt" | sort -u | wc -l)    # ids <game>_L<k>_...: 18 of v7b's 20
tag=g2seed-${ngames}g-${ntrain}t-hw-g16-b256-r4c64-fp8kv-win1-note-${RL_JUDGE_BACKEND}-fullft-lr3e-6-wu2-tk5l-topp95k20-promptmean-mis-qadR
prefix=rl-v8
if [ "${RL_V8_SMOKE:-0}" = 1 ]; then
    tag=smoke-g2seed-${ntrain}t-g16-b64-r4c64-fp8kv-win100k-chg50k-note-${RL_JUDGE_BACKEND}-fullft-lr3e-6 prefix=rl-v8s
fi
test "$(wc -l < "$RL_TASKS_DIR/lists/heldout.txt")" = 2
test "$RL_ARC3_REWARD $RL_TOPOLOGY $RL_WINDOW_CHANGE $RL_STOP_AFTER_FIRST_WINDOW $RL_CONTEXT_LIMIT $RL_SESSIONS $RL_MAMBA_SLOTS $RL_TRAINER_TRAINABLE" = "note tp1x4 1 1 none 64 384 full"
test -z "${RL_INIT_CHECKPOINT:-}${RL_INIT_MERGED:-}"                  # full FT starts from QAD-R
test -x "$RL_JUDGE_PYTHON" && "$RL_JUDGE_PYTHON" -c "import anthropic"
case "$RL_JUDGE_BACKEND" in
    api) test -f "$RL_JUDGE_ENV_FILE"
         (set -a; . "$RL_JUDGE_ENV_FILE"; set +a; test -n "${ANTHROPIC_API_KEY_2:-}") \
             || { echo "ANTHROPIC_API_KEY_2 not in $RL_JUDGE_ENV_FILE" >&2; exit 1; } ;;
    codex) test -z "${RL_JUDGE_ENV_FILE:-}" && codex --version >/dev/null && test -f "$HOME/.codex/auth.json" ;;
    *) echo "RL_JUDGE_BACKEND $RL_JUDGE_BACKEND: api or codex" >&2; exit 1 ;;
esac
[ "${RL_DRY_RUN:-0}" = 1 ] || mkdir -p "$RL_TRAINER_CHECKPOINT_STORE"
[ -z "$topology" ] || export RL_TOPOLOGY=$topology
[ -z "$gpus" ] || export RL_GPUS=$gpus
test -d "$RL_AGENT_RUNTIME/lib/python3.12/site-packages/scipy-1.18.1.dist-info"
python3 work/sglang_fork/humming_overlay.py check 0.1.18 "$ARC_DATA/humming-overlays/0.1.18" >/dev/null
test "$(git -C "$RL_FORK_TREE" rev-parse HEAD)" = f9cf529b05a27878ff59b40a16b2f5477074dbc0
test -z "$(git -C "$RL_FORK_TREE" status --porcelain --untracked-files=no)"
(cd "$RL_DFLASH_DIR" && sha256sum --quiet -c SHA256SUMS)
test -f "$RL_SERVE_MODEL/QAD_DONE" && test -f "$RL_TRAIN_MODEL/QAD_DONE"
export RL_RUN=$ARC_RUNS/qwen38fn_rl/${RL_RUN_DATE:?RL_RUN_DATE=YYYYMMDD}-arc3v8-$tag-public25-contaminated RL_UNIT_PREFIX=$prefix
case "$mode" in
    configure) test ! -e "$RL_RUN" || { echo "$RL_RUN exists" >&2; exit 1; }; export RL_STAGES="" ;;
    start) test -f "$RL_RUN/config.json" && grep -q '"max_routing_m_fraction"' "$RL_RUN/config.json" \
               || { echo "$RL_RUN: configure + apply-v8-config.py first" >&2; exit 1; } ;;
    *) echo "configure or start" >&2; exit 1 ;;
esac
echo "RL_RUN=$RL_RUN  $mode  head $(git rev-parse --short=9 HEAD)  topology $RL_TOPOLOGY gpus ${RL_GPUS:-default}"
bash work/qwen38fn_rl/launch_arc3_round2_int3qat.sh
