#!/bin/bash
# One lane of the node1 TITO collection: runs arc2 batches (claimed from the queue) or arc3 rounds in sequence against one
# server, with a tito audit after each batch.
#   bash lane.sh <lane> arc2 <port>
#   bash lane.sh <lane> arc3 <port> <max_runs>
# Stop conditions (always written to lanes/<lane>.log before exit): $R/STOP or lanes/<lane>.stop exists, /data has less
# than MIN_FREE_GB free, the server's /v1/models lacks qwen38fn, 2 consecutive batches with a nonzero harness exit, the
# queue is empty / arc3 has run max_runs. A running batch is never interrupted by this script (arc3's run_arm has no
# SIGTERM handler; an interrupt leaves no tito_manifest).
set -uo pipefail
LANE=$1 KIND=$2 PORT=$3 MAX_RUNS=${4:-0}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=$HOME/work/arc-forge/.claude/worktrees/bridge-cse_012e3ydKyiZL2YKdz5DAbzWn
cd "$WT"
source env/env.sh
MIN_FREE_GB=${MIN_FREE_GB:-120}
TOKDIR=$ARC_MODELS/Qwen3.8-Flash-Next-FP8
AGENT_PY=$ARC_RUNS/arc_code/agent-runtime/bin/python
AUDIT_PY=$HOME/work/dsv4/venv-0520/bin/python
Q=$R/queue/arc2
mkdir -p "$R/lanes" "$R/arc2" "$R/arc3" "$Q"/{pending,claimed,done,failed}
echo $$ > "$R/lanes/$LANE.pid"

log() { echo "$(date -u +%FT%TZ) [$LANE] $*" >> "$R/lanes/$LANE.log"; }
status() { echo "$(date -u +%FT%TZ) $*" > "$R/lanes/$LANE.status"; }
server_ok() { curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" | grep -q '"qwen38fn"'; }
free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }

audit() {  # $1 = run dir (has tito_manifest.json)
    local run=$1 st=$1/ledger_status.json inst
    curl -sf -m 30 "http://127.0.0.1:$PORT/session_ledger/status" > "$st" || { log "AUDIT: status unreadable"; return 1; }
    inst=$("$AUDIT_PY" -c 'import json,sys,os; s=json.load(open(sys.argv[1])); print(os.path.join(s["dir"], s["instance"]) if not s["dir"].rstrip("/").endswith(s["instance"]) else s["dir"])' "$st")
    [ -f "$run/tito_manifest.json" ] || { log "AUDIT: no tito_manifest.json in $run"; return 1; }
    PYTHONPATH=$WT/src "$AUDIT_PY" -m arcforge.tito audit "$inst" --tokenizer "$TOKDIR" \
        --manifest "$run/tito_manifest.json" --ledger-status "$st" --json "$run/tito_audit.json" \
        > "$run/tito_audit.txt" 2>&1
    local rc=$?
    log "AUDIT rc=$rc $(tail -1 "$run/tito_audit.txt" | cut -c1-200)"
    return $rc
}

fails=0
log "start kind=$KIND port=$PORT max_runs=$MAX_RUNS min_free=${MIN_FREE_GB}G"
n=0
while true; do
    if [ -e "$R/STOP" ] || [ -e "$R/lanes/$LANE.stop" ]; then log "stop file -> exit"; status "stopped (stop file)"; exit 0; fi
    f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then log "DISK /data free ${f}G < ${MIN_FREE_GB}G -> exit"; status "stopped (disk ${f}G)"; exit 3; fi
    if ! server_ok; then log "SERVER $PORT not serving qwen38fn -> exit"; status "stopped (server $PORT down)"; exit 4; fi

    if [ "$KIND" = arc2 ]; then
        batch=""
        for b in $(ls "$Q/pending" | sort); do
            if mv "$Q/pending/$b" "$Q/claimed/$b" 2>/dev/null; then batch=$b; break; fi
        done
        [ -n "$batch" ] || { log "queue empty -> exit"; status "done (queue empty)"; exit 0; }
        name=${batch%.txt}
        pool=$(sed -n 's/^pool=//p' "$Q/claimed/$batch"); ids=$(sed -n 's/^ids=//p' "$Q/claimed/$batch")
        run=$R/arc2/$name
        status "running $name ($pool, $(echo "$ids" | tr ',' '\n' | wc -l) tasks)"
        log "begin $name pool=$pool free=${f}G"
        "$AGENT_PY" "$WT/work/arc_code/arc2/run_arc2.py" \
            --config "$WT/work/arc_code/arc2/configs/node1-tito-collect-jail.json" \
            --run-dir "$run" --tasks-dir "$ARC_DATA/arc2_pools/$pool" --only "$ids" \
            --agent-python "$AGENT_PY" --driver-bin "$HOME/work/codex-0.154.0-p7/arc2-jail" \
            --model-base-url "http://127.0.0.1:$PORT/v1" --model qwen38fn \
            --tokenizer-json "$TOKDIR/tokenizer.json" > "$R/lanes/$LANE.$name.out" 2>&1
        rc=$?
        log "end $name rc=$rc"
        audit "$run"
        if [ $rc -eq 0 ]; then mv "$Q/claimed/$batch" "$Q/done/$batch"; fails=0
        else mv "$Q/claimed/$batch" "$Q/failed/$batch"; fails=$((fails + 1)); fi
    else
        n=$((n + 1))
        [ "$n" -le "$MAX_RUNS" ] || { log "arc3 ran $MAX_RUNS runs -> exit"; status "done ($MAX_RUNS runs)"; exit 0; }
        name=$(printf 'run-%02d' "$n"); run=$R/arc3/$name
        status "running arc3 $name"
        log "begin arc3 $name free=${f}G"
        .venv/bin/python work/arc_code/run_arm.py \
            --config work/arc_code/configs/node1-tito-collect-arc3.json \
            --run-dir "$run" --corpus "$ARC_ARC3_CORPUS" --expect-corpus 72fe9ffcd98ba3af \
            --prep "$ARC_DATA/arc-code-prep" \
            --python "$ARC_RUNS/arc_code/harness-venv/bin/python" --agent-python "$AGENT_PY" \
            --driver-bin "$HOME/work/codex-0.154.0-p7/prefix" --driver codex \
            --ffmpeg "$HOME/work/dsv4/ffmpeg-n8.1/bin/ffmpeg" \
            --model-base-url "http://127.0.0.1:$PORT/v1" --model qwen38fn > "$R/lanes/$LANE.$name.out" 2>&1
        rc=$?
        log "end arc3 $name rc=$rc"
        audit "$run"
        if [ $rc -eq 0 ]; then fails=0; else fails=$((fails + 1)); fi
    fi
    if [ "$fails" -ge 2 ]; then log "2 consecutive harness failures -> exit"; status "stopped (2 failures)"; exit 5; fi
done
