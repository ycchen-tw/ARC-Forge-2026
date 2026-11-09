#!/bin/bash
# Unattended watchdog: two Qwen3.8 servers (GPU 0-3 :30031, GPU 4-7 :30032) + two harness shards (a / b).
#
#   setsid nohup bash work/agentic_distill/supervise.sh > $ARC_RUNS/agentic_distill/supervisor.nohup 2>&1 < /dev/null &
#   SIDES=b LIVE=160 TLIVE=80 setsid nohup bash work/agentic_distill/supervise.sh ...   # GPU 4-7 only
#   SIDES=b LIVE=300 TLIVE=150 MAXREQ=300 SSM_DTYPE=float32 LINEAR_DECODE_BACKEND=triton REPLAY_SSM=1 ...
#     systemd-run --user --unit=agentic-distill-mix-b  # keeps out of the cursor-worker cgroup
#
# Inherits the three safeguards of work/qwen38fn_distill/supervise.sh (restart cap, cooldown + wait for /health, kill
# only its own PIDs), with these differences:
# - **No tmux, no session cgroup**: everything is `setsid nohup` + PID files (on 2026-09-03 a session crash took the
#   servers in its cgroup down together with four arms).
# - Kills only via PID files, and first checks that the PID's cmdline carries its own port / out dir; no pkill -f.
# - Server start-up goes through serve_qwen38fn.sh, which refuses if the target GPUs have someone else's process (also
#   refuses if it cannot tell).
# - Done criterion: number of tasks/*.json on disk ≥ list length (--resume makes restarts free).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

TAGS=${TAGS:-"mix"}   # mix = nvarc_training + nvarc_full mixed shuffle (each task records source); train / nvfull are old single-split lists
# SIDES=b runs only GPU 4-7 (for when 0-3 are taken). Do not call it SHARDS: the training scripts already use that name.
SIDES=${SIDES:-"a b"}
LIVE=${LIVE:-96}
TLIVE=${TLIVE:-48}
MAXREQ=${MAXREQ:-256}
SSM_DTYPE=${SSM_DTYPE:-bfloat16}
LINEAR_DECODE_BACKEND=${LINEAR_DECODE_BACKEND:-flashinfer}
# REPLAY_SSM=1: Qwen3.8-FN requires fp32 + triton verify (measured on specdata). Watchdog restarts must carry it too.
REPLAY_SSM=${REPLAY_SSM:-0}
INTERVAL=${INTERVAL:-120}
MAX_RESTARTS=${MAX_RESTARTS:-8}
COOLDOWN=${COOLDOWN:-90}
R=$ARC_RUNS/agentic_distill
LOG=$R/supervisor.log
MODEL=${MODEL:-$ARC_NFS/models/Qwen3.8-Flash-Next-FP8}
declare -A GPUS=([a]=0,1,2,3 [b]=4,5,6,7)
declare -A PORT=([a]=30031 [b]=30032)
declare -A restarts=([a]=0 [b]=0)
for s in $SIDES; do
    case "$s" in a|b) ;; *) echo "✗ SIDES may contain only a / b, got $s" >&2; exit 1;; esac
done
if [ "$REPLAY_SSM" = 1 ]; then
    "${SGLANG_VENV:-$HOME/work/dsv4/venv-qwen4}/bin/python" - <<'PY' || exit 1
import inspect
from sglang.srt.speculative import spec_utils
src = inspect.getsource(spec_utils.commit_mamba_states_after_verify)
if "commit_ple_states_after_replayssm(" not in src:
    raise SystemExit("✗ apply patch 0018 before ReplaySSM")
PY
    [ "$SSM_DTYPE" = float32 ] || { echo "✗ REPLAY_SSM=1 requires SSM_DTYPE=float32, got $SSM_DTYPE" >&2; exit 1; }
fi
echo $$ > "$R/supervisor.pid"

log() { printf '%s %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
server_up() { curl -s -m 5 "http://127.0.0.1:${PORT[$1]}/health" >/dev/null 2>&1; }
pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
# Kill only a process whose PID is in the PID file AND whose cmdline contains the given string.
kill_pidfile() {
    local f=$1 needle=$2
    [ -f "$f" ] || return 0
    local pid; pid=$(cat "$f")
    if ps -o cmd= -p "$pid" 2>/dev/null | grep -q -- "$needle"; then
        kill -TERM "$pid" 2>/dev/null; sleep 5; kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
        log "killed $pid ($needle)"
    else
        log "pidfile $f: pid $pid is not mine (cmdline lacks '$needle'); leaving it alone"
    fi
    rm -f "$f"
}

start_server() {
    local s=$1
    local extra=()
    if [ "$REPLAY_SSM" = 1 ]; then
        extra+=(--enable-linear-replayssm-spec --linear-attn-verify-backend triton)
    fi
    log "server $s: starting (GPUS=${GPUS[$s]} PORT=${PORT[$s]} MAXREQ=$MAXREQ SSM_DTYPE=$SSM_DTYPE REPLAY_SSM=$REPLAY_SSM LIVE=$LIVE TLIVE=$TLIVE)"
    setsid nohup env GPUS=${GPUS[$s]} PORT=${PORT[$s]} CTXLEN=131072 NAME=qwen38fn MODEL=$MODEL \
        MAXREQ=$MAXREQ SSM_DTYPE=$SSM_DTYPE LINEAR_DECODE_BACKEND=$LINEAR_DECODE_BACKEND \
        bash work/qwen38fn_eval/serve_qwen38fn.sh "${extra[@]}" >> "$R/serve_$s.log" 2>&1 < /dev/null &
    for _ in $(seq 1 60); do
        sleep 20
        if server_up "$s"; then
            pgrep -af 'launch_server' | grep -v 'shell-snapshots' | grep -- "--port ${PORT[$s]}" | awk '{print $1}' | head -1 > "$R/serve_${PORT[$s]}.pid"
            log "server $s: ready (pid $(cat "$R/serve_${PORT[$s]}.pid"))"
            return 0
        fi
    done
    log "server $s: not up after 20 min"
    return 1
}

corpus_for() { case "$1" in nvfull*) echo "$HOME/work/data/nvarc_synthetic/nvarc_full.jsonl.zst";; train*) echo "$HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst";; *) echo "$HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst $HOME/work/data/nvarc_synthetic/nvarc_full.jsonl.zst";; esac; }

start_runner() {
    local s=$1
    server_up "$s" || { log "runner $s: server not up; not starting the runner"; return 1; }
    log "runner $s: starting (--resume, TAG=$TAG corpus=$(corpus_for "$TAG"))"
    TAG=$TAG SHARD=$s PORT=${PORT[$s]} LIVE=$LIVE TLIVE=$TLIVE CORPUS="$(corpus_for "$TAG")" setsid nohup \
        bash work/agentic_distill/run_shard.sh >> "$R/$TAG-$s.nohup" 2>&1 < /dev/null &
    sleep 15
    pgrep -af 'agentic_tir.py' | grep -v 'shell-snapshots' | grep -E -- "--out $R/$TAG-$s( |$)" | awk '{print $1}' | head -1 > "$R/$TAG-$s.pid"
    log "runner $s: pid $(cat "$R/$TAG-$s.pid")"
}

done_shard() {
    local s=$1
    local want; want=$(grep -c . "$R/lists/${TAG}_$s.txt")
    local have; have=$(ls "$R/$TAG-$s/tasks" 2>/dev/null | wc -l)
    [ "$have" -ge "$want" ]
}

log "supervisor started pid=$$ TAGS=\"$TAGS\" SIDES=\"$SIDES\" LIVE=$LIVE TLIVE=$TLIVE MAXREQ=$MAXREQ SSM_DTYPE=$SSM_DTYPE REPLAY_SSM=$REPLAY_SSM interval=${INTERVAL}s max_restarts=$MAX_RESTARTS"
for TAG in $TAGS; do
restarts=([a]=0 [b]=0)
log "== TAG=$TAG SIDES=$SIDES: $(for s in $SIDES; do printf '%s %s tasks ' "$s" "$(grep -c . "$R/lists/${TAG}_$s.txt")"; done)"
for s in $SIDES; do
    server_up "$s" || start_server "$s" || true
    pid_alive "$R/$TAG-$s.pid" || done_shard "$s" || start_runner "$s"
done
while true; do
    sleep "$INTERVAL"
    all_done=1
    for s in $SIDES; do
        if done_shard "$s"; then
            continue   # the runner prints its summary and exits by itself; the server stays for the next TAG
        fi
        all_done=0
        if [ "${restarts[$s]}" -ge "$MAX_RESTARTS" ]; then
            log "shard $s hit the restart cap (${restarts[$s]} restarts); stopping, no more restarts until a human looks"
            continue
        fi
        if ! server_up "$s"; then
            log "server $s not responding (restarts=${restarts[$s]})"
            kill_pidfile "$R/$TAG-$s.pid" "--out $R/$TAG-$s"
            kill_pidfile "$R/serve_${PORT[$s]}.pid" "--port ${PORT[$s]}"
            restarts[$s]=$((restarts[$s] + 1))
            sleep "$COOLDOWN"
            start_server "$s" && start_runner "$s"
            continue
        fi
        if ! pid_alive "$R/$TAG-$s.pid"; then
            log "server $s healthy but runner gone (restarts=${restarts[$s]})"
            restarts[$s]=$((restarts[$s] + 1))
            sleep "$COOLDOWN"
            start_runner "$s"
        fi
    done
    runners_dead=1
    for s in $SIDES; do
        pid_alive "$R/$TAG-$s.pid" && runners_dead=0
    done
    if [ "$all_done" = 1 ] && [ "$runners_dead" = 1 ]; then
        log "TAG=$TAG SIDES=$SIDES all done"
        break
    fi
done
done
for s in $SIDES; do
    [ -f "$R/serve_${PORT[$s]}.pid" ] && { log "all done; stopping server $s to free its GPUs"; kill_pidfile "$R/serve_${PORT[$s]}.pid" "--port ${PORT[$s]}"; }
done
log "supervisor exiting"
