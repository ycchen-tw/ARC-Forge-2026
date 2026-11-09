#!/bin/bash
# Watchdog for unattended runs. 47,337 tasks take about 30 hours; a drop midway with nobody to catch it wastes a whole night.
#
#   tmux new-session -d -s fndistill_sup "bash work/qwen38fn_distill/supervise.sh"
#
# ## Why this script dares to restart automatically while watch.sh does not
#
# `watch.sh` only records, because on a shared machine the shortest failure path of "automatic restart" is
# **starting the service on someone else's GPUs**. This script is safe because it restarts via `serve.sh` →
# `serve_qwen38fn.sh`, which before start-up checks every target GPU for processes that are not its own and refuses on
# the spot if there are any (and also refuses if it cannot determine occupancy; it does not fail open). So the
# "can it start" decision is not made here but at that existing gate.
#
# ## Three safeguards
#
# 1. **Restarts are capped** (MAX_RESTARTS, default 6). On hitting the cap it stops and keeps yelling --
#    a crash loop burning GPUs all night is worse than sitting idle.
# 2. **Cooldown after a restart** (COOLDOWN), and after server start-up it waits for /health before letting the runner
#    in. The order cannot be reversed: a runner that cannot reach the server runs all 45,937 rollouts through retry and blows up.
# 3. **Only its own session names, exact match** (`-t '=NAME'`). tmux kill-session falls back to prefix matching when
#    there is no exact match -- this repo has a record of `-t dsv4` killing a job that had run for 10 hours on `dsv4b`.
#
# The runner can resume anyway (it scans (uid, attempt) in out/*.parquet), so the cost of a restart is at most
# "the unflushed batch (<100 records) + the 200 rollouts in flight at that moment".
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

RUN=$ARC_RUNS/qwen38fn_distill
PORT=${PORT:-30041}
INTERVAL=${INTERVAL:-60}
MAX_RESTARTS=${MAX_RESTARTS:-6}
COOLDOWN=${COOLDOWN:-120}
# Runner concurrency (number of live rollouts). Must match the MAXREQ the server was started with
# -- if the server's max_running_requests is smaller, the excess sits in the queue.
CONC=${CONC:-300}
LOG=$RUN/supervisor.log
restarts=0

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

server_up() { curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
have()      { tmux has-session -t "=$1" 2>/dev/null; }

start_server() {
    log "server: starting (serve_qwen38fn.sh does the GPU occupancy check and refuses GPUs held by others)"
    tmux new-session -d -s fndistill_serve \
        "cd $PWD && bash work/qwen38fn_distill/serve.sh 2>&1 | tee -a $RUN/serve.log"
    for _ in $(seq 1 45); do
        sleep 20
        server_up && { log "server: ready"; return 0; }
    done
    log "server: not up within 15 min"
    return 1
}

start_runner() {
    # Always confirm the server is alive first. A runner that cannot reach the server runs all remaining rollouts
    # through retry and then raises; the log gets flooded with tens of thousands of exception lines and nothing is produced.
    server_up || { log "runner: server not up; not starting the runner"; return 1; }
    log "runner: restarting (resume skips finished (uid, attempt) pairs)"
    tmux new-session -d -s fndistill_run \
        "cd $PWD && source env/env.sh && \$HOME/work/dsv4/venv-qwen4/bin/python \
work/qwen38fn_distill/rollout_runner.py \
--corpus \$HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst \
--out \$ARC_RUNS/qwen38fn_distill/out --tokenizer \$ARC_MODELS/Qwen3.8-Flash-Next-FP8 \
--base-url http://127.0.0.1:$PORT \
--sandbox-python \$HOME/work/dsv4/venv-qwen4/bin/python \
--concurrency $CONC --flush-every 100 --log-every 100 \
--heartbeat \$ARC_RUNS/qwen38fn_distill/heartbeat.tsv 2>&1 | tee -a \$ARC_RUNS/qwen38fn_distill/run.log"
}

log "supervisor started (interval=${INTERVAL}s max_restarts=$MAX_RESTARTS cooldown=${COOLDOWN}s)"
while true; do
    sleep "$INTERVAL"

    # Done once everything has run. The criterion is the actual record count on disk, not the log counter
    # -- the counter restarts from 1 after the runner restarts.
    shards=$(ls "$RUN"/out/*.parquet 2>/dev/null | wc -l)
    if [ "$shards" -ge 473 ]; then
        if ! have fndistill_run; then
            log "all done ($shards shards); supervisor exiting"
            exit 0
        fi
    fi

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        log "hit the restart cap ($restarts restarts); stopping, no more restarts. Do not auto-resume before a human looks."
        sleep 600
        continue
    fi

    if ! server_up; then
        log "server not responding"
        have fndistill_run && { log "stopping the runner first so it does not hammer the dead server with retries"
                                tmux kill-session -t '=fndistill_run' 2>/dev/null; }
        have fndistill_serve && tmux kill-session -t '=fndistill_serve' 2>/dev/null
        restarts=$((restarts + 1))
        sleep "$COOLDOWN"
        start_server && start_runner
        continue
    fi

    if ! have fndistill_run; then
        # Server alive but the runner is gone. It may have finished (handled above), or it may be OOM / killed /
        # an exception. Restarting once tells which: if it really finished, resume reports 0 rollouts to run and
        # exits on its own.
        log "server healthy but runner gone (restarts=$restarts)"
        restarts=$((restarts + 1))
        start_runner
    fi
done
