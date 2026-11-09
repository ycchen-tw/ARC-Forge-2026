#!/bin/bash
# Health recorder. **Records only, never restarts** -- on a shared machine, automatic restarts are the shortest path
# from "I think it is running" to "it is running on someone else's GPUs". The runner can resume anyway (it scans
# (uid, attempt) in out/*.parquet); a crash loses at most the unflushed batch, and a human decides whether to resume.
#
#   bash work/qwen38fn_distill/watch.sh &          # one row every 300 seconds by default
#
# Columns: time / done / shard count / server-running-req / KV usage / accept_len /
#          gen tok/s / /data free GB / runner alive / server alive
#
# KV usage is the most important column here: **a full KV raises no error**; the radix cache cannot keep prefixes,
# every turn re-prefills the whole context, and the only symptom is "throughput halves". So it has to be watched,
# not left to complain on its own.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

RUN=$ARC_RUNS/qwen38fn_distill
PORT=${PORT:-30041}
INTERVAL=${INTERVAL:-300}
HEALTH=$RUN/health.tsv

[ -f "$HEALTH" ] || printf 'time\tdone\tshards\trunning\tkv\taccept\tgen_tps\tdisk_gb\trunner\tserver\n' > "$HEALTH"

while true; do
    now=$(date +%H:%M:%S)
    # Progress uses **shard count × 100**, not run.log's `[n/total]`. Once the runner restarts that counter starts
    # again from 1 (it counts "records this process did", not records on disk), so progress would appear to go
    # backwards. Shards on disk are the truth. Each shard holds exactly flush_every records.
    shards=$(ls "$RUN"/out/*.parquet 2>/dev/null | wc -l)
    done=$((shards * 100))
    last=$(grep "Decode batch" "$RUN/serve.log" 2>/dev/null | tail -1)
    running=$(sed -nE 's/.*#running-req: ([0-9]+).*/\1/p' <<< "$last")
    kv=$(sed -nE 's/.*full token usage: ([0-9.]+).*/\1/p' <<< "$last")
    accept=$(sed -nE 's/.*accept len: ([0-9.]+).*/\1/p' <<< "$last")
    tps=$(sed -nE 's/.*gen throughput \(token\/s\): ([0-9.]+).*/\1/p' <<< "$last")
    disk=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
    # Recognize only our own session name, exact match (prefix matching once killed someone else's job)
    runner=$(tmux has-session -t '=fndistill_run' 2>/dev/null && echo 1 || echo 0)
    server=$(curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo 1 || echo 0)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$now" "${done:-0}" "$shards" "${running:-?}" "${kv:-?}" "${accept:-?}" \
        "${tps:-?}" "$disk" "$runner" "$server" >> "$HEALTH"
    sleep "$INTERVAL"
done
