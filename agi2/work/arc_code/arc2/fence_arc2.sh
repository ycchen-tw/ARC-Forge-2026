#!/bin/bash
# One POSIX fence per task (ARC-AGI-2). The arc2 version of fence_posix.sh: no gateway, act, record or
# arcengine; what it guards is the "answers" -- the public eval json carries the test outputs, and the
# agent has a shell.
#
#   fence_arc2.sh <workspace> <driver argv...>
#
# Inputs (all set by run_arc2.py; exit 64 if any is missing):
#   ARC2_USER              this task's dedicated user (created beforehand by run_arc2.py)
#   ARC2_RUNTIME           the agent's python prefix
#   ARC2_DRIVER_BIN        codex's install prefix
#   ARC2_DRIVER_ENV        KEY=VALUE file: variables the driver adapter has to setenv
#   ARC2_PROCESS_DATA_KB   per-process RLIMIT_DATA (KB)
#   ARC2_DENY              colon-separated: paths this user **must not** be able to read (answers, original tasks)
#   ARC2_REQUIRE           colon-separated: files this user **must** be able to read (codex's instruction file, tokenizer)
#   ARC2_PYSHADOW          directory holding the empty sitecustomize
#   ARC2_MODEL_HEALTH      the model's /health URL: fetched as this user, must return 200
#   ARC2_NETNS             1 = network cut: the driver runs in a user+net namespace this user creates (netns_inner.sh),
#                          with the model port bridged by bridge.py from ARC2_MODEL_SOCK back to 127.0.0.1:ARC2_MODEL_PORT;
#                          0 = network not cut (only for hosts without internet access, or local testing)
# With ARC2_NETNS=1 also required: ARC2_INNER (netns_inner.sh) ARC2_BRIDGE (bridge.py) ARC2_MODEL_PORT ARC2_MODEL_SOCK
#   ARC2_NET_PROBE (host:port that **must be unreachable** inside the namespace)
# Optional:
#   ARC2_OOM_SCORE_ADJ     set before dropping privileges, so the OOM killer picks the agent rather than sglang
#
# Colab G4 has no CAP_NET_ADMIN (iptables unusable), but an ordinary user can create an unprivileged user namespace
# and inside it has the capability to bring lo up: tested 2026-10-04 on the VM as nobody -- lo up, internet ENETUNREACH.
#
# Exit codes: 64 = broken declaration; 70 = answers readable or internet reachable; 71 = model unreachable.
set -euo pipefail

WORKSPACE=$1; shift
for v in ARC2_USER ARC2_RUNTIME ARC2_DRIVER_BIN ARC2_DRIVER_ENV ARC2_PROCESS_DATA_KB ARC2_DENY \
         ARC2_REQUIRE ARC2_PYSHADOW ARC2_MODEL_HEALTH ARC2_NETNS; do
    [ -n "${!v:-}" ] || { echo "fence_arc2.sh needs $v" >&2; exit 64; }
done
case "$ARC2_PROCESS_DATA_KB" in *[!0-9]*) echo "fence_arc2.sh: ARC2_PROCESS_DATA_KB is not an integer" >&2; exit 64 ;; esac
[ "$(id -u)" -eq 0 ] || { echo "fence_arc2.sh has to start as root" >&2; exit 64; }
USERNAME=$ARC2_USER
id -u "$USERNAME" >/dev/null 2>&1 || { echo "fence_arc2.sh: no user $USERNAME" >&2; exit 64; }
[ -f "$ARC2_PYSHADOW/sitecustomize.py" ] || { echo "fence_arc2.sh: no $ARC2_PYSHADOW/sitecustomize.py" >&2; exit 64; }

DRIVER_ENV=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    case "$line" in *=*) ;; *) echo "fence_arc2.sh: driver env line is not KEY=VALUE: $line" >&2; exit 64 ;; esac
    case "$line" in PYTHONPATH=*) echo "fence_arc2.sh: the driver env may not set PYTHONPATH (it would drop the pyshadow)" >&2; exit 64 ;; esac
    DRIVER_ENV+=("$line")
done < "$ARC2_DRIVER_ENV"

mkdir -p "$WORKSPACE/.tmp" "$WORKSPACE/.driver" "$WORKSPACE/answer"
chown -R "$USERNAME" "$WORKSPACE"
# Harness files stay root-owned and read-only. The agent can delete them (the directory is its own) but not modify
# them; the harness collects answers with its own copy of check.py.
for f in AGENTS.md check.py arc.py budget.txt; do
    [ -e "$WORKSPACE/$f" ] && chown root:root "$WORKSPACE/$f" && chmod 644 "$WORKSPACE/$f"
done
if [ -L "$WORKSPACE/task" ] || [ ! -d "$WORKSPACE/task" ]; then
    echo "ARC2: $WORKSPACE/task is not the task directory" >&2; exit 64
fi
chown -R root:root "$WORKSPACE/task"
chmod -R go-w,a+rX "$WORKSPACE/task"
chmod 700 "$WORKSPACE"
chmod 711 "$(dirname "$WORKSPACE")"

IFS=':' read -r -a deny <<< "$ARC2_DENY"
for path in "${deny[@]}"; do
    [ -n "$path" ] || continue
    [ -e "$path" ] || { echo "ARC2: deny path $path does not exist (a typo would make this check pass)" >&2; exit 64; }
    if runuser -u "$USERNAME" -- test -r "$path" 2>/dev/null; then
        echo "ARC2: $USERNAME can read $path -- refusing" >&2; exit 70
    fi
done
IFS=':' read -r -a need <<< "$ARC2_REQUIRE"
for path in "${need[@]}"; do
    [ -n "$path" ] || continue
    runuser -u "$USERNAME" -- test -r "$path" || { echo "ARC2: $USERNAME cannot read $path" >&2; exit 64; }
done
runuser -u "$USERNAME" -- test -x "$ARC2_DRIVER_BIN/bin/codex" || {
    echo "ARC2: $USERNAME cannot execute $ARC2_DRIVER_BIN/bin/codex" >&2; exit 64; }
if runuser -u "$USERNAME" -- test -w "$WORKSPACE/task"; then
    echo "ARC2: $USERNAME can write $WORKSPACE/task -- refusing" >&2; exit 70
fi

ulimit -d "$ARC2_PROCESS_DATA_KB"
[ "$(ulimit -Hd)" = "$ARC2_PROCESS_DATA_KB" ] || { echo "fence_arc2.sh: RLIMIT_DATA did not take" >&2; exit 64; }
if [ -n "${ARC2_OOM_SCORE_ADJ:-}" ]; then
    echo "$ARC2_OOM_SCORE_ADJ" > /proc/self/oom_score_adj
fi

# ARC2_PYTHONPATH_WORKSPACE=1 (config workspace_pythonpath): the workspace is appended after pyshadow, so scripts in
# /tmp can import arc too; the empty sitecustomize still comes first.
case "${ARC2_PYTHONPATH_WORKSPACE:-0}" in
    1) AGENT_PYTHONPATH="$ARC2_PYSHADOW:$WORKSPACE" ;;
    0) AGENT_PYTHONPATH="$ARC2_PYSHADOW" ;;
    *) echo "fence_arc2.sh: ARC2_PYTHONPATH_WORKSPACE must be 0 or 1" >&2; exit 64 ;;
esac
case "$WORKSPACE" in *:*|*[[:space:]]*) [ "$AGENT_PYTHONPATH" = "$ARC2_PYSHADOW" ] || {
    echo "fence_arc2.sh: workspace path $WORKSPACE cannot go on PYTHONPATH" >&2; exit 64; } ;; esac
BASE_ENV=(
    HOME="$WORKSPACE" TMPDIR="$WORKSPACE/.tmp" LANG=C.UTF-8 TERM=dumb MPLBACKEND=Agg
    PATH="$ARC2_RUNTIME/bin:$ARC2_DRIVER_BIN/bin:/usr/bin:/bin"
    PYTHONPATH="$AGENT_PYTHONPATH"
)
if [ "$ARC2_NETNS" = 1 ]; then
    for v in ARC2_INNER ARC2_BRIDGE ARC2_MODEL_PORT ARC2_MODEL_SOCK ARC2_NET_PROBE; do
        [ -n "${!v:-}" ] || { echo "fence_arc2.sh needs $v with ARC2_NETNS=1" >&2; exit 64; }
    done
    runuser -u "$USERNAME" -- test -r "$ARC2_INNER" -a -r "$ARC2_BRIDGE" -a -w "$ARC2_MODEL_SOCK" || {
        echo "ARC2: $USERNAME cannot use $ARC2_INNER / $ARC2_BRIDGE / $ARC2_MODEL_SOCK" >&2; exit 64; }
    NS_ENV=(ARC2_PY="$ARC2_RUNTIME/bin/python" ARC2_BRIDGE="$ARC2_BRIDGE" ARC2_MODEL_PORT="$ARC2_MODEL_PORT"
            ARC2_MODEL_SOCK="$ARC2_MODEL_SOCK" ARC2_NET_PROBE="$ARC2_NET_PROBE")
    echo "ARC2_FENCE user=$USERNAME uid=$(id -u "$USERNAME") data_kib=$(ulimit -Hd) oom_score_adj=$(cat /proc/self/oom_score_adj) netns=1" >&2
    cd "$WORKSPACE"
    exec runuser -u "$USERNAME" -- env -i "${BASE_ENV[@]}" "${DRIVER_ENV[@]}" "${NS_ENV[@]}" \
        unshare --user --map-root-user --net -- /bin/bash "$ARC2_INNER" "$@"
fi

rc=0
runuser -u "$USERNAME" -- env -i "${BASE_ENV[@]}" ARC2_MODEL_HEALTH="$ARC2_MODEL_HEALTH" \
    "$ARC2_RUNTIME/bin/python" -I - <<'PY' || rc=$?
import os, sys, urllib.request
try:
    with urllib.request.urlopen(os.environ["ARC2_MODEL_HEALTH"], timeout=10) as r:
        sys.exit(0 if r.status == 200 else 71)
except Exception as error:
    print(f"ARC2: model health failed: {error!r}", file=sys.stderr); sys.exit(71)
PY
[ "$rc" -eq 0 ] || exit "$rc"

echo "ARC2_FENCE user=$USERNAME uid=$(id -u "$USERNAME") data_kib=$(ulimit -Hd) oom_score_adj=$(cat /proc/self/oom_score_adj) netns=0" >&2
cd "$WORKSPACE"
exec runuser -u "$USERNAME" -- env -i "${BASE_ENV[@]}" "${DRIVER_ENV[@]}" "$@"
