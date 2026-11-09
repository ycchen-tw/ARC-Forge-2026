#!/bin/bash
# One bubblewrap fence per episode (ARC-AGI-2, ``fence: bwrap``, no root needed).
#
# Replaces ``codex-0.154.0-p*/arc2-jail/bin/codex`` outside the repo (10-18 audit P6: that one's ``--ro-bind / /`` only
# covered $HOME / /nfs / /tmp and did not cut the network, so the jail could fetch the ARC-AGI-2 evaluation answers from
# GitHub). This follows work/arc_code/jail.sh:
#   * Start from an empty root and mount only the system directories (read-only), the agent runtime, codex, the run's
#     harness/ (instruction files, tokenizer, check.py; no answers), this episode's socket directory (read-only) and the
#     workspace (writable). --tasks-dir, results/, other episodes' workspaces, $HOME, /nfs, /dev/shm are not inside.
#   * --unshare-user + --unshare-net: loopback only. bridge.py connects the model port from this episode's unix socket
#     back to 127.0.0.1:$ARC2_MODEL_PORT (the proxy is on the host side and lets through only the routes codex uses,
#     model_proxy.CODEX_ROUTES).
#   * Before starting, verify inside the jail: /health reaches sglang through the bridge, every path in ARC2_DENY does
#     not exist, outside DNS / TCP both fail.
#
#   jail_arc2.sh <workspace> <driver argv...>
#
# Inputs (all set by run_arc2.py; any one missing -> exit 64):
#   ARC2_RUNTIME         the agent's python prefix (read-only)
#   ARC2_DRIVER_BIN      codex's install prefix (read-only; must be the real codex, not another jail wrapper)
#   ARC2_DRIVER_ENV      KEY=VALUE file (host side, not mounted into the jail): variables the driver adapter sets,
#                        applied after the defaults
#   ARC2_HARNESS         <run>/harness (read-only)
#   ARC2_BRIDGE          bridge.py (mounted read-only as /bridge.py)
#   ARC2_SOCKETS         this episode's socket directory (read-only; holds model.sock)
#   ARC2_MODEL_PORT      the 127.0.0.1 port codex calls inside the jail (= --model-base-url's port; codex's URL unchanged)
#   ARC2_PROCESS_DATA_KB per-process RLIMIT_DATA (KB)
#   ARC2_DENY            colon-separated: paths that **must** not exist inside the jail (--tasks-dir, results/, --deny)
#   ARC2_NET_PROBE       host:port that **must** be unreachable inside the jail (an outside IP, no DNS needed)
#
# Exit codes (run_arc2.run_session goes by these): 64 = bad declaration; 70 = answers visible or outside network
# reachable; 71 = model unreachable.
set -euo pipefail

WORKSPACE=$1; shift
for v in ARC2_RUNTIME ARC2_DRIVER_BIN ARC2_DRIVER_ENV ARC2_HARNESS ARC2_BRIDGE ARC2_SOCKETS ARC2_MODEL_PORT \
         ARC2_PROCESS_DATA_KB ARC2_DENY ARC2_NET_PROBE; do
    [ -n "${!v:-}" ] || { echo "jail_arc2.sh needs $v" >&2; exit 64; }
done
case "$ARC2_PROCESS_DATA_KB" in *[!0-9]*) echo "jail_arc2.sh: ARC2_PROCESS_DATA_KB is not an integer" >&2; exit 64 ;; esac
case "$ARC2_MODEL_PORT" in *[!0-9]*) echo "jail_arc2.sh: ARC2_MODEL_PORT is not an integer" >&2; exit 64 ;; esac
[ -d "$WORKSPACE" ] || { echo "jail_arc2.sh: no workspace $WORKSPACE" >&2; exit 64; }
[ -S "$ARC2_SOCKETS/model.sock" ] || { echo "jail_arc2.sh: no socket $ARC2_SOCKETS/model.sock" >&2; exit 64; }
[ -f "$ARC2_DRIVER_ENV" ] || { echo "jail_arc2.sh: no driver env $ARC2_DRIVER_ENV" >&2; exit 64; }

DRIVER_ENV=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    case "$line" in *=*) ;; *) echo "jail_arc2.sh: driver env line is not KEY=VALUE: $line" >&2; exit 64 ;; esac
    DRIVER_ENV+=(--setenv "${line%%=*}" "${line#*=}")
done < "$ARC2_DRIVER_ENV"

# The venv's bin/python is a symlink; the install root of every hop in the chain must be mounted (reason: the same
# section of work/arc_code/jail.sh).
EXTRA=()
_bind_root() {
    local root=$1
    case "$root" in "$ARC2_RUNTIME"*|/usr|/usr/*) return ;; esac
    for seen in "${EXTRA[@]}"; do [ "$seen" = "$root" ] && return; done
    EXTRA+=(--ro-bind "$root" "$root")
}
_link=$ARC2_RUNTIME/bin/python
for _ in 1 2 3 4 5; do
    _bind_root "$(dirname "$(dirname "$_link")")"
    [ -L "$_link" ] || break
    _next=$(readlink "$_link")
    case "$_next" in /*) _link=$_next ;; *) _link=$(dirname "$_link")/$_next ;; esac
done

mkdir -p "$WORKSPACE/.tmp"
# --unshare-net only works together with --unshare-user (work/arc_code/jail.sh).
exec bwrap \
    --clearenv \
    --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /sbin /sbin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc /etc \
    --proc /proc --dev /dev --tmpfs /tmp --tmpfs /run \
    --ro-bind "$ARC2_RUNTIME" "$ARC2_RUNTIME" \
    --ro-bind "$ARC2_DRIVER_BIN" "$ARC2_DRIVER_BIN" \
    --ro-bind "$ARC2_HARNESS" "$ARC2_HARNESS" \
    --ro-bind "$ARC2_BRIDGE" /bridge.py \
    --ro-bind "$ARC2_SOCKETS" "$ARC2_SOCKETS" \
    "${EXTRA[@]}" \
    --bind "$WORKSPACE" "$WORKSPACE" \
    --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-net \
    --die-with-parent --new-session \
    --chdir "$WORKSPACE" \
    --setenv HOME "$WORKSPACE" \
    --setenv TMPDIR "$WORKSPACE/.tmp" \
    --setenv PATH "$ARC2_RUNTIME/bin:$ARC2_DRIVER_BIN/bin:/usr/bin:/bin" \
    --setenv LANG C.UTF-8 --setenv TERM dumb --setenv MPLBACKEND Agg \
    --setenv ARC_FENCE bwrap \
    --setenv ARC2_PY "$ARC2_RUNTIME/bin/python" \
    --setenv ARC2_SOCKETS "$ARC2_SOCKETS" \
    --setenv ARC2_MODEL_PORT "$ARC2_MODEL_PORT" \
    --setenv ARC2_PROCESS_DATA_KB "$ARC2_PROCESS_DATA_KB" \
    --setenv ARC2_DENY "$ARC2_DENY" \
    --setenv ARC2_NET_PROBE "$ARC2_NET_PROBE" \
    "${DRIVER_ENV[@]}" \
    /bin/bash -c '
        ulimit -d "$ARC2_PROCESS_DATA_KB" || { echo "ARC2: ulimit -d $ARC2_PROCESS_DATA_KB failed" >&2; exit 64; }
        "$ARC2_PY" -I /bridge.py --listen-tcp "127.0.0.1:$ARC2_MODEL_PORT" \
            --connect-unix "$ARC2_SOCKETS/model.sock" 2>> "$HOME/.bridge.log" &
        # -I: a struct.py or the like in the workspace cannot shadow the stdlib (lesson from the same section of work/arc_code/jail.sh).
        "$ARC2_PY" -I - <<'\''PY'\'' || exit $?
import os, socket, sys, time, urllib.request
port = os.environ["ARC2_MODEL_PORT"]
deadline = time.monotonic() + 30
last = None
while True:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            if r.status == 200:
                break
    except OSError as error:
        last = error
    if time.monotonic() > deadline:
        print(f"ARC2: model unreachable through the bridge: {last!r}", file=sys.stderr)
        sys.exit(71)
    time.sleep(0.2)
visible = [p for p in os.environ["ARC2_DENY"].split(":") if p and os.path.lexists(p)]
if visible:
    print(f"ARC2: visible inside the jail: {visible}", file=sys.stderr)
    sys.exit(70)
host, p = os.environ["ARC2_NET_PROBE"].rsplit(":", 1)
try:
    socket.create_connection((host, int(p)), timeout=5).close()
    print(f"ARC2: {host}:{p} is reachable inside the jail", file=sys.stderr)
    sys.exit(70)
except OSError:
    pass
try:
    socket.getaddrinfo("github.com", 443)
    print("ARC2: DNS resolves inside the jail", file=sys.stderr)
    sys.exit(70)
except OSError:
    pass
PY
        echo "ARC2_BWRAP bridge 127.0.0.1:$ARC2_MODEL_PORT, deny paths absent, $ARC2_NET_PROBE unreachable" >&2
        exec "$@"
    ' bash "$@"
