#!/bin/bash
# One bubblewrap fence per game.
#
# The input is a **resource declaration**, not a list of bwrap arguments -- Kaggle has no mounts, and
# there the same declaration has to be rendered by fence_posix as "dedicated user + mode bits".
# See S2 of docs/plans/arc_code_gateway.md.
#
#   ARCSEC_RUNTIME     the agent's python runtime (**not** the harness venv)
#   ARCSEC_ACT         the single file act.py
#   ARCSEC_SOCKETS     the bridges' socket directory (read-only, so they can't unlink each other)
#   ARCSEC_BRIDGE      bridge.py
#   ARCSEC_DRIVER_BIN  the driver's install prefix (codex / pi / ...)
#   ARCSEC_DRIVER_HOME this game's own driver home
#   ARCSEC_DRIVER_ENV  a KEY=VALUE file (host side, not mounted into the jail): variables the driver
#                      adapter wants setenv'd, applied line by line **after** the defaults here -- so a
#                      PATH from the adapter overrides the default PATH. This script knows nothing about
#                      drivers: CODEX_HOME / PI_CODING_AGENT_DIR and the like all live in arms/<driver>.py.
#   ARCSEC_TOKEN       this game's API key
#   ARCSEC_PROCESS_DATA_KB  per-process RLIMIT_DATA inside the jail (KB, hard cap). The scope's MemoryMax is
#                      the whole jail's total; when it is hit the OOM killer kills the driver along with
#                      the agent's script, and the agent sees no error at all (2026-09-19, two rounds, 6 / 7
#                      times each, rerunning the same script every time). RLIMIT_DATA is per process:
#                      Python hitting it raises MemoryError in place, with a line number, and the driver
#                      survives. Must be smaller than MemoryMax.
#   ARCSEC_CPU_SECONDS optional: per-process RLIMIT_CPU inside the jail (seconds, soft = hard). The posix
#                      fence's watchdog kills a single process over its CPU budget; this gives bwrap the
#                      same semantics (the brief states this cap from v12).
#   ARCSEC_ENV_PORT / ARCSEC_MODEL_PORT
#   ARCSEC_EXTRA_RO    colon-separated extra read-only paths
#   ARCSEC_RECORD      optional: the game record written by the middleware (must be $WORKSPACE/record).
#                      The workspace is writable; this layer mounts it back read-only at the same place:
#                      the agent can read it but not change it.
#   ARCSEC_ROLE        optional: player (default) or reviewer. reviewer is an empty-slot read-only review
#                      (work/arc_code/reviewer.py): the workspace is a snapshot with no game -- no
#                      ARCSEC_ACT / ARCSEC_TOKEN / ARCSEC_ENV_PORT, act.py not mounted, no environment
#                      bridge, ARC_API_KEY not set; the health check hits the model's /health instead. The
#                      snapshot's record/ shares inodes with the main game (hardlinks), so a reviewer must
#                      be given ARCSEC_RECORD so it gets mounted back read-only.
#
# The most important thing not mounted: **the harness venv**. It holds the arcengine / arc_agi
# source; the old arm derived it from --python and mounted the whole thing (run_arm.py:291), and 19/25
# games read the actuator source -- that is how `--again` was discovered. act.py is now stdlib-only,
# so that venv isn't needed here at all.
set -euo pipefail

WORKSPACE=$1; shift
ROLE=${ARCSEC_ROLE:-player}
# Operator message below, kept verbatim: "ARCSEC_ROLE=$ROLE is not player / reviewer"
case "$ROLE" in player|reviewer) ;; *) echo "jail.sh: ARCSEC_ROLE=$ROLE is not player/reviewer" >&2; exit 64 ;; esac
: "${ARCSEC_RUNTIME:?jail.sh needs ARCSEC_RUNTIME}"
: "${ARCSEC_SOCKETS:?jail.sh needs ARCSEC_SOCKETS}"
: "${ARCSEC_BRIDGE:?jail.sh needs ARCSEC_BRIDGE}"
: "${ARCSEC_DRIVER_BIN:?jail.sh needs ARCSEC_DRIVER_BIN}"
: "${ARCSEC_MODEL_PORT:?jail.sh needs ARCSEC_MODEL_PORT}"
ROLE_ARGS=()
if [ "$ROLE" = player ]; then
    : "${ARCSEC_ACT:?jail.sh needs ARCSEC_ACT}"
    : "${ARCSEC_TOKEN:?jail.sh needs ARCSEC_TOKEN}"
    : "${ARCSEC_ENV_PORT:?jail.sh needs ARCSEC_ENV_PORT}"
    ROLE_ARGS=(--ro-bind "$ARCSEC_ACT" /act.py
               --setenv ARC_BASE_URL "http://127.0.0.1:$ARCSEC_ENV_PORT"
               --setenv ARC_API_KEY "$ARCSEC_TOKEN"
               --setenv ARCSEC_ENV_PORT "$ARCSEC_ENV_PORT")
else
    for v in ARCSEC_ACT ARCSEC_TOKEN ARCSEC_ENV_PORT; do
        # Operator message below, kept verbatim: "reviewer must not be given $v"
        [ -z "${!v:-}" ] || { echo "jail.sh: reviewer must not receive $v" >&2; exit 64; }
    done
    # Operator message below, kept verbatim: "reviewer needs ARCSEC_RECORD (the snapshot record/ is mounted back read-only)"
    [ -n "${ARCSEC_RECORD:-}" ] || { echo "jail.sh: reviewer needs ARCSEC_RECORD (the snapshot record/, mounted back read-only)" >&2; exit 64; }
fi
: "${ARCSEC_PROCESS_DATA_KB:?jail.sh needs ARCSEC_PROCESS_DATA_KB}"
# Operator message below, kept verbatim: "ARCSEC_PROCESS_DATA_KB=... is not an integer KB"
case "$ARCSEC_PROCESS_DATA_KB" in ''|*[!0-9]*) echo "jail.sh: ARCSEC_PROCESS_DATA_KB=$ARCSEC_PROCESS_DATA_KB is not an integer (KB)" >&2; exit 64 ;; esac
# Operator message below, kept verbatim: "ARCSEC_CPU_SECONDS=... is not an integer number of seconds"
case "${ARCSEC_CPU_SECONDS:-}" in *[!0-9]*) echo "jail.sh: ARCSEC_CPU_SECONDS=$ARCSEC_CPU_SECONDS is not an integer (seconds)" >&2; exit 64 ;; esac
DRIVER_HOME=${ARCSEC_DRIVER_HOME:-$WORKSPACE/.driver}
mkdir -p "$DRIVER_HOME"

# Driver-specific environment variables. With bwrap's --setenv the later one wins (measured), so they go after the defaults.
DRIVER_ENV=()
if [ -n "${ARCSEC_DRIVER_ENV:-}" ]; then
    # Operator message below, kept verbatim: "ARCSEC_DRIVER_ENV=... does not exist"
    [ -f "$ARCSEC_DRIVER_ENV" ] || { echo "jail.sh: ARCSEC_DRIVER_ENV=$ARCSEC_DRIVER_ENV does not exist" >&2; exit 64; }
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        # Operator message below, kept verbatim: "a driver env line is not KEY=VALUE: $line"
        case "$line" in *=*) ;; *) echo "jail.sh: driver env has a line that is not KEY=VALUE: $line" >&2; exit 64 ;; esac
        DRIVER_ENV+=(--setenv "${line%%=*}" "${line#*=}")
    done < "$ARCSEC_DRIVER_ENV"
fi

# The venv's bin/python is a symlink to an interpreter **outside the venv** (the uv-managed one is).
# And the chain has more than one hop: bin/python -> .../cpython-3.12-.../bin/python3.12,
# and cpython-3.12-... is itself a symlink to cpython-3.12.12-....
# Mounting only the final target isn't enough -- the intermediate name doesn't resolve, python3 in the
# jail falls back to the system python, numpy is gone, and the error only says "No module named numpy",
# pointing nowhere near the real cause. So the install root of every hop on the chain must be mounted.
EXTRA=()
_bind_python_root() {
    local root=$1
    case "$root" in
        "$ARCSEC_RUNTIME"*|/usr|/usr/*) return ;;   # already inside the venv, or the system one
    esac
    for seen in "${EXTRA[@]}"; do [ "$seen" = "$root" ] && return; done
    EXTRA+=(--ro-bind "$root" "$root")
}
_link=$ARCSEC_RUNTIME/bin/python
for _ in 1 2 3 4 5; do
    _bind_python_root "$(dirname "$(dirname "$_link")")"
    [ -L "$_link" ] || break
    _next=$(readlink "$_link")
    case "$_next" in /*) _link=$_next ;; *) _link=$(dirname "$_link")/$_next ;; esac
done

if [ -n "${ARCSEC_EXTRA_RO:-}" ]; then
    IFS=: read -ra paths <<< "$ARCSEC_EXTRA_RO"
    for p in "${paths[@]}"; do [ -n "$p" ] && EXTRA+=(--ro-bind "$p" "$p"); done
fi

# A machine-wide sync(2) waits on every superblock, including NFS the jail can't see: on 10-24 the v4
# g000006/tn36__s15 agent ran `sync; df -h .`, got stuck on an NFS page on aimo2 that had long been stuck
# (D state, unkillable), the scope couldn't be stopped and the whole game failed. sync inside the jail (both
# mount points, /usr/bin and /bin) is replaced with true; files the agent writes are visible immediately
# anyway, so sync does nothing for it.
EXTRA+=(--ro-bind /usr/bin/true /usr/bin/sync --ro-bind /usr/bin/true /bin/sync)

RECORD=()
if [ -n "${ARCSEC_RECORD:-}" ]; then
    if [ "$ARCSEC_RECORD" != "$WORKSPACE/record" ] || [ -L "$ARCSEC_RECORD" ] || [ ! -d "$ARCSEC_RECORD" ]; then
        # Operator message below, kept verbatim: "ARCSEC_RECORD=... must be the directory $WORKSPACE/record (not a symlink)"
        echo "jail.sh: ARCSEC_RECORD=$ARCSEC_RECORD must be the directory $WORKSPACE/record (not a symlink)" >&2
        exit 64
    fi
    RECORD=(--ro-bind "$ARCSEC_RECORD" "$ARCSEC_RECORD")
fi

# --unshare-net only works together with --unshare-user: otherwise bwrap dies setting up loopback itself
# with `RTM_NEWADDR: No child processes`. These two always appear together.
PROC_ARGS=(--proc /proc)
[ "${ARCSEC_JAIL_PROC:-new}" = "bind" ] && PROC_ARGS=(--ro-bind /proc /proc)

exec bwrap \
    --clearenv \
    --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /sbin /sbin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc /etc \
    "${PROC_ARGS[@]}" \
    --ro-bind "$ARCSEC_RUNTIME" "$ARCSEC_RUNTIME" \
    --ro-bind "$ARCSEC_BRIDGE" /bridge.py \
    --ro-bind "$ARCSEC_DRIVER_BIN" "$ARCSEC_DRIVER_BIN" \
    --ro-bind "$ARCSEC_SOCKETS" "$ARCSEC_SOCKETS" \
    "${EXTRA[@]}" \
    --bind "$DRIVER_HOME" "$DRIVER_HOME" \
    --bind "$WORKSPACE" "$WORKSPACE" \
    "${RECORD[@]}" \
    --dev /dev --tmpfs /tmp --tmpfs /run \
    --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-net \
    --die-with-parent --new-session \
    --chdir "$WORKSPACE" \
    --setenv HOME "$WORKSPACE" \
    --setenv PATH "$ARCSEC_RUNTIME/bin:$ARCSEC_DRIVER_BIN/bin:/usr/bin:/bin" \
    --setenv MPLBACKEND Agg \
    --setenv ARC_FENCE bwrap \
    "${ROLE_ARGS[@]}" \
    --setenv ARCSEC_ROLE "$ROLE" \
    --setenv TERM dumb --setenv LANG C.UTF-8 \
    --setenv ARCSEC_RUNTIME_PY "$ARCSEC_RUNTIME/bin/python" \
    --setenv ARCSEC_RUNTIME "$ARCSEC_RUNTIME" \
    --setenv ARCSEC_MODEL_PORT "$ARCSEC_MODEL_PORT" \
    --setenv ARCSEC_SOCKETS "$ARCSEC_SOCKETS" \
    --setenv ARCSEC_PROCESS_DATA_KB "$ARCSEC_PROCESS_DATA_KB" \
    --setenv ARCSEC_CPU_SECONDS "${ARCSEC_CPU_SECONDS:-}" \
    "${DRIVER_ENV[@]}" \
    /bin/bash -c '
        # Per-process hard cap on the data segment (soft and hard both set, so the agent cannot undo it).
        # Inherited by everything in the jail, from the bridges and the driver to every agent script.
        # RLIMIT_DATA counts heap + private anonymous mmap (Linux >= 4.7), not PROT_NONE address
        # reservations like those of numpy/torch, so it does not kill normal work by mistake.
        ulimit -d "$ARCSEC_PROCESS_DATA_KB" || { echo "ARCSEC: ulimit -d $ARCSEC_PROCESS_DATA_KB failed" >&2; exit 64; }
        echo "ARCSEC: per-process data limit $(ulimit -d) KB" >&2
        if [ -n "$ARCSEC_CPU_SECONDS" ]; then
            ulimit -t "$ARCSEC_CPU_SECONDS" || { echo "ARCSEC: ulimit -t $ARCSEC_CPU_SECONDS failed" >&2; exit 64; }
            echo "ARCSEC: per-process CPU limit $(ulimit -t) s" >&2
        fi
        # Two bridges: environment and model. The agent always sees 127.0.0.1:<port>. A reviewer only gets the model one.
        if [ "$ARCSEC_ROLE" = player ]; then
            "$ARCSEC_RUNTIME_PY" -I /bridge.py --listen-tcp 127.0.0.1:$ARCSEC_ENV_PORT \
                --connect-unix "$ARCSEC_SOCKETS/env.sock" &
        fi
        "$ARCSEC_RUNTIME_PY" -I /bridge.py --listen-tcp 127.0.0.1:$ARCSEC_MODEL_PORT \
            --connect-unix "$ARCSEC_SOCKETS/model.sock" &
        # Fail closed: do not start playing unless the bridges work.
        # Note that "something is listening" is not "it works" -- the bridge TCP listener always comes up,
        # and when upstream is unreachable it just accepts the connection and closes it. So make a request
        # that **must succeed**.
        _up=0
        for _ in $(seq 1 60); do
            # -I: isolated mode, CWD not on sys.path. In a 6h real run sk48 created struct.py in the
            # workspace; the import urllib.request -> base64 -> struct here was shadowed by it and blew up,
            # all 60 health checks failed, exit 71, and the game was retired at 4/8 as a zero-action
            # session. Five games each created a struct.py of their own.
            if "$ARCSEC_RUNTIME_PY" -I - <<'\''PY'\''
import os, sys, urllib.request
if os.environ["ARCSEC_ROLE"] == "player":
    req = urllib.request.Request("http://127.0.0.1:%s/api/games" % os.environ["ARCSEC_ENV_PORT"])
    req.add_header("X-API-Key", os.environ["ARC_API_KEY"])
else:
    req = urllib.request.Request("http://127.0.0.1:%s/health" % os.environ["ARCSEC_MODEL_PORT"])
try:
    with urllib.request.urlopen(req, timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
            then _up=1; break; fi
            sleep 0.5
        done
        if [ "$_up" -ne 1 ]; then
            echo "ARCSEC: $ARCSEC_ROLE health check failed through the bridge (env port ${ARCSEC_ENV_PORT:-none}, model port $ARCSEC_MODEL_PORT)" >&2
            exit 71
        fi
        # Do not add -xdev: every bind is its own mountpoint, so adding it would miss exactly what we want to catch.
        if find / -path "*environment_files*" -print -quit 2>/dev/null | grep -q .; then
            echo "ARCSEC: corpus is visible inside the jail — refusing to play" >&2
            exit 70
        fi
        if "$ARCSEC_RUNTIME_PY" -c "import arcengine" 2>/dev/null; then
            echo "ARCSEC: arcengine is importable inside the jail — refusing" >&2
            exit 70
        fi
        exec "$@"
    ' bash "$@"
