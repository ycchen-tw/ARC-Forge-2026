#!/bin/bash
# One POSIX fence per game: for hosts without namespaces (Kaggle's ARC-AGI-3 image).
#
#   fence_posix.sh <workspace> <driver argv...>
#
# Takes **the same resource declaration** (ARCSEC_*) as jail.sh and renders it with a different
# mechanism -- this is S2 of docs/plans/arc_code_gateway.md. On Kaggle `unshare --user` is EPERM, and
# there is no bubblewrap, no CAP_SYS_ADMIN, no systemd (measured by probes, 2026-09-17); the old arm
# (work/arc_agi3/arc_code_qwen/fence_posix.sh) ran 19 submissions on that host, and every rule here is
# carried over from it; the only difference is that the input is a resource declaration.
#
# The two things bwrap gives are obtained another way here:
#
#   * **The corpus and the engine are invisible.** There, the mount namespace simply doesn't mount
#     them; here, each game has a dedicated user, the corpus stays under root's 700 (the notebook
#     closes off /kaggle), and this is verified **as that user** before the game starts, not assumed.
#   * **No way off the machine.** There, --unshare-net; here, nothing is done: a Kaggle competition
#     session has no network to begin with, so this fence is only honest on that host, not on the B200 nodes.
#
# Where it is weaker than bwrap, and must not pretend otherwise:
#
#   * No pid namespace: orphans of `nohup … &` outlive the session. The launcher kills all of this
#     user's processes after every session (one uid per game, so the kill is precise).
#   * No private /tmp: TMPDIR points into the workspace, but /tmp itself is shared.
#   * No network namespace: the agent connects to the middleware on host loopback, and can also
#     reach sglang -- which is its model anyway.
#   * Every user can read everyone else's /proc/<pid>/cmdline: so the competition key never goes in
#     argv (the middleware uses --upstream-key-file).
#
# Inputs (all set by the launcher; any missing -> exit 64):
#   ARCSEC_GAME_USER       this game's dedicated user (name chosen by the launcher, ≤ 32 chars)
#   ARCSEC_RUNTIME         the agent's python runtime (**without** arcengine)
#   ARCSEC_DRIVER_BIN      the driver's install prefix
#   ARCSEC_DRIVER_HOME     this game's driver home (inside the workspace)
#   ARCSEC_DRIVER_ENV      KEY=VALUE file (host side): variables the driver adapter wants setenv'd
#   ARCSEC_TOKEN           this game's API key (valid for this game only)
#   ARCSEC_ENV_PORT        the middleware's port on host loopback
#   ARCSEC_PROCESS_DATA_KB per-process RLIMIT_DATA (KB, hard cap)
#   ARCSEC_FENCE_DENY      colon-separated: paths this user **must not** be able to read
#   ARCSEC_PYSHADOW        directory holding the empty sitecustomize (owned by root, readable by the user)
# Optional:
#   ARCSEC_RECORD                the game record written by the middleware (root); must be $WORKSPACE/record.
#                                Left to root, 755: this user can read it but not change it; the workspace
#                                is 700, so other games can't get in.
#   ARCSEC_AGENT_LIB             act_library's arcgame directory (owned by root, 755, PYTHONPATH points
#                                there): this user can read arcgame.py but change neither it nor the directory.
#   ARCSEC_FENCE_KNOWN_READABLE  colon-separated: paths measured to be readable and deliberately accepted (printed)
#   ARCSEC_FENCE_SCAN=1          find / for the corpus as the user (done only in a game's first session)
#   ARCSEC_WORKSPACE_TOOLS=1     config workspace_tools: in the first session, confirm as the user that
#                                lib/arc.py imports and ./bg runs; prints RSI_CHECK workspace_tools ok
#   ARCSEC_OOM_SCORE_ADJ         set before dropping privileges, so the OOM killer picks the agent first, not sglang
#   ARCSEC_FENCE_REQUIRE         colon-separated: files this user **must** be able to read (the driver's instruction
#                                files etc.). If unreadable, codex refuses to start and the game has zero actions --
#                                be loud about it here first.
#   ARCSEC_ROLE                  player (default) or reviewer: empty-slot read-only review (work/arc_code/reviewer.py,
#                                same as jail.sh's reviewer). The workspace is a snapshot with no game:
#                                ARCSEC_TOKEN / ARCSEC_ENV_PORT / ARCSEC_ACT / ARCSEC_OBSERVATION_TOOLS are refused if
#                                given, and no ARC_* is set. The snapshot's record/ shares inodes with the main game
#                                (hardlinks), so ARCSEC_RECORD is required, and chowning the workspace **skips** it
#                                (otherwise the player's record files would briefly belong to the reviewer). The health
#                                check becomes a GET of ARCSEC_MODEL_HEALTH_URL (the model's /health) as the reviewer.
#                                One uid per review (decided by the launcher).
#
# Exit codes follow jail.sh: 64 = bad resource declaration; 70 = corpus / engine visible; 71 = environment unreachable.
set -euo pipefail

WORKSPACE=$1; shift
ROLE=${ARCSEC_ROLE:-player}
case "$ROLE" in
    player) ROLE_VARS=(ARCSEC_TOKEN ARCSEC_ENV_PORT) ;;
    reviewer)
        ROLE_VARS=(ARCSEC_RECORD ARCSEC_MODEL_HEALTH_URL)
        for v in ARCSEC_TOKEN ARCSEC_ENV_PORT ARCSEC_ACT ARCSEC_OBSERVATION_TOOLS; do
            # Operator message below, kept verbatim: "reviewer must not be given $v"
            [ -z "${!v:-}" ] || { echo "fence_posix.sh: reviewer must not receive $v" >&2; exit 64; }
        done ;;
    # Operator message below, kept verbatim: "ARCSEC_ROLE=$ROLE is not player / reviewer"
    *) echo "fence_posix.sh: ARCSEC_ROLE=$ROLE is not player/reviewer" >&2; exit 64 ;;
esac
for v in ARCSEC_GAME_USER ARCSEC_RUNTIME ARCSEC_DRIVER_BIN ARCSEC_DRIVER_HOME ARCSEC_DRIVER_ENV \
         ARCSEC_PROCESS_DATA_KB ARCSEC_FENCE_DENY ARCSEC_PYSHADOW "${ROLE_VARS[@]}"; do
    [ -n "${!v:-}" ] || { echo "fence_posix.sh needs $v" >&2; exit 64; }
done
# Operator message below, kept verbatim: "ARCSEC_PROCESS_DATA_KB is not an integer KB"
case "$ARCSEC_PROCESS_DATA_KB" in *[!0-9]*) echo "fence_posix.sh: ARCSEC_PROCESS_DATA_KB is not an integer (KB)" >&2; exit 64 ;; esac
[ "$(id -u)" -eq 0 ] || { echo "fence_posix.sh has to start as root to drop privileges" >&2; exit 64; }
USERNAME=$ARCSEC_GAME_USER
[ "${#USERNAME}" -le 32 ] || { echo "fence_posix.sh: user name $USERNAME longer than 32" >&2; exit 64; }
id -u "$USERNAME" >/dev/null 2>&1 || useradd -M -N -s /bin/bash "$USERNAME"

# The driver's environment variables: one KEY=VALUE per line, applied after the defaults (same rule as jail.sh).
DRIVER_ENV=()
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    # Operator message below, kept verbatim: "a driver env line is not KEY=VALUE: $line"
    case "$line" in *=*) ;; *) echo "fence_posix.sh: driver env has a line that is not KEY=VALUE: $line" >&2; exit 64 ;; esac
    DRIVER_ENV+=("$line")
done < "$ARCSEC_DRIVER_ENV"

if [ -n "${ARCSEC_RECORD:-}" ]; then
    # If the agent's previous session replaced it with a symlink or anything else, stop here: the middleware
    # only writes to the directory it opened at game start, so the record is already broken. For the reviewer
    # this must be checked before the chown (which skips it below).
    if [ "$ARCSEC_RECORD" != "$WORKSPACE/record" ] || [ -L "$ARCSEC_RECORD" ] || [ ! -d "$ARCSEC_RECORD" ]; then
        echo "ARCSEC: $ARCSEC_RECORD is not the record directory $WORKSPACE/record" >&2
        exit 64
    fi
fi
TMP="$WORKSPACE/.tmp"
mkdir -p "$TMP" "$ARCSEC_DRIVER_HOME"
if [ "$ROLE" = reviewer ]; then
    # The snapshot's record/ is a hardlink of the player's record: chowning it chowns the player's files.
    # Skip it; it is handed straight to root below.
    # -h: a symlink left by the reviewer's previous session only has the link itself changed, not followed to root's file
    find "$WORKSPACE" -path "$ARCSEC_RECORD" -prune -o -exec chown -h "$USERNAME" {} +
else
    chown -R "$USERNAME" "$WORKSPACE"
fi
# act and act.py belong to the harness: kept root-owned and read-only. The agent can delete them (the
# directory is its own) but can't change their content -- same meaning as bwrap mounting /act.py read-only.
for f in act act.py; do
    [ -e "$WORKSPACE/$f" ] && chown root:root "$WORKSPACE/$f" && chmod a+r,go-w "$WORKSPACE/$f"
done
if [ "${ARCSEC_OBSERVATION_TOOLS:-0}" == 1 ]; then
    if [ ! -f "$WORKSPACE/observation.py" ] || [ -L "$WORKSPACE/observation.py" ]; then
        echo "ARCSEC: observation.py must be the provided regular helper file" >&2
        exit 64
    fi
    chown root:root "$WORKSPACE/observation.py"
    chmod a+r,go-w "$WORKSPACE/observation.py"
fi
if [ -n "${ARCSEC_RECORD:-}" ]; then
    # player: the chown above handed it to the agent too; take it back.
    chown -R root:root "$ARCSEC_RECORD"
    chmod -R go-w,a+rX "$ARCSEC_RECORD"
fi
# 700, with the parent traverse-only: other games can't list, let alone read, this game's notes and scripts.
chmod 700 "$WORKSPACE"
chmod 711 "$(dirname "$WORKSPACE")"

# Fail closed, and ask **as the user who will do the reading**. Asking as root always passes.
IFS=':' read -r -a deny <<< "$ARCSEC_FENCE_DENY"
for path in "${deny[@]}"; do
    [ -n "$path" ] || continue
    if runuser -u "$USERNAME" -- test -r "$path" 2>/dev/null; then
        echo "ARCSEC: $USERNAME can read $path — refusing to play" >&2
        exit 70
    fi
done
if [ -n "${ARCSEC_FENCE_REQUIRE:-}" ]; then
    IFS=':' read -r -a need <<< "$ARCSEC_FENCE_REQUIRE"
    for path in "${need[@]}"; do
        [ -n "$path" ] || continue
        runuser -u "$USERNAME" -- test -r "$path" || {
            echo "ARCSEC: $USERNAME cannot read $path (the driver needs it)" >&2; exit 64; }
    done
fi
if [ -n "${ARCSEC_RECORD:-}" ]; then
    runuser -u "$USERNAME" -- test -r "$ARCSEC_RECORD" -a -x "$ARCSEC_RECORD" || {
        echo "ARCSEC: $USERNAME cannot read $ARCSEC_RECORD" >&2; exit 64; }
    if runuser -u "$USERNAME" -- test -w "$ARCSEC_RECORD"; then
        echo "ARCSEC: $USERNAME can write $ARCSEC_RECORD — refusing to play" >&2; exit 70
    fi
fi
if [ -n "${ARCSEC_AGENT_LIB:-}" ]; then
    runuser -u "$USERNAME" -- test -r "$ARCSEC_AGENT_LIB/arcgame.py" -a -x "$ARCSEC_AGENT_LIB" || {
        echo "ARCSEC: $USERNAME cannot read $ARCSEC_AGENT_LIB/arcgame.py" >&2; exit 64; }
    if runuser -u "$USERNAME" -- test -w "$ARCSEC_AGENT_LIB" -o -w "$ARCSEC_AGENT_LIB/arcgame.py"; then
        echo "ARCSEC: $USERNAME can write $ARCSEC_AGENT_LIB — refusing to play" >&2; exit 70
    fi
fi
if [ "${ARCSEC_WORKSPACE_TOOLS:-}" = 1 ] && [ "${ARCSEC_FENCE_SCAN:-}" = 1 ]; then
    # workspace_tools (RSI first batch, item C): the chown above already handed lib/ and bg to this game's
    # user; confirm as that user that they work. Checked only in the first session: at that point they are the
    # harness's originals. Afterwards they belong to the player, and the brief tells it to fix them in place if
    # wrong; breaking or deleting them is its own business and must not make the whole game refuse to start.
    # Go by the flag, not the files: on an arm without item C, the agent may write lib/arc.py itself.
    runuser -u "$USERNAME" -- test -r "$WORKSPACE/lib/arc.py" -a -r "$WORKSPACE/lib/bg.py" -a -x "$WORKSPACE/bg" || {
        echo "ARCSEC: $USERNAME cannot use lib/arc.py, lib/bg.py or ./bg" >&2; exit 64; }
    (cd "$WORKSPACE" && runuser -u "$USERNAME" -- env -i PATH=/usr/bin:/bin HOME="$WORKSPACE" \
        "$ARCSEC_RUNTIME/bin/python" -I -c "import sys; sys.path.insert(0, 'lib'); import arc" ) || {
        echo "ARCSEC: $USERNAME cannot import lib/arc.py with the agent runtime" >&2; exit 64; }
    (cd "$WORKSPACE" && runuser -u "$USERNAME" -- env -i PATH=/usr/bin:/bin HOME="$WORKSPACE" \
        "$ARCSEC_RUNTIME/bin/python" -I lib/bg.py status > /dev/null) || {
        echo "ARCSEC: $USERNAME cannot run lib/bg.py" >&2; exit 64; }
    echo "RSI_CHECK workspace_tools ok: lib/arc.py imports, ./bg runs as $USERNAME" >&2
fi
runuser -u "$USERNAME" -- test -x "$ARCSEC_DRIVER_BIN/bin/codex" -o -x "$ARCSEC_DRIVER_BIN/bin/pi" || {
    echo "ARCSEC: $USERNAME cannot execute anything under $ARCSEC_DRIVER_BIN/bin" >&2; exit 64; }
if runuser -u "$USERNAME" -- env -i PATH=/usr/bin:/bin "$ARCSEC_RUNTIME/bin/python" -I -c "import arcengine" 2>/dev/null; then
    echo "ARCSEC: arcengine is importable by $USERNAME — refusing" >&2
    exit 70
fi
if [ "${ARCSEC_FENCE_SCAN:-}" = 1 ]; then
    # Ask once more the other way round: is the corpus somewhere nobody listed? /kaggle/input is deliberately
    # **not** skipped -- the competition data is mounted there world-readable, and skipping it would skip the
    # copy most likely to leak.
    prune_args=(-path /proc -o -path /sys -o -path /dev)
    if [ -n "${ARCSEC_FENCE_KNOWN_READABLE:-}" ]; then
        IFS=':' read -r -a known <<< "$ARCSEC_FENCE_KNOWN_READABLE"
        for path in "${known[@]}"; do [ -n "$path" ] && prune_args+=(-o -path "$path"); done
        echo "ARCSEC_FENCE_KNOWN_READABLE=$ARCSEC_FENCE_KNOWN_READABLE" >&2
    fi
    visible=$(runuser -u "$USERNAME" -- timeout 300 find / \( "${prune_args[@]}" \) -prune -o \
        \( -path "*environment_files*" -name "*.py" -o -path "*/data/runs/*" \) -print 2>/dev/null \
        | head -1 || true)
    if [ -n "$visible" ]; then
        echo "ARCSEC: game source is readable as $USERNAME ($visible) — refusing to play" >&2
        exit 70
    fi
fi

# The Kaggle image's sitecustomize imports wrapt, which the agent's interpreter doesn't have. python still
# runs, but every command's output gets an extra warning line; on 2026-09-17 the agent tried to "fix" it with
# pip install and curl get-pip.py, voiding 5 of 11 passes. PYTHONPATH comes before stdlib and site-packages,
# so an empty sitecustomize shadows it -- root-owned, outside the workspace, impossible to revert.
[ -f "$ARCSEC_PYSHADOW/sitecustomize.py" ] || { echo "fence_posix.sh: no $ARCSEC_PYSHADOW/sitecustomize.py" >&2; exit 64; }

# Memory: the Kaggle container is a single 175 GiB cgroup with no swap and a read-only /sys/fs/cgroup;
# no per-game cgroup is available. RLIMIT_DATA is set soft and hard together (the session can't undo it), so
# runaway Python gets MemoryError in place; oom_score_adj is raised so that when both miss, the kernel picks the
# agent, not sglang. The whole game's total is summed per uid by the launcher's watchdog.
ulimit -d "$ARCSEC_PROCESS_DATA_KB"
[ "$(ulimit -Hd)" = "$ARCSEC_PROCESS_DATA_KB" ] || { echo "fence_posix.sh: RLIMIT_DATA did not take" >&2; exit 64; }
if [ -n "${ARCSEC_OOM_SCORE_ADJ:-}" ]; then
    echo "$ARCSEC_OOM_SCORE_ADJ" > /proc/self/oom_score_adj
fi

# ARC_FENCE / ARC_GAME_USER: before act's note_gate stops this game's other processes, it relies on these
# (plus the actual uid) to confirm it is inside the fence.
BASE_ENV=(
    HOME="$WORKSPACE" TMPDIR="$TMP" LANG=C.UTF-8 TERM=dumb MPLBACKEND=Agg
    PATH="$ARCSEC_RUNTIME/bin:$ARCSEC_DRIVER_BIN/bin:/usr/bin:/bin"
    PYTHONPATH="$ARCSEC_PYSHADOW"
    ARC_FENCE=posix ARC_GAME_USER="$USERNAME"
)
if [ "$ROLE" = player ]; then
    BASE_ENV+=(ARC_BASE_URL="http://127.0.0.1:$ARCSEC_ENV_PORT" ARC_API_KEY="$ARCSEC_TOKEN")
    CHECK_ENV=()
else
    CHECK_ENV=(ARCSEC_MODEL_HEALTH_URL="$ARCSEC_MODEL_HEALTH_URL")
fi

# Fail closed: as this user, make a request that **must succeed**. player: ask the environment with this
# game's key; reviewer: ask the model's /health.
if ! runuser -u "$USERNAME" -- env -i "${BASE_ENV[@]}" "${CHECK_ENV[@]}" "$ARCSEC_RUNTIME/bin/python" -I - <<'PY'
import os, sys, urllib.request
if "ARCSEC_MODEL_HEALTH_URL" in os.environ:
    req = urllib.request.Request(os.environ["ARCSEC_MODEL_HEALTH_URL"])
else:
    req = urllib.request.Request(os.environ["ARC_BASE_URL"] + "/api/games")
    req.add_header("X-API-Key", os.environ["ARC_API_KEY"])
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
then
    echo "ARCSEC: $ROLE health check failed as $USERNAME (env port ${ARCSEC_ENV_PORT:-none}, model ${ARCSEC_MODEL_HEALTH_URL:-none})" >&2
    exit 71
fi

echo "ARCSEC_FENCE=posix role=$ROLE user=$USERNAME uid=$(id -u "$USERNAME") data_kib=$(ulimit -Hd) oom_score_adj=$(cat /proc/self/oom_score_adj)" >&2
cd "$WORKSPACE"
exec runuser -u "$USERNAME" -- env -i "${BASE_ENV[@]}" "${DRIVER_ENV[@]}" "$@"
