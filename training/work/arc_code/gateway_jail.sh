#!/bin/bash
# Locks the official gateway into bwrap: game .py files are exec'd inside the gateway process (arc_agi's
# local_wrapper), and community games are untrusted code.
# No network (--unshare-all), no $HOME / /nfs / run directory (the key file lives there), environment cleared, only
# recordings and out are writable.
# The only way out is one unix socket: a bridge inside the jail connects it to the gateway's 127.0.0.1:<port>; on the
# host side run_arm starts another bridge connecting 127.0.0.1:<official_port> to this socket, so the address
# middleware sees is unchanged.
#   gateway_jail.sh <python> <src> <corpus/environment_files> <port> <sock dir> <recordings dir> <out dir> [--no-competition-mode]
set -euo pipefail
[ $# -ge 7 ] || { echo "usage: gateway_jail.sh PY SRC CORPUS PORT SOCKDIR RECORDINGS OUT [--no-competition-mode]" >&2; exit 64; }
PY=$1 SRC=$2 CORPUS=$3 PORT=$4 SOCKDIR=$5 REC=$6 OUT=$7
shift 7
EXTRA=("$@")
for d in "$SRC" "$CORPUS" "$SOCKDIR" "$REC" "$OUT"; do
    [ -d "$d" ] || { echo "gateway_jail: $d is not a directory" >&2; exit 64; }
done
[ "$(basename "$CORPUS")" = environment_files ] || { echo "gateway_jail: corpus must be named environment_files" >&2; exit 64; }
VENV=$(dirname "$(dirname "$PY")")
[ -f "$VENV/pyvenv.cfg" ] || { echo "gateway_jail: $PY is not inside a venv" >&2; exit 64; }
BASE=$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg")
case "$BASE" in /usr/*) ;; *) echo "gateway_jail: venv base $BASE is outside /usr (bind it explicitly first)" >&2; exit 64;; esac

exec prlimit --as=$((64 * 1024 * 1024 * 1024)) --nofile=4096 --core=0 --fsize=$((8 * 1024 * 1024 * 1024)) -- \
  bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv \
    --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /etc /etc \
    --proc /proc --dev /dev --tmpfs /tmp --tmpfs /home --tmpfs /data --tmpfs /run \
    --ro-bind "$VENV" "$VENV" --ro-bind "$SRC" "$SRC" --ro-bind "$CORPUS" "$CORPUS" \
    --bind "$SOCKDIR" /sock --bind "$REC" "$REC" --bind "$OUT" "$OUT" \
    --setenv HOME /tmp --setenv PYTHONPATH "$SRC" --setenv PYTHONDONTWRITEBYTECODE 1 --setenv LANG C.UTF-8 \
    --chdir /tmp \
    /bin/bash -c '"$0" -I "$1/arcforge/arc_code/bridge.py" --listen-unix /sock/official.sock --connect-tcp "127.0.0.1:$2" &
exec "$0" -m arcforge.arc_code.official --corpus "$3" --port "$2" --recordings-dir "$4" --scorecard-out "$5/scorecard_official.json" "${@:6}"' \
    "$PY" "$SRC" "$PORT" "$CORPUS" "$REC" "$OUT" "${EXTRA[@]}"
