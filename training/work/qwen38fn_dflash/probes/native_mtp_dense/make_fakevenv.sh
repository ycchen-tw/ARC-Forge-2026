#!/bin/bash
# make_fakevenv.sh: $MTPD_RUN/fakevenv = $ARC_SGLANG_VENV's bin/ (symlinks; serve_qwen38fn_0520.sh puts it on PATH) with
# python replaced by a shim that runs patch/mtpd_launch.py instead of work/qwen38fn_eval/int3_deploy_runtime.py launch.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export ARC_ROOT=$ROOT
source "$ROOT/env/env.sh"
: "${MTPD_RUN:?MTPD_RUN=<run dir>}" "${ARC_SGLANG_VENV:?}"
V=$MTPD_RUN/fakevenv
[ ! -e "$V" ] || { echo "✗ $V exists" >&2; exit 1; }
mkdir -p "$V/bin"
for f in "$ARC_SGLANG_VENV"/bin/*; do
    case $(basename "$f") in python|python3|python3.12) ;; *) ln -s "$f" "$V/bin/$(basename "$f")" ;; esac
done
cat > "$V/bin/python" <<SHIM
#!/bin/bash
# Shim for serve_qwen38fn_0520.sh's "\$SGLANG_VENV/bin/python": venv python, except the INT3 launch runs \$MTPD_LAUNCH.
REAL=$ARC_SGLANG_VENV/bin/python
if [ "\${1:-}" = work/qwen38fn_eval/int3_deploy_runtime.py ] && [ "\${2:-}" = launch ]; then
    : "\${MTPD_LAUNCH:?MTPD_LAUNCH unset}"
    shift
    echo "· fakevenv: launching via \$MTPD_LAUNCH" >&2
    exec "\$REAL" "\$MTPD_LAUNCH" "\$@"
fi
exec "\$REAL" "\$@"
SHIM
chmod +x "$V/bin/python"
ln -s python "$V/bin/python3"
ln -s python "$V/bin/python3.12"
echo "fakevenv ready: $V"
