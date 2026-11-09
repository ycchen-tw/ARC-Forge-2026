#!/bin/bash
# arc2 on node1 has no root, so only fence none: codex and the harness share one uid and can read the answers in
# --tasks-dir, run_dir/results, other tasks' workspaces and earlier ledgers. This wrapper around codex hides all of that
# with bwrap: the whole / read-only, $HOME, /nfs and /tmp replaced by empty tmpfs, and only codex, the agent runtime
# (read-only) and this task's workspace (writable, = fence none's cwd) mounted back. Network unchanged (the model proxy is
# on 127.0.0.1). run_arc2.py's --driver-bin points at this prefix (bin/codex is this file).
# Every codex session git-clones the openai/plugins marketplace into $CODEX_HOME/.tmp/plugins (~98 MB; on G4 / Kaggle,
# which are offline, the clone fails and takes no space; this machine is online, so it is really written). The prompt is
# unaffected (plugins_instructions=false); it only costs disk: $WS/.driver/.tmp becomes a tmpfs, codex settings unchanged.
#   ARC2_JAIL_TEST=1 codex <cmd...>   runs <cmd> in the same jail instead of codex (self-test)
set -euo pipefail
REAL_PREFIX=$HOME/work/codex-0.154.0-p7/prefix
RUNTIME=$ARC_RUNS/arc_code/agent-runtime
HOME_REAL=$HOME
if [ "${1:-}" = "--version" ]; then exec "$REAL_PREFIX/bin/codex" --version; fi
WS=$(pwd -P)
case "$WS" in "$HOME_REAL"/work/runs/*/tasks/*) ;; *) echo "arc2-jail: cwd $WS is not a task workspace" >&2; exit 70;; esac
# run_arc2 puts the driver_files codex reads (prompts/instructions.md, prompts/compact_prompt.md, tokenizer.json) in
# <run_dir>/harness; there are no answers there (they are in --tasks-dir and <run_dir>/results), so it is mounted back
# read-only.
HARNESS=$(dirname "$(dirname "$WS")")/harness
[ -d "$HARNESS/prompts" ] || { echo "arc2-jail: no $HARNESS/prompts" >&2; exit 70; }
if [ "${ARC2_JAIL_TEST:-0}" = 1 ]; then cmd=("$@"); else cmd=("$REAL_PREFIX/bin/codex" "$@"); fi
exec /usr/bin/bwrap \
    --ro-bind / / \
    --dev /dev --proc /proc \
    --tmpfs /tmp --tmpfs /nfs --tmpfs "$HOME_REAL" \
    --ro-bind "$REAL_PREFIX" "$REAL_PREFIX" \
    --ro-bind "$RUNTIME" "$RUNTIME" \
    --ro-bind "$HARNESS" "$HARNESS" \
    --bind "$WS" "$WS" \
    --tmpfs "$WS/.driver/.tmp" \
    --chdir "$WS" \
    --unshare-pid --die-with-parent \
    -- "${cmd[@]}"
