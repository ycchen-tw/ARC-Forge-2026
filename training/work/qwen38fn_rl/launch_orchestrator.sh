#!/bin/bash
# Starts the RL orchestrator in a systemd user unit. Runs --check first (config, files, task lists; no network) and starts only if
# it passes.
#
#   RL_CONFIG=work/qwen38fn_rl/configs/toy-b8k.json bash work/qwen38fn_rl/launch_orchestrator.sh
#
# Prerequisites: the rollout server (launch_rollout_server.sh) and the trainer service (work/qwen38fn_lora/train_rl_service.sh,
#       a separate line) are both up and healthy; the orchestrator checks at the start of every round and stops if they are not.
# Stop: touch <run_root>/STOP (stops after the current round); systemctl --user stop <unit> (SIGTERM: terminates the harness groups
#       it started, keeps state).
# Swap code without losing in-flight groups (stream): touch <run_root>/DETACH first, then stop (or kill -USR1 <MainPID>); the
#       orchestrator exits and the groups keep running (the unit is KillMode=process, so systemd does not kill them), and running
#       this command again adopts them (groups/*.launch.json). While the leftover groups are still in the old unit's cgroup, a unit
#       of the same name cannot start (already loaded); this script then switches to <unit>-2, -3…, prints it, and also writes it to
#       RL_UNIT_FILE (if given).
# Resume: run the same command again (rounds.jsonl is authoritative); an unfinished round left from last time needs a human
#       decision, see orchestrator.py's --abandon-incomplete-round.
# wandb: by default a separate <unit>-wandb is started after the orchestrator (wandb_sync.py --follow: follows the orchestrator's
#       PID and stops after one last read once it ends).
#       RL_WANDB=0 does not start it; RL_WANDB_ONLY=1 starts no orchestrator, only <unit>-wandb (attaching to a run that is
#       running or has ended, via orchestrator.lock).
# Optional  RL_UNIT (default rl-orchestrator), RL_PYTHON (default python3; the orchestrator uses only the standard library),
#       RL_ORCH_ARGS (extra arguments, e.g. --abandon-incomplete-round), RL_DRY_RUN=1 (only print the command),
#       RL_UNIT_FILE (receives the orchestrator unit name actually used this time, RL_UNIT or <RL_UNIT>-N; a dry run writes the one
#       it would use. launch_b200_stream.sh relies on it to find the MainPID and for disk-guard, and cannot assume RL_UNIT),
#       RL_WANDB_PROJECT (default qwen38fn-rl), RL_WANDB_PYTHON (default $HOME/work/venv-fn-train/bin/python, needs wandb),
#       RL_WANDB_ARGS (extra arguments for wandb_sync.py, e.g. --baseline <dir> ...)
# note judge (config has note_judge = prepare.arc3_reward note, RL v8):
#       backend api: the bash in the unit sources note_judge.env_file (set -a; the ANTHROPIC_API_KEY_2 the judge needs) before
#       exec'ing the orchestrator. systemd records only the file path (RL_JUDGE_ENV_FILE); the value never goes through a command
#       line and is never printed; the orchestrator checks the key at start-up.
#       backend codex: no env file is sourced and the unit's PATH / HOME stay as they are: the judge uses note_judge.codex_bin from
#       the config (the absolute path pinned with readlink -f at configure time), CODEX_HOME unset (the default ~/.codex ChatGPT
#       login), --ignore-user-config; at start-up the orchestrator checks that ~/.codex/auth.json exists and codex_bin --version =
#       codex_version.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RL_CONFIG:?}"
unit=${RL_UNIT:-rl-orchestrator}
py=${RL_PYTHON:-python3}
# a unit whose orchestrator exited but whose cgroup still holds detached harness groups (KillMode=process) stays
# loaded: systemd-run refuses the same name ("already loaded"). Take the next free <unit>-N instead.
if [ "${RL_WANDB_ONLY:-0}" = 0 ]; then
    base=$unit; n=1
    while [ "$(systemctl --user show -p LoadState --value "$unit")" = loaded ] \
          && ! systemctl --user is-active --quiet "$unit"; do
        n=$((n + 1)); unit=$base-$n
    done
    [ "$unit" = "$base" ] || echo "· unit $base is still loaded with harness groups left from the previous run (KillMode=process): using $unit this time"
    if [ -n "${RL_UNIT_FILE:-}" ]; then
        printf '%s\n' "$unit" > "$RL_UNIT_FILE.tmp" && mv "$RL_UNIT_FILE.tmp" "$RL_UNIT_FILE"
    fi
fi
wandb=${RL_WANDB:-1}
only=${RL_WANDB_ONLY:-0}
case "$wandb:$only" in
    1:0|1:1|0:0) ;;
    *) echo "✗ RL_WANDB=$wandb RL_WANDB_ONLY=$only: each must be 0/1, and RL_WANDB_ONLY=1 requires RL_WANDB=1" >&2; exit 1 ;;
esac
wunit=$unit-wandb
wpy=${RL_WANDB_PYTHON:-$HOME/work/venv-fn-train/bin/python}
dry=${RL_DRY_RUN:-0}
# shellcheck disable=SC1091
source env/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
check=$("$py" work/qwen38fn_rl/orchestrator.py --config "$RL_CONFIG" --check)
echo "$check"
run_root=${check##* run_root }
# the config passed --check above: note_judge (when present) has a backend, and the api backend an env_file that exists
judge_backend=$("$py" -c 'import json, sys; cfg = json.load(open(sys.argv[1])); print(cfg["note_judge"]["backend"] if "note_judge" in cfg else "none")' \
                "$RL_CONFIG")
judge_setenv=()
case "$judge_backend" in
    none) ;;
    api)
        judge_env=$("$py" -c 'import json, sys; print(json.load(open(sys.argv[1]))["note_judge"]["env_file"])' "$RL_CONFIG")
        [ -f "$judge_env" ] || { echo "✗ note_judge.env_file $judge_env does not exist" >&2; exit 1; }
        judge_setenv=(--setenv=RL_JUDGE_ENV_FILE="$judge_env") ;;
    codex) ;;          # the pinned binary is in the config (note_judge.codex_bin); the login is ~/.codex
    ensemble) ;;       # v8b: claude_bin / codex_bin pinned in the config; claude's login is ~/.claude (subscription)
    claude-code) ;;    # v8b, Claude Code alone: claude_bin pinned in the config; the login is ~/.claude (subscription)
    *) echo "✗ note_judge.backend $judge_backend" >&2; exit 1 ;;
esac
if [ "$wandb" = 1 ]; then
    [ -x "$wpy" ] || { echo "✗ RL_WANDB_PYTHON $wpy is not executable (to run without wandb: RL_WANDB=0)" >&2; exit 1; }
    [ -d "$run_root" ] || { echo "✗ run_root $run_root does not exist yet (the rollout server creates it at start-up)" >&2; exit 1; }
    if systemctl --user is-active --quiet "$wunit"; then
        echo "✗ unit $wunit is already running (the sidecar of the previous run wraps up within one poll after its orchestrator exits; to run without wandb: RL_WANDB=0)" >&2
        exit 1
    fi
fi
if [ "$only" = 0 ]; then
    systemctl --user is-active --quiet "$unit" && { echo "✗ unit $unit is already running" >&2; exit 1; }
    # KillMode=process: systemd itself signals only the orchestrator and does not kill the harness groups in the cgroup. On SIGTERM
    # without a DETACH file the orchestrator terminates the groups it started itself (as before); with a DETACH file (or SIGUSR1)
    # they keep running and the next orchestrator adopts them from groups/*.launch.json (SPEC §9).
    cmd=(systemd-run --user --unit="$unit" --collect --working-directory="$PWD" -p TimeoutStopSec=900 -p KillMode=process
         --setenv=PYTHONPATH="$PYTHONPATH"
         --setenv=RL_ORCH_ARGS="${RL_ORCH_ARGS:-}" "${judge_setenv[@]}"
         /bin/bash -c 'source env/env.sh && { [ -z "${RL_JUDGE_ENV_FILE:-}" ] || { set -a && . "$RL_JUDGE_ENV_FILE" && set +a; }; } && exec "$0" work/qwen38fn_rl/orchestrator.py --config "$1" ${RL_ORCH_ARGS:-}'
         "$py" "$RL_CONFIG")
    if [ "$dry" = 1 ]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
        echo "· unit $unit started; log: journalctl --user -u $unit -f, and <run_root>/orchestrator.log"
    fi
fi
[ "$wandb" = 1 ] || exit 0
wargs=(--run-root "$run_root" --config "$RL_CONFIG" --project "${RL_WANDB_PROJECT:-qwen38fn-rl}" --follow)
if [ "$only" = 0 ]; then
    # follow the unit's MainPID (bash execs into python, the PID stays the same): orchestrator.lock may still hold the PID written
    # last time
    if [ "$dry" = 1 ]; then
        pid='<MainPID>'
    else
        pid=$(systemctl --user show -p MainPID --value "$unit")
        case "$pid" in
            ''|0|*[!0-9]*)
                echo "✗ $unit has already exited (MainPID '$pid'): see journalctl --user -u $unit; to log to wandb afterwards use RL_WANDB_ONLY=1" >&2
                exit 1 ;;
        esac
    fi
    wargs+=(--orchestrator-pid "$pid")
fi
# KillMode=mixed: on stop, SIGTERM goes only to the sidecar itself (it finishes logging the current entry and waits for the wandb
# flush before leaving), and the wandb service process is left for it to reap
wcmd=(systemd-run --user --unit="$wunit" --collect --working-directory="$PWD" -p KillMode=mixed -p TimeoutStopSec=300
      --setenv=PYTHONPATH="$PYTHONPATH"
      --setenv=RL_WANDB_ARGS="${RL_WANDB_ARGS:-}"
      /bin/bash -c 'source env/env.sh && exec "$0" work/qwen38fn_rl/wandb_sync.py "$@" ${RL_WANDB_ARGS:-}'
      "$wpy" "${wargs[@]}")
if [ "$dry" = 1 ]; then printf '%q ' "${wcmd[@]}"; echo; exit 0; fi
"${wcmd[@]}"
echo "· unit $wunit started (wandb project ${RL_WANDB_PROJECT:-qwen38fn-rl}); log: journalctl --user -u $wunit -f; cursor $run_root/wandb_sync.json"
