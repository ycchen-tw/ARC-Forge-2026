#!/bin/bash
# Runs one codex arm. ARM picks the config and sampling: c0 = codex sends no sampling (server defaults 0.7 / 1.0 / none);
# the other arms set sampling explicitly through work/arc_code/sampling_proxy.py -- per configs/$ARM.sampling if it
# exists (one KEY=VALUE per line), otherwise 0.7 / 0.95 / 20 (same as Super Duck) -- and log each response's usage to
# $OUT/usage.jsonl; configs/$ARM.proxy, if it exists, holds that arm's extra proxy parameters. When the config has
# act_image (c0i) the proxy also replaces ./act do's markers with images: the signing key is generated here in $OUT
# (the jail does not mount $OUT, so the agent cannot read it), shared by the middleware and the proxy.
#   ARM=c0 SLOT=5 TAG=c0-r1 bash work/arc_code/duckx/launch_codex.sh [run_arm args, e.g. --games ft09 cn04 --wall-seconds 900]
# Other machines / other serving (node1's c0i): OUT_ROOT = output root, SERVER = model server, TOKENIZER = the
# tokenizer.json codex truncates with; unset = duckx's originals (node2's 20261008-duckx, port 30467+SLOT, NVFP4-nvidia).
# GAME_OUTPUT_TOKENS = overrides the config's game_output_tokens (a small cap for smoke tests; written into
# $OUT/config.json).
set -euo pipefail
cd $HOME/work/arc-forge-collect1022   # 20261022: the pinned collect worktree (e6d9c538, same as 10-19)
source env/env.sh
: "${STOPGAP_CFG_DIR:?}" "${CORPUS:?}" "${ARM:?ARM=c0|c1|...}" "${SLOT:?SLOT=4..7}" "${TAG:?TAG=<new output name>}"
# (stopgap 20261015) SLOT is used only to offset ports
case "$ARM" in c0|c0j) NOPROXY=1;; *) NOPROXY=0;; esac   # v17 = c0: codex connects to the server directly (sampling uses the server defaults)
CFG=$STOPGAP_CFG_DIR/$ARM.json
[ -f "$CFG" ] || { echo "no $CFG" >&2; exit 64; }
OUT=${OUT_ROOT:-$ARC_RUNS/arc_code/20261008-duckx}/$TAG
# When starting a second one on the same card (smoke test), PORT_OFFSET shifts the gateway / official / proxy ports
PORT_OFFSET=${PORT_OFFSET:-0}
[ -e "$OUT" ] && { echo "$OUT already exists; use a new TAG every time" >&2; exit 64; }
mkdir -p "$OUT"
SERVER=${SERVER:-http://127.0.0.1:$((30467 + SLOT))}
MODEL_URL=$SERVER
curl -fsS --max-time 5 "$SERVER/health" >/dev/null || { echo "server $SERVER is not up" >&2; exit 65; }
TOKENIZER=${TOKENIZER:-$ARC_MODELS/Qwen3.8-Flash-Next-NVFP4-nvidia/tokenizer.json}
[ -f "$TOKENIZER" ] || { echo "no $TOKENIZER" >&2; exit 64; }

.venv/bin/python - "$CFG" "$OUT/config.json" "$SLOT" "$TOKENIZER" "$PORT_OFFSET" "${GAME_OUTPUT_TOKENS:-}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
slot, offset = int(sys.argv[3]), int(sys.argv[5])
cfg["gateway_port"], cfg["official_port"] = 18907 + slot + offset, 18917 + slot + offset
cfg.setdefault("driver_files", {})["tool_output_tokenizer"] = sys.argv[4]
if sys.argv[6]:
    cfg["game_output_tokens"] = int(sys.argv[6])
json.dump(cfg, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
PY
# act_image: the image side length follows record.png_size (the same image as final.png)
PNG_SIZE=$(.venv/bin/python -c 'import json, sys; c = json.load(open(sys.argv[1]))
print(c["record"].get("png_size", 512) if c.get("act_image") else "")' "$OUT/config.json")
IMAGE_KEY=()
if [ -n "$PNG_SIZE" ]; then
    [ "$NOPROXY" = 0 ] || { echo "act_image needs the proxy, and c0 starts no proxy" >&2; exit 64; }
    (umask 077; .venv/bin/python -c 'import secrets; print(secrets.token_hex(32))' > "$OUT/image_ref.key")
    IMAGE_KEY=(--image-ref-key "$OUT/image_ref.key")
fi

PROXY_PID=
if [ "$NOPROXY" = 0 ]; then
    PROXY_PORT=$((31467 + SLOT + PORT_OFFSET))
    # The arm's own proxy options (e.g. c3's reasoning trim) go in configs/$ARM.proxy, one argument per line;
    # a smoke test can replace them wholesale with the env var PROXY_ARGS (recorded as is in $OUT/proxy_args.txt)
    PROXY_EXTRA=()
    if [ -n "${PROXY_ARGS:-}" ]; then
        read -r -a PROXY_EXTRA <<< "$PROXY_ARGS"
    elif [ -f "work/arc_code/duckx/configs/$ARM.proxy" ]; then
        mapfile -t PROXY_EXTRA < <(grep -v '^\s*\(#\|$\)' "work/arc_code/duckx/configs/$ARM.proxy")
    fi
    printf '%s\n' "${PROXY_EXTRA[@]}" > "$OUT/proxy_args.txt"
    # If the brief says the tail is kept after the summary on compaction (since v14j), the proxy must keep it; the
    # converse is not checked (c4 / c8 keep it but the brief does not say so)
    PROMPTS=$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompts"])' "$CFG")
    if grep -q "are kept right after the" "work/arc_code/prompts/$PROMPTS/brief.md" \
            && ! printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--carry-tail-tokens'; then
        echo "prompts $PROMPTS say the tail is kept after compaction, but the $ARM proxy has no --carry-tail-tokens" >&2; exit 64
    fi
    for arg in "${PROXY_EXTRA[@]}"; do
        case "$arg" in --trim-log|--usage-log|--tail-log|--effort-run-dir|--window-run-dir|--window-log|--level-watch-run-dir)
            echo "$ARM.proxy must not set its own log/run paths" >&2; exit 64;; esac
    done
    TRIM_LOG=()
    printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--trim-reasoning-at' && TRIM_LOG+=(--trim-log "$OUT/reasoning_trim.jsonl")
    printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--carry-tail-tokens' && TRIM_LOG+=(--tail-log "$OUT/compaction_tail.jsonl")
    # effort_schedule reads each game's record/ to identify levels
    printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--effort-schedule' && TRIM_LOG+=(--effort-run-dir "$OUT/run")
    # level_watch also reads each game's record/ to identify levels
    printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--level-watch' && TRIM_LOG+=(--level-watch-run-dir "$OUT/run")
    # context_window reads each game's notes.md; the reminder (--window-notes-after), the cut (--window-at) and the
    # latest cut (--window-hard) must all come before codex's own compaction. codex's pre_compact_reminder cannot be on
    # at the same time: it is a user message, which makes codex re-estimate the earlier reasoning and compact early
    # (context_window.py's docstring)
    if printf '%s\n' "${PROXY_EXTRA[@]}" | grep -q -- '--window-at'; then
        TRIM_LOG+=(--window-run-dir "$OUT/run" --window-log "$OUT/context_window.jsonl")
        .venv/bin/python - "$OUT/config.json" "${PROXY_EXTRA[@]}" <<'PY' || exit 64
import json, sys
cfg, args = json.load(open(sys.argv[1])), sys.argv[2:]
get = lambda k: int(args[args.index(k) + 1]) if k in args else None
high, keep, notes, hard = get("--window-at"), get("--window-keep"), get("--window-notes-after"), get("--window-hard")
limit = cfg["auto_compact_token_limit"]
if cfg.get("compact_reminder_tokens") is not None or (cfg.get("driver_config") or {}).get("features.pre_compact_reminder") == "true":
    sys.exit("context-window arms must not enable the codex pre_compact_reminder (the reminder comes from the proxy --window-notes-after)")
if not (high < limit and (notes is None or keep < notes < high) and (hard is None or high <= hard < limit)):
    sys.exit(f"need keep {keep} < notes_after {notes} < high {high} <= hard {hard} < auto_compact_token_limit {limit}")
PY
    fi
    SAMPLING=(--set temperature=0.7 --set top_p=0.95 --set top_k=20)
    if [ -f "$STOPGAP_CFG_DIR/$ARM.sampling" ]; then
        SAMPLING=()
        while read -r kv; do SAMPLING+=(--set "$kv"); done \
            < <(grep -v '^\s*\(#\|$\)' "$STOPGAP_CFG_DIR/$ARM.sampling")
        [ ${#SAMPLING[@]} -gt 0 ] || { echo "configs/$ARM.sampling has no KEY=VALUE lines" >&2; exit 64; }
    fi
    IMAGE_ARGS=()
    [ -z "$PNG_SIZE" ] || IMAGE_ARGS=("${IMAGE_KEY[@]}" --image-run-dir "$OUT/run" --image-png-size "$PNG_SIZE")
    PYTHONPATH=src .venv/bin/python work/arc_code/sampling_proxy.py --listen "$PROXY_PORT" --upstream "$SERVER" \
        "${SAMPLING[@]}" --manifest "$OUT/sampling.json" \
        --usage-log "$OUT/usage.jsonl" "${TRIM_LOG[@]}" "${IMAGE_ARGS[@]}" "${PROXY_EXTRA[@]}" \
        > "$OUT/sampling_proxy.log" 2>&1 &
    PROXY_PID=$!
    echo "$PROXY_PID" > "$OUT/sampling_proxy.pid"
    trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 50); do curl -fsS --max-time 2 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break; sleep 0.2; done
    curl -fsS --max-time 5 "http://127.0.0.1:$PROXY_PORT/health" >/dev/null || { echo "sampling proxy is not up" >&2; exit 65; }
    MODEL_URL=http://127.0.0.1:$PROXY_PORT
fi

curl -fsS --max-time 10 "$SERVER/get_server_info" > "$OUT/server_info.json"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
PYTHONPATH=src .venv/bin/python work/arc_code/run_arm.py \
    --config "$OUT/config.json" --run-dir "$OUT/run" \
    --corpus "$CORPUS" --prep "$ARC_DATA/arc-code-prep" \
    --python "$ARC_RUNS/arc_code/harness-venv/bin/python" \
    --agent-python "$ARC_RUNS/arc_code/agent-runtime/bin/python" \
    --driver-bin "$HOME/work/codex-0.154.0-p7/prefix" --driver codex \
    --ffmpeg "$HOME/work/dsv4/ffmpeg-n8.1/bin/ffmpeg" \
    --model-base-url "$MODEL_URL/v1" --model qwen38fn --tag "$TAG" "${IMAGE_KEY[@]}" "$@" \
    > "$OUT/launcher.log" 2>&1
echo "ARM_DONE $TAG" >> "$OUT/launcher.log"
