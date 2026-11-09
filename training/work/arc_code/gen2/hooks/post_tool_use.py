"""PostToolUse: after every successful tool call, decide from the runner-written window-change point (``limit`` in
``$CODEX_HOME/gen2_budget.json``) whether this window is full and whether to change it.

Flow: at ``limit``, send "window is full" once (``full``) and create the workspace's ``.context_full`` (act refuses
``do`` from then on; the next window's SessionStart removes it; if the agent deletes it, the next tool call recreates
it) -> past ``limit + buffer``, write ``$CODEX_HOME/compact_request`` (patch 0004: a new window starts when this step
ends, no summary written under token_budget; the ``limit`` in the file lets codex re-check with its own token count).
Since 10-21 there is no reminder before the window-change point any more (see GEN2_FORCE_BUFFER in run_arm).

The message is **appended after this tool output** (returning ``continue: false`` plus ``reason``; codex replaces the
output the model sees with reason). additionalContext is not used: it is a developer message, which sglang's Responses
merges into the leading system message, so the model doesn't see it and the whole prefix cache is invalidated (see
GEN2_BACKSTOP_TOKENS in run_arm). Only text output can be appended to (replacing an image loses it), and overly long
output gets spilled to a file by codex; in those two cases the message waits for the next tool call. act's refusal
message says the same thing, so the move itself also shows it.

Context size is ``last_token_usage.total_tokens`` of the rollout's last ``token_count``: the previous request's input
plus output (this request's token_count is only written after its tool finishes), so it lags one step. Within a window
it only grows; after a window change codex first records a baseline of a few thousand tokens, so as soon as it gets
smaller it is a new window and the sent record resets; by then ``.context_full`` should have been removed by
SessionStart. The exception is the window change's own PostToolUse: codex sometimes writes the new window's baseline
before running it, while SessionStart hasn't started yet (10-21 g2seed s003 sk48: hook_error 0.1 s before SessionStart).
So the first time "smaller but the flag is still there" is seen, it is only recorded and nothing is touched (state stays
on the old window, re-judged at the next tool call); if it is still there next time, that is a bug of SessionStart not
removing it, and it raises.

Long-think reminder (when the config has ``idle``, the prompt set's idle.md, from g2d): output tokens in this window
since its start or the last move (sum of each request's ``output_tokens``, ``gen2_idle.json``) trigger one appended
reminder every ``idle_tokens``; reset by a move, a window change, or waiting for note_done; not sent after the window is
full (act already refuses ``do``). 10-21 g2b: long thinks of 25k+ without a move within a window came to about 6.5 per
game, with the part above the threshold making up 17-18% of output; g2c's stuck review said "test first" only once at
the window start, and the three stuck review windows made 0, 0 and 3 moves.

Parallel tool calls (e.g. looking at two images in one step) each run a copy of this hook, starting simultaneously:
reading state, deciding and writing back must be locked (``gen2_window.lock``), one copy finishing before the other
reads. Without the lock the two race to rename the same temp file and the later one blows up on a missing file
(10-21 g2fix-r2, four times), and both would also think nothing had been sent yet, sending the same message twice.
"""

import fcntl
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TAIL_BYTES = 1 << 20      # the last token_count is always within this much of the file end (one request's rollout is far smaller)
#: codex spills replaced output beyond about 2,500 tokens (bytes/4) to a file: leave a margin
REASON_MAX_BYTES = 9_000


def requests(transcript: Path) -> list[dict]:
    """Non-empty token_counts in the rollout's tail, oldest to newest: ``total`` (context size), ``output`` (that
    request's output), ``id`` (counted once however many parallel hooks read the same row); empty if there is no request yet."""
    with open(transcript, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - TAIL_BYTES)
        f.seek(start)
        data = f.read()
    lines = data.split(b"\n")
    if start:
        lines = lines[1:]           # the line read from its middle
    if not data.endswith(b"\n"):
        lines = lines[:-1]          # the line codex is still writing
    found = []
    for line in lines:
        if b'"token_count"' not in line:
            continue
        row = json.loads(line)
        payload = row.get("payload") or {}
        if payload.get("type") == "token_count" and payload.get("info"):
            usage = payload["info"]["last_token_usage"]
            found.append({"total": int(usage["total_tokens"]), "output": int(usage["output_tokens"]),
                          "id": f'{row["timestamp"]}/{usage["total_tokens"]}'})
    return found


def appended(output, text: str) -> str | None:
    """The string to show the model: this output with text appended; None if it can't be appended (not text, too long)."""
    if not isinstance(output, str):
        return None
    reason = (output.rstrip("\n") + "\n\n" if output.strip() else "") + text
    return reason if len(reason.encode()) <= REASON_MAX_BYTES else None


def idle_update(idle: dict, seen: list[dict], new_window: bool, st: dict) -> dict:
    """Long-think count: output tokens in this window since its start or the last move (lagging one step, like context
    size): every row after the one counted last time is added (requests that called no tool don't trigger the hook and
    are counted next time; if that row is no longer in the tail, only the last row counts). Reset by a window change, a
    move, or waiting for note_done (act then refuses ``do``; what gets written is level docs)."""
    ids = [r["id"] for r in seen]
    fresh = seen[ids.index(idle["id"]) + 1:] if idle["id"] in ids else seen[-1:]
    if new_window:                          # count from the row where context dropped back: the previous window's tail doesn't count
        drop = max(i for i in range(len(fresh)) if i == 0 or fresh[i]["total"] < fresh[i - 1]["total"])
        idle, fresh = {**idle, "since": 0, "sent": 0}, fresh[drop:]
    idle = {**idle, "since": idle["since"] + sum(r["output"] for r in fresh), "id": seen[-1]["id"]}
    if st["actions"] != idle["step"] or st["waiting"] is not None:
        idle = {**idle, "since": 0, "sent": 0, "step": st["actions"]}
    return idle


inp = common.read_input()
home = common.codex_home()
budget = common.budget()
seen = requests(Path(inp["transcript_path"]))
if seen:
    tokens = seen[-1]["total"]
    ws = Path(inp["cwd"])
    with open(home / "gen2_window.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = home / "gen2_window.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"tokens": -1, "sent": []}
        gate = ws / common.CONTEXT_FULL
        if tokens < state["tokens"] and gate.exists() and not state.get("gate_after_drop"):
            state["gate_after_drop"] = tokens
            tmp = state_path.with_name(state_path.name + ".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(state_path)
            common.log("PostToolUse", action="drop_before_session_start", tokens=tokens, previous=state["tokens"],
                       tool=inp["tool_name"])
            sys.exit(0)
        new_window = tokens < state["tokens"]
        if new_window:
            if gate.exists():
                raise RuntimeError(f"a new window began but {gate} is still there (SessionStart removes it)")
            state = {"tokens": -1, "sent": []}
        limit = budget["limit"]
        if state["tokens"] < 0:
            common.log("PostToolUse", action="window", tokens=tokens, limit=limit, active=budget["active"])
        state["tokens"] = tokens
        action, text = None, None
        # Once created it stays until the next window: even if the dynamic window-change point (g2dyn) grows midway it
        # isn't taken back; the model has already been told the window is full
        if tokens >= limit and not gate.exists():
            gate.write_text(json.dumps({"tokens": tokens, "limit": limit}))
            common.log("PostToolUse", action="gate", tokens=tokens, limit=limit, active=budget["active"],
                       again="full" in state["sent"])
        if tokens >= limit + budget["buffer"]:
            (home / "compact_request").write_text(str(limit))
            action = "new_window"
        elif tokens >= limit and "full" not in state["sent"]:
            action, text = "full", budget["full"]
        idle, idle_path = None, home / "gen2_idle.json"
        if budget["idle"] is not None:
            st = common.status(ws, common.history(ws))
            idle = json.loads(idle_path.read_text()) if idle_path.exists() else \
                {"id": None, "since": 0, "sent": 0, "step": st["actions"]}
            idle = idle_update(idle, seen, new_window, st)
            due = budget["idle_tokens"] * (idle["sent"] + 1)
            if action is None and not gate.exists() and st["state"] != "WIN" and idle["since"] >= due:
                action, text = "idle", budget["idle"].format(tokens=f"{idle['since'] // 1000 * 1000:,}")
        reason = appended(inp["tool_response"], text) if text is not None else None
        if text is not None and reason is None:
            action = None                   # can't append to this output: send at the next tool call
        if reason is not None:
            if action == "idle":
                idle["sent"] += 1
            else:
                state["sent"].append(action)
        for path, body in ((state_path, state), (idle_path, idle)):
            if body is not None:
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(body))
                tmp.replace(path)
        if action is not None:
            common.log("PostToolUse", action=action, tokens=tokens, limit=limit, active=budget["active"],
                       tool=inp["tool_name"], **({"since": idle["since"]} if action == "idle" else {}))
    if reason is not None:
        print(json.dumps({"continue": False, "stopReason": "context window " + action, "reason": reason}))
