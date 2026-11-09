#!/usr/bin/env python
"""Agentic TIR: explorers (parallel, each ≤ ctx cap) → harness verification → refiner (≤ ctx cap, at most 2 attempts per test).

    python work/agentic_tir/agentic_tir.py --task-dir ... --task-list ... --out $ARC_RUNS/agentic_tir/<tag> \
        --base-url http://127.0.0.1:30031/v1 --model qwen38fn --n-explorers 3 --ctx-cap 50000

Flow for one task
-----------------
1. **explore**: K independent rollouts, same prompt (all test inputs given together), each with its own sandbox.
   Each one's context (prompt+completion, counted by the total_tokens the server returns) is kept within `ctx_cap`:
   per-turn max_tokens = ctx_cap − report_reserve − last_total − margin. When the remainder is too small, the model
   is truncated, or turns run out → append a user message asking for a handoff report (handoff; thinking off and
   tool_choice=none by default). The final reply of a normal finish is itself the report
   (RULE/FINDINGS/CODE/CONFIDENCE/ANSWER, see prompts.py).
2. **verify**: open the sandbox used by refine, define each report's `transform` as `transform_k`, run it on all train
   pairs (pass/fail/error) and all test inputs (predicted grids); together with the ANSWER each explorer states, dedup
   into candidates (A, B, ...). This is the "fact table" the refiner sees.
3. **refine**: one rollout, prompt = task + K reports + check table + candidates; same ctx cap; the final report's
   ANSWER is ATTEMPT 1 (required) / ATTEMPT 2 (optional) per test. Tests refine gives no answer for are filled from
   candidates by vote count; with `--fill-attempt2`, a missing attempt 2 is also filled from candidates (both sources
   recorded in attempt_src so analysis can tell them apart). If refine runs out of budget without answering → its
   report joins the reports and another round opens (≤ max_refine_rounds).

Scoring is the same as work/qwen38_eval/score_partial.py: pairs/*.json can be fed to it directly.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import collections
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402

from arcforge.tir.driver import Sampling, _chat  # noqa: E402
from arcforge.tir.protocol import load_config  # noqa: E402
from arcforge.tir.sandbox import Sandbox, SandboxError  # noqa: E402
from arcforge.tir.submission import SubmissionStore  # noqa: E402

GRID_MARK = "__GRID__"
ERR_MARK = "__ERR__"
DEF_ERR_MARK = "__DEF_ERR__"


# ---------------------------------------------------------------------------
# One episode (a single ≤ ctx_cap TIR conversation)
# ---------------------------------------------------------------------------

class Episode:
    """Record of one conversation. All counts come from the server's usage, never self-estimated."""

    def __init__(self, label: str, lineage: int = 0, hop: int = 1):
        self.label = label
        self.lineage = lineage        # explorer number (0 for the refiner)
        self.hop = hop                # which session of the same explorer
        self.messages: list[dict] = []
        self.calls: list[dict] = []
        self.stop_reason = ""
        self.forced_handoff = False
        self.n_tool_calls = 0
        self.n_tool_errors = 0
        self.n_sandbox_restarts = 0    # times the model's code left the sandbox unresponsive for 300s and it was killed and restarted (all state lost)
        self.n_continuations = 0
        self.n_compactions = 0             # --compact: times the harness folded the oldest tool round and the same conversation continued
        self.compact_dropped_tokens = 0    # cumulative estimated tokens folded away (the part the model can no longer see)
        self.compact_dropped_rounds = 0
        self.compact_dropped_calls = 0
        self.n_checkpoints = 0             # --compact-checkpoint: bookkeeping turns before folding
        self.compact_history: list[str] = []   # --compact-rich: one "what was tried → result" line per folded round
        self.compact_transform_src = ""        # --compact-rich: source of the last `def transform` in the folded rounds
        self.n_outputs_cleared = 0             # --compact-rich: number of tool outputs cleared
        self.output_heads: dict[str, str] = {}  # heads of cleared tool outputs (for _round_summary; not saved)
        self.task_user = ""                # original text of the first user message (compaction rebuilds from it each time, so notices never stack)
        self.task_id = ""                  # for logging only
        self.notes_structured = False      # --notes-structured (for rendering in _report_for_refiner)
        self.sub_episodes: list[dict] = []  # --delegate: helper conversations (to_dict, incl. messages)
        self.n_delegations = 0
        self.segments: list[list[dict]] = []  # --save-segments: the full messages before each compaction (the view the model actually saw); the last segment is the messages at the end
        self.final_content = ""
        self.final_reasoning = ""
        self.report: dict | None = None
        self.notes: list[str] = []          # `notes` in the sandbox when the session ends (the ledger, passed verbatim to the successor)
        self.home_check: dict | None = None  # result of running the report's CODE in its own sandbox at session end (helpers all present)
        self.alt_check: dict | None = None   # alt variant: same, but checks the counter-hypothesis `transform_alt` (not for the refiner)
        self.code_src = ""
        self.answers: list[list] = []          # answers[t] = [grid, ...] (in attempt order)
        self.answer_resolution: list[list[str]] = []
        self.t0 = time.monotonic()
        self.latency_s = 0.0

    def usage(self, key: str) -> int:
        return sum(c[key] for c in self.calls)

    @property
    def peak_total(self) -> int:
        return max((c["total_tokens"] for c in self.calls), default=0)

    def to_dict(self, keep_messages: bool) -> dict:
        d = {
            "label": self.label, "lineage": self.lineage, "hop": self.hop,
            "stop_reason": self.stop_reason, "forced_handoff": self.forced_handoff,
            "n_calls": len(self.calls), "n_tool_calls": self.n_tool_calls, "n_tool_errors": self.n_tool_errors,
            "n_sandbox_restarts": self.n_sandbox_restarts,
            "n_continuations": self.n_continuations,
            "n_compactions": self.n_compactions, "compact_dropped_tokens": self.compact_dropped_tokens,
            "n_checkpoints": self.n_checkpoints, "n_outputs_cleared": self.n_outputs_cleared,
            "compact_history": self.compact_history,
            "n_delegations": self.n_delegations, "sub_episodes": self.sub_episodes,
            "tok_prompt": self.usage("prompt_tokens"), "tok_completion": self.usage("completion_tokens"),
            "tok_reasoning": self.usage("reasoning_tokens"), "tok_cached": self.usage("cached_tokens"),
            "peak_total": self.peak_total, "latency_s": round(self.latency_s, 1),
            "report": self.report, "notes": self.notes, "home_check": self.home_check, "alt_check": self.alt_check,
            "code_src": self.code_src, "answers": self.answers,
            "answer_resolution": self.answer_resolution,
            "final_content": self.final_content, "final_reasoning": self.final_reasoning,
            "calls": self.calls,
        }
        if keep_messages:
            d["messages"] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.messages]
        if self.segments:
            d["segments"] = self.segments   # write_task moves this to segments/<task>.jsonl.gz
        return d


def _copy_private(asst: dict, msg: dict) -> None:
    """Private keys the backend attaches to a response (the token backend's `_ids` / `_gen_len`) must go into messages, so the segments carry tokens when written."""
    for k, v in msg.items():
        if k.startswith("_"):
            asst[k] = v


def _record_usage(ep: Episode, data: dict, finish: str, kind: str, max_tokens: int) -> int:
    u = data["usage"]
    details = u.get("prompt_tokens_details") or {}
    cdetails = u.get("completion_tokens_details") or {}
    rec = {
        "kind": kind, "finish": finish, "max_tokens": max_tokens,
        "prompt_tokens": int(u["prompt_tokens"]), "completion_tokens": int(u["completion_tokens"]),
        "total_tokens": int(u["total_tokens"]),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "reasoning_tokens": int(u.get("reasoning_tokens") or cdetails.get("reasoning_tokens") or 0),
        "segment": ep.n_compactions,   # which segment this call saw (segment_idx of --save-segments)
    }
    ep.calls.append(rec)
    return rec["total_tokens"]


def _base_payload(args, messages, tool, max_tokens: int) -> dict:
    p = {
        "model": args.model, "messages": messages, "tools": [tool, *args.extra_tools],
        "temperature": args.sampling.temperature, "top_p": args.sampling.top_p, "top_k": args.sampling.top_k,
        "max_tokens": max_tokens,
    }
    if args.reasoning_effort:
        p["reasoning_effort"] = args.reasoning_effort
    if args.dp_rank is not None:
        # sglang's DP is round-robin with no session affinity: without pinning a rank, every turn re-prefills the whole
        # context on another replica (the naive driver pins it in run_rollout; this path did not before). All episodes of
        # one task pin the same rank, so the system + task prefix still hits the radix cache across explorer / refiner.
        p["routed_dp_rank"] = args.dp_rank
    return p


def _time_room(args) -> int:
    """How many tokens this stream can still generate before the deadline (estimated with a conservative per-stream decode rate); unlimited without a deadline."""
    if args.deadline is None:
        return 10**9
    return int((args.deadline - time.monotonic()) * args.deadline_tok_per_s)


def _cutoff(args) -> str | None:
    """Run-level stop condition: the caller says it is hopeless (server dead), or the deadline has arrived -- "arrived"
    means the remaining time cannot finish even one minimal turn. Returns a reason string; None means continue."""
    if args.should_abort is not None and args.should_abort():
        return "abort"
    if args.deadline is not None and _time_room(args) < args.min_turn_tokens:
        return "deadline"
    return None


SANDBOX_KILLED = "[sandbox killed: no reply within 300 s; restarted, all state lost]"

# Mirror of `notes`: the sandbox's RLIMIT_FSIZE=0 blocks all file writes, so the mirror is kept on the harness side --
# after each tool call ask `len(notes)` and read back only the new entries one by one (each ≤ NOTE_ENTRY_CHARS, so they
# never hit the stdout cap); the harness writes them into the run directory itself, and pipes them back via stdin when
# the sandbox restarts. Program state (functions, variables) cannot be saved, but knowledge (notes) should not die with
# the process -- after compaction, notes are the only memory that spans the folded rounds.
NOTE_ENTRY_CHARS = 3000
_NOTES_LEN = "print(len(notes) if isinstance(notes, (list, tuple)) else -1)"
_NOTES_ITEM = "import json as _j; print(_j.dumps(str(notes[{i}])[:{n}]))"
_NOTES_LOAD = "import json as _j; notes = _j.loads({blob!r}); print(len(notes))"


async def sb_run(sb: Sandbox, code: str) -> tuple[str, bool]:
    """A layer over `sb.run`: if the subprocess does not answer for 300s (SIGALRM cannot stop code that swallows
    KeyboardInterrupt or is stuck in a C call), kill and restart it and return (SANDBOX_KILLED, True), handled as a
    timeout. v7's da515329 and v9's 88e364bc each scrapped a whole task because of this -- the latter hung in the home
    check the harness itself runs, so every call site must go through here. Restart count goes to sb.n_restarts; after
    a restart the notes mirror is piped back and the entry count recorded in sb.notes_restored."""
    try:
        return await asyncio.to_thread(sb.run, code)
    except SandboxError as e:
        print(f"!! {e}; restarting sandbox, state lost", file=sys.stderr)
        await asyncio.to_thread(sb.restart)
        sb.notes_synced = 0
        blob = json.dumps(sb.notes_mirror)
        out, timed = await asyncio.to_thread(sb.run, _NOTES_LOAD.format(blob=blob))
        assert not timed and out.strip().isdigit(), f"notes restore failed: {out[-200:]!r}"
        sb.notes_restored = int(out.strip())
        sb.notes_synced = sb.notes_restored
        return SANDBOX_KILLED, True


async def sync_notes(sb: Sandbox) -> None:
    """Read new entries of the sandbox's notes back into the harness mirror (sb.notes_mirror) and write them to the run
    directory. If the model replaced notes with a shorter list (or a non-list), reread the whole thing. If the sandbox
    is dead, skip; the next sb_run handles the restart."""
    try:
        out, timed = await asyncio.to_thread(sb.run, _NOTES_LEN)
        if timed or not out.strip().lstrip("-").isdigit():
            return
        n = int(out.strip())
        if n < 0:   # not a list: treat as a single entry
            out, _ = await asyncio.to_thread(sb.run, "import json as _j; print(_j.dumps(str(notes)[:%d]))" % NOTE_ENTRY_CHARS)
            sb.notes_mirror = [json.loads(out.strip())]
            sb.notes_synced = 1
        else:
            if n < sb.notes_synced:
                sb.notes_mirror, sb.notes_synced = [], 0
            for i in range(sb.notes_synced, n):
                out, _ = await asyncio.to_thread(sb.run, _NOTES_ITEM.format(i=i, n=NOTE_ENTRY_CHARS))
                sb.notes_mirror.append(json.loads(out.strip()))
            sb.notes_synced = n
    except SandboxError:
        return
    sb.notes_path.write_text(json.dumps(sb.notes_mirror, ensure_ascii=False))


async def _tool_round(ep: Episode, sb: Sandbox, tool_calls: list, tool_name: str, hint: str,
                      delegate_ctx: tuple | None = None):
    """delegate_ctx = (client, args, n_test, task): when given, the `delegate` tool is available (not to the helper's own _tool_round)."""
    for call in tool_calls:
        ep.n_tool_calls += 1
        fn = call["function"]
        if fn["name"] == "delegate" and delegate_ctx is not None:
            client, args, n_test, task = delegate_ctx
            try:
                a = fn["arguments"]
                question = (json.loads(a) if isinstance(a, str) else a)["question"]
                assert isinstance(question, str) and question.strip()
            except (json.JSONDecodeError, KeyError, TypeError, AssertionError) as e:
                ep.n_tool_errors += 1
                out = f"Error: delegate needs a non-empty string 'question' ({e})."
            else:
                out = await _delegate(ep, sb, client, args, n_test, task, question)
            ep.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": out + hint})
            continue
        if fn["name"] != tool_name:
            ep.n_tool_errors += 1
            out = f"Error: no tool named {fn['name']!r}; the only tool is {tool_name!r}."
        else:
            args = fn["arguments"]
            try:
                code = (json.loads(args) if isinstance(args, str) else args)["code"]
                if not isinstance(code, str):   # {"code": null} and the like: v8's 0934a4d8 left `out` undefined and scrapped the whole task
                    raise TypeError(f"'code' must be a string, got {type(code).__name__}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                ep.n_tool_errors += 1
                code = None
                out = f"Error: could not parse the tool arguments as JSON with a 'code' field ({e})."
            if code is not None:
                out, timed = await sb_run(sb, code)
                if out == SANDBOX_KILLED:
                    ep.n_sandbox_restarts += 1
                    notes_msg = (f"`notes` was restored from the harness's copy ({sb.notes_restored} entries)"
                                 if sb.notes_restored >= 0 else "`notes` is empty again")
                    out = ("Error: the sandbox stopped responding (your code could not be interrupted, or the process "
                           "died) and was killed and restarted. All functions and variables you defined are gone -- "
                           f"train_pairs / test_inputs / np are back, {notes_msg}; redefine anything else you need.")
                else:
                    await sync_notes(sb)
                ep.n_tool_errors += int(timed)
        ep.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": out + hint})


async def run_episode(client: httpx.AsyncClient, args, *, label: str, system: str, user: str,
                      sb: Sandbox, tool: dict, tool_name: str, n_test: int, lineage: int = 0, hop: int = 1,
                      task_id: str = "", handoff_user: str | None = None, finalize: bool = True,
                      task: dict | None = None) -> Episode:
    """`task` given and args.delegate on: the main loop's tool round may dispatch `delegate`. The helper itself (task=None) cannot."""
    ep = Episode(label, lineage, hop)
    delegate_ctx = (client, args, n_test, task) if (task is not None and args.delegate) else None
    ep.task_id = task_id
    ep.notes_structured = args.notes_structured
    ep.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ep.task_user = user
    # The first turn has no server-reported total_tokens yet, so estimate the prompt length with the tokenizer (+ the
    # fixed template/tool schema overhead). Without the estimate, the first turn's max_tokens treats the whole cap as
    # generatable and a 7k prompt eats report_reserve (smoke2's 142ca369 explorer 2: one turn of pure reasoning 41.5k,
    # only 988 tokens left for the handoff, report truncated to RULE only).
    last_total = args.count_tokens(system + user) + args.template_overhead
    need_handoff = None       # None = normal finish; otherwise the reason that triggered the handoff
    for _turn in range(args.max_turns):
        if args.compact and ep.usage("completion_tokens") >= args.compact_gen_budget:
            need_handoff = "gen_budget"
            break
        cut = _cutoff(args)
        if cut is not None:
            need_handoff = cut
            break
        room = args.ctx_cap - args.report_reserve - last_total - args.context_margin
        if room < args.min_turn_tokens and args.compact:
            # Context full: no handoff; fold the oldest tool round and continue the same conversation (sandbox state intact).
            if args.compact_checkpoint:
                last_total = await _checkpoint_before_compact(ep, args, sb, last_total, client, tool, tool_name)
            new_total = await _compact(ep, args, sb, n_test)
            if new_total is not None:
                last_total = new_total
                room = args.ctx_cap - args.report_reserve - last_total - args.context_margin
        if room < args.min_turn_tokens:
            need_handoff = "context_exhausted"
            break
        # Near the deadline also cap the per-turn max_tokens: a 20k turn takes several minutes on L4, and requests in
        # flight are not cut at the deadline, so the only option is to keep them from being that long up front.
        mt = min(args.max_turn_tokens, room, _time_room(args))
        if args.compact and args.gen_budget_hard:
            # Hard budget: one turn must not carry this conversation's generation past compact_gen_budget. Originally
            # checked only at the start of a turn, so one turn could overshoot by up to max_turn_tokens (20k); the fp8
            # sweep measured a 40k refiner spending 55k. A truncated turn goes through CONTINUE, and at the start of the
            # next turn gen_budget triggers the handoff (the handoff has its own --handoff-budget). At least
            # min_turn_tokens is given, which is the overshoot cap.
            mt = max(args.min_turn_tokens, min(mt, args.compact_gen_budget - ep.usage("completion_tokens")))
        payload = _base_payload(args, ep.messages, tool, mt)
        data = await args.chat(client, payload)
        choice = data["choices"][0]
        finish = choice.get("finish_reason")
        msg = choice["message"]
        last_total = _record_usage(ep, data, finish, "turn", payload["max_tokens"])
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            asst = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
            if msg.get("reasoning_content"):
                asst["reasoning_content"] = msg["reasoning_content"]
            _copy_private(asst, msg)
            ep.messages.append(asst)
            left = (args.compact_gen_budget - ep.usage("completion_tokens") if args.compact
                    else args.ctx_cap - args.report_reserve - last_total)
            hint = f"\n\n[budget: ~{max(left, 0) / 1000:.1f}k tokens of working context left]"
            if finish == "length":
                # Truncated tool call arguments are mostly broken; do not execute.
                for call in tool_calls:
                    ep.n_tool_calls += 1
                    ep.n_tool_errors += 1
                    ep.messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                        "content": "Error: your reply was cut off by the per-reply limit; "
                                                   "this call was not executed." + hint})
                if not _continue_after_cut(ep, args, last_total):
                    if args.compact and args.compact_checkpoint:
                        last_total = await _checkpoint_before_compact(ep, args, sb, last_total, client, tool, tool_name)
                    new_total = await _compact(ep, args, sb, n_test) if args.compact else None
                    if new_total is None or not _continue_after_cut(ep, args, new_total):
                        need_handoff = "length"
                        break
                    last_total = new_total
                continue
            await _tool_round(ep, sb, tool_calls, tool_name, hint, delegate_ctx)
            continue
        # No tool call: final reply.
        asst = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            asst["reasoning_content"] = msg["reasoning_content"]
        _copy_private(asst, msg)
        ep.messages.append(asst)
        if finish == "length":
            # One turn of thinking hit max_turn_tokens (107/185 handoffs in v1 came from this): if the total budget
            # remains, ask for a checkpoint and continue, no handoff. preserve_thinking keeps the truncated thinking in context.
            if _continue_after_cut(ep, args, last_total):
                continue
            if args.compact:
                if args.compact_checkpoint:
                    last_total = await _checkpoint_before_compact(ep, args, sb, last_total, client, tool, tool_name)
                new_total = await _compact(ep, args, sb, n_test)
                if new_total is not None and _continue_after_cut(ep, args, new_total):
                    last_total = new_total
                    continue
            need_handoff = "length_final"
            break
        ep.final_content = msg.get("content") or ""
        ep.final_reasoning = msg.get("reasoning_content") or ""
        ep.stop_reason = "answered"
        break
    else:
        need_handoff = "turn_cap"

    if need_handoff in ("deadline", "abort"):
        # Do not call the model again. The handoff report itself needs one generation of ≥3.5k tokens, and there is no
        # such budget after the deadline; the caller collects the answer from the sandbox's existing `transform` instead
        # (run_hop's home_check with code=None takes exactly this path), and the report is empty.
        ep.forced_handoff = True
        ep.stop_reason = f"handoff:{need_handoff}"
    elif need_handoff is not None:
        ep.forced_handoff = True
        ep.messages.append({"role": "user", "content": handoff_user or prompts.HANDOFF_USER.format(n_calls=args.handoff_tool_calls)})
        calls_left = 0 if handoff_user else args.handoff_tool_calls
        refused = 0          # tool_choice=none has no counterpart in the tokens/dsv4 backends: a model still calling tools after STOP is stopped here
        handoff_used = 0     # cumulative generation in the handoff phase (denominator of --handoff-budget)
        while True:
            if _cutoff(args) is not None:
                break        # deadline reached during handoff: the report counts as far as it got; no further generation
            mt = args.ctx_cap - last_total - args.handoff_margin
            if mt < args.min_handoff_tokens:
                # report_reserve is set aside for this and normally never reached; if it is, prefer slightly exceeding the cap to get a full report.
                mt = args.min_handoff_tokens
            if calls_left > 0:
                # Checkpoint tool calls must be short: in smoke3 without a cap this turn thought for 6–12k and squeezed the report into truncation.
                mt = min(mt, args.handoff_turn_tokens)
            if args.handoff_budget:
                # Hard cap on the whole handoff phase. A model that never writes the report and keeps calling tools
                # generates up to the ctx remainder (at most ~30k) in each of 2 checkpoint + 3 refusal turns: local
                # plumbing measured an explorer with a 20k budget spending 30–58k, all of the overshoot here.
                # The Kaggle budget is hard; prefer a truncated report (the answer can still come from the sandbox's transform).
                left_h = args.handoff_budget - handoff_used
                if left_h < 256:
                    print(f"!! {ep.task_id} {ep.label}: handoff budget {args.handoff_budget} used up; forcing final",
                          file=sys.stderr)
                    break
                mt = min(mt, left_h)
            payload = _base_payload(args, ep.messages, tool, mt)
            if calls_left <= 0:
                payload["tool_choice"] = "none"
            if args.handoff_thinking == "off":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            elif args.handoff_thinking == "low":
                payload["reasoning_effort"] = "low"
            data = await args.chat(client, payload)
            choice = data["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason")
            last_total = _record_usage(ep, data, finish, "handoff", mt)
            handoff_used += ep.calls[-1]["completion_tokens"]
            tool_calls = msg.get("tool_calls") or []
            asst = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                asst["reasoning_content"] = msg["reasoning_content"]
            _copy_private(asst, msg)
            if tool_calls and finish != "length" and calls_left > 0:
                asst["tool_calls"] = tool_calls
                ep.messages.append(asst)
                calls_left -= len(tool_calls)
                hint = (f"\n\n[handoff: {max(calls_left, 0)} tool call(s) left, then write the FINAL REPORT]"
                        if calls_left > 0 else "\n\n[handoff: no tool calls left -- write the FINAL REPORT now]")
                await _tool_round(ep, sb, tool_calls, tool_name, hint)
                continue
            if tool_calls or (finish == "length" and calls_left > 0):
                # The checkpoint turn was truncated (tool call or pure thinking): do not execute, ask for the report directly.
                if tool_calls:
                    asst["tool_calls"] = tool_calls
                ep.messages.append(asst)
                for call in tool_calls:
                    ep.n_tool_calls += 1
                    ep.n_tool_errors += 1
                    ep.messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                        "content": "Error: cut off; not executed."})
                ep.messages.append({"role": "user", "content": "Cut off. No more tool calls -- write the FINAL REPORT now."})
                calls_left = 0
                refused += 1
                if refused <= 2:
                    continue
                # Third consecutive turn still calling tools instead of writing the report (2026-09-03 audit: 20/14k episodes
                # called tools another 3–23 times after STOP, context over cap): take this turn's content as the final reply, no more chances.
                print(f"!! {ep.task_id} {ep.label}: still calling tools after STOP x{refused}; forcing final", file=sys.stderr)
            else:
                ep.messages.append(asst)
            ep.final_content = msg.get("content") or ""
            ep.final_reasoning = msg.get("reasoning_content") or ""
            break
        ep.stop_reason = f"handoff:{need_handoff}"

    if finalize:
        await finalize_report(ep, sb, n_test)
    if args.save_segments:
        if args.annotate_ids is not None:
            args.annotate_ids(ep.messages, args.reasoning_effort)
        ep.segments.append(copy.deepcopy(ep.messages))
    ep.latency_s = time.monotonic() - ep.t0
    return ep


async def _delegate(ep: Episode, sb: Sandbox, client, args, n_test: int, task: dict, question: str) -> str:
    """--delegate: open a helper conversation with a clean context (same sandbox); when it finishes, only its short
    conclusion goes back to the explorer as the tool result. The helper's tokens are merged into the parent episode's
    calls (kind prefix delegate:), and the conversation itself is stored in ep.sub_episodes."""
    if ep.n_delegations >= args.delegate_max:
        ep.n_tool_errors += 1
        return f"Error: delegation limit ({args.delegate_max} per conversation) reached; continue yourself."
    ep.n_delegations += 1
    names_out, _ = await sb_run(sb, "print(sorted(k for k, v in globals().items() if not k.startswith('_') "
                                    "and callable(v) and getattr(v, '__module__', None) == '__main__'))")
    notes = await read_notes(sb, args.tool_max_output_chars, max_items=20, max_chars=2500)
    test_inputs = [t["input"] for t in task["test"]]
    system = prompts.helper_system(n_test, args.delegate_budget, args.tool_timeout, args.tool_max_output_chars)
    if args.compact:
        system += prompts.COMPACT_SYSTEM_NOTE
    user = prompts.render_helper_user(task["train"], test_inputs, names_out.strip()[-300:], notes, args.notes_structured, question)
    sub_args = argparse.Namespace(**vars(args))
    sub_args.extra_tools = []                         # the helper cannot delegate again
    sub_args.compact_gen_budget = args.delegate_budget
    sub_args.max_turns = args.delegate_max_turns
    sub = await run_episode(client, sub_args, label=f"{ep.label} helper {ep.n_delegations}", system=system, user=user,
                            sb=sb, tool=args.tool, tool_name=args.tool_name, n_test=n_test, lineage=ep.lineage, hop=ep.hop,
                            task_id=ep.task_id, handoff_user=prompts.HELPER_HANDOFF, finalize=False)
    for c in sub.calls:
        ep.calls.append(dict(c, kind="delegate:" + c["kind"]))
    ep.n_tool_calls += sub.n_tool_calls
    ep.sub_episodes.append(sub.to_dict(keep_messages=args.save_messages))
    reply = (sub.final_content or "").strip()
    if len(reply) > args.delegate_reply_chars:
        reply = reply[:args.delegate_reply_chars] + "\n... (helper reply truncated)"
    print(f"·· {ep.task_id} {ep.label}: delegation #{ep.n_delegations} used {sub.usage('completion_tokens') / 1000:.0f}k gen, "
          f"{sub.n_tool_calls} tool calls, stop={sub.stop_reason}", file=sys.stderr)
    return (f"[helper finished: {sub.usage('completion_tokens') / 1000:.0f}k tokens, {sub.n_tool_calls} tool calls, "
            f"stop={sub.stop_reason}]\n" + (reply or "(the helper produced no reply)"))


def _msg_tokens(args, m: dict) -> int:
    """Estimated tokens of one message (content + reasoning + tool call arguments + 8 tokens of template overhead per message)."""
    n = 8 + args.count_tokens(m.get("content") or "")
    if m.get("reasoning_content"):
        n += args.count_tokens(m["reasoning_content"])
    for call in m.get("tool_calls") or []:
        n += args.count_tokens(call["function"]["arguments"]) + 12
    return n


OUTPUT_CLEARED = ("[output cleared by the harness to save context -- the code above already ran and everything it "
                  "defined is still in the sandbox; re-run it if you need to see the output again]")


def _round_summary(r: list[dict], ep: Episode) -> str:
    """One line kept per folded round: head of the first tool call's code → head of its output (Manus: failures must stay in context, or they get retried)."""
    a = r[0]
    calls = a.get("tool_calls") or []
    if calls:
        args = calls[0]["function"]["arguments"]
        try:
            code = (json.loads(args) if isinstance(args, str) else args).get("code") or ""
        except (json.JSONDecodeError, AttributeError, TypeError):
            code = str(args)
        lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        head = " | ".join(lines[:2])[:110]
    else:
        head = "(no tool call) " + (a.get("content") or "")[:80].replace("\n", " ")
    outs = [m for m in r if m["role"] == "tool"]
    if not outs:
        return head
    out = outs[0]["content"]
    if out == OUTPUT_CLEARED:
        out = ep.output_heads.get(outs[0].get("tool_call_id", ""), "(output cleared)")
    out = out.split("\n\n[budget")[0].strip().replace("\n", " / ")
    return f"{head}  ->  {out[:110]}"


def _clear_tool_outputs(r: list[dict], ep: Episode) -> None:
    """Anthropic's context-editing approach: replace a round's tool outputs with a placeholder, keeping the model's own
    reasoning and code. Outputs are recoverable (the code and the sandbox are still there); reasoning is not."""
    for m in r:
        if m["role"] == "tool" and m["content"] != OUTPUT_CLEARED:
            ep.output_heads[m.get("tool_call_id", "")] = m["content"][:160]
            m["content"] = OUTPUT_CLEARED
            ep.n_outputs_cleared += 1


COMPACT_NOTE_HEAD = """\
[harness note -- context compacted]
Your earliest {n_rounds} tool round(s) ({n_calls} tool calls, about {dropped_k}k tokens) were removed from this \
conversation to stay within the context limit; only your most recent rounds are shown below. The sandbox was NOT \
reset: whatever those rounds successfully defined is still there. The list below is the authoritative inventory -- \
a name that is not in it does not exist (its definition failed or was never run), so if you need it, define it again \
rather than calling it. Do not redo work the sandbox already holds; re-run a function if you need its output again.
Sandbox now: functions defined: {names}. `transform` on the training pairs: {train}."""


async def _checkpoint_before_compact(ep: Episode, args, sb: Sandbox, last_total: int, client, tool, tool_name) -> int:
    """--compact-checkpoint: before folding old rounds, give the model one turn (≤ compact_checkpoint_tokens) to write
    the reasoning about to disappear into `notes`. This turn with its tool outputs is the newest round, which compaction
    keeps; the cost comes out of the idle space in report_reserve. Returns the new last_total."""
    starts = [i for i, m in enumerate(ep.messages) if m["role"] == "assistant"]
    n_rounds = len(starts)
    tmpl = prompts.CHECKPOINT_USER_STRUCTURED if args.notes_structured else prompts.CHECKPOINT_USER
    ep.messages.append({"role": "user", "content": tmpl.format(n_rounds=max(n_rounds - 1, 1))})
    mt = min(args.compact_checkpoint_tokens, args.ctx_cap - last_total - args.context_margin)
    if mt < 1000:
        # Not even a checkpoint fits (one turn ate the reserve too): skip it and fold directly.
        ep.messages.pop()
        return last_total
    payload = _base_payload(args, ep.messages, tool, mt)
    data = await args.chat(client, payload)
    choice = data["choices"][0]
    msg = choice["message"]
    finish = choice.get("finish_reason")
    last_total = _record_usage(ep, data, finish, "checkpoint", mt)
    tool_calls = msg.get("tool_calls") or []
    asst = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        asst["reasoning_content"] = msg["reasoning_content"]
    _copy_private(asst, msg)
    if tool_calls:
        asst["tool_calls"] = tool_calls
    ep.messages.append(asst)
    ep.n_checkpoints += 1
    if tool_calls and finish != "length":
        await _tool_round(ep, sb, tool_calls, tool_name, "")
    elif tool_calls:
        for call in tool_calls:
            ep.n_tool_calls += 1
            ep.n_tool_errors += 1
            ep.messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": "Error: cut off; not executed."})
    return last_total


async def _compact(ep: Episode, args, sb: Sandbox, n_test: int) -> int | None:
    """--compact: when the context is full, fold the oldest tool rounds out of messages (no sandbox state is lost),
    insert a harness notice after the task message (how much was folded, sandbox state, notes), and continue the same
    conversation. --compact-rich adds three things: (1) first replace tool outputs outside the most recent rounds with
    placeholders (reasoning and code kept), without dropping whole rounds if that is enough; (2) the notice carries the
    source of the last `def transform` in the folded rounds; (3) one "what was tried → result" line per folded round.
    Returns the estimated context tokens after folding; returns None if only one round is left and nothing can be
    folded (the caller then hands off as before)."""
    msgs = ep.messages
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[1]["content"] == ep.task_user, ep.label
    starts = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(starts) < 2:
        return None
    rounds = [msgs[a:b] for a, b in zip(starts, starts[1:] + [len(msgs)])]
    # --save-segments: before folding, save the full messages the model currently sees (rich mode then clears tool outputs in place, hence the deepcopy here).
    if args.save_segments and args.annotate_ids is not None:
        args.annotate_ids(msgs, args.reasoning_effort)   # the last round's tool messages have not been sent to the server yet; fill in their tokens first
    snap = copy.deepcopy(msgs) if args.save_segments else None
    # Target context after folding = compact_target_frac of the working window; minus system + task + notice (est. 3k)
    # is what can be kept for the most recent rounds, floor compact_keep_tokens. The refiner's task message (K reports +
    # candidate table) is much larger than the explorer's; a fixed keep would leave it nearly full again right after folding.
    base = _msg_tokens(args, msgs[0]) + _msg_tokens(args, msgs[1]) + 3000 + args.template_overhead
    window = args.ctx_cap - args.report_reserve - args.context_margin
    keep_budget = max(args.compact_keep_tokens, int(window * args.compact_target_frac) - base)
    if args.compact_rich:
        # Stage 1: clear tool outputs outside the most recent compact_keep_full_rounds rounds.
        for r in rounds[:max(len(rounds) - args.compact_keep_full_rounds, 0)]:
            _clear_tool_outputs(r, ep)
    sizes = [sum(_msg_tokens(args, m) for m in r) for r in rounds]
    # Stage 2: keep rounds walking back from the newest; keep at least 1. In rich mode, if clearing outputs is enough, drop no round.
    keep, acc = 0, 0
    for size in reversed(sizes):
        if keep >= 1 and acc + size > keep_budget:
            break
        keep += 1
        acc += size
    keep = min(keep, len(rounds) - 1) if not args.compact_rich else min(keep, len(rounds))
    dropped = rounds[:len(rounds) - keep]
    if not dropped and ep.n_outputs_cleared == 0:
        return None
    n_calls = sum(len(m.get("tool_calls") or []) for r in dropped for m in r)
    ep.compact_dropped_tokens += sum(sizes[:len(dropped)])
    ep.compact_dropped_rounds += len(dropped)
    ep.compact_dropped_calls += n_calls
    ep.n_compactions += 1
    if snap is not None:
        ep.segments.append(snap)
    if args.compact_rich and dropped:
        ep.compact_history.extend(_round_summary(r, ep) for r in dropped)
        ep.compact_history = ep.compact_history[-40:]
        src = _salvage_transform_code([m for r in dropped for m in r])
        if src:
            ep.compact_transform_src = src

    names_out, _ = await sb_run(sb, "print(sorted(k for k, v in globals().items() if not k.startswith('_') "
                                    "and callable(v) and getattr(v, '__module__', None) == '__main__'))")
    chk = await _check_transform(sb, None, "transform", n_test)
    notes = await read_notes(sb, args.tool_max_output_chars, max_items=12, max_chars=1500)
    parts = [COMPACT_NOTE_HEAD.format(
        n_rounds=ep.compact_dropped_rounds, n_calls=ep.compact_dropped_calls,
        dropped_k=f"{ep.compact_dropped_tokens / 1000:.0f}", names=_trim_names(names_out.strip()),
        train=chk["summary"])]
    if args.compact_rich and ep.compact_transform_src and chk["defined"]:
        src = ep.compact_transform_src
        if len(src) > 2500:
            src = src[:1600] + "\n# ... (middle elided) ...\n" + src[-800:]
        parts.append("Current `transform` in the sandbox (the last definition from the removed rounds, verbatim):\n"
                     "```python\n" + src.rstrip() + "\n```")
    if args.compact_rich and ep.compact_history:
        parts.append("What the removed rounds tried (oldest first; one line each: code -> output):\n"
                     + "\n".join(f"- {h}" for h in ep.compact_history))
    parts.append(f"`notes` ({len(notes)} entries, newest last):\n" + prompts.render_notes(notes, args.notes_structured))
    parts.append("Continue from where you left off.")
    note = "\n".join(parts)
    # The notice is a separate user message and the task message stays unchanged: only then does the system + task
    # prefix keep hitting the radix cache (in smoke, with the notice appended to the task, the hit rate fell back to
    # only the system prompt after every fold).
    new_msgs = [msgs[0], msgs[1], {"role": "user", "content": note}]
    for r in rounds[len(dropped):]:
        new_msgs.extend(r)
    ep.messages = new_msgs
    est = sum(_msg_tokens(args, m) for m in new_msgs) + args.template_overhead
    print(f"·· {ep.task_id} {ep.label}: compaction #{ep.n_compactions} dropped {len(dropped)} rounds "
          f"(~{sum(sizes[:len(dropped)]) / 1000:.0f}k), outputs cleared so far {ep.n_outputs_cleared}, "
          f"context est {est / 1000:.0f}k", file=sys.stderr)
    return est


def _trim_names(names: str, limit: int = 900) -> str:
    """Sandbox function list for the notice: keep the head, and when over the limit say how many were cut (it used to be `[-300:]`, cutting the head, so 11% of notice lists started with half a name)."""
    if not names:
        return "(none)"
    if len(names) <= limit:
        return names
    cut = names[:limit]
    n_more = names[limit:].count("', '") + 1
    return cut.rsplit("', '", 1)[0] + f"', ... (list truncated, {n_more} more)"


def _continue_after_cut(ep: Episode, args, last_total: int) -> bool:
    """One turn truncated by max_turn_tokens: if the total budget allows and the count is not used up, append a CONTINUE message so it keeps going."""
    left = args.ctx_cap - args.report_reserve - last_total - args.context_margin
    # In compact mode the total is governed by --compact-gen-budget and continuations are unlimited (in smoke one
    # explorer handed off at 104k after using up its 4 continuations, throwing away 100k of budget).
    if (not args.compact and ep.n_continuations >= args.max_continuations) or left < args.min_turn_tokens:
        return False
    ep.n_continuations += 1
    ep.messages.append({"role": "user", "content": prompts.CONTINUE_USER.format(left_k=f"{left / 1000:.0f}")})
    return True


def _salvage_transform_code(messages: list[dict]) -> str | None:
    """When the report has no CODE, fetch the code from the last tool call that defined `transform`."""
    for m in reversed(messages):
        if m["role"] != "assistant":
            continue
        for call in reversed(m.get("tool_calls") or []):
            args = call["function"]["arguments"]
            try:
                code = (json.loads(args) if isinstance(args, str) else args)["code"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if re.search(r"^\s*def transform\s*\(", code, re.MULTILINE):
                return code
    return None


async def eval_grid_expr(sb: Sandbox, expr: str):
    """Evaluate an expression in the sandbox to a grid; on failure return (None, reason)."""
    code = (f"import json as _json\n_v = ({expr})\n"
            f"print({GRID_MARK!r} + _json.dumps(np.asarray(_v).astype(int).tolist(), separators=(',', ':')))")
    out, timed = await sb_run(sb, code)
    if timed:
        return None, "timeout"
    return _grid_from_output(out)


def _grid_from_output(out: str):
    i = out.find(GRID_MARK)
    if i < 0:
        return None, "error: " + out.strip()[-300:]
    line = out[i + len(GRID_MARK):].split("\n", 1)[0]
    if "[truncated" in out:
        return None, "output truncated"
    try:
        g = json.loads(line)
    except json.JSONDecodeError:
        return None, "unparseable"
    if not prompts.is_grid(g):
        return None, "not a grid"
    return g, "ok"


async def finalize_report(ep: Episode, sb: Sandbox, n_test: int):
    rep = prompts.parse_report(ep.final_content)
    src = "content"
    if not rep["has_sections"] and ep.final_reasoning:
        alt = prompts.parse_report(ep.final_reasoning)
        if alt["has_sections"]:
            rep, src = alt, "reasoning"
    rep["source"] = src
    if rep["code"] is None:
        salvaged = _salvage_transform_code(ep.messages)
        if salvaged is not None:
            rep["code"] = salvaged
            ep.code_src = "salvaged_tool_call"
    else:
        ep.code_src = f"report_{src}"
    ep.report = rep
    ep.answers = [[] for _ in range(n_test)]
    ep.answer_resolution = [[] for _ in range(n_test)]
    for key, (kind, body) in sorted(rep["answer_blocks"].items(), key=lambda kv: tuple(map(int, kv[0].split(":")))):
        t, a = map(int, key.split(":"))
        if t >= n_test or a >= 2:
            continue
        if kind == "none":
            continue
        if kind == "grid":
            g, why = json.loads(body), "ok"
        else:
            g, why = await eval_grid_expr(sb, body)
            if g is None and "NameError" in why and rep["code"]:
                # The report's CODE was never run in the sandbox (or used another name): define it there and evaluate again.
                out, _ = await sb_run(sb, rep["code"])
                g, why2 = await eval_grid_expr(sb, body)
                why = f"{why2} after defining report code"
        ep.answer_resolution[t].append(f"{kind}:{why}")
        if g is not None and g not in ep.answers[t]:
            ep.answers[t].append(g)


# ---------------------------------------------------------------------------
# Verification of explorer code in the refine sandbox
# ---------------------------------------------------------------------------

_DEFINE_TMPL = """\
import numpy as np, itertools, math, re, copy, functools, collections
from collections import Counter, deque, defaultdict
_ns = {{"np": np, "itertools": itertools, "math": math, "re": re, "copy": copy, "functools": functools,
        "collections": collections, "Counter": Counter, "deque": deque, "defaultdict": defaultdict,
        "__name__": "__explorer__"}}
try:
    exec({code!r}, _ns)
    {name} = _ns["transform"]
    print("defined")
except BaseException as _e:
    print({def_err!r} + repr(_e)[:300])
"""

# For continuation sessions: CODE is exec'd straight into the sandbox globals; report code often depends on helpers already defined in the sandbox.
_DEFINE_GLOBAL_TMPL = """\
try:
    exec({code!r}, globals())
    assert callable(transform)
    print("defined")
except BaseException as _e:
    print({def_err!r} + repr(_e)[:300])
"""

_TRAIN_CHECK_TMPL = """\
import json as _json
_res = []
for _p in train_pairs:
    try:
        _o = np.asarray({name}([r[:] for r in _p["input"]])).astype(int).tolist()
        _e = _p["output"]
        if _o == _e:
            _res.append("pass")
        elif len(_o) != len(_e) or len(_o[0]) != len(_e[0]):
            _res.append("FAIL shape %dx%d vs %dx%d" % (len(_o), len(_o[0]), len(_e), len(_e[0])))
        else:
            _n = sum(1 for i in range(len(_e)) for j in range(len(_e[0])) if _o[i][j] != _e[i][j])
            _res.append("FAIL %d cells" % _n)
    except BaseException as _x:
        _res.append("ERROR " + repr(_x)[:120])
print({mark!r} + _json.dumps(_res))
"""

_TEST_PRED_TMPL = """\
import json as _json
try:
    _o = np.asarray({name}([r[:] for r in test_inputs[{t}]])).astype(int).tolist()
    print({mark!r} + _json.dumps(_o, separators=(',', ':')))
except BaseException as _x:
    print({err!r} + repr(_x)[:200])
"""


NOTES_MARK = "__NOTES__"


async def _check_transform(sb: Sandbox, code: str | None, name: str, n_test: int, in_globals: bool = False,
                           load_only: bool = False) -> dict:
    """Define `code` as `name` in the sandbox (when code is None use what the sandbox already has), and run all train
    pairs and test inputs. `in_globals`: exec directly in the sandbox globals (only for name == "transform") so helpers
    can be found."""
    chk = {"name": name, "defined": False, "train": [], "test": [None] * n_test, "summary": "no code"}
    if code:
        if in_globals:
            assert name == "transform"
            define = _DEFINE_GLOBAL_TMPL.format(code=code, def_err=DEF_ERR_MARK)
        else:
            define = _DEFINE_TMPL.format(code=code, name=name, def_err=DEF_ERR_MARK)
        out, timed = await sb_run(sb, define)
        if timed or DEF_ERR_MARK in out or "defined" not in out:
            chk["summary"] = "code failed to load: " + out.replace(DEF_ERR_MARK, "").strip()[-200:]
            return chk
    else:
        out, _ = await sb_run(sb, f"print('defined' if callable(globals().get({name!r})) else 'undefined')")
        if "defined" not in out or "undefined" in out:
            chk["summary"] = f"no `{name}` in the sandbox"
            return chk
    chk["defined"] = True
    if load_only:
        chk["summary"] = "loaded"
        return chk
    out, timed = await sb_run(sb, _TRAIN_CHECK_TMPL.format(name=name, mark=GRID_MARK))
    if timed or GRID_MARK not in out:
        chk["train"] = ["TIMEOUT" if timed else "ERROR"]
    else:
        chk["train"] = json.loads(out.split(GRID_MARK, 1)[1].split("\n", 1)[0])
    for t in range(n_test):
        out, timed = await sb_run(
            sb, _TEST_PRED_TMPL.format(name=name, t=t, mark=GRID_MARK, err=ERR_MARK))
        if timed:
            chk["test"][t] = {"grid": None, "why": "timeout"}
        elif ERR_MARK in out:
            chk["test"][t] = {"grid": None, "why": out.split(ERR_MARK, 1)[1].split("\n", 1)[0][:200]}
        else:
            g, why = _grid_from_output(out)
            chk["test"][t] = {"grid": g, "why": why}
    tr = chk["train"]
    npass = sum(1 for r in tr if r == "pass")
    bits = [f"train pairs {npass}/{len(tr)} pass" + ("" if npass == len(tr) else
            " (" + "; ".join(f"pair {i + 1}: {r}" for i, r in enumerate(tr) if r != "pass") + ")")]
    for t in range(n_test):
        r = chk["test"][t]
        if r["grid"] is None:
            bits.append(f"test {t + 1}: {r['why']}")
        else:
            bits.append(f"test {t + 1}: {len(r['grid'])}x{len(r['grid'][0])} grid")
    chk["summary"] = "; ".join(bits)
    return chk


def _checks_from_home(reports: list[dict]) -> list[dict]:
    """Check table without opening a sandbox: use each explorer's own-sandbox home_check directly (all verify_reports
    adds is loading CODE into the refine sandbox). Used by two paths: "write a submission before the refiner" and "the
    deadline arrived and the refiner did not run"."""
    out = []
    for k, rep in enumerate(reports, start=1):
        home = rep["home_check"]
        assert home is not None, rep["label"]
        out.append({"name": f"transform_{k}", "defined": home["defined"], "load_err": None,
                    "train": home["train"], "test": home["test"], "summary": home["summary"]})
    return out


async def verify_reports(sb: Sandbox, reports: list[dict], n_test: int) -> list[dict]:
    """One check dict per report: train/test results come from the home_check in the explorer's own sandbox (v4–v6
    re-ran CODE in the refine sandbox, and 28% hit NameError because helpers / imports were missing, so the refiner was
    told usable code was broken); the refine sandbox only tries to load CODE as transform_k, and if it cannot, the
    summary says so."""
    checks = []
    for k, rep in enumerate(reports, start=1):
        home = rep["home_check"]
        assert home is not None, rep["label"]
        name = f"transform_{k}"
        chk = {"name": name, "defined": False, "load_err": None, "train": home["train"], "test": home["test"]}
        if rep["code"]:
            here = await _check_transform(sb, rep["code"], name, n_test, in_globals=False, load_only=True)
            chk["defined"] = here["defined"]
            chk["load_err"] = None if here["defined"] else here["summary"]
        if not home["defined"]:
            chk["summary"] = home["summary"]
        else:
            chk["summary"] = "run in the explorer's own sandbox (where its helpers exist): " + home["summary"]
            if chk["defined"]:
                chk["summary"] += f" -- also loaded in your sandbox as `{name}`"
            else:
                chk["summary"] += (f" -- NOTE: it does NOT load in your sandbox ({chk['load_err']}); "
                                   "re-implement the missing pieces if you need to run it")
        checks.append(chk)
    return checks


async def read_notes(sb: Sandbox, out_limit: int, max_items: int = 40, max_chars: int = 3000) -> list[str]:
    """Contents of `notes` in the sandbox (the model may turn it into a non-list or stuff odd things into it; always
    convert to strings and truncate). Truncation happens inside the sandbox: the printed line must be shorter than the
    sandbox's output cap `out_limit`, or it gets cut mid-line and the JSON breaks."""
    budget = min(max_chars, out_limit - len(NOTES_MARK) - 200)
    assert budget > 0, out_limit
    code = f"""import json as _json
_n = notes if isinstance(notes, (list, tuple)) else [notes]
_items = [str(_x)[:400] for _x in list(_n)[-{max_items}:]]
_kept, _text = [], 0
for _s in _items:
    _text += len(_s)
    if _text > {budget}:
        break
    _kept.append(_s)
_out = _json.dumps([_kept, len(_items) - len(_kept)])
while len(_out) > {budget} and _kept:
    _kept.pop()
    _out = _json.dumps([_kept, len(_items) - len(_kept)])
print({NOTES_MARK!r} + _out)"""
    out, timed = await sb_run(sb, code)
    if timed or NOTES_MARK not in out:
        return [f"(could not read notes: {out.strip()[-200:]})"]
    if "[truncated:" in out:
        raise RuntimeError(f"notes output truncated despite budget {budget}: {out[:300]!r}")
    kept, dropped = json.loads(out.split(NOTES_MARK, 1)[1].split("\n", 1)[0])
    if dropped:
        kept.append(f"... ({dropped} more entries not shown)")
    return kept


def _is_alt_source(src: str) -> bool:
    return src.endswith(" alt") or src.startswith("transform_alt_")


def build_candidates(reports: list[dict], checks: list[dict], n_test: int) -> list[dict]:
    """Per test: deduped candidate grids, A, B, ... ordered by (votes, first appearance). The alt variant's
    counter-hypothesis (ATTEMPT 2, a `transform_alt` passing all train pairs) also enters the candidates but gets no
    vote: it is the same explorer's second reading."""
    cands = []
    for t in range(n_test):
        pool: list[tuple[list, str, int | None]] = []
        for k, (rep, chk) in enumerate(zip(reports, checks), start=1):
            for i, g in enumerate(rep["answers"][t]):
                pool.append((g, f"explorer {k} answer" if i == 0 else f"explorer {k} alt", rep["confidence"]))
            pred = chk["test"][t]
            if pred is not None and pred["grid"] is not None:
                pool.append((pred["grid"], f"transform_{k}", rep["confidence"]))
            alt = rep["alt_check"]
            if _all_train_pass(alt) and alt["test"][t]["grid"] is not None:
                pool.append((alt["test"][t]["grid"], f"transform_alt_{k}", rep["confidence"]))
        distinct: list[dict] = []
        for g, src, conf in pool:
            for d in distinct:
                if d["grid"] == g:
                    d["sources"].append(src)
                    d["confs"].append(conf)
                    break
            else:
                distinct.append({"grid": g, "sources": [src], "confs": [conf]})
        distinct.sort(key=lambda d: (-sum(1 for s in d["sources"] if not _is_alt_source(s)),
                                     -max((c for c in d["confs"] if c is not None), default=-1)))
        keyed = {}
        for i, d in enumerate(distinct):
            key = chr(ord("A") + i) if i < 26 else f"C{i}"
            keyed[key] = d
        cands.append(keyed)
    return cands


# ---------------------------------------------------------------------------
# Per-task orchestration
# ---------------------------------------------------------------------------

def load_task(task_dir: Path, task_id: str) -> dict:
    d = json.loads((task_dir / f"{task_id}.json").read_text())
    return {"id": task_id, "train": d["train"], "test": d["test"]}


# ---------------------------------------------------------------- --task-source kaggle (ARC Prize submission)
def load_challenges(path: Path) -> dict:
    """The competition's single big JSON (`arc-agi_test_challenges.json`, tests without outputs); also accepts the local
    `<dir>/<task_id>.json` layout (public eval embeds test outputs, so it is scored). Same conventions as
    work/kaggle_submit/run_submission.py."""
    if path.is_dir():
        return {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}
    return json.loads(path.read_text())


# ---------------------------------------------------------------- --task-source nvarc (distillation corpus)
def choose_pairs(uid: str, attempt: int, n_pairs: int, n_train: int, n_test: int = 1):
    """Matches work/qwen38fn_distill/rollout_runner.py::choose_pairs exactly (identical return for n_test=1): the same
    uid/attempt draws the same train/test here as in the naive corpus, so the two datasets can be paired task by task."""
    h = hashlib.sha256(f"{uid}:{attempt}".encode()).digest()
    order = sorted(range(n_pairs), key=lambda i: hashlib.sha256(h + i.to_bytes(2, "big")).digest())
    chosen = order[: n_train + n_test]
    return sorted(chosen[:n_train]), chosen[n_train:n_train + n_test]


def load_nvarc_corpus(path: str) -> dict[str, list]:
    """`nvarc_training.jsonl.zst`: one {"uid", "pairs": [[input, output], ...]} per line. About 47k tasks in total, 90 MB compressed."""
    import io
    import zstandard
    corpus: dict[str, list] = {}
    with open(path, "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        for line in text:
            d = json.loads(line)
            assert d["uid"] not in corpus, f"duplicate uid {d['uid']}"
            corpus[d["uid"]] = d["pairs"]
    return corpus


def nvarc_task(uid: str, pairs: list, attempt: int, n_train: int, n_test: int) -> dict:
    tr, te = choose_pairs(uid, attempt, len(pairs), n_train, n_test)
    return {"id": uid,
            "train": [{"input": pairs[i][0], "output": pairs[i][1]} for i in tr],
            "test": [{"input": pairs[i][0], "output": pairs[i][1]} for i in te],
            "train_idx": tr, "test_idx": te}


# --hparam-menu: draw one set of hyperparameters per task deterministically (each axis hashed independently, axes
# uncorrelated), so the corpus covers different context caps / effort / K. The weights are the data-collection
# distribution, not the best setting; the drawn values are stored with the uid in hparams of tasks/<uid>.json.
HPARAM_MENU: dict[str, list[tuple]] = {
    "reasoning_effort": [("xhigh", 0.6), ("medium", 0.3), ("low", 0.1)],
    "ctx_cap": [(50_000, 0.6), (40_000, 0.2), (65_000, 0.2)],
    "n_explorers": [(2, 0.7), (3, 0.3)],           # base_explorers fixed at 2: the 3rd opens only when the first two disagree
    "compact_target_frac": [(0.6, 0.6), (0.5, 0.2), (0.7, 0.2)],
    "compact_gen_budget": [(100_000, 0.7), (150_000, 0.3)],
}


def draw_hparams(uid: str, attempt: int, menu: dict = HPARAM_MENU) -> dict:
    out = {}
    for axis, table in menu.items():
        total = sum(w for _, w in table)
        h = hashlib.sha256(f"{uid}:{attempt}|{axis}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2 ** 64 * total
        acc = 0.0
        for value, w in table:
            acc += w
            if u < acc:
                out[axis] = value
                break
        else:
            out[axis] = table[-1][0]
    return out


def _sandbox(args, task, extra, tag: str):
    sb = Sandbox(task["train"], None, python=args.sandbox_python, timeout=args.tool_timeout,
                 max_output_chars=args.tool_max_output_chars,
                 extra_ns={"test_inputs": [t["input"] for t in task["test"]], "notes": [], **extra})
    notes_dir = Path(args.out) / "notes"
    notes_dir.mkdir(exist_ok=True)
    sb.notes_path = notes_dir / f"{task['id']}_{tag}.json"   # notes mirror (see sync_notes / sb_run)
    sb.notes_mirror: list[str] = []
    sb.notes_synced = 0
    sb.notes_restored = -1
    return sb


def _with_gen_budget(args, gen_budget: int):
    """Shallow copy of args overriding only this session's compact_gen_budget (closures such as chat / count_tokens are
    shared, as with main_async's targs). run_hop / run_episode / prompts all read the budget from
    args.compact_gen_budget, so passing the copy as args is enough, no signature changes."""
    assert gen_budget > 0, gen_budget
    a = copy.copy(args)
    a.compact_gen_budget = gen_budget
    return a


async def run_hop(client, args, task, k: int, hop: int, sb: Sandbox, prev: Episode | None,
                  live: asyncio.Semaphore) -> Episode:
    """Session `hop` of explorer k; the sandbox is shared across sessions. For hop>1 the prompt carries the previous
    session's report, the `transform` check the harness just ran, and the sandbox's `notes` (v4: cut long-chain
    thinking into several 50k windows)."""
    n_test = len(task["test"])
    test_inputs = [t["input"] for t in task["test"]]
    system = prompts.explorer_system(n_test, args.compact_gen_budget if args.compact else args.ctx_cap,
                                     args.max_turn_tokens, args.tool_timeout, args.tool_max_output_chars,
                                     variant=args.prompt_variant, notes_structured=args.notes_structured)
    if args.compact:
        system += prompts.COMPACT_SYSTEM_NOTE
    if args.delegate:
        system += prompts.DELEGATE_SYSTEM_NOTE
    async with live:
        if hop == 1:
            assert prev is None
            user = prompts.render_explorer_user(task["train"], test_inputs)
        else:
            assert prev is not None and prev.forced_handoff
            # The report's CODE is the `transform` the successor will read: define it back into the sandbox so text and object agree.
            assert prev.home_check is not None
            info = dict(prev.report, stop_reason=prev.stop_reason, check=prev.home_check, notes=prev.notes,
                        notes_structured=args.notes_structured)
            user = prompts.render_continue_user(task["train"], test_inputs, hop, args.max_hops, info)
        ep = await run_episode(client, args, label=f"explorer {k} session {hop}", system=system, user=user,
                               sb=sb, tool=args.tool, tool_name=args.tool_name, n_test=n_test, lineage=k, hop=hop,
                               task_id=task["id"], task=task)
        ep.notes = await read_notes(sb, args.tool_max_output_chars)
        # Define the report's CODE back into the sandbox and run train/test: the `transform` the successor reads matches
        # the text, and the check results the refiner gets were also produced with all helpers present.
        ep.home_check = await _check_transform(sb, ep.report["code"], "transform", n_test, in_globals=True)
        if args.prompt_variant == "alt":
            # The counter-hypothesis `transform_alt` (if the model defined it) also runs train/test in its own sandbox: the refiner sees it, and all-pass ones enter candidates.
            ep.alt_check = await _check_transform(sb, None, "transform_alt", n_test)
        return ep


def _all_train_pass(chk: dict | None) -> bool:
    return bool(chk and chk["defined"] and chk["train"] and all(r == "pass" for r in chk["train"]))


def _agreed_answer(finals: list[Episode], n_test: int) -> list | None:
    """If ≥2 finished explorers give exactly the same first answer for every test, return that answer set (one grid per
    test), else None. In v2, 114/126 finished explorers were right; when two independent ones finish and agree there is
    no need to spend sessions on the others (reap128 final 120 tasks: 71 agreed, 62 right; answered-only gives the same
    numbers as ignoring the stop reason)."""
    done = [ep for ep in finals if ep.stop_reason == "answered" and all(ep.answers[t] for t in range(n_test))]
    for i in range(len(done)):
        for j in range(i + 1, len(done)):
            if all(done[i].answers[t][0] == done[j].answers[t][0] for t in range(n_test)):
                return [done[i].answers[t][0] for t in range(n_test)]
    return None


def _lineages_converged(finals: list[Episode], n_test: int) -> bool:
    return _agreed_answer(finals, n_test) is not None


async def run_explorers(client, args, task, live: asyncio.Semaphore) -> tuple[list[Episode], list[Episode]]:
    """K explorers, each with at most max_hops sessions. The first base_explorers run first; if two of their final
    answers already agree, stop, and only otherwise open the rest (v5: in v4, of the 56 tasks where the first two
    agreed, none agreed and was wrong; the 3rd explorer was pure waste there).
    Returns (last session of each explorer, all sessions in order)."""
    n_test = len(task["test"])
    K = args.n_explorers
    base = min(args.base_explorers, K)
    all_eps: list[Episode] = []
    finals: list[Episode | None] = [None] * K

    async def run_group(ks: list[int], sbs: dict[int, Sandbox], stack: contextlib.ExitStack):
        for k in ks:
            sbs[k] = stack.enter_context(_sandbox(args, task, {}, f"e{k}"))
        active = list(ks)
        for hop in range(1, args.max_hops + 1):
            if not active:
                break
            # Budget shrinking (in-flight sessions are not cancelled): each session's gen budget = min(original, lineage
            # remainder, task remainder split across this batch). Hitting the budget writes the handoff report via the
            # usual handoff:gen_budget path, and the answer is collected by home_check as before.
            launch = [(k, session_budget(k)) for k in active]
            skipped = [k + 1 for k, gb in launch if gb < args.min_session_budget]
            launch = [(k, gb) for k, gb in launch if gb >= args.min_session_budget]
            if launch:
                share = (args.task_budget - spent()) // len(launch)   # sessions in the same gather see the same spent(): split evenly
                launch = [(k, min(gb, share)) for k, gb in launch]
                skipped += [k + 1 for k, gb in launch if gb < args.min_session_budget]
                launch = [(k, gb) for k, gb in launch if gb >= args.min_session_budget]
            if skipped:
                print(f"·· {task['id']}: hop {hop} skipped for explorer(s) {skipped} "
                      f"(session budget below {args.min_session_budget // 1000}k)", flush=True)
            if not launch:
                break
            eps = await asyncio.gather(*(run_hop(client, _with_gen_budget(args, gb), task, k + 1, hop, sbs[k], finals[k], live)
                                         for k, gb in launch))
            for (k, _), ep in zip(launch, eps):
                finals[k] = ep
                all_eps.append(ep)
            active = [k for k, _ in launch]
            # v9: deep sessions only for explorers that already have a transform passing all train pairs -- v8's hop 4
            # got 16/45 right starting from "previous session all-pass", 2/47 from not all-pass; the latter costs 35k per
            # session, cutting it saves ~14k/task.
            active = [k for k in active if finals[k].forced_handoff
                      and (hop + 1 < args.pass_gate_hop or _all_train_pass(finals[k].home_check))]
            if active and _lineages_converged([e for e in finals if e is not None], n_test):
                active = []
            if active and spent() >= args.task_budget:
                # v10: once this task's explorers have spent the budget, open no more sessions -- tasks v7/v8 got wrong
                # spent 265–342k, right ones ≈ baseline; the extra tokens all went into tasks that stayed unsolved.
                active = []
            if active and _cutoff(args) is not None:
                active = []

    def spent() -> int:
        return sum(e.usage("completion_tokens") for e in all_eps)

    def lineage_spent(k: int) -> int:
        return sum(e.usage("completion_tokens") for e in all_eps if e.lineage == k + 1)

    def session_budget(k: int) -> int:
        gb = args.compact_gen_budget
        if args.lineage_budget:
            gb = min(gb, args.lineage_budget - lineage_spent(k))
        return gb

    with contextlib.ExitStack() as stack:
        sbs: dict[int, Sandbox] = {}
        await run_group(list(range(base)), sbs, stack)
        assert all(finals[k] is not None for k in range(base)), "the first session of each base explorer must open (parse_args guarantees the budget)"
        # The 3rd (base..K) opens only when the earlier ones "disagree" or "none passes train": agreeing-but-wrong is rare
        # (v4 0/56, v5 7/72), but an answer two finished explorers agree on while neither passes train is not trustworthy.
        # Not opened on deadline / abort.
        done_finals = [e for e in finals if e is not None]
        if (base < K and spent() < args.task_budget and _cutoff(args) is None
                and (not _lineages_converged(done_finals, n_test)
                     or not any(_all_train_pass(e.home_check) for e in done_finals))):
            await run_group(list(range(base, K)), sbs, stack)
    finals = [e for e in finals if e is not None]
    assert len(finals) >= base
    return finals, all_eps


async def run_refine(client, args, task, reports: list[dict], round_no: int, live: asyncio.Semaphore):
    """Returns (episode, checks, candidates). The refiner's generation budget is passed in by solve_task via _with_gen_budget."""
    n_test = len(task["test"])
    test_inputs = [t["input"] for t in task["test"]]
    async with live:
        with _sandbox(args, task, {"candidates": []}, "refine") as sb:
            checks = await verify_reports(sb, reports, n_test)
            cands = build_candidates(reports, checks, n_test)
            cand_ns = [{k: v["grid"] for k, v in c.items()} for c in cands]
            out, _ = await sb_run(sb, f"candidates = {json.dumps(cand_ns)}\nprint('ok')")
            if "ok" not in out:
                raise RuntimeError(f"failed to set candidates in sandbox: {out[-300:]}")
            names = [c["name"] for c in checks if c["defined"]]
            system = prompts.refiner_system(n_test, args.compact_gen_budget if args.compact else args.ctx_cap,
                                            args.max_turn_tokens, args.tool_timeout,
                                            args.tool_max_output_chars, names, variant=args.prompt_variant,
                                            notes_structured=args.notes_structured)
            user = prompts.render_refiner_user(task["train"], test_inputs, reports, checks, cands, args.inline_cells,
                                               variant=args.prompt_variant)
            ep = await run_episode(client, args, label=f"refiner round {round_no}", system=system, user=user, task_id=task["id"],
                                   sb=sb, tool=args.tool, tool_name=args.tool_name, n_test=n_test)
            ep.notes = await read_notes(sb, args.tool_max_output_chars)   # the refiner's notes were not read back before (audit: 0/4,499)
    return ep, checks, cands


async def run_verify(args, task, reports: list[dict], live: asyncio.Semaphore):
    """--skip-refine-when-agree: open the refine sandbox only to run verify_reports + build_candidates, without the
    refiner LLM. Returns (checks, candidates), same format as the last two items of run_refine, so summarize()'s
    rec["checks"][0] has content."""
    n_test = len(task["test"])
    async with live:
        with _sandbox(args, task, {"candidates": []}, "refine") as sb:
            checks = await verify_reports(sb, reports, n_test)
    return checks, build_candidates(reports, checks, n_test)


def _report_for_refiner(ep: Episode) -> dict:
    rep = dict(ep.report)
    rep["label"] = ep.label
    rep["stop_reason"] = ep.stop_reason
    rep["answers"] = ep.answers
    rep["notes"] = ep.notes
    rep["home_check"] = ep.home_check
    rep["alt_check"] = ep.alt_check
    rep["notes_structured"] = ep.notes_structured
    return rep


async def solve_task(client, args, task, live: asyncio.Semaphore) -> dict:
    t0 = time.monotonic()
    n_test = len(task["test"])
    finals, explorers = await run_explorers(client, args, task, live)
    reports = [_report_for_refiner(ep) for ep in finals]
    # First assemble candidates from the explorers' own-sandbox checks and write them into the submission: the refiner
    # still has to run one more ≤ cap conversation, so if it is cut at the deadline this cell already has something
    # (highest votes first). The refiner overwrites it when it finishes.
    home_checks = _checks_from_home(reports)
    home_cands = build_candidates(reports, home_checks, n_test)
    if args.submission is not None:
        for t in range(n_test):
            args.submission.put_attempts(task["id"], t, [c["grid"] for c in home_cands[t].values()])
    refines: list[Episode] = []
    all_checks, all_cands = [], []
    final_attempts: list[list] = [[] for _ in range(n_test)]
    final_src: list[list[str]] = [[] for _ in range(n_test)]
    # If both explorers answered on their own and agree on the first answer of every test, skip the refiner: on the
    # teacher, "agree but wrong" was v4 0/56, v5 7/72, while the refiner is a whole ≤ cap conversation (sweep measured
    # 36–55k). The cheapest cut when the budget is tight; attempt 2 is filled from candidates.
    refine_skipped = ""
    agreed = _agreed_answer(finals, n_test) if args.skip_refine_when_agree else None
    if agreed is not None:
        refine_skipped = "agree"
        print(f"·· {task['id']}: explorers agree -> refiner skipped", file=sys.stderr)
        # The refiner LLM does not run, but the refine sandbox still opens for verify/candidates: only then is there a vote-ordered candidate to fill attempt 2.
        checks, cands = await run_verify(args, task, reports, live)
        all_checks.append(checks)
        all_cands.append(cands)
    for rnd in range(1, args.max_refine_rounds + 1):
        if refine_skipped:
            break
        cut = _cutoff(args)
        if cut is not None:
            refine_skipped = cut                     # deadline / abort: no refiner; the answer is the candidate set above
            break
        # The task budget includes the explorers and earlier refine rounds; round 1 always runs (the attempts selection
        # below needs at least one round), after that no escalation when it does not fit. The budget shrinks to the
        # remainder but not below min_session_budget (K3 measured: after 2×75k the refiner gets max(min(40k, remainder), 20k)).
        left = args.task_budget - sum(e.usage("completion_tokens") for e in explorers + refines)
        if rnd > 1 and left < args.min_session_budget:
            break
        gb = max(min(args.refine_gen_budget or args.compact_gen_budget, left), args.min_session_budget)
        ep, checks, cands = await run_refine(client, _with_gen_budget(args, gb), task, reports, rnd, live)
        refines.append(ep)
        all_checks.append(checks)
        all_cands.append(cands)
        complete = all(len(ep.answers[t]) >= 1 for t in range(n_test))
        conf = ep.report["confidence"]
        # v2's refiner confidence is well calibrated (≥80: 76/80 right; <50: 2/17); if it did not finish, was forced to
        # hand off, or confidence is low, open another round and merge its report into reports. Before v2 a second round
        # opened only on "did not finish", which expression ANSWERs made almost never happen (rounds/task 1.00).
        weak = (not complete) or ep.forced_handoff or conf is None or conf < args.escalate_below
        if not weak or rnd == args.max_refine_rounds:
            break
        reports.append(_report_for_refiner(ep))
    if refines:
        # Final attempts: ATTEMPT 1 of the most confident round (ties go to the later one) first; another round's ATTEMPT 1,
        # if different, becomes the 2nd -- two rounds are two independent judgements, and such a disagreement deserves a
        # slot more than an ATTEMPT 2 written by the same chain; the best round's ATTEMPT 2 comes after that.
        order = sorted(range(len(refines)), key=lambda i: ((refines[i].report["confidence"] or -1), i), reverse=True)
        for t in range(n_test):
            ranked = [(refines[i].answers[t][0], f"refine{i + 1}") for i in order if refines[i].answers[t]]
            best = refines[order[0]].answers[t]
            if len(best) > 1:
                ranked.append((best[1], f"refine{order[0] + 1}:a2"))
            for g, src in ranked:
                if g not in final_attempts[t] and len(final_attempts[t]) < 2:
                    final_attempts[t].append(g)
                    final_src[t].append(src)
    elif agreed is not None:
        # The agreed answer is attempt 1; attempt 2 is filled from the candidates below (second-highest votes).
        for t in range(n_test):
            final_attempts[t].append(agreed[t])
            final_src[t].append("agreed")
    # Everything else (deadline / abort with no refiner): relies entirely on the candidate fill below.
    # Candidate fill: tests the refiner did not answer get the top-voted candidate; --fill-attempt2 fills the second.
    # If the refiner never ran a round (deadline), only the candidates from the explorers' own checks exist.
    cands = all_cands[-1] if all_cands else home_cands
    for t in range(n_test):
        want = 2 if (args.fill_attempt2 or not final_attempts[t]) else len(final_attempts[t])
        for key, c in cands[t].items():
            if len(final_attempts[t]) >= want:
                break
            if c["grid"] not in final_attempts[t]:
                final_attempts[t].append(c["grid"])
                final_src[t].append(f"candidate:{key}")
    # The competition's hidden test has no output: all scoring fields are None (not False -- summarize would count False as wrong in the denominator).
    scored = all("output" in t for t in task["test"])
    expected = [t.get("output") for t in task["test"]]
    per_test = []
    for t in range(n_test):
        att = final_attempts[t]
        if args.submission is not None:
            args.submission.put_attempts(task["id"], t, att)
        row = {"test_idx": t, "attempts": att, "attempt_src": final_src[t], "n_candidates": len(cands[t])}
        if scored:
            row.update({
                "correct": any(g == expected[t] for g in att),
                "correct_attempt1": bool(att) and att[0] == expected[t],
                "correct_refine_only": any(g == expected[t] for g, s in zip(att, final_src[t]) if not s.startswith("candidate")),
                "correct_any_explorer": any(g == expected[t] for ep in explorers for g in ep.answers[t]),
                "correct_any_candidate": any(c["grid"] == expected[t] for c in cands[t].values()),
            })
        else:
            row.update({k: None for k in ("correct", "correct_attempt1", "correct_refine_only",
                                          "correct_any_explorer", "correct_any_candidate")})
        per_test.append(row)
    eps = explorers + refines
    rec = {
        "task_id": task["id"], "n_test": n_test, "n_train": len(task["train"]), "dp_rank": args.dp_rank,
        "refine_skipped": refine_skipped,
        "explorers": [ep.to_dict(args.save_messages) for ep in explorers],
        "refines": [ep.to_dict(args.save_messages) for ep in refines],
        "checks": all_checks or [home_checks],
        "candidates": [[{k: {"sources": v["sources"], "confs": v["confs"], "shape": [len(v["grid"]), len(v["grid"][0])]}
                         for k, v in c.items()} for c in cands_r] for cands_r in (all_cands or [home_cands])],
        "results": per_test,
        "tok_prompt": sum(e.usage("prompt_tokens") for e in eps),
        "tok_completion": sum(e.usage("completion_tokens") for e in eps),
        "tok_reasoning": sum(e.usage("reasoning_tokens") for e in eps),
        "tok_cached": sum(e.usage("cached_tokens") for e in eps),
        "peak_total": max(e.peak_total for e in eps),
        "n_turns": sum(len(e.calls) for e in eps),
        "wall_s": round(time.monotonic() - t0, 1),
    }
    return rec


def write_task(out: Path, rec: dict):
    (out / "tasks").mkdir(exist_ok=True)
    (out / "pairs").mkdir(exist_ok=True)
    # --save-segments: segments are large (each is the full messages the model saw at the time), so they stay out of
    # tasks/*.json (summarize / compare read those in bulk) and go to segments/<task>.jsonl.gz, one segment per line.
    segs = []
    for kind in ("explorers", "refines"):
        for ep in rec[kind]:
            eps = ep.pop("segments", None)
            if eps:
                for i, m in enumerate(eps):
                    segs.append({"label": ep["label"], "lineage": ep["lineage"], "hop": ep["hop"],
                                 "segment_idx": i, "n_segments": len(eps), "stop_reason": ep["stop_reason"], "messages": m})
    if segs:
        (out / "segments").mkdir(exist_ok=True)
        tmp = out / "segments" / f"{rec['task_id']}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
            for sg in segs:
                f.write(json.dumps(sg, ensure_ascii=False) + "\n")
        tmp.rename(out / "segments" / f"{rec['task_id']}.jsonl.gz")
    tmp = out / "tasks" / f"{rec['task_id']}.json.tmp"
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.rename(out / "tasks" / f"{rec['task_id']}.json")
    n = rec["n_test"]
    for r in rec["results"]:
        pair = {
            "uid": f"{rec['task_id']}#{r['test_idx']}", "task_id": rec["task_id"], "test_idx": r["test_idx"],
            **{k: r[k] for k in ("attempts", "attempt_src", "correct", "correct_attempt1", "correct_refine_only",
                                 "correct_any_explorer", "correct_any_candidate", "n_candidates")},
            # Token counts split evenly across tests, so the sum in score_partial.py equals the whole task.
            **{k: rec[k] / n for k in ("tok_prompt", "tok_completion", "tok_reasoning", "tok_cached")},
            "n_turns": rec["n_turns"] / n,
        }
        (out / "pairs" / f"{rec['task_id']}_{r['test_idx']}.json").write_text(json.dumps(pair))


def _fmt_task_line(rec: dict) -> str:
    def _hop(e):
        conf = e["report"]["confidence"]
        return f"{e['stop_reason'].split(':')[0][:6]} c={conf if conf is not None else '-'} {e['tok_completion'] // 1000}k"
    by_lin: dict[int, list] = {}
    for e in rec["explorers"]:
        by_lin.setdefault(e["lineage"], []).append(e)
    ex = " ".join(f"E{k}[" + " > ".join(_hop(e) for e in eps) + "]" for k, eps in sorted(by_lin.items()))
    rf = " ".join(f"R{i + 1}[{e['stop_reason'].split(':')[0][:6]} {e['tok_completion'] // 1000}k]" for i, e in enumerate(rec["refines"]))
    if rec.get("refine_skipped"):
        rf = "R[agreed]" if rec["refine_skipped"] == "agree" else f"R[skipped:{rec['refine_skipped']}]"
    res = "".join("?" if r["correct"] is None else "✓" if r["correct"] else "✗" for r in rec["results"])
    expl = "".join("?" if r["correct_any_explorer"] is None else "✓" if r["correct_any_explorer"] else "✗"
                   for r in rec["results"])
    return (f"{rec['task_id']} n_test={rec['n_test']} {res} (explorers {expl}) {ex} {rf} "
            f"gen={rec['tok_completion'] // 1000}k peak={rec['peak_total'] // 1000}k {rec['wall_s']:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(args).items()
                                              if k not in ("tool", "sampling", "submission", "should_abort",
                                                           "chat", "count_tokens", "annotate_ids")},
                                             indent=1, default=str))
    if args.task_source == "nvarc":
        # Several corpora are allowed (nvarc_training + nvarc_full run mixed); each task records source = file name minus .jsonl.zst, written to tasks/*.json.
        corpus: dict[str, list] = {}
        source: dict[str, str] = {}
        for path in args.corpus:
            name = Path(path).name.split(".")[0]
            part = load_nvarc_corpus(path)
            dup = set(part) & set(corpus)
            assert not dup, f"{len(dup)} uids in more than one corpus, e.g. {sorted(dup)[:3]}"
            corpus.update(part)
            source.update({u: name for u in part})
            print(f"nvarc corpus {name}: {len(part)} uids", flush=True)
        task_ids = ([l.strip() for l in Path(args.task_list).read_text().splitlines() if l.strip()]
                    if args.task_list else list(corpus))
        missing = [t for t in task_ids if t not in corpus]
        assert not missing, f"{len(missing)} uids not in corpus, e.g. {missing[:3]}"
        get_task = lambda tid: dict(nvarc_task(tid, corpus[tid], args.attempt, args.n_train, args.n_test), source=source[tid])
        print(f"nvarc: {len(corpus)} uids total; attempt={args.attempt} n_train={args.n_train} n_test={args.n_test}", flush=True)
    elif args.task_source == "kaggle":
        challenges = load_challenges(Path(args.challenges))
        task_ids = ([l.strip() for l in Path(args.task_list).read_text().splitlines() if l.strip()]
                    if args.task_list else list(challenges))
        missing = [t for t in task_ids if t not in challenges]
        assert not missing, f"{len(missing)} task ids not in {args.challenges}, e.g. {missing[:3]}"
        get_task = lambda tid: {"id": tid, "train": challenges[tid]["train"], "test": challenges[tid]["test"]}
        n_truth = sum(all("output" in t for t in challenges[u]["test"]) for u in task_ids)
        print(f"kaggle: {len(challenges)} tasks in {args.challenges}; {n_truth}/{len(task_ids)} with ground truth",
              flush=True)
    else:
        task_ids = [l.strip() for l in Path(args.task_list).read_text().splitlines() if l.strip()]
        get_task = lambda tid: load_task(Path(args.task_dir), tid)
    if args.tasks:
        want = set(args.tasks.split(","))
        task_ids = [t for t in task_ids if t in want]
    if args.limit:
        task_ids = task_ids[: args.limit]
    done = {p.stem for p in (out / "tasks").glob("*.json")} if args.resume else set()
    todo = [t for t in task_ids if t not in done]
    print(f"{len(task_ids)} tasks, {len(done)} already done, {len(todo)} to run; "
          f"explorers={args.n_explorers} ctx_cap={args.ctx_cap} live={args.max_live}", flush=True)
    if args.submission_out:
        # Note: --limit / --tasks already truncated the task list before this, so submission.json only has keys for these tasks.
        args.submission = SubmissionStore({tid: {"test": get_task(tid)["test"]} for tid in task_ids},
                                          Path(args.submission_out))
        for tid in sorted(done):
            rec = json.loads((out / "tasks" / f"{tid}.json").read_text())
            for r in rec["results"]:
                args.submission.put_attempts(tid, r["test_idx"], r["attempts"])
        print(f"submission -> {args.submission_out}: {args.submission.n_cells} cells, "
              f"{len(args.submission.filled)} restored from {len(done)} done tasks", flush=True)
    if args.deadline is not None:
        print(f"deadline: {args.deadline_min:.1f} min from start "
              f"(per-turn max_tokens shrinks at {args.deadline_tok_per_s:g} tok/s near the end)", flush=True)
    # When the server dies every task ends with ConnectError; without this guard the run would quietly leave the submission at fallback and exit normally.
    health = {"ok": 0, "errors": 0, "tripped": ""}
    args.should_abort = lambda: bool(health["tripped"])
    dp_load = collections.Counter()
    skipped: list[str] = []

    live = asyncio.Semaphore(args.max_live)
    limits = httpx.Limits(max_connections=args.max_live + 64, max_keepalive_connections=args.max_live + 64)
    t_start = time.monotonic()
    n_done = 0
    async with httpx.AsyncClient(timeout=args.http_timeout, limits=limits) as client:
        # When all 120 tasks go into gather at once, the semaphore is FIFO: 240 base explorers queue ahead of every
        # refiner, and tasks only start finishing once all of them have started (compact-mode sessions are long, so
        # the first task takes 3 hours to appear). --task-live caps the number of tasks running at once, so completion
        # progresses smoothly and --resume loses less.
        task_live = asyncio.Semaphore(args.task_live) if args.task_live > 0 else contextlib.nullcontext()

        def _check_health():
            total = health["ok"] + health["errors"]
            rate = health["errors"] / max(total, 1)
            if (args.max_error_rate > 0 and not health["tripped"] and total >= args.error_grace
                    and rate > args.max_error_rate):
                health["tripped"] = (f"{health['errors']}/{total} tasks failed "
                                     f"({100 * rate:.1f}% > {100 * args.max_error_rate:.1f}%)")
                print(f"\n{'=' * 70}\nABORT: {health['tripped']}\n  The server is most likely dead. No new tasks are dispatched, "
                      f"and conversations in flight make no further model calls.\n{'=' * 70}\n", flush=True)

        async def one(tid):
            nonlocal n_done
            task = get_task(tid)
            targs, hp = copy.copy(args), None    # shallow copy: chat / count_tokens / submission etc. are shared; only per-task things are overridden
            if args.hparam_menu:
                hp = draw_hparams(tid, args.attempt)
                vars(targs).update(hp)
            rank = None
            try:
                async with task_live:
                    if _cutoff(args) is not None:
                        skipped.append(tid)      # deadline reached / aborted: tasks not started stay unstarted, submission keeps the fallback
                        return None
                    if args.fixed_dp_rank is not None:
                        # Single-rank data collection: pin everything even if the server is DP2, leaving the other pool idle.
                        rank = args.fixed_dp_rank
                        dp_load[rank] += 1
                    elif args.dp_size > 1:
                        # Pick a rank only after getting a slot: the one with the fewest tasks in flight (ties go to the lower); explorer/refiner follow the task.
                        rank = min(range(args.dp_size), key=lambda r: (dp_load[r], r))
                        dp_load[rank] += 1
                    targs.dp_rank = rank
                    rec = await solve_task(client, targs, task, live)
            except Exception as e:  # one task blowing up must not take down the other 119, but must be logged loudly
                n_done += 1
                health["errors"] += 1
                _check_health()
                print(f"!! {tid} FAILED: {type(e).__name__}: {e}", flush=True)
                (out / "failures.jsonl").open("a").write(json.dumps({"task_id": tid, "error": repr(e)}) + "\n")
                raise
            finally:
                if rank is not None:
                    dp_load[rank] -= 1
            health["ok"] += 1
            rec["attempt"] = args.attempt
            rec["hparams"] = hp
            if "train_idx" in task:
                rec["train_idx"], rec["test_idx"] = task["train_idx"], task["test_idx"]
            rec["source"] = task.get("source", "arc")
            write_task(out, rec)
            n_done += 1
            print(f"[{n_done}/{len(todo)} {time.monotonic() - t_start:6.0f}s] " + _fmt_task_line(rec), flush=True)

        results = await asyncio.gather(*(one(t) for t in todo), return_exceptions=True)
    n_fail = sum(1 for r in results if isinstance(r, Exception))
    summarize(out)
    if skipped:
        print(f"{len(skipped)} tasks never started (deadline / abort): {skipped[:8]}{' ...' if len(skipped) > 8 else ''}",
              flush=True)
    if args.submission is not None:
        sub = args.submission
        print(f"submission: {len(sub.filled)} / {sub.n_cells * 2} attempt cells filled (rest = test input fallback) "
              f"-> {sub.out}", flush=True)
        if sub.rejected:
            print(f"  ⚠️  {len(sub.rejected)} answers were not valid grids and were NOT written: {sub.rejected[:5]}",
                  flush=True)
    if health["tripped"]:
        # Deliberately not treated as "finished": most cells in the submission are fallback, and submitting it would count as a 0 from the model.
        print(f"\nFAILED: {health['tripped']}", flush=True)
        sys.exit(2)
    if n_fail:
        print(f"!! {n_fail} tasks failed; see failures.jsonl", flush=True)
        if args.task_source != "kaggle":
            sys.exit(1)
        # kaggle: individual task failures must not make the notebook exit non-zero (a code-competition notebook that
        # errors = the whole submission is void); failed cells keep the fallback; only an over-limit error rate (above) aborts.


def summarize(out: Path):
    recs = [json.loads(p.read_text()) for p in sorted((out / "tasks").glob("*.json"))]
    if not recs:
        print("no tasks finished")
        return
    n = len(recs)
    pairs = [r for rec in recs for r in rec["results"]]
    # In kaggle mode (hidden tests have no answers) correct is None: those tasks stay out of the score denominator, and with none at all no score is printed.
    scored = [rec for rec in recs if all(r["correct"] is not None for r in rec["results"])]

    def partial(key):
        return sum(sum(1 for r in rec["results"] if r[key]) / rec["n_test"] for rec in scored) / len(scored) * 100

    def strict(key):
        return sum(all(r[key] for r in rec["results"]) for rec in scored) / len(scored) * 100

    lines = [f"== {out.name}: {n} tasks, {len(pairs)} test pairs =="]
    if scored:
        lines += [f"partial pass@2 (final attempts)     {partial('correct'):6.2f}%   strict {strict('correct'):6.2f}%"
                  + (f"   (scored {len(scored)}/{n} tasks)" if len(scored) < n else ""),
                  f"partial pass@1 (attempt 1 only)      {partial('correct_attempt1'):6.2f}%",
                  f"partial refine-only (no candidate fill) {partial('correct_refine_only'):6.2f}%",
                  f"partial any-explorer-answer          {partial('correct_any_explorer'):6.2f}%   (oracle over explorer answers)",
                  f"partial any-candidate                {partial('correct_any_candidate'):6.2f}%   (oracle over candidates)"]
    else:
        lines.append("no ground truth (kaggle hidden test): pass@k not computed")
    lines += [f"gen tokens/task: mean {sum(r['tok_completion'] for r in recs) / n / 1000:.1f}k   "
             f"peak context max {max(r['peak_total'] for r in recs) / 1000:.1f}k   "
             f"cache hit {100 * sum(r['tok_cached'] for r in recs) / max(1, sum(r['tok_prompt'] for r in recs)):.1f}%   "
             f"wall/task mean {sum(r['wall_s'] for r in recs) / n:.0f}s"]
    if any(rec.get("dp_rank") is not None for rec in recs):
        lines.append(f"dp_rank distribution: {dict(collections.Counter(rec.get('dp_rank') for rec in recs))}")
    stops = {}
    for rec in recs:
        for e in rec["explorers"]:
            stops[e["stop_reason"]] = stops.get(e["stop_reason"], 0) + 1
    rstops = {}
    for rec in recs:
        for e in rec["refines"]:
            rstops[e["stop_reason"]] = rstops.get(e["stop_reason"], 0) + 1
    n_lin = sum(len({e["lineage"] for e in rec["explorers"]}) for rec in recs)
    lines.append(f"explorer stops: {stops}   sessions/explorer mean "
                 f"{sum(len(rec['explorers']) for rec in recs) / n_lin:.2f}")
    lines.append(f"refiner stops:  {rstops}   refine rounds/task mean {sum(len(r['refines']) for r in recs) / n:.2f}"
                 f"   skipped (explorers agree): {sum(1 for r in recs if r.get('refine_skipped') == 'agree')}")
    eps = [e for rec in recs for e in rec["explorers"] + rec["refines"]]
    if json.loads((out / "args.json").read_text()).get("compact"):
        lines.append(f"compactions: {sum(e['n_compactions'] for e in eps)} over {len(eps)} episodes   "
                     f"episodes compacted: {sum(1 for e in eps if e['n_compactions'])}   "
                     f"dropped tokens/episode: {sum(e['compact_dropped_tokens'] for e in eps) / max(len(eps), 1) / 1000:.1f}k   "
                     f"checkpoints: {sum(e['n_checkpoints'] for e in eps)}   "
                     f"outputs cleared: {sum(e['n_outputs_cleared'] for e in eps)}")
    if json.loads((out / "args.json").read_text()).get("delegate"):
        subs = [d for e in eps for d in e["sub_episodes"]]
        lines.append(f"delegations: {sum(e['n_delegations'] for e in eps)} over {len(eps)} episodes   "
                     f"episodes delegating: {sum(1 for e in eps if e['n_delegations'])}   "
                     f"helper gen/delegation: {sum(d['tok_completion'] for d in subs) / max(len(subs), 1) / 1000:.1f}k   "
                     f"helper stops: {collections.Counter(d['stop_reason'] for d in subs)}")
    lines.append(f"continuations: {sum(e['n_continuations'] for e in eps)} over {len(eps)} episodes   "
                 f"code passes all train pairs: explorers "
                 f"{sum(1 for rec in recs for c in rec['checks'][0] if c['train'] and all(x == 'pass' for x in c['train']))}"
                 f"/{sum(len(rec['checks'][0]) for rec in recs)}   "
                 f"sandbox restarts: {sum(e.get('n_sandbox_restarts', 0) for e in eps)}")
    if json.loads((out / "args.json").read_text())["prompt_variant"] == "alt":
        lines.append(_alt_summary(recs))
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")


def _alt_summary(recs: list[dict]) -> str:
    """Alt-variant-specific metrics: number of explorer sessions that wrote ALTERNATIVE, number whose `transform_alt`
    passes all train pairs, and the source of the refiner's ATTEMPT 2 (transform_alt expression / copy of an existing
    candidate / self-made grid; in v8 the 36 self-made ones were right only once)."""
    ex = [e for rec in recs for e in rec["explorers"]]
    n_alt = sum(1 for e in ex if e["report"]["alt_code"])   # only those with code count: the model often writes "none (because...)" to explain why there is none
    n_alt_pass = sum(1 for e in ex if _all_train_pass(e["alt_check"]))
    src = {"transform_alt": 0, "copy": 0, "novel": 0}
    for rec in recs:
        finals = {e["lineage"]: e for e in rec["explorers"]}   # sessions are stored in hop order; later ones overwrite earlier ones
        for r in rec["results"]:
            t = r["test_idx"]
            grids = [g for e in finals.values() for g in e["answers"][t]]
            for e in finals.values():
                if e["home_check"]["defined"] and e["home_check"]["test"][t]["grid"] is not None:
                    grids.append(e["home_check"]["test"][t]["grid"])
                if _all_train_pass(e["alt_check"]) and e["alt_check"]["test"][t]["grid"] is not None:
                    grids.append(e["alt_check"]["test"][t]["grid"])
            for i, s in enumerate(r["attempt_src"]):
                m = re.fullmatch(r"refine(\d+):a2", s)
                if not m:
                    continue
                kind, body = rec["refines"][int(m.group(1)) - 1]["report"]["answer_blocks"][f"{t}:1"]
                src["transform_alt" if kind == "expr" and "transform_alt" in body else
                    "copy" if r["attempts"][i] in grids else "novel"] += 1
    return (f"alt variant: explorer sessions with ALTERNATIVE {n_alt}/{len(ex)}, transform_alt passes all train pairs "
            f"{n_alt_pass}; refiner attempt-2 source {src}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-source", choices=["arc", "nvarc", "kaggle"], default="arc",
                    help="arc: --task-dir/<id>.json; nvarc: jsonl.zst from --corpus, train/test split by choose_pairs")
    ap.add_argument("--task-dir", default="")
    ap.add_argument("--challenges", default="",
                    help="--task-source kaggle: the competition's arc-agi_test_challenges.json, or a directory of <task_id>.json "
                         "(local dry run; scored when tests carry outputs)")
    ap.add_argument("--submission-out", default="",
                    help="write ARC Prize submission.json while running: created pre-filled with the test inputs, explorers "
                         "write candidates when they finish, the refiner overwrites when it finishes; the whole file is rewritten each time")
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="minutes from now. At the deadline no more model calls (including handoff reports); explorer answers "
                         "are collected from the sandbox's existing transform, the refiner does not open; tasks not started are not started. 0 = no limit")
    ap.add_argument("--deadline-tok-per-s", type=float, default=30.0,
                    help="near the deadline, per-turn max_tokens cap = remaining seconds × this value (conservative per-stream decode rate; "
                         "Kaggle L4 at 4 streams per replica is 60–90 tok/s per stream, estimated ~30 at 12)")
    ap.add_argument("--dp-size", type=int, default=1,
                    help="the server's --dp-size; >1 pins all episodes of a task to one routed_dp_rank (sglang's DP has no "
                         "session affinity; without pinning every turn re-prefills)")
    ap.add_argument("--dp-rank", type=int, default=None, dest="fixed_dp_rank",
                    help="pin all tasks to this routed_dp_rank. Only when unset and --dp-size>1 are tasks assigned by fewest in flight. "
                         "Required for DP attention even when using a single rank, otherwise round-robin crosses ranks and the prefix cache is empty")
    ap.add_argument("--refine-gen-budget", type=int, default=0,
                    help="refiner generation budget in compact mode; 0 = same as --compact-gen-budget")
    ap.add_argument("--max-error-rate", type=float, default=0.0,
                    help=">0: once the fraction of failed tasks exceeds this, stop dispatching new tasks and exit with non-zero code (2). Guard against a dead server")
    ap.add_argument("--error-grace", type=int, default=10, help="complete at least this many tasks before checking the error rate")
    ap.add_argument("--task-list", default="", help="one task id / uid per line; for nvarc, omitted = the whole corpus")
    ap.add_argument("--corpus", nargs="*", default=[], help="jsonl.zst for --task-source nvarc, several allowed (uids must not repeat); each task records source")
    ap.add_argument("--attempt", type=int, default=0, help="nvarc pair-sampling seed (same as rollout_runner's attempt)")
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=1)
    ap.add_argument("--hparam-menu", action="store_true",
                    help="draw effort / ctx_cap / K / target_frac / gen_budget per task deterministically from HPARAM_MENU, recorded in tasks/*.json")
    ap.add_argument("--save-segments", action="store_true",
                    help="before each compaction save the full messages to segments/<task>.jsonl.gz (distillation corpus)")
    ap.add_argument("--tasks", default="", help="comma-separated subset of task ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip tasks already in out/tasks")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", default="xhigh")
    ap.add_argument("--backend", choices=["openai", "dsv4", "tokens"], default="openai",
                    help="dsv4: build the prompt with the model's own encoding and call /generate (see dsv4_backend.py)")
    ap.add_argument("--encoding-path", default=None, help="path to encoding_dsv4.py for the dsv4 backend")
    ap.add_argument("--prompt-variant", choices=list(prompts.VARIANTS), default="default",
                    help="alt: after passing all train pairs the explorer takes one extra step looking for a counter-hypothesis (ALTERNATIVE / ATTEMPT 2), and the refiner "
                         "deliberately looks for counterexamples when all candidates are train-pass; default prompts are unchanged (see prompts.py)")
    ap.add_argument("--n-explorers", type=int, default=3)
    ap.add_argument("--base-explorers", type=int, default=2,
                    help="run this many explorers first; if two of their answers already agree, do not open the rest (set to n-explorers for all in parallel)")
    ap.add_argument("--task-budget", type=int, default=10**9,
                    help="total gen tokens for this task (explorer + refine): once reached no new session / lineage opens, and each new session's "
                         "budget shrinks to the remainder (split evenly among sessions of the same parallel batch). v7/v8 simulation: saves 49k but kills the only correct answer on 11 tasks, not adopted; "
                         "reap128 final 120 tasks: the 41 unsolved average 865k, solved 293k; Kaggle uses 180k")
    ap.add_argument("--lineage-budget", type=int, default=0,
                    help="cumulative generated-token cap per explorer across sessions (0 = unlimited): each session's compact_gen_budget = "
                         "min(original, remainder); running sessions are not cut (may exceed by one max_turn_tokens turn + handoff)")
    ap.add_argument("--min-session-budget", type=int, default=20_000,
                    help="do not open a new session when the lineage / task budget left is below this; refine round 1 gets at least this much")
    ap.add_argument("--pass-gate-hop", type=int, default=99,
                    help="from this session on, only explorers whose home check passes all train pairs may continue (v9 uses 4)")
    ap.add_argument("--max-hops", type=int, default=3,
                    help="max sessions per explorer (after a handoff, a new context continues on the same sandbox)")
    ap.add_argument("--max-refine-rounds", type=int, default=1)
    ap.add_argument("--escalate-below", type=int, default=80,
                    help="open another refine round when refiner confidence is below this (or it did not finish / was forced to hand off)")
    ap.add_argument("--ctx-cap", type=int, default=50_000, help="context cap per conversation (prompt+completion)")
    ap.add_argument("--report-reserve", type=int, default=10_000, help="context reserved for the handoff (checkpoint tool call + report)")
    ap.add_argument("--context-margin", type=int, default=2_500, help="margin reserved for tool output and the template")
    ap.add_argument("--handoff-margin", type=int, default=300)
    ap.add_argument("--min-handoff-tokens", type=int, default=3_500, help="the handoff report gets at least this many max_tokens (may slightly exceed the cap)")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path, for estimating the first turn's prompt length")
    ap.add_argument("--template-overhead", type=int, default=600, help="estimated fixed token overhead of chat template + tool schema")
    ap.add_argument("--min-turn-tokens", type=int, default=4_000,
                    help="when the remainder is below this, hand off directly instead of opening a small turn that would only be truncated")
    ap.add_argument("--max-turn-tokens", type=int, default=20_000,
                    help="max_tokens of a single reply; hitting it asks for a checkpoint and continues (not a handoff)")
    ap.add_argument("--max-continuations", type=int, default=4, help="max continuations per conversation")
    ap.add_argument("--compact", action="store_true",
                    help="no handoff when the context is full: fold the oldest tool round and continue the same conversation (sandbox state untouched), "
                         "writing the report only at --compact-gen-budget or --max-turns. Use with --max-hops 1")
    ap.add_argument("--compact-keep-tokens", type=int, default=8_000, help="after folding, keep at least this many estimated tokens of recent rounds (floor)")
    ap.add_argument("--delegate", action="store_true",
                    help="give explorers an extra `delegate` tool: hand a subproblem to a helper with a clean context (same sandbox) and get back only a short conclusion")
    ap.add_argument("--delegate-budget", type=int, default=60_000, help="generated-token budget per helper call")
    ap.add_argument("--delegate-max", type=int, default=6, help="max delegate calls per conversation")
    ap.add_argument("--delegate-max-turns", type=int, default=80)
    ap.add_argument("--delegate-reply-chars", type=int, default=6000, help="character cap of the helper reply put back into the explorer context")
    ap.add_argument("--notes-structured", action="store_true",
                    help="each notes entry carries a RULE:/VERIFIED:/REFUTED:/TODO:/CODE: tag; the harness groups, dedups, and keeps only the newest RULE when rendering")
    ap.add_argument("--compact-rich", action="store_true",
                    help="compact mode: first clear old rounds' tool outputs (keeping reasoning/code), then drop whole rounds; the notice carries the `transform` source and a one-line summary per folded round")
    ap.add_argument("--compact-keep-full-rounds", type=int, default=2, help="--compact-rich: tool outputs of this many recent rounds stay intact")
    ap.add_argument("--compact-checkpoint", action="store_true",
                    help="compact mode: before folding old rounds, give the model one turn to append the reasoning about to disappear into `notes`")
    ap.add_argument("--compact-checkpoint-tokens", type=int, default=3000, help="max_tokens of that checkpoint turn")
    ap.add_argument("--compact-target-frac", type=float, default=0.6,
                    help="target fraction of the working window (ctx_cap − report_reserve − margin) the context occupies after folding")
    ap.add_argument("--compact-gen-budget", type=int, default=200_000, help="total generated-token budget per conversation in compact mode")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--handoff-thinking", choices=["off", "low", "same"], default="same")
    ap.add_argument("--handoff-tool-calls", type=int, default=2, help="max extra tool calls allowed for a checkpoint during handoff")
    ap.add_argument("--handoff-turn-tokens", type=int, default=4_000, help="max_tokens of a checkpoint tool-call turn during handoff")
    ap.add_argument("--gen-budget-hard", action="store_true",
                    help="compact mode: per-turn max_tokens may not exceed the remaining generation budget (at least min_turn_tokens); overshoot cap drops from 20k to 4k")
    ap.add_argument("--skip-refine-when-agree", action="store_true",
                    help="skip the refiner when both explorers finished with the same answer; attempt 2 filled from candidates (saves a whole conversation)")
    ap.add_argument("--handoff-budget", type=int, default=0,
                    help="generation cap for the whole handoff phase (checkpoint tool calls + report + refusal retries); 0 = no limit (original behaviour). "
                         "Kaggle uses 10k: a model that does not write the report can generate up to the ctx remainder each turn, turning a 20k budget into 58k")
    ap.add_argument("--no-fill-attempt2", dest="fill_attempt2", action="store_false",
                    help="when refine gives only 1 attempt, do not fill the 2nd from candidates")
    ap.add_argument("--inline-cells", type=int, default=400, help="candidate grids smaller than this many cells are printed directly in the refine prompt")
    ap.add_argument("--max-live", type=int, default=64, help="conversations in flight at once (incl. sandboxes)")
    ap.add_argument("--task-live", type=int, default=0, help="cap on tasks running at once (0 = unlimited; max_live/2 recommended)")
    ap.add_argument("--http-timeout", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sandbox-python", default=sys.executable)
    ap.add_argument("--no-save-messages", dest="save_messages", action="store_false")
    args = ap.parse_args()
    if args.task_source == "nvarc":
        if not args.corpus:
            raise SystemExit("--task-source nvarc requires --corpus")
    elif args.task_source == "kaggle":
        if not args.challenges:
            raise SystemExit("--task-source kaggle requires --challenges")
    elif not (args.task_dir and args.task_list):
        raise SystemExit("--task-source arc requires --task-dir and --task-list")
    if args.dp_size < 1:
        raise SystemExit(f"--dp-size must be >= 1, got {args.dp_size}")
    if args.fixed_dp_rank is not None:
        if args.fixed_dp_rank < 0:
            raise SystemExit(f"--dp-rank must be >= 0, got {args.fixed_dp_rank}")
        if args.dp_size > 1 and args.fixed_dp_rank >= args.dp_size:
            raise SystemExit(
                f"--dp-rank {args.fixed_dp_rank} out of range for --dp-size {args.dp_size}"
            )
    if (args.lineage_budget or args.refine_gen_budget or args.gen_budget_hard) and not args.compact:
        raise SystemExit("--lineage-budget / --refine-gen-budget / --gen-budget-hard are compact-mode knobs; use with --compact")
    if args.lineage_budget and args.lineage_budget < args.min_session_budget:
        raise SystemExit("--lineage-budget is below --min-session-budget; the first session could not open")
    if args.task_budget < args.base_explorers * args.min_session_budget:
        raise SystemExit("--task-budget is not enough for each base explorer to open one --min-session-budget session")
    args.dp_rank = None           # assigned per task in main_async
    args.should_abort = None      # main_async attaches the error-rate guard
    args.submission = None        # a SubmissionStore with --submission-out
    if args.hparam_menu and not args.compact:
        raise SystemExit("the axes --hparam-menu draws are compact-mode knobs; use with --compact")
    cfg = load_config()
    args.tool = cfg["tool"]
    args.tool_name = cfg["tool_name"]
    args.extra_tools = [prompts.DELEGATE_TOOL] if args.delegate else []
    args.tool_timeout = cfg["rollout"]["tool_timeout_s"]
    args.tool_max_output_chars = cfg["rollout"]["tool_max_output_chars"]
    args.sampling = Sampling()
    args.annotate_ids = None      # the tokens backend replaces this with TokenBackend.annotate
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.count_tokens = lambda text: len(tok(text)["input_ids"])
    if args.backend == "dsv4":
        from dsv4_backend import DSV4_EFFORTS, DSv4Backend
        if args.encoding_path is None:
            raise SystemExit("--backend dsv4 requires --encoding-path")
        if args.reasoning_effort not in DSV4_EFFORTS:
            raise SystemExit(f"dsv4 --reasoning-effort must be one of {DSV4_EFFORTS}")
        if args.handoff_thinking == "off":
            raise SystemExit("the dsv4 backend has no enable_thinking=False; use same or low for --handoff-thinking")
        backend = DSv4Backend(args.base_url, args.encoding_path, tok, args.retries)
        args.chat = backend.chat
    elif args.backend == "tokens":
        from token_backend import EFFORTS, TokenBackend
        if args.reasoning_effort not in EFFORTS:
            raise SystemExit(f"tokens backend --reasoning-effort must be one of {EFFORTS}")
        if args.handoff_thinking != "same":
            raise SystemExit("tokens backend --handoff-thinking supports only same (the prompt prefix must be fixed)")
        backend = TokenBackend(args.base_url, tok, [args.tool, *args.extra_tools], args.retries)
        args.chat = backend.chat
        args.annotate_ids = backend.annotate
    else:
        async def _openai_chat(client, payload):
            return await _chat(client, args.base_url, payload, args.retries)
        args.chat = _openai_chat
    if args.ctx_cap - args.report_reserve - args.context_margin < 10_000:
        raise SystemExit("ctx_cap minus reserve/margin is under 10k; not workable")
    # The deadline counts from here (tokenizer / backend loaded), so the caller's minutes are all spent running tasks.
    args.deadline = time.monotonic() + args.deadline_min * 60 if args.deadline_min > 0 else None
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
