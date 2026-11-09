#!/usr/bin/env python
"""Agentic TIR：explorers（並行、各自 ≤ ctx cap）→ harness 驗證 → refiner（≤ ctx cap、每個 test 最多 2 attempts）。

    python work/agentic_tir/agentic_tir.py --task-dir ... --task-list ... --out $ARC_RUNS/agentic_tir/<tag> \
        --base-url http://127.0.0.1:30031/v1 --model qwen38fn --n-explorers 3 --ctx-cap 50000

一題的流程
----------
1. **explore**：K 條獨立 rollout，同一份 prompt（所有 test input 一起給），各自的 sandbox。
   每條的 context（prompt+completion，以 server 回的 total_tokens 計）壓在 `ctx_cap` 內：
   每輪 max_tokens = ctx_cap − report_reserve − last_total − margin。餘額不夠、或模型被截斷、
   或輪數用完 → 追加一則 user 訊息要求寫交接報告（handoff，預設關 thinking、tool_choice=none）。
   正常結束的最終回覆本身就是報告（RULE/FINDINGS/CODE/CONFIDENCE/ANSWER，見 prompts.py）。
2. **verify**：開 refine 用的 sandbox，把每份報告的 `transform` 定義成 `transform_k`，
   跑全部 train pair（pass/fail/error）與全部 test input（預測 grid）；連同 explorer 明講的
   ANSWER 一起去重成 candidates（A、B、…）。這是 refiner 看到的「事實表」。
3. **refine**：一條 rollout，prompt = 題目 + K 份報告 + 檢查表 + candidates；同一個 ctx cap；
   最終報告的 ANSWER 是每個 test 的 ATTEMPT 1（必要）/ ATTEMPT 2（可選）。refine 沒給答案的
   test 用 candidates 依票數補；`--fill-attempt2` 開著時 attempt 2 缺的也由 candidates 補
   （兩種來源都記在 attempt_src 裡，分析時分得開）。refine 用完預算沒答完 → 它的報告加進
   reports、再開一輪（≤ max_refine_rounds）。

計分口徑與 work/qwen38_eval/score_partial.py 相同：pairs/*.json 可直接餵它。
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
import statistics
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
    """一條對話的紀錄。所有計量都來自 server 的 usage，不自己估。"""

    def __init__(self, label: str, lineage: int = 0, hop: int = 1):
        self.label = label
        self.lineage = lineage        # explorer 編號（refiner 為 0）
        self.hop = hop                # 同一條 explorer 的第幾個 session
        self.messages: list[dict] = []
        self.calls: list[dict] = []
        self.stop_reason = ""
        self.forced_handoff = False
        self.n_tool_calls = 0
        self.n_tool_errors = 0
        self.n_sandbox_restarts = 0    # 模型的 code 讓 sandbox 300s 沒回應、被砍掉重開的次數（狀態全失）
        self.n_continuations = 0
        self.n_compactions = 0             # --compact：harness 把最舊的 tool round 折掉、同一條對話續跑的次數
        self.compact_dropped_tokens = 0    # 累計被折掉的估計 token（模型看不到的部分）
        self.compact_dropped_rounds = 0
        self.compact_dropped_calls = 0
        self.n_checkpoints = 0             # --compact-checkpoint：折掉前的記帳輪數
        self.compact_history: list[str] = []   # --compact-rich：被折掉的每一輪留一行「試了什麼 → 結果」
        self.compact_transform_src = ""        # --compact-rich：被折掉的輪次裡最後一版 `def transform` 原始碼
        self.n_outputs_cleared = 0             # --compact-rich：被清掉的 tool 輸出數
        self.output_heads: dict[str, str] = {}  # 被清掉的 tool 輸出的開頭（給 _round_summary 用；不存檔）
        self.task_user = ""                # 第一則 user 訊息的原文（compaction 每次從它重組，不會疊加公告）
        self.task_id = ""                  # 只給 log 用
        self.notes_structured = False      # --notes-structured（給 _report_for_refiner 渲染用）
        self.sub_episodes: list[dict] = []  # --delegate：helper 對話（to_dict，含 messages）
        self.n_delegations = 0
        self.segments: list[list[dict]] = []  # --save-segments：每次 compaction 前的整份 messages（模型實際看到的視圖），最後一段是結束時的 messages
        self.final_content = ""
        self.final_reasoning = ""
        self.report: dict | None = None
        self.notes: list[str] = []          # session 結束時 sandbox 裡的 `notes`（帳本，逐字帶給接手者）
        self.home_check: dict | None = None  # session 結束時在自己的 sandbox 跑報告 CODE 的結果（helper 都在）
        self.alt_check: dict | None = None   # alt 變體：同上，但檢查的是反假設 `transform_alt`（refiner 沒有）
        self.code_src = ""
        self.answers: list[list] = []          # answers[t] = [grid, ...]（attempt 順序）
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
            d["segments"] = self.segments   # write_task 會把它移到 segments/<task>.jsonl.gz
        return d


def _copy_private(asst: dict, msg: dict) -> None:
    """backend 掛在回應上的私有鍵（token backend 的 `_ids` / `_gen_len`）要跟著進 messages，segments 落地時才有 token。"""
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
        "segment": ep.n_compactions,   # 這次呼叫看到的是第幾段（--save-segments 的 segment_idx）
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
        # sglang 的 DP 是 round-robin、沒有 session affinity：不釘 rank，每一輪都會在別的 replica 重 prefill
        # 整條 context（naive driver 是在 run_rollout 釘的，這裡之前沒有）。同一題的所有 episode 釘同一個
        # rank，system + 題目的前綴還能跨 explorer / refiner 命中 radix cache。
        p["routed_dp_rank"] = args.dp_rank
    return p


class StreamStats:
    """v15：每個完成的 model call 的「completion tokens / 該 call 的 wall 秒數」（最近 WINDOW 筆）與全 run 的 completion 總數。
    `--deadline-tok-per-s` 是寫死的每條 stream 速率估計；v14 C80 實測每條只有 ~6.2 tok/s，估計值 30 讓最後一題晚了 900 s。
    有了實測的中位數，_time_room 與 --time-aware-budget 才有真實速率可用。"""
    WINDOW = 200
    MIN_SAMPLES = 30

    def __init__(self):
        self.rates: collections.deque[float] = collections.deque(maxlen=self.WINDOW)
        self.total_completion = 0

    def record(self, completion_tokens: int, wall_s: float) -> None:
        assert completion_tokens >= 0 and wall_s > 0, (completion_tokens, wall_s)
        self.total_completion += completion_tokens
        # 0 token 的 call 沒有速率資訊（不是速率 0），不進樣本；總數照加（加 0）。
        if completion_tokens > 0:
            self.rates.append(completion_tokens / wall_s)

    def measured(self) -> float | None:
        """最近 WINDOW 筆的中位數；樣本不到 MIN_SAMPLES 回 None（還不能信）。"""
        if len(self.rates) < self.MIN_SAMPLES:
            return None
        m = statistics.median(self.rates)
        if not m > 0:
            raise RuntimeError(f"measured stream rate must be > 0, got {m}")
        return m


def _stream_rate(args) -> float:
    """_time_room 用的每條 stream 速率：--deadline-rate-adaptive 且樣本夠時取 min(設定值, 實測中位數)，否則用設定值。"""
    rate = args.deadline_tok_per_s
    if args.deadline_rate_adaptive:
        measured = args.stream_stats.measured()
        if measured is not None:
            rate = min(rate, measured)
    return rate


def _time_room(args) -> int:
    """deadline 之前這條 stream 還生得出多少 token（用保守的每條 decode 速率估）；沒有 deadline 就是無限。"""
    if args.deadline is None:
        return 10**9
    return int((args.deadline - time.monotonic()) * _stream_rate(args))


def _cutoff(args) -> str | None:
    """整個 run 層級的停止條件：呼叫端說沒救了（伺服器死掉）、或 deadline 到了——「到了」的定義是
    剩下的時間連一個最短回合都生不完。回傳原因字串；None 表示可以繼續。"""
    if args.should_abort is not None and args.should_abort():
        return "abort"
    if args.deadline is not None and _time_room(args) < args.min_turn_tokens:
        return "deadline"
    return None


def _metered_chat(args):
    """包住 args.chat（bridge 裝完之後才包，所以 bridge 的 chat 也量得到）：每個 call 完成時記一筆 usage 與 wall 秒數。
    全部 call site（turn / handoff / checkpoint / delegate）都走 args.chat，這是唯一的掛點。"""
    inner, stats = args.chat, args.stream_stats

    async def chat(client, payload):
        t0 = time.monotonic()
        data = await inner(client, payload)
        stats.record(int(data["usage"]["completion_tokens"]), time.monotonic() - t0)
        return data
    return chat


# v15 time-aware budget 的常數：1.8 = v14 實測「每題平均花費 / 上限」≈ 0.55 的倒數。
TAB_WARMUP_S = 1800
TAB_SUPPLY_FRAC = 0.9
TAB_OVERSPEND = 1.8
TAB_FLOOR = 40_000
TAB_EXPLORER_FRAC = 0.42
TAB_TWO_EXPLORER_BELOW = 150_000
TAB_NO_REFINE_BELOW = 90_000
TAB_STREAM_FRAC = 0.8


def _time_aware_plan(args, *, now: float, run_start: float, total_completion: int, measured: float | None,
                     n_unstarted: int, n_inflight: int) -> dict:
    """依剩餘時間決定這題的預算。回傳 {"adjusted": bool, "log": {...}, "overrides": {...}}；overrides 空 = 不改。
    供給 = 全 run 平均吞吐 × 剩餘時間 × 0.9，平分給「還沒開始的題 + 半個在飛的題」。"""
    assert args.deadline is not None and n_unstarted >= 1 and n_inflight >= 0, (args.deadline, n_unstarted, n_inflight)
    elapsed = now - run_start
    time_left = args.deadline - now
    assert elapsed >= 0 and time_left > 0, (elapsed, time_left)
    rate_agg = total_completion / elapsed if elapsed > 0 else 0.0
    log = {"elapsed": elapsed, "rate_agg": rate_agg, "measured": measured, "time_left": time_left,
           "unstarted": n_unstarted, "inflight": n_inflight, "allowance": None, "cap": args.task_budget,
           "explorer_budget": args.compact_gen_budget, "explorers": args.n_explorers, "refine": True}
    # 第一波題目（還沒有吞吐資料）用滿額預算。
    if elapsed < TAB_WARMUP_S or total_completion <= 0:
        return {"adjusted": False, "log": log, "overrides": {}}
    allowance = rate_agg * time_left * TAB_SUPPLY_FRAC / (n_unstarted + 0.5 * n_inflight)
    cap = min(args.task_budget, max(TAB_FLOOR, int(TAB_OVERSPEND * allowance)))
    explorer = max(args.min_session_budget, min(args.compact_gen_budget, int(TAB_EXPLORER_FRAC * cap)))
    if measured is not None:
        # 單條 stream 在剩餘時間內生得完的量；比 min_session_budget 還小就不再壓（交給 _cutoff / deadline 處理）。
        stream_cap = int(measured * time_left * TAB_STREAM_FRAC)
        if stream_cap >= args.min_session_budget:
            explorer = min(explorer, stream_cap)
    overrides = {"task_budget": cap, "compact_gen_budget": explorer,
                 "lineage_budget": min(args.lineage_budget, explorer + args.handoff_budget)}
    if cap < TAB_TWO_EXPLORER_BELOW:
        overrides["n_explorers"] = min(args.n_explorers, 2)   # base_explorers=2：K=2 即不開 E3
    if cap < TAB_NO_REFINE_BELOW:
        overrides["skip_refiner"] = True
    log.update(allowance=allowance, cap=cap, explorer_budget=explorer,
               explorers=overrides.get("n_explorers", args.n_explorers),
               refine=not overrides.get("skip_refiner", False))
    return {"adjusted": True, "log": log, "overrides": overrides}


SANDBOX_KILLED = "[sandbox killed: no reply within 300 s; restarted, all state lost]"

# `notes` 的鏡像：sandbox 的 RLIMIT_FSIZE=0 封鎖所有檔案寫入，所以鏡像由 harness 這端維護——每次 tool call 後問
# `len(notes)`，只把新增的條目逐條讀回（每條 ≤ NOTE_ENTRY_CHARS，不會撞 stdout 上限），harness 自己寫進 run 目錄；
# sandbox 重啟時用 stdin 灌回去。程式狀態（函式、變數）救不回來，但知識（notes）不該跟著行程一起死——
# compaction 之後 notes 是唯一跨越被折掉輪次的記憶。
NOTE_ENTRY_CHARS = 3000
_NOTES_LEN = "print(len(notes) if isinstance(notes, (list, tuple)) else -1)"
_NOTES_ITEM = "import json as _j; print(_j.dumps(str(notes[{i}])[:{n}]))"
_NOTES_LOAD = "import json as _j; notes = _j.loads({blob!r}); print(len(notes))"


async def sb_run(sb: Sandbox, code: str) -> tuple[str, bool]:
    """`sb.run` 加一層：子行程 300s 沒回應（SIGALRM 攔不住吞掉 KeyboardInterrupt 或卡在 C 呼叫裡的碼）就砍掉重開，
    回 (SANDBOX_KILLED, True) 當 timeout 處理。v7 的 da515329、v9 的 88e364bc 都因此整題報廢過——後者掛在
    harness 自己跑的 home check，所以每個 call site 都要走這裡。重開次數記在 sb.n_restarts；重開後把 notes 鏡像
    灌回去，筆數記在 sb.notes_restored。"""
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
    """把 sandbox 裡 notes 的新增條目讀回 harness 的鏡像（sb.notes_mirror），並寫到 run 目錄。
    模型把 notes 換成更短的 list（或非 list）就整份重讀。sandbox 掛了就跳過，下一次 sb_run 會處理重啟。"""
    try:
        out, timed = await asyncio.to_thread(sb.run, _NOTES_LEN)
        if timed or not out.strip().lstrip("-").isdigit():
            return
        n = int(out.strip())
        if n < 0:   # 不是 list：當成單一條目
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
    """delegate_ctx = (client, args, n_test, task)：有的話 `delegate` 工具可用（helper 自己的 _tool_round 不給）。"""
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
                if not isinstance(code, str):   # {"code": null} 之類：v8 的 0934a4d8 讓 `out` 沒被定義而整題報廢
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
    """`task` 給了且 args.delegate 開著：主迴圈的 tool round 可以分派 `delegate`。helper 自己（task=None）不能。"""
    ep = Episode(label, lineage, hop)
    delegate_ctx = (client, args, n_test, task) if (task is not None and args.delegate) else None
    ep.task_id = task_id
    ep.notes_structured = args.notes_structured
    ep.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ep.task_user = user
    # 第一輪還沒有 server 回的 total_tokens，用 tokenizer 估 prompt 長度（+ template/tool schema 的固定開銷）。
    # 不估的話第一輪的 max_tokens 會把整個 cap 當成可生成量，7k 的 prompt 就把 report_reserve 吃掉
    # （smoke2 的 142ca369 explorer 2：一輪純推理 41.5k，交接只剩 988 token，報告被截成只有 RULE）。
    last_total = args.count_tokens(system + user) + args.template_overhead
    need_handoff = None       # None = 正常結束；否則是觸發 handoff 的原因
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
            # context 滿了：不交接，把最舊的 tool round 折掉、同一條對話續跑（sandbox 狀態都在）。
            if args.compact_checkpoint:
                last_total = await _checkpoint_before_compact(ep, args, sb, last_total, client, tool, tool_name)
            new_total = await _compact(ep, args, sb, n_test)
            if new_total is not None:
                last_total = new_total
                room = args.ctx_cap - args.report_reserve - last_total - args.context_margin
        if room < args.min_turn_tokens:
            need_handoff = "context_exhausted"
            break
        # 接近 deadline 時把單輪 max_tokens 一起壓下來：一輪 20k 在 L4 上要跑好幾分鐘，deadline 之後
        # 在飛的 request 不會被砍，只能事前不讓它那麼長。
        mt = min(args.max_turn_tokens, room, _time_room(args))
        if args.compact and args.gen_budget_hard:
            # 硬預算：單輪不得把這條對話的生成量帶過 compact_gen_budget。原本只在回合開頭檢查，單輪最多再超
            # max_turn_tokens（20k）；fp8 sweep 實測 refiner 40k 花到 55k。被截斷的那輪走 CONTINUE，下一輪開頭
            # gen_budget 觸發交接（交接另有 --handoff-budget）。至少給 min_turn_tokens，超支上限就是它。
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
                # 截斷的 tool call 引數多半是壞的，不執行。
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
        # 沒有 tool call：最終回覆。
        asst = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            asst["reasoning_content"] = msg["reasoning_content"]
        _copy_private(asst, msg)
        ep.messages.append(asst)
        if finish == "length":
            # 單輪思考撞到 max_turn_tokens（v1 有 107/185 次交接是這樣來的）：總預算還有就叫它
            # checkpoint 再繼續，不要交接。preserve_thinking 讓被截斷的思考留在 context 裡。
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
        # 不再呼叫模型。交接報告本身要一次 ≥3.5k token 的生成，deadline 之後沒有這個預算；答案改由呼叫端從
        # sandbox 現有的 `transform` 收（run_hop 的 home_check 對 code=None 走的就是這條路），report 是空的。
        ep.forced_handoff = True
        ep.stop_reason = f"handoff:{need_handoff}"
    elif need_handoff is not None:
        ep.forced_handoff = True
        ep.messages.append({"role": "user", "content": handoff_user or prompts.HANDOFF_USER.format(n_calls=args.handoff_tool_calls)})
        calls_left = 0 if handoff_user else args.handoff_tool_calls
        refused = 0          # tool_choice=none 在 tokens/dsv4 backend 沒有對應物：模型在 STOP 後還叫工具就在這裡擋
        handoff_used = 0     # 交接階段累計生成量（--handoff-budget 的分母）
        while True:
            if _cutoff(args) is not None:
                break        # 交接期間到點：報告寫到哪算到哪，不再多發一次生成
            mt = args.ctx_cap - last_total - args.handoff_margin
            if mt < args.min_handoff_tokens:
                # report_reserve 是留給這裡的，正常到不了；真到了就寧可略超 cap 也要拿到完整報告。
                mt = args.min_handoff_tokens
            if calls_left > 0:
                # checkpoint 的 tool call 要短：smoke3 裡不設上限時這一輪思考 6–12k，把報告擠到被截斷。
                mt = min(mt, args.handoff_turn_tokens)
            if args.handoff_budget:
                # 交接階段整體的硬上限。不寫報告、一直叫工具的模型會把 checkpoint 2 次 + 拒絕 3 次的每一輪都生到
                # ctx 餘額（最多 ~30k）：本機 plumbing 量到 explorer 預算 20k 卻花 30–58k，超支全在這裡。
                # Kaggle 的預算是硬的，寧可拿到被截斷的報告（答案還能從 sandbox 的 transform 收）。
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
                # checkpoint 那輪被截斷（tool call 或純思考）：不執行，直接要報告。
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
                # 連續三次不寫報告還在叫工具（2026-09-03 審核：20/14k 條 episode 在 STOP 後又叫 3–23 次、context 超 cap）：
                # 就把這一輪的內容當最終回覆，不再給機會。
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
    """--delegate：開一條乾淨 context 的 helper 對話（同一個 sandbox），跑完只把它的短結論當 tool 結果回給 explorer。
    helper 的 token 併進父 episode 的 calls（kind 前綴 delegate:），對話本身存進 ep.sub_episodes。"""
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
    sub_args.extra_tools = []                         # helper 不能再 delegate
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
    """一則訊息的估計 token（內容 + reasoning + tool call 引數 + 每則 8 token 的模板開銷）。"""
    n = 8 + args.count_tokens(m.get("content") or "")
    if m.get("reasoning_content"):
        n += args.count_tokens(m["reasoning_content"])
    for call in m.get("tool_calls") or []:
        n += args.count_tokens(call["function"]["arguments"]) + 12
    return n


OUTPUT_CLEARED = ("[output cleared by the harness to save context -- the code above already ran and everything it "
                  "defined is still in the sandbox; re-run it if you need to see the output again]")


def _round_summary(r: list[dict], ep: Episode) -> str:
    """被折掉的一輪留一行：第一個 tool call 的 code 開頭 → 輸出開頭（Manus：失敗要留在 context 裡，不然會重試）。"""
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
    """Anthropic context-editing 的做法：把一輪的 tool 輸出換成佔位符，留下模型自己的 reasoning 與 code。
    輸出是可還原的（code 還在、sandbox 還在），reasoning 不是。"""
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
    """--compact-checkpoint：折掉舊輪次之前先給模型一輪（≤ compact_checkpoint_tokens）把即將消失的推理寫進 `notes`。
    這一輪連同它的 tool 輸出是最新的 round，compaction 會留下來；成本從 report_reserve 的閒置空間出。回傳新的 last_total。"""
    starts = [i for i, m in enumerate(ep.messages) if m["role"] == "assistant"]
    n_rounds = len(starts)
    tmpl = prompts.CHECKPOINT_USER_STRUCTURED if args.notes_structured else prompts.CHECKPOINT_USER
    ep.messages.append({"role": "user", "content": tmpl.format(n_rounds=max(n_rounds - 1, 1))})
    mt = min(args.compact_checkpoint_tokens, args.ctx_cap - last_total - args.context_margin)
    if mt < 1000:
        # 連 checkpoint 都塞不下（單輪把 reserve 也吃掉了）：略過，直接折。
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
    """--compact：context 滿了就把最舊的 tool round 從 messages 折掉（sandbox 裡的狀態一個都不少），
    題目訊息後面插一則 harness 公告（折掉多少、sandbox 現況、notes），同一條對話繼續。
    --compact-rich 再加三件事：(1) 先把非最近幾輪的 tool 輸出換成佔位符（reasoning 與 code 留著），夠了就不丟整輪；
    (2) 公告帶被折掉輪次裡最後一版 `def transform` 原始碼；(3) 每個被折掉的輪次留一行「試了什麼 → 結果」。
    回傳折掉後的估計 context token；只剩一輪、折不動就回 None（呼叫端照舊交接）。"""
    msgs = ep.messages
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[1]["content"] == ep.task_user, ep.label
    starts = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(starts) < 2:
        return None
    rounds = [msgs[a:b] for a, b in zip(starts, starts[1:] + [len(msgs)])]
    # --save-segments：折之前先把模型當下看到的整份 messages 存起來（rich 模式接著會就地清 tool 輸出，所以要在這裡 deepcopy）。
    if args.save_segments and args.annotate_ids is not None:
        args.annotate_ids(msgs, args.reasoning_effort)   # 最後一輪 tool 訊息還沒送過 server，先把 token 補齊
    snap = copy.deepcopy(msgs) if args.save_segments else None
    # 折完的 context 目標 = 工作視窗的 compact_target_frac；扣掉 system + 題目 + 公告（估 3k）就是能留給最近幾輪的量，
    # 下限 compact_keep_tokens。refiner 的題目訊息（K 份報告 + 候選表）比 explorer 大得多，固定 keep 會讓它折完就又快滿。
    base = _msg_tokens(args, msgs[0]) + _msg_tokens(args, msgs[1]) + 3000 + args.template_overhead
    window = args.ctx_cap - args.report_reserve - args.context_margin
    keep_budget = max(args.compact_keep_tokens, int(window * args.compact_target_frac) - base)
    if args.compact_rich:
        # 第一階段：最近 compact_keep_full_rounds 輪以外的 tool 輸出清掉。
        for r in rounds[:max(len(rounds) - args.compact_keep_full_rounds, 0)]:
            _clear_tool_outputs(r, ep)
    sizes = [sum(_msg_tokens(args, m) for m in r) for r in rounds]
    # 第二階段：從最新一輪往回留；至少留 1 輪。rich 模式清完輸出就夠的話一輪都不丟。
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
    # 公告是獨立的一則 user 訊息，題目訊息一字不動：system + 題目 的前綴才能一直命中 radix cache
    # （smoke 裡把公告掛在題目尾巴，每次折完命中率就掉回只剩 system prompt）。
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
    """公告裡的 sandbox 函式清單：保留開頭、超過就明講截掉幾個（之前是 `[-300:]` 截頭，11% 的公告清單開頭是半個名字）。"""
    if not names:
        return "(none)"
    if len(names) <= limit:
        return names
    cut = names[:limit]
    n_more = names[limit:].count("', '") + 1
    return cut.rsplit("', '", 1)[0] + f"', ... (list truncated, {n_more} more)"


def _continue_after_cut(ep: Episode, args, last_total: int) -> bool:
    """單輪被 max_turn_tokens 截斷：總預算還夠、次數未用完就追加 CONTINUE 訊息讓它接著做。"""
    left = args.ctx_cap - args.report_reserve - last_total - args.context_margin
    # compact 模式的總量由 --compact-gen-budget 管，續跑次數不設限（smoke 裡一條 explorer 在 104k 時因為
    # 用完 4 次續跑就交接，白丟了 100k 的預算）。
    if (not args.compact and ep.n_continuations >= args.max_continuations) or left < args.min_turn_tokens:
        return False
    ep.n_continuations += 1
    ep.messages.append({"role": "user", "content": prompts.CONTINUE_USER.format(left_k=f"{left / 1000:.0f}")})
    return True


def _salvage_transform_code(messages: list[dict]) -> str | None:
    """報告裡沒 CODE 時，撈最後一次在 tool call 裡定義 `transform` 的程式碼。"""
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
    """在 sandbox 算一句表達式成 grid；失敗回 (None, reason)。"""
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
                # 報告裡的 CODE 沒在 sandbox 跑過（或用了別的名字）：把它定義進去再算一次。
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

# 接續 session 用：CODE 直接 exec 進 sandbox 的 globals，報告裡的碼常依賴 sandbox 裡已定義的 helper。
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
    """把 `code` 定義成 sandbox 裡的 `name`（code 為 None 就用 sandbox 裡已有的），跑全部 train pair 與 test input。
    `in_globals`：直接在 sandbox 的 globals 裡 exec（只能是 name == "transform"），helper 才找得到。"""
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
    """不開 sandbox 的檢查表：直接用 explorer 自家 sandbox 的 home_check（verify_reports 多做的只是把 CODE 載進
    refine sandbox）。給「refiner 之前先寫一份 submission」與「deadline 到了、refiner 沒跑」兩條路用。"""
    out = []
    for k, rep in enumerate(reports, start=1):
        home = rep["home_check"]
        assert home is not None, rep["label"]
        out.append({"name": f"transform_{k}", "defined": home["defined"], "load_err": None,
                    "train": home["train"], "test": home["test"], "summary": home["summary"]})
    return out


async def verify_reports(sb: Sandbox, reports: list[dict], n_test: int) -> list[dict]:
    """每份報告一個 check dict：train/test 結果取自 explorer 自家 sandbox 的 home_check（v4–v6 在 refine sandbox
    重跑 CODE，28% 因為 helper / import 不在而 NameError，refiner 被告知能用的碼壞了）；refine sandbox 只嘗試把
    CODE 載成 transform_k，載不進來就在 summary 裡講明。"""
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
    """sandbox 裡 `notes` 的內容（模型可能把它弄成非 list 或塞進奇怪的東西，一律轉字串、截短）。
    截短在 sandbox 裡做：印出來的那一行必須小於 sandbox 的輸出上限 `out_limit`，否則會被攔腰截斷、JSON 壞掉。"""
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
    """每個 test：去重後的候選 grid，A、B、… 依（票數、首次出現）排序。
    alt 變體的反假設（ATTEMPT 2、全過 train 的 `transform_alt`）也進候選，但不算票：它是同一條 explorer的第二讀法。"""
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


# ---------------------------------------------------------------- --task-source kaggle（ARC Prize 提交）
def load_challenges(path: Path) -> dict:
    """比賽的單一大 JSON（`arc-agi_test_challenges.json`，test 沒有 output），也吃 `<dir>/<task_id>.json`
    的本地佈局（public eval 的 test 內嵌 output，會算分）。與 work/kaggle_submit/run_submission.py 同口徑。"""
    if path.is_dir():
        return {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}
    return json.loads(path.read_text())


# ---------------------------------------------------------------- --task-source nvarc（蒸餾語料）
def choose_pairs(uid: str, attempt: int, n_pairs: int, n_train: int, n_test: int = 1):
    """逐字對齊 work/qwen38fn_distill/rollout_runner.py::choose_pairs（n_test=1 時回傳完全相同）：
    同一個 uid/attempt 在這裡與 naive 語料抽到同一組 train/test，兩批資料才能逐題配對比。"""
    h = hashlib.sha256(f"{uid}:{attempt}".encode()).digest()
    order = sorted(range(n_pairs), key=lambda i: hashlib.sha256(h + i.to_bytes(2, "big")).digest())
    chosen = order[: n_train + n_test]
    return sorted(chosen[:n_train]), chosen[n_train:n_train + n_test]


def load_nvarc_corpus(path: str) -> dict[str, list]:
    """`nvarc_training.jsonl.zst`：每行 {"uid", "pairs": [[input, output], ...]}。整份約 47k 題、90 MB 壓縮。"""
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


# --hparam-menu：每題決定性抽一組超參數（各軸獨立雜湊，軸之間不相關），讓語料涵蓋不同 context 上限 / effort / K。
# 權重是收資料的分佈，不是最佳設定；抽到的值連同 uid 存在 tasks/<uid>.json 的 hparams。
HPARAM_MENU: dict[str, list[tuple]] = {
    "reasoning_effort": [("xhigh", 0.6), ("medium", 0.3), ("low", 0.1)],
    "ctx_cap": [(50_000, 0.6), (40_000, 0.2), (65_000, 0.2)],
    "n_explorers": [(2, 0.7), (3, 0.3)],           # base_explorers 固定 2：第 3 條只在前兩條不一致時開
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
    sb.notes_path = notes_dir / f"{task['id']}_{tag}.json"   # notes 鏡像（見 sync_notes / sb_run）
    sb.notes_mirror: list[str] = []
    sb.notes_synced = 0
    sb.notes_restored = -1
    return sb


def _with_gen_budget(args, gen_budget: int):
    """args 的淺拷貝，只蓋掉這個 session 的 compact_gen_budget（chat / count_tokens 等 closure 共用，同 main_async 的 targs）。
    run_hop / run_episode / prompts 讀預算全走 args.compact_gen_budget，所以把拷貝當 args 傳進去就夠，不改簽名。"""
    assert gen_budget > 0, gen_budget
    a = copy.copy(args)
    a.compact_gen_budget = gen_budget
    return a


async def run_hop(client, args, task, k: int, hop: int, sb: Sandbox, prev: Episode | None,
                  live: asyncio.Semaphore) -> Episode:
    """explorer k 的第 hop 個 session，sandbox 跨 session 共用。hop>1 時 prompt 帶上一個 session 的報告、
    harness 剛跑的 `transform` 檢查、與 sandbox 裡的 `notes`（v4：把長鏈思考切成幾個 50k 視窗）。"""
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
            # 報告裡的 CODE 就是接手者會讀到的 `transform`：把它定義回 sandbox，文字與物件才一致。
            assert prev.home_check is not None
            info = dict(prev.report, stop_reason=prev.stop_reason, check=prev.home_check, notes=prev.notes,
                        notes_structured=args.notes_structured)
            user = prompts.render_continue_user(task["train"], test_inputs, hop, args.max_hops, info)
        ep = await run_episode(client, args, label=f"explorer {k} session {hop}", system=system, user=user,
                               sb=sb, tool=args.tool, tool_name=args.tool_name, n_test=n_test, lineage=k, hop=hop,
                               task_id=task["id"], task=task)
        ep.notes = await read_notes(sb, args.tool_max_output_chars)
        # 報告裡的 CODE 定義回 sandbox 並跑 train/test：接手者讀到的 `transform` 與文字一致，refiner 拿到的
        # 檢查結果也是在 helper 都在的環境跑出來的。
        ep.home_check = await _check_transform(sb, ep.report["code"], "transform", n_test, in_globals=True)
        if args.prompt_variant == "alt":
            # 反假設 `transform_alt`（模型有定義的話）也在自家 sandbox 跑 train/test：refiner 看得到、全過的進 candidates。
            ep.alt_check = await _check_transform(sb, None, "transform_alt", n_test)
        return ep


def _all_train_pass(chk: dict | None) -> bool:
    return bool(chk and chk["defined"] and chk["train"] and all(r == "pass" for r in chk["train"]))


def _agreed_answer(finals: list[Episode], n_test: int) -> list | None:
    """≥2 條已答完的 explorer 對每個 test 的第一個答案完全一致：回傳那組答案（每個 test 一個 grid），否則 None。
    v2 裡答完的 explorer 114/126 是對的，兩條獨立答完又一致就不必再花 session 在其他條上
    （reap128 final 120 題：一致 71 題、對 62 題；answered-only 與不分停止原因的數字相同）。"""
    done = [ep for ep in finals if ep.stop_reason == "answered" and all(ep.answers[t] for t in range(n_test))]
    for i in range(len(done)):
        for j in range(i + 1, len(done)):
            if all(done[i].answers[t][0] == done[j].answers[t][0] for t in range(n_test)):
                return [done[i].answers[t][0] for t in range(n_test)]
    return None


def _lineages_converged(finals: list[Episode], n_test: int) -> bool:
    return _agreed_answer(finals, n_test) is not None


async def run_explorers(client, args, task, live: asyncio.Semaphore) -> tuple[list[Episode], list[Episode]]:
    """K 條 explorer，各自最多 max_hops 個 session。前 base_explorers 條先跑；它們的最終答案已有兩條一致就收工，
    否則才開剩下的條（v5：v4 裡前兩條一致的 56 題沒有一題一致而錯，第 3 條在那裡是純浪費）。
    回傳 (每條的最後一個 session, 全部 session 依序)。"""
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
            # 預算收縮（不取消 in-flight）：每個 session 的 gen 預算 = min(原值, lineage 剩餘, task 剩餘平分給這批)。
            # 撞到預算照原本 handoff:gen_budget 路徑寫交接報告，答案照舊由 home_check 收。
            launch = [(k, session_budget(k)) for k in active]
            skipped = [k + 1 for k, gb in launch if gb < args.min_session_budget]
            launch = [(k, gb) for k, gb in launch if gb >= args.min_session_budget]
            if launch:
                share = (args.task_budget - spent()) // len(launch)   # 同一個 gather 的 session 看到同一個 spent()：平分
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
            # v9：深的 session 只給已有全過 train 的 transform 的條——v8 的 hop 4 從「上一 session 全過」出發 16/45 答對，
            # 從沒全過出發 2/47；後者一個 session 35k，砍掉省 ~14k/題。
            active = [k for k in active if finals[k].forced_handoff
                      and (hop + 1 < args.pass_gate_hop or _all_train_pass(finals[k].home_check))]
            if active and _lineages_converged([e for e in finals if e is not None], n_test):
                active = []
            if active and spent() >= args.task_budget:
                # v10：這題的 explorer 已經花掉預算就不再開 session——v7/v8 答錯的題 265–342k、答對的 ≈ baseline，
                # 多出來的 token 全在最後沒解開的題上。
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
        assert all(finals[k] is not None for k in range(base)), "base explorers 的第一個 session 一定要開（parse_args 保證預算夠）"
        # 第 3 條（base..K）只在前面幾條「分歧」或「都沒過 train」時才開：一致而錯的很少（v4 0/56、v5 7/72），
        # 但兩條都答完、都沒過 train 又一致的答案不值得信。deadline / abort 時不開。
        done_finals = [e for e in finals if e is not None]
        if (base < K and spent() < args.task_budget and _cutoff(args) is None
                and (not _lineages_converged(done_finals, n_test)
                     or not any(_all_train_pass(e.home_check) for e in done_finals))):
            await run_group(list(range(base, K)), sbs, stack)
    finals = [e for e in finals if e is not None]
    assert len(finals) >= base
    return finals, all_eps


async def run_refine(client, args, task, reports: list[dict], round_no: int, live: asyncio.Semaphore):
    """回傳 (episode, checks, candidates)。refiner 的生成預算由 solve_task 用 _with_gen_budget 帶進來。"""
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
            ep.notes = await read_notes(sb, args.tool_max_output_chars)   # refiner 的 notes 之前沒讀回（審核：0/4,499）
    return ep, checks, cands


async def run_verify(args, task, reports: list[dict], live: asyncio.Semaphore):
    """--skip-refine-when-agree：只開 refine sandbox 跑 verify_reports + build_candidates，不跑 refiner LLM。
    回傳 (checks, candidates)，格式與 run_refine 的後兩項相同，summarize() 的 rec["checks"][0] 才有東西。"""
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
    # 先用 explorer 自家 sandbox 的檢查組一份候選、寫進 submission：refiner 還要再跑一條 ≤ cap 的對話，
    # 到點被砍時這一格已經有東西（票數最高的先）。refiner 收尾後會覆寫。
    home_checks = _checks_from_home(reports)
    home_cands = build_candidates(reports, home_checks, n_test)
    if args.submission is not None:
        for t in range(n_test):
            args.submission.put_attempts(task["id"], t, [c["grid"] for c in home_cands[t].values()])
    refines: list[Episode] = []
    all_checks, all_cands = [], []
    final_attempts: list[list] = [[] for _ in range(n_test)]
    final_src: list[list[str]] = [[] for _ in range(n_test)]
    # 兩條 explorer 都自己答完且每個 test 的第一個答案一致，就不開 refiner：teacher 上「一致而錯」v4 0/56、v5 7/72，
    # 而 refiner 是一整條 ≤ cap 的對話（sweep 實測 36–55k）。預算緊時這是最便宜的一刀；attempt 2 由候選補位。
    refine_skipped = ""
    agreed = _agreed_answer(finals, n_test) if args.skip_refine_when_agree else None
    if agreed is not None:
        refine_skipped = "agree"
        print(f"·· {task['id']}: explorers agree -> refiner skipped", file=sys.stderr)
        # 不跑 refiner LLM，但仍開 refine sandbox 做 verify/candidates：attempt 2 才有依票數排序的候選可補。
        checks, cands = await run_verify(args, task, reports, live)
        all_checks.append(checks)
        all_cands.append(cands)
    for rnd in range(1, args.max_refine_rounds + 1):
        if refine_skipped:
            break
        if args.skip_refiner:
            refine_skipped = "budget"                # v15 time-aware budget：這題預算太小，不開 refiner（同 deadline 路徑）
            break
        cut = _cutoff(args)
        if cut is not None:
            refine_skipped = cut                     # deadline / abort：不開 refiner，答案就是上面那份候選
            break
        # task 預算含 explorer 與前幾輪 refine；round 1 一定跑（下面 attempts 選擇需要至少一輪），之後不夠就不 escalate。
        # 預算收縮成剩餘量但不低於 min_session_budget（K3 實測：2×75k 後 refiner 拿 max(min(40k, 剩餘), 20k)）。
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
        # v2 的 refiner 信心很準（≥80：76/80 對；<50：2/17）；沒答完、被迫交接、或信心不足就再開一輪，
        # 它的報告併入 reports。v2 之前只在「沒答完」時開第二輪，表達式 ANSWER 讓這幾乎不發生（rounds/task 1.00）。
        weak = (not complete) or ep.forced_handoff or conf is None or conf < args.escalate_below
        if not weak or rnd == args.max_refine_rounds:
            break
        reports.append(_report_for_refiner(ep))
    if refines:
        # 最終 attempts：信心最高的一輪（同分取後面的）的 ATTEMPT 1 先；其他輪的 ATTEMPT 1 若不同就當第 2 個
        # ——兩輪是兩次獨立的判斷，這種分歧比同一條鏈自己寫的 ATTEMPT 2 更值得一個名額；再來才是最佳輪的 ATTEMPT 2。
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
        # 一致的答案就是 attempt 1；attempt 2 由下面的候選補位（票數次高的）。
        for t in range(n_test):
            final_attempts[t].append(agreed[t])
            final_src[t].append("agreed")
    # 其餘（deadline / abort 沒開 refiner）：全靠下面的候選補位。
    # candidates 補位：refine 沒答的 test 用票數最高的；--fill-attempt2 補第二個。
    # refiner 一輪都沒跑（deadline）就只有 explorer 自家檢查出來的候選。
    cands = all_cands[-1] if all_cands else home_cands
    for t in range(n_test):
        want = 2 if (args.fill_attempt2 or not final_attempts[t]) else len(final_attempts[t])
        for key, c in cands[t].items():
            if len(final_attempts[t]) >= want:
                break
            if c["grid"] not in final_attempts[t]:
                final_attempts[t].append(c["grid"])
                final_src[t].append(f"candidate:{key}")
    # 比賽的 hidden test 沒有 output：計分欄位全部 None（不是 False——False 會被 summarize 當成答錯算進分母）。
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
    # --save-segments：segments 很大（每段是模型當時的整份 messages），不放進 tasks/*.json（summarize / compare 會整批讀），
    # 另存 segments/<task>.jsonl.gz，一行一段。
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
            # token 計量按 test 均分，讓 score_partial.py 加總後等於整題。
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
                                                           "chat", "count_tokens", "annotate_ids", "stream_stats")},
                                             indent=1, default=str))
    if args.task_source == "nvarc":
        # 可以給多個 corpus（nvarc_training + nvarc_full 混著跑）；每題記 source = 檔名去掉 .jsonl.zst，落地在 tasks/*.json。
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
        # 注意：--limit / --tasks 在這之前就截斷了題目，所以 submission.json 只會有這些題的 key。
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
              f"(per-turn max_tokens shrinks at {args.deadline_tok_per_s:g} tok/s near the end"
              f"{', capped by the measured median stream rate' if args.deadline_rate_adaptive else ''})", flush=True)
    if args.deadline_rate_adaptive or args.time_aware_budget:
        args.chat = _metered_chat(args)   # bridge 已在 parse_args 之後裝好自己的 chat，這裡包在最外層
    # 伺服器死掉時每一題都以 ConnectError 收場；沒有這個護欄就會安靜地把 submission 留在 fallback 然後正常結束。
    health = {"ok": 0, "errors": 0, "tripped": ""}
    args.should_abort = lambda: bool(health["tripped"])
    dp_load = collections.Counter()
    skipped: list[str] = []

    live = asyncio.Semaphore(args.max_live)
    limits = httpx.Limits(max_connections=args.max_live + 64, max_keepalive_connections=args.max_live + 64)
    t_start = time.monotonic()
    n_done = 0
    # time-aware budget 的排程計數：asyncio 單執行緒，更新不會交錯。
    sched = {"unstarted": len(todo), "inflight": 0}
    async with httpx.AsyncClient(timeout=args.http_timeout, limits=limits) as client:
        # 一次把 120 題全丟進 gather 時，semaphore 是 FIFO：240 條 base explorer 排在所有 refiner 前面，
        # 題目要等它們全部起跑才會開始完成（compact 模式 session 長，第一題要 3 小時才出現）。
        # --task-live 限制同時在跑的題數，完成進度就平滑、--resume 也少丟。
        task_live = asyncio.Semaphore(args.task_live) if args.task_live > 0 else contextlib.nullcontext()

        def _check_health():
            total = health["ok"] + health["errors"]
            rate = health["errors"] / max(total, 1)
            if (args.max_error_rate > 0 and not health["tripped"] and total >= args.error_grace
                    and rate > args.max_error_rate):
                health["tripped"] = (f"{health['errors']}/{total} tasks failed "
                                     f"({100 * rate:.1f}% > {100 * args.max_error_rate:.1f}%)")
                print(f"\n{'=' * 70}\nABORT: {health['tripped']}\n  伺服器多半已經死了。停止發新的題目，"
                      f"在飛的對話不再呼叫模型。\n{'=' * 70}\n", flush=True)

        async def one(tid):
            nonlocal n_done
            task = get_task(tid)
            targs, hp = copy.copy(args), None    # 淺拷貝：chat / count_tokens / submission 等共用，只蓋掉每題的東西
            if args.hparam_menu:
                hp = draw_hparams(tid, args.attempt)
                vars(targs).update(hp)
            rank = None
            counted = False
            try:
                async with task_live:
                    n_unstarted = sched["unstarted"]   # 含本題
                    sched["unstarted"] -= 1
                    if _cutoff(args) is not None:
                        skipped.append(tid)      # deadline 到了 / 已中止：沒開始的題不開，submission 留 fallback
                        return None
                    n_inflight = sched["inflight"]     # 不含本題
                    sched["inflight"] += 1
                    counted = True
                    if args.time_aware_budget:
                        plan = _time_aware_plan(
                            args, now=time.monotonic(), run_start=t_start,
                            total_completion=args.stream_stats.total_completion, measured=args.stream_stats.measured(),
                            n_unstarted=n_unstarted, n_inflight=n_inflight)
                        vars(targs).update(plan["overrides"])
                        g = plan["log"]
                        allowance = "-" if g["allowance"] is None else f"{g['allowance']:.0f}"
                        measured = "-" if g["measured"] is None else f"{g['measured']:.2f}"
                        print(f"TIME_BUDGET tid={tid} elapsed={g['elapsed']:.0f} rate_agg={g['rate_agg']:.1f} "
                              f"measured={measured} time_left={g['time_left']:.0f} unstarted={g['unstarted']} "
                              f"inflight={g['inflight']} allowance={allowance} cap={g['cap']} "
                              f"explorer_budget={g['explorer_budget']} explorers={g['explorers']} "
                              f"refine={'on' if g['refine'] else 'off'}", flush=True)
                    if args.fixed_dp_rank is not None:
                        # 單 rank 收資料：即使 server 是 DP2 也全部釘死，另一個 pool 空著。
                        rank = args.fixed_dp_rank
                        dp_load[rank] += 1
                    elif args.dp_size > 1:
                        # 拿到名額才選 rank：以在飛的題數最少者為準（同分取小），explorer/refiner 都跟著這題走。
                        rank = min(range(args.dp_size), key=lambda r: (dp_load[r], r))
                        dp_load[rank] += 1
                    targs.dp_rank = rank
                    rec = await solve_task(client, targs, task, live)
            except Exception as e:  # 一題炸掉不該連坐其他 119 題，但要大聲記下來
                n_done += 1
                health["errors"] += 1
                _check_health()
                print(f"!! {tid} FAILED: {type(e).__name__}: {e}", flush=True)
                (out / "failures.jsonl").open("a").write(json.dumps({"task_id": tid, "error": repr(e)}) + "\n")
                raise
            finally:
                if counted:
                    sched["inflight"] -= 1
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
        # 刻意不當成「跑完」：submission 裡多數格子是 fallback，交出去會被當成模型的 0 分。
        print(f"\nFAILED: {health['tripped']}", flush=True)
        sys.exit(2)
    if n_fail:
        print(f"!! {n_fail} tasks failed; see failures.jsonl", flush=True)
        if args.task_source != "kaggle":
            sys.exit(1)
        # kaggle：個別題目失敗不能讓 notebook 以非零碼收場（code competition 的 notebook 出錯 = 整份提交作廢），
        # 失敗的格子留 fallback；只有錯誤率超標（上面）才中止。


def summarize(out: Path):
    recs = [json.loads(p.read_text()) for p in sorted((out / "tasks").glob("*.json"))]
    if not recs:
        print("no tasks finished")
        return
    n = len(recs)
    pairs = [r for rec in recs for r in rec["results"]]
    # kaggle 模式（hidden test 沒有答案）的 correct 是 None：那些題不進分數的分母，一題都沒有就不印分數。
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
    """alt 變體的專屬計量：有寫 ALTERNATIVE 的 explorer session 數、其 `transform_alt` 全過 train 的數、
    refiner ATTEMPT 2 的來源（transform_alt 表達式 / 抄既有 candidate / 自造 grid；v8 自造的 36 個只對 1 個）。"""
    ex = [e for rec in recs for e in rec["explorers"]]
    n_alt = sum(1 for e in ex if e["report"]["alt_code"])   # 有 code 的才算：模型常寫 "none (因為…)" 交代為何沒有
    n_alt_pass = sum(1 for e in ex if _all_train_pass(e["alt_check"]))
    src = {"transform_alt": 0, "copy": 0, "novel": 0}
    for rec in recs:
        finals = {e["lineage"]: e for e in rec["explorers"]}   # session 依 hop 順序存，後面的蓋掉前面的
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
                    help="arc：--task-dir/<id>.json；nvarc：--corpus 的 jsonl.zst，用 choose_pairs 切 train/test")
    ap.add_argument("--task-dir", default="")
    ap.add_argument("--challenges", default="",
                    help="--task-source kaggle：比賽的 arc-agi_test_challenges.json，或 <task_id>.json 的目錄"
                         "（本地 dry run；test 帶 output 就算分）")
    ap.add_argument("--submission-out", default="",
                    help="邊跑邊寫 ARC Prize 的 submission.json：建檔先用 test input 填滿，explorer 收完先寫候選、"
                         "refiner 收完覆寫，每次整份重寫")
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="從現在起算的分鐘數。到點不再呼叫模型（含交接報告），explorer 的答案改從 sandbox 現有的 "
                         "transform 收、refiner 不開；沒開始的題不開。0 = 不設限")
    ap.add_argument("--deadline-tok-per-s", type=float, default=30.0,
                    help="接近 deadline 時單輪 max_tokens 的上限 = 剩餘秒數 × 這個值（每條 stream 的保守 decode 速率；"
                         "Kaggle L4 每 replica 4 條時每條 60–90 tok/s，12 條時估 ~30）")
    ap.add_argument("--deadline-rate-adaptive", action="store_true",
                    help="v15：_time_room 的速率改用 min(--deadline-tok-per-s, 最近 200 個完成 call 的 completion/wall 中位數)；"
                         "樣本不到 30 筆時用 --deadline-tok-per-s。預設關")
    ap.add_argument("--time-aware-budget", action="store_true",
                    help="v15：每題開始時依 (全 run 吞吐 × 剩餘時間 × 0.9) / (未開始題數 + 0.5 × 在飛題數) 縮小這題的 task / explorer / "
                         "lineage 預算，預算小時只開 2 條 explorer、更小時不開 refiner；前 1800 s 不改。需要 --deadline-min 與 --compact。預設關")
    ap.add_argument("--dp-size", type=int, default=1,
                    help="server 的 --dp-size；>1 時每題的所有 episode 釘同一個 routed_dp_rank（sglang 的 DP 沒有 "
                         "session affinity，不釘就每輪重 prefill）")
    ap.add_argument("--dp-rank", type=int, default=None, dest="fixed_dp_rank",
                    help="釘全部題到這個 routed_dp_rank。不設且 --dp-size>1 時才按在飛最少分配。"
                         "DP attention 就算只用一個 rank 也要帶，否則 round-robin 跨 rank、前綴快取全空")
    ap.add_argument("--refine-gen-budget", type=int, default=0,
                    help="compact 模式下 refiner 的生成預算；0 = 同 --compact-gen-budget")
    ap.add_argument("--max-error-rate", type=float, default=0.0,
                    help=">0 時：整題失敗的比例超過它就停止發新題、以非零碼（2）結束。伺服器死掉的護欄")
    ap.add_argument("--error-grace", type=int, default=10, help="至少完成這麼多題才開始檢查錯誤率")
    ap.add_argument("--task-list", default="", help="一行一個 task id / uid；nvarc 不給就是整份語料")
    ap.add_argument("--corpus", nargs="*", default=[], help="--task-source nvarc 的 jsonl.zst，可多個（uid 不能重複）；每題記 source")
    ap.add_argument("--attempt", type=int, default=0, help="nvarc 的 pair 抽樣種子（同 rollout_runner 的 attempt）")
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=1)
    ap.add_argument("--hparam-menu", action="store_true",
                    help="每題按 HPARAM_MENU 決定性抽 effort / ctx_cap / K / target_frac / gen_budget，記在 tasks/*.json")
    ap.add_argument("--save-segments", action="store_true",
                    help="每次 compaction 前把整份 messages 存進 segments/<task>.jsonl.gz（蒸餾語料）")
    ap.add_argument("--tasks", default="", help="逗號分隔的 task id 子集")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="跳過 out/tasks 裡已有的題")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", default="xhigh")
    ap.add_argument("--backend", choices=["openai", "dsv4", "tokens"], default="openai",
                    help="dsv4：用模型自帶 encoding 組 prompt 打 /generate（見 dsv4_backend.py）")
    ap.add_argument("--encoding-path", default=None, help="dsv4 backend 用的 encoding_dsv4.py 路徑")
    ap.add_argument("--prompt-variant", choices=list(prompts.VARIANTS), default="default",
                    help="alt：explorer 全過 train 後多一步找反假設（ALTERNATIVE / ATTEMPT 2）、refiner 在 candidates 全是 "
                         "train-pass 時刻意找反例；default 的 prompt 一字不動（見 prompts.py）")
    ap.add_argument("--n-explorers", type=int, default=3)
    ap.add_argument("--base-explorers", type=int, default=2,
                    help="先跑這幾條 explorer；它們的答案已有兩條一致就不開剩下的（設成 n-explorers 即全部並行）")
    ap.add_argument("--task-budget", type=int, default=10**9,
                    help="這題的 gen token 總量（explorer + refine）：到了就不再開新 session / 新 lineage，而且每個新 session 的"
                         "預算收縮成剩餘量（同一批並行的 session 平分）。v7/v8 模擬：省 49k 但砍掉 11 題的唯一正解，不採用；"
                         "reap128 final 120 題：沒解出的 41 題平均 865k、解出的 293k；Kaggle 用 180k")
    ap.add_argument("--lineage-budget", type=int, default=0,
                    help="每條 explorer 跨 session 的生成 token 累計上限（0 = 不限）：每個 session 的 compact_gen_budget = "
                         "min(原值, 剩餘)，不砍正在跑的 session（可超一輪 max_turn_tokens + 交接）")
    ap.add_argument("--min-session-budget", type=int, default=20_000,
                    help="lineage / task 剩餘預算不到這麼多就不開新 session；refine 第 1 輪至少給這麼多")
    ap.add_argument("--pass-gate-hop", type=int, default=99,
                    help="從第幾個 session 起只有 home check 全過 train 的條才能繼續（v9 用 4）")
    ap.add_argument("--max-hops", type=int, default=3,
                    help="每條 explorer 最多幾個 session（交接後在同一個 sandbox 開新 context 接著做）")
    ap.add_argument("--max-refine-rounds", type=int, default=1)
    ap.add_argument("--escalate-below", type=int, default=80,
                    help="refiner 信心低於這個（或沒答完 / 被迫交接）就再開一輪 refine")
    ap.add_argument("--ctx-cap", type=int, default=50_000, help="每條對話的 context 上限（prompt+completion）")
    ap.add_argument("--report-reserve", type=int, default=10_000, help="留給交接（checkpoint tool call + 報告）的 context")
    ap.add_argument("--context-margin", type=int, default=2_500, help="留給 tool 輸出與 template 的餘裕")
    ap.add_argument("--handoff-margin", type=int, default=300)
    ap.add_argument("--min-handoff-tokens", type=int, default=3_500, help="交接報告至少給這麼多 max_tokens（可略超 cap）")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer 路徑，估第一輪 prompt 長度")
    ap.add_argument("--template-overhead", type=int, default=600, help="chat template + tool schema 的固定 token 開銷估計")
    ap.add_argument("--min-turn-tokens", type=int, default=4_000,
                    help="餘額低於這個就直接交接，不再開一輪只會被截斷的小回合")
    ap.add_argument("--max-turn-tokens", type=int, default=20_000,
                    help="單一回覆的 max_tokens；撞到就要求 checkpoint 後繼續（不是交接）")
    ap.add_argument("--max-continuations", type=int, default=4, help="每條對話最多接續幾次")
    ap.add_argument("--compact", action="store_true",
                    help="context 滿了不交接：把最舊的 tool round 折掉、同一條對話續跑（sandbox 狀態不動），"
                         "直到 --compact-gen-budget 或 --max-turns 才寫報告。配 --max-hops 1 用")
    ap.add_argument("--compact-keep-tokens", type=int, default=8_000, help="折掉之後至少保留最近幾輪的估計 token（下限）")
    ap.add_argument("--delegate", action="store_true",
                    help="explorer 多一個 `delegate` 工具：把子問題交給乾淨 context 的 helper（同 sandbox），只拿回短結論")
    ap.add_argument("--delegate-budget", type=int, default=60_000, help="helper 每次的生成 token 預算")
    ap.add_argument("--delegate-max", type=int, default=6, help="每條對話最多 delegate 幾次")
    ap.add_argument("--delegate-max-turns", type=int, default=80)
    ap.add_argument("--delegate-reply-chars", type=int, default=6000, help="helper 回覆塞回 explorer context 的字元上限")
    ap.add_argument("--notes-structured", action="store_true",
                    help="notes 每條帶 RULE:/VERIFIED:/REFUTED:/TODO:/CODE: 標籤；harness 渲染時分組、去重、RULE 只留最新")
    ap.add_argument("--compact-rich", action="store_true",
                    help="compact 模式：先清舊輪的 tool 輸出（留 reasoning/code）再丟整輪；公告帶 `transform` 原始碼與被折掉輪次的一行摘要")
    ap.add_argument("--compact-keep-full-rounds", type=int, default=2, help="--compact-rich：最近幾輪的 tool 輸出保持完整")
    ap.add_argument("--compact-checkpoint", action="store_true",
                    help="compact 模式：折掉舊輪次之前先給模型一輪把即將消失的推理 append 進 `notes`")
    ap.add_argument("--compact-checkpoint-tokens", type=int, default=3000, help="checkpoint 那一輪的 max_tokens")
    ap.add_argument("--compact-target-frac", type=float, default=0.6,
                    help="折完的 context 目標佔工作視窗（ctx_cap − report_reserve − margin）的比例")
    ap.add_argument("--compact-gen-budget", type=int, default=200_000, help="compact 模式下每條對話的生成 token 總預算")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--handoff-thinking", choices=["off", "low", "same"], default="same")
    ap.add_argument("--handoff-tool-calls", type=int, default=2, help="交接時最多再給幾次 tool call 做 checkpoint")
    ap.add_argument("--handoff-turn-tokens", type=int, default=4_000, help="交接期 checkpoint tool call 那一輪的 max_tokens")
    ap.add_argument("--gen-budget-hard", action="store_true",
                    help="compact 模式：單輪 max_tokens 不得超過剩餘生成預算（至少 min_turn_tokens），超支上限從 20k 變 4k")
    ap.add_argument("--skip-refine-when-agree", action="store_true",
                    help="兩條 explorer 都答完且答案一致就不開 refiner，attempt 2 由候選補位（省一整條對話）")
    ap.add_argument("--handoff-budget", type=int, default=0,
                    help="交接階段（checkpoint tool call + 報告 + 拒絕重試）合計的生成上限；0 = 不設限（原行為）。"
                         "Kaggle 用 10k：模型不寫報告時每輪可生到 ctx 餘額，20k 的預算會花成 58k")
    ap.add_argument("--no-fill-attempt2", dest="fill_attempt2", action="store_false",
                    help="refine 只給 1 個 attempt 時，不用 candidates 補第 2 個")
    ap.add_argument("--inline-cells", type=int, default=400, help="candidate grid 小於這麼多格才直接印進 refine prompt")
    ap.add_argument("--max-live", type=int, default=64, help="同時在飛的對話數（含 sandbox）")
    ap.add_argument("--task-live", type=int, default=0, help="同時在跑的題數上限（0 = 不限；建議 max_live/2）")
    ap.add_argument("--http-timeout", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sandbox-python", default=sys.executable)
    ap.add_argument("--no-save-messages", dest="save_messages", action="store_false")
    args = ap.parse_args()
    if args.task_source == "nvarc":
        if not args.corpus:
            raise SystemExit("--task-source nvarc 需要 --corpus")
    elif args.task_source == "kaggle":
        if not args.challenges:
            raise SystemExit("--task-source kaggle 需要 --challenges")
    elif not (args.task_dir and args.task_list):
        raise SystemExit("--task-source arc 需要 --task-dir 與 --task-list")
    if args.dp_size < 1:
        raise SystemExit(f"--dp-size 要 >= 1，拿到 {args.dp_size}")
    if args.fixed_dp_rank is not None:
        if args.fixed_dp_rank < 0:
            raise SystemExit(f"--dp-rank 要 >= 0，拿到 {args.fixed_dp_rank}")
        if args.dp_size > 1 and args.fixed_dp_rank >= args.dp_size:
            raise SystemExit(
                f"--dp-rank {args.fixed_dp_rank} 超出 --dp-size {args.dp_size}"
            )
    if (args.lineage_budget or args.refine_gen_budget or args.gen_budget_hard) and not args.compact:
        raise SystemExit("--lineage-budget / --refine-gen-budget / --gen-budget-hard 是 compact 模式的旋鈕，要配 --compact")
    if args.lineage_budget and args.lineage_budget < args.min_session_budget:
        raise SystemExit("--lineage-budget 小於 --min-session-budget，第一個 session 就開不了")
    if args.task_budget < args.base_explorers * args.min_session_budget:
        raise SystemExit("--task-budget 不夠 base explorers 各開一個 --min-session-budget 的 session")
    if args.time_aware_budget:
        if args.deadline_min <= 0:
            raise SystemExit("--time-aware-budget 需要 --deadline-min")
        if not args.compact or not args.lineage_budget:
            raise SystemExit("--time-aware-budget 縮的是 compact 模式的 task / explorer / lineage 預算，要配 --compact 與 --lineage-budget")
    args.stream_stats = StreamStats()   # --deadline-rate-adaptive / --time-aware-budget 時由 main_async 包 chat 填入
    args.skip_refiner = False           # --time-aware-budget 每題在 main_async 的 targs 上覆寫
    args.dp_rank = None           # 每題在 main_async 裡指定
    args.should_abort = None      # main_async 接上錯誤率護欄
    args.submission = None        # --submission-out 時是 SubmissionStore
    if args.hparam_menu and not args.compact:
        raise SystemExit("--hparam-menu 抽的軸是 compact 模式的旋鈕，要配 --compact")
    cfg = load_config()
    args.tool = cfg["tool"]
    args.tool_name = cfg["tool_name"]
    args.extra_tools = [prompts.DELEGATE_TOOL] if args.delegate else []
    args.tool_timeout = cfg["rollout"]["tool_timeout_s"]
    args.tool_max_output_chars = cfg["rollout"]["tool_max_output_chars"]
    args.sampling = Sampling()
    args.annotate_ids = None      # tokens backend 會換成 TokenBackend.annotate
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.count_tokens = lambda text: len(tok(text)["input_ids"])
    if args.backend == "dsv4":
        from dsv4_backend import DSV4_EFFORTS, DSv4Backend
        if args.encoding_path is None:
            raise SystemExit("--backend dsv4 需要 --encoding-path")
        if args.reasoning_effort not in DSV4_EFFORTS:
            raise SystemExit(f"dsv4 的 --reasoning-effort 只有 {DSV4_EFFORTS}")
        if args.handoff_thinking == "off":
            raise SystemExit("dsv4 backend 沒有 enable_thinking=False；--handoff-thinking 用 same 或 low")
        backend = DSv4Backend(args.base_url, args.encoding_path, tok, args.retries)
        args.chat = backend.chat
    elif args.backend == "tokens":
        from token_backend import EFFORTS, TokenBackend
        if args.reasoning_effort not in EFFORTS:
            raise SystemExit(f"tokens backend 的 --reasoning-effort 只有 {EFFORTS}")
        if args.handoff_thinking != "same":
            raise SystemExit("tokens backend 的 --handoff-thinking 只支援 same（prompt 前綴要固定）")
        backend = TokenBackend(args.base_url, tok, [args.tool, *args.extra_tools], args.retries)
        args.chat = backend.chat
        args.annotate_ids = backend.annotate
    else:
        async def _openai_chat(client, payload):
            return await _chat(client, args.base_url, payload, args.retries)
        args.chat = _openai_chat
    if args.ctx_cap - args.report_reserve - args.context_margin < 10_000:
        raise SystemExit("ctx_cap 扣掉 reserve/margin 後不到 10k，沒得解")
    # deadline 從這裡起算（tokenizer / backend 都載好了），呼叫端給的分鐘數才全是跑題的時間。
    args.deadline = time.monotonic() + args.deadline_min * 60 if args.deadline_min > 0 else None
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

