"""跑 TIR rollout 的引擎：多輪對話 + python sandbox，回傳解出來的 grid。

與 `work/expert_prune/eval_tir_prune.py`（2026-08-22 那批 eval 的凍結 provenance）相比，
這裡多了三件提交才需要的東西：

1. **每條 rollout 的生成量上限**（`Budget.gen_cap`）。b200 的 334 條 transcript 顯示答對的鏈
   平均 59.5k token、答錯的 112.2k，所以砍尾巴主要砍掉的是已經卡住的 rollout。上限是把
   固定的牆鐘預算換成「跑幾條」的唯一旋鈕。
2. **全域 deadline**。比賽 rerun 到點就砍，`submission.json` 沒寫出來等於 0 分，
   所以排程必須在還來得及的時候停止發新的 rollout。
3. **sandbox 呼叫改走 thread**。原版直接在 coroutine 裡呼叫阻塞的 `sb.run()`，
   一個 20 秒的 tool call 會把整個 event loop 連同其他數十條 rollout 一起卡住。
   L4 的 CPU 比 b200 少得多，這在提交環境會是實際傷害。

答案來源分 `content` 與 `reasoning` 兩種並分開記：受過 SFT 的模型應該把 grid 放進 content，
放在 thinking block 裡代表協定壞了（量化或剪枝過頭的典型症狀），那是要看見的訊號而不是
要默默吸收的細節。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from . import protocol
from .parsing import backscan_json_parser


@dataclass
class Budget:
    """一條 rollout 花得起多少。"""

    gen_cap: int = 131_072
    """整條 rollout 的生成 token 上限（累加每輪 completion_tokens）。"""
    max_turns: int = 120
    max_tokens: int = 65_536
    """單輪生成上限，再被 context 與 gen_cap 夾一次。"""
    context_length: int = 262_144
    """必須等於 serving 的 --context-length。"""
    context_margin: int = 6_000
    """留給 tool 輸出與 chat template 的餘裕。"""
    min_turn_tokens: int = 2_048
    """低於這個數就沒必要再發一輪。"""


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    """預設與 teacher 產生訓練資料時一致（`tir_config.json` 的 sampling 段：
    temperature 1.0 / top_p 0.95，**沒有 top_k**）。調低就偏離訓練分布。

    `top_k` 一定要明確送出，不能不送。sglang 的 `--sampling-defaults` 預設是 `model`，
    會把 checkpoint 的 `generation_config.json` 當成沒指定欄位的預設值——而這顆的
    generation_config 寫著 `top_k: 20`。不送就等於在 248k 詞表上只留前 20 個候選，
    十萬 token 的推理鏈會變得重複且收不了尾，而且**請求裡看不出來**。
    -1 表示不截斷。"""


@dataclass
class Rollout:
    task_id: str
    test_idx: int
    attempt: int
    train_pairs: list[dict]
    test_input: list[list[int]]
    expected: list[list[int]] | None = None
    """本地 dry run 才有；比賽 test set 沒有答案。"""

    @property
    def uid(self) -> str:
        return f"{self.task_id}#{self.test_idx}"


@dataclass
class Result:
    task_id: str
    test_idx: int
    attempt: int
    grid: list[list[int]] | None
    answer_src: str
    stop_reason: str
    n_turns: int
    n_tool_calls: int
    n_tool_errors: int
    gen_tokens: int
    latency_s: float
    correct: bool | None = None
    replica: str = ""
    """這條 rollout 從頭到尾用的那個 base_url（DP 下附 #dpN）。看負載有沒有歪掉。"""
    n_prompt_tokens: int = 0
    n_cached_tokens: int = 0
    """整條 rollout 累加的 prompt / 其中命中前綴快取的 token 數。

    需要 server 開 `--enable-cache-report`，否則 `prompt_tokens_details` 是 null、
    這個值恆為 0。**這是整條管線最重要的健康指標**：TIR 每輪重送完整歷史，靠
    radix cache 命中那 ~130k 前綴。命中率一崩，每輪都變成整條 re-prefill——慢十倍
    且不報錯，而 Kaggle 上看不到 server log。有了它就不必靠 log 也看得見。"""
    messages: list[dict] = field(default_factory=list, repr=False)


async def _chat(client, base_url, payload, retries, backoff=3.0):
    """4xx 直接炸——那是我們自己的請求有問題，重試只會浪費 deadline。"""
    for retry in range(retries + 1):
        try:
            r = await client.post(f"{base_url}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status and status < 500) or retry == retries:
                raise
            await asyncio.sleep(backoff * (retry + 1))
    raise AssertionError("unreachable")


async def run_rollout(
    client: httpx.AsyncClient,
    rollout: Rollout,
    *,
    base_url: str,
    model: str,
    cfg: dict,
    sandbox_cls: type,
    budget: Budget,
    sampling: Sampling,
    deadline: float | None = None,
    retries: int = 3,
    keep_messages: bool = False,
    should_abort: Callable[[], bool] | None = None,
    dp_rank: int | None = None,
) -> Result:
    """跑一條 rollout。`deadline` 是 `time.monotonic()` 座標。

    `should_abort` 與 deadline 走同一條路：呼叫端用它表達「整個 run 已經沒救了」
    （例如 server 死掉、每條 rollout 都是 ConnectError），讓在飛的 rollout 收手
    而不是把剩下的預算全部燒在重試上。
    """
    sys_prompt = protocol.system_prompt(cfg, budget.max_turns)
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": protocol.render_user_prompt(
            rollout.train_pairs, rollout.test_input)},
    ]
    n_turns = n_calls = n_errs = 0
    gen_used = 0
    prompt_tok = cached_tok = 0
    last_total = 0
    stop = "turn_cap"
    grid: list[list[int]] | None = None
    answer_src = ""
    t0 = time.monotonic()

    with sandbox_cls(
        rollout.train_pairs,
        rollout.test_input,
        timeout=cfg["rollout"]["tool_timeout_s"],
        max_output_chars=cfg["rollout"]["tool_max_output_chars"],
    ) as sb:
        for _turn in range(budget.max_turns):
            if deadline is not None and time.monotonic() >= deadline:
                stop = "deadline"
                break
            if should_abort is not None and should_abort():
                stop = "aborted"
                break

            # 三個上限取最小：單輪上限、context 餘額、整條 rollout 的生成餘額。
            # last_total 是上一輪的 total_tokens，即這輪 prompt 在附上 tool 結果前的長度。
            ctx_budget = budget.context_length - last_total - budget.context_margin
            gen_budget = budget.gen_cap - gen_used
            mt = min(budget.max_tokens, ctx_budget, gen_budget)
            if mt < budget.min_turn_tokens:
                stop = "gen_cap" if gen_budget <= ctx_budget else "context_exhausted"
                break

            n_turns += 1
            payload = {
                "model": model,
                "messages": messages,
                "tools": [cfg["tool"]],
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "max_tokens": mt,
            }
            if dp_rank is not None:
                # 這條 rollout 的每一輪都送同一個 rank，radix cache 才留得住前綴。
                # rank 不合法時 sglang 回 ValueError（4xx），_chat 不會重試——那是
                # 我們自己的請求有問題，重試只會浪費 deadline。
                payload["routed_dp_rank"] = dp_rank
            data = await _chat(client, base_url, payload, retries)

            usage = data.get("usage") or {}
            last_total = int(usage.get("total_tokens") or last_total)
            gen_used += int(usage.get("completion_tokens") or 0)
            prompt_tok += int(usage.get("prompt_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            cached_tok += int(details.get("cached_tokens") or 0)

            msg = data["choices"][0]["message"]
            finish = data["choices"][0].get("finish_reason")
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # reasoning_content 一定要回灌。訓練資料保留了每一輪的 <think>（TIR 只有一次
                # user query，chat template 會為 last_query_index 之後的每則 assistant 訊息
                # 渲染 thinking；tool 結果是 role "tool"，不算 query）。丟掉的話先前每一輪都
                # 會被渲染成空的 <think>\n\n</think>，模型就被 condition 在一段它訓練時沒見過
                # 的推理歷史上。
                asst: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                }
                if msg.get("reasoning_content"):
                    asst["reasoning_content"] = msg["reasoning_content"]
                messages.append(asst)

                for call in tool_calls:
                    n_calls += 1
                    fn = call["function"]
                    if fn["name"] != cfg["tool_name"]:
                        n_errs += 1
                        out = f"Error: no tool named {fn['name']!r}."
                    else:
                        args = fn["arguments"]
                        code = (json.loads(args) if isinstance(args, str) else args)["code"]
                        # 阻塞的 subprocess 往返移出 event loop，否則一個慢的 tool call
                        # 會連坐所有同時在飛的 rollout。
                        out, timed = await asyncio.to_thread(sb.run, code)
                        n_errs += int(timed)
                    messages.append({"role": "tool",
                                     "tool_call_id": call.get("id", ""),
                                     "content": out})
                if finish == "length":
                    # 這一輪的 max_tokens 是三個上限取最小，所以 finish_reason="length"
                    # 可能只是「整條 rollout 的生成餘額用完了」。照字面記成 length 會讓
                    # Kaggle 上的 stop_reason 直接說謊：看起來像模型話太多被單輪截斷，
                    # 實際上是 gen_cap 設太緊。這兩者要調的旋鈕不同。
                    stop = "gen_cap" if gen_used >= budget.gen_cap else "length"
                    break
                continue

            # 沒有 tool call = 最終答案。
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            grid = backscan_json_parser(msg.get("content") or "")
            answer_src = "content" if grid is not None else ""
            if grid is None:
                grid = backscan_json_parser(msg.get("reasoning_content") or "")
                answer_src = "reasoning" if grid is not None else ""
            if grid is not None:
                stop = "answered" if answer_src == "content" else "answered_in_reasoning"
            elif finish == "length":
                stop = "gen_cap" if gen_used >= budget.gen_cap else "length"
            else:
                stop = "malformed"
            break

    # 被 cap／deadline／turn_cap 中斷的，最後一則 assistant 訊息裡可能已經有答案了。
    if grid is None and messages:
        for m in reversed(messages):
            if m["role"] != "assistant":
                continue
            for key, src in (("content", "content"), ("reasoning_content", "reasoning")):
                found = backscan_json_parser(m.get(key) or "")
                if found is not None:
                    grid, answer_src = found, f"salvage_{src}"
                    break
            break

    correct = None if rollout.expected is None else (grid == rollout.expected)
    return Result(
        task_id=rollout.task_id, test_idx=rollout.test_idx, attempt=rollout.attempt,
        grid=grid, answer_src=answer_src, stop_reason=stop, n_turns=n_turns,
        n_tool_calls=n_calls, n_tool_errors=n_errs, gen_tokens=gen_used,
        latency_s=time.monotonic() - t0, correct=correct,
        n_prompt_tokens=prompt_tok, n_cached_tokens=cached_tok,
        messages=messages if keep_messages else [],
    )


async def run_all(
    rollouts: list[Rollout],
    *,
    base_url: str | Sequence[str],
    model: str,
    cfg: dict,
    sandbox_cls: type,
    budget: Budget,
    sampling: Sampling,
    concurrency: int,
    dp_size: int = 1,
    deadline: float | None = None,
    on_result: Callable[[Result], None] | None = None,
    http_timeout: float = 3600.0,
    retries: int = 3,
    keep_messages: bool = False,
    should_abort: Callable[[], bool] | None = None,
) -> list[Result]:
    """照順序發 rollout，每完成一條就回呼一次。到 deadline 就取消還在飛的。

    `on_result` 是提交管線把答案落地的地方——不要等全部跑完才寫檔，跑到一半被砍是
    預期中的結局而不是意外。

    Data parallel：一條 rollout 從頭到尾必須落在同一個 DP worker
    ----------------------------------------------------------
    這不是最佳化，是正確性需求。TIR 每一輪都把完整歷史重送，靠那個 worker 的 radix
    cache 命中 ~130k 的前綴、只 prefill 新的 tool 輸出。**sglang 的 DP 預設是
    round-robin，沒有 session affinity**（`data_parallel_controller.py` 的
    `round_robin_scheduler`），所以第二輪很可能落到另一個 worker——那邊的 cache 裡
    什麼都沒有，整條 context 重新 prefill。而且不報錯，只是慢十倍。

    釘法有兩種，這裡都支援，用「路由目標」統一表示：

    * `dp_size > 1`：單一 server 開 `--dp-size`，每個 request 帶
      `routed_dp_rank`。`DataParallelController.maybe_external_dp_rank_routing`
      在 round-robin **之前**短路（0.5.17 實測），而且 rank 不合法會 `ValueError`
      而不是默默 fallback。這是首選——一個 process、一個 port。
    * `base_url` 給一串：各自獨立的 server，各自一個 URL。

    指派用「當下 in-flight 最少的目標」，在取得全域併發名額的當下決定一次，之後不再
    更動。**刻意不用 hash(session_id) % dp_size**：那是給無狀態/分散式 client 的做法，
    而我們是單一 driver、知道每個目標當下的 in-flight 數，least-loaded 嚴格更均勻
    （hash 會讓幾條長 rollout 剛好撞在同一個 rank 上）。`Result.replica` 記下用的是
    哪一個，負載歪掉看得見。
    """
    urls = [base_url] if isinstance(base_url, str) else list(base_url)
    if not urls:
        raise ValueError("base_url 是空的")
    if dp_size < 1:
        raise ValueError(f"dp_size 要 >= 1，拿到 {dp_size}")
    # 路由目標 = (url, dp_rank)。dp_size == 1 時 rank 是 None，請求裡就不帶那個欄位。
    targets = [(u, r if dp_size > 1 else None) for u in urls for r in range(dp_size)]
    # httpx 預設 Limits(max_connections=100, max_keepalive_connections=20)，會在
    # concurrency > 100 時靜默卡在 100；keepalive 太小則讓連線在每一輪之間被拆掉重撥。
    limits = httpx.Limits(max_connections=concurrency + 16,
                          max_keepalive_connections=concurrency + 16)
    sem = asyncio.Semaphore(concurrency)
    inflight = [0] * len(targets)
    results: list[Result] = []

    async with httpx.AsyncClient(timeout=http_timeout, limits=limits) as client:
        async def one(r: Rollout) -> None:
            async with sem:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if should_abort is not None and should_abort():
                    return
                idx = min(range(len(targets)), key=lambda i: inflight[i])
                inflight[idx] += 1
                url, dp_rank = targets[idx]
                try:
                    res = await run_rollout(
                        client, r, base_url=url, model=model, cfg=cfg,
                        sandbox_cls=sandbox_cls, budget=budget, sampling=sampling,
                        deadline=deadline, retries=retries, keep_messages=keep_messages,
                        should_abort=should_abort, dp_rank=dp_rank)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    res = Result(task_id=r.task_id, test_idx=r.test_idx, attempt=r.attempt,
                                 grid=None, answer_src="", stop_reason=f"error:{type(e).__name__}",
                                 n_turns=0, n_tool_calls=0, n_tool_errors=0,
                                 gen_tokens=0, latency_s=0.0)
                finally:
                    inflight[idx] -= 1
                res.replica = url if dp_rank is None else f"{url}#dp{dp_rank}"
            results.append(res)
            if on_result is not None:
                on_result(res)

        tasks = [asyncio.create_task(one(r)) for r in rollouts]
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return results

