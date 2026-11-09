"""The engine that runs TIR rollouts: multi-turn chat + a python sandbox, returning the solved grid.

Compared with `work/expert_prune/eval_tir_prune.py` (the frozen provenance of the 2026-08-22 eval batch),
this adds three things only a submission needs:

1. **A per-rollout generation cap** (`Budget.gen_cap`). The 334 B200 transcripts show correct chains average
   59.5k tokens and wrong ones 112.2k, so cutting the tail mostly cuts rollouts that are already stuck. The cap
   is the only knob that converts a fixed wall-clock budget into "how many rollouts".
2. **A global deadline**. The competition rerun is killed on time, and an unwritten `submission.json` scores 0,
   so scheduling must stop issuing new rollouts while there is still time.
3. **Sandbox calls go through a thread**. The original called the blocking `sb.run()` directly inside the
   coroutine, so one 20-second tool call stalled the whole event loop along with dozens of other rollouts.
   L4 machines have far fewer CPUs than the B200 nodes, so in the submission environment this does real damage.

Answer sources are `content` and `reasoning`, recorded separately: an SFT-trained model should put the grid in
content; a grid in the thinking block means the protocol broke (the typical symptom of over-quantization or
over-pruning) -- a signal to be seen, not a detail to be silently absorbed.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from . import protocol
from .parsing import backscan_json_parser


@dataclass
class Budget:
    """How much one rollout can afford."""

    gen_cap: int = 131_072
    """Generated-token cap for the whole rollout (completion_tokens summed over turns)."""
    max_turns: int = 120
    max_tokens: int = 65_536
    """Per-turn generation cap, further clamped by the context and gen_cap."""
    context_length: int = 262_144
    """Must equal the serving --context-length."""
    context_margin: int = 6_000
    """Headroom left for tool output and the chat template."""
    min_turn_tokens: int = 2_048
    """Below this there is no point issuing another turn."""


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    """Defaults match the teacher's when it generated the training data (the sampling section of `tir_config.json`:
    temperature 1.0 / top_p 0.95, **no top_k**). Lowering them departs from the training distribution.

    `top_k` must always be sent explicitly, never omitted. sglang's `--sampling-defaults` defaults to `model`, which
    uses the checkpoint's `generation_config.json` as the default for unspecified fields -- and this checkpoint's
    generation_config says `top_k: 20`. Not sending it means keeping only the top 20 candidates of a 248k vocabulary,
    and 100k-token reasoning chains become repetitive and fail to wrap up, **with nothing visible in the request**.
    -1 means no truncation."""


@dataclass
class Rollout:
    task_id: str
    test_idx: int
    attempt: int
    train_pairs: list[dict]
    test_input: list[list[int]]
    expected: list[list[int]] | None = None
    """Only in local dry runs; the competition test set has no answers."""

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
    """The base_url this rollout used from start to end (with #dpN under DP). Shows whether the load is skewed."""
    n_prompt_tokens: int = 0
    n_cached_tokens: int = 0
    """Prompt tokens summed over the whole rollout / how many of them hit the prefix cache.

    Needs the server started with `--enable-cache-report`, otherwise `prompt_tokens_details` is null and this
    stays 0. **This is the most important health metric of the whole pipeline**: TIR resends the full history every
    turn and relies on the radix cache hitting that ~130k prefix. If the hit rate collapses, every turn becomes a
    full re-prefill -- ten times slower with no error, and server logs are not visible on Kaggle. With this it is
    visible without the logs."""
    messages: list[dict] = field(default_factory=list, repr=False)
    session_id: str = ""
    session_close_status: int | None = None
    session_close_error: str = ""
    error: str = ""


async def _chat(client, base_url, payload, retries, backoff=3.0):
    """A 4xx blows up immediately -- it means our own request is wrong, and retrying would only waste the deadline."""
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


async def _run_rollout(
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
    session_id: str | None = None,
    reasoning_effort: str | None = None,
    force_first_tool: bool = False,
    return_token_ids: bool = False,
) -> Result:
    """Run one rollout. `deadline` is in `time.monotonic()` coordinates.

    `should_abort` takes the same path as the deadline: the caller uses it to say "the whole run is beyond saving"
    (e.g. the server died and every rollout gets ConnectError), so rollouts in flight stop instead of burning the
    remaining budget on retries.
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

            # Take the minimum of three caps: the per-turn cap, the remaining context, the rollout's remaining generation budget.
            # last_total is the previous turn's total_tokens, i.e. this turn's prompt length before the tool results are appended.
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
                # Every turn of this rollout goes to the same rank so the radix cache keeps the prefix.
                # An invalid rank makes sglang return ValueError (4xx) and _chat does not retry -- that would
                # be our own request being wrong, and retrying would only waste the deadline.
                payload["routed_dp_rank"] = dp_rank
            if session_id is not None:
                payload["session_id"] = session_id
                payload["return_meta_info"] = True
            if return_token_ids:
                payload["return_token_ids"] = True
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            if force_first_tool and _turn == 0:
                payload["tool_choice"] = "required"
            data = await _chat(client, base_url, payload, retries)
            if session_id is not None and data["choices"][0]["meta_info"]["dp_rank"] != dp_rank:
                raise RuntimeError("HiSparse response arrived from a different attention DP")

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
                # reasoning_content must be fed back. The training data kept every turn's <think> (TIR has a single
                # user query, and the chat template renders thinking for every assistant message after
                # last_query_index; tool results are role "tool", not queries). Dropping it would render every
                # earlier turn as an empty <think>\n\n</think>, conditioning the model on a reasoning history it never
                # saw in training.
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
                        # Move the blocking subprocess round trip off the event loop, otherwise one slow tool call
                        # drags down every rollout in flight.
                        out, timed = await asyncio.to_thread(sb.run, code)
                        n_errs += int(timed)
                    messages.append({"role": "tool",
                                     "tool_call_id": call.get("id", ""),
                                     "content": out})
                if finish == "length":
                    # This turn's max_tokens is the minimum of three caps, so finish_reason="length" may just mean
                    # "the rollout's generation budget ran out". Recording it literally as length would make the
                    # stop_reason on Kaggle lie outright: it would look like the model talked too much and was cut
                    # per turn, when actually gen_cap was set too tight. The two need different knobs.
                    stop = "gen_cap" if gen_used >= budget.gen_cap else "length"
                    break
                continue

            # No tool call = final answer.
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

    # When interrupted by cap / deadline / turn_cap, the last assistant message may already contain an answer.
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
    session_id: str | None = None,
    session_close_timeout: float = 15,
    reasoning_effort: str | None = None,
    force_first_tool: bool = False,
    return_token_ids: bool = False,
) -> Result:
    """Optional HiSparse SID: fixed DP, no ambiguous retries, bounded own-SID close."""
    params = dict(base_url=base_url, model=model, cfg=cfg, sandbox_cls=sandbox_cls,
                  budget=budget, sampling=sampling, deadline=deadline,
                  retries=retries if session_id is None else 0, keep_messages=keep_messages,
                  should_abort=should_abort, dp_rank=dp_rank, session_id=session_id,
                  reasoning_effort=reasoning_effort, force_first_tool=force_first_tool,
                  return_token_ids=return_token_ids)
    if session_id is None:
        return await _run_rollout(client, rollout, **params)
    if not session_id or type(dp_rank) is not int or dp_rank < 0 or session_close_timeout <= 0:
        raise ValueError("HiSparse needs a nonempty owned SID, fixed nonnegative DP, and positive close timeout")
    if not base_url.rstrip("/").endswith("/v1"):
        raise ValueError("HiSparse base_url must end with /v1 for its sibling /close_session control")
    primary = None
    cleanup = {"session_id": session_id, "status": None, "error": ""}
    try:
        result = await _run_rollout(client, rollout, **params)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            async with asyncio.timeout(session_close_timeout):
                response = await client.post(base_url.rstrip("/")[:-3] + "/close_session",
                                             json={"session_id": session_id}, timeout=session_close_timeout)
                response.raise_for_status()
                if response.content.strip() not in (b"", b"null"):
                    raise RuntimeError(f"unexpected close_session ACK: {response.content!r}")
                cleanup["status"] = response.status_code
        except Exception as exc:
            cleanup["error"] = f"{type(exc).__name__}: {exc}"
            if primary is not None:
                primary.add_note(f"owned session close failed: {cleanup['error']}")
        if primary is not None:
            primary.hisparse_session_cleanup = cleanup
    result.session_id = session_id
    result.session_close_status = cleanup["status"]
    result.session_close_error = cleanup["error"]
    return result


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
    hisparse_sessions: bool = False,
    session_close_timeout: float = 15,
    reasoning_effort: str | None = None,
    force_first_tool: bool = False,
    return_token_ids: bool = False,
    event_hooks: dict | None = None,
) -> list[Result]:
    """Issue rollouts in order, calling back once per completed rollout. At the deadline, cancel those still in flight.

    `on_result` is where the submission pipeline persists answers -- do not wait for everything to finish before writing
    the file; being killed halfway is an expected ending, not an accident.

    Data parallel: a rollout must stay on the same DP worker from start to end
    --------------------------------------------------------------------------
    This is not an optimisation, it is a correctness requirement. TIR resends the full history every turn and relies on
    that worker's radix cache hitting the ~130k prefix so only the new tool output is prefilled. **sglang's DP defaults
    to round-robin with no session affinity** (`round_robin_scheduler` in `data_parallel_controller.py`), so the second
    turn very likely lands on another worker -- whose cache holds nothing, and the whole context is prefilled again.
    With no error, just ten times slower.

    There are two ways to pin, both supported here and expressed uniformly as "routing targets":

    * `dp_size > 1`: a single server started with `--dp-size`, every request carrying `routed_dp_rank`.
      `DataParallelController.maybe_external_dp_rank_routing` short-circuits **before** round-robin (measured on
      0.5.17), and an invalid rank raises `ValueError` instead of silently falling back. This is preferred -- one
      process, one port.
    * `base_url` given as a list: independent servers, one URL each.

    Assignment uses "the target with the fewest requests in flight right now", decided once when the global concurrency
    slot is acquired and never changed afterwards. **Deliberately not hash(session_id) % dp_size**: that is for stateless
    / distributed clients, whereas we are a single driver that knows each target's in-flight count, and least-loaded is
    strictly more even (hashing lets a few long rollouts collide on the same rank). `Result.replica` records which one
    was used, so skewed load is visible.
    """
    urls = [base_url] if isinstance(base_url, str) else list(base_url)
    if not urls:
        raise ValueError("base_url is empty")
    if dp_size < 1:
        raise ValueError(f"dp_size must be >= 1, got {dp_size}")
    # Routing target = (url, dp_rank). With dp_size == 1 the rank is None and the request does not carry that field.
    if hisparse_sessions and (session_close_timeout <= 0 or any(not u.rstrip("/").endswith("/v1") for u in urls)):
        raise ValueError("HiSparse needs /v1 endpoints and a positive session close timeout")
    targets = [(u, r if dp_size > 1 or hisparse_sessions else None) for u in urls for r in range(dp_size)]
    # httpx defaults to Limits(max_connections=100, max_keepalive_connections=20), which silently caps at 100 when
    # concurrency > 100; too small a keepalive pool tears connections down and redials them between turns.
    limits = httpx.Limits(max_connections=concurrency + 16,
                          max_keepalive_connections=concurrency + 16)
    sem = asyncio.Semaphore(concurrency)
    inflight = [0] * len(targets)
    results: list[Result] = []

    async with httpx.AsyncClient(timeout=http_timeout, limits=limits, event_hooks=event_hooks) as client:
        async def one(r: Rollout) -> None:
            async with sem:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if should_abort is not None and should_abort():
                    return
                idx = min(range(len(targets)), key=lambda i: inflight[i])
                inflight[idx] += 1
                url, dp_rank = targets[idx]
                sid = "tir-" + uuid.uuid4().hex if hisparse_sessions else None
                try:
                    res = await run_rollout(
                        client, r, base_url=url, model=model, cfg=cfg,
                        sandbox_cls=sandbox_cls, budget=budget, sampling=sampling,
                        deadline=deadline, retries=retries, keep_messages=keep_messages,
                        should_abort=should_abort, dp_rank=dp_rank, session_id=sid,
                        session_close_timeout=session_close_timeout, reasoning_effort=reasoning_effort,
                        force_first_tool=force_first_tool, return_token_ids=return_token_ids)
                except asyncio.CancelledError as exc:
                    if hisparse_sessions:
                        cleanup = exc.hisparse_session_cleanup
                        res = Result(task_id=r.task_id, test_idx=r.test_idx, attempt=r.attempt,
                                     grid=None, answer_src="", stop_reason="deadline_cancelled",
                                     n_turns=0, n_tool_calls=0, n_tool_errors=0, gen_tokens=0, latency_s=0,
                                     session_id=sid, session_close_status=cleanup["status"],
                                     session_close_error=cleanup["error"])
                        res.replica = f"{url}#dp{dp_rank}"
                        results.append(res)
                        if on_result is not None:
                            on_result(res)
                    raise
                except Exception as e:  # noqa: BLE001
                    res = Result(task_id=r.task_id, test_idx=r.test_idx, attempt=r.attempt,
                                 grid=None, answer_src="", stop_reason=f"error:{type(e).__name__}",
                                 n_turns=0, n_tool_calls=0, n_tool_errors=0,
                                 gen_tokens=0, latency_s=0.0, error=str(e))
                    if hisparse_sessions:
                        cleanup = e.hisparse_session_cleanup
                        res.session_id, res.session_close_status = sid, cleanup["status"]
                        res.session_close_error = cleanup["error"]
                finally:
                    inflight[idx] -= 1
                res.replica = url if dp_rank is None else f"{url}#dp{dp_rank}"
            results.append(res)
            if on_result is not None:
                on_result(res)

        tasks = [asyncio.create_task(one(r)) for r in rollouts]
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            await asyncio.wait(tasks, timeout=timeout)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    raise outcome  # Callback/persistence failures must reach the caller.
    return results
