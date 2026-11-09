"""Offline ARC2 TIR runner with owned HiSparse sessions; never calls a Kaggle API.

Save&Run defaults to two public validation tasks. The output still contains every
challenge/test and both attempts, with ungenerated cells explicitly counted as
fallbacks. competition-rerun processes the entire supplied test roster.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from arcforge.tir import driver, protocol
from arcforge.tir.sandbox import Sandbox
from arcforge.tir.submission import SubmissionStore, validate_grid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_challenges(path):
    challenges = json.loads(path.read_text())
    require(isinstance(challenges, dict) and challenges, "challenge roster must be a nonempty object")
    for task_id, task in challenges.items():
        require(isinstance(task_id, str) and isinstance(task, dict), "invalid task")
        require(isinstance(task["train"], list) and task["train"]
                and isinstance(task["test"], list) and task["test"], f"{task_id}: empty train/test")
        for pair in task["train"]:
            require(validate_grid(pair["input"]) is None and validate_grid(pair["output"]) is None,
                    f"{task_id}: invalid training grid")
        for test in task["test"]:
            require(validate_grid(test["input"]) is None, f"{task_id}: invalid test input")
    return challenges


def validate_submission(path, challenges):
    data = json.loads(path.read_text())
    require(set(data) == set(challenges), "submission task roster differs from challenges")
    for task_id, task in challenges.items():
        require(len(data[task_id]) == len(task["test"]), f"{task_id}: wrong test count")
        for cell in data[task_id]:
            require(set(cell) == {"attempt_1", "attempt_2"}, f"{task_id}: needs exactly two attempts")
            require(all(validate_grid(grid) is None for grid in cell.values()), f"{task_id}: invalid attempt grid")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Trace:
    def __init__(self, output):
        self.root, self.count, self.phase = output / "http", 0, "preflight"
        self.root.mkdir()
        self.chats = {}

    async def request(self, request):
        self.count += 1
        request.extensions["arcforge_trace_seq"] = self.count
        directory = self.root / f"{self.count:05d}"
        directory.mkdir()
        body = await request.aread()
        (directory / "request.body").write_bytes(body)
        write_json(directory / "request.json", {"method": request.method, "url": str(request.url),
                                                "phase": self.phase, "monotonic_s": time.monotonic()})

    async def response(self, response):
        directory = self.root / f"{response.request.extensions['arcforge_trace_seq']:05d}"
        raw = await response.aread()
        (directory / "response.body").write_bytes(raw)
        write_json(directory / "response.json", {"status": response.status_code, "monotonic_s": time.monotonic()})
        if response.request.url.path.endswith("/chat/completions") and response.status_code == 200:
            body = json.loads(response.request.content)
            self.chats.setdefault(body["session_id"], []).append(json.loads(raw))

    @property
    def hooks(self):
        return {"request": [self.request], "response": [self.response]}


async def preflight(args, trace):
    async with httpx.AsyncClient(timeout=15, event_hooks=trace.hooks) as client:
        response = await client.get(args.base_url.rstrip("/")[:-3] + "/server_info")
        response.raise_for_status()
        info = response.json()
    require(info["tp_size"] == info["ep_size"] == 4 and info["dp_size"] == args.dp_size == 2
            and info["enable_dp_attention"], "runner requires TP4/EP4 attention-DP2")
    require(info["radix_cache_backend"] == "qsa_hisparse_session" and info["enable_session_radix_cache"],
            "runner requires the retained HiSparse session backend")
    require(info["context_length"] >= args.context_length, "requested client context exceeds server cap")
    config = info["hisparse_config"]
    config = json.loads(config) if isinstance(config, str) else config
    require(config["qsa_session_resume"] and config["qsa_session_idle_timeout_s"] == 0,
            "runner requires retained sessions with expiry disabled")
    per_dp = math.ceil(args.concurrency / args.dp_size)
    needed = per_dp * (math.ceil(args.context_length / info["page_size"]) * info["page_size"] + info["page_size"])
    states = {}
    for state in info["internal_states"]:
        ledger = state["qsa_hisparse"]
        if ledger["attn_tp_rank"] != 0:
            continue
        dp = ledger["dp_rank"]
        require(dp in (0, 1) and dp not in states, "ambiguous DP pool receipt")
        require(not ledger["running_requests"], "server already has active model requests")
        require(state["effective_max_running_requests_per_dp"] >= per_dp,
                f"DP{dp} cannot admit requested client concurrency")
        require(ledger["logical_available_tokens"] >= needed,
                f"DP{dp} lacks {needed} logical slots; choose an explicit client/server profile")
        require(ledger["mamba_available_slots"] >= per_dp, f"DP{dp} lacks live GDN slots")
        states[dp] = ledger
    require(set(states) == {0, 1}, "missing physical DP pool")
    return info


async def session_check(args, trace, cfg):
    """Real shared-driver tool round trip; no model-output equivalence comparison."""
    cfg = copy.deepcopy(cfg)
    cfg["system_prompt_template"] += (
        "\nBefore answering, call python exactly once to execute print(17 * 23). "
        "After the tool returns, give the puzzle's output grid."
    )
    jobs = [driver.Rollout(f"session-check-dp{dp}", 0, 0,
                           [{"input": [[1]], "output": [[2]]}], [[3]]) for dp in (0, 1)]
    trace.phase = "session-check"
    results = await driver.run_all(jobs, base_url=args.base_url, model=args.model, cfg=cfg, sandbox_cls=Sandbox,
                                   budget=driver.Budget(context_length=args.context_length),
                                   sampling=driver.Sampling(), concurrency=2, dp_size=2,
                                   deadline=None if args.deadline_min is None else time.monotonic() + args.deadline_min * 60,
                                   http_timeout=args.request_timeout, keep_messages=True,
                                   hisparse_sessions=True, session_close_timeout=args.session_close_timeout,
                                   reasoning_effort=args.reasoning_effort, force_first_tool=True,
                                   return_token_ids=True, event_hooks=trace.hooks)
    write_json(args.output_dir / "session-results.json", [asdict(result) for result in results])
    require(len(results) == 2 and {result.replica.rsplit("#dp", 1)[1] for result in results} == {"0", "1"},
            "both actual DP routes must complete")
    require(len({result.session_id for result in results}) == 2, "session-check SIDs are not distinct")
    evidence = []
    for result in results:
        replies = trace.chats[result.session_id]
        require(result.n_turns >= 2 and result.n_tool_calls >= 1 and result.n_tool_errors == 0,
                "bridge did not complete an actual tool round trip")
        require(any(message["role"] == "tool" and message["content"].strip() == "391"
                    for message in result.messages), "actual Python tool did not return391")
        require(len(replies) >= 2, "missing raw same-session chat responses")
        first_usage, second_usage = replies[0]["usage"], replies[1]["usage"]
        returned_dp = int(result.replica.rsplit("#dp", 1)[1])
        for reply in replies:
            choice = reply["choices"][0]
            require(choice["meta_info"]["dp_rank"] == returned_dp, "actual response DP differs")
            ids = choice["response_token_ids"]
            require(isinstance(ids, list) and len(ids) == reply["usage"]["completion_tokens"]
                    and all(type(token) is int and token >= 0 for token in ids), "native output IDs missing/invalid")
        cached = second_usage["prompt_tokens_details"]["cached_tokens"]
        require(cached > 0, "second turn has no same-session cache reuse")
        require(result.session_close_status in (200, 204) and not result.session_close_error,
                "owned SID close failed")
        require(result.grid is not None and validate_grid(result.grid) is None, "final response contains no valid grid")
        evidence.append({"sid": result.session_id, "replica": result.replica, "turns": result.n_turns,
                         "actual_dp_rank": returned_dp,
                         "tool_calls": result.n_tool_calls, "first_prompt_tokens": first_usage["prompt_tokens"],
                         "cached_tokens_second_turn": cached, "close_status": result.session_close_status})
    return evidence


async def run(args):
    require(args.base_url.rstrip("/").endswith("/v1"), "--base-url must end with /v1")
    require(args.dp_size == 2 and args.concurrency > 0
            and (args.deadline_min is None or math.isfinite(args.deadline_min) and args.deadline_min > 0)
            and args.session_close_timeout > 0 and args.request_timeout > 0, "invalid bounded runner configuration")
    require(not args.output_dir.exists(), "refusing to overwrite earlier runner evidence")
    args.output_dir.mkdir(parents=True)
    trace, store, challenges = Trace(args.output_dir), None, None
    summary = {"status": "running", "scope": "ARC2 inference and owned-session lifecycle",
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "records": [], "kaggle_api_called": False}
    started = time.monotonic()
    try:
        cfg = protocol.load_config(args.config_json)
        if not args.session_check_only:
            require(args.challenges is not None, "--challenges is required for ARC2 generation")
            challenges = load_challenges(args.challenges)
            store = SubmissionStore(challenges, args.output_dir / "submission.json")
            require(args.mode != "competition-rerun" or args.solutions is None,
                    "competition-rerun must not receive hidden/test solutions")
            require(args.mode != "saverun" or args.solutions is not None,
                    "Save&Run uses public validation challenges and explicit public solutions")
            selected = list(challenges)[:args.validation_limit] if args.mode == "saverun" else list(challenges)
            required_attempts = {(task_id, index, attempt) for task_id in selected
                                 for index in range(len(challenges[task_id]["test"])) for attempt in (0, 1)}
            solutions = json.loads(args.solutions.read_text()) if args.solutions is not None else None
            if solutions is not None:
                for task_id in selected:
                    require(len(solutions[task_id]) == len(challenges[task_id]["test"]), "validation solution count differs")
                    require(all(validate_grid(grid) is None for grid in solutions[task_id]), "invalid validation solution")
            summary.update(selected_tasks=selected, total_tasks=len(challenges),
                           unselected_tasks=[task_id for task_id in challenges if task_id not in selected],
                           required_selected_attempt_cells=len(required_attempts),
                           unselected_fallback_attempt_cells=2 * sum(len(challenges[task_id]["test"])
                                                                   for task_id in challenges if task_id not in selected),
                           challenges_sha256=hashlib.sha256(args.challenges.read_bytes()).hexdigest(),
                           solutions_sha256=None if args.solutions is None else hashlib.sha256(args.solutions.read_bytes()).hexdigest())
        summary["server_info"] = await preflight(args, trace)
        if args.session_check_only:
            summary["sessions"] = await session_check(args, trace, cfg)
            summary["status"] = "passed"
        else:
            budget = driver.Budget(gen_cap=args.gen_cap, max_turns=args.max_turns, max_tokens=args.max_tokens,
                                   context_length=args.context_length, context_margin=args.context_margin,
                                   min_turn_tokens=args.min_turn_tokens)
            deadline = None if args.deadline_min is None else time.monotonic() + args.deadline_min * 60
            with (args.output_dir / "records.jsonl").open("w") as records:
                def on_result(result):
                    store.put(result.task_id, result.test_idx, result.attempt, result.grid)
                    record = asdict(result)
                    summary["records"].append(record)
                    records.write(json.dumps(record) + "\n")
                    records.flush()
                for attempt in (0, 1):
                    trace.phase = f"attempt-{attempt + 1}"
                    jobs = [driver.Rollout(task_id, index, attempt, challenges[task_id]["train"], test["input"],
                                           None if solutions is None else solutions[task_id][index])
                            for task_id in selected for index, test in enumerate(challenges[task_id]["test"])]
                    await driver.run_all(jobs, base_url=args.base_url, model=args.model, cfg=cfg, sandbox_cls=Sandbox,
                                         budget=budget, sampling=driver.Sampling(args.temperature, args.top_p, -1),
                                         concurrency=args.concurrency, dp_size=args.dp_size, deadline=deadline,
                                         on_result=on_result, http_timeout=args.request_timeout, keep_messages=True,
                                         hisparse_sessions=True, session_close_timeout=args.session_close_timeout,
                                         reasoning_effort=args.reasoning_effort, event_hooks=trace.hooks)
                    if any(row["error"] or row["session_close_error"] for row in summary["records"]):
                        raise RuntimeError("inference/cleanup failed; inspect records and raw HTTP, no blind retry")
                    if deadline is not None and time.monotonic() >= deadline:
                        break
            missing = sorted(required_attempts - store.filled)
            summary.update(status="incomplete" if missing else "completed", generated_attempt_cells=len(store.filled),
                           filled_selected_attempt_cells=len(required_attempts & store.filled),
                           unfilled_selected_attempts=[{"task_id": task_id, "test_idx": index,
                                                       "attempt": f"attempt_{attempt + 1}"}
                                                      for task_id, index, attempt in missing],
                           fallback_attempt_cells=2 * store.n_cells - len(store.filled), rejected_grids=store.rejected,
                           completed_rollouts=len(summary["records"]),
                           generated_tokens=sum(row["gen_tokens"] for row in summary["records"]))
            if solutions is not None:
                summary["public_validation_solved_tasks"] = sum(
                    all(any(row["correct"] for row in summary["records"]
                            if row["task_id"] == task_id and row["test_idx"] == index)
                        for index in range(len(challenges[task_id]["test"]))) for task_id in selected)
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if store is not None:
            summary["submission_sha256"] = validate_submission(store.out, challenges)
        summary["elapsed_s"] = time.monotonic() - started
        write_json(args.output_dir / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("saverun", "competition-rerun"), default="saverun")
    parser.add_argument("--validation-limit", type=int, default=2)
    parser.add_argument("--session-check-only", action="store_true")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=100000)
    baseline_budget = driver.Budget()
    parser.add_argument("--context-margin", type=int, default=baseline_budget.context_margin)
    parser.add_argument("--gen-cap", type=int, default=baseline_budget.gen_cap)
    parser.add_argument("--max-tokens", type=int, default=baseline_budget.max_tokens)
    parser.add_argument("--max-turns", type=int, default=baseline_budget.max_turns)
    parser.add_argument("--min-turn-tokens", type=int, default=baseline_budget.min_turn_tokens)
    parser.add_argument("--deadline-min", type=float, help="optional explicit wall-clock limit; omitted means no added limit")
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--session-close-timeout", type=float, default=15)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", type=float, default=.95)
    args = parser.parse_args()
    require(args.validation_limit > 0, "validation limit must be positive")
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in ("status", "elapsed_s")}))
    raise SystemExit(1 if result["status"] in ("failed", "incomplete") else 0)


if __name__ == "__main__":
    main()
