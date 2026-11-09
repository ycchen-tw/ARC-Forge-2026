"""Run the unchanged original ARC2 agentic harness with opt-in HiSparse transport.

All solver flags pass to the original parser. --hisparse-sessions only adds an
owned SID per run_episode, a least-loaded DP per episode, native-ID receipts and close.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    # A monitor never sees half a live state while a nonstreaming call waits.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
    temporary.replace(path)


def write_gzip(path, value):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as output:
        json.dump(value, output, ensure_ascii=False)


def load_original():
    manifest = json.loads((ROOT / "ORIGINAL_MANIFEST.json").read_text())
    for name, identity in manifest["files"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == identity["sha256"],
                f"original frozen source changed: {name}")
    # A payload CLI runs in a fresh interpreter. Refuse a cached, different arcforge.
    for name, module in sys.modules.items():
        if name == "arcforge" or name.startswith("arcforge."):
            require(Path(module.__file__).resolve().is_relative_to(ROOT / "src"),
                    "launch the frozen harness in a fresh process with its own src on PYTHONPATH")
    sys.path.insert(0, str(ROOT / "src"))
    source = ROOT / "work/agentic_tir/agentic_tir.py"
    spec = importlib.util.spec_from_file_location("frozen_arc2_original", source)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    return harness, manifest


class HiSparseTransport:
    def __init__(self, harness, args):
        require(args.backend == "openai", "HiSparse bridge supports the original OpenAI backend only")
        require(args.base_url.rstrip("/").endswith("/v1"), "HiSparse base-url must end with /v1")
        require(args.http_timeout > 0, "original HTTP timeout must be positive")
        require(type(args.dp_size) is int and args.dp_size >= 1, "dp_size must be a positive integer")
        self.harness, self.args = harness, args
        # Live owned episodes per attention DP; asyncio runs every update on one thread.
        self.dp_live = [0] * args.dp_size
        self.context = contextvars.ContextVar("frozen_arc2_owned_episode", default=None)
        self.root = Path(args.out) / "hisparse-transport"
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / f"bridge-{uuid.uuid4().hex}.json", {
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_manifest": json.loads((ROOT / "ORIGINAL_MANIFEST.json").read_text()),
            "scope": "transport only; original solver/prompts/scoring/budgets unchanged",
            "effective_chat_retries": 0, "close_timeout": args.http_timeout,
            "native_ids": "SGLang sglext.output_ids, never re-tokenized",
        })

    def progress(self, state, event, phase, phase_state, started, **fields):
        stamp = time.monotonic()
        row = {"event": event, "phase": phase, "state": phase_state,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "monotonic_s": stamp, "started_monotonic_s": started,
               "elapsed_s": max(0, stamp - started),
               "episode_started_monotonic_s": state["episode_started_monotonic_s"],
               "call_started_monotonic_s": state["call_started_monotonic_s"],
               "task_id": state["task_id"], "label": state["label"],
               "lineage": state["lineage"], "hop": state["hop"],
               "sid": state["sid"], "dp_rank": state["dp"], "rid": state["rid"],
               "call_index": state["call"], "requested_max_tokens": state["requested_max_tokens"],
               "message_count": state["message_count"], "active": event != "episode_end",
               "inflight": phase == "model_call" and phase_state == "waiting", **fields}
        write_json(state["directory"] / "inflight.json", row)
        with (self.root / "events.jsonl").open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
        # Metadata only: prompts, responses, exception text and credentials stay out.
        print("ARC2_TRANSPORT=" + json.dumps(row, ensure_ascii=False), flush=True)

    async def chat(self, client, payload):
        state = self.context.get()
        require(state is not None, "model chat outside an owned original run_episode")
        require(not state["inflight"], "two requests attempted to share a live HiSparse SID")
        # The original harness stamps its per-task DP; the episode owner's DP replaces it below.
        harness_dp = payload.get("routed_dp_rank")
        require(payload.get("stream", False) is False and payload.get("n", 1) == 1,
                "native-ID receipts require one nonstreaming choice")
        state["call"] += 1
        rid = f"{state['sid']}-call-{state['call']:04d}"
        require("rid" not in payload or payload["rid"] is None or payload["rid"] == rid,
                "caller RID is outside the owned episode/call")
        state["rid"] = rid
        state["requested_max_tokens"] = payload.get("max_tokens")
        state["message_count"] = len(payload["messages"])
        request = dict(payload, session_id=state["sid"], routed_dp_rank=state["dp"], rid=rid,
                       return_meta_info=True, return_output_ids_in_sglext=True)
        prefix = state["directory"] / f"call-{state['call']:04d}"
        write_gzip(prefix.with_suffix(".request.json.gz"), request)
        started = time.monotonic()
        state["call_started_monotonic_s"] = started
        receipt = {"sid": state["sid"], "rid": rid, "requested_dp": state["dp"], "harness_task_dp": harness_dp,
                   "task_id": state["task_id"],
                   "label": state["label"], "lineage": state["lineage"], "hop": state["hop"],
                   "call_index": state["call"], "started_monotonic_s": started}
        state["inflight"] = True
        try:
            self.progress(state, "model_call_start", "model_call", "waiting", started)
            # Original shared transport, with ambiguous execution retries disabled only here.
            response = await self.harness._chat(client, self.args.base_url, request, retries=0)
            write_gzip(prefix.with_suffix(".response.json.gz"), response)
            actual_dp = response["choices"][0]["meta_info"]["dp_rank"]
            require(actual_dp == state["dp"], "actual response DP differs from episode owner")
            native = response["sglext"]["output_ids"]
            require(isinstance(native, list) and len(native) == 1, "missing one-choice native output IDs")
            ids = native[0]
            require(isinstance(ids, list) and len(ids) == response["usage"]["completion_tokens"]
                    and all(type(token) is int and token >= 0 for token in ids), "native output IDs/count invalid")
            receipt.update(actual_dp=actual_dp, native_output_id_count=len(ids),
                           native_output_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                           usage=response["usage"])
            self.progress(state, "model_call_end", "model_call", "completed", started,
                          actual_dp=actual_dp, native_output_id_count=len(ids))
            return response  # The original harness consumes its original choices/usage.
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            self.progress(state, "model_call_error", "model_call", "error", started, error_type=type(exc).__name__)
            raise
        finally:
            state["inflight"] = False
            receipt["ended_monotonic_s"] = time.monotonic()
            write_json(prefix.with_suffix(".receipt.json"), receipt)

    def install(self):
        original = self.harness.run_episode
        async def run_episode(client, episode_args, **kwargs):
            require(episode_args.dp_size == len(self.dp_live), "episode dp_size differs from the bridge")
            if episode_args.fixed_dp_rank is not None:
                dp = episode_args.fixed_dp_rank
            else:
                # Each episode owns its SID and the server reuses KV only inside a session, so
                # the original per-task pin buys no cache. Pinning whole tasks (2-3 explorers
                # plus refiner each) left DP0 queueing 4-7 requests while DP1 idled 4-7 slots
                # (Kaggle v6). Pick the DP with the fewest live episodes instead.
                dp = min(range(len(self.dp_live)), key=lambda rank: (self.dp_live[rank], rank))
            require(type(dp) is int and 0 <= dp < len(self.dp_live), "episode DP outside the server DP range")
            self.dp_live[dp] += 1
            try:
                return await owned_episode(client, episode_args, dp, **kwargs)
            finally:
                self.dp_live[dp] -= 1

        async def owned_episode(client, episode_args, dp, **kwargs):
            sid = "arc2-" + uuid.uuid4().hex
            directory = self.root / sid
            directory.mkdir()
            state = {"sid": sid, "dp": dp, "directory": directory, "call": 0, "inflight": False,
                     "task_id": kwargs.get("task_id", ""), "label": kwargs["label"],
                     "lineage": kwargs.get("lineage", 0), "hop": kwargs.get("hop", 1),
                     "episode_started_monotonic_s": time.monotonic(),
                     "call_started_monotonic_s": None, "rid": None,
                     "requested_max_tokens": None, "message_count": 0}
            token = self.context.set(state)
            receipt = {"sid": sid, "dp_rank": dp, "task_id": kwargs.get("task_id", ""), "label": kwargs["label"],
                       "lineage": kwargs.get("lineage", 0), "hop": kwargs.get("hop", 1),
                       "close_acknowledged": False, "close_status": None,
                       "started_monotonic_s": state["episode_started_monotonic_s"]}
            primary = None
            try:
                self.progress(state, "episode_start", "episode", "running", state["episode_started_monotonic_s"])
                return await original(client, episode_args, **kwargs)
            except BaseException as exc:
                primary = exc
                receipt["episode_error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                close_started, close_error = time.monotonic(), None
                try:
                    body = {"session_id": sid}
                    write_json(directory / "close-request.json", body)
                    self.progress(state, "session_close_start", "session_close", "waiting", close_started)
                    async with asyncio.timeout(episode_args.http_timeout):
                        closed = await client.post(episode_args.base_url.rstrip("/")[:-3] + "/close_session",
                                                   json=body, timeout=episode_args.http_timeout)
                        (directory / "close-response.body").write_bytes(closed.content)
                        closed.raise_for_status()
                        require(closed.content.strip() in (b"", b"null"), "unexpected close_session ACK")
                        receipt.update(close_acknowledged=True, close_status=closed.status_code)
                    self.progress(state, "session_close_end", "session_close", "completed", close_started,
                                  close_status=closed.status_code)
                except Exception as exc:
                    close_error = exc
                    receipt["close_error"] = f"{type(exc).__name__}: {exc}"
                    self.progress(state, "session_close_error", "session_close", "error", close_started,
                                  error_type=type(exc).__name__)
                    if primary is None:
                        raise
                    primary.add_note(receipt["close_error"])
                except BaseException as exc:
                    close_error = exc
                    self.progress(state, "session_close_error", "session_close", "error", close_started,
                                  error_type=type(exc).__name__)
                    raise
                finally:
                    receipt.update(calls=state["call"], ended_monotonic_s=time.monotonic())
                    try:
                        write_json(directory / "session.json", receipt)
                        self.progress(state, "episode_end", "episode", "error" if primary or close_error else "completed",
                                      state["episode_started_monotonic_s"],
                                      close_acknowledged=receipt["close_acknowledged"], close_status=receipt["close_status"])
                    finally:
                        self.context.reset(token)
        self.harness.run_episode = run_episode
        self.args.chat = self.chat


def main():
    controls = argparse.ArgumentParser(add_help=False)
    controls.add_argument("--hisparse-sessions", action="store_true")
    own, original_argv = controls.parse_known_args()
    harness, _ = load_original()
    sys.argv = [str(harness.__file__), *original_argv]
    args = harness.parse_args()
    if own.hisparse_sessions:
        HiSparseTransport(harness, args).install()
    asyncio.run(harness.main_async(args))


if __name__ == "__main__":
    main()
