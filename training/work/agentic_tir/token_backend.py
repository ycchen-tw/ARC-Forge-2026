"""Qwen token-in / token-out backend: the harness's OpenAI chat payload → render + tokenize ourselves → sglang
`/generate` (`input_ids`) → `output_ids` appended to the context **as-is**. `--backend tokens`.

Why this path exists (for the distillation corpus): the chat API does "decode → re-render → re-tokenize" every turn,
so the stored target sequence differs from the tokens the teacher actually emitted (qwen38fn_distill 48-task smoke:
18.75% of trajectories differ after re-tokenize, all within assistant spans). This follows the seam rules of
`work/qwen38fn_distill/rollout_runner.py`:

    P0  = apply_chat_template([system, task_user], tools, add_generation_prompt=False)  → tokenize
    G_i = /generate output_ids appended as-is (ends with <|im_end|>; if truncated append one <|im_end|>, which a template re-render also has)
    T   = '\\n' + (the string delta the template renders for this user / tool message)  ← follows G's <|im_end|>
    cue = '<|im_start|>assistant\\n<think>\\n'                                 ← appended before each generation

Each message's tokens are stored in its `_ids` (assistant also has `_gen_len` = cue length) and travel with the
segments when written; `render_segments.py` concatenates `_ids` directly without re-tokenizing. A message's delta is
computed with the template (`render(msgs[:k+1])` minus `render(msgs[:k])`, prefix asserted per message), not
hand-written strings -- the template merges consecutive tool messages into one user turn, so the delta is computed per
"group of consecutive tool messages" and the tokens hang on the group's first message.

Differences from the OpenAI path (same as dsv4_backend): `tool_choice: "none"` has no counterpart;
`enable_thinking=False` is not supported; output truncated by `max_tokens` has no `</think>`: all reasoning,
tool_calls always empty. Tool definitions are rendered in the shape of sglang `Tool.model_dump()` (extra
`strict: false` / `defer_loading: null`), so the training prompt matches token-for-token what the student later sees
when evaluated through the chat API.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

import httpx

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
GEN_CUE = "<|im_start|>assistant\n<think>\n"
EFFORTS = ("xhigh", "medium", "low")


def sglang_tool_shape(tool: dict) -> dict:
    """sglang serving_chat.py does `item.model_dump()` (pydantic Tool/Function) before passing to the template; field
    order and filled-in defaults must match. Compared character by character against `Tool(**tool).model_dump()` of
    sglang-qwen4 on 2026-09-03."""
    fn = tool["function"]
    return {"type": "function",
            "function": {"description": fn.get("description"), "name": fn["name"],
                         "parameters": fn.get("parameters"), "strict": False},
            "defer_loading": None}


def strip_private(m: dict) -> dict:
    """Message for the template / for writing: drop private keys like `_ids`; tool_calls arguments go from JSON string back to dict (the template wants `|items`)."""
    d = {k: v for k, v in m.items() if not k.startswith("_")}
    if d.get("tool_calls"):
        calls = []
        for c in d["tool_calls"]:
            c = json.loads(json.dumps(c))
            fn = c["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
            calls.append(c)
        d["tool_calls"] = calls
    return d


def split_generation(text: str) -> tuple[str, str]:
    """Same as rollout_runner: the cue ends with '<think>\\n', so G starts with the reasoning body."""
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
    else:
        reasoning, rest = text, ""
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
    return reasoning, rest


class TokenBackend:
    def __init__(self, base_url: str, tokenizer, tool_defs: list[dict], retries: int):
        self.root = base_url[:-3] if base_url.endswith("/v1") else base_url
        self.tok = tokenizer
        self.tools = [sglang_tool_shape(t) for t in tool_defs]
        self.retries = retries
        self.gen_ids = self._ids(GEN_CUE)
        self.nl_ids = self._ids("\n")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eot = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        for name, ids in (("cue", self.gen_ids), ("\\n", self.nl_ids)):
            assert ids, name
        assert len(self.nl_ids) == 1, self.nl_ids
        assert isinstance(self.im_end, int) and self.im_end >= 0
        self.n_malformed = 0

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _render(self, clean: list[dict], effort: str, gen: bool) -> str:
        return self.tok.apply_chat_template(clean, tools=self.tools, tokenize=False, add_generation_prompt=gen,
                                            reasoning_effort=effort)

    # -- per-message token spans ------------------------------------------------
    def annotate(self, messages: list[dict], effort: str) -> None:
        """Fill in tokens for non-assistant messages without `_ids` (or whose content changed); assistant `_ids` can
        only come from chat(). `_key` records the context the delta was computed in (content, whether the previous
        message is an assistant, the contents of the same tool group); recomputed when it changes."""
        assert effort in EFFORTS, effort
        assert messages and messages[0]["role"] == "system" and len(messages) >= 2 and messages[1]["role"] == "user", \
            "token backend expects [system, user, ...]"
        clean = [strip_private(m) for m in messages]
        # P0: system + task together (the template rejects system-only messages), tokens hang on messages[1].
        key = ("prefix", effort, messages[0]["content"], messages[1]["content"])
        if messages[1].get("_key") != key:
            messages[1]["_ids"] = self._ids(self._render(clean[:2], effort, gen=False))
            messages[1]["_key"] = key
        k = 2
        while k < len(messages):
            m = messages[k]
            if m["role"] == "assistant":
                assert "_ids" in m, f"assistant message {k} has no _ids (not produced by TokenBackend.chat)"
                k += 1
                continue
            # A group: consecutive tool messages, or a single user message.
            g1 = k + 1
            if m["role"] == "tool":
                while g1 < len(messages) and messages[g1]["role"] == "tool":
                    g1 += 1
            prev_is_asst = messages[k - 1]["role"] == "assistant"
            key = ("group", effort, prev_is_asst, tuple((x["role"], x.get("content") or "") for x in messages[k:g1]))
            if m.get("_key") != key:
                before = self._render(clean[:k], effort, gen=False)
                after = self._render(clean[:g1], effort, gen=False)
                assert after.startswith(before), f"template not append-only at message {k} ({m['role']})"
                delta = after[len(before):]
                # G stops at <|im_end|> while the template has a '\n' after the assistant; prepend it to the delta (same as rollout_runner's T_i).
                m["_ids"] = self._ids(("\n" if prev_is_asst else "") + delta)
                m["_key"] = key
                for x in messages[k + 1:g1]:
                    x["_ids"] = []
                    x["_key"] = key
            k = g1

    def context_ids(self, messages: list[dict], effort: str) -> list[int]:
        self.annotate(messages, effort)
        ids: list[int] = []
        for m in messages[1:]:
            ids.extend(m["_ids"])
        return ids

    # -- completion text -> message --------------------------------------------
    def parse(self, out_ids: list[int], truncated: bool) -> tuple[dict, str]:
        text = self.tok.decode(out_ids, skip_special_tokens=False)
        reasoning, content = split_generation(text)
        if truncated:
            return {"role": "assistant", "content": content if "</think>" in text else "",
                    "reasoning_content": reasoning, "tool_calls": []}, "length"
        calls = []
        for i, (name, body) in enumerate(_CALL_RE.findall(content)):
            params = {k: v for k, v in _PARAM_RE.findall(body)}
            calls.append({"id": f"call_{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(params, ensure_ascii=False)}})
        if calls:
            # Same as sglang's parser: content keeps only the plain text before tool_call
            content = content[: content.find("<tool_call>")].strip()
        return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}, "stop"

    # -- the call ------------------------------------------------------------------
    async def chat(self, client: httpx.AsyncClient, payload: dict) -> dict:
        effort = payload.get("reasoning_effort") or "xhigh"
        if payload.get("chat_template_kwargs"):
            raise RuntimeError("token backend does not support chat_template_kwargs (use same for --handoff-thinking)")
        messages = payload["messages"]
        ids = self.context_ids(messages, effort) + self.gen_ids
        if not any(m["role"] == "assistant" for m in messages):
            # A prompt with no G yet must equal the full re-render token for token -- the foundation of the seam rules, checked on the first turn.
            ref = self._ids(self._render([strip_private(m) for m in messages], effort, gen=True))
            assert ids == ref, f"token stream != rendered prompt ({len(ids)} vs {len(ref)} tokens)"
        req = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": payload["temperature"], "top_p": payload["top_p"], "top_k": payload["top_k"],
                "max_new_tokens": payload["max_tokens"], "skip_special_tokens": False,
            },
        }
        # /generate does not copy the routing field from the chat payload automatically; without it sglang DP goes
        # round-robin and multi-turn TIR re-prefills the whole radix prefix. 0 is a valid rank, hence `is not None`.
        if payload.get("routed_dp_rank") is not None:
            req["routed_dp_rank"] = payload["routed_dp_rank"]
        data = await _post(client, f"{self.root}/generate", req, self.retries)
        out = data["output_ids"]
        meta = data["meta_info"]
        assert len(out) == meta["completion_tokens"], (len(out), meta["completion_tokens"])
        assert meta["prompt_tokens"] == len(ids), (meta["prompt_tokens"], len(ids))
        fr = meta["finish_reason"]
        ftype = fr["type"] if isinstance(fr, dict) else fr
        assert ftype in ("length", "stop"), f"unexpected finish_reason {fr!r}"
        msg, finish = self.parse(out, truncated=(ftype == "length"))
        asst_ids = self.gen_ids + list(out)
        if not out or out[-1] not in (self.im_end, self.eot):
            asst_ids.append(self.im_end)      # a template re-render always has <|im_end|>; append it to truncated output so the seam lines up
        msg["_ids"] = asst_ids
        msg["_gen_len"] = len(self.gen_ids)
        msg["_out_len"] = len(out)
        n_reason = out.index(self.think_end) if self.think_end in out else len(out)
        return {
            "choices": [{"message": msg, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": meta["prompt_tokens"], "completion_tokens": meta["completion_tokens"],
                "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
                "prompt_tokens_details": {"cached_tokens": meta.get("cached_tokens", 0)},
                "completion_tokens_details": {"reasoning_tokens": n_reason},
            },
        }


async def _post(client: httpx.AsyncClient, url: str, req: dict, retries: int, backoff: float = 3.0) -> dict:
    """Same as driver._chat: 4xx raises immediately, only 5xx / connection errors are retried."""
    for retry in range(retries + 1):
        try:
            r = await client.post(url, json=req)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status and status < 500) or retry == retries:
                raise
            await asyncio.sleep(backoff * (retry + 1))
    raise AssertionError("unreachable")
