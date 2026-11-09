"""DeepSeek-V4 backend: convert the harness's OpenAI chat payload into a native DSv4 prompt and call sglang `/generate`.

Why not `/v1/chat/completions`: through sglang's chat endpoint DSv4 silently drops `reasoning_effort` and thinking
leaks into content (docs/research/dsv4_flash_serving_b200.md). So, following tooluse/tir_rollout.py: build the prompt
with the model's own `encoding_dsv4.encode_messages` (tools attached to the system message, tool results merged into
the user turn, old-round thinking kept when tools are present), tokenize and send `input_ids`, decode the returned
tokens and split them with `parse_message_from_completion_text` into reasoning_content / content / tool_calls (OpenAI
format), then wrap them in a chat/completions-shaped response; the rest of the harness is unchanged.

Differences from the OpenAI path (what the harness side needs to know):
- `tool_choice: "none"` has no counterpart: tool calls cannot be forbidden during handoff; if the model calls a tool
  again the harness runs it.
- `chat_template_kwargs.enable_thinking=False` is not supported (`--handoff-thinking off` is rejected at start-up).
- Output truncated by `max_tokens` has no `</think>`/EOS and the official parser asserts; split it here instead:
  no `</think>` → all reasoning, content empty; with it → the text after it is content, an incomplete DSML block is dropped.
  tool_calls always empty (the harness never executes tool calls with finish=length anyway).
- A normal finish that fails to parse (tir_rollout calls it "malformed"; never seen in 656 earlier attempts):
  finish_reason returns "malformed", content is the text after `</think>` with special tokens removed, and one line goes
  to stderr -- not swallowed, but a task is not scrapped over it either.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

import httpx

DSV4_EFFORTS = ("low", "high", "max")


def load_encoding(path: str):
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DSv4Backend:
    def __init__(self, base_url: str, encoding_path: str, tokenizer, retries: int):
        # The harness gives `http://host:port/v1`; /generate is at the root path.
        self.root = base_url[:-3] if base_url.endswith("/v1") else base_url
        self.enc = load_encoding(encoding_path)
        self.tok = tokenizer
        self.retries = retries
        self.n_malformed = 0

    # -- prompt ---------------------------------------------------------------
    def build_prompt(self, payload: dict) -> str:
        messages = [dict(m) for m in payload["messages"]]
        assert messages and messages[0]["role"] == "system", "DSv4 backend expects a leading system message"
        messages[0]["tools"] = payload["tools"]
        effort = payload.get("reasoning_effort")
        assert effort in DSV4_EFFORTS, f"reasoning_effort must be one of {DSV4_EFFORTS}, got {effort!r}"
        return self.enc.encode_messages(messages, thinking_mode="thinking", reasoning_effort=effort)

    # -- completion text -> message -------------------------------------------
    def parse(self, text: str, truncated: bool) -> tuple[dict, str]:
        enc = self.enc
        if truncated:
            reasoning, sep, rest = text.partition(enc.thinking_end_token)
            if not sep:
                return {"role": "assistant", "content": "", "reasoning_content": text, "tool_calls": []}, "length"
            content = rest.split(f"\n\n<{enc.dsml_token}")[0]
            return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": []}, "length"
        if not text.endswith(enc.eos_token):
            text += enc.eos_token
        try:
            msg = enc.parse_message_from_completion_text(text, thinking_mode="thinking")
        except (ValueError, AssertionError) as e:
            self.n_malformed += 1
            print(f"!! dsv4 malformed completion ({e}); treating as final text", file=sys.stderr)
            body = text.partition(enc.thinking_end_token)[2]
            for sp in (enc.bos_token, enc.eos_token, enc.thinking_start_token, enc.thinking_end_token):
                body = body.replace(sp, "")
            reasoning = text.partition(enc.thinking_end_token)[0]
            return {"role": "assistant", "content": body, "reasoning_content": reasoning, "tool_calls": []}, "malformed"
        for i, call in enumerate(msg["tool_calls"]):
            call["id"] = f"call_{i}"
        return msg, "stop"

    # -- the call --------------------------------------------------------------
    async def chat(self, client: httpx.AsyncClient, payload: dict) -> dict:
        prompt = self.build_prompt(payload)
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        req = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": payload["temperature"], "top_p": payload["top_p"], "top_k": payload["top_k"],
                "max_new_tokens": payload["max_tokens"], "skip_special_tokens": False,
            },
        }
        data = await _post(client, f"{self.root}/generate", req, self.retries)
        out_ids = data["output_ids"]
        meta = data["meta_info"]
        assert len(out_ids) == meta["completion_tokens"], (len(out_ids), meta["completion_tokens"])
        fr = meta["finish_reason"]
        ftype = fr["type"] if isinstance(fr, dict) else fr
        assert ftype in ("length", "stop"), f"unexpected finish_reason {fr!r}"
        text = self.tok.decode(out_ids, skip_special_tokens=False)
        msg, finish = self.parse(text, truncated=(ftype == "length"))
        n_reason = len(self.tok(msg["reasoning_content"], add_special_tokens=False)["input_ids"]) if msg["reasoning_content"] else 0
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
