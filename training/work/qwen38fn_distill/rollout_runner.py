"""Qwen3.8-Flash-Next teacher rollouts over NVARC synthetic puzzles, token-level append-only.

CoT (no tools, single shot) and TIR (multi-turn tools) share this script: they differ only in which mode the setting
draws, and the storage schema is identical (flat `ids` + span indices), so downstream rendering needs no second path.

## Why native `/generate` instead of `/v1/chat/completions`

The chat API does "decode to text → re-render with the template → re-tokenize" every turn. Harmless for evaluation,
wrong for distillation: the stored target sequence may differ from the tokens the teacher actually emitted. In the
DSv4 batch, of 31.6M tokens, **2.0% of assistant spans and 28% of trajectories** differed after re-tokenising, and the
divergence fell **entirely** on the assistant spans that carry the loss. The `output_ids` that `/generate` returns are
the tokens the teacher really emitted; appending them as-is avoids the problem.

## The append-only contract (measured offline, see preflight.py)

    P0  = apply_chat_template(sys+user, tools, add_generation_prompt=True)
          necessarily ends with '<|im_start|>assistant\\n<think>\\n'
    G_i = /generate output_ids, appended as-is (ending with <|im_end|>)
    T_i = '\\n<|im_start|>user' + per result '\\n<tool_response>\\n{out}\\n</tool_response>'
          + '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'

Measured on a three-turn conversation: the ids assembled this way are **bit-identical** to a single-shot
`apply_chat_template`. This holds because every seam falls on an atomic token -- on this model `<|im_start|>` /
`<|im_end|>` / `<think>` / `<tool_response>` / `</tool_response>` are all single tokens (DSv4's `<tool_result>` is 4
tokens, so there a tool segment must be tokenized as one piece). Even so, this script still **tokenizes each segment in
one piece**, because that does not depend on "which markers happen to be atomic".

Each turn asserts `ids[:len(prev)] == prev`. It always holds today because the chat template's `preserve_thinking`
is on by default and tool results have role `tool` (not a user query, so `last_query_index` stays at the first user
turn and every turn's `<think>` is kept). If that premise ever changes, this assert is the only check that catches it
in one go.

## Concurrency is "rollouts at once", not "requests at once"

The semaphore wraps the **whole rollout**. Wrapping a single HTTP call is the defect fixed in `work/qwen38_eval/`:
`asyncio.gather` opens every rollout at once, each one's next turn queues behind hundreds of requests and its prefix
gets evicted -- radix hit rate 27.7% vs 94.1%, 2.2× end to end.
The httpx connection pool must be enlarged accordingly -- the default is 100, beyond which requests queue silently.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pack_menu = _load(Path(__file__).parent / "pack_menu.py", "pack_menu")

OUT_SCHEMA = pa.schema([
    ("uid", pa.string()), ("attempt", pa.int16()),
    # setting: the drawn configuration, stored column by column. Even if the weight tables change later, existing records can still tell which prompt they used.
    ("mode", pa.string()), ("instr", pa.string()), ("grid_fmt", pa.string()),
    ("label_shape", pa.bool_()), ("effort", pa.string()), ("pack_sha", pa.string()),
    ("train_idx", pa.list_(pa.int16())), ("test_idx", pa.int16()),
    # Trajectory: flat ids + spans. **Do not store (input_ids, output_ids) per turn** -- that is O(n²).
    ("ids", pa.list_(pa.int32())),
    ("span_start", pa.list_(pa.int32())), ("span_end", pa.list_(pa.int32())),
    ("span_role", pa.list_(pa.int8())),          # 0=prompt 1=assistant 2=tool
    ("n_turns", pa.int16()), ("n_tool_calls", pa.int16()), ("n_tool_errors", pa.int16()),
    ("n_prompt", pa.int32()), ("n_completion", pa.int32()),
    ("stop_reason", pa.string()), ("parsed", pa.bool_()), ("correct", pa.bool_()),
    ("correct_strict", pa.bool_()), ("answer_src", pa.string()),
    # Extra verification signal for rulecode arms: the answer the model reports is not necessarily what its own
    # transform() produces. The prompt ablation measured "trust transform()" +5.7pp more accurate than "trust the
    # reported answer" (27:8, p=0.0019). These cannot be recovered from ids (they require execution), so they must be stored.
    ("confidence", pa.int16()), ("code_train_ok", pa.int16()), ("code_train_n", pa.int16()),
    ("code_test_match", pa.int8()),              # -1=not run 0=mismatch 1=match
    ("retok_ok", pa.int8()),                     # -1=not checked 0=different 1=same
    ("latency_s", pa.float32()),
])

# The model emits tool calls in XML form; the format is set by the system section of the chat template:
#   <tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>
_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_RULE_RE = re.compile(r"RULE\s*:\s*(.+?)(?=\n\s*(?:CODE|CONFIDENCE)\s*:|\Z)", re.S)
_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*(\d{1,3})")

# Run the model's own transform() in the sandbox. train_pairs / test_input / np are preloaded in the sandbox, so grids
# need not be re-sent. Only the marker on the last line of the output is read; everything else (prints and warnings
# from the model's code) is ignored.
_CODE_HARNESS = '''
import json as _j
_r = {"train_ok": 0, "train_n": 0, "test": None, "err": None}
try:
    _f = transform
except Exception:
    _r["err"] = "no_transform"
else:
    for _p in train_pairs:
        _r["train_n"] += 1
        try:
            _o = _f([list(_row) for _row in _p["input"]])
            _r["train_ok"] += int([[int(_c) for _c in _row] for _row in _o] == _p["output"])
        except Exception:
            pass
    try:
        _o = _f([list(_row) for _row in test_input])
        _r["test"] = [[int(_c) for _c in _row] for _row in _o]
    except Exception as _e:
        _r["err"] = type(_e).__name__
print("<<<CODEVERIFY>>>" + _j.dumps(_r))
'''


def choose_pairs(uid, attempt, n_pairs, n_train):
    """**Line-for-line identical** to nvarc_distill/run_infer.py and tooluse/tir_rollout.py.

    The same (uid, attempt) always gets the same pairs, so the prompt rebuilt on resume is byte-identical to the
    original and any single record can be reproduced offline. To add samples, raise --attempts.
    """
    h = hashlib.sha256(f"{uid}:{attempt}".encode()).digest()
    order = sorted(range(n_pairs), key=lambda i: hashlib.sha256(h + i.to_bytes(2, "big")).digest())
    chosen = order[: n_train + 1]
    return sorted(chosen[:n_train]), chosen[n_train]


def read_corpus(path, limit=None, skip=0):
    import zstandard

    with open(os.path.expanduser(path), "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for i, line in enumerate(text):
            if i < skip:
                continue
            if limit is not None and i >= skip + limit:
                break
            yield json.loads(line)


def tool_segment(outputs):
    """The tool feedback the teacher sees, character-for-character what the chat template would render.

    `.strip()` is not cosmetic: the template applies `|trim` to message content; without trimming here as well, the
    training data (append-only) and evaluation (chat API re-render) would diverge on whitespace.
    """
    body = "".join(f"\n<tool_response>\n{o.strip()}\n</tool_response>" for o in outputs)
    return ("\n<|im_start|>user" + body
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n")


def parse_tool_calls(text, tool_name):
    """Used only for control flow, never to rebuild the context."""
    calls = []
    for name, body in _CALL_RE.findall(text):
        params = {k: v for k, v in _PARAM_RE.findall(body)}
        calls.append({"name": name, "params": params})
    return calls


def split_generation(text):
    """How G is split internally. P0 ends with '<think>\\n', so G starts with the reasoning body."""
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
    else:
        reasoning, rest = text, ""
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
    return reasoning, rest


def extract_code(text):
    blocks = _FENCE_RE.findall(text)
    for b in reversed(blocks):
        if "def transform" in b:
            return b
    return blocks[-1] if blocks else ""


def extract_confidence(text):
    m = _CONF_RE.findall(text)
    if not m:
        return -1
    v = int(m[-1])
    return v if 0 <= v <= 100 else -1


class Runner:
    def __init__(self, args, cfg, tok, Sandbox):
        self.a = args
        self.cfg = cfg
        self.tok = tok
        self.Sandbox = Sandbox
        self.tool_schema = cfg["tool"]
        self.tool_name = cfg["tool_name"]

    # ---------------------------------------------------------------- prompt

    def build_prompt(self, setting, train, test_input):
        msgs = []
        sysp = setting.system(self.cfg, self.a.max_turns)
        if sysp:
            msgs.append({"role": "system", "content": sysp})
        msgs.append({"role": "user", "content": setting.user(train, test_input)})
        kw = {}
        if setting.uses_tools:
            kw["tools"] = [self.tool_schema]
        text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            reasoning_effort=setting.effort, **kw)
        # This ending is the foundation of append-only: G appended right after it is the teacher's first token.
        if not text.endswith("<|im_start|>assistant\n<think>\n"):
            raise RuntimeError(f"P0 does not end with the assistant/<think> cue: {text[-60:]!r}")
        return text

    # ---------------------------------------------------------------- HTTP

    async def generate(self, client, ids, max_new):
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": self.a.temperature,
                "top_p": self.a.top_p,
                # **top_k must be sent explicitly.** generation_config.json says top_k=20, keeping only 20
                # candidates of the 248k vocabulary; without sending it the checkpoint's default applies.
                "top_k": -1,
                "max_new_tokens": max_new,
                "skip_special_tokens": False,
            },
        }
        for retry in range(self.a.retries + 1):
            try:
                r = await client.post(f"{self.a.base_url}/generate", json=payload)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                st = getattr(getattr(e, "response", None), "status_code", None)
                # A 4xx is a contract bug and must not be swallowed; only transport errors and 5xx back off and retry.
                if (st and st < 500) or retry == self.a.retries:
                    raise
                await asyncio.sleep(5 * (retry + 1))

    # ---------------------------------------------------------------- rollout

    async def one(self, client, task, attempt):
        t0 = time.time()
        uid = task["uid"]
        pairs = task["pairs"]
        setting = pack_menu.draw(uid, attempt)
        tr_idx, te_idx = choose_pairs(uid, attempt, len(pairs), self.a.n_train)
        train = [{"input": pairs[i][0], "output": pairs[i][1]} for i in tr_idx]
        test_input, expected = pairs[te_idx][0], pairs[te_idx][1]

        rec = {"uid": uid, "attempt": attempt, **setting.as_dict(),
               "pack_sha": setting.fingerprint(self.cfg, self.a.max_turns)[:16],
               "train_idx": tr_idx, "test_idx": te_idx,
               "n_turns": 0, "n_tool_calls": 0, "n_tool_errors": 0,
               "parsed": False, "correct": False, "correct_strict": False,
               "answer_src": "", "confidence": -1,
               "code_train_ok": -1, "code_train_n": -1, "code_test_match": -1,
               "retok_ok": -1}

        p0 = self.build_prompt(setting, train, test_input)
        ids = self.tok(p0, add_special_tokens=False)["input_ids"]
        spans = [[0, len(ids), 0]]
        rec["n_prompt"] = len(ids)

        cap = self.a.context_cap
        if len(ids) + self.a.min_turn_tokens > cap:
            # The task itself does not fit the budget. Keep the record, not the trajectory -- the only evidence of "which tasks cannot get in".
            rec.update(ids=ids, span_start=[0], span_end=[len(ids)], span_role=[0],
                       n_completion=0, stop_reason="prompt_too_long",
                       latency_s=time.time() - t0)
            return rec

        max_turns = self.a.max_turns if setting.uses_tools else 1
        sb = None
        stop = "turn_cap"
        final_content = final_reasoning = ""
        try:
            if setting.uses_tools:
                sb = self.Sandbox(train, test_input,
                                  python=self.a.sandbox_python,
                                  timeout=self.cfg["rollout"]["tool_timeout_s"],
                                  max_output_chars=self.cfg["rollout"]["tool_max_output_chars"])
                sb.__enter__()
            for _turn in range(max_turns):
                rec["n_turns"] += 1
                budget = cap - len(ids)
                if budget < self.a.min_turn_tokens:
                    stop = "context_cap"
                    break
                data = await self.generate(client, ids, budget)
                g = data["output_ids"]
                meta = data["meta_info"]
                # If the server says it generated N tokens, there must really be N. A mismatch is a broken contract,
                # not a tolerable error -- every span index of this trajectory would be misaligned.
                if len(g) != meta["completion_tokens"]:
                    raise RuntimeError(
                        f"{uid}: output_ids {len(g)} != completion_tokens {meta['completion_tokens']}")
                prev = len(ids)
                ids = ids + g
                spans.append([prev, len(ids), 1])
                text = self.tok.decode(g, skip_special_tokens=False)
                reasoning, content = split_generation(text)
                fr = meta.get("finish_reason") or {}
                fr_type = fr.get("type") if isinstance(fr, dict) else fr

                calls = parse_tool_calls(content, self.tool_name) if setting.uses_tools else []
                if not calls:
                    final_reasoning, final_content = reasoning, content
                    ans = setting.parse(content)
                    src = "content" if ans is not None else ""
                    if ans is None:
                        ans = setting.parse(reasoning)
                        src = "reasoning" if ans is not None else ""
                    rec["answer_src"] = src
                    rec["parsed"] = ans is not None
                    rec["correct"] = bool(ans is not None and ans == expected)
                    rec["correct_strict"] = bool(rec["correct"] and src == "content")
                    # **Truncation takes priority over "an answer can be fished out".** When generation hits the
                    # budget, even if a grid can be backscanned from the reasoning, it is only something the model wrote
                    # halfway through thinking, not a finished answer. This used to be recorded as
                    # answered_in_reasoning; measured, 106 of 1,400 records (7.6%) were overestimated this way, and
                    # downstream filtering on stop_reason would take truncated trajectories as complete answers.
                    # parsed / correct are still recorded, no information lost; stop_reason tells the truth.
                    if fr_type == "length":
                        stop = "context_cap"
                    elif ans is not None:
                        stop = "answered" if src == "content" else "answered_in_reasoning"
                    else:
                        stop = "malformed"
                    break

                outs = []
                for call in calls:
                    rec["n_tool_calls"] += 1
                    if call["name"] != self.tool_name:
                        rec["n_tool_errors"] += 1
                        outs.append(f"Error: no tool named {call['name']!r}.")
                        continue
                    code = call["params"].get("code")
                    if code is None:
                        rec["n_tool_errors"] += 1
                        outs.append("Error: the python tool needs a 'code' parameter.")
                        continue
                    # sb.run blocks synchronously (selectors.select + readline). Calling it directly in the coroutine
                    # would serialize tool execution across all rollouts, with zero HTTP handling meanwhile.
                    out, timed = await asyncio.to_thread(sb.run, code)
                    rec["n_tool_errors"] += int(timed)
                    outs.append(out)

                if fr_type == "length":
                    stop = "context_cap"
                    break
                seg = tool_segment(outs)
                seg_ids = self.tok(seg, add_special_tokens=False)["input_ids"]
                if len(ids) + len(seg_ids) >= cap:
                    stop = "context_cap"
                    break
                prev = len(ids)
                ids = ids + seg_ids
                spans.append([prev, len(ids), 2])
        finally:
            if sb is not None:
                sb.__exit__(None, None, None)

        rec["stop_reason"] = stop
        rec["n_completion"] = sum(b - a for a, b, r in spans if r == 1)
        rec["ids"] = ids
        rec["span_start"] = [s[0] for s in spans]
        rec["span_end"] = [s[1] for s in spans]
        rec["span_role"] = [s[2] for s in spans]

        blob = final_content or final_reasoning
        if blob:
            rec["confidence"] = extract_confidence(blob)
        if setting.wants_code and blob and not self.a.no_verify_code:
            await self.verify_code(blob, train, test_input, expected, rec)
        if self.a.audit:
            rec["retok_ok"] = int(
                self.tok(self.tok.decode(ids, skip_special_tokens=False),
                         add_special_tokens=False)["input_ids"] == ids)
        rec["latency_s"] = time.time() - t0
        return rec

    async def verify_code(self, blob, train, test_input, expected, rec):
        """Run the model's own transform(). In a clean sandbox, not the one it solved in -- that one's namespace is
        polluted by the model's intermediate variables, so the result would not be the behaviour of "this code itself"."""
        code = extract_code(blob)
        if not code:
            return
        try:
            with self.Sandbox(train, test_input,
                              python=self.a.sandbox_python,
                              timeout=self.cfg["rollout"]["tool_timeout_s"],
                              max_output_chars=self.cfg["rollout"]["tool_max_output_chars"]) as sb:
                out, _ = await asyncio.to_thread(sb.run, code + "\n" + _CODE_HARNESS)
        except Exception:  # noqa: BLE001  a sandbox failing to start must not void the whole rollout
            return
        if "<<<CODEVERIFY>>>" not in out:
            return
        try:
            r = json.loads(out.rsplit("<<<CODEVERIFY>>>", 1)[1].strip().splitlines()[0])
        except Exception:  # noqa: BLE001  the model's code may have dirtied stdout
            return
        rec["code_train_ok"] = int(r["train_ok"])
        rec["code_train_n"] = int(r["train_n"])
        rec["code_test_match"] = int(r["test"] == expected) if r.get("test") is not None else -1


# --------------------------------------------------------------------- writing to disk

class ShardWriter:
    """One shard every N records, written to .tmp first, then os.replace(). A hard kill leaves no half-written file for resume to trust."""

    def __init__(self, out_dir, flush_every, zstd_level):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.level = zstd_level
        self.buf = []
        self.n = self._next_index()

    def _next_index(self):
        existing = sorted(self.dir.glob("part-*.parquet"))
        return int(existing[-1].name[5:10]) + 1 if existing else 0

    def add(self, rec):
        self.buf.append(rec)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cols = {f.name: [r[f.name] for r in self.buf] for f in OUT_SCHEMA}
        table = pa.table(cols, schema=OUT_SCHEMA)
        final = self.dir / f"part-{self.n:05d}.parquet"
        tmp = self.dir / f".part-{self.n:05d}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd", compression_level=self.level,
                       use_dictionary=False)
        os.replace(tmp, final)
        self.n += 1
        self.buf = []


def done_keys(out_dir):
    keys = set()
    for f in sorted(Path(out_dir).glob("part-*.parquet")):
        t = pq.read_table(f, columns=["uid", "attempt"])
        keys.update(zip(t.column("uid").to_pylist(), t.column("attempt").to_pylist()))
    return keys


# --------------------------------------------------------------------- main

async def main_async(args):
    from transformers import AutoTokenizer

    cfg = json.loads(Path(args.config_json).read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sandbox_mod = _load(args.sandbox_path, "arc_sandbox")
    runner = Runner(args, cfg, tok, sandbox_mod.Sandbox)

    keys = done_keys(args.out) if not args.no_resume else set()
    if keys:
        print(f"resuming: {len(keys)} records already on disk", flush=True)

    todo = []
    for task in read_corpus(args.corpus, args.limit or None, args.skip):
        for at in range(args.attempts):
            if (task["uid"], at) not in keys:
                todo.append((task, at))
    print(f"{len(todo)} rollouts to run  (concurrency={args.concurrency} rollouts, "
          f"cap={args.context_cap})", flush=True)
    if not todo:
        return

    writer = ShardWriter(args.out, args.flush_every, args.zstd_level)
    # The semaphore wraps the whole rollout. The connection pool must be enlarged accordingly -- httpx defaults to
    # 100 and queues silently beyond that; the symptom is "concurrency set to 200 but only 100 running", with no error.
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency + 64,
                          max_keepalive_connections=args.concurrency + 64)
    t_start = time.time()
    state = {"done": 0, "fail": 0, "gen": 0, "ok": 0, "fit": 0}
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=args.http_timeout, limits=limits) as client:
        async def work(task, at):
            async with sem:
                try:
                    rec = await runner.one(client, task, at)
                except Exception as e:  # noqa: BLE001
                    async with lock:
                        state["fail"] += 1
                    print(f"  ! {task['uid']}#{at} {type(e).__name__}: {e}", flush=True)
                    return
            async with lock:
                state["done"] += 1
                state["gen"] += rec["n_completion"]
                state["ok"] += int(rec["correct"])
                state["fit"] += int(rec["stop_reason"].startswith("answered"))
                writer.add(rec)
                n = state["done"]
                el = time.time() - t_start
                if n % args.log_every == 0 or n <= 20:
                    print(f"[{n}/{len(todo)}] {rec['uid']}#{at} "
                          f"{rec['mode']}/{rec['instr']}/{rec['grid_fmt']}/{rec['effort']} "
                          f"turns={rec['n_turns']} ids={len(rec['ids']):,} "
                          f"gen={rec['n_completion']:,} {rec['stop_reason']} "
                          f"{'OK' if rec['correct'] else 'MISS'} "
                          f"| acc {state['ok']/n*100:.1f}% fit {state['fit']/n*100:.1f}% "
                          f"{state['gen']/max(el,1):,.0f} tok/s {el/60:.1f}m", flush=True)
                if args.heartbeat and n % args.log_every == 0:
                    with open(args.heartbeat, "a") as fh:
                        fh.write(f"{time.time():.0f}\t{n}\t{state['fail']}\t{state['ok']}\t"
                                 f"{state['fit']}\t{state['gen']}\t{state['gen']/max(el,1):.0f}\n")

        await asyncio.gather(*(work(t, a) for t, a in todo))

    writer.flush()
    el = time.time() - t_start
    print(f"done in {el:.0f}s  ok={state['done']} fail={state['fail']} "
          f"acc={state['ok']/max(state['done'],1)*100:.1f}% "
          f"fit={state['fit']/max(state['done'],1)*100:.1f}% "
          f"{state['gen']/max(el,1):,.0f} tok/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30031")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config-json", default=str(_ROOT / "src/arcforge/tir/tir_config.json"))
    ap.add_argument("--sandbox-path", default=str(_ROOT / "src/arcforge/tir/sandbox.py"))
    ap.add_argument("--sandbox-python", default=sys.executable)
    ap.add_argument("--context-cap", type=int, default=40960,
                    help="token cap per rollout for prompt + generation + tool output; truncated and ended when exceeded")
    ap.add_argument("--min-turn-tokens", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=200,
                    help="number of live rollouts at once (not HTTP requests)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--http-timeout", type=float, default=3600.0)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--zstd-level", type=int, default=12)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--heartbeat", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-verify-code", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="run the re-tokenise fidelity check on every record (slow, for sampling)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
