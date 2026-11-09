#!/usr/bin/env python
"""Render agentic_distill segments (the full messages the model saw before each fold of the compact harness) into
parquet with the same schema as qwen38fn_distill.

    $HOME/work/dsv4/venv-qwen4/bin/python work/agentic_distill/render_segments.py \
        --run $ARC_RUNS/agentic_distill/pilot-a $ARC_RUNS/agentic_distill/pilot-b \
        --corpus $HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst \
        --out $ARC_DATA/agentic_nvarc_segments --tokenizer $ARC_MODELS/Qwen3.8-Flash-Next-FP8 --workers 32

One segment = one row: `ids` is the tokenization of the segment's messages rendered one by one with the **teacher's chat
template**; `span_role` 0=prompt (system / task / harness notice / CONTINUE / the assistant generation prompt),
1=assistant (reasoning + content + tool call), 2=tool output. Each assistant span starts at the length of
`render(messages[:k], add_generation_prompt=True)` and ends at the length of `render(messages[:k+1])` -- this requires
the template to be append-only (each prefix's render is a prefix of the next one's); asserted per message in the loop,
and the whole task raises if it does not hold.

Two consistency checks, both stored as columns; smoke requires all of them to be 1:
- `prompt_match`: the segment's first API call's `usage.prompt_tokens` (the length the server actually saw) == the
  number of tokens I render before that assistant message. A mismatch means template kwargs (reasoning_effort) or the
  tool schema differ from the server -- the whole batch would then be "the prompt fed in training differs from what the
  teacher saw".
- `retok_ok`: token-backend segments (messages carry `_ids`) concatenate the raw tokens directly; this column records
  "does re-render + re-tokenize of the whole segment match token for token" (a mismatch is not an error, it is exactly
  why we take the token path; in smoke about 1/4 of segments differ, in how the assistant text is split). Fixed at -1
  for chat-API segments.

No per-call copy of the full messages (O(n²)): all calls in a segment share one prefix, and the span table is enough to
reconstruct each call's view.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agentic_tir"))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from arcforge.tir.protocol import load_config  # noqa: E402

SCHEMA = pa.schema([
    # ---- columns with the same names and types as qwen38fn_distill/rollout_runner.py OUT_SCHEMA (the downstream converter reads them directly)
    ("uid", pa.string()), ("attempt", pa.int16()),
    ("mode", pa.string()), ("instr", pa.string()), ("grid_fmt", pa.string()),
    ("label_shape", pa.bool_()), ("effort", pa.string()), ("pack_sha", pa.string()),
    ("train_idx", pa.list_(pa.int16())), ("test_idx", pa.int16()),
    ("ids", pa.list_(pa.int32())),
    ("span_start", pa.list_(pa.int32())), ("span_end", pa.list_(pa.int32())), ("span_role", pa.list_(pa.int8())),
    ("n_turns", pa.int16()), ("n_tool_calls", pa.int16()), ("n_tool_errors", pa.int16()),
    ("n_prompt", pa.int32()), ("n_completion", pa.int32()),
    ("stop_reason", pa.string()), ("parsed", pa.bool_()), ("correct", pa.bool_()),
    ("correct_strict", pa.bool_()), ("answer_src", pa.string()),
    ("confidence", pa.int16()), ("code_train_ok", pa.int16()), ("code_train_n", pa.int16()),
    ("code_test_match", pa.int8()), ("retok_ok", pa.int8()), ("latency_s", pa.float32()),
    # ---- agentic-only
    ("episode", pa.string()),          # "explorer 1 session 1" / "refiner round 1"
    ("lineage", pa.int8()), ("segment_idx", pa.int16()), ("n_segments", pa.int16()),
    ("n_compactions", pa.int16()),     # how many times this episode was folded in total
    ("ctx_cap", pa.int32()), ("n_explorers", pa.int8()),
    ("compact_target_frac", pa.float32()), ("compact_gen_budget", pa.int32()),
    ("task_correct", pa.bool_()),      # final pass@2 of the whole task (the refiner's correct is exactly this)
    ("prompt_match", pa.int8()),       # 1/0/-1 (-1 = this segment has no API call, should not happen)
    ("ep_peak_total", pa.int32()),     # peak context of this episode (server usage); > ctx_cap is old data from a runaway handoff loop, filterable
    ("run", pa.string()),
    ("source", pa.string()),           # nvarc_training / nvarc_full (the tasks json's source; old data without it looks up the uid in the corpus)
])

_TOK = None
_TOOL = None
_TOKENIZER_PATH = None


def _init(tokenizer_path: str):
    global _TOK, _TOOL, _TOKENIZER_PATH
    from transformers import AutoTokenizer
    _TOKENIZER_PATH = tokenizer_path
    _TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    assert _TOK.is_fast, "need a fast tokenizer for offsets_mapping"
    # The server side (sglang serving_chat.py) does `[item.model_dump() for item in request.tools]` before passing to
    # the template: pydantic adds "strict": false and "defer_loading": null, and the field order changes. Convert with
    # the same model, otherwise the prompt is off by 11 tokens (smoke: raw 3623 / model_dump 3634 = server usage).
    from sglang.srt.entrypoints.openai.protocol import Tool
    _TOOL = Tool(**load_config()["tool"]).model_dump()


def _normalize(messages: list[dict]) -> list[dict]:
    """The chat API returns tool_calls arguments as a JSON string; the template wants a mapping (`tool_call.arguments|items`).
    Broken JSON raises -- on the harness side the same case scraps the whole task, so such a segment cannot exist."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c["function"]
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def _render(msgs: list[dict], effort: str, gen: bool) -> str:
    return _TOK.apply_chat_template(msgs, tools=[_TOOL], tokenize=False, add_generation_prompt=gen,
                                    reasoning_effort=effort)


def render_segment(messages: list[dict], effort: str) -> tuple[list[int], list[tuple[int, int, int]], int]:
    """Returns (ids, spans[(start, end, role)], first_assistant_prompt_tokens)."""
    msgs = _normalize(messages)
    text = _render(msgs, effort, gen=False)
    enc = _TOK(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    starts = [o[0] for o in enc["offset_mapping"]]

    def n_tokens(prefix: str) -> int:
        c = len(prefix)
        if c >= len(text):
            assert c == len(text), (c, len(text))
            return len(ids)
        i = bisect.bisect_left(starts, c)
        assert i < len(starts) and starts[i] == c, f"char boundary {c} is inside a token (token {i} starts at {starts[i]})"
        return i

    spans: list[tuple[int, int, int]] = []
    prev_text = ""
    prev_end = 0
    first_prompt = -1
    for k, m in enumerate(msgs):
        if k == 0:
            continue   # the template rejects system-only messages (No user query); system + the first user form one prompt span
        cur = _render(msgs[:k + 1], effort, gen=False)
        assert cur.startswith(prev_text), f"template is not append-only at message {k} ({m['role']})"
        assert text.startswith(cur), f"prefix render of {k + 1} messages is not a prefix of the full render"
        end = n_tokens(cur)
        if m["role"] == "assistant":
            gp = _render(msgs[:k], effort, gen=True)
            assert cur.startswith(gp), f"generation prompt is not a prefix of the assistant render at message {k}"
            a0 = n_tokens(gp)
            if first_prompt < 0:
                first_prompt = a0
            if a0 > prev_end:
                spans.append((prev_end, a0, 0))
            spans.append((a0, end, 1))
        else:
            role = 2 if m["role"] == "tool" else 0
            if end > prev_end:
                spans.append((prev_end, end, role))
        prev_text, prev_end = cur, end
    assert prev_end == len(ids), (prev_end, len(ids))
    return ids, normalize_roles(spans), first_prompt


def normalize_roles(spans: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Downstream `convert_distill_shards._check_row` requires the role sequence `0 (1 2)* 1?` (the naive corpus shape:
    the cue is folded into the T span's role 2). Here everything before the first assistant (system / task / notice /
    cue) merges into one role 0, and everything between two assistants (tool output, CONTINUE / STOP and other user
    messages, the appended <|im_end|>, the next cue) merges into one role 2."""
    out: list[tuple[int, int, int]] = []
    seen_asst = False
    for s, e, r in spans:
        if r == 1:
            seen_asst = True
            out.append((s, e, 1))
            continue
        role = 2 if seen_asst else 0
        if out and out[-1][2] == role and out[-1][1] == s:
            out[-1] = (out[-1][0], e, role)
        else:
            out.append((s, e, role))
    return out


def assemble_segment(messages: list[dict]) -> tuple[list[int], list[tuple[int, int, int]], int]:
    """`--backend tokens` segments: each message carries `_ids` (assistant also has `_gen_len`); concatenated directly,
    no re-tokenize. Returns (ids, spans, tokens before the first assistant + cue length) -- the latter equals that
    /generate call's prompt_tokens."""
    assert messages[0]["role"] == "system" and "_ids" not in messages[0]
    ids: list[int] = []
    spans: list[tuple[int, int, int]] = []
    first_prompt = -1
    for m in messages[1:]:
        t = m["_ids"]
        if not t:
            continue
        s = len(ids)
        ids.extend(t)
        if m["role"] == "assistant":
            g = int(m["_gen_len"])
            n_out = int(m["_out_len"])
            assert 0 < g and g + n_out <= len(t) <= g + n_out + 1, (g, n_out, len(t))
            if first_prompt < 0:
                first_prompt = s + g
            spans.append((s, s + g, 0))
            spans.append((s + g, s + g + n_out, 1))       # only tokens the model actually generated are supervised
            if len(t) > g + n_out:
                spans.append((s + g + n_out, len(ids), 2))  # the <|im_end|> the harness appends on truncation: kept out of the loss
        else:
            spans.append((s, len(ids), 2 if m["role"] == "tool" else 0))
    return ids, normalize_roles(spans), first_prompt


def _expected_for(task: dict) -> list | None:
    return task.get("test_output")


def load_corpus_outputs(corpora: list[str], uids: set[str], attempt: int, n_train: int, n_test: int) -> tuple[dict[str, list], dict[str, str]]:
    """The output of the test pair drawn for each uid (tasks/*.json does not store the answer; the harness's correct is
    precomputed, but each explorer has to be compared on its own), plus uid → source (corpus file name)."""
    import agentic_tir as A
    out, src = {}, {}
    for corpus in corpora:
        name = Path(corpus).name.split(".")[0]
        pairs = A.load_nvarc_corpus(corpus)
        for u in uids:
            if u in pairs:
                t = A.nvarc_task(u, pairs[u], attempt, n_train, n_test)
                out[u] = t["test"][0]["output"]
                src[u] = name
    missing = uids - set(out)
    assert not missing, f"{len(missing)} uids not in any corpus, e.g. {sorted(missing)[:3]}"
    return out, src


def rows_for_task(run: str, task_path: str, expected: list | None, source: str = "") -> list[dict]:
    rec = json.loads(Path(task_path).read_text())
    uid = rec["task_id"]
    source = rec.get("source") or source
    hp = rec["hparams"] or {}
    seg_path = Path(run) / "segments" / f"{uid}.jsonl.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"{uid}: in tasks/ but not in segments/ -- did this run omit --save-segments?")
    segs = [json.loads(l) for l in gzip.open(seg_path, "rt", encoding="utf-8")]
    eps = {(e["label"], e["lineage"], e["hop"]): e for e in rec["explorers"] + rec["refines"]}
    task_correct = all(r["correct"] for r in rec["results"])
    effort = hp.get("reasoning_effort", "xhigh")
    rows = []
    for sg in segs:
        ep = eps[(sg["label"], sg["lineage"], sg["hop"])]
        token_mode = "_ids" in sg["messages"][1]
        if token_mode:
            ids, spans, first_prompt = assemble_segment(sg["messages"])
        else:
            ids, spans, first_prompt = render_segment(sg["messages"], effort)
        calls = [c for c in ep["calls"] if c.get("segment") == sg["segment_idx"]]
        # A folded segment starts with old kept rounds (already generated assistants); the prompt this segment's first
        # API call saw is all tokens before the first **new** assistant + cue: number of new assistants = number of calls
        # in this segment, so count from the end.
        asst_starts = [s for s, _, r in spans if r == 1]
        if calls and len(calls) <= len(asst_starts):
            first_prompt = asst_starts[len(asst_starts) - len(calls)]
        prompt_match = -1 if not calls else int(calls[0]["prompt_tokens"] == first_prompt)
        retok_ok = -1
        if token_mode:
            # Informational: re-render + tokenize this segment's messages and compare with the raw tokens. A mismatch is not an error (it is exactly why we take the token path); stored to measure the rate.
            try:
                re_ids, _, _ = render_segment(sg["messages"], effort)
                # The template has a '\n' after the last assistant's <|im_end|>, while the token stream stops at <|im_end|>; that tail is not a difference.
                if len(re_ids) == len(ids) + 1 and sg["messages"][-1]["role"] == "assistant":
                    re_ids = re_ids[:-1]
                retok_ok = int(re_ids == ids)
            except AssertionError:
                retok_ok = 0
        is_last = sg["segment_idx"] == sg["n_segments"] - 1
        is_refiner = sg["label"].startswith("refiner")
        if is_refiner:
            ep_correct = bool(rec["results"][0]["correct"])
        else:
            ans = ep["answers"][0] if ep["answers"] else []
            ep_correct = expected is not None and any(g == expected for g in ans)
        hc = ep.get("home_check") or {}
        train = hc.get("train") or []
        test_grid = None
        try:
            test_grid = hc["test"][0]["grid"]
        except (KeyError, IndexError, TypeError):
            pass
        n_asst = sum(1 for m in sg["messages"] if m["role"] == "assistant")
        n_tool = sum(1 for m in sg["messages"] if m["role"] == "tool")
        n_err = sum(1 for m in sg["messages"] if m["role"] == "tool"
                    and (m.get("content") or "").lstrip().startswith(("Error", "Traceback")))
        n_completion = sum(e - s for s, e, r in spans if r == 1)
        system = sg["messages"][0]["content"]
        rows.append({
            "uid": uid, "attempt": int(rec.get("attempt", 0)),
            "mode": "agentic", "instr": "refiner" if is_refiner else "explorer", "grid_fmt": "json",
            "label_shape": False, "effort": effort,
            "pack_sha": hashlib.sha256((system + "\n" + json.dumps(hp, sort_keys=True)).encode()).hexdigest()[:16],
            "train_idx": list(rec.get("train_idx") or []), "test_idx": int((rec.get("test_idx") or [0])[0]),
            "ids": ids, "span_start": [s for s, _, _ in spans], "span_end": [e for _, e, _ in spans],
            "span_role": [r for _, _, r in spans],
            "n_turns": n_asst, "n_tool_calls": n_tool, "n_tool_errors": n_err,
            "n_prompt": len(ids) - n_completion, "n_completion": n_completion,
            "stop_reason": ep["stop_reason"] if is_last else "compacted",
            "parsed": bool((ep.get("report") or {}).get("has_sections")) if is_last else False,
            "correct": bool(ep_correct), "correct_strict": bool(task_correct),
            "answer_src": ep.get("code_src") or "",
            "confidence": int((ep.get("report") or {}).get("confidence") or -1),
            "code_train_ok": sum(1 for r in train if r == "pass"), "code_train_n": len(train),
            "code_test_match": -1 if (test_grid is None or expected is None) else int(test_grid == expected),
            "retok_ok": retok_ok, "latency_s": float(ep.get("latency_s") or 0.0),
            "episode": sg["label"], "lineage": int(sg["lineage"]), "segment_idx": int(sg["segment_idx"]),
            "n_segments": int(sg["n_segments"]), "n_compactions": int(ep.get("n_compactions") or 0),
            "ctx_cap": int(hp.get("ctx_cap", 0)), "n_explorers": int(hp.get("n_explorers", 0)),
            "compact_target_frac": float(hp.get("compact_target_frac", 0.0)),
            "compact_gen_budget": int(hp.get("compact_gen_budget", 0)),
            "task_correct": bool(task_correct), "prompt_match": prompt_match,
            "ep_peak_total": int(ep.get("peak_total") or 0), "run": Path(run).name, "source": source,
        })
    return rows


def _work(item):
    run, task_path, expected, source = item
    try:
        return rows_for_task(run, task_path, expected, source), None
    except Exception as e:  # one broken task must be recorded, but must not waste the other tens of thousands
        return [], f"{task_path}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", nargs="+", required=True, help="nvarc jsonl.zst used by that run (several allowed; per-explorer correct needs the test output)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=200, help="tasks per parquet file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = []
    for run in args.run:
        a = json.loads((Path(run) / "args.json").read_text())
        for p in sorted(glob.glob(str(Path(run) / "tasks" / "*.json"))):
            items.append((run, p, a))
    if args.limit:
        items = items[: args.limit]
    uids = {Path(p).stem for _, p, _ in items}
    a0 = items[0][2]
    expected, source = load_corpus_outputs(args.corpus, uids, a0["attempt"], a0["n_train"], a0["n_test"])
    work = [(run, p, expected[Path(p).stem], source[Path(p).stem]) for run, p, _ in items]
    print(f"{len(work)} tasks from {len(args.run)} run(s); workers={args.workers}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    n_shard = 0
    stats = collections.Counter()
    errors = []

    def flush():
        nonlocal buf, n_shard
        if not buf:
            return
        cols = {f.name: [r[f.name] for r in buf] for f in SCHEMA}
        table = pa.table(cols, schema=SCHEMA)
        tmp = out / f"part-{n_shard:05d}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out / f"part-{n_shard:05d}.parquet")
        n_shard += 1
        buf = []

    n_tasks = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for rows, err in pool.imap_unordered(_work, work, chunksize=1):
            n_tasks += 1
            if err:
                errors.append(err)
                stats["tasks_failed"] += 1
                continue
            buf.extend(rows)
            stats["rows"] += len(rows)
            stats["tokens"] += sum(len(r["ids"]) for r in rows)
            stats["sup_tokens"] += sum(r["n_completion"] for r in rows)
            stats["prompt_match_1"] += sum(1 for r in rows if r["prompt_match"] == 1)
            stats["prompt_match_0"] += sum(1 for r in rows if r["prompt_match"] == 0)
            if n_tasks % args.shard_size == 0:
                flush()
            if n_tasks % 500 == 0:
                print(f"  {n_tasks}/{len(work)} tasks, {stats['rows']} rows", flush=True)
    flush()
    print(f"done: {n_tasks} tasks -> {stats['rows']} rows, {stats['tokens'] / 1e6:.1f}M tokens "
          f"({stats['sup_tokens'] / 1e6:.1f}M supervised), prompt_match 1/0 = {stats['prompt_match_1']}/{stats['prompt_match_0']}, "
          f"{n_shard} shards, {stats['tasks_failed']} tasks failed", flush=True)
    if errors:
        (out / "render_errors.txt").write_text("\n".join(errors) + "\n")
        print(f"!! {len(errors)} tasks failed to render; see {out / 'render_errors.txt'}", flush=True)
        for e in errors[:5]:
            print("   ", e[:300])
    if stats["prompt_match_0"]:
        print(f"!! {stats['prompt_match_0']} segments have a prompt length that does not match server usage -- do not train on them; check template kwargs / tool schema first",
              flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
