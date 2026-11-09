"""Nemotron prompt preparation and resumable Qwen token-in/token-out generation."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import shutil
import time
from pathlib import Path

DATASET = "nvidia/Nemotron-Post-Training-Dataset-v2"
EFFORTS = ("low", "medium", "xhigh")
SPLITS = ("chat", "code", "math", "stem", "multilingual_de", "multilingual_es", "multilingual_fr", "multilingual_it", "multilingual_ja")
SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical(value) + "\n")
    os.replace(tmp, path)


@contextlib.contextmanager
def exclusive(out, name):
    out.mkdir(parents=True, exist_ok=True)
    with (out / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prompt_prefixes(row):
    """Preserve source chat context, but never include the answer to the new turn."""
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    prefix = []
    for i, msg in enumerate(messages):
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported role {msg['role']!r}; tools need explicit schemas")
        if not isinstance(msg["content"], str):
            raise ValueError("only text messages are supported")
        if msg["role"] == "system" and i != 0:
            raise ValueError("system message must be first")
        if msg.get("tool_calls"):
            raise ValueError("tool calls require a tool schema")
        item = {"role": msg["role"], "content": msg["content"]}
        if msg.get("reasoning_content"):
            item["reasoning_content"] = msg["reasoning_content"]
        prefix.append(item)
        if msg["role"] == "user" and msg["content"].strip():
            yield i, list(prefix)


def prepare(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, get_token

    out = args.out
    with exclusive(out, "prepare"):
        if args.local_data:
            files = sorted(args.local_data.rglob("*.parquet"))
            if not files:
                raise ValueError("no source parquet files")
            # Local source identity includes size/mtime; remote sources use a pinned SHA.
            source = {"local": str(args.local_data.resolve()), "files": [
                [str(p.relative_to(args.local_data)), p.stat().st_size, p.stat().st_mtime_ns]
                for p in files]}
            opener = lambda p: p.open("rb")
        else:
            token = get_token()
            local_token = Path.home() / ".hf" / "token"
            if token is None and local_token.is_file():
                token = local_token.read_text().strip()
            info = HfApi(token=token).dataset_info(DATASET, revision=args.revision)
            files = sorted(x.rfilename for x in info.siblings if x.rfilename.endswith(".parquet"))
            source = {"dataset": DATASET, "revision": info.sha, "files": files}
            fs = HfFileSystem(token=token)
            opener = lambda p: fs.open(f"datasets/{DATASET}@{info.sha}/{p}", "rb")
        if args.limit_per_split:
            source["limit_per_split"] = args.limit_per_split
        files = [f for f in files if Path(f).name.split('-')[0] in SPLITS]
        exclude = set()
        exclude_runs = [Path(p).resolve() for p in (getattr(args, "exclude_runs", None) or [])]
        for run in exclude_runs:
            prior_db = run / "prompts.sqlite"
            if not prior_db.is_file():
                raise ValueError(f"exclude run missing prompts.sqlite: {run}")
            with sqlite3.connect(f"file:{prior_db}?mode=ro", uri=True) as prior:
                found = {uid for (uid,) in prior.execute("SELECT uid FROM prompts")}
            if not found:
                raise ValueError(f"exclude run has no prompts: {run}")
            exclude.update(found)
        if exclude_runs:
            source["exclude_runs"] = [str(p) for p in exclude_runs]
            source["exclude_uid_count"] = len(exclude)
            source["exclude_uids_sha256"] = digest(sorted(exclude))
        manifest = out / "source.json"
        if manifest.exists() and json.loads(manifest.read_text()) != source:
            raise ValueError("source differs from existing run")
        atomic_json(manifest, source)
        db = sqlite3.connect(out / "prompts.sqlite")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS prompts(uid TEXT PRIMARY KEY, messages TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources(uid TEXT, file TEXT, row_index INT, turn INT,
                metadata TEXT, PRIMARY KEY(file,row_index,turn));
            CREATE TABLE IF NOT EXISTS files(file TEXT PRIMARY KEY, rows INT);
            CREATE TABLE IF NOT EXISTS selected(split TEXT, uid TEXT, PRIMARY KEY(split,uid));
        """)
        for file in files:
            key = str(file)
            split = Path(file).name.split('-')[0]
            selected = db.execute("SELECT COUNT(*) FROM selected WHERE split=?", (split,)).fetchone()[0]
            if args.limit_per_split and selected >= args.limit_per_split:
                continue
            if db.execute("SELECT 1 FROM files WHERE file=?", (key,)).fetchone():
                continue
            print(f"reading {file}", flush=True)
            # One transaction per shard: interrupted shards can be replayed safely.
            with db, opener(file) as stream:
                if shutil.disk_usage(out).free < 5 * 1024**3:
                    raise RuntimeError("less than 5 GiB disk free; preparation stopped safely")
                parquet = pq.ParquetFile(stream)
                required = {"messages", "uuid", "license", "category"}
                if not required <= set(parquet.schema_arrow.names):
                    raise ValueError(f"unexpected Nemotron schema: {parquet.schema_arrow}")
                count = 0
                for batch in parquet.iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        meta = {k: v for k, v in row.items() if k != "messages"}
                        for turn, messages in prompt_prefixes(row):
                            uid = digest(messages)
                            if uid in exclude:
                                continue
                            db.execute("INSERT OR IGNORE INTO prompts VALUES (?,?)", (uid, canonical(messages)))
                            db.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?)",
                                       (uid, key, count, turn, canonical(meta)))
                            selected += db.execute("INSERT OR IGNORE INTO selected VALUES (?,?)", (split, uid)).rowcount
                            if args.limit_per_split and selected >= args.limit_per_split:
                                break
                        count += 1
                        if args.limit_per_split and selected >= args.limit_per_split:
                            break
                    if args.limit_per_split and selected >= args.limit_per_split:
                        break
                db.execute("INSERT INTO files VALUES (?,?)", (key, count))
            print(f"committed {file}: {count} rows", flush=True)
        db.execute("CREATE INDEX IF NOT EXISTS sources_uid ON sources(uid)")
        db.commit()
        result = {"source_sha256": digest(source), "prompts": db.execute("SELECT COUNT(*) FROM prompts").fetchone()[0],
                  "source_turns": db.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                  "per_split": dict(db.execute("SELECT split,COUNT(*) FROM selected GROUP BY split"))}
        atomic_json(out / "prepared.json", result)
        db.close()
        print(canonical(result), flush=True)


def render(tok, messages, effort):
    if effort not in EFFORTS:
        raise ValueError(effort)
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                  enable_thinking=True, reasoning_effort=effort)
    tail = tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    if ids[-len(tail):] != tail:
        raise ValueError("unexpected teacher generation prefix")
    return ids


def model_eos_ids(model, tok):
    """Use the checkpoint's complete EOS list, not only the chat-turn EOS."""
    config = json.loads((Path(model) / 'generation_config.json').read_text())
    declared = config['eos_token_id']
    declared = [declared] if isinstance(declared, int) else declared
    if not isinstance(declared, list) or not declared:
        raise ValueError('generation_config.eos_token_id must be an int or nonempty list')
    if any(type(token) is not int or not 0 <= token < len(tok) for token in declared):
        raise ValueError('invalid EOS token IDs in generation_config')
    return set(declared) | {tok.eos_token_id}


def assemble(uid, effort, prefix, response, eos, vocab_size):
    output = response["output_ids"]
    meta = response["meta_info"]
    if meta["prompt_tokens"] != len(prefix) or meta["completion_tokens"] != len(output):
        raise ValueError("server token counts do not match raw IDs")
    if not output or any(type(t) is not int or not 0 <= t < vocab_size for t in output):
        raise ValueError("empty or invalid output_ids")
    reason = meta["finish_reason"]["type"]
    if reason not in ("stop", "length"):
        raise ValueError(f"unexpected finish reason {meta['finish_reason']}")
    matched = meta['finish_reason'].get('matched')
    if reason == 'stop' and (type(matched) is not int or matched not in eos or matched != output[-1]):
        raise ValueError(f"invalid token stop: {uid}/{effort}, finish={meta['finish_reason']}, "
                         f"tail={output[-8:]}")
    if reason == "stop" and output[-1] not in eos:
        raise ValueError(f"normal completion must retain EOS in output_ids: {uid}/{effort}, "
                         f"tail={output[-8:]}, finish={meta['finish_reason']}, expected={sorted(eos)}")
    return {"uid": uid, "effort": effort, "input_ids": prefix + output,
            "loss_mask": [0] * len(prefix) + [1] * len(output),
            "n_prompt": len(prefix), "n_completion": len(output),
            "finish_reason": reason, "meta_info": canonical(meta),
            "split": "validation" if int(uid[:8], 16) % 100 == 0 else "train"}


class Writer:
    def __init__(self, out):
        import pyarrow as pa
        self.out = out
        out.mkdir(exist_ok=True)
        self.buffer = []
        self.index = len(list(out.glob("part-*.parquet")))
        self.last_flush = time.monotonic()
        self.schema = pa.schema([
            ("uid", pa.string()), ("effort", pa.string()), ("input_ids", pa.list_(pa.int32())),
            ("loss_mask", pa.list_(pa.int8())), ("n_prompt", pa.int32()), ("n_completion", pa.int32()),
            ("finish_reason", pa.string()), ("meta_info", pa.string()), ("split", pa.string()),
            ("provenance", pa.string()),
        ])

    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= 32 or time.monotonic() - self.last_flush > 30:
            self.flush()

    def flush(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not self.buffer:
            return
        dest = self.out / f"part-{self.index:07d}.parquet"
        if dest.exists():
            raise FileExistsError(dest)
        tmp = dest.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema), tmp, compression="zstd")
        os.replace(tmp, dest)
        self.index += 1
        self.buffer.clear()
        self.last_flush = time.monotonic()


async def generate(args):
    import httpx
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    out = args.out
    with exclusive(out, "generate"):
        prepared = json.loads((out / "prepared.json").read_text())
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        rendered = [render(tok, [{"role": "user", "content": "Hello"}], e) for e in EFFORTS]
        if len({tuple(x) for x in rendered}) != 3:
            raise ValueError("teacher tokenizer ignores reasoning_effort")
        model = Path(args.model)
        fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [model / n for n in ("config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json")] if p.is_file()}
        config = {"schema_version": SCHEMA_VERSION, "source": prepared, "model": str(model.resolve()),
                  "tokenizer_files": fingerprints, "template_sha256": digest(tok.chat_template),
                  "temperature": args.temperature, "top_p": 1.0, "top_k": -1,
                  "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
                  "all_efforts": args.all_efforts, "seed": args.seed}
        manifest = out / "generation.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError("generation config changed; choose a new output directory")
        atomic_json(manifest, config)
        done = set()
        skipped = set()
        skipped_path = out / "skipped.jsonl"
        if skipped_path.exists():
            for line in skipped_path.read_text().splitlines():
                item = json.loads(line)
                skipped.add((item["uid"], item["effort"]))
        for p in sorted((out / "tokens").glob("part-*.parquet")):
            table = pq.read_table(p, columns=["uid", "effort"])
            for r in table.to_pylist():
                key = (r["uid"], r["effort"])
                if key in done:
                    raise ValueError(f"duplicate completed key {key}")
                done.add(key)
        writer = Writer(out / "tokens")
        db = sqlite3.connect(f"file:{out / 'prompts.sqlite'}?mode=ro", uri=True)
        queue = asyncio.Queue(maxsize=args.concurrency * 2)
        state = {"completed_this_run": 0, "generated_tokens": 0, "skipped_long": 0, "resumed": len(done)}
        limits = httpx.Limits(max_connections=args.concurrency + 16, max_keepalive_connections=args.concurrency + 16)
        start = time.monotonic()
        eos = model_eos_ids(args.model, tok)
        termination = {'eos_token_ids': sorted(eos)}
        termination_file = out / 'termination.json'
        if termination_file.exists() and json.loads(termination_file.read_text()) != termination:
            raise ValueError('model EOS token set changed')
        atomic_json(termination_file, termination)
        async with httpx.AsyncClient(timeout=7200, limits=limits) as client:
            r = await client.get(args.server + "/get_server_info")
            r.raise_for_status()
            info = r.json()
            actual = info
            # This SGLang build resolves the NEXTN CLI alias to EAGLE at runtime.
            if actual["speculative_algorithm"] not in ("NEXTN", "EAGLE") or actual["speculative_eagle_topk"] != 1:
                raise ValueError("server is not running the native MTP linear chain")
            for key, expected in {"enable_linear_replayssm_spec": True,
                                  "mamba_ssm_dtype": "float32", "speculative_num_steps": 3}.items():
                if actual[key] != expected:
                    raise ValueError(f"server {key}={actual[key]!r}, expected {expected!r}")
            if Path(actual["model_path"]).resolve() != model.resolve():
                raise ValueError("server model path differs from tokenizer model")
            if actual["max_running_requests"] < args.concurrency:
                raise ValueError("server admission below requested concurrency")
            atomic_json(out / "server.json", info)

            async def producer():
                count = 0
                for uid, messages in db.execute("SELECT uid,messages FROM prompts ORDER BY uid"):
                    efforts = EFFORTS if args.all_efforts else (EFFORTS[int(digest([args.seed, uid, "effort"])[:8], 16) % 3],)
                    for effort in efforts:
                        if (uid, effort) in done or (uid, effort) in skipped:
                            continue
                        if args.limit and count >= args.limit:
                            break
                        provenance = [json.loads(x[0]) for x in db.execute(
                            "SELECT metadata FROM sources WHERE uid=?", (uid,))]
                        await queue.put((uid, json.loads(messages), effort, provenance))
                        count += 1
                    if args.limit and count >= args.limit:
                        break
                for _ in range(args.concurrency):
                    await queue.put(None)

            async def worker():
                while (job := await queue.get()) is not None:
                    uid, messages, effort, provenance = job
                    if shutil.disk_usage(out).free < 5 * 1024**3:
                        raise RuntimeError("less than 5 GiB disk free; generation stopped safely")
                    prefix = await asyncio.to_thread(render, tok, messages, effort)
                    budget = min(args.max_new_tokens, args.max_length - len(prefix))
                    if budget < 128:
                        state["skipped_long"] += 1
                        with (out / "skipped.jsonl").open("a") as f:
                            f.write(canonical({"uid": uid, "effort": effort, "n_prompt": len(prefix)}) + "\n")
                        continue
                    request = {"input_ids": prefix, "sampling_params": {
                        "temperature": args.temperature, "top_p": 1.0, "top_k": -1,
                        "max_new_tokens": budget, "skip_special_tokens": False}}
                    for retry in range(4):
                        try:
                            response = await client.post(args.server + "/generate", json=request)
                            response.raise_for_status()
                            break
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                                raise
                            if retry == 3:
                                raise
                            await asyncio.sleep(2 ** retry)
                    payload = response.json()
                    try:
                        row = assemble(uid, effort, prefix, payload, eos, len(tok))
                    except ValueError as exc:
                        errors = out / 'contract-errors'
                        errors.mkdir(exist_ok=True)
                        atomic_json(errors / f'{uid}-{effort}-{time.time_ns()}.json',
                                    {'error': str(exc), 'request': request, 'response': payload})
                        raise
                    row["provenance"] = canonical(provenance)
                    if row["n_completion"] > budget:
                        raise ValueError("server exceeded output budget")
                    writer.add(row)
                    state["completed_this_run"] += 1
                    state["generated_tokens"] += row["n_completion"]
                    if state["completed_this_run"] % 10 == 0:
                        progress = {**state, "elapsed_s": time.monotonic() - start}
                        atomic_json(out / "progress.json", progress)
                        print(canonical(progress), flush=True)
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(producer())
                    for _ in range(args.concurrency):
                        group.create_task(worker())
            finally:
                writer.flush()
                db.close()
                atomic_json(out / "progress.json", {**state, "elapsed_s": time.monotonic() - start})
        atomic_json(out / "invocation_complete.json", {**state, "limit": args.limit,
                    "total_completed": len(done) + state["completed_this_run"],
                    "total_skipped": len(skipped) + state["skipped_long"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--out", type=Path, required=True)
    prepare_p.add_argument("--local-data", type=Path)
    prepare_p.add_argument("--revision", default="5c89e01dd720ae0f4058445ed49c5fb68a03c76e")
    prepare_p.add_argument("--limit-per-split", type=int, default=0)
    prepare_p.add_argument("--exclude-runs", type=Path, nargs="*", default=[])
    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--model", required=True)
    gen.add_argument("--server", default="http://127.0.0.1:30471")
    gen.add_argument("--concurrency", type=int, default=300)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--max-length", type=int, default=40960)
    gen.add_argument("--max-new-tokens", type=int, default=4096)
    gen.add_argument("--seed", type=int, default=20260907)
    gen.add_argument("--all-efforts", action="store_true")
    gen.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        if min(args.concurrency, args.max_length, args.max_new_tokens) <= 0 or args.limit < 0 or args.temperature < 0:
            p.error("invalid numeric arguments")
        args.server = args.server.rstrip("/")
        asyncio.run(generate(args))


if __name__ == "__main__":
    main()
