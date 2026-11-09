"""Capture a data plan's jobs (make_data_plan.py) through one profile-mode capture server -- ledger segments and token
rows alike, in plan order -- into a queue root train.py --pool reads.

    python capture_plan.py --plan <plan dir>/plan.jsonl --shard i/n --url http://127.0.0.1:<port> --queue <root> \
        --max-queue-gib <n> [--split train|heldout] [--skip-consumed <consumed.jsonl>] [--limit N] [--corpora a,b]

One producer per capture server (one GPU), each with its own queue root, the server's TITO_CAPTURE_ROOT (start it
with capture_tito.sh, TARGET_PROFILE = the plan's profile, CONTEXT_LENGTH = the plan's context length). ``--split``
defaults to train; held-out jobs are captured by a separate ``--split heldout`` run into a root no trainer reads.

Checked at start, all fatal: plan.jsonl is the plan plan_meta.json describes; the server serves the plan's profile
(capture_tito_runtime.profile_served_name: the profile file's sha256) at the plan's context length; the root's
capture_server.json names the same profile; check_target passes for every ledger instance of the plan against the
profile; media blobs match their sha256. Per job, fatal: the source item is the one the plan saw (ids sha256, length,
supervised tokens, anchors, its key), it is still capturable, and capture_tito's capture checks (prompt_tokens == len,
the chunks tile [0, n), positions == index, captured ids and M-RoPE positions, finite GR).

Sharding (data_plan.shard_of): in plan order each job of the split goes to the shard with the fewest tokens so far, so
producers stay balanced and move through the plan together. The job id is ``<split>-<plan index:09d>`` and the record's
sequence_index is the plan index, so pool.py interleaves the roots in plan order (the plan's corpus mix) and a restart
never reuses an id. record.json adds key, pass, corpus, subset, group, split, plan_index and the plan's counts to what
capture_tito records (queue_wait_seconds, seconds = request + finalize).

Resume protocol (``--skip-consumed``): the trainer appends one line ``{"key": ..., "pass": ...}`` per job it has
finished training on (data_plan.read_consumed: every line must name a job of the plan; only a torn last line is
tolerated, and reported). A restarted producer, with the same plan and the same --shard i/n (producer.json in its root
records them and a mismatch is fatal), skips the consumed jobs and the ones still READY in its root, and removes and
recaptures a job directory without READY (a capture it died in). An active.json in the root means the server may still
be writing that job: inspect, then delete it. To change n, stop producers and trainer, delete the pool, and start every
producer with --skip-consumed on fresh roots.

The next jobs are read and assembled on a worker thread while the server runs the current one, and each job's
assembly, checks and write run on another while the server runs the next.
"""

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import capture_tito as ct
import capture_tito_runtime as rt
import torch
from data_plan import (
    CORPORA,
    LEDGER_CORPORA,
    Markers,
    RowReader,
    anchor_count,
    ids_sha256,
    job_id,
    load_plan,
    parquet_key,
    read_consumed,
    row_ids_mask,
    shard_of,
    text_skip,
)

from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import Vocab, load_instance, sha256_file

LOOKAHEAD = 2


class Sources:
    """What a producer needs to turn a plan job into a request, its features and its record."""

    def __init__(self, meta, needed_instances):
        model_dir = Path(meta["target_profile"]["model_path"])
        self.meta = meta
        self.vocab = Vocab(model_dir)
        if self.vocab.sha256 != meta["tokenizer"]["sha256"]:
            raise RuntimeError(f"{self.vocab.path} is not the tokenizer the plan was built with")
        self.spec = ct.ModelSpec(model_dir, self.vocab)
        self.markers = Markers(self.vocab)
        profile = json.loads(Path(meta["target_profile"]["path"]).read_text())
        self.instances = {}
        for entry in meta["tito_instances"]:
            path = Path(entry["path"])
            if sha256_file(path / "server.json") != entry["server_json_sha256"]:
                raise RuntimeError(f"{path}/server.json changed since the plan was built")
            rt.check_target(json.loads((path / "server.json").read_text()), profile, model_dir)
            if entry["name"] in needed_instances:
                inst = load_instance(path, self.vocab)
                bad = verify_media_blobs(inst)
                if bad:
                    raise RuntimeError(f"{path}: {len(bad)} media blobs do not match their sha256, e.g. {bad[0]}")
                self.instances[entry["name"]] = inst
        self.rows = RowReader(meta["sources"])

    def prepare(self, job):
        """(S int64, media items, loss mask, request body, record fields of the source), checked against the plan."""
        ref, corpus, context_length = job["ref"], job["corpus"], self.meta["context_length"]
        where = f"plan job {job['index']} ({job['key']})"
        if corpus in LEDGER_CORPORA:
            inst = self.instances[ref["instance"]]
            seg = inst.segments[ref["seg"]]
            if job["key"] != f"{corpus}:{inst.name}:{seg.seg}":
                raise RuntimeError(f"{where}: the ref names another segment")
            reason = ct.skip_reason(seg, context_length)
            if reason is not None:
                raise RuntimeError(f"{where}: {reason} -- the plan took it, this server cannot")
            mask = ct.loss_mask(seg)
            ids, items, body = ct.segment_request(inst, seg, self.vocab, self.spec, self.meta["image_detail"])
            extra = {"instance": inst.name, "seg": seg.seg, "sid": seg.sid, "seg_status": seg.status,
                     "n_gen": seg.n_gen, "n_images": sum(it.modality == "image" for it in items),
                     "n_videos": sum(it.modality == "video" for it in items),
                     "image_detail": self.meta["image_detail"], "image_pad": self.vocab.image_pad}
        else:
            src = self.meta["sources"][ref["file"]]
            if src["corpus"] != corpus:
                raise RuntimeError(f"{where}: source file {ref['file']} is {src['corpus']}")
            row = self.rows.row(ref["file"], ref["row"])
            if parquet_key(corpus, row, src["batch"]) != job["key"]:
                raise RuntimeError(f"{where}: {src['path']} row {ref['row']} is {parquet_key(corpus, row, src['batch'])}")
            kept = ref["kept_rounds"] if corpus == "agentic" else 0
            ids, mask, _, _ = row_ids_mask(corpus, row, self.markers, where, kept=kept)
            reason = text_skip(ids, mask, self.markers, context_length)
            if reason is not None:
                raise RuntimeError(f"{where}: {reason} -- the plan took it, this server cannot")
            items, body = [], {"input_ids": ids.tolist()}
            extra = {"source_file": src["path"], "source_row": ref["row"]}
        got = {"num_tokens": len(ids), "supervised": int(mask.sum()), "eligible_anchors": anchor_count(mask),
               "ids_sha256": ids_sha256(ids)}
        want = {k: job[k] for k in got}
        if got != want:
            raise RuntimeError(f"{where}: the source item is not the one the plan saw: {got} != {want}")
        return ids, items, mask, body, extra


def existing_jobs(queue, split, by_id):
    """READY job ids already in the root (checked against the plan); directories without READY are removed."""
    ready, removed = set(), []
    for directory in sorted(queue.glob(f"{split}-*")):
        if not (directory / "READY").is_file():
            shutil.rmtree(directory)
            removed.append(directory.name)
            continue
        record = json.loads((directory / "record.json").read_text())
        job = by_id.get(directory.name)
        if job is None or (record["key"], record["pass"], record["plan_index"]) != (job["key"], job["pass"],
                                                                                    job["index"]):
            raise RuntimeError(f"{directory} is not a job of this plan's shard")
        ready.add(directory.name)
    return ready, removed


def finalize(sources, directory, prepared, record, meta, request_seconds):
    began = time.monotonic()
    ids, items, mask, _, _ = prepared
    ct.write_features(directory, ids, items, mask, sources.vocab, sources.spec)
    record.update(request_seconds=request_seconds, finalize_seconds=time.monotonic() - began,
                  cached_tokens=meta.get("cached_tokens"))
    record["seconds"] = record["request_seconds"] + record["finalize_seconds"]
    ct.publish(directory, record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", required=True, type=Path, help="plan.jsonl (plan_meta.json beside it)")
    p.add_argument("--shard", required=True, help="i/n: this producer's share of the split")
    p.add_argument("--url", required=True)
    p.add_argument("--queue", required=True, type=Path, help="this server's queue root")
    p.add_argument("--max-queue-gib", type=float, required=True)
    p.add_argument("--split", default="train", choices=("train", "heldout"))
    p.add_argument("--skip-consumed", type=Path, help="the trainer's JSONL of finished {key, pass}")
    p.add_argument("--limit", type=int, help="stop after this many jobs (smoke)")
    p.add_argument("--corpora", help="only these corpora (measurement; a production producer takes every job)")
    args = p.parse_args()
    torch.set_num_threads(8)
    url = args.url.rstrip("/")
    shard, shards = map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        raise ValueError(f"--shard {args.shard}")
    meta, jobs = load_plan(args.plan)
    profile = meta["target_profile"]
    if hashlib.sha256(Path(profile["path"]).read_bytes()).hexdigest() != profile["sha256"]:
        raise RuntimeError(f"{profile['path']} changed since the plan was built")
    context_length, _ = ct.check_server(url, rt.profile_served_name(profile["path"]))
    if context_length != meta["context_length"]:
        raise RuntimeError(f"{url} has context length {context_length}, the plan was built for {meta['context_length']}")
    server, kept = ct.capture_server_meta(args.queue)
    if (server["instance"], server["target_profile"], server["target_profile_sha256"]) != (
            None, profile["path"], profile["sha256"]):
        raise RuntimeError(f"{args.queue} is the root of another capture server: {server['target_profile']}")
    if (args.queue / "active.json").exists():
        raise RuntimeError(f"{args.queue / 'active.json'} exists: a capture is active, or one died; inspect it")
    producer = {"plan_sha256": meta["plan_sha256"], "shard": args.shard, "split": args.split}
    if (args.queue / "producer.json").exists():
        before = json.loads((args.queue / "producer.json").read_text())
        if before != producer:
            raise RuntimeError(f"{args.queue} was produced as {before}; this producer is {producer}")
    (args.queue / "producer.json").write_text(json.dumps(producer) + "\n")
    done_marker = args.queue / f"{args.split}.producer_done"
    if done_marker.exists():
        done_marker.unlink()  # producing again (a resume): the trainer must wait for this run's jobs

    split_jobs = [job for job in jobs if job["split"] == args.split]
    owner = shard_of(split_jobs, shards)
    mine = [job for job in split_jobs if owner[job["index"]] == shard]
    if args.corpora is not None:
        corpora = set(args.corpora.split(","))
        if not corpora <= set(CORPORA):
            raise ValueError(f"--corpora {args.corpora}")
        mine = [job for job in mine if job["corpus"] in corpora]
    consumed, torn = (set(), None) if args.skip_consumed is None else read_consumed(args.skip_consumed, jobs)
    ready, removed = existing_jobs(args.queue, args.split, {job_id(args.split, j["index"]): j for j in mine})
    todo = [job for job in mine if (job["key"], job["pass"]) not in consumed
            and job_id(args.split, job["index"]) not in ready]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(json.dumps({"shard": args.shard, "split": args.split, "jobs": len(mine),
                      "tokens": sum(j["num_tokens"] for j in mine), "consumed": len(consumed),
                      "ready": len(ready), "removed_partial": removed, "torn_consumed_line": torn,
                      "todo": len(todo), "todo_tokens": sum(j["num_tokens"] for j in todo)}), flush=True)

    sources = Sources(meta, {j["ref"]["instance"] for j in todo if j["corpus"] in LEDGER_CORPORA})
    prep = ThreadPoolExecutor(1, thread_name_prefix="prepare")
    fin = ThreadPoolExecutor(1, thread_name_prefix="finalize")
    ahead = deque(prep.submit(sources.prepare, job) for job in todo[:LOOKAHEAD])
    queued = len(ahead)
    totals = Counter()
    by_corpus = {}
    began = time.monotonic()
    pending = None

    def account(record):
        totals["jobs"] += 1
        totals["tokens"] += record["num_tokens"]
        c = by_corpus.setdefault(record["corpus"], Counter())
        for k in ("num_tokens", "request_seconds", "finalize_seconds", "queue_wait_seconds"):
            c[k] += record[k]
        c["jobs"] += 1
        print(json.dumps({k: record[k] for k in ("job_id", "corpus", "num_tokens")} | {
            k: round(record[k], 2) for k in ("request_seconds", "finalize_seconds", "queue_wait_seconds")}), flush=True)

    try:
        for job in todo:
            prepared = ahead.popleft().result()
            if queued < len(todo):
                ahead.append(prep.submit(sources.prepare, todo[queued]))
                queued += 1
            waited = ct.wait_for_room(args.queue, job["num_tokens"], args.max_queue_gib)
            name = job_id(args.split, job["index"])
            directory = ct.open_job(args.queue, name)
            ids, items, mask, body, extra = prepared
            record = {"job_id": name, "sequence_index": job["index"], "plan_index": job["index"],
                      "scope": "plan_capture", "split": job["split"], "key": job["key"], "pass": job["pass"],
                      "corpus": job["corpus"], "subset": job["subset"], "group": job["group"], "ref": job["ref"],
                      "num_tokens": job["num_tokens"], "supervised": job["supervised"],
                      "eligible_anchors": job["eligible_anchors"], "ids_sha256": job["ids_sha256"],
                      "plan_sha256": meta["plan_sha256"], "shard": args.shard, "capture_server": kept,
                      "queue_wait_seconds": waited, **extra}
            body = dict(body, rid=f"plan-capture-{name}", sampling_params={"max_new_tokens": 1, "temperature": 0})
            start = time.monotonic()
            response = ct.run_request(url, args.queue, record, body)
            request_seconds = time.monotonic() - start
            if pending is not None:
                account(pending.result())
            pending = fin.submit(finalize, sources, directory, prepared, record, response, request_seconds)
        if pending is not None:
            account(pending.result())
    finally:
        prep.shutdown(wait=False, cancel_futures=True)
        fin.shutdown(wait=True)
    done_marker.touch()
    seconds = time.monotonic() - began
    report = {"shard": args.shard, "split": args.split, "jobs": totals["jobs"], "tokens": totals["tokens"],
              "seconds": round(seconds, 1), "tokens_per_second": round(totals["tokens"] / max(seconds, 1e-9))}
    for corpus, c in sorted(by_corpus.items()):
        report[corpus] = {"jobs": c["jobs"], "tokens": c["num_tokens"],
                          "serving_tokens_per_second": round(c["num_tokens"] / max(c["request_seconds"], 1e-9)),
                          "request_seconds": round(c["request_seconds"], 1),
                          "finalize_seconds": round(c["finalize_seconds"], 1),
                          "queue_wait_seconds": round(c["queue_wait_seconds"], 1)}
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
