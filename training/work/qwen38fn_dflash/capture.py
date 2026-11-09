"""Bounded final-GR producer; replays full exact trajectories through target."""

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
import torch
from data import validate_row
from safetensors.torch import load_file, save_file


def queue_bytes(root):
    total = 0
    for path in root.rglob("*.safetensors"):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # The consumer released this completed record after enumeration.
            # ENOENT is the expected concurrent-deletion case, not corruption.
            continue
    return total


def read_row(shard, index):
    f = pq.ParquetFile(shard)
    for rg in range(f.num_row_groups):
        size = f.metadata.row_group(rg).num_rows
        if index < size:
            return f.read_row_group(rg).slice(index, 1).to_pylist()[0]
        index -= size
    raise IndexError("row index outside parquet")


def assemble(parts, row):
    chunks = [load_file(str(p)) for p in sorted(parts.glob("*.safetensors"))]
    if not chunks:
        raise RuntimeError("runtime final-GR capture hook did not run")
    positions = torch.cat([c["positions"] for c in chunks])
    order = positions.argsort()
    if not torch.equal(positions[order], torch.arange(len(row["input_ids"]))):
        raise RuntimeError("missing/duplicate target replay positions; full prior context is required")
    ids = torch.cat([c["input_ids"] for c in chunks])[order]
    if not torch.equal(ids, torch.tensor(row["input_ids"])):
        raise RuntimeError("target replay changed input tokens")
    gr = torch.cat([c["target_gr"] for c in chunks])[order]
    if not torch.isfinite(gr).all():
        raise RuntimeError("nonfinite target GR")
    return {
        "input_ids": ids,
        "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.bool),
        "positions": positions[order],
        "target_gr": gr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--url", required=True)
    p.add_argument("--split", choices=("train", "heldout"), default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-queue-gib", type=float, default=24)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--check-logprobs", action="store_true")
    p.add_argument("--uid", help="single original prompt UID for a bounded correctness replay")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.queue.mkdir(parents=True, exist_ok=True)
    if (args.queue / "active.json").exists():
        raise RuntimeError("another capture is active; inspect its owner")
    db = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT shard,row_index,uid,effort FROM rows WHERE split=? AND n_completion>=7 AND (? IS NULL OR uid=?) ORDER BY uid,effort LIMIT ? OFFSET ?",
        (args.split, args.uid, args.uid, args.limit if args.limit is not None else -1, args.offset),
    )
    for number, (shard, idx, uid, effort) in enumerate(rows, args.offset):
        job_id = f"{args.split}-{number:09d}"
        directory = args.queue / job_id
        if directory.exists():
            raise FileExistsError(directory)
        row = read_row(shard, idx)
        if row["uid"] != uid or row["effort"] != effort:
            raise ValueError("parquet row does not match the immutable dataset index")
        validate_row(row, 248320)
        estimated = len(row["input_ids"]) * (20480 + 17)
        if estimated * 3 > args.max_queue_gib * 2**30:
            raise RuntimeError("one record exceeds bounded queue capacity including assembly workspace")
        while queue_bytes(args.queue) + 3 * estimated > args.max_queue_gib * 2**30:
            time.sleep(1)
        (directory / "parts").mkdir(parents=True)
        job = {
            "job_id": job_id,
            "sequence_index": number,
            "scope": "single_uid_gate" if args.uid else "dataset_epoch",
            "num_tokens": len(row["input_ids"]),
            "uid": uid,
            "effort": effort,
            "source_shard": shard,
            "source_row": idx,
            "split": args.split,
            "n_prompt": row["n_prompt"],
        }
        control = args.queue / "active.json"
        temp_control = args.queue / "active.partial"
        temp_control.write_text(json.dumps(job))
        temp_control.rename(control)
        start = time.monotonic()
        try:
            response = requests.post(
                args.url.rstrip("/") + "/generate",
                json={
                    "input_ids": row["input_ids"],
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_hidden_states": False,
                    "return_logprob": args.check_logprobs,
                    "logprob_start_len": 0,
                    "top_logprobs_num": 1 if args.check_logprobs else 0,
                },
                timeout=1800,
            )
            response.raise_for_status()
            payload = response.json()
            features = assemble(directory / "parts", row)
            save_file(features, str(directory / "features.partial"))
            (directory / "features.partial").rename(directory / "features.safetensors")
            job.update(
                seconds=time.monotonic() - start,
                captured_tokens=len(row["input_ids"]),
                target_meta=payload["meta_info"],
            )
            (directory / "record.json").write_text(json.dumps(job, indent=2) + "\n")
            shutil.rmtree(directory / "parts")
            (directory / "READY").touch()
            print(json.dumps({k: job[k] for k in ("job_id", "uid", "captured_tokens", "seconds")}), flush=True)
        finally:
            control.unlink()
    (args.queue / f"{args.split}.producer_done").touch()


if __name__ == "__main__":
    main()
