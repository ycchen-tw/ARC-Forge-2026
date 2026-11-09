"""OPD jobs from a rollout round: the DFlash worker's records (sglang fork arc_dflash/opd_log.py, ARC_DFLASH_OPD_DIR,
format arc-dflash-opd/1) joined with opd_rollout.py's outputs, written as pool jobs train.py --pool trains on
(docs/research/qwen38fn_dflash_opd_plan_20261007.md).

    python opd_assemble.py --records <ARC_DFLASH_OPD_DIR> --rollouts <rollouts.jsonl>... --queue <queue root> \
        --opd-plan <make_opd_plan.py dir> --pass P [--done] [--heldout]

Per finished turn (rid ``opd-r<R>-<index>``): the sequence is rebuilt from the records alone -- prefill pieces lay down
the prompt (the drafter's view: media pads as the image token), each cycle starts at the sequence's end (its anchor is
the last token) and appends its committed tokens (d1..dL, the bonus) -- and its generated part must equal the server's
output ids; cycles past that (overlap scheduling runs one more after a request finished) are dropped. The job covers
positions [s, end - 1), s = prompt_len - the recorder's prompt tail: GR from the prefill tail and, past the prompt,
each cycle's verify rows 0..L (the committed positions). The final token is left out -- its GR is never recorded, and
nothing reads it (a clean pass reads gr[:-1]; every anchor is before it). Positions are counted from s: rotary
attention depends on differences only, and the drafter's receptive field (layers x window) fits the tail.

Job (train-<sequence_index:012d>/, READY last): features.safetensors {input_ids, positions, target_gr, loss_mask
(generated)}, cycles.safetensors {anchor [C] (job coordinates), drafted [C, b-1], accepted [C] (L), verify_gr
[C, b, 4, D], q [C, b-1] (the rollout drafter's q of each draft), candidates [C, b-1, K] (its candidates)},
record.json {key, pass (--pass), corpus "opd", split "train", sequence_index (P * 10**6 + the plan index), num_tokens,
cycles, seconds, queue_wait_seconds, rid, prompt_len, generated, observed_accept}. The trainer deletes a job's directory once it has read it, so training a round's records
again (another epoch) means assembling them again under another --pass; (key, pass) is what the trainer's consumed-job
list names, so it is unique per queue root (an existing job directory is fatal).

The queue root is one root of train.py --pool: it gets a capture_server.json (pool.Planner wants one per root; the
capture servers' runtime writes theirs, this one names the OPD plan) on the first assemble, and every later one must
bring the same plan. --done marks the root's producer finished (train.producer_done) after the last READY, so the
trainer drains the pool and ends; no job can be added afterwards. The plan must be a train plan (make_opd_plan.py
--split train): a held-out plan's cycles never enter training. --heldout takes a held-out plan instead and writes
heldout-<index>/ jobs of split "heldout" (train.py --opd-heldout: the importance estimate of the trained drafter's
accepted length on turns it never trained on), marked done by heldout.producer_done.
Every inconsistency is fatal.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "arc-dflash-opd/1"


class Index:
    """Where each rid's records are, from the chunk sidecars alone (no tensor is read)."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = {}
        for path in sorted(self.directory.glob("*.meta.json")):
            value = json.loads(path.read_text())
            if value["format"] != FORMAT:
                raise RuntimeError(f"{path}: format {value['format']!r}, this reader reads {FORMAT}")
            self.meta[value["source"]] = value
        self.decode = defaultdict(list)   # rid -> [(source, seq, file, row)]
        self.prefill = defaultdict(list)  # rid -> [(source, seq, file, piece)]
        self.finished = {}
        for path in sorted(self.directory.glob("*.chunk.json")):
            side = json.loads(path.read_text())
            if side["source"] not in self.meta:
                raise RuntimeError(f"{path}: no {side['source']}.meta.json")
            if side["decode"] is not None:
                for cycle in side["decode"]["cycles"]:
                    for row in range(cycle["row"], cycle["row"] + cycle["rows"]):
                        rid = side["decode"]["rid"][row]
                        self.decode[rid].append((side["source"], cycle["seq"], side["decode"]["file"], row))
            if side["prefill"] is not None:
                for piece in side["prefill"]["pieces"]:
                    self.prefill[piece["rid"]].append((side["source"], piece["seq"], side["prefill"]["file"], piece))
            for event in side["events"]:
                if event["kind"] == "finish":
                    if event["rid"] in self.finished:
                        raise RuntimeError(f"{event['rid']}: finished twice")
                    self.finished[event["rid"]] = event
        (source,) = self.meta  # one server process per directory: the rid join is per source
        self.tail = self.meta[source]["prompt_tail"]
        self.width = self.meta[source]["width"]


class Files:
    """Open safetensors files, rows read on demand."""

    def __init__(self, directory):
        self.directory = directory
        self.open = {}

    def rows(self, file, name, index):
        handle = self.open.get(file)
        if handle is None:
            handle = self.open[file] = safe_open(str(self.directory / file), "pt")
        return handle.get_slice(name)[index : index + 1][0]

    def span(self, file, name, start, length):
        handle = self.open.get(file)
        if handle is None:
            handle = self.open[file] = safe_open(str(self.directory / file), "pt")
        return handle.get_slice(name)[start : start + length]


def assemble(index, files, rid, output_ids):
    """(features, cycles, info) of one finished turn, checked against the server's output ids."""
    items = [(s, q, 0, f, p) for s, q, f, p in index.prefill[rid]] + [(s, q, 1, f, r) for s, q, f, r in index.decode[rid]]
    items.sort(key=lambda x: x[:3])
    tokens, rows, gr_by_pos, prompt_len = [], [], {}, None
    for _, seq, kind, file, ref in items:
        if kind == 0:
            piece = ref
            if prompt_len is not None and piece["prompt_len"] != prompt_len:
                raise RuntimeError(f"{rid}: prompt length {piece['prompt_len']} after {prompt_len}")
            prompt_len = piece["prompt_len"]
            ids = files.span(file, "ids", piece["ids_offset"], piece["n_ids"]).tolist()
            if piece["start"] > len(tokens):
                raise RuntimeError(f"{rid}: prefill piece at {piece['start']}, the sequence has {len(tokens)} tokens")
            del tokens[piece["start"] :]  # a re-prefill (retraction) overwrites from its start
            for pos in [p for p in gr_by_pos if p >= piece["start"]]:
                del gr_by_pos[pos]
            tokens += ids
            gr = files.span(file, "gr", piece["gr_offset"], piece["n_gr"])
            for k in range(piece["n_gr"]):
                gr_by_pos[piece["gr_start"] + k] = gr[k]
            if piece["last"] and piece["chunk_end"] == piece["fill_len"]:
                tokens.append(int(files.span(file, "next_token", piece["piece"], 1)[0]))
        else:
            pos = int(files.rows(file, "position", ref))
            length = int(files.rows(file, "accept_len", ref))
            block = files.rows(file, "block", ref).tolist()
            committed = files.rows(file, "committed", ref).tolist()
            if not 1 <= length <= index.width:
                raise RuntimeError(f"{rid}: accept length {length}")
            if len(tokens) != pos + 1 or tokens[pos] != block[0]:
                if prompt_len is not None and len(tokens) >= prompt_len + len(output_ids):
                    continue  # an overshoot cycle after the finish: its tokens were discarded
                raise RuntimeError(f"{rid}: cycle anchor at {pos}, the sequence has {len(tokens)} tokens")
            verify = files.rows(file, "gr", ref)
            for k in range(length):
                gr_by_pos[pos + k] = verify[k]
            tokens += committed[:length]
            rows.append((file, ref, pos, block, length, verify))
    if prompt_len is None:
        raise RuntimeError(f"{rid}: no prefill record")
    end = prompt_len + len(output_ids)
    generated = tokens[prompt_len:end]
    if generated != list(output_ids):
        raise RuntimeError(f"{rid}: rebuilt output differs from the server's at {next(i for i, (a, b) in enumerate(zip(generated, output_ids)) if a != b) if len(generated) == len(output_ids) else 'length'}")
    start = prompt_len - index.tail if prompt_len > index.tail else 0
    stop = end - 1
    missing = [p for p in range(start, stop) if p not in gr_by_pos]
    if missing:
        raise RuntimeError(f"{rid}: no GR at {len(missing)} positions, first {missing[0]} (a cached prefix?)")
    # An anchor whose token and predecessor GR are in the job: the proposal reads ids[anchor] and gr[anchor - 1], the
    # clean pass rows up to anchor - 1. The last feature position (end - 2) qualifies -- a final cycle that accepted
    # nothing, or whose first token ended the turn, anchors there; the overshoot cycle anchors at end - 1 and does not.
    cycles = [r for r in rows if r[2] < stop]
    if not cycles:
        raise RuntimeError(f"{rid}: no cycle")
    q = torch.stack([files.rows(f, "q", ref) for f, ref, *_ in cycles])
    candidates = torch.stack([files.rows(f, "cand", ref) for f, ref, *_ in cycles])
    feature_ids = torch.tensor(tokens[start:stop], dtype=torch.long)
    features = {
        "input_ids": feature_ids,
        "positions": torch.arange(stop - start),
        "target_gr": torch.stack([gr_by_pos[p] for p in range(start, stop)]),
        "loss_mask": torch.arange(start, stop) >= prompt_len,
    }
    cycle_tensors = {
        "anchor": torch.tensor([pos - start for _, _, pos, *_ in cycles], dtype=torch.long),
        "drafted": torch.tensor([block[1:] for *_, block, _, _ in cycles], dtype=torch.long),
        "accepted": torch.tensor([length - 1 for *_, length, _ in cycles], dtype=torch.long),
        "verify_gr": torch.stack([verify for *_, verify in cycles]),
        "q": q.float(),
        "candidates": candidates.long(),
    }
    if (cycle_tensors["anchor"] < 1).any():
        raise RuntimeError(f"{rid}: an anchor at the job's first position has no predecessor GR")
    info = {"prompt_len": prompt_len, "generated": len(output_ids), "cycles": len(cycles), "num_tokens": stop - start,
            "observed_accept": float(cycle_tensors["accepted"].float().mean())}
    return features, cycle_tensors, info


SEQUENCE_STRIDE = 10**6  # sequence_index = pass * stride + plan index


def sequence_index(pass_index, plan_index):
    if not 0 <= pass_index < SEQUENCE_STRIDE:
        raise ValueError(f"--pass {pass_index}: must be in [0, {SEQUENCE_STRIDE})")
    if not 0 <= plan_index < SEQUENCE_STRIDE:
        raise ValueError(f"plan index {plan_index}: must be in [0, {SEQUENCE_STRIDE}) to fit its pass's numbering")
    return pass_index * SEQUENCE_STRIDE + plan_index


def load_opd_plan(directory, split="train"):
    """(opd_meta.json, {index: line}) of a make_opd_plan.py directory: the lines are the file the meta describes, and
    meta and lines are all of `split` (a held-out turn's cycles must not train the drafter)."""
    meta = json.loads((directory / "opd_meta.json").read_text())
    raw = (directory / "opd_plan.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["opd_plan_sha256"]:
        raise RuntimeError(f"{directory}: opd_plan.jsonl is not the plan opd_meta.json describes (sha256 differs)")
    if "split" not in meta:
        raise RuntimeError(f"{directory}: opd_meta.json has no split (a plan made before make_opd_plan.py --split; "
                           "make it again: the same --seed draws the same turns)")
    if meta["split"] != split:
        raise RuntimeError(f"{directory}: an OPD plan of split {meta['split']!r}, this assembles {split!r} jobs")
    lines = [json.loads(line) for line in raw.decode().splitlines()]
    if any(line["split"] != split for line in lines):
        raise RuntimeError(f"{directory}: opd_plan.jsonl has turns outside the {split} split")
    return meta, {line["index"]: line for line in lines}


def claim_queue_root(queue, opd_plan_dir, meta, split="train"):
    """Refuse a root whose producer is done; write the root's capture_server.json on the first assemble, and on a later
    one require that it is an OPD root of this very plan."""
    if (queue / f"{split}.producer_done").is_file():
        raise RuntimeError(f"{queue}: {split}.producer_done is set, no job can be added after --done")
    marker = queue / "capture_server.json"
    if not marker.is_file():
        marker.write_text(json.dumps({"producer": "opd_assemble", "opd_plan": str(opd_plan_dir.resolve()),
                                      "opd_plan_sha256": meta["opd_plan_sha256"],
                                      "target_profile": meta["target_profile"]}, indent=1) + "\n")
        return
    found = json.loads(marker.read_text())
    if found.get("producer") != "opd_assemble":
        raise RuntimeError(f"{marker}: producer {found.get('producer')!r}, not an opd_assemble queue root")
    if found["opd_plan_sha256"] != meta["opd_plan_sha256"]:
        raise RuntimeError(f"{marker}: the root's jobs are of OPD plan {found['opd_plan_sha256']}, this is "
                           f"{meta['opd_plan_sha256']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--rollouts", required=True, type=Path, nargs="+")
    p.add_argument("--queue", required=True, type=Path, help="one root of the trainer's --pool directory")
    p.add_argument("--opd-plan", required=True, type=Path)
    p.add_argument("--pass", dest="pass_index", required=True, type=int,
                   help="the epoch: the same records can be assembled again under another pass")
    p.add_argument("--done", action="store_true",
                   help="after the last job's READY, mark the root's producer done (no job can follow)")
    p.add_argument("--heldout", action="store_true",
                   help="a held-out plan (make_opd_plan.py --split heldout): heldout-<index> jobs for train.py "
                   "--opd-heldout, never for training")
    args = p.parse_args()
    began = time.monotonic()
    split = "heldout" if args.heldout else "train"
    meta, plan = load_opd_plan(args.opd_plan, split)
    sequence_index(args.pass_index, 0)
    args.queue.mkdir(parents=True, exist_ok=True)
    claim_queue_root(args.queue, args.opd_plan, meta, split)
    index = Index(args.records)
    files = Files(args.records)
    totals = defaultdict(float)
    for rollouts in args.rollouts:
        for line in rollouts.read_text().splitlines():
            turn = json.loads(line)
            if turn["rid"] not in index.finished:
                raise RuntimeError(f"{turn['rid']}: the server returned it but the records have no finish event")
            if plan[turn["index"]]["key"] != turn["key"]:
                raise RuntimeError(f"{turn['rid']}: plan index {turn['index']} is not {turn['key']}")
            number = sequence_index(args.pass_index, turn["index"])
            features, cycles, info = assemble(index, files, turn["rid"], turn["output_ids"])
            job = args.queue / f"{split}-{number:012d}"
            job.mkdir(exist_ok=False)
            save_file({k: v.contiguous() for k, v in features.items()}, str(job / "features.safetensors"))
            save_file({k: v.contiguous() for k, v in cycles.items()}, str(job / "cycles.safetensors"))
            record = {"key": turn["key"], "pass": args.pass_index, "corpus": "opd", "split": split,
                      "sequence_index": number, "rid": turn["rid"],
                      "seconds": turn["seconds"], "queue_wait_seconds": 0.0, **info}
            (job / "record.json").write_text(json.dumps(record) + "\n")
            (job / "READY").touch()
            for k in ("cycles", "generated", "num_tokens"):
                totals[k] += info[k]
            totals["jobs"] += 1
            totals["accepted"] += info["observed_accept"] * info["cycles"]
    if args.done:
        (args.queue / f"{split}.producer_done").touch()
    print(json.dumps({**totals, "mean_accepted": totals["accepted"] / max(totals["cycles"], 1), "pass": args.pass_index,
                      "done": args.done, "seconds": time.monotonic() - began}))


if __name__ == "__main__":
    main()
