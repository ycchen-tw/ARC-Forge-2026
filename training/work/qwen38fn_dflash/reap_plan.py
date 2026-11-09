"""REAP calibration plan over TITO ledgers: which segments reap_capture.py replays, in which split, in what order.

    python reap_plan.py inventory --ledger <instance dir> ... --model-dir <recorded model> --context-length <n> \
        --out <new dir>
    python reap_plan.py plan --inventory <dir>/items.jsonl --out <new dir> --calib-tokens 20e6 --heldout-tokens 3e6

CPU only. ``inventory`` reads every segment once (capture_tito's skip rule, data_plan's workspace classification and
grouping) and writes ``items.jsonl`` + ``summary.json``: per segment its pool, task, group, tokens, sampled tokens and
image / video pad tokens. ``plan`` picks from it and writes ``plan.jsonl`` + ``plan_meta.json`` (plan sha256 inside).

Pools: only ARC-AGI-2 ``training1000`` and ``nvarc_training`` (data_plan.CLEAN_ARC2). Never used, in any split:
independent1000 and synth108 (bench B and public-eval parents), arc3 (public-25 = bench C), smoke / probe sessions.
A pruned model's bench numbers stay clean only if its saliency never saw those tasks.

Held-out: whole groups (an ARC task with its NVARC variants, data_plan.node1_group) in a fixed hash order until
``--heldout-tokens``; their segments are never calibration. Calibration: the remaining eligible segments in a fixed
hash order until ``--calib-tokens``, so a prefix of the plan is a uniform sample and a run cut short is still usable.
Per-group weighting is left to the aggregation (every job is stored on its own).

``diag`` (``--diag-tokens``, optional): ARC-AGI-3 segments, **public-25 = bench C, tainted**. The clean pools hold no
video at all and only small grid images (~85 tokens an image; an ARC-AGI-3 frame is ~400), so the only video and game
frames in the ledgers are here: every segment with a video (hash order), then the most image-dense ones. Meant for
measuring how much of the media's REAP mass a clean mask keeps; a mask selected with it is not clean for bench C.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import capture_tito as ct
import capture_tito_runtime as rt
import numpy as np
from data_plan import (
    CLEAN_ARC2,
    node1_group,
    classify_threads,
    classify_workspace,
    ids_sha256,
    tito_mask,
    unit_hash,
    workspace,
)

from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import Vocab, load_instance, sha256_file

SALT = "reap-mm-plan-v1"  # the held-out choice and the calibration order; change it and plans stop being comparable


def media_tokens(seg):
    out = {"image": 0, "video": 0}
    for m in seg.media:
        out[m.modality] += sum(b - a + 1 for a, b in m.ranges)
    return out


def scan_instance(task):
    path, model_dir, context_length = task
    vocab = Vocab(model_dir)
    inst = load_instance(path, vocab)
    bad = verify_media_blobs(inst)
    if bad:
        raise RuntimeError(f"{path}: {len(bad)} media blobs do not match their sha256, e.g. {bad[0]}")
    items = []
    for seg in sorted(inst.segments.values(), key=lambda s: s.seg):
        mask, _ = tito_mask(seg, seg.key)
        if not np.array_equal(mask, ct.loss_mask(seg)):
            raise RuntimeError(f"{seg.key}: tito_mask and capture_tito.loss_mask disagree")
        found = workspace(seg, vocab)
        mt = media_tokens(seg)
        items.append({"key": seg.key, "instance": inst.name, "seg": seg.seg, "status": seg.status,
                      "num_tokens": len(seg), "ids_sha256": ids_sha256(seg.ids), "n_gen": int(mask.sum()),
                      "skip": ct.skip_reason(seg, context_length),
                      "thread": seg.thread if seg.sid is not None else None,
                      "where": None if found is None else classify_workspace(found),
                      "n_images": sum(m.modality == "image" for m in seg.media),
                      "n_videos": sum(m.modality == "video" for m in seg.media),
                      "image_tokens": mt["image"], "video_tokens": mt["video"]})
    return items


def summarize(items):
    by = defaultdict(Counter)
    for it in items:
        b = by[f"{it['pool']}|{'skip:' + it['skip'] if it['skip'] else 'ok'}"]
        b["segments"] += 1
        for k in ("num_tokens", "n_gen", "image_tokens", "video_tokens", "n_images", "n_videos"):
            b[k] += it[k]
        b["segments_with_video"] += it["n_videos"] > 0
        b["segments_with_image"] += it["n_images"] > 0
    return {k: dict(v) for k, v in sorted(by.items())}


def cmd_inventory(args):
    out = Path(args.out)
    out.mkdir(parents=True)  # never into an existing directory
    instances = []
    for d in args.ledger:
        d = Path(d)
        recorded = json.loads((d / "server.json").read_text())
        if recorded["instance"] != d.name:
            raise RuntimeError(f"{d}: server.json names instance {recorded['instance']}")
        rt.check_model_files(recorded, args.model_dir)
        instances.append({"name": d.name, "path": str(d.resolve()), "server_json_sha256": sha256_file(d / "server.json")})
    with mp.get_context("fork").Pool(len(instances)) as pool:
        parts = pool.map(scan_instance, [(i["path"], args.model_dir, args.context_length) for i in instances],
                         chunksize=1)
    items = [it for part in parts for it in part]
    where = classify_threads([(it["key"], it["thread"], it["where"]) for it in items])
    for it in items:
        it["where"] = where[it["key"]]
        it["pool"] = it["where"]["pool"]
        it["task"] = it["where"]["task"]
        it["group"], _ = node1_group(it["where"])
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    meta = {"model_dir": str(args.model_dir), "context_length": args.context_length, "instances": instances,
            "items": len(items), "items_sha256": sha256_file(out / "items.jsonl")}
    (out / "inventory_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    summary = summarize(items)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def cmd_plan(args):
    inv = Path(args.inventory)
    meta = json.loads((inv.parent / "inventory_meta.json").read_text())
    if sha256_file(inv) != meta["items_sha256"]:
        raise RuntimeError(f"{inv} is not the inventory inventory_meta.json describes")
    items = [json.loads(line) for line in inv.read_text().splitlines()]
    eligible = [it for it in items if it["pool"] in CLEAN_ARC2 and it["skip"] is None]
    groups = defaultdict(list)
    for it in eligible:
        groups[it["group"]].append(it)
    held, held_tokens = set(), 0
    for g in sorted(groups, key=lambda g: (unit_hash(SALT, "heldout", g), g)):
        if held_tokens >= args.heldout_tokens:
            break
        held.add(g)
        held_tokens += sum(it["num_tokens"] for it in groups[g])
    calib, calib_tokens = [], 0
    pool = sorted((it for it in eligible if it["group"] not in held), key=lambda it: (unit_hash(SALT, "calib", it["key"]),
                                                                                       it["key"]))
    for it in pool:
        if calib_tokens >= args.calib_tokens:
            break
        calib.append(it)
        calib_tokens += it["num_tokens"]
    heldout = sorted((it for it in eligible if it["group"] in held), key=lambda it: (unit_hash(SALT, "ho", it["key"]),
                                                                                      it["key"]))
    arc3 = [it for it in items if it["pool"] == "arc3" and it["skip"] is None]
    videos = sorted((it for it in arc3 if it["n_videos"]), key=lambda it: (unit_hash(SALT, "diag", it["key"]), it["key"]))
    images = sorted((it for it in arc3 if it["n_images"] and not it["n_videos"]),
                    key=lambda it: (-it["image_tokens"] / it["num_tokens"], it["key"]))
    diag, diag_tokens = [], 0
    for it in videos + images:
        if diag_tokens >= args.diag_tokens:
            break
        diag.append(it)
        diag_tokens += it["num_tokens"]
    out = Path(args.out)
    out.mkdir(parents=True)
    keep = ("key", "instance", "seg", "num_tokens", "ids_sha256", "n_gen", "pool", "task", "group", "n_images",
            "n_videos", "image_tokens", "video_tokens")
    lines = []
    for split, rows in (("calib", calib), ("heldout", heldout), ("diag", diag)):
        for it in rows:
            lines.append(json.dumps({"index": len(lines), "split": split, **{k: it[k] for k in keep}}))
    raw = ("\n".join(lines) + "\n").encode()
    (out / "plan.jsonl").write_bytes(raw)

    def totals(rows):
        c = Counter()
        for it in rows:
            c["segments"] += 1
            for k in ("num_tokens", "n_gen", "image_tokens", "video_tokens"):
                c[k] += it[k]
        c["groups"] = len({it["group"] for it in rows})
        c.update({f"pool:{p}": n for p, n in Counter(it["pool"] for it in rows).items()})
        return dict(c)

    plan_meta = {"salt": SALT, "inventory": str(inv.resolve()), "inventory_items_sha256": meta["items_sha256"],
                 "instances": meta["instances"], "model_dir": meta["model_dir"],
                 "context_length": meta["context_length"], "pools": list(CLEAN_ARC2),
                 "calib_tokens_target": args.calib_tokens, "heldout_tokens_target": args.heldout_tokens,
                 "diag_tokens_target": args.diag_tokens, "diag_tainted": "arc3 public-25 = bench C",
                 "jobs": len(lines), "plan_sha256": hashlib.sha256(raw).hexdigest(),
                 "calib": totals(calib), "heldout": totals(heldout), "diag": totals(diag),
                 "eligible": totals(eligible)}
    (out / "plan_meta.json").write_text(json.dumps(plan_meta, indent=1) + "\n")
    print(json.dumps({k: plan_meta[k] for k in ("jobs", "calib", "heldout", "diag", "eligible")}, indent=1))


def load_plan(plan_path):
    """(meta, jobs) of a plan, checked: plan.jsonl is the one plan_meta.json describes, indices 0..n-1."""
    plan_path = Path(plan_path)
    meta = json.loads((plan_path.parent / "plan_meta.json").read_text())
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["plan_sha256"]:
        raise RuntimeError(f"{plan_path} is not the plan plan_meta.json describes (sha256 differs)")
    jobs = [json.loads(line) for line in raw.decode().splitlines()]
    if len(jobs) != meta["jobs"] or any(job["index"] != i for i, job in enumerate(jobs)):
        raise RuntimeError(f"{plan_path}: {len(jobs)} jobs, meta says {meta['jobs']}, or indices are not 0..n-1")
    return meta, jobs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory")
    a.add_argument("--ledger", nargs="+", required=True, help="ledger instance directories (media unpacked)")
    a.add_argument("--model-dir", required=True, help="the recorded model (tokenizer, config; file digests checked)")
    a.add_argument("--context-length", type=int, required=True, help="the capture server's; longer segments skip")
    a.add_argument("--out", required=True)
    b = sub.add_parser("plan")
    b.add_argument("--inventory", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--calib-tokens", type=float, required=True)
    b.add_argument("--heldout-tokens", type=float, required=True)
    b.add_argument("--diag-tokens", type=float, default=0, help="arc3 (bench C, tainted) media segments; 0 = none")
    args = p.parse_args()
    {"inventory": cmd_inventory, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    main()
