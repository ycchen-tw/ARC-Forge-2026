"""Replay recorded TITO segments through capture_tito_runtime.py and write DFlash training features.

    python capture_tito.py --ledger <instance dir> --url http://127.0.0.1:<port> --queue <dir> --split train \
        --image-detail high --max-queue-gib <n> [--check-logprobs] [--only <seg>,...] [--kinds text,image,video] \
        [--shard i/n] [--limit N] [--passes N]

One segment per job, one job at a time (the server runs one request). The request is the segment's ids S with each
image's pad run collapsed to one <|image_pad|> and each video block (frame groups and their timestamps) collapsed to
<|vision_start|><|video_pad|><|vision_end|>, plus the image and video bytes. The server expands every placeholder from
the media itself and keeps every other id (SGLANG_MM_AVOID_RETOKENIZE; with a video since sglang-arc-2 4b57f3c45), so
the target sees S and the pixels it saw when the segment was generated.
Every check is fatal:

- the server is the capture server for this ledger instance (its served model name);
- prompt_tokens == len(S) (otherwise the server fell back to retokenizing);
- the captured chunks tile [0, len(S)) and their 1-D positions are the index (the drafter's positions);
- captured ids == S off the media; on media pads vocab_size - 1, or S itself (<|image_pad|> / <|video_pad|>) when the
  server runs --ple-mm-placeholder-ids (what the target read there, see check_capture);
- captured M-RoPE positions == the Qwen-VL positions recomputed from S and the media grids;
- finite GR.

Segments that cannot be replayed exactly are skipped and counted: media recorded by reference or unmapped, too long
for the server, fewer than 7 sampled tokens. Video frame sampling is the server's (Qwen's processor ignores per-request
``max_frames``), so a video replays from its bytes alone.

``--image-detail`` is the detail every image was sent with (codex sends "high"); the ledger does not record it. It only
enters each image's pad value (the radix key), which the target never reads (check_capture), so it is sent for fidelity.

``--passes N`` replays the selected segments N times with continuing job numbers: a soak load for the capture ->
train pipeline when the real corpus is elsewhere (each pass writes the same features again). Pass k shards by
``(seg + k) % n``, so over the passes every producer gets every segment and the producers stay even.

Each record carries ``queue_wait_seconds``, the time this producer waited on ``--max-queue-gib`` before the job: the
trainer was the bottleneck for that long.

``--save-logprobs <dir>`` stores every input token's logprob per segment (``<instance>__<seg>.npy``, position p = log
P(token p+1 | prefix)); with ``--logprob-sampling raw`` they are the model's own distribution (temperature 1, no
filters), the counterpart of ``work/qwen38fn_lora/train_tito_mm.py --mode score``.
``--check-logprobs`` also asks for input logprobs, at the recording's temperature/top-p/top-k, and compares them with the
logprobs the ledger recorded at sampling time; per-job stats go into record.json.

Output, in the layout train.py reads: ``<queue>/<split>-<n:09d>/features.safetensors`` -- ``input_ids`` = S with every
media pad (image or video) as <|image_pad|>, the drafter's view (arc_dflash/worker.py maps every out-of-vocab pad value
to it), ``loss_mask`` = the sampled spans, ``positions`` = arange, ``target_gr`` [N, 4, 2560] bf16 -- plus
``record.json`` and ``READY``.
"""

import argparse
import base64
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import Vocab, load_instance
from capture_tito_runtime import ledger_served_name
from safetensors.torch import load_file, save_file

MIN_GEN = 7
MAX_STAMP_TOKENS = 32  # a video frame group's timestamp text between its two <|vision_start|>
GR_SHAPE = (4, 2560)
GR_BYTES_PER_TOKEN = GR_SHAPE[0] * GR_SHAPE[1] * 2


class ModelSpec:
    """What the checks need from the model's config.json."""

    def __init__(self, model_dir, vocab):
        config = json.loads((Path(model_dir) / "config.json").read_text())
        text = config["text_config"]
        self.vocab_size = int(text["vocab_size"])
        self.merge = int(config["vision_config"]["spatial_merge_size"])
        for key, name in (("image_token_id", "image_pad"), ("video_token_id", "video_pad"),
                          ("vision_start_token_id", "vision_start"), ("vision_end_token_id", "vision_end")):
            if int(config[key]) != getattr(vocab, name):
                raise ValueError(f"config {key}={config[key]} but the tokenizer's {name} is {getattr(vocab, name)}")


def skip_reason(seg, max_len):
    if seg.n_gen < MIN_GEN:
        return "few_gen"
    if len(seg) + 1 > max_len:
        return "too_long"
    for m in seg.media:
        if m.unmapped or m.sha256 is None:
            return "media_unrecorded"
    return None


def outer_vision_start(ids, a0, vs, ve):
    """A video's outer <|vision_start|>: left of its first frame group's own <|vision_start|> (at a0 - 1), across only
    the timestamp text. None when another <|vision_end|> or nothing comes first."""
    for j in range(a0 - 2, max(a0 - 2 - MAX_STAMP_TOKENS, -1), -1):
        if ids[j] == vs:
            return j
        if ids[j] == ve:
            return None
    return None


@dataclass
class Media:
    ref: object                 # the ledger's MediaRef
    modality: str
    runs: list                  # pad runs [(a, b)]: one per image, one per video frame group
    grids: list                 # each run's M-RoPE grid (t, h, w); a video frame group is (1, h, w)
    block: tuple                # outer <|vision_start|> .. <|vision_end|>, inclusive


def media_items(ids, media, vocab, spec):
    """The segment's media in order, each checked against S. An image is <vs> t*h*w/merge^2 pads <ve>; a video is <vs>,
    then per frame group (grid t of them) timestamp text <vs> h*w/merge^2 pads <ve>, then <ve>."""
    items = []
    for m in media:
        if m.grid_thw is None:
            raise ValueError(f"media without a grid: {m}")
        t, h, w = m.grid_thw
        if m.modality == "image":
            pad, per_run, grids = vocab.image_pad, t * h * w // spec.merge**2, [(t, h, w)]
            if len(m.ranges) != 1:
                raise ValueError(f"an image with {len(m.ranges)} ranges")
        elif m.modality == "video":
            pad, per_run, grids = vocab.video_pad, h * w // spec.merge**2, [(1, h, w)] * t
            if len(m.ranges) != t:
                raise ValueError(f"a video with grid t={t} and {len(m.ranges)} frame groups")
        else:
            raise ValueError(f"unsupported modality {m.modality}")
        runs = [tuple(r) for r in m.ranges]
        for a, b in runs:
            if b - a + 1 != per_run or a < 1 or b + 1 >= len(ids):
                raise ValueError(f"{m.modality} run [{a}, {b}] has {b - a + 1} pads, grid {m.grid_thw} needs {per_run}")
            if ids[a - 1] != vocab.vision_start or ids[b + 1] != vocab.vision_end or not (ids[a:b + 1] == pad).all():
                raise ValueError(f"{m.modality} run [{a}, {b}] is not <|vision_start|> pads <|vision_end|>")
        (a0, _), (_, bn) = runs[0], runs[-1]
        if m.modality == "image":
            block = (a0 - 1, bn + 1)
        else:
            outer = outer_vision_start(ids, a0, vocab.vision_start, vocab.vision_end)
            if outer is None or bn + 2 >= len(ids) or ids[bn + 2] != vocab.vision_end:
                raise ValueError(f"video runs {runs[:2]} lack the outer <|vision_start|> ... <|vision_end|>")
            block = (outer, bn + 2)
        items.append(Media(m, m.modality, runs, grids, block))
    items.sort(key=lambda it: it.block[0])
    for x, y in zip(items, items[1:]):
        if y.block[0] <= x.block[1]:
            raise ValueError(f"media blocks {x.block} and {y.block} overlap")
    return items


def collapse(ids, items, vocab):
    """The request ids: an image's pad run becomes one <|image_pad|>; a video's block becomes
    <|vision_start|><|video_pad|><|vision_end|>, the placeholder its chat template rendered."""
    pieces, prev = [], 0
    for it in items:
        if it.modality == "image":
            (a, b), = it.runs
            pieces += [ids[prev:a], np.array([vocab.image_pad], ids.dtype)]
            prev = b + 1
        else:
            start, end = it.block
            pieces += [ids[prev:start + 1], np.array([vocab.video_pad], ids.dtype)]
            prev = end
    pieces.append(ids[prev:])
    out = np.concatenate(pieces)
    for pad, modality in ((vocab.image_pad, "image"), (vocab.video_pad, "video")):
        if (out == pad).sum() != sum(it.modality == modality for it in items):
            raise ValueError(f"a {modality} pad outside the recorded media")
    return out


def media_mask(n, items):
    mask = np.zeros(n, bool)
    for it in items:
        for a, b in it.runs:
            mask[a:b + 1] = True
    return mask


def expected_mrope(n, items, merge):
    """Qwen-VL M-RoPE positions (qwen_vl.py, mrope_rope_index.py for the qwen3_vl family): text advances all three rows
    by one; each pad run is a (t, h/merge, w/merge) grid at the running position -- a video frame group with t = 1, its
    timestamp being text -- after which the position advances by the grid's largest side."""
    pos = np.empty((3, n), np.int64)
    st = nxt = 0
    for (a, b), (t, h, w) in sorted((r, g) for it in items for r, g in zip(it.runs, it.grids)):
        pos[:, st:a] = nxt + np.arange(a - st)
        nxt += a - st
        gh, gw = h // merge, w // merge
        pos[0, a:b + 1] = nxt + np.repeat(np.arange(t), gh * gw)
        pos[1, a:b + 1] = nxt + np.tile(np.repeat(np.arange(gh), gw), t)
        pos[2, a:b + 1] = nxt + np.tile(np.arange(gw), t * gh)
        nxt += max(t, gh, gw)
        st = b + 1
    pos[:, st:] = nxt + np.arange(n - st)
    return pos


def assemble(parts, n):
    """The job's chunks, concatenated in order; they must tile [0, n)."""
    files = sorted(parts.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"{parts}: the capture hook wrote nothing (is the server capture_tito_runtime.py?)")
    chunks = [load_file(str(f)) for f in files]
    keys = set(chunks[0])
    if any(set(c) != keys for c in chunks):
        raise RuntimeError(f"{parts}: chunks disagree on what they carry")
    out = {k: torch.cat([c[k] for c in chunks], dim=-1 if k == "mrope_positions" else 0) for k in keys}
    if not torch.equal(out["positions"], torch.arange(n)):
        raise RuntimeError(f"{parts}: captured positions do not tile [0, {n}) -- missing, duplicated or reordered chunks")
    return out


def check_capture(ids, items, cap, spec):
    """Captured ids and M-RoPE positions against S.

    The hook reads ``forward_batch.input_ids`` after the forward, and the multimodal embed routine rewrites its media
    positions in place (managers/mm_utils.py): by default ``input_ids.clamp_(max=vocab_size - 1)``, so every media pad
    position the target -- its PLE n-gram embedding included -- read as vocab_size - 1, not SGLang's per-item pad
    value; under --ple-mm-placeholder-ids (``ple_placeholder_ids``, one flag per chunk) the modality's placeholder,
    i.e. S itself. So each pad run must read exactly that."""
    n = len(ids)
    got = cap["input_ids"].numpy()
    modes = cap["ple_placeholder_ids"]
    if modes.dtype != torch.bool or not (modes == modes[0]).all():
        raise RuntimeError(f"chunks disagree on the media ids the target read: {modes.tolist()}")
    on_media = media_mask(n, items)
    off = ~on_media
    if not np.array_equal(got[off], ids[off]):
        bad = int(np.flatnonzero(off & (got != ids))[0])
        raise RuntimeError(f"target saw id {got[bad]} at {bad}, the segment has {ids[bad]}")
    if bool(modes[0]):
        if not np.array_equal(got[on_media], ids[on_media]):
            raise RuntimeError(f"media positions read {np.unique(got[on_media])[:4]}, not the placeholders "
                               f"{np.unique(ids[on_media])[:4]}")
    elif not (got[on_media] == spec.vocab_size - 1).all():
        raise RuntimeError(f"media positions read {np.unique(got[on_media])[:4]}, not the clamped {spec.vocab_size - 1}")
    if "mrope_positions" not in cap:
        if items:
            raise RuntimeError("no M-RoPE positions captured for a request with media")
    elif not np.array_equal(cap["mrope_positions"].numpy(), expected_mrope(n, items, spec.merge)):
        raise RuntimeError("captured M-RoPE positions differ from the ones recomputed from S and the media grids")
    if not torch.isfinite(cap["target_gr"]).all():
        raise RuntimeError("nonfinite target GR")


def drafter_ids(ids, items, vocab):
    """S as the drafter reads it in serving: every media pad (its SGLang pad value is out of vocab) is <|image_pad|>."""
    out = ids.copy()
    out[media_mask(len(ids), items)] = vocab.image_pad
    return out


def loss_mask(seg):
    mask = np.zeros(len(seg), bool)
    for s in seg.gen_spans:
        mask[s.start:s.end] = True
    return mask


def recorded_sampling(seg):
    """The segment's one (temperature, top_p, top_k, min_p), or None when its requests disagree."""
    keys = ("temperature", "top_p", "top_k", "min_p")
    seen = {tuple(a.sampling[k] for k in keys) for a in seg.appends if a.gen_len}
    return dict(zip(keys, seen.pop())) if len(seen) == 1 else None


def logprob_stats(seg, ids, input_logprobs, items):
    """Capture-time input logprobs vs the ledger's sampling-time logprobs, on S's sampled tokens."""
    recorded = seg.gen_logprobs()
    if recorded is None:
        return {"skipped": "the ledger has no logprobs for this segment"}
    by_k = {a.k: a for a in seg.appends}
    pos = np.concatenate([np.arange(s.start, s.end) for s in seg.gen_spans])
    sampled = np.concatenate([by_k[s.k].gen[:s.end - s.start] for s in seg.gen_spans])
    keep = sampled == ids[pos]  # a substituted stop token was sampled as another id
    if len(input_logprobs) != len(ids):
        raise RuntimeError(f"{len(input_logprobs)} input logprobs for {len(ids)} tokens")
    returned = np.array([int(t) for _, t, *_ in input_logprobs], np.int64)
    on_media = media_mask(len(ids), items)
    if not np.array_equal(returned[~on_media], ids[~on_media]):  # media pads report SGLang's pad values
        raise RuntimeError("input logprob token ids differ from S off the media")
    cap = np.array([input_logprobs[p][0] for p in pos[keep]], np.float64)
    delta = cap - recorded[keep]
    first_media_end = min((it.block[1] for it in items), default=len(ids))
    after = pos[keep] > first_media_end
    absd = np.abs(delta)

    def summary(d):
        if not len(d):
            return None
        a = np.abs(d)
        return {"n": int(len(d)), "mean": float(d.mean()), "mean_abs": float(a.mean()),
                "p50_abs": float(np.quantile(a, 0.5)), "p99_abs": float(np.quantile(a, 0.99)), "max_abs": float(a.max())}

    return {"all": summary(delta), "after_first_media": summary(delta[after]), "before_first_media": summary(delta[~after]),
            "substituted_skipped": int((~keep).sum()), "max_abs_position": int(pos[keep][absd.argmax()]) if len(absd) else None,
            "media_position_ids": sorted({int(t) for t in returned[on_media]})[:16]}


def data_url(inst, m):
    return f"data:{m.mime};base64," + base64.b64encode(inst.media_path(m.sha256).read_bytes()).decode()


# ----------------------------------------------------------------------------- shared with capture_plan.py


def check_server(url, served_name):
    """The capture server's context length and /server_info; raises unless it is the server named served_name."""
    info = requests.get(url + "/server_info", timeout=30)
    info.raise_for_status()
    info = info.json()
    if info["served_model_name"] != served_name:
        raise RuntimeError(f"{url} serves {info['served_model_name']!r}, not {served_name!r}")
    return int(info["context_length"]), info


def server_context(url, instance):
    return check_server(url, ledger_served_name(instance))


def segment_request(inst, seg, vocab, spec, image_detail):
    """A segment's capture request: S (int64), its media items, and the request body's ids and media."""
    ids = seg.ids.astype(np.int64)
    items = media_items(ids, seg.media, vocab, spec)
    body = {"input_ids": collapse(ids, items, vocab).tolist()}
    images = [it.ref for it in items if it.modality == "image"]
    videos = [it.ref for it in items if it.modality == "video"]
    if images:
        body["image_data"] = [{"url": data_url(inst, m), "detail": image_detail} for m in images]
    if videos:
        body["video_data"] = [data_url(inst, m) for m in videos]
    return ids, items, body


def queue_bytes(root):
    total = 0
    for path in root.rglob("*.safetensors"):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue  # the trainer released this record after enumeration
    return total


def wait_for_room(root, num_tokens, max_queue_gib):
    """Seconds spent waiting until the queue root has room for one more job of num_tokens (parts, features and the
    assembly copy: 3x its GR)."""
    estimated = num_tokens * GR_BYTES_PER_TOKEN
    if estimated * 3 > max_queue_gib * 2**30:
        raise RuntimeError(f"a job of {num_tokens} tokens exceeds the queue bound with assembly space")
    began = time.monotonic()
    while queue_bytes(root) + 3 * estimated > max_queue_gib * 2**30:
        time.sleep(1)
    return time.monotonic() - began


def open_job(queue, job_id):
    """The job's directory, with the parts/ the server's hook writes into. Never reuses one."""
    directory = queue / job_id
    if directory.exists():
        raise FileExistsError(directory)
    (directory / "parts").mkdir(parents=True)
    return directory


def run_request(url, queue, job, body):
    """One capture request with job active (the hook writes its chunks under job["job_id"]); the response's meta_info.
    Exactly len(S) prompt tokens or it raises: anything else means the server re-tokenized."""
    control = queue / "active.json"
    temp = queue / "active.partial"
    temp.write_text(json.dumps(job))
    temp.rename(control)
    try:
        response = requests.post(url + "/generate", json=body, timeout=3600)
        response.raise_for_status()
        meta = response.json()["meta_info"]
    finally:
        control.unlink()
    if meta["prompt_tokens"] != job["num_tokens"]:
        raise RuntimeError(f"{job['job_id']}: the server saw {meta['prompt_tokens']} prompt tokens, the job has "
                           f"{job['num_tokens']} (it re-tokenized instead of expanding the placeholders?)")
    return meta


def write_features(directory, ids, items, mask, vocab, spec, check_only=False):
    """Assemble and check the captured chunks, then write the features train.py reads
    (``check_only``, --logprobs-only: assemble and check, write nothing)."""
    if mask.shape != ids.shape or mask.dtype != np.bool_:
        raise ValueError(f"loss mask {mask.dtype}{mask.shape} for {len(ids)} ids")
    cap = assemble(directory / "parts", len(ids))
    check_capture(ids, items, cap, spec)
    if not check_only:
        features = {"input_ids": torch.from_numpy(drafter_ids(ids, items, vocab)),
                    "loss_mask": torch.from_numpy(mask),
                    "positions": cap["positions"], "target_gr": cap["target_gr"]}
        save_file(features, str(directory / "features.partial"))
        (directory / "features.partial").rename(directory / "features.safetensors")
    shutil.rmtree(directory / "parts")


def publish(directory, record):
    """record.json, then READY: the trainer may take the job from here on."""
    (directory / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    (directory / "READY").touch()


def capture_server_meta(queue):
    """What the server's runtime wrote into the queue root (capture_tito_runtime.py), with the fields a record keeps."""
    meta = json.loads((queue / "capture_server.json").read_text())
    return meta, {k: meta[k] for k in ("sglang_commit", "sglang_dirty", "argv", "target_profile")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ledger", required=True, type=Path, help="one ledger instance directory")
    p.add_argument("--vocab", type=Path, help="tokenizer/model dir (default: the recorded --model-path)")
    p.add_argument("--url", required=True)
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("train", "heldout"))
    p.add_argument("--image-detail", required=True, choices=("auto", "low", "high"))
    p.add_argument("--check-logprobs", action="store_true")
    p.add_argument("--save-logprobs", type=Path,
                   help="also write every input token's logprob to <dir>/<instance>__<seg>.npy (train/serve parity)")
    p.add_argument("--logprobs-only", action="store_true",
                   help="with --save-logprobs: still check the captured ids/M-RoPE, but write no features")
    p.add_argument("--logprob-sampling", choices=("recorded", "raw"), default="recorded",
                   help="raw = temperature 1, no top-p/top-k/min-p: the model's own distribution")
    p.add_argument("--only", help="comma-separated segment numbers")
    p.add_argument("--kinds", default="text,image,video", help="segments with no media / images only / a video")
    p.add_argument("--shard", help="i/n: this producer takes segments with seg %% n == i (one queue and server per shard)")
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0, help="first job number")
    p.add_argument("--max-queue-gib", type=float, required=True)
    p.add_argument("--passes", type=int, default=1, help="replay the selected segments this many times (soak load)")
    args = p.parse_args()
    if args.passes < 1:
        raise ValueError("--passes must be positive")
    if args.logprob_sampling == "raw" and not (args.check_logprobs or args.save_logprobs):
        raise ValueError("--logprob-sampling raw needs --check-logprobs or --save-logprobs")
    if args.save_logprobs is not None:
        args.save_logprobs.mkdir(parents=True, exist_ok=True)
    if args.logprobs_only and args.save_logprobs is None:
        raise ValueError("--logprobs-only needs --save-logprobs")
    torch.set_num_threads(8)
    url = args.url.rstrip("/")
    server = json.loads((args.ledger / "server.json").read_text())
    model_dir = args.vocab or Path(server["server_args"]["model_path"])
    vocab = Vocab(model_dir)
    spec = ModelSpec(model_dir, vocab)
    inst = load_instance(args.ledger, vocab)
    bad = verify_media_blobs(inst)
    if bad:
        raise RuntimeError(f"{len(bad)} media blobs do not match their sha256, e.g. {bad[0]}")
    max_len, info = server_context(url, inst.name)
    kinds = set(args.kinds.split(","))
    if not kinds <= {"text", "image", "video"}:
        raise ValueError(f"--kinds {args.kinds}")
    only = None if args.only is None else {int(x) for x in args.only.split(",")}
    shard, shards = (0, 1) if args.shard is None else map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        raise ValueError(f"--shard {args.shard}")

    capture_server, kept = capture_server_meta(args.queue)
    if capture_server["instance"] != inst.name:
        raise RuntimeError(f"{args.queue} is the root of a capture server for {capture_server['instance']}")
    capture_server = kept
    if (args.queue / "active.json").exists():
        raise RuntimeError(f"{args.queue / 'active.json'} exists: another capture is active, or one died; inspect it")
    counts = {"jobs": 0, "tokens": 0, "skipped": {}, "queue_wait_seconds": 0.0}
    began = time.monotonic()
    number = args.offset
    ordered = sorted(inst.segments.values(), key=lambda s: s.seg)
    for pass_index, seg in ((k, s) for k in range(args.passes) for s in ordered):
        if (only is not None and seg.seg not in only) or (seg.seg + pass_index) % shards != shard:
            continue
        modalities = {m.modality for m in seg.media}
        if ("video" if "video" in modalities else "image" if modalities else "text") not in kinds:
            continue
        reason = skip_reason(seg, max_len)
        if reason is not None:
            counts["skipped"][reason] = counts["skipped"].get(reason, 0) + 1
            continue
        if args.limit is not None and counts["jobs"] >= args.limit:
            break
        ids, items, body = segment_request(inst, seg, vocab, spec, args.image_detail)
        waited = wait_for_room(args.queue, len(ids), args.max_queue_gib)
        counts["queue_wait_seconds"] += waited

        job_id = f"{args.split}-{number:09d}"
        directory = open_job(args.queue, job_id)
        raw = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
        wants_logprobs = args.check_logprobs or args.save_logprobs is not None
        sampling = (raw if args.logprob_sampling == "raw" else recorded_sampling(seg)) if wants_logprobs else None
        body.update(rid=f"tito-capture-{job_id}", sampling_params={"max_new_tokens": 1, **(sampling or {"temperature": 0})})
        if sampling is not None:
            body.update(return_logprob=True, logprob_start_len=0)
        job = {"job_id": job_id, "sequence_index": number, "scope": "tito_capture", "split": args.split,
               "instance": inst.name, "seg": seg.seg, "sid": seg.sid, "seg_status": seg.status,
               "num_tokens": len(ids), "n_gen": seg.n_gen, "n_images": sum(it.modality == "image" for it in items),
               "n_videos": sum(it.modality == "video" for it in items),
               "image_detail": args.image_detail, "image_pad": vocab.image_pad,
               "capture_server": capture_server, "queue_wait_seconds": waited}
        start = time.monotonic()
        meta = run_request(url, args.queue, job, body)
        write_features(directory, ids, items, loss_mask(seg), vocab, spec, check_only=args.logprobs_only)
        job.update(seconds=time.monotonic() - start, cached_tokens=meta.get("cached_tokens"))
        if sampling is not None and args.save_logprobs is not None:
            served = meta["input_token_logprobs"]
            if len(served) != len(ids) or served[0][0] is not None:
                raise RuntimeError(f"segment {seg.seg}: {len(served)} input logprobs for {len(ids)} tokens")
            off_media = ~media_mask(len(ids), items)
            returned = np.array([int(t) for _, t, *_ in served], np.int64)
            if not np.array_equal(returned[off_media], ids[off_media]):
                raise RuntimeError(f"segment {seg.seg}: input logprob token ids differ from S off the media")
            np.save(args.save_logprobs / f"{inst.name}__{seg.seg}.npy",
                    np.array([lp for lp, *_ in served[1:]], np.float32))
        if sampling is not None and args.check_logprobs:
            job["sampling"] = sampling
            job["logprobs"] = logprob_stats(seg, ids, meta["input_token_logprobs"], items)
        elif args.check_logprobs:
            job["logprobs"] = {"skipped": "the segment's requests used different sampling"}
        publish(directory, job)
        counts["jobs"] += 1
        counts["tokens"] += len(ids)
        number += 1
        brief = {k: job[k] for k in ("job_id", "seg", "num_tokens", "n_images", "n_videos")} | {
            "seconds": round(job["seconds"], 1)}
        if "logprobs" in job and "all" in job["logprobs"]:
            brief["dlogp_mean_abs"] = job["logprobs"]["all"] and round(job["logprobs"]["all"]["mean_abs"], 4)
        print(json.dumps(brief), flush=True)
    (args.queue / f"{args.split}.producer_done").touch()
    seconds = time.monotonic() - began
    counts["queue_wait_seconds"] = round(counts["queue_wait_seconds"], 1)
    print(json.dumps({"instance": inst.name, **counts, "seconds": round(seconds, 1),
                      "tokens_per_second": round(counts["tokens"] / max(seconds, 1e-9))}), flush=True)


if __name__ == "__main__":
    main()
