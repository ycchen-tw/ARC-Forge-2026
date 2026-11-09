"""Replay a REAP plan (reap_plan.py) through reap_runtime.py and store each job's REAP cumulants.

    python reap_capture.py --plan <dir>/plan.jsonl --split calib --url http://127.0.0.1:<port> --root <server root> \
        --out <jobs dir> --image-detail high [--limit N] [--check-routing N]

One job (a ledger segment) at a time, the plan's order within the split. The request is capture_tito's
(segment_request: S with each image / video collapsed to its placeholder, plus the media bytes), max_new_tokens 1.
Before it the driver writes ``<root>/<job>/labels.npy``: per token of S its category, reap_runtime.CATEGORIES --
media pads by modality, the segment's sampled spans (capture_tito.loss_mask) as "gen", everything else "prompt".

Checked per job, all fatal: the segment is the one the plan saw (ids sha256, still not skipped); the server saw
len(S) prompt tokens; rank 0's chunks tile [0, len(S)) and read S off the media and the placeholder (or the clamped
id) on it, with the M-RoPE positions recomputed from S (capture_tito's checks without the GR); on every rank the REAP
chunks tile [0, len(S)), each chunk hooked every MoE layer once and counted exactly the job's category tokens; the
ranks' EP slices partition the experts. Then ``<out>/<job>/reap.safetensors`` -- ``acc`` [layers, fields, categories,
experts] float32 over global expert ids, ``tokens`` per category -- plus ``record.json`` and ``DONE``.

``--check-routing N``: for the first N jobs, also compare the per-(layer, expert) routing counts with the routing the
recording server logged (flashinfer_trtllm, the production path) on the same positions: the per-layer share of
routed slots that land on different experts. Small and nonzero is expected (reduction order); large means the Triton
capture does not route like production.

Resume: a job directory with DONE is skipped; one without DONE (a crash) is removed and redone. An active.json in
the root means the server may still be writing a job: inspect, then delete it.

Fidelity mode: ``--router-mask <mask.pt>`` makes the server route as the pruned model would (reap_runtime: -inf on
the dropped experts' logits); ``--logprobs`` asks for every input token's logprob under the raw distribution
(temperature 1, no filters) and stores ``logprobs.npy`` (float32, position p-1 = log P(token p | S[:p])) and
``labels.npy`` next to the cumulants. Run the same split once without a mask and once per candidate mask, then compare
with qwen38fn_eval/reap_mm_fidelity.py.

Expert patch: ``--expert-patch <file>`` swaps the file's routed experts into the served model for the run (e.g. a
merge's rebuilt centroids; with --router-mask dropping the absorbed experts); the server restores them before the
next job without that patch. ``--expert-patch-identity`` checks the loading path with the checkpoint's own tensors.

Pairs mode: ``--pairs-hidden-stride S`` also has the server sum reap_runtime.Pairs over the whole run (expert-pair
counts / norm products / output Gram, mean outputs, router logit Gram, the MoE inputs of every S-th position) under a
tag of this run; the last job flushes it, and it lands in ``<out>/pairs.safetensors`` + ``pairs.json`` (the jobs, in
order -- checked against this run's). No resume: the out directory must hold no finished job of the run.
"""

import argparse
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import capture_tito as ct
import capture_tito_runtime as rt
import numpy as np
import torch
from reap_plan import load_plan
from reap_runtime import CATEGORIES, FIELDS, reap_served_name
from safetensors.torch import load_file, save_file

from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import ROUTING_R, Vocab, ids_sha256, load_instance, segment_routing_status


def job_labels(seg, ids, items):
    labels = np.zeros(len(ids), np.int8)
    gen = ct.loss_mask(seg)
    labels[gen] = CATEGORIES.index("gen")
    media = np.zeros(len(ids), bool)
    for it in items:
        for a, b in it.runs:
            labels[a:b + 1] = CATEGORIES.index(it.modality)
            media[a:b + 1] = True
    if (media & gen).any():
        raise RuntimeError(f"{seg.key}: a media pad inside a sampled span")
    return labels


def check_replay(ids, items, cap, spec):
    """capture_tito.check_capture without the GR: ids as the target read them, and M-RoPE positions."""
    n = len(ids)
    got = cap["input_ids"].numpy()
    modes = cap["ple_placeholder_ids"]
    if modes.dtype != torch.bool or not (modes == modes[0]).all():
        raise RuntimeError(f"chunks disagree on the media ids the target read: {modes.tolist()}")
    on_media = ct.media_mask(n, items)
    off = ~on_media
    if not np.array_equal(got[off], ids[off]):
        bad = int(np.flatnonzero(off & (got != ids))[0])
        raise RuntimeError(f"target saw id {got[bad]} at {bad}, the segment has {ids[bad]}")
    if bool(modes[0]):
        if not np.array_equal(got[on_media], ids[on_media]):
            raise RuntimeError("media positions do not read the placeholders")
    elif not (got[on_media] == spec.vocab_size - 1).all():
        raise RuntimeError(f"media positions read {np.unique(got[on_media])[:4]}, not the clamped {spec.vocab_size - 1}")
    if "mrope_positions" not in cap:
        if items:
            raise RuntimeError("no M-RoPE positions captured for a request with media")
    elif not np.array_equal(cap["mrope_positions"].numpy(), ct.expected_mrope(n, items, spec.merge)):
        raise RuntimeError("captured M-RoPE positions differ from the ones recomputed from S and the media grids")


def assemble_reap(reap_dir, labels, tp, n_experts, n_layers):
    """Sum each rank's chunks (checked to tile the job) and place the ranks' EP slices: [L, F, C, E] float64."""
    want_tokens = np.bincount(labels, minlength=len(CATEGORIES))
    ranks = sorted(p.name for p in reap_dir.iterdir())
    if ranks != [f"rank{r}" for r in range(tp)]:
        raise RuntimeError(f"{reap_dir}: rank directories {ranks}, want rank0..rank{tp - 1}")
    out = torch.zeros((n_layers, len(FIELDS), len(CATEGORIES), n_experts), dtype=torch.float64)
    owner = np.full(n_experts, -1)
    for r in range(tp):
        files = sorted((reap_dir / f"rank{r}").glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"{reap_dir}/rank{r}: no chunks")
        pos, tokens, acc, ep = 0, np.zeros(len(CATEGORIES), np.int64), None, None
        for f in files:
            c = load_file(str(f))
            ep_rank, n_local, prefix, n = (int(x) for x in c["ep"])
            if prefix != pos:
                raise RuntimeError(f"{f}: chunk at {prefix}, expected {pos} (missing, duplicated or reordered)")
            if ep is None:
                ep = (ep_rank, n_local)
            elif ep != (ep_rank, n_local):
                raise RuntimeError(f"{f}: EP {ep_rank}/{n_local} != {ep} of the rank's other chunks")
            if not (c["calls"] == 1).all() or len(c["calls"]) != n_layers:
                raise RuntimeError(f"{f}: hook calls {c['calls'].tolist()}")
            pos += n
            tokens += c["tokens"].numpy()
            acc = c["acc"] if acc is None else acc + c["acc"]
        if pos != len(labels) or not np.array_equal(tokens, want_tokens):
            raise RuntimeError(f"rank{r}: chunks cover {pos} tokens {tokens.tolist()}, job has {len(labels)} "
                               f"{want_tokens.tolist()}")
        ep_rank, n_local = ep
        sl = slice(ep_rank * n_local, (ep_rank + 1) * n_local)
        if (owner[sl] != -1).any():
            raise RuntimeError(f"rank{r}: experts {sl} already owned by another rank")
        owner[sl] = r
        out[:, :, :, sl] = acc
    if (owner == -1).any():
        raise RuntimeError(f"experts {np.flatnonzero(owner == -1)[:5]} on no rank")
    return out, want_tokens


def routing_check(inst, seg, labels, counts, n_experts):
    """Per category and layer: the share of routed slots that differ between the ledger's production routing and the
    capture's counts, on the positions the ledger has (status R)."""
    st = segment_routing_status(inst.path, seg, layers=counts.shape[0], top_k=10, num_experts=n_experts)
    ok = st.status == ROUTING_R
    out = {"positions_R": int(ok.sum()), "positions": len(seg), "reasons": st.reasons}
    for c, name in enumerate(CATEGORIES):
        sel = ok & (labels == c)
        if not sel.any():
            continue
        prod = np.zeros((counts.shape[0], n_experts))
        rows = st.routing[sel].astype(np.int64)  # (m, L, k)
        for layer in range(counts.shape[0]):
            prod[layer] = np.bincount(rows[:, layer].reshape(-1), minlength=n_experts)
        cap = counts[:, c].numpy()
        # the capture counts every position; the ledger lacks the last token and any M rows of this category
        extra = int((~ok & (labels == c)).sum())
        diff = np.abs(prod - cap).sum(axis=1) / np.maximum(prod.sum(axis=1) + cap.sum(axis=1), 1)
        out[name] = {"tokens_R": int(sel.sum()), "tokens_not_R": extra, "mean": float(diff.mean()),
                     "max": float(diff.max())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--split", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--root", required=True, type=Path, help="the server's REAP_CAPTURE_ROOT")
    p.add_argument("--out", required=True, type=Path, help="per-job results (persistent)")
    p.add_argument("--ledger-root", type=Path, help="directory holding the plan's instance dirs (default: their "
                                                     "plan paths)")
    p.add_argument("--image-detail", required=True, choices=("auto", "low", "high"))
    p.add_argument("--limit", type=int)
    p.add_argument("--check-routing", type=int, default=0)
    p.add_argument("--router-mask", type=Path, help="route as the model pruned by this mask (drop [layers, experts])")
    p.add_argument("--logprobs", action="store_true", help="store every input token's raw logprob")
    p.add_argument("--router-renorm", choices=("renorm", "decouple"), default="renorm",
                   help="with --router-mask: renorm = the kept top-k weights sum to 1 (physical pruning); decouple = "
                        "each kept expert keeps its 512E weight p / (512E top-k mass), the dropped mass is lost")
    p.add_argument("--router-gain", type=Path, help="with --router-mask: a file with alpha [layers]; layer l's routed "
                                                    "weights times alpha[l] (reap_runtime router_gain)")
    p.add_argument("--expert-patch", type=Path, help="serve with these routed experts swapped in (reap_runtime "
                                                      "set_patch: checkpoint-named FP8 tensors), restored after")
    p.add_argument("--expert-patch-identity", action="store_true",
                   help="the patch is the checkpoint's own bytes: the server checks the loaded slices are unchanged")
    p.add_argument("--pairs-hidden-stride", type=int, help="pairs mode (see above): sample the MoE inputs of "
                                                            "positions that are multiples of this")
    p.add_argument("--only-groups", type=Path, help="a file of task groups (one a line): replay only their jobs "
                                                    "(job ids stay the plan's)")
    args = p.parse_args()
    mask_sha = None
    if args.pairs_hidden_stride is not None and (args.router_mask is not None or args.pairs_hidden_stride < 1):
        raise ValueError("--pairs-hidden-stride: a positive stride, and the 512E model (no --router-mask)")
    if args.router_renorm != "renorm" and args.router_mask is None:
        raise ValueError(f"--router-renorm {args.router_renorm} needs --router-mask")
    gain = None
    if args.router_gain is not None:
        if args.router_mask is None:
            raise ValueError("--router-gain needs --router-mask")
        path = args.router_gain.resolve()
        gain = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    patch = None
    if args.expert_patch is not None:
        path = args.expert_patch.resolve()
        with open(path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        patch = {"path": str(path), "sha256": digest,
                 "identity": args.expert_patch_identity}
    elif args.expert_patch_identity:
        raise ValueError("--expert-patch-identity needs --expert-patch")
    if args.router_mask is not None:
        args.router_mask = args.router_mask.resolve()
        mask_sha = hashlib.sha256(args.router_mask.read_bytes()).hexdigest()
    torch.set_num_threads(8)
    url = args.url.rstrip("/")
    meta, jobs = load_plan(args.plan)
    jobs = [j for j in jobs if j["split"] == args.split]
    if not jobs:
        raise ValueError(f"the plan has no {args.split!r} jobs")
    if args.only_groups is not None:
        groups = set(args.only_groups.read_text().split())
        jobs = [j for j in jobs if j["group"] in groups]
        if not jobs:
            raise ValueError(f"no {args.split!r} jobs in the groups of {args.only_groups}")

    server = json.loads((args.root / "capture_server.json").read_text())
    profile_path = Path(server["target_profile"])
    if reap_served_name(profile_path) != server["served_model_name"]:
        raise RuntimeError(f"{args.root}: served name {server['served_model_name']} is not the profile's")
    max_len, _ = ct.check_server(url, server["served_model_name"])
    if max_len < meta["context_length"]:
        raise RuntimeError(f"server context {max_len} < the plan's {meta['context_length']}")
    flags = dict(rt.parse_flags(server["argv"]))
    model_dir = Path(flags["--model-path"][0])
    tp = int(flags["--tp"][0]) if "--tp" in flags else int(flags["--tp-size"][0])
    profile = json.loads(profile_path.read_text())
    vocab = Vocab(model_dir)
    spec = ct.ModelSpec(model_dir, vocab)
    config = json.loads((model_dir / "config.json").read_text())["text_config"]
    n_experts, n_layers = int(config["num_experts"]), int(config["num_hidden_layers"])
    instances = {}
    for rec in meta["instances"]:
        path = (args.ledger_root / rec["name"]) if args.ledger_root else Path(rec["path"])
        recorded = json.loads((path / "server.json").read_text())
        rt.check_target(recorded, profile, model_dir)
        if rec["name"] in {j["instance"] for j in jobs}:
            inst = load_instance(path, vocab)
            bad = verify_media_blobs(inst)
            if bad:
                raise RuntimeError(f"{path}: {len(bad)} media blobs do not match their sha256, e.g. {bad[0]}")
            instances[rec["name"]] = inst
    if (args.root / "active.json").exists():
        raise RuntimeError(f"{args.root / 'active.json'} exists: another capture is active, or one died; inspect it")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "capture_server.json").write_text(json.dumps(server, indent=1) + "\n")

    run_jobs = jobs if args.limit is None else jobs[:args.limit]
    run_ids = [f"{args.split}-{j['index']:09d}" for j in run_jobs]
    pairs_tag = None
    if args.pairs_hidden_stride is not None:
        finished = [i for i in run_ids if (args.out / i / "DONE").exists()]
        if finished or (args.out / "pairs.safetensors").exists():
            raise RuntimeError(f"pairs mode does not resume: {args.out} has {len(finished)} finished jobs / a pairs file")
        pairs_tag = f"{args.split}-{uuid.uuid4().hex[:12]}"
    done = tokens_total = 0
    began = time.monotonic()
    for job in jobs:
        if args.limit is not None and done >= args.limit:
            break
        job_id = f"{args.split}-{job['index']:09d}"
        out = args.out / job_id
        if (out / "DONE").exists():
            continue
        if out.exists():
            shutil.rmtree(out)  # a job this driver died in
        work = args.root / job_id
        if work.exists():
            shutil.rmtree(work)
        inst = instances[job["instance"]]
        seg = inst.segments[job["seg"]]
        if ids_sha256(seg.ids) != job["ids_sha256"] or len(seg) != job["num_tokens"]:
            raise RuntimeError(f"{seg.key} is not the segment the plan saw")
        reason = ct.skip_reason(seg, max_len)
        if reason is not None:
            raise RuntimeError(f"{seg.key}: planned but now skipped ({reason})")
        ids, items, body = ct.segment_request(inst, seg, vocab, spec, args.image_detail)
        labels = job_labels(seg, ids, items)
        (work / "parts").mkdir(parents=True)
        (work / "reap").mkdir()
        np.save(work / "labels.npy", labels)
        if args.logprobs:
            raw = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
            body.update(rid=f"reap-{job_id}", sampling_params={"max_new_tokens": 1, **raw}, return_logprob=True,
                        logprob_start_len=0)
        else:
            body.update(rid=f"reap-{job_id}", sampling_params={"max_new_tokens": 1, "temperature": 0})
        record = {"job_id": job_id, "num_tokens": len(ids), **{k: job[k] for k in (
            "index", "split", "key", "instance", "seg", "pool", "task", "group", "n_gen", "n_images", "n_videos",
            "image_tokens", "video_tokens")}, "sglang_commit": server["sglang_commit"],
            "router_mask": None if args.router_mask is None else str(args.router_mask), "router_mask_sha256": mask_sha,
            "router_renorm": args.router_renorm, "router_gain": gain, "expert_patch": patch}
        if pairs_tag is not None:
            record["pairs"] = {"tag": pairs_tag, "hidden_stride": args.pairs_hidden_stride,
                               "flush": job_id == run_ids[-1]}
        start = time.monotonic()
        meta_info = ct.run_request(url, args.root, record, body)
        request_seconds = time.monotonic() - start
        cap = ct.assemble(work / "parts", len(ids))
        check_replay(ids, items, cap, spec)
        acc, tokens = assemble_reap(work / "reap", labels, tp, n_experts, n_layers)
        record.update(request_seconds=request_seconds, cached_tokens=meta_info.get("cached_tokens"),
                      tokens_by_category=dict(zip(CATEGORIES, tokens.tolist())))
        if done < args.check_routing:
            record["routing_check"] = routing_check(inst, seg, labels, acc[:, FIELDS.index("counts")], n_experts)
        out.mkdir(parents=True)
        if args.logprobs:
            served = meta_info["input_token_logprobs"]
            if len(served) != len(ids) or served[0][0] is not None:
                raise RuntimeError(f"{seg.key}: {len(served)} input logprobs for {len(ids)} tokens")
            returned = np.array([int(t) for _, t, *_ in served], np.int64)
            off_media = ~ct.media_mask(len(ids), items)
            if not np.array_equal(returned[off_media], ids[off_media]):
                raise RuntimeError(f"{seg.key}: input logprob token ids differ from S off the media")
            lp = np.array([x for x, *_ in served[1:]], np.float32)
            if not np.isfinite(lp[off_media[1:]]).all():
                raise RuntimeError(f"{seg.key}: nonfinite logprob on a text token")
            np.save(out / "logprobs.npy", lp)
            np.save(out / "labels.npy", labels)
        save_file({"acc": acc.float().contiguous(), "tokens": torch.from_numpy(tokens)}, str(out / "reap.partial"))
        (out / "reap.partial").rename(out / "reap.safetensors")
        (out / "record.json").write_text(json.dumps(record, indent=1) + "\n")
        (out / "DONE").touch()
        shutil.rmtree(work)
        done += 1
        tokens_total += len(ids)
        brief = {"job": job_id, "tokens": len(ids), "img": int(tokens[2]), "vid": int(tokens[3]),
                 "s": round(request_seconds, 1), "tok/s": round(len(ids) / request_seconds)}
        if "routing_check" in record:
            brief["route_diff"] = {k: round(v["mean"], 4) for k, v in record["routing_check"].items()
                                   if isinstance(v, dict) and "mean" in v}
        print(json.dumps(brief), flush=True)
    if pairs_tag is not None:
        flushed = args.root / "pairs" / f"{pairs_tag}.safetensors"
        meta_path = flushed.with_suffix(".json")
        if not flushed.is_file() or not meta_path.is_file():
            raise RuntimeError(f"the server did not flush the pairs of {pairs_tag} ({flushed})")
        pairs_meta = json.loads(meta_path.read_text())
        if pairs_meta["jobs"] != run_ids:
            raise RuntimeError(f"pairs {pairs_tag}: the server summed {len(pairs_meta['jobs'])} jobs, this run "
                               f"{len(run_ids)} (or another order)")
        shutil.move(flushed, args.out / "pairs.safetensors")
        shutil.move(meta_path, args.out / "pairs.json")
    seconds = time.monotonic() - began
    print(json.dumps({"split": args.split, "jobs": done, "tokens": tokens_total, "seconds": round(seconds, 1),
                      "tokens_per_second": round(tokens_total / max(seconds, 1e-9))}), flush=True)


if __name__ == "__main__":
    main()
