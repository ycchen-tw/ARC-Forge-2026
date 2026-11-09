"""REAP saliency server for TITO replay: capture_tito_runtime's isolated capture server, with the MoE on the Triton
runner and a hook that accumulates per-expert REAP statistics, split by token category.

    REAP_CAPTURE_ROOT=<root> python reap_runtime.py --target-profile <server.json> --port <p> --tp 2 \
        --model-path <copy of the recorded model> --vlm-cache-mb 16384 [--context-length <n>]

One server for every ledger instance of a plan (profile mode: its served name is reap_served_name(profile)); the
driver, reap_capture.py, runs check_target for each instance against the profile. Isolation is capture_tito's: one
running request, no radix, no CUDA graph, no overlap, no speculative decoding, no warmup -- so every token of a
segment is forwarded exactly once and nothing else is.

Why Triton: REAP needs each routed expert's output norm before the router weight, ||f_j(x)||. The production MoE
(flashinfer_trtllm) does routing and combine inside one kernel and never shows it. On the Triton runner the down GEMM
writes intermediate_cache3[t, j] = g_j * f_j(x_t) (router weight applied, before the top-k sum), and the runner's
LoRA hook point ``after_down`` receives it -- provided the fused-func shortcut is off (SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1,
otherwise MoeRunner.run never reaches the runner core). Routing itself is the same softmax top-10 renormalized; the
expert GEMMs differ in reduction order only.

Per forward (one prefill chunk of the active job), on every rank:

- before: the job's token categories (``<root>/<job>/labels.npy``, int8, written by the driver: CATEGORIES) for the
  chunk's positions go to the GPU;
- in each MoE layer (TritonRunnerCore.run, hooks=after_down): per (category, local expert) the five REAP cumulants
  FIELDS -- counts, sum g, sum ||f||, sum ||f||^2, sum g*||f|| -- over the chunk's routed (token, expert) pairs whose
  expert lives on this rank (EP: topk_ids are local, -1 elsewhere). ||f|| = ||g*f|| / g;
- after: ``<root>/<job>/reap/rank<r>/<prefix:012d>.safetensors`` with ``acc`` [layers, fields, categories, n_local]
  float64, the hook call count per layer, tokens per category, and this rank's EP rank / n_local; TP rank 0 also
  writes ``<root>/<job>/parts/<prefix:012d>.safetensors`` (ids as the model read them, positions, M-RoPE) for the
  driver's replay checks. No GR: REAP needs none, and it would be 20 KB a token.

Every inconsistency raises: a chunk whose MoE token count is not the chunk's, a layer hooked twice or never, a routed
weight <= 0, LoRA hooks already present, an expert layer split by TP, a forward that is not one EXTEND request.

Router mask (a job's ``router_mask``: a mask file, ``drop`` bool [layers, experts]): every MoE layer's router logits
get -inf on the dropped experts before top-k. With renormalized top-k (Qwen's norm_topk_prob) the selected experts
and their weights are then exactly those of the physically pruned model -- exp(l_j) / sum over the top-k of exp(l)
does not depend on the other logits -- so one 512E server scores every candidate keep-set without writing a
checkpoint. Each layer must be masked exactly once a chunk. Every layer keeps the same count, except in a mask whose
meta says ``partial_probe``: there unpruned layers (all experts kept) are allowed -- sequential re-estimation. ``router_renorm: decouple`` instead keeps each surviving
expert's 512E weight, p_j / (sum of the 512E top-k p): the masked top-k weights times (kept top-k mass) / (512E top-k
mass), so the dropped experts' share is lost rather than spread over the survivors. ``router_gain`` (a file with
``alpha`` [layers]) then multiplies layer l's routed weights by alpha[l] -- a per-layer scale a pruned checkpoint could
fold into its down projections.

Pairs (a job's ``pairs``: tag, hidden_stride, flush; 512E only): Pairs sums second-order statistics -- expert-pair
counts, norm products and output Gram, mean outputs, router logit Gram, sampled MoE inputs -- across a driver run's
jobs, all-reducing the routed outputs over the EP group so each rank sees all top-k slots; the job marked flush
writes ``<root>/pairs/<tag>.safetensors`` (TP rank 0).
"""

import argparse
import functools
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import capture_tito_runtime as rt

ROOT_ENV = "REAP_CAPTURE_ROOT"
CATEGORIES = ("prompt", "gen", "image", "video")  # labels.npy values 0..3: text the model read, text it sampled, media
FIELDS = ("counts", "sum_g", "sum_f", "sum_f2", "sum_gf")


def reap_served_name(profile_path):
    raw = Path(profile_path).read_bytes()
    return f"reap-capture.{json.loads(raw)['instance']}.{hashlib.sha256(raw).hexdigest()[:16]}"


def reap_argv(target, port, *, served_name, model_path, tp, context_length):
    """capture_tito's isolated argv for the target, with the MoE on the Triton runner."""
    argv = rt.capture_argv(target, port, served_name=served_name, model_path=model_path, tp=tp,
                           context_length=context_length)
    if "--moe-runner-backend" in argv:
        argv[argv.index("--moe-runner-backend") + 1] = "triton"
    else:
        argv += ["--moe-runner-backend", "triton"]
    return argv


class Chunk:
    """The active job's chunk while its forward runs: categories on the GPU and the per-layer accumulators."""

    def __init__(self, job_id, prefix, n, labels, n_layers, ep_rank, bias, renorm="renorm", pairs=None, gain=None):
        self.job_id, self.prefix, self.n, self.labels, self.n_layers = job_id, prefix, n, labels, n_layers
        self.ep_rank = ep_rank
        self.pairs = pairs  # None, or the open Pairs accumulation this chunk adds to
        self.route = {}     # pairs: layer_id -> (global topk_ids, topk_weights) from the router
        self.hidden = {}    # pairs: layer_id -> the sampled tokens' MoE inputs
        self.bias = bias    # None, or [layers, experts] with -inf on dropped experts (router mask)
        self.renorm = renorm  # "renorm" (physical pruning) or "decouple" (kept weights over the 512E top-k mass)
        self.gain = gain      # None, or [layers]: every routed weight of layer l times gain[l] (after renorm)
        self.masked = {}    # layer_id -> router mask applications
        self.acc = {}       # layer_id -> [fields, categories * n_local] float64
        self.calls = {}     # layer_id -> hook calls
        self.n_local = None  # the same for every layer

    def accumulator(self, layer_id, n_local, device):
        import torch

        if self.n_local is None:
            self.n_local = n_local
        elif self.n_local != n_local:
            raise RuntimeError(f"layer {layer_id}: {n_local} local experts, the other layers {self.n_local}")
        if layer_id not in self.acc:
            self.acc[layer_id] = torch.zeros((len(FIELDS), len(CATEGORIES) * n_local), dtype=torch.float64,
                                             device=device)
        self.calls[layer_id] = self.calls.get(layer_id, 0) + 1
        return self.acc[layer_id]


def after_down(chunk, layer_id, n_local, intermediate_cache2, intermediate_cache3, topk_weights, topk_ids):
    import torch

    t, k = topk_ids.shape
    if t != chunk.n:
        raise RuntimeError(f"layer {layer_id}: the MoE saw {t} tokens, the chunk has {chunk.n} "
                           "(tokens scattered across ranks? padding?)")
    if intermediate_cache3 is None or tuple(intermediate_cache3.shape[:2]) != (t, k):
        got = None if intermediate_cache3 is None else tuple(intermediate_cache3.shape)
        raise RuntimeError(f"layer {layer_id}: intermediate_cache3 {got}, want ({t}, {k}, hidden)")
    acc = chunk.accumulator(layer_id, n_local, intermediate_cache3.device)
    ids = topk_ids.long()
    g = topk_weights.double()
    gf = intermediate_cache3.float().norm(dim=-1).double()  # (t, k) = g * ||f||; garbage where ids == -1
    valid = (ids >= 0) & (ids < n_local)
    if bool((g[valid] <= 0).any()):
        raise RuntimeError(f"layer {layer_id}: a routed (token, expert) with router weight <= 0")
    w = valid.double()
    f = torch.where(valid, gf / torch.where(valid, g, torch.ones_like(g)), torch.zeros_like(g))
    gf = torch.where(valid, gf, torch.zeros_like(gf))
    idx = torch.where(valid, chunk.labels[:, None] * n_local + ids, torch.zeros_like(ids)).reshape(-1)
    for i, v in enumerate((w, g * w, f, f * f, gf)):
        acc[i].index_add_(0, idx, v.reshape(-1))
    if chunk.pairs is not None:
        chunk.pairs.add_moe(chunk, layer_id, n_local, intermediate_cache3, topk_weights, ids, valid)


class Pairs:
    """Per-(layer, expert pair) statistics over the routed slots of a run of jobs (a driver's ``--pairs`` tag), summed
    on the GPU across the jobs' chunks and written by TP rank 0 when the job marked ``flush`` ends.

    For every token x and every two of its top-k slots (a, b) -- a == b included -- with global experts i, j and
    routed outputs y = g * f (intermediate_cache3, all-reduced over the EP group so every rank sees all k slots):
    counts[i, j] += 1, norms[i, j] += ||y_a|| ||y_b|| (HOPE's F), gram[i, j] += <y_a, y_b> (the dropped output's
    exact energy: ||sum_{dropped} y||^2 = p^T gram p per token). The diagonals are counts, sum g^2 ||f||^2.
    Per expert: sum_y [H] = sum g f, sum_f [H] = sum f (mean outputs, for merge similarity). Per layer: the router
    logits' Gram l^T l and sum l (512E, before any mask). And the MoE inputs of every token whose position is a
    multiple of hidden_stride, with the job and position."""

    BLOCK = 2048  # tokens a block, bounds the [tokens, k, hidden] float32 temporaries

    def __init__(self, tag, hidden_stride, n_layers):
        self.tag, self.hidden_stride, self.n_layers = tag, hidden_stride, n_layers
        self.jobs = []      # completed jobs, in order
        self.open = None    # (job_id, next prefix) of the job being accumulated
        self.acc = {}       # layer_id -> dict of float64 accumulators
        self.samples = []   # (job_id, positions [s], labels [s], hidden [layers, s, H] cpu bf16) per chunk

    def layer(self, layer_id, n_experts, hidden, device):
        import torch

        if layer_id not in self.acc:
            z = functools.partial(torch.zeros, dtype=torch.float64, device=device)
            self.acc[layer_id] = {"counts": z(n_experts * n_experts), "norms": z(n_experts * n_experts),
                                  "gram": z(n_experts * n_experts), "sum_y": z((n_experts, hidden)),
                                  "sum_f": z((n_experts, hidden)), "logit_gram": z((n_experts, n_experts)),
                                  "logit_sum": z(n_experts), "tokens": z(1)}
        a = self.acc[layer_id]
        if a["sum_y"].shape != (n_experts, hidden):
            raise RuntimeError(f"pairs layer {layer_id}: {n_experts} experts x {hidden}, earlier {a['sum_y'].shape}")
        return a

    def add_router(self, chunk, layer_id, hidden_states, router_logits, out, sample_rows):
        if layer_id in chunk.route:
            raise RuntimeError(f"pairs: layer {layer_id} routed twice in a chunk")
        if hidden_states.shape[0] != chunk.n or router_logits.shape[0] != chunk.n:
            raise RuntimeError(f"pairs layer {layer_id}: hidden {tuple(hidden_states.shape)}, logits "
                               f"{tuple(router_logits.shape)} for a chunk of {chunk.n}")
        chunk.route[layer_id] = (out.topk_ids, out.topk_weights)
        chunk.hidden[layer_id] = hidden_states[sample_rows].clone()
        a = self.layer(layer_id, router_logits.shape[1], hidden_states.shape[1], router_logits.device)
        lg = router_logits.double()
        a["logit_gram"] += lg.T @ lg
        a["logit_sum"] += lg.sum(0)
        a["tokens"] += chunk.n

    def add_moe(self, chunk, layer_id, n_local, ic3, topk_weights, ids, valid):
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed import get_moe_ep_group

        if layer_id not in chunk.route:
            raise RuntimeError(f"pairs: layer {layer_id} reached the experts without passing the router")
        gid, gw = chunk.route.pop(layer_id)
        gid = gid.long()
        if gid.shape != ids.shape:
            raise RuntimeError(f"pairs layer {layer_id}: router ids {tuple(gid.shape)}, MoE ids {tuple(ids.shape)}")
        if not torch.equal(gid[valid], ids[valid] + chunk.ep_rank * n_local):
            raise RuntimeError(f"pairs layer {layer_id}: this rank's slots are not router id - ep_rank * n_local "
                               "(EP slices not contiguous?)")
        if not torch.allclose(topk_weights.float()[valid], gw.float()[valid]):
            raise RuntimeError(f"pairs layer {layer_id}: the MoE's router weights are not the router's")
        group = get_moe_ep_group()
        x = torch.where(valid[..., None], ic3, torch.zeros_like(ic3))
        cover = valid.to(torch.int32)
        dist.all_reduce(x, group=group.device_group)
        dist.all_reduce(cover, group=group.device_group)
        if not bool((cover == 1).all()):
            raise RuntimeError(f"pairs layer {layer_id}: slots covered {cover.unique().tolist()} times over the EP "
                               "ranks, want each once")
        t, k, hidden = x.shape
        n_experts = n_local * group.world_size
        a = self.layer(layer_id, n_experts, hidden, x.device)
        for s in range(0, t, self.BLOCK):
            xb = x[s:s + self.BLOCK].float()
            ib = gid[s:s + self.BLOCK]
            pair = (ib[:, :, None] * n_experts + ib[:, None, :]).reshape(-1)
            norm = xb.norm(dim=-1)
            a["counts"].index_add_(0, pair, torch.ones_like(pair, dtype=torch.float64))
            a["norms"].index_add_(0, pair, (norm[:, :, None] * norm[:, None, :]).reshape(-1).double())
            a["gram"].index_add_(0, pair, torch.bmm(xb, xb.transpose(1, 2)).reshape(-1).double())
            sy = torch.zeros((n_experts, hidden), dtype=torch.float32, device=x.device)
            sy.index_add_(0, ib.reshape(-1), xb.reshape(-1, hidden))
            a["sum_y"] += sy
            sf = torch.zeros_like(sy)
            sf.index_add_(0, ib.reshape(-1), (xb / gw[s:s + self.BLOCK].float()[..., None]).reshape(-1, hidden))
            a["sum_f"] += sf

    def end_chunk(self, chunk, positions, labels):
        import torch

        if chunk.route:
            raise RuntimeError(f"pairs: layers {sorted(chunk.route)} routed but never reached the experts")
        if sorted(chunk.hidden) != list(range(self.n_layers)):
            raise RuntimeError(f"pairs: sampled MoE inputs of layers {sorted(chunk.hidden)}")
        if len(positions):
            hidden = torch.stack([chunk.hidden[i] for i in range(self.n_layers)]).cpu()
            self.samples.append((chunk.job_id, positions.cpu(), labels.cpu(), hidden))

    def save(self, path):
        import torch
        from safetensors.torch import save_file

        if sorted(self.acc) != list(range(self.n_layers)):
            raise RuntimeError(f"pairs {self.tag}: accumulated layers {sorted(self.acc)}")
        tensors = {}
        for key in self.acc[0]:
            stacked = torch.stack([self.acc[i][key] for i in range(self.n_layers)])
            if key in ("counts", "norms", "gram"):
                e = int(round(stacked.shape[1] ** 0.5))
                stacked = stacked.view(self.n_layers, e, e)
            tensors[key] = stacked.cpu().contiguous()
        job_index = {j: i for i, j in enumerate(self.jobs)}
        if self.samples:
            tensors["hidden"] = torch.cat([h for *_, h in self.samples], dim=1).contiguous()
            tensors["hidden_job"] = torch.cat([torch.full((len(p),), job_index[j]) for j, p, *_ in self.samples])
            tensors["hidden_pos"] = torch.cat([p for _, p, *_ in self.samples])
            tensors["hidden_label"] = torch.cat([lab for _, _, lab, _ in self.samples])
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path.with_suffix(".partial")))
        path.with_suffix(".partial").rename(path)
        path.with_suffix(".json").write_text(json.dumps({"tag": self.tag, "hidden_stride": self.hidden_stride,
                                                          "jobs": self.jobs}) + "\n")


def install():
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
    from sglang.srt.lora.lora_moe_runners import LoRAHooks
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from sglang.srt.multimodal.processors.base_processor import splice_timestamped_media_blocks  # noqa: F401
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import ImageData

    if os.environ.get("SGLANG_CI_DISABLE_MOE_FUSED_FUNC") != "1":
        raise RuntimeError("SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1 is required: the fused func bypasses the runner hooks")
    root = Path(os.environ[ROOT_ENV])
    state = {"chunk": None, "labels": (None, None), "masks": {}, "gains": {}, "pairs": None, "patch": None,
             "moe": None}

    def router_bias(path, device):
        key = (path, str(device))
        if key not in state["masks"]:
            blob = torch.load(path, map_location="cpu")
            drop = blob["drop"]
            if drop.dtype != torch.bool or drop.dim() != 2:
                raise RuntimeError(f"{path}: drop {drop.dtype}{tuple(drop.shape)}, want bool [layers, experts]")
            kept = (~drop).sum(1)
            # a probe mask may leave whole layers unpruned (sequential re-estimation); a deployable one prunes all
            partial = (blob.get("meta") or {}).get("partial_probe") is True
            pruned = kept[kept < drop.shape[1]] if partial else kept
            if len(pruned) == 0 or not (pruned == pruned[0]).all() or int(pruned[0]) < 10:
                raise RuntimeError(f"{path}: keeps {kept.unique().tolist()} experts per layer (partial_probe {partial})")
            state["masks"][key] = torch.where(drop, float("-inf"), 0.0).to(device=device, dtype=torch.float32)
        return state["masks"][key]

    def router_gain(spec, device, n_layers):
        key = (spec["path"], spec["sha256"], str(device))
        if key not in state["gains"]:
            alpha = torch.load(spec["path"], map_location="cpu")["alpha"]
            if alpha.shape != (n_layers,) or not bool(torch.isfinite(alpha).all()) or not bool((alpha > 0).all()):
                raise RuntimeError(f"{spec['path']}: alpha {alpha.dtype}{tuple(alpha.shape)}, want positive [{n_layers}]")
            state["gains"][key] = alpha.to(device=device, dtype=torch.float32)
        return state["gains"][key]

    def job_labels(job):
        cached_id, labels = state["labels"]
        if cached_id != job["job_id"]:
            labels = np.load(root / job["job_id"] / "labels.npy")
            if labels.dtype != np.int8 or labels.shape != (job["num_tokens"],):
                raise RuntimeError(f"{job['job_id']}: labels {labels.dtype}{labels.shape}, want int8 "
                                   f"({job['num_tokens']},)")
            if labels.min() < 0 or labels.max() >= len(CATEGORIES):
                raise RuntimeError(f"{job['job_id']}: label values outside 0..{len(CATEGORIES) - 1}")
            state["labels"] = (job["job_id"], labels)
        return labels

    def open_pairs(job, prefix, n_layers):
        """The Pairs accumulation of the job's tag, checked to continue the job (or start it at 0) in order. A new
        tag replaces an unflushed one: that driver died (each driver run has its own tag)."""
        spec = job["pairs"]
        acc = state["pairs"]
        if acc is None or acc.tag != spec["tag"]:
            if acc is not None:
                print(f"[reap] pairs: dropping the unflushed accumulation {acc.tag} ({len(acc.jobs)} jobs, open "
                      f"{acc.open}): its driver died", flush=True)
            if int(spec["hidden_stride"]) < 1:
                raise RuntimeError(f"pairs hidden_stride {spec['hidden_stride']}")
            acc = state["pairs"] = Pairs(spec["tag"], int(spec["hidden_stride"]), n_layers)
        if acc.open is None:
            if prefix != 0 or job["job_id"] in acc.jobs:
                raise RuntimeError(f"pairs {acc.tag}: job {job['job_id']} starts at {prefix} (done before: "
                                   f"{job['job_id'] in acc.jobs})")
            acc.open = (job["job_id"], 0)
        if acc.open != (job["job_id"], prefix):
            raise RuntimeError(f"pairs {acc.tag}: chunk ({job['job_id']}, {prefix}), expected {acc.open}")
        return acc

    def close_pairs(job, end):
        acc = state["pairs"]
        if end < job["num_tokens"]:
            acc.open = (job["job_id"], end)
            return
        if end != job["num_tokens"]:
            raise RuntimeError(f"pairs: job {job['job_id']} ran to {end} of {job['num_tokens']} tokens")
        acc.jobs.append(job["job_id"])
        acc.open = None
        if job["pairs"]["flush"]:
            if get_tensor_model_parallel_rank() == 0:
                acc.save(root / "pairs" / f"{acc.tag}.safetensors")
            state["pairs"] = None

    def moe_layers(model, n_layers):
        if state["moe"] is None:
            found = {}
            for name, m in model.named_modules():
                if isinstance(m, FusedMoE):
                    if m.layer_id in found:
                        raise RuntimeError(f"expert patch: two FusedMoE with layer id {m.layer_id} ({name})")
                    found[m.layer_id] = m
            if sorted(found) != list(range(n_layers)):
                raise RuntimeError(f"expert patch: FusedMoE layer ids {sorted(found)}, want 0..{n_layers - 1}")
            state["moe"] = found
        return state["moe"]

    def set_patch(model, spec, n_layers):
        """Make the routed experts those of the job's expert_patch (None: the checkpoint's). A patch file holds
        checkpoint-named tensors ``layers.<l>.experts.<e>.<gate|up|down>_proj.<weight|weight_scale_inv>`` (FP8 E4M3
        + BF16 128x128 block scales, all six per expert); each rank loads its experts through FusedMoE's own weight
        loader after saving the slices it overwrites, and restores them before the next patch. ``identity``: the
        patch is the checkpoint's own bytes; the loaded slices must equal the saved ones (checks the path)."""
        key = None if spec is None else (spec["path"], spec["sha256"])
        cur = state["patch"]
        if (cur["key"] if cur is not None else None) == key:
            return
        if cur is not None:
            for m, local, saved in cur["saved"]:
                for pname, tensor in saved.items():
                    getattr(m, pname).data[local].copy_(tensor)
            state["patch"] = None
            print(f"[reap] expert patch restored: {cur['key'][0]} ({len(cur['saved'])} local experts)", flush=True)
        if spec is None:
            return
        tensors = load_file(spec["path"])
        experts = {}
        for name, tensor in tensors.items():
            parts = name.split(".")
            if len(parts) != 6 or parts[0] != "layers" or parts[2] != "experts":
                raise RuntimeError(f"expert patch {spec['path']}: tensor name {name}")
            experts.setdefault((int(parts[1]), int(parts[3])), {})[f"{parts[4]}.{parts[5]}"] = tensor
        mods = moe_layers(model, n_layers)
        loads = (("gate_proj", "w1", "w13_"), ("up_proj", "w3", "w13_"), ("down_proj", "w2", "w2_"))
        saved_all = []
        for (layer, expert), ts in sorted(experts.items()):
            if sorted(ts) != sorted(f"{proj}.{kind}" for proj, *_ in loads for kind in ("weight", "weight_scale_inv")):
                raise RuntimeError(f"expert patch: layer {layer} expert {expert} has {sorted(ts)}")
            m = mods[layer]
            local = expert - m.moe_ep_rank * m.num_local_experts
            if not 0 <= local < m.num_local_experts:
                continue  # another rank's expert
            pnames = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
            saved = {pname: getattr(m, pname).data[local].clone() for pname in pnames}
            saved_all.append((m, local, saved))
            for proj, shard, base in loads:
                for kind in ("weight", "weight_scale_inv"):
                    param = getattr(m, base + kind)
                    param.weight_loader(param, ts[f"{proj}.{kind}"].to(param.device), f"experts.{base}{kind}",
                                        shard_id=shard, expert_id=expert)
            if spec.get("identity"):
                for pname in pnames:
                    if not torch.equal(getattr(m, pname).data[local], saved[pname]):
                        raise RuntimeError(f"expert patch identity: layer {layer} expert {expert} {pname} differs "
                                           "after loading the checkpoint's own tensors (post-load transform?)")
        state["patch"] = {"key": key, "saved": saved_all}
        print(f"[reap] expert patch applied: {spec['path']} ({len(experts)} experts, {len(saved_all)} local)",
              flush=True)

    cls = Qwen4ExpForConditionalGeneration
    original = cls.forward

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        active = root / "active.json"
        if not active.is_file():
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        job = json.loads(active.read_text())
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            raise RuntimeError(f"REAP capture saw a {forward_batch.forward_mode} forward; it replays prefill only")
        if forward_batch.batch_size != 1:
            raise RuntimeError(f"REAP capture needs one request per forward, got {forward_batch.batch_size}")
        n = int(forward_batch.extend_seq_lens_cpu[0])
        prefix = int(forward_batch.extend_prefix_lens_cpu[0])
        ids = forward_batch.input_ids
        if tuple(ids.shape) != (n,) or tuple(positions.shape) != (n,):
            raise RuntimeError(f"chunk of {n}: ids {tuple(ids.shape)}, positions {tuple(positions.shape)}")
        labels = job_labels(job)
        if prefix + n > len(labels):
            raise RuntimeError(f"chunk [{prefix}, {prefix + n}) beyond the job's {len(labels)} tokens")
        lab = torch.from_numpy(labels[prefix:prefix + n].astype(np.int64)).to(ids.device)
        n_layers = int(self.config.num_hidden_layers)  # the language model's config (Qwen4ExpTextConfig)
        par = get_parallel()
        if int(par.moe_tp_size) != 1:
            raise RuntimeError(f"experts split by TP (moe_tp_size {par.moe_tp_size}); ||f|| would be a partial sum. "
                               "Run with EP size == TP size")
        bias = None if job.get("router_mask") is None else router_bias(job["router_mask"], ids.device)
        if bias is not None and bias.shape[0] != n_layers:
            raise RuntimeError(f"router mask has {bias.shape[0]} layers, the model {n_layers}")
        if prefix == 0:
            set_patch(self, job.get("expert_patch"), n_layers)
        elif (state["patch"] or {}).get("key") != (None if job.get("expert_patch") is None else
                                                    (job["expert_patch"]["path"], job["expert_patch"]["sha256"])):
            raise RuntimeError(f"chunk at {prefix} of {job['job_id']}: the applied expert patch is not the job's")
        renorm = job.get("router_renorm", "renorm")
        if renorm not in ("renorm", "decouple") or (renorm != "renorm" and bias is None):
            raise RuntimeError(f"router_renorm {renorm!r} (needs a router mask unless renorm)")
        pairs = open_pairs(job, prefix, n_layers) if job.get("pairs") is not None else None
        if pairs is not None and bias is not None:
            raise RuntimeError("pairs statistics are of the 512E model: no router mask")
        gain = None
        if job.get("router_gain") is not None:
            if bias is None:
                raise RuntimeError("router_gain needs a router mask")
            gain = router_gain(job["router_gain"], ids.device, n_layers)
        chunk = Chunk(job["job_id"], prefix, n, lab, n_layers, int(par.moe_ep_rank), bias, renorm, pairs, gain)
        if pairs is not None:
            pos = prefix + torch.arange(n, device=ids.device)
            chunk.sample_rows = torch.nonzero(pos % pairs.hidden_stride == 0).flatten()
        if state["chunk"] is not None:
            raise RuntimeError("a forward started while another chunk was active")
        state["chunk"] = chunk
        try:
            output = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            state["chunk"] = None
        if sorted(chunk.calls) != list(range(n_layers)) or set(chunk.calls.values()) != {1}:
            raise RuntimeError(f"chunk at {prefix}: MoE hook calls per layer {chunk.calls}, want each of "
                               f"0..{n_layers - 1} once (is the MoE on the Triton runner, fused func off?)")
        if bias is not None and (sorted(chunk.masked) != list(range(n_layers)) or set(chunk.masked.values()) != {1}):
            raise RuntimeError(f"chunk at {prefix}: router mask applications per layer {chunk.masked}")
        if pairs is not None:
            pairs.end_chunk(chunk, prefix + chunk.sample_rows, lab[chunk.sample_rows])
            close_pairs(job, prefix + n)
        rank = get_tensor_model_parallel_rank()
        job_dir = root / job["job_id"]
        acc = torch.stack([chunk.acc[i] for i in range(n_layers)])  # [L, F, C * n_local]
        ep_rank, n_local = chunk.ep_rank, chunk.n_local
        tensors = {"acc": acc.view(n_layers, len(FIELDS), len(CATEGORIES), n_local).cpu().contiguous(),
                   "calls": torch.tensor([chunk.calls[i] for i in range(n_layers)]),
                   "tokens": torch.bincount(lab, minlength=len(CATEGORIES)).cpu(),
                   "ep": torch.tensor([ep_rank, n_local, prefix, n])}
        out = job_dir / "reap" / f"rank{rank}"
        out.mkdir(parents=True, exist_ok=True)
        destination = out / f"{prefix:012d}.safetensors"
        if destination.exists():
            raise RuntimeError(f"{destination} already written: a chunk was replayed")
        save_file(tensors, str(destination.with_suffix(".partial")))
        destination.with_suffix(".partial").rename(destination)
        if rank == 0:
            placeholder = getattr(self, "mm_media_token_ids", None) is not None
            parts = {"ple_placeholder_ids": torch.tensor([placeholder]),
                     "input_ids": ids.detach().cpu().contiguous(),
                     "positions": positions.detach().cpu().contiguous()}
            mrope = forward_batch.mrope_positions
            if mrope is not None:
                if tuple(mrope.shape) != (3, n):
                    raise RuntimeError(f"M-RoPE positions {tuple(mrope.shape)} for a chunk of {n}")
                parts["mrope_positions"] = mrope.detach().cpu().contiguous()
            destination = job_dir / "parts" / f"{prefix:012d}.safetensors"
            if destination.exists():
                raise RuntimeError(f"{destination} already written: a chunk was replayed")
            save_file(parts, str(destination.with_suffix(".partial")))
            destination.with_suffix(".partial").rename(destination)
        return output

    cls.forward = forward

    run = TritonRunnerCore.run

    @functools.wraps(run)
    def run_with_reap(self, runner_input, quant_info, running_state, hooks=None):
        chunk = state["chunk"]
        if chunk is None:
            return run(self, runner_input, quant_info, running_state, hooks=hooks)
        if hooks is not None:
            raise RuntimeError("REAP capture met MoE hooks that are already set (LoRA?)")
        cfg = self.config
        if cfg.layer_id is None or cfg.num_local_experts is None or cfg.num_experts is None:
            raise RuntimeError(f"REAP capture: MoE runner config without layer_id / expert counts: {cfg}")
        if cfg.num_fused_shared_experts:
            raise RuntimeError("REAP capture: a fused shared expert is in the routed slots")
        # A draft (MTP) MoE would share layer ids with the target; capture has no speculative decoding, and the
        # per-layer call count (each layer exactly once a chunk) catches a second MoE at the same id.
        hook = LoRAHooks(after_down=functools.partial(after_down, chunk, int(cfg.layer_id),
                                                      int(cfg.num_local_experts)))
        return run(self, runner_input, quant_info, running_state, hooks=hook)

    TritonRunnerCore.run = run_with_reap

    topk_cuda = TopK.forward_cuda

    @functools.wraps(topk_cuda)
    def topk_masked(self, hidden_states, router_logits, **kwargs):
        chunk = state["chunk"]
        if chunk is None:
            return topk_cuda(self, hidden_states, router_logits, **kwargs)
        if chunk.bias is None:
            out = topk_cuda(self, hidden_states, router_logits, **kwargs)
            if chunk.pairs is not None:
                if not isinstance(out, StandardTopKOutput) or self.layer_id is None:
                    raise RuntimeError(f"pairs need the standard top-k output and a layer id, got "
                                       f"{type(out).__name__}, layer {self.layer_id}")
                chunk.pairs.add_router(chunk, self.layer_id, hidden_states, router_logits, out, chunk.sample_rows)
            return out
        if self.layer_id is None or router_logits.shape[-1] != chunk.bias.shape[1]:
            raise RuntimeError(f"router mask: TopK layer {self.layer_id}, logits {tuple(router_logits.shape)}, "
                               f"mask {tuple(chunk.bias.shape)}")
        bias = chunk.bias[self.layer_id]
        out = topk_cuda(self, hidden_states, router_logits + bias.to(router_logits.dtype), **kwargs)
        chunk.masked[self.layer_id] = chunk.masked.get(self.layer_id, 0) + 1
        if chunk.renorm == "renorm" and chunk.gain is None:
            return out
        if not isinstance(out, StandardTopKOutput):
            raise RuntimeError(f"router_renorm decouple / router_gain need the standard top-k output, got "
                               f"{type(out).__name__}")
        w = out.topk_weights.float()
        if not torch.allclose(w.sum(-1), torch.ones_like(w[:, 0]), atol=1e-3):
            raise RuntimeError(f"layer {self.layer_id}: top-k weights do not sum to 1 (not renormalized?)")
        if chunk.renorm == "decouple":
            # weight of a kept expert = p_512E / (512E top-k mass) instead of p / (kept top-k mass)
            p = torch.softmax(router_logits.float(), dim=-1)
            k = w.shape[-1]
            full = p.topk(k, dim=-1).values.sum(-1, keepdim=True)
            kept = torch.where(torch.isinf(bias), torch.zeros_like(p), p).topk(k, dim=-1).values.sum(-1, keepdim=True)
            w = w * (kept / full)
        if chunk.gain is not None:
            w = w * chunk.gain[self.layer_id]
        return out._replace(topk_weights=w.to(out.topk_weights.dtype))

    TopK.forward_cuda = topk_masked

    tokenize = TokenizerManager._tokenize_one_request

    @functools.wraps(tokenize)
    async def tokenize_one(self, obj):
        if obj.image_data is not None:
            obj.image_data = rt.as_image_data(obj.image_data, ImageData)
        return await tokenize(self, obj)

    TokenizerManager._tokenize_one_request = tokenize_one


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target-profile", required=True, type=Path, help="server.json of the deployment to capture with")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--tp", required=True, type=int, help="TP = EP size (each expert whole on one rank)")
    p.add_argument("--model-path", required=True, help="a copy of the target model (checked against its digests)")
    p.add_argument("--context-length", type=int)
    p.add_argument("--vlm-cache-mb", type=int, required=True)
    args = p.parse_args()
    target = json.loads(args.target_profile.read_text())
    if not Path(os.environ[ROOT_ENV]).is_dir():
        raise FileNotFoundError(f"{ROOT_ENV}={os.environ[ROOT_ENV]} is not a directory")
    served_name = reap_served_name(args.target_profile)
    argv = reap_argv(target, args.port, served_name=served_name, model_path=args.model_path, tp=args.tp,
                     context_length=args.context_length)
    flags = dict(rt.parse_flags(argv))
    rt.check_model_files(target, flags["--model-path"][0])
    for key, value in rt.recorded_env(target).items():
        if os.environ.get(key, value) != value:
            print(f"· {key}={value} (recorded; was {os.environ[key]})", file=sys.stderr)
        os.environ[key] = value
    os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = str(args.vlm_cache_mb)
    os.environ["SGLANG_CI_DISABLE_MOE_FUSED_FUNC"] = "1"
    install()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    isolation = {
        "max_running_requests": server_args.max_running_requests == 1,
        "disable_radix_cache": server_args.disable_radix_cache,
        "disable_cuda_graph": server_args.disable_cuda_graph,
        "disable_prefill_cuda_graph": server_args.disable_prefill_cuda_graph,
        "disable_overlap_schedule": server_args.disable_overlap_schedule,
        "no speculative decoding": server_args.speculative_algorithm is None,
        "no session ledger": server_args.session_ledger_dir is None and not server_args.session_replay,
        "triton MoE": server_args.moe_runner_backend == "triton",
        "EP == TP": server_args.ep_size == server_args.tp_size,
    }
    broken = [k for k, ok in isolation.items() if not ok]
    if broken:
        raise ValueError(f"REAP capture needs {broken}")
    import sglang

    tree = Path(sglang.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"], check=True,
                           capture_output=True, text=True).stdout.strip()
    meta = {"target_profile": str(args.target_profile.resolve()),
            "target_profile_sha256": hashlib.sha256(args.target_profile.read_bytes()).hexdigest(),
            "target_instance": target["instance"], "served_model_name": served_name, "argv": argv,
            "env": rt.recorded_env(target), "sglang_tree": str(tree), "sglang_commit": commit,
            "sglang_dirty": bool(dirty), "port": args.port, "categories": list(CATEGORIES), "fields": list(FIELDS)}
    (Path(os.environ[ROOT_ENV]) / "capture_server.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"· REAP capture server, target of {target['instance']} (sglang {commit[:12]}{' dirty' if dirty else ''}): "
          f"{shlex.join(argv)}", file=sys.stderr)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
elif __name__ == "__mp_main__":
    install()
