"""FSDP2 decoder training with FP32 masters/moments and exact global losses."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from checkpointing import load_compressed, save_compressed
from data import block_targets, eligible_anchors, sample_anchors
from init_mtp3 import load_checkpoint
from model import MTP3, CleanCache, LayerCache, append_cache
from objective import objective
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


@dataclass(frozen=True)
class Objective:
    """--objective and its knobs. dpace: hard CE with detached D-PACE weights, covered-label selector CE and an early
    KL on two slots, all at T = 1 (objective.objective, the reference engine's). acceptance: objective.
    acceptance_objective at the deployment sampling temperature -- WTV on the selector's actual proposal plus
    cov_weight x the coverage KL on every slot; joint lets the acceptance term into the backbone."""

    name: str
    temperature: float = 1.0
    cov_weight: float = 0.0
    joint: bool = False
    # OPD (objective.opd_objective): recorded verify cycles in the pool (opd_assemble.py jobs) add opd_weight x the
    # reached-state TV (opd_loss "reached"), the path WTV ("wtv", over the step's cycles) or minus the importance
    # estimate of the expected accepted drafts ("importance", objective.importance_terms: its gradient is unbiased for
    # E[L]'s off the recorded rollouts) -- "reached" and "importance" over the step's reached slots, so a slot's a_j
    # gets the same weight in both at a fresh replay.
    opd_weight: float = 0.0
    opd_loss: str = "reached"
    # The captured (offline) anchors' WTV term weight. 0 with OPD is the design doc's post-training form (section 23):
    # acceptance from the replayed cycles only, the offline anchors keep the candidates through coverage alone.
    accept_weight: float = 1.0

    def __post_init__(self):
        if self.name not in ("dpace", "acceptance"):
            raise ValueError(f"unknown objective {self.name!r}")
        if self.name == "acceptance" and not (self.temperature > 0 and self.cov_weight >= 0):
            raise ValueError("the acceptance objective needs a positive temperature and a nonnegative coverage weight")
        if self.opd_loss not in ("reached", "wtv", "importance") or self.opd_weight < 0:
            raise ValueError(f"invalid OPD settings: weight {self.opd_weight}, loss {self.opd_loss!r}")
        if self.opd_weight and self.name != "acceptance":
            raise ValueError("OPD trains with the acceptance objective")
        if self.accept_weight < 0 or (self.accept_weight != 1 and self.name != "acceptance"):
            raise ValueError(f"the offline accept weight {self.accept_weight} belongs to the acceptance objective")
        if self.name == "acceptance" and not (self.accept_weight or self.cov_weight or self.opd_weight):
            raise ValueError("every term of the acceptance objective is weighted 0")


DPACE = Objective("dpace")


def clean_trace(model, ids, gr, chunk_size=1024):
    """The drafter's clean cache over positions 1 .. len(ids) - 1, built as serving builds it (MTP3.refresh's
    arithmetic: chunk by chunk, each attending the window of history before it) but under autograd when grad is
    enabled -- the full-gradient reference the fast engine is checked against. Every position's K/V and conv
    boundaries are kept."""
    if chunk_size > model.config.window:
        raise ValueError("trace chunks cannot exceed the retained KV ring")
    if len(ids) < 2:
        raise ValueError("clean trace needs at least one position with a predecessor")
    if model.config.context == "inject":
        # Injected rows depend on their own token and predecessor GR only: one pass, no chunks, no boundaries.
        positions = torch.arange(1, len(ids), device=ids.device)[None]
        return model.context(model.embed_tokens(ids[1:][None]), gr[:-1][None], positions)
    window, pieces = None, []
    # Token t pairs with the target GR at t-1. Position 0 has no predecessor, so
    # the draft context starts at 1; see MTP3.refresh.
    for start in range(1, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        positions = torch.arange(start, end, device=ids.device)[None]
        _, new = MTP3.forward(model, model.embed_tokens(ids[start:end][None]), gr[start - 1 : end - 1][None],
                              positions, window)
        window = append_cache(window, new, positions, model.config.window)
        pieces.append(new)
    return CleanCache(
        torch.cat([p.positions for p in pieces], 1),
        tuple(
            LayerCache(
                torch.cat([p.layers[i].key for p in pieces], 2),
                torch.cat([p.layers[i].value for p in pieces], 2),
                tuple(torch.cat([p.layers[i].boundaries[j] for p in pieces], 1)
                      for j in range(len(pieces[0].layers[i].boundaries))),
            )
            for i in range(model.config.num_layers)
        ),
    )


def chunked_terms(model, hidden, labels, predecessors, teacher_gr, chunk_size, valid):
    keys = None
    total = None
    for start in range(0, len(hidden), chunk_size):
        end = min(start + chunk_size, len(hidden))

        def compute(h, y, prev, teacher, ok):
            return tuple(objective(model, h, y, prev, teacher, ok).values())

        terms = checkpoint(
            compute,
            hidden[start:end],
            labels[start:end],
            predecessors[start:end],
            teacher_gr[start:end],
            valid[start:end],
            use_reentrant=False,
        )
        if keys is None:
            keys = (
                "ce_num",
                "ce_den",
                "selector_num",
                "selector_den",
                "kl_num",
                "kl_den",
                "covered",
                "positions",
                "selector_correct",
                "unary_correct",
                "accept_prefix",
                "accept_position",
                "covered_position",
                "accept_histogram",
                "blocks",
            )
            total = terms
        else:
            total = tuple(a + b for a, b in zip(total, terms, strict=True))
    if total is None:
        raise ValueError("no supervised anchors")
    return dict(zip(keys, total, strict=True))


def reduce_loss(terms, world):
    den = torch.stack([terms[k].detach().float() for k in ("ce_den", "selector_den", "kl_den")])
    dist.all_reduce(den)
    return world * (
        terms["ce_num"] / den[0].clamp_min(1e-9)
        + terms["selector_num"] / den[1].clamp_min(1e-9)
        + 0.1 * terms["kl_num"] / den[2].clamp_min(1)
    )


def sync_clip(model, rank, world):
    squared = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise RuntimeError(f"frozen parameter got gradient: {name}")
            continue
        if p.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        g = local(p.grad)
        if not isinstance(p, DTensor):
            dist.all_reduce(g)
            g.div_(world)
        if isinstance(p, DTensor) or rank == 0:
            squared += g.norm(dtype=torch.float32).double().square()
    dist.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError(f"nonfinite global gradient norm: {norm}")
    scale = (1 / (norm + 1e-6)).clamp(max=1).float()
    for p in model.parameters():
        if p.requires_grad:
            local(p.grad).mul_(scale)
    return norm.item()


def sync_clip_replicated(model, world):
    """sync_clip for replicated parameters (the fast engine): one all-reduce over every gradient flattened instead of
    one per parameter (21 -> a few ms a step), same averaging, per-parameter FP32 norms summed in FP64, same clip."""
    params, sizes = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise RuntimeError(f"frozen parameter got gradient: {name}")
            continue
        if p.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        if isinstance(p, DTensor):
            raise RuntimeError(f"{name} is sharded; sync_clip_replicated is for replicated parameters")
        params.append(p)
        sizes.append(p.numel())
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat.div_(world)
    parts = flat.split(sizes)
    norm = torch.stack(torch._foreach_norm(list(parts))).double().square().sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError(f"nonfinite global gradient norm: {norm}")
    flat.mul_((1 / (norm + 1e-6)).clamp(max=1).float())
    for p, part in zip(params, parts, strict=True):
        p.grad.copy_(part.view_as(p.grad))
    return norm.item()


# Token counts and anchor counts spanning the ranges the fast path's compiled graphs get specialized on -- 64-bit
# indexing past 2^31 elements, and range guards inductor's cache carries over from earlier compiles (seen: tokens
# <= 10239; anchors <= 255, >= 640) -- up to beyond the longest recorded segment (209k tokens).
# A segment's clean pass runs at most pool.MAX_SEGMENT_ROWS rows; inductor's 64-bit indexing starts past ~52k.
WARMUP_TOKENS = (1100, 6000, 12000, 30000, 60000)
WARMUP_ANCHORS = (16, 37, 100, 300, 500, 700, 1000, 2048)


def warm_up(model, micro_anchors, world, objective, opd_eval=False, robust=None, gate=None):
    """Compile every shape regime of the fast engine before the first real step. Each rank compiles for itself, and a
    regime first met mid-run (a rank's first short sequence) stalled every rank ~25 s in the soak test. Synthetic
    sequences over WARMUP_TOKENS with anchors spread over them, so the clean pass (under autograd) spans the sequence
    (one segment: the anchors are closer than the receptive field);
    a full and a tiny anchor count, one and two micro-batches; then evaluate()'s no-grad variants (and evaluate_opd()'s,
    opd_eval). The gradients are
    dropped. Every rank runs the same sequence, so the collectives inside fast_step line up. Anchor counts sweep
    WARMUP_ANCHORS on one sequence (their ranges do not depend on its length)."""
    import fast_train

    c = model.config
    generator = torch.Generator(device="cuda").manual_seed(0)
    robust_mix = torch.tensor(0.37, device="cuda") if robust is not None else None
    for tokens in WARMUP_TOKENS:
        ids = torch.randint(c.vocab_size, (tokens,), device="cuda", generator=generator)
        gr = torch.randn(tokens, c.hc_count, c.hidden_size, device="cuda", generator=generator).bfloat16()
        mask = torch.ones(tokens, dtype=torch.bool, device="cuda")
        eligible = eligible_anchors(mask)
        picks = eligible[torch.randperm(len(eligible), generator=generator, device="cuda")[: micro_anchors + 37]]
        picks = picks.sort().values
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sizes = [micro_anchors, *(WARMUP_ANCHORS if tokens == WARMUP_TOKENS[3] else (37,))]
            if gate is not None and tokens == WARMUP_TOKENS[3]:
                # the anchor tails' head chunks are 256 anchors: 257 leaves a chunk of one, a shape of its own (size 1
                # is specialized) that a production micro-batch meets sooner or later
                sizes.append(257)
            for part in [picks[:size] for size in sizes] + [picks]:  # one micro-batch at each size, then two
                fast_step(model, [Segment(ids, gr, mask, part)], micro_anchors, world, objective, robust, robust_mix,
                          gate=gate)
                model.zero_grad(set_to_none=True)
            if objective.opd_weight:  # the OPD terms over the same sizes, alone and in a step with a segment
                b = c.block_size
                drafted = torch.randint(c.vocab_size, (len(picks), b - 1), device="cuda", generator=generator)
                accepted = torch.randint(b, (len(picks),), device="cuda", generator=generator)
                teacher = torch.randn(len(picks), b - 1, c.hc_count, c.hidden_size, device="cuda",
                                      generator=generator).bfloat16()

                sampler_q = torch.full((len(picks), b - 1), 0.5, device="cuda")
                sampler_candidates = torch.randint(c.vocab_size, (len(picks), b - 1, model.candidate_selector.top_k),
                                                   device="cuda", generator=generator)
                cycles = {n: OpdUnit(ids, gr, picks[:n], drafted[:n], accepted[:n], teacher[:n], sampler_q[:n],
                                     sampler_candidates[:n]) for n in sizes}
                for size in sizes:
                    fast_step(model, [cycles[size]], micro_anchors, world, objective)
                    model.zero_grad(set_to_none=True)
                fast_step(model, [Segment(ids, gr, mask, picks[:37]), cycles[37]], micro_anchors, world, objective)
                model.zero_grad(set_to_none=True)
                del drafted, accepted, teacher, sampler_q, sampler_candidates, cycles
            if objective.name == "acceptance":  # evaluate()'s no-grad variants
                with torch.no_grad():
                    trace = fast_train.clean_pass(model, ids, gr, picks)
                    for part in (picks[:micro_anchors], picks[:37]):
                        b = c.block_size
                        labels, predecessors, teacher, valid = block_targets(ids, gr, part, b, b - 1, mask=mask)
                        fast_train.acceptance_terms(model, fast_train.propose(model, trace, ids, gr, part), labels,
                                                    predecessors, teacher, objective.temperature, objective.joint,
                                                    valid=valid, per_anchor=True)
                    del trace
            if opd_eval:  # evaluate_opd()'s: the OPD terms with a sampler's record, no grad
                with torch.no_grad():
                    b = c.block_size
                    trace = fast_train.clean_pass(model, ids, gr, picks)
                    for part in (picks[:micro_anchors], picks[:37]):
                        n = len(part)
                        fast_train.opd_terms(
                            model, fast_train.propose(model, trace, ids, gr, part),
                            torch.randint(c.vocab_size, (n, b - 1), device="cuda", generator=generator), ids[part],
                            torch.randn(n, b - 1, c.hc_count, c.hidden_size, device="cuda",
                                        generator=generator).bfloat16(),
                            torch.randint(b, (n,), device="cuda", generator=generator), objective.temperature,
                            objective.joint, sampler_q=torch.full((n, b - 1), 0.5, device="cuda"),
                            sampler_candidates=torch.randint(c.vocab_size, (n, b - 1, model.candidate_selector.top_k),
                                                             device="cuda", generator=generator))
                    del trace
        del ids, gr
    torch.cuda.synchronize()


def save_state(model, optimizer, out, step, rank, world, source, dataset_manifest, replicated=False, extra_files=None,
               extra_state=None):
    # Save local tensors, preserving exact FP32 masters and Adam moments.
    # extra_files: {name: text} rank 0 writes into the checkpoint before it is
    # published (the pool's consumed-job list), so they are exactly as durable;
    # extra_state: values saved with the tensors (the pool's trained anchors).
    # Two generations exist only during atomic publication; never silently
    # drop optimizer state or reset the continuous production schedule.
    # Replicated parameters (the fast engine) and their moments are identical
    # on every rank after each step, so rank 0 writes the one copy.
    writer = rank == 0 or not replicated
    need = sum(local(p).numel() * 12 for p in model.parameters() if p.requires_grad) if writer else 0
    needed = torch.tensor(need, device="cuda", dtype=torch.int64)
    dist.all_reduce(needed)
    reserve = float(os.environ.get("DFLASH_DISK_RESERVE_GIB", "8"))
    if not math.isfinite(reserve) or reserve < 8:
        raise ValueError("DFLASH_DISK_RESERVE_GIB must be finite and at least 8")
    if shutil.disk_usage(out).free < needed.item() + reserve * 2**30:
        raise RuntimeError(f"insufficient checkpoint space with {reserve:g} GiB reserve; retaining previous checkpoint")
    temp = out / f"step-{step:09d}.partial"
    if rank == 0:
        # This run is the checkpoint's only writer, so a leftover temp of this step is a save that died mid-write
        # (PR #6 F6): drop it.
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
    dist.barrier()
    if writer:
        parameters = {n: local(p).detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        opt_state = optimizer.state_dict()
        states = {
            idx: {k: local(v).cpu() if torch.is_tensor(v) else v for k, v in state.items()}
            for idx, state in opt_state["state"].items()
        }
        compressed_bytes = save_compressed(
            {
                "parameters": parameters,
                "optimizer": {"state": states, "param_groups": opt_state["param_groups"]},
                # the optimizer state is indexed by position: resume checks the names in that order
                "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
                "state_digest": state_digest(model, optimizer) if replicated else None,
                "step": step,
                **(extra_state or {}),
                "rank": rank,
                "world_size": world,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            temp / ("replicated.pt.zst" if replicated else f"rank-{rank}.pt.zst"),
        )
        print(json.dumps({"checkpoint_step": step, "rank": rank, "compressed_bytes": compressed_bytes}), flush=True)
    dist.barrier()
    if rank == 0:
        for name, text in (extra_files or {}).items():
            (temp / name).write_text(text)
        (temp / "manifest.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "format": "torch-zstd-v1",
                    "world_size": world,
                    "replicated": replicated,
                    "source_init": str(source),
                    "dataset_manifest": str(dataset_manifest),
                },
                indent=2,
            )
        )
        destination = out / f"step-{step:09d}"
        if destination.exists():
            # Published by a segment that died before `latest` named it; this run resumed from an earlier checkpoint
            # and has just recomputed the step.
            if (out / "latest").is_file() and (out / "latest").read_text().strip() == destination.name:
                raise RuntimeError(f"{destination} is the latest checkpoint; refusing to overwrite it")
            shutil.rmtree(destination)
        temp.rename(destination)
        link = out / "latest.partial"
        link.write_text(destination.name)
        link.replace(out / "latest")
        for old in out.glob("step-*"):
            if old != destination and old.is_dir() and not old.name.endswith(".partial"):
                shutil.rmtree(old)
    dist.barrier()


def decays(name, parameter):
    """Weight decay pulls toward zero: right for the matrices, wrong for gains (RMSNorm's 1 + w is already pulled to
    identity by decaying w, but a 1-D tensor is no matrix), the convs' identity taps (base_kernel starts at 1), the
    selector's codebooks (rows the data never touches would only shrink) and the mask embedding."""
    return not (parameter.dim() < 2 or name.endswith("base_kernel") or name.endswith("_codebook")
                or name == "mask_embedding")


def state_digest(model, optimizer):
    """Sums of squares of every trainable parameter and both Adam moments, in optimizer order (FP64 on the device):
    saved with a checkpoint and recomputed after a resume loads it, so a resume that restores anything else fails."""
    params = [p for group in optimizer.param_groups for p in group["params"]]
    digest = {"parameters": torch.stack([local(p).double().square().sum() for p in params]).sum()}
    for key in ("exp_avg", "exp_avg_sq"):
        digest[key] = torch.stack([local(optimizer.state[p][key]).double().square().sum() for p in params]).sum()
    return {k: v.item() for k, v in digest.items()}


def optimizer_parameter_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [names[id(p)] for group in optimizer.param_groups for p in group["params"]]


def restore_training_state(model, optimizer, saved, world, rank, replicated):
    """Restore the exact masters, Adam groups/moments and RNG; shared by the transition gate."""
    if saved["world_size"] != world or (not replicated and saved["rank"] != rank):
        raise ValueError("resume topology mismatch")
    if saved["optimizer_parameter_names"] != optimizer_parameter_names(model, optimizer):
        raise ValueError("the checkpoint's optimizer state is for a different parameter order")
    with torch.no_grad():
        for name, value in saved["parameters"].items():
            local(model.get_parameter(name)).copy_(value)
    # DTensor state must use the matching parameter's sharding placement.
    flat_params = [param for group in optimizer.param_groups for param in group["params"]]
    for idx, state in saved["optimizer"]["state"].items():
        parameter = flat_params[idx]
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key].to("cuda")
            if isinstance(parameter, DTensor):
                value = DTensor.from_local(
                    value,
                    device_mesh=parameter.device_mesh,
                    placements=parameter.placements,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
            state[key] = value
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    if replicated:
        loaded = state_digest(model, optimizer)
        if any(abs(loaded[k] - v) > 1e-9 * abs(v) for k, v in saved["state_digest"].items()):
            raise RuntimeError(f"resume restored a different state: {loaded} != {saved['state_digest']}")


class FeaturePrefetch:
    """Loads a job's features on a worker thread into pinned host memory, so the next sequence's read and copy
    overlap this step's compute. Waits for READY like the inline loop did and fails the same way."""

    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.pending = {}

    @staticmethod
    def _load(directory):
        waiting = time.monotonic()
        while not (directory / "READY").is_file():
            if (directory.parent / "train.producer_done").is_file():
                raise RuntimeError(f"producer completed without required feature {directory.name}")
            time.sleep(0.25)
        ready = time.monotonic()
        host = {k: v.pin_memory() for k, v in load_file(str(directory / "features.safetensors")).items()}
        if (directory / "cycles.safetensors").is_file():  # an OPD job's recorded verify cycles
            host |= {f"cycle_{k}": v.pin_memory() for k, v in load_file(str(directory / "cycles.safetensors")).items()}
        metadata = json.loads((directory / "record.json").read_text())
        return host, metadata, ready - waiting

    def request(self, directory):
        if directory not in self.pending:
            self.pending[directory] = self.pool.submit(self._load, directory)

    def take(self, directory):
        self.request(directory)
        begin = time.monotonic()
        host, metadata, producer_wait = self.pending.pop(directory).result()
        blocked = time.monotonic() - begin
        return {k: v.to("cuda", non_blocking=True) for k, v in host.items()}, metadata, producer_wait, blocked


def accept_metrics(report):
    """The reduced terms as the per-step metrics (train.py's metrics.jsonl fields)."""
    blocks = report["blocks"].clamp_min(1)
    # Per-slot rates over the blocks that reach the slot (a block stops at the end of its supervised run).
    reach = report["valid_position"].clamp_min(1) if "valid_position" in report else blocks
    greedy = {
        "selector_coverage": (report["covered"] / report["positions"]).item(),
        "selector_conditional_accuracy": (report["selector_correct"] / report["covered"].clamp_min(1)).item(),
        "blocks": int(report["blocks"].item()),
        # The selector's argmax against the recorded continuation, which was sampled (at T = 1 or 0.7, the corpus's):
        # the leading run of matches. Not a greedy verify's acceptance -- that compares with the target's argmax.
        "argmax_match_length": (report["accept_prefix"].sum() / blocks).item(),
        # match_rate_prefix[j] = P(slots 0..j all match); match_rate_position[j] = P(slot j matches).
        "match_rate_prefix": (report["accept_prefix"] / reach).tolist(),
        "match_rate_position": (report["accept_position"] / reach).tolist(),
        "recall_at_k_position": (report["covered_position"] / reach).tolist(),
        "match_prefix_histogram": [int(v) for v in report["accept_histogram"].tolist()],
    }
    if "accept_num" not in report:
        return {
            "ce": (report["ce_num"] / report["ce_den"]).item(),
            "early_kl": (report["kl_num"] / report["kl_den"]).item(),
            **greedy,
        }
    positions = report["positions"].clamp_min(1)
    renewal = {}
    if "renewal_cycles" in report:
        # Serving's next anchor is a + L + 1, so it visits hard stretches more often than a uniform average over
        # positions: tokens per cycle along the recorded path from per-anchor survival (evaluate's renewal).
        tokens_per_cycle = (report["renewal_tokens"] / report["renewal_cycles"].clamp_min(1e-9)).item()
        renewal = {"renewal_accept_length": tokens_per_cycle - 1.0, "renewal_cycles": report["renewal_cycles"].item()}
    robust_metrics = {}
    if "accept_trust_num" in report:
        from robust_objective import CORPORA

        robust_metrics = {
            "robust_accept_optimization": (report["accept_opt_num"] / blocks).item(),
            "robust_accept_trust": (report["accept_trust_num"] / blocks).item(),
            "robust_tail_hit_position": (report["robust_tail_hits"] / reach).tolist(),
            "robust_trust_position": (report["robust_trust_sum"] / reach).tolist(),
            "robust_floor_position": (report["robust_floor_hits"] / reach).tolist(),
            "robust_band_mass_fraction": (report["robust_band_mass_num"]
                                          / report["robust_candidate_mass_num"].clamp_min(1e-20)).item(),
        }
        for corpus in CORPORA:
            robust_metrics[f"robust_tail_hit/{corpus}"] = (
                report[f"robust_tail_hits/{corpus}"] / report[f"robust_positions/{corpus}"].clamp_min(1)).item()
            robust_metrics[f"robust_positions/{corpus}"] = int(report[f"robust_positions/{corpus}"].item())
    if "gate_weight_sum" in report:
        from prefix_gate import metrics as gate_metrics

        robust_metrics |= gate_metrics(report)
    return {
        "loss_accept": (report["accept_num"] / blocks).item(),
        "coverage_kl": (report["cov_num"] / positions).item(),
        # Rejection sampling at the objective's temperature, teacher-forced on the recorded prefix: per slot the
        # acceptance probability a_j, the teacher mass the candidates cover p(C_j) (a_j <= p(C_j)), and the survival
        # prod_{i<=j} a_i, whose sum is the expected accepted drafts per cycle -- averaged uniformly over anchors.
        "expected_accept_length": (report["survival_position"].sum() / blocks).item(),
        **renewal,
        **robust_metrics,
        # 1 - a_j = (1 - p(C_j)) + (p(C_j) - a_j): teacher mass outside the candidates (only coverage can fix it) plus
        # mass inside them the proposal misallocates (the selector's part), averaged over the slots.
        "gap_outside_candidates": (1 - report["mass_position"] / reach).mean().item(),
        "gap_inside_candidates": ((report["mass_position"] - report["overlap_position"]) / reach).mean().item(),
        "accept_probability_position": (report["overlap_position"] / reach).tolist(),
        "candidate_mass_position": (report["mass_position"] / reach).tolist(),
        "survival_position": (report["survival_position"] / reach).tolist(),
        "ce_t1": (report["ce_t1_num"] / positions).item(),
        "ce_t": (report["ce_t_num"] / positions).item(),
        **greedy,
        **(opd_metrics(report) if "opd_blocks" in report else {}),
    }


def opd_metrics(report):
    """The step's recorded cycles: reached-state TV, path WTV and the importance loss as trained, the path's
    teacher-forced survival, the rho estimate of the expected accepted drafts (unbiased only for a fresh replay) and the
    importance one (objective.importance_terms: the sampler is the drafter that rolled them out), the share of weights
    above objective.WEIGHT_DRIFT, the accepted count observed at rollout, and how many recorded drafts are still among the current
    drafter's candidates."""
    cycles = report["opd_blocks"].clamp_min(1)
    slots = len(report["opd_overlap_position"])
    return {
        "opd_cycles": int(report["opd_blocks"].item()),
        "opd_reached_tv": (report["opd_reach_num"] / report["opd_reach_den"].clamp_min(1)).item(),
        "opd_path_wtv": (report["opd_wtv_num"] / cycles).item(),
        "opd_expected_accept_length": (report["opd_survival_position"].sum() / cycles).item(),
        "opd_rho_accept_length": (report["opd_rho_survival_position"].sum() / cycles).item(),
        "opd_observed_accept_length": (report["opd_accepted_observed"] / cycles).item(),
        "opd_importance_loss": (-report["opd_is_num"] / report["opd_reach_den"].clamp_min(1)).item(),
        "opd_is_accept_length": (report["opd_is_num"] / cycles).item(),
        "opd_is_weights_drifted": (report["opd_is_weight_drifted"] / (cycles * (slots - 1))).item(),
        "opd_sampler_covered": (report["opd_sampler_covered_position"].sum() / (cycles * slots)).item(),
        "opd_drafts_in_candidates": (report["opd_drafted_candidate_position"].sum() / (cycles * slots)).item(),
        "opd_accept_probability_position": (report["opd_overlap_position"] / cycles).tolist(),
    }


def opd_heldout_metrics(report):
    """evaluate_opd's report: the importance estimate of this drafter's expected accepted drafts per cycle on the
    held-out rollouts (objective.importance_terms; its standard error treats cycles as independent -- a lower bound,
    a turn's cycles are not) next to what the rollout drafter accepted there, the naive replay rho (biased once the
    weights moved), and the share of this drafter's proposal on the rollout drafter's candidates (below 1: the estimate
    reads low by what it cannot see)."""
    cycles = report["blocks"].clamp_min(1)
    slots = len(report["is_survival_position"])
    estimate = report["is_survival_position"].sum() / cycles
    return {
        "cycles": int(report["blocks"].item()),
        "accept_length": estimate.item(),
        "accept_length_se": ((report["is_length_sq"] / cycles - estimate**2).clamp_min(0) / cycles).sqrt().item(),
        "rollout_accept_length": (report["accepted_observed"] / cycles).item(),
        "naive_rho_accept_length": (report["rho_survival_position"].sum() / cycles).item(),
        "reached_tv": (report["reach_num"] / report["reach_den"].clamp_min(1)).item(),
        "drafts_in_candidates": (report["drafted_candidate_position"].sum() / (cycles * slots)).item(),
        "sampler_covered": (report["sampler_covered_position"].sum() / (cycles * slots)).item(),
        "weights_drifted": (report["is_weight_drifted"] / (cycles * (slots - 1))).item(),
        "survival_position": (report["is_survival_position"] / cycles).tolist(),
        "sampler_covered_position": (report["sampler_covered_position"] / cycles).tolist(),
    }


def write_record(out, record, run):
    with (out / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if run is not None:
        flat = {k: v for k, v in record.items() if isinstance(v, (int, float))}
        flat.update({f"time/{k}": v for k, v in record["stage_gpu_ms"].items()})
        flat.update({f"match_prefix/pos{j + 1}": v for j, v in enumerate(record["match_rate_prefix"])})
        flat.update({f"match_position/pos{j + 1}": v for j, v in enumerate(record["match_rate_position"])})
        flat.update({f"recall16/pos{j + 1}": v for j, v in enumerate(record["recall_at_k_position"])})
        flat.update({f"match_hist/{j}": v for j, v in enumerate(record["match_prefix_histogram"])})
        flat.update(position_series(record))
        for key in ("learning_rates", "consumed_tokens_by_corpus", "consumed_jobs_by_corpus", "sink_logits"):
            if key in record:
                flat.update({f"{key}/{name}": v for name, v in record[key].items()})
        run.log(flat, step=record["step"])


def position_series(record):
    """The acceptance objective's per-slot lists as wandb scalars (slot j + 1 = the (j + 1)-th draft)."""
    flat = {}
    for key, prefix in (("accept_probability_position", "accept_t"), ("candidate_mass_position", "mass16"),
                        ("survival_position", "survival_t")):
        if key in record:
            flat.update({f"{prefix}/pos{j + 1}": v for j, v in enumerate(record[key])})
    return flat


def reduce_report(terms):
    """Every term summed over ranks in one all-reduce, in sorted key order: a rank's dict may be built in a different
    order (zero_terms), and per-key all-reduces in dict order then pair different tensors and hang. All ranks check the
    flattened length first, so a mismatch raises everywhere instead of hanging."""
    keys = sorted(terms)
    flat = torch.cat([terms[k].detach().float().reshape(-1) for k in keys])
    bounds = torch.tensor([flat.numel(), -flat.numel()], device=flat.device)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
    if bounds[0] != -bounds[1]:
        raise RuntimeError(f"ranks disagree on the reported terms: {flat.numel()} values here, {bounds.tolist()}")
    dist.all_reduce(flat)
    parts = flat.split([terms[k].numel() for k in keys])
    return {k: part.view(terms[k].shape) for k, part in zip(keys, parts, strict=True)}


def zero_terms(block_size, objective, robust=None, gate=None):
    """The terms of a rank with nothing to do this step: it still joins every reduction."""
    if objective.name == "acceptance":
        scalars = ("cov_num", "accept_num", "blocks", "positions", "ce_t1_num", "ce_t_num", "covered",
                   "selector_correct", "unary_correct")
        vectors = ("valid_position", "overlap_position", "mass_position", "survival_position", "accept_prefix",
                   "accept_position", "covered_position")
    else:
        scalars = ("ce_num", "ce_den", "selector_num", "selector_den", "kl_num", "kl_den", "covered", "positions",
                   "selector_correct", "unary_correct", "blocks")
        vectors = ("accept_prefix", "accept_position", "covered_position")
    terms = {k: torch.zeros((), device="cuda") for k in scalars}
    for k in vectors:
        terms[k] = torch.zeros(block_size - 1, device="cuda")
    terms["accept_histogram"] = torch.zeros(block_size, device="cuda")
    if objective.opd_weight:
        terms |= zero_opd_terms(block_size)
    if robust is not None:
        from robust_objective import zero_terms as robust_zeros

        terms |= robust_zeros(block_size - 1, "cuda")
    if gate is not None:
        from prefix_gate import zero_terms as gate_zeros

        terms |= gate_zeros(block_size - 1, "cuda")
    return terms


OPD_SCALARS = ("reach_num", "reach_den", "wtv_num", "blocks", "cov_num", "accepted_observed", "is_num", "is_length_sq",
               "is_weight_drifted")
OPD_VECTORS = ("overlap_position", "mass_position", "survival_position", "reached_position",
               "drafted_candidate_position", "rho_survival_position", "is_survival_position",
               "sampler_covered_position")


def zero_opd_terms(block_size, prefix="opd_"):
    """An OPD-enabled step's OPD terms when this rank has no recorded cycle, "opd_"-prefixed like the real ones
    (evaluate_opd's unprefixed). Training and evaluate_opd both pass the sampler's record, so objective.importance_terms'
    are among them."""
    terms = {f"{prefix}{k}": torch.zeros((), device="cuda") for k in OPD_SCALARS}
    terms |= {f"{prefix}{k}": torch.zeros(block_size - 1, device="cuda") for k in OPD_VECTORS}
    terms[f"{prefix}accepted_histogram"] = torch.zeros(block_size, device="cuda")
    return terms


@dataclass
class Segment:
    """One job's anchors in a step, ascending (a contiguous run of its anchor order): the step runs the clean pass over
    them under autograd (fast_train.clean_pass from clean_start) and their blocks."""

    ids: torch.Tensor
    gr: torch.Tensor
    mask: torch.Tensor
    anchors: torch.Tensor
    corpus: str = "other"


@dataclass
class OpdUnit:
    """A slice of an OPD job's recorded verify cycles, ascending: anchors (positions in the job's sequence), the drafts,
    the accepted counts, the verify block's GR rows 0..b-2 (the teacher of each slot on the recorded path), and the
    rollout drafter's q of each draft and candidates (objective.importance_terms' sampler)."""

    ids: torch.Tensor
    gr: torch.Tensor
    anchors: torch.Tensor
    drafted: torch.Tensor
    accepted: torch.Tensor
    teacher: torch.Tensor
    sampler_q: torch.Tensor
    sampler_candidates: torch.Tensor


def unit_anchors(unit):
    return len(unit.anchors)


def _event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _clean_trace_of(model, unit, timing):
    """The unit's clean pass under autograd and its leaves (fast_train.trace_leaves), timed into timing."""
    import fast_train

    began = _event()
    trace = fast_train.clean_pass(model, unit.ids, unit.gr, unit.anchors,
                                  fast_train.clean_start(unit.anchors, model.config))
    timing["clean_forward"].append((began, _event()))
    return trace, fast_train.trace_leaves(trace)


def _backward_trace(trace, leaves, timing):
    import fast_train

    began = _event()
    fast_train.backward_trace(trace, leaves)
    timing["clean_backward"].append((began, _event()))


def fast_step(model, units, micro_anchors, world, objective, robust=None, robust_mix=None, gate=None, cov_weight=None):
    """This rank's anchors for one step through the fast engine, with gradient accumulation. units: Segments (one or
    more sequences). Returns the summed (detached) terms, CUDA events after the wait for the other ranks / the first
    pass and after the units, and the clean passes' (start, end) events.

    Each unit's clean pass runs once under autograd; its micro-batches of proposals read the pass's leaves
    (fast_train.trace_leaves) and their gradients go back through it once at the end (backward_trace), so the loss
    reaches every parameter through the context too.

    The loss is the reference's: every term normalized by its total over everything every rank does in the step. The
    denominators (the D-PACE weight sums) exist only after the forward passes. When every rank's anchors fit in
    micro_anchors, every micro-batch's graph is kept until reduce_loss has them, then one backward. Otherwise a first
    no-grad pass computes them with the head statistics (fast_train.head_statistics), and the second recomputes each
    proposal with grad and backpropagates the stored head gradients: the proposal's forward runs twice (measured +18%
    on the anchors). Normalizing each micro-batch by its own totals instead would weight a 2-anchor micro-batch like a
    4096-anchor one. A rank with nothing to do still joins the reductions and contributes zero gradients.

    The acceptance objective's normalizers are the step's blocks and supervised slots over all ranks, known before any
    forward: each micro-batch is backpropagated as soon as it is computed (fast_step_acceptance). gate (prefix_gate)
    and cov_weight (this step's coverage weight, the objective's when None) belong to it.
    """
    import fast_train

    if objective.name == "acceptance":
        return fast_step_acceptance(model, units, micro_anchors, world, objective, robust, robust_mix, gate, cov_weight)
    if robust is not None or gate is not None or cov_weight is not None:
        raise ValueError("robust training, the prefix gate and a coverage weight belong to the acceptance objective")
    timing = {"clean_forward": [], "clean_backward": []}
    traces, parts = [], []
    for unit in units:
        trace, leaves = _clean_trace_of(model, unit, timing)
        traces.append((trace, leaves))
        parts += [(leaves, unit, unit.anchors[i : i + micro_anchors]) for i in range(0, len(unit.anchors), micro_anchors)]
    most = torch.tensor(sum(len(part) for _, _, part in parts), device="cuda")
    dist.all_reduce(most, op=dist.ReduceOp.MAX)
    single = most.item() <= micro_anchors
    marks = [_event()]
    report = None

    def accumulate(terms):
        nonlocal report
        detached = {k: v.detach() for k, v in terms.items()}
        report = detached if report is None else {k: report[k] + v for k, v in detached.items()}

    if single:
        marks.append(_event())
        total = None
        for leaves, unit, part in parts:
            labels, predecessors, teacher, valid = block_targets(unit.ids, unit.gr, part, mask=unit.mask)
            hidden = fast_train.propose(model, leaves, unit.ids, unit.gr, part)
            terms = fast_train.head_terms(model, hidden, labels, predecessors, teacher, valid=valid)
            total = terms if total is None else {k: total[k] + v for k, v in terms.items()}
        total = zero_terms(model.config.block_size, objective) if total is None else total
        loss = reduce_loss(total, world)
        if parts:
            loss.backward()
        accumulate(total)
    else:
        statistics = []
        with torch.no_grad():
            for leaves, unit, part in parts:
                labels, predecessors, teacher, valid = block_targets(unit.ids, unit.gr, part, mask=unit.mask)
                hidden = fast_train.propose(model, leaves, unit.ids, unit.gr, part)
                stats = fast_train.head_statistics(model, hidden, labels, teacher, valid=valid)
                accumulate(fast_train.objective_from_statistics(model, stats, hidden, labels, predecessors, valid))
                statistics.append(stats)
        marks.append(_event())
        if report is None:
            report = zero_terms(model.config.block_size, objective)
        den = torch.stack([report[k].float() for k in ("ce_den", "selector_den", "kl_den")])
        dist.all_reduce(den)
        for (leaves, unit, part), stats in zip(parts, statistics, strict=True):
            labels, predecessors, _, valid = block_targets(unit.ids, unit.gr, part, mask=unit.mask)
            hidden = fast_train.propose(model, leaves, unit.ids, unit.gr, part)
            terms = fast_train.objective_from_statistics(model, stats, hidden, labels, predecessors, valid)
            loss = world * (
                terms["ce_num"] / den[0].clamp_min(1e-9)
                + terms["selector_num"] / den[1].clamp_min(1e-9)
                + 0.1 * terms["kl_num"] / den[2].clamp_min(1)
            )
            loss.backward()
    for trace, leaves in traces:
        _backward_trace(trace, leaves, timing)
    if not parts:  # sync_clip requires a gradient for every trainable parameter
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    marks.append(_event())
    return report, marks, timing


def fast_step_acceptance(model, units, micro_anchors, world, objective, robust=None, robust_mix=None, gate=None,
                         cov_weight=None):
    """fast_step for the acceptance objective: loss = accept_num / blocks + cov_weight * cov_num / supervised slots over
    the whole step on every rank, each micro-batch backpropagated right after its forward, each unit's clean pass
    once after its micro-batches. With the prefix gate the numerators are accept_opt_num and cov_opt_num (both
    weighted, the normalizers unchanged)."""
    import fast_train
    from data import slot_mask

    if robust is not None and (objective.opd_weight or robust_mix is None):
        raise ValueError("robust training requires offline-only data and a shared mix tensor")
    if gate is not None and (robust is not None or objective.opd_weight):
        raise ValueError("the prefix gate trains offline data only, without the robust objective")
    if cov_weight is None:
        cov_weight = objective.cov_weight
    elif not cov_weight >= 0:
        raise ValueError(f"coverage weight {cov_weight}")
    b = model.config.block_size
    timing = {"clean_forward": [], "clean_backward": []}
    marks = [_event()]
    segments = [u for u in units if isinstance(u, Segment)]
    cycles = [u for u in units if isinstance(u, OpdUnit)]
    if len(segments) + len(cycles) != len(units):
        raise TypeError("units are Segments and OpdUnits")
    if cycles and not objective.opd_weight:
        raise RuntimeError("OPD jobs in the pool but --opd-weight is 0")

    def total(values):
        return torch.stack(values).sum().double() if values else torch.zeros((), device="cuda", dtype=torch.float64)

    # Reached slots per cycle: the accepted drafts and the first rejection, min(L + 1, b - 1) -- on the device.
    counts = torch.stack([
        torch.tensor(float(sum(len(u.anchors) for u in segments)), device="cuda", dtype=torch.float64),
        total([slot_mask(u.mask, u.anchors, b).sum() for u in segments]),
        torch.tensor(float(sum(len(u.anchors) for u in cycles)), device="cuda", dtype=torch.float64),
        total([(u.accepted.clamp(max=b - 2) + 1).sum() for u in cycles]),
    ])
    dist.all_reduce(counts)
    blocks, slots, opd_blocks, reach_den = counts.tolist()
    if blocks + opd_blocks == 0:
        raise RuntimeError("a step with no anchors on any rank")
    marks.append(_event())
    report = None

    def accumulate(terms):
        nonlocal report
        detached = {k: v.detach() for k, v in terms.items()}
        report = detached if report is None else {k: report[k] + v for k, v in detached.items()}

    for unit in units:
        trace, leaves = _clean_trace_of(model, unit, timing)
        for start in range(0, len(unit.anchors), micro_anchors):
            rows = slice(start, start + micro_anchors)
            part = unit.anchors[rows]
            hidden = fast_train.propose(model, leaves, unit.ids, unit.gr, part)
            if isinstance(unit, Segment):
                labels, predecessors, teacher, valid = block_targets(unit.ids, unit.gr, part, b, b - 1, mask=unit.mask)
                gated = {} if gate is None else {"gate": gate, "anchor_tokens": unit.ids[part],
                                                 "anchor_gr": unit.gr[part - 1]}
                terms = fast_train.acceptance_terms(model, hidden, labels, predecessors, teacher,
                                                    objective.temperature, objective.joint, valid=valid,
                                                    robust=robust, robust_mix=robust_mix, corpus=unit.corpus, **gated)
                optimized = robust is not None or gate is not None
                loss = world * (objective.accept_weight * terms["accept_opt_num" if optimized else "accept_num"] / blocks
                                + cov_weight * terms["cov_opt_num" if gate is not None else "cov_num"] / slots)
                accumulate(terms | (zero_opd_terms(b) if objective.opd_weight else {}))
            else:
                terms = fast_train.opd_terms(model, hidden, unit.drafted[rows], unit.ids[part], unit.teacher[rows],
                                             unit.accepted[rows], objective.temperature, objective.joint,
                                             sampler_q=unit.sampler_q[rows],
                                             sampler_candidates=unit.sampler_candidates[rows])
                if objective.opd_loss == "reached":
                    opd = terms["reach_num"] / reach_den
                elif objective.opd_loss == "wtv":
                    opd = terms["wtv_num"] / opd_blocks
                else:
                    opd = -terms["is_num"] / reach_den
                loss = world * objective.opd_weight * opd
                accumulate({k: v for k, v in zero_terms(b, objective).items() if not k.startswith("opd_")}
                           | {f"opd_{k}": v for k, v in terms.items()})
            loss.backward()
            del hidden, terms, loss
        _backward_trace(trace, leaves, timing)
        del trace, leaves
    if report is None:
        report = zero_terms(b, objective, robust, gate)
        for parameter in model.parameters():  # sync_clip requires a gradient for every trainable parameter
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    marks.append(_event())
    return report, marks, timing


WARMUP_FRACTION = 0.03


def lr_factor(progress):
    """The production schedule at a fraction `progress` of the run: linear warm-up over the first 3%, then cosine to
    zero. A pool run measures progress in trained anchors over the plan's (--total-anchors), which the plan knows
    exactly; a smoke in steps."""
    progress = min(progress, 1.0)
    w = WARMUP_FRACTION
    return progress / w if progress < w else 0.5 * (1 + math.cos(math.pi * (progress - w) / (1 - w)))


def wsd_lr_factor(progress, decay_fraction):
    """Warmup-stable-decay (--lr-schedule wsd): linear warm-up over the first 3%, the peak until 1 - decay_fraction,
    then linear to zero at the end."""
    if not 0 < decay_fraction <= 1 - WARMUP_FRACTION:
        raise ValueError(f"decay fraction {decay_fraction} must be in (0, {1 - WARMUP_FRACTION}]")
    progress = min(progress, 1.0)
    if progress < WARMUP_FRACTION:
        return progress / WARMUP_FRACTION
    return min(1.0, (1.0 - progress) / decay_fraction)


def schedule_factor(args, progress):
    return (wsd_lr_factor(progress, args.lr_decay_fraction) if args.lr_schedule == "wsd" else lr_factor(progress))


def coverage_weight_at(args, progress):
    """This step's coverage weight: --cov-weight, or with --cov-weight-final linear from it to the final one over the
    run's progress (the step's start; a smoke's progress is 0, so it keeps --cov-weight)."""
    start = args.objective.cov_weight
    if args.cov_weight_final is None:
        return start
    return start + (args.cov_weight_final - start) * min(max(progress, 0.0), 1.0)


def renewal(anchors, survival):
    """Expected cycles and emitted tokens when serving drafts along a recorded path. anchors: ascending positions,
    survival [anchors, b - 1]: P(L >= k) at each, zero past a block's supervised slots. A chain starts at the first
    anchor and at every run's first anchor (a turn starts after the target's own token; no chain lands there), a cycle
    at a accepts L drafts and the next anchors at a + L + 1, and a chain whose next anchor is not among the anchors
    ends (the turn or the window did). Returns (sum of visit probabilities, sum of visit x (E[L] + 1))."""
    import numpy as np

    positions = anchors.tolist()
    index = {p: i for i, p in enumerate(positions)}
    tail = np.concatenate((np.ones((len(positions), 1)), survival.double().cpu().numpy()), 1)  # P(L >= k), k = 0..m
    exact = np.concatenate((tail[:, :-1] - tail[:, 1:], tail[:, -1:]), 1)  # P(L = k), k = 0..m
    visit = np.zeros(len(positions))
    for i, p in enumerate(positions):
        if i == 0 or p - 1 not in index:
            visit[i] += 1.0
        for k, probability in enumerate(exact[i]):
            j = index.get(p + k + 1)
            if j is not None:
                visit[j] += visit[i] * probability
    return float(visit.sum()), float((visit * (tail[:, 1:].sum(1) + 1)).sum())


@torch.no_grad()
def evaluate(model, jobs, objective, micro_anchors, max_anchors, rank, world):
    """The objective's terms on held-out jobs, reduced over ranks: each rank takes jobs[rank::world]; per job every
    eligible anchor, or a contiguous window of max_anchors of them placed by the job's name (contiguous so that the
    acceptance objective's renewal estimate -- renewal_cycles, renewal_tokens -- can follow serving's cycles). Clean
    passes per segment (pool.segments), as training runs them."""
    import zlib

    import fast_train
    from pool import segments

    b = model.config.block_size
    overlap = fast_train.receptive_field(model.config)
    total = None
    # The renewal sums are per job (a job's cycles chain through its anchors), so they accumulate apart from the
    # micro-batch terms: folded into `total`, the next job's first micro-batch rebuilt the dict from its own keys and
    # dropped them, leaving each rank only its last job's renewal (2026-10-08 review F1).
    renewal_cycles = renewal_tokens = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for directory in jobs[rank::world]:
            features = {k: v.cuda() for k, v in load_file(str(directory / "features.safetensors")).items()}
            ids, gr, mask = features["input_ids"], features["target_gr"], features["loss_mask"]
            if not torch.equal(features["positions"], torch.arange(len(ids), device="cuda")):
                raise ValueError(f"{directory}: capture position mismatch")
            anchors = eligible_anchors(mask)
            if len(anchors) > max_anchors:
                start = zlib.crc32(directory.name.encode()) % (len(anchors) - max_anchors + 1)
                anchors = anchors[start : start + max_anchors]
            if len(anchors) == 0:
                continue
            survival = []
            for lo, hi in segments(anchors.cpu().numpy(), 0, len(anchors), overlap):
                segment = anchors[lo:hi]
                trace = fast_train.clean_pass(model, ids, gr, segment, fast_train.clean_start(segment, model.config))
                for start in range(0, len(segment), micro_anchors):
                    part = segment[start : start + micro_anchors]
                    hidden = fast_train.propose(model, trace, ids, gr, part)
                    if objective.name == "acceptance":
                        labels, predecessors, teacher, valid = block_targets(ids, gr, part, b, b - 1, mask=mask)
                        terms = fast_train.acceptance_terms(model, hidden, labels, predecessors, teacher,
                                                            objective.temperature, objective.joint, valid=valid,
                                                            per_anchor=True)
                        survival.append(terms.pop("survival_anchor"))
                    else:
                        labels, predecessors, teacher, valid = block_targets(ids, gr, part, mask=mask)
                        terms = fast_train.head_terms(model, hidden, labels, predecessors, teacher, valid=valid)
                    terms = {k: v.detach().float() for k, v in terms.items()}
                    total = add_terms(total, terms)
                del trace
            if survival:
                cycles, tokens = renewal(anchors, torch.cat(survival))
                renewal_cycles += cycles
                renewal_tokens += tokens
            del features
    if total is None:  # a rank without held-out jobs: the offline terms only, like the ranks that had some
        total = {k: v for k, v in zero_terms(b, objective).items() if not k.startswith("opd_")}
    if objective.name == "acceptance":
        total["renewal_cycles"] = torch.tensor(renewal_cycles, device="cuda", dtype=torch.float32)
        total["renewal_tokens"] = torch.tensor(renewal_tokens, device="cuda", dtype=torch.float32)
    return reduce_report(total)


def add_terms(total, terms):
    """total + terms, key by key; both must carry the same keys (a key in only one is a bug, not a term to drop)."""
    if total is None:
        return dict(terms)
    if total.keys() != terms.keys():
        raise RuntimeError(f"term keys differ: {sorted(total.keys() ^ terms.keys())}")
    return {k: total[k] + v for k, v in terms.items()}


@torch.no_grad()
def evaluate_opd(model, jobs, objective, micro_anchors, rank, world):
    """--opd-heldout (opd_assemble.py --heldout jobs: held-out turns another drafter rolled out, its q and candidates
    recorded): this drafter at every recorded cycle, judged on the rollout's paths -- fast_train.opd_terms with the
    sampler's record, so objective.importance_terms estimates its own serving acceptance there (opd_heldout_metrics).
    Each rank takes jobs[rank::world]; clean passes per segment, as training runs them."""
    import fast_train
    from pool import segments

    b = model.config.block_size
    overlap = fast_train.receptive_field(model.config)
    total = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for directory in jobs[rank::world]:
            features = {k: v.cuda() for k, v in load_file(str(directory / "features.safetensors")).items()}
            cycles = {k: v.cuda() for k, v in load_file(str(directory / "cycles.safetensors")).items()}
            ids, gr = features["input_ids"], features["target_gr"]
            if not torch.equal(features["positions"], torch.arange(len(ids), device="cuda")):
                raise ValueError(f"{directory}: position mismatch")
            anchors = cycles["anchor"]
            for lo, hi in segments(anchors.cpu().numpy(), 0, len(anchors), overlap):
                segment = anchors[lo:hi]
                trace = fast_train.clean_pass(model, ids, gr, segment, fast_train.clean_start(segment, model.config))
                for start in range(lo, hi, micro_anchors):
                    rows = slice(start, min(start + micro_anchors, hi))
                    hidden = fast_train.propose(model, trace, ids, gr, anchors[rows])
                    terms = fast_train.opd_terms(model, hidden, cycles["drafted"][rows], ids[anchors[rows]],
                                                 cycles["verify_gr"][rows, : b - 1], cycles["accepted"][rows],
                                                 objective.temperature, objective.joint, sampler_q=cycles["q"][rows],
                                                 sampler_candidates=cycles["candidates"][rows])
                    terms = {k: v.detach().float() for k, v in terms.items()}
                    total = terms if total is None else {k: total[k] + v for k, v in terms.items()}
                del trace
            del features, cycles
    if total is None:  # a rank without held-out cycles
        total = zero_opd_terms(b, prefix="")
    return reduce_report(total)


def pool_loop(args, model, optimizer, rank, world, run, start_step, consumed, anchors_done):
    """--pool: jobs streamed from capture queue roots; rank 0 plans every step so all ranks do the same work
    (pool.Planner). Each rank keeps the features of the jobs it is on until their anchors run out, and runs each step's
    clean passes afresh under autograd; rank 0 deletes a job's directory once its rank has loaded it, which frees room
    for capture under its --max-queue-gib. Runs until every producer is done and the pool is drained (a smoke/soak: or
    --steps).

    Production (not --smoke): the learning rate follows lr_factor over the trained fraction of the plan's anchors
    (--total-anchors; anchors_done carries over a resume);
    held-out evaluation at the start, every --eval-every steps and at the end (eval.jsonl); a checkpoint every
    --save-every steps, on STOP and at the end. Resume protocol: rank 0 lists every job it has finished training
    ({"key", "pass", "step"} from its record.json) in `consumed`, and each checkpoint carries the whole list as
    consumed.jsonl. A resumed run starts from an empty pool; its producers skip the checkpoint's consumed jobs
    (capture_plan.py --skip-consumed) and capture everything else again -- jobs that were in flight at the checkpoint
    are trained again from their first anchor.

    Per step, beyond train.py's fields: the anchors and the tokens of newly started jobs per rank, how long rank 0
    waited for capture (starved), the capture time its released jobs spent blocked on the queue bound (the trainer was
    the bottleneck then), and running throughput.
    """
    import fast_train
    from pool import ANCHOR_SECONDS, Planner, clean_row_seconds, clean_rows, segments

    warmup_began = time.monotonic()
    warm_up(model, args.micro_anchors, world, args.objective, opd_eval=args.opd_heldout is not None,
            robust=args.robust_config, gate=args.prefix_gate)
    print(f"[rank{rank}] warm-up done in {time.monotonic() - warmup_began:.0f} s", flush=True)
    heldout = []
    for root in args.heldout or []:
        if not (root / "heldout.producer_done").is_file():
            raise ValueError(f"{root}: held-out capture has not finished")
        heldout += sorted(d for d in root.glob("heldout-*") if (d / "READY").is_file())
    if args.heldout is not None and not heldout:
        raise ValueError(f"{args.heldout} hold no ready held-out jobs")
    heldout.sort(key=lambda d: d.name)
    opd_heldout = []
    for root in args.opd_heldout or []:
        if not (root / "heldout.producer_done").is_file():
            raise ValueError(f"{root}: held-out OPD assembly has not finished (opd_assemble.py --heldout --done)")
        for directory in sorted(d for d in root.glob("heldout-*") if (d / "READY").is_file()):
            if json.loads((directory / "record.json").read_text())["split"] != "heldout":
                raise ValueError(f"{directory}: not a held-out job")
            opd_heldout.append(directory)
    if args.opd_heldout is not None and not opd_heldout:
        raise ValueError(f"{args.opd_heldout} hold no ready held-out OPD jobs")

    # --eval-by-corpus (plan-captured held-out: every record names its corpus): one report per corpus, and their sum
    # (the terms are additive) as the overall one.
    groups = {"all": heldout}
    if args.eval_by_corpus:
        groups = {}
        for directory in heldout:
            groups.setdefault(json.loads((directory / "record.json").read_text())["corpus"], []).append(directory)

    def run_eval(at_step):
        began_eval = time.monotonic()
        reports = {name: evaluate(model, jobs, args.objective, args.micro_anchors, args.eval_anchors, rank, world)
                   for name, jobs in sorted(groups.items())}
        total = None
        for report in reports.values():
            total = add_terms(total, report)
        opd_report = (evaluate_opd(model, opd_heldout, args.objective, args.micro_anchors, rank, world)
                      if opd_heldout else None)
        if rank == 0:
            seconds = time.monotonic() - began_eval
            records = {"": {"step": at_step, "seconds": seconds, "jobs": len(heldout), **accept_metrics(total)}}
            if args.eval_by_corpus:
                records.update({f"{name}/": {"step": at_step, "jobs": len(groups[name]), **accept_metrics(report)}
                                for name, report in reports.items()})
                # "all" sums the corpora's blocks, so the corpus with the most held-out anchors dominates it; the
                # macro average weights every corpus alike.
                corpora = [records[f"{name}/"] for name in reports]
                records["macro/"] = {"step": at_step, **{
                    k: sum(r[k] for r in corpora) / len(corpora)
                    for k in ("expected_accept_length", "renewal_accept_length", "coverage_kl", "loss_accept")
                    if all(k in r for r in corpora)}}
            if opd_report is not None:
                records["opd_heldout/"] = {"step": at_step, "jobs": len(opd_heldout), **opd_heldout_metrics(opd_report)}
                print(json.dumps({"eval_opd_heldout": records["opd_heldout/"]}), flush=True)
            with (args.out / "eval.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": at_step, "seconds": seconds,
                                         "groups": {k.rstrip("/") or "all": v for k, v in records.items()}}) + "\n")
            print(json.dumps({"eval": records[""]}), flush=True)
            if run is not None:
                flat = {}
                for prefix, record in records.items():
                    flat.update({f"eval/{prefix}{k}": v for k, v in record.items()
                                 if isinstance(v, (int, float)) and k != "step"})
                    if prefix == "opd_heldout/":
                        flat.update({f"eval/opd_heldout/{name}/pos{j + 1}": v
                                     for name in ("survival_position", "sampler_covered_position")
                                     for j, v in enumerate(record[name])})
                    if prefix in ("macro/", "opd_heldout/"):
                        continue
                    series = position_series(record)
                    series.update({f"match_position/pos{j + 1}": v
                                   for j, v in enumerate(record["match_rate_position"])})
                    series.update({f"recall16/pos{j + 1}": v for j, v in enumerate(record["recall_at_k_position"])})
                    flat.update({f"eval/{prefix}{k}": v for k, v in series.items()})
                run.log(flat, step=at_step)

    production = not args.smoke
    # Checkpoints carry the consumed-job list; a --checkpoint-smoke run exercises exactly that (probes/
    # pool_resume_check.py), so it tracks it too.
    tracking = production or args.checkpoint_smoke

    def checkpoint_now(at_step):
        if tracking:
            listing = "".join(json.dumps(entry) + "\n" for entry in consumed) if rank == 0 else None
            extra_files, extra_state = {"consumed.jsonl": listing}, {"anchors_done": anchors_done}
            if args.robust_config is not None:
                from robust_transition import checkpoint_state

                extra_state |= checkpoint_state(args.robust_spec, args.robust_digest)
                extra_files["objective-transition.json"] = args.robust_transition.read_text() if rank == 0 else None
            save_state(model, optimizer, args.out, at_step, rank, world, args.checkpoint, args.dataset_manifest, True,
                       extra_files=extra_files, extra_state=extra_state)

    if heldout:
        run_eval(start_step)
    overlap = fast_train.receptive_field(model.config)
    row_seconds = clean_row_seconds(model.config)
    planner = (Planner(args.pool, world, args.step_anchors, overlap, stop_file=args.out / "STOP",
                       jobs_per_rank=args.jobs_per_rank, max_rank_tokens=args.max_rank_tokens,
                       opd_fraction=args.opd_fraction, opd_jobs_per_rank=args.opd_jobs_per_rank,
                       row_seconds=row_seconds)
               if rank == 0 else None)
    prefetch = FeaturePrefetch()
    jobs = {}  # path -> (ids, gr, mask, ascending anchors, the same on the host, recorded cycles) of this rank's jobs
    began = time.monotonic()
    totals = {"anchors": 0, "job_tokens": 0, "jobs": 0, "capture_seconds": 0.0, "capture_queue_wait": 0.0}
    by_corpus = {"tokens": {}, "jobs": {}}  # this run's finished jobs per corpus (plan-captured records carry it)
    records, release = {}, []  # rank 0: started jobs' records until they finish; directories to delete
    completed, plan, stop_requested = start_step, None, False
    robust_mix = torch.zeros((), device="cuda") if args.robust_config is not None else None
    job_corpora = {}
    step = start_step
    # A smoke/soak stops at --steps. Production runs until the plan is drained (PR #6 F3); its schedule is the plan's
    # anchor total, which it cannot outrun by more than the jobs in flight at a resume. --step-schedule: production
    # that runs --steps steps on that schedule (an A/B at equal steps whose arms mix their units differently).
    anchor_schedule = production and not args.step_schedule
    while anchor_schedule or step < args.steps:
        if anchor_schedule and anchors_done > args.total_anchors * 1.01:
            raise RuntimeError(f"{anchors_done} anchors trained, the plan has {args.total_anchors}: --total-anchors is off")
        plan_began = time.monotonic()
        message = [planner.plan() if rank == 0 else None]
        dist.broadcast_object_list(message, src=0)
        plan = message[0]
        plan_seconds = time.monotonic() - plan_began
        if plan is None:
            break
        if "stop" in plan:  # STOP while rank 0 waited for capture (a producer died, or a manual stop): save and exit
            stop_requested = True
            if completed > start_step:
                checkpoint_now(completed)
            break
        if rank == 0:
            # A started job is loaded by its rank within this iteration; its directory then only holds queue space
            # (with many jobs per rank, in-progress jobs would fill the producers' bound and deadlock them against a
            # planner waiting for jobs). Keep its record and delete it at the next barrier. Resume is unchanged: a
            # job counts as consumed only when finished, so an in-flight one is captured again.
            for path in (u.path for per_rank in plan["units"] for u in per_rank if u.start):
                records[path] = json.loads((Path(path) / "record.json").read_text())
                release.append(path)
        for path in plan["reserve"][rank]:
            prefetch.request(Path(path))
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        marks[0].record()
        units, new_tokens, load_wait, rows_here = [], 0, 0.0, 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for unit in plan["units"][rank]:
                if unit.start:
                    features, metadata, _, blocked = prefetch.take(Path(unit.path))
                    load_wait += blocked
                    if metadata["split"] != "train":
                        raise ValueError("held-out features cannot enter training")
                    ids, gr, mask = features["input_ids"], features["target_gr"], features["loss_mask"]
                    if not torch.equal(features["positions"], torch.arange(len(ids), device="cuda")):
                        raise ValueError("capture position mismatch")
                    if "cycle_anchor" in features:  # OPD: the units are the recorded cycles, anchors ascending
                        cycles = {k.removeprefix("cycle_"): v for k, v in features.items() if k.startswith("cycle_")}
                        anchors = cycles["anchor"]
                    else:
                        cycles, anchors = None, eligible_anchors(mask)
                    # The planner's positions are the same, read from the same files.
                    jobs[unit.path] = (ids, gr, mask, anchors, anchors.cpu().numpy(), cycles)
                    if args.robust_config is not None or args.prefix_gate is not None:
                        job_corpora[unit.path] = metadata["corpus"]
                    new_tokens += len(ids)
                ids, gr, mask, anchors, positions, cycles = jobs[unit.path]
                if unit.hi > len(anchors) or (unit.last and unit.hi != len(anchors)):
                    raise RuntimeError(f"{unit.path}: plan [{unit.lo}, {unit.hi}) vs {len(anchors)} anchors")
                if unit.hi == unit.lo:
                    continue
                rows_here += clean_rows(positions, unit.lo, unit.hi, overlap)
                # one unit per segment: anchors more than the receptive field apart get separate clean passes
                for lo, hi in segments(positions, unit.lo, unit.hi, overlap):
                    span = slice(lo, hi)
                    if cycles is None:
                        units.append(Segment(ids, gr, mask, anchors[span], job_corpora.get(unit.path, "other")))
                    else:
                        units.append(OpdUnit(ids, gr, anchors[span], cycles["drafted"][span], cycles["accepted"][span],
                                             cycles["verify_gr"][span, : model.config.block_size - 1],
                                             cycles["q"][span], cycles["candidates"][span]))
            marks[1].record()
            if not any(u.hi > u.lo for per_rank in plan["units"] for u in per_rank):
                # Every rank only started jobs whose clean pass took its whole budget (sequences past ~200k tokens):
                # preprocessing, not an optimizer step -- no update and no schedule step (PR #6 F2).
                if plan["finished"]:
                    raise RuntimeError("a plan without anchors cannot finish a job")
                continue
            if robust_mix is not None:
                from robust_transition import mix_at

                robust_mix.fill_(mix_at(args.robust_spec, anchors_done))
            # the step's start: the coverage weight is in the loss, so it cannot wait for this step's anchors
            progress_before = (anchors_done / args.total_anchors if anchor_schedule
                               else step / args.steps if production else 0.0)
            cov_weight = (coverage_weight_at(args, progress_before) if args.objective.name == "acceptance" else None)
            terms, step_marks, timing = fast_step(model, units, args.micro_anchors, world, args.objective,
                                                  args.robust_config, robust_mix, gate=args.prefix_gate,
                                                  cov_weight=cov_weight)
            if args.prefix_gate is not None:
                # Trust floor 0: a step whose every supervised slot weighs 0 would still move Adam's moments. With tens
                # of thousands of anchors a step it cannot happen short of a broken gate, so it stops the run.
                weight = terms["gate_weight_sum"].sum().double()
                dist.all_reduce(weight)
                if not weight.item() > 0:
                    raise RuntimeError(f"prefix gate: the step's total weight is {weight.item()} on every rank")
        marks += step_marks
        grad_norm = sync_clip_replicated(model, world)
        marks.append(torch.cuda.Event(enable_timing=True))
        marks[-1].record()
        # The anchors every rank trains this step (the plan hands them out, so rank 0 knows them for all).
        step_anchors = torch.tensor(sum(unit_anchors(u) for u in units), device="cuda", dtype=torch.float64)
        dist.all_reduce(step_anchors)
        anchors_done += int(step_anchors.item())
        # A smoke/soak keeps a constant rate: its short budget has nothing to do with a schedule.
        factor = (schedule_factor(args, anchors_done / args.total_anchors) if anchor_schedule
                  else schedule_factor(args, (step + 1) / args.steps) if production else 1.0)
        if args.rewarm_steps:  # a warm start's fresh Adam moments: ramp back up to the schedule
            factor *= min(1.0, (step + 1) / args.rewarm_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        before = None if production else model.fc_hidden.weight.detach().clone()
        # Every 100 steps rank 0 measures each group's weight norm and the step's relative update (the parameters are
        # replicated, so its copy is everyone's).
        watch = rank == 0 and (step + 1) % 100 == 0
        snapshot = [[q.detach().clone() for q in group["params"]] for group in optimizer.param_groups] if watch else None
        optimizer.step()
        marks.append(torch.cuda.Event(enable_timing=True))
        marks[-1].record()
        if before is not None and torch.equal(before, model.fc_hidden.weight):
            raise RuntimeError("FP32 master did not update")
        weights = {}
        if watch:
            for group, old in zip(optimizer.param_groups, snapshot, strict=True):
                norm = torch.stack(torch._foreach_norm(group["params"])).norm()
                delta = torch.stack(torch._foreach_norm(torch._foreach_sub(group["params"], old))).norm()
                weights[f"param_norm/{group['name']}"] = norm.item()
                weights[f"update_ratio/{group['name']}"] = (delta / norm).item()
            del snapshot
        report = reduce_report(terms)
        if args.prefix_gate is not None and report["gate_nonfinite"].item():
            raise RuntimeError(f"prefix gate: {int(report['gate_nonfinite'].item())} non-finite tails this step")
        work = torch.tensor([sum(unit_anchors(u) for u in units), new_tokens, rows_here], device="cuda",
                            dtype=torch.float64)
        gathered = [torch.empty_like(work) for _ in range(world)]
        dist.all_gather(gathered, work)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        for unit in plan["units"][rank]:
            if unit.last:
                del jobs[unit.path]
        dist.barrier()  # every rank has loaded this step's (and any skipped plan's) started jobs
        if rank == 0:
            for path in release:
                shutil.rmtree(path)
            release.clear()
            for path in plan["finished"]:
                record = records.pop(path)
                totals["capture_seconds"] += record["seconds"]
                totals["capture_queue_wait"] += record["queue_wait_seconds"]
                totals["job_tokens"] += record["num_tokens"]
                totals["jobs"] += 1
                if tracking:
                    consumed.append({"key": record["key"], "pass": record["pass"], "step": step + 1})
                    corpus = record["corpus"]
                    by_corpus["tokens"][corpus] = by_corpus["tokens"].get(corpus, 0) + record["num_tokens"]
                    by_corpus["jobs"][corpus] = by_corpus["jobs"].get(corpus, 0) + 1
            per_rank = [g.tolist() for g in gathered]
            costs = [a * ANCHOR_SECONDS + r * row_seconds for a, _, r in per_rank]
            totals["anchors"] += sum(a for a, _, _ in per_rank)
            wall = time.monotonic() - began
            # prepare: the acceptance objective's count all-reduce (the wait for the slowest rank) or D-PACE's first
            # pass; units: the clean passes and proposals, forward and backward
            stages = ("load", "handoff", "prepare", "units", "clip_sync", "optimizer")
            record = {
                "step": step + 1,
                "seconds": elapsed,
                "plan_seconds": plan_seconds,  # includes rank 0 waiting for capture
                "starved_seconds_total": planner.starved_seconds,
                "load_wait_seconds": load_wait,
                "anchors_per_rank": [int(a) for a, _, _ in per_rank],
                "new_job_tokens_per_rank": [int(t) for _, t, _ in per_rank],
                # each unit's clean pass: its anchors' span plus the receptive field before it
                "clean_rows_per_rank": [int(r) for _, _, r in per_rank],
                "clean_forward_ms": sum(a.elapsed_time(b) for a, b in timing["clean_forward"]),
                "clean_backward_ms": sum(a.elapsed_time(b) for a, b in timing["clean_backward"]),
                # planned cost of the busiest rank over the mean: 1.0 is a perfectly balanced step
                "planned_imbalance": max(costs) / max(sum(costs) / world, 1e-9),
                "stage_gpu_ms": dict(zip(stages, (a.elapsed_time(b) for a, b in zip(marks, marks[1:])), strict=True)),
                "gradient_norm": grad_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "consumed_tokens_by_corpus": dict(by_corpus["tokens"]),
                "consumed_jobs_by_corpus": dict(by_corpus["jobs"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "pending_jobs": planner.pending_jobs(),
                "released_jobs_total": totals["jobs"],
                "released_empty_jobs_total": planner.released_empty,
                "anchors_per_second": totals["anchors"] / wall,
                "released_tokens_per_second": totals["job_tokens"] / wall,
                "capture_queue_wait_seconds_total": totals["capture_queue_wait"],
                # what the capture servers achieve while serving (the trainer may be what holds them back)
                "capture_tokens_per_busy_second": totals["job_tokens"] / max(totals["capture_seconds"], 1e-9),
                "step_anchors_per_second": sum(a for a, _, _ in per_rank) / elapsed,
                "anchors_done": anchors_done,
                "progress": anchors_done / args.total_anchors if anchor_schedule else (step + 1) / args.steps,
                **weights,
                **accept_metrics(report),
            }
            if model.config.attention_sink and (step == 0 or (step + 1) % 100 == 0):
                record["sink_logits"] = {
                    f"layer{i}_head{h}": value for i, layer in enumerate(model.layers)
                    for h, value in enumerate(layer.self_attn.sink_logits.detach().float().cpu().tolist())}
            if args.objective.name == "acceptance":  # the optimized scalar
                o = args.objective
                record["cov_weight"] = cov_weight
                record["loss"] = o.accept_weight * record["loss_accept"] + cov_weight * record["coverage_kl"]
                if args.prefix_gate is not None:
                    record["loss_optimization"] = (o.accept_weight * record["gate_accept_optimization"]
                                                   + cov_weight * record["gate_coverage_kl_optimization"])
                    record["gradient_clipped"] = bool(grad_norm > 1.0)
                if args.robust_config is not None:
                    record["robust_mix"] = float(robust_mix.item())
                    record["loss_optimization"] = (o.accept_weight * record["robust_accept_optimization"]
                                                   + cov_weight * record["coverage_kl"])
                    record["gradient_clipped"] = bool(grad_norm > 1.0)
                if o.opd_weight and record["opd_cycles"]:
                    key = {"reached": "opd_reached_tv", "wtv": "opd_path_wtv", "importance": "opd_importance_loss"}[
                        o.opd_loss]
                    record["loss"] += o.opd_weight * record[key]
            write_record(args.out, record, run)
        completed = step = step + 1
        stop_flag = torch.tensor(int(rank == 0 and (args.out / "STOP").exists()), device="cuda", dtype=torch.int32)
        dist.broadcast(stop_flag, src=0)
        stop_requested = bool(stop_flag.item())
        if heldout and completed % args.eval_every == 0:
            run_eval(completed)
        if completed % args.save_every == 0 or stop_requested:
            checkpoint_now(completed)
        if stop_requested:
            break
    if not stop_requested and completed > start_step:
        if heldout and completed % args.eval_every:
            run_eval(completed)
        if completed % args.save_every:
            checkpoint_now(completed)
    if rank == 0:
        result = {"steps": completed, "start_step": start_step, "drained": plan is None, "stopped": stop_requested,
                  "seconds": time.monotonic() - began, **totals, "starved_seconds": planner.starved_seconds,
                  "released_empty_jobs": planner.released_empty}
        temp = args.out / "run-result.partial"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out / "run-result.json")
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--queue", type=Path, help="one capture queue read in epoch order (or --pool)")
    p.add_argument("--pool", type=Path, help="fast engine: capture queue roots streamed with balanced steps (pool.py)")
    p.add_argument("--step-anchors", type=int, default=16384, help="--pool: anchors per step over all ranks")
    p.add_argument("--jobs-per-rank", type=int, default=1,
                   help="--pool: sequences each rank works on at once, the step's anchors spread over them (pool.py)")
    p.add_argument("--max-rank-tokens", type=int,
                   help="--pool: cap on the tokens of a rank's jobs in progress (their features stay on the GPU, ~20 KB "
                   "a token); a lone job may exceed it")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="multiplies every group's learning rate (copied 1e-4, new 3e-4 x --new-lr-mult / 3, reducer "
                   "--reducer-lr)")
    p.add_argument("--new-lr-mult", type=float, default=3.0,
                   help="the new group's (convs, selector, mask embedding) rate over the copied group's: 3 when they "
                   "start untrained (init_mtp3), 1 when the init carries them trained (an MTP export)")
    p.add_argument("--reducer-lr", type=float, default=3e-5,
                   help="the drafter's output reducer's (hyper_connection_mixer) rate before --lr-scale: 3e-5 keeps a "
                   "copied native reducer close to its init; a random init has nothing to keep (1e-4, the copied "
                   "group's)")
    p.add_argument("--total-anchors", type=int,
                   help="--pool production: the plan's trained anchors over all passes (its summary); the learning "
                   "rate's schedule runs over them")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dataset-manifest", required=True, type=Path)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--total-rows", type=int, help="--queue: rows in the epoch")
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--objective-chunk", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--checkpoint-smoke", action="store_true")
    p.add_argument("--fp32-smoke", action="store_true", help="diagnostic only; requires --smoke")
    p.add_argument("--gate-report", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--robust-transition", type=Path,
                   help="prepared robust-v2 objective-transition.json; requires full-state --resume")
    p.add_argument("--prefix-gate-alpha", type=float,
                   help="acceptance + --pool: prefix gate v3 (prefix_gate.py) with this tail threshold, on from the "
                   "first step; with --prefix-gate-temperature")
    p.add_argument("--prefix-gate-temperature", type=float,
                   help="--prefix-gate-alpha: the temperature of the target distribution the tails are taken from")
    p.add_argument("--lr-schedule", choices=("cosine", "wsd"), default="cosine",
                   help="--pool production: cosine (3%% warm-up, cosine to 0) or wsd (3%% warm-up, the peak, then "
                   "linear to 0 over the last --lr-decay-fraction)")
    p.add_argument("--lr-decay-fraction", type=float, help="--lr-schedule wsd: the share of the run that decays")
    p.add_argument("--cov-weight-final", type=float,
                   help="acceptance + --pool: the coverage weight goes linearly from --cov-weight to this over the "
                   "run's progress (production; a smoke keeps --cov-weight)")
    p.add_argument("--warm-start", type=Path,
                   help="--pool production: a pool checkpoint (another run's) whose trained parameters start this run, "
                   "with a fresh optimizer; its consumed jobs count as done (the producers skip them, the schedule starts "
                   "after their anchors). --checkpoint still gives the config and the frozen tensors")
    p.add_argument("--rewarm-steps", type=int, default=0,
                   help="--warm-start: the learning rate ramps up linearly over this run's first steps (fresh Adam moments)")
    p.add_argument("--step-schedule", action="store_true",
                   help="--pool production: run --steps steps with the learning rate's schedule over them, not over "
                   "--total-anchors (arms of an A/B that spend their steps on different units train equally long)")
    p.add_argument("--warm-start-new-data", action="store_true",
                   help="--warm-start takes the parameters only: this run's data is its own (e.g. another pass over the "
                   "jobs the warm start trained, plus OPD cycles), nothing counts as done and the schedule starts at 0")
    p.add_argument("--attention-backend", choices=("sdpa", "fa4", "stable"), default="sdpa")
    p.add_argument(
        "--engine",
        choices=("reference", "fast"),
        default="reference",
        help="fast: fast_train.py (FlexAttention, compiled layers, the head's gradient taken in the forward pass) "
        "on replicated parameters; --smoke only until it has its own correctness gates",
    )
    p.add_argument(
        "--residual-dtype",
        choices=("float32", "bfloat16"),
        help="fast engine: the residual stream's storage (fast_train.RESIDUAL_DTYPE). Default bfloat16, rounded like "
        "serving's all-BF16 drafter: the 2026-10-06 A/B (probes/residual_ab.py) trained the same as float32 in 13 GiB "
        "less; float32 keeps autocast's promotion",
    )
    p.add_argument(
        "--micro-anchors",
        type=int,
        default=4096,
        help="fast engine: anchors per forward/backward. More anchors on a rank than this take two passes "
        "(fast_step); 4096 peaks near 75 GiB on a 92k-token sequence",
    )
    p.add_argument(
        "--objective",
        choices=("dpace", "acceptance"),
        default="dpace",
        help="dpace: hard CE with D-PACE weights + covered-label selector CE + early KL, at T = 1 (objective.objective). "
        "acceptance (fast engine): WTV on the selector's actual proposal + --cov-weight x the coverage KL on every "
        "slot, at --temperature (objective.acceptance_objective)",
    )
    p.add_argument("--temperature", type=float, help="acceptance: the deployment sampling temperature")
    p.add_argument("--cov-weight", type=float, help="acceptance: weight of the coverage KL against the WTV term")
    p.add_argument(
        "--selector-grad",
        choices=("joint", "selector"),
        help="acceptance: joint lets the WTV term into the backbone (candidate logits, selector hidden projection)",
    )
    p.add_argument("--opd-weight", type=float, default=0.0,
                   help="acceptance + --pool: weight of the OPD term on recorded verify cycles (opd_assemble.py jobs)")
    p.add_argument("--opd-loss", choices=("reached", "wtv", "importance"), default="reached",
                   help="the OPD term: reached-state TV (accepted drafts + the first rejection), the path WTV, or "
                   "minus the importance estimate of the expected accepted drafts (objective.importance_terms)")
    p.add_argument("--opd-fraction", type=float,
                   help="--opd-weight: the share of every step's budget the OPD jobs get (pool.Planner), so the cycles "
                   "spread over the run at a fixed share; 1: OPD only, the captured jobs get what the OPD ones leave")
    p.add_argument("--opd-jobs-per-rank", type=int, help="--opd-fraction: OPD jobs a rank works on at once")
    p.add_argument("--offline-accept-weight", type=float, default=1.0,
                   help="acceptance: weight of the captured anchors' WTV term; 0 with --opd-weight is the design doc's "
                   "post-training form (acceptance from the replayed cycles, coverage from the offline anchors)")
    p.add_argument("--heldout", type=Path, nargs="+",
                   help="--pool: capture queues of heldout-* jobs evaluated during training")
    p.add_argument("--eval-every", type=int, default=2000, help="--pool: steps between held-out evaluations")
    p.add_argument("--eval-anchors", type=int, default=4096, help="--pool: held-out anchors per job (fixed subsample)")
    p.add_argument("--opd-heldout", type=Path, nargs="+",
                   help="with --heldout: opd_assemble.py --heldout roots; every eval adds the importance estimate of "
                   "this drafter's serving acceptance on those held-out rollouts (eval.jsonl group opd_heldout)")
    p.add_argument("--eval-by-corpus", action="store_true",
                   help="--pool: also evaluate each corpus of the held-out records (capture_plan.py records name it)")
    p.add_argument("--profile-step", type=int, default=0)
    p.add_argument("--wandb-project", default="qwen38fn-dflash")
    p.add_argument("--wandb-name")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-extra", type=Path, help="a JSON object recorded in the wandb config as `launch` (the "
                   "launcher's data plan summary, init and gate digests, GPU split, git commit)")
    args = p.parse_args()
    args.prefix_gate = None
    if (args.prefix_gate_alpha is None) != (args.prefix_gate_temperature is None):
        raise ValueError("--prefix-gate-alpha and --prefix-gate-temperature go together")
    if args.prefix_gate_alpha is not None:
        from prefix_gate import PrefixGateConfig

        if args.pool is None or args.robust_transition is not None or args.opd_weight:
            raise ValueError("the prefix gate trains a pool's offline data, without the robust objective or OPD")
        args.prefix_gate = PrefixGateConfig(args.prefix_gate_alpha, args.prefix_gate_temperature)
    if (args.lr_schedule == "wsd") != (args.lr_decay_fraction is not None):
        raise ValueError("--lr-decay-fraction goes with --lr-schedule wsd")
    if args.lr_schedule == "wsd":
        wsd_lr_factor(0.5, args.lr_decay_fraction)  # validates the fraction
    if (args.lr_schedule != "cosine" or args.cov_weight_final is not None) and args.pool is None:
        raise ValueError("--lr-schedule wsd and --cov-weight-final belong to --pool")
    if args.cov_weight_final is not None and not args.cov_weight_final >= 0:
        raise ValueError(f"--cov-weight-final {args.cov_weight_final} must be nonnegative")
    args.robust_config, args.robust_spec, args.robust_digest = None, None, None
    if args.robust_transition is not None:
        from robust_objective import RobustConfig
        from robust_transition import load_spec

        if args.resume is None or args.pool is None or args.warm_start is not None or args.smoke:
            raise ValueError("robust transition requires production pool resume, not warm-start or smoke")
        args.robust_spec, args.robust_digest = load_spec(args.robust_transition)
        args.robust_config = RobustConfig(**args.robust_spec["robust_objective"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    # Allocation guard on a shared box: exactly one visible card per rank. The
    # launcher chooses the cards and verifies they are free; this catches the
    # two disagreeing, which would land two ranks on one GPU.
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != world:
        raise ValueError(f"trainer sees {len(visible)} GPUs ({visible}) but runs {world} ranks")
    # The rank count is a launch decision and not a reversible one: every
    # checkpoint records its own world size and `--resume` refuses to load under
    # a different one, so changing it means restarting from the initialization.
    # Deeper drafters need more ranks to hold FP32 masters and Adam moments, but
    # each rank consumes its own sequence, so the capture side has to keep up.
    if (args.queue is None) == (args.pool is None):
        raise ValueError("give exactly one of --queue and --pool")
    if args.pool is not None:
        # The pool's length is the data plan's (pool.planned_steps gives --steps); it resumes through the consumed-job
        # list its checkpoints carry (pool_loop).
        if args.engine != "fast" or args.total_rows is not None:
            raise ValueError("--pool runs the fast engine and takes its length from --steps, not --total-rows")
        if args.eval_every <= 0 or args.eval_anchors <= 0 or args.jobs_per_rank < 1:
            raise ValueError("--eval-every, --eval-anchors and --jobs-per-rank must be positive")
        if args.resume is not None and args.smoke and not args.checkpoint_smoke:
            raise ValueError("a pool smoke resumes only as --checkpoint-smoke (probes/pool_resume_check.py)")
        if args.step_schedule and args.smoke:
            raise ValueError("--step-schedule is a production schedule; a smoke keeps a constant rate")
        if (args.smoke or args.step_schedule) != (args.total_anchors is None):
            raise ValueError("--pool production needs --total-anchors (the plan's), or --step-schedule; a smoke runs "
                             "--steps and takes none")
        if args.total_anchors is not None and args.total_anchors <= 0:
            raise ValueError("--total-anchors must be positive")
        if args.warm_start is not None and (args.smoke and not args.checkpoint_smoke):
            raise ValueError("--warm-start belongs to production or a --checkpoint-smoke")
        if (args.rewarm_steps > 0) != (args.warm_start is not None) or args.rewarm_steps < 0:
            raise ValueError("--rewarm-steps (positive) goes with --warm-start")
        if args.warm_start_new_data and args.warm_start is None:
            raise ValueError("--warm-start-new-data qualifies --warm-start")
        if args.opd_fraction is not None and not args.opd_weight:
            raise ValueError("--opd-fraction goes with --opd-weight")
        if (args.opd_fraction is None) != (args.opd_jobs_per_rank is None):
            raise ValueError("--opd-fraction and --opd-jobs-per-rank go together")
        # Without a share the OPD jobs, ready at the start, sit in one queue with the captured ones (pool.Planner)
        if args.opd_weight and not args.smoke and args.opd_fraction is None:
            raise ValueError("production OPD needs --opd-fraction")
        if args.opd_heldout is not None and args.heldout is None:
            raise ValueError("--opd-heldout evaluates beside --heldout")
    elif (args.heldout is not None or args.jobs_per_rank != 1 or args.max_rank_tokens is not None
          or args.warm_start is not None or args.warm_start_new_data or args.opd_fraction is not None
          or args.opd_jobs_per_rank is not None or args.opd_heldout is not None):
        raise ValueError("--heldout, --opd-heldout, --jobs-per-rank, --max-rank-tokens, --warm-start(-new-data) and "
                         "--opd-fraction belong to --pool")
    elif args.total_rows is None or args.total_rows <= 0:
        raise ValueError("--queue needs a positive --total-rows")
    if not args.lr_scale > 0 or not args.new_lr_mult > 0 or not args.reducer_lr > 0:
        raise ValueError("--lr-scale, --new-lr-mult and --reducer-lr must be positive")
    if world < 1 or args.steps <= 0:
        raise ValueError("requires a positive budget and at least one FSDP2 rank")
    if args.fp32_smoke and not args.smoke:
        raise ValueError("the supplied plan fixes BF16; FP32 is currently a correctness diagnostic only")
    fast = args.engine == "fast"
    if fast:
        # gates.py binds the stable reference path and the queue epoch; the fast engine's production gate is
        # fast_gates.py, for the pool.
        if not args.smoke and args.pool is None:
            raise RuntimeError("the fast engine runs production through --pool (fast_gates.py); --smoke otherwise")
        if args.fp32_smoke or args.attention_backend != "sdpa":
            raise ValueError("the fast engine is BF16 with FlexAttention; --attention-backend is the reference's")
        if args.micro_anchors <= 0:
            raise ValueError("--micro-anchors must be positive")
        args.residual_dtype = args.residual_dtype or "bfloat16"
        if args.residual_dtype == "bfloat16":
            import fast_train

            fast_train.set_residual_dtype(torch.bfloat16)
    elif args.micro_anchors != 4096 or args.residual_dtype is not None:
        raise ValueError("--micro-anchors and --residual-dtype apply to the fast engine")
    knobs = (args.temperature, args.cov_weight, args.selector_grad)
    if args.objective == "acceptance":
        if not fast:
            raise ValueError("the acceptance objective runs in the fast engine")
        if any(k is None for k in knobs):
            raise ValueError("--objective acceptance needs --temperature, --cov-weight and --selector-grad")
        args.objective = Objective("acceptance", args.temperature, args.cov_weight, args.selector_grad == "joint",
                                   args.opd_weight, args.opd_loss, args.offline_accept_weight)
    else:
        if any(k is not None for k in knobs) or args.opd_weight or args.offline_accept_weight != 1:
            raise ValueError("--temperature, --cov-weight, --selector-grad, --opd-weight and --offline-accept-weight "
                             "belong to the acceptance objective")
        args.objective = DPACE
        if args.prefix_gate is not None or args.cov_weight_final is not None:
            raise ValueError("--prefix-gate-* and --cov-weight-final belong to the acceptance objective")
    if not args.smoke and args.pool is not None:
        if args.gate_report is None:
            raise RuntimeError("all correctness gates must pass before production (fast_gates.py)")
        gate = json.loads(args.gate_report.read_text())
        from supervise import validate_source

        validate_source(Path(__file__).resolve().parent, gate)
        if not gate["production_ready"] or gate["engine"] != "fast" or gate["objective"] != args.objective.name:
            raise RuntimeError("production needs the fast engine's gates for this objective")
        if args.robust_config is not None:
            from robust_objective import VERSION

            if gate.get("robust_version") != VERSION:
                raise RuntimeError("robust training requires fresh GPU parity and full-state transition gates")
        if args.prefix_gate is not None:
            from prefix_gate import VERSION

            if gate["prefix_gate_version"] != VERSION or gate["prefix_gate"] != args.prefix_gate.spec():
                raise RuntimeError(f"the prefix gate {args.prefix_gate.spec()} needs fast_gates.py run on its own pool "
                                   f"resume evidence (the gate report has {gate['prefix_gate']})")
    elif not args.smoke:
        if args.gate_report is None:
            raise RuntimeError("all correctness gates must pass before production")
        gate = json.loads(args.gate_report.read_text())
        from supervise import validate_source

        validate_source(Path(__file__).resolve().parent, gate)
        if (
            not gate["production_ready"]
            or gate["compute_dtype"] != "bfloat16"
            or gate["attention_backend"] != args.attention_backend
        ):
            raise RuntimeError(
                "production needs passing gates for the actual precision/backend, not an FP32 diagnostic"
            )
        if args.steps != math.ceil(args.total_rows / world):
            raise ValueError("production scheduler must span the complete epoch")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    torch.set_num_threads(8)
    torch.manual_seed(20260909)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=args.resume is not None)
    dist.barrier()
    model = load_checkpoint(args.checkpoint, dtype=torch.float32, device="cuda")
    if args.prefix_gate is not None and model.config.attention_sink:
        raise ValueError("the prefix gate's first version trains without the attention sink")
    if fast and model.config.ffn_type not in ("dense_shared", "dense"):
        # The MoE router is why the stable kernels exist: one BF16 rounding difference can change a routed expert
        # and grow through the layers. The fast engine's tolerance-based equivalence holds for the dense drafter.
        raise ValueError("the fast engine is validated for the dense drafter only")
    if args.attention_backend == "fa4":
        sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    for layer in model.layers:
        layer.self_attn.backend = args.attention_backend
    if args.attention_backend == "stable" and not fast:
        if args.fp32_smoke:
            raise ValueError("stable kernels are BF16; use sdpa for an FP32 control")
        from stable_kernels import enable_stable_bf16

        enable_stable_bf16(model)
    if not args.fp32_smoke:
        model.embed_tokens.bfloat16()
        model.lm_head.bfloat16()
        model.target_reducer.bfloat16()
    mesh = init_device_mesh("cuda", (world,))
    # The fast engine calls the layers' submodules directly, around FSDP's forward hooks, so its parameters stay
    # replicated (0.9B: FP32 masters, grads and moments ~14 GiB a rank); sync_clip all-reduces their gradients.
    for layer in [] if fast else model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32 if args.fp32_smoke else torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
    # Small fusion/reducer/selector parameters stay replicated and are explicitly
    # reduced; this also keeps the frozen head available for chunked objectives.
    rates = {"copied": 1e-4, "new": 1e-4 * args.new_lr_mult, "reducer": args.reducer_lr}
    members = {(kind, decay): [] for kind in rates for decay in (True, False)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "conv." in name or name.startswith("candidate_selector.") or name == "mask_embedding":
            kind = "new"
        elif name.startswith("hyper_connection_mixer."):
            kind = "reducer"
        else:
            kind = "copied"
        members[kind, decays(name, parameter)].append(parameter)
    groups = [
        {"params": params, "lr": rates[kind] * args.lr_scale, "base_lr": rates[kind] * args.lr_scale,
         "weight_decay": 0.01 if decay else 0.0, "name": kind if decay else f"{kind}_no_decay"}
        for (kind, decay), params in members.items() if params
    ]
    # The fast engine's parameters are plain replicated tensors: fused AdamW, one kernel over all of them.
    optimizer = torch.optim.AdamW(groups, **({"fused": True} if fast else {"foreach": False}))
    start_step, anchors_done = 0, 0
    if args.resume:
        manifest = json.loads((args.resume / "manifest.json").read_text())
        if manifest["format"] != "torch-zstd-v1":
            raise ValueError("unsupported checkpoint format")
        # Checkpoints from before the fast engine carry no "replicated" key: they are all per-rank FSDP shards.
        replicated = manifest.get("replicated", False)
        if replicated != fast:
            raise ValueError("checkpoint parameter layout (replicated or FSDP shards) does not match the engine")
        saved = load_compressed(args.resume / ("replicated.pt.zst" if replicated else f"rank-{rank}.pt.zst"))
        if args.robust_config is not None:
            from robust_transition import sha256, validate_restore

            validate_restore(saved, args.robust_spec, args.robust_digest)
            if "robust_transition_sha256" not in saved:
                # All ranks validate the result before using state; hash the large file only on rank 0.
                hashes = [None]
                if rank == 0:
                    hashes[0] = {name: sha256(args.resume / name) for name in args.robust_spec["checkpoint_hashes"]}
                dist.broadcast_object_list(hashes, src=0)
                if hashes[0] != args.robust_spec["checkpoint_hashes"]:
                    raise ValueError("resume checkpoint differs from the pinned objective parent")
        elif "robust_transition_sha256" in saved:
            raise ValueError("robust checkpoint requires its original --robust-transition")
        restore_training_state(model, optimizer, saved, world, rank, replicated)
        start_step = saved["step"]
        anchors_done = saved["anchors_done"] if args.pool is not None else 0
        del saved
        if args.pool is None and start_step >= args.steps:
            raise ValueError("checkpoint already completed the requested schedule")
    inherited = []
    if args.warm_start is not None:
        manifest = json.loads((args.warm_start / "manifest.json").read_text())
        if manifest["format"] != "torch-zstd-v1" or not manifest["replicated"]:
            raise ValueError("--warm-start takes a replicated pool checkpoint")
        # The jobs it trained: done for this run too (they count toward the schedule, the producers skip them, and this
        # run's checkpoints carry them on with step 0, before this run) -- unless this run brings its own data
        # (--warm-start-new-data: the parameters only).
        if args.warm_start_new_data:
            anchors_before = 0
        else:
            inherited = [{"key": e["key"], "pass": e["pass"], "step": 0}
                         for e in map(json.loads, (args.warm_start / "consumed.jsonl").read_text().splitlines())]
            rows = {(r["key"], r["pass"]): r
                    for r in map(json.loads, (args.dataset_manifest.parent / "plan.jsonl").open())}
            missing = [e for e in inherited if rows.get((e["key"], e["pass"]), {}).get("split") != "train"]
            if missing:
                raise ValueError(f"{len(missing)} of the warm start's consumed jobs are not train jobs of this plan: "
                                 f"{missing[0]}")
            anchors_before = sum(rows[e["key"], e["pass"]]["eligible_anchors"] for e in inherited)
        if args.resume is None:
            saved = load_compressed(args.warm_start / "replicated.pt.zst")
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            if set(saved["parameters"]) != trainable:
                raise ValueError(f"warm start parameters differ: {sorted(set(saved['parameters']) ^ trainable)[:5]}")
            with torch.no_grad():
                for name, value in saved["parameters"].items():
                    local(model.get_parameter(name)).copy_(value)
            del saved
            anchors_done = anchors_before
    run = None
    if rank == 0 and not args.smoke and not args.no_wandb:
        import wandb

        # Deterministic id from the run directory so every resume continues the
        # same wandb run instead of forking a new one per checkpoint restart.
        run = wandb.init(
            project=args.wandb_project,
            id=hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:16],
            name=args.wandb_name or args.out.parent.name,
            resume="allow",
            config={
                "steps": args.steps,
                "total_rows": args.total_rows,
                "anchors": args.anchors,
                "layers": model.config.num_layers,
                "block_size": model.config.block_size,
                "window": model.config.window,
                "attention_backend": "flex" if fast else args.attention_backend,
                "engine": args.engine,
                "micro_anchors": args.micro_anchors if fast else None,
                "residual_dtype": args.residual_dtype,
                "objective": args.objective.name,
                "temperature": args.objective.temperature,
                "cov_weight": args.objective.cov_weight,
                "joint": args.objective.joint,
                "opd_weight": args.objective.opd_weight,
                "opd_loss": args.objective.opd_loss,
                "step_anchors": args.step_anchors if args.pool is not None else None,
                "jobs_per_rank": args.jobs_per_rank if args.pool is not None else None,
                "max_rank_tokens": args.max_rank_tokens,
                "lr_scale": args.lr_scale,
                "compute_dtype": "float32" if args.fp32_smoke else "bfloat16",
                "gr_alignment": "predecessor",
                "resumed_from_step": start_step,
                "checkpoint": str(args.checkpoint),
                "init_manifest_sha256": hashlib.sha256(
                    (args.checkpoint / "initialization_manifest.json").read_bytes()).hexdigest(),
                "learning_rates": {group["name"]: group["base_lr"] for group in optimizer.param_groups},
                "weight_decay": "0.01; none on 1-D tensors, conv base kernels, selector codebooks, mask embedding",
                "new_lr_mult": args.new_lr_mult,
                "reducer_lr": args.reducer_lr,
                "total_anchors": args.total_anchors,
                "warm_start": str(args.warm_start) if args.warm_start is not None else None,
                "rewarm_steps": args.rewarm_steps,
                "gradient_clip": 1.0,
                "schedule": (("3% linear warm-up, cosine to 0" if args.lr_schedule == "cosine" else
                              f"wsd: 3% linear warm-up, peak, linear to 0 over the last {args.lr_decay_fraction:g}")
                             + (" over --steps" if args.step_schedule else " over the plan's anchors")
                             if args.pool is not None else "epoch cosine"),
                "lr_schedule": args.lr_schedule,
                "lr_decay_fraction": args.lr_decay_fraction,
                "cov_weight_final": args.cov_weight_final,
                "prefix_gate": args.prefix_gate.spec() if args.prefix_gate is not None else None,
                "world_size": world,
                "trainer_gpus": os.environ["CUDA_VISIBLE_DEVICES"],
                "pool": str(args.pool) if args.pool is not None else None,
                "heldout": [str(root) for root in args.heldout] if args.heldout is not None else None,
                "opd_heldout": [str(root) for root in args.opd_heldout] if args.opd_heldout is not None else None,
                "eval_every": args.eval_every,
                "eval_anchors_per_job": args.eval_anchors,
                "save_every": args.save_every,
                "gate_report": str(args.gate_report) if args.gate_report is not None else None,
                "launch": json.loads(args.wandb_extra.read_text()) if args.wandb_extra is not None else None,
            },
        )
    if args.pool is not None:
        # What a resumed pool run must keep: the schedule, the objective, the step's size, the init.
        config = {"steps": args.steps, "total_anchors": args.total_anchors, "step_anchors": args.step_anchors,
                  "micro_anchors": args.micro_anchors, "jobs_per_rank": args.jobs_per_rank,
                  "max_rank_tokens": args.max_rank_tokens, "lr_scale": args.lr_scale, "new_lr_mult": args.new_lr_mult,
                  "reducer_lr": args.reducer_lr,
                  "warm_start": str(args.warm_start.resolve()) if args.warm_start is not None else None,
                  "warm_start_sha256": hashlib.sha256((args.warm_start / "manifest.json").read_bytes()).hexdigest()
                  if args.warm_start is not None else None, "rewarm_steps": args.rewarm_steps,
                  "warm_start_new_data": args.warm_start_new_data, "opd_fraction": args.opd_fraction,
                  "step_schedule": args.step_schedule,
                  "opd_jobs_per_rank": args.opd_jobs_per_rank,
                  "objective": vars(args.objective), "residual_dtype": args.residual_dtype,
                  "checkpoint": str(args.checkpoint.resolve()), "world_size": world, "smoke": args.smoke,
                  # content, not only paths (PR #6): the data plan's summary and the init's manifest
                  "dataset_manifest_sha256": hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest(),
                  "init_manifest_sha256": hashlib.sha256(
                      (args.checkpoint / "initialization_manifest.json").read_bytes()).hexdigest()}
        # Only when set: a run configured before these existed resumes with its stored config unchanged.
        if args.lr_schedule != "cosine":
            config |= {"lr_schedule": args.lr_schedule, "lr_decay_fraction": args.lr_decay_fraction}
        if args.cov_weight_final is not None:
            config["cov_weight_final"] = args.cov_weight_final
        if args.prefix_gate is not None:
            config["prefix_gate"] = args.prefix_gate.spec()
        if args.robust_config is not None:
            from robust_transition import validate_config

            config |= {"robust_objective": args.robust_spec["robust_objective"],
                       "robust_transition_sha256": args.robust_digest}
            validate_config(config, args.robust_spec)
        consumed = list(inherited) if args.resume is None else []
        if rank == 0:
            if args.resume is None:
                with (args.out / "run-config.json").open("x") as stream:
                    json.dump(config, stream, indent=2)
            else:
                stored = json.loads((args.out / "run-config.json").read_text())
                if stored != config:
                    raise ValueError(f"resume changes the run's configuration: {stored} -> {config}")
                consumed = [json.loads(line) for line in (args.resume / "consumed.jsonl").read_text().splitlines()]
                if any(entry["step"] > start_step for entry in consumed):
                    raise RuntimeError("the checkpoint's consumed list runs past its own step")
        pool_loop(args, model, optimizer, rank, world, run, start_step, consumed, anchors_done)
        if run is not None:
            run.finish()
        dist.destroy_process_group()
        return
    if fast:
        import fast_train

        warmup_began = time.monotonic()
        warm_up(model, args.micro_anchors, world, args.objective)
        print(f"[rank{rank}] warm-up done in {time.monotonic() - warmup_began:.0f} s", flush=True)
    prefetch = FeaturePrefetch()

    def row_of(step):
        return min(step * world + rank, args.total_rows - 1)

    for step in range(start_step, args.steps):
        row_number = row_of(step)
        directory = args.queue / f"train-{row_number:09d}"
        features, metadata, producer_wait, load_wait = prefetch.take(directory)
        if step + 1 < args.steps and row_of(step + 1) != row_number:
            prefetch.request(args.queue / f"train-{row_of(step + 1):09d}")
        if metadata["split"] != "train":
            raise ValueError("held-out features cannot enter training")
        if not args.smoke and (metadata["scope"] != "dataset_epoch" or metadata["sequence_index"] != row_number):
            raise ValueError("production requires the exact indexed epoch order")
        ids, gr, mask = features["input_ids"], features["target_gr"], features["loss_mask"]
        if args.fp32_smoke:
            gr = gr.float()
        if not torch.equal(features["positions"], torch.arange(len(ids), device="cuda")):
            raise ValueError("capture position mismatch")
        generator = torch.Generator(device="cuda").manual_seed(20260909 + row_number)
        anchors = sample_anchors(mask, args.anchors, generator=generator)
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        profiler = None
        if step + 1 == args.profile_step:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            profiler.start()
        if fast:
            stages = ("load", "handoff", "prepare", "units", "clip_sync", "optimizer")
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            marks[0].record()
            if len(anchors) == 0:
                raise ValueError("no supervised anchors")
            anchors = anchors.sort().values
            marks[1].record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terms, step_marks, _ = fast_step(model, [Segment(ids, gr, mask, anchors)], args.micro_anchors, world,
                                                 args.objective)
            marks += step_marks
        else:
            stages = ("clean", "proposal", "objective", "backward", "clip_sync", "optimizer")
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            marks[0].record()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32_smoke):
                trace = clean_trace(model, ids, gr)
                prefix = trace.before(anchors, model.config.window)
                del trace
                marks[1].record()
                hidden = model.propose(ids[anchors], gr[anchors - 1], anchors, prefix)
                marks[2].record()
                labels, predecessors, teacher, valid = block_targets(ids, gr, anchors, mask=mask)
                terms = chunked_terms(model, hidden, labels, predecessors, teacher, args.objective_chunk, valid)
                loss = reduce_loss(terms, world)
                marks[3].record()
            loss.backward()
            marks[4].record()
            del prefix, hidden, labels, predecessors, teacher, loss
        grad_norm = sync_clip_replicated(model, world) if fast else sync_clip(model, rank, world)
        marks.append(torch.cuda.Event(enable_timing=True))
        marks[-1].record()
        progress = (step + 1) / args.steps
        factor = progress / 0.03 if progress < 0.03 else 0.5 * (1 + math.cos(math.pi * (progress - 0.03) / 0.97))
        # A correctness smoke must actually update; its short budget has no
        # relevance to the full production cosine schedule.
        if args.smoke:
            factor = 1.0
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        before = model.fc_hidden.weight.detach().clone() if args.smoke else None
        optimizer.step()
        marks.append(torch.cuda.Event(enable_timing=True))
        marks[-1].record()
        if args.smoke:
            if torch.equal(before, model.fc_hidden.weight):
                raise RuntimeError("FP32 master did not update")
            for parameter, state in optimizer.state.items():
                if (
                    parameter.dtype != torch.float32
                    or state["exp_avg"].dtype != torch.float32
                    or state["exp_avg_sq"].dtype != torch.float32
                ):
                    raise RuntimeError("master/optimizer precision contract violated")
        torch.cuda.synchronize()
        if profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(str(args.out / f"trace-step{step + 1}-rank{rank}.json.gz"))
            (args.out / f"profile-step{step + 1}-rank{rank}.txt").write_text(
                profiler.key_averages().table(sort_by="self_device_time_total", row_limit=35)
            )
        report = reduce_report(terms)
        elapsed = time.monotonic() - begin
        if rank == 0:
            record = {
                "step": step + 1,
                "compute_dtype": "float32" if args.fp32_smoke else "bfloat16",
                "sequence_tokens_rank0": len(ids),
                "sampled_anchors_rank0": len(anchors),
                "seconds": elapsed,
                "stage_gpu_ms": dict(zip(stages, (a.elapsed_time(b) for a, b in zip(marks, marks[1:])), strict=True)),
                "wait_seconds": producer_wait,
                "load_wait_seconds": load_wait,
                "gradient_norm": grad_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                **accept_metrics(report),
            }
            write_record(args.out, record, run)
        dist.barrier()
        # Smoke features remain available for cache/quality gates.
        (directory / f"CONSUMED-{rank}").touch()
        stop_flag = torch.tensor(int(rank == 0 and (args.out / "STOP").exists()), device="cuda", dtype=torch.int32)
        dist.broadcast(stop_flag, src=0)
        stop_requested = bool(stop_flag.item())
        due = (
            (step + 1) % args.save_every == 0
            or step + 1 == args.steps
            or stop_requested
            or (not args.smoke and step == 0)
        )
        if (not args.smoke or args.checkpoint_smoke) and due:
            save_state(
                model, optimizer, args.out, step + 1, rank, world, args.checkpoint, args.dataset_manifest, fast
            )
        # Capture is reproducible. On resume the producer replays from the last
        # durable step; its state is never inferred from disposable features.
        if not args.smoke:
            dist.barrier()
            if rank == 0:
                for n in range(step * world, min((step + 1) * world, args.total_rows)):
                    shutil.rmtree(args.queue / f"train-{n:09d}")
            dist.barrier()
        del features, ids, gr, mask, terms
        if stop_requested:
            break
    if args.smoke and rank == 0:
        (args.out / "smoke-gate.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "steps": args.steps,
                    "world_size": world,
                    "fp32_master_and_moments": True,
                    "attention_backend": args.attention_backend,
                    "compute_dtype": "float32" if args.fp32_smoke else "bfloat16",
                    "anchors": args.anchors,
                    "resumed_from": str(args.resume) if args.resume else None,
                    "production_ready": False,
                },
                indent=2,
            )
        )
    if rank == 0:
        result = {
            "completed": step + 1 == args.steps,
            "last_step": step + 1,
            "reason": "stop_request" if stop_requested else "epoch_complete",
        }
        temp = args.out / "run-result.partial"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out / "run-result.json")
    if run is not None:
        run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
