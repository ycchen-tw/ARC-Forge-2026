"""Frozen row-sharded PLE lookup and replicated-adapter collectives."""

import time

import torch
import torch.distributed as dist
from torch import nn


class ForwardTimers:
    """Bounded CUDA event timing; avoids collecting millions of profiler events."""

    def __init__(self, layers):
        self.events, self.handles = [], []
        for layer in layers:
            for name, module in layer.named_children():
                if name not in {"linear_attn", "self_attn", "mlp", "ple"}:
                    continue
                pending = []

                def before(mod, inputs, pending=pending):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    pending.append((event, time.perf_counter()))

                def after(mod, inputs, output, pending=pending, name=name):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    start, cpu_start = pending.pop()
                    self.events.append((name, start, event, (time.perf_counter() - cpu_start) * 1000))

                self.handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    def finish(self):
        for handle in self.handles:
            handle.remove()
        result = {}
        for name, start, end, cpu_ms in self.events:
            group = result.setdefault(name, {"cuda_elapsed_ms": 0, "cpu_wall_ms": 0, "calls": 0})
            group["cuda_elapsed_ms"] += start.elapsed_time(end)
            group["cpu_wall_ms"] += cpu_ms
            group["calls"] += 1
        return result


def observe_flash_qla():
    """Count executed calls, not just the registry's preferred backend."""
    from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend

    original = FlashQLABackend.chunk_gated_delta_rule
    report = {"calls": 0, "shapes": []}

    def observed(self, *args, **kwargs):
        report["calls"] += 1
        q = args[0] if args else kwargs["q"]
        shape = list(q.shape)
        if shape not in report["shapes"]:
            report["shapes"].append(shape)
        return original(self, *args, **kwargs)

    FlashQLABackend.chunk_gated_delta_rule = observed
    return report


class RowShardedEmbedding(nn.Module):
    """Exchange variable-size DP requests, then reduce-scatter their vectors.

    Exactly one owner supplies each vector: reduction is exact even in BF16.
    Only this frozen lookup omits autograd; downstream projections keep gradients.
    """

    def __init__(self, weight, device):
        super().__init__()
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.start = self.num_embeddings * self.rank // self.world
        self.end = self.num_embeddings * (self.rank + 1) // self.world
        self.register_buffer("weight", weight[self.start:self.end].to(device, copy=True))

    @torch.no_grad()
    def forward(self, ids):
        count = torch.tensor([ids.numel()], device=ids.device, dtype=torch.long)
        counts = [torch.empty_like(count) for _ in range(self.world)]
        dist.all_gather(counts, count)
        width = max(int(x) for x in counts)
        requests = torch.zeros(width, dtype=torch.long, device=ids.device)
        requests[:ids.numel()] = ids.flatten()
        gathered = torch.empty(self.world * width, dtype=torch.long, device=ids.device)
        dist.all_gather_into_tensor(gathered, requests)
        owned = (gathered >= self.start) & (gathered < self.end)
        local_ids = (gathered - self.start).clamp(0, self.end - self.start - 1)
        vectors = torch.nn.functional.embedding(local_ids, self.weight)
        vectors.mul_(owned[:, None])
        output = torch.empty((width, self.embedding_dim), device=ids.device, dtype=self.weight.dtype)
        dist.reduce_scatter_tensor(output, vectors)
        return output[:ids.numel()].reshape(*ids.shape, self.embedding_dim)


def sync_gradients(parameters):
    """One FP32 bucket; caller scales local mean by local/global tokens."""
    assert_valid_gradients(parameters, "before synchronization")
    flat = torch.cat([p.grad.flatten().float() for p in parameters])
    dist.all_reduce(flat)
    offset = 0
    for p in parameters:
        p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
        offset += p.numel()
    assert_valid_gradients(parameters, "after synchronization")


def _assert_on_every_rank(valid, device, message):
    flag = torch.as_tensor(valid, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag):
        raise RuntimeError(message)


def assert_valid_gradients(parameters, stage):
    """Every rank rejects together, including a missing gradient on one rank."""
    valid = torch.ones((), dtype=torch.bool, device=parameters[0].device)
    for p in parameters:
        if p.grad is None:
            valid.fill_(False)
        else:
            valid.logical_and_(torch.isfinite(p.grad).all())
    _assert_on_every_rank(valid, parameters[0].device, f"missing/nonfinite gradients {stage}")


def sync_and_clip_gradients(parameters, max_norm=1.0, observe=None):
    """Refuse invalid gradients/norms on all ranks before AdamW can mutate state. ``observe(parameters)`` (debug) sees
    the summed gradients before clipping."""
    sync_gradients(parameters)
    if observe is not None:
        observe(parameters)
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters], error_if_nonfinite=False)
    _assert_on_every_rank(torch.isfinite(norm) & (norm > 0), parameters[0].device,
                          "invalid gradient norm after synchronization")
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
    assert_valid_gradients(parameters, "after clipping")
    return float(norm)


@torch.no_grad()
def assert_replicated(parameters):
    flat = torch.cat([p.flatten() for p in parameters])
    reference = flat.clone()
    dist.broadcast(reference, 0)
    error = (flat - reference).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if float(error) != 0:
        raise RuntimeError(f"LoRA replicas differ: max_abs={float(error)}")
    return float(error)
