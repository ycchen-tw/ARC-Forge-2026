"""gEAN320 INT3 routed experts with trainable group scales (QAD v2: codes fixed, scales learned).

The deployed expert weight is ``W[e, n, k] = (c[e, n, k] - 4) * s[e, n, k // 128]`` with Humming uint3 codes ``c`` and a
signed BF16 scale ``s`` per output row and 128-wide K group (work/qwen38fn_lora/int3_moe.py). Here the codes stay the
package's bytes and every scale has an FP32 master. The forward is exactly the frozen W3A8 path of int3_moe.py
(per-row E4M3 activations at both GEMMs, BF16 SwiGLU staging, SGLang's fused_moe_kernel) run with ``bf16(master)``,
the scale the exporter writes. The cast is a straight-through estimator: the BF16 scale's gradient goes to the master.

Gradient of a scale (fixed codes, no rounding of the weight is involved):

    dL/ds[e, n, g] = sum_{k in group g} (c[e, n, k] - 4) * dL/dW[e, n, k]
    dL/dW2_e  = (score * dY)_e^T @ A_dq,e      A_dq = the forward's dequantized E4M3 SwiGLU rows
    dL/dW13_e = dPre_e^T @ X_dq,e              X_dq = the forward's dequantized E4M3 input rows

i.e. the weight gradient seen through the activation quantizers (STE) with the inputs the forward actually multiplied.
dX and dRouting are int3_moe.FrozenInt3Function's (dequantized-BF16 weights, the forward's `pre` / `y`).

W3A16 (``int3_moe.configure("bf16", ...)``, the trainer's --int3-act bf16): the forward is int3_moe.a16_experts, the
Humming A16 product with the effective weight ``bf16((c - 4) * bf16(master))``; the scale gradient is the same formula
with A_dq / X_dq replaced by what that forward multiplied: the BF16 SwiGLU rows (int3_moe.swiglu_bf16 of `pre`,
recomputed in backward, bitwise the forward's) and the BF16 token rows gathered by the routing.

The weight gradients are formed per expert in FP32 from the BF16 / E4M3-dequantized operands (exact in FP32), reduced
to the group scales and dropped; no [E, N, K] gradient bank is kept.

Invariants checked at every optimizer step (``check_scales``): no scale changes sign or casts to a BF16 zero (a
scale that is zero in the package is never trained away from zero either: its codes multiply nothing useful and the
format has no way to express a sign flip of a whole group without re-choosing the codes).
"""

import sys
from pathlib import Path

import torch
import triton
from torch import nn

from qad_timing import section
from scale_grad_triton import scale_gradients_fused, scale_gradients_quack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen38fn_lora"))
from fp8_moe import _routing_backward, _swiglu_backward, padded_routing, routing_metadata  # noqa: E402
import int3_moe  # noqa: E402
from int3_moe import (  # noqa: E402
    GROUP,
    FrozenInt3Experts,
    a16_experts,
    bf16_grouped,
    grouped_w3a8,
    quantize_rows,
    swiglu_bf16,
    swiglu_quantize,
)

COUNTERS = {"forward": 0, "backward": 0, "scale_grad_experts": 0}
# Scale-gradient kernel: "fp32" = per-expert FP32 GEMM on the exact FP32 images of the BF16 / dequantized E4M3 operands
# (TF32 off, Python loop over experts; the reference). "fused" = scale_grad_triton (one launch, bf16x2 operands, FP32
# register accumulation, deterministic; ~1e-5 relative to float64, tests/test_scale_grad_triton.py); "quack" = the same
# math as a pre-pass plus quack's SM100 tcgen05 GEMM (scale_grad_triton.scale_gradients_quack). "precision" is the
# fused kernel's operand precision (bf16x2, or bf16: one dot, G' rounded to 2^-9). Set by the trainer.
SCALE_GRAD = {"mode": "fp32", "precision": "bf16x2"}


def _group_reduce(dw, codes):
    """[n, k] fp32 weight gradient, [n, k] E4M3 (code - 4) -> [n, k / 128] fp32 scale gradient."""
    n, k = dw.shape
    return (dw.view(n, k // GROUP, GROUP) * codes.view(n, k // GROUP, GROUP).float()).sum(-1)


def scale_gradients(g_rows, a_rows, codes, starts, out):
    """out[e] = group_reduce(G_e^T @ A_e, codes[e]) for every expert with rows; experts without rows get 0.

    ``g_rows(lo, hi)`` / ``a_rows(lo, hi)`` return the output-side gradient rows [hi-lo, N] and the forward's
    dequantized input rows [hi-lo, K] (both fp32, routing order) of one expert, so no [TK, K] fp32 copy is formed.
    codes [E, N, K] E4M3 (code - 4), starts [E+1] row offsets, out [E, N, K/128] fp32 (written)."""
    if SCALE_GRAD["mode"] != "fp32":
        raise ValueError(f"scale gradient mode {SCALE_GRAD['mode']!r}")
    bounds = starts.tolist()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out.zero_()
        for e in range(codes.shape[0]):
            lo, hi = bounds[e], bounds[e + 1]
            if hi > lo:
                out[e] = _group_reduce(g_rows(lo, hi).T @ a_rows(lo, hi), codes[e])
                COUNTERS["scale_grad_experts"] += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


def dequant_rows(q, scale):
    """E4M3 [rows, K] with one fp32 scale per row -> fp32 [rows, K] (exact)."""
    return q.float() * scale[:, None]


class ScaleInt3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, m1, m2, indices, scores):
        from liger_kernel.ops.fused_moe import _token_aggregation

        if w1.requires_grad or w2.requires_grad:
            raise RuntimeError("INT3 codes must stay frozen")
        if m1.dtype != torch.float32 or m2.dtype != torch.float32:
            raise TypeError(f"scale masters must be fp32, got {m1.dtype} / {m2.dtype}")
        s1 = m1.to(torch.bfloat16).contiguous()  # the deployed (exported) scale; STE in backward
        s2 = m2.to(torch.bfloat16).contiguous()
        x = x.contiguous()
        routing = padded_routing(routing_metadata(indices.contiguous().int(), w1.shape[0]))
        mode = int3_moe.ACT["mode"]
        if mode == "bf16":
            pre, y = a16_experts(x, w1, w2, s1, s2, routing)
            int3_moe.COUNTERS["a16_forward"] += 1
            saved = (x,)   # the backward re-forms its A operands from x and pre (module docstring)
        elif mode == "fp8":
            gather = routing[2].long()
            q, a_scale = quantize_rows(x)
            q_g, a_g = q.index_select(0, gather), a_scale.index_select(0, gather)
            pre = grouped_w3a8(q_g, a_g, w1, s1, routing)
            q2, a2 = swiglu_quantize(pre)
            y = grouped_w3a8(q2, a2, w2, s2, routing)
            int3_moe.COUNTERS["a8_forward"] += 1
            saved = (q_g, a_g, q2, a2)
        else:
            raise ValueError(f"INT3 activation format {mode!r}")
        out = _token_aggregation(y, scores.flatten().contiguous(), routing[4], x.shape[0], indices.shape[1], x.shape[1])
        ctx.mode = mode
        ctx.save_for_backward(w1, w2, s1, s2, pre, y, scores, *saved, *routing)
        ctx.want_scales = (m1.requires_grad, m2.requires_grad)
        COUNTERS["forward"] += 1
        return out

    @staticmethod
    def backward(ctx, dy):
        from liger_kernel.ops.fused_moe_kernels import _token_gather_weighted_sum_kernel

        w1, w2, s1, s2, pre, y, scores, *rest = ctx.saved_tensors
        if ctx.mode == "bf16":
            (x,), routing = rest[:1], rest[1:]
        else:
            (q_g, a_g, q2, a2), routing = rest[:4], rest[4:]
        dy = dy.contiguous()
        gather = routing[2].long()
        with section("moe_dx"):
            dpost = bf16_grouped(dy, w2, s2, routing, gather=True)
        dpre = torch.empty_like(pre)
        i = pre.shape[1] // 2
        _swiglu_backward[(pre.shape[0],)](pre, dpost, scores, routing[3], dpre, i, triton.next_power_of_2(i))
        ds = torch.empty_like(scores)
        _routing_backward[(pre.shape[0],)](
            dy, y, routing[2], routing[3], ds, dy.shape[1], triton.next_power_of_2(dy.shape[1])
        )
        expanded = bf16_grouped(dpre, w1, s1, routing)
        dx = torch.empty_like(dy)
        _token_gather_weighted_sum_kernel[(dy.shape[0],)](
            expanded, ds, routing[4], dx, H_dim=dy.shape[1], K_dim=scores.shape[1],
            stride_Y_TK=expanded.stride(0), stride_Y_H=1, stride_out_T=dx.stride(0), stride_out_H=1, w_is_None=True,
        )
        dm1 = dm2 = None
        starts = routing[1]
        mode = SCALE_GRAD["mode"]
        if mode not in ("fused", "quack", "fp32"):
            raise ValueError(f"scale gradient mode {mode!r}")
        fused = {"fused": scale_gradients_fused, "quack": scale_gradients_quack}.get(mode)
        if ctx.mode == "bf16":
            # A16: the A operands are BF16 already (row scale 1, exact): the SwiGLU rows the w2 GEMM read, and the
            # token rows of x the w13 GEMM gathered (quack gathers them per expert chunk; the others form the copy)
            ones = torch.ones(pre.shape[0], dtype=torch.float32, device=dy.device)
            q2, a2 = swiglu_bf16(pre) if ctx.want_scales[1] else None, ones
            if ctx.want_scales[0]:
                if mode == "quack":
                    q_g, a_g, a_index = x, ones, routing[2]
                else:
                    q_g, a_g, a_index = x.index_select(0, gather), ones, None
        else:
            a_index = None
        if ctx.want_scales[1]:
            # rows of dL/dY_e: the token gradient times this row's routing weight (bf16 scores, as the forward used)
            weight = scores.flatten().index_select(0, routing[3].long())
            with section("scale_grad_w2"):
                dm2 = torch.empty(s2.shape, dtype=torch.float32, device=dy.device)
                if fused:
                    fused(dy, q2, a2, w2, starts, dm2, g_index=routing[2], g_weight=weight,
                          precision=SCALE_GRAD["precision"])
                else:
                    weight = weight.float()
                    scale_gradients(lambda lo, hi: dy.index_select(0, gather[lo:hi]).float() * weight[lo:hi, None],
                                    lambda lo, hi: dequant_rows(q2[lo:hi], a2[lo:hi]), w2, starts, dm2)
        if ctx.want_scales[0]:
            with section("scale_grad_w13"):
                dm1 = torch.empty(s1.shape, dtype=torch.float32, device=dy.device)
                if mode == "quack":
                    fused(dpre, q_g, a_g, w1, starts, dm1, precision=SCALE_GRAD["precision"], a_index=a_index)
                elif fused:
                    fused(dpre, q_g, a_g, w1, starts, dm1, precision=SCALE_GRAD["precision"])
                else:
                    scale_gradients(lambda lo, hi: dpre[lo:hi].float(),
                                    lambda lo, hi: dequant_rows(q_g[lo:hi], a_g[lo:hi]), w1, starts, dm1)
        COUNTERS["backward"] += 1
        return dx, None, None, dm1, dm2, None, ds


class ScaleInt3Experts(nn.Module):
    """FrozenInt3Experts with the two scale banks as FP32 Parameters (plus their package signs, BF16, frozen)."""

    def __init__(self, frozen: FrozenInt3Experts):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = (
            frozen.num_experts, frozen.hidden_dim, frozen.intermediate_dim)
        self.gate_up_proj = frozen.gate_up_proj
        self.down_proj = frozen.down_proj
        for name in ("gate_up_scale", "down_scale"):
            source = getattr(frozen, name)
            if source.dtype != torch.bfloat16:
                raise TypeError(f"{name}: package scales must be BF16, got {source.dtype}")
            setattr(self, name, nn.Parameter(source.detach().float(), requires_grad=True))
            # FSDP shards Parameters on dim 0 like the scales, so each rank checks its own rows; BF16 (+-1 / 0 exact)
            # because FSDP2 cannot hold an integer Parameter
            setattr(self, name + "_sign0", nn.Parameter(torch.sign(source.detach()), requires_grad=False))
            # the package scale, for the update statistics (how many deployed BF16 scales moved, and how far)
            setattr(self, name + "_init", nn.Parameter(source.detach().clone(), requires_grad=False))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        shape = hidden_states.shape
        out = ScaleInt3Function.apply(
            hidden_states.reshape(-1, self.hidden_dim), self.gate_up_proj, self.down_proj,
            self.gate_up_scale, self.down_scale, top_k_index, top_k_weights)
        return out.reshape(shape)


def install(layers):
    """Swap every decoder layer's FrozenInt3Experts for ScaleInt3Experts (before FSDP). Returns the new modules."""
    from types import MethodType

    swapped = []
    for layer in layers:
        frozen = layer.mlp.experts
        if not isinstance(frozen, FrozenInt3Experts):
            raise RuntimeError(f"expected FrozenInt3Experts, found {type(frozen).__name__}")
        module = ScaleInt3Experts(frozen)
        module.forward = MethodType(ScaleInt3Experts.forward, module)
        layer.mlp.experts = module
        swapped.append(module)
    return swapped


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


@torch.no_grad()
def mask_zero_scale_grads(modules):
    """Zero the gradient of every scale that is zero in the package (it stays zero; see the module docstring)."""
    for module in modules:
        for name in ("gate_up_scale", "down_scale"):
            p = getattr(module, name)
            if p.grad is not None:
                _local(p.grad).masked_fill_(_local(getattr(module, name + "_sign0")) == 0, 0.0)


@torch.no_grad()
def check_scales(modules):
    """Raise if any trained scale changed sign or casts to a BF16 zero where the package scale was nonzero.
    Returns local-shard sums: {'elements', 'changed_bf16' (deployed BF16 != package), 'rel_update_sum'
    (sum of |master - package| / |package|)}."""
    changed = total = 0
    rel = 0.0
    for module in modules:
        for name in ("gate_up_scale", "down_scale"):
            master = _local(getattr(module, name))
            sign0 = _local(getattr(module, name + "_sign0"))
            init = _local(getattr(module, name + "_init"))
            deployed = master.to(torch.bfloat16)
            now = torch.sign(deployed).to(sign0.dtype)
            bad = (now != sign0) & (sign0 != 0)
            if bool(bad.any()):
                raise RuntimeError(f"{int(bad.sum())} {name} scales changed sign or cast to zero")
            if bool(((sign0 == 0) & (deployed != 0)).any()):
                raise RuntimeError(f"a zero package {name} scale moved away from zero")
            changed += int((deployed != init).sum())
            nz = init != 0
            rel += float(((master - init.float()).abs()[nz] / init.float().abs()[nz]).sum())
            total += master.numel()
    return {"elements": total, "changed_bf16": changed, "rel_update_sum": rel}
