"""Shape-independent BF16 forward kernels with ordinary analytic gradients.

Fixed reduction layouts matter for inherited GR: one BF16 rounding difference
can change a routed expert and grow through three identical warm-start layers.
These kernels never tune a reduction layout based on batch/sequence length.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F


@triton.jit(do_not_specialize=["M"])
def _matmul(
    X,
    W,
    Y,
    Offsets,
    Tiles,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    GROUPED: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    tile, col = tl.program_id(0), tl.program_id(1)
    begin, end, expert, local_tile = 0, M, 0, tile
    if GROUPED:
        valid_tile = tile < tl.load(Tiles + E - 1)
        if valid_tile:
            lo, hi = 0, E
            while lo < hi:
                mid = (lo + hi) // 2
                upto = tl.load(Tiles + mid)
                right = tile >= upto
                lo = tl.where(right, mid + 1, lo)
                hi = tl.where(right, hi, mid)
            expert = lo
            previous_tiles = tl.load(Tiles + expert - 1, expert > 0, other=0)
            begin = tl.load(Offsets + expert - 1, expert > 0, other=0)
            end = tl.load(Offsets + expert)
            local_tile = tile - previous_tiles
        else:
            begin, end = 0, 0
    rows = begin + local_tile * BM + tl.arange(0, BM)
    cols = col * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    if begin < end:
        for offset in range(tl.cdiv(K, BK)):
            inner = offset * BK + kk
            a = tl.load(X + rows[:, None] * K + inner[None, :], (rows[:, None] < end) & (inner[None, :] < K), 0)
            b = tl.load(
                W + expert * N * K + cols[None, :] * K + inner[:, None], (cols[None, :] < N) & (inner[:, None] < K), 0
            )
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        tl.store(Y + rows[:, None] * N + cols[None, :], accumulator, (rows[:, None] < end) & (cols[None, :] < N))


def _gemm(x, weight, offsets=None):
    x, weight = x.contiguous(), weight.contiguous()
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("stable GEMM requires BF16 operands")
    m, k = x.shape
    n = weight.shape[-2]
    if weight.shape[-1] != k:
        raise ValueError("GEMM inner dimension mismatch")
    out = torch.empty(m, n, device=x.device, dtype=x.dtype)
    if offsets is None:
        _matmul[(triton.cdiv(m, 16), triton.cdiv(n, 64))](x, weight, out, x, x, m, n, k, 1, False)
    else:
        counts = torch.diff(F.pad(offsets, (1, 0)))
        tiles = ((counts + 15) // 16).cumsum(0, dtype=torch.int32)
        groups = len(offsets)
        # This upper bound avoids a GPU->CPU synchronization and supports graphs.
        _matmul[(triton.cdiv(m, 16) + groups, triton.cdiv(n, 64))](
            x, weight, out, offsets, tiles, m, n, k, groups, True
        )
    return out


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return _gemm(x.reshape(-1, x.shape[-1]), weight).view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        grad = dy.reshape(-1, dy.shape[-1]).to(x.dtype)
        with torch.autocast("cuda", enabled=False):
            dx = (grad @ weight).view_as(x) if ctx.needs_input_grad[0] else None
            dw = grad.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
        return dx, dw


class StableLinear(nn.Linear):
    def forward(self, x):
        if self.bias is not None:
            raise ValueError("native MTP has no linear biases")
        if not x.is_cuda:
            return F.linear(x, self.weight)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if dtype != torch.bfloat16:
            return F.linear(x, self.weight)
        return _Linear.apply(x.to(dtype), self.weight.to(dtype))


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets):
        ctx.save_for_backward(x, weight, offsets)
        return _gemm(x, weight, offsets)

    @staticmethod
    def backward(ctx, dy):
        x, weight, offsets = ctx.saved_tensors
        # Use PyTorch's tested grouped backward. Its extra forward is temporary;
        # forward cache numerics remain fixed by the Triton path above.
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            rx = x.detach().requires_grad_(True)
            rw = weight.detach().requires_grad_(True)
            reference = torch._grouped_mm(rx, rw.transpose(1, 2), offs=offsets)
            dx, dw = torch.autograd.grad(reference, (rx, rw), dy.to(x.dtype))
        return dx if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None, None


def grouped_linear(x, weight, offsets):
    return _Grouped.apply(x, weight, offsets)


@triton.jit
def _norm(X, W, Y, Inv, WIDTH: tl.constexpr, GROUPS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    weight = tl.load(W + (row % GROUPS) * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    tl.store(Inv + row, inv)
    tl.store(Y + row * WIDTH + col, x * inv * (1 + weight), col < WIDTH)


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, width, eps):
        x = x.contiguous()
        groups = x.shape[-1] // width
        inv = torch.empty(x.numel() // width, device=x.device, dtype=torch.float32)
        y = torch.empty_like(x)
        _norm[(len(inv),)](x, weight, y, inv, width, groups, eps, triton.next_power_of_2(width), num_warps=8)
        ctx.save_for_backward(x, weight, inv)
        ctx.width = width
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, inv = ctx.saved_tensors
        width = ctx.width
        shape = (-1, x.shape[-1] // width, width)
        z = x.float().view(shape)
        g = dy.float().reshape(shape)
        r = inv.view(*z.shape[:-1], 1)
        gw = g * (1 + weight.float().view(1, -1, width))
        dx = r * (gw - z * (gw * z).mean(-1, keepdim=True) * r.square())
        dw = (g * z * r).sum(0).flatten()
        return dx.view_as(x).to(x.dtype), dw.to(weight.dtype), None, None


def rms_norm(x, weight, width, eps):
    return _Norm.apply(x, weight, width, eps)


@triton.jit(do_not_specialize=["QL", "KL"])
def _swa_fwd(
    Q,
    K,
    V,
    QP,
    KP,
    Out,
    OF,
    LSE,
    SINK,
    QL,
    KL,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr = 64,
):
    qi, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (HQ // HK)
    dd = tl.arange(0, BD)
    nn = tl.arange(0, BN)
    qoff = ((b * HQ + h) * QL + qi) * D + dd
    q = tl.load(Q + qoff, dd < D, 0).to(tl.float32)
    pos = tl.load(QP + b * QL + qi)
    left = KL - QL + qi - W + 1
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    acc = tl.full((BD,), 0, tl.float32)
    if HAS_SINK:
        maximum = tl.load(SINK + h)
        denominator = tl.full((), 1, tl.float32)
    for block in range(tl.cdiv(W, BN)):
        offset = block * BN + nn
        ki = left + offset
        physical = (ki >= 0) & (ki < KL) & (offset < W)
        kp = tl.load(KP + b * KL + ki, physical, other=-1)
        valid = physical & (kp >= 0) & (kp <= pos) & (pos - kp < W)
        address = ((b * HK + kh) * KL + ki[:, None]) * D + dd[None, :]
        key = tl.load(K + address, physical[:, None] & (dd[None, :] < D), 0).to(tl.float32)
        value = tl.load(V + address, physical[:, None] & (dd[None, :] < D), 0).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * (D**-0.5)
        score = tl.where(valid, score, float("-inf"))
        newmax = tl.maximum(maximum, tl.max(score, 0))
        shift = tl.where(newmax == float("-inf"), 0.0, newmax)
        correction = tl.exp(maximum - shift)
        probability = tl.exp(score - shift)
        acc = acc * correction + tl.sum(probability[:, None] * value, 0)
        denominator = denominator * correction + tl.sum(probability, 0)
        maximum = newmax
    y = acc / denominator
    tl.store(Out + qoff, y, dd < D)
    tl.store(OF + qoff, y, dd < D)
    tl.store(LSE + (b * HQ + h) * QL + qi, maximum + tl.log(denominator))


@triton.jit(do_not_specialize=["QL", "KL"])
def _swa_bwd(
    Q,
    K,
    V,
    QP,
    KP,
    OF,
    LSE,
    DO,
    DQ,
    DK,
    DV,
    SINK,
    DSINK,
    QL,
    KL,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    PREFIX: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr = 32,
):
    qi, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kh = h // (HQ // HK)
    dd = tl.arange(0, BD)
    nn = tl.arange(0, BN)
    qoff = ((b * HQ + h) * QL + qi) * D + dd
    q = tl.load(Q + qoff, dd < D, 0).to(tl.float32)
    dy = tl.load(DO + qoff, dd < D, 0).to(tl.float32)
    y = tl.load(OF + qoff, dd < D, 0)
    delta = tl.sum(dy * y, 0)
    lse = tl.load(LSE + (b * HQ + h) * QL + qi)
    if HAS_SINK:
        sink_probability = tl.exp(tl.load(SINK + h) - lse)
        tl.atomic_add(DSINK + h, -sink_probability * delta, sem="relaxed")
    pos = tl.load(QP + b * QL + qi)
    left = KL - QL + qi - W + 1
    dq = tl.full((BD,), 0, tl.float32)
    for block in range(tl.cdiv(W, BN)):
        offset = block * BN + nn
        ki = left + offset
        physical = (ki >= 0) & (ki < KL) & (offset < W)
        kp = tl.load(KP + b * KL + ki, physical, other=-1)
        valid = physical & (kp >= 0) & (kp <= pos) & (pos - kp < W)
        address = ((b * HK + kh) * KL + ki[:, None]) * D + dd[None, :]
        key = tl.load(K + address, physical[:, None] & (dd[None, :] < D), 0).to(tl.float32)
        value = tl.load(V + address, physical[:, None] & (dd[None, :] < D), 0).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * (D**-0.5)
        probability = tl.where(valid, tl.exp(score - lse), 0.0)
        dp = tl.sum(value * dy[None, :], 1)
        ds = probability * (dp - delta)
        dq += tl.sum(ds[:, None] * key, 0) * (D**-0.5)
        writable = valid[:, None] & (ki[:, None] >= PREFIX) & (dd[None, :] < D)
        tl.atomic_add(DK + address, ds[:, None] * q[None, :] * (D**-0.5), writable, sem="relaxed")
        tl.atomic_add(DV + address, probability[:, None] * dy[None, :], writable, sem="relaxed")
    tl.store(DQ + qoff, dq, dd < D)


class _SWA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, key_positions, window, prefix_length, sink_logits):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        positions, key_positions = positions.contiguous(), key_positions.contiguous()
        b, hq, ql, d = q.shape
        hk, kl = k.shape[1:3]
        has_sink = sink_logits is not None
        sink = sink_logits.float().contiguous() if has_sink else q.new_empty(0, dtype=torch.float32)
        if has_sink and sink.shape != (hq,):
            raise ValueError("one attention sink is required per query head")
        if (
            k.shape != v.shape
            or kl - ql != prefix_length
            or positions.shape != (b, ql)
            or key_positions.shape != (b, kl)
        ):
            raise ValueError("stable SWA requires an aligned causal prefix plus query block")
        out = torch.empty_like(q)
        precise = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(b, hq, ql, device=q.device, dtype=torch.float32)
        _swa_fwd[(ql, hq, b)](
            q,
            k,
            v,
            positions,
            key_positions,
            out,
            precise,
            lse,
            sink,
            ql,
            kl,
            hq,
            hk,
            d,
            window,
            has_sink,
            triton.next_power_of_2(d),
            num_warps=8,
        )
        ctx.save_for_backward(q, k, v, positions, key_positions, precise, lse, sink)
        ctx.has_sink = has_sink
        ctx.window, ctx.prefix_length = window, prefix_length
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k, v, positions, key_positions, precise, lse, sink = ctx.saved_tensors
        b, hq, ql, d = q.shape
        hk, kl = k.shape[1:3]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dsink = torch.zeros_like(sink)
        _swa_bwd[(ql, hq, b)](
            q,
            k,
            v,
            positions,
            key_positions,
            precise,
            lse,
            dy.contiguous(),
            dq,
            dk,
            dv,
            sink,
            dsink,
            ql,
            kl,
            hq,
            hk,
            d,
            ctx.window,
            ctx.prefix_length,
            ctx.has_sink,
            triton.next_power_of_2(d),
            num_warps=8,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, dsink if ctx.has_sink else None


def causal_swa(q, k, v, positions, key_positions, window, prefix_length, sink_logits=None):
    return _SWA.apply(q, k, v, positions, key_positions, window, prefix_length, sink_logits)


def enable_stable_bf16(model, *, attention=True, experts=True, norms=True):
    """Change numerical kernels only; names, weights and BF16 storage stay intact."""
    from model import Experts, RMSNorm

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            module.__class__ = StableLinear
        elif isinstance(module, Experts):
            module.stable_gemm = experts
        elif isinstance(module, RMSNorm):
            module.stable_kernel = norms
    if attention:
        for layer in model.layers:
            layer.self_attn.backend = "stable"
