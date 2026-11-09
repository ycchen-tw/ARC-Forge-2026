"""Integer (INT3 by default) group GPTQ for batched routed experts.

Same structure as gptq_fast.py (MXFP4): Cholesky-inverse H, 128-column blocks,
Triton error feedback inside a block, torch GEMM between blocks. The group size
must equal the 128-column block, so every group's (scale, zero) is fixed from the
block-start weights -- identical to reference GPTQ with groupsize == blocksize.

Codes are unsigned ``q in [0, 2**bits - 1]``; dequant is ``(q - zero) * scale``.
``sym=True`` pins ``zero = 2**(bits-1)``: Humming's uint3 without zero point,
dequant ``(q - 4) * scale`` (humming/utils/weight.py dequantize_weight). Humming's own
RTN uses signed scales (largest-magnitude element -> code 0), so the grid
``[-4, 3] * scale`` can put its long end on either side; both signs are searched.
Scales are rounded to ``scale_dtype`` (bf16, what Humming stores) before any code is chosen.
Scale search: per-group clipping grid (GPTQ ``mse=True``, norm 2.4).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP = 128


def _qdq(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, (q - zero) * scale


@torch.no_grad()
def find_params(w: torch.Tensor, bits: int, sym: bool, mse: bool,
                grid: int = 100, maxshrink: float = 0.8, norm: float = 2.4,
                scale_dtype: torch.dtype = torch.bfloat16):
    """Per-row (scale, zero) for the last dim of ``w`` (``[..., GROUP]``), fp32 values
    exactly representable in ``scale_dtype``."""
    maxq = 2 ** bits - 1
    half = 2 ** (bits - 1)
    w = w.float()
    xmin = w.amin(-1).clamp_max(0)
    xmax = w.amax(-1).clamp_min(0)

    def params(lo, hi, sign):
        if sym:
            # Grid [-half, half-1] * scale. sign=+1: long end negative; sign=-1: the
            # negative scale flips it. Smallest |scale| that clips neither end.
            if sign > 0:
                mag = torch.maximum(-lo / half, hi / (half - 1))
            else:
                mag = torch.maximum(hi / half, -lo / (half - 1))
            scale = (sign * mag.clamp_min(1e-8)).to(scale_dtype).float()
            zero = torch.full_like(scale, float(half))
        else:
            scale = ((hi - lo) / maxq).clamp_min(1e-8).to(scale_dtype).float()
            zero = torch.round(-lo / scale)
        return scale, zero

    signs = (1, -1) if sym else (1,)
    best = scale = zero = None
    for sign in signs:
        for i in range(int(maxshrink * grid) if mse else 1):
            p = 1 - i / grid
            s1, z1 = params(p * xmin, p * xmax, sign)
            _, dq = _qdq(w, s1[..., None], z1[..., None], maxq)
            err = (dq - w).abs().pow(norm).sum(-1)
            if best is None:
                best, scale, zero = err, s1, z1
                continue
            better = err < best
            best = torch.where(better, err, best)
            scale = torch.where(better, s1, scale)
            zero = torch.where(better, z1, zero)
    return scale, zero


@triton.jit
def _block(W, HI, S, Z, C, ERR, O: tl.constexpr, I: tl.constexpr,  # noqa: E741
           START, MAXQ: tl.constexpr, B: tl.constexpr, R: tl.constexpr):
    expert = tl.program_id(0)
    rows = tl.program_id(1) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    w = tl.load(W + (expert * O + rows[:, None]) * I + START + cols[None, :],
                rows[:, None] < O, other=0)
    g = START // B
    scale = tl.load(S + (expert * O + rows) * (I // B) + g, rows < O, other=1)
    zero = tl.load(Z + (expert * O + rows) * (I // B) + g, rows < O, other=0)
    codes = tl.full((R, B), 0, tl.int32)
    errors = tl.full((R, B), 0, tl.float32)
    for j in range(B):
        v = tl.sum(tl.where(cols[None, :] == j, w, 0), axis=1)
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(v, scale)) + zero
        q = tl.minimum(tl.maximum(q, 0.0), MAXQ)
        dq = (q - zero) * scale
        diagonal = tl.load(HI + expert * I * I + (START + j) * I + START + j)
        error = tl.div_rn(v - dq, diagonal)
        hrow = tl.load(HI + expert * I * I + (START + j) * I + START + cols)
        w = tl.where(cols[None, :] > j, w - error[:, None] * hrow[None, :], w)
        codes = tl.where(cols[None, :] == j, q.to(tl.int32)[:, None], codes)
        errors = tl.where(cols[None, :] == j, error[:, None], errors)
    tl.store(C + (expert * O + rows[:, None]) * I + START + cols[None, :],
             codes.to(tl.uint8), rows[:, None] < O)
    tl.store(ERR + (expert * O + rows[:, None]) * B + cols[None, :],
             errors, rows[:, None] < O)


def rtn_batch(w: torch.Tensor, bits: int = 3, sym: bool = True, mse: bool = True):
    e, o, i = w.shape
    g = w.float().reshape(e, o, i // GROUP, GROUP)
    scale, zero = find_params(g, bits, sym, mse)
    q, _ = _qdq(g, scale[..., None], zero[..., None], 2 ** bits - 1)
    return q.reshape(e, o, i).to(torch.uint8), scale, zero


def dequant_batch(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    e, o, i = codes.shape
    q = codes.float().reshape(e, o, i // GROUP, GROUP)
    return ((q - zero[..., None]) * scale[..., None]).reshape(e, o, i)


def _prepare(w, h, nsamp, damping):
    if w.ndim != 3 or h.shape != (w.shape[0], w.shape[2], w.shape[2]):
        raise ValueError("incompatible W/H shapes")
    if w.shape[2] % GROUP:
        raise ValueError(f"input width must be divisible by {GROUP}")
    if not torch.isfinite(w).all() or not torch.isfinite(h).all():
        raise ValueError("nonfinite calibration or weights")
    n = nsamp.to(w.device)
    if torch.any((n > 0) & (h.diagonal(dim1=1, dim2=2).sum(-1) <= 0)):
        raise ValueError("positive samples with a zero Hessian")
    keep = (n > 0).nonzero().flatten()
    ww = w[keep].float().clone().contiguous()
    hh = h[keep].float().clone().contiguous()
    diagonal = hh.diagonal(dim1=1, dim2=2)
    dead = diagonal == 0
    ww.masked_fill_(dead[:, None, :], 0)
    diagonal.masked_fill_(dead, 1)
    diagonal.add_(damping * diagonal.mean(-1, keepdim=True))
    hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hh)),
                               upper=True).contiguous()
    return keep, ww, hi


@torch.no_grad()
def gptq_batch(w: torch.Tensor, h: torch.Tensor, nsamp: torch.Tensor, bits: int = 3,
               sym: bool = True, mse: bool = True, damping: float = 0.01,
               metrics: bool = True):
    """Experts with zero samples fall back to RTN (same scale search)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    e, o, i = w.shape
    maxq = 2 ** bits - 1
    codes, scale, zero = rtn_batch(w, bits, sym, mse)
    keep, ww, hi = _prepare(w, h, nsamp, damping)
    ee = ww.shape[0]
    if ee:
        c = torch.empty_like(ww, dtype=torch.uint8)
        s = torch.empty((ee, o, i // GROUP), device=w.device, dtype=torch.float32)
        z = torch.empty_like(s)
        errors = torch.empty((ee, o, GROUP), device=w.device, dtype=torch.float32)
        for start in range(0, i, GROUP):
            g = start // GROUP
            s[:, :, g], z[:, :, g] = find_params(ww[:, :, start:start + GROUP], bits, sym, mse)
            _block[(ee, triton.cdiv(o, 8))](
                ww, hi, s, z, c, errors, o, i, start, maxq, GROUP, 8,
                num_warps=4, enable_fp_fusion=False)
            end = start + GROUP
            if end < i:
                ww[:, :, end:] -= errors @ hi[:, start:end, end:]
        codes[keep], scale[keep], zero[keep] = c, s, z
    result = {"n_gptq_experts": int(ee), "n_rtn_experts": e - int(ee)}
    if metrics:
        w32, h32 = w.float(), h.float()
        dq = dequant_batch(codes, scale, zero)
        rdq = dequant_batch(*rtn_batch(w, bits, sym, mse))
        denom = ((w32 @ h32) * w32).sum().clamp_min(1e-20)
        diff, rdiff = dq - w32, rdq - w32
        result.update(rel_h_eval=float(((diff @ h32) * diff).sum() / denom),
                      rtn_h_eval=float(((rdiff @ h32) * rdiff).sum() / denom),
                      rel_w=float(diff.square().sum() / w32.square().sum().clamp_min(1e-20)))
    return codes, scale, zero, result


@torch.no_grad()
def gptq_reference(w: torch.Tensor, h: torch.Tensor, nsamp: torch.Tensor, bits: int = 3,
                   sym: bool = True, mse: bool = True, damping: float = 0.01):
    """Column-by-column torch GPTQ (no Triton) for parity tests."""
    e, o, i = w.shape
    maxq = 2 ** bits - 1
    codes, scale, zero = rtn_batch(w, bits, sym, mse)
    keep, W, Hinv = _prepare(w, h, nsamp, damping)
    if W.shape[0]:
        C = torch.zeros_like(W, dtype=torch.uint8)
        S = torch.zeros(W.shape[0], o, i // GROUP, device=w.device)
        Z = torch.zeros_like(S)
        for i1 in range(0, i, GROUP):
            i2 = i1 + GROUP
            g = i1 // GROUP
            S[:, :, g], Z[:, :, g] = find_params(W[:, :, i1:i2], bits, sym, mse)
            W1 = W[:, :, i1:i2].clone()
            E1 = torch.zeros_like(W1)
            Hi = Hinv[:, i1:i2, i1:i2]
            for j in range(GROUP):
                q, dq = _qdq(W1[:, :, j], S[:, :, g], Z[:, :, g], maxq)
                C[:, :, i1 + j] = q.to(torch.uint8)
                err = (W1[:, :, j] - dq) / Hi[:, j, j].unsqueeze(1)
                W1[:, :, j + 1:] -= err.unsqueeze(2) * Hi[:, j, j + 1:].unsqueeze(1)
                E1[:, :, j] = err
            if i2 < i:
                W[:, :, i2:] -= E1 @ Hinv[:, i1:i2, i2:]
        codes[keep], scale[keep], zero[keep] = C, S, Z
    return codes, scale, zero
