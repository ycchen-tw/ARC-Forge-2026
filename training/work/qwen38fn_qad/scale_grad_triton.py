"""Fused INT3 group-scale gradient (QAD v2): dS[e, n, g] = sum_{k in g} code[e, n, k] * sum_{r in e} G[r, n] * A[r, k].

The reference (int3_scale_qat.scale_gradients, mode "fp32") forms each expert's [N, K] FP32 weight gradient with a
TF32-off GEMM in a Python loop over experts and reduces it with the codes. This kernel never materializes dW: one
program owns (expert, 128 rows of N, BK columns of K), accumulates G^T A over the expert's rows in FP32 registers
with tensor-core dots, multiplies by the codes and sums each 128-wide K group. The accumulation order is fixed (rows
in routing order, one program per output tile, no atomics), so results are deterministic run to run.

Operands as the backward has them:
  G rows   ``g[g_index[r]] * g_weight[r]`` (w2: token gradient gathered by routing, times the BF16 routing weight) or
           ``g[r]`` (w13: dPre, already in routing order), BF16;
  A rows   ``a_q[r] * a_scale[r]``: the forward's E4M3 activation and its per-row FP32 scale (W3A16: the BF16
           activation itself with a scale of 1, exact in BF16 like the E4M3 values).
The per-row factors (``g_weight * a_scale``) are folded into G in FP32. ``precision``:
  "bf16"  G' rounded to BF16 (relative 2^-9 per element), A exact in BF16 (E4M3 values), BF16 tensor-core dot;
  "bf16x2" G' split into BF16 hi + lo (residual ~2^-17), two BF16 dots against the exact A: near-FP32 operands;
  "tf32"  G' and A as FP32 into a TF32 dot (G' rounded to 10 mantissa bits; A exact);
  "ieee"  FP32 FMA dot, no rounding of the operands (slow; the reference for this kernel).
Rounding of the operands is unbiased per element; the long reductions (over rows, over the group) stay in FP32 in
registers, which is the property that cuBLAS BF16-in / FP32-out reductions over tokens do not guarantee.
"""

import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _scale_grad_kernel(
    g_ptr, g_index_ptr, g_weight_ptr, a_ptr, a_scale_ptr, codes_ptr, ends_ptr, out_ptr,
    N, K,
    stride_g, stride_a, stride_ce, stride_cn, stride_oe, stride_on,
    HAS_INDEX: tl.constexpr, HAS_WEIGHT: tl.constexpr, PRECISION: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    e = tl.program_id(0)
    nb = tl.program_id(1)
    kb = tl.program_id(2)
    lo = tl.load(ends_ptr + e)
    hi = tl.load(ends_ptr + e + 1)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_k = kb * BK + tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for r0 in range(lo, hi, BR):
        rows = r0 + tl.arange(0, BR)
        live = rows < hi
        if HAS_INDEX:
            src = tl.load(g_index_ptr + rows, mask=live, other=0).to(tl.int64)
        else:
            src = rows.to(tl.int64)
        g = tl.load(g_ptr + src[:, None] * stride_g + offs_n[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        mult = tl.load(a_scale_ptr + rows, mask=live, other=0.0)
        if HAS_WEIGHT:
            mult = mult * tl.load(g_weight_ptr + rows, mask=live, other=0.0).to(tl.float32)
        g = g * mult[:, None]
        a = tl.load(a_ptr + rows.to(tl.int64)[:, None] * stride_a + offs_k[None, :], mask=live[:, None], other=0.0)
        if PRECISION == "bf16":
            acc += tl.dot(tl.trans(g.to(tl.bfloat16)), a.to(tl.bfloat16), out_dtype=tl.float32)
        elif PRECISION == "bf16x2":
            g_hi = g.to(tl.bfloat16)
            g_lo = (g - g_hi.to(tl.float32)).to(tl.bfloat16)
            ab = a.to(tl.bfloat16)
            acc += tl.dot(tl.trans(g_hi), ab, out_dtype=tl.float32)
            acc += tl.dot(tl.trans(g_lo), ab, out_dtype=tl.float32)
        elif PRECISION == "tf32":
            acc += tl.dot(tl.trans(g), a.to(tl.float32), input_precision="tf32", out_dtype=tl.float32)
        else:
            acc += tl.dot(tl.trans(g), a.to(tl.float32), input_precision="ieee", out_dtype=tl.float32)
    codes = tl.load(codes_ptr + e.to(tl.int64) * stride_ce + offs_n[:, None] * stride_cn + offs_k[None, :]).to(tl.float32)
    prod = tl.reshape(acc * codes, (BN, BK // 128, 128))
    ds = tl.sum(prod, axis=2)                                                  # [BN, BK / 128]
    offs_g = kb * (BK // 128) + tl.arange(0, BK // 128)
    tl.store(out_ptr + e.to(tl.int64) * stride_oe + offs_n[:, None] * stride_on + offs_g[None, :], ds)


def _scale_tile(k):
    """(bk, num_warps), B200 sweep 2026-10-15 at 1.2M routed rows: K=2560 (w13) 256/8 (28.7 ms vs 43.9 at 128/8),
    K=640 (w2) 128/4 (21.3 ms vs 34.9)."""
    return (256, 8) if k % 256 == 0 and k >= 2048 else (128, 4)


def scale_gradients_fused(g, a_q, a_scale, codes, ends, out, g_index=None, g_weight=None, precision="bf16x2",
                          bn=128, bk=None, br=64, num_warps=None, num_stages=3):
    """Writes out[e, n, k / 128] (fp32) for every expert; experts without rows get 0. See the module docstring."""
    if precision not in ("bf16", "bf16x2", "tf32", "ieee"):
        raise ValueError(precision)
    e, n, k = codes.shape
    tile_bk, tile_warps = _scale_tile(k)
    bk = bk or tile_bk
    num_warps = num_warps or tile_warps
    if g.dtype != torch.bfloat16 or g.shape[1] != n or not g.is_contiguous():
        raise TypeError(f"G must be contiguous BF16 [*, {n}], got {g.dtype} {tuple(g.shape)}")
    if a_q.dtype not in (torch.float8_e4m3fn, torch.bfloat16) or a_q.shape[1] != k or not a_q.is_contiguous():
        raise TypeError(f"A must be contiguous E4M3 or BF16 [rows, {k}], got {a_q.dtype} {tuple(a_q.shape)}")
    if a_scale.dtype != torch.float32 or a_scale.shape != (a_q.shape[0],):
        raise TypeError("a_scale must be fp32 [rows]")
    if codes.dtype != torch.float8_e4m3fn or not codes.is_contiguous():
        raise TypeError("codes must be contiguous E4M3 [E, N, K]")
    if out.dtype != torch.float32 or out.shape != (e, n, k // GROUP) or not out.is_contiguous():
        raise TypeError(f"out must be contiguous fp32 [{e}, {n}, {k // GROUP}]")
    if n % bn or k % bk or bk % GROUP:
        raise ValueError(f"N={n} / K={k} not divisible by the tile ({bn}, {bk})")
    if ends.dtype != torch.int32 or ends.shape != (e + 1,):
        raise TypeError("ends must be int32 [E + 1]")
    if g_index is not None and (g_index.dtype != torch.int32 or g_index.shape != (a_q.shape[0],)):
        raise TypeError("g_index must be int32 [rows]")
    if g_index is None and g.shape[0] != a_q.shape[0]:
        raise ValueError("G and A rows differ and no g_index maps them")
    if g_weight is not None and g_weight.shape != (a_q.shape[0],):
        raise TypeError("g_weight must be [rows]")
    dummy = a_scale
    _scale_grad_kernel[(e, n // bn, k // bk)](
        g, g_index if g_index is not None else dummy, g_weight if g_weight is not None else dummy,
        a_q, a_scale, codes, ends, out, n, k,
        g.stride(0), a_q.stride(0), codes.stride(0), codes.stride(1), out.stride(0), out.stride(1),
        HAS_INDEX=g_index is not None, HAS_WEIGHT=g_weight is not None, PRECISION=precision,
        BN=bn, BK=bk, BR=br, num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _dense_dw_kernel(
    dy_ptr, xq_ptr, sx_ptr, part_ptr, T, N, K, rows_per_split,
    stride_dy, stride_x, stride_sx, stride_ps, stride_pn,
    PRECISION: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    nb = tl.program_id(0)
    kb = tl.program_id(1)
    sp = tl.program_id(2)
    lo = sp * rows_per_split
    hi = tl.minimum(lo + rows_per_split, T)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_k = kb * BK + tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for r0 in range(lo, hi, BR):
        rows = r0 + tl.arange(0, BR)
        live = rows < hi
        r64 = rows.to(tl.int64)
        g = tl.load(dy_ptr + r64[:, None] * stride_dy + offs_n[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        g = g * tl.load(sx_ptr + r64 * stride_sx + kb, mask=live, other=0.0)[:, None]   # BK == 128: one block scale
        a = tl.load(xq_ptr + r64[:, None] * stride_x + offs_k[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        g_hi = g.to(tl.bfloat16)
        acc += tl.dot(tl.trans(g_hi), a, out_dtype=tl.float32)
        if PRECISION == "bf16x2":
            g_lo = (g - g_hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(g_lo), a, out_dtype=tl.float32)
    tl.store(part_ptr + sp.to(tl.int64) * stride_ps + offs_n[:, None] * stride_pn + offs_k[None, :], acc)


def dense_weight_grad(dy, xq, sx, splits=None, precision="bf16x2", bn=128, br=128, num_warps=8, num_stages=3):
    """dW [N, K] fp32 = dY^T @ (x_q * s_x) for a block-FP8 projection's backward (STE through the activation
    quantizer). ``x_q`` E4M3 [T, K] with an fp32 scale per token and 128-column block (s_x [T, K/128], SGLang's 1x128
    quantizer); a program's K tile is one block, whose scale is folded into dY in FP32 and split
    into BF16 hi + lo (bf16x2: near-FP32 operand; "bf16": hi only, 2^-9 per element), the E4M3 values are exact in
    BF16. Rows are cut into ``splits``
    contiguous ranges reduced in FP32 registers and the partials summed in a fixed order (deterministic)."""
    if precision not in ("bf16", "bf16x2"):
        raise ValueError(precision)
    t, n = dy.shape
    k = xq.shape[1]
    bk = 128
    if dy.dtype != torch.bfloat16 or not dy.is_contiguous() or xq.dtype != torch.float8_e4m3fn or not xq.is_contiguous():
        raise TypeError(f"dy {dy.dtype}, xq {xq.dtype}: need contiguous BF16 / E4M3")
    if xq.shape[0] != t or sx.shape != (t, k // 128) or sx.dtype != torch.float32:
        raise TypeError(f"x_q [T, K] and fp32 s_x [T, K/128] must match dY's rows, got s_x {tuple(sx.shape)}")
    if n % bn or k % bk:
        raise ValueError(f"N={n} / K={k} not divisible by the tile ({bn}, {bk})")
    tiles = (n // bn) * (k // bk)
    if splits is None:   # enough programs to fill the GPU, at least ~4096 rows each
        sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
        splits = max(1, min(triton.cdiv(2 * sms, tiles), triton.cdiv(t, 4096)))
    per = triton.cdiv(triton.cdiv(t, splits), br) * br
    splits = triton.cdiv(t, per)
    part = torch.empty((splits, n, k), dtype=torch.float32, device=dy.device)
    _dense_dw_kernel[(n // bn, k // bk, splits)](
        dy, xq, sx, part, t, n, k, per, dy.stride(0), xq.stride(0), sx.stride(0), part.stride(0), part.stride(1),
        PRECISION=precision, BN=bn, BK=bk, BR=br, num_warps=num_warps, num_stages=num_stages,
    )
    return part[0] if splits == 1 else part.sum(0)


# ------------------------------------------------------------------------------------------- quack (tcgen05) path
# The Triton kernels above run tcgen05 MMAs but convert the operand in registers every K step (scale multiply, BF16
# cast, STS, barrier), which serializes load / convert / MMA: ~15% of the BF16 peak at 1 CTA per SM. The quack path
# (``--grad-kernels quack``) instead forms the scaled operand once in a memory-bound Triton pre-pass and hands both
# operands to quack's SM100 GEMM (Tri Dao's CuTe DSL: TMA, warp-specialized, persistent tcgen05; in the training venv),
# split along the reduction axis with its varlen_k (one segment per T split or per expert), FP32 outputs reduced in a
# fixed order. Same rounding as the Triton kernels' "bf16" / "bf16x2" (the scaled operand rounded to BF16 once, or
# hi + lo), deterministic (each output tile is one CTA's fixed-order K loop). B200, T = 120k (kernel probe 10-15,
# $ARC_RUNS/qwen38fn_qad/20261015-dw-tcgen05): dense 10240x2560 18.5 -> 4.4 ms, w13 scales 18.4 -> 7.6 ms,
# w2 15.8 -> 4.0 ms. bf16x2 needs the T split (>= 4 segments): one FP32 accumulator over 120k rows drifts to ~1e-4.


@triton.jit
def _scaled_rows_kernel(x_ptr, s_ptr, w_ptr, hi_ptr, lo_ptr, R, K, s_stride, s_col_step,
                        HAS_W: tl.constexpr, TWO: tl.constexpr, BR: tl.constexpr):
    """hi (+ lo) = bf16(x[r, k] * s[r, k // 128 * s_col_step] (* w[r])) for one 128-column block."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    kb = tl.program_id(1)
    cols = kb * 128 + tl.arange(0, 128)
    live = rows < R
    r64 = rows.to(tl.int64)
    x = tl.load(x_ptr + r64[:, None] * K + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    s = tl.load(s_ptr + r64 * s_stride + kb * s_col_step, mask=live, other=0.0)
    if HAS_W:
        s = s * tl.load(w_ptr + rows, mask=live, other=0.0).to(tl.float32)
    v = x * s[:, None]
    h = v.to(tl.bfloat16)
    tl.store(hi_ptr + r64[:, None] * K + cols[None, :], h, mask=live[:, None])
    if TWO:
        tl.store(lo_ptr + r64[:, None] * K + cols[None, :], (v - h.to(tl.float32)).to(tl.bfloat16), mask=live[:, None])


def _scaled_rows(x, scale, weight, two, per_block):
    """BF16 hi (and lo) of x * scale (* weight): scale [R, K/128] per block, or [R] per row."""
    r, k = x.shape
    hi = torch.empty(r, k, dtype=torch.bfloat16, device=x.device)
    lo = torch.empty_like(hi) if two else hi
    if r:
        _scaled_rows_kernel[(triton.cdiv(r, 64), k // 128)](
            x, scale, weight if weight is not None else scale, hi, lo, r, k,
            scale.stride(0) if per_block else 1, 1 if per_block else 0,
            HAS_W=weight is not None, TWO=two, BR=64, num_warps=4)
    return hi, lo


@triton.jit
def _sum_parts_kernel(p_ptr, o_ptr, S, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(S):
        acc += tl.load(p_ptr + s * numel + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, acc, mask=m)


@triton.jit
def _code_reduce_kernel(dw_ptr, dw2_ptr, c_ptr, o_ptr, NG, TWO: tl.constexpr, BG: tl.constexpr):
    grp = tl.program_id(0).to(tl.int64) * BG + tl.arange(0, BG)
    live = grp < NG
    offs = grp[:, None] * 128 + tl.arange(0, 128)[None, :]
    d = tl.load(dw_ptr + offs, mask=live[:, None], other=0.0)
    if TWO:
        d += tl.load(dw2_ptr + offs, mask=live[:, None], other=0.0)
    c = tl.load(c_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
    tl.store(o_ptr + grp, tl.sum(d * c, axis=1), mask=live)


def _quack_config(n):
    from quack.gemm_config import GemmConfig

    tile_n = 128 if n <= 512 else 256
    return GemmConfig(tile_m=256, tile_n=tile_n, cluster_m=2, cluster_n=1, pingpong=False, is_dynamic_persistent=True,
                      device_capacity=10)


def dense_weight_grad_quack(dy, xq, sx, precision="bf16", splits=None):
    """dense_weight_grad on quack: x' = bf16(x_q * s_x) (+ lo), dW = dY^T x' with the T axis split into ``splits``
    varlen_k segments (FP32 partials, fixed-order sum). Default splits: 1 for N x K >= 10240 x 2560 at bf16
    (enough tiles), 16 for N <= 512, else 4; bf16x2 never below 4 (accumulator drift)."""
    from quack.gemm_interface import gemm_tuned

    if precision not in ("bf16", "bf16x2"):
        raise ValueError(precision)
    t, n = dy.shape
    k = xq.shape[1]
    if dy.dtype != torch.bfloat16 or not dy.is_contiguous() or xq.dtype != torch.float8_e4m3fn or not xq.is_contiguous():
        raise TypeError(f"dy {dy.dtype}, xq {xq.dtype}: need contiguous BF16 / E4M3")
    if xq.shape[0] != t or sx.shape != (t, k // 128) or sx.dtype != torch.float32 or k % 128:
        raise TypeError(f"x_q [T, K] and fp32 s_x [T, K/128] must match dY's rows, got s_x {tuple(sx.shape)}")
    two = precision == "bf16x2"
    if splits is None:
        splits = 16 if n <= 512 else 1 if n * k >= 10240 * 2560 else 4
        if two:
            splits = max(splits, 4)
    hi, lo = _scaled_rows(xq, sx, None, two, per_block=True)
    per = triton.cdiv(triton.cdiv(t, splits), 128) * 128
    bounds = torch.tensor(list(range(0, t, per)) + [t], dtype=torch.int32, device=dy.device)
    s = bounds.numel() - 1
    cfg = _quack_config(n)
    part = torch.empty((2 * s if two else s, n, k), dtype=torch.float32, device=dy.device)
    a = dy.t()                                  # (N, T): T is the reduction (varlen_k) axis
    gemm_tuned.fn(a, hi, part[:s], cu_seqlens_k=bounds, config=cfg)
    if two:
        gemm_tuned.fn(a, lo, part[s:], cu_seqlens_k=bounds, config=cfg)
    if part.shape[0] == 1:
        return part[0]
    out = torch.empty((n, k), dtype=torch.float32, device=dy.device)
    _sum_parts_kernel[(triton.cdiv(n * k, 2048),)](part, out, part.shape[0], n * k, BLOCK=2048, num_warps=8)
    return out


def scale_gradients_quack(g, a_q, a_scale, codes, ends, out, g_index=None, g_weight=None, precision="bf16",
                          chunk=None, a_index=None):
    """scale_gradients_fused on quack: B' = bf16(a_q * a_scale (* g_weight)), dW_e = G_e^T B'_e for every expert as
    one varlen_k GEMM (segments = the experts' row ranges; w2 gathers its token rows with A_idx), then
    out[e, n, g] = sum_{k in g} code * dW_e. Experts are processed ``chunk`` at a time to bound the FP32 dW buffer
    (default: ~2 GB). Experts without rows get exactly 0. ``a_index`` (int32 [rows], W3A16 w13): row r of A is
    ``a_q[a_index[r]]``, gathered one chunk at a time (no [rows, K] copy of the token rows), with a_scale per row r."""
    from quack.gemm_interface import gemm_tuned

    if precision not in ("bf16", "bf16x2"):
        raise ValueError(precision)
    e, n, k = codes.shape
    if g.dtype != torch.bfloat16 or g.shape[1] != n or not g.is_contiguous():
        raise TypeError(f"G must be contiguous BF16 [*, {n}], got {g.dtype} {tuple(g.shape)}")
    if a_q.dtype not in (torch.float8_e4m3fn, torch.bfloat16) or a_q.shape[1] != k or not a_q.is_contiguous() \
            or k % 128:
        raise TypeError(f"A must be contiguous E4M3 or BF16 [rows, {k}], got {a_q.dtype} {tuple(a_q.shape)}")
    if a_index is not None and (a_index.dtype != torch.int32 or a_index.shape != a_scale.shape):
        raise TypeError("a_index must be int32 [rows], one source row per A row")
    if a_index is None and a_scale.shape != (a_q.shape[0],):
        raise TypeError("a_scale must be [rows of A]")
    if out.shape != (e, n, k // 128) or out.dtype != torch.float32 or not out.is_contiguous():
        raise TypeError(f"out must be contiguous fp32 [{e}, {n}, {k // 128}]")
    if (g_index is None) != (g_weight is None):
        raise TypeError("g_index and g_weight come together (w2) or not at all (w13)")
    two = precision == "bf16x2"
    chunk = chunk or max(1, (2 << 30) // (n * k * 4 * (2 if two else 1)))
    bounds = ends.tolist()
    cfg = _quack_config(n)
    a = g.t()
    for e0 in range(0, e, chunk):
        e1 = min(e, e0 + chunk)
        r0, r1 = bounds[e0], bounds[e1]
        rows_a = a_q[r0:r1] if a_index is None else a_q.index_select(0, a_index[r0:r1].long())
        hi, lo = _scaled_rows(rows_a, a_scale[r0:r1], g_weight[r0:r1] if g_weight is not None else None, two,
                              per_block=False)
        del rows_a
        cu = (ends[e0:e1 + 1] - r0).to(torch.int32).contiguous()
        m = e1 - e0
        dw = torch.empty(((2 if two else 1) * m, n, k), dtype=torch.float32, device=g.device)
        if g_index is None:
            a_c = a[:, r0:r1]
            gemm_tuned.fn(a_c, hi, dw[:m], cu_seqlens_k=cu, config=cfg)
            if two:
                gemm_tuned.fn(a_c, lo, dw[m:], cu_seqlens_k=cu, config=cfg)
        else:
            idx = g_index[r0:r1].to(torch.int32).clone()   # a fresh, 16-byte aligned index buffer for the TMA path
            gemm_tuned.fn(a, hi, dw[:m], cu_seqlens_k=cu, A_idx=idx, config=cfg)
            if two:
                gemm_tuned.fn(a, lo, dw[m:], cu_seqlens_k=cu, A_idx=idx, config=cfg)
        groups = m * n * (k // 128)
        _code_reduce_kernel[(triton.cdiv(groups, 32),)](dw, dw[m:] if two else dw, codes[e0:e1], out[e0:e1], groups,
                                                        TWO=two, BG=32, num_warps=4)
    return out
