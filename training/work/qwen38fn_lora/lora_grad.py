"""LoRA weight gradients without fp32 copies of the activations (``--fast-kernels lora_grad``, trainer_kernels.py).

dA = ds^T x and dB = dY^T s reduce over the T tokens. cuBLAS cannot take them on BF16 operands (a BF16 GEMM with fp32
output whose reduction axis is T silently drops to BF16-level accuracy for some (N, T), see trainer_kernels.LoRABF16),
so the reference paths cast x and dY to fp32 ([T, K] and [T, N] copies) and run fp32 SIMT GEMMs -- ~1.4 s of a 100k-token
RL step. Here one Triton kernel reads the big operand once, in its storage dtype:

    tgrad(P, Q[, SP]) = P^T (Q * SP)    P [T, M] BF16, or e4m3 with fp32 scales SP [T, M / 128]; Q [T, R] fp32 -> [M, R]

Q (times the block scale) is split in registers into BF16 hi + lo (hi + lo = Q to ~2^-16 relative, as LoRABF16 does for
A and B), so the MMA runs [P^T] x [hi | lo] with fp32 accumulation; each program sums one T range and the per-range
partials are added in a fixed order, so reruns are bitwise. ``fp8_rows_times(xq, sx, A)`` is the dense-QAT forward
activation times A^T: x_dq A^T = sum over 128-column blocks of sx * (xq_blk A_blk^T), the same split for A.
"""

import torch
import triton
import triton.language as tl

FP8_BLOCK = 128
# (BM, BT, num_warps, num_stages); every config gives the same result for a fixed split count
TGRAD_CONFIG = {'BM': 128, 'BT': 64, 'num_warps': 4, 'num_stages': 3}
ROWS_CONFIG = {'BT': 128, 'num_warps': 4, 'num_stages': 3}
TARGET_PROGRAMS = 1184  # ~8 per SM on B200: the T split count is chosen to reach it
# and no program sums more rows than this (one long fp32 accumulation over 7.8k rows was 4.5x the fp32 GEMM's error)
MAX_SPLIT_ROWS = 2048


@triton.jit
def _split_bf16(x):
    """fp32 -> BF16 (hi, lo), hi + lo = x to ~2^-16 relative. hi is rounded with integer ops: a .to(bf16).to(fp32) round
    trip can be folded away by the compiler, which would leave lo = 0."""
    b = x.to(tl.uint32, bitcast=True)
    hi = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).to(tl.float32, bitcast=True)
    return hi.to(tl.bfloat16), (x - hi).to(tl.bfloat16)


@triton.jit
def _tgrad_kernel(P, SP, Q, OUT, T, M, rows_per_split, s_pt, s_spt, s_qt,
                  R: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr, SCALED: tl.constexpr):
    pid_m = tl.program_id(0)
    split = tl.program_id(1)
    cols = pid_m * BM + tl.arange(0, BM)
    cmask = cols < M
    rr = tl.arange(0, R)
    t0 = split * rows_per_split
    t1 = tl.minimum(t0 + rows_per_split, T)
    acc = tl.zeros((BM, 2 * R), dtype=tl.float32)
    for t in range(t0, t1, BT):
        rows = t + tl.arange(0, BT)
        rmask = rows < t1
        rows64 = rows.to(tl.int64)
        p = tl.load(P + rows64[:, None] * s_pt + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0)
        q = tl.load(Q + rows64[:, None] * s_qt + rr[None, :], mask=rmask[:, None], other=0.0)
        if SCALED:  # BM divides the 128-column scale block: one scale per row for this program
            sc = tl.load(SP + rows64 * s_spt + (pid_m * BM) // 128, mask=rmask, other=0.0)
            q = q * sc[:, None]
        hi, lo = _split_bf16(q)
        acc = tl.dot(tl.trans(p.to(tl.bfloat16)), tl.reshape(tl.join(hi, lo), (BT, 2 * R)), acc)
    out = OUT + split.to(tl.int64) * M * (2 * R) + cols[:, None] * (2 * R) + tl.arange(0, 2 * R)[None, :]
    tl.store(out, acc, mask=cmask[:, None])


@triton.jit
def _rows_times_kernel(XQ, SX, AHL, S, T, K, s_xt, s_st, s_ot, R: tl.constexpr, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    kk = tl.arange(0, 128)
    jj = tl.arange(0, 2 * R)
    acc = tl.zeros((BT, 2 * R), dtype=tl.float32)
    for kb in range(0, K // 128):
        x = tl.load(XQ + rows64[:, None] * s_xt + kb * 128 + kk[None, :], mask=rmask[:, None], other=0.0)
        a = tl.load(AHL + jj[None, :] * K + kb * 128 + kk[:, None])          # [128, 2R] BF16 (hi, lo interleaved)
        sc = tl.load(SX + rows64 * s_st + kb, mask=rmask, other=0.0)
        acc += tl.dot(x.to(tl.bfloat16), a) * sc[:, None]
    s = tl.sum(tl.reshape(acc, (BT, R, 2)), axis=2)
    tl.store(S + rows64[:, None] * s_ot + tl.arange(0, R)[None, :], s, mask=rmask[:, None])


def _rank(r):
    if r < 8 or r & (r - 1):
        raise ValueError(f'lora_grad: LoRA rank {r} must be a power of two >= 8 (the MMA N dimension is 2r)')
    return r


def tgrad(p, q, sp=None):
    """P^T (Q * SP) -> [M, R] fp32. p [T, M] BF16 (or e4m3 with sp [T, M / 128] fp32), q [T, R] fp32."""
    t, m = p.shape
    r = _rank(q.shape[1])
    if q.shape[0] != t or q.dtype != torch.float32 or p.stride(1) != 1 or q.stride(1) != 1:
        raise ValueError(f'tgrad: p {tuple(p.shape)} {p.stride()}, q {tuple(q.shape)} {q.dtype} {q.stride()}')
    scaled = sp is not None
    if scaled != (p.dtype == torch.float8_e4m3fn) or (not scaled and p.dtype != torch.bfloat16):
        raise ValueError(f'tgrad: p {p.dtype} needs block scales exactly when it is e4m3')
    c = TGRAD_CONFIG
    if scaled and (sp.shape != (t, m // FP8_BLOCK) or m % FP8_BLOCK or FP8_BLOCK % c['BM'] or sp.stride(1) != 1
                   or sp.dtype != torch.float32):
        raise ValueError(f'tgrad: scales {tuple(sp.shape)} {sp.dtype} for p {tuple(p.shape)}, BM {c["BM"]}')
    tiles = triton.cdiv(m, c['BM'])
    splits = max(1, min(triton.cdiv(t, c['BT']), max(triton.cdiv(TARGET_PROGRAMS, tiles), triton.cdiv(t, MAX_SPLIT_ROWS))))
    rows_per_split = triton.cdiv(triton.cdiv(t, splits), c['BT']) * c['BT']
    splits = triton.cdiv(t, rows_per_split)
    part = torch.empty(splits, m, 2 * r, device=p.device, dtype=torch.float32)
    _tgrad_kernel[(tiles, splits)](p, sp if scaled else q, q, part, t, m, rows_per_split, p.stride(0),
                                   sp.stride(0) if scaled else 0, q.stride(0), R=r, BM=c['BM'], BT=c['BT'],
                                   SCALED=scaled, num_warps=c['num_warps'], num_stages=c['num_stages'])
    return part.sum(0).view(m, r, 2).sum(-1)


def fp8_rows_times(xq, sx, a32):
    """x_dq A^T -> [T, R] fp32, x_dq = xq (e4m3 [T, K]) * sx ([T, K / 128] fp32), A [R, K] fp32."""
    t, k = xq.shape
    r = _rank(a32.shape[0])
    if xq.dtype != torch.float8_e4m3fn or sx.shape != (t, k // FP8_BLOCK) or k % FP8_BLOCK or a32.shape[1] != k \
            or a32.dtype != torch.float32 or xq.stride(1) != 1 or sx.stride(1) != 1:
        raise ValueError(f'fp8_rows_times: xq {tuple(xq.shape)} {xq.dtype}, sx {tuple(sx.shape)}, A {tuple(a32.shape)}')
    hi = a32.to(torch.bfloat16)
    ahl = torch.stack([hi, (a32 - hi.float()).to(torch.bfloat16)], dim=1).reshape(2 * r, k).contiguous()
    s = torch.empty(t, r, device=xq.device, dtype=torch.float32)
    c = ROWS_CONFIG
    _rows_times_kernel[(triton.cdiv(t, c['BT']),)](xq, sx, ahl, s, t, k, xq.stride(0), sx.stride(0), s.stride(0),
                                                   R=r, BT=c['BT'], num_warps=c['num_warps'],
                                                   num_stages=c['num_stages'])
    return s
