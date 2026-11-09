"""Weight-only INT8 linear for decode (sm_89): y[M,N] = x[M,K] (bf16) @ dequant(Wq[N,K] int8, s[N,K/128])^T.

Symmetric per-(row, 128-column group) scales (RTN). Activations stay BF16 (no activation quantization, unlike the
block-FP8 W8A8 path). Each CTA streams a [BLOCK_N, K-slice] tile of int8 weights once, dequantizes in registers and
accumulates in fp32; split-K partials go through an fp32 workspace. Configs are fixed per call (no autotune).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP = 128  # default; K % 128 != 0 (e.g. hyper-connection up, K=320) uses 64


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, ws_ptr, M, N, K, stride_xm, stride_wn, stride_sn, stride_ym,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr, K_PER_SPLIT: tl.constexpr,
                  G: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, G):
        kb = k_start + k0
        offs_k = kb + tl.arange(0, G)
        kmask = offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0, eviction_policy="evict_first")
        s = tl.load(s_ptr + offs_n * stride_sn + kb // G, mask=(offs_n < N) & (kb < K), other=0.0)
        wd = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(wd))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)
    else:
        tl.atomic_add(ws_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask, sem="relaxed")


@torch.no_grad()
def group_of(k: int) -> int:
    for g in (128, 64, 32):
        if k % g == 0:
            return g
    raise ValueError(f"K={k} is not a multiple of 32")


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16 -> ([N, K] int8, [N, K/g] fp32 scales), symmetric RTN per (row, g columns), g = group_of(K)."""
    n, k = weight.shape
    g = group_of(k)
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    s = torch.empty((n, k // g), dtype=torch.float32, device=weight.device)
    for r in range(0, n, 8192):
        w = weight[r:r + 8192].float().view(-1, k // g, g)
        amax = w.abs().amax(dim=-1)
        sc = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        q[r:r + 8192] = torch.round(w / sc[..., None]).clamp(-127, 127).to(torch.int8).view(-1, k)
        s[r:r + 8192] = sc
    return q, s


def pick(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """(BLOCK_N, SPLIT_K, warps, stages) for the default tile. Shared memory per stage is BLOCK_M*G*2 + BLOCK_N*G
    bytes (sm_89 opt-in limit 99 KiB), so the 128-row tile gets 2 stages. Split-K only for narrow N in decode."""
    bm = block_m(M)
    bn = 64
    g = group_of(K)
    split = 1
    if M <= 128:
        tiles = triton.cdiv(N, bn)
        while tiles * split < 116 and K // (split * 2) >= 512 and (K // (split * 2)) % g == 0:
            split *= 2
    stages = 2 if bm >= 128 else 3
    return bn, split, 4, stages


def block_m(M: int) -> int:
    # decode: one M tile up to 128 rows (a second tile would stream every weight byte twice);
    # prefill (M > 128) is compute-heavy, tile M by 64 instead
    return 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128 if M <= 128 else 64


_WS: dict = {}


def w8a16_linear(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor, cfg=None) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, K = x2.shape
    N = wq.shape[0]
    bn, split, warps, stages = cfg or pick(M, N, K)
    bm = block_m(M)
    g = group_of(K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    k_per = triton.cdiv(triton.cdiv(K, split), g) * g
    ws = y
    if split > 1:
        key = (x.device.index, torch.cuda.current_stream().cuda_stream)
        buf = _WS.get(key)
        if buf is None or buf.numel() < M * N:
            buf = torch.empty(max(M * N, 256 * 16384), dtype=torch.float32, device=x.device)
            _WS[key] = buf
        ws = buf[: M * N].view(M, N)
        ws.zero_()
    grid = (triton.cdiv(N, bn), split, triton.cdiv(M, bm))
    _w8a16_kernel[grid](x2, wq, scale, y, ws, M, N, K, x2.stride(0), wq.stride(0), scale.stride(0), y.stride(0),
                        BLOCK_M=bm, BLOCK_N=bn, SPLIT_K=split, K_PER_SPLIT=k_per, G=g, num_warps=warps,
                        num_stages=stages)
    if split > 1:
        y.copy_(ws)
    return y.view(*lead, N)
