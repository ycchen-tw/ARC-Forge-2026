"""Bandwidth-oriented bf16 GEMM for decode (M <= 128): y[M,N] = x[M,K] @ W[N,K]^T.

cuBLAS picks wmma/split-K kernels that reach only ~50-70% of L4 DRAM bandwidth at
M=40. Here every CTA streams a [BLOCK_N, K-slice] tile of W exactly once, keeps the
small x in L2, accumulates in fp32. Split-K (fp32 atomics into a workspace, then cast)
gives enough CTAs for small-N/large-K shapes. Configs are fixed per shape (no runtime
autotune: autotune allocations caused OOMs on the 24 GB cards).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, y_ptr, ws_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   SPLIT_K: tl.constexpr, K_PER_SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0, eviction_policy="evict_first")
        acc += tl.dot(x, tl.trans(w))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)
    else:
        tl.atomic_add(ws_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask, sem="relaxed")


def choose(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) heuristic: target >= ~2 waves of CTAs on 58 SMs."""
    block_m = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    block_n = 64
    block_k = 128
    n_tiles = triton.cdiv(N, block_n) * triton.cdiv(M, block_m)
    split = 1
    while n_tiles * split < 116 and K // (split * 2) >= 512:
        split *= 2
    return block_m, block_n, block_k, split


def skinny_matmul(x: torch.Tensor, w: torch.Tensor, cfg: tuple[int, int, int, int] | None = None,
                  ws: torch.Tensor | None = None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.dim() == 2 and w.dim() == 2
    assert x.stride(1) == 1 and w.stride(1) == 1
    M, K = x.shape
    N = w.shape[0]
    bm, bn, bk, split = cfg or choose(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    k_per = triton.cdiv(triton.cdiv(K, split), bk) * bk
    if split > 1:
        if ws is None or ws.numel() < M * N:
            ws = torch.zeros((M, N), dtype=torch.float32, device=x.device)
        else:
            ws = ws[: M * N].view(M, N)
            ws.zero_()
    grid = (triton.cdiv(N, bn), split, triton.cdiv(M, bm))
    _skinny_kernel[grid](x, w, y, ws if split > 1 else y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
                         BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, K_PER_SPLIT=k_per,
                         num_warps=4, num_stages=4)
    if split > 1:
        y.copy_(ws)
    return y


_WS = {}


def load_table(path):
    """JSON list of {"N","K","bn","split","warps","stages"} from bench_skinny (only shapes that won)."""
    import json
    return {(int(r["N"]), int(r["K"])): (int(r["bn"]), int(r["split"]), int(r["warps"]), int(r["stages"]))
            for r in json.load(open(path))}


def skinny_linear(x: torch.Tensor, w: torch.Tensor, cfg) -> torch.Tensor:
    """x[..., K] @ w[N, K]^T with a fixed per-shape config; x leading dims flattened (M <= 128)."""
    bn, split, warps, stages = cfg
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, K = x2.shape
    N = w.shape[0]
    bm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    k_per = triton.cdiv(triton.cdiv(K, split), 128) * 128
    ws = y
    if split > 1:
        key = (x.device.index, torch.cuda.current_stream().cuda_stream)
        buf = _WS.get(key)
        if buf is None or buf.numel() < M * N:
            # sized for the largest decode shape up front (graph capture must not reallocate)
            buf = torch.empty(max(M * N, 128 * 16384), dtype=torch.float32, device=x.device)
            _WS[key] = buf
        ws = buf[: M * N].view(M, N)
        ws.zero_()
    grid = (triton.cdiv(N, bn), split, triton.cdiv(M, bm))
    _skinny_kernel[grid](x2, w, y, ws, M, N, K, x2.stride(0), w.stride(0), y.stride(0),
                         BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, SPLIT_K=split, K_PER_SPLIT=k_per,
                         num_warps=warps, num_stages=stages)
    if split > 1:
        y.copy_(ws)
    return y.view(*lead, N)
