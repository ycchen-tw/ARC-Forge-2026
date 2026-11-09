"""The verify lm_head GEMM (rows x 2560 @ (124,160 x 2560)^T, bf16, fp32 accumulation) for verify-size row counts on
sm_89: one program per (N block, M block), the M blocks of an N block adjacent so its weight block is read from DRAM
once. L4 (10-11 lmhead_l4.py, spaced single-call graphs): 192 rows, cuBLAS (cutlass 256x128_32x3, legacy and Lt
alike) 3,016 us = 211 GB/s of weight, this kernel at BM 64 / BN 64 / BK 128 / 4 warps / 4 stages 2,765 us = 230 GB/s;
at 384 rows cuBLAS wins (3,430 vs 3,644).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lm_head_kernel(x_ptr, w_ptr, o_ptr, M, N, K, NUM_M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    pid = tl.program_id(0)
    pn = pid // NUM_M
    pm = pid % NUM_M
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(x_ptr + om[:, None] * K + ok[None, :], mask=om[:, None] < M, other=0.0)
        b = tl.load(w_ptr + on[:, None] * K + ok[None, :], mask=on[:, None] < N, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(o_ptr + om[:, None] * N + on[None, :], acc.to(tl.bfloat16), mask=(om[:, None] < M) & (on[None, :] < N))


CFG = {"BM": 64, "BN": 64, "BK": 128, "num_warps": 4, "num_stages": 4}


def lm_head_gemm(x: torch.Tensor, w: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """x [M, K] bf16 contiguous, w [N, K] bf16 contiguous (K a multiple of BK) -> x @ w^T [M, N] bf16."""
    c = cfg or CFG
    if (x.dim() != 2 or w.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16
            or not x.is_contiguous() or not w.is_contiguous() or x.shape[1] != w.shape[1] or x.shape[1] % c["BK"]):
        raise ValueError(f"lm_head_gemm: x {tuple(x.shape)} {x.dtype} w {tuple(w.shape)} {w.dtype}")
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    num_m = triton.cdiv(M, c["BM"])
    _lm_head_kernel[(num_m * triton.cdiv(N, c["BN"]),)](
        x, w, out, M, N, K, NUM_M=num_m, BM=c["BM"], BN=c["BN"], BK=c["BK"], num_warps=c["num_warps"],
        num_stages=c["num_stages"])
    return out
