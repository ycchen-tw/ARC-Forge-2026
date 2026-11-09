"""Bit-exact CPU reproduction of CUDA ``upsample_bicubic2d_aa`` (float32): the deployed HF processor does the bicubic +
antialias resize on GPU; ``prepare_tito_mm --device cpu-exact`` uses this to get the same uint8 frames without a GPU.

Follows ``upsample_gen2d_aa_out_frame`` in torch 2.13.0 (cf30153c) ``aten/src/ATen/native/cuda/UpSampleBilinear2d.cu``
and ``upsample_antialias::{BicubicFilterFunctor,_compute_weights_span,_compute_weights,interpolate_aa_single_dim}`` in
``UpSample.cuh``; the scale is the host-side ``area_pixel_compute_scale<float>`` (align_corners=False, no
scale_factor → ``(float)in / out``). accscalar_t = float, so every step is a float32 IEEE operation; nvcc's default
``--fmad=true`` contracts ``a * b + c`` into an FMA (single rounding; it even contracts the adds/subtracts that follow
``center = scale * (i + 0.5)`` across statements). Here float32 FMA is emulated with an exact float64 product + TwoSum
+ round-to-odd (53 ≥ 24+2, so the double rounding is correct). Which multiply-adds get contracted was determined by
measuring against frames produced on the GPU (see ``axis_weights``), not read from PTX. Per output pixel the order
follows the kernel: for each row in ``ymin..ymin+ysize``, first a horizontal FMA chain over ``xsize`` taps (stored as
float), then a vertical FMA chain over ``ysize`` taps on those values. The horizontal result depends only on (input
row, output column), so it is computed once for all input rows.

Covers only float32, NCHW, bicubic, antialias=True, align_corners=False, size given (no scale_factor); anything else
raises.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32
F64 = np.float64
A = F32(-0.5)                   # the a of BicubicFilterFunctor
SIZE = 4                        # BicubicFilterFunctor::size
HALF = F32(0.5)


def fma32(a, b, c) -> np.ndarray:
    """float32 fma(a, b, c): a single rounding. a·b is exact in float64 (24+24 bits); TwoSum on p + c gives the exact
    sum s + e, then s is made round-to-odd (if inexact and the mantissa is even, move one ulp toward e); the final
    float64 → float32 is then a correct RNE."""
    a, b, c = (np.asarray(x, dtype=F32) for x in (a, b, c))
    p = a.astype(F64) * b.astype(F64)
    cc = c.astype(F64)
    s = p + cc
    bb = s - p
    e = (p - (s - bb)) + (cc - bb)
    bits = s.view(np.int64)
    fix = (e != 0) & ((bits & 1) == 0)
    if fix.any():
        step = np.where(np.signbit(e) == np.signbit(s), 1, -1).astype(np.int64)
        bits = np.where(fix, bits + step, bits)
    if not np.isfinite(s).all():
        raise FloatingPointError("fma32: non-finite value")
    return bits.view(F64).astype(F32)


def _filter(x: np.ndarray) -> np.ndarray:
    """BicubicFilterFunctor (a=−0.5): |x|<1: ((a+2)x−(a+3))·x·x+1; |x|<2: (((x−5)x+8)x−4)·a; else 0.
    nvcc's contraction: fma(1.5,x,−2.5) → times x → fma(·,x,1); x−5 → fma(·,x,8) → fma(·,x,−4) → times a (multiplying
    by −0.5 is exact)."""
    x = np.abs(x)                                    # if (x < 0) x = -x; (−0 and +0 give the same result)
    near = fma32(fma32(F32(1.5), x, F32(-2.5)) * x, x, F32(1.0))
    far = fma32(fma32(x - F32(5), x, F32(8)), x, F32(-4)) * A
    return np.where(x < 1, near, np.where(x < 2, far, F32(0))).astype(F32)


def axis_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xmin, xsize, weights[out, k]) for one axis: ``_compute_weights_span`` and ``_compute_weights`` (float32)."""
    if in_size <= 0 or out_size <= 0 or in_size >= 2 ** 24 or out_size >= 2 ** 24:
        raise ValueError(f"size {in_size}→{out_size}")
    scale = F32(in_size) / F32(out_size)             # area_pixel_compute_scale<float>: (float)src / dst
    support = F32(SIZE) * HALF * scale if scale >= 1.0 else F32(SIZE) * HALF
    ih = np.arange(out_size, dtype=F32) + HALF       # i + 0.5f (exact)
    # center = scale * ih is not rounded on its own on the GPU: all three of its uses (center − support,
    # center + support, xmin − center) are contracted into FMAs by nvcc. The uncontracted version (center rounded to
    # float first) is off by 13 / 1,870 uint8 values on 1170→1184 and 392→384; contracted, 0 differences.
    xmin = np.maximum(np.trunc(fma32(scale, ih, -support) + HALF).astype(np.int64), 0)   # static_cast<int>: truncation toward zero
    xsize = np.minimum(np.trunc(fma32(scale, ih, support) + HALF).astype(np.int64), in_size) - xmin
    if (xsize <= 0).any():
        raise RuntimeError(f"{in_size}→{out_size}: xsize ≤ 0")
    xmc = fma32(-scale, ih, xmin.astype(F32))       # xmin - xcenter
    invscale = F32(1.0 / F64(scale)) if scale >= 1.0 else F32(1.0)   # 1.0 / scale is a double division, then stored as float
    k = int(xsize.max())
    w = np.zeros((out_size, k), dtype=F32)
    total = np.zeros(out_size, dtype=F32)
    for j in range(k):
        valid = j < xsize
        wj = _filter(((F32(j) + xmc) + HALF) * invscale)   # (j + xmin_m_center + 0.5f) * invscale
        wj = np.where(valid, wj, F32(0)).astype(F32)
        w[:, j] = wj
        total = np.where(valid, total + wj, total).astype(F32)      # total_w += w (in j order)
    nz = total != 0
    w = np.where(nz[:, None] & (np.arange(k)[None, :] < xsize[:, None]), w / np.where(nz, total, F32(1))[:, None], w)
    return xmin, xsize, w.astype(F32)


def _interp_last_axis(src: np.ndarray, xmin: np.ndarray, xsize: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``interpolate_aa_single_dim`` along the last axis: out = t0·w0; then out = fma(t_j, w_j, out) for j < xsize."""
    out = src[..., xmin] * w[:, 0]
    for j in range(1, w.shape[1]):
        valid = j < xsize
        idx = np.where(valid, xmin + j, xmin)
        out = np.where(valid, fma32(src[..., idx], w[:, j], out), out)
    return out.astype(F32)


def upsample_bicubic2d_aa(x: np.ndarray, out_h: int, out_w: int, planes_per_chunk: int = 8) -> np.ndarray:
    """float32 NCHW → float32 N×C×out_h×out_w, bit-identical to CUDA ``_upsample_bicubic2d_aa(align_corners=False)``."""
    if x.dtype != F32 or x.ndim != 4:
        raise TypeError(f"expected float32 NCHW, got {x.dtype} {x.shape}")
    n, c, h, wd = x.shape
    ymin, ysize, wy = axis_weights(h, out_h)
    xmin, xsize, wx = axis_weights(wd, out_w)
    planes = np.ascontiguousarray(x).reshape(n * c, h, wd)
    out = np.empty((n * c, out_h, out_w), dtype=F32)
    for s in range(0, n * c, planes_per_chunk):
        horiz = _interp_last_axis(planes[s:s + planes_per_chunk], xmin, xsize, wx)       # (p, h, out_w): the source of buffer2
        vert = _interp_last_axis(np.swapaxes(horiz, 1, 2), ymin, ysize, wy)              # (p, out_w, out_h)
        out[s:s + planes_per_chunk] = np.swapaxes(vert, 1, 2)
    return out.reshape(n, c, out_h, out_w)
