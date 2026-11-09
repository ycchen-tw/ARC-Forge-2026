"""QSA indexer selection after the score GEMM in one Triton kernel (``sparse_qsa.COMPACT_SELECT['kernel'] = 'fused'``).

``sparse_qsa.select_compact`` per query chunk: fp32 GEMM [rows*4, 128] x [128, width] -> relu -> sum over the 4 heads ->
* 1/sqrt(128) -> -inf for blocks >= complete -> topk(512) -> ``compact_rows_into`` (drop the -inf picks, sort ascending,
append the tail). That is ~12 small kernels per chunk, and at the 256-row chunks the GPU waits on the launches.

Here the GEMM is the same torch call on the same operands (``_indexer_choices(raw=True)``); everything after it is
``_select_kernel``, one program per query row:
  pass 0   relu-sum-scale of the row's ``complete`` blocks in the original order ((h0 + h1) + h2) + h3, times the fp32
           reciprocal as torch's divide-by-scalar does, into a scratch row; histogram of the top byte of the
           order-preserving uint32 key; any NaN score fails the step (``error``);
  pass 1-3 exact radix select (8 bits per pass) of the 512th largest key;
  collect  one pass in ascending block order: every key above the threshold, then the lowest block ids among the keys
           equal to it -> the row comes out sorted, no sort needed.
Rows with complete <= 512 keep every complete block, as the original. The selection is bitwise the original's on the
real 64k / 120k captures, packed segments, random and all-tied inputs (check_qsa_gather.py gate s2) and deterministic;
57.6 vs 126 ms per layer at 120k (kernel probe 2026-10-15). Ties at the threshold resolve to the lowest block ids, which
is what torch.topk gave on every case the gate has (torch documents no tie order, so the gate is the evidence).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _key(x):
    b = x.to(tl.uint32, bitcast=True)
    return tl.where((b >> 31) != 0, ~b, b | 0x80000000)


@triton.jit
def _select_kernel(S4, s4_row, SC, sc_row, BLOCKS, s_b, COUNTS, ERR, row0, pos0, inv_scale,
                   TOPK: tl.constexpr, BW: tl.constexpr):
    r = tl.program_id(0)
    pos = pos0 + r
    complete = (pos + 1) // 4
    tail = ((pos + 1) % 4 != 0).to(tl.int32)
    out = BLOCKS + (row0 + r).to(tl.int64) * s_b
    k_idx = tl.arange(0, TOPK)
    cols0 = tl.arange(0, BW)
    if complete <= TOPK:
        tl.store(out + k_idx, tl.where(k_idx < complete, k_idx, complete).to(BLOCKS.dtype.element_ty))
        tl.store(out + TOPK, complete.to(BLOCKS.dtype.element_ty))
        tl.store(COUNTS + row0 + r, complete + tail)
    else:
        base = S4 + (r * 4).to(tl.int64) * s4_row
        srow = SC + r.to(tl.int64) * sc_row
        bins = tl.arange(0, 256)
        hist = tl.zeros([256], tl.int32)
        nan = 0
        for c in range(0, complete, BW):
            cols = c + cols0
            m = cols < complete
            x0 = tl.maximum(tl.load(base + cols, mask=m, other=0.0), 0.0)
            x1 = tl.maximum(tl.load(base + s4_row + cols, mask=m, other=0.0), 0.0)
            x2 = tl.maximum(tl.load(base + 2 * s4_row + cols, mask=m, other=0.0), 0.0)
            x3 = tl.maximum(tl.load(base + 3 * s4_row + cols, mask=m, other=0.0), 0.0)
            s = (((x0 + x1) + x2) + x3) * inv_scale
            nan += tl.sum((m & (s != s)).to(tl.int32), 0)
            tl.store(srow + cols, s, mask=m)
            hist += tl.histogram((_key(s) >> 24).to(tl.int32), 256, mask=m)
        tl.store(ERR, 1, mask=nan != 0)
        need = TOPK
        prefix = tl.zeros([], tl.uint32)
        for p in tl.static_range(4):
            shift = 24 - 8 * p
            if p > 0:
                hist = tl.zeros([256], tl.int32)
                for c in range(0, complete, BW):
                    cols = c + cols0
                    m = cols < complete
                    key = _key(tl.load(srow + cols, mask=m, other=0.0))
                    match = m & ((key >> (shift + 8)) == prefix)
                    hist += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=match)
            ge = tl.cumsum(hist, 0, reverse=True)            # keys with digit >= d
            d = tl.max(tl.where(ge >= need, bins, -1), 0)
            above = tl.sum(tl.where(bins > d, hist, 0), 0)
            need = need - above
            prefix = (prefix << 8) | d.to(tl.uint32)
        # prefix is the 512th largest key: take every key above it and the first `need` keys equal to it
        written = 0
        eq_seen = 0
        for c in range(0, complete, BW):
            cols = c + cols0
            m = cols < complete
            key = _key(tl.load(srow + cols, mask=m, other=0.0))
            gt = m & (key > prefix)
            eq = m & (key == prefix)
            eq_rank = tl.cumsum(eq.to(tl.int32), 0) - 1 + eq_seen
            take = gt | (eq & (eq_rank < need))
            slot = tl.cumsum(take.to(tl.int32), 0) - 1 + written
            tl.store(out + slot, cols.to(BLOCKS.dtype.element_ty), mask=take)
            written += tl.sum(take.to(tl.int32), 0)
            eq_seen += tl.sum(eq.to(tl.int32), 0)
        tl.store(ERR, 1, mask=written != TOPK)
        tl.store(out + TOPK, complete.to(BLOCKS.dtype.element_ty))
        tl.store(COUNTS + row0 + r, TOPK + tail)


def select_rows(s4, scratch, blocks, counts, error, row0, pos0, rows, head_dim, topk, bw=512, num_warps=4):
    """Selection of ``rows`` query rows at segment positions pos0.. from the raw score GEMM ``s4`` [rows * 4, width]
    (fp32, the 4 index heads of a query adjacent) into ``blocks`` / ``counts`` rows row0.. (``QSASelection`` layout).
    ``scratch`` fp32 [>= rows, >= width]. Sets ``error`` on a NaN score or a short selection."""
    if s4.dtype != torch.float32 or s4.shape[0] != 4 * rows or s4.stride(1) != 1:
        raise ValueError(f'raw scores must be fp32 [4 * {rows}, width] row-major, got {s4.dtype} {tuple(s4.shape)}')
    if scratch.shape[0] < rows or scratch.shape[1] < s4.shape[1] or scratch.dtype != torch.float32:
        raise ValueError(f'scratch {tuple(scratch.shape)} too small for {rows} x {s4.shape[1]}')
    if topk & (topk - 1) or blocks.shape[1] != topk + 1:
        raise ValueError(f'top-k {topk} must be a power of two with blocks [T, top-k + 1]')
    inv_scale = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(head_dim ** 0.5, dtype=torch.float32))
    _select_kernel[(rows,)](s4, s4.stride(0), scratch, scratch.stride(0), blocks, blocks.stride(0), counts, error,
                            row0, pos0, inv_scale, TOPK=topk, BW=bw, num_warps=num_warps)
