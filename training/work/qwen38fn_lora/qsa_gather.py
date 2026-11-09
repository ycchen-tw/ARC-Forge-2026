"""QSA gather attention for training: every query reads only its own selected keys.

Compact selection (``QSASelection``): ``blocks`` [T, W] segment-local 4-token block ids,
ascending (int16 while every segment is shorter than 2**17 tokens, else int32); ``counts`` int32
[T]. The selected complete blocks come first; the last counted entry is the query's incomplete
tail block when that block already holds tokens <= the query. W = block_topk + 1. Key token =
segment start + 4 * block + j.

Forward and dQ: one program = one query x one KV head. The GROUP query heads that share the
KV head and the selection form the MMA M dimension (12 -> 16 rows), so the keys of one query
can be gathered BN at a time and fed to ``tl.dot``. The causal test only bites the tail block.

dK/dV: one program = one 16-token key tile (four blocks of one segment) x one KV head. It walks
a reverse list built by a two-pass counting sort (no atomics): the queries that use any of the
tile's four blocks, in ascending query order, each with a 4-bit block mask. Every reduction
order is fixed, so reruns are bitwise identical.

``KV_FP8['on']`` (train_rl_service ``--qsa-kv-fp8``): attend the way the FP8-KV rollout does
(work/qwen38fn_lora/QSA_GATHER_KERNEL.md, "FP8 KV"): K/V are cast to float8_e4m3fn unscaled (torch's
round-to-nearest cast, what sglang's KV pool stores) and widened back to BF16 (exact) for the kernels, and
the forward / dQ score MMA takes Q pre-multiplied by scale * log2(e) and rounded to BF16 (``rollout``;
sglang's decode / verify / prefill kernels). Gradients reach the BF16 K/V straight through the cast. The
kernels also take FP8 K/V rows directly (widened per gathered tile), which is exact too but 2x slower.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

LOG2E = 1.4426950408889634
BLOCK = 4          # QSA compress ratio (tokens per block)
TILE_BLOCKS = 4    # blocks per dK/dV key tile (16 keys)
ROWS_PER_CHUNK = 256   # reverse-index counting-sort row chunk
KV_DTYPE_FP8 = torch.float8_e4m3fn
# Off: BF16 K/V, the score scaled after the MMA (bitwise the kernels before the flag existed).
KV_FP8 = {'on': False}


@dataclass
class QSASelection:
    blocks: torch.Tensor   # [T, W] int16/int32 segment-local block ids, ascending
    counts: torch.Tensor   # [T] int32 valid entries (selected complete blocks + non-empty tail)

    def check(self, total):
        if self.blocks.dtype not in (torch.int16, torch.int32) or self.counts.dtype != torch.int32:
            raise TypeError(f'QSA selection dtypes {self.blocks.dtype}/{self.counts.dtype}')
        if self.blocks.shape[0] != total or self.counts.shape != (total,):
            raise ValueError(f'QSA selection shape {tuple(self.blocks.shape)} does not match {total} tokens')
        if not self.blocks.is_contiguous() or not self.counts.is_contiguous():
            raise ValueError('QSA selection must be contiguous')


class SelectionReplay:
    """Per-batch store: the forward records each QSA layer's selection once; the activation
    checkpoint recompute (which runs inside the autograd engine) replays it instead of rerunning
    the indexer. An opaque object, so pytree/FSDP input handling passes it through by reference.

    ``verify=True`` also recomputes during the replay and raises unless it is bitwise equal.
    """

    def __init__(self, verify=False):
        self.verify = verify
        self.records = {}
        self.stats = {'recorded': 0, 'replayed': 0, 'verified': 0}
        self._layout = None

    def select(self, layer_idx, compute):
        recompute = torch._C._current_graph_task_id() != -1
        if not recompute:
            if layer_idx in self.records:
                raise RuntimeError(f'QSA selection for layer {layer_idx} already recorded: batch metadata reused '
                                   'for a second forward?')
            selection = compute()
            self.records[layer_idx] = selection
            self.stats['recorded'] += 1
            return selection
        if layer_idx not in self.records:
            raise RuntimeError(f'checkpoint recompute of layer {layer_idx} has no recorded QSA selection')
        selection = self.records.pop(layer_idx)
        if self.verify:
            fresh = compute()
            if not (torch.equal(fresh.blocks, selection.blocks) and torch.equal(fresh.counts, selection.counts)):
                raise RuntimeError(f'recomputed QSA selection of layer {layer_idx} differs from the forward')
            self.stats['verified'] += 1
        self.stats['replayed'] += 1
        return selection

    def layout(self, meta, device):
        if self._layout is None:
            self._layout = segment_layout(meta['lengths'], meta['cu'], meta['rows'], device, dkdv_tile_blocks())
        if self._layout['total'] != meta['total']:
            raise RuntimeError('replay store reused across different packed batches')
        return self._layout


def block_dtype(max_length):
    return torch.int16 if max_length < 2**17 else torch.int32


def compact_rows(chosen, positions, topk):
    """Top-k block choices of queries at segment ``positions`` -> (blocks [n, topk+1], counts [n]).

    ``chosen`` are the indices returned by top-k over scores masked to -inf at blocks >= the
    query's complete-block count; -inf picks (only when fewer than topk blocks are complete)
    are dropped. The kept set is unchanged, only reordered ascending; the tail block follows.
    """
    complete = (positions + 1) // BLOCK
    valid = chosen < complete[:, None]
    nsel = complete.clamp(max=topk)
    # Fail loudly on the device (no host sync): every complete block must be finite-scored.
    torch._assert_async((valid.sum(1) == nsel).all())
    ordered = torch.where(valid, chosen, torch.iinfo(torch.int32).max).sort(dim=1).values
    col = torch.arange(topk, device=chosen.device)
    head = torch.where(col[None, :] < nsel[:, None], ordered, complete[:, None])
    blocks = torch.cat([head, complete[:, None]], 1)
    counts = nsel + ((positions + 1) % BLOCK != 0)
    return blocks, counts.to(torch.int32)


@triton.jit
def _compact_kernel(CHOSEN, s_c, BLOCKS, s_b, COUNTS, ERR, row0, pos0, TOPK: tl.constexpr):
    """One query row of ``compact_rows``: drop -inf picks, sort ascending, append the tail."""
    r = tl.program_id(0)
    pos = pos0 + r
    complete = (pos + 1) // 4
    nsel = tl.minimum(complete, TOPK)
    idx = tl.arange(0, TOPK)
    chosen = tl.load(CHOSEN + r.to(tl.int64) * s_c + idx)
    valid = chosen < complete
    # Every complete block must be finite-scored: flag, checked once per selection call.
    tl.store(ERR, 1, mask=tl.sum(valid.to(tl.int32)) != nsel)
    key = tl.sort(tl.where(valid, chosen, 2147483647).to(tl.int32))
    out = BLOCKS + (row0 + r).to(tl.int64) * s_b
    tl.store(out + idx, tl.where(idx < nsel, key, complete).to(BLOCKS.dtype.element_ty))
    tl.store(out + TOPK, complete.to(BLOCKS.dtype.element_ty))
    tl.store(COUNTS + row0 + r, (nsel + ((pos + 1) % 4 != 0)).to(tl.int32))


def compact_rows_into(chosen, first_position, blocks, counts, row0, error):
    """Fused ``compact_rows`` writing straight into the selection (no per-chunk torch ops)."""
    rows, topk = chosen.shape
    if blocks.shape[1] != topk + 1 or chosen.stride(1) != 1 or topk & (topk - 1):
        raise ValueError('compact_rows_into expects [n, topk] contiguous choices, topk a power of 2')
    _compact_kernel[(rows,)](chosen, chosen.stride(0), blocks, blocks.stride(0), counts, error, row0, first_position,
                             TOPK=topk, num_warps=4)


def dense_rows(positions, topk):
    """Rows of queries whose complete blocks are all selected (dense causal)."""
    complete = (positions + 1) // BLOCK
    torch._assert_async((complete <= topk).all())
    col = torch.arange(topk + 1, device=positions.device)
    blocks = torch.minimum(col[None, :], complete[:, None])
    counts = complete + ((positions + 1) % BLOCK != 0)
    return blocks, counts.to(torch.int32)


def segment_layout(lengths, cu, rows, device, tile_blocks=TILE_BLOCKS):
    """Per-query segment start and the (4 * tile_blocks)-token dK/dV key tiles of every segment.

    ``lengths`` is the host list (no sync); ``cu``/``rows`` are the packed GPU metadata.
    """
    if tile_blocks not in (4, 8, 16):
        raise ValueError('tile_blocks must be 4, 8 or 16')
    TILE = BLOCK * tile_blocks
    n_tiles = sum((n + TILE - 1) // TILE for n in lengths)
    total = sum(lengths)
    cu = cu.to(device=device, dtype=torch.int64)
    rows = rows.to(device=device, dtype=torch.int64)
    sizes = cu[1:] - cu[:-1]
    tiles = (sizes + TILE - 1) // TILE
    tile_base = torch.cumsum(tiles, 0) - tiles
    tile_ids = torch.arange(n_tiles, device=device, dtype=torch.int64)
    tile_segment = torch.searchsorted(tile_base, tile_ids, right=True) - 1
    tile_start = cu[tile_segment] + (tile_ids - tile_base[tile_segment]) * TILE
    tile_end = torch.minimum(tile_start + TILE, cu[tile_segment + 1])
    # Dense global numbering of every segment's 4-token blocks (the query-union kernels, qsa_union.py): per query the
    # number of blocks before its segment, and the first key of every block.
    n_blocks = sum((n + BLOCK - 1) // BLOCK for n in lengths)
    seg_blocks = (sizes + BLOCK - 1) // BLOCK
    block_base = torch.cumsum(seg_blocks, 0) - seg_blocks
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.int64)
    block_segment = torch.searchsorted(block_base, block_ids, right=True) - 1
    block_key = cu[block_segment] + (block_ids - block_base[block_segment]) * BLOCK
    return {'seg_start': cu[rows].to(torch.int32).contiguous(), 'tile_base': tile_base[rows].to(torch.int32).contiguous(),
            'tile_start': tile_start.to(torch.int32), 'tile_end': tile_end.to(torch.int32),
            'n_tiles': n_tiles, 'total': total, 'tile_blocks': tile_blocks,
            'block_base': block_base[rows].to(torch.int32).contiguous(), 'block_key': block_key.to(torch.int32),
            'n_blocks': n_blocks, 'lengths': list(lengths), 'cu': cu}


@triton.jit
def _row_tiles(BLOCKS, COUNTS, TILEBASE, q, s_b, W, BW: tl.constexpr, TB: tl.constexpr):
    """Merged (global tile, TB-bit block mask) entries of one query row; valid at group tails."""
    idx = tl.arange(0, BW)
    count = tl.load(COUNTS + q)
    row = BLOCKS + q.to(tl.int64) * s_b
    valid = idx < count
    blk = tl.load(row + idx, mask=valid, other=0).to(tl.int32)
    tile = blk // TB
    bits = tl.where(valid, 1 << (blk % TB), 0)
    # A sorted row holds at most TB blocks of one tile, so a group spans <= TB entries.
    merged = bits
    for back in tl.static_range(1, TB):
        prev = tl.load(row + idx - back, mask=valid & (idx >= back), other=-TB).to(tl.int32)
        merged += tl.where((prev // TB) == tile, 1 << (prev % TB), 0)
    nxt = tl.load(row + idx + 1, mask=idx + 1 < count, other=-TB).to(tl.int32)
    tail = valid & ((idx + 1 >= count) | ((nxt // TB) != tile))
    gtile = tile + tl.load(TILEBASE + q)
    return gtile, merged, tail


@triton.jit
def _reverse_count(BLOCKS, COUNTS, TILEBASE, HIST, T, n_tiles, s_b, W,
                   ROWS: tl.constexpr, BW: tl.constexpr, TB: tl.constexpr):
    c = tl.program_id(0)
    hist = HIST + c.to(tl.int64) * n_tiles
    for r in range(0, ROWS):
        q = c * ROWS + r
        if q < T:
            gtile, merged, tail = _row_tiles(BLOCKS, COUNTS, TILEBASE, q, s_b, W, BW, TB)
            # Tiles are distinct within a row and this program owns its histogram row.
            old = tl.load(hist + gtile, mask=tail, other=0)
            tl.store(hist + gtile, old + 1, mask=tail)
        # Another thread may own the same tile counter in the next row.
        tl.debug_barrier()


@triton.jit
def _reverse_fill(BLOCKS, COUNTS, TILEBASE, BASE, QUERIES, BITS, T, n_tiles, s_b, W,
                  ROWS: tl.constexpr, BW: tl.constexpr, TB: tl.constexpr):
    c = tl.program_id(0)
    base = BASE + c.to(tl.int64) * n_tiles
    for r in range(0, ROWS):
        q = c * ROWS + r
        if q < T:
            gtile, merged, tail = _row_tiles(BLOCKS, COUNTS, TILEBASE, q, s_b, W, BW, TB)
            pos = tl.load(base + gtile, mask=tail, other=0)
            tl.store(QUERIES + pos, q, mask=tail)
            tl.store(BITS + pos, merged.to(BITS.dtype.element_ty), mask=tail)
            tl.store(base + gtile, pos + 1, mask=tail)
        tl.debug_barrier()


def build_reverse(selection, layout):
    """For every key tile, the (query, block mask) uses in ascending query order."""
    blocks, counts = selection.blocks, selection.counts
    total, width = blocks.shape
    n_tiles, tb = layout['n_tiles'], layout['tile_blocks']
    chunks = triton.cdiv(total, ROWS_PER_CHUNK)
    bw = triton.next_power_of_2(width)
    hist = torch.zeros((chunks, n_tiles), device=blocks.device, dtype=torch.int32)
    _reverse_count[(chunks,)](blocks, counts, layout['tile_base'], hist, total, n_tiles, blocks.stride(0), width,
                              ROWS=ROWS_PER_CHUNK, BW=bw, TB=tb, num_warps=4)
    per_tile = hist.sum(0, dtype=torch.int64)
    offsets = torch.zeros(n_tiles + 1, device=blocks.device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(per_tile, 0)
    # base[c, t] = offsets[t] + sum_{c' < c} hist[c', t]  (stable: chunk order, then row order)
    base = torch.cumsum(hist, 0, dtype=torch.int64) - hist + offsets[:-1][None, :]
    if total * width >= 2**31:
        raise ValueError('reverse index positions exceed int32')
    base = base.to(torch.int32)
    capacity = total * width
    queries = torch.empty(capacity, device=blocks.device, dtype=torch.int32)
    # Bit 7 / bit 15 land in the sign bit; the kernels sign-extend, so (bits >> b) & 1 stays exact.
    bits = torch.empty(capacity, device=blocks.device, dtype=torch.int8 if tb <= 8 else torch.int16)
    _reverse_fill[(chunks,)](blocks, counts, layout['tile_base'], base, queries, bits, total, n_tiles,
                             blocks.stride(0), width, ROWS=ROWS_PER_CHUNK, BW=bw, TB=tb, num_warps=4)
    return {'queries': queries, 'bits': bits, 'offsets': offsets}


@triton.jit
def _rollout_q(q, sm_scale):
    """sglang's QSA kernels: Q * scale * log2(e) in fp32, rounded to BF16, then the score MMA unscaled."""
    return ((q.to(tl.float32) * sm_scale) * 1.4426950408).to(tl.bfloat16)


@triton.jit
def _fwd_kernel(Q, K, V, BLOCKS, COUNTS, SEGSTART, OUT, LSE,
                s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_ot, s_oh, s_b, H,
                qk_scale, sm_scale,
                GROUP: tl.constexpr, BM: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, ROLLOUT: tl.constexpr):
    qi = tl.program_id(0)
    g = tl.program_id(1)
    qi64 = qi.to(tl.int64)
    rows = tl.arange(0, BM)
    head = g * GROUP + rows
    hmask = rows < GROUP
    offs_d = tl.arange(0, D)
    q = tl.load(Q + qi64 * s_qt + head[:, None] * s_qh + offs_d[None, :], mask=hmask[:, None], other=0.0)
    if ROLLOUT:
        q = _rollout_q(q, sm_scale)
    count = tl.load(COUNTS + qi)
    seg = tl.load(SEGSTART + qi)
    row = BLOCKS + qi64 * s_b
    offs_n = tl.arange(0, BN)
    m_i = tl.full([BM], float('-inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for first in range(0, count * 4, BN):
        entry = (first + offs_n) // 4
        live = entry < count
        key = seg + tl.load(row + entry, mask=live, other=0).to(tl.int32) * 4 + (offs_n % 4)
        live = live & (key <= qi)
        key64 = key.to(tl.int64)
        # .to(bf16): FP8 rows widened exactly (a no-op on BF16 K/V)
        k = tl.load(K + key64[:, None] * s_kt + g * s_kh + offs_d[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        if ROLLOUT:
            s = tl.dot(q, tl.trans(k))
        else:
            s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(live[None, :], s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + key64[:, None] * s_vt + g * s_vh + offs_d[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + qi64 * s_ot + head[:, None] * s_oh + offs_d[None, :], out.to(OUT.dtype.element_ty),
             mask=hmask[:, None])
    tl.store(LSE + qi64 * H + head, m_i + tl.log2(l_i), mask=hmask)


@triton.jit
def _dq_kernel(Q, K, V, OUT, DO, LSE, DELTA, BLOCKS, COUNTS, SEGSTART, DQ,
               s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_ot, s_oh, s_dot, s_doh, s_dqt, s_dqh, s_b, H,
               qk_scale, sm_scale,
               GROUP: tl.constexpr, BM: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, ROLLOUT: tl.constexpr):
    """dQ for one query x KV head; also writes Delta = rowsum(dO * O) for the dK/dV kernel.
    ROLLOUT: the forward's score (pre-scaled BF16 Q); dQ flows straight through that rounding."""
    qi = tl.program_id(0)
    g = tl.program_id(1)
    qi64 = qi.to(tl.int64)
    rows = tl.arange(0, BM)
    head = g * GROUP + rows
    hmask = rows < GROUP
    offs_d = tl.arange(0, D)
    q = tl.load(Q + qi64 * s_qt + head[:, None] * s_qh + offs_d[None, :], mask=hmask[:, None], other=0.0)
    if ROLLOUT:
        q = _rollout_q(q, sm_scale)
    do = tl.load(DO + qi64 * s_dot + head[:, None] * s_doh + offs_d[None, :], mask=hmask[:, None], other=0.0)
    lse = tl.load(LSE + qi64 * H + head, mask=hmask, other=0.0)
    o = tl.load(OUT + qi64 * s_ot + head[:, None] * s_oh + offs_d[None, :], mask=hmask[:, None], other=0.0)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    tl.store(DELTA + qi64 * H + head, delta, mask=hmask)
    count = tl.load(COUNTS + qi)
    seg = tl.load(SEGSTART + qi)
    row = BLOCKS + qi64 * s_b
    offs_n = tl.arange(0, BN)
    dq = tl.zeros([BM, D], tl.float32)
    for first in range(0, count * 4, BN):
        entry = (first + offs_n) // 4
        live = entry < count
        key = seg + tl.load(row + entry, mask=live, other=0).to(tl.int32) * 4 + (offs_n % 4)
        live = live & (key <= qi)
        key64 = key.to(tl.int64)
        k = tl.load(K + key64[:, None] * s_kt + g * s_kh + offs_d[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(V + key64[:, None] * s_vt + g * s_vh + offs_d[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        if ROLLOUT:
            s = tl.dot(q, tl.trans(k))
        else:
            s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.where(live[None, :], tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(tl.bfloat16), k, dq)
    tl.store(DQ + qi64 * s_dqt + head[:, None] * s_dqh + offs_d[None, :], (dq * sm_scale).to(DQ.dtype.element_ty),
             mask=hmask[:, None])


@triton.jit
def _dkdv_tile_start(K, V, TILE_START, TILE_END, OFFSETS, t, g, s_kt, s_kh, s_vt, s_vh,
                     D: tl.constexpr, BK: tl.constexpr):
    kstart = tl.load(TILE_START + t)
    kend = tl.load(TILE_END + t)
    key = kstart + tl.arange(0, BK)
    kin = key < kend
    key64 = key.to(tl.int64)
    offs_d = tl.arange(0, D)
    # FP8 K/V (KV_FP8) widened to BF16 once per tile; the dK/dV score uses Q unrounded (see gather_backward)
    k = tl.load(K + key64[:, None] * s_kt + g * s_kh + offs_d[None, :], mask=kin[:, None], other=0.0).to(tl.bfloat16)
    v = tl.load(V + key64[:, None] * s_vt + g * s_vh + offs_d[None, :], mask=kin[:, None], other=0.0).to(tl.bfloat16)
    return key, kin, key64, k, v, tl.load(OFFSETS + t), tl.load(OFFSETS + t + 1)


@triton.jit
def _dkdv_kernel(QD, DOD, K, V, LSE, DELTA, DK, DV, TILE_START, TILE_END, OFFSETS, QUERIES, BITS,
                 rq_t, rq_h, rd_t, rd_h, s_kt, s_kh, s_vt, s_vh, s_dkt, s_dkh, s_dvt, s_dvh, H,
                 qk_scale, sm_scale,
                 GROUP: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, QC: tl.constexpr):
    """Q/dO rows arrive by TMA gather (Blackwell); dK^T/dV^T accumulate as [D, BK] so every dot has M >= 64."""
    t = tl.program_id(0)
    g = tl.program_id(1)
    key, kin, key64, k, v, lo, hi = _dkdv_tile_start(K, V, TILE_START, TILE_END, OFFSETS, t, g,
                                                     s_kt, s_kh, s_vt, s_vh, D, BK)
    offs_d = tl.arange(0, D)
    dkt = tl.zeros([D, BK], tl.float32)
    dvt = tl.zeros([D, BK], tl.float32)
    kblock = tl.arange(0, BK) // 4
    offs_c = tl.arange(0, QC)
    for first in range(lo, hi, QC):
        idx = first + offs_c
        im = idx < hi
        qid = tl.load(QUERIES + idx, mask=im, other=0)
        bits = tl.load(BITS + idx, mask=im, other=0).to(tl.int32)
        ok = (((bits[:, None] >> kblock[None, :]) & 1) != 0) & kin[None, :] & (key[None, :] <= qid[:, None])
        qid64 = qid.to(tl.int64)
        for h in range(GROUP):
            head = g * GROUP + h
            # Padding lanes gather row qid=0 (in bounds); their bits are 0, so p and ds are 0.
            qh = QD.gather(qid * rq_t + head * rq_h, 0)
            doh = DOD.gather(qid * rd_t + head * rd_h, 0)
            lse = tl.load(LSE + qid64 * H + head, mask=im, other=0.0)
            delta = tl.load(DELTA + qid64 * H + head, mask=im, other=0.0)
            s = tl.dot(qh, tl.trans(k)) * qk_scale
            p = tl.where(ok, tl.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(doh, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dvt = tl.dot(tl.trans(doh), p.to(doh.dtype), dvt)
            dkt = tl.dot(tl.trans(qh), ds.to(qh.dtype), dkt)
    tl.store(DK + key64[None, :] * s_dkt + g * s_dkh + offs_d[:, None], (dkt * sm_scale).to(DK.dtype.element_ty),
             mask=kin[None, :])
    tl.store(DV + key64[None, :] * s_dvt + g * s_dvh + offs_d[:, None], dvt.to(DV.dtype.element_ty),
             mask=kin[None, :])


@triton.jit
def _dkdv_plain_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, TILE_START, TILE_END, OFFSETS, QUERIES, BITS,
                       s_qt, s_qh, s_kt, s_kh, s_vt, s_vh, s_dot, s_doh, s_dkt, s_dkh, s_dvt, s_dvh, H,
                       qk_scale, sm_scale,
                       GROUP: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, QC: tl.constexpr):
    """Pre-Blackwell fallback (``DKDV_CONFIG['tma'] = False``): plain pointer gathers."""
    t = tl.program_id(0)
    g = tl.program_id(1)
    key, kin, key64, k, v, lo, hi = _dkdv_tile_start(K, V, TILE_START, TILE_END, OFFSETS, t, g,
                                                     s_kt, s_kh, s_vt, s_vh, D, BK)
    offs_d = tl.arange(0, D)
    dk = tl.zeros([BK, D], tl.float32)
    dv = tl.zeros([BK, D], tl.float32)
    kblock = tl.arange(0, BK) // 4
    offs_c = tl.arange(0, QC)
    for first in range(lo, hi, QC):
        idx = first + offs_c
        im = idx < hi
        qid = tl.load(QUERIES + idx, mask=im, other=0)
        bits = tl.load(BITS + idx, mask=im, other=0).to(tl.int32)
        ok = (((bits[:, None] >> kblock[None, :]) & 1) != 0) & kin[None, :] & (key[None, :] <= qid[:, None])
        qid64 = qid.to(tl.int64)
        for h in range(GROUP):
            head = g * GROUP + h
            qh = tl.load(Q + qid64[:, None] * s_qt + head * s_qh + offs_d[None, :], mask=im[:, None], other=0.0)
            doh = tl.load(DO + qid64[:, None] * s_dot + head * s_doh + offs_d[None, :], mask=im[:, None], other=0.0)
            lse = tl.load(LSE + qid64 * H + head, mask=im, other=0.0)
            delta = tl.load(DELTA + qid64 * H + head, mask=im, other=0.0)
            s = tl.dot(qh, tl.trans(k)) * qk_scale
            p = tl.where(ok, tl.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(doh, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dv = tl.dot(tl.trans(p.to(doh.dtype)), doh, dv)
            dk = tl.dot(tl.trans(ds.to(qh.dtype)), qh, dk)
    tl.store(DK + key64[:, None] * s_dkt + g * s_dkh + offs_d[None, :], (dk * sm_scale).to(DK.dtype.element_ty),
             mask=kin[:, None])
    tl.store(DV + key64[:, None] * s_dvt + g * s_dvh + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=kin[:, None])


FWD_CONFIG = {'BN': 64, 'num_warps': 4, 'num_stages': 2}
DQ_CONFIG = {'BN': 64, 'num_warps': 4, 'num_stages': 2}
DKDV_CONFIG = {'tma': True, 'QC': 64, 'num_warps': 4, 'num_stages': 1}
DKDV_PLAIN_CONFIG = {'QC': 64, 'num_warps': 8, 'num_stages': 3}
# 'gluon' (trainer_kernels qsa_dkdv): qsa_dkdv_gluon.py on 64-key tiles instead of _dkdv_kernel on 16-key tiles
DKDV_KERNEL = {'kernel': 'triton'}
DKDV_GLUON_CONFIG = {'QC': 64, 'STAGES': 2, 'num_warps': 8}
# 'on' (trainer_kernels qsa_union): forward and dQ by the query-union kernels of qsa_union.py (Blackwell; BF16 K/V only):
# 'group' consecutive tokens of a segment share one 128-row tcgen05 tile and iterate the union of their selections.
# dK/dV stay the kernels above. Off: bitwise the kernels before the switch existed.
UNION = {'on': False, 'group': 10}


def dkdv_tile_blocks():
    return 16 if DKDV_KERNEL['kernel'] == 'gluon' else TILE_BLOCKS


def _row_view(x):
    """[T, H, D] -> ([T*H, D] rows for a TMA descriptor, row step per t, per h).

    Token-major and head-major storage are both rows of D contiguous elements; any other
    layout (e.g. a strided gradient) is copied once into token-major order.
    """
    total, heads, dim = x.shape
    span = (total - 1) * x.stride(0) + (heads - 1) * x.stride(1) + dim
    if (x.stride(2) != 1 or x.stride(0) % dim or x.stride(1) % dim or x.storage_offset() % 8
            or span != total * heads * dim):
        x = x.contiguous()
    return x.as_strided((total * heads, dim), (dim, 1)), x.stride(0) // dim, x.stride(1) // dim


def _check_inputs(q, k, v):
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError('QSA gather attention expects bf16 q/k/v')
    if q.dim() != 3 or k.dim() != 3 or v.shape != k.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f'expected q [T,H,D], k/v [T,Hkv,D], got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}')
    if q.shape[1] % k.shape[1]:
        raise ValueError('query heads must be a multiple of KV heads')
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        raise ValueError('head_dim must be contiguous')


def _kv_fp8_rows(k, v):
    """True for float8_e4m3fn k/v (the kernels widen each gathered row), False for BF16; anything else raises."""
    if k.dtype != v.dtype or k.dtype not in (torch.bfloat16, KV_DTYPE_FP8):
        raise TypeError(f'QSA gather K/V must both be bf16 or float8_e4m3fn, got {k.dtype}/{v.dtype}')
    return k.dtype == KV_DTYPE_FP8


def _union_check(k, v, rollout):
    if rollout or _kv_fp8_rows(k, v):
        raise NotImplementedError('the query-union kernels (UNION) do not implement the FP8-KV rollout score')
    if torch.cuda.get_device_capability(k.device)[0] < 10:
        raise RuntimeError('the query-union kernels (UNION) need Blackwell (sm_100+)')


def gather_forward(q, k, v, selection, layout, scale, rollout=False):
    """``rollout``: the score of the FP8-KV rollout (Q pre-scaled and rounded to BF16, see the module docstring)."""
    if UNION['on']:
        import qsa_union
        _union_check(k, v, rollout)
        union = qsa_union.build_union(selection, layout, q.shape[0], UNION['group'], qsa_union.FWD_CONFIG['BN'])
        return qsa_union.union_forward(q, k, v, union, scale)
    _kv_fp8_rows(k, v)
    total, heads, dim = q.shape
    group = heads // k.shape[1]
    out = torch.empty((total, heads, dim), device=q.device, dtype=q.dtype)
    lse = torch.empty((total, heads), device=q.device, dtype=torch.float32)
    cfg = dict(FWD_CONFIG)
    bn = cfg.pop('BN')
    _fwd_kernel[(total, k.shape[1])](
        q, k, v, selection.blocks, selection.counts, layout['seg_start'], out, lse,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1), selection.blocks.stride(0), heads, scale * LOG2E, scale,
        GROUP=group, BM=max(16, triton.next_power_of_2(group)), D=dim, BN=bn, ROLLOUT=bool(rollout), **cfg)
    return out, lse


def gather_backward(q, k, v, out, lse, grad, selection, layout, scale, rollout=False):
    """``rollout``: dQ recomputes the forward's rollout score exactly (pre-scaled BF16 Q); the dK/dV kernels score with
    the unrounded Q (their P differs from the forward's by that BF16 rounding, ~2^-9 of the logit, far below the BF16
    floor of P itself), so the Gluon / Triton dK/dV kernels need no Q rewrite in shared memory."""
    total, heads, dim = q.shape
    group = heads // k.shape[1]
    _kv_fp8_rows(k, v)
    dk = torch.empty(k.shape, device=q.device, dtype=q.dtype)
    dv = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    if UNION['on']:
        # the union is rebuilt rather than saved by the forward: no extra memory held to the backward
        import qsa_union
        _union_check(k, v, rollout)
        union = qsa_union.build_union(selection, layout, total, UNION['group'], qsa_union.FWD_CONFIG['BN'])
        dq, delta = qsa_union.union_dq(q, k, v, out, lse, grad, union, scale)
        del union
    else:
        delta = torch.empty((total, heads), device=q.device, dtype=torch.float32)   # written by _dq_kernel
        dq = torch.empty((total, heads, dim), device=q.device, dtype=q.dtype)
        cfg = dict(DQ_CONFIG)
        bn = cfg.pop('BN')
        _dq_kernel[(total, k.shape[1])](
            q, k, v, out, grad, lse, delta, selection.blocks, selection.counts, layout['seg_start'], dq,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
            grad.stride(0), grad.stride(1), dq.stride(0), dq.stride(1), selection.blocks.stride(0), heads,
            scale * LOG2E, scale,
            GROUP=group, BM=max(16, triton.next_power_of_2(group)), D=dim, BN=bn, ROLLOUT=bool(rollout), **cfg)
    reverse = build_reverse(selection, layout)
    common = dict(GROUP=group, D=dim, BK=BLOCK * layout['tile_blocks'])
    tail = (layout['tile_start'], layout['tile_end'], reverse['offsets'], reverse['queries'], reverse['bits'])
    if DKDV_KERNEL['kernel'] == 'gluon':
        import qsa_dkdv_gluon
        dk, dv = qsa_dkdv_gluon.run_dkdv(q, k, v, lse, delta, grad, reverse, layout, scale, **DKDV_GLUON_CONFIG)
    elif DKDV_KERNEL['kernel'] != 'triton':
        raise ValueError(f"DKDV_KERNEL {DKDV_KERNEL['kernel']!r}: 'triton' or 'gluon'")
    elif DKDV_CONFIG['tma']:
        if torch.cuda.get_device_capability(q.device)[0] < 10:
            raise RuntimeError('TMA gather needs Blackwell (sm_100+); set DKDV_CONFIG["tma"] = False')
        cfg = {key: val for key, val in DKDV_CONFIG.items() if key != 'tma'}
        qrows, rq_t, rq_h = _row_view(q)
        drows, rd_t, rd_h = _row_view(grad)
        qdesc = TensorDescriptor(qrows, shape=list(qrows.shape), strides=[dim, 1], block_shape=[1, dim])
        ddesc = TensorDescriptor(drows, shape=list(drows.shape), strides=[dim, 1], block_shape=[1, dim])
        _dkdv_kernel[(layout['n_tiles'], k.shape[1])](
            qdesc, ddesc, k, v, lse, delta, dk, dv, *tail, rq_t, rq_h, rd_t, rd_h,
            k.stride(0), k.stride(1), v.stride(0), v.stride(1), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1),
            heads, scale * LOG2E, scale, **common, **cfg)
    else:
        _dkdv_plain_kernel[(layout['n_tiles'], k.shape[1])](
            q, k, v, grad, lse, delta, dk, dv, *tail,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            grad.stride(0), grad.stride(1), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1), heads,
            scale * LOG2E, scale, **common, **DKDV_PLAIN_CONFIG)
    return dq, dk, dv


class _GatherAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, blocks, counts, layout, scale, kv_fp8):
        selection = QSASelection(blocks, counts)
        kk, vv = k, v
        if kv_fp8:
            # The rollout's KV pool store: an unscaled cast (MHATokenToKVPool.set_kv_buffer). Only the FP8 copies are
            # saved; dK/dV pass straight through to the BF16 k/v. The kernels run on the FP8 values widened to BF16 in
            # one pass (exact): reading FP8 rows and widening them inside the gather loop is 2x slower than BF16 rows
            # with Triton (the conversion stops the load pipelining; QSA_GATHER_KERNEL.md "FP8 KV").
            k, v = k.to(KV_DTYPE_FP8), v.to(KV_DTYPE_FP8)
            kk, vv = k.to(torch.bfloat16), v.to(torch.bfloat16)
        out, lse = gather_forward(q, kk, vv, selection, layout, scale, rollout=kv_fp8)
        ctx.save_for_backward(q, k, v, out, lse, blocks, counts)
        ctx.layout = layout
        ctx.scale = scale
        ctx.kv_fp8 = kv_fp8
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, blocks, counts = ctx.saved_tensors
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        if ctx.kv_fp8:
            k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
        dq, dk, dv = gather_backward(q, k, v, out, lse, grad, QSASelection(blocks, counts), ctx.layout, ctx.scale,
                                     rollout=ctx.kv_fp8)
        return dq, dk, dv, None, None, None, None, None


def qsa_gather_attention(q, k, v, selection, layout, scale, kv_fp8=None):
    """q [T,H,D], k/v [T,Hkv,D] bf16 -> [T,H,D]. ``layout`` from ``segment_layout``.
    ``kv_fp8`` (default ``KV_FP8['on']``): the FP8-KV rollout numerics of the module docstring."""
    _check_inputs(q, k, v)
    selection.check(q.shape[0])
    if layout['total'] != q.shape[0]:
        raise ValueError('segment layout does not match the token count')
    kv_fp8 = KV_FP8['on'] if kv_fp8 is None else kv_fp8
    return _GatherAttention.apply(q, k, v, selection.blocks, selection.counts, layout, scale, bool(kv_fp8))


class _FP8CastSTE(torch.autograd.Function):
    """x -> bf16(float8_e4m3fn(x)) forward, identity backward."""

    @staticmethod
    def forward(ctx, x):
        return x.to(KV_DTYPE_FP8).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


def rollout_kv(k, v):
    """K/V for a QSA attention that is not the gather kernel (packed_qwen4's dense FA4 path for bins whose segments are
    all within the budget): under KV_FP8 the FP8-cast values (straight-through gradient), else k/v unchanged."""
    if not KV_FP8['on']:
        return k, v
    return _FP8CastSTE.apply(k), _FP8CastSTE.apply(v)
