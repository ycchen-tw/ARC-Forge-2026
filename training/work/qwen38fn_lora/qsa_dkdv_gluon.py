"""QSA gather attention dK/dV as a Gluon kernel on Blackwell (``--fast-kernels qsa_dkdv``, qsa_gather.gather_backward).

Same math, reverse lists and per-(chunk, head) accumulation order as qsa_gather._dkdv_kernel, on 64-key tiles
(segment_layout tile_blocks 16) and explicitly software-pipelined:

  * the Q/dO rows of iteration it+STAGES-1 are TMA-gathered into a STAGES-deep ring while iteration it computes (the
    Triton kernel waits on every gather: ~25% of its stall samples in ncu);
  * the S and dP MMAs go out back to back under one commit; the dV/dK MMAs stay in flight while the next iteration
    loads its row metadata and issues its gather (one wait per MMA pair instead of four).

Real 94k+26k bin, one QSA layer: dK/dV 151.6 -> 96.0 ms (reverse build 1.8 -> 3.6 ms). dK/dV error against fp32 within
the BF16-arithmetic floor exactly as the Triton kernel; reruns bitwise; not bitwise to the Triton kernel (the 64-query
chunks of the longer 64-key reverse lists group differently, rel-L2 4.7e-4). Prototype and gate:
$ARC_RUNS/qwen38fn_qad/20261015-qsa-bwd (gluon_dkdv1b.py, gate_proto.py).
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory,
                                                                 fence_async_shared, get_tmem_reg_layout, mbarrier,
                                                                 tcgen05_commit, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.blackwell import tma as btma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GDesc

import qsa_gather as qg
from qsa_gather import LOG2E


@gluon.jit
def _issue(QD, DOD, q_buf, do_buf, bar, QUERIES, lo, hi, it, g, rq_t, rq_h, rd_t, rd_h,
           GROUP: ttgl.constexpr, QC: ttgl.constexpr, D: ttgl.constexpr, idx_l: ttgl.constexpr, pred):
    chunk = it // GROUP
    h = it - chunk * GROUP
    idx = lo + chunk * QC + ttgl.arange(0, QC, layout=idx_l)
    qid = ttgl.load(QUERIES + idx, mask=(idx < hi) & pred, other=0)
    head = g * GROUP + h
    mbarrier.expect(bar, 2 * QC * D * 2, pred=pred)
    btma.async_gather(QD, (qid * rq_t + head * rq_h).to(ttgl.int32), 0, bar, q_buf, pred=pred)
    btma.async_gather(DOD, (qid * rd_t + head * rd_h).to(ttgl.int32), 0, bar, do_buf, pred=pred)


@gluon.jit
def _dkdv_gluon(QD, DOD, K, V, LSE, DELTA, DK, DV, TILE_START, TILE_END, OFFSETS, QUERIES, BITS,
                rq_t, rq_h, rd_t, rd_h, s_kt, s_kh, s_vt, s_vh, s_dkt, s_dkh, s_dvt, s_dvh, H,
                qk_scale, sm_scale,
                GROUP: ttgl.constexpr, D: ttgl.constexpr, BK: ttgl.constexpr, QC: ttgl.constexpr,
                STAGES: ttgl.constexpr, KV_FP8: ttgl.constexpr, num_warps: ttgl.constexpr):
    t = ttgl.program_id(0)
    g = ttgl.program_id(1)
    kv_ld: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    idx_l: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.BlockedLayout([1, 4], [32, 1], [1, num_warps], [1, 0]))
    s_tl: ttgl.constexpr = TensorMemoryLayout([QC, BK], col_stride=1)
    acc_tl: ttgl.constexpr = TensorMemoryLayout([128, BK], col_stride=1)
    s_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [QC, BK], s_tl, num_warps)
    acc_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [D, BK], acc_tl, num_warps)
    row_l: ttgl.constexpr = ttgl.SliceLayout(1, s_rl)
    col_l: ttgl.constexpr = ttgl.SliceLayout(0, s_rl)
    kv_sl: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BK, D], ttgl.bfloat16)
    p_sl: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([QC, BK], ttgl.bfloat16)

    kstart = ttgl.load(TILE_START + t)
    kend = ttgl.load(TILE_END + t)
    lo = ttgl.load(OFFSETS + t)
    hi = ttgl.load(OFFSETS + t + 1)
    kr = kstart + ttgl.arange(0, BK, layout=ttgl.SliceLayout(1, kv_ld))
    dd = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, kv_ld))
    kin_ld = kr < kend
    kr64 = kr.to(ttgl.int64)
    k = ttgl.load(K + kr64[:, None] * s_kt + g * s_kh + dd[None, :], mask=kin_ld[:, None], other=0.0)
    v = ttgl.load(V + kr64[:, None] * s_vt + g * s_vh + dd[None, :], mask=kin_ld[:, None], other=0.0)
    if KV_FP8:  # qsa_gather.KV_FP8: the tile's FP8 rows widened to BF16 once (the score uses Q unrounded)
        k = k.to(ttgl.bfloat16)
        v = v.to(ttgl.bfloat16)
    k_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [BK, D], kv_sl, k)
    v_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [BK, D], kv_sl, v)
    q_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [STAGES, QC, D], QD.layout)
    do_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [STAGES, QC, D], DOD.layout)
    p_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [QC, BK], p_sl)
    ds_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [QC, BK], p_sl)
    ld_bar = ttgl.allocate_shared_memory(ttgl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    s_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    for i in ttgl.static_range(STAGES):
        mbarrier.init(ld_bar.index(i), count=1)
    mbarrier.init(s_bar, count=1)
    mbarrier.init(mma_bar, count=1)
    s_t = allocate_tensor_memory(ttgl.float32, [QC, BK], s_tl)
    dp_t = allocate_tensor_memory(ttgl.float32, [QC, BK], s_tl)
    dkt_t = allocate_tensor_memory(ttgl.float32, [D, BK], acc_tl, ttgl.zeros([D, BK], ttgl.float32, acc_rl))
    dvt_t = allocate_tensor_memory(ttgl.float32, [D, BK], acc_tl, ttgl.zeros([D, BK], ttgl.float32, acc_rl))
    fence_async_shared()
    ttgl.thread_barrier()

    n_it = (ttgl.cdiv(hi - lo, QC) * GROUP).to(ttgl.int32)
    for j in ttgl.static_range(STAGES - 1):
        _issue(QD, DOD, q_s.index(j), do_s.index(j), ld_bar.index(j), QUERIES, lo, hi, j, g, rq_t, rq_h, rd_t, rd_h,
               GROUP, QC, D, idx_l, j < n_it)

    key = kstart + ttgl.arange(0, BK, layout=col_l)
    kin = key < kend
    kblock = ttgl.arange(0, BK, layout=col_l) // 4
    offs_r = ttgl.arange(0, QC, layout=row_l)
    for it in range(0, n_it):
        s = it % STAGES
        qb = q_s.index(s)
        dob = do_s.index(s)
        mbarrier.wait(ld_bar.index(s), (it // STAGES) & 1)
        tcgen05_mma(qb, k_s.permute((1, 0)), s_t, use_acc=False)
        tcgen05_mma(dob, v_s.permute((1, 0)), dp_t, use_acc=False)
        tcgen05_commit(s_bar)
        chunk = it // GROUP
        head = g * GROUP + (it - chunk * GROUP)
        idx = lo + chunk * QC + offs_r
        im = idx < hi
        qid = ttgl.load(QUERIES + idx, mask=im, other=0)
        bits = ttgl.load(BITS + idx, mask=im, other=0).to(ttgl.int32)
        qid64 = qid.to(ttgl.int64)
        lse = ttgl.load(LSE + qid64 * H + head, mask=im, other=0.0)
        delta = ttgl.load(DELTA + qid64 * H + head, mask=im, other=0.0)
        nxt = it + STAGES - 1
        ns = nxt % STAGES
        npred = nxt < n_it
        nchunk = nxt // GROUP
        nhead = g * GROUP + (nxt - nchunk * GROUP)
        nidx = lo + nchunk * QC + ttgl.arange(0, QC, layout=idx_l)
        nqid = ttgl.load(QUERIES + nidx, mask=(nidx < hi) & npred, other=0)
        ok = (((bits[:, None] >> kblock[None, :]) & 1) != 0) & kin[None, :] & (key[None, :] <= qid[:, None])
        # dV/dK of the previous iteration must be done before its Q/dO slot is refilled and P/dS rewritten.
        mbarrier.wait(mma_bar, (it - 1) & 1, pred=it > 0)
        nbar = ld_bar.index(ns)
        mbarrier.expect(nbar, 2 * QC * D * 2, pred=npred)
        btma.async_gather(QD, (nqid * rq_t + nhead * rq_h).to(ttgl.int32), 0, nbar, q_s.index(ns), pred=npred)
        btma.async_gather(DOD, (nqid * rd_t + nhead * rd_h).to(ttgl.int32), 0, nbar, do_s.index(ns), pred=npred)
        mbarrier.wait(s_bar, it & 1)
        sv = s_t.load(s_rl) * qk_scale
        p = ttgl.where(ok, ttgl.exp2(sv - lse[:, None]), 0.0)
        dpv = dp_t.load(s_rl)
        ds = p * (dpv - delta[:, None])
        p_s.store(p.to(ttgl.bfloat16))
        ds_s.store(ds.to(ttgl.bfloat16))
        fence_async_shared()
        ttgl.thread_barrier()
        tcgen05_mma(dob.permute((1, 0)), p_s, dvt_t, use_acc=True)
        tcgen05_mma(qb.permute((1, 0)), ds_s, dkt_t, use_acc=True)
        tcgen05_commit(mma_bar)
    mbarrier.wait(mma_bar, (n_it - 1) & 1, pred=n_it > 0)

    dkt = dkt_t.load(acc_rl)
    dvt = dvt_t.load(acc_rl)
    od = ttgl.arange(0, D, layout=ttgl.SliceLayout(1, acc_rl))
    ok_ = kstart + ttgl.arange(0, BK, layout=ttgl.SliceLayout(0, acc_rl))
    om = (ok_ < kend)[None, :]
    ok64 = ok_.to(ttgl.int64)
    ttgl.store(DK + ok64[None, :] * s_dkt + g * s_dkh + od[:, None], (dkt * sm_scale).to(ttgl.bfloat16), mask=om)
    ttgl.store(DV + ok64[None, :] * s_dvt + g * s_dvh + od[:, None], dvt.to(ttgl.bfloat16), mask=om)
    for i in ttgl.static_range(STAGES):
        mbarrier.invalidate(ld_bar.index(i))
    mbarrier.invalidate(s_bar)
    mbarrier.invalidate(mma_bar)


def run_dkdv(q, k, v, lse, delta, grad, reverse, layout, scale, QC=64, STAGES=2, num_warps=8):
    if layout['tile_blocks'] != 16:
        raise ValueError(f"the Gluon dK/dV kernel runs 64-key tiles (tile_blocks 16), got {layout['tile_blocks']}")
    total, heads, dim = q.shape
    group = heads // k.shape[1]
    dk = torch.empty(k.shape, device=q.device, dtype=q.dtype)
    dv = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    qrows, rq_t, rq_h = qg._row_view(q)
    drows, rd_t, rd_h = qg._row_view(grad)
    sl = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    qdesc = GDesc(qrows, list(qrows.shape), [dim, 1], [1, dim], sl)
    ddesc = GDesc(drows, list(drows.shape), [dim, 1], [1, dim], sl)
    _dkdv_gluon[(layout['n_tiles'], k.shape[1])](
        qdesc, ddesc, k, v, lse, delta, dk, dv, layout['tile_start'], layout['tile_end'], reverse['offsets'],
        reverse['queries'], reverse['bits'], rq_t, rq_h, rd_t, rd_h,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1),
        heads, scale * LOG2E, scale, GROUP=group, D=dim, BK=qg.BLOCK * layout['tile_blocks'], QC=QC, STAGES=STAGES,
        KV_FP8=qg._kv_fp8_rows(k, v), num_warps=num_warps)
    return dk, dv
