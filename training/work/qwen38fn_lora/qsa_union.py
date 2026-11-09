"""Query-union QSA gather attention (forward / dQ) on Blackwell tcgen05 tiles.

The gather kernels in qsa_gather.py run one query x one KV head per program with the 12 q heads as an M=16 mma.sync
tile. Here one program takes G consecutive query tokens x one KV head: M = 12 G rows (G = 10 -> 120, padded to a 128-row
tcgen05 tile), iterating the UNION of the G queries' selected 4-token blocks (ascending key order) BN keys at a time;
each row is masked to its own query's selection, so the result is QSA exactly (masked entries contribute nothing).

Union per group (``build_union``, no host sync):
  * ``_union_scan`` (persistent, one scratch bitmap per program): every selected block ORs its query's bit into a
    per-block mask (dense global block numbering, segment_layout ``block_base`` / ``block_key``); a scan over the
    group's block range then writes the union in ascending key order (popcount positions, no sort) and clears the
    scratch for the program's next group.
  * ``_union_words``: per chunk of BN/4 union entries, one 64-bit visibility word per query (bit 4e+b: key b of entry
    e is selected by the query and <= its position -- the causal cut of the tail block folded in).
The kernels read one word per row per chunk. Groups are G consecutive tokens of one segment (they restart at every
segment start), so a packed bin computes every segment bitwise as it would alone.

Kernels (Gluon, warp specialized): two load partitions TMA-gather the union's K and V rows (64 per chunk) into rings,
one warp issues the tcgen05 MMAs, the default warpgroup does the masked softmax (forward) or dS (dQ); P / dS go back
into the S accumulator's TMEM and feed the next MMA from there.

Softmax: online in log2 units like qsa_gather, with the running max raised only when a row's max grows by more than
TAU (FA4's conditional rescale; exact, P <= 2**TAU before the BF16 cast). LSE = m + log2(l) as in qsa_gather, so the
dK/dV kernels are unchanged.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory,
                                                                 fence_async_shared, get_tmem_reg_layout, mbarrier,
                                                                 tcgen05_commit, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.blackwell import tma as btma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GDesc
from triton.language.extra import libdevice

import qsa_gather as qg
from qsa_gather import LOG2E

TAU = 8.0          # log2 headroom before the running max is raised (FA4)
BIG = tl.constexpr(2147483647)
SCAN_PROGRAMS_PER_SM = 4


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _union_scan(BLOCKS, COUNTS, BLKBASE, BLKKEY, SCRATCH, UKEY, UMASK, UCOUNT, GSTART, GCOUNT, s_b, CAP, NSCR, n_groups,
                G: tl.constexpr, GP: tl.constexpr, WP: tl.constexpr, NW: tl.constexpr):
    pid = tl.program_id(0)
    scratch = SCRATCH + pid.to(tl.int64) * NSCR
    rr = tl.arange(0, GP)
    cc = tl.arange(0, WP)
    wi = tl.arange(0, NW)
    bi = tl.arange(0, 32)
    below = (1 << bi) - 1
    bit = tl.zeros([GP, WP], tl.int32) + (1 << rr)[:, None]
    for gid in range(pid, n_groups, tl.num_programs(0)):
        q = tl.load(GSTART + gid) + rr
        rok = rr < tl.load(GCOUNT + gid)
        cnt = tl.load(COUNTS + q, mask=rok, other=0)
        valid = rok[:, None] & (cc[None, :] < cnt[:, None])
        blk = tl.load(BLOCKS + q[:, None].to(tl.int64) * s_b + cc[None, :], mask=valid, other=0).to(tl.int32)
        gb = tl.load(BLKBASE + q, mask=rok, other=0)[:, None] + blk
        lo = tl.min(tl.where(valid, gb, BIG))
        hi = tl.max(tl.where(valid, gb, -1))
        # OR is order independent: the masks are deterministic whatever order the atomics land in
        tl.atomic_or(scratch + (gb - lo), bit, mask=valid, sem='relaxed')
        tl.debug_barrier()
        total = 0
        ubase = gid.to(tl.int64) * CAP
        for start in range(lo, hi + 1, NW * 32):
            idx = start + wi[:, None] * 32 + bi[None, :]
            inr = idx <= hi
            m = tl.load(scratch + (idx - lo), mask=inr, other=0, cache_modifier='.cg')
            hit = m != 0
            word = tl.reduce(tl.where(hit, 1 << bi[None, :], 0), 1, _or)
            pc = libdevice.popc(word)
            pos = total + (tl.cumsum(pc, 0) - pc)[:, None] + libdevice.popc(word[:, None] & below[None, :])
            key = tl.load(BLKKEY + idx, mask=hit, other=0)
            tl.store(UKEY + ubase + pos, key, mask=hit)
            tl.store(UMASK + ubase + pos, m.to(tl.int16), mask=hit)
            tl.store(scratch + (idx - lo), tl.zeros_like(m), mask=inr)
            total += tl.sum(pc, 0)
        tl.store(UCOUNT + gid, total)
        tl.debug_barrier()


@triton.jit
def _union_words(UKEY, UMASK, UCOUNT, GSTART, UWORD, CAP, NCH, G: tl.constexpr, NE: tl.constexpr, CB: tl.constexpr):
    gid = tl.program_id(0)
    cb = tl.program_id(1)
    U = tl.load(UCOUNT + gid)
    if cb * CB * NE < U:
        ch = cb * CB + tl.arange(0, CB)
        e = tl.arange(0, NE)
        pos = ch[:, None] * NE + e[None, :]
        ok = pos < U
        base = gid.to(tl.int64) * CAP
        t0 = tl.load(GSTART + gid)
        key = tl.load(UKEY + base + pos, mask=ok, other=0)
        m = tl.load(UMASK + base + pos, mask=ok, other=0).to(tl.int32)
        shift = (4 * e)[None, :].to(tl.int64)
        for qi in tl.static_range(G):
            vis = tl.minimum(tl.maximum(t0 + qi - key + 1, 0), 4)
            nib = tl.where(((m >> qi) & 1) != 0, (1 << vis) - 1, 0).to(tl.int64)
            word = tl.reduce(nib << shift, 1, _or)
            tl.store(UWORD + (gid.to(tl.int64) * NCH + ch) * G + qi, word, mask=ch < NCH)


def group_table(layout, group):
    """(n_groups, first token, query count) of the union groups: ``group`` consecutive tokens of one segment. From the
    host segment lengths and the GPU offsets (no host sync); cached in the layout (one per packed batch)."""
    key = ('union_groups', group)
    if key not in layout:
        n_groups = sum(triton.cdiv(n, group) for n in layout['lengths'])
        cu = layout['cu']
        seg_groups = (cu[1:] - cu[:-1] + group - 1) // group
        group_base = torch.cumsum(seg_groups, 0) - seg_groups
        group_ids = torch.arange(n_groups, device=cu.device, dtype=torch.int64)
        group_seg = torch.searchsorted(group_base, group_ids, right=True) - 1
        gstart = cu[group_seg] + (group_ids - group_base[group_seg]) * group
        gcount = torch.clamp(cu[group_seg + 1] - gstart, max=group)
        layout[key] = (n_groups, gstart.to(torch.int32), gcount.to(torch.int32))
    return layout[key]


def build_union(selection, layout, total, group, chunk_keys):
    """Per group of ``group`` consecutive tokens of one segment (groups restart at every segment start, so a packed
    bin computes every segment exactly as it would alone): union entries (global key of each 4-key block, ascending),
    their count, and the per-(chunk, query) 64-bit visibility words. ``chunk_keys`` keys (= chunk_keys/4 entries) a
    chunk."""
    blocks, counts = selection.blocks, selection.counts
    width = blocks.shape[1]
    if not 1 <= group <= 15 or chunk_keys not in (32, 64):
        raise ValueError(f'union group {group} (1..15) / chunk {chunk_keys} keys (32 or 64)')
    if layout['total'] != total:
        raise ValueError('segment layout does not match the token count')
    ne = chunk_keys // 4
    dev = blocks.device
    n_groups, gstart, gcount = group_table(layout, group)
    cap = triton.cdiv(group * width, ne) * ne
    nch = cap // ne
    nprog = min(n_groups, SCAN_PROGRAMS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count)
    scratch = torch.zeros((nprog, layout['n_blocks']), device=dev, dtype=torch.int32)
    ukey = torch.empty((n_groups, cap), device=dev, dtype=torch.int32)
    umask = torch.empty((n_groups, cap), device=dev, dtype=torch.int16)
    ucount = torch.empty(n_groups, device=dev, dtype=torch.int32)
    _union_scan[(nprog,)](blocks, counts, layout['block_base'], layout['block_key'], scratch, ukey, umask, ucount,
                          gstart, gcount, blocks.stride(0), cap, layout['n_blocks'], n_groups,
                          G=group, GP=16, WP=triton.next_power_of_2(width), NW=128, num_warps=8)
    uword = torch.empty((n_groups, nch, group), device=dev, dtype=torch.int64)
    cb = 32
    _union_words[(n_groups, triton.cdiv(nch, cb))](ukey, umask, ucount, gstart, uword, cap, nch, G=group, NE=ne, CB=cb,
                                                   num_warps=4)
    return {'ukey': ukey, 'uword': uword, 'ucount': ucount, 'gstart': gstart, 'gcount': gcount, 'cap': cap,
            'nch': nch, 'group': group, 'n_groups': n_groups, 'chunk_keys': chunk_keys}


@gluon.jit
def _chunk_keys(UKEY, ubase, U, chunk, T, NE: ttgl.constexpr, BN: ttgl.constexpr, idx_l: ttgl.constexpr):
    j = ttgl.arange(0, BN, layout=idx_l)
    e = chunk * NE + j // 4
    ok = e < U
    key = ttgl.load(UKEY + ubase + e, mask=ok, other=0) + (j % 4)
    # padding entries / keys past the end: any in-bounds row (masked by the visibility words)
    return ttgl.minimum(ttgl.where(ok, key, 0), T - 1)


@gluon.jit
def _issue_rows(DESC, buf, bar, rows, BN: ttgl.constexpr, D: ttgl.constexpr):
    mbarrier.expect(bar, BN * D * 2)
    btma.async_gather(DESC, rows, 0, bar, buf)


@gluon.jit
def _group_rows(DESC, buf, bar, t0, gcnt, kv, r_t, r_h, GROUP: ttgl.constexpr, idx_l: ttgl.constexpr):
    """TMA-gather the 128 rows of a group: row r = query t0 + r // GROUP, head kv*GROUP + r % GROUP (rows past the
    group's gcnt queries read row 0; they are masked)."""
    r = ttgl.arange(0, 128, layout=idx_l)
    qi = r // GROUP
    tq = t0 + qi
    row = ttgl.where(qi < gcnt, tq * r_t + (kv * GROUP + r - qi * GROUP) * r_h, 0)
    btma.async_gather(DESC, row.to(ttgl.int32), 0, bar, buf)


@gluon.jit
def _load_part(KD, VD, QD, DOD, UKEY, q_s, do_s, k_s, v_s, q_bar, k_ready, k_empty, v_ready, v_empty, ubase, U, n,
               t0, gcnt, kv, T, rq_t, rq_h, rd_t, rd_h, r_t, r_h, IS_K: ttgl.constexpr, WITH_DO: ttgl.constexpr,
               G: ttgl.constexpr, GROUP: ttgl.constexpr, NE: ttgl.constexpr, BN: ttgl.constexpr, D: ttgl.constexpr,
               S: ttgl.constexpr, LW: ttgl.constexpr):
    """LW warps: the group's Q (and dO) rows (K partition only), then K (or V) of every union chunk into an S-deep
    ring. The gather offsets are spread over the warps (each issues its share of the TMA gathers: a short loop body;
    one warp issuing all 64 gather4 of a tile per chunk thrashes the instruction cache)."""
    idx_l: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.BlockedLayout([1, 4], [32, 1], [1, LW], [1, 0]))
    # keys are loaded and prefetched spread over the lanes, and widened to the gather's replicated layout only at the
    # issue (64 replicated registers per prefetched chunk spill)
    key_l: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [LW], [0])
    if IS_K:
        if WITH_DO:
            mbarrier.expect(q_bar, 2 * 128 * D * 2)
            _group_rows(QD, q_s, q_bar, t0, gcnt, kv, rq_t, rq_h, GROUP, idx_l)
            _group_rows(DOD, do_s, q_bar, t0, gcnt, kv, rd_t, rd_h, GROUP, idx_l)
        else:
            mbarrier.expect(q_bar, 128 * D * 2)
            _group_rows(QD, q_s, q_bar, t0, gcnt, kv, rq_t, rq_h, GROUP, idx_l)
        desc = KD
    else:
        desc = VD
    # the union keys of a chunk are loaded two chunks ahead (global-load latency off the issue path)
    keys = _chunk_keys(UKEY, ubase, U, 0, T, NE, BN, key_l)
    keys_n = _chunk_keys(UKEY, ubase, U, 1, T, NE, BN, key_l)
    for j in range(0, n):
        keys_nn = _chunk_keys(UKEY, ubase, U, j + 2, T, NE, BN, key_l)
        s = j % S
        rows = ttgl.convert_layout(keys * r_t + kv * r_h, idx_l)
        if IS_K:
            mbarrier.wait(k_empty.index(s), ((j // S) & 1) ^ 1)
            _issue_rows(desc, k_s.index(s), k_ready.index(s), rows, BN, D)
        else:
            mbarrier.wait(v_empty.index(s), ((j // S) & 1) ^ 1)
            _issue_rows(desc, v_s.index(s), v_ready.index(s), rows, BN, D)
        keys = keys_n
        keys_n = keys_nn


@gluon.jit
def _fwd_mma(q_s, k_s, v_s, s_t, o_t, q_bar, k_ready, k_empty, v_ready, v_empty, s_ready, s_free, p_ready, pv_done, n,
             BM: ttgl.constexpr, BN: ttgl.constexpr, SK: ttgl.constexpr, SV: ttgl.constexpr):
    """One warp: S_j = Q K_j^T into TMEM buffer j%2, then O += P_{j-1} V_{j-1} with P read from TMEM (aliasing S)."""
    p_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    mbarrier.wait(q_bar, 0)
    for j in range(0, n + 1):
        if j < n:
            b = j % 2
            mbarrier.wait(s_free.index(b), ((j // 2) & 1) ^ 1)
            sk = j % SK
            mbarrier.wait(k_ready.index(sk), (j // SK) & 1)
            tcgen05_mma(q_s, k_s.index(sk).permute((1, 0)), s_t.index(b), use_acc=False,
                        mbarriers=[s_ready.index(b), k_empty.index(sk)])
        if j > 0:
            i = j - 1
            bi = i % 2
            mbarrier.wait(p_ready, i & 1)
            sv = i % SV
            mbarrier.wait(v_ready.index(sv), (i // SV) & 1)
            p_t = s_t.index(bi).slice(0, BN // 2)._reinterpret(ttgl.bfloat16, [BM, BN], p_tl)
            tcgen05_mma(p_t, v_s.index(sv), o_t, use_acc=i > 0,
                        mbarriers=[pv_done, v_empty.index(sv), s_free.index(bi)])


@gluon.jit
def _fwd_softmax(s_t, o_t, s_ready, p_ready, pv_done, UWORD, wrow, rowok, n, qk_scale,
                 G: ttgl.constexpr, BM: ttgl.constexpr, BN: ttgl.constexpr, D: ttgl.constexpr, TAU_: ttgl.constexpr,
                 num_warps: ttgl.constexpr):
    """Default partition: masked online softmax of S_j, conditional O rescale, P_j (BF16) back into S_j's TMEM."""
    H2: ttgl.constexpr = BN // 2
    OC: ttgl.constexpr = 64
    sh_tl: ttgl.constexpr = TensorMemoryLayout([BM, H2], col_stride=1)
    sh_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, H2], sh_tl, num_warps)
    p_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    ph_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.bfloat16, [BM, H2], TensorMemoryLayout([BM, H2], col_stride=1),
                                                num_warps)
    oc_tl: ttgl.constexpr = TensorMemoryLayout([BM, OC], col_stride=1)
    oc_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, OC], oc_tl, num_warps)
    row_l: ttgl.constexpr = ttgl.SliceLayout(1, sh_rl)
    col_l: ttgl.constexpr = ttgl.SliceLayout(0, sh_rl)
    jcol = ttgl.arange(0, H2, layout=col_l)
    m_i = ttgl.full([BM], -1.0e30, ttgl.float32, layout=row_l)
    l_i = ttgl.zeros([BM], ttgl.float32, layout=row_l)
    w_cur = ttgl.load(wrow, mask=rowok, other=0)
    for j in range(0, n):
        b = j % 2
        w_next = ttgl.load(wrow + (j + 1) * G, mask=rowok & (j + 1 < n), other=0)
        w_lo = w_cur.to(ttgl.int32)
        w_hi = (w_cur >> 32).to(ttgl.int32)
        mbarrier.wait(s_ready.index(b), (j // 2) & 1)
        sb = s_t.index(b)
        s_lo = sb.slice(0, H2).load(sh_rl)
        s_hi = sb.slice(H2, H2).load(sh_rl)
        # mask the raw scores; the scale is folded into the exponent (one FFMA per element)
        sm_lo = ttgl.where(((w_lo[:, None] >> jcol[None, :]) & 1) != 0, s_lo, float('-inf'))
        sm_hi = ttgl.where(((w_hi[:, None] >> jcol[None, :]) & 1) != 0, s_hi, float('-inf'))
        m_cand = ttgl.maximum(m_i, ttgl.maximum(ttgl.max(sm_lo, 1), ttgl.max(sm_hi, 1)) * qk_scale)
        upd = m_cand > m_i + TAU_
        m_new = ttgl.where(upd, m_cand, m_i)
        alpha = ttgl.exp2(m_i - m_new)
        p_lo = ttgl.exp2(sm_lo * qk_scale - m_new[:, None])
        p_hi = ttgl.exp2(sm_hi * qk_scale - m_new[:, None])
        l_i = l_i * alpha + ttgl.sum(p_lo, 1) + ttgl.sum(p_hi, 1)
        m_i = m_new
        need = ttgl.max(upd.to(ttgl.int32), 0)
        # PV_{j-1} done: O is stable (rescale) and P_{j-1} has been read
        mbarrier.wait(pv_done, (j - 1) & 1, pred=j > 0)
        if j > 0:
            if need > 0:
                a = ttgl.convert_layout(alpha, ttgl.SliceLayout(1, oc_rl))
                for c in ttgl.static_range(D // OC):
                    sl = o_t.slice(c * OC, OC)
                    sl.store(sl.load(oc_rl) * a[:, None])
        p_t = sb.slice(0, BN // 2)._reinterpret(ttgl.bfloat16, [BM, BN], p_tl)
        p_t.slice(0, H2).store(ttgl.convert_layout(p_lo.to(ttgl.bfloat16), ph_rl))
        p_t.slice(H2, H2).store(ttgl.convert_layout(p_hi.to(ttgl.bfloat16), ph_rl))
        mbarrier.arrive(p_ready, count=1)
        w_cur = w_next
    mbarrier.wait(pv_done, (n - 1) & 1)
    return m_i, l_i


@gluon.jit
def _union_fwd(QD, KD, VD, UKEY, UWORD, UCOUNT, GSTART, GCOUNT, OUT, LSE,
               T, rq_t, rq_h, rk_t, rk_h, rv_t, rv_h, s_ot, s_oh, H, CAP, NCH, qk_scale,
               G: ttgl.constexpr, GROUP: ttgl.constexpr, D: ttgl.constexpr, BN: ttgl.constexpr,
               SK: ttgl.constexpr, SV: ttgl.constexpr, LW: ttgl.constexpr, LREGS: ttgl.constexpr, TAU_: ttgl.constexpr,
               num_warps: ttgl.constexpr):
    gid = ttgl.program_id(0)
    kv = ttgl.program_id(1)
    BM: ttgl.constexpr = 128
    NE: ttgl.constexpr = BN // 4
    OC: ttgl.constexpr = 64
    s_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    o_tl: ttgl.constexpr = TensorMemoryLayout([BM, D], col_stride=1)
    oc_tl: ttgl.constexpr = TensorMemoryLayout([BM, OC], col_stride=1)
    oc_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, OC], oc_tl, num_warps)
    sh_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, BN // 2],
                                                TensorMemoryLayout([BM, BN // 2], col_stride=1), num_warps)
    row_l: ttgl.constexpr = ttgl.SliceLayout(1, sh_rl)
    orow_l: ttgl.constexpr = ttgl.SliceLayout(1, oc_rl)
    ocol_l: ttgl.constexpr = ttgl.SliceLayout(0, oc_rl)

    q_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [BM, D], QD.layout)
    k_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [SK, BN, D], KD.layout)
    v_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [SV, BN, D], VD.layout)
    q_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    k_ready = ttgl.allocate_shared_memory(ttgl.int64, [SK, 1], mbarrier.MBarrierLayout())
    k_empty = ttgl.allocate_shared_memory(ttgl.int64, [SK, 1], mbarrier.MBarrierLayout())
    v_ready = ttgl.allocate_shared_memory(ttgl.int64, [SV, 1], mbarrier.MBarrierLayout())
    v_empty = ttgl.allocate_shared_memory(ttgl.int64, [SV, 1], mbarrier.MBarrierLayout())
    s_ready = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_free = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    p_ready = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    pv_done = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(q_bar, count=1)
    mbarrier.init(p_ready, count=1)
    mbarrier.init(pv_done, count=1)
    for i in ttgl.static_range(SK):
        mbarrier.init(k_ready.index(i), count=1)
        mbarrier.init(k_empty.index(i), count=1)
    for i in ttgl.static_range(SV):
        mbarrier.init(v_ready.index(i), count=1)
        mbarrier.init(v_empty.index(i), count=1)
    for i in ttgl.static_range(2):
        mbarrier.init(s_ready.index(i), count=1)
        mbarrier.init(s_free.index(i), count=1)
    s_t = allocate_tensor_memory(ttgl.float32, [2, BM, BN], s_tl)
    o_t = allocate_tensor_memory(ttgl.float32, [BM, D], o_tl)
    fence_async_shared()

    U = ttgl.load(UCOUNT + gid)
    n = (U + NE - 1) // NE
    ubase = gid.to(ttgl.int64) * CAP
    t0 = ttgl.load(GSTART + gid)
    gcnt = ttgl.load(GCOUNT + gid)
    r = ttgl.arange(0, BM, layout=row_l)
    qi = r // GROUP
    tq = t0 + qi
    rowok = qi < gcnt
    wrow = UWORD + (gid.to(ttgl.int64) * NCH) * G + qi

    m_i, l_i = ttgl.warp_specialize([
        (_fwd_softmax, (s_t, o_t, s_ready, p_ready, pv_done, UWORD, wrow, rowok, n, qk_scale, G, BM, BN, D, TAU_,
                        num_warps)),
        (_load_part, (KD, VD, QD, QD, UKEY, q_s, q_s, k_s, v_s, q_bar, k_ready, k_empty, v_ready, v_empty, ubase, U,
                      n, t0, gcnt, kv, T, rq_t, rq_h, rq_t, rq_h, rk_t, rk_h, True, False, G, GROUP, NE, BN, D, SK, LW)),
        (_load_part, (KD, VD, QD, QD, UKEY, q_s, q_s, k_s, v_s, q_bar, k_ready, k_empty, v_ready, v_empty, ubase, U,
                      n, t0, gcnt, kv, T, rq_t, rq_h, rq_t, rq_h, rv_t, rv_h, False, False, G, GROUP, NE, BN, D, SV, LW)),
        (_fwd_mma, (q_s, k_s, v_s, s_t, o_t, q_bar, k_ready, k_empty, v_ready, v_empty, s_ready, s_free, p_ready,
                    pv_done, n, BM, BN, SK, SV)),
    ], [LW, LW, 1], [LREGS, LREGS, 64])

    ro = ttgl.arange(0, BM, layout=orow_l)
    qo = ro // GROUP
    to = t0 + qo
    ook = qo < gcnt
    head = kv * GROUP + ro - qo * GROUP
    inv = 1.0 / ttgl.convert_layout(l_i, orow_l)
    obase = OUT + to.to(ttgl.int64) * s_ot + head * s_oh
    for c in ttgl.static_range(D // OC):
        o = o_t.slice(c * OC, OC).load(oc_rl) * inv[:, None]
        cols = c * OC + ttgl.arange(0, OC, layout=ocol_l)
        ttgl.store(obase[:, None] + cols[None, :], o.to(ttgl.bfloat16), mask=ook[:, None])
    lse = m_i + ttgl.log2(l_i)
    ttgl.store(LSE + tq.to(ttgl.int64) * H + kv * GROUP + r - qi * GROUP, lse, mask=rowok)


def _desc(x):
    rows, r_t, r_h = qg._row_view(x)
    sl = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    return GDesc(rows, list(rows.shape), [x.shape[2], 1], [1, x.shape[2]], sl), r_t, r_h


FWD_CONFIG = {'BN': 64, 'SK': 3, 'SV': 2, 'LW': 4, 'LREGS': 80, 'num_warps': 4}


def union_forward(q, k, v, union, scale):
    total, heads, dim = q.shape
    hkv = k.shape[1]
    group = heads // hkv
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError('the union kernels take BF16 K/V')
    if group * union['group'] > 128:
        raise ValueError(f"union group {union['group']} x {group} heads exceeds the 128-row tile")
    if union['chunk_keys'] != FWD_CONFIG['BN']:
        raise ValueError('union built for a different chunk width')
    out = torch.empty((total, heads, dim), device=q.device, dtype=q.dtype)
    lse = torch.empty((total, heads), device=q.device, dtype=torch.float32)
    qd, rq_t, rq_h = _desc(q)
    kd, rk_t, rk_h = _desc(k)
    vd, rv_t, rv_h = _desc(v)
    cfg = dict(FWD_CONFIG)
    _union_fwd[(union['n_groups'], hkv)](
        qd, kd, vd, union['ukey'], union['uword'], union['ucount'], union['gstart'], union['gcount'], out, lse,
        total, rq_t, rq_h, rk_t, rk_h, rv_t, rv_h, out.stride(0), out.stride(1), heads, union['cap'], union['nch'],
        scale * LOG2E, G=union['group'], GROUP=group, D=dim, TAU_=TAU, **cfg)
    return out, lse


@triton.jit
def _delta_kernel(OUT, DO, DELTA, s_ot, s_oh, s_dt, s_dh, H, D: tl.constexpr, HB: tl.constexpr):
    """Delta = rowsum(dO * O) per (token, head), fp32 (what qsa_gather._dq_kernel writes for the dK/dV kernels)."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HB)
    d = tl.arange(0, D)
    hm = h < H
    o = tl.load(OUT + t * s_ot + h[:, None] * s_oh + d[None, :], mask=hm[:, None], other=0.0)
    do = tl.load(DO + t * s_dt + h[:, None] * s_dh + d[None, :], mask=hm[:, None], other=0.0)
    tl.store(DELTA + t * H + h, tl.sum(do.to(tl.float32) * o.to(tl.float32), 1), mask=hm)


@gluon.jit
def _dq_mma(q_s, do_s, k_s, v_s, s_t, dp_t, dq_t, q_bar, k_ready, k_empty, v_ready, v_empty, s_ready, s_free, ds_ready,
            dq_done, n, BM: ttgl.constexpr, BN: ttgl.constexpr, SK: ttgl.constexpr, SV: ttgl.constexpr):
    """One warp: S_j = Q K_j^T and dP_j = dO V_j^T into TMEM buffer j%2; then dQ += dS_{j-1} K_{j-1}, dS read from TMEM
    (written by the default partition over S_{j-1})."""
    ds_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    mbarrier.wait(q_bar, 0)
    for j in range(0, n + 1):
        if j < n:
            b = j % 2
            mbarrier.wait(s_free.index(b), ((j // 2) & 1) ^ 1)
            sk = j % SK
            mbarrier.wait(k_ready.index(sk), (j // SK) & 1)
            tcgen05_mma(q_s, k_s.index(sk).permute((1, 0)), s_t.index(b), use_acc=False, mbarriers=[s_ready.index(b)])
            sv = j % SV
            mbarrier.wait(v_ready.index(sv), (j // SV) & 1)
            tcgen05_mma(do_s, v_s.index(sv).permute((1, 0)), dp_t.index(b), use_acc=False,
                        mbarriers=[s_ready.index(b), v_empty.index(sv)])
        if j > 0:
            i = j - 1
            bi = i % 2
            mbarrier.wait(ds_ready, i & 1)
            ds_t = s_t.index(bi).slice(0, BN // 2)._reinterpret(ttgl.bfloat16, [BM, BN], ds_tl)
            tcgen05_mma(ds_t, k_s.index(i % SK), dq_t, use_acc=i > 0, mbarriers=[k_empty.index(i % SK), s_free.index(bi)])
    tcgen05_commit(dq_done)


@gluon.jit
def _dq_softmax(s_t, dp_t, s_ready, ds_ready, dq_done, wrow, lse, delta, rowok, n, qk_scale,
                G: ttgl.constexpr, BM: ttgl.constexpr, BN: ttgl.constexpr, num_warps: ttgl.constexpr):
    """Default partition: P_j = exp2(S_j - lse) on the row's visible keys, dS_j = P_j (dP_j - Delta), BF16 dS_j into
    S_j's TMEM. Processed in 32-key sub-chunks; sub-chunk c is half (c % 2) of 64-key visibility word c // 2."""
    SUB: ttgl.constexpr = 32
    NSUB: ttgl.constexpr = BN // SUB
    s_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    sub_tl: ttgl.constexpr = TensorMemoryLayout([BM, SUB], col_stride=1)
    s_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, SUB], sub_tl, num_warps)
    ds_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.bfloat16, [BM, SUB], sub_tl, num_warps)
    rows_l: ttgl.constexpr = ttgl.SliceLayout(1, s_rl)
    jcol = ttgl.arange(0, SUB, layout=ttgl.SliceLayout(0, s_rl))
    lse_r = ttgl.convert_layout(lse, rows_l)
    delta_r = ttgl.convert_layout(delta, rows_l)
    w_cur = ttgl.load(wrow, mask=rowok, other=0)
    for j in range(0, n):
        b = j % 2
        w_next = ttgl.load(wrow + (((j + 1) * NSUB) // 2) * G, mask=rowok & (j + 1 < n), other=0)
        mbarrier.wait(s_ready.index(b), (j // 2) & 1)
        ds_t = s_t.index(b).slice(0, BN // 2)._reinterpret(ttgl.bfloat16, [BM, BN], s_tl)
        for sub in ttgl.static_range(NSUB):
            w = ttgl.convert_layout((w_cur >> (((j * NSUB + sub) % 2) * 32)).to(ttgl.int32), rows_l)
            s = s_t.index(b).slice(sub * SUB, SUB).load(s_rl)
            dp = dp_t.index(b).slice(sub * SUB, SUB).load(s_rl)
            live = ((w[:, None] >> jcol[None, :]) & 1) != 0
            p = ttgl.where(live, ttgl.exp2(s * qk_scale - lse_r[:, None]), 0.0)
            ds = p * (dp - delta_r[:, None])
            ds_t.slice(sub * SUB, SUB).store(ttgl.convert_layout(ds.to(ttgl.bfloat16), ds_rl))
        mbarrier.arrive(ds_ready, count=1)
        w_cur = w_next
    mbarrier.wait(dq_done, 0)


@gluon.jit
def _union_dq(QD, KD, VD, DOD, UKEY, UWORD, UCOUNT, GSTART, GCOUNT, LSE, DELTA, DQ,
              T, rq_t, rq_h, rk_t, rk_h, rv_t, rv_h, rd_t, rd_h, s_qt, s_qh, H, CAP, NCH, qk_scale, sm_scale,
              G: ttgl.constexpr, GROUP: ttgl.constexpr, D: ttgl.constexpr, BN: ttgl.constexpr,
              SK: ttgl.constexpr, SV: ttgl.constexpr, LW: ttgl.constexpr, LREGS: ttgl.constexpr,
              num_warps: ttgl.constexpr):
    gid = ttgl.program_id(0)
    kv = ttgl.program_id(1)
    BM: ttgl.constexpr = 128
    NE: ttgl.constexpr = BN // 4
    OC: ttgl.constexpr = 64
    s_tl: ttgl.constexpr = TensorMemoryLayout([BM, BN], col_stride=1)
    dq_tl: ttgl.constexpr = TensorMemoryLayout([BM, D], col_stride=1)
    oc_rl: ttgl.constexpr = get_tmem_reg_layout(ttgl.float32, [BM, OC], TensorMemoryLayout([BM, OC], col_stride=1),
                                                num_warps)
    row_l: ttgl.constexpr = ttgl.SliceLayout(1, oc_rl)
    ocol_l: ttgl.constexpr = ttgl.SliceLayout(0, oc_rl)

    q_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [BM, D], QD.layout)
    do_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [BM, D], DOD.layout)
    k_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [SK, BN, D], KD.layout)
    v_s = ttgl.allocate_shared_memory(ttgl.bfloat16, [SV, BN, D], VD.layout)
    q_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    k_ready = ttgl.allocate_shared_memory(ttgl.int64, [SK, 1], mbarrier.MBarrierLayout())
    k_empty = ttgl.allocate_shared_memory(ttgl.int64, [SK, 1], mbarrier.MBarrierLayout())
    v_ready = ttgl.allocate_shared_memory(ttgl.int64, [SV, 1], mbarrier.MBarrierLayout())
    v_empty = ttgl.allocate_shared_memory(ttgl.int64, [SV, 1], mbarrier.MBarrierLayout())
    s_ready = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_free = ttgl.allocate_shared_memory(ttgl.int64, [2, 1], mbarrier.MBarrierLayout())
    ds_ready = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    dq_done = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(q_bar, count=1)
    mbarrier.init(ds_ready, count=1)
    mbarrier.init(dq_done, count=1)
    for i in ttgl.static_range(SK):
        mbarrier.init(k_ready.index(i), count=1)
        mbarrier.init(k_empty.index(i), count=1)
    for i in ttgl.static_range(SV):
        mbarrier.init(v_ready.index(i), count=1)
        mbarrier.init(v_empty.index(i), count=1)
    for i in ttgl.static_range(2):
        mbarrier.init(s_ready.index(i), count=2)   # S and dP
        mbarrier.init(s_free.index(i), count=1)
    s_t = allocate_tensor_memory(ttgl.float32, [2, BM, BN], s_tl)
    dp_t = allocate_tensor_memory(ttgl.float32, [2, BM, BN], s_tl)
    dq_t = allocate_tensor_memory(ttgl.float32, [BM, D], dq_tl)
    fence_async_shared()

    U = ttgl.load(UCOUNT + gid)
    n = (U + NE - 1) // NE
    ubase = gid.to(ttgl.int64) * CAP
    t0 = ttgl.load(GSTART + gid)
    gcnt = ttgl.load(GCOUNT + gid)
    r = ttgl.arange(0, BM, layout=row_l)
    qi = r // GROUP
    tq = t0 + qi
    rowok = qi < gcnt
    head = kv * GROUP + r - qi * GROUP
    lse = ttgl.load(LSE + tq.to(ttgl.int64) * H + head, mask=rowok, other=0.0)
    delta = ttgl.load(DELTA + tq.to(ttgl.int64) * H + head, mask=rowok, other=0.0)
    wrow = UWORD + (gid.to(ttgl.int64) * NCH) * G + qi

    ttgl.warp_specialize([
        (_dq_softmax, (s_t, dp_t, s_ready, ds_ready, dq_done, wrow, lse, delta, rowok, n, qk_scale, G, BM, BN,
                       num_warps)),
        (_load_part, (KD, VD, QD, DOD, UKEY, q_s, do_s, k_s, v_s, q_bar, k_ready, k_empty, v_ready, v_empty, ubase, U,
                      n, t0, gcnt, kv, T, rq_t, rq_h, rd_t, rd_h, rk_t, rk_h, True, True, G, GROUP, NE, BN, D, SK, LW)),
        (_load_part, (KD, VD, QD, DOD, UKEY, q_s, do_s, k_s, v_s, q_bar, k_ready, k_empty, v_ready, v_empty, ubase, U,
                      n, t0, gcnt, kv, T, rq_t, rq_h, rd_t, rd_h, rv_t, rv_h, False, True, G, GROUP, NE, BN, D, SV, LW)),
        (_dq_mma, (q_s, do_s, k_s, v_s, s_t, dp_t, dq_t, q_bar, k_ready, k_empty, v_ready, v_empty, s_ready, s_free,
                   ds_ready, dq_done, n, BM, BN, SK, SV)),
    ], [LW, LW, 1], [LREGS, LREGS, 64])

    qbase = DQ + tq.to(ttgl.int64) * s_qt + head * s_qh
    for c in ttgl.static_range(D // OC):
        dq = dq_t.slice(c * OC, OC).load(oc_rl) * sm_scale
        cols = c * OC + ttgl.arange(0, OC, layout=ocol_l)
        ttgl.store(qbase[:, None] + cols[None, :], dq.to(ttgl.bfloat16), mask=rowok[:, None])


DQ_CONFIG = {'BN': 32, 'SK': 3, 'SV': 3, 'LW': 4, 'LREGS': 80, 'num_warps': 4}


def union_dq(q, k, v, out, lse, grad, union, scale):
    """dQ [T,H,D] and Delta [T,H] (fp32, for the dK/dV kernels) of the union forward's output ``out`` / ``lse``."""
    total, heads, dim = q.shape
    hkv = k.shape[1]
    group = heads // hkv
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError('the union kernels take BF16 K/V')
    if union['chunk_keys'] != 64 or DQ_CONFIG['BN'] not in (32, 64):
        raise ValueError('the dQ kernel takes 32- or 64-key chunks of the 64-key visibility words')
    delta = torch.empty((total, heads), device=q.device, dtype=torch.float32)
    _delta_kernel[(total,)](out, grad, delta, out.stride(0), out.stride(1), grad.stride(0), grad.stride(1), heads,
                            D=dim, HB=triton.next_power_of_2(heads), num_warps=4)
    dq = torch.empty((total, heads, dim), device=q.device, dtype=q.dtype)
    qd, rq_t, rq_h = _desc(q)
    kd, rk_t, rk_h = _desc(k)
    vd, rv_t, rv_h = _desc(v)
    dd, rd_t, rd_h = _desc(grad)
    _union_dq[(union['n_groups'], hkv)](
        qd, kd, vd, dd, union['ukey'], union['uword'], union['ucount'], union['gstart'], union['gcount'], lse, delta,
        dq,
        total, rq_t, rq_h, rk_t, rk_h, rv_t, rv_h, rd_t, rd_h, dq.stride(0), dq.stride(1), heads, union['cap'],
        union['nch'], scale * LOG2E, scale, G=union['group'], GROUP=group, D=dim, **DQ_CONFIG)
    return dq, delta
