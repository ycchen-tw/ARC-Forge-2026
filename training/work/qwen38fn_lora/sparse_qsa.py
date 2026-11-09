"""No-cache QSA selection and Flex, selected-tile or gather training attention.

The indexer remains frozen (hard top-k has no task-loss gradient). Selection is
stored per four-token key block, not as a dense token-by-token attention mask.
The default Flex tiles conservatively cover same-document causal regions; arbitrary QSA
choices are applied inside each tile. This is exact sparse semantics, not a
claim of O(T * budget) attention FLOPs. ``mask_backend='gather'`` (qsa_gather.py,
selection from ``select_compact``) reads only each query's selected keys.
"""

import math

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_flex = torch.compile(flex_attention, dynamic=True)


# select_compact scores only key blocks below round_up(hi/4, quantum) per chunk (every later
# block is masked to -inf anyway): ~1.6x faster at 120k. This changes the GEMM/top-k widths, so
# equality with the full-width original is shown by tests (check_qsa_gather.py gate s2: score bits
# and top-k sets per width, end-to-end selections incl. tie-heavy inputs), not by construction.
# None restores the exact original shapes.
COMPACT_TRIM_QUANTUM = 2048

# select_compact after the score GEMM: "torch" (relu / head sum / mask / topk / compact_rows_into per chunk, the
# reference) or "fused" (qsa_select_fused, one kernel per chunk; bitwise the same selection, gate s2). The fused path
# runs ``chunk_size``-row chunks (the torch path keeps the caller's, 256 by default): fewer, larger launches.
COMPACT_SELECT = {'kernel': 'torch', 'chunk_size': 4096}

# The indexer score GEMM q [rows*heads, 128] x keys [width, 128]^T (--qsa-indexer). Both operands hold BF16 values
# (RoPE outputs). 'fp32' (the reference): widened to fp32, a SIMT sgemm. 'bf16': the BF16 operands on the tensor
# cores with fp32 output, how the BF16-index rollout scores (sglang qsa/mqa.py TileLang kernels: BF16 MMA, fp32
# accumulate); BF16 x BF16 products are exact in fp32, so the two differ only by the accumulation order. 'fp8': the
# ARC_QSA_INDEX_FP8=1 rollout -- Q and keys quantized to e4m3 with its power-of-two scales (index_fp8_roundtrip),
# dequantized exactly into BF16, then the 'bf16' GEMM; the rollout's prefill scores exactly these BF16 values, its
# FP8 x FP8 decode / verify kernels the same products (exact in fp32) up to the accumulation order.
INDEXER_GEMM = {'kernel': 'fp32'}
INDEXER_MODES = ('fp32', 'bf16', 'fp8')
# Training settings these options add (a checkpoint saved before them was trained with these values).
ADDED_SETTINGS = {'qsa_kv_fp8': False, 'qsa_indexer': 'fp32'}


def add_arguments(parser):
    """train_rl_service options that make the trainer's QSA compute the way the rollout server does."""
    parser.add_argument('--qsa-kv-fp8', action='store_true',
                        help='QSA attention over FP8 (e4m3, unscaled) K/V with BF16-rounded pre-scaled Q, as the FP8-KV '
                             'rollout reads its pool; straight-through gradients to the BF16 K/V (qsa_gather.KV_FP8). '
                             'Whole-model it lands FARTHER from the rollout (v3.1: |dlogp| +3.6 percent): re-rounding K/V that '
                             'already differ from the rollout\'s by a few bf16 ulps adds e4m3 flips; see '
                             'QSA_GATHER_KERNEL.md')
    parser.add_argument('--qsa-indexer', choices=INDEXER_MODES, default='fp32',
                        help='QSA indexer scoring (sparse_qsa.INDEXER_GEMM): fp32 SIMT (default), bf16 tensor-core GEMM '
                             'with fp32 accumulation as the rollout, or fp8 = the ARC_QSA_INDEX_FP8=1 rollout (e4m3 Q '
                             'and keys with power-of-two scales, dequantized, then the bf16 GEMM)')


def apply_arguments(args):
    import qsa_gather
    qsa_gather.KV_FP8['on'] = bool(args.qsa_kv_fp8)
    INDEXER_GEMM['kernel'] = args.qsa_indexer


def index_fp8_roundtrip(x, normal_only):
    """BF16 x [..., head_dim] through the rollout's FP8 index quantization and back, bit for bit: per row (last dim)
    scale = the smallest power of two with amax / scale <= 448 (amax = m * 2^e, m in [0.5, 1): 2^(e-9) if m <= 0.875
    else 2^(e-8)), e4m3 round to nearest even, then k8 * scale in BF16 (exact). Keys: sglang
    qsa_kv_pool.quantize_qsa_index_k / dequantize_qsa_index_k (scale 1 only for amax = 0). ``normal_only``: the index Q
    rule of mqa.quantize_qsa_index_q_fp8, which also gives scale 1 to an amax below the smallest normal fp32 (the two
    rules disagree further only for amax < 2^-119, far below a layer-normed index query)."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f'FP8 index quantization takes the BF16 values the rollout quantizes, got {x.dtype}')
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax)
    k = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    scaled = amax >= torch.finfo(torch.float32).tiny if normal_only else amax > 0
    scale = torch.where(scaled, torch.ldexp(torch.ones_like(amax), k), torch.ones_like(amax))
    return ((xf / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)


def _indexer_choices(indexer, hidden, position_embeddings, meta, chunk_size, skip_dense=False, trim_quantum=None,
                     raw_scores=False):
    """Yield (start, length, chunk) per segment; chunk is None for an all-selected segment,
    else an iterator of (lo, hi, chosen, complete) with the exact original scoring ops.
    ``raw_scores``: chosen is instead the fp32 score GEMM [(hi - lo) * heads, width] before relu / head sum / scale / mask.

    ``skip_dense``: chunks whose queries all see <= block_topk complete blocks yield chosen=None
    (top-k keeps every complete block there, whatever the scores). ``trim_quantum``: see
    ``COMPACT_TRIM_QUANTUM``."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    if any(p.requires_grad for p in indexer.parameters()) and not meta['train_indexer']:
        raise RuntimeError('indexer training requires a separate differentiable objective; this selector is frozen')
    if any(isinstance(module,torch.nn.Dropout) and module.p>0 for module in indexer.modules()):
        raise RuntimeError('indexer selection/KL replay requires dropout=0')
    if hidden.shape[0] != 1 or indexer.index_kv_heads != 1:
        raise ValueError('expected compact text input and one indexer KV head')
    ratio = indexer.compress_ratio
    projected = indexer.index_qk_proj(hidden)
    q, raw = projected.split([indexer.index_n_heads*indexer.index_head_dim,indexer.index_head_dim],dim=-1)
    q = indexer.q_layernorm(q.view(1,meta['total'],indexer.index_n_heads,indexer.index_head_dim))
    cos,sin = position_embeddings
    q = apply_rotary_pos_emb(q,cos=cos,sin=sin,unsqueeze_dim=2)[0]

    gemm = INDEXER_GEMM['kernel']
    if gemm not in INDEXER_MODES:
        raise ValueError(f"QSA indexer GEMM {gemm!r}")
    if gemm != 'fp32' and q.dtype != torch.bfloat16:
        raise TypeError(f'QSA indexer {gemm} scoring needs BF16 index queries, got {q.dtype}')
    if gemm == 'fp8':
        q = index_fp8_roundtrip(q, normal_only=True)  # per (token, head), as quantize_qsa_index_q_fp8

    def chunks(start, length):
        blocks = length//ratio
        pooled = raw[0,start:start+blocks*ratio].view(blocks,ratio,-1).float().mean(1).to(raw.dtype)
        pooled = indexer.k_layernorm(pooled)
        keys = apply_rotary_pos_emb(pooled.unsqueeze(1),cos=cos[0,start:start+blocks*ratio:ratio],
                                   sin=sin[0,start:start+blocks*ratio:ratio]).squeeze(1)
        if gemm == 'fp32':
            keys = keys.float()
        elif keys.dtype != torch.bfloat16:
            raise TypeError(f'QSA indexer {gemm} scoring needs BF16 compressed keys, got {keys.dtype}')
        elif gemm == 'fp8':
            keys = index_fp8_roundtrip(keys, normal_only=False)  # per compressed key, as quantize_qsa_index_k
        block_ids = torch.arange(blocks,device=hidden.device)
        for lo in range(0,length,chunk_size):
            hi = min(lo+chunk_size,length)
            complete = (torch.arange(lo,hi,device=hidden.device)+1)//ratio
            if skip_dense and hi//ratio <= indexer.block_topk:
                yield lo, hi, None, complete
                continue
            width = blocks
            if trim_quantum is not None:
                width = min(blocks,max(indexer.block_topk,-(-(hi//ratio)//trim_quantum)*trim_quantum))
            if gemm == 'fp32':
                scores = (q[start+lo:start+hi].float().reshape(-1,indexer.index_head_dim)@keys[:width].T)
            else:
                scores = torch.mm(q[start+lo:start+hi].reshape(-1,indexer.index_head_dim),keys[:width].T,
                                  out_dtype=torch.float32)
            if raw_scores:
                yield lo, hi, scores, complete
                continue
            scores = scores.view(hi-lo,indexer.index_n_heads,width).relu().sum(1)/math.sqrt(indexer.index_head_dim)
            scores.masked_fill_(block_ids[None,:width]>=complete[:,None],float('-inf'))
            yield lo, hi, scores.topk(indexer.block_topk,dim=-1).indices, complete

    start = 0
    for length in meta['lengths']:
        yield start, length, None if length//ratio <= indexer.block_topk else chunks(start, length)
        start += length


@torch.no_grad()
def select_blocks(indexer, hidden, position_embeddings, meta, chunk_size=256):
    ratio = indexer.compress_ratio
    table = torch.zeros((meta['total'],meta['max_length']//ratio+1),device=hidden.device,dtype=torch.bool)
    for start, length, chunks in _indexer_choices(indexer, hidden, position_embeddings, meta, chunk_size):
        blocks = length//ratio
        if chunks is None:
            table[start:start+length,:blocks] = True
            continue
        for lo, hi, chosen, complete in chunks:
            table[start+lo:start+hi,:blocks].scatter_(1,chosen,chosen<complete[:,None])
    if 'qsa_audit' in meta:
        offset=0
        records=[]
        for length in meta['lengths']:
            selected=int(table[offset+length-1].sum())*ratio+length%ratio
            expected=min(length//ratio,indexer.block_topk)*ratio+length%ratio
            if selected!=expected:
                raise RuntimeError(f'QSA selection coverage mismatch: {selected} != {expected}')
            records.append({'length':length,'last_query_selected_tokens':selected,
                            'last_query_excluded_tokens':length-selected})
            offset+=length
        meta['qsa_audit'][indexer.layer_idx]={'segments':records,'selection_bytes':table.numel()*table.element_size()}
    return table


@torch.no_grad()
def select_compact(indexer, hidden, position_embeddings, meta, chunk_size=256):
    """Same choices as ``select_blocks`` (shared scoring and top-k), stored as a compact
    ``QSASelection`` instead of a T x T/4 table. No host synchronization unless audited."""
    from qsa_gather import QSASelection, block_dtype, compact_rows_into, dense_rows

    ratio, topk = indexer.compress_ratio, indexer.block_topk
    if ratio != 4:
        raise ValueError('compact QSA selection is specialized to compress_ratio=4')
    device = hidden.device
    blocks = torch.empty((meta['total'],topk+1),device=device,dtype=block_dtype(meta['max_length']))
    counts = torch.empty(meta['total'],device=device,dtype=torch.int32)
    error = torch.zeros(1,device=device,dtype=torch.int32)
    fused = COMPACT_SELECT['kernel'] == 'fused'
    if not fused and COMPACT_SELECT['kernel'] != 'torch':
        raise ValueError(f"QSA selection kernel {COMPACT_SELECT['kernel']!r}")
    if fused:
        from qsa_select_fused import select_rows
        if indexer.index_n_heads != 4:
            raise ValueError(f'the fused QSA selection sums 4 index heads, the indexer has {indexer.index_n_heads}')
        chunk_size = COMPACT_SELECT['chunk_size']
    for start, length, chunks in _indexer_choices(indexer, hidden, position_embeddings, meta, chunk_size,
                                                  skip_dense=True, trim_quantum=COMPACT_TRIM_QUANTUM, raw_scores=fused):
        scratch = None
        for lo, hi, chosen, _complete in [(0, length, None, None)] if chunks is None else chunks:
            if chosen is None:
                rows, valid = dense_rows(torch.arange(lo,hi,device=device),topk)
                blocks[start+lo:start+hi] = rows.to(blocks.dtype)
                counts[start+lo:start+hi] = valid
            elif fused:
                if scratch is None:
                    scratch = torch.empty((chunk_size,length//ratio),device=device,dtype=torch.float32)
                select_rows(chosen,scratch,blocks,counts,error,start+lo,lo,hi-lo,indexer.index_head_dim,topk)
            else:
                compact_rows_into(chosen,lo,blocks,counts,start+lo,error)
    # Device-side check, no host sync: a complete block lost to -inf/NaN scores fails loudly.
    torch._assert_async(error[0]==0)
    selection = QSASelection(blocks,counts)
    if 'qsa_audit' in meta:
        offset=0
        records=[]
        for length in meta['lengths']:
            tail=length%ratio
            selected=(int(counts[offset+length-1])-(tail>0))*ratio+tail
            expected=min(length//ratio,topk)*ratio+tail
            if selected!=expected:
                raise RuntimeError(f'QSA selection coverage mismatch: {selected} != {expected}')
            records.append({'length':length,'last_query_selected_tokens':selected,
                            'last_query_excluded_tokens':length-selected})
            offset+=length
        meta['qsa_audit'][indexer.layer_idx]={'segments':records,
            'selection_bytes':blocks.numel()*blocks.element_size()+counts.numel()*counts.element_size()}
    return selection


def block_layout(lengths, device, tile=128):
    """Conservative coarse tiles without constructing a T-by-T token mask."""
    total = sum(lengths)
    count = math.ceil(total/tile)
    starts = np.repeat(np.cumsum([0]+lengths[:-1]),lengths)
    indices = np.zeros((count,count),dtype=np.int32)
    counts = np.zeros(count,dtype=np.int32)
    for row in range(count):
        first = int(starts[row*tile])//tile
        # A query tile can cross a document boundary: fine mask still isolates it.
        columns = np.arange(first,row+1,dtype=np.int32)
        counts[row] = len(columns)
        indices[row,:len(columns)] = columns
    return (torch.tensor(counts,device=device)[None,None],torch.tensor(indices,device=device)[None,None])


def make_mask(table, meta, ratio):
    rows, positions = meta['rows'],meta['positions']
    total = meta['total']
    # Flex supplies int32 query indices. Cast before multiplying by the table
    # row stride: at 128k the boolean table exceeds 2**31 elements.
    wide_indices = table.numel() >= 2**31

    def mask(batch, head, query, key):
        q, k = query.clamp(max=total-1),key.clamp(max=total-1)
        if wide_indices:
            q = q.to(torch.int64)
        qp,kp = positions[q],positions[k]
        complete = (qp+1)//ratio
        selected = table[q,kp//ratio] | (kp//ratio==complete)
        return (query<total)&(key<total)&(rows[q]==rows[k])&(kp<=qp)&selected

    return mask


def sparse_attention(query,key,value,table,meta,ratio,scale,mask_backend='generic'):
    import qsa_gather
    if qsa_gather.KV_FP8['on'] and mask_backend!='gather':
        raise ValueError(f'--qsa-kv-fp8 is implemented by the gather backend only, not {mask_backend!r}')
    if mask_backend=='gather':
        from qsa_gather import QSASelection, qsa_gather_attention
        if not isinstance(table,QSASelection) or ratio!=4:
            raise TypeError('gather backend needs a compact QSASelection (select_compact) and ratio 4')
        layout=meta['qsa_replay'].layout(meta,query.device)
        output=qsa_gather_attention(query[0].transpose(0,1),key[0].transpose(0,1),value[0].transpose(0,1),
                                    table,layout,scale)
        return output.unsqueeze(0),None
    if mask_backend=='block_sparse':
        from qsa_block_sparse_upstream import qsa_block_sparse_attention_triton
        if meta['segments']!=1 or ratio!=4 or meta['total']%64:
            raise ValueError('block_sparse probe requires single segment, ratio4, T multiple64')
        total=meta['total']
        rows=torch.arange(total,device=query.device,dtype=torch.int32)
        selected=table[:,:total//4].clone()
        # The indexer selects complete blocks; QSA also includes the current
        # incomplete block. Causal hi excludes its not-yet-visible tokens.
        tail=(rows+1)//4
        valid=tail<total//4
        selected[rows[valid].long(),tail[valid].long()]=True
        zeros=torch.zeros_like(rows)
        output=qsa_block_sparse_attention_triton(
            query[0].transpose(0,1),key[0].transpose(0,1),value[0].transpose(0,1),
            selected,zeros,rows,zeros,zeros,scale,4)
        return output.unsqueeze(0),None
    options={'BACKEND':'TRITON'}
    if mask_backend=='generic':
        mask = make_mask(table,meta,ratio)
    elif mask_backend in ('single','single_tma'):
        if meta['segments']!=1 or meta['lengths']!=[meta['total']]:
            raise ValueError('single-segment QSA mask requires exactly one contiguous segment')
        if ratio!=4:
            raise ValueError('specialized Qwen3.8 QSA mask requires compress_ratio=4')
        total=meta['total']
        wide=table.numel()>=2**31
        def mask(batch,head,q,k):
            qi,ki=q.clamp(max=total-1),k.clamp(max=total-1)
            if wide:
                qi=qi.to(torch.int64)
            # Literal four is intentional: dynamic=True otherwise turns a
            # captured ratio into costly runtime integer div/mod instructions.
            return (q<total)&(k<total)&(k<=q)&(table[qi,ki//4]|(ki//4==(qi+1)//4))
        if mask_backend=='single_tma':
            if total%128:
                raise ValueError('TMA diagnostic requires a multiple-of-128 context')
            # A valid QSA row can start with wholly masked tiles. Keep the
            # online-softmax -inf safety check even though the complete row
            # eventually contains keys.
            options.update(BLOCKS_ARE_CONTIGUOUS=True,USE_TMA=True)
    else:
        raise ValueError('unknown QSA mask backend')
    counts,indices = meta['qsa_block_layout']
    block_mask = BlockMask.from_kv_blocks(counts,indices,BLOCK_SIZE=128,mask_mod=mask,
                                         seq_lengths=(meta['total'],meta['total']))
    output = _flex(query,key,value,block_mask=block_mask,scale=scale,enable_gqa=True,
                   kernel_options=options)
    return output.transpose(1,2),None


def full_attention(query,key,value,meta,scale):
    """Diagnostic causal full attention with the same Flex Triton backend."""
    rows,positions,total = meta['rows'],meta['positions'],meta['total']
    def mask(batch,head,query,key):
        q,k=query.clamp(max=total-1),key.clamp(max=total-1)
        return (query<total)&(key<total)&(rows[q]==rows[k])&(positions[k]<=positions[q])
    counts,indices=meta['qsa_block_layout']
    block_mask=BlockMask.from_kv_blocks(counts,indices,BLOCK_SIZE=128,mask_mod=mask,
                                       seq_lengths=(total,total))
    output=_flex(query,key,value,block_mask=block_mask,scale=scale,enable_gqa=True,
                 kernel_options={'BACKEND':'TRITON'})
    return output.transpose(1,2),None
