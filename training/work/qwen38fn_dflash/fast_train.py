"""Fast training path for the drafter: model.MTP3's clean trace, CleanCache.before and propose, on the model's own
parameters, without a per-anchor copy of the window.

The reference (train.py: clean_trace -> CleanCache.before -> MTP3.propose) runs refresh in 1024-token chunks, clones
every position's K/V and all four conv boundaries and concatenates them, then copies a 1024-row K/V window for every
anchor (about 10 MiB an anchor over five layers) on which SDPA, given a mask, runs its FP32 math kernel with GQA
expanded. Here:

- ``clean_pass``: a span of the sequence at once, layer by layer, attention by FlexAttention with the causal
  1024-position window. Keeps every position's K/V and the four conv boundaries only at the rows anchors read
  (anchor - 1).
- ``propose``: every anchor's eight slots attend to [clean K/V ++ all the blocks' K/V] in one block-sparse
  FlexAttention -- the clean rows before the anchor that are inside the window, and the anchor's own block causally.

Everything else is the model's own modules (hyper-connection mixers, grouped convs, MLPs, norms, rotary), called in the
order MTP3.forward and Decoder.forward call them. Same contract: token t pairs with the target GR at t - 1, slot j
predicts x[a + j + 1].

Training runs both under autograd, so the loss reaches every parameter through the context as well as through the
blocks: the clean K/V the slots read and the conv boundaries at anchor - 1 are the clean pass's outputs, not constants
(serving computes them with the same weights; detaching them dropped 52-74% of the backbone's gradient). A clean pass
starting at position `first` > 1 is exact from first + receptive_field on: rows before it lack their history.
test_fast_train_gpu.py checks values and per-parameter gradients against full autograd through model.MTP3.

The plain residual (Config.residual "plain", model.PlainDecoder) runs the same steps on one stream with its norms. Under
context "inject" the clean pass is the input fusion and each layer's injected K/V only (PlainDecoder.context_kv, no
trunk rows and no conv boundaries); the proposals' convs start from zero. test_plain_fast_train_gpu.py checks it.
"""

from dataclasses import dataclass

import torch
from attention_sink import rescale_attention
from model import PlainDecoder, canonical
from prefix_gate import diagnostics as gate_diagnostics, prefix_weights
from robust_objective import additive_terms as robust_additive_terms, tail_mass_from_logits
from torch import Tensor
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

BLOCK = 128
# Past its recompile limit a compiled function silently runs eager: several times slower, and for flex_attention the
# unfused path that materializes every score. Fail instead (probes/recompile_probe_fast.py counts the variants).
torch._dynamo.config.fail_on_recompile_limit_hit = True
# The variants are bounded but more than the default 8: grad on/off x the residual's dtype x a few token-count ranges
# (inductor switches to 64-bit indexing once a tensor passes 2^31 elements, e.g. 40960 * tokens, and its cache
# carries such range guards between runs).
torch._dynamo.config.recompile_limit = 32
_flex = torch.compile(flex_attention, dynamic=True)
_rescale_sink = torch.compile(rescale_attention, dynamic=True)
_teacher_tail_mass = torch.compile(tail_mass_from_logits, dynamic=True)
_robust_terms = torch.compile(robust_additive_terms, dynamic=True)
# Tiles for head_dim 256 on B200 (probes/flex_tuning_probe.py). Inductor's sm_100 defaults are 64 x 32 / 4 warps
# forward and 64 x 64 / 4 warps backward; these run the forward 1.5-2x and the backward 2-2.3x faster. TMA helps the
# forward only. (Backward BLOCK_N1 128 dies with an unspecified launch failure; max-autotune benchmarks the block mask
# as dense and picks slower tiles.) IS_DIVISIBLE drops the bounds checks: anchor_block_mask checks that the proposal's
# queries and keys are whole tiles. The clean pass's sequence is not, so it keeps them.
_FORWARD_OPTIONS = {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2, "USE_TMA": True}
_BACKWARD_OPTIONS = {
    "bwd_BLOCK_M1": 64, "bwd_BLOCK_N1": 64, "bwd_BLOCK_M2": 128, "bwd_BLOCK_N2": 64,
    "bwd_num_warps": 8, "bwd_num_stages": 1,
}
_CLEAN_OPTIONS = {**{"fwd_" + k: v for k, v in _FORWARD_OPTIONS.items()}, **_BACKWARD_OPTIONS}
_PROPOSAL_OPTIONS = {"IS_DIVISIBLE": True, **_CLEAN_OPTIONS}


def attend_with_sink(q, k, v, mask, options, sink_logits):
    if sink_logits is None:
        return _flex(q, k, v, block_mask=mask, enable_gqa=True, kernel_options=options)
    output, aux = _flex(q, k, v, block_mask=mask, enable_gqa=True, kernel_options=options,
                        return_aux=AuxRequest(lse=True))
    return _rescale_sink(output, aux.lse, sink_logits)


def receptive_field(config):
    """How far back a proposal depends on the clean rows: per layer the attention window (the keys at a - 1 back to
    a + j - window + 1) plus one previous row for each of the grouped convs on the attention's output and the MLP's input
    and output (test_receptive_field.py measures it on the reference). An injected clean row depends on its own token
    and predecessor GR only, so with context "inject" it is the window."""
    if config.context == "inject":
        return config.window
    return config.num_layers * (config.window + 3)


def _widen(indices, columns):
    """BlockMask takes the index table's width as the key tile count; entries past each row's count are unread."""
    return torch.nn.functional.pad(indices, (0, columns - indices.shape[-1]))


def _attention_inputs(attn, x, positions):
    """Attention.forward's projections: q (normed, rotated), output gate, k (normed, rotated), v."""
    c = attn.c
    q, gate = attn.q_proj(x).unflatten(-1, (c.num_attention_heads, 2 * c.head_dim)).chunk(2, -1)
    q = attn.rotary(attn.q_norm(q).transpose(1, 2), positions)
    k = attn.rotary(
        attn.k_norm(attn.k_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim))).transpose(1, 2), positions
    )
    v = attn.v_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim)).transpose(1, 2)
    return q, gate, k, v


def _attention_output(attn, y, gate):
    """Attention.forward after the kernel: y [batch, heads, seq, dim]."""
    return attn.o_proj(y.transpose(1, 2).flatten(-2) * torch.sigmoid(gate.flatten(-2)))


def _pre_attention(layer, h, positions, boundary, pack):
    """Decoder.forward up to the attention kernel. pack: [blocks, heads, b, d] -> [1, heads, blocks * b, d]."""
    raw_a, gate = layer.attn_hyper_connection(h)
    x, delta = layer.attention_conv.prepare(raw_a, boundary)
    q, attn_gate, k, v = _attention_inputs(layer.self_attn, x, positions)
    if pack:
        q, k, v = (t.transpose(0, 1).flatten(1, 2)[None] for t in (q, k, v))
    return raw_a, gate, delta, q, attn_gate, k, v


# The residual stream's storage dtype. None keeps what autocast gives: BF16 into layer 0, FP32 after it (GroupedConv's
# FP32 base kernel promotes the update). Serving runs the whole drafter in BF16 (arc_dflash/worker.py), so
# torch.bfloat16 rounds every update to BF16 like it -- the math before each rounding stays as it is. Set it before the
# first compile (set_residual_dtype); the compiled graphs guard on it.
RESIDUAL_DTYPE = None


def set_residual_dtype(dtype):
    global RESIDUAL_DTYPE
    if dtype not in (None, torch.bfloat16):
        raise ValueError(f"residual dtype {dtype}: None (autocast's) or torch.bfloat16")
    RESIDUAL_DTYPE = dtype


def _residual(h):
    return h if RESIDUAL_DTYPE is None else h.to(RESIDUAL_DTYPE)


def _post_attention(layer, h, y, attn_gate, gate, delta, boundaries, pack):
    """Decoder.forward after the attention kernel; boundaries are the attention output's and the MLP's three."""
    if pack:
        y = y[0].unflatten(1, (h.shape[0], h.shape[1])).transpose(0, 1)
    raw_b = _attention_output(layer.self_attn, y, attn_gate)
    y = layer.attention_conv.convolve(raw_b, delta, 1, boundaries[0])
    h = _residual(h + y.unsqueeze(-2) * gate.unsqueeze(-1))
    raw_c, gate = layer.mlp_hyper_connection(h)
    x, delta = layer.mlp_conv.prepare(raw_c, boundaries[1])
    raw_d = layer.mlp(x)
    y = layer.mlp_conv.convolve(raw_d, delta, 1, boundaries[2])
    return _residual(h + y.unsqueeze(-2) * gate.unsqueeze(-1)), raw_b, raw_c, raw_d


def _pre_attention_plain(layer, h, positions, boundary, pack):
    """PlainDecoder.forward up to the attention kernel (_pre_attention on one stream)."""
    raw_a = layer.input_layernorm(h)
    x, delta = layer.attention_conv.prepare(raw_a, boundary)
    q, attn_gate, k, v = _attention_inputs(layer.self_attn, x, positions)
    if pack:
        q, k, v = (t.transpose(0, 1).flatten(1, 2)[None] for t in (q, k, v))
    return raw_a, delta, q, attn_gate, k, v


def _post_attention_plain(layer, h, y, attn_gate, delta, boundaries, pack):
    """PlainDecoder.forward after the attention kernel (_post_attention on one stream)."""
    if pack:
        y = y[0].unflatten(1, (h.shape[0], h.shape[1])).transpose(0, 1)
    raw_b = _attention_output(layer.self_attn, y, attn_gate)
    h = _residual(h + layer.attention_conv.convolve(raw_b, delta, 1, boundaries[0]))
    raw_c = layer.post_attention_layernorm(h)
    x, delta = layer.mlp_conv.prepare(raw_c, boundaries[1])
    raw_d = layer.mlp(x)
    return _residual(h + layer.mlp_conv.convolve(raw_d, delta, 1, boundaries[2])), raw_b, raw_c, raw_d


def _context_kv(layer, fused, positions):
    """PlainDecoder.context_kv: a clean row's injected K/V."""
    return layer.context_kv(fused, positions)


# Inductor fuses the norms, mixers, convolutions, rotary and layout changes around each GEMM. Parameters are graph
# inputs (inline_inbuilt_nn_modules), so the five layers share graphs. Variants: grad on (propose) or off (clean
# pass), and h in BF16 (layer 0) or FP32 (after it: the convolutions' FP32 base kernel promotes the residual).
# Only the token axes are dynamic (_tokens_dynamic); model widths stay static. dynamic=True would make them symbolic
# too, and then 256 anchors and head_dim 256 coincide into a guard that recompiles on the next anchor count.
_pre_attention_compiled = torch.compile(_pre_attention, fullgraph=True)
_post_attention_compiled = torch.compile(_post_attention, fullgraph=True)
_pre_attention_plain_compiled = torch.compile(_pre_attention_plain, fullgraph=True)
_post_attention_plain_compiled = torch.compile(_post_attention_plain, fullgraph=True)
_context_kv_compiled = torch.compile(_context_kv, fullgraph=True)


def _tokens_dynamic(*tensors, dims):
    for t in tensors:
        for d in dims:
            torch._dynamo.maybe_mark_dynamic(t, d)
    return tensors


def _layer(layer, h, positions, boundary, attend, pack=False):
    """Decoder.forward with the attention kernel supplied by the caller. Token axes: the sequence (dim 1) for one
    clean sequence, the blocks (dim 0) for packed proposal blocks."""
    axis = 0 if pack else 1
    _tokens_dynamic(h, positions, dims=(axis,))
    if pack:
        _tokens_dynamic(*boundary, dims=(0,))
    if isinstance(layer, PlainDecoder):
        raw_a, delta, q, attn_gate, k, v = _pre_attention_plain_compiled(layer, h, positions, boundary[0], pack)
        y = attend(q, k, v)
        _tokens_dynamic(y, dims=(2,))
        _tokens_dynamic(attn_gate, delta, dims=(axis,))
        h, raw_b, raw_c, raw_d = _post_attention_plain_compiled(layer, h, y, attn_gate, delta, boundary[1:], pack)
        return h, (raw_a, raw_b, raw_c, raw_d), k, v
    raw_a, gate, delta, q, attn_gate, k, v = _pre_attention_compiled(layer, h, positions, boundary[0], pack)
    y = attend(q, k, v)
    _tokens_dynamic(y, dims=(2,))
    _tokens_dynamic(attn_gate, gate, delta, dims=(axis,))
    h, raw_b, raw_c, raw_d = _post_attention_compiled(layer, h, y, attn_gate, gate, delta, boundary[1:], pack)
    return h, (raw_a, raw_b, raw_c, raw_d), k, v


def _inputs(model, embeddings, predecessor_gr):
    """MTP3.forward before the layers."""
    return model.fuse(embeddings, predecessor_gr)


@torch.compile(fullgraph=True)
def _proposal_inputs(model, embeddings, predecessor_gr):
    """_inputs for blocks whose slots share one predecessor GR [blocks, branches, hidden]: the GR half once per
    block, broadcast over the slots (MTP3.propose expands it to every slot; mask_gr "none" keeps it to slot 0)."""
    c = model.config
    if c.residual == "plain":
        gr = model.fc_hidden(model.pre_fc_norm_hidden(predecessor_gr.flatten(-2)))[:, None]
        if c.mask_gr == "none":
            gr = torch.cat((gr, torch.zeros_like(gr).expand(-1, embeddings.shape[1] - 1, -1)), 1)
        return gr + model.fc_embedding(model.pre_fc_norm_embedding(embeddings))
    h = model.pre_fc_norm_hidden(predecessor_gr.flatten(-2)).unflatten(-1, (c.hc_count, c.hidden_size))
    gr = model.fc_hidden(h)[:, None]
    if c.mask_gr == "none":  # MTP3.proposal_gr: only the anchor slot reads the GR, the mask slots' term is exactly 0
        gr = torch.cat((gr, torch.zeros_like(gr).expand(-1, embeddings.shape[1] - 1, -1, -1)), 1)
    return gr + model.fc_embedding(model.pre_fc_norm_embedding(embeddings)).unsqueeze(-2)


def window_block_mask(n, window, device):
    """Causal sliding window over one sequence of n positions: q sees k when 0 <= q - k < window."""
    if window % BLOCK:
        raise ValueError(f"window {window} must be a multiple of {BLOCK}")
    w = window // BLOCK
    blocks = -(-n // BLOCK)
    q = torch.arange(blocks, device=device, dtype=torch.int32)
    # Diagonal and the block `w` back are partial; the w - 1 blocks between are entirely inside the window.
    edge = q - w
    partial_count = 1 + (edge >= 0).to(torch.int32)
    partial = torch.stack((torch.where(edge >= 0, edge, q), q), -1)
    full_start = (q - w + 1).clamp(min=0)
    full_count = q - full_start
    offsets = torch.arange(max(w - 1, 1), device=device, dtype=torch.int32)
    full = (full_start[:, None] + offsets).clamp(max=blocks - 1)

    def mask_mod(b, h, qi, ki):
        return (qi >= ki) & (qi - ki < window)

    return BlockMask.from_kv_blocks(
        partial_count[None, None], _widen(partial, blocks)[None, None], full_count[None, None],
        _widen(full, blocks)[None, None], BLOCK_SIZE=BLOCK, mask_mod=mask_mod, seq_lengths=(n, n),
    )


def _tile_table(starts, counts, width):
    """Per query tile, contiguous key-tile ranges (start, count) [tiles, ranges] -> BlockMask's index table
    [tiles, width] (entries past a row's count are unread, zero here) and the per-tile count."""
    ends = counts.cumsum(-1)
    step = torch.arange(width, device=starts.device)
    which = (step[None, :, None] >= ends[:, None, :]).sum(-1).clamp(max=starts.shape[1] - 1)
    index = starts.gather(1, which) + step - (ends - counts).gather(1, which)
    # canonical: with one query tile, ends[:, -1] is a one-element view of stride 3 that .to() keeps (one element counts
    # as contiguous), and flex_attention's compiled graph guards on the stride: a recompile per such step.
    return canonical(torch.where(step < ends[:, -1:], index, 0).to(torch.int32)), canonical(ends[:, -1].to(torch.int32))


def anchor_block_mask(anchors, block_size, clean_first, clean_length, clean_padded, window, clean_grad):
    """Queries: anchor i's slot j at row i * block_size + j, position anchors[i] + j (anchors sorted). Keys: clean
    rows 0..clean_length-1 (position clean_first + row), padding up to clean_padded, then every anchor's block in query
    order. Slot j sees clean positions p < anchor with anchor + j - p < window -- rows [anchor + j - window + 1,
    anchor - 1] - clean_first -- and its own block's slots <= j -- rows clean_padded + i * block_size + [0, j].

    Both ranges are precomputed per query row, so mask_mod is four compares. Key tiles inside the clean range of every
    row of a query tile are full blocks (no mask_mod). The key-tile -> query-tile tables are what backward walks for
    dK/dV: with clean_grad the clean tiles' dK/dV are computed (the clean K/V are the clean pass's outputs under
    autograd); without it (no-grad clean K/V, evaluation) those tables are emptied and the clean tiles' dK/dV are
    written as zeros instead. No host sync.
    """
    if BLOCK % block_size or len(anchors) % (BLOCK // block_size) or clean_padded % BLOCK:
        raise ValueError("anchor count must fill whole query tiles and the clean keys whole key tiles")
    rows = len(anchors) * block_size
    keys = clean_padded + rows
    q = torch.arange(rows, device=anchors.device)
    anchor = anchors.repeat_interleave(block_size) - clean_first
    slot = q % block_size
    lo, hi = anchor + slot - window + 1, (anchor - 1).clamp(max=clean_length - 1)
    own_lo, own_hi = clean_padded + q - slot, clean_padded + q
    lo_min, lo_max = lo.view(-1, BLOCK).amin(1), lo.view(-1, BLOCK).amax(1)
    hi_min, hi_max = hi.view(-1, BLOCK).amin(1), hi.view(-1, BLOCK).amax(1)
    start = lo_min.clamp(min=0)
    reads = hi_max >= start                                   # the query tile reads some clean row
    c0, c1 = start // BLOCK, hi_max // BLOCK                  # clean key tiles [c0, c1]
    f0 = (lo_max.clamp(min=0) + BLOCK - 1) // BLOCK           # every row sees all of key tiles [f0, f1)
    f1 = (hi_min + 1) // BLOCK
    full = torch.where(reads, f1 - f0, 0).clamp(min=0)
    f0, f1 = (torch.where(full > 0, f, c1 + 1) for f in (f0, f1))
    o0, o1 = own_lo.view(-1, BLOCK).amin(1) // BLOCK, own_hi.view(-1, BLOCK).amax(1) // BLOCK
    # Partial: clean tiles before and after the full ones, then the tile of the anchors' own blocks.
    starts = torch.stack((c0, f1, o0), 1)
    counts = torch.stack((torch.where(reads, f0 - c0, 0), torch.where(reads, c1 - f1 + 1, 0), o1 - o0 + 1), 1)
    kv_indices, kv_count = _tile_table(starts, counts, keys // BLOCK)
    full_indices, full_count = _tile_table(f0[:, None], full[:, None], keys // BLOCK)
    lo, hi, own_lo, own_hi = (t.to(torch.int32) for t in (lo, hi, own_lo, own_hi))

    def mask_mod(b, h, qi, ki):
        return ((ki >= lo[qi]) & (ki <= hi[qi])) | ((ki >= own_lo[qi]) & (ki <= own_hi[qi]))

    mask = BlockMask.from_kv_blocks(
        kv_count[None, None], kv_indices[None, None], full_count[None, None], full_indices[None, None],
        BLOCK_SIZE=BLOCK, mask_mod=mask_mod, seq_lengths=(rows, keys),
    )
    if clean_grad:
        return mask
    clean = torch.arange(keys // BLOCK, device=anchors.device) < clean_padded // BLOCK
    return BlockMask(
        mask.seq_lengths, mask.kv_num_blocks, mask.kv_indices, mask.full_kv_num_blocks, mask.full_kv_indices,
        mask.q_num_blocks.masked_fill(clean, 0), mask.q_indices, mask.full_q_num_blocks.masked_fill(clean, 0),
        mask.full_q_indices, mask.BLOCK_SIZE, mask.mask_mod,
    )


@dataclass
class CleanTrace:
    """The drafter's clean K/V for positions first .. first + length - 1 (row = position - first), padded to whole key
    tiles, and the four conv boundaries of every layer at the rows ``anchor - 1`` of the anchors it was built for
    (none under context "inject": the clean rows do not run the trunk)."""

    keys: list[Tensor]               # per layer [1, kv heads, padded, head dim]
    values: list[Tensor]
    first: int
    length: int
    anchors: Tensor                  # sorted, unique
    boundaries: list[tuple]          # per layer, 4 x [len(anchors), 1, hidden]; () under context "inject"

    def boundaries_for(self, anchors):
        rows = torch.searchsorted(self.anchors, anchors)
        if not torch.equal(self.anchors[rows], anchors):
            raise ValueError("anchors outside the set the trace was built for")
        return [tuple(b[rows] for b in layer) for layer in self.boundaries]

    def tensors(self):
        return [*self.keys, *self.values, *(b for layer in self.boundaries for b in layer)]


def clean_start(anchors, config):
    """The first position of an exact clean pass for sorted anchors: their receptive field back from the first."""
    return max(1, int(anchors[0]) - receptive_field(config))


def clean_pass(model, ids, gr, anchors, first=1):
    """The clean trace over positions first .. anchors[-1] - 1 for the given anchors (sorted, unique, >= 1), under
    autograd when grad is enabled. A pass that starts after position 1 lacks the history of its first rows, so it must
    start at least receptive_field before the first anchor (clean_start)."""
    c = model.config
    if not torch.equal(anchors, anchors.unique()) or int(anchors[0]) < 1:
        raise ValueError("anchors must be sorted, unique and >= 1")
    if first < 1 or (first > 1 and int(anchors[0]) - first < receptive_field(c)):
        raise ValueError(f"a clean pass from {first} does not cover the receptive field of anchor {int(anchors[0])}")
    # Through anchor - 1 of the last anchor; one row at least (an anchor at position 1 reads none of them).
    n = max(int(anchors[-1]) - 1, first) - first + 1
    if first + n > len(ids):
        raise ValueError("the clean pass runs past the sequence")
    positions = torch.arange(first, first + n, device=ids.device)[None]
    h = _inputs(model, model.embed_tokens(ids[first : first + n])[None], gr[first - 1 : first + n - 1][None])
    padded = -(-n // BLOCK) * BLOCK
    if c.context == "inject":
        keys, values = [], []
        _tokens_dynamic(h, positions, dims=(1,))
        for layer in model.layers:
            k, v = _context_kv_compiled(layer, h, positions)
            pad = (0, 0, 0, padded - n)
            keys.append(torch.nn.functional.pad(k, pad))
            values.append(torch.nn.functional.pad(v, pad))
        return CleanTrace(keys, values, first, n, anchors, [() for _ in model.layers])
    zero = h.new_zeros(1, 1, c.hidden_size)
    mask = window_block_mask(n, c.window, ids.device)
    # Boundary at anchor - 1: row anchor - 1 - first; an anchor at position 1 has no clean predecessor, its boundary is 0.
    rows = anchors - 1 - first
    present = (rows >= 0)[:, None, None]
    keys, values, boundaries = [], [], []
    for layer in model.layers:
        def attend(q, k, v):
            return attend_with_sink(q, k, v, mask, _CLEAN_OPTIONS, layer.self_attn.sink_logits)

        h, raw, k, v = _layer(layer, h, positions, (zero,) * 4, attend)
        pad = (0, 0, 0, padded - n)
        keys.append(torch.nn.functional.pad(k, pad))
        values.append(torch.nn.functional.pad(v, pad))
        boundaries.append(tuple((r[0, rows.clamp(min=0)][:, None] * present).to(r.dtype) for r in raw))
    return CleanTrace(keys, values, first, n, anchors, boundaries)


def trace_leaves(trace):
    """The trace with every tensor replaced by a new autograd leaf holding the same values: several micro-batches of
    proposals read it and collect their gradients in the leaves, then backward_trace sends the sum through the clean
    pass once (instead of keeping every micro-batch's graph or running the clean pass per micro-batch)."""
    def leaf(t):
        return t.detach().requires_grad_()

    return CleanTrace([leaf(k) for k in trace.keys], [leaf(v) for v in trace.values], trace.first, trace.length,
                      trace.anchors, [tuple(leaf(b) for b in layer) for layer in trace.boundaries])


def backward_trace(trace, leaves):
    """Backpropagate the gradients the leaves (trace_leaves) collected through the clean pass that made the trace."""
    pairs = [(t, leaf.grad) for t, leaf in zip(trace.tensors(), leaves.tensors(), strict=True) if leaf.grad is not None]
    if len(pairs) != len(trace.tensors()):
        raise RuntimeError(f"{len(trace.tensors()) - len(pairs)} clean-trace tensors got no gradient from the proposals")
    torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])


@torch.compile(fullgraph=True)
def _proposal_output(model, h):
    return model.output(h)


def propose(model, trace, ids, gr, anchors):
    """MTP3.propose for sorted anchors on the clean trace: hidden [anchors, block - 1, hidden].

    The anchor count is padded to whole query tiles (16 blocks) with copies of the last anchor. A padded block sees
    only the clean keys and itself, and its outputs are dropped, so it changes nothing else and gets no gradient.
    """
    c = model.config
    b = c.block_size
    real = len(anchors)
    if real == 0:
        raise ValueError("no anchors")
    anchors = torch.cat((anchors, anchors[-1:].expand(-real % (BLOCK // b))))
    count = len(anchors)
    positions = anchors[:, None] + torch.arange(b, device=anchors.device)
    embeddings = torch.cat(
        (model.embed_tokens(ids[anchors])[:, None], model.mask_embedding.expand(count, b - 1, -1)), 1
    )
    h = _proposal_inputs(model, *_tokens_dynamic(embeddings, gr[anchors - 1], dims=(0,)))
    padded = trace.keys[0].shape[2]
    clean_grad = any(t.requires_grad for t in trace.keys + trace.values)
    mask = anchor_block_mask(anchors, b, trace.first, trace.length, padded, c.window, clean_grad)
    if c.context == "inject":  # PlainDecoder.forward: the block's convs start from zero
        boundaries = [(h.new_zeros(count, 1, c.hidden_size),) * 4 for _ in model.layers]
    else:
        boundaries = trace.boundaries_for(anchors)
    for i, layer in enumerate(model.layers):
        def attend(q, k, v, i=i):
            # One sequence of anchors * b queries after the shared clean keys.
            k = torch.cat((trace.keys[i], k), 2)
            v = torch.cat((trace.values[i], v), 2)
            return attend_with_sink(q, k, v, mask, _PROPOSAL_OPTIONS, layer.self_attn.sink_logits)

        h, _, _, _ = _layer(layer, h, positions, boundaries[i], attend, pack=True)
    return _proposal_output(model, *_tokens_dynamic(h, dims=(0,)))[:real, :-1]


class _HeadGradient(torch.autograd.Function):
    """ce_num and kl_num as functions of the proposal hidden states, their gradients computed in the forward pass.

    The head is frozen, so d(ce_num)/d(hidden) = sum_i w_i (softmax_i - onehot_i) W and d(kl_num)/d(hidden) =
    sum_i (softmax_i - p_teacher_i) W are fixed once the forward pass has the logits (the D-PACE weights w are
    detached). Backward only scales them by the incoming scalar gradients, which carry the global normalizers.
    """

    @staticmethod
    def forward(ctx, hidden, ce_num, kl_num, d_ce, d_kl):
        ctx.save_for_backward(d_ce, d_kl)
        ctx.dtype = hidden.dtype
        return ce_num.clone(), kl_num.clone()

    @staticmethod
    def backward(ctx, g_ce, g_kl):
        d_ce, d_kl = ctx.saved_tensors
        grad = g_ce * d_ce
        grad[:, : d_kl.shape[1]] += g_kl * d_kl
        return grad.to(ctx.dtype), None, None, None, None


@torch.compile(dynamic=True)
def _ce_gradient(logits, lse, labels, weight):
    """weight * (softmax - onehot), the CE gradient on the logits, in the logits' dtype."""
    p = torch.exp(logits.float() - lse[:, None])
    onehot = torch.arange(logits.shape[-1], device=logits.device)[None] == labels[:, None]
    return (weight[:, None] * (p - onehot.float())).to(logits.dtype)


@torch.compile(dynamic=True)
def _kl_and_gradient(student, student_lse, teacher):
    """KL(teacher || student) per row and its gradient on the student logits, softmax - p_teacher."""
    t = teacher.float()
    teacher_logp = t - torch.logsumexp(t, -1, keepdim=True)
    student_logp = student.float() - student_lse[:, None]
    p = teacher_logp.exp()
    return (p * (teacher_logp - student_logp)).sum(-1), (student_logp.exp() - p).to(student.dtype)


@dataclass
class HeadStatistics:
    """The head's part of the objective for one micro-batch, computed without autograd (head_statistics)."""

    ce: Tensor            # [anchors, b - 1] FP32
    weight: Tensor        # D-PACE weights, detached
    kl: Tensor            # [anchors, early slots] KL(teacher || student)
    unary: Tensor         # [anchors, b - 1, top_k] FP32 top-K logits
    candidates: Tensor    # [anchors, b - 1, top_k]
    d_ce: Tensor          # d(sum ce * weight) / d hidden, FP32
    d_kl: Tensor          # d(sum kl) / d hidden[:, :early slots], FP32


@torch.no_grad()
def head_statistics(model, hidden, labels, teacher_gr, anchors_per_chunk=256, valid=None):
    """Per chunk of anchors: one head GEMM, one read of the logits for the FP32 log-sum-exp, the label logit and the
    top-K on the BF16 logits (head_kernels.row_stats_topk: same values as the reference's FP32 copy, equal values
    lowest index first), D-PACE weights, then the CE and early-KL gradients on the logits and one GEMM each back to the
    hidden size. The head is frozen, so these gradients are final; the logits are never kept.
    (The reference keeps FP32 logits through a checkpoint and runs the head and the top-K again in backward.)
    valid: data.block_targets' slot mask (all slots when None).
    """
    from head_kernels import row_stats_topk
    from objective import dpace_weights
    from torch.nn import functional as F

    count, b = labels.shape
    weight_matrix = model.lm_head.weight
    early_slots = teacher_gr.shape[1]
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    flat = hidden.detach().to(weight_matrix.dtype)
    teacher_hidden = model.target_reducer(teacher_gr).to(weight_matrix.dtype)
    d_ce = torch.empty(count, b, hidden.shape[-1], device=hidden.device, dtype=torch.float32)
    d_kl = torch.empty(count, early_slots, hidden.shape[-1], device=hidden.device, dtype=torch.float32)
    ce, kl, weight, unary, candidates = [], [], [], [], []
    for start in range(0, count, anchors_per_chunk):
        end = min(start + anchors_per_chunk, count)
        rows = (end - start) * b
        y = labels[start:end].reshape(-1)
        logits = F.linear(flat[start:end].reshape(rows, -1), weight_matrix)
        lse, label_logit, top_values, top_indices = row_stats_topk(logits, y, model.candidate_selector.top_k)
        chunk_ce = (lse - label_logit).view(end - start, b)
        chunk_weight = dpace_weights((-chunk_ce).exp(), valid[start:end]) * valid[start:end]
        gradient = _ce_gradient(logits, lse, y, chunk_weight.reshape(-1))
        d_ce[start:end] = (gradient @ weight_matrix).view(end - start, b, -1)
        del gradient
        early = logits.view(end - start, b, -1)[:, :early_slots].reshape((end - start) * early_slots, -1)
        early_lse = lse.view(end - start, b)[:, :early_slots].reshape(-1)
        teacher_logits = F.linear(teacher_hidden[start:end].reshape(early.shape[0], -1), weight_matrix)
        chunk_kl, gradient = _kl_and_gradient(early, early_lse, teacher_logits)
        early_valid = valid[start:end, :early_slots]
        d_kl[start:end] = (gradient @ weight_matrix).view(end - start, early_slots, -1) * early_valid[..., None]
        del logits, early, teacher_logits, gradient
        ce.append(chunk_ce)
        kl.append(chunk_kl.view(end - start, early_slots) * early_valid)
        weight.append(chunk_weight)
        unary.append(top_values.float().view(end - start, b, -1))
        candidates.append(top_indices.view(end - start, b, -1))
    return HeadStatistics(torch.cat(ce), torch.cat(weight), torch.cat(kl), torch.cat(unary), torch.cat(candidates),
                          d_ce, d_kl)


def objective_from_statistics(model, stats, hidden, labels, predecessors, valid=None):
    """objective.objective's terms. ce_num and kl_num carry hidden's gradient from the precomputed statistics, the
    selector its own through autograd (on hidden.detach(), as the reference). The statistics may come from an earlier
    no-grad pass over the same micro-batch (train.py's two-pass step); hidden is then the recomputed proposal."""
    from torch.nn import functional as F

    count, b = labels.shape
    early_slots = stats.d_kl.shape[1]
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    ce_num, kl_num = _HeadGradient.apply(
        hidden, (stats.ce * stats.weight).sum(), (stats.kl * valid[:, :early_slots]).sum(), stats.d_ce, stats.d_kl
    )
    candidates, weight = stats.candidates, stats.weight
    matches = candidates == labels[..., None]
    covered = matches.any(-1) & valid
    selected = model.candidate_selector(candidates, stats.unary, hidden.detach(), predecessors)
    selector_ce = F.cross_entropy(
        selected.float().flatten(0, 1), matches.long().argmax(-1).flatten(), reduction="none"
    ).view_as(stats.ce)
    selector_weight = weight * covered
    chosen = candidates.gather(-1, selected.argmax(-1, keepdim=True)).squeeze(-1)
    accept = (chosen == labels) & valid
    run = accept.long().cumprod(-1)
    return {
        "ce_num": ce_num,
        "ce_den": weight.sum(),
        "selector_num": (selector_ce * selector_weight).sum(),
        "selector_den": selector_weight.sum(),
        "kl_num": kl_num,
        "kl_den": valid[:, :early_slots].sum(),
        "covered": covered.sum(),
        "positions": valid.sum(),
        "selector_correct": ((chosen == labels) & covered).sum(),
        "unary_correct": ((candidates[..., 0] == labels) & valid).sum(),
        "accept_prefix": run.sum(0),
        "accept_position": accept.sum(0),
        "covered_position": covered.sum(0),
        "accept_histogram": torch.bincount(run.sum(-1), minlength=b + 1),
        "blocks": torch.tensor(count, device=labels.device),
    }


def head_terms(model, hidden, labels, predecessors, teacher_gr, anchors_per_chunk=256, valid=None):
    """objective.objective's terms for a whole micro-batch, the head never kept for backward."""
    stats = head_statistics(model, hidden, labels, teacher_gr, anchors_per_chunk, valid)
    return objective_from_statistics(model, stats, hidden, labels, predecessors, valid)


# The acceptance objective (objective.acceptance_objective is the reference). Every slot has a teacher distribution,
# so the head's two big tensors per chunk are the drafter's and the teacher's logits; both are read once for the
# coverage KL and its gradient and never kept. Its normalizers (the step's blocks and positions) are known before any
# forward, so train.fast_step backpropagates each micro-batch as soon as it is computed: no two passes.


class _CoverageGradient(torch.autograd.Function):
    """cov_num as a function of the proposal hidden states; d(cov_num)/d(hidden) = sum_i (r_i - p_i) / T W was taken
    in the forward pass (the head is frozen), backward only scales it."""

    @staticmethod
    def forward(ctx, hidden, cov_num, d_cov):
        ctx.save_for_backward(d_cov)
        ctx.dtype = hidden.dtype
        return cov_num.clone()

    @staticmethod
    def backward(ctx, g):
        (d_cov,) = ctx.saved_tensors
        return (g * d_cov).to(ctx.dtype), None, None


class _CandidateLogits(torch.autograd.Function):
    """The head's logits at the candidates, U[r, k] = hidden[r] . W[C[r, k]], as a function of hidden. Forward returns
    the values the top-K already read (no second GEMM); backward gathers the candidates' head rows a chunk at a time,
    never all of W[C] at once (4096 anchors x 7 x 16 x 2560 would be 2.3 GB)."""

    @staticmethod
    def forward(ctx, hidden, weight, candidates, unary):
        ctx.save_for_backward(weight, candidates)
        ctx.dtype = hidden.dtype
        return unary.clone()

    @staticmethod
    def backward(ctx, g):
        weight, candidates = ctx.saved_tensors
        flat_c = candidates.reshape(-1, candidates.shape[-1])
        flat_g = g.reshape(-1, g.shape[-1])
        grad = torch.empty(flat_c.shape[0], weight.shape[1], device=g.device, dtype=torch.float32)
        for start in range(0, flat_c.shape[0], 4096):
            rows = slice(start, start + 4096)
            grad[rows] = torch.einsum("rk,rkd->rd", flat_g[rows].float(), weight[flat_c[rows]].float())
        return grad.view(*candidates.shape[:-1], -1).to(ctx.dtype), None, None, None


@torch.compile(dynamic=True)
def _coverage_and_gradient(student, teacher, inverse):
    """Per row at temperature 1 / inverse: KL(p_teacher || r_student), the gradient on the student logits
    (r - p) * inverse in their dtype, the teacher's and the student's log-sum-exp of logits * inverse."""
    s = student.float() * inverse
    t = teacher.float() * inverse
    s_lse = torch.logsumexp(s, -1, keepdim=True)
    t_lse = torch.logsumexp(t, -1, keepdim=True)
    p = torch.exp(t - t_lse)
    kl = (p * ((t - t_lse) - (s - s_lse))).sum(-1)
    return kl, ((torch.exp(s - s_lse) - p) * inverse).to(student.dtype), t_lse.squeeze(-1), s_lse.squeeze(-1)


@dataclass
class AcceptanceStatistics:
    """The head's part of the acceptance objective for one micro-batch (acceptance_statistics), no autograd."""

    kl: Tensor            # [anchors, b - 1] FP32 KL(p || r) at the temperature
    ce_t1: Tensor         # [anchors, b - 1] hard CE of the data token at T = 1 (diagnostic)
    ce_t: Tensor          # [anchors, b - 1] hard CE of the data token at the temperature (diagnostic)
    unary: Tensor         # [anchors, b - 1, top_k] FP32 top-K logits
    candidates: Tensor    # [anchors, b - 1, top_k]
    teacher_mass: Tensor  # [anchors, b - 1, top_k] FP32 p(v) at the candidates, p normalized over the vocabulary
    d_cov: Tensor         # d(sum kl) / d hidden, FP32
    teacher_tail_mass: Tensor | None = None  # full-vocabulary T_gate tail, never candidate-normalized
    anchor_tail_mass: Tensor | None = None   # [anchors] the anchor token's T_gate tail from H[a - 1] (prefix gate)


@torch.no_grad()
def acceptance_statistics(model, hidden, labels, teacher_gr, temperature, anchors_per_chunk=256, robust=None, gate=None,
                          anchor_tokens=None, anchor_gr=None):
    """Per chunk of anchors: the drafter's and the teacher's head GEMMs, the drafter's log-sum-exp / label logit / top-K
    in one read (head_kernels.row_stats_topk), the coverage KL and its gradient on the logits in one compiled pass, one
    GEMM back to the hidden size, and the teacher's probabilities at the candidates.

    robust / gate (prefix_gate.PrefixGateConfig): also every label's full-vocabulary tail at the gate temperature from
    the teacher logits already in hand; gate adds the anchors' (anchor_tokens [anchors] from anchor_gr = GR[a - 1]
    [anchors, branches, hidden]): one more teacher head GEMM per chunk, a seventh of the slots', reduced to a scalar
    per anchor at once."""
    from head_kernels import row_stats_topk
    from torch.nn import functional as F

    if robust is not None and gate is not None:
        raise ValueError("the robust objective and the prefix gate are separate objectives")
    if (gate is None) != (anchor_tokens is None) or (gate is None) != (anchor_gr is None):
        raise ValueError("the prefix gate needs the anchors' tokens and predecessor GR, and only it takes them")
    count, b = labels.shape
    inverse = 1.0 / temperature
    weight_matrix = model.lm_head.weight
    flat = hidden.detach().to(weight_matrix.dtype)
    teacher_hidden = model.target_reducer(teacher_gr).to(weight_matrix.dtype)
    if teacher_hidden.shape[:2] != labels.shape:
        raise ValueError("the acceptance objective needs the teacher GR of every slot")
    gate_temperature = (robust.gate_temperature if robust is not None
                        else gate.gate_temperature if gate is not None else None)
    if gate is not None:
        if anchor_tokens.shape != (count,) or anchor_gr.shape[0] != count:
            raise ValueError("one anchor token and one predecessor GR per block")
        anchor_hidden = model.target_reducer(anchor_gr).to(weight_matrix.dtype)
    d_cov = torch.empty(count, b, hidden.shape[-1], device=hidden.device, dtype=torch.float32)
    kl, ce_t1, ce_t, unary, candidates, mass = [], [], [], [], [], []
    tails, anchor_tails = [], []
    for start in range(0, count, anchors_per_chunk):
        end = min(start + anchors_per_chunk, count)
        rows = (end - start) * b
        y = labels[start:end].reshape(-1)
        logits = F.linear(flat[start:end].reshape(rows, -1), weight_matrix)
        lse, label_logit, top_values, top_indices = row_stats_topk(logits, y, model.candidate_selector.top_k)
        teacher_logits = F.linear(teacher_hidden[start:end].reshape(rows, -1), weight_matrix)
        chunk_kl, gradient, teacher_lse, scaled_lse = _coverage_and_gradient(logits, teacher_logits, inverse)
        d_cov[start:end] = (gradient @ weight_matrix).view(end - start, b, -1)
        mass.append(torch.exp(teacher_logits.gather(-1, top_indices).float() * inverse - teacher_lse[:, None])
                    .view(end - start, b, -1))
        if gate_temperature is not None:
            tails.append(_teacher_tail_mass(teacher_logits, y, gate_temperature).view(end - start, b))
        del logits, teacher_logits, gradient
        if gate is not None:
            anchor_logits = F.linear(anchor_hidden[start:end], weight_matrix)
            anchor_tails.append(_teacher_tail_mass(anchor_logits, anchor_tokens[start:end], gate_temperature))
            del anchor_logits
        kl.append(chunk_kl.view(end - start, b))
        ce_t1.append((lse - label_logit).view(end - start, b))
        ce_t.append((scaled_lse - label_logit * inverse).view(end - start, b))
        unary.append(top_values.float().view(end - start, b, -1))
        candidates.append(top_indices.view(end - start, b, -1))
    return AcceptanceStatistics(torch.cat(kl), torch.cat(ce_t1), torch.cat(ce_t), torch.cat(unary),
                                torch.cat(candidates), torch.cat(mass), d_cov, torch.cat(tails) if tails else None,
                                torch.cat(anchor_tails) if anchor_tails else None)


def acceptance_from_statistics(model, stats, hidden, labels, predecessors, temperature, joint, valid=None,
                               per_anchor=False, robust=None, robust_mix=None, corpus="other", gate=None):
    """objective.acceptance_objective's terms. cov_num carries hidden's gradient from the statistics; the acceptance
    term its own through autograd -- on the candidates' logits (_CandidateLogits) and the selector's hidden projection
    when joint, on the selector alone otherwise. valid: as objective.acceptance_objective's. per_anchor adds
    "survival_anchor" [anchors, b - 1] (zero past a block's valid slots; evaluation's renewal estimate), which is not
    an additive term.

    gate (prefix_gate): the optimized numerators are accept_opt_num and cov_opt_num, both weighted by the same detached
    prefix weights -- the coverage one in its value and in d_cov alike; accept_num and cov_num stay the raw,
    unweighted diagnostics (cov_num then without a gradient)."""
    count, b = labels.shape
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    if gate is None:
        cov_num = _CoverageGradient.apply(hidden, (stats.kl * valid).sum(), stats.d_cov * valid[..., None])
    else:
        if robust is not None or stats.teacher_tail_mass is None or stats.anchor_tail_mass is None:
            raise ValueError("the prefix gate needs the statistics' anchor and label tails, and no robust objective")
        weights = prefix_weights(stats.anchor_tail_mass, stats.teacher_tail_mass, valid, gate)
        scale = torch.where(valid, weights, torch.zeros_like(weights))
        cov_num = (stats.kl * valid).sum()
        cov_opt_num = _CoverageGradient.apply(hidden, (stats.kl * scale).sum(), stats.d_cov * scale[..., None])
    if joint:
        unary = _CandidateLogits.apply(hidden, model.lm_head.weight, stats.candidates, stats.unary)
        selector_hidden = hidden
    else:
        unary, selector_hidden = stats.unary, hidden.detach()
    scores = model.candidate_selector(stats.candidates, unary, selector_hidden, predecessors).float()
    proposal = (scores / temperature).softmax(-1)
    overlap = torch.where(valid, torch.minimum(stats.teacher_mass, proposal).sum(-1), 1.0)
    survival = overlap.cumprod(-1)
    raw_accept = ((1 - survival) * valid).sum() / b
    candidates = stats.candidates
    with torch.no_grad():
        covered = (candidates == labels[..., None]).any(-1) & valid
        chosen = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        accept = (chosen == labels) & valid
        run = accept.long().cumprod(-1)
    extra = {}
    if robust is not None:
        if stats.teacher_tail_mass is None or robust_mix is None:
            raise ValueError("robust acceptance requires teacher tails and a step-wide mix tensor")
        extra = _robust_terms(stats.teacher_mass, proposal, stats.teacher_tail_mass, valid,
                              torch.ones_like(robust_mix), robust, corpus)
        # Preserve the existing eager raw-WTV backward at ramp start. Compiling a
        # second raw numerator can change BF16 rounding even though its algebra agrees.
        extra["accept_opt_num"] = (1 - robust_mix) * raw_accept + robust_mix * extra["accept_opt_num"]
    if gate is not None:
        extra = {"accept_opt_num": ((1 - survival) * scale).sum() / b, "cov_opt_num": cov_opt_num,
                 **gate_diagnostics(stats.anchor_tail_mass, stats.teacher_tail_mass, weights, valid, gate, corpus)}
    return {
        "cov_num": cov_num,
        "accept_num": raw_accept,
        "blocks": torch.tensor(count, device=labels.device),
        "positions": valid.sum(),
        "valid_position": valid.sum(0),
        "overlap_position": (overlap.detach() * valid).sum(0),
        "mass_position": (stats.teacher_mass.sum(-1) * valid).sum(0),
        "survival_position": (survival.detach() * valid).sum(0),
        "ce_t1_num": (stats.ce_t1 * valid).sum(),
        "ce_t_num": (stats.ce_t * valid).sum(),
        "covered": covered.sum(),
        "selector_correct": (accept & covered).sum(),
        "unary_correct": ((candidates[..., 0] == labels) & valid).sum(),
        "accept_prefix": run.sum(0),
        "accept_position": accept.sum(0),
        "covered_position": covered.sum(0),
        "accept_histogram": torch.bincount(run.sum(-1), minlength=b + 1),
    } | extra | ({"survival_anchor": survival.detach() * valid} if per_anchor else {})


def acceptance_terms(model, hidden, labels, predecessors, teacher_gr, temperature, joint, anchors_per_chunk=256,
                     valid=None, per_anchor=False, robust=None, robust_mix=None, corpus="other", gate=None,
                     anchor_tokens=None, anchor_gr=None):
    """objective.acceptance_objective's terms for a whole micro-batch, the head never kept for backward."""
    stats = acceptance_statistics(model, hidden, labels, teacher_gr, temperature, anchors_per_chunk, robust, gate,
                                  anchor_tokens, anchor_gr)
    return acceptance_from_statistics(model, stats, hidden, labels, predecessors, temperature, joint, valid, per_anchor,
                                      robust, robust_mix, corpus, gate)


def opd_from_statistics(model, stats, hidden, drafted, anchor_tokens, accepted, temperature, joint, sampler_q=None,
                        sampler_candidates=None):
    """objective.opd_objective's terms from acceptance_statistics(hidden, labels=drafted, teacher_gr=the verify block's
    GR): the head part is the offline one (the backbone's states, candidates and teacher are path-independent given the
    teacher rows); what changes is the selector's predecessor (the sampled draft) and the reached-slot mask. cov_num
    carries hidden's gradient on the reached slots only. sampler_q / sampler_candidates: objective.importance_terms (its
    is_num with the gradient, the OPD loss "importance")."""
    from objective import importance_terms

    if (sampler_q is None) != (sampler_candidates is None):
        raise ValueError("sampler_q and sampler_candidates go together")
    count, b = drafted.shape
    if ((accepted < 0) | (accepted > b)).any():
        raise ValueError("accepted drafts out of range")
    predecessors = torch.cat((anchor_tokens[:, None], drafted[:, :-1]), 1)
    reached = torch.arange(b, device=drafted.device)[None] <= accepted.clamp(max=b - 1)[:, None]
    cov_num = _CoverageGradient.apply(hidden, (stats.kl * reached).sum(), stats.d_cov * reached[..., None])
    if joint:
        unary = _CandidateLogits.apply(hidden, model.lm_head.weight, stats.candidates, stats.unary)
        selector_hidden = hidden
    else:
        unary, selector_hidden = stats.unary, hidden.detach()
    scores = model.candidate_selector(stats.candidates, unary, selector_hidden, predecessors).float()
    proposal = (scores / temperature).softmax(-1)
    overlap = torch.minimum(stats.teacher_mass, proposal).sum(-1)
    survival = overlap.cumprod(-1)
    with torch.no_grad():
        picked = stats.candidates == drafted[..., None]
        # p(y^) from the teacher's log-sum-exp is not kept; at a candidate it is teacher_mass at that index (outside
        # the candidates q(y^) = 0, so min(p, q) = 0 either way).
        p_drafted = (stats.teacher_mass * picked).sum(-1)
    q_drafted = (proposal * picked).sum(-1)  # with its gradient: importance_terms' weights train through it
    with torch.no_grad():
        rho = torch.where(q_drafted > 0, (p_drafted / q_drafted.clamp_min(1e-30)).clamp(max=1), 0.0)
    importance = {} if sampler_q is None else importance_terms(
        overlap, q_drafted, p_drafted, sampler_q, stats.candidates, proposal, sampler_candidates)
    return importance | {
        "reach_num": ((1 - overlap) * reached).sum(),
        "reach_den": reached.sum(),
        "wtv_num": (1 - survival.sum(-1) / b).sum(),
        "blocks": torch.tensor(count, device=drafted.device),
        "cov_num": cov_num,
        "overlap_position": overlap.detach().sum(0),
        "mass_position": stats.teacher_mass.sum(-1).sum(0),
        "survival_position": survival.detach().sum(0),
        "reached_position": reached.sum(0),
        "drafted_candidate_position": picked.any(-1).sum(0),
        "rho_survival_position": rho.cumprod(-1).sum(0),
        "accepted_observed": accepted.sum(),
        "accepted_histogram": torch.bincount(accepted, minlength=b + 1),
    }


def opd_terms(model, hidden, drafted, anchor_tokens, teacher_gr, accepted, temperature, joint, anchors_per_chunk=256,
              sampler_q=None, sampler_candidates=None):
    """objective.opd_objective's terms for a micro-batch of recorded cycles, the head never kept for backward."""
    stats = acceptance_statistics(model, hidden, drafted, teacher_gr, temperature, anchors_per_chunk)
    return opd_from_statistics(model, stats, hidden, drafted, anchor_tokens, accepted, temperature, joint, sampler_q,
                               sampler_candidates)
