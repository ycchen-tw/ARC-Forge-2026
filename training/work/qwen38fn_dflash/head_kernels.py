"""One read of the head's BF16 logits for log-sum-exp, the label's logit and the top-K.

``row_stats_topk(logits, labels, k)`` replaces a compiled FP32 log-sum-exp with a label gather, and
``logits.topk(k, -1)`` with one Triton kernel, one program (one warp) per row streaming the vocabulary in tiles of
[GROUPS, 8] (one 16-byte vector per group):

- log-sum-exp: online max over tiles, FP32 sums of exp2((x - r) * log2(e)) kept per lane (``r`` is the running max,
  or 0 when it is infinite, like ``torch.logsumexp``), lse = r + log(sum).
- label logit: the BF16 value at the label, exactly; an out-of-range label gives NaN (no host sync to raise on it).
- top-K: a running set of K (value, index) pairs. A tile is only looked at when its max beats the set's minimum, then
  its groups are visited largest max first while a group can still beat the set's minimum; a visited group's 8 values
  are read again (cache) and inserted one at a time, each replacing the set's minimum.

Order and ties: the K largest under (value descending, index ascending) -- among equal values the lowest index wins,
both for which elements make the cut at the K-th place and for the output order. -0.0 and +0.0 are equal. Values are
the logits' own BF16 bits, read back by index. torch.topk returns the same values; it leaves the order of equal values
unspecified, so its indices can differ from these only among equal values. NaN logits are not supported: the row's
log-sum-exp is NaN (so the loss shows it), but the top-K skips a NaN among numbers where torch.topk ranks it first.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MAX_K = 128
_GROUP = 8  # one 16-byte vector of BF16


@triton.jit
def _order(bits):
    """BF16 bits (int32, sign-extended) -> an int32 in [-32767, 32767] ordered like the value; -0.0 -> 0."""
    return tl.where(bits < 0, -(bits & 0x7FFF), bits)


@triton.jit
def _value_of_order(order):
    """Inverse of _order, as FP32."""
    bits = tl.where(order < 0, 0x8000 | -order, order)
    return (bits << 16).to(tl.float32, bitcast=True)


@triton.jit
def _beats(order, index, min_order, min_index):
    """(order, index) ranks above (min_order, min_index): larger value, or equal value and lower index."""
    return (order > min_order) | ((order == min_order) & (index < min_index))


@triton.jit
def _set_min(set_order, set_index):
    """The set's lowest-ranked entry: lowest value, among those the highest index."""
    min_order = tl.min(set_order, 0)
    return min_order, tl.max(tl.where(set_order == min_order, set_index, -1), 0)


@triton.jit
def _consume(base, x, start, groups, lanes, acc, m, r, set_order, set_index, min_order, min_index,
             threshold_value, V: tl.constexpr, G: tl.constexpr, LOG2G: tl.constexpr, S: tl.constexpr,
             LOG2S: tl.constexpr, MASKED: tl.constexpr):
    """Fold one [G, S] tile starting at ``start`` into the running log-sum-exp and the running top-K."""
    LOG2E: tl.constexpr = 1.4426950408889634
    xf = x.to(tl.float32)
    group_max = tl.max(xf, 1)
    tile_max = tl.max(group_max, 0)
    if tile_max > m:
        r_new = tl.where(tl.abs(tile_max) == float("inf"), 0.0, tile_max)
        # acc is all zeros while m is -inf; exp2 may overflow there, so do not multiply by it.
        acc = tl.where(m == -float("inf"), 0.0, acc * tl.exp2((r - r_new) * LOG2E))
        m = tile_max
        r = r_new
    # x - r is exact for BF16 x and r (a BF16 value or 0), so the terms near the max are exact to exp2's rounding.
    acc += tl.exp2((xf - r) * LOG2E)
    # Every element here has a higher index than every set entry, so only a strictly larger value can enter.
    if (min_order == -32768) | (tile_max > threshold_value):
        group_order = _order(group_max.to(tl.int32, bitcast=True) >> 16)
        group_key = group_order * G + (G - 1 - groups)
        if MASKED:
            group_key = tl.where(groups * S < V - start, group_key, -2147483648)
        best = tl.max(group_key, 0)
        group = G - 1 - (best & (G - 1))
        # The group's best possible entry: its max at its lowest index. Groups come largest max first, so the
        # first one that cannot beat the set's minimum ends the tile.
        while _beats(best >> LOG2G, start + group * S, min_order, min_index):
            offsets = start + group * S + lanes
            if MASKED:
                v = tl.load(base + offsets, mask=offsets < V, other=0.0)
            else:
                v = tl.load(base + offsets)
            key = _order(v.to(tl.int16, bitcast=True).to(tl.int32)) * S + (S - 1 - lanes)
            if MASKED:
                key = tl.where(offsets < V, key, -2147483648)
            top = tl.max(key, 0)
            index = start + group * S + S - 1 - (top & (S - 1))
            while _beats(top >> LOG2S, index, min_order, min_index):
                replace = (set_order == min_order) & (set_index == min_index)
                set_order = tl.where(replace, top >> LOG2S, set_order)
                set_index = tl.where(replace, index, set_index)
                min_order, min_index = _set_min(set_order, set_index)
                key = tl.where(key == top, -2147483648, key)
                top = tl.max(key, 0)
                index = start + group * S + S - 1 - (top & (S - 1))
            group_key = tl.where(group_key == best, -2147483648, group_key)
            best = tl.max(group_key, 0)
            group = G - 1 - (best & (G - 1))
        threshold_value = tl.where(min_order == -32768, -float("inf"), _value_of_order(min_order))
    return acc, m, r, set_order, set_index, min_order, min_index, threshold_value


@triton.jit
def _write_top(base, TopValues, TopIndices, row, keys, K: tl.constexpr, N: tl.constexpr):
    """The K largest of N keys (order << 32 | ~index; non-entries INT64_MIN): value descending, then index ascending."""
    ordered = tl.sort(keys, descending=True)
    index = 4294967295 - (ordered & 4294967295)
    out = tl.arange(0, N)
    real = out < K
    tl.store(TopValues + row * K + out, tl.load(base + index, mask=real), mask=real)
    tl.store(TopIndices + row * K + out, index, mask=real)


# The row count is only the grid, not an argument; Labels is read one scalar per row, so a slice at any offset must
# not compile a variant for its alignment.
@triton.jit(do_not_specialize=["Labels"])
def _row_stats_topk_kernel(
    X,
    Labels,
    Lse,
    LabelLogit,
    TopValues,
    TopIndices,
    V: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    G: tl.constexpr,
    LOG2G: tl.constexpr,
    S: tl.constexpr,
    LOG2S: tl.constexpr,
    STAGES: tl.constexpr,
):
    TILE: tl.constexpr = G * S
    FULL_TILES: tl.constexpr = V // TILE
    row = tl.program_id(0).to(tl.int64)
    base = X + row * V
    groups = tl.arange(0, G)
    lanes = tl.arange(0, S)
    tile_offsets = groups[:, None] * S + lanes[None, :]
    slots = tl.arange(0, KP)
    # Empty slots: order -32768 (below every value), distinct indices. Padding slots K..KP: above every value.
    set_order = tl.where(slots < K, -32768, 32768 + slots)
    set_index = (1 << 30) + slots
    min_order, min_index = _set_min(set_order, set_index)
    threshold_value = tl.full([], -float("inf"), tl.float32)
    m = tl.full([], -float("inf"), tl.float32)
    r = tl.full([], 0.0, tl.float32)  # the reference the sum is kept relative to: m, or 0 while m is infinite
    acc = tl.zeros([G, S], tl.float32)

    for t in tl.range(FULL_TILES, num_stages=STAGES):
        start = t * TILE
        x = tl.load(base + start + tile_offsets)
        acc, m, r, set_order, set_index, min_order, min_index, threshold_value = _consume(
            base, x, start, groups, lanes, acc, m, r, set_order, set_index, min_order, min_index,
            threshold_value, V, G, LOG2G, S, LOG2S, False)
    if V % TILE != 0:
        start = FULL_TILES * TILE
        x = tl.load(base + start + tile_offsets, mask=tile_offsets < V - start, other=-float("inf"))
        acc, m, r, set_order, set_index, min_order, min_index, threshold_value = _consume(
            base, x, start, groups, lanes, acc, m, r, set_order, set_index, min_order, min_index,
            threshold_value, V, G, LOG2G, S, LOG2S, True)

    tl.store(Lse + row, tl.where(m == float("inf"), float("inf"), r + libdevice.log(tl.sum(tl.sum(acc, 1), 0))))
    label = tl.load(Labels + row)
    in_range = (label >= 0) & (label < V)
    label_logit = tl.load(base + label, mask=in_range, other=0.0).to(tl.float32)
    tl.store(LabelLogit + row, tl.where(in_range, label_logit, float("nan")))
    keys = tl.where(slots < K, set_order.to(tl.int64) * 4294967296 + (4294967295 - set_index.to(tl.int64)),
                    -9223372036854775808)
    _write_top(base, TopValues, TopIndices, row, keys, K, KP)


def row_stats_topk(logits: torch.Tensor, labels: torch.Tensor, k: int = 16, tile: int = 1024, stages: int = 3):
    """(lse fp32 [rows], label logit fp32 [rows], top values bf16 [rows, k] descending, top indices int64 [rows, k]).

    ``logits``: contiguous BF16 [rows, V] on CUDA. ``labels``: contiguous int64 [rows]. One kernel, one read of the
    logits. Equal values are ordered (and cut at the k-th place) lowest index first; see the module docstring.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be tensors")
    if logits.dtype != torch.bfloat16:
        raise TypeError(f"logits must be bfloat16, got {logits.dtype}")
    if labels.dtype != torch.int64:
        raise TypeError(f"labels must be int64, got {labels.dtype}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, V], got shape {tuple(logits.shape)}")
    rows, vocab = logits.shape
    if labels.shape != (rows,):
        raise ValueError(f"labels must be [{rows}], got shape {tuple(labels.shape)}")
    if not logits.is_cuda or labels.device != logits.device:
        raise ValueError(f"logits and labels must be on the same CUDA device, got {logits.device} and {labels.device}")
    if not logits.is_contiguous() or not labels.is_contiguous():
        raise ValueError("logits and labels must be contiguous")
    if not 1 <= k <= min(vocab, _MAX_K):
        raise ValueError(f"k must be in [1, min(V, {_MAX_K})], got k={k}, V={vocab}")
    if vocab >= 2**30:
        raise ValueError(f"V={vocab} does not fit the kernel's indices")
    if tile & (tile - 1) or not 256 <= tile <= 2**16:
        raise ValueError(f"tile must be a power of two in [256, 65536], got {tile}")
    device = logits.device
    lse = torch.empty(rows, device=device, dtype=torch.float32)
    label_logit = torch.empty(rows, device=device, dtype=torch.float32)
    top_values = torch.empty(rows, k, device=device, dtype=torch.bfloat16)
    top_indices = torch.empty(rows, k, device=device, dtype=torch.int64)
    if rows:
        groups = tile // _GROUP
        with torch.cuda.device(device):
            _row_stats_topk_kernel[(rows,)](
                logits, labels, lse, label_logit, top_values, top_indices,
                V=vocab, K=k, KP=triton.next_power_of_2(k), G=groups, LOG2G=groups.bit_length() - 1, S=_GROUP,
                LOG2S=_GROUP.bit_length() - 1, STAGES=stages, num_warps=1,
            )
    return lse, label_logit, top_values, top_indices
