"""Qwen3.8-FN PLE on a packed sequence, without padding segments to the longest one.

The previous packed path (packed_qwen4.ple_forward, ``ple_mode='padded'``) scattered every segment into a
[segments, max_length] batch so HF's per-row code could run unchanged. A 3-segment 115k bin then runs PLE on
3 x max_length rows and holds several [rows, 10240] intermediates: up to ~30 GB of the 173 GiB peak, and
the row count is what pushed the FP8 PLE lookup past 2**31 (fp8_base._lookup, fixed separately).

Here every token stays in place. HF semantics per row: two EOS tokens are prepended as n-gram context and an
EOS inside the row restarts the n-gram history (``_shift_right_ignore_eos``); the dilated causal conv zero-pads
on the left. So a segment start acts exactly like an EOS boundary, and the conv runs per segment. The n-gram
ids are integer-identical to the padded path; the conv is HF's own ``_short_conv`` on each segment slice.
"""

import itertools
import math

import torch


def packed_ngram_ids(embedding, ids, segment_start):
    """[T, heads] n-gram ids of HF's Qwen4ExpTextNGramEmbedding for a packed token sequence."""
    total = ids.shape[0]
    positions = torch.arange(total, device=ids.device)
    eos_positions = torch.where(ids == embedding.eos_token_id, positions, -1)
    inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([inclusive.new_full((1,), -1), inclusive[:-1]])
    start = torch.maximum(segment_start, previous_eos + 1)
    position_in_segment = positions - start
    eos = ids.new_full((), embedding.eos_token_id)
    shifted = []
    for shift in range(embedding.ngram_size):
        source = (positions - shift).clamp_min(0)
        shifted.append(torch.where(position_in_segment >= shift, ids[source], eos))
    blocks = []
    for ngram in range(2, embedding.ngram_size + 1):
        lo = (ngram - 2) * embedding.heads_per_ngram
        hi = lo + embedding.heads_per_ngram
        mixed = shifted[0] * embedding.layer_multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * embedding.layer_multipliers[position])
        sizes = embedding.ngram_heads_vocab_sizes[lo:hi]
        offsets = embedding.ngram_heads_offsets[lo:hi]
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, -1)) + offsets.view(1, -1))
    return torch.cat(blocks, dim=-1)


def packed_ple_forward(ple, hidden_states, input_ids, lengths):
    """HF Qwen4ExpTextPLELayer.forward for B=1 packed input with the given segment lengths (no cache, no mask)."""
    if hidden_states.shape[0] != 1 or input_ids.shape != hidden_states.shape[:2]:
        raise ValueError('packed PLE expects B=1 hidden states and matching ids')
    total = hidden_states.shape[1]
    if sum(lengths) != total:
        raise ValueError(f'segment lengths sum to {sum(lengths)}, the pack has {total} tokens')
    ids = input_ids[0].long()
    starts = torch.repeat_interleave(
        torch.tensor([0] + list(itertools.accumulate(lengths))[:-1], device=ids.device),
        torch.tensor(lengths, device=ids.device))
    ngram_ids = packed_ngram_ids(ple.ple_embedding, ids, starts)
    embedding = ple.ple_embedding.ngram_embedding
    embeddings = embedding(ngram_ids.unsqueeze(0).to(embedding.weight.device)).flatten(-2)
    key_normed = ple.norm_key(ple.key_proj(embeddings)).unflatten(-1, (ple.hc_count, ple.hidden_size))
    value = ple.value_proj(embeddings)
    query_normed = ple.norm_query(hidden_states).unflatten(-1, (ple.hc_count, ple.hidden_size))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = ple.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    convolved, offset = [], 0
    for length in lengths:
        convolved.append(ple._short_conv(gated_value_normed[:, offset:offset + length], None))
        offset += length
    return gated_value + torch.cat(convolved, dim=1)
