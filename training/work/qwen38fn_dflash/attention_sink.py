"""Per-query-head zero-value softmax sink. No position, key or persistent cache entry.

The reference appends a zero K/V with additive logit s_h. The fast path uses
ordinary attention and its differentiable natural-log LSE: o_sink = o * sigmoid(LSE - s_h).
"""

import torch
from torch.nn import functional as F


def rescale_attention(output, lse, sink_logits):
    # Keep the LSE attached: its gradient contributes to Q/K as well as the sink.
    dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
    scale = torch.sigmoid(lse.to(dtype) - sink_logits.to(dtype)[None, :, None])
    return (output.to(dtype) * scale[..., None]).to(output.dtype)


def reference_attention(q, k, v, visible, sink_logits):
    """SDPA oracle and bounded-query serving path; visible is [batch, query, key]."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    bias = torch.zeros_like(visible, dtype=dtype).masked_fill(~visible, float("-inf"))
    bias = bias[:, None].expand(-1, q.shape[1], -1, -1)
    sink = sink_logits.to(dtype)[None, :, None, None].expand(q.shape[0], -1, q.shape[2], 1)
    bias = torch.cat((bias, sink), -1)
    return F.scaled_dot_product_attention(
        q, F.pad(k, (0, 0, 0, 1)), F.pad(v, (0, 0, 0, 1)), attn_mask=bias, enable_gqa=True,
    )


def keep_sinks_float32(model):
    """Call after bulk dtype conversion, before loading trained FP32 sink values."""
    for layer in model.layers:
        sink = layer.self_attn.sink_logits
        if sink is not None:
            sink.data = sink.data.float()
