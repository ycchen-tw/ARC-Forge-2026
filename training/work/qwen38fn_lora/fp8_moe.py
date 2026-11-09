"""Frozen block-FP8 MoE using unmodified SGLang GEMM/quantization kernels.

Original E4M3 weights and 128x128 scales are never dequantized as full matrices.
Dynamic 1x128 E4M3 activation/gradient quantization; no persistent W transpose.
Routing and SwiGLU derivatives remain high precision, output/residuals BF16.
Local code supplies frozen-weight autograd and routing/pointwise glue, not GEMM. The one local kernel,
`_quant_groups` (`--fast-kernels act_quant`), is the pinned quantizer's math over 32 groups per program, bitwise equal
to it (check_trainer_kernels.py).
"""

import torch
import triton
import triton.language as tl
from torch import nn

COUNTERS = {"backend": "unmodified_sglang_73a2552", "forward": 0, "backward": 0, "fp8_gemm": 0, "bf16_dx_gemm": 0}
# "fp8": backward GEMMs quantize dY/dPre to 1x128 E4M3 like the forward (default, fastest).
# "bf16": forward unchanged (deployment numerics), backward GEMMs on per-expert dequantized BF16 weights,
# so only the forward's activation quantization differs from a BF16 base. For gradient-fidelity studies.
DX_MODE = {"mode": "fp8"}


def set_dx_mode(mode):
    if mode not in ("fp8", "bf16"):
        raise ValueError(f"FP8 MoE dX mode {mode!r}")
    DX_MODE["mode"] = mode


def dequant_expert(weight, scale, e):
    n, k = weight.shape[1:]
    full = weight[e].float().view(n // 128, 128, k // 128, 128) * scale[e].float()[:, None, :, None]
    return full.reshape(n, k).to(torch.bfloat16)


def bf16_grouped(a, weight, scale, routing, gather=False):
    """Backward-direction grouped GEMM (out = a @ W_e) with dequantized BF16 expert weights."""
    _, starts, indices = routing[:3]
    rows = a.index_select(0, indices.long()) if gather else a
    out = torch.empty((indices.numel(), weight.shape[2]), device=a.device, dtype=torch.bfloat16)
    bounds = starts.tolist()
    for e in range(weight.shape[0]):
        lo, hi = bounds[e], bounds[e + 1]
        if hi > lo:
            out[lo:hi] = rows[lo:hi] @ dequant_expert(weight, scale, e)
    COUNTERS["bf16_dx_gemm"] += 1
    return out


# "sglang": the pinned _per_token_group_quant_8bit, one 128-group per program (the reference).
# "grouped": _quant_groups, the same math over QUANT_GROUPS groups per program (8-9x faster at trainer sizes, where one
# group per program leaves the GPU mostly scheduling programs).
QUANT_KERNEL = {"kernel": "sglang"}
QUANT_GROUPS = 32


def set_quant_kernel(kernel):
    if kernel not in ("sglang", "grouped"):
        raise ValueError(f"activation quantizer {kernel!r}")
    QUANT_KERNEL["kernel"] = kernel


@triton.jit
def _quant_groups(y_ptr, q_ptr, s_ptr, n_groups, eps, bit8_min, bit8_max, G: tl.constexpr, BLOCK: tl.constexpr):
    # The pinned SGLang _per_token_group_quant_8bit (y_stride = N = BLOCK = 128) for G consecutive groups.
    rows = tl.program_id(0).to(tl.int64) * G + tl.arange(0, G)
    cols = tl.arange(0, BLOCK)
    live = rows < n_groups
    offs = rows[:, None] * BLOCK + cols[None, :]
    y = tl.load(y_ptr + offs, mask=live[:, None], other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv[:, None], bit8_min, bit8_max).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + offs, y_q, mask=live[:, None])
    tl.store(s_ptr + rows, y_s, mask=live)


def quantize(a, kernel=None):
    from upstream_fp8 import kernels

    if a.ndim != 2 or a.shape[1] % 128 or a.dtype != torch.bfloat16 or not a.is_contiguous():
        raise ValueError("quantization requires contiguous BF16 [rows, K%128=0]")
    q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    s = torch.empty((a.shape[0], a.shape[1] // 128), dtype=torch.float32, device=a.device)
    kernel = QUANT_KERNEL["kernel"] if kernel is None else kernel
    if kernel == "grouped":
        groups = a.numel() // 128
        _quant_groups[(triton.cdiv(groups, QUANT_GROUPS),)](
            a, q, s, groups, 448e-30, -448.0, 448.0, G=QUANT_GROUPS, BLOCK=128, num_warps=4
        )
    elif kernel == "sglang":
        _, quant = kernels()
        quant[(a.numel() // 128,)](a, q, s, 128, 128, 448e-30, -448.0, 448.0, BLOCK=128, num_warps=4)
    else:
        raise ValueError(f"activation quantizer {kernel!r}")
    return q, s


TILE_ROWS = 64  # liger's BLOCK_M_TOKEN = the grouped GEMMs' BLOCK_SIZE_M: one M-tile of an expert's rows


def routing_metadata(topk_indices, num_experts):
    """liger ``compute_routing_metadata`` without its host sync, plus the real M-tile count on the device.

    liger reads the tile count back (``.item()``) to size the two tile arrays: one D2H copy per MoE call that also
    waits behind whatever else is on the copy engine -- in backward the checkpoint offload's 2.46 GB H2D prefetch,
    which stalled the compute stream ~30-45 ms per layer (QAD 10-17 profile: 2.1 s of a 21 s step). Here the tile
    arrays take the upper bound cdiv(T*K, 64) + E (an expert ends at most one partial tile), the tiles past the real
    count stay zero (padded_routing masks their rows), and the count stays on the device. Same three liger kernels,
    same order: the real tiles are liger's bit for bit. Returns liger's seven tensors and the [1] int32 tile count.
    """
    from liger_kernel.ops import fused_moe as fm

    if fm.BLOCK_M_TOKEN != TILE_ROWS:
        raise RuntimeError(f"liger BLOCK_M_TOKEN {fm.BLOCK_M_TOKEN} != {TILE_ROWS} (padded_routing, the GEMM tiles)")
    tokens, k = topk_indices.shape
    tk = tokens * k
    device = topk_indices.device
    e_pow2, k_pow2 = triton.next_power_of_2(num_experts), triton.next_power_of_2(k)
    per_block = max(1, 1024 // k_pow2)
    n_tiles = triton.cdiv(tokens, per_block)
    tile_counts = torch.empty(num_experts, n_tiles, dtype=torch.int32, device=device)
    fm._moe_router_histogram_kernel[(n_tiles,)](topk_indices, tile_counts, tokens, E=num_experts, n_tiles=n_tiles,
                                                TOKENS_PER_TILE=per_block, K_POW2=k_pow2, K=k, E_POW2=e_pow2)
    expert_count = tile_counts.sum(dim=1, dtype=torch.int32)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    tile_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    fm._moe_router_prefix_sum_kernel[(num_experts + 2,)](expert_count, expert_start, tile_offset, E=num_experts,
                                                         partial_sum_ptr=tile_counts, n_tiles=n_tiles, TK=tk,
                                                         BLOCK_M=128, BLOCK_N=e_pow2, BLOCK_M_TOKEN=TILE_ROWS)
    bound = triton.cdiv(tk, TILE_ROWS) + num_experts
    tile_start = torch.zeros(bound, dtype=torch.int32, device=device)
    tile_expert = torch.zeros(bound, dtype=torch.int32, device=device)
    scatter = torch.empty(tk, dtype=torch.int32, device=device)
    reverse = torch.empty(tk, dtype=torch.int32, device=device)
    gather = torch.empty(tk, dtype=torch.int32, device=device)
    if tk > 0:
        fm._moe_router_scatter_kernel[(n_tiles,)](scatter, reverse, gather, tile_start, tile_expert, topk_indices,
                                                  tokens, tile_counts, n_tiles, expert_start[:num_experts],
                                                  tile_offset[:num_experts], K_POW2=k_pow2, K=k,
                                                  TOKENS_PER_BLOCK=per_block, BLOCK_M_TOKEN=TILE_ROWS)
    return expert_count, expert_start, gather, scatter, reverse, tile_start, tile_expert, tile_offset[num_experts:]


def padded_routing(routing):
    """routing_metadata's eight tensors -> the nine the grouped GEMMs read: liger's seven, the token row of every
    tile slot (T*K = skip for slots past an expert's end and for every slot of the tiles past the real count), and the
    real slot count on the device (``num_tokens_post_padded``: the GEMM programs of the tiles past it return at once).
    Takes its own output back unchanged."""
    if len(routing) == 9:
        return routing
    if len(routing) != 8:
        raise ValueError(f"padded_routing takes routing_metadata's 8 tensors, got {len(routing)}")
    *routing, tiles = routing
    _, ends, indices, _, _, starts, experts = routing
    rows = starts[:, None] + torch.arange(TILE_ROWS, device=starts.device)[None, :]
    real = (torch.arange(starts.numel(), device=starts.device) < tiles)[:, None]
    padded = torch.where(real & (rows < ends[experts.long() + 1, None]), rows, indices.numel()).flatten().contiguous()
    count = (tiles * TILE_ROWS).to(torch.int32)  # stays on the device: no host sync
    return (*routing, padded, count)


def grouped(a, weight, scale, routing, transpose=True, gather=False):
    from upstream_fp8 import kernels

    if weight.dtype != torch.float8_e4m3fn or weight.requires_grad:
        raise ValueError("frozen E4M3 weight required")
    n, k = weight.shape[1:] if transpose else weight.shape[1:][::-1]
    if a.shape[1] != k or n % 128 or k % 128:
        raise ValueError("FP8 grouped shape mismatch")
    routing = padded_routing(routing)
    _, _, indices, _, _, starts, experts, padded, count = routing
    q, s = quantize(a)
    if gather:
        # Route quantized inputs only: no repeated BF16 activation buffer.
        q = q.index_select(0, indices.long())
        s = s.index_select(0, indices.long())
    out = torch.empty((indices.numel(), n), device=a.device, dtype=torch.bfloat16)
    gemm, _ = kernels()
    bk, bn = (weight.stride(2), weight.stride(1)) if transpose else (weight.stride(1), weight.stride(2))
    sk, sn = (scale.stride(2), scale.stride(1)) if transpose else (scale.stride(1), scale.stride(2))
    compiled = gemm[(starts.numel() * triton.cdiv(n, 128),)](
        a_ptr=q,
        a_desc=None,
        b_ptr=weight,
        b_desc=None,
        bias_ptr=None,
        c_ptr=out,
        a_scale_ptr=s,
        b_scale_ptr=scale,
        topk_weights_ptr=None,
        sorted_token_ids_ptr=padded,
        expert_ids_ptr=experts,
        num_tokens_post_padded_ptr=count,
        add_mask_ptr=None,
        N=n,
        K=k,
        EM=padded.numel(),
        num_valid_tokens=indices.numel(),
        stride_am=q.stride(0),
        stride_ak=1,
        stride_be=weight.stride(0),
        stride_bk=bk,
        stride_bn=bn,
        stride_bias_e=0,
        stride_bias_n=0,
        stride_cm=out.stride(0),
        stride_cn=1,
        stride_asm=s.stride(0),
        stride_ask=1,
        stride_bse=scale.stride(0),
        stride_bsk=sk,
        stride_bsn=sn,
        group_n=128,
        group_k=128,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=8,
        MUL_ROUTED_WEIGHT=False,
        top_k=1,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=True,
        c_sorted=False,
        filter_expert=False,
        swap_ab=False,
        FUSE_ADD_TO_OUTPUT=False,
        MASK_OUTPUT=False,
        FUSE_SUM_ALL_REDUCE=False,
        LORA_PRESERVE_BASE=False,
        ROUTER_TOPK=1,
        FUSE_SWIGLU=False,
        num_warps=4,
        num_stages=3,
    )
    COUNTERS["fp8_gemm"] += 1
    return out, compiled


@triton.jit
def _swiglu(Pre, Post, Width: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    gate = tl.load(Pre + row * 2 * Width + i, i < Width, other=0).to(tl.float32)
    up = tl.load(Pre + row * 2 * Width + Width + i, i < Width, other=0).to(tl.float32)
    tl.store(Post + row * Width + i, gate * tl.sigmoid(gate) * up, i < Width)


@triton.jit
def _swiglu_backward(Pre, DPost, Weights, Scatter, DPre, Width: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    gate = tl.load(Pre + row * 2 * Width + i, i < Width, other=0).to(tl.float32)
    up = tl.load(Pre + row * 2 * Width + Width + i, i < Width, other=0).to(tl.float32)
    d = tl.load(DPost + row * Width + i, i < Width, other=0).to(tl.float32)
    slot = tl.load(Scatter + row)
    routing = tl.load(Weights + slot).to(tl.float32)
    sig = tl.sigmoid(gate)
    tl.store(DPre + row * 2 * Width + i, d * routing * up * sig * (1 + gate * (1 - sig)), i < Width)
    tl.store(DPre + row * 2 * Width + Width + i, d * routing * gate * sig, i < Width)


@triton.jit
def _routing_backward(DO, Y, Indices, Scatter, DS, H: tl.constexpr, B: tl.constexpr):
    # Y is [tokens * top_k, H]; 128k * 10 * 2560 exceeds int32 indexing.
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, B)
    token = tl.load(Indices + row).to(tl.int64)
    slot = tl.load(Scatter + row)
    d = tl.load(DO + token * H + h, h < H, other=0).to(tl.float32)
    y = tl.load(Y + row * H + h, h < H, other=0).to(tl.float32)
    tl.store(DS + slot, tl.sum(d * y))


class FrozenFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, s1, s2, indices, scores):
        from liger_kernel.ops.fused_moe import _token_aggregation

        if w1.requires_grad or w2.requires_grad:
            raise RuntimeError("FP8 base must stay frozen")
        x = x.contiguous()
        routing = padded_routing(routing_metadata(indices.contiguous().int(), w1.shape[0]))
        pre, _ = grouped(x, w1, s1, routing, gather=True)
        post = torch.empty((pre.shape[0], pre.shape[1] // 2), dtype=x.dtype, device=x.device)
        _swiglu[(post.shape[0],)](pre, post, post.shape[1], triton.next_power_of_2(post.shape[1]))
        y, _ = grouped(post, w2, s2, routing)
        out = _token_aggregation(y, scores.flatten().contiguous(), routing[4], x.shape[0], indices.shape[1], x.shape[1])
        ctx.save_for_backward(w1, w2, s1, s2, pre, y, scores, *routing)
        COUNTERS["forward"] += 1
        return out

    @staticmethod
    def backward(ctx, dy):
        from liger_kernel.ops.fused_moe_kernels import _token_gather_weighted_sum_kernel

        w1, w2, s1, s2, pre, y, scores, *routing = ctx.saved_tensors
        dy = dy.contiguous()
        bf16 = DX_MODE["mode"] == "bf16"
        if bf16:
            dpost = bf16_grouped(dy, w2, s2, routing, gather=True)
        else:
            dpost, _ = grouped(dy, w2, s2, routing, transpose=False, gather=True)
        dpre = torch.empty_like(pre)
        i = pre.shape[1] // 2
        _swiglu_backward[(pre.shape[0],)](pre, dpost, scores, routing[3], dpre, i, triton.next_power_of_2(i))
        ds = torch.empty_like(scores)
        _routing_backward[(pre.shape[0],)](
            dy, y, routing[2], routing[3], ds, dy.shape[1], triton.next_power_of_2(dy.shape[1])
        )
        if bf16:
            expanded = bf16_grouped(dpre, w1, s1, routing)
        else:
            expanded, _ = grouped(dpre, w1, s1, routing, transpose=False)
        dx = torch.empty_like(dy)
        _token_gather_weighted_sum_kernel[(dy.shape[0],)](
            expanded,
            ds,
            routing[4],
            dx,
            H_dim=dy.shape[1],
            K_dim=scores.shape[1],
            stride_Y_TK=expanded.stride(0),
            stride_Y_H=1,
            stride_out_T=dx.stride(0),
            stride_out_H=1,
            w_is_None=True,
        )
        COUNTERS["backward"] += 1
        return dx, None, None, None, None, None, ds


class FrozenFP8Experts(nn.Module):
    def __init__(self, experts, hidden, intermediate, device="cpu"):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = experts, hidden, intermediate
        self.gate_up_proj = nn.Parameter(
            torch.empty(experts, 2 * intermediate, hidden, device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(experts, hidden, intermediate, device=device, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.register_buffer(
            "gate_up_scale",
            torch.empty(experts, 2 * intermediate // 128, hidden // 128, device=device, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "down_scale", torch.empty(experts, hidden // 128, intermediate // 128, device=device, dtype=torch.bfloat16)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        shape = hidden_states.shape
        out = FrozenFP8Function.apply(
            hidden_states.reshape(-1, self.hidden_dim),
            self.gate_up_proj,
            self.down_proj,
            self.gate_up_scale,
            self.down_scale,
            top_k_index,
            top_k_weights,
        )
        return out.reshape(shape)
