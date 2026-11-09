"""Frozen INT3 routed experts (gEAN320 deployment model, W3A8) for the RL trainer.

The server runs Humming uint3 g128 weights with dynamic per-row E4M3 activations (`input_scale_group_size=0`):
x_q = e4m3(x / s) with s = max(absmax(row), 1e-30) / 448, e4m3 x e4m3 GEMM with fp32 accumulation, the K-group
(128) weight scale applied per group and the row scale at the end, bf16 output; silu_and_mul in fp32 stored bf16;
the second GEMM's input is quantized per row again; top-k weighted sum. The experts are frozen, so the trainer only
has to reproduce that forward and supply dX / dRouting for the LoRA layers around them.

Storage: the Humming code `c` in 0..7 means (c - 4) * scale, and (c - 4) in -4..3 is exact in E4M3, so the experts
are kept as E4M3 `(code - 4)` (1 B/weight, lossless) next to the signed BF16 `[E, N, K/128]` scale. The forward is
then SGLang's own `fused_moe_kernel` in w8a8 mode (the kernel `fp8_moe.py` already uses) with `group_n=1`: a weight
scale per output row and K-group, one activation scale per row (`stride_ask=0`).

W3A16 (`--int3-act bf16`, set by `configure`; the rollout's Humming A16, `ARC_INT3_ACT=bf16`): Humming multiplies the
group scale onto the weight in BF16 (`__hmul2`), so the effective weight is `bf16((code - 4) * scale)` -- exactly the
`dequant_bank` the backward already multiplies with -- and the BF16 activation goes in unquantized, fp32 accumulation,
BF16 out; between the GEMMs sgl_kernel `silu_and_mul` (fp32 math, BF16 out, no requantization). The trainer runs it as
`dequant_bank` + quack's SM100 varlen-M grouped GEMM (`grouped_a16`: the w13 GEMM gathers its token rows itself, so
no [T*K, H] BF16 copy is formed), and the w2 GEMM in row chunks with the SwiGLU of each chunk only, so at every stage
the forward holds no more than the W3A8 one besides `pre` / `y` (T = 180k tokens, top-10: w13 bank 2.1 GB vs the
E4M3 input and its routed copy 5.1 GB; w2 bank 1.05 GB + a 0.34 GB SwiGLU chunk vs 1.6 GB of E4M3 rows). The backward
is the same for both: the activation quantizer is straight-through, dX on the dequantized BF16 bank with the forward's
own `pre` / `y`.
"""

import torch
import triton
import triton.language as tl
from fp8_moe import _routing_backward, _swiglu_backward, padded_routing, routing_metadata
from torch import nn
from triton.language.extra.cuda import libdevice

COUNTERS = {"forward": 0, "backward": 0, "w3a8_gemm": 0, "bf16_dx_gemm": 0, "a8_forward": 0, "a16_forward": 0,
            "a16_gemm": 0}
GROUP = 128
CODE_ZERO = 4  # Humming uint3 without zero point: dequantized value = (code - 4) * scale
# The routed experts' activation format (configure): "fp8" = W3A8 (default), "bf16" = W3A16. ``layers`` = the number
# of FrozenInt3Experts of this process (0 = not configured: assert_step_calls checks nothing).
ACT_MODES = ("fp8", "bf16")
ACT = {"mode": "fp8", "layers": 0}
_SEEN = {}
# Rows per w2 chunk of the A16 forward: one BF16 SwiGLU chunk (1.3 KB/row) is alive at a time (module docstring).
A16_W2_ROWS = 1 << 18
# quack SM100 tile for both A16 GEMMs: fastest of the 256x{128,160,256} / 128x256 tiles at 1.8M rows (L21, v3.1 p001
# routing: w13 with the gather 8.4 ms vs torch gather + grouped_mm 14.0 ms; w2 4.4 vs 5.2 ms), bitwise equal to
# torch.nn.functional.grouped_mm on the same operands.
A16_TILE = {"tile_m": 256, "tile_n": 256, "cluster_m": 2, "cluster_n": 1}


def configure(mode, layers):
    """Set the activation format of this process's ``layers`` FrozenInt3Experts (the trainers' --int3-act)."""
    if mode not in ACT_MODES:
        raise ValueError(f"INT3 activation format {mode!r}, expected one of {ACT_MODES}")
    if layers <= 0:
        raise ValueError(f"INT3 experts in {layers} layers")
    ACT.update(mode=mode, layers=layers)
    _SEEN.update(COUNTERS)


def assert_step_calls(update, frozen_layers=0):
    """After one forward (and backward): every INT3 expert layer ran the configured format once per forward -- twice
    with the checkpoint recompute when training, except the ``frozen_layers`` no gradient reaches -- and none ran the
    other one. Nothing to check before configure."""
    if not ACT["layers"]:
        return
    ran = {key: COUNTERS[key] - _SEEN[key] for key in ("a8_forward", "a16_forward")}
    _SEEN.update(COUNTERS)
    if not 0 <= frozen_layers <= ACT["layers"]:
        raise ValueError(f"{frozen_layers} frozen of {ACT['layers']} INT3 expert layers")
    expected = (ACT["layers"] - frozen_layers) * 2 + frozen_layers if update else ACT["layers"]
    want = {"a8_forward": expected if ACT["mode"] == "fp8" else 0, "a16_forward": expected if ACT["mode"] == "bf16" else 0}
    if ran != want:
        raise RuntimeError(f"INT3 experts ran {ran} this step; --int3-act {ACT['mode']} expects {want}")


def unpack_uint3(qweight, k):
    """Humming little-endian 3-bit stream along the last dim -> uint8 codes `[..., k]`.

    `qweight` is int32 `[..., 3k/32]`: every 32 codes occupy 3 words, code i sits at bits 3i..3i+2 of the stream
    (LSB first); the codes at offsets 10 and 21 of each group of 32 straddle a word boundary.
    """
    if qweight.dtype != torch.int32 or k % 32 or qweight.shape[-1] * 32 != 3 * k:
        raise ValueError(
            f"uint3 stream needs int32 [..., 3k/32] with k % 32 == 0, got {tuple(qweight.shape)} for k={k}"
        )
    lead = qweight.shape[:-1]
    words = qweight.reshape(-1, k // 32, 3)
    out = torch.empty((words.shape[0], k // 32, 32), dtype=torch.uint8, device=qweight.device)
    j = torch.arange(32, device=qweight.device)
    word, shift = (3 * j) // 32, ((3 * j) % 32).long()
    # int64 chunks keep the 64-bit window (word | next word << 32) simple; the temporaries stay ~2 GB
    rows = max(1, (1 << 26) // k)
    for lo in range(0, words.shape[0], rows):
        w = words[lo : lo + rows].long() & 0xFFFFFFFF
        w = torch.cat([w, torch.zeros_like(w[..., :1])], dim=-1)
        out[lo : lo + rows] = (((w[..., word] | (w[..., word + 1] << 32)) >> shift) & 7).to(torch.uint8)
    return out.reshape(*lead, k)


def codes_to_e4m3(codes):
    """(code - 4) as E4M3; the integers -4..3 are exactly representable."""
    if codes.dtype != torch.uint8 or int(codes.max()) > 7:
        raise ValueError("uint3 codes must be uint8 in 0..7")
    return (codes.to(torch.float16) - CODE_ZERO).to(torch.float8_e4m3fn)


@triton.jit
def _quant_rows(X, Q, S, K: tl.constexpr, B: tl.constexpr):
    # Same formula and casts as humming.ops.quant_input(group_size=row): scale = max(absmax, 1e-30) / 448,
    # inv = 1 / scale in fp32, x_q = (x * inv) -> float8e4nv.
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    x = tl.load(X + row * K + i, i < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-30) / 448
    inv = 1 / scale
    tl.store(Q + row * K + i, (x * inv).to(tl.float8e4nv), i < K)
    tl.store(S + row, scale)


@triton.jit
def _swiglu_quant(Pre, Q, S, Width: tl.constexpr, B: tl.constexpr):
    # silu_and_mul exactly as the server's sgl_kernel (fast-math exp / divide, as fused_moe_kernel's FUSE_SWIGLU
    # asm documents), stored BF16 first because the server quantizes the BF16 activation buffer, then _quant_rows.
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    gate = tl.load(Pre + row * 2 * Width + i, i < Width, other=0).to(tl.float32)
    up = tl.load(Pre + row * 2 * Width + Width + i, i < Width, other=0).to(tl.float32)
    exp_neg = tl.inline_asm_elementwise(
        "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
        "=f,f",
        [gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    silu = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [gate, 1.0 + exp_neg],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    act = (silu * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(act), axis=0), 1e-30) / 448
    inv = 1 / scale
    tl.store(Q + row * Width + i, (act * inv).to(tl.float8e4nv), i < Width)
    tl.store(S + row, scale)


@triton.jit
def _swiglu_bf16(Pre, Out, Width: tl.constexpr, B: tl.constexpr):
    # sgl_kernel silu_and_mul as the server runs it on B200: x / (1 + expf(-x)) * y in fp32, BF16 out, from its SM100
    # build ("precise math": CUDA expf = libdevice __nv_expf, IEEE division; the fast-math exp / divide of
    # _swiglu_quant is the SM90 build's) -- what the A16 Humming runner feeds its w2 GEMM.
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    gate = tl.load(Pre + row * 2 * Width + i, i < Width, other=0).to(tl.float32)
    up = tl.load(Pre + row * 2 * Width + Width + i, i < Width, other=0).to(tl.float32)
    silu = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(Out + row * Width + i, (silu * up).to(tl.bfloat16), i < Width)


def quantize_rows(a):
    """Dynamic per-row E4M3 (Humming W3A8 activation): `(q [rows, K] e4m3, scale [rows] fp32)`."""
    if a.ndim != 2 or a.dtype != torch.bfloat16 or not a.is_contiguous():
        raise ValueError("per-row quantization requires contiguous BF16 [rows, K]")
    q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    s = torch.empty(a.shape[0], dtype=torch.float32, device=a.device)
    _quant_rows[(a.shape[0],)](a, q, s, a.shape[1], triton.next_power_of_2(a.shape[1]), num_warps=8)
    return q, s


def swiglu_quantize(pre):
    """BF16 `[rows, 2I]` gate|up -> (E4M3 `[rows, I]`, scale `[rows]`) of the BF16 silu(gate) * up."""
    if pre.ndim != 2 or pre.dtype != torch.bfloat16 or not pre.is_contiguous():
        raise ValueError("swiglu_quantize requires contiguous BF16 [rows, 2I]")
    width = pre.shape[1] // 2
    q = torch.empty((pre.shape[0], width), dtype=torch.float8_e4m3fn, device=pre.device)
    s = torch.empty(pre.shape[0], dtype=torch.float32, device=pre.device)
    _swiglu_quant[(pre.shape[0],)](pre, q, s, width, triton.next_power_of_2(width), num_warps=8)
    return q, s


def swiglu_bf16(pre):
    """BF16 `[rows, 2I]` gate|up -> BF16 `[rows, I]` silu(gate) * up (sgl_kernel silu_and_mul, the A16 activation)."""
    if pre.ndim != 2 or pre.dtype != torch.bfloat16 or not pre.is_contiguous():
        raise ValueError("swiglu_bf16 requires contiguous BF16 [rows, 2I]")
    width = pre.shape[1] // 2
    out = torch.empty((pre.shape[0], width), dtype=torch.bfloat16, device=pre.device)
    _swiglu_bf16[(pre.shape[0],)](pre, out, width, triton.next_power_of_2(width), num_warps=8)
    return out


def grouped_w3a8(q, a_scale, weight, scale, routing):
    """Sorted-row grouped GEMM `out[r] = (q[r] * a_scale[r]) @ (weight[e(r)] * scale[e(r)]).T`, BF16 out.

    q: E4M3 `[TK, K]` already in routing order, a_scale: fp32 `[TK]`, weight: E4M3 `[E, N, K]` holding (code - 4),
    scale: signed BF16 `[E, N, K/128]`. Unmodified SGLang fused_moe_kernel, w8a8 with group_n=1 / group_k=128.
    """
    from upstream_fp8 import kernels

    e, n, k = weight.shape
    if weight.dtype != torch.float8_e4m3fn or weight.requires_grad or scale.dtype != torch.bfloat16:
        raise ValueError("frozen E4M3 weight with BF16 scale required")
    if scale.shape != (e, n, k // GROUP) or k % GROUP or q.shape[1] != k or a_scale.shape != (q.shape[0],):
        raise ValueError("INT3 grouped shape mismatch")
    if not (weight.is_contiguous() and scale.is_contiguous() and q.is_contiguous() and a_scale.is_contiguous()):
        raise ValueError("INT3 grouped GEMM expects contiguous operands")
    _, _, indices, _, _, starts, experts, padded, count = padded_routing(routing)
    out = torch.empty((indices.numel(), n), device=q.device, dtype=torch.bfloat16)
    gemm, _ = kernels()
    gemm[(starts.numel() * triton.cdiv(n, 128),)](
        a_ptr=q,
        a_desc=None,
        b_ptr=weight,
        b_desc=None,
        bias_ptr=None,
        c_ptr=out,
        a_scale_ptr=a_scale,
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
        stride_bk=1,
        stride_bn=weight.stride(1),
        stride_bias_e=0,
        stride_bias_n=0,
        stride_cm=out.stride(0),
        stride_cn=1,
        stride_asm=1,
        stride_ask=0,  # one activation scale per row for every K group
        stride_bse=scale.stride(0),
        stride_bsk=1,
        stride_bsn=scale.stride(1),
        group_n=1,  # a weight scale per output row
        group_k=GROUP,
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
    COUNTERS["w3a8_gemm"] += 1
    return out


@triton.jit
def _dequant_bank_kernel(W, S, OUT, rows, K: tl.constexpr, G: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    g = tl.program_id(1)
    live = r < rows
    r64 = r.to(tl.int64)
    c = g * G + tl.arange(0, G)
    w = tl.load(W + r64[:, None] * K + c[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    s = tl.load(S + r64 * (K // G) + g, mask=live, other=0.0).to(tl.float32)
    tl.store(OUT + r64[:, None] * K + c[None, :], (w * s[:, None]).to(tl.bfloat16), mask=live[:, None])


def dequant_bank(weight, scale):
    """Whole-bank BF16 `(code - 4) * scale` `[E, N, K]` (backward only), one Triton pass.

    Bitwise the same bank as `dequant_bank_reference` (torch's BF16 multiply forms the product in fp32 and rounds it
    RTNE; the E4M3 code and the BF16 scale are exact in fp32): ~9x faster at gEAN320 sizes (w13 5.9 -> 0.6 ms)."""
    e, n, k = weight.shape
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.bfloat16 or scale.shape != (e, n, k // GROUP):
        raise ValueError(f"E4M3 [E, N, K] codes with BF16 [E, N, K/128] scales expected, got {weight.dtype} "
                         f"{tuple(weight.shape)} / {scale.dtype} {tuple(scale.shape)}")
    if k % GROUP or not (weight.is_contiguous() and scale.is_contiguous()):
        raise ValueError("dequant_bank needs contiguous operands and K % 128 == 0")
    out = torch.empty((e, n, k), dtype=torch.bfloat16, device=weight.device)
    rows = e * n
    _dequant_bank_kernel[(triton.cdiv(rows, 32), k // GROUP)](weight, scale, out, rows, K=k, G=GROUP, BR=32,
                                                               num_warps=4)
    return out


def dequant_bank_reference(weight, scale):
    """The torch formulation (expert chunks); tests check `dequant_bank` against it bit for bit."""
    e, n, k = weight.shape
    out = torch.empty((e, n, k), dtype=torch.bfloat16, device=weight.device)
    step = max(1, (1 << 28) // (n * k))
    for lo in range(0, e, step):
        hi = min(e, lo + step)
        out[lo:hi] = (
            weight[lo:hi].view(hi - lo, n, k // GROUP, GROUP).to(torch.bfloat16) * scale[lo:hi, :, :, None]
        ).view(hi - lo, n, k)
    return out


def grouped_a16(a, bank, starts, out, gather=None):
    """Sorted-row grouped GEMM `out[r] = a[gather[r] if gather else r] @ bank[e(r)].T`, fp32 accumulation, BF16 out.

    a: BF16 `[*, K]`, bank: BF16 `[E, N, K]` (dequant_bank), starts: int32 `[E+1]` row offsets of the experts in
    `out` (`[rows, N]`, contiguous, written), gather: int32 `[rows]` source row of every output row (or None: `a` is
    already in routing order). quack's SM100 varlen-M GEMM (TMA, tcgen05, fixed tile A16_TILE, no autotuning)."""
    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_tuned

    e, n, k = bank.shape
    if a.dtype != torch.bfloat16 or bank.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"A16 grouped GEMM takes BF16 operands, got {a.dtype} / {bank.dtype} -> {out.dtype}")
    if a.ndim != 2 or a.shape[1] != k or out.shape != ((a.shape[0] if gather is None else gather.shape[0]), n):
        raise ValueError(f"A16 grouped GEMM shapes: a {tuple(a.shape)}, bank {tuple(bank.shape)}, out {tuple(out.shape)}")
    if starts.dtype != torch.int32 or starts.shape != (e + 1,) or (gather is not None and gather.dtype != torch.int32):
        raise ValueError("A16 grouped GEMM: int32 starts [E+1] and int32 gather rows")
    if not (a.is_contiguous() and bank.is_contiguous() and out.is_contiguous() and starts.is_contiguous()):
        raise ValueError("A16 grouped GEMM expects contiguous operands")
    if torch.cuda.get_device_capability(a.device)[0] != 10:
        raise RuntimeError("the A16 grouped GEMM's quack tile is configured for SM100 (B200)")
    config = GemmConfig(**A16_TILE, pingpong=False, is_dynamic_persistent=True, device_capacity=10)
    gemm_tuned.fn(a, bank.transpose(-2, -1), out, cu_seqlens_m=starts, A_idx=gather, config=config)
    COUNTERS["a16_gemm"] += 1
    return out


def a16_experts(x, w1, w2, s1, s2, routing):
    """The W3A16 routed-expert forward in routing order: (`pre` [T*K, 2I], `y` [T*K, H]), both BF16 (module docstring).

    w13 reads the token rows through the routing's gather; w2 runs in A16_W2_ROWS chunks, each with its own SwiGLU
    and its experts' row offsets clipped to the chunk (a row's result does not depend on the chunking: every output
    row is one K-ordered dot product per column, the kernel splits no K)."""
    starts, gather = routing[1], routing[2]
    rows = gather.numel()
    bank = dequant_bank(w1, s1)
    pre = grouped_a16(x, bank, starts, torch.empty((rows, w1.shape[1]), dtype=torch.bfloat16, device=x.device),
                      gather=gather)
    del bank
    bank = dequant_bank(w2, s2)
    y = torch.empty((rows, w2.shape[1]), dtype=torch.bfloat16, device=x.device)
    for lo in range(0, rows, A16_W2_ROWS):
        hi = min(rows, lo + A16_W2_ROWS)
        grouped_a16(swiglu_bf16(pre[lo:hi]), bank, (starts - lo).clamp_(0, hi - lo), y[lo:hi])
    return pre, y


def bf16_grouped(a, weight, scale, routing, gather=False):
    """Backward-direction grouped GEMM (out = a @ W_e) on the dequantized BF16 bank."""
    _, starts, indices = routing[:3]
    rows = a.index_select(0, indices.long()) if gather else a
    bank = dequant_bank(weight, scale)
    COUNTERS["bf16_dx_gemm"] += 1
    return torch.nn.functional.grouped_mm(rows, bank, offs=starts[1:])


class FrozenInt3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, s1, s2, indices, scores):
        from liger_kernel.ops.fused_moe import _token_aggregation

        if w1.requires_grad or w2.requires_grad:
            raise RuntimeError("INT3 base must stay frozen")
        x = x.contiguous()
        routing = padded_routing(routing_metadata(indices.contiguous().int(), w1.shape[0]))
        if ACT["mode"] == "bf16":
            pre, y = a16_experts(x, w1, w2, s1, s2, routing)
            COUNTERS["a16_forward"] += 1
        elif ACT["mode"] == "fp8":
            gather = routing[2].long()
            # the row quantization is per token: quantize once, then route the E4M3 rows and their scales
            q, a_scale = quantize_rows(x)
            pre = grouped_w3a8(q.index_select(0, gather), a_scale.index_select(0, gather), w1, s1, routing)
            q2, a_scale2 = swiglu_quantize(pre)
            y = grouped_w3a8(q2, a_scale2, w2, s2, routing)
            COUNTERS["a8_forward"] += 1
        else:
            raise ValueError(f"INT3 activation format {ACT['mode']!r}")
        out = _token_aggregation(y, scores.flatten().contiguous(), routing[4], x.shape[0], indices.shape[1], x.shape[1])
        ctx.save_for_backward(w1, w2, s1, s2, pre, y, scores, *routing)
        COUNTERS["forward"] += 1
        return out

    @staticmethod
    def backward(ctx, dy):
        from liger_kernel.ops.fused_moe_kernels import _token_gather_weighted_sum_kernel

        # The activation quantizers are straight-through; dX uses the dequantized BF16 weights and the
        # forward's own `pre` / `y`, so the only difference to a BF16 base is the forward's rounding.
        w1, w2, s1, s2, pre, y, scores, *routing = ctx.saved_tensors
        dy = dy.contiguous()
        dpost = bf16_grouped(dy, w2, s2, routing, gather=True)
        dpre = torch.empty_like(pre)
        i = pre.shape[1] // 2
        _swiglu_backward[(pre.shape[0],)](pre, dpost, scores, routing[3], dpre, i, triton.next_power_of_2(i))
        ds = torch.empty_like(scores)
        _routing_backward[(pre.shape[0],)](
            dy, y, routing[2], routing[3], ds, dy.shape[1], triton.next_power_of_2(dy.shape[1])
        )
        expanded = bf16_grouped(dpre, w1, s1, routing)
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


class FrozenInt3Experts(nn.Module):
    def __init__(self, experts, hidden, intermediate, device="cpu"):
        super().__init__()
        if hidden % GROUP or intermediate % GROUP:
            raise ValueError("INT3 experts need hidden and intermediate sizes that are multiples of 128")
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
            torch.empty(experts, 2 * intermediate, hidden // GROUP, device=device, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "down_scale", torch.empty(experts, hidden, intermediate // GROUP, device=device, dtype=torch.bfloat16)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        shape = hidden_states.shape
        out = FrozenInt3Function.apply(
            hidden_states.reshape(-1, self.hidden_dim),
            self.gate_up_proj,
            self.down_proj,
            self.gate_up_scale,
            self.down_scale,
            top_k_index,
            top_k_weights,
        )
        return out.reshape(shape)
