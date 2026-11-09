"""Frozen Qwen4ExpTextGatedResidual (hyper-connection read: grouped RMSNorm, mix gate, inject gate) as one custom op,
forward and dX-only backward (``--fast-kernels hc``, trainer_kernels.py).

The eager module reads the 4 x 2560 residual stream about a dozen times per call (norm, two projections, silu, sigmoid,
product, mean, each with a backward); here the norm, the gated mix and their backward are three Triton passes and the
two projections stay cuBLAS:

    zb, r = rmsnorm_g(h) * (1 + w)          # _norm_fwd_kernel: BF16 normed stream, fp32 1/rms per group
    du = zb @ [W_down; W_inject]^T          # [T, 320 + 4] BF16 (one GEMM for both projections)
    s = silu(du[:, :320] / 4)               # BF16
    pre = s @ W_up^T                        # [T, 4 x 2560] BF16, saved for backward
    x = mean_g(sigmoid(pre) * zb)           # _mix_kernel, fp32 accumulation, one BF16 rounding
    gate = 2 sigmoid(du[:, 320:] / 4)

Rounding: HF eager rounds sigmoid(pre) and its product with the normed stream to BF16 before the mean; here they stay
fp32 and the mean is rounded once (closer to an fp64 reference, check_trainer_kernels.py). The weights are frozen, so
no dW is formed; the op raises if one requires grad. Derived from work/qwen38fn_dflash/probes/hc_fused.py ("cublas").
"""

import torch
import triton
import triton.language as tl

G, H, R = 4, 2560, 320
RC = 336  # R + G rows of W_down / W_inject, padded to a multiple of 16
COUNTERS = {'forward': 0, 'backward': 0, 'inject_forward': 0, 'inject_backward': 0}


@triton.jit
def _norm_fwd_kernel(h_ptr, w_ptr, zb_ptr, r_ptr, eps, H: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    for g in tl.static_range(G):
        x = tl.load(h_ptr + row * (G * H) + g * H + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, 0) / H + eps)
        w = tl.load(w_ptr + g * H + offs, mask=mask, other=0.0)
        z = x * r * (1.0 + w)
        tl.store(zb_ptr + row * (G * H) + g * H + offs, z.to(tl.bfloat16), mask=mask)
        tl.store(r_ptr + row * G + g, r)


def _mix_configs():
    return [triton.Config({'BM': bm, 'BN': bn}, num_warps=nw) for bm in (16, 32, 64) for bn in (128, 256)
            for nw in (4, 8)]


# Every output element is computed independently (fixed order over g), so the autotuned tile does not change results.
@triton.autotune(configs=_mix_configs(), key=['H_BF16', 'BACKWARD'])
@triton.jit
def _mix_kernel(pre_ptr, h_ptr, r_ptr, w_ptr, gx_ptr, x_ptr, dpre_ptr, dzmix_ptr, M,
                H: tl.constexpr, G: tl.constexpr, H_BF16: tl.constexpr, BACKWARD: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    rmask = rows < M
    if BACKWARD:
        gx = tl.load(gx_ptr + rows[:, None] * H + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        gx = gx * 0.25
    else:
        xacc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in tl.static_range(G):
        offs = rows[:, None] * (G * H) + g * H + cols[None, :]
        m = tl.sigmoid(tl.load(pre_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32))
        hv = tl.load(h_ptr + offs, mask=rmask[:, None], other=0.0)
        rv = tl.load(r_ptr + rows * G + g, mask=rmask, other=0.0)
        wv = tl.load(w_ptr + g * H + cols)
        z = hv.to(tl.float32) * rv[:, None] * (1.0 + wv[None, :])
        if H_BF16:
            z = z.to(tl.bfloat16).to(tl.float32)  # RMSNorm returns the input dtype
        if BACKWARD:
            dpre = gx * z * m * (1.0 - m)
            tl.store(dpre_ptr + offs, dpre.to(tl.bfloat16), mask=rmask[:, None])
            tl.store(dzmix_ptr + offs, gx * m, mask=rmask[:, None])
        else:
            xacc += m * z
    if not BACKWARD:
        x = xacc * 0.25
        tl.store(x_ptr + rows[:, None] * H + cols[None, :], x.to(x_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _norm_bwd_kernel(dz_ptr, h_ptr, r_ptr, w_ptr, dh_ptr, M, ROWS, H: tl.constexpr, G: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    for g in tl.static_range(G):
        w1 = 1.0 + tl.load(w_ptr + g * H + offs, mask=mask, other=0.0)
        for i in range(ROWS):
            row = (pid * ROWS + i).to(tl.int64)
            ok = mask & (row < M)
            base = row * (G * H) + g * H
            hv = tl.load(h_ptr + base + offs, mask=ok, other=0.0).to(tl.float32)
            dz = tl.load(dz_ptr + base + offs, mask=ok, other=0.0)
            r = tl.load(r_ptr + row * G + g, mask=row < M, other=0.0)
            c = tl.sum(dz * w1 * hv, 0)
            dh = w1 * r * dz - hv * (r * r * r / H) * c
            tl.store(dh_ptr + base + offs, dh.to(dh_ptr.dtype.element_ty), mask=ok)


def _cat_weight(w_down, w_inject):
    wc = torch.zeros(RC, G * H, device=w_down.device, dtype=torch.bfloat16)
    wc[:R] = w_down
    if w_inject is not None:
        wc[R:R + G] = w_inject
    return wc


def _grid(rows):
    return lambda meta: (triton.cdiv(rows, meta['BM']), H // meta['BN'])


@torch.library.custom_op('fn_trainer::hc_mix_fwd', mutates_args=())
def mix_fwd(h: torch.Tensor, norm_w: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor,
            w_inject: torch.Tensor, combine: bool, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                    torch.Tensor, torch.Tensor]:
    """h [..., G, H] -> (mixed [..., H], gate [..., G] BF16, and r / du / pre saved for mix_bwd)."""
    with torch.autocast('cuda', enabled=False):
        shape = h.shape
        h2 = h.reshape(-1, G * H)
        rows = h2.shape[0]
        zb = torch.empty(rows, G * H, device=h.device, dtype=torch.bfloat16)
        r = torch.empty(rows, G, device=h.device, dtype=torch.float32)
        _norm_fwd_kernel[(rows,)](h2, norm_w, zb, r, eps, H=H, G=G, BLOCK=4096, num_warps=8)
        du = zb @ _cat_weight(w_down, w_inject if combine else None).T  # [T, 336] BF16
        s = torch.nn.functional.silu(du[:, :R].float() * 0.25).bfloat16()
        gate = (2 * torch.sigmoid(du[:, R:R + G].float() * 0.25)).bfloat16()
        x = torch.empty(rows, H, device=h.device, dtype=h.dtype)
        pre = s @ w_up.bfloat16().T
        _mix_kernel[_grid(rows)](pre, h2, r, norm_w, None, x, None, None, rows, H=H, G=G,
                                 H_BF16=h.dtype == torch.bfloat16, BACKWARD=False)
        return x.view(*shape[:-2], H), gate.view(*shape[:-2], G), r, du, pre


@mix_fwd.register_fake
def _(h, norm_w, w_down, w_up, w_inject, combine, eps):
    lead = h.shape[:-2]
    rows = h.numel() // (G * H)
    return (h.new_empty(*lead, H), h.new_empty(*lead, G, dtype=torch.bfloat16),
            h.new_empty(rows, G, dtype=torch.float32), h.new_empty(rows, RC, dtype=torch.bfloat16),
            h.new_empty(rows, G * H, dtype=torch.bfloat16))


@torch.library.custom_op('fn_trainer::hc_mix_bwd', mutates_args=())
def mix_bwd(gx: torch.Tensor, ggate: torch.Tensor, h: torch.Tensor, norm_w: torch.Tensor, w_down: torch.Tensor,
            w_up: torch.Tensor, w_inject: torch.Tensor, r: torch.Tensor, du: torch.Tensor, pre: torch.Tensor,
            combine: bool) -> torch.Tensor:
    with torch.autocast('cuda', enabled=False):
        shape = h.shape
        h2 = h.reshape(-1, G * H)
        rows = h2.shape[0]
        gx = gx.reshape(rows, H).contiguous()
        dpre = torch.empty(rows, G * H, device=h.device, dtype=torch.bfloat16)
        dzmix = torch.empty(rows, G * H, device=h.device, dtype=torch.float32)
        _mix_kernel[_grid(rows)](pre, h2, r, norm_w, gx, None, dpre, dzmix, rows, H=H, G=G,
                                 H_BF16=h.dtype == torch.bfloat16, BACKWARD=True)
        ds = dpre @ w_up.bfloat16()                       # [T, 320] BF16
        # silu(v)' = sigmoid(v) (1 + v (1 - sigmoid(v))), v = du / 4
        v = du[:, :R].float() * 0.25
        sv = torch.sigmoid(v)
        dcat = torch.zeros(rows, RC, device=h.device, dtype=torch.bfloat16)
        dcat[:, :R] = (ds.float() * sv * (1 + v * (1 - sv)) * 0.25).bfloat16()
        if combine:
            sg = torch.sigmoid(du[:, R:R + G].float() * 0.25)
            dcat[:, R:R + G] = (ggate.reshape(rows, G).float() * 2 * sg * (1 - sg) * 0.25).bfloat16()
        dz = torch.addmm(dzmix, dcat, _cat_weight(w_down, w_inject if combine else None), out_dtype=torch.float32)
        per_program = 32
        dh = torch.empty(rows, G * H, device=h.device, dtype=h.dtype)
        _norm_bwd_kernel[(triton.cdiv(rows, per_program),)](dz, h2, r, norm_w, dh, rows, per_program, H=H, G=G,
                                                            BLOCK=4096, num_warps=8)
        return dh.view(shape)


@mix_bwd.register_fake
def _(gx, ggate, h, norm_w, w_down, w_up, w_inject, r, du, pre, combine):
    return torch.empty_like(h)


def _setup(ctx, inputs, output):
    h, norm_w, w_down, w_up, w_inject, combine, eps = inputs
    _, _, r, du, pre = output
    ctx.mark_non_differentiable(r, du, pre)
    ctx.save_for_backward(h, norm_w, w_down, w_up, w_inject, r, du, pre)
    ctx.combine = combine


def _backward(ctx, gx, ggate, _r, _du, _pre):
    h, norm_w, w_down, w_up, w_inject, r, du, pre = ctx.saved_tensors
    if any(ctx.needs_input_grad[1:5]):
        raise RuntimeError('hc_mix: the hyper-connection weights must be frozen (dX-only backward)')
    if ggate is None:  # no gate output (the final readout) or the gate did not reach the loss
        ggate = torch.zeros(h.shape[:-1], device=h.device, dtype=torch.bfloat16)
    COUNTERS['backward'] += 1
    return mix_bwd(gx, ggate, h, norm_w, w_down, w_up, w_inject, r, du, pre, ctx.combine), None, None, None, None, \
        None, None


mix_fwd.register_autograd(_backward, setup_context=_setup)


def gated_residual_forward(module, hyper_input):
    """``Qwen4ExpTextGatedResidual.forward`` through mix_fwd: (mixed, hyper_input, inject gate), or mixed alone for the
    final readout (no block_inject_weight)."""
    if (module.hc_count, module.hidden_size, module.input_mix_weight_down.weight.shape[0]) != (G, H, R) \
            or module.hc_norm.group_size != module.hidden_size:
        raise RuntimeError('hc_mix: kernel constants G/H/R do not match this module')
    if hyper_input.shape[-1] != G * H:
        raise ValueError(f'Expected {G * H} hyper-connection features, got {hyper_input.shape[-1]}.')
    combine = module.block_inject_weight is not None
    # without an inject projection the op still needs a [G, G*H] operand; its rows of du are never read
    w_inject = module.block_inject_weight.weight if combine else module.input_mix_weight_down.weight[:G]
    COUNTERS['forward'] += 1
    mixed, gate, *_ = mix_fwd(hyper_input.unflatten(-1, (G, H)), module.hc_norm.weight,
                              module.input_mix_weight_down.weight, module.input_mix_weight_up.weight, w_inject,
                              combine, module.hc_norm.eps)
    return (mixed, hyper_input, gate) if combine else mixed


# ------------------------------------------------------------------------------------- residual write-back (hc_inject)
# Qwen4ExpTextDecoderLayer adds each block's output y back into the stream through the inject gate:
#     out = hyper_input + (y[..., None, :] * gate[..., :, None]).flatten(-2)
# Eager is a broadcast mul and an add forward, two [T, 4, 2560] products and two reductions backward (~1.6 s of a
# 120k-token QAD step). Here one pass each way. Forward is bitwise eager (the product is rounded to BF16 before the
# add, as eager does). Backward rounds each product to BF16 like eager's MulBackward and sums in fp32 in a fixed order:
# dy over the 4 groups, dgate over the 2560 columns in BN chunks (the config is pinned so reruns stay bitwise).
INJECT_FWD = {'BM': 8, 'BN': 512, 'num_warps': 4}
INJECT_BWD = {'BM': 4, 'BN': 256, 'num_warps': 4}


@triton.jit
def _round_bf16(x):
    """fp32 -> the nearest-even BF16 value, kept in fp32. A plain .to(bf16).to(fp32) round trip is folded away when an
    add follows (22% of write-back outputs then missed eager's rounding)."""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _inject_fwd_kernel(h_ptr, y_ptr, w_ptr, out_ptr, M, H: tl.constexpr, G: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    rmask = rows < M
    y = tl.load(y_ptr + rows[:, None] * H + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    for g in tl.static_range(G):
        wv = tl.load(w_ptr + rows * G + g, mask=rmask, other=0.0).to(tl.float32)
        offs = rows[:, None] * (G * H) + g * H + cols[None, :]
        hv = tl.load(h_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        inj = _round_bf16(y * wv[:, None])
        tl.store(out_ptr + offs, (hv + inj).to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _inject_bwd_kernel(dout_ptr, y_ptr, w_ptr, dy_ptr, dw_ptr, M, H: tl.constexpr, G: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    rmask = rows < M
    gidx = tl.arange(0, G)
    dw = tl.zeros((BM, G), dtype=tl.float32)
    for chunk in tl.static_range(H // BN):
        cols = chunk * BN + tl.arange(0, BN)
        y = tl.load(y_ptr + rows[:, None] * H + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        dy = tl.zeros((BM, BN), dtype=tl.float32)
        for g in tl.static_range(G):
            wv = tl.load(w_ptr + rows * G + g, mask=rmask, other=0.0).to(tl.float32)
            d = tl.load(dout_ptr + rows[:, None] * (G * H) + g * H + cols[None, :], mask=rmask[:, None],
                        other=0.0).to(tl.float32)
            dy += _round_bf16(d * wv[:, None])
            part = tl.sum(_round_bf16(d * y), axis=1)
            dw += tl.where(gidx[None, :] == g, part[:, None], 0.0)
        tl.store(dy_ptr + rows[:, None] * H + cols[None, :], dy.to(tl.bfloat16), mask=rmask[:, None])
    tl.store(dw_ptr + rows[:, None] * G + gidx[None, :], dw.to(tl.bfloat16), mask=rmask[:, None])


def _inject_rows(h, y, gate):
    lead = y.shape[:-1]
    if h.shape != (*lead, G * H) or y.shape[-1] != H or gate.shape != (*lead, G) \
            or {h.dtype, y.dtype, gate.dtype} != {torch.bfloat16}:
        raise ValueError(f'hc_inject: expected BF16 [..., {G * H}], [..., {H}], [..., {G}]; got '
                         f'{tuple(h.shape)} {h.dtype}, {tuple(y.shape)} {y.dtype}, {tuple(gate.shape)} {gate.dtype}')
    return y.numel() // H


@torch.library.custom_op('fn_trainer::hc_inject', mutates_args=())
def inject(h: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """h [..., G*H], y [..., H], gate [..., G] (BF16) -> h + (y[..., None, :] * gate[..., None]).flatten(-2)."""
    rows = _inject_rows(h, y, gate)
    h2, y2, w2 = h.reshape(rows, G * H).contiguous(), y.reshape(rows, H).contiguous(), gate.reshape(rows, G).contiguous()
    out = torch.empty_like(h2)
    c = INJECT_FWD
    _inject_fwd_kernel[(triton.cdiv(rows, c['BM']), H // c['BN'])](h2, y2, w2, out, rows, H=H, G=G, BM=c['BM'],
                                                                    BN=c['BN'], num_warps=c['num_warps'])
    return out.view(h.shape)


@inject.register_fake
def _(h, y, gate):
    _inject_rows(h, y, gate)
    return torch.empty_like(h)


@torch.library.custom_op('fn_trainer::hc_inject_bwd', mutates_args=())
def inject_bwd(dout: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _inject_rows(dout, y, gate)
    d2, y2, w2 = dout.reshape(rows, G * H).contiguous(), y.reshape(rows, H).contiguous(), gate.reshape(rows, G).contiguous()
    dy = torch.empty(rows, H, device=y.device, dtype=torch.bfloat16)
    dw = torch.empty(rows, G, device=y.device, dtype=torch.bfloat16)
    c = INJECT_BWD
    if H % c['BN']:
        raise RuntimeError(f'hc_inject: backward chunk {c["BN"]} does not divide {H}')
    _inject_bwd_kernel[(triton.cdiv(rows, c['BM']),)](d2, y2, w2, dy, dw, rows, H=H, G=G, BM=c['BM'], BN=c['BN'],
                                                      num_warps=c['num_warps'])
    return dy.view(y.shape), dw.view(gate.shape)


@inject_bwd.register_fake
def _(dout, y, gate):
    return torch.empty_like(y), torch.empty_like(gate)


def _inject_setup(ctx, inputs, output):
    _, y, gate = inputs
    ctx.save_for_backward(y, gate)


def _inject_backward(ctx, dout):
    y, gate = ctx.saved_tensors
    COUNTERS['inject_backward'] += 1
    dy, dgate = inject_bwd(dout, y, gate)
    return dout, dy, dgate


inject.register_autograd(_inject_backward, setup_context=_inject_setup)


def residual_inject(hyper_input, y, gate):
    """The decoder layer's write-back ``hyper_input + (y.unsqueeze(-2) * gate.unsqueeze(-1)).flatten(-2)``."""
    COUNTERS['inject_forward'] += 1
    return inject(hyper_input, y, gate)
