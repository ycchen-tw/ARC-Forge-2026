"""Opt-in faster kernels for the FN LoRA trainers: ``--fast-kernels`` of train_tito_mm.py and train_rl_service.py.

Each replaces one eager region of the frozen base or of the LoRA path and reproduces it to BF16 rounding; the
reference is the code it replaces. check_trainer_kernels.py holds every kernel against an fp64 reference on the real
layer shapes (and act_quant bitwise against the pinned SGLang kernel); ``install(..., shadow=True)`` re-runs the
reference forward next to every bound call inside the real model and records the difference.

  hc         Qwen4ExpTextGatedResidual (frozen; 96 layer connections + the final readout) -> hc_mix.py: grouped
             RMSNorm, gated mix and inject gate in three Triton passes around two cuBLAS GEMMs, dX-only backward.
  hc_inject  the decoder layers' residual write-back hyper_input + (y x inject gate) -> hc_mix.inject: one Triton pass
             forward (bitwise eager) and one backward (dy bitwise, dgate by summation order, 2.4e-6 relative). Bound
             inside packed_qwen4's layer wrapper; not shadowed. At 120k tokens fwd+bwd 12.0 -> 2.5 ms per call.
  lora       PEFT lora.Linear (fp32 A/B, BF16 base) -> LoRABF16: s = x A^T, the add-back, ds = dY B and dX on BF16
             operands with fp32 accumulation (fp32 operands as hi/lo BF16 pairs); dA and dB, which reduce over the
             tokens, stay fp32 GEMMs (see LoRABF16 for the cuBLAS accuracy trap that rules out BF16 there). The
             --dense-qat projections keep their own forward (fp8_dense_qat.py) and are not touched.
  lora_grad  the LoRA weight gradients dA = ds^T x, dB = dY^T s (reduce over T) of LoRABF16 and of fp8_dense_qat ->
             lora_grad.py: one Triton pass over the BF16 / e4m3 activations with the fp32 factor as BF16 hi + lo, fp32
             accumulation, fixed-order split-T sum, instead of fp32 copies of x / dY and fp32 SIMT GEMMs. 2.5-3.2e-6
             relative to fp64 at every (T, width) of the sweep (fp32 GEMM 0.2-2e-6); per module at 100k tokens
             LoRABF16 3.97 -> 0.33 ms, dense QAT 10.1 -> 1.0 ms.
  gdn_norm   Qwen4ExpTextRMSNormGated (GDN output norm, sigmoid gate, frozen weight) -> FLA's fused gated RMSNorm
             kernels, NB = 1 and one pinned config. FLA's own wrapper passes NB = cdiv(T, 65536) as a constexpr that
             only keys its autotuner, so every new packed length recompiled and the GPU idled behind it.
  act_quant  fp8_moe.quantize (1x128 E4M3 activation quantizer of the FP8 experts and of the dense-QAT GEMMs): the
             pinned SGLang kernel's math with 32 groups per program instead of one (bitwise equal).
  qsa_select sparse_qsa.select_compact after the score GEMM -> qsa_select_fused.py (one kernel per 4096-row chunk;
             bitwise the same selection, check_qsa_gather.py gate s2). Per QSA layer at 120k: 126 -> 58 ms.
  qsa_dkdv   qsa_gather dK/dV -> qsa_dkdv_gluon.py: Gluon, 64-key tiles, TMA-gather ring (Blackwell only). Within the
             BF16 floor like the Triton kernel, not bitwise to it (4.7e-4). Per QSA layer at 120k: 152 -> 96 ms.
  gdn_gva    Qwen4ExpTextGatedDeltaNet.forward without the q/k repeat_interleave 16 -> 48 heads: FlashQLA runs the
             grouped-value layout itself (dq / dk group-reduced in the kernel). Output, dv, dg, dbeta bitwise equal;
             dq / dk differ by summation order only (3.3e-3 relative, the same 4.7e-3 against fp64 as the reference).
             Per GDN layer fwd+bwd at 120k 33.6 -> 27.6 ms, 4 GiB less peak.
  qsa_union  (named only, not in 'all') qsa_gather forward and dQ -> qsa_union.py: 10 consecutive queries of a segment
             share a 128-row tcgen05 tile and iterate the union of their selected blocks, each row masked to its own
             selection (Blackwell, BF16 K/V only). Within the BF16 floor like the gather kernels, not bitwise to them;
             reruns and packed == per-segment bitwise. Per QSA layer and micro step at 120k 310 -> 207 ms; v3.1
             whole model (W3A16, world 3) -1.46 s per micro step (-5.7%). QSA_GATHER_KERNEL.md "Query union".

2026-10-11, FP8 base, 4 x ~100k-token RL bins (node1): 25.15 -> 14.90 s per step (hc -5.6 s, lora -2.8 s,
gdn_norm -1.3 s, act_quant -0.6 s, additive), peak 132.7 -> 118.8 GiB. In-model relative L2 of each call against
the replaced code: hc 3.8e-3, inject gate 2.5e-3, lora 1.3e-4, gdn_norm 3.3e-3, act_quant 0. Whole-model logp /
gradient differences under decision replay equal those of a 1-ulp perturbation of 0.1% of the input embedding, so
they say nothing about kernel correctness (docs/research/qwen38fn_rl_trainer_speed_20261011.md).
"""

from types import MethodType

import torch
import triton

KERNELS = ('hc', 'hc_inject', 'lora', 'lora_grad', 'gdn_norm', 'act_quant', 'qsa_select', 'qsa_dkdv', 'gdn_gva')
NAMED_ONLY = ('qsa_union',)   # accepted by parse() when named, not in 'all'
ADAPTER = 'default'
# lora_grad: LoRABF16 and fp8_dense_qat take their dA / dB from lora_grad.tgrad
LORA_GRAD = {'on': False}
COUNTERS = {'lora_forward': 0, 'lora_backward': 0, 'gdn_norm_forward': 0, 'gdn_norm_backward': 0, 'gdn_gva_forward': 0}
# Set by install(): {'hc': {'layer': n, 'other': n}, 'lora': n, 'gdn_norm': n, 'act_quant': True}, minus what is off.
INSTALLED = {}
_SEEN = {}
# install(shadow=True): {'<kernel>[<output>]': {'calls', 'exact', 'rel_sum', 'rel_max'}} of fast vs reference forward.
SHADOW = {}
# (BT, num_warps), pinned so reruns stay bitwise; the fastest pair at 100k tokens x 48 heads (every pair gives the same
# error against fp64): fwd+bwd 1.89 ms vs 3.22 ms for (32, 4) / (32, 4)
GDN_NORM_CONFIG = {'fwd': (32, 4), 'bwd': (64, 16)}


def parse(value):
    """'none' | 'all' | 'all,<NAMED_ONLY...>' | comma list of KERNELS + NAMED_ONLY -> the kernels in that order.
    qsa_union is gated with the Gluon dK/dV only, so it needs qsa_dkdv (named or through 'all')."""
    if value == 'none':
        return ()
    names = value.split(',')
    if names[0] == 'all':
        extra = names[1:]
        if not set(extra) <= set(NAMED_ONLY) or len(set(extra)) != len(extra):
            raise ValueError(f"--fast-kernels {value!r}: after 'all' only {NAMED_ONLY}, each once")
        names = list(KERNELS) + extra
    unknown = [n for n in names if n not in KERNELS + NAMED_ONLY]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f'--fast-kernels {value!r}: unknown or repeated {unknown or names}; choose from '
                         f'{KERNELS + NAMED_ONLY}')
    if 'qsa_union' in names and 'qsa_dkdv' not in names:
        raise ValueError(f'--fast-kernels {value!r}: qsa_union is validated with the Gluon dK/dV only; add qsa_dkdv')
    return tuple(k for k in KERNELS + NAMED_ONLY if k in names)


def add_arguments(parser, debug=False):
    parser.add_argument('--fast-kernels', default='none',
                        help=f"'none', 'all', 'all,{','.join(NAMED_ONLY)}' or a comma list of "
                             f"{', '.join(KERNELS + NAMED_ONLY)} ({', '.join(NAMED_ONLY)}: only when named, not in "
                             f"'all'; qsa_union needs qsa_dkdv): faster trainer kernels that "
                             'reproduce the reference to BF16 rounding (trainer_kernels.py)')
    if debug:
        parser.add_argument('--fast-kernels-shadow', action='store_true',
                            help='debug: every fast-kernel call also runs the reference forward on the same inputs; '
                                 'the per-kernel relative L2 goes into the step record (slow)')


def check_arguments(parser, args, debug=False):
    try:
        kernels = parse(args.fast_kernels)
    except ValueError as error:
        parser.error(str(error))
    if debug and args.fast_kernels_shadow and not set(kernels) - {'act_quant'}:
        parser.error('--fast-kernels-shadow needs hc, lora or gdn_norm in --fast-kernels')


# ---------------------------------------------------------------------------------------------------------- gdn_gva
def _gva_forward():
    """The installed transformers' Qwen4ExpTextGatedDeltaNet.forward with its two q / k repeat_interleave lines removed
    (built from the source so it follows that version; fails if the lines are not there exactly once)."""
    import inspect
    import textwrap

    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    src = textwrap.dedent(inspect.getsource(M.Qwen4ExpTextGatedDeltaNet.forward))
    for name in ('query', 'key'):
        line = f'{name} = {name}.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)'
        if src.count(line) != 1:
            raise RuntimeError(f'gdn_gva: {line!r} is not in {M.__file__} exactly once; re-derive the GVA forward')
        src = src.replace(line, f'pass  # gdn_gva: {name} stays at num_k_heads, FlashQLA groups the value heads')
    namespace = {}
    exec(compile(src, f'<gdn_gva: {M.__file__}>', 'exec'), M.__dict__, namespace)
    hf_gva = namespace['forward']

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None:
            raise RuntimeError('gdn_gva is a training forward (no cache); the cached decode path needs repeated heads')
        COUNTERS['gdn_gva_forward'] += 1
        return hf_gva(self, hidden_states, None, attention_mask, **kwargs)

    return forward


# -------------------------------------------------------------------------------------------------------- hc_inject
def _inject_forward():
    """The installed transformers' Qwen4ExpTextDecoderLayer.forward with its two residual write-backs through
    hc_mix.residual_inject (built from the source like gdn_gva; fails if the lines are not there exactly twice)."""
    import inspect
    import textwrap

    import hc_mix
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    src = textwrap.dedent(inspect.getsource(M.Qwen4ExpTextDecoderLayer.forward))
    old = ('    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)\n'
           '    hidden_states = hyper_input + injection.flatten(-2)\n')
    if src.count(old) != 2:
        raise RuntimeError(f'hc_inject: the residual write-back is not in {M.__file__} exactly twice; re-derive it')
    src = src.replace(old, '    hidden_states = _hc_residual_inject(hyper_input, hidden_states, injection_weights)\n')
    namespace = {}
    exec(compile(src, f'<hc_inject: {M.__file__}>', 'exec'), dict(M.__dict__, _hc_residual_inject=hc_mix.residual_inject),
         namespace)
    return namespace['forward']


# ------------------------------------------------------------------------------------------------------------- lora
def _split(t32):
    """fp32 -> (hi, lo) BF16 with hi + lo equal to t32 to ~2^-16 relative."""
    hi = t32.to(torch.bfloat16)
    return hi, (t32 - hi.float()).to(torch.bfloat16)


class LoRABF16(torch.autograd.Function):
    """y = base + scale * (x A^T) B^T. The GEMMs that reduce over the feature axes (s = x A^T over K, ds = dY B over N)
    run on BF16 operands with fp32 accumulation, the fp32 operands (A, B) entering as hi/lo BF16 pairs stacked on the
    rank axis, so s and ds keep fp32-level accuracy; the forward add-back uses BF16 s and B (its output is BF16 anyway).
    dA = ds^T x and dB = dY^T s reduce over the T tokens and stay fp32 GEMMs, as in PEFT: on cuBLAS (torch 2.11, cu130,
    B200) a BF16 GEMM with fp32 output whose reduction axis is T silently drops to BF16-level accuracy for some
    (N, T) -- dB off by 2-12% at N=48 / T=2049, 4099, 99999, 100000 and N=10240 / T=4099 -- whatever
    allow_bf16_reduced_precision_reduction says (check_trainer_kernels.py sweeps every LoRA shape over T)."""

    @staticmethod
    def forward(ctx, x, base, a32, b32, scale):
        if x.dtype != torch.bfloat16 or base.dtype != torch.bfloat16 or a32.dtype != torch.float32 \
                or b32.dtype != torch.float32:
            raise TypeError(f'LoRABF16: x {x.dtype}, base {base.dtype}, A {a32.dtype}, B {b32.dtype}')
        r = a32.shape[0]
        x2 = x.reshape(-1, x.shape[-1])
        a_hi, a_lo = _split(a32)
        s2 = torch.mm(x2, torch.cat([a_hi, a_lo]).t(), out_dtype=torch.float32)            # [T, 2r]
        s32 = s2[:, :r] + s2[:, r:]
        y = torch.addmm(base.reshape(-1, base.shape[-1]), s32.to(torch.bfloat16), b32.to(torch.bfloat16).t(),
                        alpha=scale)
        ctx.save_for_backward(x2, s32, a_hi, b32)
        ctx.scale, ctx.x_shape = scale, x.shape
        COUNTERS['lora_forward'] += 1
        return y.view(base.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, s32, a_hi, b32 = ctx.saved_tensors
        r = s32.shape[1]
        dy2 = dy.reshape(-1, dy.shape[-1])
        if dy2.dtype != torch.bfloat16:
            raise TypeError(f'LoRABF16 backward got a {dy2.dtype} output gradient')
        d2 = torch.mm(dy2, torch.cat(_split(b32), dim=1), out_dtype=torch.float32)          # [T, 2r]
        ds32 = (d2[:, :r] + d2[:, r:]) * ctx.scale
        if LORA_GRAD['on']:
            import lora_grad
            da = lora_grad.tgrad(x2, ds32).t()
            db = lora_grad.tgrad(dy2, s32) * ctx.scale
        else:
            da = torch.mm(ds32.t(), x2.float())                                             # [r, K], fp32 over T
            db = torch.mm(dy2.float().t(), s32) * ctx.scale                                 # [N, r], fp32 over T
        dx = torch.mm(ds32.to(torch.bfloat16), a_hi).view(ctx.x_shape)
        COUNTERS['lora_backward'] += 1
        return dx, dy, da, db, None


def lora_forward(module, x, *args, **kwargs):
    """``forward`` of a plain single-adapter PEFT lora.Linear through LoRABF16 (bound on the instance)."""
    if module.disable_adapters or module.merged or args or kwargs:
        raise RuntimeError('fast lora: adapters disabled/merged or extra forward arguments')
    result = module.base_layer(x)
    return LoRABF16.apply(x, result, module.lora_A[ADAPTER].weight, module.lora_B[ADAPTER].weight,
                          module.scaling[ADAPTER])


def _check_plain_lora(name, module):
    a, b = module.lora_A[ADAPTER].weight, module.lora_B[ADAPTER].weight
    if (list(module.lora_A) != [ADAPTER] or module.active_adapters != [ADAPTER] or module.merged
            or module.disable_adapters or module.use_dora.get(ADAPTER, False)
            or not isinstance(module.lora_dropout[ADAPTER], torch.nn.Identity)
            or module.lora_A[ADAPTER].bias is not None or module.lora_B[ADAPTER].bias is not None
            or a.dtype != torch.float32 or b.dtype != torch.float32 or 'forward' in module.__dict__):
        raise RuntimeError(f'{name}: fast lora needs a plain bias-free single-adapter fp32 LoRA with no bound forward')


# --------------------------------------------------------------------------------------------------------- gdn_norm
class GatedRMSNormFLA(torch.autograd.Function):
    """``fla.modules.fused_norm_gate``'s D <= 512 kernels called directly: y = rmsnorm(x) * w * sigmoid(g), fp32 inside,
    one BF16 rounding; dX and dG only (the weight is frozen)."""

    @staticmethod
    def forward(ctx, x, g, w, eps):
        import fla.modules.fused_norm_gate as F
        x, g = x.contiguous(), g.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        F.layer_norm_gated_fwd_kernel[lambda meta: (triton.cdiv(rows, meta['BT']),)](
            x=x, g=g, y=y, w=w, b=None, residual=None, residual_out=None, mean=None, rstd=rstd, eps=eps, T=rows, D=d,
            BD=triton.next_power_of_2(d), NB=1, ACTIVATION='sigmoid', IS_RMS_NORM=True)
        ctx.save_for_backward(x, g, w, rstd)
        COUNTERS['gdn_norm_forward'] += 1
        return y

    @staticmethod
    def backward(ctx, dy):
        import fla.modules.fused_norm_gate as F
        from fla.utils import get_multiprocessor_count
        x, g, w, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        rows, d = x.shape
        programs = min(get_multiprocessor_count(x.device.index), rows)
        dx, dg = torch.empty_like(x), torch.empty_like(g)
        dw = torch.empty((programs, d), dtype=torch.float32, device=x.device)  # frozen weight: written, never read
        F.layer_norm_gated_bwd_kernel[(programs,)](
            x=x, g=g, w=w, b=None, y=None, dy=dy, dx=dx, dg=dg, dw=dw, db=None, dresidual=None, dresidual_in=None,
            mean=None, rstd=rstd, T=rows, D=d, BS=triton.cdiv(rows, programs), BD=triton.next_power_of_2(d), NB=1,
            ACTIVATION='sigmoid', IS_RMS_NORM=True, STORE_DRESIDUAL=False)
        COUNTERS['gdn_norm_backward'] += 1
        return dx, dg, None, None


def gdn_norm_forward(module, hidden_states, gate=None):
    if gate is None or hidden_states.ndim != 2:
        raise RuntimeError(f'fast gdn_norm: expects a 2-D input and a gate, got {tuple(hidden_states.shape)}')
    return GatedRMSNormFLA.apply(hidden_states, gate, module.weight, module.variance_epsilon)


def _pin_fla_norm_configs():
    import fla.modules.fused_norm_gate as F

    for kernel, (bt, warps) in ((F.layer_norm_gated_fwd_kernel, GDN_NORM_CONFIG['fwd']),
                                (F.layer_norm_gated_bwd_kernel, GDN_NORM_CONFIG['bwd'])):
        # @triton.heuristics(@triton.autotune(@triton.jit)): the configs live on the inner Autotuner; with a single
        # config it launches that one without benchmarking (a timed pick could change the reduction order per run)
        tuner = getattr(kernel, 'fn', None)
        if not isinstance(kernel, triton.runtime.autotuner.Heuristics) \
                or not isinstance(tuner, triton.runtime.Autotuner):
            raise RuntimeError(f'fla {kernel}: expected heuristics over an autotuner, the layout this pins')
        tuner.configs = [triton.Config({'BT': bt}, num_warps=warps)]


# ----------------------------------------------------------------------------------------------------------- shadow
def _record(kind, out, ref):
    if out.shape != ref.shape or out.dtype != ref.dtype:
        raise RuntimeError(f'shadow {kind}: fast {tuple(out.shape)} {out.dtype}, reference {tuple(ref.shape)} {ref.dtype}')
    rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
    s = SHADOW.setdefault(kind, {'calls': 0, 'exact': 0, 'rel_sum': 0.0, 'rel_max': 0.0})
    s['calls'] += 1
    s['exact'] += int(torch.equal(out, ref))
    s['rel_sum'] += rel
    s['rel_max'] = max(s['rel_max'], rel)


def _shadowed(kind, fast, reference):
    def forward(module, *args, **kwargs):
        out = fast(module, *args, **kwargs)
        with torch.no_grad():
            ref = reference(module, *args, **kwargs)
        outs, refs = (out, ref) if isinstance(out, tuple) else ((out,), (ref,))
        if len(outs) != len(refs):
            raise RuntimeError(f'shadow {kind}: {len(outs)} outputs vs {len(refs)}')
        for i, (o, r) in enumerate(zip(outs, refs, strict=True)):
            _record(f'{kind}[{i}]', o.detach(), r)
        return out
    return forward


# ---------------------------------------------------------------------------------------------------------- install
def install(model, kernels, skip_lora=(), shadow=False):
    """Bind ``kernels`` (parse()) on the modules of PEFT ``model``: after the LoRA is attached and fp8_dense_qat is
    installed (its modules go in ``skip_lora``), before FSDP. Every module is checked to be in the state the kernel
    assumes (frozen weights, plain LoRA, sigmoid gate, no forward bound yet). Returns what was bound."""
    from peft.tuners.lora.layer import Linear as LoraLinear
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    if INSTALLED:
        raise RuntimeError('trainer_kernels.install ran twice')
    if shadow and not set(kernels) - {'act_quant'}:
        raise ValueError('shadow compares hc / lora / gdn_norm calls; none is selected')
    bound, binds = {}, []
    if 'hc' in kernels:
        import hc_mix
        counts = {'layer': 0, 'other': 0}
        for name, module in model.named_modules():
            if isinstance(module, M.Qwen4ExpTextGatedResidual):
                if any(p.requires_grad for p in module.parameters()) or 'forward' in module.__dict__:
                    raise RuntimeError(f'{name}: fast hc needs a frozen hyper connection with no bound forward')
                binds.append(('hc', module, hc_mix.gated_residual_forward, M.Qwen4ExpTextGatedResidual.forward))
                # the decoder layers' connections run again in the checkpoint recompute; the final readout does not
                counts['layer' if '.layers.' in name else 'other'] += 1
        if counts != {'layer': 96, 'other': 1}:
            raise RuntimeError(f'fast hc: found hyper connections {counts}, expected 96 in layers + 1 readout')
        bound['hc'] = counts
    if 'hc_inject' in kernels:
        inject = _inject_forward()
        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, M.Qwen4ExpTextDecoderLayer)]
        for name, layer in layers:
            inner = layer.__dict__.get('packed_inner_forward')
            if getattr(inner, '__func__', None) is not M.Qwen4ExpTextDecoderLayer.forward:
                raise RuntimeError(f'{name}: hc_inject needs packed_qwen4\'s wrapper around the stock layer forward')
        if len(layers) != 48:
            raise RuntimeError(f'hc_inject: {len(layers)} decoder layers, expected 48')
        bound['hc_inject'] = len(layers)
    if 'gdn_norm' in kernels:
        _pin_fla_norm_configs()
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, M.Qwen4ExpTextRMSNormGated):
                if module.activation != 'sigmoid' or module.weight.requires_grad or module.weight.shape[0] > 512 \
                        or 'forward' in module.__dict__:
                    raise RuntimeError(f'{name}: fast gdn_norm needs a frozen <=512-wide sigmoid-gated norm')
                binds.append(('gdn_norm', module, gdn_norm_forward, M.Qwen4ExpTextRMSNormGated.forward))
                count += 1
        if count != 36:
            raise RuntimeError(f'fast gdn_norm: {count} gated norms, expected 36 (one per GDN layer)')
        bound['gdn_norm'] = count
    if 'lora' in kernels:
        skip = {id(m) for m in skip_lora}
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear) and id(module) not in skip:
                _check_plain_lora(name, module)
                binds.append(('lora', module, lora_forward, LoraLinear.forward))
                count += 1
        if not count:
            raise RuntimeError('fast lora: no LoRA module outside dense QAT')
        bound['lora'] = count
    if 'lora_grad' in kernels:
        import fp8_dense_qat
        LORA_GRAD['on'] = True
        fp8_dense_qat.GRAD_KERNEL['kernel'] = 'lora_grad'
        bound['lora_grad'] = True
    if 'act_quant' in kernels:
        import fp8_moe
        fp8_moe.set_quant_kernel('grouped')
        bound['act_quant'] = True
    if 'qsa_select' in kernels:
        import sparse_qsa
        sparse_qsa.COMPACT_SELECT['kernel'] = 'fused'
        bound['qsa_select'] = True
    if 'qsa_dkdv' in kernels:
        import qsa_gather
        if torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError('qsa_dkdv: the Gluon dK/dV kernel needs Blackwell (sm_100+)')
        qsa_gather.DKDV_KERNEL['kernel'] = 'gluon'
        bound['qsa_dkdv'] = True
    if 'qsa_union' in kernels:
        import qsa_gather
        if torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError('qsa_union: the query-union kernels need Blackwell (sm_100+)')
        qsa_gather.UNION['on'] = True
        bound['qsa_union'] = True
    if 'gdn_gva' in kernels:
        gva = _gva_forward()
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, M.Qwen4ExpTextGatedDeltaNet):
                if 'forward' in module.__dict__ or module.num_v_heads // module.num_k_heads < 2:
                    raise RuntimeError(f'{name}: gdn_gva needs grouped value heads and no bound forward')
                binds.append(('gdn_gva', module, gva, M.Qwen4ExpTextGatedDeltaNet.forward))
                count += 1
        if count != 36:
            raise RuntimeError(f'gdn_gva: {count} gated delta nets, expected 36')
        bound['gdn_gva'] = count
    for kind, module, fast, reference in binds:
        module.forward = MethodType(_shadowed(kind, fast, reference) if shadow else fast, module)
    if 'hc_inject' in bound:
        for _, layer in layers:
            layer.packed_inner_forward = MethodType(inject, layer)
    INSTALLED.update(bound)
    _SEEN.update(COUNTERS, **_hc_counters())
    return bound


def _hc_counters():
    import hc_mix
    return {'hc_' + k: v for k, v in hc_mix.COUNTERS.items()}


def assert_step_calls(update, frozen_layers=0, frozen_gdn=0, stream_grad=True):
    """After one forward (and backward) of the model: every bound module ran its fast kernel once per forward -- the
    in-layer ones twice with the checkpoint recompute when training -- and once in backward. ``frozen_layers`` (of which
    ``frozen_gdn`` are GDN layers): the first decoder layers no gradient reaches (train_rl_service --freeze-first-layers),
    whose modules run one forward and no recompute or backward. ``stream_grad`` False: the residual stream entering the
    first trained layer carries no gradient (full-weight trainer: frozen embeddings, no input-require-grads hook)."""
    if not INSTALLED:
        return
    now = dict(COUNTERS, **_hc_counters())
    ran = {k: now[k] - _SEEN[k] for k in now}
    _SEEN.update(now)

    def calls(n, frozen):
        """(forwards, backwards) of n per-layer modules, ``frozen`` of them in frozen layers."""
        if not 0 <= frozen <= n:
            raise ValueError(f'{frozen} frozen of {n} modules')
        return ((n - frozen) * 2 + frozen, n - frozen) if update else (n, 0)

    if frozen_layers and 'lora' in INSTALLED:
        raise RuntimeError('frozen layers are a full-weight trainer setting; the LoRA kernel counts assume none')
    want = {}
    if 'hc' in INSTALLED:   # two hyper connections per layer, and the final readout (one forward, one backward)
        layer, other = INSTALLED['hc']['layer'], INSTALLED['hc']['other']
        f, b = calls(layer, 2 * frozen_layers)
        if update and not stream_grad:
            # the first trained layer's attention-side connection reads a stream without gradient with its frozen
            # weights: it is recomputed with its layer but nothing upstream of it needs a gradient, so no backward
            b -= 1
        want.update(hc_forward=f + other, hc_backward=(b + other) if update else 0)
    if 'hc_inject' in INSTALLED:  # two write-backs per layer
        f, b = calls(2 * INSTALLED['hc_inject'], 2 * frozen_layers)
        want.update(hc_inject_forward=f, hc_inject_backward=b)
    if 'lora' in INSTALLED:
        f, b = calls(INSTALLED['lora'], 0)
        want.update(lora_forward=f, lora_backward=b)
    if 'gdn_norm' in INSTALLED:
        f, b = calls(INSTALLED['gdn_norm'], frozen_gdn)
        want.update(gdn_norm_forward=f, gdn_norm_backward=b)
    if 'gdn_gva' in INSTALLED:
        want['gdn_gva_forward'] = calls(INSTALLED['gdn_gva'], frozen_gdn)[0]
    wrong = {k: (ran[k], v) for k, v in want.items() if ran[k] != v}
    if wrong:
        raise RuntimeError(f'fast kernels ran (got, expected) {wrong} this step: a module bypassed its kernel')
